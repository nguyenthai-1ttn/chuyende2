"""
modules/groq_translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Groq cloud LLM translator — 30 RPM free tier.

Rate strategy:
  • Token bucket: min 2.0s between consecutive API calls
  • Debounce:     0.8s quiet window before flushing batch
  • Backoff:      1s → 2s → 4s on 429
  • On exhaustion: log warning, drop segment

Input:  asyncio.Queue[GroupedSegment]
Output: asyncio.Queue[TranslatedSubtitle]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import GroqConfig
from modules.stt_engine import WordTimestamp
from utils.logger import get_logger

log = get_logger("GroqTranslator")

try:
    from groq import AsyncGroq, RateLimitError, APIStatusError  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    AsyncGroq = None
    RateLimitError = Exception
    APIStatusError = Exception


# ─────────────────────────────────────────────────────────────
#  Output contract (shared with pipeline + formatter)
# ─────────────────────────────────────────────────────────────

@dataclass
class TranslatedSubtitle:
    original_text: str
    translated_text: str
    start: float
    end: float
    segment_id: int
    words: List[WordTimestamp] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.lines:
            self.lines = [
                ln.strip()
                for ln in self.translated_text.split("\n")
                if ln.strip()
            ]


# ─────────────────────────────────────────────────────────────
#  Groq Translator
# ─────────────────────────────────────────────────────────────

class GroqTranslatorModule:
    """
    Translates GroupedSegment → TranslatedSubtitle via Groq API.

    Implements proactive pacing (min_interval_s) and reactive
    exponential backoff on 429 errors.
    """

    def __init__(self, cfg: GroqConfig):
        self._cfg = cfg
        self._client: Optional[AsyncGroq] = None
        self._last_call_time: float = 0.0

    # ── Validation ────────────────────────────────────────────

    def validate(self) -> None:
        if not _GROQ_AVAILABLE:
            raise RuntimeError(
                "groq package not installed. Run: pip install groq"
            )
        if not self._cfg.api_key.strip():
            raise ValueError("Groq API key is required.")
        # Quick reachability check by creating client (no network call yet)
        self._ensure_client()
        log.info(f"Groq validator OK (model={self._cfg.model!r})")

    # ── Main loop ─────────────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        self._ensure_client()
        log.info(
            f"GroqTranslator running "
            f"(model={self._cfg.model!r}, "
            f"interval={self._cfg.min_interval_s}s, "
            f"debounce={self._cfg.debounce_s}s) …"
        )

        while not stop_event.is_set():
            try:
                segment = await asyncio.wait_for(
                    input_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                subtitle = await self._translate_with_retry(segment)
                if subtitle:
                    await output_queue.put(subtitle)
                    log.debug(
                        f"[seg#{segment.segment_id}] → "
                        f"\"{subtitle.translated_text[:80]}\""
                    )
            except Exception as exc:
                log.error(
                    f"Unexpected error for seg#{segment.segment_id}: {exc}"
                )
            finally:
                input_queue.task_done()

        log.info("GroqTranslator stopped.")

    # ── Translation with retry ────────────────────────────────

    async def _translate_with_retry(self, segment) -> Optional[TranslatedSubtitle]:
        cfg = self._cfg
        backoff_delays = [1.0, 2.0, 4.0]  # 3 retries

        for attempt in range(cfg.max_retries + 1):
            # ── Proactive pacing ─────────────────────────────
            await self._pace()

            try:
                t0 = time.monotonic()
                translated = await asyncio.wait_for(
                    self._call_api(segment.text),
                    timeout=cfg.timeout_s,
                )
                self._last_call_time = time.monotonic()
                elapsed_ms = (time.monotonic() - t0) * 1_000
                log.debug(
                    f"Groq latency: {elapsed_ms:.0f} ms | "
                    f"in={len(segment.text)}c out={len(translated)}c"
                )

                translated = self._sanitise(translated)
                if not translated:
                    log.debug("Empty translation — skipping segment.")
                    return None

                return TranslatedSubtitle(
                    original_text=segment.text,
                    translated_text=translated,
                    start=segment.start,
                    end=segment.end,
                    segment_id=segment.segment_id,
                    words=list(getattr(segment, "words", [])),
                )

            except Exception as exc:
                is_rate_limit = self._is_rate_limit(exc)
                is_last = attempt >= cfg.max_retries

                if is_last:
                    if is_rate_limit:
                        log.warning(
                            f"[Groq] Đã hết rate limit. "
                            f"Xin chuyển sang model khác hoặc đợi."
                        )
                    else:
                        log.error(
                            f"[Groq] Lỗi sau {attempt + 1} lần thử "
                            f"(seg#{segment.segment_id}): {exc}"
                        )
                    return None

                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                log.warning(
                    f"[Groq] Lỗi (attempt {attempt + 1}): "
                    f"{type(exc).__name__} — retry sau {delay}s"
                )
                await asyncio.sleep(delay)

        return None

    # ── API call ──────────────────────────────────────────────

    async def _call_api(self, text: str) -> str:
        cfg = self._cfg
        client = self._ensure_client()

        response = await client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user",   "content": text},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    # ── Helpers ───────────────────────────────────────────────

    async def _pace(self) -> None:
        """Enforce minimum interval between API calls."""
        elapsed = time.monotonic() - self._last_call_time
        wait = self._cfg.min_interval_s - elapsed
        if wait > 0:
            log.debug(f"Pacing: waiting {wait:.2f}s before next Groq call")
            await asyncio.sleep(wait)

    def _ensure_client(self) -> "AsyncGroq":
        if self._client is None:
            if not _GROQ_AVAILABLE:
                raise RuntimeError("groq package not installed.")
            self._client = AsyncGroq(api_key=self._cfg.api_key.strip())
        return self._client

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        if _GROQ_AVAILABLE and isinstance(exc, RateLimitError):
            return True
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "quota" in msg

    def _sanitise(self, text: str) -> str:
        cfg = self._cfg
        for ch in ["*", "_", "`", '"', "'"]:
            text = text.replace(ch, "")
        text = text.strip()

        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not raw_lines:
            return ""

        merged = " ".join(raw_lines)
        lines = self._wrap(merged, cfg.max_chars_per_line)[: cfg.max_lines]
        return "\n".join(lines)

    @staticmethod
    def _wrap(text: str, max_chars: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
