"""
modules/gemini_translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google Gemini cloud LLM translator — 15 RPM free tier.

Rate strategy:
  • Token bucket: min 4.0s between consecutive API calls
  • Debounce:     1.5s quiet window before flushing batch
  • Backoff:      2s → 4s → 8s on 429
  • On exhaustion: log warning, drop segment

Input:  asyncio.Queue[GroupedSegment]
Output: asyncio.Queue[TranslatedSubtitle]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from config import GeminiConfig
from modules.groq_translator import TranslatedSubtitle   # reuse same dataclass
from utils.logger import get_logger

log = get_logger("GeminiTranslator")

try:
    import google.generativeai as genai               # type: ignore
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    genai = None
    ResourceExhausted = Exception
    TooManyRequests = Exception


class GeminiTranslatorModule:
    """
    Translates GroupedSegment → TranslatedSubtitle via Gemini API.

    Implements proactive pacing (min_interval_s) and reactive
    exponential backoff on quota errors.
    """

    def __init__(self, cfg: GeminiConfig):
        self._cfg = cfg
        self._model = None
        self._last_call_time: float = 0.0

    # ── Validation ────────────────────────────────────────────

    def validate(self) -> None:
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
        if not self._cfg.api_key.strip():
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=self._cfg.api_key.strip())
        log.info(f"Gemini validator OK (model={self._cfg.model!r})")

    # ── Main loop ─────────────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        self._ensure_model()
        log.info(
            f"GeminiTranslator running "
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

        log.info("GeminiTranslator stopped.")

    # ── Translation with retry ────────────────────────────────

    async def _translate_with_retry(self, segment) -> Optional[TranslatedSubtitle]:
        cfg = self._cfg
        backoff_delays = [2.0, 4.0, 8.0]

        for attempt in range(cfg.max_retries + 1):
            await self._pace()

            try:
                t0 = time.monotonic()
                translated = await asyncio.get_event_loop().run_in_executor(
                    None, self._call_api_sync, segment.text
                )
                self._last_call_time = time.monotonic()
                elapsed_ms = (time.monotonic() - t0) * 1_000
                log.debug(
                    f"Gemini latency: {elapsed_ms:.0f} ms | "
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
                            f"[Gemini] Đã hết rate limit. "
                            f"Xin chuyển sang model khác hoặc đợi."
                        )
                    else:
                        log.error(
                            f"[Gemini] Lỗi sau {attempt + 1} lần thử "
                            f"(seg#{segment.segment_id}): {exc}"
                        )
                    return None

                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                log.warning(
                    f"[Gemini] Lỗi (attempt {attempt + 1}): "
                    f"{type(exc).__name__} — retry sau {delay}s"
                )
                await asyncio.sleep(delay)

        return None

    # ── Sync API call (run in executor) ──────────────────────

    def _call_api_sync(self, text: str) -> str:
        model = self._ensure_model()
        response = model.generate_content(
            contents=text,
            generation_config={
                "temperature": self._cfg.temperature,
                "max_output_tokens": self._cfg.max_tokens,
            },
        )
        return (response.text or "").strip()

    # ── Helpers ───────────────────────────────────────────────

    async def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        wait = self._cfg.min_interval_s - elapsed
        if wait > 0:
            log.debug(f"Pacing: waiting {wait:.2f}s before next Gemini call")
            await asyncio.sleep(wait)

    def _ensure_model(self):
        if self._model is None:
            if not _GEMINI_AVAILABLE:
                raise RuntimeError("google-generativeai not installed.")
            genai.configure(api_key=self._cfg.api_key.strip())
            self._model = genai.GenerativeModel(
                model_name=self._cfg.model,
                system_instruction=self._cfg.system_prompt,
            )
        return self._model

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        if _GEMINI_AVAILABLE and isinstance(exc, (ResourceExhausted, TooManyRequests)):
            return True
        msg = str(exc).lower()
        return "429" in msg or "quota" in msg or "resource exhausted" in msg

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
