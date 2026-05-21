"""
modules/translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 3 — Natural Language Processing & LLM Translation

Updated to support:
  • GroupedSegment input (from SentenceGrouper)
  • Rolling Vietnamese context — last N translations are injected
    into each request so the LLM maintains terminology consistency.

Input:  asyncio.Queue[GroupedSegment | TranscriptSegment]
Output: asyncio.Queue[TranslatedSubtitle]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from config import TranslationConfig
from modules.stt_engine import TranscriptSegment, WordTimestamp
from utils.logger import get_logger

log = get_logger("Translator")

# Support both raw TranscriptSegment and grouped segments
try:
    from modules.sentence_grouper import GroupedSegment
    SegmentInput = Union[TranscriptSegment, GroupedSegment]
except ImportError:
    GroupedSegment = None
    SegmentInput = TranscriptSegment


# ─────────────────────────────────────────────────────────────
#  Output data contract
# ─────────────────────────────────────────────────────────────

@dataclass
class TranslatedSubtitle:
    original_text: str
    translated_text: str              # Already line-broken by LLM
    start: float                      # Seconds (from pipeline start)
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
#  Ollama HTTP client (async)
# ─────────────────────────────────────────────────────────────

class _OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int):
        self._url   = f"{base_url.rstrip('/')}/api/generate"
        self._model = model
        self._timeout = timeout_s

    async def generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        top_p: float,
        num_predict: int,
    ) -> str:
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p":       top_p,
                "num_predict": num_predict,
                "stop":        ["\n\n", "---"],
            },
        }

        try:
            import aiohttp  # type: ignore
            return await self._generate_aiohttp(payload, aiohttp)
        except ImportError:
            return await self._generate_fallback(payload)

    async def _generate_aiohttp(self, payload: dict, aiohttp) -> str:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return data.get("response", "").strip()

    async def _generate_fallback(self, payload: dict) -> str:
        import urllib.request
        body = json.dumps(payload).encode()

        def _blocking():
            req = urllib.request.Request(
                self._url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                return json.loads(r.read()).get("response", "").strip()

        return await asyncio.get_event_loop().run_in_executor(None, _blocking)


# ─────────────────────────────────────────────────────────────
#  Translation module
# ─────────────────────────────────────────────────────────────

class TranslatorModule:
    """
    Async translation module.

    Context window: keeps last `context_window` Vietnamese translations
    and prepends them to each new request so the LLM can maintain
    consistent wording across sentences.
    """

    def __init__(self, cfg: TranslationConfig, context_window: int = 2):
        self._cfg    = cfg
        self._client = _OllamaClient(cfg.ollama_base_url, cfg.model, cfg.timeout_s)

        self._context_window = context_window
        self._vi_context: List[str] = []     # Rolling Vietnamese translation history

    # ── Main async loop ───────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        await self._check_ollama_alive()
        log.info("TranslatorModule pipeline running …")

        while not stop_event.is_set():
            try:
                segment = await asyncio.wait_for(input_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                subtitle = await self._translate_with_retry(segment)
                if subtitle:
                    self._update_context(subtitle.translated_text)
                    await output_queue.put(subtitle)
                    log.debug(
                        f"[seg#{segment.segment_id}] → \"{subtitle.translated_text[:80]}\""
                    )
            except Exception as exc:
                log.error(f"Translation failed for seg#{segment.segment_id}: {exc}")
                await output_queue.put(
                    TranslatedSubtitle(
                        original_text=segment.text,
                        translated_text="[Lỗi dịch]",
                        start=segment.start,
                        end=segment.end,
                        segment_id=segment.segment_id,
                        words=list(getattr(segment, "words", [])),
                    )
                )
            finally:
                input_queue.task_done()

        log.info("TranslatorModule stopped.")

    # ── Context management ────────────────────────────────────

    def _update_context(self, vi_text: str) -> None:
        """Add latest translation to rolling context buffer."""
        # Keep only the first line of each subtitle for context brevity
        first_line = vi_text.split("\n")[0].strip()
        if first_line:
            self._vi_context.append(first_line)
            if len(self._vi_context) > self._context_window:
                self._vi_context.pop(0)

    def _build_prompt(self, segment) -> str:
        """
        Build the user prompt, injecting Vietnamese context when available.

        Format:
            [Phụ đề trước: <vi_line_1> | <vi_line_2>]
            <English text to translate>
        """
        parts: List[str] = []

        if self._vi_context:
            ctx_str = " | ".join(self._vi_context)
            parts.append(f"[Phụ đề trước: {ctx_str}]")

        parts.append(segment.text)
        return "\n".join(parts)

    # ── Translation with retry ────────────────────────────────

    async def _translate_with_retry(self, segment) -> Optional[TranslatedSubtitle]:
        cfg = self._cfg
        last_exc: Optional[Exception] = None

        for attempt in range(cfg.translation_max_retries + 1):
            if attempt > 0:
                delay = cfg.translation_retry_delay_s * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
                log.debug(f"Retry {attempt} for seg#{segment.segment_id}")

            try:
                t0 = time.monotonic()
                prompt = self._build_prompt(segment)

                translated = await self._client.generate(
                    prompt=prompt,
                    system=cfg.system_prompt,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    num_predict=cfg.num_predict,
                )
                elapsed_ms = (time.monotonic() - t0) * 1_000
                log.debug(
                    f"LLM latency: {elapsed_ms:.0f} ms "
                    f"| in={len(segment.text)} chars, out={len(translated)} chars"
                )

                translated = self._sanitise(translated)
                if not translated:
                    log.debug("Empty translation — skipping.")
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
                last_exc = exc
                log.warning(
                    f"Ollama error (attempt {attempt + 1}): {type(exc).__name__}: {exc}"
                )

        log.error(f"All retries exhausted for seg#{segment.segment_id}: {last_exc}")
        return None

    # ── Text sanitisation ─────────────────────────────────────

    def _sanitise(self, text: str) -> str:
        """
        Post-process LLM output:
          • Strip markdown / quotes
          • Enforce max_lines × max_chars_per_line
        """
        cfg = self._cfg

        for ch in ["*", "_", "`", '"', "'"]:
            text = text.replace(ch, "")
        text = text.strip()

        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not raw_lines:
            return ""

        merged = " ".join(raw_lines)
        lines  = self._wrap(merged, cfg.max_chars_per_line)[: cfg.max_lines]
        return "\n".join(lines)

    @staticmethod
    def _wrap(text: str, max_chars: int) -> List[str]:
        words   = text.split()
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

    # ── Health check ──────────────────────────────────────────

    async def _check_ollama_alive(self) -> None:
        url = f"{self._cfg.ollama_base_url.rstrip('/')}/api/tags"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        log.info(f"Ollama reachable at {self._cfg.ollama_base_url} ✔")
                        return
        except Exception:
            pass
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=5)
            log.info(f"Ollama reachable at {self._cfg.ollama_base_url} ✔")
        except Exception as exc:
            log.warning(
                f"Ollama not reachable ({exc}). Run: `ollama serve`"
            )
