"""
modules/translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 3 — LLM Translation (Ollama / Qwen local)

  • GroupedSegment input (from TranslationBatcher)
  • Rolling Vietnamese context in each request

Input:  asyncio.Queue[GroupedSegment]
Output: asyncio.Queue[TranslatedSubtitle]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from config import TranslationConfig
from modules.ollama_client import OllamaClient
from modules.stt_engine import TranscriptSegment, WordTimestamp
from utils.logger import get_logger

log = get_logger("Translator")

try:
    from modules.sentence_grouper import GroupedSegment
    SegmentInput = Union[TranscriptSegment, GroupedSegment]
except ImportError:
    GroupedSegment = None
    SegmentInput = TranscriptSegment


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


class TranslatorModule:
    """
    Async translation via Ollama (local GPU, quantized GGUF tags).
    """

    def __init__(self, cfg: TranslationConfig, context_window: int = 2):
        self._cfg = cfg
        self._client: Optional[OllamaClient] = None
        self._context_window = context_window
        self._vi_context: List[str] = []

    def validate(self) -> None:
        client = self._ensure_client()
        client.check_reachable()
        client.warn_if_model_missing()

    async def preload(self) -> None:
        """Load model into VRAM before pipeline runs."""
        await self._ensure_client().preload()

    def _ensure_client(self) -> OllamaClient:
        if self._client is None:
            self._client = OllamaClient(self._cfg)
        return self._client

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        self._ensure_client()
        log.info(
            f"TranslatorModule running (Ollama model={self._cfg.model!r}) …"
        )

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

    def _update_context(self, vi_text: str) -> None:
        first_line = vi_text.split("\n")[0].strip()
        if first_line:
            self._vi_context.append(first_line)
            if len(self._vi_context) > self._context_window:
                self._vi_context.pop(0)

    def _build_prompt(self, segment) -> str:
        parts: List[str] = []
        if self._vi_context:
            ctx_str = " | ".join(self._vi_context)
            parts.append(f"[Phụ đề trước: {ctx_str}]")
        parts.append(segment.text)
        return "\n".join(parts)

    async def _translate_with_retry(self, segment) -> Optional[TranslatedSubtitle]:
        cfg = self._cfg
        client = self._ensure_client()
        last_exc: Optional[Exception] = None

        for attempt in range(cfg.translation_max_retries + 1):
            if attempt > 0:
                delay = cfg.translation_retry_delay_s * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
                log.debug(f"Retry {attempt} for seg#{segment.segment_id}")

            try:
                t0 = time.monotonic()
                prompt = self._build_prompt(segment)

                translated = await asyncio.wait_for(
                    client.generate(
                        prompt=prompt,
                        system=cfg.system_prompt,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        num_predict=cfg.num_predict,
                    ),
                    timeout=cfg.timeout_s,
                )
                elapsed_ms = (time.monotonic() - t0) * 1_000
                log.debug(
                    f"Ollama latency: {elapsed_ms:.0f} ms "
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
                    f"Ollama error (attempt {attempt + 1}): "
                    f"{type(exc).__name__}: {exc}"
                )

        log.error(f"All retries exhausted for seg#{segment.segment_id}: {last_exc}")
        return None

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
