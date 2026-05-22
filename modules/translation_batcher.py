"""
modules/translation_batcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 2.5 — Translation batching & debounce

Merges multiple GroupedSegment objects and emits one batch after
a quiet period (debounce), reducing Ollama calls per minute.

Input:  asyncio.Queue[GroupedSegment]
Output: asyncio.Queue[GroupedSegment]  (merged batches)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from config import TranslationConfig
from modules.sentence_grouper import GroupedSegment
from utils.logger import get_logger

log = get_logger("TranslationBatcher")


class TranslationBatcherModule:
    """
    Debounces grouped STT text before it reaches the translator.

    Parameters
    ----------
    debounce_s      : Seconds of silence before flushing the buffer.
    max_batch_words : Split early if the buffer exceeds this word count.
    """

    def __init__(self, cfg: TranslationConfig):
        self._debounce_s = cfg.debounce_s
        self._max_batch_words = cfg.max_batch_words
        self._buffer: List[GroupedSegment] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._batch_counter = 0

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        log.info(
            f"TranslationBatcher running (debounce={self._debounce_s}s, "
            f"max_words={self._max_batch_words}) …"
        )

        try:
            while not stop_event.is_set():
                try:
                    segment: GroupedSegment = await asyncio.wait_for(
                        input_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                input_queue.task_done()
                self._buffer.append(segment)
                log.debug(
                    f"Buffered group#{segment.segment_id} "
                    f"(buffer={len(self._buffer)} segs)"
                )

                if self._word_count() >= self._max_batch_words:
                    await self._cancel_flush_task()
                    await self._flush(output_queue)
                else:
                    self._schedule_flush(output_queue)

        finally:
            await self._cancel_flush_task()
            if self._buffer:
                await self._flush(output_queue)

        log.info("TranslationBatcher stopped.")

    def _schedule_flush(self, output_queue: asyncio.Queue) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

        async def _debounced() -> None:
            try:
                await asyncio.sleep(self._debounce_s)
                await self._flush(output_queue)
            except asyncio.CancelledError:
                pass

        self._flush_task = asyncio.create_task(_debounced(), name="BatchDebounce")

    async def _cancel_flush_task(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        self._flush_task = None

    async def _flush(self, output_queue: asyncio.Queue) -> None:
        if not self._buffer:
            return

        while self._buffer and self._word_count() > self._max_batch_words:
            batch = self._take_until_word_limit(self._max_batch_words)
            merged = self._merge(batch)
            if merged:
                await output_queue.put(merged)

        if self._buffer:
            merged = self._merge(self._buffer)
            self._buffer.clear()
            if merged:
                await output_queue.put(merged)

    def _take_until_word_limit(self, limit: int) -> List[GroupedSegment]:
        taken: List[GroupedSegment] = []
        words = 0
        while self._buffer and words < limit:
            seg = self._buffer.pop(0)
            taken.append(seg)
            words += len(seg.text.split())
        return taken

    def _word_count(self) -> int:
        return sum(len(s.text.split()) for s in self._buffer)

    def _merge(self, segments: List[GroupedSegment]) -> Optional[GroupedSegment]:
        if not segments:
            return None

        merged_text = " ".join(s.text.strip() for s in segments)
        merged_text = re.sub(r"\s{2,}", " ", merged_text).strip()
        if not merged_text:
            return None

        all_words = []
        for s in segments:
            all_words.extend(s.words)

        self._batch_counter += 1
        return GroupedSegment(
            text=merged_text,
            start=segments[0].start,
            end=segments[-1].end,
            segment_id=self._batch_counter,
            words=all_words,
            source_count=sum(s.source_count for s in segments),
        )
