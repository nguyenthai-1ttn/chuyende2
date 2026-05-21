"""
modules/sentence_grouper.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 1.5 — Sentence Boundary Detection & Semantic Grouping

Sits between STTEngine and TranslatorModule.
Buffers TranscriptSegment objects and flushes them as one
GroupedSegment when a natural sentence boundary is detected.

Flush triggers (any one is enough):
  1. Text ends with sentence-ending punctuation  (. ! ?)
  2. Speech gap between consecutive segments > gap_threshold_s
  3. Word count exceeds max_words
  4. Hard timeout (max_wait_s) — so display is never stuck

Input:  asyncio.Queue[TranscriptSegment]
Output: asyncio.Queue[GroupedSegment]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from modules.stt_engine import TranscriptSegment, WordTimestamp
from utils.logger import get_logger

log = get_logger("SentenceGrouper")

# Regex: text ends with sentence-closing punctuation
_SENTENCE_END_RE = re.compile(r"[.!?…]+\s*$")

# Common sentence-medial abbreviations — don't flush after these
_ABBREV_RE = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est|avg|max|min)\.$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
#  Output data contract
# ─────────────────────────────────────────────────────────────

@dataclass
class GroupedSegment:
    """
    One complete (or near-complete) sentence, formed by merging
    one or more TranscriptSegment objects.
    """
    text: str                               # Merged English text
    start: float                            # Earliest segment start (seconds)
    end: float                              # Latest segment end (seconds)
    segment_id: int
    words: List[WordTimestamp] = field(default_factory=list)
    source_count: int = 1                   # How many STT segments were merged


# ─────────────────────────────────────────────────────────────
#  SentenceGrouperModule
# ─────────────────────────────────────────────────────────────

class SentenceGrouperModule:
    """
    Accumulates TranscriptSegment objects and emits GroupedSegment
    objects at natural sentence boundaries.

    Parameters
    ----------
    max_wait_s        : Hard timeout — flush even if no sentence end found.
    gap_threshold_s   : If the gap between two consecutive segments exceeds
                        this, treat it as a sentence boundary.
    max_words         : Word-count cap; flush early to avoid very long lines.
    """

    def __init__(
        self,
        max_wait_s: float = 1.8,
        gap_threshold_s: float = 1.0,
        max_words: int = 35,
    ):
        self._max_wait_s      = max_wait_s
        self._gap_threshold_s = gap_threshold_s
        self._max_words       = max_words

        self._buffer: List[TranscriptSegment] = []
        self._buffer_start_time: float = 0.0
        self._seg_counter: int = 0

    # ── Main async loop ───────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        log.info("SentenceGrouper running …")

        while not stop_event.is_set():
            # Dynamic timeout: how long until we must force-flush?
            remaining = self._time_until_forced_flush()

            try:
                segment: TranscriptSegment = await asyncio.wait_for(
                    input_queue.get(), timeout=remaining
                )
                input_queue.task_done()
                self._ingest(segment)

                if self._should_flush_on_content(segment):
                    grouped = self._flush()
                    if grouped:
                        await output_queue.put(grouped)
                        log.debug(
                            f"[group#{grouped.segment_id}] Flushed on content "
                            f"({grouped.source_count} segs) — \"{grouped.text[:60]}\""
                        )

            except asyncio.TimeoutError:
                # Hard timeout expired — emit whatever we have
                if self._buffer:
                    grouped = self._flush()
                    if grouped:
                        await output_queue.put(grouped)
                        log.debug(
                            f"[group#{grouped.segment_id}] Flushed on timeout "
                            f"({grouped.source_count} segs) — \"{grouped.text[:60]}\""
                        )

        # Final drain
        if self._buffer:
            grouped = self._flush()
            if grouped:
                await output_queue.put(grouped)

        log.info("SentenceGrouper stopped.")

    # ── Ingestion ─────────────────────────────────────────────

    def _ingest(self, segment: TranscriptSegment) -> None:
        if not self._buffer:
            self._buffer_start_time = time.monotonic()

        # Check speech gap BEFORE appending (gap between prev end and new start)
        if self._buffer and self._has_large_gap(segment):
            # Emit current buffer first, then start fresh with this segment
            # We can't await here, so mark for flush on NEXT call.
            # Simple approach: treat gap as a flush trigger (handled in should_flush_on_content).
            pass

        self._buffer.append(segment)

    # ── Flush decision ────────────────────────────────────────

    def _should_flush_on_content(self, latest: TranscriptSegment) -> bool:
        if not self._buffer:
            return False

        text = latest.text.strip()

        # 1. Sentence-ending punctuation (exclude abbreviations)
        if _SENTENCE_END_RE.search(text) and not _ABBREV_RE.search(text):
            return True

        # 2. Speech gap before the latest segment
        if len(self._buffer) >= 2:
            prev = self._buffer[-2]
            if (latest.start - prev.end) >= self._gap_threshold_s:
                return True

        # 3. Word count cap
        if self._total_words() >= self._max_words:
            return True

        return False

    def _has_large_gap(self, new_seg: TranscriptSegment) -> bool:
        if not self._buffer:
            return False
        return (new_seg.start - self._buffer[-1].end) >= self._gap_threshold_s

    def _time_until_forced_flush(self) -> float:
        """Seconds until hard timeout expires. Never < 0.1 s."""
        if not self._buffer:
            return self._max_wait_s
        elapsed = time.monotonic() - self._buffer_start_time
        return max(0.1, self._max_wait_s - elapsed)

    # ── Flushing ──────────────────────────────────────────────

    def _flush(self) -> Optional[GroupedSegment]:
        if not self._buffer:
            return None

        segs = self._buffer
        self._buffer = []
        self._buffer_start_time = 0.0

        # Merge text; strip duplicate spaces
        merged_text = " ".join(s.text.strip() for s in segs)
        merged_text = re.sub(r"\s{2,}", " ", merged_text).strip()

        if not merged_text:
            return None

        # Merge word-level timestamps (already absolute within each segment)
        all_words: List[WordTimestamp] = []
        for s in segs:
            all_words.extend(s.words)

        self._seg_counter += 1

        return GroupedSegment(
            text=merged_text,
            start=segs[0].start,
            end=segs[-1].end,
            segment_id=self._seg_counter,
            words=all_words,
            source_count=len(segs),
        )

    # ── Helpers ───────────────────────────────────────────────

    def _total_words(self) -> int:
        return sum(len(s.text.split()) for s in self._buffer)
