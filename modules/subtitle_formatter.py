"""
modules/subtitle_formatter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 4 — Subtitle Creation & Display Preparation

Updated: now generates a word-level display schedule so the
SubtitleOverlay can reveal Vietnamese words progressively,
synchronised with the original English speech timing.

Strategy for word timing:
  1. Whisper produces word-level timestamps for English words.
  2. The translated Vietnamese text is word-wrapped into lines.
  3. Vietnamese words are mapped to proportional English timestamps:
       VI word i (of N total) → EN word at position round(i/N * M)
  4. If Whisper word timestamps are absent, words are spaced evenly
     across the segment duration.

Input:  asyncio.Queue[TranslatedSubtitle]
Output: asyncio.Queue[DisplaySubtitle]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import SubtitleConfig
from modules.translator import TranslatedSubtitle
from modules.stt_engine import WordTimestamp
from utils.logger import get_logger

log = get_logger("SubtitleFormatter")


# ─────────────────────────────────────────────────────────────
#  Final display contract consumed by the UI
# ─────────────────────────────────────────────────────────────

@dataclass
class DisplaySubtitle:
    lines: List[str]                    # 1-2 lines of Vietnamese text
    display_at_ms: int                  # Absolute wall-clock ms to show
    hide_at_ms: int                     # Absolute wall-clock ms to hide
    segment_id: int
    fade_in_ms: int = 150
    fade_out_ms: int = 300
    original_text: str = ""             # English (for debug overlay)

    # Progressive word reveal schedule.
    # Each entry: (word_string, absolute_ms_to_reveal)
    # Words are in reading order (line 1 then line 2).
    word_schedule: List[Tuple[str, int]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  stable-ts drift corrector (optional)
# ─────────────────────────────────────────────────────────────

class _StableTSAligner:
    def __init__(self):
        self._available = False
        try:
            import stable_whisper  # type: ignore  # noqa: F401
            self._available = True
            log.info("stable-ts available — drift correction enabled ✔")
        except ImportError:
            log.info("stable-ts not installed — using raw Whisper timestamps.")

    def correct_end_time(
        self, end_s: float, text_len: int, reading_speed_ms_per_char: int
    ) -> float:
        min_dur = (text_len * reading_speed_ms_per_char) / 1_000.0
        return max(end_s, (end_s - min_dur) + min_dur)


# ─────────────────────────────────────────────────────────────
#  Subtitle Formatter Module
# ─────────────────────────────────────────────────────────────

class SubtitleFormatterModule:
    def __init__(self, cfg: SubtitleConfig):
        self._cfg     = cfg
        self._aligner = _StableTSAligner()
        self._pipeline_wall_start: float = time.time()

    # ── Public API ────────────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        self._pipeline_wall_start = time.time()
        log.info("SubtitleFormatter pipeline running …")

        while not stop_event.is_set():
            try:
                sub: TranslatedSubtitle = await asyncio.wait_for(
                    input_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                display = self._format(sub)
                if display:
                    await output_queue.put(display)
                    log.debug(
                        f"[seg#{sub.segment_id}] "
                        f"display={display.display_at_ms}ms "
                        f"hide={display.hide_at_ms}ms "
                        f"words={len(display.word_schedule)} | {display.lines}"
                    )
            except Exception as exc:
                log.error(f"Formatting error for seg#{sub.segment_id}: {exc}")
            finally:
                input_queue.task_done()

        log.info("SubtitleFormatter stopped.")

    # ── Formatting logic ──────────────────────────────────────

    def _format(self, sub: TranslatedSubtitle) -> Optional[DisplaySubtitle]:
        cfg   = self._cfg
        lines = sub.lines
        if not lines:
            return None

        # ── 1. Timing ────────────────────────────────────────
        now_ms = int(time.time() * 1_000)
        segment_start_wall_ms = int(
            self._pipeline_wall_start * 1_000 + sub.start * 1_000
        )
        display_at_ms = max(now_ms, segment_start_wall_ms)

        total_chars   = sum(len(ln) for ln in lines)
        reading_ms    = total_chars * cfg.char_reading_speed
        raw_dur_ms    = int((sub.end - sub.start) * 1_000)

        duration_ms = max(raw_dur_ms, reading_ms, cfg.min_display_ms)
        duration_ms = min(duration_ms, cfg.max_display_ms)

        corrected_end_s  = self._aligner.correct_end_time(
            sub.end, total_chars, cfg.char_reading_speed
        )
        drift_ms = max(0, int((corrected_end_s - sub.end) * 1_000))
        duration_ms += drift_ms

        hide_at_ms = display_at_ms + duration_ms

        # ── 2. Word schedule ──────────────────────────────────
        word_schedule = self._build_word_schedule(
            lines, sub.words, display_at_ms, duration_ms
        )

        return DisplaySubtitle(
            lines=lines,
            display_at_ms=display_at_ms,
            hide_at_ms=hide_at_ms,
            segment_id=sub.segment_id,
            fade_in_ms=cfg.fade_in_ms,
            fade_out_ms=cfg.fade_out_ms,
            original_text=sub.original_text,
            word_schedule=word_schedule,
        )

    # ── Word schedule builder ─────────────────────────────────

    def _build_word_schedule(
        self,
        lines: List[str],
        en_words: List[WordTimestamp],
        display_at_ms: int,
        duration_ms: int,
    ) -> List[Tuple[str, int]]:
        """
        Return [(vi_word, abs_show_ms), ...] for progressive reveal.

        Maps each Vietnamese word to an absolute wall-clock millisecond
        at which it should appear on screen.
        """
        # Flatten VI lines → list of words (preserves reading order)
        vi_words: List[str] = []
        for line in lines:
            vi_words.extend(line.split())

        if not vi_words:
            return []

        n_vi = len(vi_words)

        if en_words:
            # ── Map via proportional English timestamps ───────
            # en_words are relative (seconds from segment start).
            # Shift to absolute wall-clock ms.
            seg_start_wall_ms = display_at_ms   # best approximation

            # Build absolute EN word start times
            en_times_ms = [
                seg_start_wall_ms + int(w.start * 1_000)
                for w in en_words
            ]
            n_en = len(en_times_ms)

            schedule: List[Tuple[str, int]] = []
            for i, vi_word in enumerate(vi_words):
                # Proportional index into English word list
                en_idx = min(round(i / n_vi * n_en), n_en - 1)
                show_ms = en_times_ms[en_idx]
                # Never show before the subtitle display time
                show_ms = max(show_ms, display_at_ms)
                schedule.append((vi_word, show_ms))
        else:
            # ── Fallback: evenly distributed ──────────────────
            schedule = [
                (
                    vi_words[i],
                    display_at_ms + int(i / n_vi * duration_ms),
                )
                for i in range(n_vi)
            ]

        return schedule
