"""
modules/stt_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 2 — Speech-to-Text Conversion

Uses Faster-Whisper (CTranslate2 backend) for ultra-low latency
transcription of audio segments produced by AudioCaptureModule.

Model recommendation:
  • distil-small.en  → ~4× faster than Whisper small, best for latency
  • base             → slightly more accurate, still fast

Input:  asyncio.Queue[np.ndarray]   (float32, 16 kHz, from AudioCapture)
Output: asyncio.Queue[TranscriptSegment]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import STTConfig
from utils.logger import get_logger

log = get_logger("STTEngine")


# ─────────────────────────────────────────────────────────────
#  Data contract shared across downstream modules
# ─────────────────────────────────────────────────────────────

@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds from segment start
    end: float
    probability: float


@dataclass
class TranscriptSegment:
    text: str                              # Full segment text (raw English)
    start: float                           # Absolute timestamp (seconds since pipeline start)
    end: float
    words: List[WordTimestamp] = field(default_factory=list)
    confidence: float = 1.0               # Average word probability
    segment_id: int = 0


# ─────────────────────────────────────────────────────────────
#  STT Engine
# ─────────────────────────────────────────────────────────────

class STTEngine:
    """
    Wraps Faster-Whisper and exposes an async `run()` coroutine
    that consumes audio segments and emits TranscriptSegment objects.
    """

    def __init__(self, cfg: STTConfig):
        self._cfg = cfg
        self._model = None
        self._pipeline_start_time: float = time.monotonic()
        self._segment_counter: int = 0

    # ── Initialisation ────────────────────────────────────────

    def load(self) -> None:
        """Load the Faster-Whisper model (call once before run())."""
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed.\n"
                "Install: pip install faster-whisper"
            )

        log.info(
            f"Loading Faster-Whisper model '{self._cfg.model_name}' "
            f"({self._cfg.device}, {self._cfg.compute_type}) …"
        )
        self._model = WhisperModel(
            self._cfg.model_name,
            device=self._cfg.device,
            compute_type=self._cfg.compute_type,
        )
        self._pipeline_start_time = time.monotonic()
        log.info("Faster-Whisper model loaded ✔")

    # ── Main async loop ───────────────────────────────────────

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        transcript_cb=None,
    ) -> None:
        """
        Continuously consume audio segments, transcribe them, and push
        TranscriptSegment objects to output_queue.
        """
        if self._model is None:
            raise RuntimeError("Call STTEngine.load() before run().")

        log.info("STTEngine pipeline running …")

        while not stop_event.is_set():
            try:
                audio: np.ndarray = await asyncio.wait_for(
                    input_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # Skip very short or silent segments early
            if len(audio) < 1600:
                log.debug(f"Skipped short segment ({len(audio)} samples).")
                input_queue.task_done()
                continue

            try:
                segment = await asyncio.get_event_loop().run_in_executor(
                    None, self._transcribe, audio
                )
                if segment is not None:
                    await output_queue.put(segment)
                    if transcript_cb:
                        transcript_cb(segment.text)
                    log.debug(
                        f"[seg#{segment.segment_id}] {segment.start:.2f}s–{segment.end:.2f}s "
                        f"| conf={segment.confidence:.2f} | \"{segment.text[:60]}\""
                    )
            except Exception as exc:
                log.error(f"Transcription error: {exc}")
            finally:
                input_queue.task_done()

        log.info("STTEngine stopped.")

    # ── Transcription (blocking, runs in thread pool) ─────────

    def _transcribe(self, audio: np.ndarray) -> Optional[TranscriptSegment]:
        cfg = self._cfg
        now = time.monotonic() - self._pipeline_start_time

        segments, info = self._model.transcribe(
            audio,
            language=cfg.language,
            beam_size=cfg.beam_size,
            best_of=cfg.best_of,
            word_timestamps=cfg.word_timestamps,
            condition_on_previous_text=cfg.condition_on_previous_text,
            vad_filter=cfg.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": cfg.vad_min_silence_duration_ms,
            },
        )

        # Collect all segments (generator — must iterate to get results)
        all_segs = list(segments)
        if not all_segs:
            return None

        # Merge all Whisper segments into one TranscriptSegment
        full_text = " ".join(s.text.strip() for s in all_segs).strip()
        if not full_text:
            return None

        # Aggregate word-level timestamps
        word_list: List[WordTimestamp] = []
        probs: List[float] = []

        for seg in all_segs:
            if seg.words:
                for w in seg.words:
                    word_list.append(
                        WordTimestamp(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )
                    probs.append(w.probability)

        avg_conf = float(np.mean(probs)) if probs else 1.0

        # Confidence gate
        if avg_conf < cfg.min_confidence:
            log.debug(f"Dropped low-confidence segment (conf={avg_conf:.2f}): {full_text}")
            return None

        self._segment_counter += 1
        seg_start = all_segs[0].start + now
        seg_end   = all_segs[-1].end  + now

        return TranscriptSegment(
            text=full_text,
            start=seg_start,
            end=seg_end,
            words=word_list,
            confidence=avg_conf,
            segment_id=self._segment_counter,
        )
