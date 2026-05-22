"""
pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline Orchestrator — now with SentenceGrouper stage.

Updated architecture:

  AudioCapture  ──[pcm_q]──▶    DeepgramSTT (live)
  DeepgramSTT   ──[stt_q]──▶    SentenceGrouper
  Grouper       ──[group_q]──▶  TranslationBatcher
  Batcher       ──[batch_q]──▶  TranslatorModule (Ollama)
  Ollama        ──[trans_q]──▶  SubtitleFormatter
  Formatter     ──[disp_q]──▶   UI (Tkinter via callback)

Each module runs as a separate asyncio Task inside a shared event
loop that lives in its own daemon thread.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

from config import AppConfig
from modules.audio_capture import AudioCaptureModule
from modules.sentence_grouper import SentenceGrouperModule
from modules.deepgram_stt import DeepgramSTTEngine
from modules.subtitle_formatter import DisplaySubtitle, SubtitleFormatterModule
from modules.translation_batcher import TranslationBatcherModule
from modules.translator import TranslatorModule
from utils.logger import get_logger

log = get_logger("Pipeline")


# ─────────────────────────────────────────────────────────────
#  Pipeline state
# ─────────────────────────────────────────────────────────────

class PipelineState(Enum):
    IDLE     = auto()
    STARTING = auto()
    RUNNING  = auto()
    STOPPING = auto()
    ERROR    = auto()


# ─────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────

class SubtitlePipeline:
    """
    Top-level pipeline orchestrator.

    Parameters
    ----------
    cfg            : AppConfig
    subtitle_cb    : Callable[[DisplaySubtitle], None]
    status_cb      : Callable[[str, str], None]   (module_name, status_text)
    transcript_cb  : Callable[[str], None]         (raw English text)
    """

    def __init__(
        self,
        cfg: AppConfig,
        subtitle_cb: Callable[[DisplaySubtitle], None],
        status_cb: Optional[Callable[[str, str], None]] = None,
        transcript_cb: Optional[Callable[[str], None]] = None,
    ):
        self._cfg          = cfg
        self._subtitle_cb  = subtitle_cb
        self._status_cb    = status_cb or (lambda m, s: None)
        self._transcript_cb = transcript_cb or (lambda t: None)

        self.state = PipelineState.IDLE
        self._stop_event: Optional[asyncio.Event]               = None
        self._loop:       Optional[asyncio.AbstractEventLoop]   = None
        self._thread:     Optional[threading.Thread]            = None

        # Asyncio inter-module queues (created in start())
        self._audio_q:  Optional[asyncio.Queue] = None
        self._stt_q:    Optional[asyncio.Queue] = None
        self._group_q:  Optional[asyncio.Queue] = None
        self._batch_q:  Optional[asyncio.Queue] = None
        self._trans_q:  Optional[asyncio.Queue] = None
        self._disp_q:   Optional[asyncio.Queue] = None

        # Modules
        self._audio_mod:  Optional[AudioCaptureModule]      = None
        self._stt_mod:    Optional[DeepgramSTTEngine]       = None
        self._group_mod:  Optional[SentenceGrouperModule]   = None
        self._batch_mod:  Optional[TranslationBatcherModule] = None
        self._trans_mod:  Optional[TranslatorModule]        = None
        self._fmt_mod:    Optional[SubtitleFormatterModule] = None

    # ── Public API ────────────────────────────────────────────

    def start(self) -> None:
        if self.state not in (PipelineState.IDLE, PipelineState.ERROR):
            log.warning("Pipeline already running.")
            return

        self.state  = PipelineState.STARTING
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="PipelineThread"
        )
        self._thread.start()
        log.info("Pipeline thread started.")

    def stop(self) -> None:
        if self.state not in (PipelineState.RUNNING, PipelineState.STARTING):
            return

        self.state = PipelineState.STOPPING
        log.info("Pipeline stopping …")

        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        if self._thread:
            self._thread.join(timeout=10)

        self.state = PipelineState.IDLE
        log.info("Pipeline stopped.")

    def is_running(self) -> bool:
        return self.state == PipelineState.RUNNING

    # ── Event loop ────────────────────────────────────────────

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:
            log.error(f"Pipeline fatal error: {exc}")
            self.state = PipelineState.ERROR
            self._status_cb("Pipeline", f"ERROR: {exc}")
        finally:
            self._loop.close()

    # ── Async main ────────────────────────────────────────────

    async def _main(self) -> None:
        cfg  = self._cfg
        pcfg = cfg.pipeline

        self._stop_event = asyncio.Event()

        # ── Create queues ────────────────────────────────────
        self._audio_q = asyncio.Queue(maxsize=pcfg.audio_queue_maxsize)
        self._stt_q   = asyncio.Queue(maxsize=pcfg.stt_queue_maxsize)
        self._group_q = asyncio.Queue(maxsize=pcfg.stt_queue_maxsize)
        self._batch_q = asyncio.Queue(maxsize=pcfg.translation_queue_maxsize)
        self._trans_q = asyncio.Queue(maxsize=pcfg.translation_queue_maxsize)
        self._disp_q  = asyncio.Queue(maxsize=pcfg.subtitle_queue_maxsize)

        # ── Instantiate modules ──────────────────────────────
        self._audio_mod = AudioCaptureModule(cfg.audio)
        self._stt_mod   = DeepgramSTTEngine(cfg.stt)
        gcfg = cfg.grouper
        self._group_mod = SentenceGrouperModule(
            max_wait_s=gcfg.max_wait_s,
            gap_threshold_s=gcfg.gap_threshold_s,
            max_words=gcfg.max_words,
            flush_on_gap=gcfg.flush_on_gap,
        )
        self._batch_mod = TranslationBatcherModule(cfg.translation)
        self._trans_mod = TranslatorModule(cfg.translation, context_window=2)
        self._fmt_mod   = SubtitleFormatterModule(cfg.subtitle)

        # ── Validate cloud APIs ───────────────────────────────
        self._status_cb("Deepgram", "Checking …")
        try:
            self._stt_mod.validate()
            self._status_cb("Deepgram", "Ready ✔")
        except Exception as exc:
            log.error(f"Deepgram STT setup failed: {exc}")
            self._status_cb("Deepgram", f"Error: {exc}")
            self.state = PipelineState.ERROR
            return

        self._status_cb("Ollama", "Checking …")
        try:
            self._trans_mod.validate()
            if cfg.translation.preload_on_start:
                self._status_cb("Ollama", "Preloading model …")
                await self._trans_mod.preload()
            self._status_cb("Ollama", "Ready ✔")
        except Exception as exc:
            log.error(f"Ollama translator setup failed: {exc}")
            self._status_cb("Ollama", f"Error: {exc}")
            self.state = PipelineState.ERROR
            return

        # ── Start audio capture ──────────────────────────────
        self._status_cb("AudioCapture", "Starting …")
        await self._audio_mod.start(self._audio_q)
        self._status_cb("AudioCapture", "Running ✔")

        self.state = PipelineState.RUNNING
        self._status_cb("Pipeline", "Running ✔")

        # ── Launch async tasks ───────────────────────────────
        tasks = [
            asyncio.create_task(
                self._stt_mod.run(
                    self._audio_q, self._stt_q,
                    self._stop_event, self._transcript_cb,
                ),
                name="Deepgram",
            ),
            asyncio.create_task(
                self._group_mod.run(
                    self._stt_q, self._group_q, self._stop_event
                ),
                name="Grouper",
            ),
            asyncio.create_task(
                self._batch_mod.run(
                    self._group_q, self._batch_q, self._stop_event
                ),
                name="Batcher",
            ),
            asyncio.create_task(
                self._trans_mod.run(
                    self._batch_q, self._trans_q, self._stop_event
                ),
                name="Ollama",
            ),
            asyncio.create_task(
                self._fmt_mod.run(
                    self._trans_q, self._disp_q, self._stop_event
                ),
                name="Formatter",
            ),
            asyncio.create_task(
                self._dispatch_subtitles(),
                name="Dispatcher",
            ),
        ]

        await self._stop_event.wait()

        for t in tasks:
            t.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        await self._audio_mod.stop()

        self._status_cb("Pipeline", "Stopped.")
        log.info("All pipeline tasks finished.")

    # ── Subtitle dispatcher ───────────────────────────────────

    async def _dispatch_subtitles(self) -> None:
        while not self._stop_event.is_set():
            try:
                sub: DisplaySubtitle = await asyncio.wait_for(
                    self._disp_q.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            now_ms  = int(time.time() * 1_000)
            wait_ms = sub.display_at_ms - now_ms
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1_000.0)

            try:
                self._subtitle_cb(sub)
            except Exception as exc:
                log.warning(f"Subtitle callback error: {exc}")
            finally:
                self._disp_q.task_done()
