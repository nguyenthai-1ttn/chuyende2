"""
pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline Orchestrator — Dual-provider (Groq / Gemini).

Architecture:
  AudioCapture  ──[pcm_q]──▶  DeepgramSTT (live)
  DeepgramSTT   ──[stt_q]──▶  SentenceGrouper
  Grouper       ──[group_q]─▶  TranslationBatcher
  Batcher       ──[batch_q]─▶  Groq OR Gemini (via TranslatorFactory)
  Translator    ──[trans_q]─▶  SubtitleFormatter
  Formatter     ──[disp_q]──▶  UI (Tkinter via callback)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional, Union

from config import AppConfig
from modules.audio_capture import AudioCaptureModule
from modules.sentence_grouper import SentenceGrouperModule
from modules.deepgram_stt import DeepgramSTTEngine
from modules.subtitle_formatter import DisplaySubtitle, SubtitleFormatterModule
from modules.translation_batcher import TranslationBatcherModule
from modules.translator_factory import TranslatorFactory
from utils.logger import get_logger

log = get_logger("Pipeline")


class PipelineState(Enum):
    IDLE     = auto()
    STARTING = auto()
    RUNNING  = auto()
    STOPPING = auto()
    ERROR    = auto()


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
        self._cfg           = cfg
        self._subtitle_cb   = subtitle_cb
        self._status_cb     = status_cb or (lambda m, s: None)
        self._transcript_cb = transcript_cb or (lambda t: None)

        self.state = PipelineState.IDLE
        self._stop_event: Optional[asyncio.Event]             = None
        self._loop:       Optional[asyncio.AbstractEventLoop] = None
        self._thread:     Optional[threading.Thread]          = None

        self._audio_q: Optional[asyncio.Queue] = None
        self._stt_q:   Optional[asyncio.Queue] = None
        self._group_q: Optional[asyncio.Queue] = None
        self._batch_q: Optional[asyncio.Queue] = None
        self._trans_q: Optional[asyncio.Queue] = None
        self._disp_q:  Optional[asyncio.Queue] = None

        self._audio_mod: Optional[AudioCaptureModule]       = None
        self._stt_mod:   Optional[DeepgramSTTEngine]        = None
        self._group_mod: Optional[SentenceGrouperModule]    = None
        self._batch_mod: Optional[TranslationBatcherModule] = None
        self._trans_mod  = None   # GroqTranslatorModule | GeminiTranslatorModule
        self._fmt_mod:   Optional[SubtitleFormatterModule]  = None

    # ── Public API ────────────────────────────────────────────

    def start(self) -> None:
        if self.state not in (PipelineState.IDLE, PipelineState.ERROR):
            log.warning("Pipeline already running.")
            return

        self.state   = PipelineState.STARTING
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

        # ── Queues ───────────────────────────────────────────
        self._audio_q = asyncio.Queue(maxsize=pcfg.audio_queue_maxsize)
        self._stt_q   = asyncio.Queue(maxsize=pcfg.stt_queue_maxsize)
        self._group_q = asyncio.Queue(maxsize=pcfg.stt_queue_maxsize)
        self._batch_q = asyncio.Queue(maxsize=pcfg.translation_queue_maxsize)
        self._trans_q = asyncio.Queue(maxsize=pcfg.translation_queue_maxsize)
        self._disp_q  = asyncio.Queue(maxsize=pcfg.subtitle_queue_maxsize)

        # ── Batcher config from active provider ──────────────
        provider = cfg.active_provider.lower()
        if provider == "groq":
            prov_cfg = cfg.groq
        else:
            prov_cfg = cfg.gemini

        # Build a minimal TranslationConfig-compatible shim for batcher
        # (batcher only needs debounce_s and max_batch_words)
        class _BatcherShim:
            debounce_s      = prov_cfg.debounce_s
            max_batch_words = prov_cfg.max_batch_words

        # ── Modules ──────────────────────────────────────────
        self._audio_mod = AudioCaptureModule(cfg.audio)
        self._stt_mod   = DeepgramSTTEngine(cfg.stt)

        gcfg = cfg.grouper
        self._group_mod = SentenceGrouperModule(
            max_wait_s=gcfg.max_wait_s,
            gap_threshold_s=gcfg.gap_threshold_s,
            max_words=gcfg.max_words,
            flush_on_gap=gcfg.flush_on_gap,
        )

        self._batch_mod = TranslationBatcherModule(_BatcherShim())
        self._trans_mod = TranslatorFactory.create(cfg)
        self._fmt_mod   = SubtitleFormatterModule(cfg.subtitle)

        # ── Validate APIs ────────────────────────────────────
        self._status_cb("Deepgram", "Checking …")
        try:
            self._stt_mod.validate()
            self._status_cb("Deepgram", "Ready ✔")
        except Exception as exc:
            log.error(f"Deepgram setup failed: {exc}")
            self._status_cb("Deepgram", f"Error: {exc}")
            self.state = PipelineState.ERROR
            return

        provider_label = provider.capitalize()
        self._status_cb(provider_label, "Checking …")
        try:
            TranslatorFactory.validate(cfg)
            self._status_cb(provider_label, "Ready ✔")
        except Exception as exc:
            log.error(f"{provider_label} setup failed: {exc}")
            self._status_cb(provider_label, f"Error: {exc}")
            self.state = PipelineState.ERROR
            return

        # ── Start audio ──────────────────────────────────────
        self._status_cb("AudioCapture", "Starting …")
        await self._audio_mod.start(self._audio_q)
        self._status_cb("AudioCapture", "Running ✔")

        self.state = PipelineState.RUNNING
        self._status_cb("Pipeline", "Running ✔")

        # ── Tasks ────────────────────────────────────────────
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
                name=provider_label,
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
