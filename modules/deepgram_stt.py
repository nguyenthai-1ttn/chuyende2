"""
modules/deepgram_stt.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 2 — Speech-to-Text (Deepgram Live)

Consumes a continuous stream of PCM-16 LE mono frames and opens a
Deepgram listen/v1 WebSocket. Emits TranscriptSegment on final results.

Input:  asyncio.Queue[bytes]   (PCM-16 LE chunks, 16 kHz)
Output: asyncio.Queue[TranscriptSegment]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional

from config import STTConfig
from modules.stt_engine import TranscriptSegment, WordTimestamp
from utils.logger import get_logger

log = get_logger("DeepgramSTT")

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
    from deepgram.listen.v1.types import ListenV1Results
except ImportError as exc:
    AsyncDeepgramClient = None  # type: ignore
    EventType = None  # type: ignore
    ListenV1Results = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class DeepgramSTTEngine:
    """
    Live streaming STT via Deepgram WebSocket API.

    Usage:
        engine = DeepgramSTTEngine(cfg)
        await engine.run(pcm_queue, stt_queue, stop_event, transcript_cb)
    """

    def __init__(self, cfg: STTConfig):
        self._cfg = cfg
        self._segment_counter = 0
        self._pipeline_start: float = 0.0
        self._stream_time_offset: float = 0.0
        self._last_final_text: str = ""

    # ── Validation ────────────────────────────────────────────

    def validate(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "deepgram-sdk is not installed. Run: pip install deepgram-sdk"
            ) from _IMPORT_ERROR
        if not self._cfg.api_key.strip():
            raise ValueError("Deepgram API key is required.")

    # ── Main async loop ───────────────────────────────────────

    async def run(
        self,
        pcm_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        transcript_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.validate()
        cfg = self._cfg
        self._pipeline_start = time.monotonic()
        self._segment_counter = 0
        self._last_final_text = ""

        client = AsyncDeepgramClient(api_key=cfg.api_key.strip())
        log.info(
            f"Connecting Deepgram live (model={cfg.model!r}, lang={cfg.language!r}) …"
        )

        try:
            async with client.listen.v1.connect(
                model=cfg.model,
                encoding="linear16",
                sample_rate=str(cfg.sample_rate),
                language=cfg.language,
                interim_results="true" if cfg.interim_results else "false",
                punctuate="true" if cfg.punctuate else "false",
                smart_format="true" if cfg.smart_format else "false",
                endpointing=str(cfg.endpointing_ms),
            ) as dg_socket:
                listen_task = asyncio.create_task(
                    dg_socket.start_listening(), name="DeepgramListen"
                )

                async def on_message(message) -> None:
                    await self._handle_message(
                        message, output_queue, transcript_cb
                    )

                async def on_error(exc) -> None:
                    log.error(f"Deepgram WebSocket error: {exc}")

                dg_socket.on(EventType.MESSAGE, on_message)
                dg_socket.on(EventType.ERROR, on_error)

                sender_task = asyncio.create_task(
                    self._send_pcm_loop(dg_socket, pcm_queue, stop_event),
                    name="DeepgramSender",
                )

                await stop_event.wait()

                sender_task.cancel()
                try:
                    await dg_socket.send_close_stream()
                except Exception as exc:
                    log.debug(f"CloseStream: {exc}")

                listen_task.cancel()
                await asyncio.gather(sender_task, listen_task, return_exceptions=True)

        except Exception as exc:
            log.error(f"Deepgram session failed: {type(exc).__name__}: {exc}")
            raise

        log.info("DeepgramSTT stopped.")

    # ── PCM sender ────────────────────────────────────────────

    async def _send_pcm_loop(
        self,
        dg_socket,
        pcm_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        cfg = self._cfg
        last_keepalive = time.monotonic()
        first_chunk = True

        while not stop_event.is_set():
            try:
                chunk: bytes = await asyncio.wait_for(
                    pcm_queue.get(), timeout=0.25
                )
            except asyncio.TimeoutError:
                chunk = None
            else:
                if first_chunk:
                    self._stream_time_offset = time.monotonic() - self._pipeline_start
                    first_chunk = False
                await dg_socket.send_media(chunk)
                pcm_queue.task_done()

            now = time.monotonic()
            if now - last_keepalive >= cfg.keepalive_interval_s:
                try:
                    await dg_socket.send_keep_alive()
                except Exception as exc:
                    log.debug(f"KeepAlive: {exc}")
                last_keepalive = now

    # ── Result handler ────────────────────────────────────────

    async def _handle_message(
        self,
        message,
        output_queue: asyncio.Queue,
        transcript_cb: Optional[Callable[[str], None]],
    ) -> None:
        if ListenV1Results is None or not isinstance(message, ListenV1Results):
            return

        channel = message.channel
        if not channel.alternatives:
            return

        alt = channel.alternatives[0]
        text = (alt.transcript or "").strip()
        if not text:
            return

        is_final    = bool(message.is_final)
        speech_final = getattr(message, "speech_final", None)

        # ── DEBUG: log mọi message để thấy luồng dữ liệu ──────────
        log.info(
            f"DG msg | is_final={is_final} | speech_final={speech_final} "
            f"| require_sf={self._cfg.require_speech_final} | text='{text[:50]}'"
        )

        if transcript_cb and not is_final:
            try:
                transcript_cb(text)
            except Exception as exc:
                log.debug(f"transcript_cb error: {exc}")

        if not is_final:
            return

        if self._cfg.require_speech_final:
            if speech_final is False:
                log.info(f"Dropped: require_speech_final=True but speech_final=False")
                return

        if self._is_duplicate_final(text):
            log.info(f"Dropped duplicate: '{text[:50]}'")
            return

        if transcript_cb:
            try:
                transcript_cb(text)
            except Exception as exc:
                log.debug(f"transcript_cb error: {exc}")

        words = self._map_words(alt.words)
        confidences = [w.probability for w in words if w.probability > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

        abs_start = self._stream_time_offset + float(message.start)
        abs_end   = abs_start + float(message.duration)

        self._segment_counter += 1
        segment = TranscriptSegment(
            text=text,
            start=abs_start,
            end=abs_end,
            words=words,
            confidence=avg_conf,
            segment_id=self._segment_counter,
        )

        await output_queue.put(segment)
        log.info(
            f"[seg#{segment.segment_id}] PUSHED to queue "
            f"{segment.start:.2f}s–{segment.end:.2f}s | \"{text[:60]}\""
        )

    def _is_duplicate_final(self, text: str) -> bool:
        """Skip repeated or regressive Deepgram finals."""
        norm = text.strip().lower()
        if not norm:
            return True
        prev = self._last_final_text.strip().lower()
        if prev and norm == prev:
            return True
        if prev and norm.startswith(prev) and len(norm) <= len(prev):
            return True
        self._last_final_text = text
        return False

    @staticmethod
    def _map_words(raw_words) -> List[WordTimestamp]:
        if not raw_words:
            return []
        out: List[WordTimestamp] = []
        for w in raw_words:
            word = (getattr(w, "punctuated_word", None) or getattr(w, "word", "") or "").strip()
            if not word:
                continue
            out.append(
                WordTimestamp(
                    word=word,
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(getattr(w, "confidence", 1.0) or 1.0),
                )
            )
        return out
