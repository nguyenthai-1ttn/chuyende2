"""
modules/audio_capture.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 1 — Audio Signal Acquisition

Captures audio from mic / system / file / stream and streams raw
PCM-16 LE frames to the downstream STT module (Deepgram live).

Output: asyncio.Queue[bytes]  (PCM-16 LE, 16 kHz, mono chunks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from typing import Optional

from config import AudioConfig
from utils.logger import get_logger

log = get_logger("AudioCapture")


def _build_ffmpeg_cmd(cfg: AudioConfig) -> list[str]:
    """
    Build an FFmpeg command that reads from the configured source and
    outputs raw PCM-16 LE, mono, 16 kHz on stdout.
    """
    sr = cfg.sample_rate
    ch = cfg.channels

    if cfg.source_type == "mic":
        return []
    elif cfg.source_type == "system":
        if sys.platform.startswith("linux"):
            input_args = ["-f", "pulse", "-i", "default"]
        elif sys.platform == "darwin":
            input_args = ["-f", "avfoundation", "-i", ":0"]
        else:
            device = cfg.source_path or "audio=Stereo Mix (Realtek(R) Audio)"
            if device and not device.lower().startswith("audio="):
                device = f"audio={device}"
            input_args = ["-f", "dshow", "-i", device]
    elif cfg.source_type in ("file", "stream"):
        input_args = ["-i", cfg.source_path]
    else:
        raise ValueError(f"Unknown source_type: {cfg.source_type!r}")

    return [
        "ffmpeg", "-loglevel", "quiet",
        *input_args,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", str(ch),
        "-f", "s16le",
        "pipe:1",
    ]


class AudioCaptureModule:
    """
    Async-compatible audio capture — continuous PCM frame streaming.

    Usage:
        module = AudioCaptureModule(cfg)
        await module.start(output_queue)
        await module.stop()
    """

    def __init__(self, cfg: AudioConfig):
        self._cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._out_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self, output_queue: asyncio.Queue) -> None:
        self._out_queue = output_queue
        self._loop = asyncio.get_event_loop()
        self._running = True

        if self._cfg.source_type == "mic":
            self._thread = threading.Thread(
                target=self._capture_sounddevice, daemon=True
            )
        else:
            self._thread = threading.Thread(
                target=self._capture_ffmpeg, daemon=True
            )

        self._thread.start()
        log.info(f"AudioCapture started (source={self._cfg.source_type!r}, stream=pcm) ✔")

    async def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("AudioCapture stopped.")

    def _capture_sounddevice(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            log.error("sounddevice not installed. Install with: pip install sounddevice")
            return

        sr = self._cfg.sample_rate
        frame_samples = int(sr * self._cfg.chunk_duration_ms / 1_000)

        def callback(indata, frames, time_info, status):
            if not self._running:
                raise sd.CallbackStop()
            if status:
                log.debug(f"sounddevice status: {status}")
            raw = (indata[:, 0] * 32_768).astype("int16").tobytes()
            self._emit(raw)

        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            device=self._cfg.input_device,
            callback=callback,
        ):
            log.info("sounddevice stream open.")
            while self._running:
                time.sleep(0.05)

    def _capture_ffmpeg(self) -> None:
        cmd = _build_ffmpeg_cmd(self._cfg)
        sr = self._cfg.sample_rate
        frame_bytes = int(sr * self._cfg.chunk_duration_ms / 1_000) * 2

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.error("FFmpeg not found. Install FFmpeg and ensure it is on PATH.")
            return

        log.info(f"FFmpeg process started (cmd={' '.join(cmd)[:80]}…)")

        try:
            while self._running:
                raw = proc.stdout.read(frame_bytes)
                if not raw:
                    break
                self._emit(raw)
        finally:
            proc.terminate()
            log.info("FFmpeg process terminated.")

    def _emit(self, pcm_bytes: bytes) -> None:
        """Thread-safe enqueue of a PCM frame."""
        if not pcm_bytes or not self._loop or not self._out_queue:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._out_queue.put(pcm_bytes), self._loop
            )
        except Exception as exc:
            log.warning(f"Failed to enqueue PCM frame: {exc}")
