"""
modules/audio_capture.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 1 — Audio Signal Acquisition & Preprocessing

Two-stage design:
  • Technical Capture  — FFmpeg reads any source (mic / system audio /
                         file / RTSP stream) and delivers raw PCM-16 chunks.
  • Intelligent Capture — DeepFilterNet removes noise; webrtcvad detects
                          speech and segments the stream into utterances.

Output: asyncio.Queue[np.ndarray]  (float32, 16 kHz, mono)
        Each item is one complete speech segment ready for Whisper.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

from config import AudioConfig
from utils.audio_utils import (
    bytes_to_float32,
    float32_to_bytes,
    frame_generator,
    is_silent,
    normalise,
)
from utils.logger import get_logger

log = get_logger("AudioCapture")


# ─────────────────────────────────────────────────────────────
#  DeepFilterNet wrapper (optional — graceful fallback)
# ─────────────────────────────────────────────────────────────

class _NoiseFilter:
    """Thin wrapper around DeepFilterNet with graceful degradation."""

    def __init__(self, enabled: bool, post_filter: bool):
        self._enabled = enabled
        self._model = None
        self._df_state = None

        if not enabled:
            log.info("Noise filter disabled by config.")
            return

        try:
            from df.enhance import enhance_audio_array, init_df  # type: ignore
            self._model, self._df_state, _ = init_df(post_filter=post_filter)
            log.info("DeepFilterNet initialised ✔")
        except Exception as exc:
            log.warning(f"DeepFilterNet not available ({exc}). Running without noise filter.")
            self._enabled = False

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self._enabled or self._model is None:
            return audio
        try:
            from df.enhance import enhance_audio_array  # type: ignore
            # DeepFilterNet expects (channels, samples); we have (samples,)
            tensor = audio[np.newaxis, :]
            enhanced = enhance_audio_array(self._model, self._df_state, tensor)
            return enhanced[0]
        except Exception as exc:
            log.warning(f"DeepFilterNet processing error: {exc}")
            return audio


# ─────────────────────────────────────────────────────────────
#  VAD helper  (webrtcvad → speech segment collector)
# ─────────────────────────────────────────────────────────────

class _VADCollector:
    """
    Collects PCM frames; when a speech segment ends (enough silence),
    yields the accumulated float32 audio as a numpy array.
    """

    def __init__(self, cfg: AudioConfig):
        self._cfg = cfg
        self._vad = None
        self._frames: list[bytes] = []
        self._triggered = False
        self._ring: list[bytes] = []           # padding ring buffer
        self._num_padding = int(
            cfg.vad_padding_ms / cfg.chunk_duration_ms
        )
        self._silence_count = 0
        self._max_silence_frames = int(
            cfg.vad_max_silence_ms / cfg.chunk_duration_ms
        )

        try:
            import webrtcvad  # type: ignore
            self._vad = webrtcvad.Vad(cfg.vad_aggressiveness)
            log.info(f"webrtcvad initialised (aggressiveness={cfg.vad_aggressiveness}) ✔")
        except ImportError:
            log.warning("webrtcvad not installed. Falling back to energy-based VAD.")

    def feed(self, frame_bytes: bytes, sample_rate: int) -> Optional[np.ndarray]:
        """Feed one frame; returns a complete segment or None."""
        is_speech = self._classify(frame_bytes, sample_rate)

        if not self._triggered:
            # Pre-buffer for padding
            self._ring.append(frame_bytes)
            if len(self._ring) > self._num_padding:
                self._ring.pop(0)

            if is_speech:
                self._triggered = True
                self._frames = list(self._ring)
                self._ring = []
                self._silence_count = 0
        else:
            self._frames.append(frame_bytes)
            if not is_speech:
                self._silence_count += 1
                if self._silence_count > self._max_silence_frames:
                    # Segment complete
                    segment_bytes = b"".join(self._frames)
                    self._frames = []
                    self._triggered = False
                    self._silence_count = 0
                    return bytes_to_float32(segment_bytes, sample_rate)
            else:
                self._silence_count = 0

        return None

    def flush(self) -> Optional[np.ndarray]:
        """Call on shutdown to emit any buffered audio."""
        if self._frames:
            seg = bytes_to_float32(b"".join(self._frames))
            self._frames = []
            self._triggered = False
            return seg
        return None

    def _classify(self, frame: bytes, sample_rate: int) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, sample_rate)
            except Exception:
                pass
        # Energy fallback
        audio = bytes_to_float32(frame)
        return not is_silent(audio, threshold=0.008)


# ─────────────────────────────────────────────────────────────
#  FFmpeg source reader
# ─────────────────────────────────────────────────────────────

def _build_ffmpeg_cmd(cfg: AudioConfig) -> list[str]:
    """
    Build an FFmpeg command that reads from the configured source and
    outputs raw PCM-16 LE, mono, 16 kHz on stdout.
    """
    sr = cfg.sample_rate
    ch = cfg.channels

    # Input arguments differ by source type
    if cfg.source_type == "mic":
        # Use default audio device via sounddevice (handled separately)
        return []   # signals: use sounddevice path
    elif cfg.source_type == "system":
        # Linux: pulse; macOS: avfoundation; Windows: dshow
        if sys.platform.startswith("linux"):
            input_args = ["-f", "pulse", "-i", "default"]
        elif sys.platform == "darwin":
            input_args = ["-f", "avfoundation", "-i", ":0"]
        else:
            input_args = ["-f", "dshow", "-i", "audio=Stereo Mix (Realtek(R) Audio)"]
    elif cfg.source_type in ("file", "stream"):
        input_args = ["-i", cfg.source_path]
    else:
        raise ValueError(f"Unknown source_type: {cfg.source_type!r}")

    return [
        "ffmpeg", "-loglevel", "quiet",
        *input_args,
        "-vn",                          # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", str(ch),
        "-f", "s16le",
        "pipe:1",                       # output to stdout
    ]


# ─────────────────────────────────────────────────────────────
#  AudioCaptureModule
# ─────────────────────────────────────────────────────────────

class AudioCaptureModule:
    """
    Async-compatible audio capture module.

    Usage:
        module = AudioCaptureModule(cfg)
        await module.start(output_queue)
        # … later …
        await module.stop()
    """

    def __init__(self, cfg: AudioConfig):
        self._cfg = cfg
        self._noise_filter = _NoiseFilter(cfg.noise_filter_enabled, cfg.df_post_filter)
        self._vad_collector = _VADCollector(cfg)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._out_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public API ────────────────────────────────────────────

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
        log.info(f"AudioCapture started (source={self._cfg.source_type!r}) ✔")

    async def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("AudioCapture stopped.")

    # ── Microphone via sounddevice ────────────────────────────

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
            raw = (indata[:, 0] * 32_768).astype(np.int16).tobytes()
            self._process_frame(raw)

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

        # Flush remaining audio
        seg = self._vad_collector.flush()
        if seg is not None:
            self._emit(seg)

    # ── Any source via FFmpeg ─────────────────────────────────

    def _capture_ffmpeg(self) -> None:
        cmd = _build_ffmpeg_cmd(self._cfg)
        sr = self._cfg.sample_rate
        frame_bytes = int(sr * self._cfg.chunk_duration_ms / 1_000) * 2  # 16-bit

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.error("FFmpeg not found. Install FFmpeg and ensure it is on PATH.")
            return

        log.info(f"FFmpeg process started (cmd={' '.join(cmd)[:60]}…)")

        try:
            while self._running:
                raw = proc.stdout.read(frame_bytes)
                if not raw:
                    break
                self._process_frame(raw)
        finally:
            proc.terminate()
            seg = self._vad_collector.flush()
            if seg is not None:
                self._emit(seg)
            log.info("FFmpeg process terminated.")

    # ── Shared frame processing pipeline ─────────────────────

    def _process_frame(self, raw_bytes: bytes) -> None:
        """
        Per-frame pipeline:
          PCM bytes → DeepFilterNet noise reduction → VAD → emit segment
        """
        # 1. Convert to float32
        audio = bytes_to_float32(raw_bytes, self._cfg.sample_rate)

        # 2. Noise filter (operates on individual frames for streaming)
        audio = self._noise_filter.process(audio, self._cfg.sample_rate)

        # 3. Re-encode to bytes for webrtcvad (needs PCM-16)
        vad_bytes = float32_to_bytes(audio)

        # 4. VAD — collect frames into speech segments
        segment = self._vad_collector.feed(vad_bytes, self._cfg.sample_rate)
        if segment is not None:
            # Normalise loudness before sending downstream
            segment = normalise(segment)
            self._emit(segment)

    def _emit(self, segment: np.ndarray) -> None:
        """Thread-safe enqueue into the asyncio output queue."""
        if self._loop and self._out_queue:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._out_queue.put(segment), self._loop
                )
            except Exception as exc:
                log.warning(f"Failed to enqueue audio segment: {exc}")
