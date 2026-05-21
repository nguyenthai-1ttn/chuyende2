"""
utils/audio_utils.py — Shared audio helpers used across modules.
"""

from __future__ import annotations

import array
import collections
import struct
from typing import Iterator

import numpy as np


# ─────────────────────────────────────────────
#  Byte ↔ NumPy helpers
# ─────────────────────────────────────────────

def bytes_to_float32(raw: bytes, sample_rate: int = 16_000) -> np.ndarray:
    """Convert PCM-16 LE bytes → float32 array in [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples / 32_768.0


def float32_to_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 array in [-1, 1] → PCM-16 LE bytes."""
    pcm = (audio * 32_768.0).clip(-32_768, 32_767).astype(np.int16)
    return pcm.tobytes()


def chunk_audio(audio: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield equal-sized chunks from a 1-D audio array."""
    for start in range(0, len(audio), chunk_size):
        yield audio[start : start + chunk_size]


# ─────────────────────────────────────────────
#  RMS energy gate (fast silence detection)
# ─────────────────────────────────────────────

def rms_energy(audio: np.ndarray) -> float:
    """Return RMS energy (0–1) of float32 audio."""
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0


def is_silent(audio: np.ndarray, threshold: float = 0.005) -> bool:
    return rms_energy(audio) < threshold


# ─────────────────────────────────────────────
#  Ring buffer for streaming audio accumulation
# ─────────────────────────────────────────────

class RingBuffer:
    """Fixed-size ring buffer for float32 audio samples."""

    def __init__(self, capacity_samples: int):
        self._buf: collections.deque[float] = collections.deque(
            maxlen=capacity_samples
        )
        self.capacity = capacity_samples

    def write(self, data: np.ndarray) -> None:
        self._buf.extend(data.tolist())

    def read_all(self) -> np.ndarray:
        return np.array(list(self._buf), dtype=np.float32)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────
#  VAD frame formatter
# ─────────────────────────────────────────────

def frame_generator(
    frame_duration_ms: int,
    audio: bytes,
    sample_rate: int,
) -> Iterator[bytes]:
    """Yield fixed-size PCM-16 frames for webrtcvad."""
    n = int(sample_rate * (frame_duration_ms / 1_000.0) * 2)  # *2 for 16-bit
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset : offset + n]
        offset += n


# ─────────────────────────────────────────────
#  Normalise loudness
# ─────────────────────────────────────────────

def normalise(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalise audio to a target RMS level."""
    cur_rms = rms_energy(audio)
    if cur_rms < 1e-6:
        return audio
    return (audio * (target_rms / cur_rms)).clip(-1.0, 1.0)
