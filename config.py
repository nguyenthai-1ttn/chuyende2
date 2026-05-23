"""
config.py — Centralized configuration for the AI Subtitle Agent system.
All module settings live here for easy tuning without touching core logic.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


# ─────────────────────────────────────────────
#  AUDIO CAPTURE CONFIG
# ─────────────────────────────────────────────
@dataclass
class AudioConfig:
    """Technical & intelligent audio capture settings."""

    # --- Technical Capture (FFmpeg / sounddevice) ---
    sample_rate: int = 16_000          # Hz; Deepgram linear16 expects 16 kHz
    channels: int = 1                  # Mono
    chunk_duration_ms: int = 30        # PCM frame size (10 / 20 / 30 ms)
    input_device: Optional[int] = None # None → system default; set index for specific mic

    # Input source:  "mic" | "system" | "file" | "stream"
    source_type: Literal["mic", "system", "file", "stream"] = "mic"
    source_path: str = ""              # Path or URL used when source_type ∈ {file, stream}


# ─────────────────────────────────────────────
#  SPEECH-TO-TEXT CONFIG (Deepgram live)
# ─────────────────────────────────────────────
@dataclass
class STTConfig:
    """Deepgram live WebSocket STT — continuous PCM stream."""

    api_key: str = ""
    model: str = "nova-2"
    language: str = "en"
    sample_rate: int = 16_000

    interim_results: bool = True
    punctuate: bool = True
    smart_format: bool = True
    endpointing_ms: int = 600          # Deepgram utterance endpointing (ms)
    require_speech_final: bool = False  # Let is_final drive segments instead

    # Network
    connect_timeout_s: float = 10.0
    keepalive_interval_s: float = 8.0


# ─────────────────────────────────────────────
#  TRANSLATION (LLM) CONFIG
# ─────────────────────────────────────────────
@dataclass
class TranslationConfig:
    """Ollama local translation (Qwen2.5-1.5B-Instruct quant on GPU)."""

    ollama_base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:1.5b-instruct-q4_K_M"
    timeout_s: int = 60

    target_language: str = "Vietnamese"
    max_lines: int = 2
    max_chars_per_line: int = 40

    system_prompt: str = (
        "You are a Vietnamese subtitle translator. "
        "Translate the English text to Vietnamese ONLY. "
        "STRICT RULES:\n"
        "1. Output ONLY Vietnamese text. No Chinese, no English, no other language.\n"
        "2. Maximum 2 lines, each line maximum 40 characters.\n"
        "3. If the input is unclear or noisy, output your best guess in Vietnamese.\n"
        "4. NO punctuation at line breaks. NO hyphens between words.\n"
        "5. Output ONLY the subtitle text, nothing else."
    )

    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 100
    keep_alive: int = -1               # Keep model loaded on GPU (-1 = indefinite)
    preload_on_start: bool = True

    translation_max_retries: int = 2
    translation_retry_delay_s: float = 0.5

    # Batching before Ollama (merge utterances; no cloud RPM limit)
    debounce_s: float = 0.8
    min_interval_s: float = 0.0        # Unused for local; kept for batcher compat
    max_batch_words: int = 80


# ─────────────────────────────────────────────
#  SENTENCE GROUPER CONFIG
# ─────────────────────────────────────────────
@dataclass
class GrouperConfig:
    max_wait_s: float = 4.0
    gap_threshold_s: float = 1.8
    max_words: int = 50
    flush_on_gap: bool = False         # Off for live Deepgram (gaps ≠ sentence end)


# ─────────────────────────────────────────────
#  SUBTITLE FORMATTING / DISPLAY CONFIG
# ─────────────────────────────────────────────
@dataclass
class SubtitleConfig:
    """Visual styling and timing for the subtitle overlay."""

    # --- Timing ---
    min_display_ms: int = 1_200        # Never flash shorter than this
    max_display_ms: int = 6_000        # Auto-clear after this
    fade_in_ms: int = 150
    fade_out_ms: int = 300
    char_reading_speed: int = 20       # ms per character (reading time estimate)

    # --- Typography ---
    font_family: str = "Arial"         # Fallback; overridden in UI if richer font present
    font_size: int = 26
    font_weight: str = "bold"

    # --- Colors ---
    text_color: str = "#FFFFFF"
    text_shadow_color: str = "#000000"
    bg_color: str = "#1A1A1A"
    bg_opacity: float = 0.75           # 0.0 = fully transparent, 1.0 = opaque

    # --- Layout ---
    position: Literal["bottom", "top"] = "bottom"
    margin_bottom: int = 60            # px from screen edge
    max_width_ratio: float = 0.80      # Subtitle box = 80 % of screen width
    padding_x: int = 20
    padding_y: int = 10
    line_spacing: int = 6
    max_chars_per_line: int = 40 


# ─────────────────────────────────────────────
#  PIPELINE CONFIG
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Async queue sizes and fault-tolerance settings."""

    audio_queue_maxsize: int = 200     # PCM frames buffered before back-pressure
    stt_queue_maxsize: int = 20
    translation_queue_maxsize: int = 20
    subtitle_queue_maxsize: int = 30

    # If a module stalls longer than this, emit a warning
    module_timeout_s: float = 15.0

    # Max retries for transient API/network errors
    translation_max_retries: int = 2
    translation_retry_delay_s: float = 0.5

    debug: bool = False                # Verbose pipeline logging


# ─────────────────────────────────────────────
#  TOP-LEVEL APP CONFIG
# ─────────────────────────────────────────────
@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    grouper: GrouperConfig = field(default_factory=GrouperConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Singleton-style default used across the app
DEFAULT_CONFIG = AppConfig()
