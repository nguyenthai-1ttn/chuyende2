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
    sample_rate: int = 16_000          # Hz; Whisper requires 16 kHz
    channels: int = 1                  # Mono
    chunk_duration_ms: int = 30        # VAD frame size (10 / 20 / 30 ms allowed)
    input_device: Optional[int] = None # None → system default; set index for specific mic

    # Input source:  "mic" | "system" | "file" | "stream"
    source_type: Literal["mic", "system", "file", "stream"] = "mic"
    source_path: str = ""              # Path or URL used when source_type ∈ {file, stream}

    # --- Intelligent Capture (DeepFilterNet noise filter) ---
    noise_filter_enabled: bool = False
    df_post_filter: bool = True        # Extra clarity pass in DeepFilterNet

    # VAD (webrtcvad): aggressiveness 0–3 (3 = most aggressive, rejects more non-speech)
    vad_aggressiveness: int = 2
    vad_padding_ms: int = 300          # Silence padding kept around speech segments
    vad_min_speech_ms: int = 150       # Ignore bursts shorter than this
    vad_max_silence_ms: int = 600   # End segment after N ms of silence


# ─────────────────────────────────────────────
#  SPEECH-TO-TEXT CONFIG
# ─────────────────────────────────────────────
@dataclass
class STTConfig:
    """Faster-Whisper settings tuned for ultra-low latency."""

    # Model: "distil-small.en" (fastest) or "base" (more accurate)
    model_name: str = "distil-small.en"
    device: str = "cpu"                # "cuda" if GPU available
    compute_type: str = "int8"         # int8 → fastest CPU; float16 → GPU

    language: str = "en"
    beam_size: int = 3                 # Lower = faster; 1 = greedy
    best_of: int = 1
    word_timestamps: bool = True       # Required for subtitle sync
    condition_on_previous_text: bool = False  # Avoid hallucination loops

    # Built-in Whisper VAD filter (second guard after webrtcvad)
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500

    # Confidence gate — discard low-confidence segments
    min_confidence: float = -1.0       # log-prob; -1.0 lets almost everything through


# ─────────────────────────────────────────────
#  TRANSLATION (LLM) CONFIG
# ─────────────────────────────────────────────
@dataclass
class TranslationConfig:
    """Ollama / DeepSeek-V2 translation settings."""

    ollama_base_url: str = "http://localhost:11434"
    model: str = "deepseek-v2:latest"  # or "deepseek-r1:latest"
    timeout_s: int = 60                 # Hard deadline per request

    target_language: str = "Vietnamese"
    max_lines: int = 2
    max_chars_per_line: int = 40

    # System prompt injected into every request
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

    # Ollama generation params
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 100
    translation_max_retries: int = 2
    translation_retry_delay_s: float = 0.5


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

    audio_queue_maxsize: int = 50      # Frames buffered before back-pressure
    stt_queue_maxsize: int = 20
    translation_queue_maxsize: int = 20
    subtitle_queue_maxsize: int = 30

    # If a module stalls longer than this, emit a warning
    module_timeout_s: float = 15.0

    # Max retries for transient Ollama/network errors
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
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Singleton-style default used across the app
DEFAULT_CONFIG = AppConfig()
