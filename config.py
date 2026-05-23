"""
config.py — Centralized configuration for the AI Subtitle Agent system.
All module settings live here for easy tuning without touching core logic.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List


# ─────────────────────────────────────────────
#  AUDIO CAPTURE CONFIG
# ─────────────────────────────────────────────
@dataclass
class AudioConfig:
    """Technical & intelligent audio capture settings."""

    sample_rate: int = 16_000
    channels: int = 1
    chunk_duration_ms: int = 30
    input_device: Optional[int] = None

    source_type: Literal["mic", "system", "file", "stream"] = "mic"
    source_path: str = ""


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
    endpointing_ms: int = 600
    require_speech_final: bool = False

    connect_timeout_s: float = 10.0
    keepalive_interval_s: float = 8.0


# ─────────────────────────────────────────────
#  GROQ TRANSLATION CONFIG
# ─────────────────────────────────────────────
@dataclass
class GroqConfig:
    """Groq cloud LLM — 30 RPM free tier."""

    api_key: str = ""
    model: str = "llama-3.1-8b-instant"
    available_models: List[str] = field(default_factory=lambda: [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ])

    # Rate limiting (30 RPM → safe at ~2s interval)
    min_interval_s: float = 2.0
    debounce_s: float = 0.8
    max_batch_words: int = 60
    max_retries: int = 3

    # Generation
    temperature: float = 0.1
    max_tokens: int = 150
    timeout_s: int = 30

    # Subtitle format
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
    max_lines: int = 2
    max_chars_per_line: int = 40


# ─────────────────────────────────────────────
#  GEMINI TRANSLATION CONFIG
# ─────────────────────────────────────────────
@dataclass
class GeminiConfig:
    """Google Gemini cloud LLM — 15 RPM free tier."""

    api_key: str = ""
    model: str = "gemini-2.0-flash"
    available_models: List[str] = field(default_factory=lambda: [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ])

    # Rate limiting (15 RPM → safe at ~4s interval)
    min_interval_s: float = 4.0
    debounce_s: float = 1.5
    max_batch_words: int = 80
    max_retries: int = 3

    # Generation
    temperature: float = 0.1
    max_tokens: int = 150
    timeout_s: int = 30

    # Subtitle format
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
    max_lines: int = 2
    max_chars_per_line: int = 40


# ─────────────────────────────────────────────
#  SENTENCE GROUPER CONFIG
# ─────────────────────────────────────────────
@dataclass
class GrouperConfig:
    max_wait_s: float = 4.0
    gap_threshold_s: float = 1.8
    max_words: int = 50
    flush_on_gap: bool = False


# ─────────────────────────────────────────────
#  SUBTITLE FORMATTING / DISPLAY CONFIG
# ─────────────────────────────────────────────
@dataclass
class SubtitleConfig:
    """Visual styling and timing for the subtitle overlay."""

    min_display_ms: int = 1_200
    max_display_ms: int = 6_000
    fade_in_ms: int = 150
    fade_out_ms: int = 300
    char_reading_speed: int = 20

    font_family: str = "Arial"
    font_size: int = 26
    font_weight: str = "bold"

    text_color: str = "#FFFFFF"
    text_shadow_color: str = "#000000"
    bg_color: str = "#1A1A1A"
    bg_opacity: float = 0.75

    position: Literal["bottom", "top"] = "bottom"
    margin_bottom: int = 60
    max_width_ratio: float = 0.80
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

    audio_queue_maxsize: int = 200
    stt_queue_maxsize: int = 20
    translation_queue_maxsize: int = 20
    subtitle_queue_maxsize: int = 30
    module_timeout_s: float = 15.0
    debug: bool = False


# ─────────────────────────────────────────────
#  TOP-LEVEL APP CONFIG
# ─────────────────────────────────────────────
@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    grouper: GrouperConfig = field(default_factory=GrouperConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Which LLM provider is active: "groq" | "gemini"
    active_provider: str = "groq"


DEFAULT_CONFIG = AppConfig()
