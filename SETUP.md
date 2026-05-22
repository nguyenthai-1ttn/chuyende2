# 🎬 AI Subtitle Agent — Setup & Architecture Guide
## Real-Time English → Vietnamese Subtitle System

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI SUBTITLE AGENT                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MODULE 1 — AUDIO CAPTURE                   │   │
│  │                                                         │   │
│  │  [Source]  →  FFmpeg (Technical Capture)               │   │
│  │               ↓                                         │   │
│  │            DeepFilterNet (Noise Reduction)              │   │
│  │               ↓                                         │   │
│  │            webrtcvad (Speech Detection / VAD)           │   │
│  │               ↓                                         │   │
│  │         [audio_q: asyncio.Queue[np.ndarray]]            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MODULE 2 — SPEECH-TO-TEXT                  │   │
│  │                                                         │   │
│  │       Faster-Whisper (distil-small.en / base)           │   │
│  │       Word-level timestamps · VAD filter                │   │
│  │               ↓                                         │   │
│  │        [stt_q: asyncio.Queue[TranscriptSegment]]        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          MODULE 3 — LLM TRANSLATION (AGENT)             │   │
│  │                                                         │   │
│  │       Ollama → Qwen2.5-1.5B-Instruct (local GPU, Q4)   │   │
│  │       System prompt: "max 2 lines, 40 chars each"       │   │
│  │       temperature=0.1, num_predict=100                  │   │
│  │               ↓                                         │   │
│  │       [trans_q: asyncio.Queue[TranslatedSubtitle]]      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          MODULE 4 — SUBTITLE FORMATTING                 │   │
│  │                                                         │   │
│  │   Technical Alignment: Whisper timestamps + stable-ts   │   │
│  │   Agentic Formatting: LLM line-breaks (already done)    │   │
│  │   Display timing heuristic: reading speed × char count  │   │
│  │               ↓                                         │   │
│  │        [disp_q: asyncio.Queue[DisplaySubtitle]]         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TKINTER UI (main thread)                   │   │
│  │                                                         │   │
│  │   ControlPanel (configure, status, logs)                │   │
│  │   SubtitleOverlay (borderless, always-on-top, fade)     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Thread model:
  Main thread   → Tkinter event loop (UI)
  Daemon thread → asyncio event loop (entire pipeline)
  Communication → threading.Queue (subtitle_queue)
```

---

## Prerequisites

### 1. Python 3.9+
```bash
python --version   # must be 3.9 or higher
```

### 2. FFmpeg
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# Add ffmpeg to PATH
ffmpeg -version   # verify
```

### 3. Ollama + Qwen2.5 (GPU, quantized)
```bash
# Install Ollama — https://ollama.com/download
# Windows: install then run `ollama serve` in a terminal

# Recommended: 1.5B instruct, Q4 quant (~1 GB VRAM, fast on GPU)
ollama pull qwen2.5:1.5b-instruct-q4_K_M

# Alternatives:
# ollama pull qwen2.5:1.5b-instruct
# ollama pull qwen2.5:1.5b

ollama serve
```

Quantization is selected by the **model tag** when you `pull` (GGUF Q4_K_M, etc.) — not in app code.

### 4. Deepgram API (STT cloud)
Get an API key at https://console.deepgram.com/ and enter it in the app UI.

### 5. Python packages

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Ensure Ollama is running
ollama serve &

# 2. Launch the agent
python main.py
```

The control panel opens. Configure your source, then click **▶ Start**.

---

## Configuration Guide

### Audio Source Options

| Source   | Description                              | Requirement           |
|----------|------------------------------------------|-----------------------|
| `mic`    | System microphone via sounddevice        | sounddevice installed |
| `system` | System audio loopback (what's playing)   | PulseAudio / WASAPI   |
| `file`   | Video or audio file                      | FFmpeg on PATH        |
| `stream` | RTSP / HLS / HTTP stream URL             | FFmpeg on PATH        |

### Model Selection

| Whisper Model    | Speed  | Accuracy | VRAM  | Use Case               |
|------------------|--------|----------|-------|------------------------|
| distil-small.en  | ★★★★★ | ★★★★☆   | 250MB | **Recommended default**|
| base             | ★★★★☆ | ★★★★☆   | 150MB | CPU-only low memory    |
| base.en          | ★★★★☆ | ★★★★☆   | 150MB | English-only, faster   |
| small            | ★★★☆☆ | ★★★★★   | 500MB | Higher accuracy        |
| tiny             | ★★★★★ | ★★★☆☆   | 75MB  | Raspberry Pi / weak CPU|

### Ollama / DeepSeek Models

| Model                | Size  | Speed  | Quality | Notes                     |
|----------------------|-------|--------|---------|---------------------------|
| deepseek-v2:latest   | ~8GB  | ★★★★☆ | ★★★★★  | Best translation quality  |
| deepseek-r1:1.5b     | ~1GB  | ★★★★★ | ★★★☆☆  | Ultra-fast, lighter        |
| deepseek-r1:7b       | ~4GB  | ★★★★☆ | ★★★★☆  | Good balance              |
| qwen2.5:3b           | ~2GB  | ★★★★★ | ★★★★☆  | Excellent EN→VI           |

---

## Latency Tuning

### Target: < 2 seconds end-to-end

| Stage           | Typical Latency | How to Reduce                              |
|-----------------|-----------------|--------------------------------------------|
| VAD detection   | 30–300 ms       | Lower `vad_max_silence_ms` in config       |
| STT (Whisper)   | 200–800 ms      | Use `distil-small.en`, device=`cuda`       |
| LLM translation | 300–1500 ms     | Use smaller model, CUDA for Ollama         |
| Display         | < 30 ms         | UI polled every 30 ms                      |

### CPU-only optimisation
```python
# config.py
stt.model_name = "distil-small.en"  # fastest
stt.compute_type = "int8"           # quantized
stt.beam_size = 1                   # greedy decode
translation.num_predict = 80        # cap output tokens
translation.timeout_s = 5           # fail fast
```

### GPU optimisation
```python
stt.device = "cuda"
stt.compute_type = "float16"
# Ollama automatically uses GPU if available
```

---

## Project Structure

```
subtitle_agent/
├── main.py                    # Tkinter UI (ControlPanel + SubtitleOverlay)
├── pipeline.py                # Async pipeline orchestrator
├── config.py                  # All configuration dataclasses
├── requirements.txt
├── SETUP.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── audio_capture.py       # Module 1: FFmpeg + DeepFilterNet + VAD
│   ├── stt_engine.py          # Module 2: Faster-Whisper
│   ├── translator.py          # Module 3: Ollama/DeepSeek LLM
│   └── subtitle_formatter.py  # Module 4: Timing + display prep
└── utils/
    ├── __init__.py
    ├── logger.py              # Coloured console logger
    └── audio_utils.py         # PCM helpers, ring buffer, VAD frames
```

---

## Extending the System

### Add a new language
In `config.py`, change:
```python
translation.target_language = "Japanese"
translation.system_prompt = "Translate English to Japanese. Max 2 lines, 20 characters each."
```

### Custom subtitle prompt
```python
translation.system_prompt = (
    "Dịch tiếng Anh sang tiếng Việt. "
    "Tối đa 2 dòng, mỗi dòng không quá 40 ký tự. "
    "Giữ nguyên tên riêng. "
    "Chỉ trả về phụ đề, không giải thích."
)
```

### Connect to a live stream
```python
cfg.audio.source_type = "stream"
cfg.audio.source_path = "rtsp://192.168.1.100:8554/stream"
# or HLS:
cfg.audio.source_path = "https://example.com/live/stream.m3u8"
```

---

## Troubleshooting

| Problem                      | Fix                                                    |
|------------------------------|--------------------------------------------------------|
| `FFmpeg not found`           | Install FFmpeg, add to PATH                            |
| `Ollama not reachable`       | Run `ollama serve` in a separate terminal              |
| Model not found in Ollama    | `ollama pull qwen2.5:1.5b-instruct-q4_K_M`             |
| Slow first translation     | Use **Preload now** or enable **On Start** in UI       |
| Subtitles appear too fast    | Increase `subtitle.char_reading_speed` in config       |
| High latency                 | Switch to smaller model, reduce `vad_max_silence_ms`   |
| No audio on `system` source  | Install PulseAudio (Linux) or VB-Cable (Windows)       |
| DeepFilterNet import error   | `pip install deepfilternet`                            |
| webrtcvad import error       | `pip install webrtcvad-wheels`                         |
| CUDA out of memory           | Use `compute_type="int8"` or smaller model             |

---

## License
MIT — Free for personal and commercial use.
