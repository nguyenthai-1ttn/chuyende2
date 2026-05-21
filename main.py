"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI Subtitle Agent — Tkinter Interface

Two windows:
  1. ControlPanel   — configure pipeline, monitor module status,
                      start / stop, log output.
  2. SubtitleOverlay — borderless, always-on-top, semi-transparent
                       window at the bottom of the screen showing
                       live Vietnamese subtitles with fade animations.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue
import sys
import threading
import time
import tkinter as tk
from typing import List, Optional, Tuple
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Optional

from config import AppConfig, AudioConfig, STTConfig, SubtitleConfig, TranslationConfig
from modules.subtitle_formatter import DisplaySubtitle
from pipeline import PipelineState, SubtitlePipeline
from utils.logger import get_logger

log = get_logger("UI")

# ─────────────────────────────────────────────────────────────
#  THEME CONSTANTS
# ─────────────────────────────────────────────────────────────

DARK    = "#0D0D0D"
PANEL   = "#151515"
CARD    = "#1C1C1C"
BORDER  = "#2A2A2A"
ACCENT  = "#00C896"   # teal-green
ACCENT2 = "#0088FF"   # blue
WARN    = "#FF6B35"
TEXT    = "#E8E8E8"
MUTED   = "#6B6B6B"
WHITE   = "#FFFFFF"
_GHOST_COLOUR = "#444444"   # dim placeholder colour
_LIVE_COLOUR  = "#FFFFFF"   # bright revealed colour
_SHADOW_COLOUR = "#000000"

FONT_MONO = ("Consolas", 10) if sys.platform == "win32" else ("Menlo", 10)
FONT_UI   = ("Segoe UI", 10) if sys.platform == "win32" else ("SF Pro Display", 10)
FONT_BIG  = ("Segoe UI Semibold", 12) if sys.platform == "win32" else ("SF Pro Display", 12)


# ─────────────────────────────────────────────────────────────
#  SUBTITLE OVERLAY WINDOW
# ─────────────────────────────────────────────────────────────

class SubtitleOverlay:
    """
    Floating subtitle window — borderless, always on top.

    Adds progressive word-reveal animation driven by the
    word_schedule field of DisplaySubtitle.
    """

    def __init__(self, root: tk.Tk, cfg):
        self._cfg  = cfg
        self._root = root
        self._win:    Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas]  = None

        # Hide / fade state
        self._hide_job: Optional[str] = None
        self._current_seg_id: int = -1
        self._target_alpha: float = 0.0
        self._fade_step_ms: int   = 16

        # Progressive reveal state
        self._word_jobs: List[str]  = []    # pending after() IDs
        self._revealed: List[str]   = []    # words shown so far
        self._full_lines: List[str] = []    # all lines of current subtitle

        # Drag support
        self._drag_x = 0
        self._drag_y = 0

        self._build()

    # ── Build window ─────────────────────────────────────────

    def _build(self) -> None:
        win = tk.Toplevel(self._root)
        win.title("Subtitles")
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg="#0D0D0D")

        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        cfg = self._cfg

        box_w = int(sw * cfg.max_width_ratio)
        box_h = (cfg.font_size + cfg.line_spacing) * 2 + cfg.padding_y * 2 + 16

        x = (sw - box_w) // 2
        y = sh - box_h - cfg.margin_bottom

        win.geometry(f"{box_w}x{box_h}+{x}+{y}")
        win.attributes("-alpha", 0.0)

        canvas = tk.Canvas(
            win, width=box_w, height=box_h,
            bg="#0D0D0D", highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        canvas.bind("<ButtonPress-1>", self._drag_start)
        canvas.bind("<B1-Motion>",     self._drag_move)

        self._win    = win
        self._canvas = canvas
        self._box_w  = box_w
        self._box_h  = box_h

    # ── Public API ────────────────────────────────────────────

    def show(self, subtitle) -> None:
        """Display a new subtitle with progressive word reveal."""
        if self._win is None:
            return

        # Cancel pending hide and word-reveal jobs
        if self._hide_job:
            self._root.after_cancel(self._hide_job)
            self._hide_job = None
        self._cancel_word_jobs()

        # Reset reveal state
        self._current_seg_id = subtitle.segment_id
        self._full_lines     = list(subtitle.lines)
        self._revealed       = []

        # Draw ghost (dim placeholder of full text)
        self._draw_ghost(subtitle.lines)

        # Fade in window
        self._target_alpha = self._cfg.bg_opacity
        self._animate_fade("in")

        # Schedule each word reveal
        self._schedule_words(subtitle.word_schedule, subtitle.segment_id)

        # Schedule hide
        duration_ms = subtitle.hide_at_ms - int(time.time() * 1_000)
        duration_ms = max(duration_ms, self._cfg.min_display_ms)
        self._hide_job = self._root.after(
            duration_ms,
            lambda: self._begin_fade_out(subtitle.segment_id),
        )

    # ── Ghost layer ───────────────────────────────────────────

    def _draw_ghost(self, lines: List[str]) -> None:
        """Draw the full subtitle in dim colour as a placeholder."""
        canvas = self._canvas
        canvas.delete("all")
        cfg = self._cfg
        w, h = self._box_w, self._box_h

        # Background
        canvas.create_rectangle(
            cfg.padding_x - 8, cfg.padding_y - 4,
            w - cfg.padding_x + 8, h - cfg.padding_y + 4,
            fill="#111111", outline="", width=0,
        )

        line_h  = cfg.font_size + cfg.line_spacing
        total_h = line_h * len(lines)
        start_y = (h - total_h) // 2 + cfg.font_size // 2

        for i, line in enumerate(lines):
            y = start_y + i * line_h
            # Shadow
            canvas.create_text(
                w // 2 + 2, y + 2,
                text=line, anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_SHADOW_COLOUR, tags=f"shadow_{i}",
            )
            # Ghost text
            canvas.create_text(
                w // 2, y,
                text=line, anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_GHOST_COLOUR, tags=f"ghost_{i}",
            )
            # Live text placeholder (empty, filled by reveal)
            canvas.create_text(
                w // 2, y,
                text="", anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_LIVE_COLOUR, tags=f"live_{i}",
            )

    # ── Progressive word reveal ───────────────────────────────

    def _schedule_words(
        self,
        word_schedule: List[Tuple[str, int]],
        seg_id: int,
    ) -> None:
        """Schedule one after() call per word."""
        if not word_schedule:
            # No schedule — reveal everything immediately
            self._reveal_all(seg_id)
            return

        now_ms = int(time.time() * 1_000)
        for idx, (word, show_at_ms) in enumerate(word_schedule):
            delay = max(0, show_at_ms - now_ms)
            jid = self._root.after(
                delay,
                lambda i=idx, w=word, sid=seg_id: self._on_word_reveal(i, w, sid),
            )
            self._word_jobs.append(jid)

    def _on_word_reveal(self, idx: int, word: str, seg_id: int) -> None:
        if seg_id != self._current_seg_id:
            return

        self._revealed.append(word)
        self._redraw_live()

    def _reveal_all(self, seg_id: int) -> None:
        """Immediately show all words (no word-level timestamps)."""
        if seg_id != self._current_seg_id:
            return
        self._revealed = [w for line in self._full_lines for w in line.split()]
        self._redraw_live()

    def _redraw_live(self) -> None:
        """
        Update the live canvas text items to show revealed words.
        Keeps ghost layer intact; live layer overwrites the matching
        portion of each line.
        """
        canvas = self._canvas
        if canvas is None:
            return

        # Re-wrap revealed words into the same line structure as the original
        revealed_text  = " ".join(self._revealed)
        max_chars      = getattr(self._cfg, "max_chars_per_line", 40)
        live_lines     = self._wrap(revealed_text, max_chars)

        for i in range(len(self._full_lines)):
            live_text = live_lines[i] if i < len(live_lines) else ""
            canvas.itemconfig(f"live_{i}", text=live_text)

    @staticmethod
    def _wrap(text: str, max_chars: int) -> List[str]:
        words   = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    # ── Fade animation ────────────────────────────────────────

    def _begin_fade_out(self, seg_id: int) -> None:
        if seg_id != self._current_seg_id:
            return
        self._target_alpha = 0.0
        self._animate_fade("out")

    def _animate_fade(self, direction: str) -> None:
        if self._win is None:
            return
        try:
            current = float(self._win.attributes("-alpha"))
        except Exception:
            current = 0.0

        step   = 0.08 if direction == "in" else 0.06
        target = self._target_alpha
        new_alpha = (
            min(current + step, target) if direction == "in"
            else max(current - step, target)
        )
        self._win.attributes("-alpha", new_alpha)

        if abs(new_alpha - target) > 0.01:
            self._root.after(
                self._fade_step_ms,
                lambda: self._animate_fade(direction),
            )

    # ── Helpers ───────────────────────────────────────────────

    def _cancel_word_jobs(self) -> None:
        for jid in self._word_jobs:
            try:
                self._root.after_cancel(jid)
            except Exception:
                pass
        self._word_jobs.clear()

    def toggle_visible(self) -> None:
        if self._win is None:
            return
        alpha = float(self._win.attributes("-alpha"))
        self._win.attributes(
            "-alpha", 0.0 if alpha > 0.0 else self._cfg.bg_opacity
        )

    # ── Drag support ──────────────────────────────────────────

    def _drag_start(self, event: tk.Event) -> None:
        self._drag_x = event.x_root - self._win.winfo_x()
        self._drag_y = event.y_root - self._win.winfo_y()

    def _drag_move(self, event: tk.Event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self._win.geometry(f"+{x}+{y}")

# ─────────────────────────────────────────────────────────────
#  STATUS INDICATOR WIDGET
# ─────────────────────────────────────────────────────────────

class StatusDot(tk.Canvas):
    COLOURS = {"idle": MUTED, "ok": ACCENT, "warn": WARN, "error": "#FF3333"}

    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=12, height=12, bg=PANEL,
                         highlightthickness=0, **kwargs)
        self._dot = self.create_oval(2, 2, 10, 10, fill=self.COLOURS["idle"], outline="")

    def set_state(self, state: str) -> None:
        self.itemconfig(self._dot, fill=self.COLOURS.get(state, MUTED))


# ─────────────────────────────────────────────────────────────
#  CONTROL PANEL
# ─────────────────────────────────────────────────────────────

class ControlPanel:
    """Main control window of the AI Subtitle Agent."""

    def __init__(self):
        self._cfg = AppConfig()
        self._pipeline: Optional[SubtitlePipeline] = None
        self._subtitle_queue: queue.Queue[DisplaySubtitle] = queue.Queue(maxsize=30)

        # ── Root window ───────────────────────────────────────
        self._root = tk.Tk()
        self._root.title("🎬 AI Subtitle Agent — EN → VI")
        self._root.configure(bg=DARK)
        self._root.resizable(False, False)

        self._apply_theme()

        # Subtitle overlay must exist BEFORE _build_ui() references it
        self._overlay = SubtitleOverlay(self._root, self._cfg.subtitle)

        self._build_ui()

        # Poll subtitle queue every 30 ms
        self._root.after(30, self._poll_subtitle_queue)

    # ── Theme ─────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")

        style.configure("TFrame",          background=PANEL)
        style.configure("Card.TFrame",     background=CARD)
        style.configure("TLabel",          background=PANEL, foreground=TEXT,
                         font=FONT_UI)
        style.configure("Title.TLabel",    background=PANEL, foreground=ACCENT,
                         font=(FONT_BIG[0], 14, "bold"))
        style.configure("Muted.TLabel",    background=CARD, foreground=MUTED,
                         font=(FONT_UI[0], 9))
        style.configure("TCombobox",       fieldbackground=CARD, background=CARD,
                         foreground=TEXT, selectbackground=ACCENT)
        style.configure("TEntry",          fieldbackground=CARD, foreground=TEXT,
                         insertcolor=TEXT)
        style.configure("TCheckbutton",    background=CARD, foreground=TEXT)
        style.configure("TScale",          background=CARD, troughcolor=BORDER)

        style.configure("Start.TButton",
                        background=ACCENT, foreground=DARK,
                        font=(FONT_BIG[0], 11, "bold"),
                        relief="flat", padding=(16, 8))
        style.map("Start.TButton",
                  background=[("active", "#00A87E"), ("disabled", BORDER)])

        style.configure("Stop.TButton",
                        background=WARN, foreground=WHITE,
                        font=(FONT_BIG[0], 11, "bold"),
                        relief="flat", padding=(16, 8))
        style.map("Stop.TButton",
                  background=[("active", "#E05020"), ("disabled", BORDER)])

        style.configure("Outline.TButton",
                        background=CARD, foreground=ACCENT,
                        relief="flat", padding=(8, 4))

    # ── UI Build ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self._root

        # ── Header ────────────────────────────────────────────
        header = tk.Frame(root, bg=PANEL, padx=20, pady=14)
        header.pack(fill="x")

        tk.Label(
            header, text="🎬  AI SUBTITLE AGENT",
            bg=PANEL, fg=ACCENT, font=(FONT_BIG[0], 15, "bold")
        ).pack(side="left")

        tk.Label(
            header, text="EN → VI  |  Real-Time",
            bg=PANEL, fg=MUTED, font=FONT_UI
        ).pack(side="left", padx=12)

        self._overlay_btn = tk.Button(
            header, text="☐ Overlay",
            bg=CARD, fg=TEXT,
            relief="flat", padx=8,
            command=self._overlay.toggle_visible,
            cursor="hand2",
        )
        self._overlay_btn.pack(side="right")

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # ── Main content ──────────────────────────────────────
        body = tk.Frame(root, bg=DARK, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        left  = tk.Frame(body, bg=DARK)
        right = tk.Frame(body, bg=DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        # ---- Source Card ----
        self._build_source_card(left)

        # ---- Model Card ----
        self._build_model_card(left)

        # ---- Status Card ----
        self._build_status_card(right)

        # ---- Subtitle Style Card ----
        self._build_style_card(right)

        # ── Controls ──────────────────────────────────────────
        ctrl = tk.Frame(root, bg=PANEL, padx=16, pady=10)
        ctrl.pack(fill="x")
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        self._start_btn = ttk.Button(
            ctrl, text="▶  Start", style="Start.TButton",
            command=self._start, cursor="hand2"
        )
        self._start_btn.pack(side="left", padx=(0, 8))

        self._stop_btn = ttk.Button(
            ctrl, text="■  Stop", style="Stop.TButton",
            command=self._stop, cursor="hand2", state="disabled"
        )
        self._stop_btn.pack(side="left")

        self._pipeline_status = tk.Label(
            ctrl, text="Idle", bg=PANEL, fg=MUTED, font=FONT_UI
        )
        self._pipeline_status.pack(side="right", padx=8)

        # ── Log pane ──────────────────────────────────────────
        self._build_transcript_pane(root)
        self._build_log_pane(root)

    def _scan_audio_devices(self) -> list[str]:
        """Dùng FFmpeg để liệt kê tất cả thiết bị audio trên Windows."""
        import subprocess, re
        devices = []
        try:
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stderr
            # Tìm tất cả tên thiết bị audio
            pattern = r'"([^"]+)"\s*\(audio\)'
            matches = re.findall(pattern, output)
            devices = matches
        except Exception as exc:
            log.warning(f"Could not scan audio devices: {exc}")
        return devices
    
    def _scan_ollama_models(self) -> list[str]:
        """Lấy danh sách model đang có trong Ollama."""
        try:
            import urllib.request, json
            url = f"{self._ollama_url_var.get().rstrip('/')}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                models = [m["name"] for m in data.get("models", [])]
                return models if models else []
        except Exception as exc:
            log.warning(f"Could not scan Ollama models: {exc}")
            return []

    def _card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=DARK, pady=4)
        outer.pack(fill="x")
        tk.Label(outer, text=title, bg=DARK, fg=MUTED, font=(FONT_UI[0], 9, "bold")).pack(
            anchor="w"
        )
        inner = tk.Frame(outer, bg=CARD, padx=12, pady=10)
        inner.pack(fill="x", pady=(2, 0))
        return inner

    def _row(self, parent: tk.Widget, label: str) -> tk.Frame:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=CARD, fg=MUTED, width=18, anchor="w",
                 font=(FONT_UI[0], 9)).pack(side="left")
        return row

    # ── Source Card ───────────────────────────────────────────

    def _build_source_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "AUDIO SOURCE")

        self._source_var = tk.StringVar(value="mic")

        row = self._row(card, "Input source")
        combo = ttk.Combobox(
            row, textvariable=self._source_var,
            values=["mic", "system", "file", "stream"],
            state="readonly", width=14
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._on_source_change)

        self._path_row = self._row(card, "File / Stream URL")
        self._path_var = tk.StringVar()
        self._path_entry = ttk.Entry(self._path_row, textvariable=self._path_var, width=22)
        self._path_entry.pack(side="left")
        self._browse_btn = ttk.Button(
            self._path_row, text="Browse", style="Outline.TButton",
            command=self._browse, cursor="hand2"
        )
        self._browse_btn.pack(side="left", padx=4)

        # Device dropdown — hiện khi chọn "system"
        self._device_row = self._row(card, "Audio device")
        self._audio_device_var = tk.StringVar(value="Scanning...")
        self._audio_device_combo = ttk.Combobox(
            self._device_row, textvariable=self._audio_device_var,
            state="readonly", width=30
        )
        self._audio_device_combo.pack(side="left")
        self._audio_device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        ttk.Button(
            self._device_row, text="↺ Refresh", style="Outline.TButton",
            command=self._refresh_devices, cursor="hand2"
        ).pack(side="left", padx=4)

        self._path_row_frame = self._path_row
        self._toggle_path_row()
        # Scan devices in background on startup
        self._root.after(500, self._refresh_devices)

        noise_row = self._row(card, "Noise filter")
        self._noise_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            noise_row, text="DeepFilterNet",
            variable=self._noise_var, bg=CARD, fg=TEXT,
            selectcolor=CARD, activebackground=CARD, activeforeground=ACCENT,
        ).pack(side="left")

    def _on_source_change(self, _event=None) -> None:
        self._toggle_path_row()

    def _toggle_path_row(self) -> None:
        src = self._source_var.get()
        if src == "system":
            # Ẩn text entry, hiện device dropdown
            self._path_entry.configure(state="disabled")
            self._browse_btn.configure(state="disabled")
            self._audio_device_combo.configure(state="readonly")
        elif src in ("file", "stream"):
            # Hiện text entry, ẩn device dropdown
            self._path_entry.configure(state="normal")
            self._browse_btn.configure(state="normal" if src == "file" else "disabled")
            self._audio_device_combo.configure(state="disabled")
        else:
            # mic — ẩn tất cả
            self._path_entry.configure(state="disabled")
            self._browse_btn.configure(state="disabled")
            self._audio_device_combo.configure(state="disabled")

    def _refresh_devices(self) -> None:
        """Quét lại danh sách thiết bị audio."""
        import threading
        def scan():
            devices = self._scan_audio_devices()
            if not devices:
                devices = ["Stereo Mix (Realtek(R) Audio)", "CABLE Output (VB-Audio Virtual Cable)"]
            self._root.after(0, lambda: self._update_device_list(devices))
        threading.Thread(target=scan, daemon=True).start()
        self._audio_device_var.set("Scanning...")

    def _update_device_list(self, devices: list) -> None:
        self._audio_device_combo["values"] = devices
        if devices:
            self._audio_device_combo.current(0)
            self._audio_device_var.set(devices[0])
        self._log(f"Found {len(devices)} audio device(s): {', '.join(devices)}")

    def _on_device_selected(self, _event=None) -> None:
        """Khi chọn thiết bị, tự điền vào path_var để pipeline dùng."""
        selected = self._audio_device_var.get()
        self._path_var.set(selected)
    
    def _refresh_ollama_models(self) -> None:
        """Quét lại danh sách model Ollama trong background."""
        import threading
        def scan():
            models = self._scan_ollama_models()
            self._root.after(0, lambda: self._update_ollama_list(models))
        threading.Thread(target=scan, daemon=True).start()
        self._ollama_var.set("Scanning...")

    def _update_ollama_list(self, models: list) -> None:
        if not models:
            models = ["qwen2.5:3b", "deepseek-v2:latest"]
            self._log("Could not reach Ollama — showing defaults.")
        else:
            self._log(f"Found {len(models)} Ollama model(s): {', '.join(models)}")
        self._ollama_combo["values"] = models
        # Chọn model đầu tiên nếu chưa có lựa chọn
        current = self._ollama_var.get()
        if current not in models:
            self._ollama_combo.current(0)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video / audio file",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)

    # ── Model Card ────────────────────────────────────────────

    def _build_model_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "MODELS")

        row1 = self._row(card, "Whisper model")
        self._whisper_var = tk.StringVar(value="distil-small.en")
        ttk.Combobox(
            row1, textvariable=self._whisper_var,
            values=["distil-small.en", "base", "small", "base.en", "tiny"],
            state="readonly", width=18
        ).pack(side="left")

        row2 = self._row(card, "Device")
        self._device_var = tk.StringVar(value="cpu")
        ttk.Combobox(
            row2, textvariable=self._device_var,
            values=["cpu", "cuda"],
            state="readonly", width=10
        ).pack(side="left")

        row3 = self._row(card, "Ollama model")
        self._ollama_var = tk.StringVar(value="")
        self._ollama_combo = ttk.Combobox(
            row3, textvariable=self._ollama_var,
            state="readonly", width=22
        )
        self._ollama_combo.pack(side="left")
        ttk.Button(
            row3, text="↺", style="Outline.TButton",
            command=self._refresh_ollama_models, cursor="hand2", width=2
        ).pack(side="left", padx=4)
        # Tự quét khi khởi động
        self._root.after(800, self._refresh_ollama_models)

        row4 = self._row(card, "Ollama URL")
        self._ollama_url_var = tk.StringVar(value="http://localhost:11434")
        ttk.Entry(row4, textvariable=self._ollama_url_var, width=22).pack(side="left")

    # ── Status Card ───────────────────────────────────────────

    def _build_status_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "MODULE STATUS")

        modules = [
            ("AudioCapture", "audio"),
            ("STTEngine",    "stt"),
            ("Translator",   "translator"),
            ("Formatter",    "formatter"),
        ]
        self._status_dots:  dict[str, StatusDot]  = {}
        self._status_labels: dict[str, tk.Label]  = {}

        for name, key in modules:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=3)

            dot = StatusDot(row)
            dot.pack(side="left")
            tk.Label(row, text=name, bg=CARD, fg=TEXT,
                     font=(FONT_UI[0], 9), width=14, anchor="w").pack(side="left", padx=4)
            lbl = tk.Label(row, text="idle", bg=CARD, fg=MUTED, font=(FONT_UI[0], 9))
            lbl.pack(side="left")

            self._status_dots[name]  = dot
            self._status_labels[name] = lbl

        # Token / latency stats
        sep = tk.Frame(card, bg=BORDER, height=1)
        sep.pack(fill="x", pady=6)

        self._lat_label = tk.Label(card, text="Last latency: —", bg=CARD, fg=MUTED,
                                   font=(FONT_UI[0], 9))
        self._lat_label.pack(anchor="w")
        self._seg_label = tk.Label(card, text="Segments: 0", bg=CARD, fg=MUTED,
                                   font=(FONT_UI[0], 9))
        self._seg_label.pack(anchor="w")

        self._seg_count = 0

    # ── Subtitle Style Card ───────────────────────────────────

    def _build_style_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "SUBTITLE STYLE")

        row1 = self._row(card, "Font size")
        self._font_size_var = tk.IntVar(value=26)
        tk.Scale(
            row1, from_=16, to=48,
            variable=self._font_size_var, orient="horizontal",
            length=130, bg=CARD, fg=TEXT, troughcolor=BORDER,
            highlightthickness=0, command=self._update_subtitle_style,
        ).pack(side="left")

        row2 = self._row(card, "Text color")
        self._text_color_var = tk.StringVar(value="#FFFFFF")
        for color, label in [("#FFFFFF", "White"), ("#FFE566", "Yellow"), ("#88FF88", "Green")]:
            tk.Radiobutton(
                row2, text=label, value=color, variable=self._text_color_var,
                bg=CARD, fg=TEXT, selectcolor=CARD,
                activebackground=CARD, activeforeground=ACCENT,
                command=self._update_subtitle_style,
            ).pack(side="left")

        row3 = self._row(card, "Position")
        self._pos_var = tk.StringVar(value="bottom")
        for pos in ("bottom", "top"):
            tk.Radiobutton(
                row3, text=pos.capitalize(), value=pos, variable=self._pos_var,
                bg=CARD, fg=TEXT, selectcolor=CARD,
                activebackground=CARD, activeforeground=ACCENT,
            ).pack(side="left")

        row4 = self._row(card, "Opacity")
        self._opacity_var = tk.DoubleVar(value=0.75)
        tk.Scale(
            row4, from_=0.3, to=1.0, resolution=0.05,
            variable=self._opacity_var, orient="horizontal",
            length=130, bg=CARD, fg=TEXT, troughcolor=BORDER,
            highlightthickness=0,
        ).pack(side="left")

        row5 = self._row(card, "Show original")
        self._show_orig_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row5, text="English debug",
            variable=self._show_orig_var, bg=CARD, fg=TEXT,
            selectcolor=CARD, activebackground=CARD,
        ).pack(side="left")

    def _update_subtitle_style(self, _event=None) -> None:
        self._cfg.subtitle.font_size = self._font_size_var.get()
        self._cfg.subtitle.text_color = self._text_color_var.get()

    def _build_transcript_pane(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=DARK, padx=16, pady=4)
        frame.pack(fill="x")

        header = tk.Frame(frame, bg=DARK)
        header.pack(fill="x")
        tk.Label(header, text="HEARD (Speech-to-Text)",
                 bg=DARK, fg=MUTED, font=(FONT_UI[0], 9, "bold")).pack(side="left")
        tk.Button(
            header, text="Clear", bg=DARK, fg=MUTED,
            relief="flat", font=(FONT_UI[0], 8),
            command=self._clear_transcript, cursor="hand2",
        ).pack(side="right")

        self._transcript_text = tk.Text(
            frame, height=4,
            bg="#0A0F0A", fg="#00FF88",
            font=FONT_MONO,
            relief="flat", wrap="word",
            insertbackground=ACCENT,
        )
        self._transcript_text.pack(fill="both", expand=True, pady=(4, 0))
        self._transcript_text.configure(state="disabled")

    def _clear_transcript(self) -> None:
        self._transcript_text.configure(state="normal")
        self._transcript_text.delete("1.0", "end")
        self._transcript_text.configure(state="disabled")

    def _on_transcript(self, text: str) -> None:
        """Called from pipeline thread — schedule UI update safely."""
        self._root.after(0, lambda: self._show_transcript(text))

    def _show_transcript(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._transcript_text.configure(state="normal")
        self._transcript_text.insert("end", f"[{ts}] {text}\n")
        self._transcript_text.see("end")
        self._transcript_text.configure(state="disabled")

    # ── Log Pane ──────────────────────────────────────────────

    def _build_log_pane(self, parent: tk.Widget) -> None:
        log_frame = tk.Frame(parent, bg=DARK, padx=16, pady=8)
        log_frame.pack(fill="both", expand=True)

        header = tk.Frame(log_frame, bg=DARK)
        header.pack(fill="x")
        tk.Label(header, text="LOG", bg=DARK, fg=MUTED,
                 font=(FONT_UI[0], 9, "bold")).pack(side="left")
        tk.Button(
            header, text="Clear", bg=DARK, fg=MUTED,
            relief="flat", font=(FONT_UI[0], 8),
            command=self._clear_log, cursor="hand2",
        ).pack(side="right")

        self._log_text = tk.Text(
            log_frame, height=7,
            bg="#0A0A0A", fg="#5FBF5F",
            font=FONT_MONO,
            relief="flat", wrap="word",
            insertbackground=ACCENT,
            selectbackground=ACCENT,
        )
        self._log_text.pack(fill="both", expand=True, pady=(4, 0))

        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)

        self._log("AI Subtitle Agent ready. Configure source and press Start.")

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] {msg}\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ── Pipeline control ──────────────────────────────────────

    def _build_config(self) -> AppConfig:
        """Read UI values into a fresh AppConfig."""
        cfg = AppConfig()

        # Audio
        cfg.audio.source_type = self._source_var.get()
        cfg.audio.source_path = self._path_var.get()
        cfg.audio.noise_filter_enabled = self._noise_var.get()

        # STT
        cfg.stt.model_name = self._whisper_var.get()
        cfg.stt.device     = self._device_var.get()

        # Translation
        cfg.translation.model           = self._ollama_var.get()
        cfg.translation.ollama_base_url = self._ollama_url_var.get()

        # Subtitle
        cfg.subtitle.font_size  = self._font_size_var.get()
        cfg.subtitle.text_color = self._text_color_var.get()
        cfg.subtitle.bg_opacity = self._opacity_var.get()

        return cfg

    def _start(self) -> None:
        cfg = self._build_config()
        self._cfg = cfg

        self._pipeline = SubtitlePipeline(
            cfg=cfg,
            subtitle_cb=self._on_subtitle,
            status_cb=self._on_status,
            transcript_cb=self._on_transcript,
        )

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._pipeline_status.configure(text="Starting …", fg=ACCENT)

        # Reset status dots
        for dot in self._status_dots.values():
            dot.set_state("idle")

        self._pipeline.start()
        self._log(f"Pipeline started — {cfg.audio.source_type} → {cfg.stt.model_name} → {cfg.translation.model}")

    def _stop(self) -> None:
        if self._pipeline:
            threading.Thread(target=self._pipeline.stop, daemon=True).start()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._pipeline_status.configure(text="Stopped", fg=MUTED)
        self._log("Pipeline stopped.")

    # ── Callbacks from pipeline (may be called from async thread) ─

    def _on_subtitle(self, sub: DisplaySubtitle) -> None:
        """Thread-safe: push to queue; UI polls it."""
        try:
            self._subtitle_queue.put_nowait(sub)
        except queue.Full:
            pass

    def _on_status(self, module: str, status: str) -> None:
        """Called from the pipeline thread — schedule UI update on main thread."""
        self._root.after(0, lambda: self._apply_status(module, status))

    def _apply_status(self, module: str, status: str) -> None:
        if module == "Pipeline":
            self._pipeline_status.configure(
                text=status,
                fg=ACCENT if "Running" in status else MUTED,
            )
            return

        if module in self._status_labels:
            self._status_labels[module].configure(text=status)
            if "✔" in status or "Running" in status or "Ready" in status:
                self._status_dots[module].set_state("ok")
            elif "Error" in status:
                self._status_dots[module].set_state("error")
            elif "Loading" in status or "Starting" in status:
                self._status_dots[module].set_state("warn")

        self._log(f"[{module}] {status}")

    # ── Subtitle queue polling ────────────────────────────────

    def _poll_subtitle_queue(self) -> None:
        t0 = time.time()
        while not self._subtitle_queue.empty():
            try:
                sub: DisplaySubtitle = self._subtitle_queue.get_nowait()
            except queue.Empty:
                break

            self._overlay.show(sub)
            self._seg_count += 1
            self._seg_label.configure(text=f"Segments: {self._seg_count}")
            lat_ms = int((time.time() - t0) * 1_000)
            self._lat_label.configure(
                text=f"Last dispatch: {lat_ms} ms | ID #{sub.segment_id}"
            )

            if self._show_orig_var.get():
                self._log(f"[EN] {sub.original_text}")
            self._log(f"[VI] {' | '.join(sub.lines)}")

        self._root.after(30, self._poll_subtitle_queue)

    # ── Run ───────────────────────────────────────────────────

    def run(self) -> None:
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_close(self) -> None:
        if self._pipeline and self._pipeline.is_running():
            if messagebox.askyesno("Quit", "Pipeline is running. Stop and quit?"):
                self._stop()
            else:
                return
        self._root.destroy()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ControlPanel()
    app.run()
