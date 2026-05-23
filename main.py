"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI Subtitle Agent — Tkinter Interface (Dual-provider: Groq / Gemini)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from config import AppConfig
from modules.subtitle_formatter import DisplaySubtitle
from pipeline import PipelineState, SubtitlePipeline
from utils.logger import get_logger

log = get_logger("UI")

# ─────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────

DARK    = "#0D0D0D"
PANEL   = "#151515"
CARD    = "#1C1C1C"
BORDER  = "#2A2A2A"
ACCENT  = "#00C896"
ACCENT2 = "#0088FF"
WARN    = "#FF6B35"
TEXT    = "#E8E8E8"
MUTED   = "#6B6B6B"
WHITE   = "#FFFFFF"
_GHOST_COLOUR  = "#444444"
_LIVE_COLOUR   = "#FFFFFF"
_SHADOW_COLOUR = "#000000"

INPUT_BG = "#F5F5F5"
INPUT_FG = "#111111"

FONT_MONO = ("Consolas", 10) if sys.platform == "win32" else ("Menlo", 10)
FONT_UI   = ("Segoe UI", 10) if sys.platform == "win32" else ("SF Pro Display", 10)
FONT_BIG  = ("Segoe UI Semibold", 12) if sys.platform == "win32" else ("SF Pro Display", 12)

WIN_W, WIN_H = 980, 540
WIN_MIN_W, WIN_MIN_H = 860, 500

# ─────────────────────────────────────────────────────────────
#  SUBTITLE OVERLAY
# ─────────────────────────────────────────────────────────────

class SubtitleOverlay:
    def __init__(self, root: tk.Tk, cfg):
        self._cfg  = cfg
        self._root = root
        self._win:    Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas]  = None

        self._hide_job: Optional[str] = None
        self._current_seg_id: int = -1
        self._target_alpha: float = 0.0
        self._fade_step_ms: int   = 16

        self._word_jobs: List[str]  = []
        self._revealed: List[str]   = []
        self._full_lines: List[str] = []

        self._drag_x = 0
        self._drag_y = 0
        self._build()

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

    def show(self, subtitle) -> None:
        if self._win is None:
            return
        if self._hide_job:
            self._root.after_cancel(self._hide_job)
            self._hide_job = None
        self._cancel_word_jobs()

        self._current_seg_id = subtitle.segment_id
        self._full_lines     = list(subtitle.lines)
        self._revealed       = []

        self._draw_ghost(subtitle.lines)
        self._target_alpha = self._cfg.bg_opacity
        self._animate_fade("in")
        self._schedule_words(subtitle.word_schedule, subtitle.segment_id)

        duration_ms = subtitle.hide_at_ms - int(time.time() * 1_000)
        duration_ms = max(duration_ms, self._cfg.min_display_ms)
        self._hide_job = self._root.after(
            duration_ms,
            lambda: self._begin_fade_out(subtitle.segment_id),
        )

    def _draw_ghost(self, lines: List[str]) -> None:
        canvas = self._canvas
        canvas.delete("all")
        cfg = self._cfg
        w, h = self._box_w, self._box_h

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
            canvas.create_text(
                w // 2 + 2, y + 2,
                text=line, anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_SHADOW_COLOUR, tags=f"shadow_{i}",
            )
            canvas.create_text(
                w // 2, y,
                text=line, anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_GHOST_COLOUR, tags=f"ghost_{i}",
            )
            canvas.create_text(
                w // 2, y,
                text="", anchor="center",
                font=(cfg.font_family, cfg.font_size, cfg.font_weight),
                fill=_LIVE_COLOUR, tags=f"live_{i}",
            )

    def _schedule_words(self, word_schedule: List[Tuple[str, int]], seg_id: int) -> None:
        if not word_schedule:
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
        if seg_id != self._current_seg_id:
            return
        self._revealed = [w for line in self._full_lines for w in line.split()]
        self._redraw_live()

    def _redraw_live(self) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        revealed_text = " ".join(self._revealed)
        max_chars     = getattr(self._cfg, "max_chars_per_line", 40)
        live_lines    = self._wrap(revealed_text, max_chars)

        for i in range(len(self._full_lines)):
            live_text = live_lines[i] if i < len(live_lines) else ""
            canvas.itemconfig(f"live_{i}", text=live_text)

    @staticmethod
    def _wrap(text: str, max_chars: int) -> List[str]:
        words = text.split()
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
            self._root.after(self._fade_step_ms, lambda: self._animate_fade(direction))

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
        self._win.attributes("-alpha", 0.0 if alpha > 0.0 else self._cfg.bg_opacity)

    def _drag_start(self, event: tk.Event) -> None:
        self._drag_x = event.x_root - self._win.winfo_x()
        self._drag_y = event.y_root - self._win.winfo_y()

    def _drag_move(self, event: tk.Event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self._win.geometry(f"+{x}+{y}")


# ─────────────────────────────────────────────────────────────
#  STATUS DOT
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
    def __init__(self):
        self._cfg = AppConfig()
        self._pipeline: Optional[SubtitlePipeline] = None
        self._subtitle_queue: queue.Queue[DisplaySubtitle] = queue.Queue(maxsize=30)

        self._root = tk.Tk()
        self._root.title("🎬 AI Subtitle Agent — EN → VI")
        self._root.configure(bg=DARK)
        self._root.minsize(WIN_MIN_W, WIN_MIN_H)
        self._root.resizable(True, True)

        self._apply_theme()
        self._overlay = SubtitleOverlay(self._root, self._cfg.subtitle)
        self._build_ui()
        self._root.after(30, self._poll_subtitle_queue)

    # ── Theme ─────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")

        self._root.option_add("*TCombobox*Listbox.background", INPUT_BG)
        self._root.option_add("*TCombobox*Listbox.foreground", INPUT_FG)
        self._root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self._root.option_add("*TCombobox*Listbox.selectForeground", INPUT_FG)

        style.configure("TFrame",       background=PANEL)
        style.configure("Card.TFrame",  background=CARD)
        style.configure("TLabel",       background=PANEL, foreground=TEXT, font=FONT_UI)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                        font=(FONT_UI[0], 9))
        style.configure("TCombobox",
            fieldbackground=INPUT_BG, background=INPUT_BG,
            foreground=INPUT_FG, arrowcolor=INPUT_FG,
            selectbackground=ACCENT, selectforeground=INPUT_FG,
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", INPUT_BG), ("disabled", BORDER)],
            foreground=[("readonly", INPUT_FG), ("disabled", MUTED)],
            background=[("readonly", INPUT_BG), ("active", INPUT_BG)],
        )
        style.configure("TEntry",
            fieldbackground=INPUT_BG, foreground=INPUT_FG, insertcolor=INPUT_FG,
        )
        style.configure("TCheckbutton", background=CARD, foreground=TEXT)
        style.configure("TScale",       background=CARD, troughcolor=BORDER)
        style.configure("Start.TButton",
            background=ACCENT, foreground=DARK,
            font=(FONT_BIG[0], 11, "bold"), relief="flat", padding=(16, 8))
        style.map("Start.TButton",
            background=[("active", "#00A87E"), ("disabled", BORDER)])
        style.configure("Stop.TButton",
            background=WARN, foreground=WHITE,
            font=(FONT_BIG[0], 11, "bold"), relief="flat", padding=(16, 8))
        style.map("Stop.TButton",
            background=[("active", "#E05020"), ("disabled", BORDER)])
        style.configure("Outline.TButton",
            background=CARD, foreground=ACCENT, relief="flat", padding=(8, 4))

        # Provider tab button styles
        style.configure("TabActive.TButton",
            background=ACCENT, foreground=DARK,
            font=(FONT_UI[0], 9, "bold"), relief="flat", padding=(10, 4))
        style.configure("TabInactive.TButton",
            background=BORDER, foreground=MUTED,
            font=(FONT_UI[0], 9), relief="flat", padding=(10, 4))

    # ── UI Build ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self._root

        # Header
        header = tk.Frame(root, bg=PANEL, padx=20, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="🎬  AI SUBTITLE AGENT",
                 bg=PANEL, fg=ACCENT, font=(FONT_BIG[0], 15, "bold")).pack(side="left")
        tk.Label(header, text="EN → VI  |  Real-Time",
                 bg=PANEL, fg=MUTED, font=FONT_UI).pack(side="left", padx=12)
        self._overlay_btn = tk.Button(
            header, text="☐ Overlay", bg=CARD, fg=TEXT,
            relief="flat", padx=8, command=self._overlay.toggle_visible, cursor="hand2",
        )
        self._overlay_btn.pack(side="right")
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # Body
        body = tk.Frame(root, bg=DARK, padx=12, pady=8)
        body.pack(fill="both", expand=True)
        left  = tk.Frame(body, bg=DARK)
        right = tk.Frame(body, bg=DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        self._build_source_card(left)
        self._build_model_card(left)
        self._build_status_card(right)
        self._build_style_card(right)

        # Controls
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

        # Bottom: transcript + log
        bottom = tk.Frame(root, bg=DARK, padx=12, pady=8)
        bottom.pack(fill="both", expand=True)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        heard_col = tk.Frame(bottom, bg=DARK)
        log_col   = tk.Frame(bottom, bg=DARK)
        heard_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        log_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._build_transcript_pane(heard_col)
        self._build_log_pane(log_col)
        self._center_window(WIN_W, WIN_H)

    def _center_window(self, width: int, height: int) -> None:
        self._root.update_idletasks()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self._root.geometry(f"{width}x{height}+{x}+{y}")

    @staticmethod
    def _style_combobox(combo: ttk.Combobox) -> None:
        try:
            combo.configure(style="TCombobox")
        except tk.TclError:
            pass

    def _card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=DARK, pady=4)
        outer.pack(fill="x")
        tk.Label(outer, text=title, bg=DARK, fg=MUTED,
                 font=(FONT_UI[0], 9, "bold")).pack(anchor="w")
        inner = tk.Frame(outer, bg=CARD, padx=10, pady=8)
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
            state="readonly", width=14,
        )
        self._style_combobox(combo)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._on_source_change)

        self._path_row = self._row(card, "File / Stream URL")
        self._path_var = tk.StringVar()
        self._path_entry = ttk.Entry(self._path_row, textvariable=self._path_var, width=22)
        self._path_entry.pack(side="left")
        self._browse_btn = ttk.Button(
            self._path_row, text="Browse", style="Outline.TButton",
            command=self._browse, cursor="hand2",
        )
        self._browse_btn.pack(side="left", padx=4)

        self._device_row = self._row(card, "Audio device")
        self._audio_device_var = tk.StringVar(value="Scanning...")
        self._audio_device_combo = ttk.Combobox(
            self._device_row, textvariable=self._audio_device_var,
            state="readonly", width=30,
        )
        self._style_combobox(self._audio_device_combo)
        self._audio_device_combo.pack(side="left")
        self._audio_device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)
        ttk.Button(
            self._device_row, text="↺ Refresh", style="Outline.TButton",
            command=self._refresh_devices, cursor="hand2",
        ).pack(side="left", padx=4)

        self._toggle_path_row()
        self._root.after(500, self._refresh_devices)

    def _on_source_change(self, _=None) -> None:
        self._toggle_path_row()

    def _toggle_path_row(self) -> None:
        src = self._source_var.get()
        if src == "system":
            self._path_entry.configure(state="disabled")
            self._browse_btn.configure(state="disabled")
            self._audio_device_combo.configure(state="readonly")
        elif src in ("file", "stream"):
            self._path_entry.configure(state="normal")
            self._browse_btn.configure(
                state="normal" if src == "file" else "disabled"
            )
            self._audio_device_combo.configure(state="disabled")
        else:
            self._path_entry.configure(state="disabled")
            self._browse_btn.configure(state="disabled")
            self._audio_device_combo.configure(state="disabled")

    def _scan_audio_devices(self) -> list:
        import subprocess, re
        devices = []
        try:
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True, text=True, timeout=5,
            )
            devices = re.findall(r'"([^"]+)"\s*\(audio\)', result.stderr)
        except Exception as exc:
            log.warning(f"Could not scan audio devices: {exc}")
        return devices

    def _refresh_devices(self) -> None:
        def scan():
            devices = self._scan_audio_devices()
            if not devices:
                devices = ["Stereo Mix (Realtek(R) Audio)",
                           "CABLE Output (VB-Audio Virtual Cable)"]
            self._root.after(0, lambda: self._update_device_list(devices))
        threading.Thread(target=scan, daemon=True).start()
        self._audio_device_var.set("Scanning...")

    def _update_device_list(self, devices: list) -> None:
        self._audio_device_combo["values"] = devices
        if devices:
            self._audio_device_combo.current(0)
            self._audio_device_var.set(devices[0])
        self._log(f"Found {len(devices)} audio device(s).")

    def _on_device_selected(self, _=None) -> None:
        self._path_var.set(self._audio_device_var.get())

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

    # ── Model Card (Deepgram + Dual-tab LLM) ─────────────────

    def _build_model_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "MODELS")

        # ── Deepgram ──────────────────────────────────────────
        row_dg_key = self._row(card, "Deepgram API key")
        self._deepgram_key_var = tk.StringVar(value="")
        ttk.Entry(
            row_dg_key, textvariable=self._deepgram_key_var,
            width=28, show="*",
        ).pack(side="left")

        row_dg_model = self._row(card, "Deepgram model")
        self._deepgram_model_var = tk.StringVar(value="nova-2")
        dg_combo = ttk.Combobox(
            row_dg_model, textvariable=self._deepgram_model_var,
            values=["nova-2", "nova-3", "nova", "enhanced", "base"],
            state="readonly", width=18,
        )
        self._style_combobox(dg_combo)
        dg_combo.pack(side="left")

        # ── LLM Provider separator ────────────────────────────
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))

        # Provider tab row
        tab_row = tk.Frame(card, bg=CARD)
        tab_row.pack(fill="x", pady=(0, 6))
        tk.Label(tab_row, text="LLM Provider", bg=CARD, fg=MUTED,
                 font=(FONT_UI[0], 9), width=18, anchor="w").pack(side="left")

        self._active_provider_var = tk.StringVar(value="groq")

        self._groq_tab_btn = tk.Button(
            tab_row, text="Groq", bg=ACCENT, fg=DARK,
            font=(FONT_UI[0], 9, "bold"), relief="flat", padx=12, pady=3,
            cursor="hand2", command=lambda: self._switch_provider("groq"),
        )
        self._groq_tab_btn.pack(side="left", padx=(0, 2))

        self._gemini_tab_btn = tk.Button(
            tab_row, text="Gemini", bg=BORDER, fg=MUTED,
            font=(FONT_UI[0], 9), relief="flat", padx=12, pady=3,
            cursor="hand2", command=lambda: self._switch_provider("gemini"),
        )
        self._gemini_tab_btn.pack(side="left")

        # ── Groq fields frame ─────────────────────────────────
        self._groq_frame = tk.Frame(card, bg=CARD)
        self._groq_frame.pack(fill="x")

        row_groq_key = tk.Frame(self._groq_frame, bg=CARD)
        row_groq_key.pack(fill="x", pady=3)
        tk.Label(row_groq_key, text="Groq API key", bg=CARD, fg=MUTED,
                 width=18, anchor="w", font=(FONT_UI[0], 9)).pack(side="left")
        self._groq_key_var = tk.StringVar(value="")
        ttk.Entry(
            row_groq_key, textvariable=self._groq_key_var,
            width=28, show="*",
        ).pack(side="left")

        row_groq_model = tk.Frame(self._groq_frame, bg=CARD)
        row_groq_model.pack(fill="x", pady=3)
        tk.Label(row_groq_model, text="Groq model", bg=CARD, fg=MUTED,
                 width=18, anchor="w", font=(FONT_UI[0], 9)).pack(side="left")
        self._groq_model_var = tk.StringVar(value="llama-3.1-8b-instant")
        groq_combo = ttk.Combobox(
            row_groq_model, textvariable=self._groq_model_var,
            values=[
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
            ],
            state="readonly", width=34,
        )
        self._style_combobox(groq_combo)
        groq_combo.pack(side="left")

        # ── Gemini fields frame ───────────────────────────────
        self._gemini_frame = tk.Frame(card, bg=CARD)
        # Hidden initially

        row_gem_key = tk.Frame(self._gemini_frame, bg=CARD)
        row_gem_key.pack(fill="x", pady=3)
        tk.Label(row_gem_key, text="Gemini API key", bg=CARD, fg=MUTED,
                 width=18, anchor="w", font=(FONT_UI[0], 9)).pack(side="left")
        self._gemini_key_var = tk.StringVar(value="")
        ttk.Entry(
            row_gem_key, textvariable=self._gemini_key_var,
            width=28, show="*",
        ).pack(side="left")

        row_gem_model = tk.Frame(self._gemini_frame, bg=CARD)
        row_gem_model.pack(fill="x", pady=3)
        tk.Label(row_gem_model, text="Gemini model", bg=CARD, fg=MUTED,
                 width=18, anchor="w", font=(FONT_UI[0], 9)).pack(side="left")
        self._gemini_model_var = tk.StringVar(value="gemini-2.0-flash")
        gem_combo = ttk.Combobox(
            row_gem_model, textvariable=self._gemini_model_var,
            values=[
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash",
            ],
            state="readonly", width=34,
        )
        self._style_combobox(gem_combo)
        gem_combo.pack(side="left")

    def _switch_provider(self, provider: str) -> None:
        self._active_provider_var.set(provider)

        if provider == "groq":
            self._groq_tab_btn.configure(bg=ACCENT, fg=DARK,
                                          font=(FONT_UI[0], 9, "bold"))
            self._gemini_tab_btn.configure(bg=BORDER, fg=MUTED,
                                            font=(FONT_UI[0], 9))
            self._gemini_frame.pack_forget()
            self._groq_frame.pack(fill="x")
        else:
            self._gemini_tab_btn.configure(bg=ACCENT, fg=DARK,
                                            font=(FONT_UI[0], 9, "bold"))
            self._groq_tab_btn.configure(bg=BORDER, fg=MUTED,
                                          font=(FONT_UI[0], 9))
            self._groq_frame.pack_forget()
            self._gemini_frame.pack(fill="x")

        # Update the LLM status dot label
        label = "Groq" if provider == "groq" else "Gemini"
        if hasattr(self, "_llm_status_label_widget"):
            self._llm_status_label_widget.configure(text=label)

    # ── Status Card ───────────────────────────────────────────

    def _build_status_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "MODULE STATUS")

        modules = [
            ("AudioCapture", "audio"),
            ("Deepgram",     "stt"),
            ("Groq",         "llm"),      # label updated when provider switches
            ("Formatter",    "formatter"),
        ]
        self._status_dots:   dict = {}
        self._status_labels: dict = {}

        for name, key in modules:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=3)

            dot = StatusDot(row)
            dot.pack(side="left")

            lbl_name = tk.Label(
                row, text=name, bg=CARD, fg=TEXT,
                font=(FONT_UI[0], 9), width=14, anchor="w",
            )
            lbl_name.pack(side="left", padx=4)

            # Keep reference to LLM name label so we can rename it
            if key == "llm":
                self._llm_status_label_widget = lbl_name

            lbl_status = tk.Label(row, text="idle", bg=CARD, fg=MUTED,
                                   font=(FONT_UI[0], 9))
            lbl_status.pack(side="left")

            self._status_dots[name]   = dot
            self._status_labels[name] = lbl_status

        sep = tk.Frame(card, bg=BORDER, height=1)
        sep.pack(fill="x", pady=6)

        self._lat_label = tk.Label(card, text="Last latency: —", bg=CARD, fg=MUTED,
                                   font=(FONT_UI[0], 9))
        self._lat_label.pack(anchor="w")
        self._seg_label = tk.Label(card, text="Segments: 0", bg=CARD, fg=MUTED,
                                   font=(FONT_UI[0], 9))
        self._seg_label.pack(anchor="w")
        self._seg_count = 0

    # ── Style Card ────────────────────────────────────────────

    def _build_style_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "SUBTITLE STYLE")

        row1 = self._row(card, "Font size")
        self._font_size_var = tk.IntVar(value=26)
        tk.Scale(
            row1, from_=16, to=48, variable=self._font_size_var,
            orient="horizontal", length=130, bg=CARD, fg=TEXT,
            troughcolor=BORDER, highlightthickness=0,
            command=self._update_subtitle_style,
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

    def _update_subtitle_style(self, _=None) -> None:
        self._cfg.subtitle.font_size  = self._font_size_var.get()
        self._cfg.subtitle.text_color = self._text_color_var.get()

    # ── Transcript & Log ──────────────────────────────────────

    def _build_transcript_pane(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=DARK)
        frame.pack(fill="both", expand=True)

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
            frame, height=3, bg="#0A0F0A", fg="#00FF88",
            font=FONT_MONO, relief="flat", wrap="word",
            insertbackground=ACCENT,
        )
        self._transcript_text.pack(fill="both", expand=True, pady=(4, 0))
        self._transcript_text.configure(state="disabled")

    def _clear_transcript(self) -> None:
        self._transcript_text.configure(state="normal")
        self._transcript_text.delete("1.0", "end")
        self._transcript_text.configure(state="disabled")

    def _on_transcript(self, text: str) -> None:
        self._root.after(0, lambda: self._show_transcript(text))

    def _show_transcript(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._transcript_text.configure(state="normal")
        self._transcript_text.insert("end", f"[{ts}] {text}\n")
        self._transcript_text.see("end")
        self._transcript_text.configure(state="disabled")

    def _build_log_pane(self, parent: tk.Widget) -> None:
        log_frame = tk.Frame(parent, bg=DARK)
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

        log_body = tk.Frame(log_frame, bg=DARK)
        log_body.pack(fill="both", expand=True, pady=(4, 0))

        sb = ttk.Scrollbar(log_body, orient="vertical")
        sb.pack(side="right", fill="y")

        self._log_text = tk.Text(
            log_body, height=5, bg="#0A0A0A", fg="#5FBF5F",
            font=FONT_MONO, relief="flat", wrap="word",
            insertbackground=ACCENT, selectbackground=ACCENT,
            yscrollcommand=sb.set,
        )
        self._log_text.pack(side="left", fill="both", expand=True)
        sb.configure(command=self._log_text.yview)
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
        cfg = AppConfig()

        cfg.audio.source_type = self._source_var.get()
        cfg.audio.source_path = self._path_var.get()

        cfg.stt.api_key = self._deepgram_key_var.get().strip()
        cfg.stt.model   = self._deepgram_model_var.get()

        provider = self._active_provider_var.get()
        cfg.active_provider = provider

        cfg.groq.api_key = self._groq_key_var.get().strip()
        cfg.groq.model   = self._groq_model_var.get().strip()

        cfg.gemini.api_key = self._gemini_key_var.get().strip()
        cfg.gemini.model   = self._gemini_model_var.get().strip()

        cfg.subtitle.font_size  = self._font_size_var.get()
        cfg.subtitle.text_color = self._text_color_var.get()
        cfg.subtitle.bg_opacity = self._opacity_var.get()

        return cfg

    def _start(self) -> None:
        cfg = self._build_config()
        self._cfg = cfg

        if not cfg.stt.api_key:
            self._log("Error: Deepgram API key is required.")
            return

        provider = cfg.active_provider
        if provider == "groq" and not cfg.groq.api_key:
            self._log("Error: Groq API key is required.")
            return
        if provider == "gemini" and not cfg.gemini.api_key:
            self._log("Error: Gemini API key is required.")
            return

        self._pipeline = SubtitlePipeline(
            cfg=cfg,
            subtitle_cb=self._on_subtitle,
            status_cb=self._on_status,
            transcript_cb=self._on_transcript,
        )

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._pipeline_status.configure(text="Starting …", fg=ACCENT)

        for dot in self._status_dots.values():
            dot.set_state("idle")

        self._pipeline.start()

        llm_model = cfg.groq.model if provider == "groq" else cfg.gemini.model
        self._log(
            f"Pipeline started — {cfg.audio.source_type} → "
            f"Deepgram/{cfg.stt.model} → {provider.capitalize()}/{llm_model}"
        )

    def _stop(self) -> None:
        if self._pipeline:
            threading.Thread(target=self._pipeline.stop, daemon=True).start()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._pipeline_status.configure(text="Stopped", fg=MUTED)
        self._log("Pipeline stopped.")

    # ── Callbacks ─────────────────────────────────────────────

    def _on_subtitle(self, sub: DisplaySubtitle) -> None:
        try:
            self._subtitle_queue.put_nowait(sub)
        except queue.Full:
            pass

    def _on_status(self, module: str, status: str) -> None:
        self._root.after(0, lambda: self._apply_status(module, status))

    def _apply_status(self, module: str, status: str) -> None:
        if module == "Pipeline":
            self._pipeline_status.configure(
                text=status,
                fg=ACCENT if "Running" in status else MUTED,
            )
            return

        # Map provider name to the "Groq" dot (which may be labelled differently)
        lookup = module
        if module in ("Groq", "Gemini"):
            lookup = "Groq"   # both use the same dot slot

        if lookup in self._status_labels:
            self._status_labels[lookup].configure(text=status)
            dot = self._status_dots[lookup]
            if any(k in status for k in ("✔", "Running", "Ready")):
                dot.set_state("ok")
            elif "Error" in status:
                dot.set_state("error")
            elif any(k in status for k in ("Loading", "Starting", "Checking")):
                dot.set_state("warn")
        elif module in self._status_labels:
            self._status_labels[module].configure(text=status)

        self._log(f"[{module}] {status}")

    # ── Subtitle poll ─────────────────────────────────────────

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
