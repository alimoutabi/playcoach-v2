#!/usr/bin/env python3
"""
Modern Tkinter GUI for PlayCoach v2 (OOP transcription pipeline).

What’s improved vs. the basic GUI:
- Uses ttk widgets (cleaner look)
- Header bar + “card” panels
- Better spacing + consistent typography
- Status bar + indeterminate progress bar
- Runs transcription in a background thread (UI stays responsive)
- “Open output folder” button
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from pathlib import Path
import subprocess
import sys

from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig


# -----------------------------
# Preset mapping (same as yours)
# -----------------------------
def filter_cfg_from_preset(preset: str) -> FilterConfig:
    preset = preset.lower().strip()

    if preset == "raw":
        return FilterConfig(enable_A=False, enable_B=False, enable_D=False)

    if preset == "clean":
        return FilterConfig(
            enable_A=False, enable_B=True, enable_D=True,
            cluster_window=0.04, dedupe_window=0.08,
            max_notes_per_cluster=6, harmonic_velocity_ratio=1.15,
        )

    if preset == "aggressive":
        return FilterConfig(
            enable_A=True, enable_B=True, enable_D=True,
            cluster_window=0.04, dedupe_window=0.08,
            max_notes_per_cluster=5, harmonic_velocity_ratio=1.10,
            min_occurrences=2, min_total_dur_ratio_of_max=0.10,
        )

    raise ValueError("preset must be raw|clean|aggressive")


# -----------------------------
# Small OS helpers
# -----------------------------
def open_folder(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        return
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


# -----------------------------
# UI helpers
# -----------------------------
def make_card(parent: tk.Widget, *, title: str) -> ttk.Frame:
    """A simple 'card' style frame with a title label."""
    outer = ttk.Frame(parent, padding=12, style="Card.TFrame")
    lbl = ttk.Label(outer, text=title, style="CardTitle.TLabel")
    lbl.pack(anchor="w", pady=(0, 8))
    return outer


# -----------------------------
# App
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PlayCoach v2 – Transcriber")
        self.geometry("980x720")
        self.minsize(880, 620)

        self.audio_path: Path | None = None
        self.outdir: Path = (Path.cwd() / "outputs").resolve()

        self._build_style()
        self._build_layout()

        self._toggle_hop()
        self._set_status("Ready.")

    # ---- Styling
    def _build_style(self):
        self.configure(bg="#0f172a")  # deep slate background

        style = ttk.Style(self)
        # On macOS, "aqua" is fine; on others, "clam" looks clean.
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Base
        style.configure(".", font=("SF Pro Text", 12))
        style.configure("TLabel", background="#0f172a", foreground="#e5e7eb")
        style.configure("TFrame", background="#0f172a")

        # Header
        style.configure("Header.TFrame", background="#0b1224")
        style.configure("HeaderTitle.TLabel", background="#0b1224", foreground="#ffffff", font=("SF Pro Display", 18, "bold"))
        style.configure("HeaderSub.TLabel", background="#0b1224", foreground="#cbd5e1")

        # Cards
        style.configure("Card.TFrame", background="#111827", relief="flat")
        style.configure("CardTitle.TLabel", background="#111827", foreground="#ffffff", font=("SF Pro Display", 13, "bold"))

        # Inputs
        style.configure("TEntry", fieldbackground="#0b1224", foreground="#e5e7eb")
        style.configure("TCombobox", fieldbackground="#0b1224", foreground="#e5e7eb")
        style.map("TCombobox", fieldbackground=[("readonly", "#0b1224")])

        # Buttons
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff", padding=(12, 8), font=("SF Pro Text", 12, "bold"))
        style.map("Primary.TButton",
                  background=[("active", "#1d4ed8"), ("disabled", "#334155")],
                  foreground=[("disabled", "#94a3b8")])

        style.configure("TButton", background="#1f2937", foreground="#e5e7eb", padding=(10, 8))
        style.map("TButton", background=[("active", "#374151")])

        # Status
        style.configure("Status.TLabel", background="#0b1224", foreground="#cbd5e1")

    # ---- Layout
    def _build_layout(self):
        # Header bar
        header = ttk.Frame(self, style="Header.TFrame", padding=(16, 14))
        header.pack(fill="x")

        left = ttk.Frame(header, style="Header.TFrame")
        left.pack(side="left", fill="x", expand=True)

        ttk.Label(left, text="PlayCoach v2", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(left, text="Transcribe piano audio → notes.txt (+ optional chords.txt)", style="HeaderSub.TLabel").pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(header, style="Header.TFrame")
        right.pack(side="right")

        self.btn_open_out = ttk.Button(right, text="Open Output Folder", command=lambda: open_folder(self.outdir))
        self.btn_open_out.pack()

        # Main container
        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        # Top row: Inputs + Options
        grid = ttk.Frame(main)
        grid.pack(fill="x")

        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)

        # --- Inputs card
        inputs = make_card(grid, title="Inputs")
        inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        row1 = ttk.Frame(inputs, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, 8))

        self.btn_pick_audio = ttk.Button(row1, text="Choose Audio…", command=self.pick_audio)
        self.btn_pick_audio.pack(side="left")

        self.audio_label = ttk.Label(row1, text="No file selected", background="#111827", foreground="#cbd5e1")
        self.audio_label.pack(side="left", padx=10, fill="x", expand=True)

        row2 = ttk.Frame(inputs, style="Card.TFrame")
        row2.pack(fill="x")

        self.btn_pick_out = ttk.Button(row2, text="Choose Output Folder…", command=self.pick_outdir)
        self.btn_pick_out.pack(side="left")

        self.outdir_label = ttk.Label(row2, text=f"{self.outdir}", background="#111827", foreground="#cbd5e1")
        self.outdir_label.pack(side="left", padx=10, fill="x", expand=True)

        # --- Options card
        opts = make_card(grid, title="Options")
        opts.grid(row=0, column=1, sticky="nsew")

        r1 = ttk.Frame(opts, style="Card.TFrame")
        r1.pack(fill="x", pady=(0, 10))
        ttk.Label(r1, text="Preset", background="#111827").pack(side="left")

        self.preset_var = tk.StringVar(value="clean")
        self.preset_combo = ttk.Combobox(r1, textvariable=self.preset_var, values=["raw", "clean", "aggressive"], state="readonly", width=14)
        self.preset_combo.pack(side="right")

        r2 = ttk.Frame(opts, style="Card.TFrame")
        r2.pack(fill="x", pady=(0, 10))
        self.chords_var = tk.BooleanVar(value=True)
        self.chk_chords = ttk.Checkbutton(r2, text="Write chords (frame-based)", variable=self.chords_var, command=self._toggle_hop)
        self.chk_chords.pack(side="left")

        r3 = ttk.Frame(opts, style="Card.TFrame")
        r3.pack(fill="x", pady=(0, 12))
        ttk.Label(r3, text="Hop (sec)", background="#111827").pack(side="left")

        self.hop_var = tk.StringVar(value="0.05")
        self.hop_entry = ttk.Entry(r3, textvariable=self.hop_var, width=8)
        self.hop_entry.pack(side="right")

        r4 = ttk.Frame(opts, style="Card.TFrame")
        r4.pack(fill="x")

        self.btn_run = ttk.Button(r4, text="Run Transcription", style="Primary.TButton", command=self.run_transcription)
        self.btn_run.pack(fill="x")

        # Middle: Tabs for outputs
        out_card = make_card(main, title="Output")
        out_card.pack(fill="both", expand=True, pady=(16, 0))

        self.tabs = ttk.Notebook(out_card)
        self.tabs.pack(fill="both", expand=True)

        notes_tab = ttk.Frame(self.tabs)
        chords_tab = ttk.Frame(self.tabs)
        self.tabs.add(notes_tab, text="Notes")
        self.tabs.add(chords_tab, text="Chords")

        self.notes_box = ScrolledText(notes_tab, height=18, wrap="none")
        self.notes_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.notes_box.insert("end", "Notes output will appear here…\n")

        self.chords_box = ScrolledText(chords_tab, height=14, wrap="none")
        self.chords_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.chords_box.insert("end", "Chords output will appear here…\n")

        # Bottom status bar
        status_bar = ttk.Frame(self, style="Header.TFrame", padding=(12, 8))
        status_bar.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left", fill="x", expand=True)

        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    # ---- UI actions
    def _toggle_hop(self):
        state = "normal" if self.chords_var.get() else "disabled"
        self.hop_entry.configure(state=state)

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _set_running(self, running: bool):
        if running:
            self.btn_run.configure(state="disabled")
            self.btn_pick_audio.configure(state="disabled")
            self.btn_pick_out.configure(state="disabled")
            self.preset_combo.configure(state="disabled")
            self.chk_chords.configure(state="disabled")
            self.hop_entry.configure(state="disabled")
            self.progress.start(12)
        else:
            self.btn_run.configure(state="normal")
            self.btn_pick_audio.configure(state="normal")
            self.btn_pick_out.configure(state="normal")
            self.preset_combo.configure(state="readonly")
            self.chk_chords.configure(state="normal")
            self._toggle_hop()
            self.progress.stop()

    def pick_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a"), ("All files", "*.*")]
        )
        if path:
            self.audio_path = Path(path).expanduser().resolve()
            self.audio_label.config(text=str(self.audio_path))

    def pick_outdir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.outdir = Path(folder).expanduser().resolve()
            self.outdir_label.config(text=str(self.outdir))

    # ---- Worker
    def run_transcription(self):
        if not self.audio_path or not self.audio_path.exists():
            messagebox.showerror("Missing file", "Please choose an audio file first.")
            return

        # Validate hop early
        write_chords = bool(self.chords_var.get())
        try:
            hop = float(self.hop_var.get()) if write_chords else 0.05
            if write_chords and hop <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid hop", "Hop must be a positive number (e.g. 0.05).")
            return

        self._set_running(True)
        self._set_status("Running transcription…")

        def job():
            try:
                preset = self.preset_var.get()
                filter_cfg = filter_cfg_from_preset(preset)

                frame_cfg = FrameConfig(write_chords=write_chords, frame_hop=hop)

                self.outdir.mkdir(parents=True, exist_ok=True)

                app = TranscriptionApp(
                    filter_cfg=filter_cfg,
                    frame_cfg=frame_cfg,
                    print_raw=False,
                    print_audio_info=False,
                )

                stem = self.audio_path.stem
                app.run(audio_path=self.audio_path, outdir=self.outdir, stem=stem)

                notes_txt = self.outdir / f"{stem}_notes.txt"
                chords_txt = self.outdir / f"{stem}_chords.txt"

                notes_content = notes_txt.read_text(encoding="utf-8") if notes_txt.exists() else "No notes txt created."
                if write_chords and chords_txt.exists():
                    chords_content = chords_txt.read_text(encoding="utf-8")
                else:
                    chords_content = "(Chords disabled or chords file not created.)"

                # Update UI from main thread
                self.after(0, lambda: self._show_results(notes_content, chords_content))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self._set_status("Error."))
                self.after(0, lambda: self._set_running(False))

        threading.Thread(target=job, daemon=True).start()

    def _show_results(self, notes_content: str, chords_content: str):
        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("end", notes_content)

        self.chords_box.delete("1.0", "end")
        self.chords_box.insert("end", chords_content)

        self._set_running(False)
        self._set_status("Done ✅")
        messagebox.showinfo("Done", "Transcription finished successfully ✅")


if __name__ == "__main__":
    App().mainloop()