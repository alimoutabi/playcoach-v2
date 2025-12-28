#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from pathlib import Path
import subprocess
import sys

import numpy as np
import sounddevice as sd
from piano_transcription_inference import sample_rate

from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig


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


def make_card(parent: tk.Widget, *, title: str) -> ttk.Frame:
    outer = ttk.Frame(parent, padding=12, style="Card.TFrame")
    ttk.Label(outer, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
    return outer


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PlayCoach v2 – Transcriber")
        self.geometry("980x740")
        self.minsize(880, 620)

        self.audio_path: Path | None = None
        self.outdir: Path = (Path.cwd() / "outputs").resolve()

        # Live mic state
        self.live_running = False
        self.live_stream = None
        self.live_thread = None
        self.buf_lock = threading.Lock()
        self.buffer_sec = 10.0
        self.buf = np.zeros(int(self.buffer_sec * sample_rate), dtype=np.float32)

        # buffer fill count (warm-up)
        self.filled = 0

        # ✅ UI lock to prevent double-click races
        self.ui_lock = False

        self._build_style()
        self._build_layout()
        self._toggle_hop()
        self._toggle_mode_ui()
        self._set_status("Ready.")

        print("[GUI] Started. sample_rate =", sample_rate)
        print("[GUI] Default outdir =", self.outdir)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        self.configure(bg="#0f172a")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", font=("SF Pro Text", 12))
        style.configure("TLabel", background="#0f172a", foreground="#e5e7eb")
        style.configure("TFrame", background="#0f172a")

        style.configure("Header.TFrame", background="#0b1224")
        style.configure("HeaderTitle.TLabel", background="#0b1224", foreground="#ffffff",
                        font=("SF Pro Display", 18, "bold"))
        style.configure("HeaderSub.TLabel", background="#0b1224", foreground="#cbd5e1")

        style.configure("Card.TFrame", background="#111827", relief="flat")
        style.configure("CardTitle.TLabel", background="#111827", foreground="#ffffff",
                        font=("SF Pro Display", 13, "bold"))

        style.configure("TEntry", fieldbackground="#0b1224", foreground="#e5e7eb")
        style.configure("TCombobox", fieldbackground="#0b1224", foreground="#e5e7eb")
        style.map("TCombobox", fieldbackground=[("readonly", "#0b1224")])

        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff",
                        padding=(12, 8), font=("SF Pro Text", 12, "bold"))
        style.map("Primary.TButton",
                  background=[("active", "#1d4ed8"), ("disabled", "#334155")],
                  foreground=[("disabled", "#94a3b8")])

        style.configure("TButton", background="#1f2937", foreground="#e5e7eb", padding=(10, 8))
        style.map("TButton", background=[("active", "#374151")])

        style.configure("Status.TLabel", background="#0b1224", foreground="#cbd5e7eb")

    def _build_layout(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=(16, 14))
        header.pack(fill="x")

        left = ttk.Frame(header, style="Header.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="PlayCoach v2", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(left, text="File transcription + Live microphone mode",
                  style="HeaderSub.TLabel").pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(header, style="Header.TFrame")
        right.pack(side="right")
        ttk.Button(right, text="Open Output Folder", command=lambda: open_folder(self.outdir)).pack()

        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        # Top grid
        grid = ttk.Frame(main)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)

        # Inputs card
        inputs = make_card(grid, title="Inputs")
        inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        row_mode = ttk.Frame(inputs, style="Card.TFrame")
        row_mode.pack(fill="x", pady=(0, 10))

        ttk.Label(row_mode, text="Mode", background="#111827").pack(side="left")
        self.mode_var = tk.StringVar(value="file")
        self.mode_combo = ttk.Combobox(
            row_mode, textvariable=self.mode_var, values=["file", "live"], state="readonly", width=10
        )
        self.mode_combo.pack(side="left", padx=10)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_mode_ui())

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

        # Options card
        opts = make_card(grid, title="Options")
        opts.grid(row=0, column=1, sticky="nsew")

        r1 = ttk.Frame(opts, style="Card.TFrame")
        r1.pack(fill="x", pady=(0, 10))
        ttk.Label(r1, text="Preset", background="#111827").pack(side="left")
        self.preset_var = tk.StringVar(value="clean")
        self.preset_combo = ttk.Combobox(
            r1, textvariable=self.preset_var, values=["raw", "clean", "aggressive"], state="readonly", width=14
        )
        self.preset_combo.pack(side="right")

        r2 = ttk.Frame(opts, style="Card.TFrame")
        r2.pack(fill="x", pady=(0, 10))
        self.chords_var = tk.BooleanVar(value=True)
        self.chk_chords = ttk.Checkbutton(
            r2, text="Write chords (frame-based)", variable=self.chords_var, command=self._toggle_hop
        )
        self.chk_chords.pack(side="left")

        r3 = ttk.Frame(opts, style="Card.TFrame")
        r3.pack(fill="x", pady=(0, 10))
        ttk.Label(r3, text="Hop (sec)", background="#111827").pack(side="left")
        self.hop_var = tk.StringVar(value="0.05")
        self.hop_entry = ttk.Entry(r3, textvariable=self.hop_var, width=8)
        self.hop_entry.pack(side="right")

        # Live controls
        r_live = ttk.Frame(opts, style="Card.TFrame")
        r_live.pack(fill="x", pady=(0, 10))
        self.live_row = r_live

        ttk.Label(r_live, text="Window (sec)", background="#111827").pack(side="left")
        self.window_var = tk.StringVar(value="4.0")
        self.window_entry = ttk.Entry(r_live, textvariable=self.window_var, width=8)
        self.window_entry.pack(side="right")

        r_buttons = ttk.Frame(opts, style="Card.TFrame")
        r_buttons.pack(fill="x")
        self.btn_run = ttk.Button(r_buttons, text="Run (File)", style="Primary.TButton", command=self.run_file)
        self.btn_run.pack(fill="x")

        self.btn_live = ttk.Button(r_buttons, text="Start Live Mic", style="Primary.TButton", command=self.toggle_live)
        self.btn_live.pack(fill="x", pady=(8, 0))

        # ✅ Reset button (as you already had)
        self.btn_reset = ttk.Button(r_buttons, text="Reset", command=self.reset_all)
        self.btn_reset.pack(fill="x", pady=(8, 0))

        # Output card
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

        # Status bar
        status_bar = ttk.Frame(self, style="Header.TFrame", padding=(12, 8))
        status_bar.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").pack(
            side="left", fill="x", expand=True
        )

        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    def _toggle_hop(self):
        self.hop_entry.configure(state=("normal" if self.chords_var.get() else "disabled"))

    def _toggle_mode_ui(self):
        mode = self.mode_var.get()
        is_file = (mode == "file")

        print(f"[GUI] Mode changed -> {mode}")

        self.btn_pick_audio.configure(state=("normal" if is_file else "disabled"))
        self.btn_run.configure(state=("normal" if is_file else "disabled"))

        self.window_entry.configure(state=("normal" if not is_file else "disabled"))
        self.btn_live.configure(state=("normal" if not is_file else "disabled"))

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _set_busy(self, busy: bool):
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    # ✅ UI lock: extended a tiny bit to also cover file-run clicks
    def _lock_ui_temporarily(self, ms: int = 250):
        self.ui_lock = True
        try:
            self.btn_live.configure(state="disabled")
            self.btn_reset.configure(state="disabled")
            self.btn_run.configure(state="disabled")
        except Exception:
            pass

        def unlock():
            self.ui_lock = False
            try:
                is_live_mode = (self.mode_var.get() == "live")
                is_file_mode = (self.mode_var.get() == "file")

                self.btn_live.configure(state=("normal" if is_live_mode else "disabled"))
                self.btn_run.configure(state=("normal" if is_file_mode else "disabled"))
                self.btn_reset.configure(state="normal")
            except Exception:
                pass

        self.after(ms, unlock)

    # ✅ Reset action
    def reset_all(self):
        if self.ui_lock:
            return
        print("[GUI] Reset pressed")
        self._lock_ui_temporarily(250)

        if self.live_running:
            print("[GUI] Reset: stopping live mode first…")
            self.stop_live()

        self.notes_box.delete("1.0", "end")
        self.chords_box.delete("1.0", "end")
        self.notes_box.insert("end", "Notes output will appear here…\n")
        self.chords_box.insert("end", "Chords output will appear here…\n")

        with self.buf_lock:
            self.buf[:] = 0.0
            self.filled = 0

        self._set_busy(False)
        self._set_status("Reset ✅ (ready for new run)")
        print("[GUI] Reset done")

    def pick_audio(self):
        print("[GUI] Choose Audio clicked")
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a"), ("All files", "*.*")]
        )
        if path:
            self.audio_path = Path(path).expanduser().resolve()
            self.audio_label.config(text=str(self.audio_path))
            print("[GUI] Selected audio:", self.audio_path)

    def pick_outdir(self):
        print("[GUI] Choose Output Folder clicked")
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.outdir = Path(folder).expanduser().resolve()
            self.outdir_label.config(text=str(self.outdir))
            print("[GUI] Selected outdir:", self.outdir)

    # --------------------
    # File mode
    # --------------------
    def run_file(self):
        if self.ui_lock:
            return
        self._lock_ui_temporarily(250)

        if not self.audio_path or not self.audio_path.exists():
            messagebox.showerror("Missing file", "Please choose an audio file first.")
            return

        write_chords = bool(self.chords_var.get())
        try:
            hop = float(self.hop_var.get()) if write_chords else 0.05
            if write_chords and hop <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid hop", "Hop must be a positive number (e.g. 0.05).")
            return

        print("[GUI] Run (File) starting…")
        print("      preset =", self.preset_var.get())
        print("      write_chords =", write_chords, "hop =", hop)
        print("      audio_path =", self.audio_path)
        print("      outdir =", self.outdir)

        self._set_busy(True)
        self._set_status("Running file transcription…")

        def job():
            try:
                filter_cfg = filter_cfg_from_preset(self.preset_var.get())
                frame_cfg = FrameConfig(write_chords=write_chords, frame_hop=hop)
                self.outdir.mkdir(parents=True, exist_ok=True)

                app = TranscriptionApp(
                    filter_cfg=filter_cfg, frame_cfg=frame_cfg,
                    print_raw=False, print_audio_info=False
                )

                stem = self.audio_path.stem
                app.run(audio_path=self.audio_path, outdir=self.outdir, stem=stem)

                self.after(0, lambda: self._load_outputs(stem, write_chords))
                self.after(0, lambda: self._set_status("Done ✅"))
                self.after(0, lambda: messagebox.showinfo("Done", "File transcription finished ✅"))
                print("[GUI] Run (File) done ✅")

            except Exception as e:
                print("[GUI] Run (File) error:", repr(e))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self._set_status("Error."))

            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=job, daemon=True).start()

    def _load_outputs(self, stem: str, write_chords: bool):
        print("[GUI] Loading output files for stem =", stem)
        notes_txt = self.outdir / f"{stem}_notes.txt"
        chords_txt = self.outdir / f"{stem}_chords.txt"

        notes_content = notes_txt.read_text(encoding="utf-8") if notes_txt.exists() else "No notes txt created."
        if write_chords and chords_txt.exists():
            chords_content = chords_txt.read_text(encoding="utf-8")
        else:
            chords_content = "(Chords disabled or chords file not created.)"

        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("end", notes_content)

        self.chords_box.delete("1.0", "end")
        self.chords_box.insert("end", chords_content)

    # --------------------
    # Live mic mode
    # --------------------
    def _audio_callback(self, indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32)
        with self.buf_lock:
            n = len(x)
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = x
            self.filled = min(len(self.buf), self.filled + n)

    def toggle_live(self):
        if self.ui_lock:
            return
        self._lock_ui_temporarily(250)

        if self.live_running:
            print("[GUI] Stop Live Mic clicked")
            self.stop_live()
        else:
            print("[GUI] Start Live Mic clicked")
            self.start_live()

    def start_live(self):
        try:
            window_sec = float(self.window_var.get())
            if window_sec <= 0:
                raise ValueError("Window must be > 0")
            hop = float(self.hop_var.get())
            if hop <= 0:
                raise ValueError("Hop must be > 0")
        except Exception as e:
            messagebox.showerror("Bad live settings", str(e))
            return

        self.outdir.mkdir(parents=True, exist_ok=True)

        with self.buf_lock:
            self.filled = 0
            self.buf[:] = 0.0

        print("[LIVE] Starting stream…")
        print("       window_sec =", window_sec)
        print("       update interval ~0.5s")
        print("       outdir =", self.outdir)

        self.live_running = True
        self.btn_live.configure(text="Stop Live Mic")
        self._set_status("Listening… (warming up mic buffer)")
        self._set_busy(True)

        self.live_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.live_stream.start()

        def worker():
            while self.live_running:
                t_start = time.time()

                with self.buf_lock:
                    n = int(window_sec * sample_rate)
                    audio = None if self.filled < n else self.buf[-n:].copy()

                if audio is None:
                    self.after(0, lambda: self._set_status("Listening… (warming up mic buffer)"))
                    time.sleep(0.2)
                    continue

                rms = float(np.sqrt(np.mean(audio * audio)))
                if rms < 0.005:
                    self.after(0, lambda: self._set_status("Listening… (too quiet — play louder / closer to mic)"))
                    time.sleep(0.2)
                    continue

                filter_cfg = filter_cfg_from_preset(self.preset_var.get())
                frame_cfg = FrameConfig(
                    write_chords=bool(self.chords_var.get()),
                    frame_hop=float(self.hop_var.get()),
                )

                try:
                    app = TranscriptionApp(
                        filter_cfg=filter_cfg,
                        frame_cfg=frame_cfg,
                        print_raw=False,
                        print_audio_info=False,
                    )

                    stem = "live"
                    print(f"[LIVE] Transcribing window… rms={rms:.3f}, samples={len(audio)}")
                    app.run_audio(audio, outdir=self.outdir, stem=stem)

                    notes_path = self.outdir / f"{stem}_notes.txt"
                    chords_path = self.outdir / f"{stem}_chords.txt"

                    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(no notes)"
                    chords = chords_path.read_text(encoding="utf-8") if chords_path.exists() else "(no chords)"

                    self.after(0, lambda n=notes, c=chords: self._show_live(n, c))
                    self.after(0, lambda: self._set_status(f"Listening… (rms={rms:.3f})"))

                except Exception as e:
                    print("[LIVE] ERROR:", repr(e))
                    self.after(0, lambda: messagebox.showerror("Live error", str(e)))
                    self.after(0, self.stop_live)
                    return

                elapsed = time.time() - t_start
                time.sleep(max(0.0, 0.5 - elapsed))

        self.live_thread = threading.Thread(target=worker, daemon=True)
        self.live_thread.start()

    def _show_live(self, notes: str, chords: str):
        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("end", notes)

        self.chords_box.delete("1.0", "end")
        self.chords_box.insert("end", chords)

    def stop_live(self):
        print("[LIVE] Stopping stream…")
        self.live_running = False
        self.btn_live.configure(text="Start Live Mic")
        self._set_busy(False)
        self._set_status("Live stopped.")

        if self.live_stream:
            try:
                self.live_stream.stop()
                self.live_stream.close()
            except Exception:
                pass
            self.live_stream = None

        print("[LIVE] Stopped ✅")

    def _on_close(self):
        print("[GUI] Closing…")
        try:
            self.stop_live()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()