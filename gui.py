#!/usr/bin/env python3
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageTk
import numpy as np
import sounddevice as sd
from piano_transcription_inference import sample_rate
from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig
from PIL import Image, ImageTk
from transcribe.sheet_render import render_grand_staff_from_notes_txt
import winsound
import time
import threading

class MetronomeEngine:
    def __init__(self):
        self.running = False
        self.bpm = 120

    def start(self, bpm, tick_callback):
        self.bpm = bpm
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(tick_callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self, tick_callback):
        while self.running:
            winsound.Beep(1000, 50) 
            tick_callback()
            time.sleep(60.0 / self.bpm)




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
        self.buf_lock = threading.Lock()



        # ✅ record-from-start buffer (no window)
        self.recorded_chunks: list[np.ndarray] = []

        # ✅ UI lock to prevent double-click races
        self.ui_lock = False

        # ✅ Keep reference to Tk image to prevent GC
        self._sheet_imgtk = None

        self._build_style()
        self._build_layout()
        self._toggle_hop()
        self._toggle_mode_ui()
        self._set_status("Ready.")

        print("[GUI] Started. sample_rate =", sample_rate)
        print("[GUI] Default outdir =", self.outdir)

        self.protocol("WM_DELETE_WINDOW", self._on_close)


    def toggle_metronome(self):
        if not self.metro_engine.running:
            try:
                bpm = int(self.bpm_var.get())
                if bpm < 30 or bpm > 300: raise ValueError
                self.metro_engine.start(bpm, self._metro_tick)
                self.btn_metro.configure(text="Stop")
            except ValueError:
                messagebox.showwarning("BPM Error", "Bitte eine Zahl zwischen 30 und 300 eingeben.")
        else:
            self.metro_engine.stop()
            self.btn_metro.configure(text="Start")
            self.metro_canvas.itemconfig(self.metro_light, fill="#374151")

    def _metro_tick(self):
        # Visueller Effekt: Kurz aufleuchten (Blau)
        self.after(0, lambda: self.metro_canvas.itemconfig(self.metro_light, fill="#2563eb"))
        # Nach 150ms wieder abdunkeln
        self.after(150, lambda: self.metro_canvas.itemconfig(self.metro_light, fill="#374151"))
        
        # Optional: Ein kurzer System-Sound
        # self.bell() # Einfachster Weg für einen Klick-Ton


    


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



     ######### Metronome ##########
        self.metro_engine = MetronomeEngine()
        r_metro = ttk.Frame(opts, style="Card.TFrame")
        r_metro.pack(fill="x", pady=(10, 0))
        ttk.Label(r_metro, text="Metronom (BPM)", background="#111827").pack(side="left")
        self.bpm_var = tk.StringVar(value="120")
        self.bpm_entry = ttk.Entry(r_metro, textvariable=self.bpm_var, width=6)
        self.bpm_entry.pack(side="left", padx=5)
        self.metro_canvas = tk.Canvas(r_metro, width=20, height=20, bg="#111827", highlightthickness=0)
        self.metro_canvas.pack(side="left", padx=10)
        self.metro_light = self.metro_canvas.create_oval(4, 4, 16, 16, fill="#374151")
        self.btn_metro = ttk.Button(r_metro, text="Start", width=8, command=self.toggle_metronome)
        self.btn_metro.pack(side="right")






        # ✅ Window removed (no live controls row)
        self.live_row = None

        r_buttons = ttk.Frame(opts, style="Card.TFrame")
        r_buttons.pack(fill="x")
        self.btn_run = ttk.Button(r_buttons, text="Run (File)", style="Primary.TButton", command=self.run_file)
        self.btn_run.pack(fill="x")

        self.btn_live = ttk.Button(r_buttons, text="Start Live Mic", style="Primary.TButton", command=self.toggle_live)
        self.btn_live.pack(fill="x", pady=(8, 0))

        self.btn_reset = ttk.Button(r_buttons, text="Reset", command=self.reset_all)
        self.btn_reset.pack(fill="x", pady=(8, 0))

        # Output card
        out_card = make_card(main, title="Output")
        out_card.pack(fill="both", expand=True, pady=(16, 0))

        self.tabs = ttk.Notebook(out_card)
        self.tabs.pack(fill="both", expand=True)

        notes_tab = ttk.Frame(self.tabs)
        chords_tab = ttk.Frame(self.tabs)
        sheet_tab = ttk.Frame(self.tabs)

        self.tabs.add(notes_tab, text="Notes")
        self.tabs.add(chords_tab, text="Chords")
        self.tabs.add(sheet_tab, text="Sheet")

        self.notes_box = ScrolledText(notes_tab, height=18, wrap="none")
        self.notes_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.notes_box.insert("end", "Notes output will appear here…\n")

        self.chords_box = ScrolledText(chords_tab, height=14, wrap="none")
        self.chords_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.chords_box.insert("end", "Chords output will appear here…\n")

        # ✅ Sheet view
        self.sheet_label = ttk.Label(sheet_tab)
        self.sheet_label.pack(fill="both", expand=True, padx=8, pady=8)

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

        self.btn_live.configure(state=("normal" if not is_file else "disabled"))

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _set_busy(self, busy: bool):
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

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

    # ✅ Render sheet from notes text -> PNG -> show in tab
    def _update_sheet_from_notes_txt(self, notes_txt: str):
        try:
            img = render_grand_staff_from_notes_txt(notes_txt)

            # Fit image into current sheet tab size
            w = max(300, self.sheet_label.winfo_width())
            h = max(200, self.sheet_label.winfo_height())

            img = img.copy()
            img.thumbnail((w - 20, h - 20))  # keep aspect ratio

            self._sheet_imgtk = ImageTk.PhotoImage(img)
            self.sheet_label.configure(image=self._sheet_imgtk, text="")
        except Exception as e:
            self.sheet_label.configure(text=f"Sheet render error: {e}", image="")
            self._sheet_imgtk = None

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

        self.sheet_label.configure(image="", text="")
        self._sheet_imgtk = None

        with self.buf_lock:
            self.recorded_chunks.clear()

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

        # ✅ Update sheet tab from notes txt
        self._update_sheet_from_notes_txt(notes_content)

    # --------------------
    # Live mic mode (no window)
    # Start = record everything
    # Stop  = stop recording and analyze once
    # --------------------
    def _audio_callback(self, indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32).copy()
        with self.buf_lock:
            self.recorded_chunks.append(x)

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
        self.outdir.mkdir(parents=True, exist_ok=True)

        with self.buf_lock:
            self.recorded_chunks.clear()

        print("[LIVE] Starting stream (record-from-start)…")
        print("       outdir =", self.outdir)

        self.live_running = True
        self.btn_live.configure(text="Stop + Analyze")
        self._set_status("Listening… (records from Start; press Stop + Analyze when done)")
        self._set_busy(True)

        self.live_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.live_stream.start()

    def _show_live(self, notes: str, chords: str):
        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("end", notes)

        self.chords_box.delete("1.0", "end")
        self.chords_box.insert("end", chords)

    def stop_live(self):
        print("[LIVE] Stopping stream…")
        self.live_running = False
        self.btn_live.configure(text="Start Live Mic")

        if self.live_stream:
            try:
                self.live_stream.stop()
                self.live_stream.close()
            except Exception:
                pass
            self.live_stream = None

        self._set_status("Analyzing last take…")
        self._set_busy(True)

        with self.buf_lock:
            if not self.recorded_chunks:
                audio = np.zeros(0, dtype=np.float32)
            else:
                audio = np.concatenate(self.recorded_chunks, axis=0)

        def job():
            try:
                if audio.size == 0 or len(audio) < int(0.2 * sample_rate):
                    empty_notes = "Filtered notes\n\n(No audio captured — press Start and play a bit)\n"
                    empty_chords = "Chord segments (frame-based)\n\n(No audio captured)\n"
                    self.after(0, lambda: self._show_live(empty_notes, empty_chords))
                    self.after(0, lambda: self._update_sheet_from_notes_txt(empty_notes))
                    self.after(0, lambda: self._set_status("Done ✅ (no audio)"))
                    return

                filter_cfg = filter_cfg_from_preset(self.preset_var.get())
                frame_cfg = FrameConfig(
                    write_chords=bool(self.chords_var.get()),
                    frame_hop=float(self.hop_var.get()),
                )

                app = TranscriptionApp(
                    filter_cfg=filter_cfg,
                    frame_cfg=frame_cfg,
                    print_raw=False,
                    print_audio_info=False,
                )

                stem = "live"
                rms = float(np.sqrt(np.mean(audio * audio)))
                print(f"[LIVE] Analyzing once… rms={rms:.3f}, samples={len(audio)}")
                app.run_audio(audio, outdir=self.outdir, stem=stem)

                notes_path = self.outdir / f"{stem}_notes.txt"
                chords_path = self.outdir / f"{stem}_chords.txt"

                notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(no notes)"
                chords = chords_path.read_text(encoding="utf-8") if chords_path.exists() else "(no chords)"

                self.after(0, lambda n=notes, c=chords: self._show_live(n, c))
                self.after(0, lambda n=notes: self._update_sheet_from_notes_txt(n))
                self.after(0, lambda: self._set_status("Done ✅"))

            except Exception as e:
                print("[LIVE] ERROR:", repr(e))
                self.after(0, lambda: messagebox.showerror("Live error", str(e)))
                self.after(0, lambda: self._set_status("Error."))
            finally:
                self.after(0, lambda: self._set_busy(False))
                print("[LIVE] Stopped ✅ (analyzed once)")

        threading.Thread(target=job, daemon=True).start()

    def _on_close(self):
        print("[GUI] Closing…")
        try:
            if self.live_running:
                self.stop_live()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()