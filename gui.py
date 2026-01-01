#!/usr/bin/env python3
#from future import annotations

import json
import re
import threading
from pathlib import Path
from collections import defaultdict

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk
from piano_transcription_inference import sample_rate

import subprocess
import sys

from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig

# ✅ Sheet rendering
from transcribe.sheet_render import render_grand_staff_from_notes_txt


# -------------------------
# Presets
# -------------------------
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


# -------------------------
# OS helpers
# -------------------------
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


# -------------------------
# NOTE-ONLY comparison
# - No RH/LH
# - No rhythm validation
# - Order-based matching with lookahead
# - Extra detected notes are ignored
# -------------------------
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_pc(m: int) -> int:
    return int(m) % 12


def pc_to_name(pc: int) -> str:
    return PC_NAMES[int(pc) % 12]


def _extract_expected_blocks(obj) -> dict:
    """
    Accepts multiple possible expected.json formats:
      A) {"events":{"RH":{meas:[{"offset":..., "midis":[...]}]}, "LH":{...}}}
      😎 {"RH":{meas:[(offset,[midis...]), ...]}, "LH":{...}}  (older)
      C) {"events":{"RH":{meas:[(offset,[midis...]), ...]}, ...}} (mixed)
    Returns dict with keys RH/LH (if present), each mapping measure->list of events.
    """
    if isinstance(obj, dict) and "events" in obj and isinstance(obj["events"], dict):
        base = obj["events"]
    else:
        base = obj

    out = {}
    for hand in ("RH", "LH"):
        hand_dict = base.get(hand, {})
        if isinstance(hand_dict, dict):
            out[hand] = hand_dict
    return out


def load_expected_sequence_by_measure(expected_path: Path) -> dict[int, list[int]]:
    """
    Returns: {measure_no: [pc, pc, pc, ...]} in "sheet order"
    - merges RH+LH
    - uses 'offset' only to sort events inside measure (NOT for timing validation)
    - flattens chords: if one event has multiple midis, we add them in sorted pc order
    """
    data = json.loads(expected_path.read_text(encoding="utf-8"))
    blocks = _extract_expected_blocks(data)

    # temp: per measure store list of (offset, [pcs...])
    tmp: dict[int, list[tuple[float, list[int]]]] = defaultdict(list)
    for _, meas_dict in blocks.items():
        if not isinstance(meas_dict, dict):
            continue
        for meas_key, events in meas_dict.items():
            try:
                meas = int(meas_key)
            except Exception:
                continue
            if not isinstance(events, list):
                continue

            for ev in events:
                off = 0.0
                midis = None

                if isinstance(ev, dict):
                    off = float(ev.get("offset", 0.0))
                    midis = ev.get("midis", None)
                elif isinstance(ev, (list, tuple)) and len(ev) >= 2:
                    try:
                        off = float(ev[0])
                    except Exception:
                        off = 0.0
                    midis = ev[1]

                if not midis:
                    continue

                pcs = []
                for m in midis:
                    try:
                        pcs.append(midi_to_pc(int(m)))
                    except Exception:
                        pass

                if pcs:
                    pcs_sorted = sorted(pcs)  # chord: stable order
                    tmp[meas].append((off, pcs_sorted))

    out: dict[int, list[int]] = {}
    for meas, items in tmp.items():
        items.sort(key=lambda x: x[0])  # sort by offset within measure
        seq: list[int] = []
        for _, pcs in items:
            seq.extend(pcs)
        out[meas] = seq

    return out


def parse_notes_txt(notes_txt: str) -> list[dict]:
    """
    Parse notes txt (Filtered notes) into events:
      idx midi name onset offset dur velocity
    We DO NOT validate rhythm; we use onset only for ordering / rough measure splitting.
    """
    evs = []
    for line in notes_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("filtered notes"):
            continue
        if line.lower().startswith("idx"):
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 7:
            continue

        try:
            midi = int(parts[1])
            onset = float(parts[3])
            dur = float(parts[5])
            vel = int(parts[6])
        except Exception:
            continue

        if not (21 <= midi <= 108):
            continue

        evs.append({"midi": midi, "pc": midi_to_pc(midi), "onset": onset, "dur": dur, "vel": vel})
    evs.sort(key=lambda x: x["onset"])
    return evs


def split_detected_into_measure_sequences(
    evs: list[dict],
    *,
    meas_from: int,
    meas_to: int,
    min_velocity: int = 20,
    min_dur: float = 0.05,
) -> dict[int, list[int]]:
    """
    Split recording into N equal time bins (measures) and return per measure
    a sequence of pitch-classes (ordered by onset).
    Extras are fine; we keep them because the matcher will ignore them.
    """
    out: dict[int, list[int]] = {m: [] for m in range(meas_from, meas_to + 1)}
    if not evs:
        return out

    fevs = [e for e in evs if e["vel"] >= min_velocity and e["dur"] >= min_dur]
    if not fevs:
        return out

    t0 = fevs[0]["onset"]
    t1 = fevs[-1]["onset"]
    n_meas = (meas_to - meas_from + 1)

    total = max(1e-6, (t1 - t0))
    seg = total / n_meas

    for e in fevs:
        rel = e["onset"] - t0
        idx = int(rel / seg) if seg > 0 else 0
        if idx < 0:
            idx = 0
        if idx >= n_meas:
            idx = n_meas - 1
        meas = meas_from + idx
        out[meas].append(int(e["pc"]))

    return out


def match_sequence_with_lookahead(
    expected: list[int],
    detected: list[int],
    *,
    lookahead: int = 8,
) -> tuple[list[tuple[int, str, int | None]], int]:
    """
    For each expected pc, scan forward in detected (with lookahead window).
    Returns rows: (expected_pc, status, matched_detected_index_or_None)
    Extra detected notes are automatically ignored.
    Also returns 'used_detected' count (progress in detected).
    """
    rows: list[tuple[int, str, int | None]] = []
    j = 0


    for pc in expected:
        found = None
        # search forward within window
        for k in range(j, min(len(detected), j + lookahead)):
            if detected[k] == pc:
                found = k
                break

        if found is None:
            rows.append((pc, "MISS", None))
        else:
            rows.append((pc, "OK", found))
            j = found + 1

    return rows, j


def build_feedback_table(
    expected_by_meas: dict[int, list[int]],
    detected_by_meas: dict[int, list[int]],
    meas_from: int,
    meas_to: int,
    *,
    lookahead: int = 8,
) -> str:
    """
    Print like the older style: one line per expected note.
    """
    lines = [
        f"Feedback – measures {meas_from}..{meas_to}",
        "Rule: order-based matching (no rhythm). Extra detected notes are ignored (lookahead).",
        "",
        "meas\t#\texpected\tstatus\t(found@idx)"
    ]

    ok = 0
    total = 0

    for m in range(meas_from, meas_to + 1):
        exp = expected_by_meas.get(m, [])
        det = detected_by_meas.get(m, [])
        if not exp:
            continue

        rows, _ = match_sequence_with_lookahead(exp, det, lookahead=lookahead)

        for i, (pc, status, found_idx) in enumerate(rows, start=1):
            total += 1
            if status == "OK":
                ok += 1

            found_str = "-" if found_idx is None else str(found_idx)
            lines.append(f"{m}\t{i}\t{pc_to_name(pc)}\t{status}\t{found_str}")

    lines.append("")
    lines.append(f"OK: {ok}/{total}")
    return "\n".join(lines)


# -------------------------
# GUI App
# -------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PlayCoach v2 – Transcriber")
        self.geometry("980x740")
        self.minsize(880, 620)

        self.audio_path: Path | None = None
        self.outdir: Path = (Path.cwd() / "outputs").resolve()

        # ✅ expected.json (sheet reference)
        self.expected_path: Path = (Path.cwd() / "outputs" / "expected.json").resolve()
        self.measure_from_var = tk.StringVar(value="1")
        self.measure_to_var = tk.StringVar(value="5")

        # Live mic state
        self.live_running = False
        self.live_stream = None
        self.buf_lock = threading.Lock()

        # record-from-start buffer (no window)
        self.recorded_chunks: list[np.ndarray] = []

        # UI lock to prevent double-click races
        self.ui_lock = False

        # Keep reference to Tk image to prevent GC
        self._sheet_imgtk = None

        self._build_style()
        self._build_layout()
        self._toggle_hop()
        self._toggle_mode_ui()
        self._set_status("Ready.")

        print("[GUI] Started. sample_rate =", sample_rate)
        print("[GUI] Default outdir =", self.outdir)
        print("[GUI] expected.json =", self.expected_path)

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

        grid = ttk.Frame(main)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)

        # Inputs
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

        # expected.json picker
        row3 = ttk.Frame(inputs, style="Card.TFrame")
        row3.pack(fill="x", pady=(8, 0))
        self.btn_pick_expected = ttk.Button(row3, text="Choose expected.json…", command=self.pick_expected)
        self.btn_pick_expected.pack(side="left")
        self.expected_label = ttk.Label(
            row3, text=str(self.expected_path), background="#111827", foreground="#cbd5e1"
        )
        self.expected_label.pack(side="left", padx=10, fill="x", expand=True)

        # Options
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

        # ✅ measures selection (FIXED ORDER using grid)
        r4 = ttk.Frame(opts, style="Card.TFrame")
        r4.pack(fill="x", pady=(0, 10))
        ttk.Label(r4, text="Measures (from..to)", background="#111827").grid(row=0, column=0, sticky="w")

        self.meas_from_entry = ttk.Entry(r4, textvariable=self.measure_from_var, width=6)
        self.meas_from_entry.grid(row=0, column=1, padx=(10, 6), sticky="e")

        ttk.Label(r4, text="to", background="#111827").grid(row=0, column=2, padx=(0, 6))

        self.meas_to_entry = ttk.Entry(r4, textvariable=self.measure_to_var, width=6)
        self.meas_to_entry.grid(row=0, column=3, sticky="e")

        r4.columnconfigure(0, weight=1)

        # Buttons
        r_buttons = ttk.Frame(opts, style="Card.TFrame")
        r_buttons.pack(fill="x")
        self.btn_run = ttk.Button(r_buttons, text="Run (File)", style="Primary.TButton", command=self.run_file)
        self.btn_run.pack(fill="x")

        self.btn_live = ttk.Button(r_buttons, text="Start Live Mic", style="Primary.TButton", command=self.toggle_live)
        self.btn_live.pack(fill="x", pady=(8, 0))

        self.btn_reset = ttk.Button(r_buttons, text="Reset", command=self.reset_all)
        self.btn_reset.pack(fill="x", pady=(8, 0))

        # Output
        out_card = make_card(main, title="Output")
        out_card.pack(fill="both", expand=True, pady=(16, 0))

        self.tabs = ttk.Notebook(out_card)
        self.tabs.pack(fill="both", expand=True)

        notes_tab = ttk.Frame(self.tabs)
        chords_tab = ttk.Frame(self.tabs)
        sheet_tab = ttk.Frame(self.tabs)
        feedback_tab = ttk.Frame(self.tabs)

        self.tabs.add(notes_tab, text="Notes")
        self.tabs.add(chords_tab, text="Chords")
        self.tabs.add(sheet_tab, text="Sheet")
        self.tabs.add(feedback_tab, text="Feedback")

        self.notes_box = ScrolledText(notes_tab, height=18, wrap="none")
        self.notes_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.notes_box.insert("end", "Notes output will appear here…\n")

        self.chords_box = ScrolledText(chords_tab, height=14, wrap="none")
        self.chords_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.chords_box.insert("end", "Chords output will appear here…\n")

        self.sheet_label = ttk.Label(sheet_tab)
        self.sheet_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.feedback_box = ScrolledText(feedback_tab, height=14, wrap="none")
        self.feedback_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.feedback_box.insert("end", "Feedback will appear here…\n")

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

    def pick_expected(self):
        path = filedialog.askopenfilename(
            title="Select expected.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.expected_path = Path(path).expanduser().resolve()
            self.expected_label.config(text=str(self.expected_path))
            print("[GUI] expected.json =", self.expected_path)

    def _update_sheet_from_notes_txt(self, notes_txt: str):
        try:
            img = render_grand_staff_from_notes_txt(notes_txt)

            w = max(300, self.sheet_label.winfo_width())
            h = max(200, self.sheet_label.winfo_height())

            img = img.copy()
            img.thumbnail((w - 20, h - 20))

            self._sheet_imgtk = ImageTk.PhotoImage(img)
            self.sheet_label.configure(image=self._sheet_imgtk, text="")
        except Exception as e:
            self.sheet_label.configure(text=f"Sheet render error: {e}", image="")
            self._sheet_imgtk = None

    def _run_compare_and_show(self, notes_txt: str):
        if not self.expected_path.exists():
            self.feedback_box.delete("1.0", "end")
            self.feedback_box.insert("end", f"expected.json not found:\n{self.expected_path}\n")
            return

        try:
            m_from = int(self.measure_from_var.get())
            m_to = int(self.measure_to_var.get())
            if m_from <= 0 or m_to < m_from:
                raise ValueError
        except Exception:
            self.feedback_box.delete("1.0", "end")
            self.feedback_box.insert("end", "Invalid measures range. Use e.g. 1..5\n")
            return

        try:
            exp_seq_by_meas = load_expected_sequence_by_measure(self.expected_path)
        except Exception as e:
            self.feedback_box.delete("1.0", "end")
            self.feedback_box.insert("end", f"Could not read expected.json:\n{e}\n")
            return

        exp_sel = {m: exp_seq_by_meas.get(m, []) for m in range(m_from, m_to + 1)}
        if all(len(v) == 0 for v in exp_sel.values()):
            self.feedback_box.delete("1.0", "end")
            self.feedback_box.insert("end", f"No expected notes found for measures {m_from}..{m_to}\n")
            return
        
        evs = parse_notes_txt(notes_txt)
        det_seq_by_meas = split_detected_into_measure_sequences(
            evs, 
            meas_from=m_from, 
            meas_to=m_to
        )

        # Helper function to convert MIDI pitch to Note Name
        def get_note_name(midi_pitch):
            if midi_pitch is None: return "N/A"
            # International Note Names
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            letter = names[midi_pitch % 12]
            octave = (midi_pitch // 12) - 1
            return f"{letter}{octave}"

        # --- UPDATED LOGIC (ENGLISH & NOTE NAMES) ---
        feedback_lines = []
        skipped_notes = []

        for m in range(m_from, m_to + 1):
            exp_notes = exp_sel.get(m, [])
            det_notes = det_seq_by_meas.get(m, [])
            
            i, j = 0, 0  # i = expected, j = detected
            while i < len(exp_notes) and j < len(det_notes):
                # Resolve Pitches
                e_item = exp_notes[i]
                e_pitch = e_item['pitch'] if isinstance(e_item, dict) else e_item
                
                d_item = det_notes[j]
                d_pitch = d_item['pitch'] if isinstance(d_item, dict) else d_item
                
                e_name = get_note_name(e_pitch)
                d_name = get_note_name(d_pitch)

                # 1. Note is Correct
                if e_pitch == d_pitch:
                    feedback_lines.append(f"Measure {m}: Note {e_name} [OK]")
                    i += 1
                    j += 1
                # 2. SKIP DETECTION: Is the next expected note the one played?
                elif i + 1 < len(exp_notes):
                    next_item = exp_notes[i+1]
                    next_pitch = next_item['pitch'] if isinstance(next_item, dict) else next_item
                    
                    if next_pitch == d_pitch:
                        feedback_lines.append(f"Measure {m}: Note {e_name} [SKIPPED!]")
                        skipped_notes.append(f"Measure {m}: {e_name}")
                        i += 2  # Sync to next note
                        j += 1
                    else:
                        feedback_lines.append(f"Measure {m}: Wrong (Expected:{e_name} Played:{d_name})")
                        i += 1
                        j += 1
                else:
                    feedback_lines.append(f"Measure {m}: Wrong (Expected:{e_name} Played:{d_name})")
                    i += 1
                    j += 1

        # Final Feedback Assembly (English)
        feedback_text = "\n".join(feedback_lines)
        if skipped_notes:
            feedback_text += "\n\n--- SUMMARY ---"
            feedback_text += "\nYou skipped the following notes:"
            for note in skipped_notes:
                feedback_text += f"\n  - {note}"
        else:
            feedback_text += "\n\n--- SUMMARY ---\nNo skips detected. Well done!"

        self.feedback_box.delete("1.0", "end")
        self.feedback_box.insert("end", feedback_text)

    def reset_all(self):
        if self.ui_lock:
            return
        self._lock_ui_temporarily(250)

        if self.live_running:
            self.stop_live()

        self.notes_box.delete("1.0", "end")
        self.chords_box.delete("1.0", "end")
        self.feedback_box.delete("1.0", "end")

        self.notes_box.insert("end", "Notes output will appear here…\n")
        self.chords_box.insert("end", "Chords output will appear here…\n")
        self.feedback_box.insert("end", "Feedback will appear here…\n")

        self.sheet_label.configure(image="", text="")
        self._sheet_imgtk = None

        with self.buf_lock:
            self.recorded_chunks.clear()

        self._set_busy(False)
        self._set_status("Reset ✅ (ready for new run)")

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

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self._set_status("Error."))

            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=job, daemon=True).start()

    def _load_outputs(self, stem: str, write_chords: bool):
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

        self._update_sheet_from_notes_txt(notes_content)

        # ✅ per-note feedback (order-based)
        self._run_compare_and_show(notes_content)

    # --------------------
    # Live mic mode
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
            self.stop_live()
        else:
            self.start_live()

    def start_live(self):
        self.outdir.mkdir(parents=True, exist_ok=True)

        with self.buf_lock:
            self.recorded_chunks.clear()

        self.live_running = True
        self.btn_live.configure(text="Stop + Analyze")
        self._set_status("Listening… (press Stop + Analyze when done)")
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
            audio = np.concatenate(self.recorded_chunks, axis=0) if self.recorded_chunks else np.zeros(0, dtype=np.float32)

        def job():
            try:
                if audio.size == 0 or len(audio) < int(0.2 * sample_rate):
                    empty_notes = "Filtered notes\n\n(No audio captured — press Start and play a bit)\n"
                    empty_chords = "Chord segments (frame-based)\n\n(No audio captured)\n"
                    self.after(0, lambda: self._show_live(empty_notes, empty_chords))
                    self.after(0, lambda: self._update_sheet_from_notes_txt(empty_notes))
                    self.after(0, lambda: self._run_compare_and_show(empty_notes))
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
                app.run_audio(audio, outdir=self.outdir, stem=stem)

                notes_path = self.outdir / f"{stem}_notes.txt"
                chords_path = self.outdir / f"{stem}_chords.txt"

                notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(no notes)"
                chords = chords_path.read_text(encoding="utf-8") if chords_path.exists() else "(no chords)"

                self.after(0, lambda n=notes, c=chords: self._show_live(n, c))
                self.after(0, lambda n=notes: self._update_sheet_from_notes_txt(n))
                self.after(0, lambda n=notes: self._run_compare_and_show(n))
                self.after(0, lambda: self._set_status("Done ✅"))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Live error", str(e)))
                self.after(0, lambda: self._set_status("Error."))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=job, daemon=True).start()

    def _on_close(self):
        try:
            if self.live_running:
                self.stop_live()
        except Exception:
            pass
        self.destroy()


App().mainloop()