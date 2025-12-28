from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

from .filters import NoteFilters, FilterConfig
from .frame import FrameConfig, FrameChordExtractor, FrameChord, ChordSegment
from .io_utils import IOWriter


class TranscriptionApp:
    """
    Orchestrates the full pipeline:
      1) load audio
      2) transcribe (piano_transcription_inference)
      3) clamp note times to audio duration
      4) apply A/B/D filters
      5) (optional) frame-based chords
      6) write TXT outputs
    """

    def __init__(
        self,
        *,
        filter_cfg: FilterConfig,
        frame_cfg: FrameConfig,
        print_raw: bool = False,
        print_audio_info: bool = False,
    ):
        # We keep device fixed to CPU since you removed --device from CLI.
        self.device = "cpu"

        self.filter_cfg = filter_cfg
        self.frame_cfg = frame_cfg
        self.print_raw = print_raw
        self.print_audio_info = print_audio_info

        self.filters = NoteFilters()
        self.frame_extractor = FrameChordExtractor()
        self.io = IOWriter()

    def run(self, audio_path: Path, outdir: Path, stem: Optional[str] = None) -> None:
        # ---- resolve paths ----
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        outdir = outdir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        stem = stem or audio_path.stem
        out_mid = outdir / f"{stem}.mid"
        out_notes_txt = outdir / f"{stem}_notes.txt"
        out_chords_txt = outdir / f"{stem}_chords.txt"

        # ---- load audio ----
        audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        audio_dur = len(audio) / sample_rate

        if self.print_audio_info:
            print(f"Audio samples: {len(audio)}")
            print(f"Audio duration (s): {audio_dur:.3f}")

        # ---- transcribe ----
        transcriptor = PianoTranscription(device=self.device)
        result = transcriptor.transcribe(audio, str(out_mid))

        note_events_raw = result.get("est_note_events", [])
        pedal_events = result.get("est_pedal_events", [])

        # ---- clamp note times to real audio length ----
        note_events_raw = self.filters.clamp_events_to_audio(note_events_raw, audio_dur=audio_dur)

        if self.print_raw:
            print(self.io.build_notes_txt(note_events_raw, title="RAW notes (clamped to audio duration)"))

        # ---- apply ABD filters (based on preset) ----
        note_events_filtered = self.filters.apply_ABD(note_events_raw, self.filter_cfg)
        filtered_txt = self.io.build_notes_txt(note_events_filtered, title="Filtered notes")
        print(filtered_txt)

        # ---- pedal events (console only) ----
        if pedal_events:
            print("Pedal events (est_pedal_events):")
            for p in pedal_events:
                onset = float(p["onset_time"])
                offset = min(float(p["offset_time"]), audio_dur)
                print(f"  onset={onset:.3f}s  offset={offset:.3f}s")

        # ---- optional frame-based chords ----
        if self.frame_cfg.write_chords:
            frame_chords: List[FrameChord] = self.frame_extractor.events_to_frame_chords(
                note_events_filtered, audio_dur=audio_dur, cfg=self.frame_cfg
            )
            chord_segments: List[ChordSegment] = self.frame_extractor.merge_frames(frame_chords)

            chords_txt = self.frame_extractor.build_chords_txt(chord_segments)
            print(chords_txt)
            self.io.save_text(out_chords_txt, chords_txt)
            print(f"Saved CHORDS TXT: {out_chords_txt}")

        # ---- write outputs (TXT only) ----
        self.io.save_text(out_notes_txt, filtered_txt)

        print(f"\nSaved NOTES TXT: {out_notes_txt}")
        print(f"Wrote MIDI      : {out_mid}")