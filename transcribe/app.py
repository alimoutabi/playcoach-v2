from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

from .filters import NoteFilters, FilterConfig
from .frame import FrameConfig, FrameChordExtractor, FrameChord, ChordSegment
from .io_utils import IOWriter
from .utils import midi_to_name, make_json_safe


class TranscriptionApp:
    def __init__(
        self,
        *,
        device: str,
        filter_cfg: FilterConfig,
        frame_cfg: FrameConfig,
        no_midi: bool,
        full_json: bool,
        print_raw: bool,
        print_audio_info: bool,
    ):
        self.device = device
        self.filter_cfg = filter_cfg
        self.frame_cfg = frame_cfg
        self.no_midi = no_midi
        self.full_json = full_json
        self.print_raw = print_raw
        self.print_audio_info = print_audio_info

        self.filters = NoteFilters()
        self.frame_extractor = FrameChordExtractor()
        self.io = IOWriter()

    def run(self, audio_path: Path, outdir: Path, stem: Optional[str] = None) -> None:
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        outdir = outdir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        stem = stem or audio_path.stem
        out_mid = outdir / f"{stem}.mid"
        out_txt = outdir / f"{stem}_notes.txt"
        out_json = outdir / f"{stem}_result.json"
        out_chords = outdir / f"{stem}_chords.txt"

        # Load audio
        audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        audio_dur = len(audio) / sample_rate

        if self.print_audio_info:
            print(f"Audio samples: {len(audio)}")
            print(f"Audio duration (s): {audio_dur:.3f}")

        # Transcribe
        transcriptor = PianoTranscription(device=self.device)
        result = transcriptor.transcribe(audio, str(out_mid))

        note_events_raw = result.get("est_note_events", [])
        pedal_events = result.get("est_pedal_events", [])

        # Clamp
        note_events_raw = self.filters.clamp_events_to_audio(note_events_raw, audio_dur=audio_dur)

        if self.print_raw:
            print(self.io.build_notes_txt(note_events_raw, title="RAW notes (clamped to audio duration)"))

        # Filters
        note_events_filtered = self.filters.apply_ABD(note_events_raw, self.filter_cfg)
        print(self.io.build_notes_txt(note_events_filtered, title="Filtered notes"))

        if pedal_events:
            print("Pedal events (est_pedal_events):")
            for p in pedal_events:
                onset = float(p["onset_time"])
                offset = min(float(p["offset_time"]), audio_dur)
                print(f"  onset={onset:.3f}s  offset={offset:.3f}s")

        # Frame-based
        frame_chords: List[FrameChord] = []
        chord_segments: List[ChordSegment] = []
        if self.frame_cfg.write_chords:
            frame_chords = self.frame_extractor.events_to_frame_chords(
                note_events_filtered, audio_dur=audio_dur, cfg=self.frame_cfg
            )
            chord_segments = self.frame_extractor.merge_frames(frame_chords, cfg=self.frame_cfg)

            chords_txt = self.frame_extractor.build_chords_txt(chord_segments)
            print(chords_txt)
            self.io.save_text(out_chords, chords_txt)
            print(f"Saved CHORDS TXT: {out_chords}")

        # Save TXT
        self.io.save_text(out_txt, self.io.build_notes_txt(note_events_filtered, title="Filtered notes"))

        # Save JSON
        if self.full_json:
            payload = make_json_safe(result)
        else:
            payload = {
                "audio_file": str(audio_path),
                "audio_duration_s": audio_dur,
                "note_events_raw": note_events_raw if self.print_raw else None,
                "note_events": note_events_filtered,
                "pedal_events": pedal_events,
                "filters": asdict(self.filter_cfg),
                "frame_based": asdict(self.frame_cfg),
                "frame_chords": [
                    {"t0": fc.t0, "t1": fc.t1, "midis": list(fc.midis), "notes": [midi_to_name(m) for m in fc.midis]}
                    for fc in frame_chords
                ] if (self.frame_cfg.write_chords and self.frame_cfg.write_frame_chords) else None,
                "chord_segments": [
                    {"t0": cs.t0, "t1": cs.t1, "midis": list(cs.midis), "notes": [midi_to_name(m) for m in cs.midis]}
                    for cs in chord_segments
                ] if self.frame_cfg.write_chords else None,
            }

        self.io.save_json(out_json, payload)

        # Handle --no-midi
        if self.no_midi and out_mid.exists():
            out_mid.unlink()

        print(f"\nSaved TXT : {out_txt}")
        print(f"Saved JSON: {out_json}")
        if not self.no_midi:
            print(f"Wrote MIDI: {out_mid}")