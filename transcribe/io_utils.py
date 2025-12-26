from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from .filters import NoteFilters
from .utils import midi_to_name, make_json_safe


class IOWriter:
    @staticmethod
    def build_notes_txt(note_events: List[dict], title: str) -> str:
        note_events = NoteFilters.sort_by_onset(note_events)
        lines = [title, "", "idx\tmidi\tname\tonset(s)\toffset(s)\tdur(s)\tvelocity"]
        for i, n in enumerate(note_events):
            onset = float(n["onset_time"])
            offset = float(n["offset_time"])
            dur = NoteFilters.note_duration(n)
            midi = int(n["midi_note"])
            vel = n.get("velocity", "")
            name = midi_to_name(midi)
            lines.append(f"{i}\t{midi}\t{name}\t{onset:.3f}\t{offset:.3f}\t{dur:.3f}\t{vel}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def save_text(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=make_json_safe)