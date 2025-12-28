from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PIL import Image

import music21 as m21


@dataclass
class NoteEvent:
    midi: int
    onset: float


def _parse_notes_txt(notes_txt: str) -> List[NoteEvent]:
    """
    Expects your notes txt format like:

    idx    midi    name    onset(s) offset(s) dur(s) velocity
    0      60      C4      0.123    0.456     0.333  80
    ...
    """
    events: List[NoteEvent] = []
    for line in notes_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("idx"):
            continue
        if line.lower().startswith("filtered notes"):
            continue

        parts = re.split(r"\s+", line)
        # We expect at least: idx midi name onset ...
        if len(parts) < 4:
            continue

        try:
            midi = int(parts[1])
            onset = float(parts[3])
        except Exception:
            continue

        events.append(NoteEvent(midi=midi, onset=onset))

    events.sort(key=lambda e: e.onset)
    return events


def _group_by_onset(events: List[NoteEvent], tol: float = 0.05) -> List[List[NoteEvent]]:
    """
    Groups near-simultaneous notes (chords) by onset time tolerance.
    """
    if not events:
        return []

    groups: List[List[NoteEvent]] = []
    cur: List[NoteEvent] = [events[0]]
    base = events[0].onset

    for e in events[1:]:
        if abs(e.onset - base) <= tol:
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
            base = e.onset
    groups.append(cur)
    return groups


def _build_part(groups: List[List[NoteEvent]], clef: m21.clef.Clef) -> m21.stream.Part:
    """
    Rhythm is intentionally simplified:
    - each group becomes one chord/note
    - each step advances by quarterLength=1
    """
    p = m21.stream.Part()
    p.insert(0, clef)
    p.insert(0, m21.meter.TimeSignature("4/4"))

    offset = 0.0
    for g in groups:
        midis = sorted({e.midi for e in g})
        if not midis:
            continue

        if len(midis) == 1:
            n = m21.note.Note(midis[0])
            n.quarterLength = 1
            p.insert(offset, n)
        else:
            ch = m21.chord.Chord(midis)
            ch.quarterLength = 1
            p.insert(offset, ch)

        offset += 1.0

    return p


def render_grand_staff_from_notes_txt(notes_txt: str) -> Image.Image:
    """
    Builds a simple piano grand staff (treble + bass) from your *_notes.txt
    and renders it to PNG via MuseScore (configured in music21 environment).
    Returns a PIL.Image.
    """
    events = _parse_notes_txt(notes_txt)

    # Split into treble/bass by MIDI (middle C = 60)
    treble_events = [e for e in events if e.midi >= 60]
    bass_events = [e for e in events if e.midi < 60]

    treble_groups = _group_by_onset(treble_events, tol=0.05)
    bass_groups = _group_by_onset(bass_events, tol=0.05)

    score = m21.stream.Score()

    treble = _build_part(treble_groups, m21.clef.TrebleClef())
    bass = _build_part(bass_groups, m21.clef.BassClef())

    # Make it explicitly piano-like
    treble.partName = "Treble"
    bass.partName = "Bass"

    score.insert(0, treble)
    score.insert(0, bass)

    # Render to PNG (requires MuseScore path set in music21 env)
    with tempfile.TemporaryDirectory() as td:
        xml_path = os.path.join(td, "playcoach.xml")
        score.write("musicxml", fp=xml_path)

        # This produces PNG(s). music21 usually creates something like:
        # playcoach-1.png  (or playcoach.png depending on setup)
        out = score.write("musicxml.png", fp=os.path.join(td, "playcoach"))

        # Find a PNG we can open
        png_candidates = []
        if isinstance(out, str) and out.lower().endswith(".png") and os.path.exists(out):
            png_candidates.append(out)

        # Also scan temp folder for pngs
        for fn in os.listdir(td):
            if fn.lower().endswith(".png"):
                png_candidates.append(os.path.join(td, fn))

        if not png_candidates:
            raise RuntimeError(
                "No PNG produced. Check MuseScore path and that music21 can call it."
            )

        png_candidates.sort()
        img = Image.open(png_candidates[0]).convert("RGBA")
        return img