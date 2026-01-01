#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, Counter

from music21 import converter, note, chord, stream


# -------------------------
# 1) Run Oemer
# -------------------------
def run_oemer(image_path: Path, workdir: Path) -> Path:
    """
    Runs oemer on an image and returns the newest produced MusicXML path.
    Uses the current python interpreter to avoid env/path issues.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "oemer.ete", str(image_path)]
    subprocess.run(cmd, cwd=str(workdir), check=True)

    candidates = (
        list(workdir.glob("*.musicxml"))
        + list(workdir.glob("*.mxl"))
        + list(workdir.glob("*.xml"))
    )
    if not candidates:
        raise FileNotFoundError(f"No MusicXML produced in {workdir}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest


# -------------------------
# 2) Helpers
# -------------------------
def midi_to_name(m: int) -> str:
    n = note.Note()
    n.pitch.midi = int(m)
    return n.nameWithOctave


def group_midis_by_offset(items, tol=1e-6):
    """
    items: list of (offset, [midi...])
    returns grouped: list of (offset, sorted_unique_midis)
    """
    items = sorted(items, key=lambda x: x[0])
    grouped = []
    cur_off = None
    cur_set = set()

    for off, midis in items:
        off = float(off)
        if cur_off is None:
            cur_off = off

        if abs(off - cur_off) <= tol:
            for m in midis:
                cur_set.add(int(m))
        else:
            grouped.append((cur_off, sorted(cur_set)))
            cur_off = off
            cur_set = set(int(m) for m in midis)

    if cur_off is not None:
        grouped.append((cur_off, sorted(cur_set)))

    return grouped


def part_pitch_profile(part) -> float:
    """
    Rough heuristic: median MIDI pitch of a part.
    Used to guess RH vs LH if parts are not labeled.
    """
    pitches = []
    for el in part.recurse():
        if isinstance(el, note.Note):
            pitches.append(int(el.pitch.midi))
        elif isinstance(el, chord.Chord):
            pitches.extend(int(p.midi) for p in el.pitches)

    if not pitches:
        return 0.0

    pitches.sort()
    return pitches[len(pitches) // 2]


def assign_hands(score):
    """
    Returns mapping: { "RH": part_index, "LH": part_index } if >=2 parts.
    If only one part exists, both map to that one.
    """
    parts = list(score.parts)
    if len(parts) == 0:
        return {"RH": None, "LH": None}
    if len(parts) == 1:
        return {"RH": 0, "LH": 0}

    # Most piano scores: 2 parts, but order isn't guaranteed in OMR
    medians = [(i, part_pitch_profile(p)) for i, p in enumerate(parts)]
    medians_sorted = sorted(medians, key=lambda x: x[1])  # low -> high

    lh = medians_sorted[0][0]
    rh = medians_sorted[-1][0]
    return {"RH": rh, "LH": lh}


# -------------------------
# 3) Extraction: By Hand -> Measure -> Offset
# -------------------------
def extract_piano_events(musicxml_path: Path, offset_tol=1e-6, midi_min=21, midi_max=108):
    """
    Returns dict:
      {
        "RH": { measure_no: [(offset_in_measure, [midis...]), ...] },
        "LH": { measure_no: [(offset_in_measure, [midis...]), ...] }
      }
    """
    score = converter.parse(str(musicxml_path))
    hand_map = assign_hands(score)

    out = {"RH": defaultdict(list), "LH": defaultdict(list)}

    for hand in ["RH", "LH"]:
        p_idx = hand_map[hand]
        if p_idx is None:
            continue

        part = score.parts[p_idx]

        for meas in part.getElementsByClass(stream.Measure):
            meas_no = meas.number if meas.number is not None else 0

            # collect raw events inside this measure
            raw = []
            for el in meas.recurse():
                if isinstance(el, note.Note):
                    m = int(el.pitch.midi)
                    if midi_min <= m <= midi_max:
                        raw.append((float(el.offset), [m]))
                elif isinstance(el, chord.Chord):
                    midis = [int(p.midi) for p in el.pitches]
                    midis = [m for m in midis if midi_min <= m <= midi_max]
                    if midis:
                        raw.append((float(el.offset), midis))

            # group to chord-events & dedupe duplicates
            grouped = group_midis_by_offset(raw, tol=offset_tol)
            out[hand][meas_no] = grouped

    return out, hand_map


# -------------------------
# 4) Pretty print
# -------------------------
def print_events(events_by_hand, hand_map):
    print("\n=== Hand assignment ===")
    print(f"RH part index: {hand_map['RH']}")
    print(f"LH part index: {hand_map['LH']}")

    for hand in ["RH", "LH"]:
        print(f"\n=== {hand} (by Measure -> Offset) ===")
        measures = sorted(events_by_hand[hand].keys())
        for meas_no in measures:
            print(f"\n-- Measure {meas_no} --")
            for off, midis in events_by_hand[hand][meas_no]:
                names = [midi_to_name(m) for m in midis]
                print(f"{off:6.3f} | {names}")


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to PNG/JPG of sheet music (best: clear, cropped, straight)")
    ap.add_argument("--workdir", default="outputs_oemer", help="Where oemer writes outputs")
    ap.add_argument("--midi-min", type=int, default=21, help="Lowest MIDI allowed (piano A0=21)")
    ap.add_argument("--midi-max", type=int, default=108, help="Highest MIDI allowed (piano C8=108)")
    args = ap.parse_args()

    img = Path(args.image).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()

    mx = run_oemer(img, workdir)
    print(f"✅ MusicXML: {mx}")

    events_by_hand, hand_map = extract_piano_events(
        mx,
        offset_tol=1e-6,
        midi_min=args.midi_min,
        midi_max=args.midi_max
    )

    print_events(events_by_hand, hand_map)


if __name__ == "__main__":
    main()