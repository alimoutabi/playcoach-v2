#!/usr/bin/env python3
"""
transcribe_oop.py

Minimal CLI for the OOP transcription pipeline.

Outputs:
- <audio_stem>_notes.txt
- <audio_stem>.mid
- optionally <audio_stem>_chords.txt   (frame-based chord segments)

Filters are controlled via one switch: --preset {raw, clean, aggressive}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig


def _filter_cfg_from_preset(preset: str) -> FilterConfig:
    """
    Map a simple preset name -> filter parameters.

    raw:
      - no A/B/D filters (just clamping)

    clean:
      - B + D (good default): chord clustering + harmonic cleanup

    aggressive:
      - A + B + D (strong cleanup): also removes one-off ghost pitches globally
    """
    preset = preset.lower().strip()

    if preset == "raw":
        return FilterConfig(enable_A=False, enable_B=False, enable_D=False)

    if preset == "clean":
        return FilterConfig(
            enable_A=False,
            enable_B=True,
            enable_D=True,
            cluster_window=0.04,
            dedupe_window=0.08,
            max_notes_per_cluster=6,
            harmonic_velocity_ratio=1.15,
        )

    if preset == "aggressive":
        return FilterConfig(
            enable_A=True,
            enable_B=True,
            enable_D=True,
            cluster_window=0.04,
            dedupe_window=0.08,
            max_notes_per_cluster=5,       # a bit stricter than "clean"
            harmonic_velocity_ratio=1.10,  # drops harmonics a bit more easily
            min_occurrences=2,
            min_total_dur_ratio_of_max=0.10,
        )

    raise ValueError(f"Unknown preset: {preset}. Use raw|clean|aggressive.")


def parse_args():
    p = argparse.ArgumentParser(description="Piano transcription -> TXT (+ optional frame-based chords).")

    p.add_argument("--audio", required=True, help="Input audio file (wav/mp3/ogg/...).")
    p.add_argument("--outdir", default=None, help="Output directory (default: same folder as audio).")

    p.add_argument(
        "--preset",
        default="clean",
        choices=["raw", "clean", "aggressive"],
        help="Filter strength preset (default: clean).",
    )

    p.add_argument("--chords", action="store_true", help="Write *_chords.txt (frame-based chord segments).")
    p.add_argument(
        "--hop",
        type=float,
        default=0.05,
        help="Frame hop size in seconds (only used if --chords). Default: 0.05 (=50ms).",
    )

    p.add_argument("--print-raw", action="store_true", help="Also print raw notes to console.")
    p.add_argument("--print-audio-info", action="store_true", help="Print audio duration and sample count.")

    return p.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else audio_path.parent

    filter_cfg = _filter_cfg_from_preset(args.preset)

    frame_cfg = FrameConfig(
        write_chords=bool(args.chords),
        frame_hop=float(args.hop),
    )

    app = TranscriptionApp(
        filter_cfg=filter_cfg,
        frame_cfg=frame_cfg,
        print_raw=bool(args.print_raw),
        print_audio_info=bool(args.print_audio_info),
    )

    app.run(audio_path=audio_path, outdir=outdir)


if __name__ == "__main__":
    main()