#!/usr/bin/env python3
"""
transcribe_oop.py

CLI entrypoint (thin wrapper) around the OOP transcription pipeline.

High-level flow:
1) Parse CLI arguments (input audio, output folder, filter toggles, frame-based options)
2) Convert CLI args -> configuration objects (FilterConfig + FrameConfig)
3) Create TranscriptionApp (the “orchestrator”)
4) Run the app (loads audio, transcribes, filters, optionally extracts frame-based chords, writes outputs)

Note:
- This file intentionally contains almost no “signal processing logic”.
- All real work lives in transcribe/app.py, transcribe/filters.py, transcribe/frame.py, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Our project modules:
# - TranscriptionApp: orchestrates the whole pipeline end-to-end
# - FilterConfig: holds parameters for the A/B/D note-cleanup filters
# - FrameConfig: holds parameters for optional frame-based chord extraction
from transcribe.app import TranscriptionApp
from transcribe.filters import FilterConfig
from transcribe.frame import FrameConfig


def parse_args():
    """
    Define and parse command-line options.

    Why this is separate:
    - Keeps main() clean
    - Makes it easy to review/extend CLI options without touching pipeline code

    Returns:
        argparse.Namespace with all CLI parameters.
    """
    p = argparse.ArgumentParser(
        description="OOP piano transcription (notes + optional frame-chords)."
    )

    # -----------------------------
    # Core inputs / outputs
    # -----------------------------
    p.add_argument("--audio", required=True, help="Path to input audio file.")
    p.add_argument("--outdir", default=None, help="Output directory (default: audio folder).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    p.add_argument("--stem", default=None, help="Base filename for outputs (default: audio stem).")

    # Output control
    p.add_argument("--no-midi", action="store_true", help="Do not keep the MIDI file.")
    p.add_argument("--full-json", action="store_true", help="Write JSON-safe full model result (large).")

    # Debug / logging
    p.add_argument("--print-raw", action="store_true", help="Also print the raw (unfiltered) notes.")
    p.add_argument("--print-audio-info", action="store_true", help="Print audio duration and sample count.")

    # -----------------------------
    # A/B/D FILTER TOGGLES
    # -----------------------------
    # A: Adaptive consistency (remove one-off ghost pitches across the whole audio)
    # B: Onset clustering (group notes by onset time => chord moments) + dedupe + keep top-K notes
    # D: Harmonic/overtone drop (remove likely octave/harmonic hallucinations inside a chord moment)
    p.add_argument("--enable-A", action="store_true", help="Enable filter A: adaptive consistency.")
    p.add_argument("--enable-B", action="store_true", help="Enable filter B: onset clustering + dedupe + top-K.")
    p.add_argument("--enable-D", action="store_true", help="Enable filter D: drop likely harmonics/overtones.")

    # Parameters used mainly by (B)
    p.add_argument("--cluster-window", type=float, default=0.04,
                   help="(B) Cluster onset window in seconds (e.g. 0.04 = 40ms).")
    p.add_argument("--dedupe-window", type=float, default=0.08,
                   help="(B) Dedupe time window in seconds for repeated same notes.")
    p.add_argument("--max-notes-per-cluster", type=int, default=6,
                   help="(B) Keep only the strongest K notes per chord cluster.")

    # Parameters used by (D)
    p.add_argument("--harmonic-velocity-ratio", type=float, default=1.15,
                   help="(D) Drop harmonic if base velocity >= ratio * harmonic velocity.")

    # Parameters used by (A)
    p.add_argument("--min-occurrences", type=int, default=2,
                   help="(A) Keep pitches that appear at least N times.")
    p.add_argument("--min-total-dur-ratio-of-max", type=float, default=0.10,
                   help="(A) Keep pitches with total duration >= ratio * max_total_duration.")

    # -----------------------------
    # FRAME-BASED CHORD EXTRACTION (optional)
    # -----------------------------
    # This is “real-time-ready”: we compute the active notes per short frame (e.g. every 50ms),
    # then merge consecutive similar frames into stable chord segments.
    p.add_argument("--write-chords", action="store_true",
                   help="Enable frame-based chord extraction and write *_chords.txt + JSON chord segments.")
    p.add_argument("--frame-hop", type=float, default=0.05,
                   help="Frame size/hop in seconds (e.g. 0.05 = 50ms).")
    p.add_argument("--frame-min-vel", type=int, default=0,
                   help="Ignore notes with velocity below this when building frames.")
    p.add_argument("--frame-min-active", type=int, default=2,
                   help="Drop frames that have fewer than this many active notes.")
    p.add_argument("--merge-min-jaccard", type=float, default=0.85,
                   help="Merge consecutive frames if chord sets are similar (Jaccard >= threshold).")
    p.add_argument("--merge-min-dur", type=float, default=0.10,
                   help="Drop chord segments shorter than this duration (seconds).")
    p.add_argument("--write-frame-chords", action="store_true",
                   help="Also store per-frame chords in JSON (bigger output).")

    return p.parse_args()


def main():
    """
    Entry point of the CLI program.

    Responsibilities:
    - Read CLI args
    - Resolve input/output paths
    - Build config objects from CLI args
    - Construct TranscriptionApp and execute it
    """
    args = parse_args()

    # -----------------------------
    # Resolve paths
    # -----------------------------
    # Path(args.audio) is the input file. We keep it as a Path for easy handling.
    audio_path = Path(args.audio)

    # If user did not provide --outdir, we default to "same folder as the input audio".
    outdir = Path(args.outdir).expanduser() if args.outdir else audio_path.expanduser().resolve().parent

    # -----------------------------
    # Build FilterConfig (A/B/D)
    # -----------------------------
    # This config is passed into TranscriptionApp.
    # The app will apply these filters after transcription (post-processing stage).
    filter_cfg = FilterConfig(
        enable_A=bool(args.enable_A),
        enable_B=bool(args.enable_B),
        enable_D=bool(args.enable_D),
        cluster_window=float(args.cluster_window),
        dedupe_window=float(args.dedupe_window),
        max_notes_per_cluster=int(args.max_notes_per_cluster),
        harmonic_velocity_ratio=float(args.harmonic_velocity_ratio),
        min_occurrences=int(args.min_occurrences),
        min_total_dur_ratio_of_max=float(args.min_total_dur_ratio_of_max),
    )

    # -----------------------------
    # Build FrameConfig (optional chord extraction)
    # -----------------------------
    # If write_chords is True, the app will:
    # 1) slice time into frames of length frame_hop
    # 2) collect active notes per frame
    # 3) merge similar frames -> chord segments
    # 4) write *_chords.txt and chord segments into JSON
    frame_cfg = FrameConfig(
        write_chords=bool(args.write_chords),
        frame_hop=float(args.frame_hop),
        frame_min_vel=int(args.frame_min_vel),
        frame_min_active=int(args.frame_min_active),
        merge_min_jaccard=float(args.merge_min_jaccard),
        merge_min_dur=float(args.merge_min_dur),
        write_frame_chords=bool(args.write_frame_chords),
    )

    # -----------------------------
    # Create and run the app
    # -----------------------------
    # TranscriptionApp hides all implementation details:
    # - loads audio
    # - runs piano_transcription_inference
    # - clamps note times to the audio duration
    # - applies filters A/B/D
    # - optionally performs frame-based chord extraction
    # - writes TXT/JSON/MIDI outputs
    app = TranscriptionApp(
        device=str(args.device),
        filter_cfg=filter_cfg,
        frame_cfg=frame_cfg,
        no_midi=bool(args.no_midi),
        full_json=bool(args.full_json),
        print_raw=bool(args.print_raw),
        print_audio_info=bool(args.print_audio_info),
    )

    # stem controls output filenames, e.g. <stem>_notes.txt, <stem>_result.json, <stem>.mid, etc.
    app.run(audio_path=audio_path, outdir=outdir, stem=args.stem)


# Standard Python convention:
# This ensures main() only runs when this file is executed directly:
#   python transcribe_oop.py ...
# and not when imported as a module.
if __name__ == "__main__":
    main()