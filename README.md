# PlayCoach v2 – Piano Transcription & Chord Detection

HOW TO RUN (MOST IMPORTANT)

python main.py \
  --audio "/path/to/audio.ogg" \
  --outdir "/path/to/output" \
  --enable-A --enable-B --enable-D \
  --write-chords \
  --frame-hop 0.05 \
  --frame-min-vel 20 \
  --frame-min-active 2 \
  --merge-min-jaccard 0.85 \
  --merge-min-dur 0.10


ARGUMENTS (SHORT)

--audio              Input audio file
--outdir             Output directory
--device             cpu or cuda
--no-midi            Do not save MIDI
--print-raw          Print raw notes
--print-audio-info   Show audio info

FILTERS
A: Consistency filter (removes one-off ghost notes)
B: Onset clustering (groups notes played together)
D: Harmonic filter (removes octave / overtone artifacts)

FRAME-BASED CHORDS

Audio is split into short frames (e.g. every 50ms).
For each frame, active notes are collected.
Similar consecutive frames are merged into chord segments.
Very short or unstable segments are removed.


PROJECT FILES

main.py
- CLI entry point
- Parses arguments
- Starts the app

transcribe/app.py
- Full transcription pipeline
- Audio → notes → filters → chords → output

transcribe/filters.py
- Note-level filtering (A / B / D)

transcribe/frame.py
- Frame-based chord detection
- Real-time ready logic

transcribe/utils.py
- Helper functions (MIDI names, JSON, formatting)
