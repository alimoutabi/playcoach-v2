PlayCoach v2 – Piano Transcription (OOP)
=======================================

What this project does
---------------------
Desktop tool to transcribe piano audio into:
- notes.txt  → detected notes with timing
- chords.txt → optional frame-based chord segments

Built fully object-oriented (OOP).
Usable via CLI or a simple Tkinter desktop GUI.


Filters (A / B / D)
------------------
A – Adaptive consistency
Removes short, one-off “ghost” notes across the whole audio.

B – Onset clustering
Groups notes that start together (chords), removes duplicates,
keeps only the strongest notes.

D – Harmonic filter
Drops likely harmonics / octave overtones if a stronger base note exists.


Main entry files
----------------
transcribe_oop.py
CLI entrypoint. Parses arguments, selects filter preset,
runs the full transcription pipeline.

gui.py
Tkinter desktop GUI (for testing).
Select audio, preset, hop size, and view notes & chords.


Core OOP modules (transcribe/)
------------------------------
app.py
Main orchestrator: loads audio, runs the model,
applies filters, extracts chords, writes TXT outputs.

filters.py
Implements A/B/D note-cleanup logic and FilterConfig.

frame.py
Frame-based chord extraction (real-time-ready).
Slices time into frames and merges them into chord segments.

io_utils.py
Handles writing TXT files (notes and chords).

utils.py
Small helpers (MIDI → note name, misc utilities).


Typical usage
-------------
CLI:
python transcribe_oop.py --audio path/to/file.ogg --preset clean --chords --hop 0.05

GUI:
python gui.py