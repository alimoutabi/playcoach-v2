from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Set

from .filters import NoteFilters
from .utils import midi_to_name


@dataclass(frozen=True)
class FrameConfig:
    write_chords: bool = False
    frame_hop: float = 0.05
    frame_min_vel: int = 0
    frame_min_active: int = 2
    merge_min_jaccard: float = 0.85
    merge_min_dur: float = 0.10
    write_frame_chords: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameChord:
    t0: float
    t1: float
    midis: Tuple[int, ...]


@dataclass
class ChordSegment:
    t0: float
    t1: float
    midis: Tuple[int, ...]


class FrameChordExtractor:
    @staticmethod
    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return 0.0 if not u else (len(a & b) / len(u))

    def events_to_frame_chords(
        self,
        note_events: List[dict],
        audio_dur: float,
        cfg: FrameConfig,
    ) -> List[FrameChord]:
        if cfg.frame_hop <= 0:
            raise ValueError("frame_hop must be > 0")

        norm = []
        for ev in note_events:
            onset = float(ev["onset_time"])
            offset = float(ev["offset_time"])
            midi = int(ev["midi_note"])
            vel = NoteFilters.note_velocity(ev)
            if vel < cfg.frame_min_vel:
                continue
            if offset <= onset:
                continue
            norm.append((onset, offset, midi))

        frames: List[FrameChord] = []
        t = 0.0
        while t < audio_dur:
            t0 = t
            t1 = min(t + cfg.frame_hop, audio_dur)

            active: Set[int] = set()
            for onset, offset, midi in norm:
                if onset < t1 and offset > t0:
                    active.add(midi)

            if len(active) >= cfg.frame_min_active:
                frames.append(FrameChord(t0=t0, t1=t1, midis=tuple(sorted(active))))

            t += cfg.frame_hop

        return frames

    def merge_frames(self, frames: List[FrameChord], cfg: FrameConfig) -> List[ChordSegment]:
        if not frames:
            return []

        segs: List[ChordSegment] = []

        cur_t0 = frames[0].t0
        cur_t1 = frames[0].t1
        cur_set: Set[int] = set(frames[0].midis)

        for fr in frames[1:]:
            fr_set = set(fr.midis)
            sim = self._jaccard(cur_set, fr_set)

            if sim >= cfg.merge_min_jaccard:
                cur_t1 = fr.t1
                cur_set = (cur_set & fr_set) if (cur_set and fr_set) else fr_set
            else:
                if (cur_t1 - cur_t0) >= cfg.merge_min_dur and len(cur_set) > 0:
                    segs.append(ChordSegment(t0=cur_t0, t1=cur_t1, midis=tuple(sorted(cur_set))))
                cur_t0 = fr.t0
                cur_t1 = fr.t1
                cur_set = fr_set

        if (cur_t1 - cur_t0) >= cfg.merge_min_dur and len(cur_set) > 0:
            segs.append(ChordSegment(t0=cur_t0, t1=cur_t1, midis=tuple(sorted(cur_set))))

        return segs

    @staticmethod
    def build_chords_txt(segments: List[ChordSegment], title: str = "Chord segments (frame-based)") -> str:
        def fmt(midis: Tuple[int, ...]) -> str:
            return "-".join(midi_to_name(m) for m in midis)

        lines = [title, "", "idx\tstart(s)\tend(s)\tdur(s)\tnotes"]
        for i, s in enumerate(segments):
            dur = s.t1 - s.t0
            lines.append(f"{i}\t{s.t0:.3f}\t{s.t1:.3f}\t{dur:.3f}\t{fmt(s.midis)}")
        return "\n".join(lines) + "\n"