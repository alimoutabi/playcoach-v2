from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FilterConfig:
    enable_A: bool = False
    enable_B: bool = False
    enable_D: bool = False

    cluster_window: float = 0.04
    dedupe_window: float = 0.08
    max_notes_per_cluster: int = 6

    harmonic_velocity_ratio: float = 1.15

    min_occurrences: int = 2
    min_total_dur_ratio_of_max: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


class NoteFilters:
    @staticmethod
    def note_duration(ev: dict) -> float:
        onset = float(ev["onset_time"])
        offset = float(ev["offset_time"])
        return max(0.0, offset - onset)

    @staticmethod
    def note_velocity(ev: dict) -> int:
        v = ev.get("velocity", 0)
        return 0 if v is None else int(v)

    @staticmethod
    def sort_by_onset(events: List[dict]) -> List[dict]:
        return sorted(events, key=lambda x: float(x["onset_time"]))

    @staticmethod
    def clamp_events_to_audio(note_events: List[dict], audio_dur: float) -> List[dict]:
        clamped = []
        for ev in note_events:
            onset = float(ev["onset_time"])
            offset = float(ev["offset_time"])

            if onset >= audio_dur:
                continue
            offset = min(offset, audio_dur)
            offset = max(offset, onset)

            ev2 = dict(ev)
            ev2["onset_time"] = onset
            ev2["offset_time"] = offset
            clamped.append(ev2)

        return NoteFilters.sort_by_onset(clamped)

    @staticmethod
    def cluster_by_onset(note_events: List[dict], cluster_window: float) -> List[List[dict]]:
        events = NoteFilters.sort_by_onset(note_events)
        clusters: List[List[dict]] = []

        current: List[dict] = []
        cluster_start_onset: Optional[float] = None

        for ev in events:
            onset = float(ev["onset_time"])
            if not current:
                current = [ev]
                cluster_start_onset = onset
                continue

            if onset - float(cluster_start_onset) <= cluster_window:
                current.append(ev)
            else:
                clusters.append(current)
                current = [ev]
                cluster_start_onset = onset

        if current:
            clusters.append(current)

        return clusters

    @staticmethod
    def dedupe_same_pitch_in_cluster(cluster: List[dict], dedupe_window: float) -> List[dict]:
        def better(a: dict, b: dict) -> dict:
            sa = (NoteFilters.note_velocity(a), NoteFilters.note_duration(a))
            sb = (NoteFilters.note_velocity(b), NoteFilters.note_duration(b))
            return a if sa >= sb else b

        by_midi: Dict[int, List[dict]] = defaultdict(list)
        for ev in NoteFilters.sort_by_onset(cluster):
            by_midi[int(ev["midi_note"])].append(ev)

        kept: List[dict] = []
        for _, evs in by_midi.items():
            evs = NoteFilters.sort_by_onset(evs)
            best = evs[0]
            for ev in evs[1:]:
                if abs(float(ev["onset_time"]) - float(best["onset_time"])) <= dedupe_window:
                    best = better(best, ev)
                else:
                    kept.append(best)
                    best = ev
            kept.append(best)

        return NoteFilters.sort_by_onset(kept)

    @staticmethod
    def dedupe_pitch_class_in_cluster(cluster: List[dict]) -> List[dict]:
        if not cluster:
            return cluster

        mids = sorted(int(ev["midi_note"]) for ev in cluster)
        median_midi = mids[len(mids) // 2]

        by_pc: Dict[int, List[dict]] = defaultdict(list)
        for ev in cluster:
            by_pc[int(ev["midi_note"]) % 12].append(ev)

        def rank(ev: dict):
            midi = int(ev["midi_note"])
            dist = abs(midi - median_midi)
            vel = NoteFilters.note_velocity(ev)
            dur = NoteFilters.note_duration(ev)
            return (dist, -vel, -dur)

        kept = [sorted(evs, key=rank)[0] for evs in by_pc.values()]
        return NoteFilters.sort_by_onset(kept)

    @staticmethod
    def keep_top_k_in_cluster(cluster: List[dict], max_notes: int) -> List[dict]:
        ranked = sorted(
            cluster,
            key=lambda ev: (NoteFilters.note_velocity(ev), NoteFilters.note_duration(ev)),
            reverse=True,
        )
        return NoteFilters.sort_by_onset(ranked[:max_notes])

    @staticmethod
    def dedupe_same_midi_globally(note_events: List[dict], dedupe_window: float) -> List[dict]:
        events = NoteFilters.sort_by_onset(note_events)

        def better(a: dict, b: dict) -> dict:
            return a if (NoteFilters.note_velocity(a), NoteFilters.note_duration(a)) >= (
                NoteFilters.note_velocity(b), NoteFilters.note_duration(b)
            ) else b

        last_by_midi: Dict[int, dict] = {}
        kept: List[dict] = []

        for ev in events:
            midi = int(ev["midi_note"])
            onset = float(ev["onset_time"])

            if midi not in last_by_midi:
                last_by_midi[midi] = ev
                continue

            prev = last_by_midi[midi]
            prev_onset = float(prev["onset_time"])

            if abs(onset - prev_onset) <= dedupe_window:
                last_by_midi[midi] = better(prev, ev)
            else:
                kept.append(prev)
                last_by_midi[midi] = ev

        kept.extend(last_by_midi.values())
        return NoteFilters.sort_by_onset(kept)

    @staticmethod
    def drop_likely_harmonics(
        cluster: List[dict],
        harmonic_intervals=(12, 19, 24, 31),
        min_base_velocity_ratio: float = 1.15,
    ) -> List[dict]:
        if not cluster:
            return cluster

        best_by_midi: Dict[int, dict] = {}
        for ev in cluster:
            midi = int(ev["midi_note"])
            if midi not in best_by_midi:
                best_by_midi[midi] = ev
            else:
                a = best_by_midi[midi]
                b = ev
                best_by_midi[midi] = a if (NoteFilters.note_velocity(a), NoteFilters.note_duration(a)) >= (
                    NoteFilters.note_velocity(b), NoteFilters.note_duration(b)
                ) else b

        to_drop = set()
        for midi_high, ev_high in best_by_midi.items():
            v_high = NoteFilters.note_velocity(ev_high)
            for interval in harmonic_intervals:
                midi_base = midi_high - int(interval)
                if midi_base in best_by_midi:
                    ev_base = best_by_midi[midi_base]
                    v_base = NoteFilters.note_velocity(ev_base)
                    if v_base >= v_high * min_base_velocity_ratio:
                        to_drop.add(midi_high)
                        break

        kept = [ev for ev in cluster if int(ev["midi_note"]) not in to_drop]
        return NoteFilters.sort_by_onset(kept)

    @staticmethod
    def adaptive_consistency_filter(
        note_events: List[dict],
        min_occurrences: int = 2,
        min_total_dur_ratio_of_max: float = 0.10,
        keep_if_velocity_ge: Optional[int] = None,
    ) -> List[dict]:
        if not note_events:
            return note_events

        occ = defaultdict(int)
        tot_dur = defaultdict(float)

        for ev in note_events:
            midi = int(ev["midi_note"])
            occ[midi] += 1
            tot_dur[midi] += NoteFilters.note_duration(ev)

        max_total = max(tot_dur.values()) if tot_dur else 0.0
        dur_threshold = (min_total_dur_ratio_of_max * max_total) if max_total > 0 else 0.0

        filtered = []
        for ev in note_events:
            midi = int(ev["midi_note"])
            v = NoteFilters.note_velocity(ev)

            if keep_if_velocity_ge is not None and v >= int(keep_if_velocity_ge):
                filtered.append(ev)
                continue
            if occ[midi] >= int(min_occurrences):
                filtered.append(ev)
                continue
            if tot_dur[midi] >= float(dur_threshold):
                filtered.append(ev)
                continue

        return NoteFilters.sort_by_onset(filtered)

    @classmethod
    def apply_ABD(cls, note_events: List[dict], cfg: FilterConfig) -> List[dict]:
        events = cls.sort_by_onset(note_events)

        if cfg.enable_B:
            clusters = cls.cluster_by_onset(events, cluster_window=cfg.cluster_window)
            pruned: List[dict] = []
            for c in clusters:
                c = cls.dedupe_same_pitch_in_cluster(c, dedupe_window=cfg.dedupe_window)
                c = cls.dedupe_pitch_class_in_cluster(c)
                c = cls.keep_top_k_in_cluster(c, max_notes=cfg.max_notes_per_cluster)
                pruned.extend(c)
            events = cls.dedupe_same_midi_globally(pruned, dedupe_window=cfg.dedupe_window)

        if cfg.enable_D:
            clusters = cls.cluster_by_onset(events, cluster_window=cfg.cluster_window)
            pruned: List[dict] = []
            for c in clusters:
                c = cls.drop_likely_harmonics(c, min_base_velocity_ratio=cfg.harmonic_velocity_ratio)
                pruned.extend(c)
            events = cls.sort_by_onset(pruned)

        if cfg.enable_A:
            events = cls.adaptive_consistency_filter(
                events,
                min_occurrences=cfg.min_occurrences,
                min_total_dur_ratio_of_max=cfg.min_total_dur_ratio_of_max,
                keep_if_velocity_ge=None,
            )

        return events