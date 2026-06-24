"""Compliance logic — per-person violation state, deduplicated and debounced.

Turns noisy per-frame detections into a stable answer to "who is currently
violating what". Two responsibilities:

1. **Association** — a violation box (a bare head = No-Helmet, an unvested torso =
   No-Vest) is attributed to the tracked person whose box contains the largest
   fraction of it (>= `association_containment`). So a violation becomes
   "person #5: No hard hat", not a free-floating box.

2. **Debounce / clear** — a (person, violation) pair must be observed for
   `debounce_frames` consecutive-ish frames before it FIRES, and stays active
   until `clear_frames` frames pass with it absent. This kills single-frame
   flicker and gives stable alerts. Because state is keyed by the persistent
   tracker_id, a person keeps their violation state through a brief occlusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv

from .config import SEVERITY_RANK, ViolationRule


@dataclass
class ActiveViolation:
    person_id: int
    class_name: str
    severity: str
    label: str


@dataclass
class PersonStatus:
    tracker_id: int
    bbox: tuple[float, float, float, float]          # x1, y1, x2, y2
    active: list[ActiveViolation] = field(default_factory=list)

    @property
    def worst_severity(self) -> Optional[str]:
        if not self.active:
            return None
        # An unknown/typo'd severity is treated as 'high' (the documented contract),
        # consistent with overlay.severity_color and the HUD counter.
        return max((v.severity for v in self.active),
                   key=lambda s: SEVERITY_RANK.get(s, SEVERITY_RANK["high"]))

    @property
    def is_compliant(self) -> bool:
        return not self.active


@dataclass
class FrameCompliance:
    persons: list[PersonStatus] = field(default_factory=list)
    events: list[ActiveViolation] = field(default_factory=list)       # all active this frame
    new_events: list[ActiveViolation] = field(default_factory=list)   # fired THIS frame


class _Counter:
    __slots__ = ("present_streak", "absent_streak", "active")

    def __init__(self):
        self.present_streak = 0
        self.absent_streak = 0
        self.active = False


class ComplianceMonitor:
    # Bound the session accumulator. Realistic sessions never approach this; the cap
    # only stops pathological unbounded growth over a multi-hour run with ID churn.
    _FIRE_CAP = 100_000

    def __init__(self, rules_by_id: dict[int, ViolationRule],
                 debounce_frames: int = 5, clear_frames: int = 15,
                 containment_thresh: float = 0.30, lost_track_buffer: int = 30):
        self.rules_by_id = dict(rules_by_id)
        self.debounce_frames = max(1, int(debounce_frames))
        self.clear_frames = max(1, int(clear_frames))
        self.containment_thresh = float(containment_thresh)

        # tracker_id -> {violation_class_id -> _Counter}
        self._state: dict[int, dict[int, _Counter]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame_idx = 0
        # Keep a vanished person's state at least as long as ByteTrack can keep their
        # id alive through occlusion (lost_track_buffer), so a returning person's
        # debounced violation state is never discarded while the same id is still live.
        self._gc_after = max(self.clear_frames, int(lost_track_buffer), 30) * 3

        # session accumulators (for the exit summary / stretch goal)
        self.ever_fired: set[tuple[int, str]] = set()
        self._summary_truncated = False

    def update(self, tracked_persons: sv.Detections,
               violation_dets: sv.Detections) -> FrameCompliance:
        self._frame_idx += 1

        person_ids, person_boxes = self._persons(tracked_persons)
        # (person_id, violation_class_id) pairs observed this frame
        present = self._associate(person_ids, person_boxes, violation_dets)

        frame = FrameCompliance()
        for pid, box in zip(person_ids, person_boxes):
            self._last_seen[pid] = self._frame_idx
            counters = self._state.setdefault(pid, {})
            status = PersonStatus(tracker_id=pid, bbox=tuple(float(v) for v in box))
            for vid, rule in self.rules_by_id.items():
                c = counters.setdefault(vid, _Counter())
                was_active = c.active
                if (pid, vid) in present:
                    c.present_streak += 1
                    c.absent_streak = 0
                    if c.present_streak >= self.debounce_frames:
                        c.active = True
                else:
                    c.absent_streak += 1
                    # Decay rather than hard-reset, so brief detector flicker (a single
                    # dropped frame) doesn't endlessly restart the debounce and hide a
                    # real, persistent violation — 'consecutive-ish', per the docstring.
                    c.present_streak = max(0, c.present_streak - 1)
                    if c.absent_streak >= self.clear_frames:
                        c.active = False
                if c.active:
                    av = ActiveViolation(pid, rule.class_name, rule.severity, rule.label)
                    status.active.append(av)
                    frame.events.append(av)
                    if not was_active:
                        frame.new_events.append(av)
                        if len(self.ever_fired) < self._FIRE_CAP:
                            self.ever_fired.add((pid, rule.class_name))
                        else:
                            self._summary_truncated = True
            frame.persons.append(status)

        self._gc()
        return frame

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _persons(tracked: sv.Detections):
        ids: list[int] = []
        boxes: list[np.ndarray] = []
        if len(tracked) == 0 or tracked.tracker_id is None:
            return ids, boxes
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue
            ids.append(int(tid))
            boxes.append(np.asarray(box, dtype=float))
        return ids, boxes

    def _associate(self, person_ids, person_boxes,
                   violation_dets: sv.Detections) -> set[tuple[int, int]]:
        present: set[tuple[int, int]] = set()
        if not person_ids or len(violation_dets) == 0 or violation_dets.class_id is None:
            return present
        P = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
        V = np.asarray(violation_dets.xyxy, dtype=float).reshape(-1, 4)
        cont = _containment_matrix(V, P)  # (N_viol, M_person)
        for vi, vid in enumerate(violation_dets.class_id):
            if int(vid) not in self.rules_by_id:
                continue
            j = int(np.argmax(cont[vi]))
            # Require real overlap: with containment_thresh == 0 a zero-overlap box
            # would otherwise be falsely attributed to person #0 (argmax of an all-0 row).
            if cont[vi, j] > 0.0 and cont[vi, j] >= self.containment_thresh:
                present.add((person_ids[j], int(vid)))
        return present

    def _gc(self) -> None:
        stale = [pid for pid, seen in self._last_seen.items()
                 if self._frame_idx - seen > self._gc_after]
        for pid in stale:
            self._state.pop(pid, None)
            self._last_seen.pop(pid, None)

    def summary(self) -> dict:
        """Session totals for the exit summary."""
        by_type: dict[str, int] = {}
        by_person: dict[int, list[str]] = {}
        for pid, cls in sorted(self.ever_fired):
            by_type[cls] = by_type.get(cls, 0) + 1
            by_person.setdefault(pid, []).append(cls)
        return {
            "unique_violations": len(self.ever_fired),
            "by_type": by_type,
            "by_person": {pid: v for pid, v in sorted(by_person.items())},
            "people_tracked": len(self._last_seen),  # currently-retained states
            "truncated": self._summary_truncated,     # True if the session cap was hit
        }


def _containment_matrix(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Fraction of each violation box V that lies inside each person box P.

    Returns an (N_viol, M_person) matrix of intersection_area / area(V).
    """
    N, M = len(V), len(P)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)
    vx1, vy1, vx2, vy2 = (V[:, k][:, None] for k in range(4))
    px1, py1, px2, py2 = (P[:, k][None, :] for k in range(4))
    iw = np.clip(np.minimum(vx2, px2) - np.maximum(vx1, px1), 0, None)
    ih = np.clip(np.minimum(vy2, py2) - np.maximum(vy1, py1), 0, None)
    inter = iw * ih
    varea = np.maximum((vx2 - vx1) * (vy2 - vy1), 1e-6)
    return inter / varea
