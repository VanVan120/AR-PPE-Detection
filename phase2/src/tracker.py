"""Multi-object tracking — persistent person IDs via ByteTrack (supervision).

Only *persons* are tracked: each person gets a stable `tracker_id` that survives
movement and brief occlusion (governed by `lost_track_buffer`). Violation boxes
(No-Helmet / No-Vest) are NOT tracked — they are per-frame observations that the
compliance module attributes to whichever tracked person contains them. This keeps
IDs meaningful ("person #5") and avoids ID churn on small, flickery PPE boxes.
"""
from __future__ import annotations

import warnings

import numpy as np
import supervision as sv


def _make_byte_track(frame_rate: int, lost_track_buffer: int):
    """Construct sv.ByteTrack, tolerating signature drift and silencing the
    cosmetic deprecation warning (supervision pinned <0.30 in requirements)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return sv.ByteTrack(frame_rate=int(frame_rate),
                                lost_track_buffer=int(lost_track_buffer))
        except TypeError:
            return sv.ByteTrack()


class PersonTracker:
    def __init__(self, person_ids, violation_ids, frame_rate: int = 30,
                 lost_track_buffer: int = 30):
        self.person_ids = list(person_ids)
        self.violation_ids = list(violation_ids)
        self.tracker = _make_byte_track(frame_rate, lost_track_buffer)

    def update(self, detections: sv.Detections):
        """Split detections, advance the tracker on persons, return (tracked, violations).

        Must be called every frame — even when no person is visible — so the
        tracker ages out lost tracks and keeps Kalman predictions current.
        """
        persons = self._subset(detections, self.person_ids)
        violations = self._subset(detections, self.violation_ids)
        tracked = self.tracker.update_with_detections(persons)
        return tracked, violations

    @staticmethod
    def _subset(detections: sv.Detections, class_ids) -> sv.Detections:
        if len(detections) == 0 or detections.class_id is None or not class_ids:
            return detections[np.zeros(len(detections), dtype=bool)]
        mask = np.isin(detections.class_id, class_ids)
        return detections[mask]
