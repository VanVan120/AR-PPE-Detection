"""Work ID — bind each tracked person to a persistent worker identity.

Each worker wears a printed **ArUco marker** (on the helmet or vest). Every frame
we detect markers, attribute each marker to the tracked person whose box most
*contains* it (reusing the same intersection/area helper the compliance module
uses), and remember the binding `track_id -> worker_label`. The binding is
**sticky**: once a worker is identified it persists even on frames where the
marker is hidden. But if a track disappears for longer than the tracker's
id-reuse window and then reappears, the binding is dropped until a marker
re-confirms it — so a recycled track id can never hand a new person someone
else's identity. This turns anonymous "Person #5" into a named "Worker" — Jian's
"Work ID as the main detection object".

ArUco lives in `cv2.aruco` (opencv-contrib-python). If it isn't present the binder
refuses to construct with a clear message, and run.py disables Work ID gracefully.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import supervision as sv

from .compliance import _containment_matrix


def aruco_available() -> bool:
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


class WorkIdBinder:
    def __init__(self, dictionary: str = "DICT_4X4_50", markers: Optional[dict] = None,
                 containment: float = 0.5, gc_after: int = 600, reacquire_after: int = 30):
        if not aruco_available():
            raise RuntimeError(
                "cv2.aruco unavailable — install opencv-contrib-python for Work ID "
                "(pip install 'opencv-contrib-python>=4.9'), or set workid.enabled: false.")
        dict_id = getattr(cv2.aruco, dictionary, None)
        if dict_id is None:
            raise ValueError(f"unknown ArUco dictionary '{dictionary}' "
                             "(e.g. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250)")
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
        self.markers = {int(k): str(v) for k, v in (markers or {}).items()}
        self._configured = set(self.markers.values())   # names a stray tag must not steal
        self.containment = float(containment)
        self._gc_after = int(gc_after)
        # A track absent longer than ByteTrack's id-reuse window may be a DIFFERENT
        # person who recycled the id; drop its sticky label until a marker re-confirms.
        self._reacquire_after = max(1, int(reacquire_after))
        self._bound: dict[int, str] = {}        # track_id -> worker label
        self._last_seen: dict[int, int] = {}
        self._frame_idx = 0

    def label_for_marker(self, marker_id: int) -> str:
        """Configured worker name, else a stable auto label so an unmapped tag still works."""
        return self.markers.get(int(marker_id), f"W-{int(marker_id):03d}")

    def resolve(self, frame: np.ndarray, tracked: sv.Detections) -> dict[int, str]:
        """Detect markers, (re)bind to the person each one is inside, and return the
        current {track_id: worker_label} for the persons visible this frame."""
        self._frame_idx += 1
        track_ids, person_boxes = self._persons(tracked)

        # Re-acquisition guard: if a track reappears after an absence longer than the
        # tracker's id-reuse window, treat it as possibly a new person and forget its
        # sticky label, so an unmarked recycled id can never wear a prior worker's name.
        for tid in track_ids:
            last = self._last_seen.get(tid)
            if last is not None and (self._frame_idx - last) > self._reacquire_after:
                self._bound.pop(tid, None)
        for tid in track_ids:
            self._last_seen[tid] = self._frame_idx

        if track_ids and frame is not None:
            marker_ids, marker_boxes = self._detect_markers(frame)
            if marker_ids:
                P = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
                M = np.asarray(marker_boxes, dtype=float).reshape(-1, 4)
                cont = _containment_matrix(M, P)   # (n_markers, n_persons): fraction of marker in person
                for tid, mid in self._assign(M, P, cont, track_ids, marker_ids).items():
                    label = self.label_for_marker(mid)
                    current = self._bound.get(tid)
                    # Stickiness: an established CONFIGURED identity is never overwritten
                    # by a different (stray/second) tag; only re-confirm or upgrade an
                    # anonymous 'W-<id>' auto label.
                    if current is not None and current in self._configured and label != current:
                        continue
                    self._bound[tid] = label

        self._gc()
        return {tid: self._bound[tid] for tid in track_ids if tid in self._bound}

    def _assign(self, M, P, cont, track_ids, marker_ids) -> dict[int, int]:
        """Deterministically map each marker to at most one person, then each person to
        a single marker, independent of detector ordering.

        * A marker binds to a person only when it passes the containment threshold AND
          its centre is inside that person's box; among such persons the SMALLEST
          (tightest / nearest) box wins — raw argmax silently prefers the lowest index
          (often a large background box that encloses the real worker).
        * A genuinely ambiguous marker (two comparably-tight boxes, comparable overlap)
          is left unassigned this frame rather than guessed.
        * When two markers fall on one person, the higher-containment marker wins
          (tie-broken by lowest id) so the identity can't flip frame-to-frame.
        """
        p_area = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
        m_cx = (M[:, 0] + M[:, 2]) / 2.0
        m_cy = (M[:, 1] + M[:, 3]) / 2.0
        best: dict[int, tuple] = {}   # tid -> (containment, marker_id)
        for mi, mid in enumerate(marker_ids):
            cand = []
            for j in range(len(track_ids)):
                c = float(cont[mi, j])
                if c <= 0.0 or c < self.containment:
                    continue
                if not (P[j, 0] <= m_cx[mi] <= P[j, 2] and P[j, 1] <= m_cy[mi] <= P[j, 3]):
                    continue
                cand.append((float(p_area[j]), -c, j))
            if not cand:
                continue
            cand.sort()                                  # smallest box, then highest containment
            if len(cand) >= 2:
                (a0, nc0, _), (a1, nc1, _) = cand[0], cand[1]
                if abs(a0 - a1) <= 0.05 * max(a0, a1, 1.0) and abs(nc0 - nc1) < 0.05:
                    continue                              # too close to call — don't guess
            j = cand[0][2]
            tid = track_ids[j]
            c = float(cont[mi, j])
            prev = best.get(tid)
            if prev is None or c > prev[0] or (c == prev[0] and mid < prev[1]):
                best[tid] = (c, mid)
        return {tid: mid for tid, (c, mid) in best.items()}

    def worker_for(self, track_id) -> Optional[str]:
        return self._bound.get(int(track_id)) if track_id is not None else None

    def all_bindings(self) -> dict[int, str]:
        return dict(self._bound)

    # --- helpers -------------------------------------------------------------
    def _detect_markers(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return [], []
        marker_ids = [int(x) for x in np.asarray(ids).flatten()]
        boxes = []
        for c in corners:
            pts = np.asarray(c, dtype=float).reshape(-1, 2)
            boxes.append([float(pts[:, 0].min()), float(pts[:, 1].min()),
                          float(pts[:, 0].max()), float(pts[:, 1].max())])
        return marker_ids, boxes

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

    def _gc(self) -> None:
        stale = [tid for tid, seen in self._last_seen.items()
                 if self._frame_idx - seen > self._gc_after]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._bound.pop(tid, None)
