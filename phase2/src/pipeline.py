"""One frame in, one annotated frame out — the whole safety pipeline as a reusable unit.

`run.py` owns a live loop: window, keys, recording, FPS accounting. The phone server and
the offline clip analyser in `phase7_mobile/` need the *same* per-frame work but none of
that. Rather than reimplement detect -> track -> comply -> identify -> log in three
places (where they would quietly drift apart, and a fix in one would silently miss the
others), that core lives here and every entry point calls it.

    pipe = SafetyPipeline(cfg, frame_rate=30)
    result = pipe.process(frame, frame_no, elapsed_s)      # state + annotated frame
    ...
    pipe.close(elapsed_s, frame_no); pipe.report()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FrameResult:
    frame: np.ndarray                       # annotated, ready to show / encode
    persons: int = 0
    alerts: List[dict] = field(default_factory=list)      # currently active
    new_alerts: List[dict] = field(default_factory=list)  # fired THIS frame
    workers: List[dict] = field(default_factory=list)     # roster rows
    fps: float = 0.0
    # Per-person boxes in NORMALISED coordinates (0..1 of frame width/height). The phone
    # app draws its own overlay from these onto its own live camera preview, so the video
    # stays smooth and only the boxes carry the round-trip lag. Normalised because the
    # phone uploads a downscaled frame but displays at its own screen size.
    people: List[dict] = field(default_factory=list)


class SafetyPipeline:
    """Detector + tracker + compliance + worker identity + per-worker history."""

    def __init__(self, cfg, frame_rate: int = 30, render: bool = True,
                 quiet: bool = False):
        from .config import validate_against_model
        from .detector import Detector

        self.cfg = cfg
        self.render_enabled = bool(render)
        self._quiet = bool(quiet)

        self.detector = Detector(cfg.weights_path, cfg.device,
                                 cfg.confidence_threshold, cfg.imgsz)
        self.detector.warmup()

        self.person_ids = self.detector.class_ids_for([cfg.person_class])
        self.rules_by_id: Dict[int, Any] = {}
        for name, rule in cfg.violation_rules.items():
            for cid in self.detector.class_ids_for([name]):
                self.rules_by_id[cid] = rule
        if not self.person_ids:
            raise ValueError(f"person_class '{cfg.person_class}' is not a model class "
                             f"{sorted(self.detector.names.values())}")
        if not self.rules_by_id:
            raise ValueError("none of the violation_rules match a model class "
                             f"{sorted(self.detector.names.values())}")
        for issue in validate_against_model(cfg, self.detector.names):
            if issue.level != "ok" and not self._quiet:
                print(issue)

        self._frame_rate = int(frame_rate) or 30
        self.reset()

    def reset(self, frame_rate: Optional[int] = None) -> None:
        """Begin a fresh session — new tracks, no violation history — WITHOUT reloading
        the detector. Loading the model is the expensive part (seconds on CPU); the state
        that actually needs clearing between two site walks is everything downstream of
        it. Rebuilding the whole pipeline instead would make "start a new walk" feel like
        restarting the app."""
        from .compliance import ComplianceMonitor
        from .tracker import PersonTracker

        if frame_rate:
            self._frame_rate = int(frame_rate)
        cfg = self.cfg
        self.tracker = PersonTracker(self.person_ids, list(self.rules_by_id.keys()),
                                     frame_rate=self._frame_rate,
                                     lost_track_buffer=cfg.lost_track_buffer)
        self.monitor = ComplianceMonitor(self.rules_by_id, cfg.debounce_frames,
                                         cfg.clear_frames, cfg.association_containment,
                                         cfg.lost_track_buffer)
        self.binder = self._build_binder()
        self.identity, self.history = self._build_identity()
        self._fps = 0.0
        self._last_t = None

    # --- optional subsystems --------------------------------------------------
    def _build_binder(self):
        if not self.cfg.workid_enabled:
            return None
        try:
            from .workid import WorkIdBinder, aruco_available
            if not aruco_available():
                raise RuntimeError("cv2.aruco unavailable (install opencv-contrib-python)")
            return WorkIdBinder(self.cfg.workid_dictionary, self.cfg.workid_markers,
                                self.cfg.workid_containment,
                                gc_after=max(self.cfg.lost_track_buffer, 30) * 3,
                                reacquire_after=self.cfg.lost_track_buffer)
        except Exception as e:                             # noqa: BLE001
            if not self._quiet:
                print(f"[warn] Work ID disabled: {e}")
            return None

    def _build_identity(self):
        if not self.cfg.identity_enabled:
            return None, None
        try:
            from .identity import IdentityManager
            from .reid import build_embedder
            from .workerlog import WorkerHistory
            embedder = build_embedder(self.cfg.identity_method, device=self.cfg.device)
            mgr = IdentityManager(
                embedder,
                match_threshold=self.cfg.identity_match_threshold,
                margin=self.cfg.identity_margin,
                max_exemplars=self.cfg.identity_max_exemplars,
                forget_after=self.cfg.identity_forget_after,
                min_box_height=self.cfg.identity_min_box_height,
                appearance_enabled=self.cfg.identity_appearance)
            # A worker the detector drops for a frame is not a compliant worker: tolerate
            # the same gap the compliance monitor uses before clearing a violation, so a
            # dropped detection cannot split one violation into two.
            return mgr, WorkerHistory(absence_tolerance=self.cfg.clear_frames)
        except Exception as e:                             # noqa: BLE001
            if not self._quiet:
                print(f"[warn] worker identity disabled: {e}")
            return None, None

    # --- the per-frame core ---------------------------------------------------
    def process(self, frame: np.ndarray, frame_no: int, elapsed_s: float,
                activity_res: Optional[object] = None) -> FrameResult:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:                                     # smoothed, like the live HUD
                self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_t = now

        dets = self.detector.detect(frame)
        tracked, violations = self.tracker.update(dets)
        fc = self.monitor.update(tracked, violations)

        badge_of = self.binder.resolve(frame, tracked) if self.binder is not None else {}
        worker_of = dict(badge_of)
        ident_by_track: Dict[int, Any] = {}
        if self.identity is not None:
            ids, boxes = _tracked_arrays(tracked)
            # `frame` is still the clean camera image here: appearance must never be
            # described from our own overlay.
            ident_by_track = self.identity.update(frame, ids, boxes,
                                                  marker_labels=badge_of)
            worker_of = {t: r.label for t, r in ident_by_track.items()}
            if self.history is not None:
                self.history.update(frame_no, elapsed_s, fc, ident_by_track)

        workers = _roster_rows(self.identity, ident_by_track, fc)
        hud = {"fps": self._fps, "stage_ms": {}, "recording": False,
               "device": self.cfg.device, "activity": activity_res, "workers": workers}

        out = frame
        if self.render_enabled:
            out = self.render(frame, fc, hud, worker_of)

        def _row(ev, tid):
            return {"worker": worker_of.get(tid) or f"Person #{tid}",
                    "violation": ev.class_name, "label": ev.label,
                    "severity": ev.severity, "identified": tid in badge_of}

        return FrameResult(
            frame=out,
            persons=len(fc.persons),
            alerts=[_row(ev, ev.person_id) for ev in fc.events],
            new_alerts=[_row(ev, ev.person_id) for ev in fc.new_events],
            workers=workers,
            fps=self._fps,
            people=_people_rows(fc, worker_of, badge_of, frame.shape),
        )

    def render(self, frame, fc, hud, worker_of):
        """Draw the configured AR view (composite / seethrough / glasses)."""
        from . import overlay
        from .arview import draw_safe_zone, render_seethrough, simulate_glasses
        mode = getattr(self.cfg, "arview_mode", "composite")
        if mode == "composite":
            overlay.annotate(frame, fc, hud, worker_of)
            if getattr(self.cfg, "arview_show_fov", False):
                draw_safe_zone(frame, self.cfg.arview_fov_ratio)
            return frame
        layer = render_seethrough(frame.shape, fc, hud, worker_of,
                                  self.cfg.arview_fov_ratio, self.cfg.arview_scale)
        if mode == "seethrough":
            return layer
        return simulate_glasses(frame, layer, self.cfg.arview_fov_ratio)

    # --- session ---------------------------------------------------------------
    def close(self, elapsed_s: float, frame_no: int = -1) -> None:
        if self.history is not None:
            self.history.close(elapsed_s=elapsed_s, frame_no=frame_no)

    def report(self, now_s=None) -> dict:
        """Non-destructive snapshot. Safe to call repeatedly during a live session --
        pass `now_s` so still-running violations report their elapsed time instead of 0."""
        base = {"violations": self.monitor.summary()}
        if self.history is not None:
            base["workers"] = self.history.report(now_s)
        if self.identity is not None:
            base["identity_stats"] = dict(self.identity.stats)
        return base

    def format_report(self, now_s=None) -> str:
        return (self.history.format_report(now_s=now_s)
                if self.history is not None else "")


# --- helpers shared with run.py -----------------------------------------------
def _tracked_arrays(tracked):
    ids, boxes = [], []
    if len(tracked) == 0 or tracked.tracker_id is None:
        return ids, boxes
    for box, tid in zip(tracked.xyxy, tracked.tracker_id):
        if tid is None:
            continue
        ids.append(int(tid))
        boxes.append([float(v) for v in box])
    return ids, boxes


def _people_rows(fc, worker_of: dict, badge_of: dict, shape) -> list:
    """Per-person boxes normalised to the frame, for a client that draws its own overlay.

    Normalising here rather than on the client is deliberate: the client does not know
    what resolution it was processed at. The phone uploads a downscaled JPEG and displays
    the full-resolution preview, so pixel coordinates would be silently wrong — boxes
    offset by the scale factor, which looks like a tracking bug rather than a units bug.
    """
    h, w = float(shape[0]), float(shape[1])
    if h <= 0 or w <= 0:
        return []
    rows = []
    for p in getattr(fc, "persons", []):
        x1, y1, x2, y2 = p.bbox
        rows.append({
            "id": int(p.tracker_id),
            "box": [round(x1 / w, 4), round(y1 / h, 4),
                    round(x2 / w, 4), round(y2 / h, 4)],
            "label": worker_of.get(p.tracker_id) or f"Person #{p.tracker_id}",
            "badge": p.tracker_id in badge_of,
            "severity": p.worst_severity or "",
            "violations": [v.label for v in p.active],
        })
    return rows


def _roster_rows(identity, ident_by_track: dict, fc) -> list:
    if identity is None:
        return []
    violating = set()
    for person in getattr(fc, "persons", []):
        res = ident_by_track.get(person.tracker_id)
        if res is not None and person.active:
            violating.add(res.uid)
    present = {r.uid for r in ident_by_track.values()}
    rows = [{"label": w.label, "badge": w.marker_confirmed,
             "present": w.uid in present, "violating": w.uid in violating}
            for w in identity.roster()]
    rows.sort(key=lambda r: (not r["present"], not r["violating"], r["label"]))
    return rows
