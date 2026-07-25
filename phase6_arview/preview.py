"""Render the three AR views side by side, so the glasses design can be reviewed
without a headset.

    python -m phase6_arview.preview                  # synthetic scene, zero downloads
    python -m phase6_arview.preview --image shot.jpg # run the real detector on a photo
    python -m phase6_arview.preview --fov 0.5        # tighter lens

Produces `outputs/ar_preview.png` with three panels:

  1. **MONITOR** — the Phase 2 composite HUD, with the glasses FOV outlined. Everything
     outside that dashed box is invisible to a headset, which is the point of the panel.
  2. **SEE-THROUGH LAYER** — what the projector actually emits: bright graphics on black.
     Black = transparent. If a UI element vanishes here, it would vanish on the lens.
  3. **THROUGH THE GLASSES** — the layer additively blended over the dimmed world and
     cropped to the lens FOV: an approximation of what the wearer sees.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from src import arview, overlay                                        # noqa: E402
from src.compliance import ActiveViolation, FrameCompliance, PersonStatus  # noqa: E402


def synthetic_scene(w: int = 960, h: int = 540):
    """A stand-in site scene: three workers, two of them unsafe, one near the edge so the
    off-display indicator is exercised."""
    rng = np.random.default_rng(0)
    frame = np.full((h, w, 3), 96, np.uint8)
    cv2.rectangle(frame, (0, 0), (w, int(h * 0.34)), (150, 135, 118), -1)   # sky/wall
    cv2.rectangle(frame, (0, int(h * 0.34)), (w, h), (86, 92, 100), -1)     # ground
    for _ in range(60):                                                     # texture
        x, y = int(rng.integers(0, w)), int(rng.integers(int(h * 0.34), h))
        cv2.circle(frame, (x, y), int(rng.integers(2, 9)), (70, 76, 84), -1)

    people = [(120, 190, 250, 470, "Alice Tan", [("No-Helmet", "high", "No hard hat")]),
              (430, 210, 552, 480, "Bob Lim", []),
              (835, 200, 950, 470, "Worker 3", [("No-Vest", "medium", "No hi-vis vest")])]
    fc = FrameCompliance()
    worker_of = {}
    for i, (x1, y1, x2, y2, name, viols) in enumerate(people, start=1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 70, 90), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 40), (60, 190, 235), -1)   # helmet-ish
        st = PersonStatus(tracker_id=i, bbox=(float(x1), float(y1), float(x2), float(y2)))
        for cls, sev, label in viols:
            st.active.append(ActiveViolation(i, cls, sev, label))
        fc.persons.append(st)
        fc.events.extend(st.active)
        worker_of[i] = name
    return frame, fc, worker_of


def detected_scene(path: str):
    """Run the real Phase 2 detector + compliance on one image."""
    from src.compliance import ComplianceMonitor
    from src.config import load_config
    from src.detector import Detector

    cfg = load_config(os.path.join(_ROOT, "phase2", "config.yaml"))
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"could not read image: {path}")
    det = Detector(cfg.weights_path, cfg.confidence_threshold, cfg.imgsz, cfg.device)
    names = {v: k for k, v in det.names.items()} if isinstance(det.names, dict) else {}
    person_ids = [k for k, v in det.names.items() if v == cfg.person_class]
    rules_by_id = {k: r for k, v in det.names.items()
                   for r in cfg.violation_rules if r.class_name == v}
    dets = det.detect(frame)
    persons = dets[np.isin(dets.class_id, person_ids)] if len(dets) else dets
    persons.tracker_id = np.arange(1, len(persons) + 1)
    viols = dets[np.isin(dets.class_id, list(rules_by_id))] if len(dets) else dets
    mon = ComplianceMonitor(rules_by_id, debounce_frames=1, clear_frames=1,
                            containment_thresh=cfg.association_containment)
    fc = mon.update(persons, viols)
    return frame, fc, {p.tracker_id: f"Worker {p.tracker_id}" for p in fc.persons}


def _label(img, text, colour=(240, 240, 245)):
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def build_preview(frame, fc, worker_of, fov: float, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    hud = {"fps": 24.0, "stage_ms": {}, "recording": False, "device": "cuda",
           "activity": None,
           "workers": [{"label": n, "badge": n[0].isupper() and " " in n,
                        "present": True,
                        "violating": any(p.active for p in fc.persons
                                         if worker_of.get(p.tracker_id) == n)}
                       for n in dict.fromkeys(worker_of.values())]}

    monitor = overlay.annotate(frame.copy(), fc, hud, worker_of)
    arview.draw_safe_zone(monitor, fov)

    layer = arview.render_seethrough(frame.shape, fc, hud, worker_of, fov, scale)
    through = arview.simulate_glasses(frame, layer, fov)
    through = cv2.resize(through, (w, int(through.shape[0] * w / through.shape[1])),
                         interpolation=cv2.INTER_LINEAR)

    panels = [_label(monitor, "1  MONITOR  - Phase 2 HUD (dashed box = what glasses can show)"),
              _label(layer, "2  SEE-THROUGH LAYER  - what the projector emits (black = transparent)"),
              _label(through, "3  THROUGH THE GLASSES  - additive over the world, cropped to FOV")]
    hh = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((hh - p.shape[0], p.shape[1], 3), np.uint8)])
              for p in panels]
    return np.hstack(panels)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Preview the AR-glasses render modes")
    ap.add_argument("--image", default="", help="run the real detector on this photo")
    ap.add_argument("--fov", type=float, default=arview.DEFAULT_FOV_RATIO,
                    help="fraction of the camera frame the lens shows (0.62 default)")
    ap.add_argument("--scale", type=float, default=1.25,
                    help="see-through text/stroke scale (glasses need bigger)")
    ap.add_argument("--out", default="outputs/ar_preview.png")
    args = ap.parse_args(argv)

    if args.image:
        frame, fc, worker_of = detected_scene(args.image)
    else:
        frame, fc, worker_of = synthetic_scene()
        print("[info] synthetic scene (pass --image to run the real detector on a photo)")

    combo = build_preview(frame, fc, worker_of, args.fov, args.scale)
    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, combo)
    print(f"wrote {out}  ({combo.shape[1]}x{combo.shape[0]})")
    print(f"lens FOV ratio {args.fov}  ->  a headset shows only the middle "
          f"{args.fov * 100:.0f}% of the camera view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
