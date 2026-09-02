"""How far away can a worker's badge be read?

The ArUco badge is the authoritative identity signal, so every extra metre of read range
is an extra metre over which a worker has a real name rather than "Worker N". This
measures the range of the two detection paths in `phase2/src/workid.py`:

  * **full frame** — the marker detector run once on the whole image;
  * **+ crop** — additionally, for every person not yet bound, the upper part of their
    box cropped, upscaled, and searched again (`workid.crop_detect`).

A synthetic figure (the same crude rendering as `reid_eval.py`) wears a REAL ArUco
marker on the helmet band, scaled as a printed badge would be: `badge_cm / 170 cm` of the
person's pixel height. The figure is rendered at decreasing heights — further away — with
lighting jitter, sensor noise and a touch of optical blur, and the read rate of each path
is reported per height.

**Sampling matters.** A marker pasted at an integer pixel size aligns its module grid with
the pixel grid at some sizes and not others, and a detector's read rate then oscillates
with height — a rendering artifact, not a range. So the card is rendered at high
resolution, anti-aliased, and placed with a random sub-pixel offset and a few percent of
random scale per trial, the way a real badge lands on a real sensor. The marker itself is
genuine (`cv2.aruco`), so this measures the detector; the figure and the optics are
simulated, so treat the pixel sizes as the finding and any metre conversion as a rule of
thumb.

    python -m phase5_workid.badge_eval
    python -m phase5_workid.badge_eval --badge-cm 8 --trials 20
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phase2"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5_workid.reid_eval import (FRAME_H, FRAME_W, SIMILAR, _background,  # noqa: E402
                                     _draw_worker)

RULE = "-" * 70
HEIGHTS = (420, 320, 260, 220, 180, 150, 120, 100)
PERSON_CM = 170.0
_HI_MARKER = 256                      # marker side in the high-resolution card
_HI_QUIET = 43                        # quiet zone: side / 6, as the printed tag has
_HI_CARD = _HI_MARKER + 2 * _HI_QUIET


def aruco_available() -> bool:
    import cv2
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


def _card_hi(marker_id: int, dictionary: str) -> np.ndarray:
    import cv2
    dict_id = getattr(cv2.aruco, dictionary)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dict_id),
                                           int(marker_id), _HI_MARKER)
    card = np.full((_HI_CARD, _HI_CARD), 255, np.uint8)
    card[_HI_QUIET:_HI_QUIET + _HI_MARKER, _HI_QUIET:_HI_QUIET + _HI_MARKER] = marker
    return card


def render_badged_figure(height_px: int, marker_id: int, badge_cm: float = 10.0,
                         dictionary: str = "DICT_4X4_50", seed: int = 0,
                         blur: float = 0.6) -> Tuple[np.ndarray, List[float], List[float]]:
    """A frame with one figure of `height_px` wearing marker `marker_id` on the helmet.
    Returns (frame, person box, marker card box) -- the card box in frame coordinates,
    fractional, so a test can check that a detection lands where the card was put."""
    import cv2
    rng = np.random.default_rng(seed)
    frame = _background(rng)
    hh = float(height_px)
    ww = hh * 0.42
    x1 = float(rng.uniform(20, FRAME_W - ww - 20))
    y1 = float(rng.uniform(10, max(11, FRAME_H - hh - 10)))
    box = [x1, y1, x1 + ww, y1 + hh]
    outfit = SIMILAR[marker_id % len(SIMILAR)]
    _draw_worker(frame, box, outfit, rng, jitter=0.10)

    # Printed size, with a few percent of scale jitter and a sub-pixel position.
    side_f = hh * badge_cm / PERSON_CM * float(rng.uniform(0.96, 1.04))
    card_f = side_f * _HI_CARD / _HI_MARKER
    scale = card_f / _HI_CARD
    mx = x1 + (ww - card_f) / 2.0 + float(rng.uniform(-0.5, 0.5))
    my = y1 + max(0.0, hh * 0.11 - card_f / 2.0) + float(rng.uniform(-0.5, 0.5))
    mx = float(np.clip(mx, 0.0, FRAME_W - card_f - 1.0))
    my = float(np.clip(my, 0.0, FRAME_H - card_f - 1.0))

    card = _card_hi(marker_id, dictionary)
    # Anti-alias before the downscale, as a lens and a sensor's pixel area would.
    sigma = 0.5 / scale if scale < 1.0 else 0.0
    if sigma > 0.3:
        card = cv2.GaussianBlur(card, (0, 0), sigma)
    card_bgr = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    M = np.array([[scale, 0.0, mx], [0.0, scale, my]], dtype=np.float64)
    cv2.warpAffine(card_bgr, M, (FRAME_W, FRAME_H), dst=frame, flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_TRANSPARENT)

    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    noise = rng.integers(-8, 9, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame, box, [mx, my, mx + card_f, my + card_f]


def _tracked(box, tid: int = 1):
    import supervision as sv
    return sv.Detections(xyxy=np.asarray([box], dtype=np.float32),
                         confidence=np.asarray([0.9], dtype=np.float32),
                         class_id=np.asarray([0]), tracker_id=np.asarray([tid]))


def read_rate(height_px: int, crop_detect: bool, trials: int = 12,
              badge_cm: float = 10.0, dictionary: str = "DICT_4X4_50",
              crop_min_height: int = 320) -> float:
    """Fraction of trials in which a fresh binder reads the badge and binds it to the
    person. A fresh binder per trial, because binding is sticky by design."""
    from src.workid import WorkIdBinder
    ok = 0
    for t in range(trials):
        mid = t % 3
        frame, box, _card = render_badged_figure(height_px, mid, badge_cm, dictionary, seed=t)
        binder = WorkIdBinder(dictionary, {0: "A", 1: "B", 2: "C"}, containment=0.5,
                              crop_detect=crop_detect, crop_min_height=crop_min_height)
        got = binder.resolve(frame, _tracked(box))
        ok += got.get(1) == {0: "A", 1: "B", 2: "C"}[mid]
    return 100.0 * ok / max(1, trials)


def sweep(heights: Sequence[int] = HEIGHTS, trials: int = 12, badge_cm: float = 10.0,
          dictionary: str = "DICT_4X4_50", crop_min_height: int = 320) -> List[dict]:
    rows = []
    for h in heights:
        rows.append({
            "height_px": int(h),
            "badge_px": round(h * badge_cm / PERSON_CM, 1),
            "full_frame": read_rate(h, False, trials, badge_cm, dictionary, crop_min_height),
            "with_crop": read_rate(h, True, trials, badge_cm, dictionary, crop_min_height),
        })
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Badge read range: full frame vs crop pass")
    ap.add_argument("--badge-cm", type=float, default=10.0, help="printed badge side (cm)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--crop-min-height", type=int, default=320)
    args = ap.parse_args(argv)
    if not aruco_available():
        print("cv2.aruco unavailable — install opencv-contrib-python.", file=sys.stderr)
        return 1
    rows = sweep(HEIGHTS, args.trials, args.badge_cm, args.dictionary, args.crop_min_height)
    print(RULE)
    print(f"Badge read rate vs. distance  ({args.badge_cm:.0f} cm badge, {args.dictionary}, "
          f"{args.trials} trials/row, {FRAME_W}x{FRAME_H} frame)")
    print(RULE)
    print(f"{'person px':>10}{'badge px':>10}{'full frame':>13}{'+ crop':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['height_px']:>10}{r['badge_px']:>10.1f}{r['full_frame']:>12.0f}%"
              f"{r['with_crop']:>9.0f}%")
    print(RULE)
    print("  person px : the figure's height in the frame (smaller = further away)")
    print("  badge px  : the marker's side in the frame, as a printed badge would scale")
    print("  full frame: detector run once on the whole image (the original path)")
    print("  + crop    : plus an upscaled search of each unbound person's head/torso")
    print()
    print("  A real ArUco marker on a synthetic figure, anti-aliased and placed at a random")
    print("  sub-pixel offset and scale per trial, with noise and blur: this measures the")
    print("  detector's range in pixels. Metres depend on the lens; on a phone at 640 px")
    print("  wide a 10 cm badge is roughly 8-9 px at 5 m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
