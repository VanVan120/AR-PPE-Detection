"""Unit tests for the badge read-range measurement (phase5_workid/badge_eval.py) and the
crop-detection path it measures (phase2/src/workid.py).

Real ArUco markers on synthetic figures, so this needs cv2.aruco (opencv-contrib-python)
and nothing else: no weights, no dataset, no camera. The assertions are on properties
that must hold for any ArUco build -- a close badge reads, the crop pass never loses a
marker the full frame found, and a crop-path detection is mapped back to where the card
actually is -- not on the exact read rate at a marginal size, which the OpenCV version
decides.

    python phase5_workid/tests/test_badge_eval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase2"))

from phase5_workid.badge_eval import (aruco_available, read_rate, render_badged_figure,  # noqa: E402
                                      _tracked)

results = {}


def test_large_badge_reads_on_both_paths():
    results["badge: a close badge is read by both paths"] = (
        read_rate(420, False, trials=4) == 100.0 and read_rate(420, True, trials=4) == 100.0)


def test_crop_path_never_worse():
    """The crop pass adds a second look; it must never lose a marker the full frame found,
    at any distance."""
    ok = True
    for h in (420, 260, 180, 150):
        ok = ok and read_rate(h, True, trials=6) >= read_rate(h, False, trials=6)
    results["badge: the crop pass never reads fewer badges than the full frame"] = ok


def test_crop_path_maps_back_to_the_card():
    """A detection made in the upscaled crop must come back in FRAME coordinates, on the
    card, or it could never bind to the right person. Checked directly on the crop path,
    with the full-frame result withheld, at a size where the crop reads reliably."""
    from src.workid import WorkIdBinder
    b = WorkIdBinder("DICT_4X4_50", {0: "A", 1: "B", 2: "C"}, crop_detect=True)
    checked, ok = 0, True
    for seed in range(6):
        mid = seed % 3
        frame, box, card = render_badged_figure(300, mid, 10.0, seed=seed)
        ids, boxes = b._detect_in_crops(frame, [1], [box], [], [])
        if not ids:
            continue
        checked += 1
        cx, cy = (card[0] + card[2]) / 2.0, (card[1] + card[3]) / 2.0
        mx, my = (boxes[0][0] + boxes[0][2]) / 2.0, (boxes[0][1] + boxes[0][3]) / 2.0
        ok = ok and ids[0] == mid and abs(mx - cx) <= 3.0 and abs(my - cy) <= 3.0
        # ... and the mapped marker binds to the person through the normal path.
        got = WorkIdBinder("DICT_4X4_50", {0: "A", 1: "B", 2: "C"},
                           crop_detect=True).resolve(frame, _tracked(box))
        ok = ok and got.get(1) == {0: "A", 1: "B", 2: "C"}[mid]
    results["badge: a crop-path detection lands on the card, in frame coordinates"] = (
        checked >= 3 and ok)


def test_crop_path_extends_range():
    """Somewhere between 'reads on the full frame' and 'too small for anything' the crop
    pass must read badges the full frame misses -- that is its only reason to exist.
    Summed across the marginal sizes so no single size decides."""
    full = sum(read_rate(h, False, trials=8) for h in (220, 180, 150))
    crop = sum(read_rate(h, True, trials=8) for h in (220, 180, 150))
    results["badge: across the marginal sizes the crop pass reads more than the full frame"] = (
        crop > full)


def test_bound_person_is_not_re_searched():
    """Once a person is bound the crop search stops for them (it is the expensive part),
    and the sticky binding carries on without it."""
    from src.workid import WorkIdBinder
    frame, box, _card = render_badged_figure(420, 1, 10.0, seed=3)
    b = WorkIdBinder("DICT_4X4_50", {1: "B"}, crop_detect=True)
    first = b.resolve(frame, _tracked(box))
    calls = []
    orig = b._detect_in_crops

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    b._detect_in_crops = spy
    b._detect_markers = lambda f: ([], [])       # hide the badge on the second frame
    second = b.resolve(frame, _tracked(box))
    results["badge: a bound person keeps the name and is not crop-searched again"] = (
        first.get(1) == "B" and second.get(1) == "B" and len(calls) == 1)


def main() -> int:
    if not aruco_available():
        print("SKIP - cv2.aruco unavailable (install opencv-contrib-python)")
        print("ALL_BADGE_EVAL True")
        return 0
    test_large_badge_reads_on_both_paths()
    test_crop_path_never_worse()
    test_crop_path_maps_back_to_the_card()
    test_crop_path_extends_range()
    test_bound_person_is_not_re_searched()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_BADGE_EVAL", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
