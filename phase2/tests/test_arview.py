"""Unit tests for glasses-ready rendering (src/arview.py).

The properties tested here are the ones that decide whether a design is *displayable* on
an optical see-through lens at all, and they are easy to break by accident when editing
the overlay:

  * the layer must be genuinely black outside the graphics (black = transparent);
  * nothing may be drawn outside the lens safe zone, because it would never be seen;
  * a worker outside the FOV must still be signalled, not silently dropped.

    python phase2/tests/test_arview.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import arview
from src.compliance import ActiveViolation, FrameCompliance, PersonStatus

results = {}
H, W = 480, 854


def _fc(persons):
    """persons = [(tid, x1, y1, x2, y2, [(cls, sev, label), ...]), ...]"""
    fc = FrameCompliance()
    for tid, x1, y1, x2, y2, viols in persons:
        st = PersonStatus(tracker_id=tid, bbox=(float(x1), float(y1), float(x2), float(y2)))
        for cls, sev, label in viols:
            st.active.append(ActiveViolation(tid, cls, sev, label))
        fc.persons.append(st)
        fc.events.extend(st.active)
    return fc


HUD = {"fps": 24.0, "stage_ms": {}, "recording": False, "device": "cpu", "activity": None}
NO_HELMET = ("No-Helmet", "high", "No hard hat")


# ---- the safe zone is what the lens can show ---------------------------------
def test_safe_rect_is_centred():
    x1, y1, x2, y2 = arview.safe_rect(W, H, 0.5)
    results["arview: safe zone is centred and the right size"] = (
        (x2 - x1) == W // 2 and (y2 - y1) == H // 2
        and abs((x1 + x2) // 2 - W // 2) <= 1 and abs((y1 + y2) // 2 - H // 2) <= 1)


def test_safe_rect_clamped():
    """A nonsense FOV must not produce an empty or inverted rectangle."""
    x1, y1, x2, y2 = arview.safe_rect(W, H, 0.0)
    results["arview: an out-of-range FOV is clamped, not degenerate"] = (x2 > x1 and y2 > y1)


# ---- black is transparent ----------------------------------------------------
def test_layer_is_black_outside_graphics():
    """The projector emits only where graphics are. If the layer were built by copying
    the frame, or a dark 'card' were drawn, most pixels would be non-zero here -- and on
    an additive lens that would wash the world out."""
    fc = _fc([(1, 300, 200, 400, 430, [NO_HELMET])])
    layer = arview.render_seethrough((H, W, 3), fc, HUD, {1: "Alice Tan"})
    lit = float((layer.max(axis=2) > 12).mean())
    results["arview: see-through layer is mostly black (transparent)"] = (lit < 0.10)


def test_nothing_drawn_outside_safe_zone():
    """Anything outside the lens FOV is invisible to the wearer, so drawing there is a
    silent loss of information."""
    fc = _fc([(1, 300, 200, 400, 430, [NO_HELMET]),
              (2, 20, 210, 90, 420, []),           # far left, outside the FOV
              (3, 800, 205, 850, 425, [NO_HELMET])])  # far right, outside
    layer = arview.render_seethrough((H, W, 3), fc, HUD, {1: "A", 2: "B", 3: "C"})
    x1, y1, x2, y2 = arview.safe_rect(W, H, arview.DEFAULT_FOV_RATIO)
    outside = layer.copy()
    outside[y1:y2, x1:x2] = 0
    results["arview: nothing is drawn outside the lens safe zone"] = (
        int(outside.max()) == 0)


# ---- off-display workers are still signalled ---------------------------------
def test_offscreen_worker_gets_an_indicator():
    inside_only = _fc([(1, 380, 200, 470, 430, [])])
    with_outside = _fc([(1, 380, 200, 470, 430, []),
                        (2, 800, 205, 850, 425, [NO_HELMET])])
    a = arview.render_seethrough((H, W, 3), inside_only, HUD, {1: "A"})
    b = arview.render_seethrough((H, W, 3), with_outside, HUD, {1: "A", 2: "B"})
    results["arview: an unsafe worker outside the FOV still gets an indicator"] = (
        int(b.astype(int).sum()) > int(a.astype(int).sum()))


def test_visible_worker_not_duplicated_in_offview_list():
    """A worker inside the FOV is already labelled on their body; repeating them in the
    OFF-VIEW list would waste scarce display space."""
    # Place the worker on the RIGHT of the safe zone so their own reticle and labels
    # cannot land in the bottom-left corner this test inspects.
    fc = _fc([(1, 560, 150, 650, 340, [NO_HELMET])])
    layer = arview.render_seethrough((H, W, 3), fc, HUD, {1: "A"})
    x1, y1, x2, y2 = arview.safe_rect(W, H, arview.DEFAULT_FOV_RATIO)
    strip = layer[y2 - 90:y2, x1:x1 + 240]        # the OFF-VIEW corner
    results["arview: a visible worker is not repeated in the OFF-VIEW list"] = (
        int(strip.max()) == 0)


# ---- the glasses simulation --------------------------------------------------
def test_simulate_crops_to_fov():
    frame = np.full((H, W, 3), 120, np.uint8)
    layer = np.zeros((H, W, 3), np.uint8)
    out = arview.simulate_glasses(frame, layer, 0.5)
    results["arview: the simulated view is cropped to the lens FOV"] = (
        out.shape[0] == H // 2 and out.shape[1] == W // 2)


def test_simulate_is_additive():
    """Graphics must BRIGHTEN the world, never replace it -- an additive combiner cannot
    render black, so UI can only add light."""
    frame = np.full((H, W, 3), 100, np.uint8)
    layer = np.zeros((H, W, 3), np.uint8)
    layer[:, :] = (40, 40, 40)
    plain = arview.simulate_glasses(frame, np.zeros_like(layer), 1.0)
    lit = arview.simulate_glasses(frame, layer, 1.0)
    results["arview: graphics add light rather than occluding the world"] = (
        int(lit.mean()) > int(plain.mean()))


def test_empty_scene_is_blank():
    fc = _fc([])
    layer = arview.render_seethrough((H, W, 3), fc, HUD, {})
    lit = float((layer.max(axis=2) > 12).mean())
    results["arview: an empty scene emits almost nothing"] = (lit < 0.02)


def main() -> int:
    test_safe_rect_is_centred()
    test_safe_rect_clamped()
    test_layer_is_black_outside_graphics()
    test_nothing_drawn_outside_safe_zone()
    test_offscreen_worker_gets_an_indicator()
    test_visible_worker_not_duplicated_in_offview_list()
    test_simulate_crops_to_fov()
    test_simulate_is_additive()
    test_empty_scene_is_blank()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_ARVIEW", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
