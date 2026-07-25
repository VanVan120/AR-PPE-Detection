"""Unit tests for appearance re-ID (src/reid.py) and persistent identity (src/identity.py).

Synthetic frames only -- coloured rectangles standing in for differently-dressed
workers -- so this runs with no camera, no weights and no dataset.

The tests target the failure modes that actually matter for a worker-tracking system:
losing someone across a track-id change, merging two different people into one identity,
and letting appearance overrule a physical badge.

    python phase2/tests/test_identity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.identity import IdentityManager
from src.reid import ColorHistogramEmbedder, cosine_similarity, crop_person

results = {}

FRAME_H, FRAME_W = 480, 640
# BGR body colours for three "workers"
RED, BLUE, GREEN = (40, 40, 200), (200, 60, 40), (60, 190, 60)


def _frame(people):
    """people = [(x1, y1, x2, y2, bgr), ...] painted on a mid-grey background."""
    f = np.full((FRAME_H, FRAME_W, 3), 120, dtype=np.uint8)
    for x1, y1, x2, y2, colour in people:
        f[int(y1):int(y2), int(x1):int(x2)] = colour
    return f


def _person(x, colour, w=60, h=160, y=150):
    return (x, y, x + w, y + h, colour)


def _mgr(**kw):
    kw.setdefault("match_threshold", 0.62)
    return IdentityManager(ColorHistogramEmbedder(), **kw)


# ---- descriptor sanity -------------------------------------------------------
def test_same_colour_matches_better_than_different():
    emb = ColorHistogramEmbedder()
    f = _frame([_person(50, RED), _person(300, BLUE)])
    a = emb.embed_one(crop_person(f, [50, 150, 110, 310]))
    b = emb.embed_one(crop_person(f, [300, 150, 360, 310]))
    # same worker seen again, slightly shifted / different size
    f2 = _frame([_person(120, RED, w=64, h=170, y=140)])
    a2 = emb.embed_one(crop_person(f2, [120, 140, 184, 310]))
    same = float(cosine_similarity(a[None], a2[None])[0, 0])
    diff = float(cosine_similarity(a[None], b[None])[0, 0])
    results["reid: same worker scores far above a different worker"] = (
        same > 0.9 and diff < 0.5 and same > diff)


def test_cosine_handles_zero_vector():
    z = np.zeros((1, 8), dtype=np.float32)
    v = np.ones((1, 8), dtype=np.float32)
    s = cosine_similarity(z, v)
    results["reid: zero vector yields 0.0, not NaN"] = bool(np.isfinite(s).all())


def test_crop_person_rejects_out_of_frame():
    f = _frame([])
    results["reid: degenerate / off-frame boxes return None"] = (
        crop_person(f, [10, 10, 11, 11]) is None
        and crop_person(f, [-50, -50, -10, -10]) is None
        and crop_person(f, [10, 10, 100, 200]) is not None)


# ---- the core capability: survive a track-id change --------------------------
def test_identity_survives_track_id_change():
    """The whole point. Same person, tracker gives a NEW id after an occlusion gap;
    the worker identity must be recovered rather than a stranger invented."""
    m = _mgr()
    f = _frame([_person(50, RED)])
    first = m.update(f, [1], [[50, 150, 110, 310]])
    label_before = first[1].label
    for _ in range(5):                                   # person absent entirely
        m.update(_frame([]), [], [])
    f2 = _frame([_person(70, RED)])
    again = m.update(f2, [9], [[70, 150, 130, 310]])     # new track id
    results["identity: same worker recovered after a track-id change"] = (
        again[9].label == label_before and again[9].source == "appearance")


def test_two_lookalikes_stay_separate():
    """Mutual exclusion: two people visible at once can never collapse into one worker,
    even when they look identical -- the more dangerous error."""
    m = _mgr()
    f = _frame([_person(50, RED), _person(300, RED)])
    out = m.update(f, [1, 2], [[50, 150, 110, 310], [300, 150, 360, 310]])
    results["identity: two simultaneous lookalikes get distinct workers"] = (
        out[1].uid != out[2].uid)


def test_distinct_people_not_merged():
    m = _mgr()
    f = _frame([_person(50, RED), _person(300, BLUE)])
    out = m.update(f, [1, 2], [[50, 150, 110, 310], [300, 150, 360, 310]])
    results["identity: differently-dressed people get distinct workers"] = (
        out[1].uid != out[2].uid and len(m.workers) == 2)


# ---- markers are authoritative ----------------------------------------------
def test_marker_wins_and_promotes():
    """An anonymous appearance identity must be PROMOTED in place when a badge finally
    confirms it -- same record, real name, history intact -- not duplicated."""
    m = _mgr()
    f = _frame([_person(50, RED)])
    m.update(f, [1], [[50, 150, 110, 310]])              # anonymous first
    seen_before = list(m.workers.values())[0].frames_seen
    out = m.update(f, [1], [[50, 150, 110, 310]], marker_labels={1: "Alice Tan"})
    w = m.workers[out[1].uid]
    results["identity: a badge promotes the anonymous worker in place"] = (
        out[1].label == "Alice Tan" and out[1].source == "marker"
        and len(m.workers) == 1 and w.frames_seen > seen_before
        and m.stats["promotions"] == 1)


def test_uid_survives_promotion():
    """The uid must NOT change when an anonymous worker is renamed by a badge. Anything
    holding it -- above all the violation history -- would otherwise be orphaned at
    exactly the moment the worker finally gets a real name."""
    m = _mgr()
    f = _frame([_person(50, RED)])
    before = m.update(f, [1], [[50, 150, 110, 310]])[1]
    after = m.update(f, [1], [[50, 150, 110, 310]], marker_labels={1: "Alice Tan"})[1]
    results["identity: uid is stable across a rename (history stays attached)"] = (
        before.uid == after.uid and before.label != after.label
        and after.label == "Alice Tan")


def test_marker_never_duplicated_across_tracks():
    """One badge, two tracks: the name must live on exactly one of them."""
    m = _mgr()
    f = _frame([_person(50, RED), _person(300, BLUE)])
    boxes = [[50, 150, 110, 310], [300, 150, 360, 310]]
    m.update(f, [1, 2], boxes, marker_labels={1: "Alice Tan"})
    out = m.update(f, [1, 2], boxes, marker_labels={2: "Alice Tan"})
    holders = [t for t, r in out.items() if r.label == "Alice Tan"]
    results["identity: one badge name is never worn by two tracks at once"] = (
        holders == [2])


def test_marker_identity_persists_without_tag():
    """Stickiness: once badged, the name survives frames where the tag isn't visible."""
    m = _mgr()
    f = _frame([_person(50, RED)])
    m.update(f, [1], [[50, 150, 110, 310]], marker_labels={1: "Bob Lim"})
    out = m.update(f, [1], [[50, 150, 110, 310]])        # tag hidden now
    results["identity: badge name persists when the tag is not visible"] = (
        out[1].label == "Bob Lim")


# ---- refuse to guess ---------------------------------------------------------
def test_different_person_not_matched_on_re_entry():
    """Threshold enforcement on the re-entry path, where mutual exclusion cannot help:
    one worker leaves, a differently-dressed one arrives. Matching them would silently
    hand a stranger someone else's identity and violation history."""
    m = _mgr()
    m.update(_frame([_person(50, RED)]), [1], [[50, 150, 110, 310]])
    m.update(_frame([]), [], [])
    out = m.update(_frame([_person(50, BLUE)]), [4], [[50, 150, 110, 310]])
    results["identity: a different person on re-entry is not matched"] = (
        out[4].source == "new" and len(m.workers) == 2)


def test_ambiguous_lookalikes_refuse_to_guess():
    """The margin rule, and the realistic construction-site case: two workers in
    IDENTICAL PPE are both known, then one returns alone. The system cannot tell which,
    so it must refuse rather than pick one at random and corrupt that worker's record."""
    m = _mgr()
    f = _frame([_person(50, RED), _person(300, RED)])
    m.update(f, [1, 2], [[50, 150, 110, 310], [300, 150, 360, 310]])
    m.update(_frame([]), [], [])
    out = m.update(_frame([_person(170, RED)]), [5], [[170, 150, 230, 310]])
    results["identity: ambiguous lookalikes -> refuses to guess, opens a new worker"] = (
        out[5].source == "new" and len(m.workers) == 3)


def test_small_boxes_rejected():
    """A tiny distant box must not enter the gallery and pollute it."""
    m = _mgr(min_box_height=100)
    f = _frame([_person(50, RED, w=20, h=40, y=100)])
    out = m.update(f, [1], [[50, 100, 70, 140]])
    w = m.workers[out[1].uid]
    results["identity: undersized boxes contribute no exemplar"] = (len(w.exemplars) == 0)


def test_roster_and_stats():
    m = _mgr()
    f = _frame([_person(50, RED), _person(300, BLUE)])
    m.update(f, [1, 2], [[50, 150, 110, 310], [300, 150, 360, 310]])
    roster = m.roster()
    results["identity: roster lists every known worker"] = (
        len(roster) == 2 and m.stats["new_workers"] == 2)


def main() -> int:
    test_same_colour_matches_better_than_different()
    test_cosine_handles_zero_vector()
    test_crop_person_rejects_out_of_frame()
    test_identity_survives_track_id_change()
    test_two_lookalikes_stay_separate()
    test_distinct_people_not_merged()
    test_marker_wins_and_promotes()
    test_uid_survives_promotion()
    test_marker_never_duplicated_across_tracks()
    test_marker_identity_persists_without_tag()
    test_different_person_not_matched_on_re_entry()
    test_ambiguous_lookalikes_refuse_to_guess()
    test_small_boxes_rejected()
    test_roster_and_stats()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_IDENTITY", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
