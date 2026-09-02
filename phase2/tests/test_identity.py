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


# =============================================================================
# Tracking enhancements (spec: docs/superpowers/specs/2026-09-02-workid-tracking-design.md)
# =============================================================================
class _MeanColourEmbedder:
    """A deliberately simple descriptor for the DECISION-RULE tests below: the unit
    vector of the crop's mean BGR colour. It makes every similarity exactly predictable,
    so a test can place a return crop between the floor and the threshold on purpose --
    something a real histogram cannot be made to do reliably."""
    name = "mean-colour"
    dim = 3

    def embed(self, crops):
        out = []
        for c in crops:
            v = np.asarray(c, dtype=np.float32).reshape(-1, 3).mean(axis=0)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return (np.stack(out).astype(np.float32) if out
                else np.zeros((0, self.dim), dtype=np.float32))


PURE_RED, PURE_BLUE = (0, 0, 200), (200, 0, 0)
# unit (0, .835, .55): cosine 0.55 against PURE_RED -- below the 0.62 threshold, above
# the 0.45 gate floor -- and 0.0 against PURE_BLUE.
HALF_RED = (0, 167, 110)
BOX = [50, 150, 110, 310]


def _box(x, w=60, h=160, y=150):
    return [x, y, x + w, y + h]


def _rule_mgr(**kw):
    kw.setdefault("match_threshold", 0.62)
    return IdentityManager(_MeanColourEmbedder(), **kw)


# ---- probation: a track must earn a worker ----------------------------------
def test_probation_phantom_creates_no_worker():
    """A one-frame false detection must not become a permanent 'Worker N'. Today's
    behaviour commits on the first frame; with probation the phantom never reaches
    the roster while the real worker still gets identified."""
    m = _mgr(probation_frames=3)
    f = _frame([_person(50, RED)])
    o1 = m.update(f, [1], [BOX])
    o2 = m.update(_frame([_person(50, RED), _person(300, BLUE)]), [1, 9],
                  [BOX, _box(300)])                       # 9 = one-frame phantom
    o3 = m.update(f, [1], [BOX])
    results["probation: a one-frame phantom track creates no worker"] = (
        1 not in o1 and 1 not in o2 and 9 not in o2
        and 1 in o3 and o3[1].source == "new"
        and len(m.workers) == 1 and m.stats["new_workers"] == 1)


def test_probation_recovers_with_pooled_descriptor():
    """During probation the embeddings are pooled; the commit must still re-identify
    the returning worker, just a few frames later."""
    m = _mgr(probation_frames=3)
    f = _frame([_person(50, RED)])
    for _ in range(3):
        m.update(f, [1], [BOX])
    before = m.update(f, [1], [BOX])[1]
    for _ in range(4):
        m.update(_frame([]), [], [])
    f2 = _frame([_person(70, RED)])
    o1 = m.update(f2, [9], [_box(70)])
    o2 = m.update(f2, [9], [_box(70)])
    o3 = m.update(f2, [9], [_box(70)])
    w = m.workers[before.uid]
    results["probation: the returning worker is recovered on commit, history intact"] = (
        9 not in o1 and 9 not in o2 and o3[9].uid == before.uid
        and o3[9].source == "appearance" and len(m.workers) == 1
        and len(w.exemplars) >= 1)


def test_probation_marker_commits_immediately():
    """A badge is authoritative and must not wait out the probation."""
    m = _mgr(probation_frames=3)
    out = m.update(_frame([_person(50, RED)]), [1], [BOX], marker_labels={1: "Alice Tan"})
    results["probation: a badge read commits the track at once"] = (
        out[1].label == "Alice Tan" and out[1].source == "marker" and len(m.workers) == 1)


def test_probation_abandoned_track_is_forgotten():
    """A track that dies during probation must not leak a pending entry that could bind
    a recycled track id to stale evidence later."""
    m = _mgr(probation_frames=3)
    m.update(_frame([_person(50, RED)]), [1], [BOX])
    for _ in range(10):
        m.update(_frame([]), [], [])
    o = m.update(_frame([_person(300, BLUE)]), [1], [_box(300)])      # id 1 recycled
    o = m.update(_frame([_person(300, BLUE)]), [1], [_box(300)])
    o = m.update(_frame([_person(300, BLUE)]), [1], [_box(300)])
    results["probation: an abandoned pending track does not taint a recycled id"] = (
        1 in o and len(m.workers) == 1 and len(m._pending) == 0)


# ---- spatio-temporal gating ---------------------------------------------------
def test_gate_vetoes_implausible_lookalike():
    """Same colour, but 450 px away two frames after the worker was last seen: nobody
    walks that fast. Without the gate the appearance match silently hands the newcomer
    the other worker's identity and history."""
    def run(gate):
        m = _rule_mgr(gate=gate)
        for _ in range(3):
            m.update(_frame([_person(50, PURE_RED)]), [1], [BOX])
        m.update(_frame([]), [], [])
        return m.update(_frame([_person(500, PURE_RED)]), [9], [_box(500)])[9].source
    results["gate: a recently-seen worker who cannot have got here is vetoed"] = (
        run(False) == "appearance" and run(True) == "new")


def test_gate_position_assists_below_threshold():
    """Appearance alone is inconclusive (0.55 < 0.62), but exactly one recently-seen
    worker can physically be at this spot -> accept, and say so in `source`."""
    def run(gate):
        m = _rule_mgr(gate=gate, gate_floor=0.45)
        f = _frame([_person(50, PURE_RED), _person(400, PURE_BLUE)])
        for _ in range(3):
            m.update(f, [1, 2], [BOX, _box(400)])
        a_uid = m.update(f, [1, 2], [BOX, _box(400)])[1].uid
        for _ in range(2):
            m.update(_frame([]), [], [])
        out = m.update(_frame([_person(60, HALF_RED)]), [9], [_box(60)])[9]
        return out, a_uid
    off, a_off = run(False)
    on, a_on = run(True)
    results["gate: a lone plausible worker is recovered on position + weak appearance"] = (
        off.source == "new" and on.uid == a_on and on.source == "position"
        and 0.5 < on.similarity < 0.62)


def test_gate_two_plausible_lookalikes_still_refuse():
    """Two identical workers stood side by side, both vanished, one is back between the
    two spots: position cannot say which, appearance cannot either -> refuse to guess."""
    m = _rule_mgr(gate=True)
    f = _frame([_person(50, PURE_RED), _person(130, PURE_RED)])
    for _ in range(3):
        m.update(f, [1, 2], [BOX, _box(130)])
    for _ in range(2):
        m.update(_frame([]), [], [])
    out = m.update(_frame([_person(90, PURE_RED)]), [9], [_box(90)])[9]
    results["gate: two plausible lookalikes -> still refuses to guess"] = (
        out.source == "new" and len(m.workers) == 3)


def test_gate_compensates_camera_pan():
    """A hand-held or head-worn camera moves the whole scene. The gate must predict
    where a lost worker's image has DRIFTED to, using the other tracks' motion -- or a
    correct match 160 px away would be vetoed as impossible."""
    m = _rule_mgr(gate=True)
    f = _frame([_person(50, PURE_RED), _person(300, PURE_BLUE)])
    for _ in range(3):
        a_uid = m.update(f, [1, 2], [BOX, _box(300)])[1].uid
    for x in (340, 380, 420, 460):                        # B carries the pan: +40 px/frame
        f = _frame([_person(x, PURE_BLUE)])
        ids, boxes = [2], [_box(x)]
        if x == 460:                                     # A returns, 160 px along
            f = _frame([_person(x, PURE_BLUE), _person(210, PURE_RED)])
            ids, boxes = [2, 9], [_box(x), _box(210)]
        out = m.update(f, ids, boxes)
    results["gate: camera pan is compensated, so a drifted true match is kept"] = (
        9 in out and out[9].uid == a_uid and len(m.workers) == 2)


def test_gate_decisive_appearance_overrides_veto():
    """Three distinctly dressed workers are known. The red one reappears 450 px away two
    frames later -- impossible by position, but red is unmistakably red and nothing else
    known comes close. Decisive appearance must win; the veto is for lookalikes."""
    m = _rule_mgr(gate=True)
    f = _frame([_person(50, PURE_RED), _person(300, PURE_BLUE), _person(500, (0, 200, 0))])
    boxes = [BOX, _box(300), _box(500)]
    for _ in range(3):
        o = m.update(f, [1, 2, 3], boxes)
    a_uid = o[1].uid
    f2 = _frame([_person(300, PURE_BLUE), _person(500, (0, 200, 0))])
    m.update(f2, [2, 3], boxes[1:])
    out = m.update(_frame([_person(500, PURE_RED)]), [9], [_box(500)])[9]
    results["gate: decisive appearance overrides the spatial veto"] = (
        out.uid == a_uid and out.source == "appearance" and len(m.workers) == 3)


def test_gate_co_visible_worker_is_never_a_candidate():
    """A worker seen on the very frame a new track appeared was, by definition, not that
    track. Even with identical colours, a lone candidate and a plausible position, the
    new track must open a new worker -- and no appearance evidence may bring the
    co-visible worker back."""
    m = _rule_mgr(gate=True, probation_frames=3)
    f = _frame([_person(50, PURE_RED)])
    for _ in range(3):
        o = m.update(f, [1], [BOX])                  # commits on the third sighting
    a_uid = o[1].uid
    both = _frame([_person(50, PURE_RED), _person(80, PURE_RED)])
    m.update(both, [1, 9], [BOX, _box(80)])              # one frame of co-visibility
    o2 = m.update(_frame([_person(80, PURE_RED)]), [9], [_box(80)])
    o3 = m.update(_frame([_person(80, PURE_RED)]), [9], [_box(80)])
    results["gate: a co-visible worker is excluded outright, not merely vetoed"] = (
        9 not in o2 and o3[9].source == "new" and o3[9].uid != a_uid
        and len(m.workers) == 2)


def test_position_rule_rejects_a_stranger_in_the_spot():
    """Handover: a red worker leaves a spot and, eight frames later, a blue one steps into
    it. Position says 'plausible', but blue does not look like red -- the position rule
    must not hand the newcomer the red worker's identity and violation history."""
    m = _rule_mgr(gate=True)
    f = _frame([_person(50, PURE_RED)])
    for _ in range(3):
        a_uid = m.update(f, [1], [BOX])[1].uid
    for _ in range(8):
        m.update(_frame([]), [], [])
    out = m.update(_frame([_person(55, PURE_BLUE)]), [9], [_box(55)])[9]
    results["gate: a stranger stepping into a vacated spot is not matched by position"] = (
        out.source == "new" and out.uid != a_uid and len(m.workers) == 2)


def test_gate_falls_back_to_appearance_beyond_horizon():
    """A worker gone for longer than the horizon carries no position information; the
    decision must be purely by appearance, exactly as before."""
    m = _rule_mgr(gate=True, gate_horizon=5)
    for _ in range(3):
        a_uid = m.update(_frame([_person(50, PURE_RED)]), [1], [BOX])[1].uid
    for _ in range(10):
        m.update(_frame([]), [], [])
    out = m.update(_frame([_person(500, PURE_RED)]), [9], [_box(500)])[9]
    results["gate: beyond the horizon position is ignored and appearance decides"] = (
        out.uid == a_uid and out.source == "appearance")


def test_time_mode_uses_seconds_not_frames():
    """A laptop that processes 3 of every 30 camera frames must not run a gate three
    times tighter than configured. Given `now_s`, the gate measures the gap in seconds:
    the same two-frame gap is 'far too fast' at 0.07 s and 'an easy walk' at 4 s."""
    def run(seconds_per_frame):
        m = _rule_mgr(gate=True, gate_speed_s=0.9, gate_horizon_s=10.0)
        t = 0.0
        for _ in range(3):
            a_uid = m.update(_frame([_person(50, PURE_RED)]), [1], [BOX], now_s=t)[1].uid
            t += seconds_per_frame
        m.update(_frame([]), [], [], now_s=t)
        t += seconds_per_frame
        # 300 px is ~1.9 body heights: allowed after ~1.5 s of walking, not after 0.07 s.
        out = m.update(_frame([_person(350, PURE_RED)]), [9], [_box(350)], now_s=t)[9]
        return out.uid == a_uid
    results["gate: with timestamps the tolerance grows with seconds gone, not frames"] = (
        run(0.033) is False and run(2.0) is True)


# ---- merges that carry history ---------------------------------------------
def test_badge_merges_anonymous_record_into_named():
    """Alice was badged earlier. Later a person is seen without the badge visible and
    becomes 'Worker 1'; when the badge is finally read on them, that anonymous record
    must be MERGED into Alice's -- not stranded with its own history."""
    m = _rule_mgr()
    for _ in range(2):
        a = m.update(_frame([_person(50, PURE_RED)]), [1], [BOX],
                     marker_labels={1: "Alice Tan"})[1].uid
    for _ in range(2):
        m.update(_frame([]), [], [])
    anon = m.update(_frame([_person(300, PURE_BLUE)]), [5], [_box(300)])[5]
    out = m.update(_frame([_person(300, PURE_BLUE)]), [5], [_box(300)],
                   marker_labels={5: "Alice Tan"})[5]
    merges = m.take_merges()
    results["merge: a badge on an anonymous record folds it into the named worker"] = (
        anon.source == "new" and anon.uid != a and out.uid == a
        and out.label == "Alice Tan" and len(m.workers) == 1
        and merges == [(anon.uid, a)] and m.take_merges() == []
        and m.stats["merges"] == 1)


# ---- offline consolidation ---------------------------------------------------
def test_consolidate_uses_presence_overlap():
    """Online, a returning worker was ambiguous between A and badged Bob, so a fragment
    'C' was opened. Bob then returned elsewhere and was confirmed by his badge -- so
    Bob was visible at the same time as C and cannot be C. Offline that leaves A as the
    only candidate, and C is folded into A."""
    for gate in (False, True):
        m = _rule_mgr(gate=gate)
        f = _frame([_person(50, PURE_RED), _person(130, PURE_RED)])
        for _ in range(3):
            o = m.update(f, [1, 2], [BOX, _box(130)], marker_labels={2: "Bob Lim"})
        a_uid, b_uid = o[1].uid, o[2].uid
        for _ in range(2):
            m.update(_frame([]), [], [])
        c = m.update(_frame([_person(90, PURE_RED)]), [9], [_box(90)])[9]
        for _ in range(3):
            o = m.update(_frame([_person(90, PURE_RED), _person(140, PURE_RED)]), [9, 10],
                         [_box(90), _box(140)], marker_labels={10: "Bob Lim"})
        ok_online = (c.source == "new" and o[10].uid == b_uid and len(m.workers) == 3)
        merges = m.consolidate()
        results[f"consolidate: overlap constraint folds the fragment into A (gate={gate})"] = (
            ok_online and merges == [(c.uid, a_uid)] and len(m.workers) == 2
            and a_uid in m.workers and m.workers[a_uid].label.startswith("Worker"))


def test_consolidate_never_merges_co_visible_records():
    """Two identical people seen at the same moment are provably two people."""
    m = _rule_mgr(gate=True)
    f = _frame([_person(50, PURE_RED), _person(130, PURE_RED)])
    for _ in range(3):
        m.update(f, [1, 2], [BOX, _box(130)])
    results["consolidate: co-visible lookalikes are never merged"] = (
        m.consolidate() == [] and len(m.workers) == 2)


def test_consolidate_respects_badges_and_ambiguity():
    """Two badge-confirmed records never merge with each other. An anonymous fragment
    merges into a named record only when the decision rule is satisfied: with position
    it picks the neighbour; without, two identical names are ambiguous and it refuses."""
    def run(gate):
        # Live matching runs WITHOUT the gate so the fragment actually exists; the gate is
        # then switched on (or not) for the offline pass, which is what this test is about.
        m = _rule_mgr(gate=False)
        for _ in range(3):
            o = m.update(_frame([_person(50, PURE_RED)]), [1], [BOX],
                         marker_labels={1: "Alice Tan"})
        a_uid = o[1].uid
        for _ in range(2):
            m.update(_frame([]), [], [])
        for _ in range(3):
            m.update(_frame([_person(400, PURE_RED)]), [2], [_box(400)],
                     marker_labels={2: "Bob Lim"})
        for _ in range(2):
            m.update(_frame([]), [], [])
        for _ in range(3):
            o = m.update(_frame([_person(60, PURE_RED)]), [3], [_box(60)])
        c_uid = o[3].uid
        m.gate = gate
        return m, a_uid, c_uid
    m_off, a_off, c_off = run(False)
    m_on, a_on, c_on = run(True)
    off = m_off.consolidate()
    on = m_on.consolidate()
    results["consolidate: named records stay apart; a fragment joins only when unambiguous"] = (
        off == [] and len(m_off.workers) == 3
        and on == [(c_on, a_on)] and len(m_on.workers) == 2
        and m_on.workers[a_on].marker_confirmed and m_on.workers[a_on].label == "Alice Tan")


# ---- coasting labels ---------------------------------------------------------
def test_ghosts_reported_while_lost():
    """A worker just lost by the tracker keeps a predicted, clearly-marked box for a
    short while, then disappears -- never while they are actually visible."""
    m = _rule_mgr(coast_frames=3)
    o = m.update(_frame([_person(50, PURE_RED)]), [1], [BOX])
    visible = m.ghosts()
    m.update(_frame([]), [], [])
    g1 = m.ghosts()
    for _ in range(3):
        m.update(_frame([]), [], [])
    gone = m.ghosts()
    ok = (visible == [] and len(g1) == 1 and gone == []
          and g1[0]["uid"] == o[1].uid and g1[0]["label"] == o[1].label
          and g1[0]["age"] == 1 and [round(v) for v in g1[0]["box"]] == BOX)
    results["ghosts: a lost worker coasts as a predicted box, then expires"] = ok


def test_ghosts_follow_camera_pan():
    """The predicted box moves with the scene, so the coasting label stays on the spot
    where the person is, not where they were."""
    m = _rule_mgr(coast_frames=5)
    f = _frame([_person(50, PURE_RED), _person(300, PURE_BLUE)])
    m.update(f, [1, 2], [BOX, _box(300)])
    m.update(f, [1, 2], [BOX, _box(300)])
    for x in (340, 380):                                  # pan +40/frame, A lost
        m.update(_frame([_person(x, PURE_BLUE)]), [2], [_box(x)])
    g = [gh for gh in m.ghosts() if gh["label"].startswith("Worker")]
    results["ghosts: the predicted box drifts with the camera pan"] = (
        len(g) == 1 and abs(g[0]["box"][0] - 130) < 1.0 and g[0]["age"] == 2)


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
    test_probation_phantom_creates_no_worker()
    test_probation_recovers_with_pooled_descriptor()
    test_probation_marker_commits_immediately()
    test_probation_abandoned_track_is_forgotten()
    test_gate_vetoes_implausible_lookalike()
    test_gate_position_assists_below_threshold()
    test_gate_two_plausible_lookalikes_still_refuse()
    test_gate_compensates_camera_pan()
    test_gate_falls_back_to_appearance_beyond_horizon()
    test_gate_decisive_appearance_overrides_veto()
    test_gate_co_visible_worker_is_never_a_candidate()
    test_position_rule_rejects_a_stranger_in_the_spot()
    test_time_mode_uses_seconds_not_frames()
    test_badge_merges_anonymous_record_into_named()
    test_consolidate_uses_presence_overlap()
    test_consolidate_never_merges_co_visible_records()
    test_consolidate_respects_badges_and_ambiguity()
    test_ghosts_reported_while_lost()
    test_ghosts_follow_camera_pan()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_IDENTITY", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
