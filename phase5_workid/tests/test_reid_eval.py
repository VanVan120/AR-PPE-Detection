"""Unit tests for the worker re-ID evaluation harness (phase5_workid/reid_eval.py).

A measurement harness that flatters itself is worse than no harness, so most of these
tests attack the SCORING rather than the matcher. The headline guard is
`test_recall_is_zero_without_appearance`: it pins down a real bug this harness shipped
with, where each forced re-entry was compared against the identity assigned on the very
same frame, reporting 100% recall no matter how the matcher behaved.

    python phase5_workid/tests/test_reid_eval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase2"))

from phase5_workid.reid_eval import (SIMILAR, evaluate, make_sequence,  # noqa: E402
                                     run_scenario, _fresh)
from src.identity import IdentityManager                                # noqa: E402
from src.reid import ColorHistogramEmbedder                             # noqa: E402

results = {}


# ---- the generated sequence is what it claims to be --------------------------
def test_sequence_shape():
    seq = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0)
    results["harness: one forced re-entry per worker"] = (
        len(seq.frames) == 60 and len(seq.tracks) == 60 and len(seq.reentries) == 3)


def test_reentry_uses_a_new_track_id():
    """The gap must actually change the track id -- otherwise nothing is being tested,
    because the identity layer would trivially keep its existing binding."""
    seq = make_sequence(n_workers=2, frames=60, scenario="distinct", gap=8, seed=1)
    ok = True
    for f, w, new_tid in seq.reentries:
        before = {t.track_id for fr in seq.tracks[:f] for t in fr if t.worker == w}
        ok = ok and (new_tid not in before)
    results["harness: a re-entry really does get an unseen track id"] = ok


def test_worker_absent_during_gap():
    seq = make_sequence(n_workers=2, frames=60, scenario="distinct", gap=8, seed=2)
    f, w, _tid = seq.reentries[0]
    gap_frames = seq.tracks[f - 8:f]
    results["harness: the worker is genuinely absent during the gap"] = (
        all(all(t.worker != w for t in fr) for fr in gap_frames))


# ---- the scoring cannot flatter itself ---------------------------------------
def test_recall_is_zero_without_appearance():
    """THE regression guard. With appearance matching disabled every re-entry must open
    a brand-new worker, so re-ID recall is 0% by construction. A scoring bug that
    compares the re-entry against the identity assigned on the same frame reports 100%
    here -- which is exactly what this harness did before the fix."""
    seq = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0)
    mgr = IdentityManager(ColorHistogramEmbedder(), appearance_enabled=False,
                          forget_after=10 ** 9)
    res = evaluate(mgr, seq)
    results["harness: recall is 0% when appearance is off (no self-comparison)"] = (
        res["reentries"] == 3 and res["recovered"] == 0 and res["reid_recall"] == 0.0)


def test_fragmentation_without_appearance():
    """Same run: each worker should end up split into exactly 2 identities (before and
    after their gap), so fragmentation must be 2.0 -- not 1.0."""
    seq = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0)
    mgr = IdentityManager(ColorHistogramEmbedder(), appearance_enabled=False,
                          forget_after=10 ** 9)
    res = evaluate(mgr, seq)
    results["harness: fragmentation is 2.00 when every re-entry is a new worker"] = (
        abs(res["fragmentation"] - 2.0) < 1e-6)


def test_no_false_merges_when_appearance_off():
    """Every identity is fresh, so none can belong to another worker."""
    seq = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0)
    mgr = IdentityManager(ColorHistogramEmbedder(), appearance_enabled=False,
                          forget_after=10 ** 9)
    res = evaluate(mgr, seq)
    results["harness: false merges are 0 when identities are never reused"] = (
        res["false_merges"] == 0)


# ---- the matcher does what the README claims ---------------------------------
def test_distinct_recovers():
    res = run_scenario("distinct", "histogram", 0.62, workers=4, frames=90, gap=12, seed=0)
    results["re-ID: distinct clothing is fully recovered with no false merge"] = (
        res["reid_recall"] == 100.0 and res["false_merge_rate"] == 0.0)


def test_uniform_is_harder_than_distinct():
    """The honest headline: identical PPE defeats appearance re-ID. If this ever stops
    holding, the synthetic 'uniform' scenario has stopped being realistic."""
    d = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 0)
    u = run_scenario("uniform", "histogram", 0.62, 4, 90, 12, 0)
    results["re-ID: identical PPE scores far worse than distinct clothing"] = (
        u["reid_recall"] < 50.0 < d["reid_recall"]
        and u["fragmentation"] > d["fragmentation"])


def test_similar_outfits_share_a_vest():
    """The 'similar' scenario must actually hold the shirt constant -- that is the whole
    point of it (same issued vest, personal helmet)."""
    shirts = {o.shirt for o in SIMILAR}
    helmets = {o.helmet for o in SIMILAR}
    results["harness: 'similar' shares one vest but varies the helmet"] = (
        len(shirts) == 1 and len(helmets) == len(SIMILAR))


def test_threshold_guards_false_merges():
    """The configured default must not be beaten on safety by a looser threshold: a low
    threshold buys recall by merging different people, which is the error that moves a
    violation onto the wrong worker."""
    loose = run_scenario("similar", "histogram", 0.40, 4, 90, 12, 0)
    default = run_scenario("similar", "histogram", 0.62, 4, 90, 12, 0)
    results["re-ID: the default threshold merges fewer people than a loose one"] = (
        default["false_merge_rate"] <= loose["false_merge_rate"])


def main() -> int:
    test_sequence_shape()
    test_reentry_uses_a_new_track_id()
    test_worker_absent_during_gap()
    test_recall_is_zero_without_appearance()
    test_fragmentation_without_appearance()
    test_no_false_merges_when_appearance_off()
    test_distinct_recovers()
    test_uniform_is_harder_than_distinct()
    test_similar_outfits_share_a_vest()
    test_threshold_guards_false_merges()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_REID_EVAL", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
