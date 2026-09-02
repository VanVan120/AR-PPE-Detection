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


# ---- the published baseline is pinned ----------------------------------------
def test_baseline_rows_are_pinned():
    """The numbers in the READMEs (3 seeds, 4 workers, gap 12, threshold 0.62). Any
    change to the identity layer's DEFAULT behaviour, or to the sequence generator,
    shows up here first."""
    from phase5_workid.reid_eval import aggregate
    got = {}
    for scen in ("distinct", "similar", "uniform"):
        runs = [run_scenario(scen, "histogram", 0.62, 4, 90, 12, s) for s in range(3)]
        a = aggregate(runs)
        got[scen] = (a["recovered"], a["reentries"], round(a["false_merge_rate"], 1),
                     round(a["fragmentation"], 2))
    results["baseline: the published re-ID table reproduces exactly"] = (
        got["distinct"] == (12, 12, 0.0, 1.0)
        and got["similar"] == (9, 12, 0.0, 1.25)
        and got["uniform"] == (1, 12, 9.6, 1.92))


# ---- delayed scoring for probation -------------------------------------------
def test_probation_reentry_scored_at_commit():
    """With probation the returning track has no identity for two frames. It must be
    scored when the identity ARRIVES, not skipped -- otherwise probation would hide
    every re-entry from the recall metric and flatter itself."""
    res = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 0, probation_frames=3)
    results["harness: re-entries under probation are scored at commit, not skipped"] = (
        res["reentries"] == 4 and res["reid_recall"] == 100.0)


def test_never_committed_reentry_counts_as_missed():
    """A probation so long that nothing is ever committed must read as 0% recall, with
    every re-entry still counted."""
    res = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 0, probation_frames=10 ** 6)
    results["harness: a re-entry never identified is a miss, not an omission"] = (
        res["reentries"] == 4 and res["recovered"] == 0 and res["workers_created"] == 0)


# ---- phantoms and relocation ------------------------------------------------
def test_phantoms_are_short_and_do_not_change_the_workers():
    base = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0)
    ph = make_sequence(n_workers=3, frames=60, scenario="distinct", gap=8, seed=0, phantoms=5)
    same_workers = all(
        [(t.worker, t.track_id) for t in fr_b] == [(t.worker, t.track_id) for t in fr_p
                                                    if t.worker >= 0]
        for fr_b, fr_p in zip(base.tracks, ph.tracks))
    ids = {}
    for fr in ph.tracks:
        for t in fr:
            if t.worker < 0:
                ids[t.track_id] = ids.get(t.track_id, 0) + 1
    results["harness: phantoms are 1-2 frame tracks that leave the real workers untouched"] = (
        same_workers and len(ids) == 5 and all(1 <= n <= 2 for n in ids.values()))


def test_probation_absorbs_phantoms():
    """The mechanism's whole purpose: spurious one-frame tracks must not become workers."""
    base = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 0, phantoms=6)
    enh = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 0, phantoms=6,
                       probation_frames=3)
    results["harness: probation stops phantom tracks from inflating the worker count"] = (
        base["workers_created"] > 4 and enh["workers_created"] == 4
        and enh["reid_recall"] == 100.0)


def test_relocate_moves_the_reentry():
    base = make_sequence(n_workers=2, frames=60, scenario="distinct", gap=8, seed=3)
    rel = make_sequence(n_workers=2, frames=60, scenario="distinct", gap=8, seed=3,
                        relocate=True)
    moved = 0
    for (f, w, tid) in rel.reentries:
        xb = [t.box[0] for t in base.tracks[f] if t.worker == w][0]
        xr = [t.box[0] for t in rel.tracks[f] if t.worker == w][0]
        moved += abs(xb - xr) > 1.0
    results["harness: --relocate makes returning workers re-enter elsewhere"] = (
        moved == len(rel.reentries) and len(rel.reentries) == 2)


# ---- offline consolidation is scored honestly --------------------------------
def test_offline_can_only_reduce_fragmentation():
    """Merging records can never increase the number of identities per worker, and on
    distinct clothing it must not invent a false merge."""
    on = run_scenario("similar", "histogram", 0.62, 4, 90, 12, 1, probation_frames=3, gate=True)
    off = run_scenario("similar", "histogram", 0.62, 4, 90, 12, 1, probation_frames=3,
                       gate=True, consolidate=True)
    d = run_scenario("distinct", "histogram", 0.62, 4, 90, 12, 1, probation_frames=3,
                     gate=True, consolidate=True)
    results["harness: the offline pass never raises fragmentation, never merges distinct people"] = (
        off["fragmentation"] <= on["fragmentation"]
        and off["workers_created"] <= on["workers_created"]
        and d["false_merge_rate"] == 0.0 and d["fragmentation"] == 1.0)


def test_pipeline_eval_accepts_enhanced_settings():
    from phase5_workid.reid_eval import pipeline_eval
    seq = make_sequence(n_workers=3, frames=60, scenario="similar", gap=8, seed=0, motion=8.0)
    base = pipeline_eval(seq, "histogram", 0.62)
    enh = pipeline_eval(seq, "histogram", 0.62, probation_frames=3, gate=True)
    keys = ("bytetrack_ids_per_worker", "identity_uids_per_worker", "false_merge_rate")
    results["harness: the head-motion pipeline runs with the enhanced settings"] = (
        all(k in base and k in enh for k in keys)
        and base["bytetrack_ids_per_worker"] == enh["bytetrack_ids_per_worker"])


def main() -> int:
    test_baseline_rows_are_pinned()
    test_probation_reentry_scored_at_commit()
    test_never_committed_reentry_counts_as_missed()
    test_phantoms_are_short_and_do_not_change_the_workers()
    test_probation_absorbs_phantoms()
    test_relocate_moves_the_reentry()
    test_offline_can_only_reduce_fragmentation()
    test_pipeline_eval_accepts_enhanced_settings()
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
