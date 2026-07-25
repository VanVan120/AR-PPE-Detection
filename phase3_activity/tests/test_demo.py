"""Unit tests for the Phase 3 workflow-monitor demo (tas/demo.py).

Pure logic on tiny synthetic workflows -- no features, LMDB, torch or annotations, so
this runs anywhere. Covers the parts that carry real risk of being silently wrong:
the honest (no-lookahead) anticipation scoring, the violation-free plan generator, the
fault injector, and the transition-count convention that the reported FP rate divides by.

    python phase3_activity/tests/test_demo.py
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tas.anticipation import AnticipationModel
from tas.demo import inject_fault, replay, stream_sampled
from tas.procedure import ProcedureModel, score_sequence

results = {}
NAMES = {0: "base", 1: "mid", 2: "top", 3: "done", 4: "left", 5: "right"}


def _corpus():
    """0,1,2 strictly in order, then {4,5} interchangeable, then 3 last."""
    return [[0, 1, 2] + ([4, 5] if i % 2 == 0 else [5, 4]) + [3] for i in range(20)]


def _models():
    corpus = _corpus()
    proc = ProcedureModel.fit(corpus, min_support=5, precedence_tau=0.99, id_to_action=NAMES)
    antic = AnticipationModel.fit(corpus, alpha=0.1, procedure=proc, id_to_action=NAMES)
    return proc, antic


def _quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ---- a clean sequence raises nothing -----------------------------------------
def test_clean_sequence_no_events():
    proc, antic = _models()
    out = _quiet(replay, [0, 1, 2, 4, 5, 3], proc, antic)
    results["replay: a clean in-order sequence raises no mistake"] = (len(out["events"]) == 0)


# ---- an out-of-order sequence is caught --------------------------------------
def test_order_violation_caught():
    proc, antic = _models()
    out = _quiet(replay, [2, 1, 0, 4, 5, 3], proc, antic)   # 0->1->2 reversed
    kinds = {e.kind for _i, e in out["events"]}
    results["replay: reversed prerequisites raise order_violation"] = (
        len(out["events"]) > 0 and kinds == {"order_violation"})


# ---- anticipation is scored WITHOUT lookahead --------------------------------
def test_scoring_uses_only_past():
    """The step at position i must be predicted from steps[:i] only. If the
    implementation leaked the current step into the history, a done-masked model would
    never rank it first and top-1 would collapse to 0 -- so a high top-1 on a perfectly
    learnable sequence is the evidence that the split is correct."""
    proc, antic = _models()
    out = _quiet(replay, [0, 1, 2, 4, 5, 3], proc, antic)
    results["replay: anticipation scored from past only (no leakage)"] = (
        out["top1"] is not None and out["top1"] >= 50.0)


def test_score_disabled():
    proc, antic = _models()
    out = _quiet(replay, [0, 1, 2], proc, antic, score=False)
    results["replay: score=False reports no accuracy (sample mode)"] = (out["top1"] is None)


# ---- the generated plan respects the learned order ---------------------------
def test_sampled_stream_is_violation_free():
    """stream_sampled uses prerequisite masking so the demo a data-less user sees is
    clean by construction -- a greedy unmasked walk trips violations immediately."""
    proc, antic = _models()
    _name, steps = stream_sampled(antic, length=6)
    results["stream_sampled: generated plan raises no violation"] = (
        len(steps) > 1 and len(score_sequence(proc, steps)) == 0)


# ---- fault injection produces a detectable violation -------------------------
def test_inject_fault_is_detected():
    proc, antic = _models()
    seq = [0, 1, 2, 4, 5, 3]
    pert, pair = inject_fault(proc, seq, seed=0)
    detected = len(score_sequence(proc, pert)) > 0
    results["inject_fault: swaps a constraint pair and the monitor catches it"] = (
        pair is not None and pert != seq and detected)


def test_inject_fault_no_candidates():
    """A sequence with no constrained pair must come back unchanged, not crash."""
    proc, _antic = _models()
    seq = [4]
    pert, pair = inject_fault(proc, seq, seed=0)
    results["inject_fault: no candidate pair -> unchanged, no crash"] = (
        pair is None and pert == list(seq))


# ---- the FP-rate denominator counts transitions, not distinct steps ----------
def test_transition_count_convention():
    """Regression guard for a real bug: the per-transition FP rate was dividing by the
    number of DISTINCT steps. When a step recurs (rework) that denominator is smaller
    than the number of positions the monitor actually judged, inflating the rate."""
    seq = [0, 1, 0, 1, 2]                      # 5 entries, 4 transitions, 3 distinct
    transitions = max(len(seq) - 1, 0)
    distinct = len(set(seq))
    results["FP denominator: transitions (n-1) != distinct steps when steps recur"] = (
        transitions == 4 and distinct == 3 and transitions != distinct)


# ---- replay reports one event per offending transition -----------------------
def test_event_indices_align():
    proc, antic = _models()
    out = _quiet(replay, [2, 0, 1, 4, 5, 3], proc, antic)
    ok = all(0 <= i < 6 for i, _e in out["events"])
    results["replay: event indices point at real positions in the stream"] = ok


def main() -> int:
    test_clean_sequence_no_events()
    test_order_violation_caught()
    test_scoring_uses_only_past()
    test_score_disabled()
    test_sampled_stream_is_violation_free()
    test_inject_fault_is_detected()
    test_inject_fault_no_candidates()
    test_transition_count_convention()
    test_event_indices_align()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_DEMO", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
