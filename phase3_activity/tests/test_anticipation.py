"""Unit tests for next-step anticipation (tas/anticipation.py).

Pure logic — no features/torch/lmdb. Builds tiny synthetic workflows with known
sequential structure and checks that the model learns it, predicts sensible top-k next
steps, beats the marginal (frequency-only) baseline, excludes already-done steps, and
round-trips through JSON.

    python phase3_activity/tests/test_anticipation.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tas.anticipation import AnticipationModel, evaluate

results = {}
NAMES = {0: "base", 1: "mid", 2: "top", 3: "done", 4: "left", 5: "right"}


def _corpus():
    """Same structure as the mistake tests: 0,1,2 in order, then {4,5} either order,
    then 3. So the next step after 0 is almost always 1, after 1 almost always 2, etc."""
    seqs = []
    for i in range(20):
        mid = [4, 5] if i % 2 == 0 else [5, 4]
        seqs.append([0, 1, 2] + mid + [3])
    return seqs


def _fit():
    return AnticipationModel.fit(_corpus(), alpha=0.1, id_to_action=NAMES)


# ---- learns the dominant transition -----------------------------------------
def test_predict_top1():
    m = _fit()
    # after [0] the next step is essentially always 1
    top = m.predict_next([0], k=3)
    results["predict: top-1 after '0' is '1'"] = (top and top[0][0] == 1)


# ---- excludes completed steps -----------------------------------------------
def test_excludes_done():
    m = _fit()
    top_ids = [n for n, _p in m.predict_next([0, 1, 2], k=6)]
    results["predict: excludes already-done steps"] = (
        0 not in top_ids and 1 not in top_ids and 2 not in top_ids)


# ---- cold start uses the start distribution ---------------------------------
def test_cold_start():
    m = _fit()
    top = m.predict_next([], k=3)          # no history -> should predict the usual first step (0)
    results["predict: cold-start predicts first step"] = (top and top[0][0] == 0)


# ---- top-k probabilities are a ranked, normalized-ish list ------------------
def test_topk_ranked():
    m = _fit()
    top = m.predict_next([2], k=3)
    probs = [p for _n, p in top]
    results["predict: top-k is ranked descending"] = (
        len(top) == 3 and all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)))


# ---- the model beats the marginal baseline on held-out-style data -----------
def test_beats_marginal():
    m = _fit()
    val = [("v", s) for s in _corpus()[:6]]     # same distribution -> transitions should win
    r = evaluate(m, val, ks=(1, 3))
    acc = r["accuracy"]
    beats = acc["full"][1] > acc["marginal"][1] and acc["full"][3] >= acc["marginal"][3]
    results["evaluate: full model beats marginal top-1"] = beats


# ---- feasibility masking is opt-in and doesn't crash ------------------------
def test_feasibility_optional():
    from tas.procedure import ProcedureModel
    proc = ProcedureModel.fit(_corpus(), min_support=5, precedence_tau=0.99, id_to_action=NAMES)
    m = AnticipationModel.fit(_corpus(), alpha=0.1, procedure=proc, id_to_action=NAMES)
    off = m.predict_next([0], k=3, use_feasibility=False)
    on = m.predict_next([0], k=3, use_feasibility=True)
    results["predict: feasibility flag runs both ways"] = (len(off) > 0 and len(on) > 0)


# ---- serialization round-trip preserves predictions -------------------------
def test_serialization_roundtrip():
    m = _fit()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ant.json")
        m.save(p)
        m2 = AnticipationModel.load(p)
    same = (m.predict_next([0, 1], k=3) == m2.predict_next([0, 1], k=3))
    results["model: save/load round-trip preserves predictions"] = same


def main() -> int:
    test_predict_top1()
    test_excludes_done()
    test_cold_start()
    test_topk_ranked()
    test_beats_marginal()
    test_feasibility_optional()
    test_serialization_roundtrip()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_ANTICIPATION", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
