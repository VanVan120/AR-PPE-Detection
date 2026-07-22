"""Unit tests for the assembly-order mistake detector (tas/procedure.py).

Pure logic — no features, torch or lmdb needed. Builds tiny synthetic workflows with a
known order structure and checks that the model learns the right precedence
constraints, the monitor flags genuine out-of-order steps, stays quiet on correct
order, and — the key property — does NOT flag a step that is simply absent (part of a
different assembly) as a missing prerequisite.

    python phase3_activity/tests/test_mistake.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tas.procedure import (MistakeMonitor, ProcedureModel, reduce_segments_to_step_ids,
                           score_sequence)

results = {}

# steps: 0 base -> 1 mid -> 2 top -> 3 done (strict order); 4 left / 5 right free order.
NAMES = {0: "base", 1: "mid", 2: "top", 3: "done", 4: "left", 5: "right"}


def _corpus():
    """10 videos: 0,1,2 always in order, then {4,5} in EITHER order, then 3 (done last).
    So 0<1<2<{4,5}<3 are deterministic; 4 vs 5 is 50/50 (never a constraint)."""
    seqs = []
    for i in range(10):
        mid = [4, 5] if i % 2 == 0 else [5, 4]        # alternate -> interchangeable
        seqs.append([0, 1, 2] + mid + [3])
    return seqs


def _fit():
    return ProcedureModel.fit(_corpus(), min_support=5, precedence_tau=0.99, id_to_action=NAMES)


# ---- fit learns the right constraints ---------------------------------------
def test_fit_learns_precedence():
    m = _fit()
    cs = {(a, b) for (a, b) in m.constraint_stats}
    has_order = all(pair in cs for pair in [(0, 1), (1, 2), (2, 3), (0, 3), (4, 3), (2, 4)])
    # the interchangeable pair must NOT become a constraint (either direction)
    no_free = (4, 5) not in cs and (5, 4) not in cs
    results["fit: learns deterministic order, ignores interchangeable pair"] = has_order and no_free


# ---- monitor flags a genuine out-of-order step ------------------------------
def test_flags_out_of_order():
    m = _fit()
    # 2 done before 1, but 1 must precede 2 -> violation detected when 1 is entered
    events = score_sequence(m, [0, 2, 1, 3])
    ok = any(ev.kind == "order_violation" and ev.step == 1 and 2 in ev.violated
             for _i, ev in events)
    results["monitor: flags out-of-order step (order_violation)"] = ok


# ---- monitor is quiet on a correct sequence ---------------------------------
def test_quiet_on_correct():
    m = _fit()
    a = score_sequence(m, [0, 1, 2, 4, 5, 3])
    b = score_sequence(m, [0, 1, 2, 5, 4, 3])       # the other valid ordering of 4/5
    results["monitor: no flags on either correct ordering"] = (a == [] and b == [])


# ---- the key fix: an ABSENT step is not a false 'missing prerequisite' -------
def test_absent_step_no_false_positive():
    m = _fit()
    # 2 and its predecessors 0/1 are absent; 4,5,3 present and in correct order.
    # The old prerequisite rule would flag "3 before required 0/1/2"; the successor
    # rule must not, because none of 3's predecessors were actually performed.
    events = score_sequence(m, [4, 5, 3])
    results["monitor: absent step != false positive"] = (events == [])


# ---- unknown step (not in training vocab) -----------------------------------
def test_unknown_step():
    m = _fit()
    events = score_sequence(m, [0, 99])             # 99 never seen
    ok = any(ev.kind == "unknown_step" and ev.step == 99 for _i, ev in events)
    results["monitor: flags unknown step"] = ok


# ---- novel-transition signal is opt-in --------------------------------------
def test_novel_transition_opt_in():
    m = _fit()
    # 0 -> 3 is a transition never seen in training (always 2/4/5 between them), but
    # 0->3 has no precedence violation. Off by default -> quiet; on -> flagged.
    off = score_sequence(m, [0, 3], use_novel_transition=False)
    on = score_sequence(m, [0, 3], use_novel_transition=True)
    ok = (off == [] and any(ev.kind == "novel_transition" for _i, ev in on))
    results["monitor: novel_transition is opt-in"] = ok


# ---- serialization round-trip preserves behaviour ---------------------------
def test_serialization_roundtrip():
    m = _fit()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "proc.json")
        m.save(p)
        m2 = ProcedureModel.load(p)
    same_cons = set(m.constraint_stats) == set(m2.constraint_stats)
    same_behaviour = ([ev.kind for _i, ev in score_sequence(m, [0, 2, 1, 3])]
                      == [ev.kind for _i, ev in score_sequence(m2, [0, 2, 1, 3])])
    results["model: save/load round-trip preserves constraints + behaviour"] = (
        same_cons and same_behaviour)


# ---- de-dup: repeated same-step reports don't re-fire -----------------------
def test_repeat_step_dedup():
    m = _fit()
    mon = MistakeMonitor(m)
    # entering 2 then 2 then 2 (same step held) -> at most one decision, no crash
    seq = [0, 1, 2, 2, 2, 4, 5, 3]
    fired = [mon.observe(s) for s in seq]
    n_events = sum(1 for e in fired if e is not None)
    results["monitor: repeated same-step reports are de-duplicated"] = (n_events == 0)


# ---- segment reduction collapses consecutive duplicates ----------------------
def test_reduce_segments():
    segs = [(0, 5, "base"), (5, 9, "base"), (9, 12, "mid"), (12, 15, "zzz_unknown")]
    steps = reduce_segments_to_step_ids(segs, {"base": 0, "mid": 1})
    results["reduce: collapse consecutive dups + skip unknown"] = (steps == [0, 1])


def main() -> int:
    test_fit_learns_precedence()
    test_flags_out_of_order()
    test_quiet_on_correct()
    test_absent_step_no_false_positive()
    test_unknown_step()
    test_novel_transition_opt_in()
    test_serialization_roundtrip()
    test_repeat_step_dedup()
    test_reduce_segments()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_MISTAKE", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
