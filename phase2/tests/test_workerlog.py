"""Unit tests for the per-worker violation history (src/workerlog.py).

Pure logic over synthetic compliance frames -- no camera, no model, no data.

The cases that matter: a violation must be recorded as a timed episode rather than a
counter, a worker's record must survive being renamed by a badge, and an episode still
running when the session ends must not report a zero duration.

    python phase2/tests/test_workerlog.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compliance import ActiveViolation, FrameCompliance, PersonStatus
from src.identity import IdentityResult
from src.workerlog import WorkerHistory

results = {}

NO_HELMET = ("No-Helmet", "high", "No hard hat")
NO_VEST = ("No-Vest", "medium", "No hi-vis vest")


def _frame(persons):
    """persons = [(track_id, [violation_tuple, ...]), ...]"""
    fc = FrameCompliance()
    for tid, viols in persons:
        st = PersonStatus(tracker_id=tid, bbox=(0.0, 0.0, 10.0, 20.0))
        for cls, sev, label in viols:
            st.active.append(ActiveViolation(tid, cls, sev, label))
        fc.persons.append(st)
    return fc


def _ident(tid_to):
    """tid -> IdentityResult"""
    return {tid: IdentityResult(label, uid, source)
            for tid, (uid, label, source) in tid_to.items()}


# ---- episodes carry real durations -------------------------------------------
def test_episode_duration():
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Worker 1", "new")})
    h.update(0, 0.0, _frame([(1, [])]), ids)
    for i, t in enumerate([1.0, 2.0, 3.0], start=1):
        h.update(i, t, _frame([(1, [NO_HELMET])]), ids)
    h.update(4, 4.0, _frame([(1, [])]), ids)             # cleared
    rec = h.records["w1"]
    ep = rec.episodes[0]
    results["workerlog: a violation becomes a timed episode"] = (
        len(rec.episodes) == 1 and ep.closed and abs(ep.duration_s - 3.0) < 1e-6)


def test_open_episode_closed_at_session_end():
    """Still violating when the session stops: the duration must be real and flagged,
    not silently 0."""
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Worker 1", "new")})
    h.update(0, 0.0, _frame([(1, [NO_VEST])]), ids)
    h.update(1, 5.0, _frame([(1, [NO_VEST])]), ids)
    h.close(elapsed_s=5.0, frame_no=1)
    ep = h.records["w1"].episodes[0]
    results["workerlog: open episode is closed and marked at session end"] = (
        ep.closed and ep.truncated and abs(ep.duration_s - 5.0) < 1e-6)


def test_separate_episodes_not_merged():
    """Violate, clear, violate again -> two episodes, not one long one."""
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Worker 1", "new")})
    h.update(0, 0.0, _frame([(1, [NO_HELMET])]), ids)
    h.update(1, 1.0, _frame([(1, [])]), ids)
    h.update(2, 2.0, _frame([(1, [NO_HELMET])]), ids)
    h.update(3, 3.0, _frame([(1, [])]), ids)
    results["workerlog: violate/clear/violate records two episodes"] = (
        len(h.records["w1"].episodes) == 2)


def test_two_violation_types_tracked_separately():
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Worker 1", "new")})
    h.update(0, 0.0, _frame([(1, [NO_HELMET, NO_VEST])]), ids)
    h.update(1, 2.0, _frame([(1, [])]), ids)
    rec = h.records["w1"]
    results["workerlog: concurrent violation types are separate episodes"] = (
        len(rec.episodes) == 2 and rec.by_type() == {"No-Helmet": 1, "No-Vest": 1})


# ---- identity churn ----------------------------------------------------------
def test_record_survives_rename():
    """The badge promotion case: history recorded while anonymous must carry over to the
    real name, in ONE record -- this is why the log keys on uid, not on the label."""
    h = WorkerHistory()
    anon = _ident({1: ("w1", "Worker 1", "new")})
    named = _ident({1: ("w1", "Alice Tan", "marker")})
    h.update(0, 0.0, _frame([(1, [NO_HELMET])]), anon)
    h.update(1, 1.0, _frame([(1, [NO_HELMET])]), named)
    h.update(2, 2.0, _frame([(1, [])]), named)
    rec = h.records["w1"]
    results["workerlog: one record survives a rename, keeping earlier history"] = (
        len(h.records) == 1 and rec.label == "Alice Tan"
        and rec.marker_confirmed and len(rec.episodes) == 1
        and abs(rec.episodes[0].duration_s - 2.0) < 1e-6)


def test_new_track_id_same_worker_is_one_record():
    """Identity already resolved the track churn; the log must not re-split it."""
    h = WorkerHistory()
    h.update(0, 0.0, _frame([(1, [])]), _ident({1: ("w1", "Alice Tan", "marker")}))
    h.update(1, 1.0, _frame([(8, [])]), _ident({8: ("w1", "Alice Tan", "marker")}))
    results["workerlog: same worker under two track ids stays one record"] = (
        len(h.records) == 1 and h.records["w1"].frames_seen == 2)


def test_unidentified_person_skipped():
    h = WorkerHistory()
    h.update(0, 0.0, _frame([(1, [NO_HELMET])]), {})
    results["workerlog: an unidentified person adds no record"] = (len(h.records) == 0)


# ---- reporting ---------------------------------------------------------------
def test_compliance_pct():
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Worker 1", "new")})
    for i in range(3):
        h.update(i, float(i), _frame([(1, [])]), ids)
    h.update(3, 3.0, _frame([(1, [NO_HELMET])]), ids)     # 1 of 4 frames unsafe
    results["workerlog: compliance percentage is frames-based"] = (
        abs(h.records["w1"].compliance_pct - 75.0) < 1e-6)


def test_report_sorted_worst_first():
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Mild", "new"), 2: ("w2", "Severe", "new")})
    h.update(0, 0.0, _frame([(1, [NO_HELMET]), (2, [NO_HELMET])]), ids)
    h.update(1, 1.0, _frame([(1, []), (2, [NO_HELMET])]), ids)
    h.update(2, 9.0, _frame([(1, []), (2, [])]), ids)
    rep = h.report()
    results["workerlog: report lists the worst offender first"] = (
        rep["per_worker"][0]["worker"] == "Severe"
        and rep["workers_seen"] == 2 and rep["total_violation_episodes"] == 2)


def test_clean_session_report():
    h = WorkerHistory()
    h.update(0, 0.0, _frame([(1, [])]), _ident({1: ("w1", "Alice Tan", "marker")}))
    rep = h.report()
    text = h.format_report()
    results["workerlog: a clean session reports 100% and no episodes"] = (
        rep["total_violation_episodes"] == 0
        and rep["per_worker"][0]["compliance_pct"] == 100.0
        and "Alice Tan" in text)


def test_save_json_roundtrip():
    """The machine-readable report must be valid JSON with the durations intact."""
    import json
    import tempfile
    h = WorkerHistory()
    ids = _ident({1: ("w1", "Alice Tan", "marker")})
    h.update(0, 0.0, _frame([(1, [NO_HELMET])]), ids)
    h.update(1, 4.0, _frame([(1, [])]), ids)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub", "workers.json")
        h.save_json(p)
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    w = data["per_worker"][0]
    results["workerlog: save_json writes a valid report with durations"] = (
        data["workers_seen"] == 1 and w["worker"] == "Alice Tan"
        and w["identified_by_badge"] is True and w["episodes"][0]["duration_s"] == 4.0)


def test_empty_report_does_not_crash():
    h = WorkerHistory()
    results["workerlog: empty session formats without crashing"] = (
        isinstance(h.format_report(), str) and h.report()["workers_seen"] == 0)


def main() -> int:
    test_episode_duration()
    test_open_episode_closed_at_session_end()
    test_separate_episodes_not_merged()
    test_two_violation_types_tracked_separately()
    test_record_survives_rename()
    test_new_track_id_same_worker_is_one_record()
    test_unidentified_person_skipped()
    test_compliance_pct()
    test_report_sorted_worst_first()
    test_clean_session_report()
    test_save_json_roundtrip()
    test_empty_report_does_not_crash()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_WORKERLOG", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
