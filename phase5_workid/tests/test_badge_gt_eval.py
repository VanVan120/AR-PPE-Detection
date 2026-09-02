"""Unit tests for the badge-as-ground-truth scorer (phase5_workid/badge_gt_eval.py).

The CLI needs the detector weights and a real clip; the SCORER does not, and it is where
a harness can flatter itself. These tests feed hand-written streams whose right answer is
known and check the metrics come out as they must.

    python phase5_workid/tests/test_badge_gt_eval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase5_workid.badge_gt_eval import score_stream            # noqa: E402
from phase5_workid.reid_eval import _remap_from                 # noqa: E402

results = {}


def test_perfect_stream():
    """One badge name, two tracks, same uid throughout -> 100% recall, one identity."""
    frames = [[(1, "Alice", "w1")], [(1, "Alice", "w1")], [], [(5, "Alice", "w1")]]
    r = score_stream(frames)
    results["badge-gt: a re-entry that kept its uid scores 100%"] = (
        r["reentries"] == 1 and r["recovered"] == 1 and r["fragmentation"] == 1.0
        and r["false_merge_rate"] == 0.0 and r["workers_true"] == 1)


def test_fragmented_stream():
    """The uid changed when the track changed -> the re-entry was missed and the record
    is split in two."""
    frames = [[(1, "Alice", "w1")], [], [(5, "Alice", "w2")], [(5, "Alice", "w2")]]
    r = score_stream(frames)
    results["badge-gt: a uid change under the same badge is a miss and a fragment"] = (
        r["reentries"] == 1 and r["recovered"] == 0 and r["fragmentation"] == 2.0
        and r["workers_created"] == 2)


def test_delayed_uid_scored_at_first_uid():
    """Probation: the new track has no uid for two frames, then the right one."""
    frames = [[(1, "Alice", "w1")], [(5, "Alice", None)], [(5, "Alice", None)],
              [(5, "Alice", "w1")]]
    r = score_stream(frames)
    results["badge-gt: a re-entry is scored when its uid arrives, not skipped"] = (
        r["reentries"] == 1 and r["recovered"] == 1 and r["assignments"] == 2)


def test_never_identified_track_is_a_miss():
    frames = [[(1, "Alice", "w1")], [(5, "Alice", None)], [(5, "Alice", None)]]
    r = score_stream(frames)
    results["badge-gt: a returning track never given a uid counts as a miss"] = (
        r["reentries"] == 1 and r["recovered"] == 0)


def test_false_merge_across_names():
    """Two badge names wearing the same uid: the second one's frames are false merges."""
    frames = [[(1, "Alice", "w1"), (2, "Bob", "w2")],
              [(1, "Alice", "w1"), (2, "Bob", "w1")],
              [(1, "Alice", "w1"), (2, "Bob", "w1")]]
    r = score_stream(frames)
    results["badge-gt: a uid worn by two badge names is counted as a false merge"] = (
        r["false_merges"] == 2 and abs(r["false_merge_rate"] - 100.0 * 2 / 6) < 1e-6)


def test_unlabelled_rows_are_ignored():
    """Rows with no badge read carry no ground truth and must not be scored at all."""
    frames = [[(1, None, "w1"), (2, "Bob", "w2")], [(1, None, "w3"), (2, "Bob", "w2")]]
    r = score_stream(frames)
    results["badge-gt: frames without a badge read are not scored"] = (
        r["assignments"] == 2 and r["workers_true"] == 1 and r["reentries"] == 0)


def test_offline_remap_repairs_a_fragment():
    """The consolidation merge map is applied to the same log: a fragment merged back
    into the right record turns a miss into a recovery."""
    frames = [[(1, "Alice", "w1")], [], [(5, "Alice", "w2")], [(5, "Alice", "w2")]]
    r = score_stream(frames, remap=_remap_from([("w2", "w1")]))
    results["badge-gt: the offline merge map is scored on the same events"] = (
        r["recovered"] == 1 and r["fragmentation"] == 1.0 and r["workers_created"] == 1)


def main() -> int:
    test_perfect_stream()
    test_fragmented_stream()
    test_delayed_uid_scored_at_first_uid()
    test_never_identified_track_is_a_miss()
    test_false_merge_across_names()
    test_unlabelled_rows_are_ignored()
    test_offline_remap_repairs_a_fragment()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_BADGE_GT", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
