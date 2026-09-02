"""Real-footage evaluation of worker re-identification — with the badge as ground truth.

Every number in `reid_eval.py` comes from rendered figures, and the READMEs say so. This
is the path to a number from a real clip WITHOUT hand-labelling anybody: the workers wear
printed ArUco badges (`phase2/tools/make_worker_tags.py`), and **the badge reads are the
ground truth**. The appearance-only identity layer never sees them; it is scored on how
often it keeps the same worker uid on the same badge name across track breaks.

Protocol, one detector pass:

  1. detect + track as the app does (`SafetyPipeline`'s detector and tracker);
  2. the Work-ID binder reads badges and labels each track it can (`badge_of`);
  3. two identity managers — `baseline` (as first published) and `enhanced`
     (config.yaml) — are fed the same tracks with NO marker labels;
  4. for every frame, every track that carries a badge name AND received a uid is one
     scored assignment: (name, uid). The three metrics of `reid_eval.py` follow from that
     log, re-scored through the consolidation merge map for the `+offline` row.

What this does and does not measure. A track without a badge read is invisible to the
score (its identity is unknown), so the numbers describe the badge-labelled fraction of
the clip, and they are sparse when badges are small or often hidden. A badge mis-bound to
a neighbour would count as a wrong ground truth; the binder's containment rule and
stickiness make that rare, and the annotated clip lets a reviewer check. Fragmentation of
appearance identity between two badge reads of the same person IS counted (the uid changed
under the same name), which is exactly the failure the enhancements target.

    python -m phase5_workid.badge_gt_eval site_visit.mp4
    python -m phase5_workid.badge_gt_eval site_visit.mp4 --every 2 --out outputs/badge_gt.json

Needs the detector weights (`phase2/models/best.pt`), a clip in which people wear the
printed badges, and `opencv-contrib-python` (for `cv2.aruco`).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from phase5_workid.reid_eval import BASELINE, ENHANCED, RULE, _remap_from, _score  # noqa: E402

Row = Tuple[int, Optional[str], Optional[str]]     # (track_id, badge name, uid)


def score_stream(frames: Sequence[Sequence[Row]],
                 remap: Optional[Callable[[str], str]] = None,
                 workers_created: Optional[int] = None) -> dict:
    """Score a stream of per-frame rows `(track_id, badge_name or None, uid or None)`.

    A badge name's *re-entry* is the moment it appears under a track id it has not been
    seen under before, while it had a uid under an earlier track. It is scored at the
    first frame that new track has a uid; a track that ends without one is a miss. A row
    without a name carries no ground truth and is skipped; a row without a uid (the track
    is in probation, or too small to describe) is not an assignment.
    """
    assign_log: List[Tuple[str, str]] = []
    reentry_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    last_uid: Dict[str, str] = {}                  # name -> last uid it wore
    track_of: Dict[str, int] = {}                  # name -> current track id
    seen_tracks: Dict[str, set] = {}               # name -> every track id it wore
    pending: Dict[Tuple[str, int], Optional[str]] = {}   # (name, tid) -> pre uid
    uids_seen: set = set()

    for rows in frames:
        for tid, name, uid in rows:
            if name is None:
                continue
            tid = int(tid)
            tracks = seen_tracks.setdefault(name, set())
            if tid not in tracks:
                tracks.add(tid)
                if name in track_of:                # not the first track: a re-entry
                    pending[(name, tid)] = last_uid.get(name)
                track_of[name] = tid
            if uid is None:
                continue
            uids_seen.add(uid)
            key = (name, tid)
            if key in pending:
                reentry_pairs.append((pending.pop(key), uid))
            assign_log.append((name, uid))
            last_uid[name] = uid
    for key, pre in pending.items():               # never given a uid -> a miss
        reentry_pairs.append((pre, None))
    created = len(uids_seen) if workers_created is None else int(workers_created)
    if remap is not None and workers_created is None:
        created = len({remap(u) for u in uids_seen})
    return _score(assign_log, reentry_pairs, created, remap)


def _format(rows: Sequence[dict], clip: str, labelled: int, frames: int) -> str:
    out = [RULE, f"Worker re-ID on real footage, badge reads as ground truth",
           f"  {os.path.basename(clip)}: {frames} frames processed, "
           f"{labelled} badge-labelled person-frames", RULE,
           f"{'config':<18}{'re-ID recall':>14}{'false merge':>13}{'frag':>7}{'workers':>10}",
           "-" * len(RULE)]
    for r in rows:
        out.append(f"{r['config']:<18}{r['recovered']:>6}/{r['reentries']:<3}"
                   f"{r['reid_recall']:>4.0f}%{r['false_merge_rate']:>12.1f}%"
                   f"{r['fragmentation']:>7.2f}"
                   f"{r['workers_created']:>6}/{r['workers_true']:<4}")
    out += [RULE,
            "  re-ID recall : a badge name returning under a new track kept its uid",
            "  false merge  : assignments where a uid was worn by two badge names",
            "  frag         : distinct uids per badge name (1.00 = perfect)",
            "  workers      : uids created / badge names seen",
            "",
            "  Only badge-labelled frames are scored; tracks whose badge was never read are",
            "  invisible here. A small or hidden badge means a sparse score, not a bad one."]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Score appearance re-ID on a real clip, using badge reads as truth")
    ap.add_argument("video")
    ap.add_argument("--config", default=os.path.join(_ROOT, "phase2", "config.yaml"))
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--out", default="", help="write the rows as JSON here")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    import cv2
    from src.config import load_config
    from src.identity import IdentityManager
    from src.pipeline import SafetyPipeline, _tracked_arrays, identity_kwargs
    from src.reid import build_embedder
    from src.workid import aruco_available

    if not aruco_available():
        print("cv2.aruco unavailable — install opencv-contrib-python.", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    cfg.workid_enabled = True                      # the badge IS the ground truth here
    if not cfg.workid_markers:
        print("[warn] workid.markers is empty: unmapped tags get auto names 'W-<id>'.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open video: {args.video}", file=sys.stderr)
        return 1
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in != fps_in or not (0.1 <= fps_in <= 480):
        fps_in = 30.0
    every = max(1, int(args.every))
    eff_fps = max(1.0, fps_in / every)

    pipe = SafetyPipeline(cfg, frame_rate=int(round(eff_fps)), render=False, quiet=True)
    if pipe.binder is None:
        print("Work ID binder unavailable (see the warning above).", file=sys.stderr)
        return 1
    kw = identity_kwargs(cfg, eff_fps)
    # The synthetic harness's ENHANCED carries a per-FRAME gate speed for its 30 fps
    # sequences; here the manager is given clip time, so only the switches are taken
    # from it and the per-second knobs come from the config.
    switches = {k: v for k, v in ENHANCED.items() if k in ("probation_frames", "gate")}
    managers = {
        "baseline": IdentityManager(build_embedder(cfg.identity_method), **{**kw, **BASELINE}),
        "enhanced": IdentityManager(build_embedder(cfg.identity_method), **{**kw, **switches}),
    }
    streams: Dict[str, List[List[Row]]] = {k: [] for k in managers}

    frame_no = processed = labelled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if every > 1 and (frame_no - 1) % every:
            continue
        dets = pipe.detector.detect(frame)
        tracked, _v = pipe.tracker.update(dets)
        badge_of = pipe.binder.resolve(frame, tracked)
        ids, boxes = _tracked_arrays(tracked)
        labelled += sum(1 for t in ids if t in badge_of)
        clip_s = frame_no / fps_in
        for name, mgr in managers.items():
            out = mgr.update(frame, ids, boxes, now_s=clip_s)   # NO marker labels
            streams[name].append([(t, badge_of.get(t), out[t].uid if t in out else None)
                                  for t in ids])
        processed += 1
        if processed % 50 == 0:
            print(f"\r  {frame_no} frames", end="", flush=True)
        if args.max_frames and processed >= args.max_frames:
            break
    cap.release()
    print("\r" + " " * 30, end="\r")

    rows = []
    for name, mgr in managers.items():
        r = score_stream(streams[name], workers_created=len(mgr.workers))
        r["config"] = name
        rows.append(r)
    remap = _remap_from(managers["enhanced"].consolidate())
    r = score_stream(streams["enhanced"], remap=remap,
                     workers_created=len(managers["enhanced"].workers))
    r["config"] = "enhanced+offline"
    rows.append(r)

    print(_format(rows, args.video, labelled, processed))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"clip": os.path.abspath(args.video), "frames": processed,
                       "every": every, "labelled_person_frames": labelled, "rows": rows},
                      fh, indent=2)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
