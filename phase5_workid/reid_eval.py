"""Honest measurement of worker re-identification.

The claim being tested is narrow and specific: **when the tracker loses a worker and
gives them a new `tracker_id`, does the identity layer put them back together?** That is
the capability `phase2/src/identity.py` adds, so that is what gets measured — not MOTA,
not a person-re-ID leaderboard.

**Protocol — injected occlusion** (the same self-supervised shape as the Phase 3
injected-fault protocol). A sequence is generated in which every worker's true identity
is known by construction. At scripted points a worker is removed for a gap of frames and
returns under a **new track id**, exactly as ByteTrack behaves after an occlusion longer
than `lost_track_buffer`. Three things are then counted:

  * **Re-ID recall** — of those forced re-entries, how many were reunited with the right
    worker. Higher is better.
  * **False-merge rate** — how often a person was given a uid belonging to a *different*
    worker. This is the dangerous error: it silently transfers one person's violation
    history to another, so it is reported separately and never averaged away.
  * **Fragmentation** — distinct uids per true worker. 1.0 is perfect; 2.0 means each
    worker's record was split in half.

**Why synthetic.** The repository has no annotated multi-person video, and grading
re-ID against a tracker's own output would be circular. Generated sequences give exact
ground truth, run anywhere with no download, and are reproducible from a seed. The cost
is realism, and it is a real cost: these are rendered figures, not a construction site.
Treat the numbers as a measurement of the *matcher*, not a site-performance prediction.

The `uniform` scenario is the one to look at hardest: every worker in identical PPE, the
actual condition on site. It is designed to make appearance re-ID fail, because that
failure is the argument for the ArUco badge.

    python -m phase5_workid.reid_eval                     # all scenarios, default config
    python -m phase5_workid.reid_eval --sweep             # threshold sweep
    python -m phase5_workid.reid_eval --scenario uniform --workers 4
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phase2"))

from src.identity import IdentityManager          # noqa: E402
from src.reid import build_embedder               # noqa: E402

FRAME_H, FRAME_W = 480, 854
RULE = "-" * 74


# --- synthetic worker appearance ---------------------------------------------
@dataclass
class Outfit:
    helmet: Tuple[int, int, int]
    shirt: Tuple[int, int, int]
    trousers: Tuple[int, int, int]


DISTINCT = [
    Outfit((60, 200, 240), (40, 40, 200), (80, 60, 50)),      # yellow hat, red shirt
    Outfit((240, 200, 60), (200, 60, 40), (60, 50, 45)),      # blue hat, blue shirt
    Outfit((80, 220, 90), (60, 190, 60), (70, 70, 70)),       # green
    Outfit((250, 250, 250), (30, 120, 220), (40, 40, 60)),    # white hat, orange shirt
    Outfit((40, 40, 210), (200, 190, 60), (55, 55, 55)),      # red hat, cyan shirt
    Outfit((180, 90, 200), (120, 90, 180), (60, 55, 50)),     # purple
]
# The real construction site: everybody in the same kit.
UNIFORM = Outfit((60, 200, 240), (40, 150, 240), (60, 60, 65))   # yellow hat, hi-vis orange

# The realistic middle ground, and the most informative case: the SAME issued hi-vis
# vest (so the largest, most salient region is identical for everyone) but personal
# helmets and trousers. Neither trivially separable nor hopeless.
SIMILAR = [
    Outfit((60, 200, 240), (40, 150, 240), (70, 60, 50)),     # yellow helmet
    Outfit((240, 200, 60), (40, 150, 240), (50, 50, 70)),     # blue helmet
    Outfit((250, 250, 250), (40, 150, 240), (60, 70, 60)),    # white helmet
    Outfit((40, 40, 210), (40, 150, 240), (45, 45, 45)),      # red helmet
    Outfit((80, 220, 90), (40, 150, 240), (75, 65, 55)),      # green helmet
    Outfit((180, 90, 200), (40, 150, 240), (55, 60, 65)),     # purple helmet
]


@dataclass
class Track:
    """One appearance of a worker under one track id."""
    worker: int                   # TRUE identity
    track_id: int
    box: List[float]


@dataclass
class Sequence:
    frames: List[np.ndarray] = field(default_factory=list)
    tracks: List[List[Track]] = field(default_factory=list)     # per frame
    reentries: List[Tuple[int, int, int]] = field(default_factory=list)
    # (frame_idx, worker, new_track_id) for each forced re-entry after a gap


def _draw_worker(frame: np.ndarray, box, outfit: Outfit, rng, jitter: float) -> None:
    """Render a crude figure: helmet band, shirt band, trouser band, plus lighting
    jitter and noise so the descriptor is not handed a pixel-identical crop."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    h = y2 - y1
    bands = ((y1, y1 + int(h * 0.22), outfit.helmet),
             (y1 + int(h * 0.22), y1 + int(h * 0.62), outfit.shirt),
             (y1 + int(h * 0.62), y2, outfit.trousers))
    gain = 1.0 + rng.uniform(-jitter, jitter)
    for by1, by2, colour in bands:
        if by2 <= by1:
            continue
        c = np.clip(np.asarray(colour, dtype=np.float32) * gain, 0, 255)
        frame[by1:by2, x1:x2] = c.astype(np.uint8)
    patch = frame[y1:y2, x1:x2].astype(np.int16)
    patch += rng.integers(-12, 13, patch.shape, dtype=np.int16)
    frame[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)


def _background(rng) -> np.ndarray:
    """A textured, non-uniform background — a flat one would make centre weighting
    look better than it is."""
    bg = rng.integers(70, 150, (FRAME_H // 8, FRAME_W // 8, 3), dtype=np.uint8)
    import cv2
    return cv2.resize(bg, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)


def _motion_blur(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Directional blur standing in for the smear a head-mounted camera produces while
    the wearer turns. Kernel length follows the pan speed."""
    import cv2
    n = int(min(21, max(3, round(np.hypot(dx, dy)))))
    if n < 3:
        return frame
    k = np.zeros((n, n), np.float32)
    if abs(dx) >= abs(dy):
        k[n // 2, :] = 1.0
    else:
        k[:, n // 2] = 1.0
    return cv2.filter2D(frame, -1, k / k.sum())


def make_sequence(n_workers: int = 4, frames: int = 90, scenario: str = "distinct",
                  gap: int = 12, seed: int = 0, jitter: float = 0.10,
                  motion: float = 0.0) -> Sequence:
    """Build a sequence where every worker is occluded once and returns under a new id.

    `motion` (pixels/frame) adds head motion: the whole scene pans, so every worker
    translates together and the frame is motion-blurred in the direction of travel. It is
    the AR-glasses condition — a head-mounted camera is never still — and it attacks
    appearance re-ID through blur and changing background context.
    """
    rng = np.random.default_rng(seed)
    if scenario == "uniform":
        outfits = [UNIFORM] * n_workers
    elif scenario == "distinct":
        outfits = [DISTINCT[i % len(DISTINCT)] for i in range(n_workers)]
    elif scenario == "similar":
        outfits = [SIMILAR[i % len(SIMILAR)] for i in range(n_workers)]
    else:
        raise ValueError(f"unknown scenario '{scenario}' "
                         "(distinct | similar | uniform)")

    seq = Sequence()
    next_track_id = 1
    track_of = {w: (next_track_id + w) for w in range(n_workers)}
    next_track_id += n_workers
    # Each worker vanishes for `gap` frames starting at a staggered point.
    start_gap = {w: int(frames * 0.35) + w * 4 for w in range(n_workers)}

    x0 = np.linspace(60, FRAME_W - 160, n_workers)
    for f in range(frames):
        frame = _background(rng)
        # Head motion: one global offset applied to EVERY worker this frame, plus blur.
        # Amplitudes are chosen so the PEAK per-frame displacement really is `motion`
        # pixels (d/df of A*sin(f/T) peaks at A/T), keeping the units honest.
        pan_x = motion * 9.0 * np.sin(f / 9.0) if motion else 0.0
        pan_y = motion * 13.0 * 0.3 * np.sin(f / 13.0) if motion else 0.0
        prev_x = motion * 9.0 * np.sin((f - 1) / 9.0) if motion else 0.0
        prev_y = motion * 13.0 * 0.3 * np.sin((f - 1) / 13.0) if motion else 0.0
        d_x, d_y = pan_x - prev_x, pan_y - prev_y      # this frame's actual travel
        present: List[Track] = []
        for w in range(n_workers):
            g0 = start_gap[w]
            if g0 <= f < g0 + gap:
                continue                                    # occluded
            if f == g0 + gap:                               # returns as a NEW id
                track_of[w] = next_track_id
                next_track_id += 1
                seq.reentries.append((f, w, track_of[w]))
            # gentle motion + size change (distance)
            cx = x0[w] + 30.0 * np.sin((f + w * 7) / 18.0) + pan_x
            hh = 190 + 25 * np.sin((f + w * 11) / 25.0)
            ww = hh * 0.42
            y1 = 120 + 20 * np.cos((f + w * 5) / 21.0) + pan_y
            box = [float(cx), float(y1), float(cx + ww), float(y1 + hh)]
            _draw_worker(frame, box, outfits[w], rng, jitter)
            present.append(Track(worker=w, track_id=track_of[w], box=box))
        if motion:
            # Blur follows THIS frame's travel, so a fast swing smears and the
            # turnaround points (where the camera is momentarily still) do not.
            frame = _motion_blur(frame, d_x, d_y)
        seq.frames.append(frame)
        seq.tracks.append(present)
    return seq


# --- evaluation ---------------------------------------------------------------
def evaluate(manager: IdentityManager, seq: Sequence) -> dict:
    """Replay a sequence through the identity manager and score it against ground truth."""
    last_uid: Dict[int, str] = {}                # worker -> uid on the last frame seen
    uid_by_worker: Dict[int, set] = {}
    assignments = 0
    false_merges = 0
    uid_owner: Dict[str, int] = {}               # uid -> the worker that first claimed it
    recovered = 0
    reentries_scored = 0
    by_frame: Dict[int, List[Tuple[int, int]]] = {}
    for rf, w, tid in seq.reentries:
        by_frame.setdefault(rf, []).append((w, tid))

    for f, (frame, tracks) in enumerate(zip(seq.frames, seq.tracks)):
        ids = [t.track_id for t in tracks]
        boxes = [t.box for t in tracks]
        out = manager.update(frame, ids, boxes)

        # Score re-entries FIRST, while `last_uid` still holds the identity from BEFORE
        # the gap. Updating it beforehand would compare the new assignment against
        # itself and report 100% recall no matter how the matcher behaved.
        for w, tid in by_frame.get(f, []):
            res = out.get(tid)
            if res is None:
                continue
            reentries_scored += 1
            if last_uid.get(w) is not None and last_uid[w] == res.uid:
                recovered += 1

        for t in tracks:
            res = out.get(t.track_id)
            if res is None:
                continue
            assignments += 1
            uid_by_worker.setdefault(t.worker, set()).add(res.uid)
            owner = uid_owner.setdefault(res.uid, t.worker)
            if owner != t.worker:
                false_merges += 1
            last_uid[t.worker] = res.uid

    workers = [w for w, s in uid_by_worker.items() if s]
    frag = float(np.mean([len(uid_by_worker[w]) for w in workers])) if workers else 0.0
    return {
        "reentries": reentries_scored,
        "recovered": recovered,
        "reid_recall": 100.0 * recovered / max(reentries_scored, 1),
        "assignments": assignments,
        "false_merges": false_merges,
        "false_merge_rate": 100.0 * false_merges / max(assignments, 1),
        "fragmentation": frag,
        "workers_true": len(workers),
        "workers_created": len(manager.workers),
    }


def tracker_churn(seq: Sequence, lost_buffer: int = 30, frame_rate: int = 30) -> dict:
    """Run the REAL ByteTrack over the sequence's boxes and count how badly it fragments.

    This measures the thing head motion actually breaks. The re-ID protocol above scripts
    a fixed number of track breaks, so by construction it cannot show that a moving camera
    creates *more* of them — ByteTrack associates by motion continuity, and a head-mounted
    camera translates the whole scene every frame, which is exactly what its constant-
    velocity assumption does not expect.

    Ground truth is known, so each returned tracker_id is attributed to a true worker by
    best IoU against the boxes fed in that frame. Reported as distinct tracker_ids per
    true worker: 1.0 means the tracker never lost anybody.
    """
    import supervision as sv

    from src.tracker import PersonTracker

    tracker = PersonTracker([0], [], frame_rate=frame_rate, lost_track_buffer=lost_buffer)
    ids_per_worker: Dict[int, set] = {}
    for tracks in seq.tracks:
        if tracks:
            xyxy = np.asarray([t.box for t in tracks], dtype=np.float32)
            dets = sv.Detections(xyxy=xyxy,
                                 confidence=np.full(len(tracks), 0.9, dtype=np.float32),
                                 class_id=np.zeros(len(tracks), dtype=int))
        else:
            dets = sv.Detections.empty()
        tracked, _v = tracker.update(dets)
        if len(tracked) == 0 or tracked.tracker_id is None:
            continue
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue
            best, best_iou = None, 0.0
            for t in tracks:                       # attribute back to the true worker
                iou = _iou(box, t.box)
                if iou > best_iou:
                    best, best_iou = t.worker, iou
            if best is not None and best_iou > 0.5:
                ids_per_worker.setdefault(best, set()).add(int(tid))
    if not ids_per_worker:
        return {"ids_per_worker": 0.0, "workers": 0}
    return {"ids_per_worker": float(np.mean([len(v) for v in ids_per_worker.values()])),
            "workers": len(ids_per_worker)}


def pipeline_eval(seq: Sequence, method: str = "histogram", threshold: float = 0.62,
                  lost_buffer: int = 30, frame_rate: int = 30) -> dict:
    """End-to-end: real ByteTrack -> IdentityManager, scored against ground truth.

    This is the measurement that answers the AR question directly. ByteTrack alone
    fragments a worker into several tracker_ids when the camera moves; the identity layer
    exists to glue those back into one worker. Reporting both numbers side by side shows
    how much of the damage is actually repaired, rather than asserting it.
    """
    import supervision as sv

    from src.tracker import PersonTracker

    tracker = PersonTracker([0], [], frame_rate=frame_rate, lost_track_buffer=lost_buffer)
    manager = _fresh(method, threshold)
    track_ids_per_worker: Dict[int, set] = {}
    uids_per_worker: Dict[int, set] = {}
    uid_owner: Dict[str, int] = {}
    assignments = false_merges = 0

    for f_idx, tracks in enumerate(seq.tracks):
        if tracks:
            dets = sv.Detections(
                xyxy=np.asarray([t.box for t in tracks], dtype=np.float32),
                confidence=np.full(len(tracks), 0.9, dtype=np.float32),
                class_id=np.zeros(len(tracks), dtype=int))
        else:
            dets = sv.Detections.empty()
        tracked, _v = tracker.update(dets)
        if len(tracked) == 0 or tracked.tracker_id is None:
            continue
        ids, boxes, owners = [], [], {}
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue
            best, best_iou = None, 0.0
            for t in tracks:
                iou = _iou(box, t.box)
                if iou > best_iou:
                    best, best_iou = t.worker, iou
            if best is None or best_iou <= 0.5:
                continue
            ids.append(int(tid))
            boxes.append([float(v) for v in box])
            owners[int(tid)] = best
            track_ids_per_worker.setdefault(best, set()).add(int(tid))

        out = manager.update(seq.frames[f_idx], ids, boxes)
        for tid, res in out.items():
            w = owners.get(tid)
            if w is None:
                continue
            assignments += 1
            uids_per_worker.setdefault(w, set()).add(res.uid)
            if uid_owner.setdefault(res.uid, w) != w:
                false_merges += 1

    def _mean(d):
        return float(np.mean([len(v) for v in d.values()])) if d else 0.0

    return {
        "bytetrack_ids_per_worker": _mean(track_ids_per_worker),
        "identity_uids_per_worker": _mean(uids_per_worker),
        "workers": len(track_ids_per_worker),
        "false_merge_rate": 100.0 * false_merges / max(assignments, 1),
    }


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _fresh(method: str, threshold: float, margin: float = 0.04) -> IdentityManager:
    return IdentityManager(build_embedder(method), match_threshold=threshold,
                           margin=margin, forget_after=10 ** 9)


def run_scenario(scenario: str, method: str, threshold: float, workers: int,
                 frames: int, gap: int, seed: int, margin: float = 0.04,
                 motion: float = 0.0) -> dict:
    seq = make_sequence(n_workers=workers, frames=frames, scenario=scenario,
                        gap=gap, seed=seed, motion=motion)
    res = evaluate(_fresh(method, threshold, margin), seq)
    res.update({"scenario": scenario, "method": method, "threshold": threshold,
                "motion": motion})
    return res


def print_table(rows: Sequence[dict], title: str) -> None:
    print(RULE)
    print(title)
    print(RULE)
    print(f"{'scenario':<12}{'head':>6}{'thr':>6}{'re-ID recall':>14}{'false merge':>13}"
          f"{'frag':>7}{'workers':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['scenario']:<12}{r.get('motion', 0.0):>6.0f}{r['threshold']:>6.2f}"
              f"{r['recovered']:>6}/{r['reentries']:<3}{r['reid_recall']:>4.0f}%"
              f"{r['false_merge_rate']:>12.1f}%"
              f"{r['fragmentation']:>7.2f}"
              f"{r['workers_created']:>6.1f}/{r['workers_true']:<4.1f}")
    print(RULE)


def _run_pipeline(args) -> int:
    """The AR-glasses table: how much head-motion damage the identity layer repairs."""
    scen = "similar" if args.scenario == "all" else args.scenario
    motions = [0.0, 4.0, 8.0, 12.0, 20.0] if args.motion <= 0 else [args.motion]
    print(RULE)
    print("End-to-end under head motion: real ByteTrack -> identity layer")
    print(f"  scenario={scen}  workers={args.workers}  seeds={args.seeds}  "
          f"(lower is better; 1.00 = never lost anybody)")
    print(RULE)
    print(f"{'head px/frame':>14}{'ByteTrack ids/worker':>22}"
          f"{'+ identity':>13}{'false merge':>13}")
    print("-" * 74)
    for mot in motions:
        bt, uid, fm = [], [], []
        for s in range(max(1, args.seeds)):
            seq = make_sequence(n_workers=args.workers, frames=args.frames, scenario=scen,
                                gap=args.gap, seed=args.seed + s, motion=mot)
            r = pipeline_eval(seq, args.method, args.threshold)
            bt.append(r["bytetrack_ids_per_worker"])
            uid.append(r["identity_uids_per_worker"])
            fm.append(r["false_merge_rate"])
        print(f"{mot:>14.0f}{np.mean(bt):>22.2f}{np.mean(uid):>13.2f}"
              f"{np.mean(fm):>12.1f}%")
    print(RULE)
    print("  A head-mounted camera translates the whole scene every frame, which is not")
    print("  what ByteTrack's constant-velocity model expects, so it splits one worker")
    print("  across several tracker_ids. The identity layer glues them back together.")
    print("  Watch the last row: past a point the appearance gallery itself breaks down")
    print("  and false merges appear -- that is the honest operating limit, not a bug.")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Measure worker re-identification across injected occlusions.")
    ap.add_argument("--scenario", default="all",
                    choices=["distinct", "similar", "uniform", "all"])
    ap.add_argument("--method", default="histogram", help="histogram | deep")
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--margin", type=float, default=0.04)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3,
                    help="average over this many seeds (a single sequence is noisy)")
    ap.add_argument("--pipeline", action="store_true",
                    help="AR-glasses table: run REAL ByteTrack into the identity layer "
                         "at increasing head motion and show how much it repairs")
    ap.add_argument("--motion", type=float, default=0.0, metavar="PX",
                    help="head motion in px/frame: pans the whole scene and adds "
                         "directional blur (the AR-glasses condition)")
    ap.add_argument("--motion-sweep", action="store_true",
                    help="compare a still camera against increasing head motion")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the match threshold to expose the recall/false-merge trade-off")
    args = ap.parse_args(argv)

    if args.pipeline:
        return _run_pipeline(args)

    scenarios = (["distinct", "similar", "uniform"] if args.scenario == "all"
                 else [args.scenario])
    thresholds = ([0.40, 0.50, 0.55, 0.60, 0.62, 0.70, 0.80, 0.90]
                  if args.sweep else [args.threshold])
    motions = [0.0, 2.0, 4.0, 8.0] if args.motion_sweep else [args.motion]

    rows = []
    for scen in scenarios:
      for mot in motions:
        for thr in thresholds:
            runs = [run_scenario(scen, args.method, thr, args.workers, args.frames,
                                 args.gap, args.seed + s, args.margin, mot)
                    for s in range(max(1, args.seeds))]
            agg = {k: float(np.mean([r[k] for r in runs]))
                   for k in ("reid_recall", "false_merge_rate", "fragmentation",
                             "workers_created", "workers_true")}
            agg.update({
                "scenario": scen, "method": args.method, "threshold": thr,
                "motion": mot,
                "recovered": int(sum(r["recovered"] for r in runs)),
                "reentries": int(sum(r["reentries"] for r in runs)),
            })
            rows.append(agg)

    title = (f"Worker re-ID across injected occlusions  "
             f"({args.method}, {args.seeds} seed(s), {args.workers} workers, "
             f"gap {args.gap} frames)")
    if args.motion_sweep or args.motion:
        title += "\n  head = simulated head motion in px/frame (0 = tripod)"
    print_table(rows, title)
    print("  re-ID recall : forced re-entries reunited with the right worker (higher better)")
    print("  false merge  : assignments given ANOTHER worker's identity (lower better --")
    print("                 this is the error that moves a violation onto the wrong person)")
    print("  frag         : distinct identities per true worker (1.00 = perfect)")
    print("  workers      : identities created / true workers")
    print()
    print("  Synthetic sequences with exact ground truth: this measures the MATCHER, not")
    print("  site performance. distinct = different clothing; similar = SAME issued vest,")
    print("  personal helmets; uniform = identical PPE (the hardest, most realistic case).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
