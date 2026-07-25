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


def make_sequence(n_workers: int = 4, frames: int = 90, scenario: str = "distinct",
                  gap: int = 12, seed: int = 0, jitter: float = 0.10) -> Sequence:
    """Build a sequence where every worker is occluded once and returns under a new id."""
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
            cx = x0[w] + 30.0 * np.sin((f + w * 7) / 18.0)
            hh = 190 + 25 * np.sin((f + w * 11) / 25.0)
            ww = hh * 0.42
            y1 = 120 + 20 * np.cos((f + w * 5) / 21.0)
            box = [float(cx), float(y1), float(cx + ww), float(y1 + hh)]
            _draw_worker(frame, box, outfits[w], rng, jitter)
            present.append(Track(worker=w, track_id=track_of[w], box=box))
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


def _fresh(method: str, threshold: float, margin: float = 0.04) -> IdentityManager:
    return IdentityManager(build_embedder(method), match_threshold=threshold,
                           margin=margin, forget_after=10 ** 9)


def run_scenario(scenario: str, method: str, threshold: float, workers: int,
                 frames: int, gap: int, seed: int, margin: float = 0.04) -> dict:
    seq = make_sequence(n_workers=workers, frames=frames, scenario=scenario,
                        gap=gap, seed=seed)
    res = evaluate(_fresh(method, threshold, margin), seq)
    res.update({"scenario": scenario, "method": method, "threshold": threshold})
    return res


def print_table(rows: Sequence[dict], title: str) -> None:
    print(RULE)
    print(title)
    print(RULE)
    print(f"{'scenario':<12}{'thr':>6}{'re-ID recall':>14}{'false merge':>13}"
          f"{'frag':>7}{'workers':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['scenario']:<12}{r['threshold']:>6.2f}"
              f"{r['recovered']:>6}/{r['reentries']:<3}{r['reid_recall']:>4.0f}%"
              f"{r['false_merge_rate']:>12.1f}%"
              f"{r['fragmentation']:>7.2f}"
              f"{r['workers_created']:>6.1f}/{r['workers_true']:<4.1f}")
    print(RULE)


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
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the match threshold to expose the recall/false-merge trade-off")
    args = ap.parse_args(argv)

    scenarios = (["distinct", "similar", "uniform"] if args.scenario == "all"
                 else [args.scenario])
    thresholds = ([0.40, 0.50, 0.55, 0.60, 0.62, 0.70, 0.80, 0.90]
                  if args.sweep else [args.threshold])

    rows = []
    for scen in scenarios:
        for thr in thresholds:
            runs = [run_scenario(scen, args.method, thr, args.workers, args.frames,
                                 args.gap, args.seed + s, args.margin)
                    for s in range(max(1, args.seeds))]
            agg = {k: float(np.mean([r[k] for r in runs]))
                   for k in ("reid_recall", "false_merge_rate", "fragmentation",
                             "workers_created", "workers_true")}
            agg.update({
                "scenario": scen, "method": args.method, "threshold": thr,
                "recovered": int(sum(r["recovered"] for r in runs)),
                "reentries": int(sum(r["reentries"] for r in runs)),
            })
            rows.append(agg)

    title = (f"Worker re-ID across injected occlusions  "
             f"({args.method}, {args.seeds} seed(s), {args.workers} workers, "
             f"gap {args.gap} frames)")
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
