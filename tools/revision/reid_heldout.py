"""Revision R1, Part F.1: the Work ID ablation on held-out seeds and unseen configurations.

`python -m phase5_workid.reid_eval --ablation --seed 100 --seeds 20` prints only the 20-seed
means. This script runs exactly the same computation one seed at a time -- the same
`make_sequence`, `_fresh` and `evaluate` calls with the same arguments that
`reid_eval.main()` -> `run_scenario()` makes -- so it can report the spread across seeds,
and checks its means against the CLI's printed table. Nothing in phase5_workid/ or
phase2/src/ is modified.

Optional IDF1 (--idf1): a recording proxy wraps the IdentityManager, so every frame's
(true worker, assigned uid) pairs are captured without changing `evaluate()`. IDF1 is then
computed the standard way (Ristani et al., 2016): a global one-to-one assignment between true
workers and uids maximising co-occurring frames; IDTP / IDFP / IDFN from it. Boxes are perfect
by construction, so this measures identity assignment only. A track still in probation has no
uid that frame and counts as a missed detection (IDFN); for `enhanced+offline` the
consolidation merge map is applied first, exactly as `_score` does.

    .venv\\Scripts\\python.exe tools\\revision\\reid_heldout.py --idf1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from phase5_workid import reid_eval as re_          # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partF1")
SEED0, NSEEDS = 100, 20
THRESHOLD, MARGIN, METHOD = 0.62, 0.04, "histogram"      # the CLI defaults, frozen
BASE = {"workers": 4, "frames": 90, "gap": 12, "motion": 0.0}
CONFIGURATIONS = [            # (name, CLI flags, overrides of BASE)
    ("default", [], {}),
    ("gap6", ["--gap", "6"], {"gap": 6}),
    ("gap18", ["--gap", "18"], {"gap": 18}),
    ("gap24", ["--gap", "24"], {"gap": 24}),
    ("workers3", ["--workers", "3"], {"workers": 3}),
    ("workers6", ["--workers", "6"], {"workers": 6}),
    ("frames120", ["--frames", "120"], {"frames": 120}),
    ("motion4", ["--motion", "4"], {"motion": 4.0}),
]
SCENARIOS = ("distinct", "similar", "uniform")
T_975_19 = 2.093024054408263          # Student t, 0.975 quantile, 19 df (20 seeds)


class Recorder:
    """Delegates to the real IdentityManager and records what it returned each frame."""

    def __init__(self, manager):
        self._m = manager
        self.frames = []
        self.merges = None

    def update(self, frame, ids, boxes):
        out = self._m.update(frame, ids, boxes)
        self.frames.append({tid: (res.uid if res is not None else None)
                            for tid, res in out.items()})
        return out

    def consolidate(self):
        self.merges = self._m.consolidate()
        return self.merges

    def __getattr__(self, name):
        return getattr(self._m, name)


def idf1(seq, rec):
    from scipy.optimize import linear_sum_assignment

    remap = re_._remap_from(rec.merges) if rec.merges is not None else (lambda u: u)
    gt_dets = pred_dets = 0
    co = defaultdict(int)
    for f, tracks in enumerate(seq.tracks):
        out = rec.frames[f]
        for t in tracks:
            uid = out.get(t.track_id)
            uid = remap(uid) if uid is not None else None
            if t.worker >= 0:
                gt_dets += 1
            if uid is not None:
                pred_dets += 1
                if t.worker >= 0:
                    co[(t.worker, uid)] += 1
    if not co:
        return {"idtp": 0, "idfp": pred_dets, "idfn": gt_dets, "idf1": 0.0}
    gts = sorted({g for g, _ in co})
    uids = sorted({u for _, u in co})
    M = np.zeros((len(gts), len(uids)))
    for (g, u), v in co.items():
        M[gts.index(g), uids.index(u)] = v
    r, c = linear_sum_assignment(-M)
    idtp = float(M[r, c].sum())
    idfp, idfn = pred_dets - idtp, gt_dets - idtp
    return {"idtp": idtp, "idfp": idfp, "idfn": idfn,
            "idf1": 100.0 * 2 * idtp / (2 * idtp + idfp + idfn)}


def run_one(p, scenario, name, kw, consolidate, seed, with_idf1):
    """Mirror of reid_eval.run_scenario (same calls, same arguments), plus the recorder."""
    seq = re_.make_sequence(n_workers=p["workers"], frames=p["frames"], scenario=scenario,
                            gap=p["gap"], seed=seed, motion=p["motion"], phantoms=0,
                            relocate=False, swap=False)
    mgr = re_._fresh(METHOD, THRESHOLD, MARGIN, **kw)
    rec = Recorder(mgr)
    res = re_.evaluate(rec, seq, consolidate=consolidate)
    row = {k: res[k] for k in ("reentries", "recovered", "reid_recall", "assignments",
                               "false_merges", "false_merge_rate", "fragmentation",
                               "workers_true", "workers_created")}
    if with_idf1:
        row.update(idf1(seq, rec))
    return row


def stats(v):
    v = np.asarray(v, float)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    half = T_975_19 * sd / np.sqrt(len(v))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": m, "sd": sd, "ci95_mean": [m - half, m + half],
            "p2.5_p97.5": [float(lo), float(hi)], "min": float(v.min()), "max": float(v.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--idf1", action="store_true", help="also compute IDF1 (optional output)")
    ap.add_argument("--only", nargs="*", default=None, help="configuration names to run")
    ap.add_argument("--seeds", type=int, default=NSEEDS)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    confs = [c for c in CONFIGURATIONS if a.only is None or c[0] in a.only]

    t_all = time.time()
    rows = []
    timing = {}
    for cname, flags, ov in confs:
        p = dict(BASE, **ov)
        t0 = time.time()
        for scen in SCENARIOS:
            for name, kw, consolidate in re_.CONFIGS:
                for s in range(a.seeds):
                    r = run_one(p, scen, name, dict(kw), consolidate, SEED0 + s, a.idf1)
                    rows.append({"configuration": cname, "scenario": scen, "row": name,
                                 "seed": SEED0 + s, **r})
        timing[cname] = round(time.time() - t0, 1)
        print(f"[{datetime.datetime.now():%H:%M:%S}] {cname}: {timing[cname]:.0f} s", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "per_seed.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    for cname, flags, ov in confs:
        for scen in SCENARIOS:
            for name, _kw, _c in re_.CONFIGS:
                sel = [r for r in rows if r["configuration"] == cname and r["scenario"] == scen
                       and r["row"] == name]
                ent = {"configuration": cname, "cli_flags": " ".join(flags), "scenario": scen,
                       "row": name, "seeds": f"{SEED0}-{SEED0 + a.seeds - 1}",
                       "reentries_total": int(sum(r["reentries"] for r in sel)),
                       "recovered_total": int(sum(r["recovered"] for r in sel)),
                       "reid_recall": stats([r["reid_recall"] for r in sel]),
                       "false_merge_rate": stats([r["false_merge_rate"] for r in sel]),
                       "false_merges_total": int(sum(r["false_merges"] for r in sel)),
                       "fragmentation": stats([r["fragmentation"] for r in sel])}
                if a.idf1:
                    ent["idf1"] = stats([r["idf1"] for r in sel])
                summary.append(ent)

    meta = {"command": " ".join([sys.executable] + sys.argv),
            "equivalent_cli": [f"python -m phase5_workid.reid_eval --ablation --seed {SEED0} "
                               f"--seeds {a.seeds} " + " ".join(f) for _, f, _ in confs],
            "frozen": {"method": METHOD, "threshold": THRESHOLD, "margin": MARGIN,
                       "configs": [[n, kw, c] for n, kw, c in re_.CONFIGS]},
            "seconds": round(time.time() - t_all, 1), "seconds_per_configuration": timing,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "opencv": __import__("cv2").__version__,
                         "platform": platform.platform()},
            "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(f"wrote {a.out} in {meta['seconds'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
