"""Sensitivity of ids_per_gt / merged_ids to the per-frame matching rule, on TrackEval's preprocessed data.

A: as in mot17_eval.id_counts (Hungarian on raw IoU, then keep pairs >= 0.5)
B: threshold first (IoU < 0.5 zeroed before the Hungarian), the TrackEval convention
C: CLEAR's own matches (continuity bonus), i.e. the pairs that define IDSW
Reads only; writes nothing outside this script's stdout.
"""
import csv
import os
import sys

import numpy as np

for _a, _b in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _b)
import trackeval  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
TRACKERS = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge")
CSV = os.path.join(ROOT, "paper_materials", "revision_r1", "partD", "per_sequence_metrics.csv")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
EPS = np.finfo("float").eps


def dataset(bench, trackers):
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update({"GT_FOLDER": GT_TE, "TRACKERS_FOLDER": TRACKERS, "OUTPUT_FOLDER": None,
               "TRACKERS_TO_EVAL": trackers, "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": bench,
               "SPLIT_TO_EVAL": "train", "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data",
               "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
    return trackeval.datasets.MotChallenge2DBox(dc)


def counts(data):
    out = {}
    for variant in "ABC":
        gt_to, tr_to = {}, {}
        prev = {}
        for t in range(data["num_timesteps"]):
            g, p, s = data["gt_ids"][t], data["tracker_ids"][t], data["similarity_scores"][t]
            if len(g) == 0 or len(p) == 0:
                continue
            if variant == "A":
                m = s
            elif variant == "B":
                m = s.copy(); m[m < 0.5 - EPS] = 0
            else:
                bonus = np.array([[1.0 if prev.get(int(gi)) == int(pj) else 0.0 for pj in p] for gi in g])
                m = 1000 * bonus + s
                m[s < 0.5 - EPS] = 0
            r, c = linear_sum_assignment(-m)
            if variant == "C":
                prev = {}
            for i, j in zip(r, c):
                if s[i, j] >= 0.5 - EPS and (variant == "A" or m[i, j] > EPS):
                    gt_to.setdefault(int(g[i]), set()).add(int(p[j]))
                    tr_to.setdefault(int(p[j]), set()).add(int(g[i]))
                    if variant == "C":
                        prev[int(g[i])] = int(p[j])
        out[variant] = (len(gt_to), sum(len(v) for v in gt_to.values()),
                        sum(1 for v in tr_to.values() if len(v) >= 2))
    return out


def main():
    want = sys.argv[1:]
    for bench, rate in (("MOT17", "native"), ("MOT17sub7", "7hz")):
        folder = os.path.join(TRACKERS, f"{bench}-train")
        trackers = sorted(t for t in os.listdir(folder)
                          if all(os.path.isfile(os.path.join(folder, t, "data", f"{s}.txt")) for s in SEQS))
        trackers = [t for t in trackers if not want or t in want]
        if not trackers:
            continue
        ds = dataset(bench, trackers)
        for trk in trackers:
            tot = {v: [0, 0, 0] for v in "ABC"}
            for seq in SEQS:
                data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(trk, seq), "pedestrian")
                c = counts(data)
                for v in "ABC":
                    for k in range(3):
                        tot[v][k] += c[v][k]
            ref = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
                   if r["rate"] == rate and r["run"] == trk and r["sequence"] == "COMBINED"][0]
            line = f"{rate:<6} {trk:<46} csv ids/gt {float(ref['ids_per_gt']):.4f} merged {ref['merged_ids']:>4} |"
            for v in "ABC":
                m, sm, mg = tot[v]
                line += f" {v}: ids/gt {sm / m:.4f} merged {mg:>4} matchedGT {m:>3} |"
            print(line, flush=True)


if __name__ == "__main__":
    main()
