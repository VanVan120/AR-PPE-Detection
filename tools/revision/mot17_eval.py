"""Revision R1, Part D: TrackEval scoring of every tracker run, plus identity-fragmentation counts.

Runs TrackEval's MotChallenge2DBox with its standard preprocessing (pedestrians only; tracker
boxes matched to distractor classes and zero-marked ground truth removed; nothing custom) and
the HOTA, CLEAR, Identity and Count metrics, per sequence and combined, for the native-rate
benchmark (MOT17-train) and the ~7 Hz subsample (MOT17sub7-train).

Added on top, from TrackEval's own preprocessed data (so the same pedestrian/distractor
handling applies): per-frame Hungarian matching at IoU >= 0.5, then
  * ids_per_gt   : distinct predicted IDs matched to each ground-truth identity (mean over
                   identities matched at least once; never-matched identities counted apart)
  * merged_ids   : predicted IDs matched to two or more ground-truth identities
and the person detector's precision / recall at conf > 0.35, IoU 0.5: every cached detection
above 0.35 is written as its own one-box track, so TrackEval's CLEAR TP / FP / FN at IoU 0.5
are exactly detection counts under the standard MOT17 distractor handling.

TrackEval (git 12c8791) still uses np.float / np.int / np.bool, which NumPy 1.24 removed. The
three names are restored as aliases of the builtins they always were (float, int, bool) before
importing it; TrackEval itself is not modified.

    .venv-eval\\Scripts\\python.exe tools\\revision\\mot17_eval.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

for _alias, _builtin in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)          # restore removed NumPy aliases for TrackEval

import trackeval                                           # noqa: E402
from scipy.optimize import linear_sum_assignment           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TRACKERS = os.path.join(CACHE, "trackers", "mot_challenge")
OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partD")
RUNS = os.path.join(OUT, "runs")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
BENCH = {"MOT17": "native", "MOT17sub7": "7hz"}


def write_detection_tracks(det):
    """Each detection with conf > 0.35 as its own track (unique id)."""
    name = f"{det}__detections_conf035"
    d = os.path.join(TRACKERS, "MOT17-train", name, "data")
    os.makedirs(d, exist_ok=True)
    tid = 0
    for seq in SEQS:
        z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
        frames, off, dets = z["frames"], z["off"], z["dets"]
        with open(os.path.join(d, f"{seq}.txt"), "w", encoding="utf-8") as fh:
            for i, f in enumerate(frames):
                for x1, y1, x2, y2, c, _ in dets[off[i]:off[i + 1]]:
                    if c > 0.35:
                        tid += 1
                        fh.write(f"{int(f)},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n")
    return name


def derive_uid_only():
    """For every Work ID run, a '_uidonly' copy without the probation boxes.

    workid_phase2 emits a box in probation (no uid yet) under a provisional id 1,000,000 + tid,
    as the app shows 'Person #tid'; the switch to the uid then counts as an ID switch. The
    '_uidonly' variant drops those boxes instead (fewer detections, no provisional ids), which
    isolates what the identity layer does to association. Derived from the same output files.
    """
    made = []
    for split in ("MOT17-train", "MOT17sub7-train"):
        folder = os.path.join(TRACKERS, split)
        if not os.path.isdir(folder):
            continue
        for run in os.listdir(folder):
            if "workid_phase2" not in run or run.endswith("_uidonly"):
                continue
            src = os.path.join(folder, run, "data")
            dst = os.path.join(folder, run + "_uidonly", "data")
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                with open(os.path.join(src, f), encoding="utf-8") as fi, \
                        open(os.path.join(dst, f), "w", encoding="utf-8") as fo:
                    for line in fi:
                        if int(line.split(",", 2)[1]) < 1_000_000:
                            fo.write(line)
            made.append(f"{split}/{run}_uidonly")
    return made


def evaluate(benchmark, trackers):
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
               "TIME_PROGRESS": False, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True,
               "PLOT_CURVES": False, "BREAK_ON_ERROR": True, "DISPLAY_LESS_PROGRESS": True})
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update({"GT_FOLDER": GT_TE, "TRACKERS_FOLDER": TRACKERS,
               "OUTPUT_FOLDER": os.path.join(OUT, "trackeval", f"{benchmark}-train"),
               "TRACKERS_TO_EVAL": trackers, "CLASSES_TO_EVAL": ["pedestrian"],
               "BENCHMARK": benchmark, "SPLIT_TO_EVAL": "train", "DO_PREPROC": True,
               "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
    dataset = trackeval.datasets.MotChallenge2DBox(dc)
    quiet = {"PRINT_CONFIG": False}
    # Count is added by Evaluator.evaluate itself ("Count metrics are always run").
    metrics = [trackeval.metrics.HOTA(quiet), trackeval.metrics.CLEAR({**quiet, "THRESHOLD": 0.5}),
               trackeval.metrics.Identity({**quiet, "THRESHOLD": 0.5})]
    res, _msg = trackeval.Evaluator(ec).evaluate([dataset], metrics)
    return res["MotChallenge2DBox"], dataset


def id_counts(dataset, tracker, seq):
    raw = dataset.get_raw_seq_data(tracker, seq)
    data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
    gt_to, tr_to = {}, {}
    for t in range(data["num_timesteps"]):
        g, p, s = data["gt_ids"][t], data["tracker_ids"][t], data["similarity_scores"][t]
        if len(g) == 0 or len(p) == 0:
            continue
        r, c = linear_sum_assignment(-s)
        for i, j in zip(r, c):
            if s[i, j] >= 0.5 - np.finfo("float").eps:
                gt_to.setdefault(int(g[i]), set()).add(int(p[j]))
                tr_to.setdefault(int(p[j]), set()).add(int(g[i]))
    n_gt = data["num_gt_ids"]
    matched = len(gt_to)
    return {"gt_ids": int(n_gt), "gt_ids_matched": matched, "gt_ids_never_matched": int(n_gt - matched),
            "pred_ids": int(data["num_tracker_ids"]), "pred_ids_matched": len(tr_to),
            "sum_ids_over_matched_gt": int(sum(len(v) for v in gt_to.values())),
            "ids_per_gt": (sum(len(v) for v in gt_to.values()) / matched) if matched else None,
            "merged_ids": int(sum(1 for v in tr_to.values() if len(v) >= 2))}


def row_of(r):
    h, c, i, n = r["pedestrian"]["HOTA"], r["pedestrian"]["CLEAR"], r["pedestrian"]["Identity"], r["pedestrian"]["Count"]
    return {"HOTA": 100 * float(np.mean(h["HOTA"])), "DetA": 100 * float(np.mean(h["DetA"])),
            "AssA": 100 * float(np.mean(h["AssA"])), "IDF1": 100 * float(i["IDF1"]),
            "IDSW": int(c["IDSW"]), "MOTA": 100 * float(c["MOTA"]), "CLR_Re": 100 * float(c["CLR_Re"]),
            "CLR_Pr": 100 * float(c["CLR_Pr"]), "CLR_TP": int(c["CLR_TP"]), "CLR_FN": int(c["CLR_FN"]),
            "CLR_FP": int(c["CLR_FP"]), "MT": int(c["MT"]), "ML": int(c["ML"]), "Frag": int(c["Frag"]),
            "IDTP": int(i["IDTP"]), "IDFP": int(i["IDFP"]), "IDFN": int(i["IDFN"]),
            "GT_IDs": int(n["GT_IDs"]), "IDs": int(n["IDs"])}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    det_names = [write_detection_tracks(d) for d in ("grouped", "coco")
                 if os.path.isfile(os.path.join(CACHE, "detections", f"{d}__{SEQS[0]}.npz"))]
    uid_only = derive_uid_only()
    rows, combined = [], []
    for bench, rate in BENCH.items():
        folder = os.path.join(TRACKERS, f"{bench}-train")
        if not os.path.isdir(folder):
            continue
        trackers = sorted(t for t in os.listdir(folder)
                          if all(os.path.isfile(os.path.join(folder, t, "data", f"{s}.txt")) for s in SEQS))
        if not trackers:
            continue
        res, dataset = evaluate(bench, trackers)
        for trk in trackers:
            base = trk
            for suffix in ("_uidonly", "_offline"):
                base = base[:-len(suffix)] if base.endswith(suffix) else base
            run = os.path.join(RUNS, f"{base}__{rate}.json")
            runrec = json.load(open(run, encoding="utf-8")) if os.path.isfile(run) else {}
            tot = {"gt_ids": 0, "gt_ids_matched": 0, "gt_ids_never_matched": 0, "pred_ids": 0,
                   "pred_ids_matched": 0, "sum_ids_over_matched_gt": 0, "merged_ids": 0}
            for seq in SEQS:
                ic = id_counts(dataset, trk, seq)
                for k in tot:
                    tot[k] += ic[k]
                rr = runrec.get("sequences", {}).get(seq, {})
                rows.append({"rate": rate, "run": trk, "sequence": seq, **row_of(res[trk][seq]), **ic,
                             "cpu_ms_per_frame_tracker": rr.get("ms_per_frame_tracker"),
                             "cpu_ms_per_frame_identity": rr.get("ms_per_frame_identity")})
            comb = {"rate": rate, "run": trk, "sequence": "COMBINED", **row_of(res[trk]["COMBINED_SEQ"]), **tot,
                    "ids_per_gt": tot["sum_ids_over_matched_gt"] / tot["gt_ids_matched"] if tot["gt_ids_matched"] else None,
                    "cpu_ms_per_frame_tracker": runrec.get("ms_per_frame_tracker_all"),
                    "cpu_ms_per_frame_identity": runrec.get("ms_per_frame_identity_all")}
            rows.append(comb)
            combined.append(comb)
    keys = list(rows[0].keys())
    for name, data in (("per_sequence_metrics.csv", rows), ("combined_metrics.csv", combined)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(data)
    with open(os.path.join(OUT, "trackeval_versions.json"), "w", encoding="utf-8") as fh:
        json.dump({"trackeval": getattr(trackeval, "__version__", "git 12c8791"), "numpy": np.__version__,
                   "python": sys.version.split()[0], "numpy_alias_shim": ["float", "int", "bool"],
                   "detection_pseudo_trackers": det_names, "uid_only_variants": uid_only}, fh, indent=2)
    for c in combined:
        print(f"{c['rate']:<6} {c['run']:<44} HOTA {c['HOTA']:5.1f} DetA {c['DetA']:5.1f} AssA {c['AssA']:5.1f} "
              f"IDF1 {c['IDF1']:5.1f} IDSW {c['IDSW']:5d} MOTA {c['MOTA']:6.1f} ids/gt "
              f"{(c['ids_per_gt'] or 0):4.2f} merged {c['merged_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
