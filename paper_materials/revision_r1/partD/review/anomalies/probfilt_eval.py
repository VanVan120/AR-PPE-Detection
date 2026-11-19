"""Fair baseline for the '_uidonly' Work ID variants: ByteTrack output with the first two appearances
of every track dropped (what probation removes), scored with the same TrackEval setup. Output
goes to the scratchpad only."""
import os, sys, collections
import numpy as np
for _a, _b in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _b)
import trackeval
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
SRC = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge")
DST = os.path.join(HERE, "trackers")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
split = sys.argv[1] if len(sys.argv) > 1 else "MOT17-train"
bench = split.split("-")[0]
runs = []
for base in ("grouped__bytetrack_phase2", "coco__bytetrack_phase2", "oracle__bytetrack_phase2"):
    name = base + "_probfilt"
    os.makedirs(os.path.join(DST, split, name, "data"), exist_ok=True)
    for s in SEQS:
        lines = open(os.path.join(SRC, split, base, "data", f"{s}.txt"), encoding="utf-8").read().splitlines()
        seen = collections.Counter()
        with open(os.path.join(DST, split, name, "data", f"{s}.txt"), "w", encoding="utf-8") as fo:
            for ln in sorted(lines, key=lambda l: int(l.split(",")[0])):
                tid = int(ln.split(",")[1])
                seen[tid] += 1
                if seen[tid] > 2:
                    fo.write(ln + "\n")
    runs.append(name)
# also copy the reference runs so everything is scored in one TrackEval call
import shutil
for ref in ("grouped__bytetrack_phase2", "grouped__workid_phase2_uidonly", "coco__bytetrack_phase2",
            "coco__workid_phase2_uidonly", "oracle__bytetrack_phase2", "oracle__workid_phase2_uidonly"):
    d = os.path.join(DST, split, ref, "data")
    if not os.path.isdir(d):
        shutil.copytree(os.path.join(SRC, split, ref, "data"), d)
    runs.append(ref)
ec = trackeval.Evaluator.get_default_eval_config()
ec.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
           "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "BREAK_ON_ERROR": True,
           "DISPLAY_LESS_PROGRESS": True})
dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dc.update({"GT_FOLDER": GT_TE, "TRACKERS_FOLDER": DST, "OUTPUT_FOLDER": os.path.join(HERE, "te_out"),
           "TRACKERS_TO_EVAL": runs, "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": bench,
           "SPLIT_TO_EVAL": "train", "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data",
           "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
ds = trackeval.datasets.MotChallenge2DBox(dc)
q = {"PRINT_CONFIG": False}
res, _ = trackeval.Evaluator(ec).evaluate([ds], [trackeval.metrics.HOTA(q), trackeval.metrics.CLEAR({**q, "THRESHOLD": 0.5}),
                                                trackeval.metrics.Identity({**q, "THRESHOLD": 0.5})])
res = res["MotChallenge2DBox"]
def idc(trk):
    tot_pairs = tot_m = merged = 0
    for s in SEQS:
        data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(trk, s), "pedestrian")
        gt_to, tr_to = {}, {}
        for t in range(data["num_timesteps"]):
            g, p, sim = data["gt_ids"][t], data["tracker_ids"][t], data["similarity_scores"][t]
            if len(g) == 0 or len(p) == 0:
                continue
            r, c = linear_sum_assignment(-sim)
            for i, j in zip(r, c):
                if sim[i, j] >= 0.5 - np.finfo("float").eps:
                    gt_to.setdefault(int(g[i]), set()).add(int(p[j])); tr_to.setdefault(int(p[j]), set()).add(int(g[i]))
        tot_pairs += sum(len(v) for v in gt_to.values()); tot_m += len(gt_to); merged += sum(1 for v in tr_to.values() if len(v) >= 2)
    return tot_pairs / tot_m, merged
for trk in runs:
    r = res[trk]["COMBINED_SEQ"]["pedestrian"]
    h, c, i = r["HOTA"], r["CLEAR"], r["Identity"]
    ipg, mg = idc(trk)
    print(f"{split} {trk:<40} HOTA {100*np.mean(h['HOTA']):5.2f} DetA {100*np.mean(h['DetA']):5.2f} AssA {100*np.mean(h['AssA']):5.2f} "
          f"IDF1 {100*i['IDF1']:5.2f} IDSW {int(c['IDSW']):4d} MOTA {100*c['MOTA']:5.2f} TP {int(c['CLR_TP'])} ids/gt {ipg:.3f} merged {mg}")
