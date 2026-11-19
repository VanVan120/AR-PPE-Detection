"""mot17_eval.id_counts runs Hungarian on raw IoU and then drops pairs < 0.5; CLEAR/Identity zero
sub-threshold IoU before the assignment. Compare the two on a few runs."""
import os, numpy as np
for _a, _b in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _b)
import trackeval
from scipy.optimize import linear_sum_assignment
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
TRK = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
runs = ["coco__bytetrack_phase2", "coco__botsort_reid", "coco__workid_phase2_uidonly", "grouped__bytetrack_phase2", "oracle__workid_phase2_uidonly"]
dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dc.update({"GT_FOLDER": GT_TE, "TRACKERS_FOLDER": TRK, "OUTPUT_FOLDER": os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_out2"),
           "TRACKERS_TO_EVAL": runs, "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
           "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
ds = trackeval.datasets.MotChallenge2DBox(dc)
for trk in runs:
    out = {}
    for mode in ("raw", "thresholded"):
        pairs = matched = merged = 0
        for s in SEQS:
            data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(trk, s), "pedestrian")
            gt_to, tr_to = {}, {}
            for t in range(data["num_timesteps"]):
                g, p, sim = data["gt_ids"][t], data["tracker_ids"][t], data["similarity_scores"][t]
                if len(g) == 0 or len(p) == 0:
                    continue
                S = sim.copy()
                if mode == "thresholded":
                    S[S < 0.5 - np.finfo("float").eps] = 0
                r, c = linear_sum_assignment(-S)
                for i, j in zip(r, c):
                    if sim[i, j] >= 0.5 - np.finfo("float").eps:
                        gt_to.setdefault(int(g[i]), set()).add(int(p[j])); tr_to.setdefault(int(p[j]), set()).add(int(g[i]))
            pairs += sum(len(v) for v in gt_to.values()); matched += len(gt_to); merged += sum(1 for v in tr_to.values() if len(v) >= 2)
        out[mode] = (round(pairs / matched, 3), merged, matched)
    print(trk, "raw-IoU Hungarian (as mot17_eval):", out["raw"], " thresholded first (as CLEAR):", out["thresholded"])
