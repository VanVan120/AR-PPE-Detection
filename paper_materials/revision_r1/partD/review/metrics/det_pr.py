"""Direct count of detector P/R at conf > 0.35, IoU 0.5 from the detection cache (npz), MOT17 rules."""
import csv
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections")
TRK = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge", "MOT17-train")
CSV = os.path.join(ROOT, "paper_materials", "revision_r1", "partD", "per_sequence_metrics.csv")
EPS = np.finfo(float).eps


def iou_xyxy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter)


def hungarian_pairs(s):
    m = s.copy()
    m[m < 0.5 - EPS] = 0
    r, c = linear_sum_assignment(-m)
    ok = m[r, c] > EPS
    return r[ok], c[ok]


def greedy_pairs(s):
    pairs, used_r, used_c = [], set(), set()
    for k in np.argsort(-s, axis=None):
        i, j = np.unravel_index(k, s.shape)
        if s[i, j] < 0.5:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j); pairs.append((i, j))
    return pairs


def main(det, seq, round2=False):
    z = np.load(os.path.join(CACHE, f"{det}__{seq}.npz"))
    frames, off, dets = z["frames"], z["off"], z["dets"]
    g = np.loadtxt(os.path.join(GT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
    L = int(g[:, 0].max())
    tp = fp = fn = removed = 0
    tp_g = 0
    n_kept = 0
    for i, f in enumerate(frames):
        d = dets[off[i]:off[i + 1]]
        d = d[d[:, 4] > 0.35]
        n_kept += len(d)
        box = d[:, :4].astype(np.float64)
        if round2:                                   # as written to the MOT text file (x, y, w, h at 2 dp)
            x, y = np.round(box[:, 0], 2), np.round(box[:, 1], 2)
            w, h = np.round(box[:, 2] - box[:, 0], 2), np.round(box[:, 3] - box[:, 1], 2)
            box = np.stack([x, y, x + w, y + h], 1)
        gt = g[g[:, 0] == f]
        gtb = np.stack([gt[:, 2], gt[:, 3], gt[:, 2] + gt[:, 4], gt[:, 3] + gt[:, 5]], 1) if len(gt) else np.zeros((0, 4))
        keep = np.ones(len(box), bool)
        if len(gt) and len(box):
            r, c = hungarian_pairs(iou_xyxy(gtb, box))
            dis = np.isin(gt[r, 7].astype(int), (2, 7, 8, 12))
            keep[c[dis]] = False
            removed += int(dis.sum())
        box = box[keep]
        ped = (gt[:, 7] == 1) & (gt[:, 6] != 0)
        pb = gtb[ped]
        if len(pb) and len(box):
            s = iou_xyxy(pb, box)
            r, c = hungarian_pairs(s)
            n = len(r)
            tp_g += len(greedy_pairs(s))
        else:
            n = 0
        tp += n
        fp += len(box) - n
        fn += len(pb) - n
    print(f"{det} {seq}: frames in cache {len(frames)} (gt frames {L}); boxes conf>0.35: {n_kept}; "
          f"removed as distractor matches: {removed}")
    print(f"  direct (Hungarian): TP {tp} FP {fp} FN {fn}  P {100 * tp / (tp + fp):.4f} R {100 * tp / (tp + fn):.4f}"
          f"   | greedy TP {tp_g}")
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["rate"] == "native" and r["run"] == f"{det}__detections_conf035" and r["sequence"] == seq:
            print(f"  csv: TP {r['CLR_TP']} FP {r['CLR_FP']} FN {r['CLR_FN']}  P {float(r['CLR_Pr']):.4f} R {float(r['CLR_Re']):.4f}")
    nlines = sum(1 for _ in open(os.path.join(TRK, f"{det}__detections_conf035", "data", f"{seq}.txt")))
    print(f"  lines in pseudo-tracker file: {nlines}")


if __name__ == "__main__":
    rnd = "--round2" in sys.argv
    for a in [x for x in sys.argv[1:] if not x.startswith("--")]:
        det, seq = a.split(":")
        main(det, seq, rnd)
