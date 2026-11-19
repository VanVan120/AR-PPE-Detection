"""Independent re-computation of MOTA / IDF1 / HOTA / ids-per-GT / merged-ids from raw MOT files.

Does NOT import trackeval. Applies the MOT17 rules: GT kept only if class == 1 and flag != 0;
tracker boxes matched (Hungarian on all GT rows, IoU >= 0.5) to classes 2, 7, 8, 12 are removed.
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
TRK = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge")
CSV = os.path.join(ROOT, "paper_materials", "revision_r1", "partD", "per_sequence_metrics.csv")
DISTRACTORS = (2, 7, 8, 12)
EPS = np.finfo(float).eps


def iou_xywh(a, b):
    """a: N x 4, b: M x 4 in x, y, w, h -> N x M IoU (own implementation)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None] - inter
    out = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    out[area_a <= 0, :] = 0
    out[:, area_b <= 0] = 0
    return out


def seq_len(split, seq):
    for line in open(os.path.join(GT, split, seq, "seqinfo.ini")):
        if line.startswith("seqLength="):
            return int(line.split("=")[1])


def load(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a


def preprocess(split, seq, trk_path):
    L = seq_len(split, seq)
    g = load(os.path.join(GT, split, seq, "gt", "gt.txt"))
    p = load(trk_path) if os.path.getsize(trk_path) else np.zeros((0, 10))
    frames = []
    removed = 0
    for t in range(1, L + 1):
        gt = g[g[:, 0] == t]
        pr = p[p[:, 0] == t]
        keep_p = np.ones(len(pr), bool)
        if len(gt) and len(pr):
            s = iou_xywh(gt[:, 2:6], pr[:, 2:6])
            m = s.copy()
            m[m < 0.5 - EPS] = 0
            r, c = linear_sum_assignment(-m)
            ok = m[r, c] > EPS
            r, c = r[ok], c[ok]
            dis = np.isin(gt[r, 7].astype(int), DISTRACTORS)
            keep_p[c[dis]] = False
            removed += int(dis.sum())
        pr = pr[keep_p]
        gt = gt[(gt[:, 7] == 1) & (gt[:, 6] != 0)]
        frames.append((gt[:, 1].astype(np.int64), gt[:, 2:6], pr[:, 1].astype(np.int64), pr[:, 2:6]))
    return frames, removed


def clear(frames):
    tp = fp = fn = idsw = 0
    last = {}          # gt id -> last matched tracker id (any time before)
    prev_step = {}     # gt id -> tracker id matched in the immediately previous timestep
    for gid, gb, pid, pb in frames:
        if len(gid) == 0:          # TrackEval keeps prev-timestep matches across empty frames
            fp += len(pid)
            continue
        if len(pid) == 0:
            fn += len(gid)
            continue
        s = iou_xywh(gb, pb)
        bonus = np.array([[1.0 if prev_step.get(int(gi)) == int(pj) else 0.0 for pj in pid] for gi in gid])
        sc = 1000 * bonus + s
        sc[s < 0.5 - EPS] = 0
        r, c = linear_sum_assignment(-sc)
        ok = sc[r, c] > EPS
        r, c = r[ok], c[ok]
        new_step = {}
        for i, j in zip(r, c):
            gi, pj = int(gid[i]), int(pid[j])
            if gi in last and last[gi] != pj:
                idsw += 1
            last[gi] = pj
            new_step[gi] = pj
        prev_step = new_step
        tp += len(r)
        fn += len(gid) - len(r)
        fp += len(pid) - len(r)
    mota = (tp - fp - idsw) / max(1, tp + fn)
    return dict(TP=tp, FP=fp, FN=fn, IDSW=idsw, MOTA=100 * mota,
                Re=100 * tp / max(1, tp + fn), Pr=100 * tp / max(1, tp + fp))


def identity(frames):
    gts = sorted({int(x) for f in frames for x in f[0]})
    prs = sorted({int(x) for f in frames for x in f[2]})
    gi = {g: i for i, g in enumerate(gts)}
    pj = {p: j for j, p in enumerate(prs)}
    pm = np.zeros((len(gts), len(prs)))
    n_g = n_p = 0
    for gid, gb, pid, pb in frames:
        n_g += len(gid)
        n_p += len(pid)
        if len(gid) and len(pid):
            s = iou_xywh(gb, pb)
            ii, jj = np.nonzero(s >= 0.5)
            for a, b in zip(ii, jj):
                pm[gi[int(gid[a])], pj[int(pid[b])]] += 1
    if pm.size:
        r, c = linear_sum_assignment(-pm)
        idtp = int(pm[r, c].sum())
    else:
        idtp = 0
    idfn, idfp = n_g - idtp, n_p - idtp
    return dict(IDTP=idtp, IDFP=idfp, IDFN=idfn, IDF1=100 * 2 * idtp / max(1, n_g + n_p),
                GT_IDs=len(gts), IDs=len(prs), GT_Dets=n_g, Dets=n_p)


def hota(frames):
    gts = sorted({int(x) for f in frames for x in f[0]})
    prs = sorted({int(x) for f in frames for x in f[2]})
    gi = {g: i for i, g in enumerate(gts)}
    pj = {p: j for j, p in enumerate(prs)}
    G, P = len(gts), len(prs)
    alphas = np.arange(0.05, 0.99, 0.05)
    pot = np.zeros((G, P))
    gcount = np.zeros((G, 1))
    pcount = np.zeros((1, P))
    sims = []
    for gid, gb, pid, pb in frames:
        gix = np.array([gi[int(x)] for x in gid], int)
        pix = np.array([pj[int(x)] for x in pid], int)
        s = iou_xywh(gb, pb)
        sims.append((gix, pix, s))
        if len(gix) and len(pix):
            denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            sim_iou = np.zeros_like(s)
            mk = denom > EPS
            sim_iou[mk] = s[mk] / denom[mk]
            pot[gix[:, None], pix[None, :]] += sim_iou
        gcount[gix] += 1
        pcount[0, pix] += 1
    gas = pot / (gcount + pcount - pot)
    TP = np.zeros(len(alphas)); FN = np.zeros(len(alphas)); FP = np.zeros(len(alphas))
    mc = [np.zeros((G, P)) for _ in alphas]
    for gix, pix, s in sims:
        if len(gix) == 0:
            FP += len(pix)
            continue
        if len(pix) == 0:
            FN += len(gix)
            continue
        sc = gas[gix[:, None], pix[None, :]] * s
        r, c = linear_sum_assignment(-sc)
        for a, al in enumerate(alphas):
            ok = s[r, c] >= al - EPS
            n = int(ok.sum())
            TP[a] += n; FN[a] += len(gix) - n; FP[a] += len(pix) - n
            if n:
                mc[a][gix[r[ok]], pix[c[ok]]] += 1
    AssA = np.zeros(len(alphas))
    for a in range(len(alphas)):
        m = mc[a]
        ass = m / np.maximum(1, gcount + pcount - m)
        AssA[a] = (m * ass).sum() / max(1, TP[a])
    DetA = TP / np.maximum(1, TP + FN + FP)
    H = np.sqrt(DetA * AssA)
    return dict(HOTA=100 * H.mean(), DetA=100 * DetA.mean(), AssA=100 * AssA.mean())


def idcounts(frames, threshold_first):
    gt_to, tr_to = {}, {}
    for gid, gb, pid, pb in frames:
        if len(gid) == 0 or len(pid) == 0:
            continue
        s = iou_xywh(gb, pb)
        m = s.copy()
        if threshold_first:
            m[m < 0.5 - EPS] = 0
        r, c = linear_sum_assignment(-m)
        for i, j in zip(r, c):
            if s[i, j] >= 0.5 - EPS:
                gt_to.setdefault(int(gid[i]), set()).add(int(pid[j]))
                tr_to.setdefault(int(pid[j]), set()).add(int(gid[i]))
    n_match = len(gt_to)
    tot = sum(len(v) for v in gt_to.values())
    return dict(gt_ids_matched=n_match, sum_ids=tot, ids_per_gt=tot / n_match if n_match else None,
                merged_ids=sum(1 for v in tr_to.values() if len(v) >= 2), pred_ids_matched=len(tr_to))


def csv_row(rate, run, seq):
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["rate"] == rate and r["run"] == run and r["sequence"] == seq:
            return r
    return None


def main():
    cases = [a.split(":") for a in sys.argv[1:]]
    for rate, run, seq in cases:
        split = "MOT17-train" if rate == "native" else "MOT17sub7-train"
        frames, removed = preprocess(split, seq, os.path.join(TRK, split, run, "data", f"{seq}.txt"))
        c, i, h = clear(frames), identity(frames), hota(frames)
        k1, k0 = idcounts(frames, threshold_first=False), idcounts(frames, threshold_first=True)
        ref = csv_row(rate, run, seq)
        print(f"=== {rate} {run} {seq}  (tracker boxes removed as distractor matches: {removed})")
        for key, mine in [("MOTA", c["MOTA"]), ("CLR_TP", c["TP"]), ("CLR_FP", c["FP"]), ("CLR_FN", c["FN"]),
                          ("IDSW", c["IDSW"]), ("CLR_Re", c["Re"]), ("CLR_Pr", c["Pr"]),
                          ("IDF1", i["IDF1"]), ("IDTP", i["IDTP"]), ("IDFP", i["IDFP"]), ("IDFN", i["IDFN"]),
                          ("GT_IDs", i["GT_IDs"]), ("IDs", i["IDs"]),
                          ("HOTA", h["HOTA"]), ("DetA", h["DetA"]), ("AssA", h["AssA"]),
                          ("gt_ids_matched", k1["gt_ids_matched"]), ("sum_ids_over_matched_gt", k1["sum_ids"]),
                          ("ids_per_gt", k1["ids_per_gt"]), ("merged_ids", k1["merged_ids"]),
                          ("pred_ids_matched", k1["pred_ids_matched"])]:
            rv = ref[key] if ref else None
            try:
                diff = abs(float(rv) - float(mine))
            except (TypeError, ValueError):
                diff = None
            flag = "OK" if diff is not None and diff < 1e-6 else "DIFF"
            print(f"  {key:<24} mine={mine!s:<22} csv={rv!s:<22} {flag}")
        print(f"  [threshold-first Hungarian variant] ids_per_gt={k0['ids_per_gt']} merged_ids={k0['merged_ids']} "
              f"gt_ids_matched={k0['gt_ids_matched']} sum_ids={k0['sum_ids']} pred_ids_matched={k0['pred_ids_matched']}")


if __name__ == "__main__":
    main()
