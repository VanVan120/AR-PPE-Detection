import os, numpy as np
from scipy.optimize import linear_sum_assignment
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

def iou(a, b):
    # a: N x4 xyxy, b: M x4
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter)

hbins = [0, 48, 75, 100, 150, 200, 300, 2000]
vbins = [0, 0.1, 0.25, 0.5, 0.75, 1.01]
tot = {}
for det in ("grouped", "coco"):
    print("=" * 30, det)
    allconf = []
    allcls = set()
    for s in SEQS:
        z = np.load(os.path.join(CACHE, f"{det}__{s}.npz"))
        d = z["dets"]
        allconf.append(d[:, 4]); allcls |= set(np.unique(d[:, 5]).tolist())
        h = d[:, 3] - d[:, 1]; w = d[:, 2] - d[:, 0]
        print(s, "boxes", len(d), "frames", len(z["frames"]), "frames range", z["frames"].min(), z["frames"].max(),
              "conf>0.35:", int((d[:, 4] > 0.35).sum()), "median h(>0.35)", float(np.median(h[d[:, 4] > 0.35])) if (d[:,4]>0.35).any() else None,
              "min conf", float(d[:, 4].min()), "max per frame", int(np.diff(z["off"]).max()))
    c = np.concatenate(allconf)
    print("classes:", allcls, "conf quantiles", np.quantile(c, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))
    print("conf hist", np.histogram(c, bins=[0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0])

# Recall by GT height and visibility at conf>0.35, IoU>=0.5, against class1 flag1 GT; also recall at thresholds
for thr in (0.25, 0.3, 0.35, 0.5):
    for det in ("grouped", "coco"):
        tp = fn = fp = 0
        for s in SEQS:
            gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
            z = np.load(os.path.join(CACHE, f"{det}__{s}.npz"))
            frames, off, dets = z["frames"], z["off"], z["dets"]
            gfr = {}
            for r in gt:
                gfr.setdefault(int(r[0]), []).append(r)
            for i, f in enumerate(frames):
                d = dets[off[i]:off[i+1]]; d = d[d[:, 4] > thr]
                g = np.array(gfr.get(int(f), []))
                if len(g) == 0:
                    fp += len(d); continue
                ped = (g[:, 7] == 1) & (g[:, 6] == 1)
                gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
                M = iou(gb, d[:, :4]) if len(d) else np.zeros((len(gb), 0))
                Ms = M.copy(); Ms[Ms < 0.5 - 1e-9] = 0
                if Ms.size:
                    r_, c_ = linear_sum_assignment(-Ms)
                    ok = Ms[r_, c_] > 0
                    r_, c_ = r_[ok], c_[ok]
                else:
                    r_ = c_ = np.array([], int)
                # remove dets matched to distractor classes (2,7,8,12)
                dist = np.isin(g[r_, 7], [2, 7, 8, 12])
                keep_d = np.ones(len(d), bool); keep_d[c_[dist]] = False
                # now match remaining dets to pedestrians
                pidx = np.where(ped)[0]; didx = np.where(keep_d)[0]
                M2 = M[np.ix_(pidx, didx)] if len(pidx) and len(didx) else np.zeros((len(pidx), len(didx)))
                M2s = M2.copy(); M2s[M2s < 0.5 - 1e-9] = 0
                if M2s.size:
                    a, b = linear_sum_assignment(-M2s); okk = M2s[a, b] > 0; ntp = int(okk.sum())
                else:
                    ntp = 0
                tp += ntp; fn += len(pidx) - ntp; fp += len(didx) - ntp
        print(f"thr {thr} {det}: TP {tp} FN {fn} FP {fp} R {100*tp/(tp+fn):.2f} P {100*tp/(tp+fp):.2f}")
