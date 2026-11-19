import os, numpy as np
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
for det in ("grouped", "coco"):
    print("====", det, " per sequence: all | vis>=0.5&h>=100 | vis<0.25 share of GT | median GT h")
    for s in SEQS:
        gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
        z = np.load(os.path.join(CACHE, f"{det}__{s}.npz"))
        frames, off, dets = z["frames"], z["off"], z["dets"]
        P = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
        a = [0, 0]; b = [0, 0]
        for i, f in enumerate(frames):
            d = dets[off[i]:off[i+1]]; d = d[d[:, 4] > 0.35]
            g = P[P[:, 0] == f]
            if len(g) == 0: continue
            gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
            m = np.zeros(len(g), bool)
            if len(d):
                M = iou(gb, d[:, :4]); M[M < 0.5] = 0
                r, c = linear_sum_assignment(-M); ok = M[r, c] > 0; m[r[ok]] = True
            sel = (g[:, 8] >= 0.5) & (g[:, 5] >= 100)
            a[0] += m.sum(); a[1] += len(g); b[0] += m[sel].sum(); b[1] += sel.sum()
        print(f"  {s}: {100*a[0]/a[1]:5.1f}% | {100*b[0]/max(b[1],1):5.1f}% (n={b[1]}) | {100*(P[:,8]<0.25).mean():4.1f}% | {np.median(P[:,5]):.0f}")
