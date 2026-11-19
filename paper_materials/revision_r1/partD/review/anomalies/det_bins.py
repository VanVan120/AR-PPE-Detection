import os, numpy as np
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
hb = [0, 48, 75, 100, 150, 200, 300, 5000]
vb = [-0.01, 0.1, 0.25, 0.5, 0.75, 1.01]
res = {}
for det in ("grouped", "coco"):
    H = {k: [0, 0] for k in range(len(hb) - 1)}
    V = {k: [0, 0] for k in range(len(vb) - 1)}
    HV = {}
    S = {}
    for s in SEQS:
        gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
        z = np.load(os.path.join(CACHE, f"{det}__{s}.npz"))
        frames, off, dets = z["frames"], z["off"], z["dets"]
        gfr = {}
        for r in gt:
            gfr.setdefault(int(r[0]), []).append(r)
        st = [0, 0]
        for i, f in enumerate(frames):
            d = dets[off[i]:off[i+1]]; d = d[d[:, 4] > 0.35]
            g = np.array(gfr.get(int(f), []))
            if len(g) == 0: continue
            ped = (g[:, 7] == 1) & (g[:, 6] == 1)
            gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
            M = iou(gb, d[:, :4]) if len(d) else np.zeros((len(gb), 0))
            Ms = M.copy(); Ms[Ms < 0.5 - 1e-9] = 0
            if Ms.size:
                r_, c_ = linear_sum_assignment(-Ms); ok = Ms[r_, c_] > 0; r_, c_ = r_[ok], c_[ok]
            else:
                r_ = c_ = np.array([], int)
            dist = np.isin(g[r_, 7], [2, 7, 8, 12])
            keep_d = np.ones(len(d), bool); keep_d[c_[dist]] = False
            pidx = np.where(ped)[0]; didx = np.where(keep_d)[0]
            matched = np.zeros(len(pidx), bool)
            if len(pidx) and len(didx):
                M2 = M[np.ix_(pidx, didx)].copy(); M2[M2 < 0.5 - 1e-9] = 0
                a, b = linear_sum_assignment(-M2); okk = M2[a, b] > 0
                matched[a[okk]] = True
            for k, gi in enumerate(pidx):
                h = g[gi, 5]; v = g[gi, 8]
                hi = np.searchsorted(hb, h, side="right") - 1; vi = np.searchsorted(vb, v, side="right") - 1
                hi = min(hi, len(hb) - 2); vi = min(max(vi, 0), len(vb) - 2)
                H[hi][0] += matched[k]; H[hi][1] += 1
                V[vi][0] += matched[k]; V[vi][1] += 1
                key = (hi, vi >= 3)
                HV.setdefault(key, [0, 0]); HV[key][0] += matched[k]; HV[key][1] += 1
                st[0] += matched[k]; st[1] += 1
        S[s] = st
    print("=====", det)
    for k in H: print(f"  h [{hb[k]},{hb[k+1]}): n={H[k][1]:6d} recall={100*H[k][0]/max(H[k][1],1):5.1f}")
    for k in V: print(f"  vis [{vb[k]},{vb[k+1]}): n={V[k][1]:6d} recall={100*V[k][0]/max(V[k][1],1):5.1f}")
    print("  visible>=0.5 only, by height:")
    for k in sorted(HV):
        if k[1]: print(f"   h [{hb[k[0]]},{hb[k[0]+1]}): n={HV[k][1]:6d} recall={100*HV[k][0]/max(HV[k][1],1):5.1f}")
    for s in S: print("  ", s, S[s], f"{100*S[s][0]/S[s][1]:.1f}")
