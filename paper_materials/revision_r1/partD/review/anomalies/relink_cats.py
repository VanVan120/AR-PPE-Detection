import os, json, collections, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
ev = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_relinks.json")))
gts = {}
for s in SEQS:
    g = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
    gts[s] = g[(g[:, 7] == 1) & (g[:, 6] == 1)]
cats = collections.Counter(); rows = []
for e in ev:
    if not e["relink"] or e["correct"]:
        continue
    g = gts[e["seq"]]
    A = g[g[:, 1] == e["prev_gt"]]; B = g[g[:, 1] == e["gt"]]
    a_last = A[A[:, 0].argmax()]; b_first = B[B[:, 0].argmin()]
    gap = int(b_first[0] - a_last[0])
    ca = (a_last[2] + a_last[4] / 2, a_last[3] + a_last[5] / 2); cb = (b_first[2] + b_first[4] / 2, b_first[3] + b_first[5] / 2)
    hb = 0.5 * (a_last[5] + b_first[5])
    jump = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]) / hb)
    overlap = gap <= 0
    if overlap:
        c = "A and B co-exist in GT (different people for sure)"
    elif gap <= 30 and jump <= 0.5:
        c = "short gap, same spot (possible GT id fragmentation)"
    elif jump >= 2.0:
        c = "jump >= 2 body heights (implausible same person)"
    else:
        c = "other"
    cats[c] += 1
    rows.append((e["seq"], e["source"], e["sim"], gap, round(jump, 2), round(e["vis"], 2), e["clipped"], c))
print(len(rows), "wrong relinks")
for k, v in cats.most_common():
    print(f"  {v:4d}  {k}")
print("by source:", collections.Counter((r[1], r[7][:20]) for r in rows))
