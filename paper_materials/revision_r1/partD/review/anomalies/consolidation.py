import os, collections, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
tot = collections.Counter()
for s in SEQS:
    live = np.loadtxt(os.path.join(TRK, "MOT17-train", "oracle__workid_phase2", "data", f"{s}.txt"), delimiter=",", ndmin=2)
    off = np.loadtxt(os.path.join(TRK, "MOT17-train", "oracle__workid_phase2_offline", "data", f"{s}.txt"), delimiter=",", ndmin=2)
    assert len(live) == len(off) and np.allclose(live[:, [0, 2, 3, 4, 5]], off[:, [0, 2, 3, 4, 5]])
    gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2); gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    key = {(int(r[0]), round(r[2], 2), round(r[3], 2)): int(r[1]) for r in gt}
    dom = collections.defaultdict(collections.Counter)
    groups = collections.defaultdict(set)
    for a, b in zip(live, off):
        if a[1] >= 1_000_000:
            continue
        g = key.get((int(a[0]), round(a[2], 2), round(a[3], 2)), -1)
        dom[int(a[1])][g] += 1
        groups[int(b[1])].add(int(a[1]))
    merges = ok = 0
    for oid, lids in groups.items():
        if len(lids) < 2:
            continue
        lids = sorted(lids)
        # each consolidation merge folds one live uid into another: count len-1 merges per group
        d = [dom[l].most_common(1)[0][0] for l in lids]
        for x in d[1:]:
            merges += 1; ok += int(x == d[0])
    tot["merges"] += merges; tot["same_gt"] += ok
    print(s, "consolidation merges", merges, "joining the same dominant GT id", ok)
print("TOTAL", dict(tot))
