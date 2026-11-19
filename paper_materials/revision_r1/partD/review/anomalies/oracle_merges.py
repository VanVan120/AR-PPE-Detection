import os, sys, numpy as np, collections
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
run = sys.argv[1] if len(sys.argv) > 1 else "oracle__workid_phase2_uidonly"
split = sys.argv[2] if len(sys.argv) > 2 else "MOT17-train"
verbose_seq = sys.argv[3] if len(sys.argv) > 3 else None
tot = collections.Counter()
for s in SEQS:
    gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    if split != "MOT17-train":
        import json
        fm = json.load(open(os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge", "MOT17sub7-train_frame_map.json")))[s]
        new_of = {f: i + 1 for i, f in enumerate(fm["kept_frames"])}
        gt = gt[np.isin(gt[:, 0], fm["kept_frames"])]
        gt[:, 0] = [new_of[int(f)] for f in gt[:, 0]]
    tr = np.loadtxt(os.path.join(TRK, split, run, "data", f"{s}.txt"), delimiter=",", ndmin=2)
    # exact match by frame + box (oracle boxes are GT boxes rounded to 0.01)
    key = {}
    for r in gt:
        key.setdefault(int(r[0]), []).append(r)
    uid2gt = collections.defaultdict(list)   # uid -> list of (frame, gtid, h, cx, cy, vis)
    unmatched = 0
    for r in tr:
        f = int(r[0]); cands = key.get(f, [])
        best, bi = 0, None
        for g in cands:
            if abs(g[2] - r[2]) < 0.02 and abs(g[3] - r[3]) < 0.02 and abs(g[4] - r[4]) < 0.02 and abs(g[5] - r[5]) < 0.02:
                bi = g; break
        if bi is None:
            unmatched += 1; continue
        uid2gt[int(r[1])].append((f, int(bi[1]), bi[5], bi[2] + bi[4] / 2, bi[3] + bi[5] / 2, bi[8]))
    merged = 0; pairs = []
    for uid, rows in uid2gt.items():
        gids = collections.OrderedDict()
        for f, g, h, cx, cy, v in sorted(rows):
            d = gids.setdefault(g, {"first": f, "last": f, "n": 0, "h": [], "fbox": (cx, cy), "lbox": (cx, cy), "fh": h, "lh": h})
            d["last"] = f; d["n"] += 1; d["h"].append(h); d["lbox"] = (cx, cy); d["lh"] = h
        if len(gids) >= 2:
            merged += 1
            seq = list(gids.items())
            for (ga, a), (gb, b) in zip(seq, seq[1:]):
                gap = b["first"] - a["last"]
                dist = float(np.hypot(b["fbox"][0] - a["lbox"][0], b["fbox"][1] - a["lbox"][1]))
                pairs.append((uid, ga, gb, a["first"], a["last"], b["first"], b["last"], gap, dist, a["lh"], b["fh"], np.median(a["h"]), np.median(b["h"])))
    small = sum(1 for p in pairs if min(p[11], p[12]) < 48)
    tot["merged"] += merged; tot["pairs"] += len(pairs); tot["unmatched"] += unmatched; tot["small"] += small
    gaps = np.array([p[7] for p in pairs]) if pairs else np.array([0])
    dists = np.array([p[8] / max(1e-6, 0.5 * (p[9] + p[10])) for p in pairs]) if pairs else np.array([0])
    tot["overlap_in_time"] += int((gaps <= 0).sum())
    print(f"{s}: uids {len(uid2gt)}, merged uids {merged}, consecutive merge pairs {len(pairs)}, rows unmatched {unmatched}, "
          f"pairs with a median height<48: {small}; gap frames median {np.median(gaps):.0f} [min {gaps.min()}, max {gaps.max()}]; "
          f"centre distance / body-height median {np.median(dists):.2f}; pairs within 0.5 body height & <=90 frames: {int(((dists<=1.0)&(gaps<=90)).sum())}")
    if verbose_seq and s == verbose_seq:
        for p in pairs:
            print("   uid %d: GT %d [%d-%d] -> GT %d [%d-%d]; gap %d fr; jump %.0f px (%.2f body h); h last/first %.0f/%.0f; median h %.0f/%.0f" %
                  (p[0], p[1], p[3], p[4], p[2], p[5], p[6], p[7], p[8], p[8] / (0.5 * (p[9] + p[10])), p[9], p[10], p[11], p[12]))
print("TOTAL", dict(tot))
