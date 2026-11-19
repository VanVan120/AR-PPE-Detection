"""Read-only structural validation of Part D tracker outputs against the detection cache / GT.

Writes nothing outside this scratch folder.
"""
import configparser
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TR = os.path.join(CACHE, "trackers", "mot_challenge")
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
RUNS = os.path.join(ROOT, "paper_materials", "revision_r1", "partD", "runs")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
FM = json.load(open(os.path.join(GT_TE, "MOT17sub7-train_frame_map.json"), encoding="utf-8"))


def seqinfo(seq):
    c = configparser.ConfigParser()
    c.optionxform = str
    c.read(os.path.join(MOT, seq, "seqinfo.ini"))
    s = c["Sequence"]
    return float(s["frameRate"]), int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"])


_cache = {}


def cache(det, seq):
    k = (det, seq)
    if k not in _cache:
        z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
        frames, off, dets, feats = z["frames"], z["off"], z["dets"], z["feats"]
        d = {int(f): (dets[off[i]:off[i + 1]], feats[off[i]:off[i + 1]]) for i, f in enumerate(frames)}
        _cache[k] = (d, frames, off, dets, feats)
    return _cache[k]


_gt = {}


def gt_oracle(seq):
    if seq not in _gt:
        g = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
        keep = (g[:, 7] == 1) & (g[:, 6] == 1)
        out = defaultdict(list)
        for r in g[keep]:
            x, y, w, h = r[2:6]
            out[int(r[0])].append([x, y, x + w, y + h, 0.9, 0.0])
        _gt[seq] = {f: np.array(v, np.float64) for f, v in out.items()}
    return _gt[seq]


def load_trk(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a


def iou_xywh(a, B):
    """a: (4,) xywh, B: (N,4) xyxy -> (N,) IoU"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    ix1 = np.maximum(ax1, B[:, 0]); iy1 = np.maximum(ay1, B[:, 1])
    ix2 = np.minimum(ax2, B[:, 2]); iy2 = np.minimum(ay2, B[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = a[2] * a[3] + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def parse_run(name):
    """-> det, tracker, feed, variant"""
    det, rest = name.split("__", 1)
    return det, rest


report = {}

# ------------------------------------------------------------------ (A) detection cache sanity
print("== (A) detection cache sanity")
for det, pcls in (("grouped", 3), ("coco", 0)):
    for seq in SEQS:
        fps, n, W, H = seqinfo(seq)
        d, frames, off, dets, feats = cache(det, seq)
        assert list(frames) == list(range(1, n + 1)), (det, seq, "frames not 1..N")
        assert off[0] == 0 and off[-1] == len(dets) == len(feats)
        cls = np.unique(dets[:, 5])
        conf = dets[:, 4]
        per = np.diff(off)
        # order within frame descending conf
        unsorted = sum(1 for i in range(len(frames)) if np.any(np.diff(dets[off[i]:off[i + 1], 4]) > 0))
        norms = np.linalg.norm(feats, axis=1)
        oob = np.sum((dets[:, 0] < -1) | (dets[:, 1] < -1) | (dets[:, 2] > W + 1) | (dets[:, 3] > H + 1))
        print(f"  {det:7s} {seq}: frames {len(frames)}, boxes {len(dets)}, cls {cls.tolist()} (want {pcls}), "
              f"conf min {conf.min():.4f} max {conf.max():.4f} (>0.10: {bool((conf > 0.10).all())}), "
              f"max/frame {per.max()}, frames unsorted {unsorted}, feat dim {feats.shape[1]}, "
              f"zero-norm feats {int((norms < 1e-12).sum())}, nan {int(np.isnan(feats).sum())}, oob {int(oob)}")

# ------------------------------------------------------------------ (B) boxes_in vs cache counts
print("\n== (B) boxes_in recorded vs recomputed")
bad_b = 0
for fn in sorted(os.listdir(RUNS)):
    r = json.load(open(os.path.join(RUNS, fn), encoding="utf-8"))
    det, rate, mc = r["detections"], r["rate"], r["min_conf_fed"]
    for seq, s in r["sequences"].items():
        frames = list(range(1, s["frames"] * s["step"] + 1, s["step"])) if rate == "native" else FM[seq]["kept_frames"]
        if rate == "native":
            frames = list(range(1, seqinfo(seq)[1] + 1))
        assert len(frames) == s["frames"], (fn, seq)
        if det == "oracle":
            g = gt_oracle(seq)
            n = sum(len(g.get(f, [])) for f in frames)
        else:
            d = cache(det, seq)[0]
            n = sum(int((d[f][0][:, 4] > mc).sum()) if mc is not None else len(d[f][0]) for f in frames)
        if n != s["boxes_in"]:
            bad_b += 1
            print(f"  MISMATCH {fn} {seq}: recorded {s['boxes_in']} recomputed {n}")
print(f"  mismatches: {bad_b}")

# ------------------------------------------------------------------ (C) phase2/oracle outputs are input boxes
print("\n== (C) phase2 output lines vs fed boxes (exact, 2-decimal)")


def fed_boxes(det, seq, f, min_conf):
    if det == "oracle":
        b = gt_oracle(seq).get(f)
        return np.zeros((0, 6)) if b is None else b
    b = cache(det, seq)[0][f][0].astype(np.float64)
    if min_conf is not None:
        b = b[b[:, 4] > min_conf]
    return b


def fmt(v):
    return float(f"{v:.2f}")


for split in ("MOT17-train", "MOT17sub7-train"):
    for run in sorted(os.listdir(os.path.join(TR, split))):
        if "phase2" not in run:
            continue
        det = run.split("__")[0]
        feedall = "__feedall" in run
        min_conf = None if (det == "oracle" or feedall) else 0.35
        tot = unmatched = dup_id = dup_box = 0
        maxframe_bad = 0
        for seq in SEQS:
            a = load_trk(os.path.join(TR, split, run, "data", f"{seq}.txt"))
            if split == "MOT17-train":
                fmap = {f: f for f in range(1, seqinfo(seq)[1] + 1)}
            else:
                fmap = {i + 1: f for i, f in enumerate(FM[seq]["kept_frames"])}
            if len(a) and a[:, 0].max() > len(fmap):
                maxframe_bad += 1
            byf = defaultdict(list)
            for row in a:
                byf[int(row[0])].append(row)
            for nf, rows in byf.items():
                f = fmap[nf]
                fb = fed_boxes(det, seq, f, min_conf)
                cand = np.array([[fmt(x1), fmt(y1), fmt(x2 - x1), fmt(y2 - y1)] for x1, y1, x2, y2, *_ in fb]) \
                    if len(fb) else np.zeros((0, 4))
                ids = [int(r[1]) for r in rows]
                if len(ids) != len(set(ids)):
                    dup_id += 1
                used = set()
                for r in rows:
                    tot += 1
                    if len(cand) == 0:
                        unmatched += 1
                        continue
                    dd = np.abs(cand - r[2:6]).max(axis=1)
                    hits = np.where(dd <= 0.0101)[0]
                    hits = [h for h in hits if h not in used]
                    if not hits:
                        unmatched += 1
                    else:
                        used.add(hits[0])
        print(f"  {split:16s} {run:45s} lines {tot:7d} unmatched {unmatched} frames-with-dup-ids {dup_id} "
              f"seqs-with-frame>K {maxframe_bad}")

# ------------------------------------------------------------------ (D) external trackers: IoU to cached dets
print("\n== (D) BoT-SORT / boxmot output boxes: best IoU with a cached det in the same frame")
for run in sorted(os.listdir(os.path.join(TR, "MOT17-train"))):
    if not any(k in run for k in ("botsort", "strongsort", "deepocsort")):
        continue
    det = run.split("__")[0]
    ious, dupf, n_lines, maxf_bad, min_id = [], 0, 0, 0, 10 ** 9
    best_conf = []
    for seq in SEQS:
        n = seqinfo(seq)[1]
        a = load_trk(os.path.join(TR, "MOT17-train", run, "data", f"{seq}.txt"))
        n_lines += len(a)
        if len(a) and a[:, 0].max() > n:
            maxf_bad += 1
        if len(a):
            min_id = min(min_id, int(a[:, 1].min()))
        d = cache(det, seq)[0]
        byf = defaultdict(list)
        for row in a:
            byf[int(row[0])].append(row)
        for f, rows in byf.items():
            ids = [int(r[1]) for r in rows]
            if len(ids) != len(set(ids)):
                dupf += 1
            b = d[f][0].astype(np.float64)
            for r in rows:
                if len(b) == 0:
                    ious.append(0.0); best_conf.append(0.0)
                    continue
                iu = iou_xywh(r[2:6], b[:, :4])
                j = int(np.argmax(iu))
                ious.append(float(iu[j])); best_conf.append(float(b[j, 4]))
    ious = np.array(ious); best_conf = np.array(best_conf)
    print(f"  {run:28s} lines {n_lines:6d} min id {min_id} frames-dup-id {dupf} seq-frame>N {maxf_bad} | best IoU: "
          f"min {ious.min():.3f} p1 {np.percentile(ious, 1):.3f} median {np.median(ious):.4f} "
          f"frac>=0.99 {np.mean(ious >= 0.99):.3f} frac<0.5 {np.mean(ious < 0.5):.4f} | conf of best det: "
          f"min {best_conf.min():.3f} frac<0.25 {np.mean(best_conf < 0.25):.4f}")

# ------------------------------------------------------------------ (E) workid vs bytetrack box sets, offline / uidonly
print("\n== (E) workid vs bytetrack boxes; _offline / _uidonly derivations")


def lines_of(split, run, seq):
    with open(os.path.join(TR, split, run, "data", f"{seq}.txt"), encoding="utf-8") as fh:
        return [l.rstrip("\n") for l in fh]


for split in ("MOT17-train", "MOT17sub7-train"):
    for run in sorted(os.listdir(os.path.join(TR, split))):
        if "workid_phase2" not in run or run.endswith(("_offline", "_uidonly")):
            continue
        bt = run.replace("workid_phase2", "bytetrack_phase2")
        boxdiff = idfunc_bad = off_bad = uid_bad = prov_wrong = 0
        n_merged_ids = 0
        per_track_prov = Counter()
        for seq in SEQS:
            W = lines_of(split, run, seq)
            B = lines_of(split, bt, seq) if os.path.isdir(os.path.join(TR, split, bt)) else None
            if B is not None:
                ws = Counter((l.split(",")[0],) + tuple(l.split(",")[2:6]) for l in W)
                bs = Counter((l.split(",")[0],) + tuple(l.split(",")[2:6]) for l in B)
                if ws != bs:
                    boxdiff += 1
                # provisional ids must be 1e6 + the bytetrack tid of the same box
                bmap = {}
                for l in B:
                    p = l.split(",")
                    bmap[(p[0],) + tuple(p[2:6])] = int(p[1])
                tid_to_ids = defaultdict(list)
                for l in W:
                    p = l.split(",")
                    wid = int(p[1])
                    btid = bmap.get((p[0],) + tuple(p[2:6]))
                    if wid >= 1_000_000 and btid is not None and wid != 1_000_000 + btid:
                        prov_wrong += 1
                    if btid is not None:
                        tid_to_ids[btid].append((int(p[0]), wid))
                for btid, seqids in tid_to_ids.items():
                    seqids.sort()
                    n_prov = sum(1 for _, w in seqids if w >= 1_000_000)
                    per_track_prov[n_prov] += 1
            O = lines_of(split, run + "_offline", seq)
            if len(O) != len(W):
                off_bad += 1
            mp = {}
            for lw, lo in zip(W, O):
                pw, po = lw.split(","), lo.split(",")
                if pw[0] != po[0] or pw[2:] != po[2:]:
                    off_bad += 1
                    break
                a, b = int(pw[1]), int(po[1])
                if a >= 1_000_000 and a != b:
                    off_bad += 1
                if mp.setdefault(a, b) != b:
                    idfunc_bad += 1
            n_merged_ids += sum(1 for a, b in mp.items() if a != b)
            U = lines_of(split, run + "_uidonly", seq)
            if U != [l for l in W if int(l.split(",")[1]) < 1_000_000]:
                uid_bad += 1
        print(f"  {split:16s} {run:36s} seqs-boxset-differs-from-bytetrack {boxdiff}  prov-id!=1e6+tid {prov_wrong}  "
              f"offline-format-bad {off_bad}  offline-id-not-function {idfunc_bad}  ids-remapped {n_merged_ids}  "
              f"uidonly-bad {uid_bad}  provisional-boxes-per-track {dict(sorted(per_track_prov.items()))}")

# ------------------------------------------------------------------ (F) 7 Hz GT vs original GT
print("\n== (F) 7 Hz ground truth = original rows on kept frames, renumbered 1..K")
for seq in SEQS:
    fps, n, W, H = seqinfo(seq)
    k = max(1, round(fps / 7.0))
    kept = list(range(1, n + 1, k))
    fm = FM[seq]
    ok_map = fm["step"] == k and fm["kept_frames"] == kept
    orig = [l.rstrip("\n").split(",") for l in open(os.path.join(MOT, seq, "gt", "gt.txt"), encoding="utf-8") if l.strip()]
    new_of = {f: i + 1 for i, f in enumerate(kept)}
    want = [[str(new_of[int(r[0])])] + r[1:] for r in orig if int(r[0]) in new_of]
    got = [l.rstrip("\n").split(",") for l in open(os.path.join(GT_TE, "MOT17sub7-train", seq, "gt", "gt.txt"), encoding="utf-8") if l.strip()]
    c = configparser.ConfigParser(); c.optionxform = str
    c.read(os.path.join(GT_TE, "MOT17sub7-train", seq, "seqinfo.ini"))
    sl = int(c["Sequence"]["seqLength"])
    full = [l.rstrip("\n") for l in open(os.path.join(GT_TE, "MOT17-train", seq, "gt", "gt.txt"), encoding="utf-8")]
    orig_txt = [l.rstrip("\n") for l in open(os.path.join(MOT, seq, "gt", "gt.txt"), encoding="utf-8")]
    print(f"  {seq}: step {k} kept {len(kept)} map-ok {ok_map} rows-equal {want == got} ({len(got)} rows) "
          f"seqLength {sl}=={len(kept)} {sl == len(kept)} frameRate {c['Sequence']['frameRate']} "
          f"native-gt-identical {full == orig_txt}")
