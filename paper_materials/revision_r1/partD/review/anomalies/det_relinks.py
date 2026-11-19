"""Replay {det}__workid_phase2 (native, conf > 0.35 feed) as mot17_track.Phase2 does and score every
identity re-link against the GT identity matched (IoU >= 0.5, Hungarian) at the decision frame."""
import os, sys, collections, json, configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
PHASE2 = os.path.join(ROOT, "phase2"); sys.path.insert(0, PHASE2)
import cv2, supervision as sv
from src.config import load_config
from src.tracker import PersonTracker
from src.identity import IdentityManager
from src.pipeline import identity_kwargs, _tracked_arrays
from src.reid import build_embedder
cfg = load_config(os.path.join(PHASE2, "config.yaml"))
det = sys.argv[1] if len(sys.argv) > 1 else "coco"
tot = collections.Counter(); out_ids_check = {}
for seq in SEQS:
    c = configparser.ConfigParser(); c.optionxform = str; c.read(os.path.join(MOT, seq, "seqinfo.ini"))
    fps, n = float(c["Sequence"]["frameRate"]), int(c["Sequence"]["seqLength"]); fr = int(round(fps))
    gt = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    z = np.load(os.path.join(CACHE, f"{det}__{seq}.npz")); frames, off, dets = z["frames"], z["off"], z["dets"]
    tracker = PersonTracker([3], [1, 2], frame_rate=fr, lost_track_buffer=cfg.lost_track_buffer)
    mgr = IdentityManager(build_embedder(cfg.identity_method, device="cpu"), **identity_kwargs(cfg, fr))
    uid_last_gt, seen_tid = {}, set(); uids = set()
    for i, f in enumerate(frames):
        d = dets[off[i]:off[i + 1]]; d = d[d[:, 4] > 0.35]
        D = (sv.Detections(xyxy=d[:, :4].astype(np.float32), confidence=d[:, 4].astype(np.float32),
                           class_id=np.full(len(d), 3, dtype=int)) if len(d) else sv.Detections.empty())
        tracked, _ = tracker.update(D)
        img = cv2.imread(os.path.join(MOT, seq, "img1", f"{int(f):06d}.jpg"))
        ids, boxes = _tracked_arrays(tracked)
        res = mgr.update(img, ids, boxes, marker_labels={}, now_s=(int(f) - 1) / fps)
        g = gt[gt[:, 0] == f]
        gid_of = {}
        if len(g) and len(boxes):
            gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
            M = iou(np.asarray(boxes, float), gb); M[M < 0.5] = 0
            a, b = linear_sum_assignment(-M)
            for x, y in zip(a, b):
                if M[x, y] > 0: gid_of[ids[x]] = int(g[y, 1])
        for tid in ids:
            r = res.get(tid)
            if r is None: continue
            uids.add(r.uid)
            gid = gid_of.get(tid)
            if tid not in seen_tid:
                seen_tid.add(tid)
                prev = uid_last_gt.get(r.uid, "none") if r.source != "new" else "new"
                if r.source != "new":
                    tot["relinks"] += 1
                    if gid is None or prev in (None, "none"):
                        tot["relink_unscorable"] += 1
                    elif prev == gid:
                        tot["relink_same_gt"] += 1
                    else:
                        tot["relink_other_gt"] += 1
                else:
                    tot["new"] += 1
            if gid is not None:
                uid_last_gt[r.uid] = gid
    out_ids_check[seq] = len(uids)
    print(seq, dict(tot), "uids", len(uids), flush=True)
print("FINAL", det, dict(tot))
