"""Replay oracle__workid_phase2 (native) exactly as mot17_track.Phase2 does, and record every
identity decision (source, similarity) and whether a re-link joined the same GT identity."""
import os, sys, collections, json
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
PHASE2 = os.path.join(ROOT, "phase2")
sys.path.insert(0, PHASE2)
import cv2
import supervision as sv
from src.config import load_config
from src.tracker import PersonTracker
from src.identity import IdentityManager
from src.pipeline import identity_kwargs, _tracked_arrays
from src.reid import build_embedder
cfg = load_config(os.path.join(PHASE2, "config.yaml"))
seqs = sys.argv[1:] or SEQS
summary = collections.Counter()
events_all = []
for seq in seqs:
    import configparser
    c = configparser.ConfigParser(); c.optionxform = str; c.read(os.path.join(MOT, seq, "seqinfo.ini"))
    fps, n = float(c["Sequence"]["frameRate"]), int(c["Sequence"]["seqLength"])
    fr = int(round(fps))
    gt = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    byf = collections.defaultdict(list)
    for r in gt:
        byf[int(r[0])].append(r)
    tracker = PersonTracker([3], [1, 2], frame_rate=fr, lost_track_buffer=cfg.lost_track_buffer)
    mgr = IdentityManager(build_embedder(cfg.identity_method, device="cpu"), **identity_kwargs(cfg, fr))
    W, H = None, None
    uid_last_gt = {}      # uid -> last GT id seen under it
    tid_seen = set()
    out_rows = []
    for f in range(1, n + 1):
        rows = byf.get(f, [])
        if rows:
            g = np.array(rows)
            dets = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]].astype(np.float32)
            d = sv.Detections(xyxy=dets, confidence=np.full(len(g), 0.9, np.float32),
                              class_id=np.full(len(g), 3, dtype=int))
        else:
            g = np.zeros((0, 9)); d = sv.Detections.empty()
        tracked, _ = tracker.update(d)
        img = cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg"))
        if W is None:
            H, W = img.shape[:2]
        ids, boxes = _tracked_arrays(tracked)
        out = mgr.update(img, ids, boxes, marker_labels={}, now_s=(f - 1) / fps)
        # map each tracked box back to its GT row (ByteTrack outputs the raw input boxes)
        for tid, box in zip(ids, boxes):
            gi = None
            for k, r in enumerate(rows):
                if abs(r[2] - box[0]) < 0.01 and abs(r[3] - box[1]) < 0.01 and abs(r[2] + r[4] - box[2]) < 0.01:
                    gi = r; break
            gid = int(gi[1]) if gi is not None else -1
            vis = float(gi[8]) if gi is not None else -1
            res = out.get(tid)
            if res is None:
                continue
            if tid not in tid_seen:
                tid_seen.add(tid)
                prev = uid_last_gt.get(res.uid)
                clipped = box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H
                ev = {"seq": seq, "frame": f, "tid": tid, "uid": res.uid, "source": res.source,
                      "sim": round(float(res.similarity), 3), "gt": gid, "vis": vis,
                      "h": round(float(box[3] - box[1]), 1), "clipped": bool(clipped),
                      "relink": prev is not None, "prev_gt": prev,
                      "correct": (prev == gid) if prev is not None else None}
                events_all.append(ev)
            uid_last_gt[res.uid] = gid
    ev = [e for e in events_all if e["seq"] == seq]
    rel = [e for e in ev if e["relink"]]
    print(f"{seq}: decisions {len(ev)}, new {sum(1 for e in ev if not e['relink'])}, relinks {len(rel)}, "
          f"relinks to the same GT id {sum(1 for e in rel if e['correct'])}, "
          f"by source {collections.Counter((e['source'], e['correct']) for e in rel)}", flush=True)
json.dump(events_all, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_relinks.json"), "w"), indent=0)
rel = [e for e in events_all if e["relink"]]
print("ALL relinks", len(rel), "correct", sum(1 for e in rel if e["correct"]),
      "by source", collections.Counter((e["source"], e["correct"]) for e in rel))
wrong = [e for e in rel if not e["correct"]]
print("wrong relinks: vis<0.25 at decision", sum(1 for e in wrong if e["vis"] < 0.25), "of", len(wrong),
      "; box clipped by frame", sum(1 for e in wrong if e["clipped"]), "; sim quartiles", np.quantile([e["sim"] for e in wrong], [0.25, 0.5, 0.75]).round(3) if wrong else None)
right = [e for e in rel if e["correct"]]
print("correct relinks: vis<0.25", sum(1 for e in right if e["vis"] < 0.25), "of", len(right), "; sim quartiles", np.quantile([e["sim"] for e in right], [0.25, 0.5, 0.75]).round(3) if right else None)
