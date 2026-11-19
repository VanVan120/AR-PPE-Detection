"""Independent replay through the DEPLOYED SafetyPipeline.process (only the detector is stubbed with the
cached boxes), compared line-by-line with the stored Part D tracker files.

  python pipeline_replay.py <det: grouped|coco|oracle> <rate: native|7hz> <feed: pipeline|all> [seqs...]

Writes only into this scratch folder.
"""
import configparser
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2
import numpy as np

cv2.setNumThreads(1)
import torch

torch.set_num_threads(1)
import supervision as sv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
PHASE2 = os.path.join(ROOT, "phase2")
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TR = os.path.join(CACHE, "trackers", "mot_challenge")
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
sys.path.insert(0, PHASE2)
from src.config import load_config  # noqa: E402
from src.pipeline import SafetyPipeline, _tracked_arrays  # noqa: E402

det, rate, feed = sys.argv[1], sys.argv[2], sys.argv[3]
seqs = sys.argv[4:] or SEQS
os.chdir(HERE)

cfg = load_config(os.path.join(PHASE2, "config.yaml"))
assert cfg.identity_enabled and not cfg.workid_enabled, "config is not 'identity on, badges off'"
print("cfg: device", cfg.device, "conf", cfg.confidence_threshold, "ltb", cfg.lost_track_buffer,
      "identity_method", cfg.identity_method, "consolidate", getattr(cfg, "identity_consolidate", None))
pipe = SafetyPipeline(cfg, frame_rate=30, render=False, quiet=True)
assert pipe.identity is not None and pipe.binder is None
PERSON = pipe.person_ids[0]
print("person ids", pipe.person_ids, "violation ids", list(pipe.rules_by_id))


def seqinfo(seq):
    c = configparser.ConfigParser(); c.optionxform = str
    c.read(os.path.join(MOT, seq, "seqinfo.ini"))
    return float(c["Sequence"]["frameRate"]), int(c["Sequence"]["seqLength"])


def remap_from(merges):
    parent = dict(merges)

    def r(u):
        seen = set()
        while u in parent and u not in seen:
            seen.add(u); u = parent[u]
        return u
    return r


tot_diff = {}
for seq in seqs:
    fps, n = seqinfo(seq)
    if rate == "native":
        kept, step = list(range(1, n + 1)), 1
    else:
        fm = json.load(open(os.path.join(GT_TE, "MOT17sub7-train_frame_map.json"), encoding="utf-8"))[seq]
        kept, step = fm["kept_frames"], fm["step"]
    fr = int(round(fps / step))                       # analyze.py: int(round(fps_in / every))
    pipe.reset(frame_rate=fr)                         # the deployed 'new session' path
    assert pipe.identity is not None and pipe.binder is None
    # what the stubbed detector returns this frame
    holder = {}
    pipe.detector.detect = lambda frame: holder["d"]
    cap = {}
    t_upd = pipe.tracker.update

    def tracker_update(d, _f=t_upd):
        tr, vi = _f(d)
        cap["tracked"] = tr
        return tr, vi
    pipe.tracker.update = tracker_update
    i_upd = pipe.identity.update

    def ident_update(*a, _f=i_upd, **k):
        out = _f(*a, **k)
        cap["ident"] = out
        cap["now_s"] = k.get("now_s")
        cap["marker_labels"] = k.get("marker_labels")
        return out
    pipe.identity.update = ident_update
    i_cons = pipe.identity.consolidate

    def ident_consolidate(_f=i_cons):
        m = _f()
        cap["merges"] = list(m)
        return m
    pipe.identity.consolidate = ident_consolidate

    if det == "oracle":
        g = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
        g = g[(g[:, 7] == 1) & (g[:, 6] == 1)]
        D = {}
        for r in g:
            D.setdefault(int(r[0]), []).append([r[2], r[3], r[2] + r[4], r[3] + r[5], 0.9])
        D = {f: np.array(v, np.float32) for f, v in D.items()}
    else:
        z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
        D = {int(f): z["dets"][z["off"][i]:z["off"][i + 1], :5] for i, f in enumerate(z["frames"])}
    bt, wk, wk_uid = [], [], []
    marker_seen = set()
    for new_f, f in enumerate(kept, start=1):
        d = D.get(f, np.zeros((0, 5), np.float32))
        if det != "oracle" and feed == "pipeline":
            d = d[d[:, 4] > cfg.confidence_threshold]
        holder["d"] = (sv.Detections(xyxy=d[:, :4].astype(np.float32), confidence=d[:, 4].astype(np.float32),
                                     class_id=np.full(len(d), PERSON, dtype=int))
                       if len(d) else sv.Detections.empty())
        frame = cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg"))
        res = pipe.process(frame, f, (f - int(os.environ.get("NOW_OFFSET", "0"))) / fps)  # analyze.py: frame_no / fps_in (NOW_OFFSET=1 -> mot17_track.py convention)
        marker_seen.add(repr(cap["marker_labels"]))
        tr = cap["tracked"]
        for box, tid in zip(tr.xyxy, tr.tracker_id):
            x1, y1, x2, y2 = [float(v) for v in box]
            bt.append(f"{new_f},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
        ids, boxes = _tracked_arrays(tr)
        for tid, b in zip(ids, boxes):
            r = cap["ident"].get(tid)
            oid = int(r.uid[1:]) if r is not None else 1_000_000 + tid
            x1, y1, x2, y2 = b
            wk.append(f"{new_f},{oid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
            wk_uid.append((r.uid if r is not None else None, oid))
        # people rows: ghosts present in the app view, never in the stored file
    pipe.close(kept[-1] / fps, kept[-1])
    rm = remap_from(cap["merges"])
    off = []
    for line, (uid, oid) in zip(wk, wk_uid):
        p = line.split(",")
        p[1] = str(int(rm(uid)[1:]) if uid is not None else oid)
        off.append(",".join(p))
    split = "MOT17-train" if rate == "native" else "MOT17sub7-train"
    suffix = "__feedall" if (feed == "all" and det != "oracle") else ""
    for run, mine in ((f"{det}__bytetrack_phase2{suffix}", bt), (f"{det}__workid_phase2{suffix}", wk),
                      (f"{det}__workid_phase2{suffix}_offline", off)):
        path = os.path.join(TR, split, run, "data", f"{seq}.txt")
        if os.environ.get("SAVE_DIR"):
            sd = os.path.join(os.environ["SAVE_DIR"], split, run + "__nowf", "data")
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, f"{seq}.txt"), "w", encoding="utf-8") as fh:
                fh.write("".join(l + "\n" for l in mine))
        stored = [l.rstrip("\n") for l in open(path, encoding="utf-8")] if os.path.isfile(path) else None
        if stored is None:
            print(f"  {run} {seq}: no stored file")
            continue
        nd = sum(1 for a, b in zip(mine, stored) if a != b) + abs(len(mine) - len(stored))
        tot_diff[run] = tot_diff.get(run, 0) + nd
        first = next((i for i, (a, b) in enumerate(zip(mine, stored)) if a != b), None)
        print(f"  {split} {run:40s} {seq}: fr {fr} mine {len(mine)} stored {len(stored)} differing {nd}"
              + (f" first diff line {first}: mine '{mine[first]}' stored '{stored[first]}'" if first is not None else ""),
              flush=True)
    print(f"  merges {len(cap['merges'])}, marker_labels seen {marker_seen}, last now_s {cap['now_s']:.3f}", flush=True)
print("TOTAL differing lines:", tot_diff)
