"""Revision R1, Part D: run every tracker on the SAME cached MOT17 detections; write MOT results.

Trackers (default settings, nothing tuned), and the environment each needs:
  bytetrack_phase2   phase2/src/tracker.py PersonTracker (supervision ByteTrack) with the
                     phase2/config.yaml values; raw tracker IDs                      [.venv]
  workid_phase2      the same tracker + phase2's IdentityManager with the identity defaults
                     (identity_kwargs(cfg, fps), badges off, clip time as now_s); the stable
                     worker uid is the ID. Ghost (predicted) boxes are never emitted. Also
                     writes workid_phase2_offline: the same run with the end-of-session
                     consolidation merges applied                                    [.venv]
  botsort_reid       ultralytics 8.4.75 BOT-SORT, botsort.yaml with with_reid: True and
                     model: auto, replayed on the cached boxes WITH the cached native
                     features, exactly as model.track builds and feeds it            [.venv-eval]
  strongsort         boxmot 26.0.0 StrongSort(), default config, default ReID
                     (osnet_x0_25_msmt17)                                           [.venv-eval]
  deepocsort         boxmot 26.0.0 DeepOcSort(), defaults, same ReID weights         [.venv-eval]

Inputs (--det): 'grouped' and 'coco' are the cached detections (conf 0.10) from
tools/revision/mot17_detect.py; 'oracle' is the MOT17 ground truth (class 1, considered flag 1)
given as detections with confidence 0.9 (the value phase5_workid/reid_eval.py gives synthetic
boxes; ByteTrack's association gate depends on IoU x conf).

The deployed pipeline only ever feeds ByteTrack boxes with conf > 0.35 (its detector
threshold), so bytetrack_phase2 / workid_phase2 filter the cache at > 0.35 ('--feed pipeline',
the default). '--feed all' gives them every cached box (>= 0.10) as a labelled sensitivity run.
The other trackers always get the whole cache and apply their own default thresholds.

--rate 7hz keeps every round(fps/7)-th frame (frame_map from mot17_prepare.py), renumbers the
kept frames 1..K for TrackEval, and gives the phase2 tracker frame_rate = int(round(fps/k)),
the convention of phase7_mobile/analyze.py.

CPU time per frame is measured around the tracker call(s) only (detection is cached; image
decoding is excluded). For workid_phase2 it is reported for the ByteTrack call and the identity
layer separately.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import configparser
import datetime
import json
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PHASE2 = os.path.join(ROOT, "phase2")
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TRACKERS = os.path.join(CACHE, "trackers", "mot_challenge")
BOXMOT_CWD = os.path.join(CACHE, "boxmot_weights")      # holds models/osnet_x0_25_msmt17.pt
RUNS = os.path.join(ROOT, "paper_materials", "revision_r1", "partD", "runs")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
ORACLE_CONF = 0.9
PIPELINE_CONF = 0.35


def seqinfo(seq):
    c = configparser.ConfigParser()
    c.optionxform = str
    c.read(os.path.join(MOT, seq, "seqinfo.ini"))
    s = c["Sequence"]
    return float(s["frameRate"]), int(s["seqLength"])


def kept_frames(seq, rate):
    fps, n = seqinfo(seq)
    if rate == "native":
        return list(range(1, n + 1)), 1, fps
    fm = json.load(open(os.path.join(GT_TE, "MOT17sub7-train_frame_map.json"), encoding="utf-8"))[seq]
    return fm["kept_frames"], fm["step"], fps


def load_dets(det, seq):
    """{frame: (dets N x 6 float32 [x1 y1 x2 y2 conf cls], feats N x 128 or None)}."""
    out = {}
    if det == "oracle":
        gt = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
        keep = (gt[:, 7] == 1) & (gt[:, 6] == 1)
        for row in gt[keep]:
            f = int(row[0])
            x, y, w, h = row[2:6]
            out.setdefault(f, []).append([x, y, x + w, y + h, ORACLE_CONF, 0.0])
        return {f: (np.array(v, np.float32), None) for f, v in out.items()}
    z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
    frames, off, dets, feats = z["frames"], z["off"], z["dets"], z["feats"]
    for i, f in enumerate(frames):
        out[int(f)] = (dets[off[i]:off[i + 1]], feats[off[i]:off[i + 1]])
    return out


def imread(seq, f):
    import cv2
    return cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg"))


# --------------------------------------------------------------------------- trackers

class Phase2:
    """ByteTrack exactly as SafetyPipeline builds it, optionally + the identity layer."""

    def __init__(self, fr, fps, with_identity):
        sys.path.insert(0, PHASE2)
        import supervision as sv
        from src.config import load_config
        from src.tracker import PersonTracker
        self.sv = sv
        self.cfg = load_config(os.path.join(PHASE2, "config.yaml"))
        self.PERSON = 3
        self.tracker = PersonTracker([self.PERSON], [1, 2], frame_rate=fr,
                                     lost_track_buffer=self.cfg.lost_track_buffer)
        self.fps = fps
        self.mgr = None
        if with_identity:
            from src.identity import IdentityManager
            from src.pipeline import identity_kwargs, _tracked_arrays
            from src.reid import build_embedder
            self._tracked_arrays = _tracked_arrays
            self.mgr = IdentityManager(build_embedder(self.cfg.identity_method, device="cpu"),
                                       **identity_kwargs(self.cfg, fr))
            self.kwargs = identity_kwargs(self.cfg, fr)
        self.t_track = self.t_ident = 0.0

    def step(self, f, dets, feats, img):
        sv = self.sv
        if len(dets):
            d = sv.Detections(xyxy=dets[:, :4].astype(np.float32),
                              confidence=dets[:, 4].astype(np.float32),
                              class_id=np.full(len(dets), self.PERSON, dtype=int))
        else:
            d = sv.Detections.empty()
        t0 = time.perf_counter()
        tracked, _ = self.tracker.update(d)
        self.t_track += time.perf_counter() - t0
        rows = []
        if self.mgr is None:
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                rows.append((int(tid), box, None))
            return rows
        ids, boxes = self._tracked_arrays(tracked)
        t0 = time.perf_counter()
        out = self.mgr.update(img, ids, boxes, marker_labels={}, now_s=(f - 1) / self.fps)
        self.t_ident += time.perf_counter() - t0
        for tid, box in zip(ids, boxes):
            r = out.get(tid)
            # In probation the pipeline shows the person as 'Person #tid' (no uid yet); emit it
            # under a provisional ID that can never collide with a uid number.
            rows.append((int(r.uid[1:]) if r is not None else 1_000_000 + int(tid),
                         np.asarray(box, np.float32), r.uid if r is not None else None))
        return rows


class BotSortReID:
    def __init__(self):
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.bot_sort import BOTSORT
        self.Boxes = Boxes
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
        cfg.with_reid = True                      # model stays 'auto' (native detector features)
        self.cfg = dict(vars(cfg)) if hasattr(cfg, "__dict__") else str(cfg)
        self.tracker = BOTSORT(args=cfg)          # as ultralytics/trackers/track.py:77
        self.t_track = 0.0

    def step(self, f, dets, feats, img):
        if feats is None:
            raise SystemExit("[FAIL] BoT-SORT with native ReID needs cached features")
        t0 = time.perf_counter()
        out = self.tracker.update(self.Boxes(dets, img.shape[:2]), img, feats)
        self.t_track += time.perf_counter() - t0
        return [(int(r[4]), r[:4], None) for r in out]


class BoxMot:
    def __init__(self, name):
        import boxmot
        cls = {"strongsort": boxmot.StrongSort, "deepocsort": boxmot.DeepOcSort}[name]
        os.chdir(BOXMOT_CWD)                      # default ReID weights resolve from ./models
        self.tracker = cls()
        self.cfg = self.tracker.config.to_dict()
        self.t_track = 0.0

    def step(self, f, dets, feats, img):
        t0 = time.perf_counter()
        out = np.asarray(self.tracker.update(dets.astype(np.float32), img))
        self.t_track += time.perf_counter() - t0
        return [(int(r[4]) + 1, r[:4], None) for r in out]     # boxmot IDs start at 0


def run(tracker, det, rate, feed, threads):
    import torch
    torch.set_num_threads(threads)
    if tracker in ("bytetrack_phase2", "workid_phase2") and det != "oracle" and feed == "pipeline":
        min_conf = PIPELINE_CONF
    else:
        min_conf = None
    split = "MOT17-train" if rate == "native" else "MOT17sub7-train"
    name = f"{det}__{tracker}" + ("__feedall" if feed == "all" and tracker.endswith("phase2")
                                  and det != "oracle" else "")
    outdirs = {name: os.path.join(TRACKERS, split, name, "data")}
    if tracker == "workid_phase2":
        outdirs[name + "_offline"] = os.path.join(TRACKERS, split, name + "_offline", "data")
    for d in outdirs.values():
        os.makedirs(d, exist_ok=True)
    record = {"run": name, "tracker": tracker, "detections": det, "rate": rate, "feed": feed,
              "min_conf_fed": min_conf, "sequences": {}, "started": datetime.datetime.now().isoformat(timespec="seconds"),
              "command": " ".join([sys.executable] + sys.argv),
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "torch": torch.__version__, "platform": platform.platform()},
              "torch_threads": threads}
    for seq in SEQS:
        frames, step, fps = kept_frames(seq, rate)
        fr = int(round(fps / step))
        D = load_dets(det, seq)
        if tracker == "bytetrack_phase2":
            T = Phase2(fr, fps, False)
        elif tracker == "workid_phase2":
            T = Phase2(fr, fps, True)
        elif tracker == "botsort_reid":
            T = BotSortReID()
        else:
            T = BoxMot(tracker)
        lines, uid_rows = [], []
        n_in = 0
        for new_f, f in enumerate(frames, start=1):
            dets, feats = D.get(f, (np.zeros((0, 6), np.float32), np.zeros((0, 128), np.float32)))
            if feats is None and det != "oracle":
                feats = np.zeros((0, 128), np.float32)
            if min_conf is not None and len(dets):
                m = dets[:, 4] > min_conf
                dets = dets[m]
                feats = feats[m] if feats is not None else None
            n_in += len(dets)
            need_img = tracker != "bytetrack_phase2"
            img = imread(seq, f) if need_img else None
            for tid, box, uid in T.step(f, dets, feats, img):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                lines.append((new_f, tid, x1, y1, x2 - x1, y2 - y1))
                if tracker == "workid_phase2":
                    uid_rows.append((new_f, tid, uid, x1, y1, x2 - x1, y2 - y1))
        with open(os.path.join(outdirs[name], f"{seq}.txt"), "w", encoding="utf-8") as fh:
            for fr_, tid, x, y, w, h in lines:
                fh.write(f"{fr_},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        seq_rec = {"frames": len(frames), "step": step, "frame_rate_given": fr, "fps": fps,
                   "boxes_in": n_in, "boxes_out": len(lines),
                   "ids_out": len({l[1] for l in lines}),
                   "ms_per_frame_tracker": 1000 * T.t_track / max(len(frames), 1)}
        if tracker == "workid_phase2":
            from phase5_workid.reid_eval import _remap_from       # follows merge chains
            merges = T.mgr.consolidate()
            remap = _remap_from(merges)
            with open(os.path.join(outdirs[name + "_offline"], f"{seq}.txt"), "w", encoding="utf-8") as fh:
                for fr_, tid, uid, x, y, w, h in uid_rows:
                    t2 = int(remap(uid)[1:]) if uid is not None else tid
                    fh.write(f"{fr_},{t2},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
            seq_rec.update(ms_per_frame_identity=1000 * T.t_ident / max(len(frames), 1),
                           consolidation_merges=len(merges),
                           workers_created=len(T.mgr.workers),
                           provisional_id_boxes=sum(1 for r in uid_rows if r[2] is None))
            record["identity_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                         for k, v in T.kwargs.items()}
        if hasattr(T, "cfg") and not isinstance(T, Phase2):
            record["tracker_config"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k, v in (T.cfg.items() if isinstance(T.cfg, dict) else [])}
        record["sequences"][seq] = seq_rec
        print(f"[{datetime.datetime.now():%H:%M:%S}] {name} {rate} {seq}: {seq_rec['ids_out']} ids, "
              f"{seq_rec['ms_per_frame_tracker']:.1f} ms/frame", flush=True)
    tot_frames = sum(s["frames"] for s in record["sequences"].values())
    record["ms_per_frame_tracker_all"] = sum(s["ms_per_frame_tracker"] * s["frames"]
                                             for s in record["sequences"].values()) / tot_frames
    if tracker == "workid_phase2":
        record["ms_per_frame_identity_all"] = sum(s["ms_per_frame_identity"] * s["frames"]
                                                  for s in record["sequences"].values()) / tot_frames
    record["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, f"{name}__{rate}.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracker", required=True,
                    choices=["bytetrack_phase2", "workid_phase2", "botsort_reid", "strongsort", "deepocsort"])
    ap.add_argument("--det", required=True, choices=["grouped", "coco", "oracle"])
    ap.add_argument("--rate", default="native", choices=["native", "7hz"])
    ap.add_argument("--feed", default="pipeline", choices=["pipeline", "all"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seqs", nargs="*", default=None, help="smoke tests only: a subset of sequences")
    ap.add_argument("--smoke-dir", default=None, help="smoke tests only: write here, not to the run folders")
    a = ap.parse_args(argv)
    sys.path.insert(0, ROOT)
    global SEQS, TRACKERS, RUNS
    if a.seqs:
        SEQS = a.seqs
    if a.smoke_dir:
        TRACKERS, RUNS = os.path.join(a.smoke_dir, "trackers"), os.path.join(a.smoke_dir, "runs")
    run(a.tracker, a.det, a.rate, a.feed, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
