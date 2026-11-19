"""Log every IdentityManager._gap call whose elapsed time is within 1e-9 s of the 3.0 s horizon, under
both clip-time origins, for coco/MOT17-05 (pipeline feed) up to frame 80.  Read-only."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np

cv2.setNumThreads(1)
import supervision as sv

ROOT = r"C:\Users\user\Desktop\SummerInternProject"
sys.path.insert(0, os.path.join(ROOT, "phase2"))
from src.config import load_config  # noqa: E402
from src.identity import IdentityManager  # noqa: E402
from src.pipeline import identity_kwargs, _tracked_arrays  # noqa: E402
from src.reid import build_embedder  # noqa: E402
from src.tracker import PersonTracker  # noqa: E402

MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
seq, fps, LAST = "MOT17-05-FRCNN", 14.0, 80
z = np.load(os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections", f"coco__{seq}.npz"))
D = {int(f): z["dets"][z["off"][i]:z["off"][i + 1]] for i, f in enumerate(z["frames"])}
frames = {f: cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg")) for f in range(1, LAST + 1)}
cfg = load_config(os.path.join(ROOT, "phase2", "config.yaml"))
orig_gap = IdentityManager._gap

for offset in (1, 0):
    log = []

    def gap(self, last_frame, last_s, at_frame, at_s, _o=orig_gap):
        res = _o(self, last_frame, last_s, at_frame, at_s)
        if at_s is not None and last_s is not None and abs((at_s - last_s) - 3.0) < 1e-9:
            log.append((self._frame_idx, last_frame, at_frame, repr(at_s - last_s), res[0]))
        return res
    IdentityManager._gap = gap
    tr = PersonTracker([3], [1, 2], frame_rate=14, lost_track_buffer=cfg.lost_track_buffer)
    mgr = IdentityManager(build_embedder(cfg.identity_method, device="cpu"), **identity_kwargs(cfg, 14))
    uids_at_68 = None
    for f in range(1, LAST + 1):
        d = D[f][D[f][:, 4] > 0.35]
        det = (sv.Detections(xyxy=d[:, :4].astype(np.float32), confidence=d[:, 4].astype(np.float32),
                             class_id=np.full(len(d), 3, dtype=int)) if len(d) else sv.Detections.empty())
        tracked, _ = tr.update(det)
        ids, boxes = _tracked_arrays(tracked)
        out = mgr.update(frames[f], ids, boxes, marker_labels={}, now_s=(f - offset) / fps)
        if f == 68:
            uids_at_68 = {t: (r.uid, r.source) for t, r in out.items()}
    IdentityManager._gap = orig_gap
    print(f"now_s = (f - {offset}) / {fps}: _gap calls at |dt - 3.0| < 1e-9 -> "
          f"[(update#, last_seen frame, at frame, dt_s, beyond_horizon)] = {log}")
    print(f"   frame 68 identities {{tid: (uid, source)}}: {uids_at_68}")
