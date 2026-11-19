import os
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, cv2, torch
torch.set_num_threads(2)
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
from ultralytics import YOLO
from ultralytics.utils import nms, ops
W = os.path.join(ROOT, "paper_materials", "retrain", "local_out", "best_grouped.pt")
model = YOLO(W)
kw = dict(imgsz=640, conf=0.10, device="cpu", verbose=False, half=False)
model.predict(np.zeros((640, 640, 3), np.uint8), classes=[3], **kw)
p = model.predictor
print("names", model.names)

def match_count(gb, d):
    if len(gb) == 0 or len(d) == 0:
        return np.zeros(len(gb), bool)
    M = iou(gb, d); M[M < 0.5 - 1e-9] = 0
    a, b = linear_sum_assignment(-M); ok = M[a, b] > 0
    m = np.zeros(len(gb), bool); m[a[ok]] = True
    return m

STEP = 25
agg = {"n": 0, "single35": 0, "multi35": 0, "allcls_any35": 0, "cache_eq": 0, "cache_frames": 0}
other_cls_hits = {}
for s in SEQS:
    gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
    z = np.load(os.path.join(CACHE, f"grouped__{s}.npz"))
    frames, off, dets = z["frames"], z["off"], z["dets"]
    fidx = {int(f): i for i, f in enumerate(frames)}
    for f in range(1, int(frames.max()) + 1, STEP):
        img = cv2.imread(os.path.join(MOT, s, "img1", f"{f:06d}.jpg"))
        im = p.preprocess([img])
        with torch.no_grad():
            preds = p.inference(im)
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        o_single = nms.non_max_suppression(raw.clone(), 0.10, 0.7, classes=[3], multi_label=False, max_det=300)[0]
        o_multi = nms.non_max_suppression(raw.clone(), 0.10, 0.7, classes=[3], multi_label=True, max_det=300)[0]
        o_all = nms.non_max_suppression(raw.clone(), 0.10, 0.7, classes=None, multi_label=False, max_det=300)[0]
        def sc(o):
            o = o.clone()
            o[:, :4] = ops.scale_boxes(im.shape[2:], o[:, :4], img.shape)
            return o.numpy()
        os_, om, oa = sc(o_single), sc(o_multi), sc(o_all)
        # sanity: single-label reproduces the cache
        cd = dets[off[fidx[f]]:off[fidx[f] + 1]]
        agg["cache_frames"] += 1
        if len(cd) == len(os_) and (len(cd) == 0 or np.allclose(np.sort(cd[:, 4]), np.sort(os_[:, 4]), atol=1e-3)):
            agg["cache_eq"] += 1
        g = gt[(gt[:, 0] == f) & (gt[:, 7] == 1) & (gt[:, 6] == 1) & (gt[:, 8] >= 0.5) & (gt[:, 5] >= 150)]
        gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
        ms = match_count(gb, os_[os_[:, 4] > 0.35, :4])
        mm = match_count(gb, om[om[:, 4] > 0.35, :4])
        agg["n"] += len(gb); agg["single35"] += ms.sum(); agg["multi35"] += mm.sum()
        # for GT missed by single-label person: which non-person class boxes overlap IoU>=0.5 at conf>0.35?
        oth = oa[(oa[:, 5] != 3) & (oa[:, 4] > 0.35)]
        for k in np.where(~ms)[0]:
            if len(oth):
                I = iou(gb[k:k+1], oth[:, :4])[0]
                j = int(np.argmax(I))
                if I[j] >= 0.5:
                    c = int(oth[j, 5]); other_cls_hits[c] = other_cls_hits.get(c, 0) + 1
                    agg["allcls_any35"] += 1
    print(s, dict(agg), other_cls_hits, flush=True)
print("FINAL", agg, other_cls_hits)
print(f"visible(>=0.5) & h>=150 GT on sampled frames: single-label person recall {100*agg['single35']/agg['n']:.1f}%  "
      f"multi-label person recall {100*agg['multi35']/agg['n']:.1f}%  missed-with-other-class-box(IoU>=.5) {agg['allcls_any35']}")
