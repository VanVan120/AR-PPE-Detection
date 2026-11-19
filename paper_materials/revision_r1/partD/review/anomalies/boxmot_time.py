"""StrongSort on cached coco MOT17-09: all cached boxes (as Part D) vs boxes pre-filtered at its own
min_conf 0.6. Same outputs expected; timing difference = OSNet on boxes the tracker discards."""
import os, sys, time
os.environ["YOLO_AUTOINSTALL"] = "false"
import numpy as np, cv2, torch
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
os.chdir(os.path.join(HERE, "boxmot_cwd"))
import boxmot
seq = sys.argv[1] if len(sys.argv) > 1 else "MOT17-09-FRCNN"
nfr = int(sys.argv[2]) if len(sys.argv) > 2 else 200
z = np.load(os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections", f"coco__{seq}.npz"))
frames, off, dets = z["frames"], z["off"], z["dets"]
imgs = [cv2.imread(os.path.join(MOT, seq, "img1", f"{int(f):06d}.jpg")) for f in frames[:nfr]]
res = {}
for mode in ("all", "prefilter"):
    trk = boxmot.StrongSort()
    out_all, t = [], 0.0
    for i in range(nfr):
        d = dets[off[i]:off[i + 1]].astype(np.float32)
        if mode == "prefilter":
            d = d[d[:, 4] >= 0.6]
        t0 = time.perf_counter()
        o = np.asarray(trk.update(d, imgs[i]))
        t += time.perf_counter() - t0
        out_all.append(o.round(2))
    res[mode] = out_all
    nb = sum(len(dets[off[i]:off[i + 1]]) for i in range(nfr))
    print(mode, f"ms/frame {1000 * t / nfr:.1f}", "boxes/frame in", f"{nb / nfr:.1f}" if mode == 'all' else f"{sum((dets[off[i]:off[i+1]][:,4]>=0.6).sum() for i in range(nfr))/nfr:.1f}", flush=True)
same = all(a.shape == b.shape and np.allclose(a[:, :5], b[:, :5], atol=0.05) for a, b in zip(res["all"], res["prefilter"]))
print("outputs identical (boxes + ids):", same)
