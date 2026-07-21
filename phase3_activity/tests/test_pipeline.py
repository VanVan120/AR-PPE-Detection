"""End-to-end smoke test for the TAS training/eval pipeline (no real data needed).

Builds a synthetic fixture where each frame's 2048-D feature encodes its label (a
class-specific block + noise), so a correct dataset->model->loss->train->eval->score
pipeline MUST learn it. Verifies: model output shape, pooled<->frame conversions, and
that training on the synthetic set drives validation MoF well above chance.

    python phase3_activity/tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tas import dataset as D
from tas import postprocess as P
from tas.model import MSTCN
from tas.torch_dataset import TASDataset
from tas.train import train_model
from tas.evaluate import evaluate_model

import torch

results = {}
NUM_CLASSES = 4
DIM = D.FEATURE_DIM          # 2048 (loader asserts this)


class FakeFeatureStore:
    """Deterministic 2048-D feature per (video, frame): a class-specific block = 1
    plus small noise, so the label is linearly decodable from the feature."""
    def __init__(self, gt_frames, num_classes=NUM_CLASSES, dim=DIM, noise=0.05, seed=0):
        self.gt = gt_frames
        self.block = dim // num_classes
        self.noise = noise
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def vector(self, key: str) -> np.ndarray:
        seq = key.split("/")[0]
        f = int(key.rsplit("_", 1)[1].split(".")[0])
        c = int(self.gt[seq][f])
        v = self._rng.normal(0, self.noise, size=self.dim).astype(np.float32)
        v[c * self.block:(c + 1) * self.block] += 1.0
        return v


def _make_video(rng, min_len=30, max_len=50):
    """A random sequence of 3-4 labelled segments -> per-frame labels."""
    n_seg = rng.integers(3, 5)
    labels, frames = [], []
    for _ in range(n_seg):
        lbl = int(rng.integers(0, NUM_CLASSES))
        if labels and lbl == labels[-1]:
            lbl = (lbl + 1) % NUM_CLASSES        # avoid merging adjacent same-label segs
        dur = int(rng.integers(8, 15))
        labels.append(lbl)
        frames.extend([lbl] * dur)
    return frames


def _build_fixture(n_train=8, n_val=4, seed=1):
    rng = np.random.default_rng(seed)
    gt, statistic = {}, {}
    train_ids, val_ids = [], []
    for i in range(n_train + n_val):
        vid = ("assembly_train_%02d" % i) if i < n_train else ("assembly_val_%02d" % (i - n_train))
        frames = _make_video(rng)
        gt[vid] = frames
        statistic[vid] = {"C10095_rgb": [0, len(frames) - 1]}      # inclusive span
        (train_ids if i < n_train else val_ids).append(vid)
    store = FakeFeatureStore(gt, seed=seed)
    return train_ids, val_ids, store, statistic, gt


# ---- postprocess conversions ------------------------------------------------
def test_postprocess():
    down = P.downsample_labels([0, 0, 0, 1, 1], chunk_size=3)
    up = P.upsample_predictions([0, 1], orig_len=5, chunk_size=3)
    pad = P.upsample_predictions([2], orig_len=5, chunk_size=2)
    results["postprocess: down/up/pad"] = (
        list(down) == [0, 1] and list(up) == [0, 0, 0, 1, 1] and list(pad) == [2, 2, 2, 2, 2])


# ---- model forward shape ----------------------------------------------------
def test_model_shape():
    m = MSTCN(num_stages=2, num_layers=3, num_f_maps=8, dim=DIM, num_classes=NUM_CLASSES)
    out = m(torch.randn(1, DIM, 7))
    results["model: output shape (S,B,C,T)"] = (tuple(out.shape) == (2, 1, NUM_CLASSES, 7))


# ---- dataset yields aligned (feature, target) -------------------------------
def test_dataset_item():
    train_ids, _val, store, stat, gt = _build_fixture()
    ds = TASDataset(train_ids, "C10095_rgb", store, stat, gt, chunk_size=5)
    x, y, vid, orig_len = ds[0]
    ok = (x.shape[0] == DIM and x.shape[1] == y.shape[0] and orig_len == len(gt[vid]))
    results["dataset: (D,T') feature aligned with (T',) target"] = ok


# ---- the real test: train, then val MoF must beat chance --------------------
def test_train_learns():
    train_ids, val_ids, store, stat, gt = _build_fixture()
    train_ds = TASDataset(train_ids, "C10095_rgb", store, stat, gt, chunk_size=5)
    val_ds = TASDataset(val_ids, "C10095_rgb", store, stat, gt, chunk_size=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_layers=2 -> receptive field ~7 steps < pooled seq length, so the per-frame
    # decodable signal wins instead of collapsing to the dominant class.
    model = MSTCN(num_stages=2, num_layers=2, num_f_maps=32, dim=DIM, num_classes=NUM_CLASSES)
    best = train_model(model, train_ds, val_ds, device, epochs=60, lr=1e-3,
                       weight_decay=0.0, patience=60, lr_step=1000, seed=0,
                       log_every=10, verbose=True)
    # chance MoF ~ 1/NUM_CLASSES ~ 25%; a working pipeline overfits this decodable signal.
    results["pipeline: val MoF beats chance (>=60)"] = (best.get("MoF", 0.0) >= 60.0)
    results["pipeline: summary has F1@{10,25,50}"] = (
        set(best.get("F1", {}).keys()) == {0.1, 0.25, 0.5})
    print("  final val:", {k: (round(v, 1) if isinstance(v, float) else v)
                           for k, v in best.items() if k != "F1"},
          "F1", {k: round(v, 1) for k, v in best.get("F1", {}).items()})


# ---- infer seam: checkpoint round-trip + Assembly101Recognizer.infer(clip) ---
def test_infer_seam():
    import tempfile
    from tas.model import MSTCN, save_checkpoint
    from tas.infer_seam import Assembly101Recognizer, ActivityResult

    class FakeExtractor:
        def extract(self, clip):
            return np.ones((len(clip), DIM), dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "m.pt")
        save_checkpoint(MSTCN(num_stages=1, num_layers=2, num_f_maps=8,
                              dim=DIM, num_classes=4), ckpt)
        acsv = os.path.join(td, "actions.csv")
        with open(acsv, "w", encoding="utf-8") as fh:
            fh.write("action_id,verb_id,noun_id,action_cls,verb_cls,noun_cls\n")
            for i, name in enumerate(["a", "b", "c", "d"]):
                fh.write(f"{i},0,0,{name},v,n\n")
        rec = Assembly101Recognizer(ckpt, acsv, device="cpu",
                                    extractor=FakeExtractor(), chunk_size=4)
        r = rec.infer([np.zeros((8, 8, 3), np.uint8) for _ in range(10)])
        empty = rec.infer([])
    ok = (isinstance(r, ActivityResult) and r.step in {"a", "b", "c", "d"}
          and 0.0 <= r.confidence <= 1.0 and r.mistake is False
          and empty.step == "no-input")
    results["seam: Assembly101Recognizer infer(clip) -> valid ActivityResult"] = ok


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    test_postprocess()
    test_model_shape()
    test_dataset_item()
    test_train_learns()
    test_infer_seam()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_PIPELINE", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
