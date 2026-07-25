"""Appearance re-identification — recognise the same worker after the tracker loses them.

ByteTrack assigns a `tracker_id` from motion continuity alone. When a worker is occluded
for longer than `lost_track_buffer`, walks out of frame, or is missed by the detector for
a stretch, that id is retired and the person comes back as a **new** id. Everything keyed
on `tracker_id` — compliance state, Work-ID binding, violation history — is lost with it.

This module supplies the missing signal: a compact **appearance descriptor** of a person
crop, so a returning worker can be matched back to who they were. `identity.py` consumes
it; nothing here knows about tracking.

Two embedders, both returning L2-normalised vectors so cosine similarity is a dot product:

  * `ColorHistogramEmbedder` (default) — banded HSV histograms with centre weighting.
    Pure OpenCV/NumPy: no model, no download, ~0.1 ms per crop. Discriminates people by
    clothing colour layout (helmet / vest / trousers).
  * `DeepEmbedder` (opt-in) — torchvision ResNet-18 penultimate features. Stronger under
    pose and lighting change, but downloads ImageNet weights on first use and costs
    milliseconds per crop.

**The limitation that matters, stated up front.** Appearance re-ID assumes people *look
different*. On a real site in uniform PPE — same hi-vis vest, same helmet — that
assumption is weak, and colour histograms in particular will confuse two workers wearing
the same kit. This is measured, not hand-waved: see `phase5_workid/`. It is precisely why
the ArUco marker stays **authoritative** in `identity.py` and appearance is only ever used
to bridge gaps where no marker is visible.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

# Person crops are resized to this before description, so a distant worker and a close
# one produce comparable vectors.
_STD_W, _STD_H = 64, 128


def crop_person(frame: np.ndarray, box: Sequence[float],
                pad: float = 0.0) -> Optional[np.ndarray]:
    """Safely crop a person box from a frame. Returns None if the box is degenerate or
    falls outside the image (both happen with predicted/extrapolated tracker boxes)."""
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if pad:
        dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 - xi1 < 4 or yi2 - yi1 < 8:          # too small to describe meaningfully
        return None
    return frame[yi1:yi2, xi1:xi2]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between (N, D) and (M, D) L2-normalised matrices -> (N, M).

    Inputs are re-normalised defensively: a zero vector (an all-black crop) would
    otherwise produce NaN and silently poison every comparison.
    """
    A = np.atleast_2d(np.asarray(a, dtype=np.float32))
    B = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if A.size == 0 or B.size == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
    return (A @ B.T).astype(np.float32)


class ColorHistogramEmbedder:
    """Banded HSV colour histogram — the zero-dependency default.

    The crop is resized to a standard size and split into `bands` horizontal strips
    (roughly helmet / torso+vest / legs). Each strip gets a joint Hue-Saturation
    histogram, individually L1-normalised so one dominant strip cannot swamp the rest,
    then all strips are concatenated and L2-normalised.

    Two details that matter for accuracy:

    * **Centre weighting.** A person box always contains background at the corners. Every
      pixel is weighted by an elliptical falloff so the body centre dominates and the
      background contributes little. Without this the descriptor partly encodes the
      *scene* — so a worker re-entering at a different spot fails to match.
    * **Value gating.** Near-black and near-white pixels have unstable hue, so they are
      excluded from the hue histogram via the OpenCV mask rather than voting randomly.
    """

    name = "histogram"

    def __init__(self, bands: int = 3, h_bins: int = 12, s_bins: int = 4):
        self.bands = max(1, int(bands))
        self.h_bins = max(2, int(h_bins))
        self.s_bins = max(1, int(s_bins))
        self.dim = self.bands * self.h_bins * self.s_bins
        self._weight = self._elliptical_weight(_STD_H, _STD_W)

    @staticmethod
    def _elliptical_weight(h: int, w: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = ((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2
        return np.clip(1.0 - 0.75 * r, 0.05, 1.0).astype(np.float32)

    def embed_one(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        img = cv2.resize(crop, (_STD_W, _STD_H), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        # Drop pixels whose hue is meaningless (too dark, too bright, too desaturated).
        valid = ((v > 30) & (v < 240) & (s > 30)).astype(np.uint8) * 255

        band_h = _STD_H // self.bands
        parts: List[np.ndarray] = []
        for b in range(self.bands):
            y0 = b * band_h
            y1 = _STD_H if b == self.bands - 1 else (b + 1) * band_h
            sub, sub_mask = hsv[y0:y1], valid[y0:y1]
            wsub = self._weight[y0:y1]
            hist = self._weighted_hs_hist(sub, sub_mask, wsub)
            total = float(hist.sum())
            parts.append(hist / total if total > 0 else hist)
        vec = np.concatenate(parts).astype(np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec

    def _weighted_hs_hist(self, hsv: np.ndarray, mask: np.ndarray,
                          weight: np.ndarray) -> np.ndarray:
        """Joint Hue-Saturation histogram with per-pixel weights.

        cv2.calcHist has no per-pixel weighting, so this bins with np.bincount, which
        accepts weights directly.
        """
        m = mask > 0
        if not np.any(m):
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)
        hq = (hsv[:, :, 0][m].astype(np.float32) * self.h_bins / 180.0).astype(np.int32)
        sq = (hsv[:, :, 1][m].astype(np.float32) * self.s_bins / 256.0).astype(np.int32)
        np.clip(hq, 0, self.h_bins - 1, out=hq)
        np.clip(sq, 0, self.s_bins - 1, out=sq)
        idx = hq * self.s_bins + sq
        w = weight[m].astype(np.float32)
        return np.bincount(idx, weights=w,
                           minlength=self.h_bins * self.s_bins).astype(np.float32)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not len(crops):
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(c) for c in crops]).astype(np.float32)


class DeepEmbedder:
    """Opt-in torchvision ResNet-18 penultimate features (512-D).

    Stronger than colour histograms under pose/lighting change. Downloads ImageNet
    weights on first use (~45 MB), so it is never the default — a demo machine with no
    internet must still work. These are ImageNet features, NOT a re-ID-trained metric
    model; they are a reasonable general descriptor, not a person-re-ID SOTA.
    """

    name = "deep"

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        self._torch = torch
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device
        self.preprocess = weights.transforms()
        self.dim = 512

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if not len(crops):
            return np.zeros((0, self.dim), dtype=np.float32)
        tensors = []
        for c in crops:
            img = c if (c is not None and c.size) else np.zeros((8, 8, 3), np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            tensors.append(self.preprocess(t))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy().astype(np.float32)
        n = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        return feats / n


def build_embedder(method: str = "histogram", device: str = "cpu"):
    """Factory. Falls back to the histogram embedder with a clear message if the deep
    one cannot be constructed (no torch, no torchvision, or no network for weights) —
    a live demo must never die because an optional descriptor is unavailable."""
    method = (method or "histogram").strip().lower()
    if method in ("deep", "resnet", "resnet18"):
        try:
            return DeepEmbedder(device=device)
        except Exception as exc:                      # noqa: BLE001 - report and degrade
            print(f"[warn] re-ID: deep embedder unavailable ({exc.__class__.__name__}: "
                  f"{exc}); falling back to the colour-histogram embedder.")
            return ColorHistogramEmbedder()
    if method not in ("histogram", "hist", "color", "colour"):
        print(f"[warn] re-ID: unknown method '{method}' — using the colour histogram.")
    return ColorHistogramEmbedder()
