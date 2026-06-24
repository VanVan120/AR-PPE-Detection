"""Activity / dynamic-workflow recognition — SCAFFOLD (off by default).

This is the *seam* for egocentric workflow recognition, not the final recognizer.
The real model waits on the confirmed dataset (Assembly101 / Ego4D); this wires
the architecture so it drops in later without touching the pipeline:

    rolling per-stream clip buffer  ->  recognizer.infer(clip)  ->  {step, conf, mistake}

Backends:
  * `placeholder` (default) — honest no-op: returns step "pending-dataset". Use this
    until the dataset is confirmed; it proves the seam runs at full FPS.
  * `kinetics` — a generic **Kinetics-400** pretrained video model (torchvision
    r3d_18). It is a DEMO that the seam works end-to-end on real video; its labels
    are everyday actions, NOT construction steps. Replace with an Assembly101-
    fine-tuned model (same `infer(clip)` contract) when the dataset lands.

Note on egocentric framing: Assembly101/Ego4D recognize the **camera wearer's** own
activity, so the default stream key is "ego" (the whole first-person frame), not
per-tracked-person. A per-worker (third-person) variant is a later option.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ActivityResult:
    step: str
    confidence: float
    mistake: bool = False


class _ClipBuffer:
    """Per-stream rolling buffer that emits a non-overlapping clip of `clip_len`
    frames, sampling one frame every `stride` frames (keeps inference affordable)."""

    def __init__(self, clip_len: int, stride: int):
        self.clip_len = max(1, int(clip_len))
        self.stride = max(1, int(stride))
        self._buf: dict[str, deque] = {}
        self._tick: dict[str, int] = {}

    def push(self, key: str, frame: np.ndarray):
        self._tick[key] = self._tick.get(key, 0) + 1
        if (self._tick[key] - 1) % self.stride != 0:
            return None
        dq = self._buf.setdefault(key, deque(maxlen=self.clip_len))
        # Store an immutable snapshot: the caller draws the AR overlay onto this same
        # array in place after pushing, which would otherwise corrupt buffered clip
        # frames (the recognizer would classify HUD-burned, non-egocentric pixels).
        dq.append(np.ascontiguousarray(frame).copy())
        if len(dq) == self.clip_len:
            clip = list(dq)
            dq.clear()                       # non-overlapping windows
            return clip
        return None


class PlaceholderRecognizer:
    name = "placeholder"

    def infer(self, clip) -> ActivityResult:
        return ActivityResult(step="pending-dataset", confidence=0.0, mistake=False)


class KineticsRecognizer:
    """Generic Kinetics-400 action recognition (torchvision r3d_18). DEMO only."""
    name = "kinetics"

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.video import r3d_18, R3D_18_Weights
        self._torch = torch
        self.device = 0 if device == "cuda" else device
        self.weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=self.weights).eval()
        try:
            self.model = self.model.to(self.device)
        except Exception:
            self.device = "cpu"
        self.preprocess = self.weights.transforms()
        self.categories = list(self.weights.meta["categories"])

    def infer(self, clip) -> ActivityResult:
        torch = self._torch
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]   # BGR -> RGB
        arr = np.ascontiguousarray(np.stack(rgb))                  # (T, H, W, C) uint8
        vid = torch.from_numpy(arr).permute(0, 3, 1, 2)            # (T, C, H, W)
        batch = self.preprocess(vid).unsqueeze(0).to(self.device)  # (B, C, T, H, W)
        with torch.no_grad():
            prob = self.model(batch).softmax(1)[0]
            conf, idx = prob.max(0)
        return ActivityResult(step=self.categories[int(idx)], confidence=float(conf), mistake=False)


def build_recognizer(backend: str, device: str = "cpu"):
    if backend == "kinetics":
        return KineticsRecognizer(device=device)
    return PlaceholderRecognizer()


class ActivityModule:
    """Clip buffer + recognizer; call update(key, frame) every frame, read latest()."""

    def __init__(self, backend: str = "placeholder", clip_len: int = 16,
                 stride: int = 2, device: str = "cpu"):
        self.buffer = _ClipBuffer(clip_len, stride)
        self.recognizer = build_recognizer(backend, device)
        self.backend = getattr(self.recognizer, "name", backend)
        self._last: dict[str, ActivityResult] = {}

    def update(self, key: str, frame: np.ndarray) -> Optional[ActivityResult]:
        clip = self.buffer.push(key, frame)
        if clip is not None:
            self._last[key] = self.recognizer.infer(clip)
        return self._last.get(key)

    def latest(self, key: str) -> Optional[ActivityResult]:
        return self._last.get(key)
