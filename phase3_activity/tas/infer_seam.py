"""Bridge the trained Assembly101 TAS model into the Phase 2 `infer(clip)` seam.

`Assembly101Recognizer` mirrors the `phase2/src/activity.py` recognizer contract:
`.name` + `infer(clip) -> ActivityResult(step, confidence, mistake)`, where `clip` is
a list of BGR frames. It extracts per-frame features, max-pools to chunk resolution,
runs the temporal model, and reports the clip's dominant step.

IMPORTANT honesty note (same spirit as the `kinetics` demo backend): the Assembly101
TAS model was trained on Assembly101's own **TSM features**, not raw frames. A correct
*live* backend therefore needs the matching TSM feature extractor. The default
`ResNetFeatureExtractor` here is a STAND-IN (ImageNet ResNet-50) that makes the seam
run end-to-end but does NOT reproduce that feature distribution, so its live labels
are a wiring demo, not meaningful steps. The fully-correct use of the trained model is
**offline scoring on the provided features** (`tas/evaluate.py`); the live seam becomes
meaningful once a matching TSM extractor is supplied via the `extractor=` argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ActivityResult:
    step: str
    confidence: float
    mistake: bool = False


class ResNetFeatureExtractor:
    """STAND-IN per-frame 2048-D extractor (ImageNet ResNet-50 penultimate). Not the
    Assembly101 TSM distribution — see the module docstring."""

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
        self._torch = torch
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()               # -> 2048-D penultimate features
        self.model = model.eval().to(device)
        self.device = device
        self.preprocess = weights.transforms()

    def extract(self, clip) -> np.ndarray:
        import cv2
        torch = self._torch
        imgs = []
        for frame in clip:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)   # C,H,W uint8
            imgs.append(self.preprocess(t))
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)                # (T, 2048)
        return feats.cpu().numpy().astype("float32")


class Assembly101Recognizer:
    name = "assembly101"

    def __init__(self, checkpoint: str, actions_csv: str, device: str = "cpu",
                 extractor: Optional[object] = None, chunk_size: int = 20,
                 max_frames_per_video: int = 1200):
        from .dataset import load_actions_csv
        from .model import load_model
        if not checkpoint:
            raise ValueError("assembly101 backend needs a trained checkpoint (activity.checkpoint)")
        if not actions_csv:
            raise ValueError("assembly101 backend needs actions.csv (activity.actions_csv)")
        self._device = device
        self.chunk_size = int(chunk_size)
        self.max_frames = int(max_frames_per_video)
        self.model = load_model(checkpoint, device)
        _, self.id_to_action = load_actions_csv(actions_csv)
        if extractor is None:
            print("[warn] assembly101: no TSM extractor supplied — using a STAND-IN ResNet "
                  "extractor. Live labels are a wiring demo, not meaningful steps (use offline "
                  "scoring or pass a matching TSM extractor for real accuracy).")
            extractor = ResNetFeatureExtractor(device)
        self.extractor = extractor

    def infer(self, clip: List[np.ndarray]) -> ActivityResult:
        import torch

        from .dataset import chunk_maxpool
        if not clip:
            return ActivityResult("no-input", 0.0, False)
        feats = self.extractor.extract(clip)                         # (T, 2048)
        pooled = chunk_maxpool(feats, self.chunk_size, self.max_frames)   # (T', 2048)
        if pooled.shape[0] == 0:
            return ActivityResult("no-input", 0.0, False)
        x = torch.from_numpy(np.ascontiguousarray(pooled.T)).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self.model(x)[-1, 0]                            # (C, T')
            probs = torch.softmax(logits, dim=0).mean(dim=1)        # (C,) clip-level
            conf, idx = probs.max(0)
        step = self.id_to_action.get(int(idx), f"cls-{int(idx)}")
        return ActivityResult(step=step, confidence=float(conf), mistake=False)
