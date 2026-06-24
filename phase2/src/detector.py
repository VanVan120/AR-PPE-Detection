"""Per-frame inference with the trained YOLO detector.

Loads the Phase 1 weights once and runs them on each frame, returning a
`supervision.Detections` (the common currency the tracker and compliance modules
consume). The trained model carries its own class id->name map, so class names
are read from the model, never hardcoded.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import supervision as sv


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Coerce a frame to 3-channel BGR (YOLO expects 3 channels). cv2 capture always
    yields BGR, but a grayscale / BGRA frame from another source would otherwise raise
    a low-level torch channel error; convert it, or fail with a clear message."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"detect() expected a 2D grayscale or HxWx{{3,4}} frame, got shape {frame.shape}")


def _device_arg(device: str):
    """ultralytics accepts 0 / 'cpu' / 'mps' / 'cuda:0'. Normalise 'cuda' -> 0."""
    return 0 if device == "cuda" else device


class Detector:
    def __init__(self, weights_path: str, device: str = "cpu",
                 confidence_threshold: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = _device_arg(device)
        self.conf = float(confidence_threshold)
        self.imgsz = int(imgsz)
        # id -> name, straight from the trained weights.
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}

    def class_ids_for(self, names) -> list[int]:
        """Resolve class NAMES to the model's integer ids (skips names not in the model)."""
        wanted = set(names)
        return [i for i, n in self.names.items() if n in wanted]

    def warmup(self) -> None:
        """Run one inference so lazy CUDA / model init isn't billed to frame 1's FPS."""
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(blank, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a BGR frame. Returns detections at/above the threshold."""
        result = self.model.predict(
            _to_bgr(frame), conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
