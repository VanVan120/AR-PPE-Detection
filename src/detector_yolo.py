"""YOLO-World open-vocabulary detection pipeline.

Runs a single pre-trained YOLO-World model driven by the configurable prompt
list. Detections are returned with the matched prompt, its mapped dataset class
(if any), confidence, and an absolute-pixel xyxy box. The model is run at a low
confidence floor so the evaluator can integrate a full mAP curve; callers filter
to `confidence_threshold` for display/reporting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Config


@dataclass
class Detection:
    label: str                  # the prompt that fired, e.g. "person without a hard hat"
    class_name: Optional[str]   # mapped dataset class, e.g. "NO-Hardhat" (None = not scored)
    confidence: float
    bbox: list[float]           # [x1, y1, x2, y2] absolute pixels

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
        }


class YoloWorldDetector:
    """Thin wrapper around ultralytics YOLO-World with the prompt list applied."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompts = list(cfg.safety_prompts)
        self.device = cfg.device
        # Imported here so `--check`/dataset-only paths don't pay the import cost.
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(cfg.yolo_model)
        self.model.set_classes(self.prompts)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image. Returns detections >= `yolo_conf_floor`."""
        results = self.model.predict(
            image,
            conf=self.cfg.yolo_conf_floor,
            iou=0.5,
            device=self.device,
            augment=self.cfg.tta,   # test-time augmentation (config knob)
            verbose=False,
        )
        dets: list[Detection] = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, conf, cls):
            label = self.prompts[k] if 0 <= k < len(self.prompts) else str(k)
            dets.append(Detection(
                label=label,
                class_name=self.cfg.class_for_prompt(label),
                confidence=float(c),
                bbox=[float(v) for v in box],
            ))
        return dets


def annotate(image: np.ndarray, detections: list[Detection], threshold: float,
             class_thresholds: Optional[dict[str, float]] = None) -> np.ndarray:
    """Draw boxes for detections at or above their threshold onto a copy of the image.

    Each detection is kept if its confidence meets the per-class override in
    `class_thresholds` (keyed by mapped dataset class) when present, else the
    global `threshold`. Violation classes (NO-*) are drawn red, compliant PPE /
    person classes green, and unscored prompts (no dataset class) orange.
    """
    out = image.copy()

    def _keep(d: Detection) -> bool:
        t = threshold
        if class_thresholds and d.class_name in class_thresholds:
            t = class_thresholds[d.class_name]
        return d.confidence >= t

    shown = [d for d in detections if _keep(d)]
    # Draw larger boxes first so small ones stay legible on top.
    shown.sort(key=lambda d: _area(d.bbox), reverse=True)
    for d in shown:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        color = _color_for(d)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d.class_name or d.label} {d.confidence:.2f}"
        _draw_label(out, text, x1, y1, color)
    return out


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _color_for(d: Detection) -> tuple[int, int, int]:
    name = d.class_name or ""
    if name.upper().startswith("NO-"):
        return (40, 40, 220)      # red (BGR) — violation
    if d.class_name is None:
        return (0, 165, 255)      # orange — detected but not scored
    return (60, 180, 75)          # green — compliant / person


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - base - 2)
    cv2.rectangle(img, (x, y_top), (x + tw + 2, y_top + th + base + 2), color, -1)
    cv2.putText(img, text, (x + 1, y_top + th + 1), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def save_image(image: np.ndarray, path: str) -> None:
    """Write an image to disk, Unicode-path-safe on Windows.

    cv2.imwrite uses a non-Unicode path API on Windows (silently returns False
    for non-ASCII paths). Encode in-memory then write the bytes ourselves.
    """
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"failed to encode image for {path}")
    buf.tofile(path)
