"""Fine-tuned (closed-vocabulary) detection pipeline — Track A.

Unlike YOLO-World, a fine-tuned detector has fixed classes baked in by training
on the dataset's own labels. It therefore predicts dataset class indices directly
(no prompts, no prompt->class mapping). This wrapper loads such a model (produced
by train.py) and emits the same `Detection` objects the rest of the pipeline and
the evaluator already consume, so it slots into the identical harness used to
score the zero-shot model — an apples-to-apples comparison.

The model is run at the configurable confidence floor so the evaluator can
integrate a full mAP curve; display/operating-point filtering happens downstream
exactly as for YOLO-World.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config
from .detector_yolo import Detection


class FinetunedDetector:
    """Wrapper around an ultralytics detector fine-tuned on the dataset classes."""

    name = "finetuned"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        from ultralytics import YOLO
        self.model = YOLO(cfg.finetuned_model)
        # The trained model carries its own id->name map; it should match the
        # dataset's classes since it was trained on this data.yaml.
        self.names = dict(self.model.names)

    def detect(self, image: np.ndarray) -> list[Detection]:
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
            name = self.names.get(int(k), str(int(k)))
            # class_name is the dataset class when recognised (so it is scored and
            # coloured correctly); otherwise None (treated as unscored).
            class_name = name if name in self.cfg.dataset_classes else None
            dets.append(Detection(
                label=name,
                class_name=class_name,
                confidence=float(c),
                bbox=[float(v) for v in box],
            ))
        return dets


def build_detector(cfg: Config):
    """Construct the detection backend selected in config.

    `detector_backend: yolo_world` (default) -> zero-shot YOLO-World.
    `detector_backend: finetuned`            -> the model at `finetuned_model`.
    """
    backend = (cfg.detector_backend or "yolo_world").lower()
    if backend == "finetuned":
        return FinetunedDetector(cfg)
    if backend == "yolo_world":
        from .detector_yolo import YoloWorldDetector
        return YoloWorldDetector(cfg)
    raise ValueError(f"unknown detector_backend '{cfg.detector_backend}' "
                     "(expected 'yolo_world' or 'finetuned')")
