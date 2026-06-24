"""Discover images and load the evaluation split + ground-truth labels.

Two entry points:
  * `load_eval_split(cfg)` — loads the dataset's test split via `supervision`,
    returning per-image ground-truth boxes (dataset-class indexed) for scoring.
  * `load_adhoc_folder(path)` — loads loose images from a folder (no labels,
    no evaluation) so the pipelines can be run on arbitrary images.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from .config import Config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read an image as BGR, Unicode-path-safe on Windows.

    cv2.imread uses a non-Unicode path API on Windows and returns None for any
    path with non-ASCII characters (common on localized user profiles). Reading
    the bytes ourselves and decoding avoids that.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class Sample:
    """One image plus its ground truth (if available)."""
    name: str                          # file name, e.g. "0001.jpg"
    path: str                          # absolute/relative path on disk
    gt: Optional[sv.Detections] = None  # ground-truth boxes, class_id = dataset class index
    width: int = 0
    height: int = 0

    def read_image(self) -> np.ndarray:
        """Load the image as a BGR numpy array (OpenCV convention)."""
        img = imread_unicode(self.path)
        if img is None:
            raise FileNotFoundError(f"could not read image: {self.path}")
        if not self.height or not self.width:
            self.height, self.width = img.shape[:2]
        return img


def load_eval_split(cfg: Config) -> list[Sample]:
    """Load the evaluation split with ground-truth annotations via supervision.

    Returns samples sorted by file name for deterministic ordering. Ground-truth
    `class_id` values index into `cfg.dataset_classes`.
    """
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=cfg.images_dir,
        annotations_directory_path=cfg.labels_dir,
        data_yaml_path=cfg.data_yaml_path,
    )

    samples: list[Sample] = []
    for image_path, image, annotations in ds:
        h, w = (image.shape[:2] if image is not None else (0, 0))
        samples.append(Sample(
            name=os.path.basename(image_path),
            path=image_path,
            gt=annotations,
            width=int(w),
            height=int(h),
        ))
    samples.sort(key=lambda s: s.name)
    return samples


def load_split(cfg: Config, split: str) -> list[Sample]:
    """Load an arbitrary split (e.g. 'valid', 'test') with ground truth.

    Reuses `load_eval_split` against a shallow copy of the config whose
    `eval_split` points at `split`, so the derived image/label paths follow.
    """
    import copy
    c2 = copy.copy(cfg)
    c2.eval_split = split
    return load_eval_split(c2)


def load_adhoc_folder(folder: str) -> list[Sample]:
    """Load loose images from a folder. No labels -> no evaluation."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"not a folder: {folder}")
    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMG_EXTS)
    ]
    samples: list[Sample] = []
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        samples.append(Sample(name=os.path.basename(p), path=p, gt=None,
                              width=int(w), height=int(h)))
    return samples


def select_review_sample(samples: list[Sample], k: int) -> list[Sample]:
    """Pick up to `k` evenly-spaced samples for the qualitative results page.

    Even spacing (rather than the first k) gives a more representative spread
    across the split without introducing randomness (keeps runs reproducible).
    """
    if k <= 0 or k >= len(samples):
        return list(samples)
    step = len(samples) / k
    idxs = sorted({int(i * step) for i in range(k)})
    return [samples[i] for i in idxs]
