"""torch Dataset over Assembly101 TSM features for temporal action segmentation.

One sample per split entry (an assembly or disassembly annotation): reads the
annotation's features from the shared recording's LMDB view over its own absolute
frame range, max-pools to chunk resolution, and pairs it with the chunk-downsampled
GT. The full per-frame GT is kept for scoring (predictions are upsampled back).

The feature `store` needs `.read_many(core, view, frames) -> np.ndarray(T, 2048)` —
the real `LmdbFeatureStore` at runtime, or an in-memory fake in tests — so this is
testable without the gated dataset. Build `samples` with `dataset.build_samples`.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import chunk_maxpool
from .postprocess import downsample_labels


class TASDataset(Dataset):
    def __init__(self, samples: Sequence[dict], view: str, store,
                 chunk_size: int = 20, max_frames_per_video: int = 1200):
        self.samples = list(samples)
        self.view = view
        self.store = store
        self.chunk_size = chunk_size
        self.max_frames = max_frames_per_video
        self._labels_by_name = {s["name"]: s["labels"] for s in self.samples}

    def __len__(self) -> int:
        return len(self.samples)

    def gt_frames_for(self, name: str) -> List[int]:
        return self._labels_by_name[name]

    def __getitem__(self, i: int):
        s = self.samples[i]
        feats = self.store.read_many(s["core"], self.view, s["frames"])     # (T, 2048)
        pooled = chunk_maxpool(feats, self.chunk_size, self.max_frames)      # (T', 2048)
        target = downsample_labels(s["labels"], self.chunk_size, self.max_frames)
        n = min(len(pooled), len(target))
        x = torch.from_numpy(np.ascontiguousarray(pooled[:n].T)).float()     # (2048, T')
        y = torch.from_numpy(np.asarray(target[:n])).long()                 # (T',)
        return x, y, s["name"], len(s["labels"])
