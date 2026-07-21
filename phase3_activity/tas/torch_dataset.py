"""torch Dataset over Assembly101 TSM features for temporal action segmentation.

One sample per (video, view): reads the video's feature span from an injected feature
store, max-pools to chunk resolution, and pairs it with the chunk-downsampled GT.
The full-resolution per-frame GT is kept for scoring (predictions are upsampled back).

The feature `store` is any object with `.vector(key) -> np.ndarray(2048,)` — the real
`LmdbFeatureStore` at runtime, or an in-memory fake in tests — so this is testable
without the gated dataset.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import dataset as D
from .postprocess import downsample_labels


class TASDataset(Dataset):
    def __init__(self, video_ids: Sequence[str], view: str, store,
                 statistic: Dict[str, Dict[str, list]],
                 gt_frames: Dict[str, List[int]],
                 chunk_size: int = 20, max_frames_per_video: int = 1200):
        self.view = view
        self.store = store
        self.statistic = statistic
        self.gt = gt_frames
        self.chunk_size = chunk_size
        self.max_frames = max_frames_per_video
        # keep only videos we can actually load (present in stats for this view + have GT)
        self.samples: List[str] = [
            v for v in video_ids
            if v in statistic and view in statistic[v] and v in gt_frames and len(gt_frames[v]) > 0]

    def __len__(self) -> int:
        return len(self.samples)

    def gt_frames_for(self, video_id: str) -> List[int]:
        return self.gt[video_id]

    def __getitem__(self, i: int):
        vid = self.samples[i]
        span = self.statistic[vid][self.view]
        feats = D.read_video_features(self.store, vid, self.view, span)      # (T, 2048)
        pooled = D.chunk_maxpool(feats, self.chunk_size, self.max_frames)    # (T', 2048)
        target = downsample_labels(self.gt[vid], self.chunk_size, self.max_frames)  # (T'',)
        n = min(len(pooled), len(target))                                    # align (rounding)
        x = torch.from_numpy(np.ascontiguousarray(pooled[:n].T)).float()     # (2048, T')
        y = torch.from_numpy(target[:n]).long()                             # (T',)
        return x, y, vid, len(self.gt[vid])


def single_video_collate(batch):
    """Batch size 1 (variable-length sequences). Returns (x[1,D,T], y[1,T], vid, orig_len)."""
    x, y, vid, orig_len = batch[0]
    return x.unsqueeze(0), y.unsqueeze(0), vid, orig_len
