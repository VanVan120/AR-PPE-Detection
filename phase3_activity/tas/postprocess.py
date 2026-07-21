"""Chunk <-> frame resolution conversion for TAS.

Features are max-pooled into non-overlapping `chunk_size`-frame steps, so the model
predicts at the pooled resolution. Two conversions are needed:

  * training targets: densified per-FRAME GT -> one label per pooled chunk
    (`downsample_labels`, majority vote within the chunk)
  * scoring: pooled per-step predictions -> per-FRAME predictions of the original
    length (`upsample_predictions`), so they align with the frame-level GT the
    metrics expect.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def downsample_labels(frame_labels: Sequence[int], chunk_size: int = 20,
                      max_frames_per_video: int = 1200) -> np.ndarray:
    """Per-frame labels -> one label per non-overlapping `chunk_size` chunk (majority
    vote), capped at `max_frames_per_video` steps. Mirrors the pooled feature length."""
    frame_labels = np.asarray(frame_labels)
    T = len(frame_labels)
    n_steps = min(max_frames_per_video, int(np.ceil(T / chunk_size))) if T > 0 else 0
    out = np.zeros(n_steps, dtype=np.int64)
    for k in range(n_steps):
        seg = frame_labels[k * chunk_size:(k + 1) * chunk_size]
        vals, counts = np.unique(seg, return_counts=True)
        out[k] = int(vals[counts.argmax()])
    return out


def upsample_predictions(pooled_labels: Sequence[int], orig_len: int,
                         chunk_size: int = 20) -> np.ndarray:
    """Pooled per-step predictions -> per-frame predictions of length `orig_len`
    (repeat each step `chunk_size` times; trim overflow; pad the tail with the last
    label if the cap truncated the sequence)."""
    pooled_labels = np.asarray(pooled_labels)
    if len(pooled_labels) == 0:
        return np.zeros(orig_len, dtype=np.int64)
    frames = np.repeat(pooled_labels, chunk_size)
    if len(frames) >= orig_len:
        return frames[:orig_len].astype(np.int64)
    pad = np.full(orig_len - len(frames), frames[-1], dtype=np.int64)
    return np.concatenate([frames.astype(np.int64), pad])
