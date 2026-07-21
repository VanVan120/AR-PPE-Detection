"""Assembly101 temporal-action-segmentation data loading.

Coded against the source-verified contract of
`assembly-101/assembly101-temporal-action-segmentation` (dataset.py / data_stat.py):

  * Features: one LMDB per camera VIEW under `TSM_features/<view>/`. Per-frame key
    `"{sequence}/{view}/{view}_{frame:010d}.jpg"`, value = raw bytes of a 2048-D
    float32 vector. All frame indices are at 30 fps.
  * `statistic_input.pkl`: {video: {view: [min_frame, max_frame]}} (INCLUSIVE span),
    built once by `build_statistic_input`.
  * Class map: `actions.csv` (header; columns action_id,verb_id,noun_id,action_cls,
    verb_cls,noun_cls) -> 202 coarse classes.
  * Labels: `coarse_labels/<assembly|disassembly>_<seq>.txt`, TAB-separated, no
    header, 3 cols `start_frame end_frame action_cls` (segments; end-EXCLUSIVE).
  * Splits: `coarse_splits/{train,val,test}_coarse_{assembly,disassembly}.txt`,
    tab-separated, video_id = first field. (test GT is withheld — not scorable locally.)
  * Feature aggregation: max-pool over non-overlapping `chunk_size`-frame chunks,
    capped at `max_frames_per_video` pooled steps.

The heavy deps (lmdb, pandas, torch) are imported lazily so the pure transform
helpers here are unit-testable without them. Feature reads go through an injectable
store (`FeatureStore` protocol: `.vector(key) -> np.ndarray(2048,)`), so the loader
logic can be tested against an in-memory fake.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_DIM = 2048
NUM_CLASSES = 202
FRAMES_FORMAT = "/{}/{}_{:010d}.jpg"   # matches the reference verbatim
_DISASSEMBLY = "disassembly"
_DISASSEBLY = "disassebly"             # the reference's GT-filename spelling quirk


# --- key / filename helpers --------------------------------------------------
def frame_key(sequence: str, view: str, frame_no: int) -> str:
    """The exact LMDB key for one frame's feature vector."""
    return sequence + FRAMES_FORMAT.format(view, view, int(frame_no))


def gt_label_filename(video_id: str) -> str:
    """coarse_labels filename for a video id, replicating the repo's 'disassembly'
    -> 'disassebly' misspelling so GT lookups match on disassembly sequences."""
    if _DISASSEMBLY in video_id:
        video_id = video_id.replace(_DISASSEMBLY, _DISASSEBLY)
    return video_id + ".txt"


# --- annotations -------------------------------------------------------------
def load_actions_csv(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """actions.csv -> (action_cls -> id, id -> action_cls). Asserts 202 classes."""
    import pandas as pd
    df = pd.read_csv(path, header=0,
                     names=["action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls"])
    actions_dict = {str(row.action_cls): int(row.action_id) for row in df.itertuples()}
    id_to_action = {int(row.action_id): str(row.action_cls) for row in df.itertuples()}
    if len(actions_dict) != NUM_CLASSES:
        # Not fatal (older/newer dumps), but the model head assumes 202 — warn loudly.
        print(f"[warn] actions.csv has {len(actions_dict)} classes, expected {NUM_CLASSES}")
    return actions_dict, id_to_action


def parse_coarse_label_file(text: str) -> List[Tuple[int, int, str]]:
    """Parse a coarse_labels/*.txt body -> list of (start, end, action_cls). Segments;
    end is exclusive."""
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        segments.append((int(parts[0]), int(parts[1]), parts[2]))
    return segments


def densify_labels(segments: Sequence[Tuple[int, int, str]],
                   actions_dict: Dict[str, int]) -> List[int]:
    """Expand (start, end, cls) segments into a per-frame list of action ids
    (end-exclusive), exactly as the reference does."""
    frame_labels: List[int] = []
    for start, end, cls in segments:
        frame_labels.extend([actions_dict[cls]] * (int(end) - int(start)))
    return frame_labels


def parse_split_file(text: str) -> List[str]:
    """A coarse_splits/*.txt body -> list of video ids (first tab field per line)."""
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(line.split("\t")[0])
    return ids


# --- feature aggregation -----------------------------------------------------
def chunk_maxpool(feats: np.ndarray, chunk_size: int = 20,
                  max_frames_per_video: int = 1200) -> np.ndarray:
    """Max-pool (T, D) features over non-overlapping `chunk_size`-frame chunks,
    capped at `max_frames_per_video` pooled steps. Returns (T', D)."""
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"expected (T, D) features, got shape {feats.shape}")
    T = feats.shape[0]
    n_steps = min(max_frames_per_video, int(np.ceil(T / chunk_size))) if T > 0 else 0
    out = np.zeros((n_steps, feats.shape[1]), dtype=np.float32)
    for k in range(n_steps):
        seg = feats[k * chunk_size:(k + 1) * chunk_size]
        out[k] = seg.max(axis=0)
    return out


def read_video_features(store, sequence: str, view: str,
                        frame_span: Tuple[int, int]) -> np.ndarray:
    """Read the contiguous [min, max] (INCLUSIVE) frame span for one (sequence, view)
    from `store` into a (T, 2048) array. `store.vector(key)` returns a 2048-D array."""
    lo, hi = int(frame_span[0]), int(frame_span[1])
    rows = []
    for f in range(lo, hi + 1):
        vec = store.vector(frame_key(sequence, view, f))
        if vec.shape[0] != FEATURE_DIM:
            raise ValueError(f"feature at {frame_key(sequence, view, f)} has dim "
                             f"{vec.shape[0]}, expected {FEATURE_DIM}")
        rows.append(vec)
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


# --- LMDB-backed feature store (runtime) -------------------------------------
class LmdbFeatureStore:
    """Reads 2048-D float32 vectors from the per-view Assembly101 TSM LMDBs.

    Opens one env per view under `features_root/<view>/`. Lazily imports `lmdb`.
    """

    def __init__(self, features_root: str):
        import lmdb  # lazy: only needed at runtime, not for unit tests
        self._lmdb = lmdb
        self.features_root = features_root
        self._envs: Dict[str, "lmdb.Environment"] = {}

    def _env(self, view: str):
        if view not in self._envs:
            path = os.path.join(self.features_root, view)
            self._envs[view] = self._lmdb.open(path, readonly=True, lock=False,
                                               readahead=False, meminit=False)
        return self._envs[view]

    def vector(self, key: str) -> np.ndarray:
        view = key.split("/")[1]            # "{seq}/{view}/{view}_{f}.jpg"
        with self._env(view).begin(write=False) as txn:
            data = txn.get(key.encode("ascii"))
        if data is None:
            raise KeyError(f"missing feature key: {key}")
        return np.frombuffer(data, dtype="float32")

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


# --- split/fold assembly -----------------------------------------------------
_SPLIT_FILES = {
    "train": ["train_coarse_assembly.txt", "train_coarse_disassembly.txt"],
    "val": ["val_coarse_assembly.txt", "val_coarse_disassembly.txt"],
    "test": ["test_coarse_assembly.txt", "test_coarse_disassembly.txt"],
}


def video_ids_for_fold(splits_dir: str, fold: str) -> List[str]:
    """Return the video ids for a fold. `fold` in {train, train_val, val}; 'test' is
    rejected because the test GT is withheld (not locally scorable), matching the
    reference dataset.py which quits on the test branch."""
    if fold == "test":
        raise ValueError("test split GT is withheld (online leaderboard only) — "
                         "score on the 'val' fold locally.")
    if fold == "train_val":
        want = ["train", "val"]
    elif fold in ("train", "val"):
        want = [fold]
    else:
        raise ValueError(f"unknown fold '{fold}' (expected train | train_val | val)")
    ids: List[str] = []
    for group in want:
        for fname in _SPLIT_FILES[group]:
            path = os.path.join(splits_dir, fname)
            with open(path, "r", encoding="utf-8") as fh:
                ids.extend(parse_split_file(fh.read()))
    return ids


def build_statistic_input(store_root: str, views: Sequence[str],
                          out_path: str) -> Dict[str, Dict[str, list]]:
    """Scan the LMDBs and write statistic_input.pkl = {video: {view: [min, max]}}
    (inclusive span). Ports data_stat.py; asserts each span is contiguous."""
    import lmdb
    import pickle
    stat: Dict[str, Dict[str, list]] = {}
    for view in views:
        env = lmdb.open(os.path.join(store_root, view), readonly=True, lock=False)
        with env.begin() as txn:
            cur = txn.cursor()
            per_video: Dict[str, List[int]] = {}
            for raw_key, _ in cur:
                key = raw_key.decode("ascii")             # "{seq}/{view}/{view}_{f}.jpg"
                seq = key.split("/")[0]
                frame_no = int(key.rsplit("_", 1)[1].split(".")[0])
                per_video.setdefault(seq, []).append(frame_no)
        env.close()
        for seq, frames in per_video.items():
            lo, hi = min(frames), max(frames)
            assert hi - lo + 1 == len(frames), f"non-contiguous frames for {seq}/{view}"
            stat.setdefault(seq, {})[view] = [lo, hi]
    with open(out_path, "wb") as fh:
        pickle.dump(stat, fh)
    return stat
