"""Unit tests for the TAS metrics + dataset-loader logic.

No external data or heavy deps required: metrics are pure numpy; the loader's
transform logic is exercised against an in-memory fake feature store. Run:

    python phase3_activity/tests/test_tas.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tas import metrics as M
from tas import dataset as D


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


results = {}


# ---- metrics: perfect prediction -> all 100 ---------------------------------
def test_metrics_perfect():
    gt = ["A", "A", "B", "B", "C", "C", "C"]
    m = M.TASMetrics(overlaps=(0.1, 0.25, 0.5))
    m.add(list(gt), list(gt))
    s = m.summary()
    ok = _approx(s["MoF"], 100.0) and _approx(s["Edit"], 100.0) \
        and all(_approx(s["F1"][o], 100.0) for o in (0.1, 0.25, 0.5))
    results["metrics: perfect -> 100/100/100"] = ok


# ---- metrics: hand-computed case --------------------------------------------
# gt   = A A B B C   (segs A[0,2] B[2,4] C[4,5])
# pred = A A B A C   (segs A[0,2] B[2,3] A[3,4] C[4,5])
#   MoF  = 4/5 = 80        (frame 3 A vs B is the only miss)
#   Edit = (1 - lev([A,B,A,C],[A,B,C]) / 4) * 100 = (1 - 1/4)*100 = 75
#   F1   : tp=3 (A,B,C), fp=1 (spurious A), fn=0 -> P=3/4, R=1 -> F1 = 85.714...
def test_metrics_handcomputed():
    gt = ["A", "A", "B", "B", "C"]
    pred = ["A", "A", "B", "A", "C"]
    m = M.TASMetrics(overlaps=(0.1, 0.25, 0.5))
    m.add(pred, gt)
    s = m.summary()
    exp_f1 = 2 * (3 / 4) * 1.0 / ((3 / 4) + 1.0) * 100.0  # 85.714285...
    ok = _approx(s["MoF"], 80.0) and _approx(s["Edit"], 75.0) \
        and all(_approx(s["F1"][o], exp_f1, 1e-4) for o in (0.1, 0.25, 0.5))
    results["metrics: hand-computed 80 / 75 / 85.71"] = ok


# ---- metrics: back_gd=[] means no class is excluded -------------------------
def test_metrics_no_bg_exclusion():
    # If '0' were treated as background it would be dropped from segments; with the
    # Assembly101 empty bg_class it must be counted. Segments of [0,0,1,1] = two.
    labels, starts, ends = M.get_labels_start_end_time([0, 0, 1, 1], bg_class=())
    results["metrics: bg_class=() keeps class 0"] = (labels == [0, 1] and starts == [0, 2] and ends == [2, 4])


# ---- dataset: exact LMDB key format -----------------------------------------
def test_frame_key():
    k = D.frame_key("nusar_seq7", "C10095_rgb", 5)
    results["dataset: frame_key format"] = (k == "nusar_seq7/C10095_rgb/C10095_rgb_0000000005.jpg")


# ---- dataset: disassembly -> disassebly GT-filename quirk --------------------
def test_gt_filename_quirk():
    ok = (D.gt_label_filename("disassembly_abc") == "disassebly_abc.txt"
          and D.gt_label_filename("assembly_xyz") == "assembly_xyz.txt")
    results["dataset: disassembly->disassebly quirk"] = ok


# ---- dataset: GT densify is end-exclusive -----------------------------------
def test_densify():
    segs = D.parse_coarse_label_file("0\t2\ta\n2\t5\tb\n")   # tab-separated
    dense = D.densify_labels(segs, {"a": 0, "b": 1})
    results["dataset: densify end-exclusive"] = (dense == [0, 0, 1, 1, 1])


# ---- dataset: split-file parse (first tab field) ----------------------------
def test_split_parse():
    body = "seq1\t1\ttoy07\tvehicle\nseq2\t0\ttoy11\ttruck\n\n"
    results["dataset: split parse -> video ids"] = (D.parse_split_file(body) == ["seq1", "seq2"])


# ---- dataset: chunk-maxpool shape + values ----------------------------------
def test_chunk_maxpool():
    # 45 frames, feature dim 4, value = frame index in every channel.
    feats = np.tile(np.arange(45, dtype=np.float32)[:, None], (1, 4))
    pooled = D.chunk_maxpool(feats, chunk_size=20, max_frames_per_video=1200)
    # 3 chunks: [0:20]->19, [20:40]->39, [40:45]->44
    ok = (pooled.shape == (3, 4)
          and _approx(pooled[0, 0], 19) and _approx(pooled[1, 0], 39) and _approx(pooled[2, 0], 44))
    # cap is respected
    capped = D.chunk_maxpool(feats, chunk_size=1, max_frames_per_video=10)
    ok = ok and capped.shape == (10, 4)
    results["dataset: chunk_maxpool shape+values+cap"] = ok


# ---- dataset: read_video_features via a fake in-memory store -----------------
class _FakeStore:
    """Maps frame_key -> a 2048-D vector whose channels all equal the frame index."""
    def vector(self, key: str) -> np.ndarray:
        f = int(key.rsplit("_", 1)[1].split(".")[0])
        return np.full((D.FEATURE_DIM,), float(f), dtype=np.float32)


def test_read_video_features():
    store = _FakeStore()
    arr = D.read_video_features(store, "seqA", "C10095_rgb", (10, 14))  # inclusive [10,14]
    ok = (arr.shape == (5, D.FEATURE_DIM)
          and _approx(arr[0, 0], 10) and _approx(arr[-1, 0], 14))
    # end-to-end: read span then pool
    pooled = D.chunk_maxpool(arr, chunk_size=2, max_frames_per_video=1200)
    ok = ok and pooled.shape == (3, D.FEATURE_DIM) and _approx(pooled[0, 0], 11) and _approx(pooled[-1, 0], 14)
    results["dataset: read span (inclusive) + pool"] = ok


# ---- dataset: fold logic rejects test, unions train_val ----------------------
def test_fold_logic(tmpdir):
    for name, seqs in [
        ("train_coarse_assembly.txt", ["t_asm1", "t_asm2"]),
        ("train_coarse_disassembly.txt", ["t_dis1"]),
        ("val_coarse_assembly.txt", ["v_asm1"]),
        ("val_coarse_disassembly.txt", ["v_dis1"]),
    ]:
        with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"{s}\t1\ttoy\tname" for s in seqs))
    train = D.video_ids_for_fold(tmpdir, "train")
    val = D.video_ids_for_fold(tmpdir, "val")
    train_val = D.video_ids_for_fold(tmpdir, "train_val")
    rejected = False
    try:
        D.video_ids_for_fold(tmpdir, "test")
    except ValueError:
        rejected = True
    ok = (set(train) == {"t_asm1", "t_asm2", "t_dis1"} and set(val) == {"v_asm1", "v_dis1"}
          and set(train_val) == set(train) | set(val) and rejected)
    results["dataset: fold train/val/train_val + test rejected"] = ok


def main() -> int:
    import tempfile
    test_metrics_perfect()
    test_metrics_handcomputed()
    test_metrics_no_bg_exclusion()
    test_frame_key()
    test_gt_filename_quirk()
    test_densify()
    test_split_parse()
    test_chunk_maxpool()
    test_read_video_features()
    with tempfile.TemporaryDirectory() as td:
        test_fold_logic(td)

    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_TAS", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
