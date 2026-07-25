"""Unit tests for the Phase 4 edge-deployment toolkit.

No model weights, no dataset and no ONNX Runtime required: these cover the pure
logic that the export / benchmark / parity tools are built on — the latency
statistics, the CUDA-environment guard, the box-matching maths used for output
parity, and artifact discovery. The heavy paths (a real export, a real
benchmark) are validated by running the CLIs against the real detector; what is
tested here is everything that can be checked deterministically.

    python phase4_deploy/tests/test_edge.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge import common as C
from edge.parity import iou_matrix, match_detections

results = {}


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---- latency statistics ------------------------------------------------------
def test_summarize_math():
    # 1..100 -> mean 50.5, p50 50.5, p95 95.05 (numpy linear interpolation)
    s = C.summarize(list(range(1, 101)))
    ok = (s.n == 100 and _approx(s.mean, 50.5) and _approx(s.p50, 50.5)
          and _approx(s.min, 1.0) and _approx(s.max, 100.0)
          and _approx(s.p95, np.percentile(np.arange(1, 101), 95)))
    results["stats: mean/p50/p95/min/max"] = ok


def test_summarize_fps_uses_median():
    # FPS must come from the MEDIAN, so one slow outlier can't distort throughput.
    s = C.summarize([10.0] * 99 + [10_000.0])
    results["stats: fps derived from p50 (outlier-robust)"] = _approx(s.fps, 100.0)


def test_summarize_empty():
    s = C.summarize([])
    results["stats: empty input is safe"] = (s.n == 0 and s.fps == 0.0)


def test_time_calls_discards_warmup():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    times = C.time_calls(fn, warmup=5, iters=7, cuda_sync=False)
    # 5 warmup + 7 timed calls happened, but only the 7 timed ones are returned.
    results["timing: warmup runs but is discarded"] = (calls["n"] == 12 and len(times) == 7)


def test_time_calls_measures_real_duration():
    import time

    def fn():
        time.sleep(0.005)

    s = C.summarize(C.time_calls(fn, warmup=1, iters=5, cuda_sync=False))
    # a 5 ms sleep must register as >= ~4 ms (timer resolution slack)
    results["timing: measures real elapsed time"] = (s.p50 >= 4.0)


# ---- the CUDA-environment guard ---------------------------------------------
def test_preserve_cuda_env_restores():
    key = "CUDA_VISIBLE_DEVICES"
    prev = os.environ.get(key)
    os.environ[key] = "0"
    with C.preserve_cuda_env():
        os.environ[key] = ""              # what ultralytics select_device('cpu') does
    restored = os.environ.get(key) == "0"
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev
    results["cuda-guard: restores CUDA_VISIBLE_DEVICES"] = restored


def test_preserve_cuda_env_removes_when_absent():
    key = "CUDA_VISIBLE_DEVICES"
    prev = os.environ.pop(key, None)
    with C.preserve_cuda_env():
        os.environ[key] = ""
    absent = key not in os.environ
    if prev is not None:
        os.environ[key] = prev
    results["cuda-guard: unsets var that did not exist before"] = absent


# ---- IoU / detection matching (output parity) --------------------------------
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10],        # identical -> 1.0
                  [5, 0, 15, 10],        # half overlap -> 50/150 = 1/3
                  [20, 20, 30, 30]], dtype=float)   # disjoint -> 0
    m = iou_matrix(a, b)
    results["iou: identical/partial/disjoint"] = (
        m.shape == (1, 3) and _approx(m[0, 0], 1.0)
        and _approx(m[0, 1], 1.0 / 3.0, 1e-6) and _approx(m[0, 2], 0.0))


def test_iou_empty():
    results["iou: empty inputs are safe"] = (
        iou_matrix(np.zeros((0, 4)), np.zeros((2, 4))).shape == (0, 2))


def test_match_identical():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    cls = np.array([0, 1])
    conf = np.array([0.9, 0.8])
    m = match_detections(boxes, cls, conf, boxes, cls, conf)
    results["match: identical detections -> full match, zero drift"] = (
        m["matched"] == 2 and m["unmatched_a"] == 0 and _approx(m["max_conf_delta"], 0.0))


def test_match_reports_conf_drift():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    cls = np.array([0])
    m = match_detections(boxes, cls, np.array([0.90]), boxes, cls, np.array([0.75]))
    results["match: reports confidence drift"] = (
        m["matched"] == 1 and _approx(m["max_conf_delta"], 0.15, 1e-6))


def test_match_class_must_agree():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    # same box, different class -> must NOT match
    m = match_detections(boxes, np.array([0]), np.array([0.9]),
                         boxes, np.array([1]), np.array([0.9]))
    results["match: different class does not match"] = (m["matched"] == 0)


def test_match_low_iou_rejected():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[7, 0, 17, 10]], dtype=float)          # IoU ~0.21 < 0.9
    m = match_detections(a, np.array([0]), np.array([0.9]),
                         b, np.array([0]), np.array([0.9]), iou_thresh=0.9)
    results["match: below IoU threshold is unmatched"] = (
        m["matched"] == 0 and m["unmatched_a"] == 1 and m["unmatched_b"] == 1)


def test_match_no_double_counting():
    # two predictions on one ground-truth box: each target may be used once
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10]], dtype=float)
    m = match_detections(a, np.array([0, 0]), np.array([0.9, 0.8]),
                         b, np.array([0]), np.array([0.9]))
    results["match: a target is matched at most once"] = (
        m["matched"] == 1 and m["unmatched_a"] == 1)


# ---- artifact discovery / misc ----------------------------------------------
def test_discover_targets():
    from edge.bench import discover_targets
    with tempfile.TemporaryDirectory() as td:
        art = os.path.join(td, "artifacts")
        os.makedirs(art)
        wt = os.path.join(td, "best.pt")
        open(wt, "wb").write(b"x")
        for n in ("best_fp32_640.onnx", "best_int8_640.onnx", "best_fp32_320.onnx"):
            open(os.path.join(art, n), "wb").write(b"x")
        found = {t["label"] for t in discover_targets(art, wt, 640)}
        # only the 640 artifacts (plus the .pt baseline) belong to imgsz 640
        results["discover: picks the right imgsz artifacts"] = (
            found == {"pytorch fp32", "onnx fp32", "onnx int8"})


def test_file_mb():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "f.bin")
        with open(p, "wb") as fh:
            fh.write(b"\0" * 2_000_000)
        results["file_mb: reports size in MB"] = _approx(C.file_mb(p), 2.0, 0.01)


def test_onnxruntime_health_shape():
    h = C.onnxruntime_health()
    results["health: returns the expected report shape"] = (
        isinstance(h, dict) and {"installed", "advertised", "problem", "fix"} <= set(h))


def main() -> int:
    test_summarize_math()
    test_summarize_fps_uses_median()
    test_summarize_empty()
    test_time_calls_discards_warmup()
    test_time_calls_measures_real_duration()
    test_preserve_cuda_env_restores()
    test_preserve_cuda_env_removes_when_absent()
    test_iou_matrix()
    test_iou_empty()
    test_match_identical()
    test_match_reports_conf_drift()
    test_match_class_must_agree()
    test_match_low_iou_rejected()
    test_match_no_double_counting()
    test_discover_targets()
    test_file_mb()
    test_onnxruntime_health_shape()

    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_EDGE", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
