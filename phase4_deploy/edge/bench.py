"""Measure real inference latency for each exported format — honestly.

Two rules govern everything here, because both are easy to get wrong in a way
that flatters the result:

**1. Measure what the application actually pays.** The headline number is
*end-to-end* wall-clock for a full `predict()` on one BGR frame: letterbox +
BGR->RGB + normalise + host->device copy + forward pass + NMS + box rescale.
Reporting only the forward pass is the most common way benchmarks overstate
throughput — for YOLO at 640, pre/post-processing is a large fraction of the
budget, and it does not disappear when you swap the backend. We report the
end-to-end distribution as the result and print ultralytics' internal
preprocess / inference / postprocess split beside it so the gap is visible.

**2. Never mislabel the hardware.** CUDA work is asynchronous, so every timed
region on a GPU is wrapped in `torch.cuda.synchronize()` (see
`common.time_calls`), otherwise we would be timing kernel *launches*. And for
ONNX we report the execution provider the session is **actually** using, because
ONNX Runtime silently falls back to CPU when a requested GPU provider fails to
load — an unchecked run would publish CPU latency under a GPU label.

Warmup iterations are always discarded: the first calls pay for lazy CUDA
context creation, cuDNN algorithm selection and allocator growth, and are not
representative of steady state.

    python -m phase4_deploy.edge.bench
    python -m phase4_deploy.edge.bench --imgsz 640,480,320 --iters 60
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .common import (Stats, enable_utf8_stdout, file_mb, freeze_environment,
                     onnx_actual_providers, onnxruntime_health, preserve_cuda_env,
                     quiet_onnxruntime, summarize, time_calls)

DEFAULT_ARTIFACTS = "phase4_deploy/artifacts"
DEFAULT_WEIGHTS = "phase2/models/best.pt"


@dataclass
class BenchResult:
    label: str
    path: str = ""
    device: str = "cpu"
    imgsz: int = 640
    size_mb: float = 0.0
    stats: Stats = field(default_factory=Stats)
    speed: Dict[str, float] = field(default_factory=dict)
    providers: List[str] = field(default_factory=list)
    ok: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {"label": self.label, "path": self.path, "device": self.device,
                "imgsz": self.imgsz, "size_mb": round(self.size_mb, 2),
                "latency_ms": self.stats.as_dict(),
                "speed_breakdown_ms": {k: round(v, 2) for k, v in self.speed.items()},
                "providers": self.providers, "ok": self.ok, "note": self.note}


def load_frame(imgsz: int, path: Optional[str] = None):
    """A single BGR frame to benchmark on — a real image when one is available.

    The same frame is reused for every backend so differences are the backend,
    not the input."""
    import numpy as np
    if path and os.path.isfile(path):
        import cv2
        img = cv2.imread(path)
        if img is not None:
            return img
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)


def _sample_image() -> Optional[str]:
    import glob
    for pat in ("data/ppe_download/test/images/*.jpg", "data/dataset/test/images/*.jpg"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def bench_model(path: str, device: str, imgsz: int, frame, warmup: int = 10,
                iters: int = 50, conf: float = 0.35, half: bool = False,
                label: str = "") -> BenchResult:
    """Time `iters` full predict() calls on one frame. Never raises."""
    r = BenchResult(label=label or os.path.basename(path), path=path,
                    device=device, imgsz=imgsz, size_mb=file_mb(path))
    is_cuda = str(device) not in ("cpu",)
    try:
        # Everything ultralytics-touching is inside preserve_cuda_env, so a CPU run
        # cannot disable CUDA for the runs that follow it in this same process.
        with preserve_cuda_env():
            from ultralytics import YOLO
            model = YOLO(path, task="detect")
            kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)
            if half:
                kw["half"] = True

            def call():
                return model.predict(frame, **kw)

            times = time_calls(call, warmup=warmup, iters=iters, cuda_sync=is_cuda)
            r.stats = summarize(times)
            # Stage breakdown (ultralytics' own pre/inference/post split), averaged over a
            # few extra calls so one unrepresentative sample can't skew it. This is
            # context only — the authoritative figure is the end-to-end distribution above.
            acc: Dict[str, List[float]] = {}
            for _ in range(5):
                sp = getattr(call()[0], "speed", None) or {}
                for k, v in sp.items():
                    if isinstance(v, (int, float)):
                        acc.setdefault(k, []).append(float(v))
            r.speed = {k: sum(v) / len(v) for k, v in acc.items() if v}
        r.ok = True
    except Exception as e:
        r.note = f"{type(e).__name__}: {str(e)[:180]}"
    if path.endswith(".onnx"):
        provs, err = onnx_actual_providers(path)
        r.providers = provs
        if err and not r.note:
            r.note = err
    return r


def discover_targets(artifacts: str, weights: str, imgsz: int) -> List[dict]:
    """The models to benchmark: the PyTorch baseline plus whatever was exported."""
    stem = os.path.splitext(os.path.basename(weights))[0]
    out: List[dict] = []
    if os.path.isfile(weights):
        out.append({"label": "pytorch fp32", "path": weights, "kind": "pt"})
    for tag, label in (("fp32", "onnx fp32"), ("fp16", "onnx fp16"), ("int8", "onnx int8")):
        p = os.path.join(artifacts, f"{stem}_{tag}_{imgsz}.onnx")
        if os.path.isfile(p):
            out.append({"label": label, "path": p, "kind": "onnx"})
    ts = os.path.join(artifacts, f"{stem}_{imgsz}.torchscript")
    if os.path.isfile(ts):
        out.append({"label": "torchscript", "path": ts, "kind": "ts"})
    return out


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_suite(artifacts: str, weights: str, imgsz: int, iters: int, warmup: int,
              conf: float, include_cuda: bool, image: Optional[str]) -> List[BenchResult]:
    frame = load_frame(imgsz, image)
    targets = discover_targets(artifacts, weights, imgsz)
    results: List[BenchResult] = []
    for t in targets:
        # ONNX Runtime here is a CPU build; TorchScript/PyTorch can also use CUDA.
        devices = ["cpu"]
        if include_cuda and t["kind"] in ("pt", "ts"):
            devices.append("cuda")
        for dev in devices:
            label = f"{t['label']} [{'cuda' if dev == 'cuda' else 'cpu'}]"
            results.append(bench_model(t["path"], 0 if dev == "cuda" else "cpu", imgsz,
                                       frame, warmup, iters, conf, label=label))
        if t["kind"] == "pt" and include_cuda:
            results.append(bench_model(t["path"], 0, imgsz, frame, warmup, iters, conf,
                                       half=True, label="pytorch fp16 [cuda]"))
    return results


def print_table(results: Sequence[BenchResult], imgsz: int) -> None:
    print(f"\n=== imgsz {imgsz} — end-to-end latency per frame "
          f"(preprocess + inference + NMS) ===")
    print(f"{'backend':<24} {'size MB':>8} {'p50 ms':>8} {'p95 ms':>8} {'mean':>8} "
          f"{'std':>7} {'FPS':>7}   breakdown pre/inf/post ms")
    for r in results:
        if not r.ok:
            print(f"{r.label:<24} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>7} {'-':>7}   {r.note}")
            continue
        b = r.speed
        bd = (f"{b.get('preprocess', 0):.1f}/{b.get('inference', 0):.1f}/"
              f"{b.get('postprocess', 0):.1f}") if b else "-"
        print(f"{r.label:<24} {r.size_mb:>8.1f} {r.stats.p50:>8.2f} {r.stats.p95:>8.2f} "
              f"{r.stats.mean:>8.2f} {r.stats.std:>7.2f} {r.stats.fps:>7.1f}   {bd}")


def main(argv=None) -> int:
    import argparse
    import json

    enable_utf8_stdout()
    freeze_environment()
    quiet_onnxruntime()

    ap = argparse.ArgumentParser(description="Benchmark exported PPE-detector formats")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--imgsz", default="640", help="comma-separated, e.g. 640,480,320")
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per backend")
    ap.add_argument("--warmup", type=int, default=10, help="discarded warmup iterations")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--image", default="", help="benchmark frame (default: a test image)")
    ap.add_argument("--no-cuda", action="store_true", help="CPU only")
    ap.add_argument("--json", default="", help="write results as JSON here")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.weights):
        print(f"[FAIL] weights not found: {args.weights}")
        return 1

    health = onnxruntime_health()
    if health.get("problem"):
        print(f"[warn] ONNX Runtime: {health['problem']}\n       fix: {health['fix']}")

    include_cuda = (not args.no_cuda) and _cuda_available()
    image = args.image or _sample_image()
    print("=" * 78)
    print(" Phase 4 — edge latency benchmark")
    print("=" * 78)
    print(f"host        : {_host_line()}")
    print(f"frame       : {image or '(synthetic random frame)'}")
    print(f"iterations  : {args.iters} timed, {args.warmup} warmup (discarded)")
    print(f"ORT         : {health.get('version')} advertising {health.get('advertised')}")
    if not include_cuda:
        print("cuda        : disabled/unavailable — CPU numbers only")

    payload: Dict[str, object] = {"host": _host_line(), "iters": args.iters,
                                  "warmup": args.warmup, "frame": image or "synthetic",
                                  "onnxruntime": health, "by_imgsz": {}}
    sizes = [int(s) for s in str(args.imgsz).split(",") if str(s).strip()]
    any_ok = False
    all_results: List[BenchResult] = []
    for imgsz in sizes:
        results = run_suite(args.artifacts, args.weights, imgsz, args.iters,
                            args.warmup, args.conf, include_cuda, image)
        if not results:
            print(f"\n[warn] no models found for imgsz {imgsz} — run the exporter first:\n"
                  f"       python -m phase4_deploy.edge.exporter --imgsz {imgsz}")
            continue
        print_table(results, imgsz)
        any_ok = any_ok or any(r.ok for r in results)
        all_results.extend(results)
        payload["by_imgsz"][str(imgsz)] = [r.as_dict() for r in results]

    onnx_res = [r for r in all_results if r.ok and r.path.endswith(".onnx")]
    if onnx_res:
        provs = onnx_res[0].providers
        print(f"\nONNX execution provider actually in use: {provs}")
        if not any("CUDA" in p for p in provs):
            print("  (CPU build of onnxruntime — these are honest CPU numbers, which is the\n"
                  "   relevant target for AR glasses; no GPU claim is being made.)")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if any_ok else 1


def _host_line() -> str:
    import platform
    bits = [platform.processor() or platform.machine()]
    try:
        import torch
        if torch.cuda.is_available():
            bits.append(torch.cuda.get_device_name(0))
        bits.append(f"torch {torch.__version__}")
    except Exception:
        pass
    return " | ".join(b for b in bits if b)


if __name__ == "__main__":
    raise SystemExit(main())
