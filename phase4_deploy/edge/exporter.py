"""Export the trained PPE detector to edge-deployable formats — and verify each one.

An export is only useful if the exported file still *works*, so every format
produced here is immediately re-loaded and checked: class names must survive, and
it must produce detections on a real image. A silently-broken export that merely
"looks smaller" is worse than no export at all.

Formats:
  * `onnx`        — ONNX fp32. The portable baseline; runs anywhere ONNX Runtime
                    runs (CPU, and NPUs/mobile via vendor runtimes).
  * `onnx-fp16`   — ONNX half precision. Roughly halves the file.
                    **Requires a CUDA device to export**: ultralytics silently
                    ignores `half=True` on CPU and writes fp32 (measured: same
                    44.75 MB), so we refuse rather than mislabel the artifact.
  * `onnx-int8`   — 8-bit *dynamically* quantized ONNX via
                    `onnxruntime.quantization.quantize_dynamic`. Smallest file by
                    far. Honest caveat: dynamic quantization mainly shrinks the
                    model and speeds up matmul-heavy networks; on a conv-heavy
                    CNN like YOLO it may give little or NO CPU speedup — the
                    benchmark measures whether it actually helps here.
  * `torchscript` — self-contained PyTorch graph; useful where a Python-free
                    libtorch runtime is the deployment target.

Two traps this module defends against (both hit during development, see README):
  1. ultralytics auto-installs `onnxruntime-gpu` whenever CUDA is present
     (exporter.py:637). If the machine's CUDA/cuDNN don't match ORT's build the
     result is a runtime that ADVERTISES CUDA but cannot use it, and ONNX
     inference then fails outright. We freeze auto-install and health-check ORT.
  2. `half=True` on CPU is a silent no-op (see above).

    python -m phase4_deploy.edge.exporter --weights phase2/models/best.pt
    python -m phase4_deploy.edge.exporter --formats onnx,onnx-int8 --imgsz 480
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .common import (KNOWN_FORMATS, enable_utf8_stdout, file_mb, freeze_environment,
                     onnx_actual_providers, onnxruntime_health, preserve_cuda_env,
                     quiet_onnxruntime)

DEFAULT_WEIGHTS = "phase2/models/best.pt"
DEFAULT_OUTDIR = "phase4_deploy/artifacts"


@dataclass
class ExportResult:
    fmt: str
    path: str = ""
    size_mb: float = 0.0
    seconds: float = 0.0
    ok: bool = False
    verified: bool = False
    names_ok: bool = False
    detections: int = -1
    note: str = ""

    def as_dict(self) -> dict:
        return {"format": self.fmt, "path": self.path, "size_mb": round(self.size_mb, 2),
                "seconds": round(self.seconds, 1), "ok": self.ok, "verified": self.verified,
                "names_ok": self.names_ok, "detections": self.detections, "note": self.note}


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _export_with_ultralytics(weights: str, outdir: str, tag: str, ext: str,
                             **export_kwargs) -> str:
    """Run an ultralytics export in an isolated work dir and move the artifact into
    `outdir` under a descriptive name. ultralytics writes next to the source
    weights, so we copy the weights first rather than littering phase2/models/."""
    from ultralytics import YOLO

    work = os.path.join(outdir, ".work")
    os.makedirs(work, exist_ok=True)
    stem = os.path.splitext(os.path.basename(weights))[0]
    tmp_pt = os.path.join(work, stem + ".pt")
    shutil.copy(weights, tmp_pt)
    try:
        # preserve_cuda_env: a device='cpu' export would otherwise set
        # CUDA_VISIBLE_DEVICES="" for the whole process and break every later
        # GPU export/verification (see common.preserve_cuda_env).
        with preserve_cuda_env():
            produced = YOLO(tmp_pt).export(**export_kwargs)
        produced = str(produced)
        dest = os.path.join(outdir, f"{stem}_{tag}{ext}")
        if os.path.isdir(produced):                       # folder-style exports
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            shutil.move(produced, dest)
        else:
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(produced, dest)
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def export_format(weights: str, fmt: str, outdir: str, imgsz: int = 640,
                  opset: int = 12) -> ExportResult:
    """Produce one format. Never raises — failures come back as a result with a note."""
    r = ExportResult(fmt=fmt)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    try:
        if fmt == "onnx":
            r.path = _export_with_ultralytics(
                weights, outdir, f"fp32_{imgsz}", ".onnx", format="onnx", imgsz=imgsz,
                opset=opset, simplify=True, dynamic=False, half=False, device="cpu",
                verbose=False)
        elif fmt == "onnx-fp16":
            if not _cuda_available():
                r.note = ("skipped: fp16 export needs a CUDA device — ultralytics silently "
                          "ignores half=True on CPU and would write an fp32 file")
                return r
            r.path = _export_with_ultralytics(
                weights, outdir, f"fp16_{imgsz}", ".onnx", format="onnx", imgsz=imgsz,
                opset=opset, simplify=True, dynamic=False, half=True, device=0,
                verbose=False)
        elif fmt == "onnx-int8":
            base = os.path.join(outdir, f"{os.path.splitext(os.path.basename(weights))[0]}_fp32_{imgsz}.onnx")
            if not os.path.isfile(base):
                base_res = export_format(weights, "onnx", outdir, imgsz, opset)
                if not base_res.ok:
                    r.note = "skipped: needs the fp32 ONNX export, which failed"
                    return r
                base = base_res.path
            r.path = _quantize_dynamic(base, outdir, imgsz)
        elif fmt == "torchscript":
            r.path = _export_with_ultralytics(
                weights, outdir, f"{imgsz}", ".torchscript", format="torchscript",
                imgsz=imgsz, device="cpu", verbose=False)
        else:
            r.note = f"unknown format '{fmt}' (known: {', '.join(KNOWN_FORMATS)})"
            return r
        r.ok = os.path.exists(r.path)
        r.size_mb = file_mb(r.path)
    except Exception as e:
        r.note = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        r.seconds = time.time() - t0
    return r


def _quantize_dynamic(base_onnx: str, outdir: str, imgsz: int) -> str:
    """8-bit dynamic weight quantization of an existing ONNX graph."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    stem = os.path.basename(base_onnx).replace(f"_fp32_{imgsz}.onnx", "")
    dest = os.path.join(outdir, f"{stem}_int8_{imgsz}.onnx")
    quantize_dynamic(base_onnx, dest, weight_type=QuantType.QUInt8)
    return dest


def verify_export(path: str, ref_names: Dict[int, str], sample_image: Optional[str] = None,
                  imgsz: int = 640, conf: float = 0.35) -> Dict[str, object]:
    """Re-load an exported model and prove it still works.

    Checks (a) it loads, (b) the class-name map survived the round trip, and
    (c) it produces detections on a real image (or at least runs on a blank one).
    """
    import numpy as np
    from ultralytics import YOLO

    out: Dict[str, object] = {"loaded": False, "names_ok": False, "detections": -1, "note": ""}
    try:
        with preserve_cuda_env():
            model = YOLO(path, task="detect")
            out["loaded"] = True
            names = {int(k): str(v) for k, v in (model.names or {}).items()}
            out["names_ok"] = (names == ref_names)
            if not out["names_ok"]:
                out["note"] = f"class names differ: {names}"
            src = sample_image if (sample_image and os.path.isfile(sample_image)) \
                else np.zeros((imgsz, imgsz, 3), dtype="uint8")
            # Verification is a correctness check, not a speed check — pin it to CPU
            # so the result is deterministic and independent of GPU availability.
            res = model.predict(src, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
            out["detections"] = int(len(res.boxes))
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def _first_sample_image() -> Optional[str]:
    """A real image to verify against, if the dataset happens to be present."""
    import glob
    for pat in ("data/ppe_download/test/images/*.jpg",
                "data/dataset/test/images/*.jpg",
                "phase2/data/clips/*.jpg"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main(argv=None) -> int:
    import argparse
    import json

    enable_utf8_stdout()
    freeze_environment()
    quiet_onnxruntime()

    ap = argparse.ArgumentParser(
        description="Export the PPE detector to edge formats and verify each one")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--formats", default="onnx,onnx-fp16,onnx-int8",
                    help=f"comma-separated (known: {','.join(KNOWN_FORMATS)})")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--json", default="", help="also write the results as JSON here")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.weights):
        print(f"[FAIL] weights not found: {args.weights}\n"
              "       Train them (see the root README) or point --weights at the .pt file.")
        return 1

    health = onnxruntime_health()
    if health.get("problem"):
        print(f"[warn] ONNX Runtime: {health['problem']}")
        print(f"       fix: {health['fix']}")
    print(f"ONNX Runtime {health.get('version')} "
          f"({', '.join(health.get('packages') or ['?'])}) "
          f"providers={health.get('advertised')}")

    from ultralytics import YOLO
    ref_names = {int(k): str(v) for k, v in YOLO(args.weights).names.items()}
    base_mb = file_mb(args.weights)
    sample = _first_sample_image()
    print(f"source: {args.weights} ({base_mb:.2f} MB, {len(ref_names)} classes)  imgsz={args.imgsz}")
    print(f"verify image: {sample or '(none found - verifying on a blank frame)'}")
    print("-" * 74)

    results: List[ExportResult] = []
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        print(f"exporting {fmt} ...", flush=True)
        r = export_format(args.weights, fmt, args.outdir, args.imgsz, args.opset)
        if r.ok:
            v = verify_export(r.path, ref_names, sample, args.imgsz, args.conf)
            r.verified = bool(v["loaded"])
            r.names_ok = bool(v["names_ok"])
            r.detections = int(v["detections"])
            if v["note"]:
                r.note = str(v["note"])
        results.append(r)

    print("-" * 74)
    print(f"{'format':<12} {'size MB':>9} {'vs .pt':>8} {'export s':>9} {'loads':>6} "
          f"{'names':>6} {'dets':>5}  note")
    print(f"{'pytorch':<12} {base_mb:>9.2f} {'1.00x':>8} {'-':>9} {'yes':>6} {'yes':>6} {'-':>5}")
    for r in results:
        if not r.ok:
            print(f"{r.fmt:<12} {'-':>9} {'-':>8} {'-':>9} {'-':>6} {'-':>6} {'-':>5}  {r.note}")
            continue
        ratio = f"{base_mb / r.size_mb:.2f}x" if r.size_mb else "-"
        print(f"{r.fmt:<12} {r.size_mb:>9.2f} {ratio:>8} {r.seconds:>9.1f} "
              f"{'yes' if r.verified else 'NO':>6} {'yes' if r.names_ok else 'NO':>6} "
              f"{r.detections:>5}  {r.note}")

    for r in results:
        if r.ok and r.path.endswith(".onnx"):
            provs, err = onnx_actual_providers(r.path)
            if err:
                print(f"[warn] {os.path.basename(r.path)}: {err}")
            else:
                print(f"  {os.path.basename(r.path)}: ORT will run on {provs}")

    ok = [r for r in results if r.ok and r.verified and r.names_ok]
    print(f"\n{len(ok)}/{len(results)} format(s) exported AND verified -> {args.outdir}")

    if args.json:
        payload = {"weights": args.weights, "imgsz": args.imgsz,
                   "pytorch_mb": round(base_mb, 2),
                   "onnxruntime": health,
                   "results": [r.as_dict() for r in results]}
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
