"""Revision R1: check that the GPU is not clock-capped before a long training run.

Logs `nvidia-smi` every 0.5 s for 5 s idle, 30 s under a matmul load, and 5 s idle again, and
measures the matmul throughput actually achieved. Written after the first Part A launch ran at
210 MHz (paper_materials/revision_r1/partA/attempt0_gpu_power_capped/NOTE.md).

    .venv-gpu\\Scripts\\python.exe tools\\revision\\gpu_check.py --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import statistics
import subprocess
import time

FIELDS = ("timestamp,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,power.limit,"
          "temperature.gpu,pstate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(a.out, f"nvidia_smi_{stamp}.csv")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] no CUDA device")
    dev = torch.device("cuda:0")

    smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,nounits",
                            "-lms", "500"], stdout=open(log_path, "w", encoding="utf-8"),
                           stderr=subprocess.STDOUT)
    marks = {"idle_before_start": time.time()}
    time.sleep(5)
    a_ = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b_ = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    marks["load_start"] = time.time()
    n = 0
    while time.time() - marks["load_start"] < a.seconds:
        for _ in range(20):
            a_ @ b_
        torch.cuda.synchronize()
        n += 20
    marks["load_end"] = time.time()
    secs = marks["load_end"] - marks["load_start"]
    tflops = 2 * 4096 ** 3 * n / secs / 1e12
    del a_, b_
    torch.cuda.empty_cache()
    time.sleep(5)
    marks["idle_after_end"] = time.time()
    smi.terminate()
    smi.wait(timeout=10)

    rows = []
    with open(log_path, encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if len(r) < 9 or r[0].strip() == "timestamp":
                continue
            try:
                ts = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append({"t": ts, "util": float(r[1]), "sm": float(r[3]),
                             "power": float(r[5]), "temp": float(r[7]), "pstate": r[8].strip()})
            except ValueError:
                continue
    load = [r for r in rows if marks["load_start"] + 2 <= r["t"] <= marks["load_end"]]
    idle = [r for r in rows if r["t"] < marks["load_start"] or r["t"] > marks["load_end"] + 2]

    def summ(rs, k):
        v = [r[k] for r in rs]
        return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)} if v else None

    result = {
        "when": stamp, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "matmul": "fp16 4096x4096", "load_seconds": round(secs, 2), "matmuls": n,
        "achieved_tflops": round(tflops, 2),
        "load": {k: summ(load, k) for k in ("sm", "power", "util", "temp")},
        "idle": {k: summ(idle, k) for k in ("sm", "power", "util", "temp")},
        "nvidia_smi_log": os.path.basename(log_path),
    }
    with open(os.path.join(a.out, f"gpu_check_{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
