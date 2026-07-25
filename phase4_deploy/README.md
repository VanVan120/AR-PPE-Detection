# Phase 4 — Edge Deployment Readiness (export, quantization, latency, accuracy parity)

**The question this phase answers:** the Phase 1 detector is accurate (90%+ on every
metric) and Phase 2 runs it live — but AR glasses do not contain an RTX GPU. *Can this
model actually run on device, and what does it cost in accuracy to make it fast enough?*

Phase 4 turns that into measurements rather than opinion: export the detector to
edge-deployable formats, benchmark each one **end-to-end**, and prove each one is still
**accurate** on the same held-out test split that produced the Phase 1 benchmark.

> Every number below was measured on this machine by the commands shown. Nothing is
> estimated. The honest limits of these measurements are stated in
> [Caveats](#caveats--what-these-numbers-do-and-dont-prove) — read that before quoting them.

---

## Status

| # | Milestone | State |
|---|---|---|
| **M1** | Export to ONNX (fp32/fp16), INT8 quantization, TorchScript — each **re-loaded and verified** (class names survive, still detects) | ✅ `edge/exporter.py` |
| **M2** | End-to-end latency benchmark with correct CUDA sync, discarded warmup, p50/p95/std, and execution-provider honesty | ✅ `edge/bench.py` |
| **M3** | Accuracy parity: detection-level agreement + mAP on the real test split | ✅ `edge/parity.py` |
| **M4** | Exported models usable from Phase 2 (`weights: …onnx`), with the auto-install trap disarmed | ✅ `phase2/src/detector.py` |
| **M5** | Unit tests that need no weights and no dataset | ✅ `tests/test_edge.py` (17/17) |

```bash
python phase4_deploy/tests/test_edge.py          # ALL_EDGE True  (no weights/data needed)
```

---

## The headline result

> **Real-time PPE detection is achievable on a CPU alone.** Exported to ONNX and run at
> 320 px, the detector reaches **31.8 FPS (31.5 ms/frame) on the CPU**, at
> **mAP50 0.969** — versus 0.988 for the 640 px PyTorch baseline, which manages only
> 8.2 FPS on that same CPU. That is **3.9× faster for 1.9 mAP50 points** — which
> decomposes cleanly into **3.1× from dropping 640 → 320 px** and **1.26× from the ONNX
> export itself** (3.08 × 1.26 = 3.88).

The GPU numbers are the workstation upper bound (TorchScript on CUDA: **80 FPS**), but the
CPU column is the one that matters for glasses-class hardware, and it clears real-time.

---

## 1. Export — size and integrity

`python -m phase4_deploy.edge.exporter --weights phase2/models/best.pt`

| Format | Size | vs `.pt` | Loads | Names survive | Detects |
|---|---|---|---|---|---|
| PyTorch `.pt` (baseline) | 22.52 MB | 1.00× | — | — | — |
| ONNX fp32 | 44.75 MB | **0.50×** (bigger) | ✅ | ✅ | ✅ |
| ONNX fp16 | 22.41 MB | 1.00× | ✅ | ✅ | ✅ |
| ONNX INT8 (dynamic) | **11.49 MB** | **1.96×** | ✅ | ✅ | ✅ |
| TorchScript | 44.92 MB | 0.50× | ✅ | ✅ | ✅ |

**ONNX fp32 is *larger* than the `.pt`** — the checkpoint stores fp16 weights, ONNX fp32
does not. Export is not automatically a compression step; only fp16/INT8 shrink it.

Every row is verified, not assumed: each artifact is re-loaded, its class map compared to
the original, and run on a real image. A smaller file that silently stopped detecting
would otherwise look like a win.

## 2. Latency — measured end-to-end

`python -m phase4_deploy.edge.bench --imgsz 640,480,320 --iters 40 --warmup 15`

Median (p50) milliseconds for a **complete** `predict()` on one 
frame — letterbox + normalise + forward pass + NMS + box rescale — 40 timed iterations
after 15 discarded warmups, CUDA-synchronised.

| Backend | 640 px | 480 px | **320 px** |
|---|---|---|---|
| PyTorch fp32 · CPU | 121.9 ms (8.2 FPS) | 74.4 ms (13.4) | 39.5 ms (25.3) |
| **ONNX fp32 · CPU** | 108.8 ms (9.2 FPS) | 62.4 ms (16.0) | **31.5 ms (31.8 FPS)** |
| ONNX fp16 · CPU | 111.7 ms (9.0) | 62.7 ms (16.0) | 30.9 ms (32.3) |
| ONNX INT8 · CPU | 149.1 ms (6.7) | 90.4 ms (11.1) | 46.6 ms (21.5) |
| TorchScript · CPU | 117.9 ms (8.5) | — | — |
| PyTorch fp32 · CUDA | 19.3 ms (51.8) | 18.7 ms (53.4) | 20.4 ms (49.0) |
| PyTorch fp16 · CUDA | 20.4 ms (48.9) | 15.2 ms (66.0) | 16.1 ms (62.2) |
| **TorchScript · CUDA** | **12.5 ms (80.0 FPS)** | — | — |

Three findings worth stating plainly, because two of them contradict common assumptions:

- **INT8 is smaller but *slower*.** It is the smallest artifact (11.5 MB, 3.9× smaller than
  fp32 ONNX) yet consistently **~1.4× slower** on CPU at every resolution. Dynamic
  quantization mainly helps matmul-heavy networks; on a conv-heavy detector the
  quantize/dequantize overhead outweighs the cheaper arithmetic. **Choose INT8 here only
  if storage/memory is the binding constraint — not for speed.**
- **fp16 buys nothing on CPU.** x86 CPUs have no native fp16 compute path, so it matches
  fp32 (it *is* a real 2× size win, and would matter on hardware with fp16 units).
- **The GPU is overhead-bound, not compute-bound.** CUDA latency barely moves between
  640 px and 320 px (19.3 → 20.4 ms) because preprocessing, NMS and launch overhead
  dominate. Shrinking the input only helps the CPU path — which is exactly the path that
  needs help. The `pre/inf/post` breakdown printed by the tool makes this visible.

## 3. Accuracy parity — does it still work?

**Detection-level agreement** (`--mode outputs`, 25 images, matched at IoU ≥ 0.9):

| Export | Detections matched | Agreement | Max Δconfidence |
|---|---|---|---|
| ONNX fp32 | 142 / 142 | 100% | **0.000001** |
| ONNX fp16 | 142 / 142 | 100% | 0.0025 |
| ONNX INT8 | 141 / 142 | 99.3% | **0.2079** |

**Metric parity** (`--mode metrics`, 300-image random subset of the *test* split, standard
mAP protocol at `conf=0.001`):

| Model | mAP50 @640 | mAP50-95 @640 | mAP50 @320 | mAP50-95 @320 |
|---|---|---|---|---|
| PyTorch (baseline) | 0.9882 | 0.8842 | 0.9767 | 0.8169 |
| ONNX fp32 | 0.9857 (−0.0025) | 0.8744 | 0.9693 (−0.0074) | 0.7951 |
| ONNX fp16 | 0.9857 (−0.0025) | 0.8746 | 0.9693 (−0.0074) | 0.7944 |
| ONNX INT8 | 0.9817 (−0.0065) | 0.8577 | 0.9671 (−0.0096) | 0.7815 |

- **Export is essentially free**: fp32/fp16 cost ~0.25 mAP50 points, and fp32 is
  numerically identical detection-for-detection (Δconf ~1e-6).
- **INT8's cost is real but modest** at mAP50 (−0.0065) — the confidence drift (0.21) is
  the more telling signal, and it is what loses the one unmatched detection.
- **Downscaling costs more than exporting.** 640 → 320 costs ~0.016 mAP50 and ~0.08
  **mAP50-95** — the localisation metric suffers much more than the detection metric,
  which is expected when you halve the resolution. If precise box geometry matters,
  that's the trade to watch; if "is there an unhelmeted person" is the question, 320 px
  holds up well.

## Recommendation

| Deployment target | Use | Why |
|---|---|---|
| **AR glasses / CPU-only edge box** | **ONNX fp32 @ 320 px** | 31.8 FPS on CPU, mAP50 0.969 — the only combination that clears real-time without a GPU |
| Tight storage/memory budget | ONNX INT8 | 11.5 MB (3.9× smaller) — but accept ~1.4× slower |
| Edge box **with** an NVIDIA GPU | TorchScript · CUDA | fastest measured (80 FPS) |
| Development / retraining | PyTorch `.pt` | keep as the source of truth |

Use it from Phase 2 by pointing `weights` at the exported file:

```yaml
# phase2/config.yaml
weights: "../phase4_deploy/artifacts/best_fp32_320.onnx"
imgsz: 320
```
`python run.py --check` validates it. The detector auto-detects an exported graph, states
`task=detect` explicitly, and falls back to CPU with a clear message if the installed ONNX
Runtime has no usable CUDA provider (rather than crashing mid-session).

---

## Caveats — what these numbers do and don't prove

Stated plainly so nothing here is over-read:

1. **This is not AR-glasses silicon.** Measured on a laptop: AMD Ryzen 5 7640HS CPU +
   RTX 4050 Laptop GPU. Real glasses/NPU hardware will differ — often slower per core,
   sometimes much faster via a dedicated NPU. **The transferable artifact is the harness,
   not the constants**: re-run `edge/bench.py` and `edge/parity.py` on the target device
   to get its numbers. That is precisely why they are CLIs and not a hard-coded table.
2. **Single frame, batch = 1, one process.** This models the live per-frame path (which is
   how Phase 2 runs), not batched throughput. Nothing else was contending for the machine,
   and laptops thermally throttle — an early run of the same benchmark on a cold/busy
   process reported 2× the latency with std ≈ 90 ms, which is why warmup is 15 iterations
   and **p95 and std are always reported next to p50**. Treat single runs as indicative
   and re-run before quoting.
3. **The parity numbers use a 300-image random subset** of the 4,190-image test split
   (a full CPU-backend validation of every format × resolution takes hours). They are
   directly comparable *to each other* because every model saw the identical subset and
   identical arguments — but they are **not** the same measurement as Phase 1's headline
   96.1%, which used the full split. For a report-grade figure run `--limit 0`.
4. **FPS is derived from median latency**, so a single outlier cannot inflate it. It is
   detector throughput only — it excludes tracking, compliance logic and overlay drawing,
   which Phase 2 adds on top.
5. **No GPU claim is made for ONNX.** The installed ONNX Runtime is a CPU build; the tool
   reports the provider actually in use and would refuse to label a CPU run as GPU.

## Traps found while building this (all defended against in code)

These cost real debugging time and would bite anyone repeating the work:

1. **ultralytics silently pip-installs `onnxruntime-gpu`** whenever `torch.cuda.is_available()`
   (`exporter.py:637`, triggered because `simplify=True` is the default). On a machine
   whose CUDA/cuDNN don't match that build, the result is a runtime that *advertises*
   `CUDAExecutionProvider` but cannot load it — and ultralytics' AutoBackend then binds GPU
   buffers to a CPU session and **ONNX inference fails outright**. Observed live: loading an
   ONNX model from Phase 2 downloaded 241 MB and broke inference. All Phase 4 tools *and*
   `phase2/src/detector.py` now set `YOLO_AUTOINSTALL=false`.
   *Symptom:* `RuntimeError: no data transfer registered for copying tensors from Device:[DeviceType:1…]`.
   *Fix:* `pip uninstall -y onnxruntime-gpu && pip install --force-reinstall onnxruntime`.
2. **`select_device("cpu")` disables CUDA for the whole process.** It sets
   `CUDA_VISIBLE_DEVICES=""` (`torch_utils.py:182`) and never restores it, so one CPU-targeted
   call makes every later GPU call fail with `AssertionError: Invalid device id` on a
   perfectly healthy GPU. Every ultralytics call here is wrapped in
   `common.preserve_cuda_env()`.
3. **`half=True` on CPU is a silent no-op** — you get an fp32 file at full size with no
   error. The exporter refuses fp16 without CUDA rather than mislabel the artifact.
4. **ONNX Runtime falls back to CPU silently** when a requested GPU provider fails, and
   ultralytics prints the *requested* provider, not the active one. `onnx_actual_providers()`
   reports what the session really uses.
5. **`nms=True` bakes `conf=0.25`/`iou=0.7` into the graph**, which caps mAP and cannot be
   lowered at validation time (source-verified: 0.928 vs 0.981 mAP50 on identical weights).
   All exports here use `nms=False`.
6. **ONNX export does not support `int8`** in ultralytics 8.3.233 (it raises), so INT8 is
   produced with `onnxruntime.quantization.quantize_dynamic`. Ultralytics' metadata survives
   quantization, so class names are preserved.

`opset=12` is used deliberately (rather than the 8.3.233 default of 22) for the widest
compatibility with older/embedded ONNX runtimes and vendor NPU toolchains.

## Commands

```bash
pip install -r phase4_deploy/requirements.txt

# 1. export + verify every format
python -m phase4_deploy.edge.exporter --weights phase2/models/best.pt --imgsz 320

# 2. latency, across resolutions
python -m phase4_deploy.edge.bench --imgsz 640,480,320 --iters 40 --warmup 15

# 3. accuracy: fast breakage check, then the authoritative metric run
python -m phase4_deploy.edge.parity --mode outputs
python -m phase4_deploy.edge.parity --mode metrics --imgsz 640,320 --limit 300
python -m phase4_deploy.edge.parity --mode metrics --limit 0      # full split
```
Artifacts and JSON results land in `phase4_deploy/artifacts/` (gitignored — regenerate
with the commands above). Add `--json <path>` to any tool to save machine-readable output.
