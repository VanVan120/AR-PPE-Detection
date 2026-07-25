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

> **Real-time PPE detection is achievable on a CPU alone — at 480 px, not 320 px.**
> Exported to ONNX and run at **480 px**, the detector reaches **16.0 FPS (62.4 ms/frame)
> on the CPU** at **mAP50 0.9809**, against **0.9842** for the 640 px PyTorch baseline that
> manages only **8.2 FPS** on the same CPU. That is **1.95× faster for 0.3 mAP50 points** —
> very nearly free.
>
> **320 px is the speed-first option and it is not free for safety.** It does reach
> **31.8 FPS**, but recall on `No-Helmet` — the class the whole system exists to catch —
> falls from **0.938 → 0.864**, i.e. missed bare heads roughly **double** (6.2% → 13.6%).
> Aggregate mAP50 hides this; see [the per-class table](#3-accuracy-parity--does-it-still-work).
> **Prefer 480 px unless you have measured that the extra frames matter more than the
> misses.**
>
> All accuracy figures are measured on the **full 4,190-image test split** — the same one
> behind the Phase 1 benchmark. (Sanity check: this harness scores the 640 px PyTorch model
> at mAP50 0.9842, against Phase 1's independently reported 98.2% — the protocols agree.)

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

**Metric parity — authoritative**, on the **full 4,190-image test split** at the deployment
resolution, standard mAP protocol (`conf=0.001`):
`python -m phase4_deploy.edge.parity --mode metrics --imgsz 320 --limit 0`

| Model (320 px) | mAP50 | mAP50-95 | Precision | Recall | ΔmAP50 |
|---|---|---|---|---|---|
| PyTorch (baseline) | 0.9683 | 0.8046 | 0.9467 | 0.9266 | — |
| **ONNX fp32** | **0.9635** | 0.7871 | 0.9329 | 0.9258 | **−0.0048** |
| ONNX fp16 | 0.9635 | 0.7873 | 0.9331 | 0.9258 | −0.0048 |
| ONNX INT8 | 0.9590 | 0.7735 | 0.9290 | 0.9236 | −0.0094 |

**Resolution is the real trade — and aggregate metrics hide the safety-relevant part.**
Full split, ONNX fp32 vs the 640 px PyTorch baseline:

| Config | mAP50 | mAP50-95 | Recall (all) | **`No-Helmet` recall** | CPU FPS |
|---|---|---|---|---|---|
| PyTorch @ 640 (baseline) | 0.9842 | 0.8727 | 0.9623 | **0.938** | 8.2 |
| **ONNX fp32 @ 480** | **0.9809** | 0.8687 | 0.9573 | **0.920** | **16.0** |
| ONNX fp32 @ 320 | 0.9635 | 0.7871 | 0.9258 | **0.864** | 31.8 |

Per-class recall at 320 px, showing why the aggregate is misleading:

| Class | R @640 | R @320 | Δ |
|---|---|---|---|
| **No-Helmet** | 0.938 | **0.864** | **−0.074** |
| No-Vest | 0.961 | 0.926 | −0.035 |
| Helmet | 0.958 | 0.926 | −0.032 |
| Vest | 0.978 | 0.952 | −0.026 |
| Person | 0.977 | 0.960 | −0.017 |

- **Export is essentially free**: fp32/fp16 cost ~0.005 mAP50, and fp32 is numerically
  identical detection-for-detection (Δconf ~1e-6).
- **INT8's accuracy cost is small but real** (−0.009 mAP50). The confidence drift (0.21) is
  the more telling signal, and it is what loses the one unmatched detection above.
- **Downscaling is where the accuracy goes, and it is not uniform.** Aggregate recall moves
  little (0.9623 → 0.9258 at 320 px), but that average is dominated by easy classes.
  `No-Helmet` — the safety-critical one, and already the hardest class in the Phase 1
  benchmark — degrades **~4× more than `Person`**. Bare heads are small, and small objects
  are exactly what you lose first when you halve the input. **480 px keeps almost all of
  it (0.920) at 16 FPS; 320 px does not.**
- 640 → 320 also costs ~0.085 **mAP50-95** vs ~0.016 mAP50, i.e. boxes get looser faster
  than detections get lost.

A 300-image-subset sweep across resolutions (`--limit 300`, in `artifacts/parity.json`)
shows the same ordering but reads ~0.6 mAP50 points optimistic — which is why every number
above comes from the full split.

## Recommendation

| Deployment target | Use | Why |
|---|---|---|
| **AR glasses / CPU-only edge box** | **ONNX fp32 @ 480 px** | 16.0 FPS on CPU at mAP50 0.9809 and `No-Helmet` recall 0.920 — real-time *and* it keeps the safety-critical class |
| Speed-critical, misses acceptable | ONNX fp32 @ 320 px | 31.8 FPS, but `No-Helmet` recall drops to 0.864 (~2× more missed bare heads) — justify this before choosing it |
| Tight storage/memory budget | ONNX INT8 | 11.5 MB on disk (3.9× smaller) — but ~1.4× *slower*, and it is weight-only quantization so runtime memory does not shrink proportionally |
| Edge box **with** an NVIDIA GPU | TorchScript · CUDA | fastest measured (80 FPS) |
| Development / retraining | PyTorch `.pt` | keep as the source of truth |

Use it from Phase 2 by pointing `weights` at the exported file:

```yaml
# phase2/config.yaml
weights: "../phase4_deploy/artifacts/best_fp32_480.onnx"
imgsz: 480
```
`python run.py --check` validates it. The detector auto-detects an exported graph, states
`task=detect` explicitly, and falls back to CPU with a clear message if the installed ONNX
Runtime has no usable CUDA provider (rather than crashing mid-session).

---

## Caveats — what these numbers do and don't prove

Stated plainly so nothing here is over-read:

1. **This is not AR-glasses silicon, and "CPU" here means 12 threads of it.** Measured on
   a laptop: AMD Ryzen 5 7640HS (6 cores / **12 logical threads**) + RTX 4050 Laptop GPU.
   Both PyTorch and ONNX Runtime use *all* cores by default, so the CPU figures are
   all-core numbers on a desktop-class x86 CPU — glasses-class or embedded ARM cores are
   typically far weaker per core and fewer, while a dedicated NPU can be much faster.
   Treat the CPU column as "a modern laptop CPU", **not** as a prediction for the target
   device. The benchmark now prints its core/thread budget in the header for exactly this
   reason. **The transferable artifact is the harness, not the constants**: re-run
   `edge/bench.py` and `edge/parity.py` on the target hardware. That is precisely why they
   are CLIs and not a hard-coded table.
2. **Single frame, batch = 1, one process.** This models the live per-frame path (which is
   how Phase 2 runs), not batched throughput. Nothing else was contending for the machine,
   and laptops thermally throttle — an early run of the same benchmark on a cold/busy
   process reported 2× the latency with std ≈ 90 ms, which is why warmup is 15 iterations
   and **p95 and std are always reported next to p50**. Treat single runs as indicative
   and re-run before quoting.
3. **The headline accuracy figures use the full 4,190-image test split**, so they are
   report-grade and directly comparable to Phase 1's benchmark. The extra
   *cross-resolution* sweep (640 vs 320 for every format) uses a 300-image subset, because
   a full CPU-backend validation of every format × resolution takes hours — that sweep is
   internally consistent (identical images and arguments for every model) but reads ~0.6
   mAP50 points optimistic, and is labelled as such wherever it appears.
4. **FPS is derived from median latency**, so a single outlier cannot inflate it. It is
   detector throughput only — it excludes tracking, compliance logic and overlay drawing,
   which Phase 2 adds on top.
5. **No GPU claim is made for ONNX.** The installed ONNX Runtime is a CPU build. The tool
   reads the provider list off the **live session that was actually timed** (via the
   ultralytics backend), so the label always describes the measurement. If that session
   can't be reached it falls back to opening an equivalent one requesting the *same*
   providers the timed run used — never the full available list, which would report CUDA
   for a run executed on CPU.
6. **The fp16 numbers are size-and-speed results, not an fp16-hardware accuracy result.**
   The ONNX Runtime CPU provider has no native fp16 compute path, so it up-converts and
   executes in fp32. The measured accuracy therefore says the *weights* survived the
   conversion — it does not predict accuracy on hardware that genuinely computes in fp16.
7. **INT8 here is weight-only dynamic quantization.** 11.5 MB is a *disk* size; activations
   stay float at runtime, so memory does not fall by the same 3.9×. It also quantizes on
   the fly, which is part of why it measures slower.
8. **The pre/inference/post breakdown is sampled separately** (5 extra calls, averaged)
   from the timed loop, as context for *where* the time goes. It will not sum exactly to
   the p50 beside it; the end-to-end distribution is the authoritative figure.

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
