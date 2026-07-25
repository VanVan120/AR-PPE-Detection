"""Shared primitives for the Phase 4 edge-deployment tools.

Three things live here because export, benchmarking and parity all need them:

  * **Honest timing** (`summarize`, `time_calls`). A latency number is only
    meaningful if the measurement is correct. CUDA kernel launches are
    *asynchronous*, so timing a GPU call without `torch.cuda.synchronize()`
    measures the launch, not the work — and reports absurdly fast numbers. We
    synchronise around every timed region on CUDA, discard warmup iterations
    (lazy init / cuDNN autotune / allocator warmup make the first calls
    unrepresentative), and report a **distribution** (p50/p95/std), never a bare
    mean, because a laptop GPU's tail latency is what a live pipeline actually
    feels.

  * **Execution-provider honesty** (`onnx_actual_providers`,
    `onnxruntime_health`). ONNX Runtime will happily accept a request for
    `CUDAExecutionProvider`, fail to load it, and **silently run on CPU** — the
    call still succeeds and returns correct results. Benchmarks that trust the
    *requested* provider therefore publish "GPU" numbers that are really CPU
    numbers. Everything here reports the provider the session is **actually**
    using, and `onnxruntime_health` detects the specific broken state where a
    non-functional `onnxruntime-gpu` is installed (see the README's "traps").

  * **No silent environment mutation** (`freeze_environment`). ultralytics
    auto-installs missing export dependencies — and picks `onnxruntime-gpu`
    whenever `torch.cuda.is_available()` (exporter.py:637), which on a machine
    whose CUDA/cuDNN don't match ORT's build leaves a *broken* runtime that
    breaks ONNX inference. A deployment tool must never reshape the user's
    environment behind their back, so we set `YOLO_AUTOINSTALL=false` and
    report missing dependencies as actionable messages instead.

Nothing here imports torch/ultralytics/onnxruntime at module scope, so the pure
helpers are unit-testable with numpy alone.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Formats this toolkit knows how to produce, in report order.
KNOWN_FORMATS = ("onnx", "onnx-fp16", "onnx-int8", "torchscript")


def freeze_environment(quiet_yolo: bool = False) -> None:
    """Stop ultralytics from silently pip-installing things mid-run.

    Must be called BEFORE `ultralytics` is imported to be fully effective
    (`AUTOINSTALL` is read at import time in ultralytics/utils/__init__.py).

    `quiet_yolo` additionally suppresses ultralytics' banner and tqdm progress
    bars, whose carriage returns otherwise interleave with (and can swamp) our
    own report tables."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    if quiet_yolo:
        os.environ.setdefault("YOLO_VERBOSE", "false")
        os.environ.setdefault("TQDM_DISABLE", "1")


def quiet_onnxruntime() -> None:
    """Reduce ONNX Runtime's provider chatter (we report provider state ourselves)."""
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")


def enable_utf8_stdout() -> None:
    """Print UTF-8 safely on a legacy Windows console (cp1252 mangles em-dashes).

    Same approach as phase2/run.py; a no-op where the stream can't be reconfigured."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")       # type: ignore[union-attr]
        except Exception:
            pass


def pin_cuda_device_count() -> None:
    """Force torch to cache its CUDA device count while the environment is clean.

    This must run BEFORE anything can clobber `CUDA_VISIBLE_DEVICES`, and it is what
    makes `preserve_cuda_env` reliable rather than order-dependent. torch memoises
    `device_count()` on first call; if that first call happens while ultralytics has
    the variable set to `""`, the count is pinned at **0 for the lifetime of the
    process** and restoring the variable afterwards cannot undo it (a later
    `device=0` then fails with "Invalid CUDA 'device=0' requested" on a healthy GPU).
    Querying once up front pins the correct value instead."""
    try:
        import torch
        torch.cuda.device_count()
    except Exception:
        pass


@contextlib.contextmanager
def preserve_cuda_env() -> Iterator[None]:
    """Undo ultralytics' process-global CUDA side effect.

    `select_device("cpu")` sets `CUDA_VISIBLE_DEVICES=""`
    (ultralytics/utils/torch_utils.py:182) to force `torch.cuda.is_available()`
    False — and it is NEVER restored. So a single CPU-targeted call silently
    disables CUDA for everything later in the same process: subsequent GPU work
    dies with `AssertionError: Invalid device id`, and a `device=0` request fails
    with "Invalid CUDA 'device=0' requested" even on a healthy GPU.

    Any block that calls into ultralytics with an explicit device belongs inside
    this context manager, so one export can't poison the ones after it.

    **Limit, stated precisely:** restoring the variable restores torch's *view* of
    the devices only if torch has already cached its device count. It cannot undo a
    count that was cached as 0 inside the clobbered window, because that cache is
    permanent for the process. Callers therefore pair this with
    `pin_cuda_device_count()` at startup, which removes the dependency on call
    ordering. The variable is also restored on exceptions."""
    key = "CUDA_VISIBLE_DEVICES"
    had = key in os.environ
    prev = os.environ.get(key)
    try:
        yield
    finally:
        if had:
            os.environ[key] = prev            # type: ignore[assignment]
        else:
            os.environ.pop(key, None)


def file_mb(path: str) -> float:
    """Size of a file (or a directory tree, for formats that export a folder) in MB."""
    if os.path.isdir(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total / 1e6
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0.0


# --- timing ------------------------------------------------------------------
@dataclass
class Stats:
    """A latency distribution, in milliseconds (plus derived FPS)."""
    n: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @property
    def fps(self) -> float:
        """Throughput implied by the MEDIAN latency (robust to outliers)."""
        return 1000.0 / self.p50 if self.p50 > 0 else 0.0

    def as_dict(self) -> dict:
        d = {k: round(float(getattr(self, k)), 3)
             for k in ("mean", "p50", "p95", "std", "min", "max")}
        d["n"] = int(self.n)
        d["fps"] = round(self.fps, 2)
        return d


def summarize(times_ms: Sequence[float]) -> Stats:
    """Summarise raw per-iteration timings (ms) into a distribution.

    p95 uses linear interpolation (numpy's default), matching how latency SLOs
    are normally quoted."""
    arr = np.asarray([float(t) for t in times_ms], dtype=float)
    if arr.size == 0:
        return Stats()
    return Stats(
        n=int(arr.size),
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        min=float(arr.min()),
        max=float(arr.max()),
    )


def time_calls(fn: Callable[[], object], warmup: int = 10, iters: int = 50,
               cuda_sync: bool = False) -> List[float]:
    """Call `fn` repeatedly and return per-iteration wall-clock times in ms.

    `warmup` iterations run first and are DISCARDED (they pay for lazy CUDA
    context creation, cuDNN algorithm selection and allocator growth, and are
    routinely 10-100x slower than steady state).

    With `cuda_sync=True` the GPU is synchronised *before* starting the clock and
    *before* stopping it, so the measured interval contains the completed GPU
    work rather than just the asynchronous launch."""
    import time

    sync = _cuda_syncer() if cuda_sync else None
    for _ in range(max(0, int(warmup))):
        fn()
    if sync:
        sync()
    out: List[float] = []
    for _ in range(max(1, int(iters))):
        if sync:
            sync()                       # ensure prior work is done before t0
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()                       # ensure THIS work is done before t1
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _cuda_syncer() -> Optional[Callable[[], None]]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:
        pass
    return None


# --- ONNX Runtime provider honesty -------------------------------------------
def onnx_actual_providers(onnx_path: str,
                          requested: Optional[Sequence[str]] = None) -> Tuple[List[str], Optional[str]]:
    """Open an ORT session and report the providers it is ACTUALLY using.

    Returns `(providers_in_use, error)`. When `requested` names a provider that
    fails to initialise, ORT falls back silently — the returned list is the truth,
    so callers can refuse to label a CPU run as a GPU run."""
    try:
        import onnxruntime as ort
    except ImportError:
        return [], "onnxruntime is not installed"
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = list(requested) if requested else ort.get_available_providers()
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        return list(sess.get_providers()), None
    except Exception as e:                                  # unreadable/invalid model
        return [], f"{type(e).__name__}: {e}"


def onnxruntime_health() -> dict:
    """Diagnose the ONNX Runtime install, including the broken-GPU-package trap.

    `onnxruntime` and `onnxruntime-gpu` both provide the module `onnxruntime`.
    If the GPU build is installed but its CUDA/cuDNN dependencies are missing,
    `get_available_providers()` still ADVERTISES CUDAExecutionProvider while any
    attempt to use it fails — and ultralytics' AutoBackend, seeing CUDA
    advertised, takes a GPU IO-binding path that then raises. We detect that
    state so the tools can print a fix instead of a stack trace."""
    info: dict = {"installed": False, "version": None, "packages": [],
                  "advertised": [], "cuda_usable": None, "problem": None, "fix": None}
    try:
        import onnxruntime as ort
    except ImportError:
        info["problem"] = "onnxruntime is not installed"
        info["fix"] = "pip install onnxruntime"
        return info

    info["installed"] = True
    info["version"] = getattr(ort, "__version__", "?")
    try:
        import importlib.metadata as md
        for pkg in ("onnxruntime", "onnxruntime-gpu"):
            try:
                info["packages"].append(f"{pkg}=={md.version(pkg)}")
            except Exception:
                pass
    except Exception:
        pass

    info["advertised"] = list(ort.get_available_providers())
    if "CUDAExecutionProvider" in info["advertised"]:
        # Advertised != usable. Probe it on a tiny in-memory model.
        usable, err = _probe_cuda_ep()
        info["cuda_usable"] = usable
        if not usable:
            info["problem"] = ("onnxruntime-gpu is installed but its CUDA provider cannot "
                               f"load ({err}). ONNX inference via ultralytics will FAIL "
                               "because it binds GPU buffers to a CPU session.")
            info["fix"] = ("pip uninstall -y onnxruntime-gpu  &&  "
                           "pip install --force-reinstall onnxruntime")
    else:
        info["cuda_usable"] = False
    return info


def _probe_cuda_ep() -> Tuple[bool, str]:
    """Try to actually instantiate the CUDA EP on a trivial model."""
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        sess = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                    providers=["CUDAExecutionProvider"])
        return "CUDAExecutionProvider" in sess.get_providers(), ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# --- small formatting helpers -------------------------------------------------
def fmt_row(cells: Sequence[object], widths: Sequence[int]) -> str:
    out = []
    for c, w in zip(cells, widths):
        s = str(c)
        out.append(s.ljust(w) if not _is_num(s) else s.rjust(w))
    return "  ".join(out).rstrip()


def _is_num(s: str) -> bool:
    try:
        float(s.replace("%", "").replace("x", "").replace("+", ""))
        return True
    except ValueError:
        return False
