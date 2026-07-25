"""Per-frame inference with the trained YOLO detector.

Loads the Phase 1 weights once and runs them on each frame, returning a
`supervision.Detections` (the common currency the tracker and compliance modules
consume). The trained model carries its own class id->name map, so class names
are read from the model, never hardcoded.

Accepts either the PyTorch `.pt` weights or a Phase 4 exported model (`.onnx`,
`.torchscript`) — see `phase4_deploy/` for how to produce and benchmark those.
"""
from __future__ import annotations

import os

# ultralytics auto-installs missing dependencies at import/load time. Loading an
# .onnx model on a CUDA machine makes it fetch a 241 MB `onnxruntime-gpu` — which,
# if the host's CUDA/cuDNN don't match that build, ADVERTISES a CUDA provider it
# cannot use and then breaks ONNX inference outright. A live safety demo must not
# reshape its own environment mid-run, so we disable that here (before ultralytics
# is imported below, since it reads this at import time).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from typing import Optional

import cv2
import numpy as np
import supervision as sv


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Coerce a frame to 3-channel BGR (YOLO expects 3 channels). cv2 capture always
    yields BGR, but a grayscale / BGRA frame from another source would otherwise raise
    a low-level torch channel error; convert it, or fail with a clear message."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"detect() expected a 2D grayscale or HxWx{{3,4}} frame, got shape {frame.shape}")


def _device_arg(device: str):
    """ultralytics accepts 0 / 'cpu' / 'mps' / 'cuda:0'. Normalise 'cuda' -> 0."""
    return 0 if device == "cuda" else device


def _onnx_cuda_usable() -> bool:
    """Whether ONNX Runtime can REALLY run on the GPU here.

    `get_available_providers()` only reports what the build advertises. The failure
    mode that matters is precisely "advertises CUDA but cannot load it" (a mismatched
    onnxruntime-gpu), so checking the advertised list would answer True in exactly the
    broken case. We instantiate a trivial session on the CUDA provider and see whether
    it actually comes back bound to CUDA."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        import onnx
        from onnx import TensorProto, helper
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        sess = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                    providers=["CUDAExecutionProvider"])
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception:
        return False


class Detector:
    def __init__(self, weights_path: str, device: str = "cpu",
                 confidence_threshold: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO
        ext = os.path.splitext(weights_path)[1].lower()
        self.exported = ext not in ("", ".pt")
        # Exported graphs carry no task field, so state it rather than let
        # ultralytics guess (which prints a warning and can guess wrong).
        self.model = YOLO(weights_path, task="detect") if self.exported else YOLO(weights_path)
        self.device = _device_arg(device)
        if ext == ".onnx" and self.device != "cpu" and not _onnx_cuda_usable():
            # The installed ONNX Runtime is a CPU build: asking for CUDA would make
            # ultralytics bind GPU buffers to a CPU session and fail. Be explicit
            # instead of crashing mid-session.
            print("[info] ONNX Runtime has no usable CUDA provider - running the "
                  "detector on CPU (see phase4_deploy/README.md for the numbers).")
            self.device = "cpu"
        self.conf = float(confidence_threshold)
        self.imgsz = int(imgsz)
        if self.exported:
            # An exported graph carries no cached class names, so reading `.names`
            # makes ultralytics build the whole backend — on its OWN auto-selected
            # device, which is CUDA whenever torch sees a GPU, regardless of what we
            # chose. For ONNX that both ignores our CPU decision and can trigger an
            # onnxruntime-gpu requirement. Force the backend up on the chosen device
            # first, so `.names` then reads from an already-correct backend.
            try:
                self.model.predict(np.zeros((32, 32, 3), dtype=np.uint8), imgsz=self.imgsz,
                                   device=self.device, verbose=False)
            except Exception:
                pass
        # id -> name, straight from the trained weights.
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}

    def class_ids_for(self, names) -> list[int]:
        """Resolve class NAMES to the model's integer ids (skips names not in the model)."""
        wanted = set(names)
        return [i for i, n in self.names.items() if n in wanted]

    def warmup(self) -> None:
        """Run one inference so lazy CUDA / model init isn't billed to frame 1's FPS."""
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(blank, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a BGR frame. Returns detections at/above the threshold."""
        result = self.model.predict(
            _to_bgr(frame), conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
