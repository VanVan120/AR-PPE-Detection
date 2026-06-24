"""Video frame source — a uniform interface over a webcam index or a video file.

`python run.py` does not care whether frames come from a live camera (the laptop
stand-in for the eventual glasses camera) or a recorded clip; both flow through
`FrameSource`. The class degrades gracefully: a camera that won't open or a path
that isn't a video raises `SourceError` with a human message, not a stack trace.
"""
from __future__ import annotations

import os
import sys
from typing import Iterator, Union

import cv2
import numpy as np


class SourceError(RuntimeError):
    """A video source could not be opened or read — carries a user-facing message."""


class FrameSource:
    def __init__(self, source: Union[int, str], target_fps: int = 30):
        self.source = source
        self.is_live = not (isinstance(source, str) and not str(source).isdigit())
        self._cap = self._open(source)
        if self._cap is None or not self._cap.isOpened():
            raise SourceError(self._open_error_message(source))

        # Properties (best-effort; webcams often misreport).
        reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = reported_fps if reported_fps and reported_fps > 0 else float(target_fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # Frame count is only meaningful for files.
        self.frame_count = 0 if self.is_live else int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @staticmethod
    def _open(source: Union[int, str]):
        # On Windows the default MSMF backend can stall opening a webcam; DSHOW is
        # more reliable for live cameras. Files open with the default backend.
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
            index = int(source)
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    return cap
                if cap is not None:
                    cap.release()   # free the half-opened DSHOW handle before fallback
            return cv2.VideoCapture(index)
        return cv2.VideoCapture(str(source))

    @staticmethod
    def _open_error_message(source: Union[int, str]) -> str:
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
            return (f"could not open webcam index {source}. Is a camera connected and not in "
                    "use by another app? Try a different index (e.g. --source 1), or point "
                    "`source` at a video file in config.yaml.")
        if not os.path.isfile(str(source)):
            return f"video file not found: {source}"
        return (f"could not open video file: {source} (unsupported codec or corrupt file). "
                "Try re-encoding to H.264 MP4.")

    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames until the source is exhausted (file) or stopped (webcam)."""
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
