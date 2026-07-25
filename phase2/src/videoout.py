"""Write a video the person you send it to can actually open.

`cv2.VideoWriter(path, fourcc(*"mp4v"), ...)` is the usual line, and on most builds it is
the only one that works — but **MPEG-4 Part 2, which is what `mp4v` means, does not play in
any current browser**. The file opens fine in VLC and looks correct in every desktop test,
so the problem only appears at the far end: a supervisor taps the annotated clip on their
phone and gets a black rectangle or a download prompt. That was the state of the review
bundle until this module existed.

There is no way to know from the API which encoders a given OpenCV build actually has.
`isOpened()` lies in both directions — it returned False for H.264 on the development
machine (the OpenH264 DLL was missing) and True for encoders that then produce a file
nothing can read. So this **probes**: it writes a handful of real frames with each
candidate, reads them back, and keeps the first that survives the round trip. The result is
cached, because probing costs about a second.

Preference order is by how widely the result plays, not by quality:

    H.264/mp4   every browser, every phone; shares over WhatsApp and email. Needs a real
                ffmpeg — `pip install imageio-ffmpeg` supplies one, and it is worth it:
                measured at ~300 fps encode against VP8's 15 and VP9's 3.
    VP8/webm    Chrome, Edge, Firefox, Android. No extra install.
    VP9/webm    better compression, but **2.7 fps encode** on the development laptop —
                slower than the detector, so it is ranked below VP8 despite being newer.
    MPEG-4/mp4  last resort — opens in VLC, and the caller is told to say so.

    writer, path, codec, plays_in_browser = open_writer(stem, fps, (w, h))
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np


@contextlib.contextmanager
def _quiet_native_stderr():
    """Hide FFmpeg's own complaints while probing.

    A failed candidate makes OpenCV print things like "Failed to load OpenH264 library
    ... Please check environment and/or download library" — from C, so `contextlib
    .redirect_stderr` does not touch it. Those lines are the probe working as designed, but
    to anyone reading the terminal they look like the app is broken. The outcome is
    reported afterwards in one clear line instead.
    """
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):
        yield                                            # no real fd (IDE, notebook)
        return
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(saved, 2)
        except OSError:
            pass
        for fd in (saved, devnull):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

# (fourcc, extension, human name, plays in a browser?)
# VP8 before VP9 deliberately: measured on real 960x540 frames, VP8 encodes at 15 fps and
# VP9 at 2.7 — slower than the detector it is meant to keep up with, which turned a 12 s
# clip into 84 s of analysis.
CANDIDATES = [
    ("avc1", ".mp4", "H.264", True),
    ("VP80", ".webm", "VP8", True),
    ("VP90", ".webm", "VP9", True),
    ("mp4v", ".mp4", "MPEG-4 Part 2", False),
]

_probed: Optional[tuple] = None
_ffmpeg: Optional[str] = None                # "" once we know there is none


def ffmpeg_exe() -> str:
    """A real ffmpeg, if one is reachable. Optional, never required.

    OpenCV's bundled FFmpeg usually has no H.264 *encoder* (licensing), which is why the
    universal format is the one that cannot be written by default. A standalone ffmpeg has
    libx264 and is both far faster and playable on iPhones, where WebM support is patchy.
    """
    global _ffmpeg
    if _ffmpeg is not None:
        return _ffmpeg
    _ffmpeg = ""
    try:
        import imageio_ffmpeg                            # optional dependency
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            _ffmpeg = exe
    except Exception:                                    # noqa: BLE001
        pass
    if not _ffmpeg:
        import shutil
        _ffmpeg = shutil.which("ffmpeg") or ""
    return _ffmpeg


class FfmpegWriter:
    """Pipe raw frames to ffmpeg. Same three methods as `cv2.VideoWriter`."""

    def __init__(self, exe: str, path: str, fps: float, size: Tuple[int, int]):
        import subprocess
        w, h = int(size[0]), int(size[1])
        self.proc = None
        self._bad = False
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", f"{max(1.0, float(fps)):.4f}", "-i", "-",
            "-an",
            # H.264 with 4:2:0 chroma: anything else (4:4:4, 10-bit) is rejected by
            # Safari and by most Android decoders.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p",
            # libx264 + yuv420p cannot take odd dimensions, and a phone clip is often an
            # odd number of pixels tall after rotation metadata is applied.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            # Puts the index at the front so the phone can start playing before the whole
            # file has arrived — otherwise tapping play does nothing until it finishes.
            "-movflags", "+faststart",
            path,
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE)
        except OSError:
            self._bad = True

    def isOpened(self) -> bool:                          # noqa: N802  (cv2's name)
        return bool(self.proc is not None and not self._bad
                    and self.proc.poll() is None)

    def write(self, frame) -> None:
        if self._bad or self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError, ValueError):
            self._bad = True                             # ffmpeg died; stop feeding it

    def release(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=120)
        except Exception:                                # noqa: BLE001
            self.proc.kill()
        self.proc = None


def _try_codec(fourcc: str, ext: str, size=(160, 120), fps: float = 15.0) -> bool:
    """Write a few frames, read them back. Anything less is not evidence."""
    import cv2

    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not writer.isOpened():
            writer.release()
            return False
        # Noise, not a flat colour: some encoders "succeed" on a constant image and fail
        # on anything with real entropy, and a flat probe would pick them.
        rng = np.random.RandomState(0)
        for _ in range(5):
            writer.write(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        writer.release()
        if not os.path.isfile(path) or os.path.getsize(path) < 256:
            return False
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None and frame.size)
    except Exception:                                    # noqa: BLE001
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _try_ffmpeg(exe: str, size=(160, 120), fps: float = 15.0) -> bool:
    import cv2
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        w = FfmpegWriter(exe, path, fps, size)
        if not w.isOpened():
            w.release()
            return False
        rng = np.random.RandomState(0)
        for _ in range(5):
            w.write(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        w.release()
        if not os.path.isfile(path) or os.path.getsize(path) < 256:
            return False
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None and frame.size)
    except Exception:                                    # noqa: BLE001
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def best_codec(force: str = "") -> tuple:
    """(fourcc, ext, name, browser_playable) for the best encoder this build really has.

    `fourcc == "ffmpeg"` means "pipe to the standalone ffmpeg", not an OpenCV fourcc.
    """
    global _probed
    if force:
        for cand in CANDIDATES:
            if cand[0].lower() == force.lower():
                return cand
    if _probed is None:
        with _quiet_native_stderr():
            exe = ffmpeg_exe()
            if exe and _try_ffmpeg(exe):
                _probed = ("ffmpeg", ".mp4", "H.264 (ffmpeg)", True)
            else:
                _probed = next((c for c in CANDIDATES if _try_codec(c[0], c[1])), None)
        if _probed is None:
            # Nothing round-tripped. Fall back to the historic default rather than refuse
            # to write anything: a file that needs VLC beats no file at all.
            _probed = CANDIDATES[-1]
    return _probed


def open_writer(path_stem: str, fps: float, size: Tuple[int, int], force: str = ""):
    """Open a writer at `path_stem` + whatever extension the chosen codec needs.

    Returns (writer, path, codec_name, browser_playable). `writer` may report
    `isOpened() == False` if even the fallback refuses; callers should check.
    """
    import cv2

    fourcc, ext, name, playable = best_codec(force)
    path = path_stem + ext
    if fourcc == "ffmpeg":
        writer = FfmpegWriter(ffmpeg_exe(), path, fps, size)
        if writer.isOpened():
            return writer, path, name, playable
        writer.release()                                 # fall through to OpenCV
        fourcc, ext, name, playable = next(
            (c for c in CANDIDATES if _try_codec(c[0], c[1])), CANDIDATES[-1])
        path = path_stem + ext
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                             max(1.0, float(fps)), (int(size[0]), int(size[1])))
    return writer, path, name, playable


def describe() -> str:
    """One line for a start-up banner or `--check`. ASCII only — this is printed into a
    Windows console, where an em dash comes out as a replacement character."""
    fourcc, ext, name, playable = best_codec()
    if fourcc == "ffmpeg":
        return f"review video: {name} {ext} - plays on every phone, fast to write"
    if playable:
        return (f"review video: {name} {ext} - plays on Android/Chrome/Firefox. For H.264 "
                f"(iPhone-safe, ~40x faster): pip install imageio-ffmpeg")
    return (f"review video: {name} {ext} - WILL NOT play in any browser. "
            f"Fix with: pip install imageio-ffmpeg")


def content_type(path: str) -> str:
    return "video/webm" if path.lower().endswith(".webm") else "video/mp4"
