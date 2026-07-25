"""Phase 8 — the phone IS the camera.

Phase 7 put the *view* on a phone, but the camera was still a laptop webcam or a
third-party IP-camera app someone had to install and wire up by hand. This is the app
itself: the supervisor opens one link, taps **Install**, and gets an icon on the home
screen. Opening it turns on the phone's own camera and draws the safety overlay on the
live picture. Nothing to record, nothing to copy off the phone, no file to send anyone.

**How the work is split.** The phone captures a frame, sends it here, and gets back the
boxes as coordinates; it draws them over its own live preview. The preview therefore stays
perfectly smooth at the camera's native rate, and only the boxes carry the round-trip lag.
Sending the annotated picture back instead would have made the whole view stutter at the
network's pace — the picture is the part the eye notices.

**Why HTTPS with a certificate nobody signed.** Browsers only give a page the camera on a
secure origin, so a plain-HTTP page cannot do this at all. The server makes its own
certificate (see `certs.py`), which is why the phone shows a warning the first time. The
terminal prints the certificate's fingerprint so accepting it is a check, not a leap.

**When the camera is not available** — an iPhone that will not trust the certificate, a
locked-down device, a flat refusal of the permission — the app falls back to **Record**:
the phone's normal camera app records a clip, the app uploads it, and the same pipeline
analyses it and plays the annotated result back. That path needs no certificate and no
camera permission, so there is always a way to get a result from a site.

    python -m phase8_phoneapp.app
    python -m phase8_phoneapp.app --port 8443 --fps 10
    python -m phase8_phoneapp.app --http            # no camera; Record still works
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))
sys.path.insert(0, _ROOT)

from phase7_mobile.analyze import analyze_clip          # noqa: E402
from phase8_phoneapp import certs                       # noqa: E402
from src.config import load_config                      # noqa: E402
from src.pipeline import SafetyPipeline                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
STATE_DIR = os.path.join(HERE, ".state")

MAX_FRAME_BYTES = 4 * 1024 * 1024            # a 640px JPEG is ~40 KB; this is 100x slack
MAX_UPLOAD_BYTES = 400 * 1024 * 1024
CAMERA_TAKEOVER_S = 5.0                      # a silent camera client loses its claim
PROC_WAIT_S = 15.0
HANDSHAKE_TIMEOUT_S = 20
VIEWS = ("live", "seethrough", "glasses")

STATIC_TYPES = {
    "app.js": "application/javascript; charset=utf-8",
    "sw.js": "application/javascript; charset=utf-8",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
}


# ---------------------------------------------------------------- session ----
class PhoneSession:
    """The pipeline, fed by frames PUSHED from a phone instead of pulled from a camera."""

    def __init__(self, cfg, frame_rate: int = 12, jpeg_quality: int = 70):
        self.cfg = cfg
        self.frame_rate = int(frame_rate)
        self.jpeg_quality = int(jpeg_quality)
        self._pipe: SafetyPipeline | None = None
        self._proc = threading.Lock()        # one frame through the pipeline at a time
        self._lock = threading.Lock()        # guards the small shared state below
        self._error = ""
        self._ready = False
        self._frame_no = 0
        self._t0 = 0.0
        self._last_frame_at = 0.0
        self._last_jpeg = b""
        self._cam_id = ""
        self._cam_seen = 0.0
        self._view = "live"
        self._hinted = False
        self._last_state: dict = {"persons": 0, "alerts": [], "workers": [], "fps": 0.0}

    # -- lifecycle --
    def start(self) -> None:
        """Load the model off the accept loop so the link works the instant it is printed.

        Loading takes seconds on a CPU. Blocking start-up would mean the terminal prints a
        link that refuses connections for the first ten seconds — which reads as "it is
        broken" to anyone who has not been told to wait.
        """
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            pipe = SafetyPipeline(self.cfg, frame_rate=self.frame_rate, quiet=True)
        except Exception as e:                           # noqa: BLE001
            with self._lock:
                self._error = str(e)
            return
        with self._lock:
            self._pipe = pipe
            self._ready = True

    def ready(self) -> bool:
        with self._lock:
            return self._ready

    # -- the camera claim --
    def claim(self, cid: str) -> bool:
        """Only one phone may be the camera.

        Two phones posting frames into one tracker would interleave two different scenes
        into a single track history: identities would swap between them and the violation
        log would be nonsense. A second phone is still useful — it just watches — so this
        refuses the *frames*, not the viewer. The claim lapses after a few silent seconds
        so a phone that locks or walks out of range does not hold the camera forever.
        """
        now = time.monotonic()
        with self._lock:
            if self._cam_id and self._cam_id != cid and (now - self._cam_seen) < CAMERA_TAKEOVER_S:
                return False
            self._cam_id = cid
            self._cam_seen = now
            return True

    def camera_held_by_other(self, cid: str) -> bool:
        with self._lock:
            return bool(self._cam_id and self._cam_id != cid
                        and (time.monotonic() - self._cam_seen) < CAMERA_TAKEOVER_S)

    # -- the per-frame work --
    def submit(self, jpeg: bytes, cid: str, view: str = "live") -> dict:
        with self._lock:
            if self._error:
                return {"ok": False, "error": self._error}
            if not self._ready:
                return {"ok": False, "starting": True}
        if not self.claim(cid):
            return {"ok": False, "camera_busy": True}

        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "frame was not a readable image"}

        if not self._proc.acquire(timeout=PROC_WAIT_S):
            # The previous frame from this phone is still being processed (a retry after a
            # client-side timeout). Dropping this one is right: a stale frame processed
            # late would feed the tracker an out-of-order view of the scene.
            return {"ok": False, "busy": True}
        t_start = time.perf_counter()
        try:
            pipe = self._pipe
            if pipe is None:
                return {"ok": False, "starting": True}
            if self._t0 == 0.0:
                # Time starts at the first frame, not at server start-up: the report
                # should describe the walk, not the minutes the laptop sat idle first.
                self._t0 = t_start
            self._frame_no += 1
            elapsed = t_start - self._t0
            want_render = view in ("seethrough", "glasses")
            pipe.render_enabled = want_render
            if want_render:
                self.cfg.arview_mode = view
            res = pipe.process(frame, self._frame_no, elapsed)
            out = {
                "ok": True,
                "seq": self._frame_no,
                "persons": res.persons,
                "people": res.people,
                "alerts": res.alerts,
                "workers": res.workers,
                "fps": round(res.fps, 1),
                "elapsed": round(elapsed, 1),
                "view": view,
            }
            if want_render:
                ok, buf = cv2.imencode(".jpg", res.frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    data = buf.tobytes()
                    out["jpeg"] = base64.b64encode(data).decode("ascii")
                    with self._lock:
                        self._last_jpeg = data
            else:
                # Keep a snapshot-able image even in the fast path, but only every so
                # often: encoding the clean frame every time would burn time for a button
                # that is pressed once a minute.
                if self._frame_no % 15 == 0:
                    ok, buf = cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        with self._lock:
                            self._last_jpeg = buf.tobytes()
        except Exception as e:                           # noqa: BLE001
            return {"ok": False, "error": f"processing failed: {e}"}
        finally:
            self._proc.release()

        out["ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
        with self._lock:
            self._view = view
            self._last_frame_at = time.monotonic()
            self._last_state = {"persons": out["persons"], "alerts": out["alerts"],
                                "workers": out["workers"], "fps": out["fps"]}
        return out

    # -- accessors --
    def fps_hint(self) -> str:
        """Say so when the rate the tracker was told to expect is not the rate it is
        getting.

        ByteTrack converts `lost_track_buffer` into a span of real time using the frame
        rate it was given at construction. Told 15 fps while actually receiving 6, its
        memory of an occluded person covers less than half the time it should, so people
        pick up a new identity after a short occlusion — visible as workers multiplying in
        the roster, and blamed on the re-identification rather than on the units.
        """
        with self._lock:
            fps = float(self._last_state.get("fps") or 0.0)
            frames, rate = self._frame_no, self.frame_rate
        if frames < 60 or fps < 0.5:
            return ""                      # not enough evidence yet
        if 0.6 * rate <= fps <= 1.7 * rate:
            return ""
        return (f"This laptop is managing about {fps:.0f} fps, but the tracker was set up "
                f"for {rate}. Restart it with  --fps {int(round(fps))}  for steadier "
                f"identities.")

    def state(self) -> dict:
        hint = self.fps_hint()
        if hint and not self._hinted:
            self._hinted = True
            print(f"  [tune] {hint}")
        with self._lock:
            st = dict(self._last_state)
            st.update({
                "ready": self._ready,
                "error": self._error,
                "frames": self._frame_no,
                "view": self._view,
                "camera": bool(self._cam_id),
                "stale_s": (round(time.monotonic() - self._last_frame_at, 1)
                            if self._last_frame_at else -1.0),
                "target_fps": self.frame_rate,
                "fps_hint": hint,
            })
            return st

    def jpeg(self) -> bytes:
        with self._lock:
            return self._last_jpeg

    def report(self) -> dict:
        """Non-destructive, exactly as in phase 7: the phone polls this, and closing open
        violation episodes on read would shatter one violation into many."""
        with self._lock:
            pipe, t0 = self._pipe, self._t0
        if pipe is None:
            return {}
        now = (time.perf_counter() - t0) if t0 else None
        return pipe.report(now_s=now)

    def final_report(self) -> dict:
        with self._lock:
            pipe, t0, n = self._pipe, self._t0, self._frame_no
        if pipe is None:
            return {}
        # Take the processing lock so the pipeline is not mutated mid-close by a frame
        # that is still in flight.
        got = self._proc.acquire(timeout=PROC_WAIT_S)
        try:
            elapsed = (time.perf_counter() - t0) if t0 else 0.0
            pipe.close(elapsed_s=elapsed, frame_no=n)
            return pipe.report(now_s=elapsed)
        finally:
            if got:
                self._proc.release()

    def reset(self) -> bool:
        """Start a fresh walk: forget every worker, track and violation, keep the model."""
        if not self._proc.acquire(timeout=PROC_WAIT_S):
            return False
        try:
            if self._pipe is None:
                return False
            self._pipe.reset()
            with self._lock:
                self._frame_no = 0
                self._t0 = 0.0
                self._last_jpeg = b""
                self._last_state = {"persons": 0, "alerts": [], "workers": [], "fps": 0.0}
            return True
        finally:
            self._proc.release()


# ------------------------------------------------------------ clip analysis --
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class Jobs:
    """Analyse a clip the phone just recorded, and report progress while it runs."""

    def __init__(self, cfg, out_root: str):
        self.cfg = cfg
        self.out_root = out_root
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def start(self, data: bytes, name: str) -> tuple[str | None, str]:
        # One at a time. Two analyses plus the live pipeline means three copies of the
        # detector on a laptop that is also encoding video — and a phone at a site only
        # ever records one clip at a time anyway.
        if not self._busy.acquire(blocking=False):
            return None, "another clip is still being analysed"
        jid = secrets.token_urlsafe(8)
        safe = _SAFE_NAME.sub("-", os.path.splitext(name or "clip")[0]).strip("-")[:32]
        stem = f"{safe or 'clip'}-{jid[:6]}"
        up_dir = os.path.join(self.out_root, "_uploads")
        os.makedirs(up_dir, exist_ok=True)
        src = os.path.join(up_dir, f"{stem}.mp4")
        try:
            with open(src, "wb") as fh:
                fh.write(data)
        except OSError as e:
            self._busy.release()
            return None, f"could not save the clip: {e}"

        with self._lock:
            self._jobs[jid] = {"state": "running", "pct": 0, "name": stem,
                               "dir": os.path.join(self.out_root, stem), "error": ""}
        threading.Thread(target=self._run, args=(jid, src, stem), daemon=True).start()
        return jid, ""

    def _run(self, jid: str, src: str, stem: str) -> None:
        try:
            def progress(frac: float) -> None:
                with self._lock:
                    if jid in self._jobs:
                        self._jobs[jid]["pct"] = int(round(100 * frac))

            rep = analyze_clip(src, self.cfg, self.out_root, progress=False,
                               on_progress=progress)
            out_dir = os.path.join(self.out_root, stem)
            summary = ""
            spath = os.path.join(out_dir, "summary.txt")
            if os.path.isfile(spath):
                with open(spath, "r", encoding="utf-8") as fh:
                    summary = fh.read()
            with self._lock:
                self._jobs[jid].update({"state": "done", "pct": 100, "report": rep,
                                        "summary": summary, "dir": out_dir})
        except Exception as e:                           # noqa: BLE001
            with self._lock:
                self._jobs[jid].update({"state": "failed", "error": str(e)})
        finally:
            self._busy.release()

    def status(self, jid: str) -> dict:
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return {"state": "unknown"}
            out = {"state": job["state"], "pct": job["pct"], "error": job["error"]}
            if job["state"] == "done":
                rep = job.get("report", {})
                out["summary"] = job.get("summary", "")
                out["workers"] = rep.get("workers", {})
                out["violations"] = rep.get("violations", {})
                out["clip"] = rep.get("clip", {})
            return out

    def file(self, jid: str, which: str) -> str | None:
        """Resolve a result file. The client names a KIND, never a path — the path comes
        from the job record — so there is nothing here to traverse out of."""
        allowed = {"video": "annotated.mp4", "report": "report.json",
                   "summary": "summary.txt"}
        if which not in allowed:
            return None
        with self._lock:
            job = self._jobs.get(jid)
            if job is None or job["state"] != "done":
                return None
            path = os.path.join(job["dir"], allowed[which])
        return path if os.path.isfile(path) else None


# ------------------------------------------------------------------ server ---
def _handler_factory(session: PhoneSession, jobs: Jobs, token: str, tls: bool):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60                        # drop idle keep-alive connections

        def log_message(self, fmt, *a):
            pass

        # -- helpers --
        def _authorised(self, qs) -> bool:
            got = (qs.get("t") or [""])[0]
            if not token:
                return True
            # Compare bytes: `parse_qs` decodes percent-escapes with errors="replace", so
            # `?t=%FF` yields a non-ASCII str and `compare_digest` raises TypeError on
            # those — which would put a traceback in the terminal that is showing the
            # access link, triggerable by anyone on the network without the key.
            try:
                return secrets.compare_digest(got.encode("utf-8", "surrogatepass"),
                                              token.encode("utf-8"))
            except (UnicodeError, TypeError, AttributeError):
                return False

        def _deny(self):
            self.close_connection = True
            self._send(b"403 - open the link printed in the terminal (it carries the key).",
                       "text/plain; charset=utf-8", 403)

        def _send(self, body: bytes, ctype: str, code: int = 200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode("utf-8"),
                       "application/json; charset=utf-8", code)

        def _read_body(self, cap: int) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > cap:
                self.close_connection = True     # body not consumed; do not reuse
                return None
            buf = bytearray()
            while len(buf) < length:
                chunk = self.rfile.read(min(65536, length - len(buf)))
                if not chunk:
                    self.close_connection = True
                    return None
                buf += chunk
            return bytes(buf)

        # -- routes --
        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if not self._authorised(qs):
                return self._deny()
            path = u.path

            if path in ("/", "/index.html"):
                return self._page()
            if path == "/manifest.webmanifest":
                return self._manifest()
            name = path.lstrip("/")
            if name in STATIC_TYPES:
                return self._static(name)

            if path == "/api/status":
                st = session.state()
                st["tls"] = tls
                st["camera_busy"] = session.camera_held_by_other(
                    (qs.get("cid") or [""])[0])
                return self._json(st)
            if path == "/api/report":
                return self._json(session.report())
            if path == "/api/snapshot":
                data = session.jpeg()
                return (self._send(data, "image/jpeg") if data
                        else self._json({"error": "no frame yet"}, 503))
            if path == "/api/job":
                return self._json(jobs.status((qs.get("id") or [""])[0]))
            if path == "/api/result":
                return self._result(qs)
            return self._json({"error": "not found"}, 404)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if not self._authorised(qs):
                return self._deny()

            if u.path == "/api/frame":
                body = self._read_body(MAX_FRAME_BYTES)
                if body is None:
                    return self._json({"ok": False, "error": "bad frame upload"}, 400)
                cid = (qs.get("cid") or [""])[0][:64]
                view = (qs.get("v") or ["live"])[0]
                if view not in VIEWS:
                    view = "live"
                return self._json(session.submit(body, cid, view))

            if u.path == "/api/upload":
                body = self._read_body(MAX_UPLOAD_BYTES)
                if body is None:
                    return self._json({"error": "clip missing or too large "
                                                f"(max {MAX_UPLOAD_BYTES // (1024*1024)} MB)"}, 400)
                jid, err = jobs.start(body, (qs.get("name") or ["clip"])[0])
                if jid is None:
                    return self._json({"error": err}, 409)
                return self._json({"job": jid})

            if u.path == "/api/reset":
                return self._json({"ok": session.reset()})

            return self._json({"error": "not found"}, 404)

        # -- individual responses --
        def _page(self):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as fh:
                html = fh.read()
            self._send(html.replace(b"__TOKEN__", token.encode()),
                       "text/html; charset=utf-8")

        def _static(self, name: str):
            with open(os.path.join(STATIC_DIR, name), "rb") as fh:
                data = fh.read()
            if name.endswith(".js"):
                data = data.replace(b"__TOKEN__", token.encode())
            # The service worker must be able to claim the whole origin, not just its own
            # folder, or an installed app would fall outside its own cache scope.
            extra = {"Service-Worker-Allowed": "/"} if name == "sw.js" else None
            self._send(data, STATIC_TYPES[name], extra=extra)

        def _manifest(self):
            """Built here rather than shipped as a file: `start_url` and the icon URLs must
            carry the access key, or the icon on the home screen opens to a 403. `id` is
            fixed so rotating the key updates the installed app instead of installing a
            second copy of it."""
            q = f"?t={token}" if token else ""
            man = {
                "id": "/",
                "name": "AR Safety Monitor",
                "short_name": "Safety",
                "description": "PPE compliance on the live camera view.",
                "start_url": f"/{q}",
                "scope": "/",
                "display": "standalone",
                "orientation": "any",
                "background_color": "#12100e",
                "theme_color": "#12100e",
                "icons": [
                    {"src": f"/icon-192.png{q}", "sizes": "192x192", "type": "image/png",
                     "purpose": "any"},
                    {"src": f"/icon-512.png{q}", "sizes": "512x512", "type": "image/png",
                     "purpose": "any"},
                    {"src": f"/icon-512.png{q}", "sizes": "512x512", "type": "image/png",
                     "purpose": "maskable"},
                ],
            }
            self._send(json.dumps(man).encode("utf-8"), "application/manifest+json")

        def _result(self, qs):
            path = jobs.file((qs.get("id") or [""])[0], (qs.get("f") or ["video"])[0])
            if path is None:
                return self._json({"error": "no such result"}, 404)
            ctype = ("video/mp4" if path.endswith(".mp4")
                     else "application/json" if path.endswith(".json")
                     else "text/plain; charset=utf-8")
            self._serve_file(path, ctype)

        def _serve_file(self, path: str, ctype: str):
            """Serve with Range support — without it iOS Safari refuses to play the
            annotated clip at all, and Android seeks by re-downloading the whole file."""
            size = os.path.getsize(path)
            rng = self.headers.get("Range", "")
            start, end = 0, size - 1
            partial = False
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:                                   # bytes=-N  (the final N bytes)
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                partial = True

            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


class AppServer(ThreadingHTTPServer):
    """TLS is negotiated in the worker thread, not the accept loop.

    Wrapping the listening socket is the usual shortcut, but then the handshake happens
    inside `accept()` on the main thread — one phone that opens a connection and stalls
    mid-handshake freezes the app for everybody. Handing the raw socket to the worker and
    wrapping it there keeps the accept loop free.
    """
    daemon_threads = True
    ssl_context: ssl.SSLContext | None = None

    def server_bind(self):
        """Refuse to share the port — on Windows the default lets two copies bind it.

        `socketserver` sets SO_REUSEADDR, and on Windows that does not merely permit
        rebinding after a restart: it lets a *second* process bind a port another process
        is already listening on, with no error. Whoever double-clicks the launcher twice
        then has two models running, each seeing a different half of the frames, with
        identities and violations split across two sessions that neither the phone nor the
        terminal gives any hint of. It was hit here by accident, and it is silent.

        SO_EXCLUSIVEADDRUSE is the Windows way to say "this port is mine", without
        reintroducing the TIME_WAIT restart problem that SO_REUSEADDR exists to solve on
        POSIX. On POSIX two listeners on one port already fail, so nothing changes there.
        """
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.allow_reuse_address = False
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        return super().server_bind()

    def finish_request(self, request, client_address):
        if self.ssl_context is None:
            return super().finish_request(request, client_address)
        try:
            request.settimeout(HANDSHAKE_TIMEOUT_S)
            conn = self.ssl_context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            # Someone tapped "go back" on the certificate warning, or spoke plain HTTP to
            # an HTTPS port. Both are normal and neither is worth a traceback.
            return
        try:
            self.RequestHandlerClass(conn, client_address, self)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def handle_error(self, request, client_address):
        """One line, never a traceback.

        A traceback here would be dumped into the same terminal that is showing the access
        link and the certificate fingerprint — the two things the supervisor is reading
        off the screen. A real bug still gets a visible line; a phone locking its screen
        mid-request does not.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                            ssl.SSLError, TimeoutError, socket.timeout)):
            return
        print(f"  [warn] request failed: {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------- setup ---
def load_token(new: bool = False) -> str:
    """A key that survives a restart.

    The installed app keeps the key it was given. Minting a fresh one on every start-up
    would silently break the home-screen icon every time the laptop is restarted — the app
    would open straight into a 403 with no clue why. Rotate deliberately with
    `--new-token`.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, "token")
    if not new and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tok = fh.read().strip()
            if tok:
                return tok
        except OSError:
            pass
    tok = secrets.token_urlsafe(9)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tok)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass
    return tok


def print_qr(text: str) -> None:
    try:
        import qrcode                                    # optional
    except ImportError:
        print("  (pip install qrcode  for a scannable QR here)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(text)
    qr.make(fit=True)
    m = qr.get_matrix()
    for y in range(0, len(m), 2):
        row = ""
        for x in range(len(m[0])):
            top, bot = m[y][x], (m[y + 1][x] if y + 1 < len(m) else False)
            row += "█" if top and bot else "▀" if top else "▄" if bot else " "
        print("  " + row)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Phone app: the phone's own camera, with the safety overlay")
    ap.add_argument("--config", default=os.path.join(_ROOT, "phase2", "config.yaml"))
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fps", type=int, default=None,
                    help="frame rate the tracker is told to expect (default: config "
                         "target_fps). Set it near what the phone actually achieves.")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality sent back 30-95")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "site"))
    ap.add_argument("--http", action="store_true",
                    help="plain HTTP: no certificate warning, but the browser will NOT "
                         "give the page a camera. Record-and-upload still works.")
    ap.add_argument("--new-token", action="store_true",
                    help="rotate the access key (installed apps must reopen the new link)")
    ap.add_argument("--no-token", action="store_true",
                    help="disable the access key (only on a hotspot you control)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rate = args.fps or int(getattr(cfg, "target_fps", 12)) or 12
    token = "" if args.no_token else load_token(new=args.new_token)

    session = PhoneSession(cfg, frame_rate=rate, jpeg_quality=args.quality)
    session.start()
    jobs = Jobs(cfg, args.out)

    ip = certs.local_addresses()[0]
    scheme = "http" if args.http else "https"
    pair = None if args.http else certs.ensure_cert(os.path.join(STATE_DIR, "certs"))
    if pair is None and not args.http:
        print("[warn] could not create a certificate (need `cryptography` or openssl).")
        print("       Falling back to plain HTTP - the camera will NOT be available,")
        print("       but Record-and-upload still works. Fix: pip install cryptography")
        scheme = "http"

    try:
        httpd = AppServer((args.host, args.port),
                          _handler_factory(session, jobs, token, tls=(scheme == "https")))
    except OSError as e:
        print(f"\n[FAIL] port {args.port} is not available: {e}")
        print("       The app is probably already running in another window — use that")
        print(f"       one, or start this with  --port {args.port + 1}")
        return 2
    if pair is not None and scheme == "https":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(pair[0], pair[1])
        httpd.ssl_context = ctx

    url = f"{scheme}://{ip}:{args.port}/" + (f"?t={token}" if token else "")
    bar = "=" * 64
    print()
    print(bar)
    print("  SAFETY APP - open this on the phone (same WiFi or hotspot)")
    print(bar)
    print(f"  {url}")
    print()
    print_qr(url)
    print()
    if scheme == "https":
        print("  The phone WILL warn that the connection is not private. That is this")
        print("  certificate being self-signed - tap Advanced -> Proceed. Browsers only")
        print("  give a page the camera over HTTPS, so there is no way around it.")
        print("  Certificate SHA-256 (check it matches the one the phone shows):")
        print(f"    {certs.fingerprint(pair[0])}")
    else:
        print("  Plain HTTP: the browser will NOT allow camera access. Use the Record")
        print("  button instead, or restart without --http.")
    print()
    print("  Then: 'Install app' / 'Add to Home Screen' puts an icon on the phone.")
    print(f"  tracker rate : {rate} fps   |   results: {args.out}")
    if not token:
        print("  [warn] access key DISABLED - anyone on this network can open it.")
    print("  Ctrl+C to stop and print the session report.")
    print(bar)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        httpd.server_close()
        rep = session.final_report()
        workers = (rep or {}).get("workers", {})
        if workers.get("per_worker"):
            print(f"{workers.get('workers_seen', 0)} worker(s), "
                  f"{workers.get('total_violation_episodes', 0)} violation episode(s), "
                  f"{workers.get('total_violation_s', 0)}s unsafe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
