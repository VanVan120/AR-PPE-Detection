"""Phone-linked live view — open one link on a phone and watch the site.

Runs the full safety pipeline on a laptop and serves the annotated view to a phone
browser over the local network. No app to install, nothing to sideload: the supervisor
opens a link (or scans the QR printed in the terminal) and sees what the system sees.

The camera can be:
  * the laptop's own webcam (`--source 0`),
  * a **phone** running any free IP-camera app (`--source http://192.168.x.x:8080/video`),
    so the phone is the camera and the screen at the same time,
  * or a video file, for rehearsing the demo before going anywhere.

**Access.** The feed shows identifiable people, so it is not left open: every request
must carry a random token minted at start-up, and the printed link already contains it.
Someone else on the same WiFi who does not have the link gets a 403. This is deliberately
not a login — a password is one more thing to type on a phone at a site — but it is also
NOT strong security: the stream is plain HTTP, so treat it as "keeps honest people out on
a private hotspot", not as protection on an untrusted network.

    python -m phase7_mobile.server                       # laptop webcam
    python -m phase7_mobile.server --source http://192.168.0.14:8080/video
    python -m phase7_mobile.server --arview glasses --port 8000
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from src.config import load_config                    # noqa: E402
from src.pipeline import SafetyPipeline               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Session:
    """Owns the capture + processing thread and publishes the latest annotated frame."""

    def __init__(self, cfg, source, jpeg_quality: int = 70):
        self.cfg = cfg
        self.source = source
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._state: dict = {"persons": 0, "alerts": [], "workers": [], "fps": 0.0,
                             "frames": 0, "running": False, "error": "", "stale_s": 0.0,
                             "mode": getattr(cfg, "arview_mode", "composite")}
        self._last_frame_at = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._pipe = None
        self._t0 = time.perf_counter()
        self._frame_no = 0

    # -- lifecycle --
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stopping(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> bool:
        """Returns False if the capture thread refused to stop in time. The caller must
        NOT then touch the pipeline: it may still be inside `process()`, and reading its
        state concurrently is exactly the race the report snapshot exists to avoid."""
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=5)
        return not self._thread.is_alive()

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source: {self.source}")
        return cap

    def _run(self) -> None:
        try:
            cap = self._open()
            # Fall back to the CONFIGURED target_fps, not a hardcoded 30, so the tracker
            # gets the same frame rate here as it does in run.py. Many webcams and most
            # IP-camera streams report 0, and a wrong rate makes lost_track_buffer cover
            # the wrong span of real time.
            fps = cap.get(cv2.CAP_PROP_FPS) or float(getattr(self.cfg, "target_fps", 0)) or 30.0
            self._pipe = SafetyPipeline(self.cfg, frame_rate=int(round(fps)), quiet=True)
            with self._lock:
                self._state["running"] = True
        except Exception as e:                              # noqa: BLE001
            with self._lock:
                self._state["error"] = str(e)
                self._state["running"] = False
            return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                # A file ran out; a camera hiccup should not kill the session.
                if isinstance(self.source, str) and os.path.isfile(str(self.source)):
                    cap.release()
                    cap = self._open()                      # loop the rehearsal clip
                    continue
                time.sleep(0.05)
                continue
            self._frame_no += 1
            elapsed = time.perf_counter() - self._t0
            try:
                res = self._pipe.process(frame, self._frame_no, elapsed)
            except Exception as e:                          # noqa: BLE001
                with self._lock:
                    self._state["error"] = f"processing failed: {e}"
                break
            ok, buf = cv2.imencode(".jpg", res.frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._last_frame_at = time.monotonic()
                self._state.update({
                    "persons": res.persons,
                    "alerts": res.alerts,
                    "workers": res.workers,
                    "fps": round(res.fps, 1),
                    "frames": self._frame_no,
                    "elapsed": round(elapsed, 1),
                    "running": True,
                    "mode": getattr(self.cfg, "arview_mode", "composite"),
                })
        cap.release()
        with self._lock:
            self._state["running"] = False

    # -- accessors --
    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def state(self) -> dict:
        """Includes `stale_s`: seconds since the last new frame. Without it a frozen feed
        (camera unplugged, source dropped) is indistinguishable from a live one -- the
        phone would keep showing the last good frame as though nothing were wrong."""
        with self._lock:
            st = dict(self._state)
            st["stale_s"] = (round(time.monotonic() - self._last_frame_at, 1)
                             if self._last_frame_at else -1.0)
            return st

    def report(self) -> dict:
        """Live snapshot. Deliberately does NOT close open episodes: the phone polls this,
        and closing them on read would end each violation early and start a new one on the
        next frame, shattering one long violation into many short ones."""
        if self._pipe is None:
            return {}
        return self._pipe.report(now_s=time.perf_counter() - self._t0)

    def final_report(self) -> dict:
        """End of session: close what is still open, then report."""
        if self._pipe is None:
            return {}
        elapsed = time.perf_counter() - self._t0
        self._pipe.close(elapsed_s=elapsed, frame_no=self._frame_no)
        return self._pipe.report(now_s=elapsed)

    def set_mode(self, mode: str) -> bool:
        if mode not in ("composite", "seethrough", "glasses"):
            return False
        self.cfg.arview_mode = mode          # picked up on the next frame
        return True


def _handler_factory(session: Session, token: str):
    page = os.path.join(HERE, "mobile.html")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):       # keep the console readable
            pass

        # -- helpers --
        def _authorised(self, qs) -> bool:
            got = (qs.get("t") or [""])[0]
            # Compare BYTES, not str. `parse_qs` decodes percent-escapes with
            # errors="replace", so `?t=%FF` or `?t=%C3%BC` yields a non-ASCII string and
            # `compare_digest` on str raises TypeError — which would escape do_GET, dump a
            # traceback into the terminal showing the access link, and return an empty
            # body instead of the documented 403. Anyone on the network could trigger that
            # without the key.
            try:
                return secrets.compare_digest(got.encode("utf-8", "surrogatepass"),
                                              token.encode("utf-8"))
            except (UnicodeError, TypeError, AttributeError):
                return False

        def _deny(self):
            body = b"403 - open the link printed in the terminal (it carries the key)."
            self.send_response(403)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode("utf-8"),
                       "application/json; charset=utf-8", code)

        # -- routes --
        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if not self._authorised(qs):
                return self._deny()
            if u.path in ("/", "/index.html"):
                with open(page, "rb") as fh:
                    html = fh.read().replace(b"__TOKEN__", token.encode())
                return self._send(html, "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._json(session.state())
            if u.path == "/api/report":
                return self._json(session.report())
            if u.path == "/stream.mjpg":
                return self._stream()
            if u.path == "/api/mode":
                mode = (qs.get("m") or [""])[0]
                return self._json({"ok": session.set_mode(mode), "mode": mode})
            if u.path == "/api/snapshot":
                data = session.jpeg()
                return self._send(data, "image/jpeg") if data else self._json(
                    {"error": "no frame yet"}, 503)
            self._json({"error": "not found"}, 404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            # A dead client is only detectable by writing to it. If the producer stalls
            # (camera unplugged, pipeline error) this loop would otherwise skip every
            # write and spin forever on a socket nobody is reading — and the phone opens a
            # NEW stream on each view switch and each unlock, leaking a thread every time.
            # So: re-send the last frame periodically, and give up if nothing arrives.
            KEEPALIVE_S, GIVE_UP_S = 1.0, 30.0
            try:
                last = None
                last_write = time.monotonic()
                while not session.stopping():
                    data = session.jpeg()
                    fresh = data and data is not last
                    if not fresh:
                        idle = time.monotonic() - last_write
                        if idle > GIVE_UP_S:
                            break                      # producer is gone; release thread
                        if not data or idle < KEEPALIVE_S:
                            time.sleep(0.02)
                            continue                   # nothing yet, or too soon to repeat
                    last = data
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    last_write = time.monotonic()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass                                   # phone navigated away / locked

    return Handler


def lan_ip() -> str:
    """Best-guess LAN address. No packet is actually sent — connecting a UDP socket just
    makes the OS pick the interface it would route through."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def print_qr(text: str) -> None:
    """Print a scannable QR in the terminal if `qrcode` is available; otherwise just say
    so. Never a hard dependency — the URL alone is enough."""
    try:
        import qrcode                                   # optional
    except ImportError:
        print("  (pip install qrcode  to get a scannable QR here)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(text)
    qr.make(fit=True)
    m = qr.get_matrix()
    # Two rows per line using half-blocks keeps the code square in a terminal.
    for y in range(0, len(m), 2):
        row = ""
        for x in range(len(m[0])):
            top = m[y][x]
            bot = m[y + 1][x] if y + 1 < len(m) else False
            row += "█" if top and bot else "▀" if top else "▄" if bot else " "
        print("  " + row)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Serve the live safety view to a phone")
    ap.add_argument("--config", default=os.path.join(_ROOT, "phase2", "config.yaml"))
    ap.add_argument("--source", default=None,
                    help="camera index, video file, or a phone IP-camera URL")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 serves the whole local network (needed for the phone)")
    ap.add_argument("--arview", default=None,
                    choices=["composite", "seethrough", "glasses"])
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 30-95")
    ap.add_argument("--no-token", action="store_true",
                    help="disable the access key (only on a private hotspot)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.arview:
        cfg.arview_mode = args.arview
    source = args.source if args.source is not None else cfg.resolved_source()

    token = "" if args.no_token else secrets.token_urlsafe(9)
    session = Session(cfg, source, jpeg_quality=args.quality)
    session.start()

    httpd = ThreadingHTTPServer((args.host, args.port),
                                _handler_factory(session, token))
    ip = lan_ip()
    url = f"http://{ip}:{args.port}/" + (f"?t={token}" if token else "")

    print()
    print("=" * 62)
    print("  PHONE LINK - open this on the phone (same WiFi / hotspot)")
    print("=" * 62)
    print(f"  {url}")
    print()
    print_qr(url)
    print()
    print(f"  source : {source}")
    print(f"  view   : {cfg.arview_mode}")
    if not token:
        print("  [warn] access key DISABLED - anyone on this network can watch.")
    else:
        print("  Access key is in the link. Without it the page is refused.")
    print("  Plain HTTP on the local network: fine on a private hotspot, not on")
    print("  an untrusted one. Ctrl+C to stop.")
    print("=" * 62)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        httpd.server_close()
        clean = session.stop()
        if not clean:
            print("[warn] the capture thread did not stop in time - skipping the final "
                  "report rather than reading the pipeline while it is still running.")
        rep = session.final_report() if clean else {}
        workers = (rep or {}).get("workers", {})
        if workers.get("per_worker"):
            print(f"{workers.get('workers_seen', 0)} worker(s), "
                  f"{workers.get('total_violation_episodes', 0)} violation episode(s), "
                  f"{workers.get('total_violation_s', 0)}s unsafe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
