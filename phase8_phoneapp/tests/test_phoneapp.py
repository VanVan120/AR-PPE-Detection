"""Unit tests for the phone app (phase8_phoneapp/).

No camera, no weights, no phone: the HTTP layer runs against a stub session, and the
parts with real logic — the camera claim, the certificate, the coordinate normalisation,
the byte-range reader — are exercised directly.

Three of these guard mistakes that would only ever show up on someone else's phone at a
site, where nobody can debug them:

  * `test_manifest_carries_the_key` — the installed icon opens `start_url`. Without the
    key in it the app launches straight into a 403 and looks broken.
  * `test_second_phone_cannot_also_be_the_camera` — two phones feeding one tracker
    interleaves two scenes into one track history.
  * `test_boxes_are_normalised` — the phone uploads a downscaled frame and draws on a
    full-resolution preview, so pixel coordinates would be silently offset.

    python phase8_phoneapp/tests/test_phoneapp.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from phase8_phoneapp import certs                                    # noqa: E402
from phase8_phoneapp.app import (AppServer, Jobs, PhoneSession,      # noqa: E402
                                 _handler_factory, load_token)
from src.compliance import ActiveViolation, FrameCompliance, PersonStatus  # noqa: E402
from src.pipeline import _people_rows                                # noqa: E402

results = {}
TOKEN = "phone-token-abc"


# ---- the camera claim ---------------------------------------------------------
def _session():
    """A session with no config and no model: `claim` and the not-ready paths of
    `submit` are pure logic and need neither."""
    return PhoneSession(None, frame_rate=10)


def test_first_phone_gets_the_camera():
    s = _session()
    results["phone: the first phone to send a frame becomes the camera"] = s.claim("A")


def test_second_phone_cannot_also_be_the_camera():
    """Two phones pushing frames into one tracker would interleave two different scenes
    into a single track history — identities swapping between them, and a violation log
    that describes neither place."""
    s = _session()
    s.claim("A")
    results["phone: a second phone is refused the camera, not the app"] = (
        not s.claim("B") and s.camera_held_by_other("B") and not s.camera_held_by_other("A"))


def test_the_claim_lapses_when_a_phone_goes_quiet():
    """A phone that locks its screen or walks out of range must not hold the camera for
    the rest of the site visit."""
    s = _session()
    s.claim("A")
    s._cam_seen -= 99.0                        # as if A had been silent for 99 s
    results["phone: a silent phone loses the camera to the next one"] = s.claim("B")


def test_the_same_phone_keeps_the_camera():
    s = _session()
    results["phone: the holder keeps the camera frame after frame"] = (
        s.claim("A") and s.claim("A") and s.claim("A"))


def test_frames_before_the_model_is_loaded_are_not_errors():
    """Loading the model takes seconds. The link is printed immediately, so early frames
    must report 'starting', not fail."""
    s = _session()
    out = s.submit(b"not-a-jpeg", "A")
    results["phone: frames sent while the model loads report 'starting'"] = (
        out.get("ok") is False and out.get("starting") is True and "error" not in out)


def test_unreadable_frame_is_rejected_cleanly():
    s = _session()
    s._ready = True                             # pretend the model is up
    out = s.submit(b"\x00\x01\x02 not an image", "A")
    results["phone: a corrupt frame is refused, not crashed on"] = (
        out.get("ok") is False and "readable image" in out.get("error", ""))


def test_a_wrong_frame_rate_is_reported_not_hidden():
    """ByteTrack turns `lost_track_buffer` into a span of real time using the frame rate
    it was given. Told 15 while actually getting 6, its memory of an occluded person
    covers less than half the time it should and people come back as new workers — which
    gets blamed on the re-identification rather than on the units."""
    s = _session()                              # constructed for 10 fps
    quiet_early = s.fps_hint()                  # no frames yet -> nothing to say
    s._frame_no = 200
    s._last_state["fps"] = 3.0
    hint = s.fps_hint()
    s._last_state["fps"] = 9.5                  # close enough; must stay quiet
    ok_close = s.fps_hint()
    results["phone: a frame rate the tracker was not set up for is reported"] = (
        quiet_early == "" and ok_close == "" and "--fps 3" in hint)


# ---- coordinates --------------------------------------------------------------
def test_boxes_are_normalised():
    """The phone uploads a 640-wide frame but draws on a 1080-wide preview. Pixel
    coordinates would be offset by the scale factor — which looks like a tracking bug,
    not a units bug, and would be chased for hours."""
    fc = FrameCompliance()
    st = PersonStatus(tracker_id=7, bbox=(80.0, 60.0, 240.0, 300.0))
    st.active.append(ActiveViolation(7, "No-Helmet", "high", "No hard hat"))
    fc.persons.append(st)
    rows = _people_rows(fc, {7: "Worker 3"}, {}, (480, 640, 3))
    r = rows[0]
    results["phone: boxes come back as fractions of the frame, not pixels"] = (
        r["box"] == [0.125, 0.125, 0.375, 0.625] and r["label"] == "Worker 3"
        and r["severity"] == "high" and r["violations"] == ["No hard hat"]
        and all(0.0 <= v <= 1.0 for v in r["box"]))


def test_degenerate_frame_shape_is_survivable():
    results["phone: a zero-sized frame yields no boxes instead of dividing by zero"] = (
        _people_rows(FrameCompliance(), {}, {}, (0, 0, 3)) == [])


# ---- certificates -------------------------------------------------------------
def test_certificate_is_made_and_reused():
    tmp = tempfile.mkdtemp(prefix="ar-cert-")
    try:
        pair = certs.ensure_cert(tmp, ["127.0.0.1", "192.168.7.7"])
        ok_made = pair is not None and all(os.path.isfile(p) for p in pair)
        # Same hosts -> the existing certificate must be reused, or every restart would
        # be a new certificate and a fresh warning for the supervisor to click through.
        reused = certs._covers(pair[0], ["127.0.0.1", "192.168.7.7"])
        # A host it does not cover -> regenerate, or the phone gets a name-mismatch error
        # it cannot click past.
        stale = certs._covers(pair[0], ["10.99.99.99"])
        results["phone: a certificate is created, reused, and renewed when the IP moves"] = (
            ok_made and reused and not stale)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fingerprint_is_printable():
    tmp = tempfile.mkdtemp(prefix="ar-cert-")
    try:
        pair = certs.ensure_cert(tmp, ["127.0.0.1"])
        fp = certs.fingerprint(pair[0])
        parts = fp.split(":")
        results["phone: the certificate fingerprint prints in the browser's format"] = (
            len(parts) == 32 and all(len(p) == 2 for p in parts))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_addresses_are_addresses():
    addrs = certs.local_addresses()
    ok = addrs and all(len(a.split(".")) == 4 for a in addrs)
    results["phone: the printed link uses a real IPv4 address"] = bool(ok)


# ---- the access key -----------------------------------------------------------
def test_token_survives_a_restart():
    """The installed app keeps the key it was given. A key minted afresh on every start
    would break the home-screen icon every time the laptop rebooted."""
    import phase8_phoneapp.app as app
    tmp = tempfile.mkdtemp(prefix="ar-tok-")
    old = app.STATE_DIR
    try:
        app.STATE_DIR = tmp
        a = load_token()
        b = load_token()
        c = load_token(new=True)
        d = load_token()
        results["phone: the access key survives a restart, and --new-token rotates it"] = (
            a == b and c != a and d == c and len(a) > 8)
    finally:
        app.STATE_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


# ---- the HTTP layer -----------------------------------------------------------
class _StubSession:
    def __init__(self):
        self.frames = []
        self.reset_calls = 0

    def submit(self, jpeg, cid, view="live"):
        self.frames.append((len(jpeg), cid, view))
        return {"ok": True, "seq": len(self.frames), "people": [], "alerts": [],
                "workers": [], "persons": 0, "fps": 9.0, "view": view}

    def state(self):
        return {"ready": True, "persons": 0, "alerts": [], "workers": [], "fps": 9.0,
                "target_fps": 12}

    def camera_held_by_other(self, cid):
        return False

    def report(self):
        return {"workers": {"workers_seen": 0}}

    def jpeg(self):
        return b"\xff\xd8\xff-stub"

    def reset(self):
        self.reset_calls += 1
        return True


def _serve(session, jobs, token=TOKEN):
    httpd = AppServer(("127.0.0.1", 0), _handler_factory(session, jobs, token, tls=False))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _req(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_auth_including_a_non_ascii_key():
    """`parse_qs` decodes `%FF` with errors='replace', so the key arrives as a non-ASCII
    str and `compare_digest` raises TypeError on those — which would print a traceback
    into the terminal showing the access link, triggerable with no key at all."""
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    try:
        codes = [_req(f"{base}/api/status")[0],
                 _req(f"{base}/api/status?t=nope")[0],
                 _req(f"{base}/api/status?t=%FF")[0],
                 _req(f"{base}/api/status?t={TOKEN}")[0]]
        results["phone: bad, missing and non-ASCII keys all give 403, never a crash"] = (
            codes == [403, 403, 403, 200])
    finally:
        httpd.shutdown()


def test_manifest_carries_the_key():
    """The home-screen icon opens `start_url`. Without the key in it — and in the icon
    URLs the launcher fetches — the installed app opens straight into a 403."""
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    try:
        code, body, hdrs = _req(f"{base}/manifest.webmanifest?t={TOKEN}")
        man = json.loads(body)
        results["phone: the installed app's start URL carries the access key"] = (
            code == 200 and TOKEN in man["start_url"]
            and all(TOKEN in i["src"] for i in man["icons"])
            and man["display"] == "standalone" and man["id"] == "/"
            and "manifest" in hdrs.get("Content-Type", ""))
    finally:
        httpd.shutdown()


def test_service_worker_may_claim_the_whole_origin():
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    try:
        code, body, hdrs = _req(f"{base}/sw.js?t={TOKEN}")
        results["phone: the service worker is allowed to scope to the whole app"] = (
            code == 200 and hdrs.get("Service-Worker-Allowed") == "/"
            and b"addEventListener" in body)
    finally:
        httpd.shutdown()


def test_client_script_gets_the_key_substituted():
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    try:
        _c, page, _h = _req(f"{base}/?t={TOKEN}")
        _c, js, _h = _req(f"{base}/app.js?t={TOKEN}")
        text = page.decode("utf-8", "replace") + js.decode("utf-8", "replace")
        results["phone: the page and the script ship with the key, no placeholder left"] = (
            "__TOKEN__" not in text and TOKEN in text)
    finally:
        httpd.shutdown()


def test_a_frame_reaches_the_pipeline():
    stub = _StubSession()
    httpd, base = _serve(stub, Jobs(None, tempfile.gettempdir()))
    try:
        code, body, _h = _req(f"{base}/api/frame?t={TOKEN}&cid=phone1&v=glasses",
                              data=b"x" * 512, headers={"Content-Type": "image/jpeg"})
        d = json.loads(body)
        results["phone: a posted frame reaches the pipeline with its client and view"] = (
            code == 200 and d["ok"] is True and stub.frames == [(512, "phone1", "glasses")])
    finally:
        httpd.shutdown()


def test_an_unknown_view_falls_back_instead_of_failing():
    stub = _StubSession()
    httpd, base = _serve(stub, Jobs(None, tempfile.gettempdir()))
    try:
        _req(f"{base}/api/frame?t={TOKEN}&cid=p&v=../../etc", data=b"y" * 8,
             headers={"Content-Type": "image/jpeg"})
        results["phone: an unrecognised view name falls back to the normal one"] = (
            stub.frames[-1][2] == "live")
    finally:
        httpd.shutdown()


def test_oversized_and_empty_frames_are_refused():
    stub = _StubSession()
    httpd, base = _serve(stub, Jobs(None, tempfile.gettempdir()))
    try:
        empty, _b, _h = _req(f"{base}/api/frame?t={TOKEN}&cid=p", data=b"")
        # Claim a huge body without sending it: the cap must be enforced from the header,
        # before anything is read into memory.
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.putrequest("POST", f"/api/frame?t={TOKEN}&cid=p")
        conn.putheader("Content-Length", str(50 * 1024 * 1024))
        conn.endheaders()
        big = conn.getresponse().status
        conn.close()
        results["phone: empty and oversized frame uploads are refused"] = (
            empty == 400 and big == 400 and stub.frames == [])
    finally:
        httpd.shutdown()


def test_reset_starts_a_fresh_walk():
    stub = _StubSession()
    httpd, base = _serve(stub, Jobs(None, tempfile.gettempdir()))
    try:
        code, body, _h = _req(f"{base}/api/reset?t={TOKEN}", data=b"")
        results["phone: 'New walk' clears the session"] = (
            code == 200 and json.loads(body)["ok"] is True and stub.reset_calls == 1)
    finally:
        httpd.shutdown()


def test_two_copies_cannot_share_the_port():
    """Found by accident: on Windows the socketserver default (SO_REUSEADDR) lets a
    SECOND process bind a port that is already being listened on, silently. Two models
    then run, each seeing a different half of the frames, and the identities and
    violations are split across two sessions with no error anywhere — the launcher being
    double-clicked twice is all it takes."""
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    port = httpd.server_address[1]
    try:
        second = None
        try:
            second = AppServer(("127.0.0.1", port),
                               _handler_factory(_StubSession(), None, TOKEN, tls=False))
        except OSError:
            pass                                # the required outcome
        if second is not None:
            second.server_close()
        results["phone: a second copy cannot quietly bind the same port"] = second is None
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unknown_route_404s():
    httpd, base = _serve(_StubSession(), Jobs(None, tempfile.gettempdir()))
    try:
        results["phone: an unknown route 404s rather than hanging"] = (
            _req(f"{base}/api/nope?t={TOKEN}")[0] == 404)
    finally:
        httpd.shutdown()


# ---- results: byte ranges and path safety -------------------------------------
def _finished_job(jobs, tmp, payload=b"0123456789" * 200):
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "annotated.mp4"), "wb") as fh:
        fh.write(payload)
    jobs._jobs["JOB"] = {"state": "done", "pct": 100, "name": "clip", "dir": tmp,
                         "error": "", "report": {}, "summary": "ok"}
    return payload


def test_result_video_supports_byte_ranges():
    """Without Range support iOS Safari refuses to play the annotated clip at all — the
    review video would simply never appear on half the phones."""
    tmp = tempfile.mkdtemp(prefix="ar-job-")
    jobs = Jobs(None, tmp)
    payload = _finished_job(jobs, os.path.join(tmp, "clip"))
    httpd, base = _serve(_StubSession(), jobs)
    try:
        url = f"{base}/api/result?t={TOKEN}&id=JOB&f=video"
        whole, body, hdrs = _req(url)
        part_code, part, phdrs = _req(url, headers={"Range": "bytes=10-19"})
        tail_code, tail, _h = _req(url, headers={"Range": "bytes=-5"})
        results["phone: the annotated clip serves byte ranges so phones can play it"] = (
            whole == 200 and body == payload and hdrs.get("Accept-Ranges") == "bytes"
            and part_code == 206 and part == payload[10:20]
            and phdrs.get("Content-Range") == f"bytes 10-19/{len(payload)}"
            and tail_code == 206 and tail == payload[-5:])
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def test_range_past_the_end_is_refused():
    tmp = tempfile.mkdtemp(prefix="ar-job-")
    jobs = Jobs(None, tmp)
    _finished_job(jobs, os.path.join(tmp, "clip"), payload=b"abcdef")
    httpd, base = _serve(_StubSession(), jobs)
    try:
        code, _b, hdrs = _req(f"{base}/api/result?t={TOKEN}&id=JOB&f=video",
                              headers={"Range": "bytes=900-999"})
        results["phone: a range past the end of the clip gives 416, not a hang"] = (
            code == 416 and hdrs.get("Content-Range") == "bytes */6")
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def test_results_cannot_be_escaped():
    """The client names a KIND of result, never a path. Anything else must resolve to
    nothing — this server is handed to someone to run on a site network."""
    tmp = tempfile.mkdtemp(prefix="ar-job-")
    jobs = Jobs(None, tmp)
    _finished_job(jobs, os.path.join(tmp, "clip"))
    bad = [jobs.file("JOB", "../../../../Windows/win.ini"),
           jobs.file("JOB", "/etc/passwd"),
           jobs.file("JOB", "..\\..\\secrets.txt"),
           jobs.file("NOPE", "video")]
    shutil.rmtree(tmp, ignore_errors=True)
    results["phone: a result can only be one of the known kinds, never a path"] = (
        all(b is None for b in bad))


def test_only_one_clip_is_analysed_at_a_time():
    """Two analyses plus the live pipeline is three copies of the detector on a laptop
    that is also encoding video."""
    tmp = tempfile.mkdtemp(prefix="ar-job-")
    jobs = Jobs(None, tmp)
    jobs._busy.acquire()                        # as if one were already running
    try:
        jid, err = jobs.start(b"fake video bytes", "clip.mp4")
        results["phone: a second clip waits instead of loading a second model"] = (
            jid is None and "still being analysed" in err)
    finally:
        jobs._busy.release()
        shutil.rmtree(tmp, ignore_errors=True)


def test_uploaded_names_cannot_steer_the_filesystem():
    import phase8_phoneapp.app as app
    tmp = tempfile.mkdtemp(prefix="ar-job-")
    jobs = Jobs(None, tmp)
    real = app.analyze_clip
    # Stand in for the analyser: this test is about where the file LANDS, and running the
    # real one on deliberately-bogus bytes only adds an ffmpeg complaint from a worker
    # thread that lands after the results and reads like a failure.
    app.analyze_clip = lambda *a, **k: {}
    try:
        jid, _err = jobs.start(b"not really a video", "../../../evil name.mp4")
        for _ in range(60):                     # let the worker thread finish
            if jobs._busy.acquire(blocking=False):
                jobs._busy.release()
                break
            time.sleep(0.05)
        made = os.listdir(os.path.join(tmp, "_uploads"))
        results["phone: an uploaded clip's own name cannot steer where it is written"] = (
            jid is not None and len(made) == 1
            and ".." not in made[0] and "/" not in made[0] and "\\" not in made[0]
            and jobs.status(jid)["state"] == "done")
    finally:
        app.analyze_clip = real
        shutil.rmtree(tmp, ignore_errors=True)


# ---- the front end refers only to things that exist ---------------------------
def test_every_element_the_script_touches_exists():
    """There is no way to see a JavaScript error on someone else's phone at a site.

    A single mistyped id makes `$( )` return null and the first property set throws during
    load — the app then shows a dead grey page with no message, on a device nobody can
    attach a debugger to. Cheap to check here; impossible to diagnose there.
    """
    import re as _re
    with open(os.path.join(_ROOT, "phase8_phoneapp", "static", "index.html"),
              encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(_ROOT, "phase8_phoneapp", "static", "app.js"),
              encoding="utf-8") as fh:
        js = fh.read()
    have = set(_re.findall(r'\bid="([^"]+)"', html))
    want = set(_re.findall(r'\$\("([^"]+)"\)', js))
    missing = sorted(want - have)
    # Every tab button must have the panel it reveals, and every view button must name a
    # view the server will actually accept.
    tabs = set(_re.findall(r'data-tab="([^"]+)"', html))
    panels = {t for t in tabs if f'id="tab-{t}"' in html}
    views = set(_re.findall(r'data-view="([^"]+)"', html))
    from phase8_phoneapp.app import VIEWS
    if missing:
        print("   missing element ids:", missing)
    results["phone: the script only touches elements the page actually has"] = (
        not missing and tabs == panels and views <= set(VIEWS))


def test_nothing_is_revealed_by_clearing_its_inline_display():
    """`el.style.display = ""` clears the INLINE value and falls back to the stylesheet.

    For an element the stylesheet gives `display:none` — `#shot`, the picture the two
    glasses views arrive as — that means the line meant to reveal it hides it instead, and
    those views silently never appear. Shipped that way once. Anything the script reveals
    by clearing the inline value must therefore have no `display` of its own in the CSS.
    """
    import re as _re
    static = os.path.join(_ROOT, "phase8_phoneapp", "static")
    with open(os.path.join(static, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(static, "app.js"), encoding="utf-8") as fh:
        js = fh.read()
    css = html.split("<style>")[1].split("</style>")[0]
    styled = set()
    for m in _re.finditer(r"#([A-Za-z0-9_-]+)\s*\{([^}]*)\}", css):
        if _re.search(r"(^|[;\s])display\s*:", m.group(2)):
            styled.add(m.group(1))
    cleared = set(_re.findall(r'\$\("([^"]+)"\)\.style\.display\s*=\s*""', js))
    clash = sorted(cleared & styled)
    if clash:
        print("   revealed by clearing an inline display, but styled in CSS:", clash)
    results["phone: nothing is revealed by clearing a display the CSS also sets"] = not clash


def test_the_page_never_reaches_off_the_local_network():
    """No CDN, no font host, no analytics. A site laptop is often on a phone hotspot with
    no route out, and a page that waits on an external stylesheet renders unstyled or not
    at all — and it would also be quietly telling someone else's server that this is
    running."""
    import re as _re
    files = ["index.html", "app.js", "sw.js"]
    bad = []
    for name in files:
        with open(os.path.join(_ROOT, "phase8_phoneapp", "static", name),
                  encoding="utf-8") as fh:
            for m in _re.finditer(r'https?://[^\s"\'<>)]+', fh.read()):
                url = m.group(0)
                if not url.startswith(("http://127.0.0.1", "http://localhost")):
                    bad.append(f"{name}: {url}")
    if bad:
        print("   external references:", bad)
    results["phone: the app loads nothing from outside the local network"] = not bad


def main() -> int:
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_PHONEAPP", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
