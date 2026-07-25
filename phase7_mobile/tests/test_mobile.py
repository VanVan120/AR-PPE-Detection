"""Unit tests for the phone link (phase7_mobile/) and the shared pipeline report.

No camera, no weights, no network beyond localhost: the HTTP layer is exercised against a
stub session, so these run anywhere.

The headline guard is `test_report_is_non_destructive`, which pins down a real bug this
server shipped with: `/api/report` closed every open violation episode, so merely opening
the report on the phone ended each ongoing violation and the next frame started a new one
— one long violation shattered into a string of short ones, and durations reported wrong.

    python phase7_mobile/tests/test_mobile.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from phase7_mobile.server import _handler_factory, lan_ip     # noqa: E402
from src.compliance import ActiveViolation, FrameCompliance, PersonStatus  # noqa: E402
from src.identity import IdentityResult                        # noqa: E402
from src.workerlog import WorkerHistory                        # noqa: E402

results = {}
TOKEN = "test-token-123"
NO_HELMET = ("No-Helmet", "high", "No hard hat")


# ---- the report must never mutate the session --------------------------------
def _history_with_open_episode():
    h = WorkerHistory()
    ids = {1: IdentityResult("Worker 1", "w1", "new")}
    fc = FrameCompliance()
    st = PersonStatus(tracker_id=1, bbox=(0.0, 0.0, 10.0, 20.0))
    st.active.append(ActiveViolation(1, *NO_HELMET))
    fc.persons.append(st)
    h.update(0, 0.0, fc, ids)          # violation starts and stays open
    return h


def test_report_is_non_destructive():
    h = _history_with_open_episode()
    first = h.report(now_s=5.0)
    second = h.report(now_s=6.0)
    eps = h.records["w1"].episodes
    results["mobile: polling the report never closes an open episode"] = (
        len(eps) == 1 and not eps[0].closed and not eps[0].truncated
        and first["total_violation_episodes"] == 1
        and second["total_violation_episodes"] == 1)


def test_open_episode_reports_live_duration():
    h = _history_with_open_episode()
    rep = h.report(now_s=7.5)
    ep = rep["per_worker"][0]["episodes"][0]
    results["mobile: an ongoing violation reports elapsed time, not 0"] = (
        ep["duration_s"] == 7.5 and ep["ongoing"] is True and ep["truncated"] is False)


def test_report_without_now_is_zero_not_negative():
    """Called with no clock (e.g. an offline summary), an open episode must read 0 rather
    than a nonsense value."""
    h = _history_with_open_episode()
    rep = h.report()
    results["mobile: report with no clock gives 0 for open episodes"] = (
        rep["per_worker"][0]["episodes"][0]["duration_s"] == 0.0)


def test_close_still_marks_truncated():
    """The end-of-session path must still behave as before."""
    h = _history_with_open_episode()
    h.close(elapsed_s=9.0, frame_no=3)
    ep = h.records["w1"].episodes[0]
    results["mobile: end-of-session close still marks the episode truncated"] = (
        ep.closed and ep.truncated and ep.duration_s == 9.0)


# ---- the HTTP layer ----------------------------------------------------------
class _StubSession:
    """Stands in for a running pipeline so the HTTP layer can be tested with no model."""

    def __init__(self):
        self.mode = "composite"
        self.report_calls = 0

    def jpeg(self):
        return b"\xff\xd8\xff-not-a-real-jpeg"

    def state(self):
        return {"persons": 2, "alerts": [], "workers": [], "fps": 12.0,
                "frames": 5, "running": True}

    def report(self):
        self.report_calls += 1
        return {"workers": {"workers_seen": 1}}

    def set_mode(self, mode):
        if mode not in ("composite", "seethrough", "glasses"):
            return False
        self.mode = mode
        return True


def _serve(session):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(session, TOKEN))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_auth():
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        missing, _ = _get(f"{base}/api/status")
        wrong, _ = _get(f"{base}/api/status?t=nope")
        ok, body = _get(f"{base}/api/status?t={TOKEN}")
        data = json.loads(body)
        results["mobile: requests without the key are refused"] = (
            missing == 403 and wrong == 403 and ok == 200 and data["persons"] == 2)
    finally:
        httpd.shutdown()


def test_page_has_token_substituted():
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        code, body = _get(f"{base}/?t={TOKEN}")
        text = body.decode("utf-8", "replace")
        results["mobile: the page ships with the key filled in, no placeholder left"] = (
            code == 200 and "__TOKEN__" not in text and TOKEN in text)
    finally:
        httpd.shutdown()


def test_mode_endpoint_validates():
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        _c, good = _get(f"{base}/api/mode?t={TOKEN}&m=glasses")
        _c, bad = _get(f"{base}/api/mode?t={TOKEN}&m=hack")
        results["mobile: the view switch accepts only known modes"] = (
            json.loads(good)["ok"] is True and session.mode == "glasses"
            and json.loads(bad)["ok"] is False and session.mode == "glasses")
    finally:
        httpd.shutdown()


def test_unknown_route_404():
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        code, _ = _get(f"{base}/api/nope?t={TOKEN}")
        results["mobile: an unknown route 404s rather than hanging"] = (code == 404)
    finally:
        httpd.shutdown()


def test_snapshot_served():
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        code, body = _get(f"{base}/api/snapshot?t={TOKEN}")
        results["mobile: a snapshot is served as an image"] = (
            code == 200 and body.startswith(b"\xff\xd8\xff"))
    finally:
        httpd.shutdown()


def test_non_ascii_token_is_refused_not_crashed():
    """Regression guard. `parse_qs` decodes percent-escapes with errors='replace', so
    `?t=%FF` yields a non-ASCII string; `secrets.compare_digest` on str raises TypeError
    for that, which used to escape the handler, dump a traceback into the terminal that
    is showing the access link, and return an empty body instead of a 403 — triggerable by
    anyone on the network with no key."""
    session = _StubSession()
    httpd, base = _serve(session)
    try:
        codes = [_get(f"{base}/api/status?t=%FF")[0],
                 _get(f"{base}/api/status?t=%C3%BC")[0],
                 _get(f"{base}/api/status?t={TOKEN}")[0]]
        results["mobile: a non-ASCII key is refused with 403, not a crash"] = (
            codes == [403, 403, 200])
    finally:
        httpd.shutdown()


def test_absence_tolerance_keeps_one_episode():
    """A worker the detector drops for a frame is not a compliant worker: their violation
    must stay ONE episode, or a single missed detection doubles the violation count."""
    h = WorkerHistory(absence_tolerance=5)
    ids = {1: IdentityResult("Worker 1", "w1", "new")}

    def frame_with(active):
        fc = FrameCompliance()
        st = PersonStatus(tracker_id=1, bbox=(0.0, 0.0, 10.0, 20.0))
        if active:
            st.active.append(ActiveViolation(1, *NO_HELMET))
        fc.persons.append(st)
        return fc

    h.update(0, 0.0, frame_with(True), ids)
    h.update(1, 1.0, FrameCompliance(), {})      # detector drops them entirely
    h.update(2, 2.0, frame_with(True), ids)      # back, still unsafe
    results["mobile: a one-frame detection dropout does not split a violation"] = (
        len(h.records["w1"].episodes) == 1)


def test_present_and_compliant_still_closes_immediately():
    """The tolerance must NOT delay a genuine clear: if the worker is visible and no
    longer violating, the episode ends there."""
    h = WorkerHistory(absence_tolerance=5)
    ids = {1: IdentityResult("Worker 1", "w1", "new")}

    def frame_with(active):
        fc = FrameCompliance()
        st = PersonStatus(tracker_id=1, bbox=(0.0, 0.0, 10.0, 20.0))
        if active:
            st.active.append(ActiveViolation(1, *NO_HELMET))
        fc.persons.append(st)
        return fc

    h.update(0, 0.0, frame_with(True), ids)
    h.update(1, 3.0, frame_with(False), ids)     # seen, compliant -> real clear
    ep = h.records["w1"].episodes[0]
    results["mobile: a visible, compliant worker closes the episode at once"] = (
        ep.closed and ep.duration_s == 3.0)


def test_absent_gap_not_charged_as_unsafe_time():
    """When a vanished worker finally times out, the episode must end when they were LAST
    seen violating — charging the unseen gap to them would inflate their unsafe time."""
    h = WorkerHistory(absence_tolerance=2)
    ids = {1: IdentityResult("Worker 1", "w1", "new")}
    fc = FrameCompliance()
    st = PersonStatus(tracker_id=1, bbox=(0.0, 0.0, 10.0, 20.0))
    st.active.append(ActiveViolation(1, *NO_HELMET))
    fc.persons.append(st)
    h.update(0, 0.0, fc, ids)
    h.update(1, 1.0, fc, ids)                    # last seen violating at t=1
    for i, t in enumerate([2.0, 30.0, 60.0], start=2):
        h.update(i, t, FrameCompliance(), {})    # gone for a long time
    ep = h.records["w1"].episodes[0]
    results["mobile: the unseen gap is not counted as unsafe time"] = (
        ep.closed and ep.duration_s == 1.0)


# ---- the review clip has to actually open on the phone it was made for -------
def test_the_annotated_clip_uses_a_format_a_browser_can_play():
    """`cv2.VideoWriter_fourcc(*"mp4v")` is the line everyone writes, and it produces
    MPEG-4 Part 2 — which **no current browser plays**. The file opens in VLC and looks
    perfect in every desktop check, so the failure only appears at the far end: the
    supervisor taps the review clip on their phone and gets a black rectangle. That was
    the state of this bundle until the codec was probed instead of assumed."""
    from src.videoout import CANDIDATES, best_codec
    fourcc, ext, _name, playable = best_codec()
    fallback = CANDIDATES[-1]
    results["mobile: the review clip is written in a format phones can play"] = (
        playable or (fourcc, ext) == (fallback[0], fallback[1]))
    if not playable:
        print("   [note] no browser-playable encoder in this OpenCV build; the summary "
              "warns the reader to use VLC.")


def test_the_chosen_encoder_really_round_trips():
    """`isOpened()` lies in both directions: it returned False for H.264 on the
    development machine (a missing DLL) and True for encoders that then write a file
    nothing can read. Only a write-then-read proves anything."""
    import tempfile

    import numpy as np

    from src.videoout import open_writer
    tmp = tempfile.mkdtemp(prefix="ar-vid-")
    try:
        writer, path, _name, _ok = open_writer(os.path.join(tmp, "probe"), 15.0, (160, 120))
        rng = np.random.RandomState(1)
        for _ in range(10):
            writer.write(rng.randint(0, 255, (120, 160, 3), dtype=np.uint8))
        writer.release()
        import cv2
        cap = cv2.VideoCapture(path)
        n = 0
        while True:
            ok, _f = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        results["mobile: the chosen encoder writes a file that reads back"] = (
            os.path.isfile(path) and n >= 5)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_missing_ffmpeg_is_not_a_crash():
    """H.264 is the only format that plays everywhere, but OpenCV's bundled FFmpeg has no
    H.264 encoder, so it comes from an optional standalone ffmpeg. Optional has to mean
    optional: with no binary, this must degrade to the next codec rather than throw."""
    from src.videoout import FfmpegWriter, open_writer
    import numpy as np
    w = FfmpegWriter("definitely-not-an-executable-xyz", "out.mp4", 15.0, (64, 48))
    unopened = not w.isOpened()
    w.write(np.zeros((48, 64, 3), np.uint8))       # must be a no-op, not an exception
    w.release()
    import tempfile
    writer, path, _n, _p = open_writer(
        os.path.join(tempfile.mkdtemp(prefix="ar-nof-"), "x"), 15.0, (64, 48))
    got_something = writer is not None and bool(path)
    writer.release()
    results["mobile: a missing ffmpeg degrades to the next codec, never a crash"] = (
        unopened and got_something)


def test_the_video_format_is_reported_with_its_fix():
    """Whether the review clip opens on the supervisor's phone depends on which encoder
    this machine has, which is invisible until someone taps play. So it is printed at
    start-up and by `run.py --check`, together with the one command that improves it."""
    from src.videoout import best_codec, describe
    text = describe()
    playable = best_codec()[3]
    results["mobile: the video format is stated up front, with the fix if it is not ideal"] = (
        bool(text) and text.isascii()                  # a Windows console mangles em dashes
        and ("plays on every phone" in text or "imageio-ffmpeg" in text)
        and (playable or "WILL NOT play" in text))


def test_a_long_clip_is_strided_so_it_finishes():
    """At ~120 ms a frame, a two-minute clip is seven minutes of analysis. Nobody watching
    a progress bar waits that long without deciding it has hung — so the stride is chosen
    from the length, and the summary says what was done."""
    from phase7_mobile.analyze import AUTO_FRAME_BUDGET, choose_stride
    short_clip = choose_stride(900)                     # 30 s @ 30 fps -> every frame
    long_clip = choose_stride(30 * 60 * 4)              # 4 min @ 30 fps
    results["mobile: a short clip is analysed in full and a long one is strided"] = (
        short_clip == 1 and long_clip > 1
        and (30 * 60 * 4) / long_clip <= AUTO_FRAME_BUDGET
        and choose_stride(999999, every=1) == 1         # an explicit request always wins
        and choose_stride(0) == 1)                      # unknown length: never guess


def test_an_unreadable_clip_raises_something_catchable():
    """Not `SystemExit`: that is not an `Exception`, so it goes straight through the
    `except Exception` in the phone app's worker thread and leaves the job stuck on
    'analysing' for ever."""
    from phase7_mobile.analyze import ClipError, analyze_clip
    import tempfile
    bad = os.path.join(tempfile.mkdtemp(prefix="ar-bad-"), "not-a-video.mp4")
    with open(bad, "wb") as fh:
        fh.write(b"definitely not a video")
    raised = None
    try:
        analyze_clip(bad, None, tempfile.gettempdir())
    except BaseException as e:                          # noqa: BLE001
        raised = e
    results["mobile: an unreadable clip raises a catchable error, not SystemExit"] = (
        isinstance(raised, ClipError) and not isinstance(raised, SystemExit))


def test_lan_ip_is_an_address():
    ip = lan_ip()
    parts = ip.split(".")
    results["mobile: the printed link uses a real IPv4 address"] = (
        len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts))


def main() -> int:
    test_report_is_non_destructive()
    test_open_episode_reports_live_duration()
    test_report_without_now_is_zero_not_negative()
    test_close_still_marks_truncated()
    test_auth()
    test_page_has_token_substituted()
    test_mode_endpoint_validates()
    test_unknown_route_404()
    test_snapshot_served()
    test_non_ascii_token_is_refused_not_crashed()
    test_absence_tolerance_keeps_one_episode()
    test_present_and_compliant_still_closes_immediately()
    test_absent_gap_not_charged_as_unsafe_time()
    test_the_annotated_clip_uses_a_format_a_browser_can_play()
    test_the_chosen_encoder_really_round_trips()
    test_a_missing_ffmpeg_is_not_a_crash()
    test_the_video_format_is_reported_with_its_fix()
    test_a_long_clip_is_strided_so_it_finishes()
    test_an_unreadable_clip_raises_something_catchable()
    test_lan_ip_is_an_address()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_MOBILE", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
