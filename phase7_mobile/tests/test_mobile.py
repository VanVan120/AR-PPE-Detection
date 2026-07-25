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
    test_lan_ip_is_an_address()
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), "-", k)
    ok = all(results.values())
    print("ALL_MOBILE", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
