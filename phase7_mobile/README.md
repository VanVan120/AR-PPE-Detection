# Phase 7 — Phone link: take it to a site

The point of this phase is a **feedback loop**: get the system into a supervisor's hands
on a real site, so the next round of work is driven by what actually breaks there rather
than by guesses.

A development site usually has no usable WiFi, and carrying a laptop around one is
awkward — so there are **two paths**, and the offline one needs no network at all.

---

## Path A — record now, analyse later (works anywhere)

The robust option. Record normally with the phone camera at the site. No app, no pairing,
no signal. Afterwards, drop the clips in:

```bash
python -m phase7_mobile.analyze site_visit.mp4
python -m phase7_mobile.analyze clips/          # a whole folder
```

Each clip produces a review bundle:

```
outputs/site/<clip-name>/
    annotated.mp4     the clip with the AR overlay burned in  (.webm on some machines —
                      see "The codec trap" below; summary.txt names the actual file)
    report.json       per-worker violations, durations, compliance
    summary.txt       a page-long readable summary + what feedback to send back
    worst_1..3.jpg    stills of the moments with the most simultaneous violations
```

`summary.txt` ends with the five questions worth answering after a site visit (did boxes
follow the right people, was anyone missed, did a name change when someone reappeared, was
the overlay readable outdoors, what did it fail to notice). Sending back the annotated
video plus that file is enough to reproduce any issue, because everything is timestamped.

Long clips choose their own frame stride so the wait stays predictable — roughly 900
frames are analysed, so anything up to about 30 s is done in full and longer clips skip
frames, which `summary.txt` states. `--every 1` forces every frame; `--every 3` forces a
coarse pass.

### The codec trap

`cv2.VideoWriter(path, fourcc(*"mp4v"), ...)` is the line everyone writes, and it produces
**MPEG-4 Part 2 — which no current browser plays.** The file opens perfectly in VLC, so
nothing looks wrong on the machine that made it; the failure appears only when the
supervisor taps the clip on their phone and gets a black rectangle. This bundle shipped
that way until [`phase2/src/videoout.py`](../phase2/src/videoout.py) started **probing**:
it writes a few frames with each candidate encoder, reads them back, and keeps the first
that survives — because `isOpened()` lies in both directions.

Measured on the development laptop, same 12 s clip:

| encoder | needs | encode speed | size | plays on a phone |
|---|---|---|---|---|
| **H.264** | `pip install imageio-ffmpeg` | **587 fps** | **1.1 MB** | everywhere, incl. iPhone |
| VP8 / WebM | nothing extra | 15 fps | 8.8 MB | Android, Chrome, Firefox |
| VP9 / WebM | nothing extra | 2.7 fps | 8.0 MB | as VP8, but slower than the detector |
| MPEG-4 Part 2 | nothing extra | 522 fps | 3.3 MB | **no** |

`imageio-ffmpeg` is in `phase2/requirements.txt` for that reason. `python run.py --check`
prints which one this machine will actually use.

---

## Path B — live view on the phone

Needs the phone and a laptop on the same WiFi or phone hotspot. The laptop runs the
models; the phone is the screen, and optionally also the camera.

```bash
python -m phase7_mobile.server                                  # laptop webcam
python -m phase7_mobile.server --source http://192.168.0.14:8080/video   # phone camera
python -m phase7_mobile.server --arview glasses
```

The terminal prints a link (and a QR if `qrcode` is installed). Open it on the phone.
The page shows the live annotated view, active alerts, the worker roster with `badge` /
`seen` provenance, a view switcher (**Normal / Glasses layer / Through glasses**), a
snapshot button and the per-worker report.

To use the **phone as the camera**, install any free IP-camera app, start it, and pass its
URL as `--source`. The phone then films *and* displays, with the laptop in a bag doing the
work.

### Access

The feed shows identifiable workers, so it is not left open. A random key is minted at
start-up and embedded in the printed link; requests without it get **403**. The comparison
is constant-time, so the key cannot be recovered a character at a time by timing.

**This is not strong security, and the README does not pretend otherwise.** It is plain
HTTP on the local network: fine on a private hotspot, not on an untrusted or public one.
Anyone who obtains the link can watch. `--no-token` disables the key entirely and prints a
warning; use it only on a hotspot you control.

---

## Why a shared pipeline

`run.py` (live demo), the phone server, and the offline analyser all need the same
per-frame work: detect → track → compliance → worker identity → history. That core now
lives once in [`phase2/src/pipeline.py`](../phase2/src/pipeline.py) as `SafetyPipeline`.
Three copies would drift, and a fix applied to one would silently miss the others.

```python
pipe = SafetyPipeline(cfg, frame_rate=30)
res = pipe.process(frame, frame_no, elapsed_s)   # annotated frame + alerts + roster
pipe.close(elapsed_s, frame_no)
pipe.report()
```

### Bugs this phase caught

An adversarial review of the new code raised 24 candidate defects; 9 were refuted on
inspection and 15 confirmed. The ones worth knowing about:

| where | defect |
|---|---|
| `report` (below) | polling the report closed open violation episodes |
| `_authorised` | `?t=%FF` made `compare_digest` raise on a non-ASCII string — traceback into the terminal showing the access link, empty body instead of 403, triggerable **without the key** |
| `_stream` | when the producer stalled, the loop skipped every write, so it never noticed a closed socket and spun forever — and the phone opens a new stream on each view switch and unlock, leaking a thread each time |
| `Session.report` | read `WorkerHistory.records` from the HTTP thread while the capture thread inserted into it → `RuntimeError: dictionary changed size during iteration` mid-request |
| `Session._run` | a dead feed was indistinguishable from a live one; the phone kept showing the last good frame as though nothing were wrong |
| `WorkerHistory.update` | a **single** frame in which the detector dropped a worker ended their violation, so one missed detection split one violation into two |
| `analyze.main` | one unreadable file aborted an entire folder — and a folder pulled off a phone very often has exactly one |
| `analyze` stills | ties broke on frame number, so the three "worst moments" were three consecutive frames of the same incident |
| `analyze_clip` | the annotated video was written in a format **no browser plays** (above) |
| `analyze_clip` | an unreadable clip raised `SystemExit`, which is not an `Exception` — so the phone app's worker thread never caught it and the job sat on "analysing" for ever |
| `analyze_clip` | a clip whose header reports a NaN frame rate made every timestamp NaN; `or 30.0` does not catch NaN, because NaN is truthy |

All are fixed and guarded by tests. The report bug is the most instructive:

### The report bug

`/api/report` originally called `pipe.close()`, which **closes every open violation
episode**. Because the phone polls that endpoint, merely *looking* at the report ended
each ongoing violation, and the next frame started a fresh one — shattering one long
violation into a string of short ones and corrupting the durations. Observed directly:
polling three times drove the `truncated` count from 3 to 7 with no change on site.

`WorkerHistory.report(now_s)` is now **non-destructive** and reports an ongoing episode's
elapsed time instead of closing it (`"ongoing": true`). Closing still happens exactly
once, at end of session. `tests/test_mobile.py::test_report_is_non_destructive` guards it.

---

## Tests

```bash
python phase7_mobile/tests/test_mobile.py     # ALL_MOBILE True
```

Covers the non-destructive report, live durations for ongoing episodes, token enforcement
(missing and wrong), the token being substituted into the page with no placeholder left,
view-switch validation, unknown routes, snapshots, and the printed address.

## Honest limits

- **Latency.** The laptop does the work and the phone shows the result, so the view lags
  the world by the pipeline time (~60 ms) plus WiFi. Fine for reviewing compliance, not
  for anything requiring precise registration on a moving scene.
- **Not a native app.** A real Android app running the model on-device would be the
  "proper" answer, but it needs a mobile runtime, a camera pipeline and sideloading — a
  much larger piece of work with nothing measured to justify it yet. The web page works on
  any phone with no install, which is what a one-off site test actually needs.
- **Plain HTTP.** No TLS, so the stream is readable by anyone who can see the traffic on
  that network.
- **Frame striding on long clips** keeps the wait predictable, but the compliance debounce
  counts *processed* frames — so at `--every 3` a violation must persist three times longer
  in real time before it fires. `summary.txt` states the stride that was used. Pass
  `--every 1` when the exact number of violations matters.
