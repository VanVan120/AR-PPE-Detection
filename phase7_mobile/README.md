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
    annotated.mp4     the clip with the AR overlay burned in
    report.json       per-worker violations, durations, compliance
    summary.txt       a page-long readable summary + what feedback to send back
    worst_1..3.jpg    stills of the moments with the most simultaneous violations
```

`summary.txt` ends with the five questions worth answering after a site visit (did boxes
follow the right people, was anyone missed, did a name change when someone reappeared, was
the overlay readable outdoors, what did it fail to notice). Sending back `annotated.mp4`
plus that file is enough to reproduce any issue, because everything is timestamped.

Long clips: `--every 2` or `--every 3` processes every Nth frame and is much faster.

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

### A bug this phase caught

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
- **`--every` on the analyser** speeds up long clips by skipping frames. Timestamps stay
  in clip time and the tracker is told the effective rate, but the compliance debounce
  counts *processed* frames — so with `--every 3` a violation must persist three times
  longer in real time before it fires. Use `--every 1` when the exact number of violations
  matters; use 2–3 when you just want a fast look at a long clip.
