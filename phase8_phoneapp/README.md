# Phase 8 — the phone app

Phase 7 put the *view* on a phone. The camera was still a laptop webcam, or a third-party
IP-camera app that somebody had to install and wire up by hand with an IP address.

This is the app. One link, one icon on the home screen, and **the phone's own camera** with
the safety overlay drawn on the live picture. Nothing to record, nothing to copy off the
phone, nothing to send anyone.

```bash
python -m phase8_phoneapp.app
```

The terminal prints an address and a QR. Open it on a phone on the same WiFi or hotspot,
tap **Start camera**, then **Install app** to put an icon on the home screen.

---

## What the app does

| tab | what it is for |
|---|---|
| **Live** | the phone's camera with boxes, names and violations drawn on it; alerts and worker roster underneath; a switch between the normal view and the two AR-glasses views from Phase 6 |
| **Record** | records with the phone's normal camera, uploads it, analyses it here and plays the annotated result back with a summary — **no camera permission needed for this page**, so it works when the live view will not |
| **Report** | the per-worker safety record for this walk: who was seen, how long each was non-compliant, how many episodes |

**New walk** clears the workers, tracks and violations without restarting anything or
reloading the model.

---

## How the work is split, and why

The phone captures a frame, posts it to the laptop, and gets back **the boxes as
coordinates**. It draws them itself, over its own live preview.

The obvious alternative — send the annotated picture back — was rejected deliberately. The
picture is the part the eye notices; routing it through WiFi twice would make the whole
view stutter at the network's pace. This way the preview runs at the camera's own rate and
only the boxes carry the round-trip lag. When that lag grows, the boxes **fade** rather
than sitting there crisp and wrong: a stale box drawn confidently in the wrong place is
more misleading than a faint one.

Coordinates come back as **fractions of the frame**, not pixels, because the phone uploads
a 640 px frame and displays a full-resolution preview. Pixel coordinates would be offset by
the scale factor — which looks exactly like a tracking bug and would be chased as one.

---

## The certificate warning

**The phone will say "your connection is not private". That is expected. Tap Advanced →
Proceed.**

Browsers only hand a page the camera on a *secure origin*. `http://192.168.0.14:8443` is
not one, so a plain-HTTP page cannot do this at all, however it is written. The server
therefore makes its own certificate — and nobody has signed it, hence the warning.

Being told to click through a security warning is a bad habit, so the terminal prints the
certificate's **SHA-256 fingerprint**, and the phone shows the same value under the
warning's details. If they match, the warning is only saying "self-signed"; it is not
saying someone is in the middle. The certificate is cached and reused, so this is a
once-per-phone step, and it is regenerated when the laptop's IP changes.

If it will not proceed at all (some locked-down iPhones), **use the Record tab** — that
path needs neither a certificate nor camera permission.

---

## Access

A random key is minted on first run and embedded in the printed link; every request is
checked against it in constant time, and anything else gets a 403.

The key is **saved to `.state/token` and reused across restarts**. The installed home-screen
icon opens a fixed URL, so minting a fresh key on every start-up would silently break that
icon every time the laptop rebooted, with the app opening straight into a 403 and no clue
why. `--new-token` rotates it deliberately. The app also takes the key from the URL first
and remembers it, so pasting a fresh link always repairs an app whose saved key has gone
stale.

**This is not strong security.** It is a self-signed certificate on a local network: the
traffic is encrypted, but nothing proves *which* laptop you are talking to, and anyone who
has the link can watch. Fine on a hotspot you control; not a substitute for a real
deployment, and it streams identifiable people.

---

## What was measured

On the development laptop (CPU, no GPU), against `phase2/data/clips/sample_walkthrough.mp4`
replayed as if it were a phone camera:

| | |
|---|---|
| server time per frame | **121 ms** (detect → track → comply → identify → log) |
| end-to-end round trip | **6–8 fps** over loopback |
| 12 s clip analysed (Record tab) | **13 s**, producing a 2.4 MB annotated MP4 |

The **live preview is not what runs at 6–8 fps** — that is the camera's own rate and stays
smooth. 6–8 fps is how often the boxes are refreshed. On a machine with CUDA both numbers
improve; the split of work does not change.

---

## Honest limits

- **The laptop is still required.** It runs the model. A phone-only version means a mobile
  runtime (ONNX Runtime Web or TFLite), the detector converted and quantised, and NMS
  reimplemented on the client — a substantial piece of work with nothing measured yet to
  say it would be fast enough on a mid-range phone. Phase 4 already exports ONNX, so the
  starting point exists.
- **Boxes lag by the round trip**, roughly 150–300 ms on WiFi. Fine for judging compliance;
  not enough for anything that must register precisely on a fast-moving scene.
- **Identity churn is the weakest part**, and this phase makes it easy to see: 40 frames of
  a 7-person clip produced 15 named workers. That is the Phase 5 finding (appearance re-ID
  recall of **8%** when everyone is in matching PPE) meeting a low frame rate, not a new
  fault. ArUco badges fix it; nothing else measured does.
- **Set `--fps` near what the laptop actually manages.** ByteTrack turns
  `lost_track_buffer` into a span of *real time* using the frame rate it was given at
  construction — told 15 while receiving 6, its memory of an occluded person covers less
  than half the time it should, and people return as new workers. The app measures the real
  rate and says so on screen rather than correcting it silently, because the fix is a flag
  on the laptop.
- **Android Chrome is the tested path.** iOS Safari is stricter about untrusted
  certificates; if the live camera will not start there, Record still will.
- **One phone at a time can be the camera.** A second phone gets the view but is refused
  the frames — two phones feeding one tracker would interleave two different scenes into a
  single track history, and the violation log would describe neither.

---

## Tests

```bash
python phase8_phoneapp/tests/test_phoneapp.py     # ALL_PHONEAPP True
```

30 checks: the camera claim (including it lapsing when a phone goes quiet), coordinate
normalisation, certificate creation/reuse/renewal, key persistence and rotation, the
403 paths including a non-ASCII key, byte-range playback, upload-name sanitisation, and
two that exist purely because they could otherwise only fail on someone else's phone at a
site:

- **every element id the script touches exists in the page** — one typo makes `$()` return
  null, the first property set throws during load, and the app is a dead grey rectangle on
  a device nobody can attach a debugger to;
- **nothing is loaded from outside the local network** — no CDN, no font host. A site
  laptop is often on a hotspot with no route out, and it should not be announcing itself to
  anyone's server either.

### A bug this phase caught

Running two copies during testing revealed that **on Windows, `socketserver`'s default
`SO_REUSEADDR` lets a second process bind a port another process is already listening on,
with no error at all**. Requests then go to whichever process the OS picks: two models
running, each seeing a different half of the frames, identities and violations split across
two sessions, and nothing in the terminal or on the phone hinting at it. Double-clicking the
launcher twice is all it takes. Fixed with `SO_EXCLUSIVEADDRUSE` plus a clear message on
the second start, and guarded by
`test_phoneapp.py::test_two_copies_cannot_share_the_port`.
