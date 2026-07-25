# AR Safety Monitor — User Guide

**Everything you need to run this system, in order, in plain language.**

This guide is for the person *using* the system, not building it. If you want the technical
design and the measured results, read [README.md](README.md) and the per-phase READMEs it
links to.

---

## Contents

1. [What this system does](#1-what-this-system-does)
2. [What you need](#2-what-you-need)
3. [First-time setup](#3-first-time-setup)
4. [The launcher, option by option](#4-the-launcher-option-by-option)
5. [Using it on a phone at a site](#5-using-it-on-a-phone-at-a-site) ← **the main workflow**
6. [Reading the results](#6-reading-the-results)
7. [Sending feedback back](#7-sending-feedback-back)
8. [When something goes wrong](#8-when-something-goes-wrong)
9. [What it can and cannot do](#9-what-it-can-and-cannot-do)
10. [Privacy and safe use](#10-privacy-and-safe-use)
11. [Command reference](#11-command-reference)

---

## 1. What this system does

It watches a camera and reports, per person, **who is not wearing their PPE and for how
long**.

Concretely, for each frame it:

1. **detects** people, hard hats and safety vests (a YOLOv8 model trained for this);
2. **tracks** each person, so they keep the same number as they move;
3. **decides** who is non-compliant — a bare head belongs to the person whose box contains
   it, and a violation must persist for several frames before it is reported, so a single
   bad frame does not raise an alarm;
4. **names** each person and keeps that name across a brief disappearance, so their safety
   record follows them rather than resetting;
5. **logs** each violation as an episode with a start, an end and a duration.

It also contains three things beyond PPE, built and measured but **not** part of the daily
workflow: assembly-step recognition with out-of-order detection (Phase 3), an AR-glasses
display design (Phase 6), and a speed/accuracy study for running on small hardware
(Phase 4). Launcher options exist for all of them.

**What it is not.** It is not a certified safety system, it does not stop machinery, and it
should not be the only thing watching a site. Treat it as an extra pair of eyes that keeps
notes.

---

## 2. What you need

| | |
|---|---|
| **A Windows PC or laptop** | Windows 10/11. macOS and Linux work too, from the command line. |
| **Python 3.11 or newer** | From <https://www.python.org/downloads/>. During install, **tick "Add python.exe to PATH"** — nothing works without it. |
| **About 5 GB of free disk** | Mostly PyTorch. |
| **The detector file** | `phase2\models\best.pt`. If it is missing, see [Weights & data](README.md#weights--data). Without it, the options that need a camera will not start; the rest still will. |
| **A phone** | Only for [section 5](#5-using-it-on-a-phone-at-a-site). Any Android phone with Chrome, or an iPhone with Safari. Nothing to install on it. |
| **A camera** | The laptop's webcam, or the phone's camera, or a recorded video file. |

A graphics card is **not** required. On a CPU it runs at roughly 6–9 frames per second,
which is enough for compliance monitoring. With an NVIDIA card it is several times faster.

---

## 3. First-time setup

### Windows: double-click `START.bat`

Then press **`1`** — *First-time setup*. It creates a private Python environment inside the
project folder (`.venv`) and installs everything. **This takes several minutes and needs an
internet connection.** It only has to be done once, and it does not touch any other Python
on the machine.

When it finishes, press **`2`** — *Readiness check*. You want to see a list of `[ ok ]`
lines ending in:

```
READY — run `python run.py` for the live demo.
```

Two lines on that report are worth reading:

- **`Detector : found`** in the menu header. If it says MISSING, the model file is not in
  `phase2\models\`.
- **`review video: ...`** at the bottom. This says which video format your machine can
  write. If it mentions `pip install imageio-ffmpeg`, do that — it makes the review clips
  play on every phone, and writes them about 40× faster. See
  [section 8](#8-when-something-goes-wrong).

### macOS / Linux, or if you prefer the command line

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r phase2/requirements.txt
cd phase2 && python run.py --check
```

### Checking it works without a camera or the model

Every piece of logic is unit-tested against synthetic data, so you can confirm the install
before you have hardware or weights. In the launcher that is option **`9`**; by hand it is
the block of `python .../tests/test_*.py` commands in
[README.md](README.md#1-verify-the-code-with-zero-downloads). Each prints `PASS` lines and
one `ALL_… True`.

---

## 4. The launcher, option by option

Double-click **`START.bat`**. The header tells you whether the environment and the detector
are in place before you pick anything.

| key | what it does | needs |
|---|---|---|
| **1** | First-time setup — installs everything | internet |
| **2** | Readiness check — what is installed, what is missing, which video format | — |
| **3** | **Live demo on the webcam.** A window opens with the overlay. `q` quit, `s` screenshot, `r` start/stop recording | camera + model |
| **4** | Same, but on a video file. Drag the file onto the window and press Enter | model |
| **5** | **Phone app / site visit** — see [section 5](#5-using-it-on-a-phone-at-a-site) | see below |
| **6** | Worker-ID measurement: how often a worker who walks out of view is correctly recognised on the way back | — |
| **7** | AR-glasses view: renders the three display modes side by side into `outputs\ar_preview.png` | — |
| **8** | Workflow monitor: replays an assembly step by step, predicts the next step, and flags a deliberately planted out-of-order mistake | — |
| **9** | Run every test | — |
| **E** | Speed test: exports the model and measures how fast this PC runs it | model |
| **0** | Quit |

Options **6, 7, 8 and 9 need no camera, no model file and no downloads** — they are the
quickest way to show what the system does on a machine that has nothing set up.

### About option 3 (live demo)

The window shows a box per person, coloured by the worst violation they currently have
(red = no hard hat, orange = no vest, green = compliant), their name, a status band across
the top, and a panel listing everyone the system has seen. Press `r` to record; the file
lands in `phase2\outputs\` and the terminal prints its name and format.

### About option 8 (workflow monitor)

This is the Phase 3 work and it is unrelated to PPE — it understands *assembly order*.
Watch for two things in the output: `next:` on each line (what it expects to happen next,
with confidence), and `*** MISTAKE (order_violation)` where a step arrived out of order.
Near the bottom it prints `injected fault : CAUGHT` — a mistake was deliberately planted
and it found it.

---

## 5. Using it on a phone at a site

This is the part to hand to someone going to a site. **Press `5` in the launcher.**

```
   [A]  PHONE APP        the phone's OWN camera, with the overlay
   [B]  Analyse a clip   a video already on this PC
   [C]  View from here   this PC's webcam, shown on the phone
```

**Choose [A].** The other two are older paths kept because they are occasionally the right
tool: **[B]** for a video file you already have, **[C]** when the laptop has the good camera
and you just want the picture on a phone.

### 5.1 What [A] does

The **phone becomes the camera and the screen.** The laptop does the thinking. They talk
over WiFi. There is nothing to install on the phone.

You need the phone and the laptop **on the same network**. At a site with no WiFi, turn on
the phone's **personal hotspot** and connect the laptop to it — that works, and it is the
recommended setup because the laptop's address then stays the same every time.

### 5.2 Step by step

1. On the laptop, launcher → `5` → `A`. Wait for the box of text.
2. It prints a web address like `https://192.168.1.156:8443/?t=g-MKXw0eeeMq`, and a square
   QR code if you have installed `qrcode`. **Type that address into the phone's browser, or
   scan the QR.**
3. **The phone will warn you that the connection is not private.** This is expected — see
   [5.3](#53-about-that-security-warning). Tap **Advanced**, then **Proceed**.
4. The app loads. Tap **Start camera**, then **Allow** when the phone asks for camera
   permission. Point the phone at people. Boxes and names appear on the live picture.
5. Tap **Install app** (in the *Report* tab) — or, on an iPhone, **Share → Add to Home
   Screen**. You now have an icon that opens it full-screen like an app. **You only have to
   do steps 2–4 once per phone.**

### 5.3 About that security warning

You are being asked to click through a security warning, so here is exactly why.

Phone browsers only give a web page access to the camera over **HTTPS**. A plain
`http://…` page cannot use the camera at all, no matter how it is written. HTTPS needs a
certificate, and certificates are normally signed by a company the browser already trusts —
which is impossible for a laptop on a building site with no domain name. So the laptop
signs its own, and the browser correctly points out that nobody vouched for it.

To make accepting it an informed decision rather than a habit, **the laptop prints the
certificate's fingerprint**:

```
Certificate SHA-256 (check it matches the one the phone shows):
  FF:B4:41:67:68:86:7C:F5:A9:8D:10:52:4F:33:CF:0E:E9:83:FB:1F:F4:7D:43:E4:FE:DB:DC:CB:CF:69:64:8C
```

On the phone's warning screen, tap the details and compare it. **If they match, the warning
is only telling you the certificate is self-signed.** If they do *not* match, stop — someone
else is answering.

The connection is genuinely encrypted either way. What a self-signed certificate does not
prove is *which* laptop you reached.

### 5.4 The three tabs

**Live** — the camera with the overlay. Underneath: how many people are in view, how many
alerts are active, how many workers are known; the list of current violations; the worker
roster; and a switch between three views:

- **Normal** — the overlay drawn on the camera picture. Use this.
- **Glasses layer** — what an AR headset's projector would emit (bright graphics on black,
  because a see-through lens can only *add* light, so black is transparent).
- **Through glasses** — a simulation of what the wearer would actually see, cropped to the
  narrow field of view a real lens has.

The two glasses views are drawn on the laptop and sent back as pictures, so they lag more
than Normal. They are for showing the design, not for walking around.

**Also on the Live tab:** *Flip camera* switches front/rear. **New walk** forgets every
worker, track and violation and starts a fresh session — use it when you move to a
different area, so one area's numbers do not bleed into the next.

**Record** — for when the live camera will not start (see
[section 8](#8-when-something-goes-wrong)), or when you want a file to keep. Tap **Record or
choose a clip**; the phone's normal camera app opens. Record, accept, and it uploads and
analyses automatically, then plays the annotated result back with a written summary.
**Keep clips to 20–30 seconds** — the laptop analyses at roughly real time, and it does that
*instead of* the live view.

**Report** — the safety record for this walk: workers seen, violation episodes, total
unsafe seconds, and a line per worker. Also **Save photo** (the current view) and **Raw
JSON** (everything, for a spreadsheet). And the **Install** button.

### 5.5 The numbers in the top-right corner

`8.2 fps · 140 ms · live`

- **fps** — how often the boxes are refreshed. **This is not the video frame rate**: the
  picture itself is the phone's own camera and stays smooth. Only the boxes update at this
  rate.
- **ms** — how long the laptop spent on the last frame.
- the word is the state: `live`, `violation`, `camera off`, `loading model…`,
  `no link to laptop`, `another phone is the camera`, `key rejected`.

If the boxes ever look **faded**, they are out of date — the link is struggling. That is
deliberate: a crisp box in the wrong place is more misleading than a faint one.

If a line appears suggesting you restart with `--fps N`, it is worth doing. The tracker was
told to expect a frame rate it is not getting, and until that matches, workers get new
numbers after brief disappearances more often than they should.

### 5.6 Two phones

Only one phone can be **the camera**. A second phone that opens the link sees the same
alerts, roster and report, and its status line says *another phone is the camera*. Two
phones feeding one tracker would mix two different scenes into one worker history and the
report would describe neither.

---

## 6. Reading the results

### On the phone

The Report tab, described above.

### On the laptop

Press `Ctrl+C` in the terminal to stop. It prints a one-line summary. Full results are in
`outputs\site\<clip name>\`:

| file | what it is |
|---|---|
| `annotated.mp4` (or `.webm`) | the clip with the overlay burned in |
| `summary.txt` | a readable page — **start here** |
| `report.json` | everything, for a spreadsheet or another tool |
| `worst_1.jpg` … `worst_3.jpg` | stills of the moments with the most simultaneous violations, at least two seconds apart so they are three different incidents |

`summary.txt` names the video file, states whether frames were skipped, and ends with the
five questions worth answering.

### What the report fields mean

| field | meaning |
|---|---|
| `workers_seen` | how many distinct people it believes it saw. **Expect this to be too high** — see [section 9](#9-what-it-can-and-cannot-do). |
| `identified_by_badge` | how many were confirmed by a printed ArUco badge rather than by appearance. A name from appearance alone is provisional, and is marked `*`. |
| `violation_episodes` | a continuous stretch of non-compliance. One person with no hat for 30 s is **one** episode, not 900 frames. |
| `violation_s` | total seconds non-compliant. Time when the person could not be seen is **not** counted against them. |
| `compliance_pct` | share of the time they were visible in which they were compliant. |
| `truncated: true` | the episode was still open when the session ended, so its real duration is at least this. |
| `ongoing: true` | the episode is still running (a live report). |

---

## 7. Sending feedback back

This is the whole point of putting it on a phone: finding out what breaks somewhere real.

After a site visit, send back:

1. the **annotated video** (`Save video` in the app, or from `outputs\site\…`),
2. **`summary.txt`**,
3. answers to the five questions it ends with:
   - Did the boxes follow the right people?
   - Was anyone missed, or flagged wrongly? **Note the time in the clip.**
   - Did a worker's name or number change when they came back into view?
   - Was the overlay readable outdoors, and was anything in the way?
   - Anything you expected it to notice and it did not?

Everything is timestamped, so "at 0:14 it put a red box on the scaffold" is enough to
reproduce and fix. That is far more useful than "it seemed a bit off".

---

## 8. When something goes wrong

### The phone app

| what you see | what it is | what to do |
|---|---|---|
| *"Your connection is not private"* | Expected — the self-signed certificate | Tap Advanced → Proceed. Compare the fingerprint first ([5.3](#53-about-that-security-warning)) |
| The page will not load at all | Phone and laptop are on different networks | Put both on the same WiFi, or use the phone's hotspot. Corporate/guest WiFi often blocks devices from seeing each other — a hotspot avoids that |
| **"This page was opened over plain http"** | You opened `http://…`, not `https://…` | Use the exact address printed by the laptop. Or use the **Record** tab, which does not need it |
| **"Camera permission was refused"** | The browser is blocking the camera | Tap the padlock or ⓘ next to the address → Site settings → allow Camera. Then reload |
| **"The camera is in use by another app"** | Another app has it | Close that app and tap Start camera again |
| Camera will not start on an **iPhone** | iOS is stricter about untrusted certificates | Use the **Record** tab — it needs neither the certificate nor camera permission |
| **"The laptop rejected this app's access key"** | The key was rotated, or the server was set up fresh | Open the new link the laptop printed. The app picks the new key up automatically |
| **"another phone is the camera"** | Another phone got there first | Normal. Stop the camera on that phone, or just watch |
| **"loading model…"** for more than a minute | The model is still loading, or failed | Look at the laptop terminal for a `[warn]`/`[FAIL]` line |
| **"no link to laptop"** | WiFi dropped, or the laptop went to sleep | Check the laptop is awake and still on the same network |
| The picture freezes | The phone put the browser to sleep | Tap the screen. The app reconnects on its own; the status line tells you when it is stale |
| The screen keeps turning off | The wake lock was refused | Set the phone's screen timeout longer |

### The laptop

| what you see | what it is | what to do |
|---|---|---|
| **`[FAIL] port 8443 is not available`** | It is already running in another window | Use that window, or add `--port 8444` |
| **`review video: MPEG-4 Part 2 … WILL NOT play`** | Your OpenCV has no browser-playable encoder | `pip install imageio-ffmpeg` |
| **`review video: VP8 … For H.264 … pip install imageio-ffmpeg`** | Works on Android, patchy on iPhone, and slow to write | Same fix. Worth doing before a site visit |
| **`[warn] could not create a certificate`** | `cryptography` is missing and there is no `openssl` | `pip install cryptography`. Until then the camera cannot be used, but Record can |
| **`[tune] This laptop is managing about N fps…`** | The tracker was set up for a different rate | Restart with the `--fps N` it suggests |
| **`[warn] could not save the access key`** | The folder is read-only | Fix the permissions, or expect a new link each restart |
| **`weights not found: …best.pt`** | The detector file is missing | See [Weights & data](README.md#weights--data) |
| **`NOT READY`** from the readiness check | Read the `[FAIL]` lines above it | Each one names the problem |
| The video will not play when you open it | An older bundle, or no browser-playable encoder | VLC opens all of them. `pip install imageio-ffmpeg` and re-run to get one that plays anywhere |
| **`Analysing…`** never finishes | A clip that could not be read | It now reports the failure instead of hanging. If it persists, look at the terminal |
| Analysis takes minutes | Long clip on a CPU | Keep clips to 20–30 s. Long ones skip frames automatically and say so |

### Nothing is being detected

In order of likelihood: the light is too low or the people are too small in frame (get
closer — a person should be at least a couple of hundred pixels tall); the camera is
pointing at something else; the confidence threshold is too high (lower
`confidence_threshold` in `phase2\config.yaml`); or the model file is not the trained one.

### Everything is flagged as a violation

The detector is not seeing the hats or vests — usually because they are small, dark, or
oblique. Get closer, and check the annotated video to see what it *did* find. Raise
`debounce_frames` in `phase2\config.yaml` to require a violation to persist longer before
it is reported.

---

## 9. What it can and cannot do

Straight answers, because a demo that oversells itself wastes a site visit.

**It does well:** finding people, hard hats and safety vests in reasonable light (90%+ on
every metric on its test set); attributing a violation to the right person when people are
not overlapping heavily; measuring how long a violation lasted; keeping a name across a
short disappearance.

**Known weaknesses:**

- **Worker identity is the weakest part.** When everyone wears the same PPE, telling them
  apart *by appearance* barely works — measured at **8%** correct re-identification in
  matching uniforms, and it wrongly merges two people about **10%** of the time. This is
  why a 12-second clip of 7 people reports 18 "workers". The fix that works is a printed
  **ArUco badge** on the hat or vest (`phase2\tools\make_worker_tags.py` prints them);
  appearance alone is a best effort.
- **Everything about identity and AR glasses is measured on synthetic tests**, not on real
  site footage. That is the single biggest gap, and one site visit closes it.
- **Crowds confuse it.** Shoulder-to-shoulder, violation boxes get attributed to the wrong
  neighbour and the labels crowd each other.
- **Frame rate on a CPU is 6–9 fps.** Fine for compliance. Not fine for anything fast.
- **The overlay lags on the phone** by 150–300 ms. Fine for judging compliance, not for
  precise registration on a moving scene.
- **No AR glasses hardware has been tested.** Phase 6 designs and simulates the display;
  the numbers about head motion come from synthetic camera shake.
- **The assembly-workflow part (Phase 3) is a different dataset and a different problem.**
  On perfect input its mistake detector has a 6.6% false-alarm rate per step; on its own
  recogniser's output that rises to **11.4%**. It is a demonstration, not a site tool.
- **It has never been run outdoors in bright sun.** Whether the overlay is readable there
  is genuinely unknown — it is question 4 on the feedback list for exactly that reason.

---

## 10. Privacy and safe use

**This system records identifiable people at work.** Please treat it accordingly.

- **Tell people they are being recorded** before you start, and get whatever permission
  your organisation and local law require. A safety tool that surprises people is a
  different kind of problem.
- **The phone link is protected by a secret key in the link, and nothing else.** The
  encryption is real but self-signed, which means anyone who obtains the link, on that
  network, can watch. It is suitable for **a hotspot you control**. Do not run it on public,
  guest or client WiFi.
- **`--no-token` removes even that.** It prints a warning. Only use it on a network that is
  entirely yours.
- **Recordings and reports stay on the laptop**, in `outputs\`. Nothing is uploaded
  anywhere; the system makes no internet connections at all while running. Delete
  `outputs\` when you no longer need it.
- **Names are guesses.** Anything marked `seen` or `*` came from appearance matching, which
  is [unreliable](#9-what-it-can-and-cannot-do). Never treat an appearance-derived name as
  a record of who did what. Only badge-confirmed names identify a person.
- **Do not use the output for discipline** without a human checking the video. The
  false-alarm rate is low but not zero, and the identity matching is weak.

---

## 11. Command reference

Everything the launcher does, as commands. Run from the project folder with the environment
active.

```bash
# --- checks -----------------------------------------------------------------
cd phase2 && python run.py --check         # what is installed, missing, and which codec

# --- live demo (needs the model) --------------------------------------------
cd phase2 && python run.py                 # webcam;  q quit, s screenshot, r record
cd phase2 && python run.py --source clip.mp4
cd phase2 && python run.py --arview glasses     # see-through preview instead of the HUD

# --- the phone app (Phase 8) ------------------------------------------------
python -m phase8_phoneapp.app              # prints an https link + QR
python -m phase8_phoneapp.app --fps 8      # match the rate this laptop achieves
python -m phase8_phoneapp.app --port 8444  # if 8443 is taken
python -m phase8_phoneapp.app --new-token  # rotate the access key
python -m phase8_phoneapp.app --http       # no certificate; Record works, camera does not

# --- analysing clips on the laptop (Phase 7) -------------------------------
python -m phase7_mobile.analyze clip.mp4
python -m phase7_mobile.analyze clips/           # a whole folder
python -m phase7_mobile.analyze clip.mp4 --every 1     # every frame, exact counts

# --- laptop camera, phone as the screen (Phase 7) --------------------------
python -m phase7_mobile.server
python -m phase7_mobile.server --source http://192.168.0.14:8080/video   # IP-camera app

# --- measurements and demos that need nothing -------------------------------
python -m phase5_workid.reid_eval                  # worker re-ID, measured
python -m phase6_arview.preview                    # -> outputs/ar_preview.png
python -m phase3_activity.tas.demo --inject-fault  # workflow monitor
python -m phase4_deploy.edge.bench --imgsz 320     # speed on this PC (after an export)

# --- printed worker badges --------------------------------------------------
python phase2/tools/make_worker_tags.py            # ArUco tags -> reliable names
```

Useful settings live in **`phase2\config.yaml`**:

| setting | what it changes |
|---|---|
| `confidence_threshold` | lower = detects more, including more false positives |
| `debounce_frames` | how long a violation must persist before it is reported |
| `clear_frames` | how long it must be absent before the violation is closed |
| `identity.match_threshold` | higher = fewer wrong names, more workers counted twice |
| `arview.mode` | `composite` (screen), `seethrough` or `glasses` |
| `workid.markers` | maps printed badge numbers to real names |

---

*Questions this guide does not answer are probably in [README.md](README.md) or in the
README inside the relevant `phaseN_*` folder.*
