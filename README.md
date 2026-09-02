# AI-Empowered Workflow Monitoring for AR-Glasses Inspection

A computer-vision system for a civil-engineering **AR-glasses inspection assistant**. It
does two complementary jobs on a live egocentric video feed:

1. **PPE safety** — detect who is and isn't wearing a **hard hat** and **hi-vis vest**, and
   raise deduplicated per-person violation alerts in real time.
2. **Worker identity (Work ID)** — keep each worker's **name and safety record attached to
   the person**, surviving occlusion, re-entry and tracker id changes, and produce a
   per-worker safety report.
3. **Workflow understanding** — recognise the **assembly step** underway, flag **out-of-order**
   steps, and **anticipate the next** step — the "you're here / next: …" guidance an AR
   assistant surfaces.

Built in phases, each self-contained and independently runnable:

| Phase | What it is | Where | State |
|---|---|---|---|
| **1 — Detector** | YOLOv8 fine-tuned to **90%+ on every metric, all 5 PPE classes** | repo root | ✅ |
| **2 — Real-time AR** | detector + person tracking + per-person compliance + **AR HUD overlay** | [`phase2/`](phase2/) | ✅ |
| **3 — Workflow understanding** | Assembly101 step recognition + mistake detection + next-step anticipation | [`phase3_activity/`](phase3_activity/) | ✅ |
| **4 — Edge deployment** | export + quantization + measured latency/accuracy trade-off for on-device use | [`phase4_deploy/`](phase4_deploy/) | ✅ |
| **5 — Worker identity** | **Work ID**: identity that survives occlusion + per-worker safety report | [`phase5_workid/`](phase5_workid/) | ✅ |
| **6 — AR-glasses readiness** | see-through render mode, lens FOV safe zone, head-motion measurement | [`phase6_arview/`](phase6_arview/) | ✅ frozen |
| **7 — Phone link** | take it to a site: offline clip analysis + live view on a phone | [`phase7_mobile/`](phase7_mobile/) | ✅ |
| **8 — Phone app** | installable app; **the phone's own camera** with the overlay on it | [`phase8_phoneapp/`](phase8_phoneapp/) | ✅ |
| **9 — Work ID tracking** | identity that **sticks**: probation, spatio-temporal gating, history-carrying merges, coasting labels, badge range — each an ablation row | [`phase5_workid/`](phase5_workid/#making-the-identity-stick--the-tracking-enhancement) | ✅ |

> Summer-internship project for *AI-Empowered Dynamic Workflow Monitoring for Inspection via
> AR Glasses*.

> ### 📖 **Using the system? Read [USER_GUIDE.md](USER_GUIDE.md).**
> Step-by-step setup, every launcher option, the full phone-app walkthrough, how to read the
> report, a troubleshooting table for every failure it is known to have, and an honest list
> of what it cannot do. This README is the technical overview; the guide is the manual.

---

## Quick start

**Prerequisites:** Python **3.11+**. A GPU is optional (everything runs on CPU — just slower).

### Easiest (Windows): double-click `START.bat`
A menu launcher handles everything — no command line needed:

```
[1] First-time setup     installs everything into a private .venv (once)
[2] Readiness check      shows what's installed / missing
[3] Run LIVE demo        webcam
[4] Run demo on a VIDEO  point it at a file
[5] Worker ID tracking   how well workers are re-identified (Phase 5)
[6] Workflow monitor     step + mistake + next-step (Phase 3)
[7] Verify everything    runs all the tests (no downloads needed)
[8] Edge speed test      how fast it runs on this PC (Phase 4)
```

Options **[5]–[8]** need no camera, no weights and no downloads.

Run **[1]** once, then **[2]** to confirm it's ready, then **[3]**. On macOS/Linux, or to run
things by hand, use the steps below.

### 1. Verify the code with zero downloads
The Phase 3 logic is unit-tested against synthetic fixtures, so you can confirm it all works
**without any model weights or datasets**:

```bash
pip install numpy                                   # the suites below need only numpy
python phase3_activity/tests/test_tas.py            # ALL_TAS True
python phase3_activity/tests/test_mistake.py        # ALL_MISTAKE True
python phase3_activity/tests/test_anticipation.py   # ALL_ANTICIPATION True
python phase3_activity/tests/test_demo.py           # ALL_DEMO True
python phase4_deploy/tests/test_edge.py             # ALL_EDGE True
pip install opencv-python                           # Phase 5 describes image crops
python phase2/tests/test_identity.py                # ALL_IDENTITY True
python phase2/tests/test_workerlog.py               # ALL_WORKERLOG True
python phase5_workid/tests/test_reid_eval.py        # ALL_REID_EVAL True
python phase5_workid/tests/test_badge_eval.py       # ALL_BADGE_EVAL True
python phase5_workid/tests/test_badge_gt_eval.py    # ALL_BADGE_GT True
python phase2/tests/test_arview.py                  # ALL_ARVIEW True
python phase7_mobile/tests/test_mobile.py           # ALL_MOBILE True
python phase8_phoneapp/tests/test_phoneapp.py       # ALL_PHONEAPP True
pip install torch                                   # the pipeline test also trains a tiny model
python phase3_activity/tests/test_pipeline.py       # ALL_PIPELINE True
```

Two things you can *watch* run, still with no downloads:

```bash
python -m phase5_workid.reid_eval                        # worker re-ID, baseline vs enhanced
python -m phase5_workid.reid_eval --ablation             # each tracking mechanism on its own
python -m phase5_workid.badge_eval                       # badge read range
python -m phase3_activity.tas.demo --inject-fault        # workflow monitor
```

### 2. Run the live AR safety demo (Phase 2)
Needs the trained detector at `phase2/models/best.pt` (see **[Weights & data](#weights--data)**)
and a webcam or a video file.

```bash
cd phase2
pip install -r requirements.txt
python run.py --check                              # readiness check — tells you exactly what's missing
python run.py                                      # live webcam   (q quit · s screenshot · r record)
python run.py --source data/clips/site.mp4         # or a video file
```

`--check` is the friendliest starting point: it validates Python, dependencies, the model, the
camera/video source, and any optional features, and prints `READY` or the exact items to fix.

### 3. Reproduce the Phase 3 numbers (optional, needs the dataset)
See **[phase3_activity/README.md](phase3_activity/README.md)** for the Assembly101 download and:
```bash
python -m phase3_activity.tas.evaluate      --fold val           # step recognition
python -m phase3_activity.tas.mistake_eval  --procedure assembly # mistake detection
python -m phase3_activity.tas.anticipation  --procedure assembly # next-step anticipation
```

---

## Phase 1 — the detector (results)

Held-out **test split, 4,190 images**, scored with ultralytics' native validation
(`best_refined.pt`, YOLOv8s):

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Helmet | 97.0% | 95.6% | 96.3% | 97.9% | 82.1% |
| No-Helmet | 93.8% | 93.4% | 93.6% | 97.5% | 80.3% |
| No-Vest | 95.8% | 95.9% | 95.9% | 97.6% | 87.3% |
| Person | 96.6% | 97.8% | 97.2% | 98.9% | 90.4% |
| Vest | 97.3% | 97.7% | 97.5% | 99.1% | 89.2% |
| **All (mean)** | **96.1%** | **96.1%** | **96.1%** | **98.2%** | **85.9%** |

**5 / 5 classes clear 90%** on precision, recall, F1, and mAP@50. The blocker to 90% was never
the architecture — it was **data quantity** for the safety-critical *absence* classes (a person
*without* a hard hat). The full prototype that established this (zero-shot vs fine-tune vs VLM,
threshold tuning, fusion) is written up in **[docs/phase1_prototype.md](docs/phase1_prototype.md)**.

```bash
pip install -r requirements.txt
python run.py --check
python eval_ppe.py --model best_refined.pt --dataset-dir data/ppe_download   # per-class P/R/F1/mAP
```

---

## Phase 2 — real-time tracking & AR overlay ([`phase2/`](phase2/))

Takes the trained detector and makes it **deployment-aware**:

```
webcam / clip ─▶ detect (YOLO) ─▶ track persons (ByteTrack) ─▶ compliance ─▶ AR HUD overlay ─▶ screen / mp4
                                                              (debounce + dedup per person)
```

- **Stable per-person track IDs**; violations deduplicated per person, not per frame.
- **A polished AR heads-up overlay** — a status header, corner-bracket person reticles coloured
  by their worst violation (green = compliant), an active-alerts card, a workflow card (Phase 3),
  and a bottom status bar. ~57 FPS on CUDA with per-stage latency reported.
- **Reality-check** — runs on a self-recorded first-person clip and quantifies the domain gap vs
  the Phase 1 benchmark (an honest answer to "does 90%+ survive worn-camera video?").
- **Optional features** (off by default): Work-ID worker badges, a JSONL event log, and the
  Phase 3 activity backend (step recognition + mistake + anticipation). Enable in
  [`phase2/config.yaml`](phase2/config.yaml); `run.py --check` validates each.
- **Graceful** — no webcam / bad weights / headless all degrade with clear messages.

Full details in **[phase2/README.md](phase2/README.md)**.

---

## Phase 3 — workflow understanding ([`phase3_activity/`](phase3_activity/))

Egocentric **procedural-activity** understanding on the **Assembly101** dataset (Sener et al.,
CVPR 2022), following the supervisor's staged plan. All numbers are on the held-out **val** fold:

| Capability | What it does | Result |
|---|---|---|
| **Step recognition** | label every frame with the assembly step (temporal action segmentation) | MoF **40.5**, Edit 31.2, F1@10/25/50 = 32.7 / 29.1 / 21.6 (matches published C2F-TCN, MoF 37.8) |
| **Mistake detection** | learn the expected order, flag out-of-order steps | **100%** recall on injected order violations, 6.6% false positives per transition |
| **Next-step anticipation** | predict the next step from the steps done so far | top-1 15.5% / top-3 27.5%, beating the frequency baseline (11.3 / 23.2) |

**See all three run together** — no downloads needed:

```bash
python -m phase3_activity.tas.demo --source sample --inject-fault
```

It replays a workflow one step at a time and prints what the system would tell an inspector:
the current step, the anticipated next steps, and a `*** MISTAKE` line when a step arrives out
of order. (START.bat option 5 does this for you.)

The trained model plugs into the Phase 2 seam (`activity.backend: assembly101`), and the mistake /
anticipation models drive the workflow card in the overlay. **Honest scope, methods, and caveats
are documented in [phase3_activity/README.md](phase3_activity/README.md).** Two caveats worth
knowing up front: the 6.6% false-positive rate is measured on ground-truth step streams — on the
streams the trained recogniser actually produces it rises to **11.4%**; and *live* step labels
need a real Assembly101 TSM feature extractor, so offline replay is the meaningful path today.

---

## Phase 5 — worker identity / Work ID ([`phase5_workid/`](phase5_workid/))

Person tracking alone gives `tracker_id`, which is only *motion* continuity: it dies when a
worker is occluded or leaves frame, and the same person returns as a new number — taking
their name and violation history with them. Phase 5 puts a **persistent worker identity**
above the tracker, so the record follows the person.

Two signals, strict precedence: an **ArUco badge is authoritative** (a real name), and
**appearance re-ID bridges the gaps** where no tag is visible. Unbadged people still get a
durable `Worker 1`, `Worker 2`, … so **it works with no props at all**.

Measured with an injected-occlusion protocol (every worker forced to return under a new
track id, ground truth known by construction):

| scenario | re-ID recall | false merge | identities / true |
|---|---|---|---|
| distinct clothing | **100%** | 0.0% | 4.0 / 4 |
| same issued vest, personal helmet | **75%** | **0.0%** | 5.0 / 4 |
| identical PPE | 8% | 9.6% | 7.0 / 4 |

**The honest headline of that first round: appearance re-ID collapses when everyone wears
the same PPE** (8% recall). That is the physical limit of appearance matching, not a tuning
problem — and it is exactly why the badge stays authoritative for real site use. Costs
**0.6 ms/frame**. **Phase 9 below changes that table** by adding what appearance lacks:
time, position and a second look.

Identity makes a **per-worker safety report** possible — violations stored as timed
episodes, so durations are real:

```
worker                 seen  events   unsafe  compliant
Alice Tan                47s      2      12s        74%
Worker 3 *               31s      1       4s        91%
* = identity from appearance only (no badge seen) — treat the name as provisional
```

See it measured, with no camera and no downloads:

```bash
python -m phase5_workid.reid_eval
```

Full method, the threshold trade-off and the limits: **[phase5_workid/README.md](phase5_workid/README.md)**.

---

## Phase 6 — AR-glasses readiness ([`phase6_arview/`](phase6_arview/))

> **Frozen.** The supervisor's direction (September 2026) is that no further AR-glasses
> development is needed. This phase stays as a finished design study — the see-through
> renderer, the lens safe zone and the measured head-motion limits — and nothing below it
> depends on a headset. The phone app (Phase 8) is the delivery target.

Everything above runs on a laptop showing a webcam feed. This phase closes the gap to
something a headset can wear. The models don't change — the **render target** and **head
motion** do.

**Render target.** The Phase 2 HUD draws onto the camera image, which is right for a
monitor and for video-passthrough headsets (Quest 3, Vision Pro). On **optical
see-through** glasses (HoloLens, Xreal, Rokid) it is wrong: black is transparent, so
painting the camera feed would lay video over reality, and the dark translucent cards that
make text readable on a photo emit no light and simply vanish. There is now a
`seethrough` mode — bright strokes on black, larger text, confined to the lens **safe
zone**, with chevrons for workers outside the field of view.

```bash
python -m phase6_arview.preview      # monitor | see-through layer | through the glasses
cd phase2 && python run.py --arview glasses
```

**Head motion, measured.** A head-mounted camera translates the whole scene every frame,
which is not what ByteTrack's motion model expects — it splits one worker across several
track ids. Running real ByteTrack into the Phase 5 identity layer:

| head motion (px/frame) | ByteTrack ids per worker | **+ identity** | false merge |
|---|---|---|---|
| 0 (tripod) | 1.00 | **1.00** | 0.0% |
| 8 | 1.50 | **1.00** | 0.0% |
| 12 | 1.75 | **1.00** | 0.0% |
| 20 | 1.75 | 1.75 | **29.8%** |

**The identity layer completely absorbs head-motion fragmentation up to ~12 px/frame** —
and has an honest breaking point at 20, where it stops repairing *and* starts handing
workers the wrong identity. Full method and limits: **[phase6_arview/README.md](phase6_arview/README.md)**.

---

## Phase 7 — take it to a site on a phone ([`phase7_mobile/`](phase7_mobile/))

The point is a **feedback loop**: get this into a supervisor's hands on a real site so the
next round of work is driven by what actually breaks there. A site usually has no usable
WiFi, so there are two paths.

**Record now, analyse later — needs no network at all.** Record normally on the phone at
the site, then drop the clips in:

```bash
python -m phase7_mobile.analyze site_visit.mp4
```

Each clip produces `annotated.mp4`, a per-worker `report.json`, stills of the worst
moments, and a `summary.txt` that ends with the five questions worth answering after a
visit. Sending the video plus that file back is enough to reproduce any issue.

**Or live on the phone.** With a laptop on the same hotspot, the laptop runs the models and
the phone is the screen — and, with any free IP-camera app, the camera too:

```bash
python -m phase7_mobile.server                  # prints a link (and a QR) for the phone
```

The page shows the live view, alerts, the worker roster, and a switcher between the normal
HUD and the two glasses views. Access is gated by a random key in the link, because the
feed shows identifiable workers — though it is plain HTTP, so it suits a private hotspot,
not an untrusted network. Details and limits: **[phase7_mobile/README.md](phase7_mobile/README.md)**.

---

## Phase 8 — the phone app ([`phase8_phoneapp/`](phase8_phoneapp/))

Phase 7 put the *view* on a phone; the camera was still a laptop webcam or a third-party
IP-camera app somebody had to wire up by hand. This is the app: one link, one icon on the
home screen, and **the phone's own camera** with the overlay drawn on the live picture.

```bash
python -m phase8_phoneapp.app       # prints an https link and a QR
```

Open it on a phone on the same WiFi, tap **Start camera**, then **Install app**. There are
three tabs: **Live** (the camera with boxes, names, alerts and the worker roster, plus the
two AR-glasses views from Phase 6), **Record** (records with the phone's normal camera,
uploads it, and plays the annotated result back — no camera permission needed, so it works
when the live view will not), and **Report** (the per-worker safety record for this walk).

The phone sends a frame and gets back **coordinates**, not a picture, and draws the boxes
itself — so the preview stays smooth at the camera's own rate and only the boxes carry the
network lag. They fade as they age, because a crisp box in the wrong place is worse than a
faint one.

**The phone will warn that the connection is not private — that is expected.** Browsers only
give a page the camera over HTTPS, so the server signs its own certificate; the terminal
prints its fingerprint so accepting it is a check rather than a leap. Measured on this
laptop (CPU): **121 ms** of server time per frame, **6–8 fps** of box refresh, a 12 s clip
analysed in 13 s. Limits and the Windows port-sharing bug this phase uncovered:
**[phase8_phoneapp/README.md](phase8_phoneapp/README.md)**.

---

## Phase 9 — Work ID tracking that sticks ([`phase5_workid/`](phase5_workid/#making-the-identity-stick--the-tracking-enhancement))

The last engineering round, on one instruction: *enhance the tracking effects of the Work
ID of workers*. The phone app had made the weakness visible — 40 frames of a 7-person clip
produced 15 named workers — and reading the identity layer against that gave five causes.
Each became a mechanism, each mechanism an ablation row, and the first-round numbers are
the baseline they are measured against (they reproduce exactly).

- **Probation** — a track must be seen three times before it can become or match a
  worker, and its descriptors are pooled meanwhile. A one-frame false detection never
  becomes a permanent "Worker N".
- **Spatio-temporal gating** — a recently-seen worker who could not have reached the spot
  is vetoed even if the colours agree; a lone worker who could be there is accepted on
  weaker appearance. Camera pan is compensated from the other tracks' motion. Decisive
  appearance (a distinctly dressed worker) still overrides the veto.
- **Merges that carry history** — a badge that names a track holding an anonymous record
  folds that record in; an end-of-session pass merges anonymous fragments whose presence
  intervals never overlap (two records visible at the same moment are two people).
- **Coasting labels** — a lost worker keeps a dashed, dimmed, predicted box for half a
  second, on the laptop overlay and on the phone.
- **Badge read range** — unbound persons' head-and-torso crops are upscaled and searched
  for the badge as well.

Same protocol as Phase 5, 3 seeds, threshold 0.62:

| scenario | baseline: recall / false merge / identities | **enhanced** (shipped): recall / false merge / identities |
|---|---|---|
| distinct clothing | 100% / 0.0% / 4.0 of 4 | **100% / 0.0% / 4.0 of 4** |
| same vest, own helmet | 75% / 0.0% / 5.0 of 4 | **100% / 0.0% / 4.0 of 4** |
| identical PPE | 8% / 9.6% / 7.0 of 4 | **100% / 0.0% / 4.0 of 4** |
| + 6 false detections | 7.0 – 9.3 identities of 4 | **4.0 of 4** |

The ablation says *why*: probation alone fixes the same-vest case and does nothing for
identical kit; the gate alone lifts identical kit to 67% but adds false merges; only
together do they reach 100% with none. Ten seeds agree. The head-motion table is unchanged
up to 12 px/frame.

**Measured limits, on purpose.** The identical-PPE headline holds where a worker returns
near where they vanished. Six workers shoulder to shoulder in identical PPE stay at 17%
recall — people closer than the gate radius are beyond both appearance and position. And
when identical workers *swap places* the gate is confidently wrong rather than uncertain:
false merges rise to 52.7%, against 9.6% for the baseline, the one measured condition
where the enhanced layer is worse. Both are the badge's case. A worker who re-enters
*somewhere else* after 1.5 s costs distinct clothing 17 points (100% → 83%) because
position information has decayed and only decisive appearance rescues a distant return.
The badge crop pass adds 15–25 points of read rate across the 11–19 px badge range,
about half a metre of reliable range on a phone; below 9 px nothing reads.

**Real footage, without labelling anyone.** `badge_gt_eval.py` scores the appearance layer
on a real clip using the workers' printed badges as ground truth — one detector pass,
baseline and enhanced side by side. It needs the weights and a clip of people wearing the
tags; the scorer is unit-tested and the clip is the missing piece.

```bash
python -m phase5_workid.reid_eval --ablation        # the table above, per mechanism
python -m phase5_workid.reid_eval --workers 6       # the crowd limit
python -m phase5_workid.reid_eval --swap            # the swapped-places limit
python -m phase5_workid.badge_eval                  # badge range
python -m phase5_workid.badge_gt_eval site.mp4      # real footage (needs weights + badges)
```

---

## Weights & data

Model weights (`*.pt`) and datasets are **excluded from git** (size), so a fresh `git clone` has
the code but not the large files. To run the parts that need them:

| To run | You need | Where it goes |
|---|---|---|
| Phase 3 **tests** | nothing (synthetic fixtures) | — |
| Phase 2 **live demo** | the trained detector `best.pt` | `phase2/models/best.pt` |
| Phase 1 **eval** | `best_refined.pt` + the PPE dataset | repo root / `data/` |
| Phase 3 **scoring** | Assembly101 features + annotations | `phase3_activity/data/` ([guide](phase3_activity/README.md)) |

- **PPE dataset** — Roboflow `segp-fcn6m/ppe-yezzu-fwbjo` (42k images, CC BY 4.0); train with
  [kaggle_ppe.ipynb](kaggle_ppe.ipynb) → [kaggle_ppe_continue.ipynb](kaggle_ppe_continue.ipynb),
  drop `best_refined.pt` at the root and copy it to `phase2/models/best.pt`.
- **API keys** are read from an environment variable / secret, never hard-coded (see
  [.env.example](.env.example)).

---

## Tech stack
Python 3.11+ · **ultralytics** (YOLOv8) · **supervision** (ByteTrack) · **OpenCV** · PyTorch ·
NumPy · pandas · PyYAML. Phase 3's temporal model is a self-contained MS-TCN; its mistake /
anticipation models are pure-Python (no heavy deps).

## Project structure
```
.
├── README.md                       # ← this file (whole-project overview)
├── run.py · train.py · eval_ppe.py · tune.py · src/   # Phase 1: detector + eval harness
├── docs/phase1_prototype.md        # deep-dive: zero-shot vs fine-tune vs VLM
├── kaggle_ppe*.ipynb · ppe_colab.ipynb                # cloud training
├── phase2/                         # Phase 2: real-time video, tracking, AR overlay
│   ├── run.py · config.yaml · README.md
│   └── src/{detector,tracker,compliance,overlay,workid,eventlog,activity,...}.py
├── phase3_activity/                # Phase 3: Assembly101 workflow understanding
│   ├── README.md
│   ├── tas/{dataset,model,train,evaluate,procedure,anticipation,visualize,infer_seam}.py
│   └── tests/{test_tas,test_mistake,test_anticipation,test_pipeline}.py
├── phase4_deploy/                  # Phase 4: edge deployment (export/quantize/measure)
│   ├── README.md
│   ├── edge/{common,exporter,bench,parity}.py
│   └── tests/test_edge.py
├── phase5_workid/                  # Phase 5/9: worker identity + the tracking enhancement
│   ├── README.md · reid_eval.py · badge_eval.py · badge_gt_eval.py
│   └── tests/{test_reid_eval,test_badge_eval,test_badge_gt_eval}.py
├── phase6_arview/                  # Phase 6: AR-glasses readiness (frozen design study)
├── phase7_mobile/                  # Phase 7: site-clip analyser + live view on a phone
│   ├── analyze.py · server.py · mobile.html · README.md
│   └── tests/test_mobile.py
└── phase8_phoneapp/                # Phase 8: installable app, the phone IS the camera
    ├── app.py · certs.py · make_icons.py · README.md
    ├── static/{index.html,app.js,sw.js,icon-*.png}
    └── tests/test_phoneapp.py
```

## Roadmap
- ✅ **Phase 1** — PPE detector @ 90%+ on all metrics, all classes
- ✅ **Phase 2** — real-time tracking + AR overlay + reality-check (+ optional Work-ID / event log)
- ✅ **Phase 3** — workflow understanding: step recognition → mistake detection → anticipation
- ✅ **Phase 4** — edge deployment readiness: ONNX/quantized export, measured latency + accuracy parity
- ✅ **Phase 5–6** — worker identity that survives occlusion · see-through render + head-motion limits (**Phase 6 frozen**: no further AR-glasses development)
- ✅ **Phase 7–8** — site-clip analyser · installable phone app running on the phone's own camera
- ✅ **Phase 9** — Work ID tracking that sticks: probation · spatio-temporal gating · history-carrying merges · coasting labels · badge range, each measured as an ablation row
- ⬜ **Next** — a real clip with printed badges through `badge_gt_eval.py`, to put a real-footage number beside the synthetic ones · the paper

## Credits & license
- **PPE dataset:** Roboflow Universe `segp-fcn6m/ppe-yezzu-fwbjo` — **CC BY 4.0**.
- **Assembly101:** Sener et al., *"Assembly101"*, CVPR 2022 — **CC BY-NC 4.0** (non-commercial).
- **Built with:** [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 and Roboflow
  [supervision](https://github.com/roboflow/supervision).
- **Code license:** see [LICENSE](LICENSE).
