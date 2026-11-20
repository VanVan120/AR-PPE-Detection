**Training record.** Epochs are 1-based, as in `results.csv`. Training time is the sum of the epoch times `results.csv` records; it excludes start-up, evaluation and time lost to crashes.

| seed | stage | max epochs / patience | epochs run | best epoch (val mAP@50-95) | ended by | uninterrupted early stopping would end at (best) | training time (h) | sessions in `results.csv` |
|---|---|---|---:|---|---|---|---:|---:|
| 0 | 1 | 50 / 12 | 50 | 31 (0.4921) | epoch limit | 28 (16) | 8.08 | 3 |
| 0 | 2 | 20 / 10 | 20 | 13 (0.4741) | epoch limit | 11 (1) | 3.14 | 3 |
| 1 | 1 | 50 / 12 | 36 | 24 (0.5006) | early stopping | 36 (24) | 6.13 | 3 |
| 1 | 2 | 20 / 10 | 19 | 9 (0.4884) | early stopping | 19 (9) | 3.21 | 1 |
| 2 | 1 | 50 / 12 | 42 | 30 (0.4948) | early stopping | 42 (30) | 7.27 | 1 |
| 2 | 2 | 20 / 10 | 14 | 4 (0.4819) | early stopping | 14 (4) | 2.19 | 1 |

**Grouped test set, full_640, stage 2 (`best_grouped.pt` equivalent, the deployed recipe).** P and R are ultralytics' values at its F1-optimal confidence; mAP@50 and mAP@50-95 as in Table 3.

| seed | P | R | mAP@50 | mAP@50-95 | No-Helmet P | No-Helmet R | No-Helmet mAP@50 | No-Helmet mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (published) | 70.8 | 71.3 | 71.1 | 46.2 | 42.9 | 50.7 | 37.9 | 20.8 |
| 1 | 73.2 | 73.7 | 75.4 | 48.5 | 50.6 | 53.3 | 47.7 | 25.6 |
| 2 | 72.8 | 73.2 | 73.9 | 47.3 | 49.1 | 55.4 | 46.9 | 23.9 |
| **mean ± SD** | 72.3 ± 1.3 | 72.8 ± 1.3 | 73.5 ± 2.2 | 47.3 ± 1.1 | 47.6 ± 4.1 | 53.1 ± 2.3 | 44.2 ± 5.5 | 23.4 ± 2.5 |
| range | 70.8–73.2 | 71.3–73.7 | 71.1–75.4 | 46.2–48.5 | 42.9–50.6 | 50.7–55.4 | 37.9–47.7 | 20.8–25.6 |

**Grouped test set, full_640, stage 1 (`best_grouped_stage1.pt` equivalent).** P and R are ultralytics' values at its F1-optimal confidence; mAP@50 and mAP@50-95 as in Table 3.

| seed | P | R | mAP@50 | mAP@50-95 | No-Helmet P | No-Helmet R | No-Helmet mAP@50 | No-Helmet mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (published) | 71.4 | 75.1 | 74.5 | 48.3 | 46.9 | 56.6 | 42.6 | 23.0 |
| 1 | 72.6 | 75.8 | 77.0 | 49.5 | 48.3 | 53.7 | 47.6 | 25.6 |
| 2 | 72.3 | 75.6 | 75.7 | 48.3 | 49.3 | 55.2 | 47.7 | 24.4 |
| **mean ± SD** | 72.1 ± 0.6 | 75.5 ± 0.3 | 75.8 ± 1.3 | 48.7 ± 0.7 | 48.2 ± 1.2 | 55.2 ± 1.4 | 46.0 ± 3.0 | 24.3 ± 1.3 |
| range | 71.4–72.6 | 75.1–75.8 | 74.5–77.0 | 48.3–49.5 | 46.9–49.3 | 53.7–56.6 | 42.6–47.7 | 23.0–25.6 |

**Per class, full_640, stage 2 (`best_grouped.pt` equivalent, the deployed recipe): mean ± SD (range) over seeds 0, 1, 2.**

| class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 72.3 ± 1.3 (70.8–73.2) | 72.8 ± 1.3 (71.3–73.7) | 73.5 ± 2.2 (71.1–75.4) | 47.3 ± 1.1 (46.2–48.5) |
| Helmet | 80.6 ± 1.7 (78.6–81.7) | 90.0 ± 0.5 (89.5–90.4) | 89.6 ± 0.7 (88.8–90.1) | 57.2 ± 0.5 (56.6–57.5) |
| No-Helmet | 47.6 ± 4.1 (42.9–50.6) | 53.1 ± 2.3 (50.7–55.4) | 44.2 ± 5.5 (37.9–47.7) | 23.4 ± 2.5 (20.8–25.6) |
| No-Vest | 70.3 ± 1.0 (69.1–71.0) | 59.2 ± 2.4 (56.7–61.5) | 64.5 ± 3.7 (61.1–68.5) | 38.9 ± 1.9 (37.3–41.1) |
| Person | 82.2 ± 0.2 (82.1–82.4) | 82.7 ± 1.6 (80.9–83.8) | 86.4 ± 1.4 (84.9–87.4) | 64.5 ± 0.8 (63.7–65.0) |
| Vest | 80.7 ± 0.4 (80.3–81.0) | 78.8 ± 1.2 (77.5–80.0) | 82.7 ± 1.1 (81.6–83.8) | 52.7 ± 0.8 (51.7–53.3) |

**Per class, full_640, stage 1 (`best_grouped_stage1.pt` equivalent): mean ± SD (range) over seeds 0, 1, 2.**

| class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 72.1 ± 0.6 (71.4–72.6) | 75.5 ± 0.3 (75.1–75.8) | 75.8 ± 1.3 (74.5–77.0) | 48.7 ± 0.7 (48.3–49.5) |
| Helmet | 80.8 ± 1.1 (79.5–81.5) | 90.6 ± 0.3 (90.3–90.9) | 90.6 ± 0.5 (90.0–90.9) | 58.1 ± 0.5 (57.5–58.4) |
| No-Helmet | 48.2 ± 1.2 (46.9–49.3) | 55.2 ± 1.4 (53.7–56.6) | 46.0 ± 3.0 (42.6–47.7) | 24.3 ± 1.3 (23.0–25.6) |
| No-Vest | 69.7 ± 0.9 (68.7–70.3) | 64.9 ± 1.9 (63.0–66.8) | 69.5 ± 2.8 (67.3–72.7) | 41.6 ± 1.7 (40.5–43.5) |
| Person | 81.4 ± 0.4 (81.0–81.8) | 84.6 ± 1.2 (83.3–85.8) | 88.4 ± 0.6 (88.1–89.1) | 65.8 ± 0.3 (65.5–66.2) |
| Vest | 80.4 ± 1.6 (78.8–82.0) | 82.1 ± 0.1 (82.0–82.3) | 84.3 ± 0.7 (83.6–85.0) | 53.8 ± 1.0 (52.8–54.8) |

**Other test conditions (Table S20): mAP@50 per seed, and mean ± SD.**

| stage | condition | all: seed 0 / seed 1 / seed 2 | all: mean ± SD | all mAP@50-95: mean ± SD | No-Helmet: seed 0 / seed 1 / seed 2 | No-Helmet: mean ± SD |
|---|---|---|---|---|---|---|
| 2 | full_640 | 71.1 / 75.4 / 73.9 | 73.5 ± 2.2 | 47.3 ± 1.1 | 37.9 / 47.7 / 46.9 | 44.2 ± 5.5 |
| 2 | one_per_source_640 | 66.2 / 70.5 / 68.4 | 68.4 ± 2.1 | 45.2 ± 1.1 | 38.5 / 49.4 / 44.0 | 44.0 ± 5.4 |
| 2 | full_480 | 72.0 / 75.3 / 74.9 | 74.1 ± 1.8 | 48.2 ± 1.1 | 40.7 / 49.3 / 50.8 | 47.0 ± 5.4 |
| 2 | full_320 | 71.0 / 72.5 / 73.9 | 72.5 ± 1.5 | 45.7 ± 1.0 | 39.8 / 41.6 / 47.5 | 43.0 ± 4.0 |
| 1 | full_640 | 74.5 / 77.0 / 75.7 | 75.8 ± 1.3 | 48.7 ± 0.7 | 42.6 / 47.6 / 47.7 | 46.0 ± 3.0 |
| 1 | one_per_source_640 | 69.5 / 71.6 / 70.1 | 70.4 ± 1.1 | 46.5 ± 0.6 | 42.1 / 47.6 / 44.3 | 44.7 ± 2.8 |
| 1 | full_480 | 74.4 / 76.9 / 76.5 | 75.9 ± 1.3 | 49.4 ± 0.7 | 41.8 / 48.1 / 51.0 | 47.0 ± 4.7 |
| 1 | full_320 | 73.5 / 74.4 / 75.1 | 74.3 ± 0.8 | 46.9 ± 0.6 | 43.6 / 44.2 / 46.4 | 44.7 ± 1.5 |

**Sessions of the new runs** (from `launcher_seed<N>.json`; the error is the last exception line the session printed).

| seed | session | start | end | exit code | wall time (h) | resumed | error |
|---|---:|---|---|---:|---:|---|---|
| 1 | 1 | 2026-11-19 17:56:50 | 2026-11-19 20:24:25 | 1 | 2.46 | no | RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED_CUDART |
| 1 | 2 | 2026-11-19 20:26:16 | 2026-11-19 21:14:58 | 1 | 0.81 | yes | torch.AcceleratorError: CUDA error: out of memory |
| 1 | 3 | 2026-11-19 21:14:59 | 2026-11-19 21:19:41 | 1 | 0.08 | yes | RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED |
| 1 | 4 | 2026-11-19 21:24:13 | 2026-11-20 03:53:54 | 0 | 6.49 | yes |  |
| 2 | 1 | 2026-11-20 03:53:54 | 2026-11-20 13:28:57 | 0 | 9.58 | no |  |
