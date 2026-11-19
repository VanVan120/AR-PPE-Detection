**Training record.** Epochs are 1-based, as in `results.csv`. Training time is the sum of the epoch times `results.csv` records; it excludes start-up, evaluation and time lost to crashes.

| seed | stage | max epochs / patience | epochs run | best epoch (val mAP@50-95) | ended by | uninterrupted early stopping would end at (best) | training time (h) | sessions in `results.csv` |
|---|---|---|---:|---|---|---|---:|---:|
| 0 | 1 | 50 / 12 | 50 | 31 (0.4921) | epoch limit | 28 (16) | 8.08 | 3 |
| 0 | 2 | 20 / 10 | 20 | 13 (0.4741) | epoch limit | 11 (1) | 3.14 | 3 |
| 1 | 1 | 50 / 12 | 36 | 24 (0.5006) | early stopping | 36 (24) | 6.13 | 3 |
| 1 | 2 | 20 / 10 | 19 | 9 (0.4884) | early stopping | 19 (9) | 3.21 | 1 |

**Grouped test set, full_640, stage 2 (`best_grouped.pt` equivalent, the deployed recipe).** P and R are ultralytics' values at its F1-optimal confidence; mAP@50 and mAP@50-95 as in Table 3.

| seed | P | R | mAP@50 | mAP@50-95 | No-Helmet P | No-Helmet R | No-Helmet mAP@50 | No-Helmet mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (published) | 70.8 | 71.3 | 71.1 | 46.2 | 42.9 | 50.7 | 37.9 | 20.8 |
| 1 | 73.2 | 73.7 | 75.4 | 48.5 | 50.6 | 53.3 | 47.7 | 25.6 |
| **mean ± SD** | 72.0 ± 1.7 | 72.5 ± 1.7 | 73.3 ± 3.1 | 47.4 ± 1.6 | 46.8 ± 5.5 | 52.0 ± 1.8 | 42.8 ± 6.9 | 23.2 ± 3.4 |
| range | 70.8–73.2 | 71.3–73.7 | 71.1–75.4 | 46.2–48.5 | 42.9–50.6 | 50.7–53.3 | 37.9–47.7 | 20.8–25.6 |

**Grouped test set, full_640, stage 1 (`best_grouped_stage1.pt` equivalent).** P and R are ultralytics' values at its F1-optimal confidence; mAP@50 and mAP@50-95 as in Table 3.

| seed | P | R | mAP@50 | mAP@50-95 | No-Helmet P | No-Helmet R | No-Helmet mAP@50 | No-Helmet mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (published) | 71.4 | 75.1 | 74.5 | 48.3 | 46.9 | 56.6 | 42.6 | 23.0 |
| 1 | 72.6 | 75.8 | 77.0 | 49.5 | 48.3 | 53.7 | 47.6 | 25.6 |
| **mean ± SD** | 72.0 ± 0.8 | 75.4 ± 0.5 | 75.8 ± 1.8 | 48.9 ± 0.8 | 47.6 ± 1.0 | 55.1 ± 2.0 | 45.1 ± 3.6 | 24.3 ± 1.9 |
| range | 71.4–72.6 | 75.1–75.8 | 74.5–77.0 | 48.3–49.5 | 46.9–48.3 | 53.7–56.6 | 42.6–47.6 | 23.0–25.6 |

**Per class, full_640, stage 2 (`best_grouped.pt` equivalent, the deployed recipe): mean ± SD (range) over seeds 0, 1.**

| class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 72.0 ± 1.7 (70.8–73.2) | 72.5 ± 1.7 (71.3–73.7) | 73.3 ± 3.1 (71.1–75.4) | 47.4 ± 1.6 (46.2–48.5) |
| Helmet | 80.0 ± 1.9 (78.6–81.4) | 89.8 ± 0.5 (89.5–90.2) | 89.3 ± 0.7 (88.8–89.8) | 57.1 ± 0.7 (56.6–57.5) |
| No-Helmet | 46.8 ± 5.5 (42.9–50.6) | 52.0 ± 1.8 (50.7–53.3) | 42.8 ± 6.9 (37.9–47.7) | 23.2 ± 3.4 (20.8–25.6) |
| No-Vest | 70.1 ± 1.3 (69.1–71.0) | 59.1 ± 3.4 (56.7–61.5) | 64.8 ± 5.2 (61.1–68.5) | 39.2 ± 2.7 (37.3–41.1) |
| Person | 82.3 ± 0.2 (82.2–82.4) | 82.4 ± 2.0 (80.9–83.8) | 86.1 ± 1.8 (84.9–87.4) | 64.4 ± 1.0 (63.7–65.0) |
| Vest | 80.9 ± 0.2 (80.8–81.0) | 79.4 ± 0.8 (78.9–80.0) | 83.3 ± 0.7 (82.8–83.8) | 53.1 ± 0.3 (52.9–53.3) |

**Per class, full_640, stage 1 (`best_grouped_stage1.pt` equivalent): mean ± SD (range) over seeds 0, 1.**

| class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 72.0 ± 0.8 (71.4–72.6) | 75.4 ± 0.5 (75.1–75.8) | 75.8 ± 1.8 (74.5–77.0) | 48.9 ± 0.8 (48.3–49.5) |
| Helmet | 80.4 ± 1.4 (79.5–81.4) | 90.4 ± 0.1 (90.3–90.5) | 90.4 ± 0.6 (90.0–90.9) | 58.0 ± 0.6 (57.5–58.4) |
| No-Helmet | 47.6 ± 1.0 (46.9–48.3) | 55.1 ± 2.0 (53.7–56.6) | 45.1 ± 3.6 (42.6–47.6) | 24.3 ± 1.9 (23.0–25.6) |
| No-Vest | 69.5 ± 1.2 (68.7–70.3) | 64.9 ± 2.7 (63.0–66.8) | 70.0 ± 3.8 (67.3–72.7) | 42.0 ± 2.1 (40.5–43.5) |
| Person | 81.3 ± 0.4 (81.0–81.6) | 84.6 ± 1.7 (83.3–85.8) | 88.6 ± 0.7 (88.1–89.1) | 66.0 ± 0.2 (65.8–66.2) |
| Vest | 81.2 ± 1.1 (80.4–82.0) | 82.2 ± 0.2 (82.1–82.3) | 84.7 ± 0.4 (84.4–85.0) | 54.2 ± 0.8 (53.6–54.8) |

**Other test conditions (Table S20): mAP@50 per seed, and mean ± SD.**

| stage | condition | all: seed 0 / seed 1 | all: mean ± SD | all mAP@50-95: mean ± SD | No-Helmet: seed 0 / seed 1 | No-Helmet: mean ± SD |
|---|---|---|---|---|---|---|
| 2 | full_640 | 71.1 / 75.4 | 73.3 ± 3.1 | 47.4 ± 1.6 | 37.9 / 47.7 | 42.8 ± 6.9 |
| 2 | one_per_source_640 | 66.2 / 70.5 | 68.3 ± 3.0 | 45.3 ± 1.4 | 38.5 / 49.4 | 43.9 ± 7.7 |
| 2 | full_480 | 72.0 / 75.3 | 73.7 ± 2.4 | 48.1 ± 1.4 | 40.7 / 49.3 | 45.0 ± 6.1 |
| 2 | full_320 | 71.0 / 72.5 | 71.8 ± 1.1 | 45.2 ± 0.8 | 39.8 / 41.6 | 40.7 ± 1.3 |
| 1 | full_640 | 74.5 / 77.0 | 75.8 ± 1.8 | 48.9 ± 0.8 | 42.6 / 47.6 | 45.1 ± 3.6 |
| 1 | one_per_source_640 | 69.5 / 71.6 | 70.6 ± 1.5 | 46.8 ± 0.3 | 42.1 / 47.6 | 44.9 ± 4.0 |
| 1 | full_480 | 74.4 / 76.9 | 75.7 ± 1.7 | 49.3 ± 1.0 | 41.8 / 48.1 | 45.0 ± 4.4 |
| 1 | full_320 | 73.5 / 74.4 | 74.0 ± 0.6 | 46.6 ± 0.1 | 43.6 / 44.2 | 43.9 ± 0.4 |

**Sessions of the new runs** (from `launcher_seed<N>.json`; the error is the last exception line the session printed).

| seed | session | start | end | exit code | wall time (h) | resumed | error |
|---|---:|---|---|---:|---:|---|---|
| 1 | 1 | 2026-11-19 17:56:50 | 2026-11-19 20:24:25 | 1 | 2.46 | no | RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED_CUDART |
| 1 | 2 | 2026-11-19 20:26:16 | 2026-11-19 21:14:58 | 1 | 0.81 | yes | torch.AcceleratorError: CUDA error: out of memory |
| 1 | 3 | 2026-11-19 21:14:59 | 2026-11-19 21:19:41 | 1 | 0.08 | yes | RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED |
| 1 | 4 | 2026-11-19 21:24:13 | 2026-11-20 03:53:54 | 0 | 6.49 | yes |  |
