# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs_seed2\grouped_s1\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 72.30 | 75.59 | 73.91 | 73.83 | 75.75 | 48.34 |
| one_per_source_640 | 640 | 584 | 64.29 | 75.08 | 69.27 | 69.24 | 70.13 | 45.89 |
| full_480 | 480 | 4174 | 73.03 | 75.41 | 74.20 | 74.12 | 76.49 | 49.65 |
| full_320 | 320 | 4174 | 71.79 | 74.79 | 73.26 | 73.23 | 75.10 | 47.54 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 81.52 | 78.74 | 81.72 | 80.75 |
| No-Helmet | 49.31 | 45.52 | 52.12 | 49.37 |
| No-Vest | 70.06 | 57.02 | 70.07 | 67.94 |
| Person | 81.78 | 67.21 | 81.49 | 79.78 |
| Vest | 78.82 | 72.96 | 79.72 | 81.12 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.95 | 87.86 | 91.42 | 88.36 |
| No-Helmet | 55.18 | 54.48 | 55.94 | 50.40 |
| No-Vest | 65.00 | 65.35 | 64.13 | 66.34 |
| Person | 84.79 | 83.57 | 84.34 | 84.72 |
| Vest | 82.02 | 84.13 | 81.23 | 84.11 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 85.97 | 83.05 | 86.30 | 84.38 |
| No-Helmet | 52.08 | 49.60 | 53.96 | 49.88 |
| No-Vest | 67.44 | 60.90 | 66.97 | 67.13 |
| Person | 83.26 | 74.50 | 82.89 | 82.18 |
| Vest | 80.39 | 78.15 | 80.47 | 82.59 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.87 | 88.00 | 90.96 | 88.86 |
| No-Helmet | 47.74 | 44.29 | 50.96 | 46.44 |
| No-Vest | 68.42 | 58.67 | 68.41 | 67.55 |
| Person | 88.09 | 79.26 | 87.63 | 87.02 |
| Vest | 83.61 | 80.43 | 84.48 | 85.60 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 58.39 | 54.30 | 58.84 | 56.00 |
| No-Helmet | 24.36 | 25.25 | 27.27 | 23.69 |
| No-Vest | 40.58 | 35.93 | 41.72 | 39.26 |
| Person | 65.52 | 60.01 | 66.27 | 64.81 |
| Vest | 52.83 | 53.94 | 54.13 | 53.96 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
