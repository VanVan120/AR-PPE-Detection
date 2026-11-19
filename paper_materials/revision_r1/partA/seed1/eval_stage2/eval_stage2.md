# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs_seed1\grouped_s2\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 73.25 | 73.75 | 73.50 | 73.38 | 75.44 | 48.52 |
| one_per_source_640 | 640 | 584 | 64.25 | 73.85 | 68.72 | 68.68 | 70.45 | 46.36 |
| full_480 | 480 | 4174 | 73.46 | 73.92 | 73.69 | 73.58 | 75.34 | 49.07 |
| full_320 | 320 | 4174 | 70.95 | 72.31 | 71.62 | 71.56 | 72.55 | 45.76 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 81.40 | 78.60 | 82.15 | 79.34 |
| No-Helmet | 50.65 | 46.99 | 50.47 | 47.74 |
| No-Vest | 70.99 | 57.21 | 70.50 | 67.05 |
| Person | 82.18 | 67.98 | 81.93 | 79.99 |
| Vest | 81.02 | 70.49 | 82.25 | 80.63 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.17 | 86.98 | 89.85 | 86.77 |
| No-Helmet | 53.26 | 56.72 | 54.88 | 46.61 |
| No-Vest | 61.52 | 62.15 | 62.00 | 61.58 |
| Person | 83.79 | 82.80 | 82.20 | 83.23 |
| Vest | 80.01 | 80.61 | 80.66 | 83.37 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 85.56 | 82.58 | 85.83 | 82.89 |
| No-Helmet | 51.92 | 51.39 | 52.59 | 47.17 |
| No-Vest | 65.91 | 59.58 | 65.97 | 64.20 |
| Person | 82.98 | 74.66 | 82.06 | 81.58 |
| Vest | 80.51 | 75.21 | 81.45 | 81.98 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 89.77 | 87.05 | 90.33 | 87.50 |
| No-Helmet | 47.71 | 49.35 | 49.32 | 41.65 |
| No-Vest | 68.52 | 58.06 | 66.40 | 63.78 |
| Person | 87.41 | 78.32 | 86.22 | 85.17 |
| Vest | 83.78 | 79.48 | 84.41 | 84.64 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 57.53 | 53.35 | 58.26 | 54.64 |
| No-Helmet | 25.65 | 27.11 | 28.06 | 22.73 |
| No-Vest | 41.06 | 36.66 | 39.94 | 36.57 |
| Person | 65.04 | 60.29 | 65.27 | 63.38 |
| Vest | 53.33 | 54.41 | 53.84 | 51.49 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
