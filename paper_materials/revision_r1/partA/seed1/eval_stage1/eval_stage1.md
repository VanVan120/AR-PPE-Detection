# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs_seed1\grouped_s1\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 72.59 | 75.77 | 74.15 | 74.08 | 77.04 | 49.47 |
| one_per_source_640 | 640 | 584 | 64.28 | 75.06 | 69.25 | 69.20 | 71.63 | 47.01 |
| full_480 | 480 | 4174 | 73.37 | 74.60 | 73.98 | 73.92 | 76.86 | 50.01 |
| full_320 | 320 | 4174 | 71.42 | 73.69 | 72.53 | 72.51 | 74.39 | 46.65 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 81.37 | 79.04 | 83.11 | 81.73 |
| No-Helmet | 48.31 | 46.36 | 47.93 | 48.83 |
| No-Vest | 70.33 | 55.99 | 71.27 | 66.99 |
| Person | 80.98 | 66.87 | 81.38 | 79.99 |
| Vest | 81.98 | 73.15 | 83.15 | 79.54 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.52 | 87.05 | 90.01 | 87.15 |
| No-Helmet | 53.71 | 56.72 | 51.59 | 47.42 |
| No-Vest | 66.78 | 64.18 | 66.16 | 65.45 |
| Person | 85.80 | 85.54 | 83.58 | 84.78 |
| Vest | 82.06 | 81.83 | 81.67 | 83.63 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 85.70 | 82.85 | 86.43 | 84.35 |
| No-Helmet | 50.87 | 51.02 | 49.70 | 48.11 |
| No-Vest | 68.51 | 59.81 | 68.62 | 66.21 |
| Person | 83.32 | 75.06 | 82.47 | 82.32 |
| Vest | 82.02 | 77.25 | 82.40 | 81.54 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.85 | 88.24 | 91.29 | 88.68 |
| No-Helmet | 47.63 | 47.65 | 48.09 | 44.19 |
| No-Vest | 72.70 | 62.48 | 71.21 | 67.14 |
| Person | 89.06 | 80.53 | 88.38 | 87.08 |
| Vest | 84.96 | 79.23 | 85.32 | 84.86 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 58.41 | 54.25 | 58.87 | 55.22 |
| No-Helmet | 25.61 | 26.03 | 27.29 | 23.58 |
| No-Vest | 43.54 | 39.49 | 42.70 | 38.03 |
| Person | 66.17 | 61.55 | 66.51 | 64.31 |
| Vest | 53.63 | 53.73 | 54.70 | 52.11 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
