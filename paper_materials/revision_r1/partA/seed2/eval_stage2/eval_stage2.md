# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs_seed2\grouped_s2\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 72.81 | 73.21 | 73.01 | 72.82 | 73.93 | 47.26 |
| one_per_source_640 | 640 | 584 | 66.75 | 70.29 | 68.48 | 68.40 | 68.44 | 44.79 |
| full_480 | 480 | 4174 | 71.61 | 74.40 | 72.98 | 72.77 | 74.87 | 48.55 |
| full_320 | 320 | 4174 | 72.40 | 73.20 | 72.80 | 72.72 | 73.89 | 46.61 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 81.69 | 80.44 | 79.54 | 81.27 |
| No-Helmet | 49.08 | 49.20 | 48.99 | 51.33 |
| No-Vest | 70.87 | 58.95 | 69.32 | 68.05 |
| Person | 82.09 | 69.69 | 79.98 | 79.62 |
| Vest | 80.33 | 75.46 | 80.23 | 81.73 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.41 | 85.91 | 91.58 | 87.49 |
| No-Helmet | 55.38 | 50.75 | 58.07 | 53.89 |
| No-Vest | 59.29 | 55.65 | 60.15 | 60.73 |
| Person | 83.46 | 80.83 | 83.73 | 82.91 |
| Vest | 77.52 | 78.33 | 78.47 | 80.97 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 85.83 | 83.09 | 85.14 | 84.27 |
| No-Helmet | 52.04 | 49.96 | 53.15 | 52.58 |
| No-Vest | 64.57 | 57.25 | 64.41 | 64.18 |
| Person | 82.77 | 74.85 | 81.81 | 81.23 |
| Vest | 78.90 | 76.87 | 79.34 | 81.35 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.13 | 86.59 | 90.68 | 88.12 |
| No-Helmet | 46.91 | 44.00 | 50.85 | 47.48 |
| No-Vest | 63.99 | 54.67 | 63.97 | 64.36 |
| Person | 87.05 | 78.19 | 86.40 | 85.61 |
| Vest | 81.56 | 78.74 | 82.45 | 83.86 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 57.35 | 53.34 | 58.28 | 55.69 |
| No-Helmet | 23.93 | 24.45 | 27.42 | 23.44 |
| No-Vest | 38.41 | 34.12 | 38.97 | 37.36 |
| Person | 64.89 | 59.06 | 65.46 | 63.63 |
| Vest | 51.74 | 52.98 | 52.61 | 52.91 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
