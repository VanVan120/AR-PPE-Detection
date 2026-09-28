# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs\grouped_s1\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 71.39 | 75.11 | 73.20 | 73.06 | 74.47 | 48.33 |
| one_per_source_640 | 640 | 584 | 63.91 | 72.37 | 67.88 | 67.85 | 69.49 | 46.52 |
| full_480 | 480 | 4174 | 71.88 | 74.60 | 73.22 | 73.09 | 74.45 | 48.62 |
| full_320 | 320 | 4174 | 72.12 | 72.82 | 72.47 | 72.42 | 73.54 | 46.48 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 79.46 | 78.11 | 80.72 | 81.56 |
| No-Helmet | 46.89 | 46.35 | 45.27 | 49.12 |
| No-Vest | 68.69 | 56.34 | 69.76 | 68.10 |
| Person | 81.56 | 67.32 | 82.01 | 81.36 |
| Vest | 80.36 | 71.42 | 81.64 | 80.48 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 90.34 | 86.16 | 90.77 | 86.37 |
| No-Helmet | 56.57 | 52.99 | 53.13 | 49.60 |
| No-Vest | 63.03 | 59.70 | 63.71 | 62.54 |
| Person | 83.34 | 80.50 | 83.11 | 81.76 |
| Vest | 82.28 | 82.51 | 82.29 | 83.81 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 84.55 | 81.94 | 85.45 | 83.90 |
| No-Helmet | 51.28 | 49.45 | 48.88 | 49.36 |
| No-Vest | 65.73 | 57.97 | 66.60 | 65.20 |
| Person | 82.44 | 73.32 | 82.56 | 81.56 |
| Vest | 81.31 | 76.56 | 81.96 | 82.11 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 89.96 | 86.60 | 90.61 | 87.46 |
| No-Helmet | 42.56 | 42.05 | 41.82 | 43.57 |
| No-Vest | 67.31 | 58.28 | 66.91 | 65.43 |
| Person | 88.07 | 80.02 | 87.61 | 86.18 |
| Vest | 84.42 | 80.48 | 85.28 | 85.07 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 57.51 | 53.20 | 58.19 | 55.17 |
| No-Helmet | 22.97 | 25.64 | 22.18 | 21.92 |
| No-Vest | 40.55 | 36.08 | 40.59 | 37.60 |
| Person | 65.83 | 61.40 | 66.37 | 64.35 |
| Vest | 54.79 | 56.28 | 55.75 | 53.37 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
