# Segmentation Refinement Audit for v3i Proposals

## Scope

No new YOLO or segmentation training was run.

This audit applies an existing DeepLabV3+ segmentation checkpoint to YOLO proposal crops from:

```text
v3i archaeological_object
YOLOv8n 640
conf = 0.05
```

The goal was to test whether mask-derived features can filter YOLO proposals before a downstream refinement stage.

## Inputs

| Input | Value |
|---|---|
| candidates | `04_detection_yolo/reports/proposals/v3i_conf005/filter_audit/candidate_features.csv` |
| segmentation checkpoint | `03_multiclass_segmentation_deeplab/runs/kaggle/collect_models/all/deeplab_5class_43_best.pth` |
| task | `archaeology_5class` |
| encoder | `resnet34` |
| image size | `256` |
| mask threshold | `0.5` |

Important caveat: the DeepLab model was trained on segmentation `.npy` geodata patches, while current YOLO proposal crops are image crops from the detection pipeline. This is therefore an exploratory domain-transfer check, not a final segmentation benchmark.

## Baseline

Use the original proposal-stage baseline:

| Metric | Value |
|---|---:|
| Validation images | 68 |
| YOLO proposals | 229 |
| GT objects | 108 |
| Covered GT @ IoU0.3 | 69 |
| Coverage @ IoU0.3 | 0.639 |
| FP candidates @ IoU0.3 | 149 |
| FP/image @ IoU0.3 | 2.19 |

Note: `refined_candidates.csv` contains only proposals and their matched GT ids. It cannot recover the full GT denominator by itself. Earlier auto-reporting incorrectly displayed `GT = 67` and `coverage = 1.0`; the config/script now carry the true baseline values explicitly.

## Mask Feature Summary

TP/FP are defined at proposal level:

```text
TP: max_iou_with_gt >= 0.3
FP: max_iou_with_gt < 0.3
```

| Feature | TP median | TP IQR | FP median | FP IQR | Signal |
|---|---:|---:|---:|---:|---|
| `foreground_fraction` | 0.126 | 0.092-0.179 | 0.081 | 0.040-0.127 | Weak/moderate: TP tends to have more foreground, but overlap is large. |
| `largest_component_area` | 3444 px | 1382-9680 | 4407 px | 1236-11677 | Weak: FP components can be as large or larger. |
| `largest_component_fraction` | 0.833 | 0.632-0.984 | 0.773 | 0.551-0.894 | Weak: TP slightly more single-component. |
| `mask_bbox_fraction` | 0.392 | 0.285-0.674 | 0.438 | 0.229-0.703 | Not useful alone. |
| `compactness` | 0.212 | 0.117-0.369 | 0.198 | 0.118-0.313 | Very weak. |
| `num_components` | 5 | 3-8 | 5 | 3-8 | Not useful. |
| `mean_foreground_prob` | 0.800 | 0.748-0.857 | 0.776 | 0.711-0.825 | Weak/moderate. |
| `max_foreground_prob` | 0.995 | 0.984-0.999 | 0.992 | 0.954-0.998 | Weak. |

The segmentation model usually predicts a non-empty foreground mask for both TP and FP. That means simple "has mask / no mask" logic is not useful.

## Rule Simulation

Best simple mask-based rules:

| Rule | Proposals kept | FP removed | FP reduction | Covered GT kept | Covered GT loss | FP/image |
|---|---:|---:|---:|---:|---:|---:|
| `foreground_fraction >= 0.05` | 174 | 49 | 32.9% | 64 | 5 / 69 = 7.2% | 1.47 |
| `largest_component_area >= 256 AND compactness >= 0.05` | 214 | 15 | 10.1% | 69 | 0% | 1.97 |
| `largest_component_area >= 256` | 219 | 10 | 6.7% | 69 | 0% | 2.04 |
| `compactness >= 0.05` | 224 | 5 | 3.4% | 69 | 0% | 2.12 |
| `foreground_fraction >= 0.005` | 225 | 4 | 2.7% | 69 | 0% | 2.13 |

More aggressive rules can remove around half of FP, but they lose too much coverage:

| Rule | FP reduction | Covered GT loss | Interpretation |
|---|---:|---:|---|
| `compactness >= 0.2` | 51.7% | 30 / 69 = 43.5% | Too destructive. |
| `not_touch_edge AND foreground_fraction >= 0.005` | 61.1% | 38 / 69 = 55.1% | Rejects many true candidates near crop edges. |
| `num_components <= 3 AND foreground_fraction >= 0.005` | 74.5% | 45 / 69 = 65.2% | Not viable. |

## Target Check

Target:

```text
FP reduction >= 50%
covered GT loss <= 10%
```

Result:

```text
Not achieved.
```

The best usable mask-only rule is:

```text
foreground_fraction >= 0.05
```

It gives:

```text
FP reduction: 32.9%
coverage loss: 7.2%
FP/image: 2.19 -> 1.47
```

This is useful, but not a 2x FP reduction.

## Interpretation

The existing segmentation checkpoint does produce masks on YOLO crops, but the masks are not selective enough to cleanly separate true archaeological candidates from false positives.

Likely reasons:

1. Domain mismatch: DeepLab was trained on segmentation `.npy` patches, not YOLO image crops.
2. Many YOLO FP are archaeologically plausible or terrain-structure-like, so the segmentation model also sees foreground-like relief.
3. The mask features tested here are generic and simple; they do not encode whether the shape is meaningful archaeology.

## Conclusion

Segmentation refinement with the existing checkpoint is **not sufficient as a hard mask-only filter**.

It can still be useful as a soft feature source:

```text
yolo_conf
+ bbox geometry
+ region/edge flags
+ segmentation foreground_fraction
+ mask component features
```

But the current mask-only rules do not solve the FP problem.

The next practical step is a small crop-level refinement dataset:

```text
positive:
  TP proposals
  missed GT crops

negative:
  manually confirmed trash / terrain_like FP

ignore:
  plausible_object
  uncertain
```

Then train a high-recall crop classifier or hybrid classifier using both image crop and mask-derived features.

## Outputs

Generated artifacts:

```text
04_detection_yolo/reports/segmentation_refinement/v3i_conf005_deeplab_5class_43/
  refined_candidates.csv
  segmentation_refinement_summary.csv
  mask_feature_summary_tp_fp.csv
  mask_feature_summary_by_group.csv
  rule_simulation_segmentation.csv
  masks/
  overlays/
  contact_sheets/
```
