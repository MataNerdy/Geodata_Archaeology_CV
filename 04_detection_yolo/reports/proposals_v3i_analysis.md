# v3i YOLO Proposal Analysis

## Context

The `v3i` experiment uses a Li-only merged one-class target:

```text
archaeological_object =
  kurgany_tselye
  kurgany_povrezhdennye
  gorodishcha
  fortifikatsii
  arkhitektury
```

The trained model is:

```text
YOLOv8n
imgsz = 640
dataset = dataset_yolo_bbox_v3i_li_archaeological_object_merged
```

The experiment showed that this model is not reliable enough as a final detector. Standard validation metrics remain moderate and the merged target class is visually heterogeneous: round kurgans, damaged mounds, settlement-like shapes, and linear fortifications do not share one simple morphology.

However, low-confidence inference shows that the model is useful as a proposal generator. It does not stay silent: it produces many archaeologically plausible candidates that can be passed to a downstream segmentation/refinement stage.

## Proposal Mode Comparison

| Mode | conf | Recall@IoU0.5 | Coverage@IoU0.3 | TP | FP | FN | Proposals | FP/image | Candidates per found object |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled proposal review | 0.05 | 0.509 | 0.639 | 55 | 174 | 53 | 229 | 2.56 | 3.32 |
| Aggressive mining | 0.01 | 0.657 | 0.778 | 71 | 918 | 37 | 989 | 13.50 | 11.77 |

Notes:

- `Coverage@IoU0.3` counts GT objects that have at least one prediction with IoU >= 0.30.
- Recall is computed at the stricter working match threshold IoU >= 0.50.
- `Candidates per found object` is `total proposals / covered_gt_iou_0.30`.

## Why `conf = 0.05`

`conf = 0.05` is the best current default for proposal generation because it is the first threshold where the model becomes substantially useful while the false-positive stream is still reviewable:

```text
coverage@IoU0.3 = 0.639
recall@IoU0.5   = 0.509
FP/image        = 2.56
```

This is suitable for:

- manual validation of candidates;
- building a first crop dataset;
- feeding a segmentation/refinement model with a manageable number of false positives.

## Why Keep `conf = 0.01`

`conf = 0.01` is not a good final inference threshold, but it is useful for aggressive mining:

```text
coverage@IoU0.3 = 0.778
recall@IoU0.5   = 0.657
FP/image        = 13.50
```

This mode is useful when the goal is to discover missed or weakly annotated objects. It should be used only with a strong downstream filter or for manual review of hard regions.

## Visual Review Findings

The low-confidence validation review created full-image overlays and contact sheets for:

```text
runs/v3i_yolov8n640_low_conf_val_review_outputs/
```

The review confirms that bbox coordinates are visually aligned. The main issue is not coordinate drift. The main issue is semantic ambiguity: the detector responds to many geomorphologically similar shapes.

Common false-positive sources:

- large terrain edges and slopes;
- shadow/highlight transitions;
- linear relief structures;
- settlement-like or fortification-like patterns not covered by current GT;
- dense relief textures in FP-heavy regions.

Most FP-heavy validation regions:

```text
005_ЛУБНО
006_МОСКОВИТЫ
004_ДЕМИДОВКА
011_РУНА
013_БЕРВЕНЕЦ
```

## Next Pipeline Step

The next stage should not be another YOLO training run. The better direction is:

```text
LiDAR tile
  -> YOLO low-confidence proposals
  -> padded bbox crops
  -> candidate table
  -> segmentation/refinement model
  -> filtered archaeological-object candidates
```

The proposed implementation entrypoint is:

```bash
python 04_detection_yolo/scripts/generate_yolo_proposals.py \
  --config 04_detection_yolo/configs/generate_proposals_v3i_conf005.yaml
```

For aggressive mining:

```bash
python 04_detection_yolo/scripts/generate_yolo_proposals.py \
  --config 04_detection_yolo/configs/generate_proposals_v3i_conf001.yaml
```

Use dry-run mode to validate paths without running inference:

```bash
python 04_detection_yolo/scripts/generate_yolo_proposals.py \
  --config 04_detection_yolo/configs/generate_proposals_v3i_conf005.yaml \
  --dry-run
```

