# Archaeological Object Detection from LiDAR

This module investigates object detection for archaeological features on LiDAR-derived raster tiles.

The experiments did not produce a reliable high-mAP detector. They identified a more useful operating mode for YOLO:

```text
YOLO is not reliable enough as a final detector,
but low-confidence inference is useful for proposal generation and manual review.
```

Working pipeline:

```text
LiDAR tiles -> YOLO detection -> proposal generation -> manual proposal audit
```

![Case study: validation image 000444](assets/readme/figure_case_study_000444.png)

Validation image `000444` is used as a compact case study:

```text
raw LiDAR -> ground truth -> standard detector -> low-confidence proposals -> manual review
```

At the standard detection threshold, YOLO recovers only part of the annotated scene. Lowering the confidence threshold produces additional archaeologically plausible candidates. Manual review indicates that some formal false positives are not obvious noise. In this setting, the model is better treated as a proposal generator for expert review than as an autonomous final detector.

Figure sources:

| Panel | Source |
|---|---|
| A. Raw LiDAR | `datasets/dataset_yolo_bbox_v3i_li_archaeological_object_merged/images/val/000444.png` |
| B. Ground truth | `datasets/dataset_yolo_bbox_v3i_li_archaeological_object_merged/labels/val/000444.txt` |
| C. Standard detector | `runs/yolo_v3i_archaeological_object_20260618_221705/analysis/v3i_archaeological_object_yolov8n_640/predictions_all_conf.csv` |
| D. Low-confidence proposals | `reports/proposals/v3i_conf005/predictions.csv` |
| E. Manual review | `reports/refinement_manual_review.csv` and `reports/proposals/v3i_conf005/crops/000444_p*.jpg` |

## Problem

The task is to localize archaeological objects in LiDAR imagery using bounding boxes.

Target morphologies:

- `kurgany_tselye`: well-preserved burial mounds;
- `kurgany_povrezhdennye`: damaged or weakly expressed burial mounds;
- `gorodishcha`: settlement-like area features;
- `fortifikatsii`: linear or area-based fortification structures.

This setup is harder than the previous segmentation task. The detector has to search large tiles with substantial background, while target objects may be small, low-contrast, partially destroyed, or clipped by tile boundaries. Some unannotated terrain structures are visually close to archaeological features.

## Dataset

The source data is a YOLO bounding-box dataset built from geospatial rasters and polygon annotations. The final proposal experiment uses the LiDAR-only merged dataset:

```text
dataset_yolo_bbox_v3i_li_archaeological_object_merged
```

All target classes are merged into one YOLO class:

```text
0: archaeological_object
```

The split is region-aware: validation regions are separated from training regions to avoid leakage by `region`, `source_id`, and `raster_file`.

### v3i Dataset Split

| Split | Images | Positive images | Negative images | BBox |
|---|---:|---:|---:|---:|
| train | 408 | 237 | 171 | 1069 |
| val | 68 | 48 | 20 | 108 |
| total | 476 | 285 | 191 | 1177 |

Validation regions:

```text
004_ДЕМИДОВКА
005_ЛУБНО
006_МОСКОВИТЫ
011_РУНА
012_ЛИХУША
013_БЕРВЕНЕЦ
025_ШУМГОРА
037_КЧР
```

Validation boxes by source class:

| Source class | Val bbox |
|---|---:|
| `fortifikatsii` | 47 |
| `kurgany_povrezhdennye` | 28 |
| `kurgany_tselye` | 20 |
| `gorodishcha` | 13 |

Leakage check:

| Key | Train/val overlap |
|---|---:|
| region | 0 |
| source_id | 0 |
| raster_file | 0 |

Validation image `000444` is used in the title figure. The LiDAR tile contains several clearly expressed features and many terrain structures with similar morphology, while the ground-truth boxes cover only the annotated subset of the scene.

## Research Questions

The experiments focused on practical questions:

- can YOLO work as an archaeological object detector on LiDAR tiles;
- is a cleaner Li-only dataset better than a larger Li + Ae dataset;
- does expanding the target from `kurgan` to `archaeological_object` improve recall;
- does longer training help;
- does switching YOLO architecture help;
- does increasing image size to `1024` help;
- can low-confidence YOLO inference be used for proposal generation;
- can simple rules reduce the false-positive stream;
- what does manual review reveal about formal false positives.

## Experiments Overview

Detailed experiment logs are kept in `reports/` and `runs/`.

| Experiment | Goal | Result | Conclusion |
|---|---|---|---|
| Early kurgan-only runs `v2-v4` | Train a detector for intact and damaged burial mounds | Best balanced v4 run: `mAP50 = 0.21359`, `Recall = 0.20424` | YOLO learned part of the signal, but recall remained low. |
| Dataset ablation `v3b` vs `v3d` | Compare a cleaner Li-only dataset against a larger Li + Ae dataset | `v3b_li_medium`: `mAP50 = 0.33904`; `v3d_li_ae_medium`: `mAP50 = 0.16164` | The cleaner Li-only dataset performed better than mixed-modality training. |
| Longer training | Test whether 400 epochs recover low recall | `v3b_400_epoch_limit`: `mAP50 = 0.27816`, below the 100-epoch baseline | More epochs did not remove the bottleneck. |
| YOLO26 check | Test a newer nano detector variant | `YOLO26n`: `mAP50 = 0.22218` on v3b | Changing the detector variant did not improve the baseline. |
| Manual-clean kurgan dataset `v3g` | Remove broken tiles and poor boxes before training | `mAP50 = 0.20386`, `Recall = 0.20354` | Cleaning made the dataset more trustworthy, but not easier for YOLO. |
| Curated validation `v3h` | Build a manually selected region-aware validation split | `mAP50 = 0.27114`; no-Saratov sanity check: `mAP50 = 0.46433` | Metrics were highly sensitive to validation-region composition. |
| Merged class `v3i` | Test one-class target `archaeological_object` | `mAP50 = 0.35723`, `Recall = 0.32407` | The broader class added data but increased morphological heterogeneity. |
| Model and size ablation on v3i | Compare YOLOv8n/YOLO26n and `640/1024` | Best `mAP50`: YOLOv8n 640; best `mAP50-95`: YOLO26n 640 | `1024` did not improve detection metrics. |
| Proposal mode | Lower confidence to increase candidate coverage | `conf=0.05`: `coverage@IoU0.3 = 0.639`; `conf=0.01`: `0.778` | YOLO is useful as a proposal generator. |
| Proposal filtering | Test simple filters before manual review | Best global rule removed `30.9%` FP with `8.7%` covered-GT loss | Rules help, but do not resolve semantic ambiguity. |
| Manual proposal audit | Review all v3i proposals at `conf=0.05` | Only `43 / 149` formal FP were obvious terrain/noise | Standard detection metrics underestimate practical proposal value. |

## Key Results

### Final Detector Mode

The strongest detector-style result came from the curated no-Saratov kurgan sanity check. It is useful diagnostically, but the validation split is small and region-sensitive, so it should not be treated as a general solution.

| Dataset | Target | Model | imgsz | Val images | Val bbox | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3b_li_medium` | `kurgan` | YOLOv8n | 640 | 29 | 49 | 0.70544 | 0.28571 | 0.33904 | 0.11516 |
| `v3h_no_saratov` | `kurgan` | YOLOv8n | 640 | 31 | 66 | 0.68752 | 0.40909 | 0.46433 | 0.20339 |
| `v3i_archaeological_object` | `archaeological_object` | YOLOv8n | 640 | 68 | 108 | 0.65580 | 0.32407 | 0.35723 | 0.10604 |

The detector is not reliable enough for fully automated archaeological mapping. Recall remains low, and validation quality depends strongly on region composition. In Panel C of the title figure, the standard threshold recovers part of image `000444` but does not cover the full annotated scene.

### Model and Image-Size Ablation on v3i

| Experiment | Model | imgsz | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `v3i_yolov8n_img640` | YOLOv8n | 640 | 87 | 0.65580 | 0.32407 | 0.35723 | 0.10604 |
| `v3i_yolo26n_img640` | YOLO26n | 640 | 173 | 0.59923 | 0.34259 | 0.35024 | 0.15516 |
| `v3i_yolo26n_img1024` | YOLO26n | 1024 | 138 | 0.61251 | 0.26350 | 0.27636 | 0.11370 |
| `v3i_yolov8n_img1024` | YOLOv8n | 1024 | 141 | 0.58746 | 0.24074 | 0.27572 | 0.09563 |

`YOLOv8n 640` remained the most useful proposal baseline. `YOLO26n 640` improved stricter localization (`mAP50-95`), but did not improve `mAP50` or proposal coverage. Increasing image size to `1024` produced many visually plausible candidates, but did not improve the standard detection metrics.

### Proposal Mode

The current proposal baseline uses:

```text
dataset = dataset_yolo_bbox_v3i_li_archaeological_object_merged
model   = YOLOv8n
imgsz   = 640
```

Low-confidence inference changes the operating question. Instead of asking whether YOLO is an accurate final detector, the relevant question becomes:

```text
Can YOLO produce a manageable set of archaeologically meaningful candidate objects?
```

| conf | Proposals | Proposals/image | TP | FP | FN | Recall@IoU0.5 | Coverage@IoU0.3 | Coverage@IoU0.5 | FP/image | Interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 229 | 3.37 | 55 | 174 | 53 | 0.509 | 0.639 | 0.509 | 2.56 | Primary proposal mode for manual review. |
| 0.01 | 989 | 14.54 | 71 | 918 | 37 | 0.657 | 0.778 | 0.657 | 13.50 | Aggressive mining mode; too noisy for direct review. |

`conf=0.05` is the best current operating point: it covers many ground-truth objects while keeping the candidate stream small enough for visual audit.

On image `000444`, low-confidence inference changes the interpretation from a small set of strict detections to a reviewable set of candidate objects. The analysis therefore shifts from final-detector performance to the practical value of a proposal workflow.

## Failure Analysis

### Object Size

The v3h false-negative audit used the following matching rule:

```text
confidence >= 0.25 and IoU >= 0.5
```

| Group | Count | Median bbox area | Median width | Median height |
|---|---:|---:|---:|---:|
| FOUND | 24 | 32756 px | 182 px | 174.5 px |
| MISSED | 88 | 14994 px | 133 px | 118 px |

Missed objects were often smaller than found objects, but size alone did not explain the errors. The audit also found large missed objects that a robust detector should have recovered.

False negatives were frequently associated with `small_object` (`50`), `large_object` (`38`), `edge_object` (`32`), `dense_cluster` (`25`), and `isolated_object` (`11`). By error type, `metric_miss` (`43`) was the largest group, followed by `hard_miss` (`28`) and `near_miss` (`17`). In many cases, the model produced a prediction near the object, but not one that satisfied strict detection matching.

### Region Effects

In the full v3h audit, false negatives were concentrated in several regions:

| Region | FN count |
|---|---:|
| `028_САРАТОВ` | 44 |
| `008_СЕЛЯНЕ` | 17 |
| `019_ОСЕЧКИ_1` | 12 |
| `025_ШУМГОРА` | 8 |
| `037_КЧР` | 7 |

Moving `028_САРАТОВ` from validation to training sharply increased kurgan detector metrics:

| Dataset | Train images | Val images | Val bbox | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v3h_li_manual_curated_val` | 212 | 46 | 112 | 0.42761 | 0.24107 | 0.27114 | 0.12092 |
| `v3h_no_saratov` | 227 | 31 | 66 | 0.68752 | 0.40909 | 0.46433 | 0.20339 |

The result does not solve the task. It points to a strong regional domain shift in the validation protocol.

### Heterogeneous Morphology

The `archaeological_object` class combines round burial mounds, damaged mounds, settlement-like areas, and linear fortifications. The merged target increases the amount of training data, but makes the one-class detector morphologically ambiguous.

In the v3i regional audit at `conf=0.25`, several regions had very low recall:

| Region | GT | TP | FN | Recall | Dominant classes |
|---|---:|---:|---:|---:|---|
| `005_ЛУБНО` | 27 | 1 | 26 | 0.037 | `fortifikatsii; gorodishcha` |
| `037_КЧР` | 12 | 1 | 11 | 0.083 | `kurgany_povrezhdennye` |
| `006_МОСКОВИТЫ` | 22 | 3 | 19 | 0.136 | `fortifikatsii` |
| `025_ШУМГОРА` | 18 | 4 | 14 | 0.222 | mixed |
| `013_БЕРВЕНЕЦ` | 12 | 11 | 1 | 0.917 | kurgans |

The same model can work well in one region and fail in another.

## Manual Proposal Audit

All `229` v3i proposals at `conf=0.05` were manually reviewed.

Manual labels:

| Label | Count |
|---|---:|
| `object` | 132 |
| `plausible_object` | 51 |
| `terrain_like` | 30 |
| `trash` | 13 |
| `bad_crop` | 3 |

Formal false positives were defined as:

```text
max_iou_with_gt < 0.3
```

There were `149` formal FP proposals. Manual review split them as follows:

| Formal FP Category | Count |
|---|---:|
| `object` | 52 |
| `plausible_object` | 51 |
| `terrain_like + trash` | 43 |
| `bad_crop` | 3 |

Only `43 / 149` formal false positives were clear terrain-like or trash candidates.

Most formal false positives were manually classified as archaeological objects or archaeologically plausible structures.

For `000444`, manual audit is particularly informative: the low-confidence proposals include matched objects as well as additional objects that do not match the current ground truth but still look archaeologically meaningful.

![Manual review of 000444 proposal crops](assets/readme/figure_000444_manual_review.png)

In this setting, standard detection metrics understate the practical value of proposal generation. Some "false positives" may be incomplete labels, ambiguous archaeological structures, or features outside the current ground-truth definition.

## Proposal Filtering

Simple rule-based filters were tested before manual review. They can reduce the candidate stream, but they do not resolve semantic ambiguity: some formal false positives are archaeologically meaningful.

| Filter | FP reduction | Covered GT loss | Interpretation |
|---|---:|---:|---|
| `bbox_area_norm > 0.1 AND conf < 0.15` | 30.9% | 8.7% | Best global rule. |
| `region in {025_ШУМГОРА, 004_ДЕМИДОВКА, 011_РУНА} AND conf < 0.1` | 37.6% | 7.2% | Best region-aware rule. |

Rule-based filtering is useful for review prioritization, but it cannot replace expert interpretation of candidate objects.

## Final Conclusions

1. YOLO is limited as a final detector for this task.
   Standard-threshold recall remains low, and performance depends strongly on region composition and object morphology.

2. Expanding the target class to `archaeological_object` did not make YOLO a stronger final detector.
   The class became more heterogeneous: burial mounds, damaged mounds, settlements, and fortifications do not share a single simple visual form.

3. Low-confidence YOLO inference is useful for proposal generation.
   At `conf=0.05`, the model produced `229` proposals on `68` validation images, with `coverage@IoU0.3 = 0.639` and `3.37` proposals per image.

4. Manual audit changed the interpretation of false positives.
   Of `149` formal FP proposals, only `43` were clear terrain-like or trash candidates.

5. The most useful workflow at this stage is:

```text
LiDAR -> YOLO proposal generation -> human review
```

The workflow is useful for archaeological review even though the model is not yet a reliable autonomous detector.

## Repository Structure

```text
04_detection_yolo/
├── README.md
├── assets/readme/
├── configs/
├── scripts/
├── app/
├── notebooks/
├── reports/
├── runs/
└── requirements.txt
```

`configs/` contains reproducible parameters, `scripts/` contains dataset-building and proposal-generation code, `app/` contains local audit/viewer tools, `reports/` contains analysis outputs and CSV files, and `assets/readme/` contains README figures. Large datasets, model weights, and full training runs are kept out of Git tracking.

## Future Work

Natural next steps:

- train a small crop-level refinement classifier from manually reviewed proposals;
- complete labels for `plausible_object` candidates;
- turn the proposal workflow into a human-in-the-loop tool for archaeological review.

These steps are outside the scope of the current detection/proposal study.
