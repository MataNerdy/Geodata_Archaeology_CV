# Dataset Ablation Findings

## Recommended First Baseline

`v3b_medium` is the safer first controlled baseline if the goal is to isolate Li-only behavior: it keeps `142` positive images and `579` boxes with moderate raster-quality and small-area filtering. `v3d_li_ae_medium` is the best scale-up candidate because it keeps `311` positive images and `1207` boxes, but it changes the modality mix and should be compared against v3b rather than treated as a clean replacement.

## Strongest Positive-Image Filter Overall

| dataset_version | step_name | images_before | images_after | positive_before | positive_after | bbox_before | bbox_after | removed_images | removed_positive_images | removed_bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| v3a_minimal | modality_filter | 1693 | 625 | 416 | 186 | 4889 | 1886 | 1068 | 230 | 3003 |

## Strongest BBox Filter Overall

| dataset_version | step_name | images_before | images_after | positive_before | positive_after | bbox_before | bbox_after | removed_images | removed_positive_images | removed_bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| v3a_minimal | modality_filter | 1693 | 625 | 416 | 186 | 4889 | 1886 | 1068 | 230 | 3003 |

## Strongest Positive-Image Filter After Modality Choice

| dataset_version | step_name | images_before | images_after | positive_before | positive_after | bbox_before | bbox_after | removed_images | removed_positive_images | removed_bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| v3c_strict | valid_fraction_gte_0.9 | 625 | 454 | 186 | 135 | 1886 | 567 | 171 | 51 | 1319 |

## Strongest BBox Filter After Modality Choice

| dataset_version | step_name | images_before | images_after | positive_before | positive_after | bbox_before | bbox_after | removed_images | removed_positive_images | removed_bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| v3d_li_ae_medium | n_objects_lte_50 | 1311 | 1298 | 324 | 311 | 2952 | 1207 | 13 | 13 | 1745 |

## Over-Cleaning Signal

Versions with fewer than 100 positive images are likely over-cleaned for a stable YOLO baseline: `v3c_strict`.

## Remaining Label-Risk Areas

- Very small boxes may represent ambiguous or barely visible damaged kurgans.
- Edge-touching boxes can be valid partial objects, but they also encode tiling artifacts.
- High-object-count tiles may contain dense archaeological zones; dropping them may remove useful hard cases.
- Ae imagery may add useful context but also modality shift and weaker visual signal.
- Strict filtering leaves only 73 positive images, so validation becomes fragile and recall estimates are noisy.

## Manual Review Priority

- Positive tiles near the lower bbox-area tail.
- Tiles removed by the edge-bbox filter.
- Images with high `n_objects` before cutoff.
- Validation positives, because validation is small in strict settings.
- False-negative candidates: source kurgan images where bbox filters removed all selected boxes.

## Next Experiments

1. Train YOLOv8n on `v3b_medium` at `imgsz=640` as the controlled Li-only baseline.
2. Train the same config on `v3d_li_ae_medium` to test whether Ae helps or adds noise.
3. Use `v3c_strict` only as a precision-oriented stress test, not as the first baseline.
4. Inspect recall failures by bbox area bucket, especially damaged kurgans and very small boxes.
5. Tune confidence only after dataset choice is fixed.
