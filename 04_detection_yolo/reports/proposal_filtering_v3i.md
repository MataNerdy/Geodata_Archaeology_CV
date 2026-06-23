# v3i Proposal Post-Filtering Audit

## Scope

This audit uses the current `v3i` YOLO proposal output:

```text
dataset = dataset_yolo_bbox_v3i_li_archaeological_object_merged
model   = YOLOv8n
imgsz   = 640
conf    = 0.05
split   = val
```

No new YOLO training was run.

## Baseline

| Metric | Value |
|---|---:|
| Images | 68 |
| GT objects | 108 |
| Proposals | 229 |
| Proposals / image | 3.37 |
| Covered GT @ IoU 0.3 | 69 |
| Coverage @ IoU 0.3 | 0.639 |
| Covered GT @ IoU 0.5 | 55 |
| Coverage @ IoU 0.5 | 0.509 |
| FP candidates @ IoU 0.3 | 149 |
| FP / image @ IoU 0.3 | 2.19 |

## Candidate Feature Comparison

TP/FP are defined by proposal IoU with GT:

```text
TP proposal: max_iou_with_gt >= 0.3
FP proposal: max_iou_with_gt < 0.3
```

| Feature | TP median | TP IQR | FP median | FP IQR | Signal |
|---|---:|---:|---:|---:|---|
| `yolo_conf` | 0.157 | 0.083-0.348 | 0.081 | 0.063-0.119 | Strong: FP are much lower confidence. |
| `bbox_area_norm` | 0.019 | 0.005-0.055 | 0.037 | 0.014-0.137 | FP tend to be larger. |
| `bbox_w` | 156 px | 71-274 | 214 px | 129-437 | FP tend to be wider. |
| `bbox_h` | 134 px | 71-221 | 184 px | 100-355 | FP tend to be taller. |
| `aspect_ratio` | 1.19 | 1.09-1.40 | 1.27 | 1.16-1.81 | Weak/moderate. Extreme aspect ratios are usually FP. |
| `pad_area_norm` | 0.041 | 0.011-0.112 | 0.071 | 0.026-0.250 | FP tend to produce very large padded crops. |
| `valid_fraction` | 0.984 | 0.814-1.000 | 1.000 | 0.955-1.000 | Weak; not useful alone. |
| `tile_std` | 20.64 | 16.94-25.28 | 20.91 | 17.80-28.73 | Weak. |
| `tile_p98_minus_p2` | 105 | 75.75-122 | 97 | 81-131 | Weak. |
| `n_objects` | 4 | 2-6 | 1 | 0-2 | Strong but validation-specific: FP often appear in negative or sparse tiles. |

Boolean features:

| Feature | FP | TP | Interpretation |
|---|---:|---:|---|
| `bbox_touches_image_edge = True` | 67 | 22 | Useful FP signal, but not safe alone. |
| `pad_touches_image_edge = True` | 110 | 38 | Strong FP enrichment, but many true proposals also touch padded edge. |
| `tile_touches_raster_edge = True` | 39 | 32 | Weak alone. |
| `has_edge_object = True` | 50 | 51 | Not useful as reject rule; many real objects are in these tiles. |
| `is_positive = False` | 51 | 0 | Strong if trusted, but cannot be used when mining unlabeled data. |

## Region Effects

| Region | FP @ IoU0.3 | TP @ IoU0.3 | Note |
|---|---:|---:|---|
| `025_ШУМГОРА` | 34 | 11 | Many terrain/edge-like candidates. |
| `005_ЛУБНО` | 28 | 24 | Mixed: both useful candidates and many partial/large structures. |
| `004_ДЕМИДОВКА` | 24 | 2 | Strong FP-heavy region. |
| `011_РУНА` | 20 | 3 | FP-heavy, but some candidates are archaeologically plausible. |
| `012_ЛИХУША` | 13 | 8 | Mixed. |
| `037_КЧР` | 13 | 6 | Slopes and edge-like terrain. |
| `006_МОСКОВИТЫ` | 11 | 11 | Many fortification-like candidates; should be reviewed, not blindly rejected. |
| `013_БЕРВЕНЕЦ` | 6 | 15 | Strong region for the current model. |

## Rule Simulation

Baseline:

```text
proposals = 229
FP@IoU0.3 = 149
FP/image = 2.19
covered_gt_iou03 = 69
coverage_iou03 = 0.639
```

Promising filters:

| Rule | Proposals after | FP after | FP removed | Covered GT | Coverage | Lost covered GT | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| `region in {025_ШУМГОРА, 004_ДЕМИДОВКА, 011_РУНА} AND conf < 0.1` | 165 | 93 | 56 | 64 | 0.593 | 5 | 1.37 |
| `bbox_area_norm > 0.1 AND conf < 0.15` | 175 | 103 | 46 | 63 | 0.583 | 6 | 1.51 |
| `region in {025_ШУМГОРА, 004_ДЕМИДОВКА, 037_КЧР} AND bbox_area_norm > 0.1` | 193 | 116 | 33 | 66 | 0.611 | 3 | 1.71 |
| `aspect_ratio > 2` | 196 | 122 | 27 | 65 | 0.602 | 4 | 1.79 |
| `tile_touches_raster_edge AND conf < 0.1` | 199 | 125 | 24 | 64 | 0.593 | 5 | 1.84 |
| `bbox_area_norm > 0.2` | 201 | 126 | 23 | 64 | 0.593 | 5 | 1.85 |
| `aspect_ratio > 3` | 221 | 141 | 8 | 69 | 0.639 | 0 | 2.07 |

Best current candidate:

```text
IF region in {025_ШУМГОРА, 004_ДЕМИДОВКА, 011_РУНА}
AND yolo_conf < 0.1
THEN reject_or_review
```

Effect:

```text
FP removed: 56 / 149 = 37.6%
Coverage drop: 69 -> 64 covered GT
Covered GT loss: 7.2% of baseline covered GT
FP/image: 2.19 -> 1.37
```

This meets the requested target range, but it is region-aware and may overfit the current validation set. It should be treated first as `flag_for_review`, not as a final production reject rule.

## Recommended Rule-Based Filter Design

Use two levels:

### Reject Candidates

Conservative automatic rejection:

```text
IF aspect_ratio > 3
THEN reject
```

This removes only 8 FP but loses no covered GT in the current validation set. It is safe but weak.

Moderate rejection:

```text
IF bbox_area_norm > 0.2
AND yolo_conf < 0.15
THEN reject
```

This removes 19 FP and loses 5 covered GT. It is useful but should be manually inspected before production use.

### Flag For Review

High-value review filter:

```text
IF region in {025_ШУМГОРА, 004_ДЕМИДОВКА, 011_РУНА}
AND yolo_conf < 0.1
THEN flag_for_review
```

This captures the strongest FP reduction pattern. Because it is region-specific, it should not be blindly generalized.

Large terrain candidate flag:

```text
IF bbox_area_norm > 0.1
AND yolo_conf < 0.15
THEN flag_for_review
```

This catches many large slope/terrain candidates but also loses useful coverage if used as hard reject.

## Additional Features To Compute From Crops

The current metadata/proposal table is not enough to reliably distinguish:

```text
archaeologically plausible FP
vs
unlabeled missed object
vs
terrain artifact
```

Recommended automatically computable crop features:

| Feature | Why useful |
|---|---|
| Local contrast inside bbox and padded crop | Slopes/edges often have strong directional contrast. |
| Entropy / texture complexity | Noisy terrain differs from compact mound-like objects. |
| Edge density and edge orientation histogram | Linear fortifications and slopes have different edge direction patterns. |
| Candidate center vs tile/crop edge distance | Edge artifacts and truncated objects are overrepresented near edges. |
| Shape compactness after simple threshold/segmentation | Kurgans should be more compact than slopes/ridges. |
| Connected component count | Terrain noise often creates many components. |
| Radial symmetry / circularity score | Useful for kurgan-like candidates. |
| Elongation / line-likeness score | Useful to separate fortifications from slopes and mounds. |
| Hillshade multi-direction agreement | Real relief objects should persist across illumination direction. |

## Conclusion

Rule-based filtering can give a noticeable short-term improvement before segmentation:

- A conservative rule can remove a small number of FP with almost no coverage loss.
- A region-aware review rule can remove around 38% of FP with around 7% covered-GT loss.
- Pure metadata rules are not enough to solve the problem robustly because many false positives are archaeologically plausible.

The best next step is a two-stage proposal pipeline:

```text
YOLO proposals
  -> conservative rule-based reject / review flags
  -> crop-level feature extraction
  -> segmentation/refinement
  -> manual label completion for plausible unlabeled candidates
```

