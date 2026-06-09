# v3a_minimal Audit

## Dataset Level

- Images: `292`
- Positive images: `146`
- Negative images: `146`
- BBox count: `591`

## Split

| item | images |
|---|---|
| train | 258 |
| val | 34 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgan | 591 |

### Image balance by class

| item | images |
|---|---|
| kurgan | 146 |

## Modalities

| item | images |
|---|---|
| Li | 292 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 591.0 | 591.0 | 591.0 | 591.0 |
| mean | 124294.47884940778 | 223.83925549915398 | 221.13536379018612 | 1.260366469327785 |
| std | 331469.777099444 | 298.11970559353097 | 295.65918070831685 | 1.6162140574844133 |
| min | 84.0 | 6.0 | 3.0 | 0.0783132530120482 |
| 50% | 12880.0 | 118.0 | 114.0 | 1.0 |
| 90% | 409149.0 | 667.0 | 683.0 | 1.5142857142857142 |
| 95% | 682689.0 | 889.0 | 854.0 | 2.323756922483037 |
| 99% | 1848544.700000003 | 1327.4000000000008 | 1492.1000000000017 | 8.388888888888951 |
| max | 2870814.0 | 1758.0 | 1716.0 | 20.333333333333332 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 292.0 |
| mean | 2.0239726027397262 |
| std | 5.140912040101232 |
| min | 0.0 |
| 50% | 0.5 |
| 90% | 4.0 |
| 95% | 9.449999999999989 |
| 99% | 24.089999999999975 |
| max | 47.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 146.0 |
| mean | 4.0479452054794525 |
| std | 6.692592087685404 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 9.5 |
| 95% | 17.25 |
| 99% | 36.00000000000023 |
| max | 47.0 |

## Quality Metrics

- Edge bbox ratio: `0.3029`
- valid_fraction mean: `0.9784`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 292.0 |
| mean | 0.9784273631423376 |
| std | 0.044451694693177356 |
| min | 0.8067360520362854 |
| 10% | 0.9197611510753633 |
| 50% | 0.999970942735672 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 232 |
| True | 60 |

### has_edge_object

| item | images |
|---|---|
| False | 148 |
| True | 144 |

## Leakage

| item | count |
|---|---|
| region_overlap | 0 |
| source_id_overlap | 0 |
| raster_file_overlap | 0 |

## Figures

- `class_balance.png`
- `positive_negative_ratio.png`
- `bbox_area_distribution.png`
- `objects_per_image.png`
- `bbox_width_height.png`
- `train_labels_page_01..05.jpg`
- `val_labels_page_01..05.jpg`
