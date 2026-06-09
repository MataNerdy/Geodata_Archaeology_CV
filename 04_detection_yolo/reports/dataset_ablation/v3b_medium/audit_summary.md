# v3b_medium Audit

## Dataset Level

- Images: `284`
- Positive images: `142`
- Negative images: `142`
- BBox count: `579`

## Split

| item | images |
|---|---|
| train | 255 |
| val | 29 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgan | 579 |

### Image balance by class

| item | images |
|---|---|
| kurgan | 142 |

## Modalities

| item | images |
|---|---|
| Li | 284 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 579.0 | 579.0 | 579.0 | 579.0 |
| mean | 119490.597582038 | 220.01381692573403 | 217.49222797927462 | 1.2607451577429567 |
| std | 323272.54576605774 | 293.12646481721214 | 289.23474594400045 | 1.6200514865292002 |
| min | 120.0 | 6.0 | 3.0 | 0.0783132530120482 |
| 50% | 12880.0 | 118.0 | 114.0 | 1.0 |
| 90% | 364478.4000000008 | 662.0 | 668.2000000000005 | 1.5183251231527102 |
| 95% | 649986.9000000005 | 874.0 | 825.2000000000003 | 2.297447775154783 |
| 99% | 1639127.0000000114 | 1331.480000000001 | 1477.640000000003 | 8.722222222222298 |
| max | 2870814.0 | 1758.0 | 1716.0 | 20.333333333333332 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 284.0 |
| mean | 2.038732394366197 |
| std | 5.153671696606481 |
| min | 0.0 |
| 50% | 0.5 |
| 90% | 4.0 |
| 95% | 9.849999999999966 |
| 99% | 23.340000000000032 |
| max | 47.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 142.0 |
| mean | 4.077464788732394 |
| std | 6.703522884203131 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 9.900000000000006 |
| 95% | 17.849999999999966 |
| 99% | 36.210000000000065 |
| max | 47.0 |

## Quality Metrics

- Edge bbox ratio: `0.3040`
- valid_fraction mean: `0.9873`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 284.0 |
| mean | 0.9872829741640065 |
| std | 0.02979849414112955 |
| min | 0.857643723487854 |
| 10% | 0.9461056470870972 |
| 50% | 0.9999775886535645 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 238 |
| True | 46 |

### has_edge_object

| item | images |
|---|---|
| True | 149 |
| False | 135 |

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
