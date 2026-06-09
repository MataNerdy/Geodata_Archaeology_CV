# v3d_li_ae_medium Audit

## Dataset Level

- Images: `622`
- Positive images: `311`
- Negative images: `311`
- BBox count: `1207`

## Split

| item | images |
|---|---|
| train | 535 |
| val | 87 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgan | 1207 |

### Image balance by class

| item | images |
|---|---|
| kurgan | 311 |

## Modalities

| item | images |
|---|---|
| Ae | 369 |
| Li | 253 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 1207.0 | 1207.0 | 1207.0 | 1207.0 |
| mean | 85860.25517812758 | 176.82601491300747 | 171.4125932062966 | 1.2101801009432855 |
| std | 267942.55693830096 | 255.5628936087191 | 253.90384915829665 | 1.2314637477580284 |
| min | 100.0 | 6.0 | 3.0 | 0.0783132530120482 |
| 50% | 7917.0 | 88.0 | 89.0 | 1.037037037037037 |
| 90% | 207092.00000000032 | 481.2000000000003 | 445.0000000000009 | 1.5114714553738944 |
| 95% | 502580.1000000004 | 711.5000000000007 | 717.6000000000004 | 2.1454675467546784 |
| 99% | 1403898.7000000002 | 1313.2600000000016 | 1438.1000000000008 | 5.4 |
| max | 2870814.0 | 1758.0 | 1716.0 | 20.333333333333332 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 622.0 |
| mean | 1.9405144694533762 |
| std | 4.819578802293134 |
| min | 0.0 |
| 50% | 0.5 |
| 90% | 4.0 |
| 95% | 7.949999999999932 |
| 99% | 24.789999999999964 |
| max | 47.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 311.0 |
| mean | 3.8810289389067525 |
| std | 6.243086000435783 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 8.0 |
| 95% | 16.0 |
| 99% | 35.69999999999993 |
| max | 47.0 |

## Quality Metrics

- Edge bbox ratio: `0.2618`
- valid_fraction mean: `0.9938`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 622.0 |
| mean | 0.9937532264903247 |
| std | 0.021979087710028267 |
| min | 0.8502604166666666 |
| 10% | 0.9961947202682494 |
| 50% | 1.0 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 509 |
| True | 113 |

### has_edge_object

| item | images |
|---|---|
| False | 331 |
| True | 291 |

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
