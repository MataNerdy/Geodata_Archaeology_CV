# v3e Train Li+Ae / Val Li Audit

## Dataset Level

- Images: `598`
- Positive images: `302`
- Negative images: `296`
- BBox count: `1167`

## Split

| item | images |
|---|---|
| train | 558 |
| val | 40 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgan | 1167 |

### Image balance by class

| item | images |
|---|---|
| kurgan | 302 |

## Modalities

| item | images |
|---|---|
| Ae | 328 |
| Li | 270 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 1167.0 | 1167.0 | 1167.0 | 1167.0 |
| mean | 88496.76520994001 | 179.83718937446443 | 174.471293916024 | 1.2110747559927404 |
| std | 272110.20554643776 | 259.2669754337875 | 257.5570858608647 | 1.2472136724505232 |
| min | 100.0 | 6.0 | 3.0 | 0.0783132530120482 |
| 50% | 7920.0 | 88.0 | 89.0 | 1.0357142857142858 |
| 90% | 216399.0000000006 | 484.8000000000002 | 467.00000000000045 | 1.5114714553738944 |
| 95% | 508936.30000000005 | 719.1000000000001 | 732.7 | 2.176529280835061 |
| 99% | 1413605.4199999976 | 1315.6799999999998 | 1443.7599999999989 | 5.631755102040761 |
| max | 2870814.0 | 1758.0 | 1716.0 | 20.333333333333332 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 598.0 |
| mean | 1.951505016722408 |
| std | 4.889326844680374 |
| min | 0.0 |
| 50% | 1.0 |
| 90% | 4.0 |
| 95% | 8.149999999999977 |
| 99% | 25.059999999999945 |
| max | 47.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 302.0 |
| mean | 3.8642384105960264 |
| std | 6.324406600881268 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 7.900000000000034 |
| 95% | 16.0 |
| 99% | 35.97000000000003 |
| max | 47.0 |

## Quality Metrics

- Edge bbox ratio: `0.2656`
- valid_fraction mean: `0.9918`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 598.0 |
| mean | 0.991821022497034 |
| std | 0.026264614568316507 |
| min | 0.8502604166666666 |
| 10% | 0.9768601655960082 |
| 50% | 1.0 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 487 |
| True | 111 |

### has_edge_object

| item | images |
|---|---|
| False | 314 |
| True | 284 |

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
