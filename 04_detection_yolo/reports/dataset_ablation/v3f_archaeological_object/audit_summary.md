# v3f Archaeological Object Audit

## Dataset Level

- Images: `481`
- Positive images: `264`
- Negative images: `217`
- BBox count: `1045`

## Split

| item | images |
|---|---|
| train | 441 |
| val | 40 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| archaeological_object | 1045 |

### Image balance by class

| item | images |
|---|---|
| archaeological_object | 264 |

## Modalities

| item | images |
|---|---|
| Li | 481 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 1045.0 | 1045.0 | 1045.0 | 1045.0 |
| mean | 190803.0105263158 | 292.5626794258373 | 264.99808612440194 | 1.365881446195392 |
| std | 555832.5819367123 | 406.0312059149171 | 369.15520725860284 | 1.6169679334495008 |
| min | 120.0 | 4.0 | 3.0 | 0.07692307692307693 |
| 50% | 18445.0 | 139.0 | 134.0 | 1.0236686390532543 |
| 90% | 525909.0000000001 | 794.2 | 704.2 | 2.1052631578947367 |
| 95% | 1068360.799999997 | 1229.0 | 991.1999999999996 | 3.611111111111111 |
| 99% | 2826322.0799999945 | 2047.0 | 2047.0 | 9.78995169082112 |
| max | 7124160.0 | 2729.0 | 2624.0 | 20.333333333333332 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 481.0 |
| mean | 2.1725571725571724 |
| std | 4.800754661742809 |
| min | 0.0 |
| 50% | 1.0 |
| 90% | 5.0 |
| 95% | 9.0 |
| 99% | 23.399999999999977 |
| max | 48.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 264.0 |
| mean | 3.9583333333333335 |
| std | 5.913361060915813 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 8.400000000000034 |
| 95% | 14.849999999999994 |
| 99% | 29.110000000000014 |
| max | 48.0 |

## Quality Metrics

- Edge bbox ratio: `0.3837`
- valid_fraction mean: `0.9851`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 481.0 |
| mean | 0.9850806141383974 |
| std | 0.033496078141627035 |
| min | 0.8504428863525391 |
| 10% | 0.9352900981903076 |
| 50% | 0.9999780654907228 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 410 |
| True | 71 |

### has_edge_object

| item | images |
|---|---|
| False | 278 |
| True | 203 |

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

## Source Class Balance

### BBox by source class

| source_class | bbox |
|---|---|
| fortifikatsii | 401 |
| kurgany_povrezhdennye | 346 |
| kurgany_tselye | 233 |
| gorodishcha | 65 |

### BBox by split and source class

| split | source_class_name | bbox |
|---|---|---|
| train | fortifikatsii | 383 |
| train | gorodishcha | 62 |
| train | kurgany_povrezhdennye | 299 |
| train | kurgany_tselye | 231 |
| val | fortifikatsii | 18 |
| val | gorodishcha | 3 |
| val | kurgany_povrezhdennye | 47 |
| val | kurgany_tselye | 2 |

### Train / val positive-negative balance

| split | is_positive | images |
|---|---|---|
| train | False | 206 |
| train | True | 235 |
| val | False | 11 |
| val | True | 29 |
