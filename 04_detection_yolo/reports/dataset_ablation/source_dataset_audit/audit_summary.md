# Source YOLO Dataset Audit

## Dataset Level

- Images: `1693`
- Positive images: `956`
- Negative images: `737`
- BBox count: `6681`

## Split

| item | images |
|---|---|
| train | 1436 |
| val | 257 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgany_povrezhdennye | 3792 |
| fortifikatsii | 1172 |
| kurgany_tselye | 1097 |
| arkhitektury | 360 |
| gorodishcha | 260 |

### Image balance by class

| item | images |
|---|---|
| fortifikatsii | 379 |
| kurgany_povrezhdennye | 331 |
| gorodishcha | 221 |
| kurgany_tselye | 184 |
| arkhitektury | 66 |

## Modalities

| item | images |
|---|---|
| Ae | 830 |
| Li | 625 |
| Or | 123 |
| SpOr | 115 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 6681.0 | 6681.0 | 6681.0 | 6681.0 |
| mean | 168971.7807214489 | 219.7702439754528 | 216.03786858254753 | 1.1735358229582498 |
| std | 1093277.8485415792 | 387.3165600813485 | 368.16119293716645 | 1.2810755608350832 |
| min | 80.0 | 4.0 | 3.0 | 0.06862745098039216 |
| 50% | 20960.0 | 141.0 | 151.0 | 0.9847715736040609 |
| 90% | 144573.0 | 388.0 | 382.0 | 1.5751072961373391 |
| 95% | 435600.0 | 730.0 | 662.0 | 2.25 |
| 99% | 3053109.59999999 | 2039.5999999999995 | 1902.1999999999998 | 5.48 |
| max | 16769025.0 | 4095.0 | 4095.0 | 41.5 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 1693.0 |
| mean | 3.9462492616656824 |
| std | 17.482076412011835 |
| min | 0.0 |
| 50% | 1.0 |
| 90% | 5.0 |
| 95% | 14.0 |
| 99% | 78.47999999999956 |
| max | 340.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 956.0 |
| mean | 6.9884937238493725 |
| std | 22.807844554585568 |
| min | 1.0 |
| 50% | 2.0 |
| 90% | 11.0 |
| 95% | 24.0 |
| 99% | 115.70000000000027 |
| max | 340.0 |

## Quality Metrics

- Edge bbox ratio: `0.2327`
- valid_fraction mean: `0.9642`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 1693.0 |
| mean | 0.9641919300562304 |
| std | 0.1081881824142583 |
| min | 0.3590383529663086 |
| 10% | 0.8968067169189453 |
| 50% | 1.0 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 1227 |
| True | 466 |

### has_edge_object

| item | images |
|---|---|
| False | 1037 |
| True | 656 |

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
