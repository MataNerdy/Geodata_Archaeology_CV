# v3c_strict Audit


> WARNING: positive images < 100.

## Dataset Level

- Images: `146`
- Positive images: `73`
- Negative images: `73`
- BBox count: `231`

## Split

| item | images |
|---|---|
| train | 126 |
| val | 20 |

## Classes

### BBox balance

| item | bbox |
|---|---|
| kurgan | 231 |

### Image balance by class

| item | images |
|---|---|
| kurgan | 73 |

## Modalities

| item | images |
|---|---|
| Li | 146 |

## BBox Statistics

| index | area | width | height | aspect_ratio |
|---|---|---|---|---|
| count | 231.0 | 231.0 | 231.0 | 231.0 |
| mean | 40717.20346320346 | 168.69264069264068 | 157.6147186147186 | 1.0625601605141906 |
| std | 70788.31395018953 | 131.10687935867566 | 114.87033729869573 | 0.20986359968992013 |
| min | 288.0 | 17.0 | 16.0 | 0.5806451612903226 |
| 50% | 17856.0 | 138.0 | 124.0 | 1.0333333333333334 |
| 90% | 87612.0 | 302.0 | 294.0 | 1.2730923694779117 |
| 95% | 211458.5 | 486.0 | 438.0 | 1.4947368421052631 |
| 99% | 361132.0 | 659.0 | 567.0999999999997 | 1.9360927152317864 |
| max | 396116.0 | 662.0 | 658.0 | 2.0949367088607596 |

## Image Statistics

### Objects per image

| item | n_objects |
|---|---|
| count | 146.0 |
| mean | 1.582191780821918 |
| std | 3.0529959267060454 |
| min | 0.0 |
| 50% | 0.5 |
| 90% | 5.5 |
| 95% | 9.25 |
| 99% | 14.0 |
| max | 15.0 |

### Objects per positive image

| item | n_objects |
|---|---|
| count | 73.0 |
| mean | 3.164383561643836 |
| std | 3.700651964444677 |
| min | 1.0 |
| 50% | 1.0 |
| 90% | 9.399999999999991 |
| 95% | 12.0 |
| 99% | 14.280000000000001 |
| max | 15.0 |

## Quality Metrics

- Edge bbox ratio: `0.0000`
- valid_fraction mean: `0.9923`

### valid_fraction

| item | valid_fraction |
|---|---|
| count | 146.0 |
| mean | 0.9922900309050962 |
| std | 0.019989282319820133 |
| min | 0.909977912902832 |
| 10% | 0.969273567199707 |
| 50% | 0.9999878406524658 |
| 90% | 1.0 |
| 95% | 1.0 |
| 99% | 1.0 |
| max | 1.0 |

### tile_touches_raster_edge

| item | images |
|---|---|
| False | 124 |
| True | 22 |

### has_edge_object

| item | images |
|---|---|
| False | 87 |
| True | 59 |

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
