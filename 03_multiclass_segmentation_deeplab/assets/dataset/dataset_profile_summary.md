# Archaeology Dataset Profile

## Scope

- Metadata rows: 3260
- Regions: 109
- Modalities: Ae, Li, Or, SpOr
- Object counts are connected-component occurrences inside stored patch masks.
- The dataset has no global object identifier, so repeated crops of one source polygon cannot be deduplicated reliably.
- Object areas are native-mask pixel areas.
- The profile reports raw components and components with `min_area >= 8`, matching the polygon evaluator cleanup threshold.

## Object Occurrences By Class

| class_name | raw_object_occurrences | object_occurrences_min_area_8 |
| --- | --- | --- |
| kurgany_tselye | 6761 | 6722 |
| kurgany_povrezhdennye | 8576 | 8520 |
| gorodishcha | 1514 | 403 |
| fortifikatsii | 1611 | 1590 |
| arkhitektury | 1991 | 1981 |

## Samples By Primary Class

| class_name | samples |
| --- | --- |
| kurgany_tselye | 669 |
| kurgany_povrezhdennye | 1822 |
| gorodishcha | 78 |
| fortifikatsii | 473 |
| arkhitektury | 218 |

## Samples By Modality

| modality | samples |
| --- | --- |
| Li | 934 |
| Ae | 1274 |
| SpOr | 976 |
| Or | 76 |

## Class x Modality

| class_name | Li | Ae | SpOr | Or |
| --- | --- | --- | --- | --- |
| kurgany_tselye | 158 | 366 | 145 | 0 |
| kurgany_povrezhdennye | 569 | 638 | 615 | 0 |
| gorodishcha | 15 | 11 | 48 | 4 |
| fortifikatsii | 192 | 243 | 36 | 2 |
| arkhitektury | 0 | 16 | 132 | 70 |

## Top 20 Regions

| region | samples |
| --- | --- |
| 027_ТИМЕРЕВО | 885 |
| 026_БОРОВИЧИ | 194 |
| 042_ИЗБОРСК | 185 |
| 044_ГОЧЕВО | 112 |
| 033_МИЛОВИДОВО_0.1км | 109 |
| 008_СЕЛЯНЕ | 92 |
| 007_ЮШКОВО | 83 |
| 047_КАЛМЫКИЯ_1 | 82 |
| 017_ВЫШЕГЖА | 72 |
| 006_МОСКОВИТЫ | 69 |
| 116_Старый_Барах | 69 |
| 058_СЕЛЬЦО | 65 |
| 057_ШИШКИНО | 64 |
| 046_ТЫВА_2 | 63 |
| 055_КАБАРДИНО-БАЛКАРИЯ | 60 |
| 121_Курганы_2 | 52 |
| 091_Кисловодская_котловина_1 | 51 |
| 010_НОВЕНЬКОЕ | 49 |
| 041_ЧУДОВО | 48 |
| 054_КУРМЕНТУ | 42 |

## Object Area By Class

| class_name | raw_object_occurrences | raw_mean_area_px | raw_median_area_px | mean_area_px_min_area_8 | median_area_px_min_area_8 |
| --- | --- | --- | --- | --- | --- |
| kurgany_tselye | 6761 | 4298.03 | 170.0 | 4322.95 | 172.0 |
| kurgany_povrezhdennye | 8576 | 23390.82 | 1081.0 | 23544.53 | 1083.0 |
| gorodishcha | 1514 | 70249.28 | 2.0 | 263909.7 | 9509.0 |
| fortifikatsii | 1611 | 26462.73 | 3042.0 | 26812.2 | 3116.5 |
| arkhitektury | 1991 | 35649.18 | 267.0 | 35829.12 | 267.0 |

## Regional Modality Comparison

The collage uses one region and one primary class represented in `Li`, `Ae` and `SpOr`.
The metadata does not contain a shared crop coordinate identifier, so the selected patches are regional examples and are not guaranteed to be pixel-aligned.

| region | class_name | modality | sample_id |
| --- | --- | --- | --- |
| 033_МИЛОВИДОВО_0.1км | kurgany_tselye | Li | 001686 |
| 033_МИЛОВИДОВО_0.1км | kurgany_tselye | Ae | 001696 |
| 033_МИЛОВИДОВО_0.1км | kurgany_tselye | SpOr | 001794 |
