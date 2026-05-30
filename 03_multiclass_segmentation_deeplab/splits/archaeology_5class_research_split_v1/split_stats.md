# archaeology_5class_research_split_v1

Frozen split protocol artifact. This file was generated from existing split CSVs without recomputing validation regions.

## Files

- train_split.csv: 2278 samples
- val_split.csv: 601 samples
- test_split.csv: not available; TODO only when a real held-out test protocol exists

## Counts

- train regions: 74
- val regions: 30

## Train Class Counts

- kurgany_povrezhdennye: 1345
- kurgany_tselye: 423
- fortifikatsii: 309
- arkhitektury: 159
- gorodishcha: 42

## Val Class Counts

- kurgany_povrezhdennye: 290
- kurgany_tselye: 153
- fortifikatsii: 121
- arkhitektury: 24
- gorodishcha: 13

## Train Modality Counts

- Ae: 907
- Li: 739
- SpOr: 587
- Or: 45

## Val Modality Counts

- Ae: 266
- SpOr: 176
- Li: 159

## Validation Regions

- 044_ГОЧЕВО: 110
- 008_СЕЛЯНЕ: 92
- 006_МОСКОВИТЫ: 65
- 010_НОВЕНЬКОЕ: 47
- 046_ТЫВА_2: 36
- 018_СЕМИБРАТНЕЕ_1: 33
- 054_КУРМЕНТУ: 33
- 028_САРАТОВ: 30
- 123_Курганы_4: 27
- 122_Курганы_3: 25
- 075_Сары_Булун: 17
- 024_УСТЬ-РЕКА: 15
- 025_ШУМГОРА: 12
- 037_КЧР: 12
- 005_ЛУБНО: 10
- 020_ОСЕЧКИ_2: 10
- 032_ПРИМАКИ_1.3км: 7
- 021_НОВОТИТАРОВСКАЯ: 3
- 154_Постройки_5: 3
- 038_ПЕТРОВСКОЕ: 2
- 080_Белая_Гора: 2
- 141_Каменные_выкладки_2: 2
- 022_КРАСНОСЕЛЬСКАЯ_0.5км: 1
- 030_КОПАНСКОЕ: 1
- 072_Каменка: 1
- 084_Маяцкая_крепость: 1
- 087_Городище: 1
- 088_Верхний_Карабут: 1
- 152_Постройки_3: 1
- 153_Постройки_4: 1

## Rule

Do not rerun `make_region_holdout_split` for this benchmark. All model selection/postprocessing is validation-only.
