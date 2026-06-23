# Refinement Review App

Минимальное Streamlit-приложение для ручной типологии crop-level кандидатов после YOLO proposal generation.

Приложение не меняет исходный датасет, proposal CSV, crops или overlays. Все ручные решения сохраняются отдельно:

```text
04_detection_yolo/reports/refinement_manual_review.csv
```

## Как запустить

Из корня репозитория:

```bash
streamlit run 04_detection_yolo/app/refinement_review_app.py
```

По умолчанию приложение открывает текущий аудит `v3i` proposals:

```text
metadata.csv:
04_detection_yolo/reports/proposals/v3i_conf005/filter_audit/candidate_features.csv

crops:
04_detection_yolo/reports/proposals/v3i_conf005/crops

overlays:
04_detection_yolo/reports/proposals/v3i_conf005/overlays
```

Все пути можно поменять в sidebar. Это нужно для будущих refinement datasets.

## Метки ручной проверки

Используй такие метки:

| Label | Что означает | Можно использовать как negative? |
|---|---|---|
| `trash` | явный мусор: шум, нерелевантный артефакт, явно не объект | да |
| `terrain_like` | склон, овраг, обрыв, край рельефа, raster/tile artifact | да, после spot check |
| `bad_crop` | crop технически плохой или бесполезный | обычно исключать; можно как technical reject |
| `object` | реальный археологический объект или хороший TP proposal | нет, это positive |
| `plausible_object` | похоже на археологический объект, но рядом нет GT | нет |
| `uncertain` | не хватает уверенности | нет |
| `skip` | пропустить / вернуться позже | нет |

Критически важно: **не использовать `plausible_object`, `uncertain` и `skip` как negative examples**.

Иначе refinement-модель может научиться отбрасывать настоящие, но недоразмеченные археологические объекты.

## В каком порядке проверять

1. Сначала открой `group = obvious_fp`.
2. Приоритетные регионы для hard negatives:
   - `004_ДЕМИДОВКА`
   - `025_ШУМГОРА`
   - `011_РУНА`
3. Очень явный мусор размечай как `trash`.
4. Склоны, овраги, обрывы и рельефные артефакты размечай как `terrain_like`.
5. Если объект выглядит археологически правдоподобно, но GT рядом нет, ставь `plausible_object`, а не `trash`.
6. Если сомневаешься, ставь `uncertain`.
7. После `obvious_fp` отдельно проверь `group = plausible_fp`. Это самая опасная группа: там могут быть как FP, так и пропущенная разметка.
8. `missed_gt` появится после сборки refinement dataset с GT crops; эту группу стоит смотреть отдельно.

## Как потом использовать метки

Рекомендуемая логика для первого refinement dataset:

```text
positive:
  object
  automatic TP proposals
  missed GT crops

negative:
  trash
  terrain_like

ignore:
  plausible_object
  uncertain
  skip
  bad_crop, если не используем отдельный technical reject class
```

Цель первого refinement stage — не идеальная классификация, а **уменьшить FP поток, сохранив высокий recall**.

Поэтому лучше иметь меньше, но чище negative examples, чем большой шумный negative set.
