# YOLOv8 Detection of Archaeological Kurgans

## Обзор

Этот модуль посвящен задаче object detection для археологических курганов на данных дистанционного зондирования.

Цель этапа - проверить, насколько хорошо YOLOv8 может находить отдельные курганы на raster tiles, подготовленных из геоданных:

> можно ли перейти от segmentation baseline к bbox detection pipeline для маленьких, слабоконтрастных археологических объектов?

Модуль продолжает исследовательскую линию после `02_unet_segmentation` и `03_multiclass_segmentation_deeplab`, но ставит задачу иначе: вместо pixel mask модель должна предсказывать bounding boxes для отдельных объектов.

Главный вклад этого этапа - не только обучение YOLO, а dataset engineering:

- преобразование raster + polygon-разметки в YOLO bbox dataset;
- фильтрация шумных samples;
- контроль negative sampling;
- балансировка двух kurgan-классов;
- анализ precision/recall trade-off и threshold tuning.

## Постановка задачи

Необходимо детектировать два класса курганов:

| ID | Класс | Описание |
|---:|---|---|
| 0 | `kurgany_tselye` | целые / хорошо выраженные курганы |
| 1 | `kurgany_povrezhdennye` | поврежденные или частично разрушенные курганы |

Модель получает tiled raster image и предсказывает YOLO bounding boxes.

В отличие от segmentation-модулей, здесь качество определяется не пиксельной маской, а тем, насколько корректно модель локализует отдельные объекты.

## Почему detection сложнее в этом модуле

Курганы являются сложными объектами для bbox detection:

- объекты маленькие относительно размера tile;
- контраст часто слабый, особенно на aerial imagery;
- поврежденные курганы имеют нечеткую морфологию;
- плотные группы объектов приводят к duplicate detections;
- tile boundaries могут обрезать объекты и создавать шумные bbox;
- dataset небольшой и несбалансированный;
- LiDAR и aerial imagery дают разные визуальные сигналы.

Из-за этого стандартное увеличение модели не решает задачу само по себе. Основное качество появляется через аккуратную подготовку данных.

## Dataset And Modalities

Исходные данные состоят из растров и polygon-разметки.

Поддерживаемые модальности в исходном bbox pipeline:

| Модальность | Описание |
|---|---|
| `Li` | LiDAR-derived raster |
| `Ae` | aerial imagery |
| `SpOr` | satellite / orthophoto imagery |
| `Or` | дополнительная orthophoto-модальность |

Начальный bbox dataset включал пять классов:

| ID | Класс |
|---:|---|
| 0 | `kurgany_tselye` |
| 1 | `kurgany_povrezhdennye` |
| 2 | `gorodishcha` |
| 3 | `fortifikatsii` |
| 4 | `arkhitektury` |

Для текущей YOLO-серии задача была сужена до двух kurgan-классов и двух основных модальностей: `Li` и `Ae`.

## Pipeline

```text
Geospatial rasters + GeoJSON polygons
        ↓
Region discovery and CRS handling
        ↓
Raster tiling with modality-specific context
        ↓
Polygon intersection with tile bounds
        ↓
YOLO bbox label generation
        ↓
Dataset filtering and negative sampling
        ↓
YOLOv8 training
        ↓
Validation, curves, confusion matrices
        ↓
Experiment analysis: v2 / v3 / v4 / v5
```

Технически pipeline включает:

- поиск регионов и raster directories;
- чтение `UTM.json` и GeoJSON-разметки;
- приведение геометрий к raster CRS при необходимости;
- нарезку raster windows;
- clipping polygons по tile bounds;
- перевод polygon bounds в YOLO bbox format;
- сохранение `images/`, `labels/`, `metadata.csv`, `dataset.yaml`;
- region-based train/validation split;
- обучение и валидацию YOLOv8.

## Dataset Engineering

Первый bbox dataset был слишком шумным: пять классов, несколько модальностей, много background tiles и сильный дисбаланс.

Поэтому в ноутбуке была выполнена серия dataset refinement experiments.

### v2: kurgan-only Li + Ae baseline

Фильтрация:

- оставить только `Li` и `Ae`;
- оставить только классы `0` и `1`;
- пересобрать labels в двухклассовый YOLO dataset;
- сохранить часть negative tiles;
- `NEGATIVE_RATIO = 0.25`;
- `RANDOM_SEED = 42`.

Результат подготовки:

| Характеристика | Значение |
|---|---:|
| Images | 635 |
| Positive images | 368 |
| Negative images | 267 |
| BBoxes | 4288 |
| `kurgany_povrezhdennye` boxes | 3275 |
| `kurgany_tselye` boxes | 1013 |

### v3: clean dataset

v3 был экспериментом с более агрессивной чисткой.

Фильтры:

- частичная фильтрация edge-объектов;
- `MIN_VALID_FRACTION`;
- `MIN_CONTRAST`;
- ограничение dense tiles через `MAX_OBJECTS`;
- downsampling negative tiles.

Результат:

| Характеристика | Значение |
|---|---:|
| Images | 313 |
| BBoxes | 539 |

v3 повысил precision, но заметно снизил recall. Это стало важным сигналом: слишком агрессивная чистка удаляет трудные примеры, которые нужны модели для обобщения.

### v4: balanced clean dataset

v4 стал наиболее сбалансированной версией dataset.

Фильтры:

| Параметр | Значение |
|---|---:|
| Modalities | `Li`, `Ae` |
| Classes | `kurgany_tselye`, `kurgany_povrezhdennye` |
| `NEGATIVE_RATIO` | 0.15 |
| `MIN_VALID_FRACTION` | 0.25 |
| `MIN_CONTRAST` | 3 |
| `MAX_OBJECTS` | 20 |
| Edge-ratio filter | skip if `bbox_touches_tile_edge > 0.8` |
| Random seed | 42 |

Результат подготовки:

| Характеристика | Значение |
|---|---:|
| Images | 347 |
| Positive images | 226 |
| Negative images | 121 |
| BBoxes | 772 |
| `kurgany_povrezhdennye` boxes | 430 |
| `kurgany_tselye` boxes | 342 |

## Эксперименты

Основная серия экспериментов проверяла, что сильнее влияет на качество: размер модели, очистка данных, балансировка классов или confidence threshold.

### Дизайн экспериментов

| Version | Цель | Dataset | Model | Основное изменение |
|---|---|---|---|---|
| v1 | Проверить исходный bbox dataset | 5 классов, несколько модальностей | TODO: confirm | Baseline на полном шумном dataset |
| v2 | Получить рабочий kurgan-only baseline | `Li` + `Ae`, 2 класса | YOLOv8n | Удалены не-kurgan классы, negative ratio `0.25` |
| v3 | Проверить эффект агрессивной чистки | Clean `Li` + `Ae`, 2 класса | YOLOv8s | Edge/object quality filters, contrast filters, dense-tile limit |
| v4 | Сбалансировать качество и class balance | Balanced clean `Li` + `Ae`, 2 класса | YOLOv8s | Negative ratio `0.15`, class balancing, мягче сохранены сложные примеры |
| v5 | Проверить, можно ли поднять recall через threshold tuning | v4-style dataset | YOLOv8s | Validation при confidence `0.10` и `0.15` |

### Summary metrics

| Version | mAP50 | Precision | Recall | AP `tselye` | AP `povrezhdennye` | Вывод |
|---|---:|---:|---:|---:|---:|---|
| v1 | TODO | TODO | TODO | TODO | TODO | Исходный dataset оказался слишком шумным и несбалансированным. |
| v2 | ≈ 0.182 | ≈ 0.32 | ≈ 0.23 | ≈ 0.286 | ≈ 0.078 | Модель работает, но часто пропускает поврежденные курганы. |
| v3 | ≈ 0.17 | ≈ 0.455 | ≈ 0.162 | TODO | TODO | Чистка повысила precision, но снизила recall. |
| v4 | 0.21359 | 0.48504 | 0.20424 | 0.278 | 0.148 | Наиболее сбалансированный вариант; `tselye` легче, damaged-class остается bottleneck. |

### Threshold tuning

| Version | Confidence | mAP50 | Recall | AP `povrezhdennye` | Вывод |
|---|---:|---:|---:|---:|---|
| v5 | 0.10 | ≈ 0.187 | ≈ 0.195 | ≈ 0.130 | Recall не вырос относительно v4. |
| v5 | 0.15 | ≈ 0.178 | ≈ 0.195 | ≈ 0.111 | Повышение threshold ухудшило mAP50 и damaged-class AP. |

Главный вывод серии: bottleneck находится не в confidence threshold, а в данных, разметке и сложности damaged-object morphology.

Основной v4 run сохранен как Colab archive:

```text
runs/yolo_kurgan_detection_v4_kurgans_li_ae_v4_yolov8s_balanced_colab_20260605_131149/
```

Лучшие строки из `results.csv`:

| Run | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `runs_2` | 51 | 0.43985 | 0.18478 | 0.18323 | 0.12095 |
| `v4` | 48 | 0.48504 | 0.20424 | 0.21359 | 0.12590 |

Для v4 также полезно зафиксировать крайние точки обучения:

| Selection | Epoch | Precision | Recall | mAP50 | mAP50-95 | Интерпретация |
|---|---:|---:|---:|---:|---:|---|
| Last epoch | 73 | 0.31420 | 0.18134 | 0.15278 | 0.08118 | После best epoch качество просело. |
| Best mAP50 | 48 | 0.48504 | 0.20424 | 0.21359 | 0.12590 | Основной checkpoint для анализа: `best.pt`. |
| Best recall | 55 | 0.27045 | 0.21655 | 0.14643 | 0.08409 | Recall можно поднять, но ценой mAP и precision. |
| Best precision | 49 | 0.97202 | 0.12500 | 0.18717 | 0.11390 | Очень консервативный режим модели. |

Предыдущий локальный `runs_4` остается legacy artifact для сравнения, но основной v4 result в README теперь соответствует воспроизводимому Colab-run и checkpoint `best.pt`.

### Baseline comparison: Li-only vs Li + Ae

После абляции датасета были отдельно сравнены два кандидата на честный YOLO baseline:

- `dataset_yolo_bbox_v3b_li_binary_medium` - более чистый Li-only dataset;
- `dataset_yolo_bbox_v3d_li_ae_binary_medium` - более крупный Li + Ae dataset.

Оба запуска использовали одинаковую конфигурацию:

```text
model = yolov8n.pt
single_cls = True
imgsz = 640
epochs = 100
seed = 42
close_mosaic = 10
patience = 25
```

Менялся только `dataset.yaml`.

| Dataset | Images | Positive | BBox | Precision | Recall | mAP50 | mAP50-95 | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3b_li_medium` | 284 | 142 | 579 | 0.70544 | 0.28571 | 0.33904 | 0.11516 | 84 |
| `v3d_li_ae_medium` | 622 | 311 | 1207 | 0.35951 | 0.21348 | 0.16164 | 0.06106 | 76 |

Вывод: несмотря на меньший размер, `v3b_li_medium` является более сильным baseline. Добавление `Ae` почти удвоило число изображений и bbox, но ухудшило все основные метрики: precision, recall, mAP50 и mAP50-95.

Разбор ошибок подтверждает это:

| Dataset | Found GT | False negatives | False positives | Основной паттерн ошибок |
|---|---:|---:|---:|---|
| `v3b_li_medium` | 9 | 40 | 3 | Все false negatives относятся к `kurgany_povrezhdennye` на `Li`. |
| `v3d_li_ae_medium` | 14 | 75 | 31 | Много false positives на `Ae`; false negatives есть и на damaged, и на whole Ae-объектах. |

Для `v3d_li_ae_medium` модель чаще пропускает маленькие объекты: median bbox area у найденных объектов `34649.5 px`, у пропущенных - `11438 px`. Для `v3b_li_medium` картина другая: median bbox area у пропущенных объектов больше, чем у найденных (`104232 px` против `21021 px`), поэтому ошибки Li-only baseline связаны не только с размером, а также с визуальной неоднозначностью и damaged morphology.

Практический вывод: текущий baseline стоит строить от `v3b_li_medium`. `Ae` не стоит добавлять в общий train set без отдельной проверки качества, modality-aware sampling или отдельной модели/ветки эксперимента.

### Long-run and YOLO26 checks

После выбора `v3b_li_medium` были проверены две дополнительные гипотезы:

- поможет ли увеличить лимит обучения с `100` до `400` эпох;
- даст ли прирост замена `YOLOv8n` на `YOLO26n` при том же dataset и train config.

Во всех запусках сохранены те же ключевые параметры: `imgsz = 640`, `single_cls = True`, `seed = 42`, `close_mosaic = 10`, `patience = 25`, `batch = -1`. Из-за `patience = 25` оба long-run остановились раньше 400 эпох.

| Experiment | Model | Epochs ran | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Вывод |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `v3b_100_epoch_baseline` | YOLOv8n | 100 | 84 | 0.70544 | 0.28571 | 0.33904 | 0.11516 | Текущий лучший controlled baseline. |
| `v3b_400_epoch_limit` | YOLOv8n | 105 | 74 | 0.51425 | 0.26531 | 0.27816 | 0.09593 | Длиннее обучение не улучшило качество. |
| `v3b_yolo26n_400_epoch_limit` | YOLO26n | 63 | 37 | 0.52136 | 0.20408 | 0.22218 | 0.09721 | YOLO26n не дал прироста на этом dataset. |

Вывод: текущий bottleneck не в числе эпох и не в замене nano-архитектуры. Для следующего этапа разумнее проверять `imgsz = 1024`, YOLOv8s/YOLO26s или улучшение данных и разметки, а не просто увеличивать training budget.

## Результаты

В предыдущей серии v2-v5 наиболее полезной версией была v4:

```text
YOLOv8s + cleaned balanced Li/Ae kurgan dataset
```

Ключевой результат:

```text
v4 best.pt, epoch 48

mAP50 = 0.21359
mAP50-95 = 0.12590
precision = 0.48504
recall = 0.20424

AP kurgany_tselye = 0.278
AP kurgany_povrezhdennye = 0.148
```

По сравнению с v2, качество damaged-class detection улучшилось:

```text
kurgany_povrezhdennye AP: ~0.07-0.08 -> ~0.15
```

При этом задача остается recall-limited: модель лучше находит хорошо выраженные целые курганы и часто отправляет поврежденные объекты в background.

Confusion matrix подтверждает этот вывод: значительная часть объектов уходит в `background`, особенно для `kurgany_povrezhdennye`. Normalized confusion matrix показывает, что поврежденный класс распознается хуже и остается главным ограничением качества.

После отдельной dataset-ablation серии текущим более чистым binary baseline является `v3b_li_medium`: YOLOv8n на Li-only dataset дает `mAP50 = 0.33904`, `precision = 0.70544`, `recall = 0.28571` и превосходит более крупный Li + Ae вариант.

Дополнительные проверки с `epochs = 400` и `YOLO26n` не улучшили этот baseline: лучший результат остается `v3b_li_medium + YOLOv8n + 100 epochs`.

## Key Findings

- Основной прирост качества достигнут не архитектурой, а переработкой dataset.
- `Li` дает более сильный структурный сигнал, чем aerial imagery.
- В прямом сравнении baseline-кандидатов `v3b_li_medium` оказался сильнее, чем более крупный `v3d_li_ae_medium`: `mAP50 0.33904` против `0.16164`.
- Добавление `Ae` без дополнительной чистки увеличивает число false positives и не повышает recall.
- Увеличение лимита обучения до 400 эпох и замена YOLOv8n на YOLO26n не улучшили Li-only baseline.
- `kurgany_tselye` детектируются лучше, чем `kurgany_povrezhdennye`.
- Агрессивная чистка повышает precision, но может ухудшить recall.
- Балансировка v4 улучшила damaged-class AP.
- Confidence threshold tuning не решил проблему recall.
- Главный bottleneck - сложность данных: low contrast, small objects, edge truncation, ambiguity in damaged mounds.

## Visual Results

Сейчас в репозитории есть raw YOLO visual artifacts:

```text
runs/runs_2/
├── results.png
├── BoxPR_curve.png
├── BoxF1_curve.png
├── BoxP_curve.png
├── confusion_matrix.png
└── val_batch*_labels/pred.jpg

runs/yolo_kurgan_detection_v4_kurgans_li_ae_v4_yolov8s_balanced_colab_20260605_131149/
├── kurgans_li_ae_v4_yolov8s_balanced_colab/
│   ├── results.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix_normalized.png
│   ├── val_batch*_labels/pred.jpg
│   └── weights/best.pt
└── kurgans_li_ae_v4_yolov8s_balanced_colab_val_conf_*/

runs/yolo_baseline_comparison/
├── analysis/
│   ├── metrics_comparison.csv
│   ├── baseline_comparison_report.md
│   ├── object_size_found_vs_missed.csv
│   └── */false_negative_contact_sheet.jpg
└── runs/
    ├── v3b_li_medium_yolov8n_img640/
    └── v3d_li_ae_medium_yolov8n_img640/
```

Для GitHub README лучше подготовить curated figures в `reports/figures/`, а не коммитить полные `runs/`.

Наиболее полезные будущие README-фигуры:

- clean panel с LiDAR/Ae examples + predictions;
- v2/v3/v4/v5 experiment comparison chart;
- dataset engineering pipeline diagram;
- class balance / dataset version summary;
- compact confusion matrix;
- failure cases panel.

TODO: скопировать или пересобрать выбранные легкие изображения в `reports/figures/`.

## Failure Cases

Типичные ошибки:

- false negatives на поврежденных или low-contrast курганах;
- путаница между `kurgany_tselye` и `kurgany_povrezhdennye`;
- уход поврежденных курганов в background;
- duplicate detections в плотных группах;
- нестабильная локализация около tile boundaries;
- более слабое качество на aerial imagery по сравнению с LiDAR-derived imagery.

Эти ошибки показывают, что дальнейшее улучшение должно начинаться с данных: annotation review, sampling strategy, tiling strategy и targeted examples for damaged mounds.

## Streamlit Dataset Viewer

В проекте уже есть prototype viewer для проверки YOLO bbox dataset:

```bash
streamlit run skripts/visualize_yolo_labels.py
```

Viewer позволяет фильтровать samples по:

- region;
- modality;
- class;
- positive / negative samples;
- bbox area;
- edge-touching boxes.

Для portfolio-интеграции этот код стоит перенести в:

```text
app/streamlit_app.py
```

и заменить hard-coded dataset path на config или CLI argument.

## Reproducibility

Текущий notebook запускался в Google Colab и ожидает dataset archive в Google Drive.

Основной experimental notebook:

```bash
notebooks/geo_li_ae_kurgan_detection.ipynb
```

Воспроизводимый Colab notebook для обучения из GitHub и сохранения результатов обратно на Google Drive:

```bash
notebooks/colab_yolo_train_from_github_drive.ipynb
```

Новые воспроизводимые entrypoints:

```bash
# Build filtered datasets from the source 5-class YOLO bbox dataset
python scripts/filter_yolo_dataset.py --config configs/dataset_v2.yaml
python scripts/filter_yolo_dataset.py --config configs/dataset_v4.yaml

# Train YOLO models
python scripts/train_yolo.py --config configs/train_v2_yolov8n.yaml
python scripts/train_yolo.py --config configs/train_v4_yolov8s.yaml

# Validate threshold behavior
python scripts/validate_yolo.py --config configs/validate_thresholds_v4.yaml

# Summarize an Ultralytics run
python scripts/summarize_yolo_run.py runs/.../results.csv

# Dataset QA viewer
streamlit run app/streamlit_app.py
```

Старые файлы в `skripts/` оставлены как exploratory prototypes и historical reference. Основная логика v2/v4 filtering теперь вынесена в `src/dataset/filter_yolo_dataset.py`.

Обучение v2 в notebook:

| Параметр | Значение |
|---|---|
| Model | `yolov8n.pt` |
| Data | `dataset_yolo_bbox_v2_kurgans_li_ae/dataset.yaml` |
| Image size | 1024 |
| Epochs | 60 |
| Batch | 8 |
| Patience | 20 |
| Scheduler | `cos_lr=True` |
| Cache | `True` |
| `close_mosaic` | 10 |

Обучение v4 в notebook:

| Параметр | Значение |
|---|---|
| Model | `yolov8s.pt` |
| Data | `dataset_yolo_bbox_v4_kurgans_li_ae_balanced/dataset.yaml` |
| Image size | 1024 |
| Epochs | 80 |
| Batch | 8 |
| Patience | 25 |
| Scheduler | `cos_lr=True` |
| Cache | `True` |
| `close_mosaic` | 15 |
| Run name | `kurgans_li_ae_v4_yolov8s_balanced` |

Colab notebook использует те же идеи, но сохраняет результаты обучения в архив на Google Drive.

## Repository Structure

Текущая целевая структура модуля:

```text
04_detection_yolo/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset_v2.yaml
│   ├── dataset_v4.yaml
│   ├── train_v2_yolov8n.yaml
│   ├── train_v4_yolov8s.yaml
│   └── validate_thresholds_v4.yaml
├── src/
│   └── dataset/
│       └── filter_yolo_dataset.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── filter_yolo_dataset.py
│   ├── train_yolo.py
│   ├── validate_yolo.py
│   └── summarize_yolo_run.py
├── notebooks/
│   └── geo_li_ae_kurgan_detection.ipynb
├── reports/
│   ├── experiments.md
│   └── figures/
└── data_sample/
```

Папка `skripts/` пока сохранена без удаления как historical layer. Новые experiments should use `configs/`, `src/`, `scripts/` and `app/`.

## Ограничения

- Dataset небольшой: после фильтрации остаются сотни images.
- Ранние версии имеют сильный class imbalance.
- Validation set маленький, особенно для `Ae`.
- v3 и v5 metrics сейчас восстановлены из experiment notes, а не из локальных run folders.
- Notebook и scripts содержат hard-coded local/Colab paths.
- Full datasets, model weights и полные `runs/` не должны попадать в GitHub.
- Detection damaged mounds остается главным слабым местом.

## Future Work

- Вынести notebook cells в `src/` и `scripts/`.
- Добавить YAML configs для v2, v3, v4 и threshold tuning.
- Подготовить маленький `data_sample/` без больших исходных данных.
- Перенести Streamlit viewer в `app/`.
- Скопировать curated figures в `reports/figures/`.
- Улучшить sampling для `kurgany_povrezhdennye`.
- Проверить annotation quality для edge-truncated objects.
- Попробовать YOLOv8m или YOLO11 после стабилизации dataset.
- Проверить segmentation-based или hybrid detection approach.
- Добавить multi-scale training и улучшенную tiling strategy.

## Tech Stack

- Python
- pandas, NumPy
- GeoPandas, Rasterio, Shapely
- PIL / Pillow
- Ultralytics YOLOv8
- Streamlit
- Matplotlib
- Google Colab GPU

## Role In The Full Project

Этот модуль является detection-этапом всей серии:

```text
01_geodata_to_cv
        ↓
02_unet_segmentation
        ↓
03_multiclass_segmentation_deeplab
        ↓
04_detection_yolo
```

Главная роль модуля - проверить, можно ли перейти от segmentation masks к object detection для археологических объектов.

Текущий вывод:

```text
YOLOv8 работает как detector,
но качество ограничено не только моделью.

Главный bottleneck - dataset engineering:
small objects, damaged morphology, class imbalance,
negative sampling and tile-level noise.
```
