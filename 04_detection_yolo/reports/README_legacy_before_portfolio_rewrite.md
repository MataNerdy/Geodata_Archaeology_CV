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

### Хроника экспериментов из `runs/`

Ниже зафиксирована фактическая история локальных и Kaggle/Colab запусков, которые сейчас лежат в `runs/`. Таблица использует best epoch по `metrics/mAP50(B)` из `results.csv`, если не указано иначе.

| Stage | Run folder | Dataset / idea | Model | imgsz | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Основной вывод |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Early baseline | `runs/runs_2` | ранний Li/Ae kurgan baseline | YOLO | TODO | 51 | 0.43985 | 0.18478 | 0.18323 | 0.12095 | Модель обучается, но recall низкий. |
| Early v4-like run | `runs/runs_4` | более чистый/сбалансированный kurgan dataset | YOLO | TODO | 52 | 0.54479 | 0.18838 | 0.22443 | 0.12145 | Небольшой прирост mAP50 относительно `runs_2`, но recall почти не вырос. |
| v3 Li binary Kaggle | `runs/yolo_kurgan_detection_v3_li_binary_20260609_091639` | Li-only binary kurgan | YOLOv8n | 640 | 35 | 0.52643 | 0.26923 | 0.22620 | 0.06808 | Li-only направление выглядит перспективнее, но локализация остается слабой. |
| v4 Colab balanced | `runs/yolo_kurgan_detection_v4_kurgans_li_ae_v4_yolov8s_balanced_colab_20260605_131149` | balanced Li/Ae kurgan dataset | YOLOv8s | 1024 | 48 | 0.48504 | 0.20424 | 0.21359 | 0.12590 | Сбалансированный v4 улучшил damaged-class AP, но не решил recall bottleneck. |
| v3b controlled baseline | `runs/yolo_baseline_comparison/runs/v3b_li_medium_yolov8n_img640` | Li-only medium, binary kurgan | YOLOv8n | 640 | 84 | 0.70544 | 0.28571 | 0.33904 | 0.11516 | Лучший controlled baseline до ручной чистки. |
| v3d Li+Ae comparison | `runs/yolo_baseline_comparison/runs/v3d_li_ae_medium_yolov8n_img640` | Li + Ae medium, binary kurgan | YOLOv8n | 640 | 76 | 0.35951 | 0.21348 | 0.16164 | 0.06106 | Увеличение датасета за счет `Ae` ухудшило все основные метрики. |
| v3e Ae transfer check | `runs/yolo_ae_transfer_to_lidar/runs/v3e_train_li_ae_val_li_yolov8n_img640` | train Li+Ae, val Li | YOLOv8n | 640 | 60 | 0.46538 | 0.26531 | 0.25010 | 0.07726 | `Ae` как train-сигнал не превзошел чистый Li-only baseline на Li validation. |
| v3b long run | `runs/yolo_v3b_400_epochs` | v3b, лимит 400 epochs | YOLOv8n | 640 | 74 | 0.51425 | 0.26531 | 0.27816 | 0.09593 | Больше эпох не помогло; early stopping остановил run на 105 эпохах. |
| v3b YOLO26 check | `runs/yolo_v3b_yolo26_400_epochs` | v3b, YOLO26n, лимит 400 epochs | YOLO26n | 640 | 37 | 0.52136 | 0.20408 | 0.22218 | 0.09721 | YOLO26n не улучшил YOLOv8n baseline. |
| v3b high-resolution check | `runs/yolo_v3b_img1024_20260614_150032` | v3b, larger input | YOLOv8n | 1024 | 89 | 0.43482 | 0.24490 | 0.24261 | 0.10972 | `imgsz=1024` ухудшил обычные метрики; низкий threshold дает больше proposals, но с большим числом FP. |
| v3g manual-clean baseline | `runs/yolo_v3g_manual_keep_20260617_181819` | manual keep-only Li medium, new split | YOLOv8n | 640 | 49 | 0.52174 | 0.20354 | 0.20386 | 0.08053 | Ручная чистка дала более честный clean baseline, но не подняла detection mAP. |
| v3h curated validation | `runs/yolo_v3h_curated_val_20260618_131256` | manual-clean dataset with curated region validation | YOLOv8n | 640 | 60 | 0.42761 | 0.24107 | 0.27114 | 0.12092 | Curated validation поднял mAP относительно v3g, но выявил сильный региональный bottleneck. |
| v3h no-Saratov sanity check | `runs/yolo_v3h_no_saratov_20260618_163640` | same v3h dataset, `028_САРАТОВ` moved from val to train | YOLOv8n | 640 | 82 | 0.68752 | 0.40909 | 0.46433 | 0.20339 | Удаление Саратова из val резко улучшило метрики, но validation стал маленьким и менее сбалансированным. |
| v3i merged archaeological object | `runs/yolo_v3i_archaeological_object_20260618_221705` | Li-only merged one-class `archaeological_object` dataset | YOLOv8n | 640 | 87 | 0.65580 | 0.32407 | 0.35723 | 0.10604 | Расширение класса до всех археологических объектов не превзошло kurgan-only v3h/no-Saratov, но оказалось полезным для proposal-mode анализа. |
| v3i model/imgsz comparison | `runs/yolo_v3i_model_imgsz_comparison_20260620_212713` | same v3i dataset, model/image-size ablation | YOLOv8n / YOLO26n | 640 / 1024 | 87 / 173 / 138 / 141 | best: 0.65580 | best: 0.34259 | best: 0.35723 | best: 0.15516 | `YOLOv8n 640` остался лучшим proposal baseline; `YOLO26n 640` дал лучший mAP50-95, но хуже low-confidence coverage. |

Короткая интерпретация этой истории:

- лучший controlled baseline на полноценном сравнении датасетов остается `v3b_li_medium + YOLOv8n + imgsz=640`; no-Saratov показывает более высокий `mAP50`, но только как diagnostic split на маленькой validation выборке;
- `Li + Ae` не улучшил качество ни в прямом сравнении v3d, ни в transfer-проверке v3e;
- увеличение эпох, переход на YOLO26n и `imgsz=1024` не решили bottleneck;
- ручная чистка v3g убрала заведомо плохие tiles и дала более надежный validation split, но метрики стали ниже: это сигнал, что прежний v3b частично выигрывал от особенностей старого split/датасета, а не только от лучшего обобщения.
- curated split v3h показал, что качество сильно зависит от состава validation regions; регион `028_САРАТОВ` оказался главным источником false negatives.
- no-Saratov run полезен как diagnostic/sanity check, но не должен автоматически считаться финальным baseline: validation уменьшается до 31 images / 27 positive / 66 bbox и становится перекошенной в сторону `kurgany_povrezhdennye`.
- v3i проверил альтернативную постановку `archaeological_object`: один класс вместо узкого `kurgan`. Стандартные detection-метрики оказались ниже, чем у kurgan-only no-Saratov, но threshold sweep показал высокий потенциал low-confidence proposal generation.
- Последующее сравнение `YOLOv8n`/`YOLO26n` и `imgsz = 640/1024` на v3i не подтвердило пользу `1024`: большие изображения визуально дают много правдоподобных candidate objects, но по текущей GT-разметке резко увеличивают число false positives.

### Inference-only и proposal mode

После обучения v3g был проведен отдельный inference-only анализ без переобучения. Цель была не повысить mAP, а проверить, может ли YOLO работать как генератор кандидатов для будущего пайплайна:

```text
LiDAR tile
    ↓
YOLO low-confidence proposal generation
    ↓
bbox crop
    ↓
segmentation refinement
```

Для `v3g_li_manual_keep_yolov8n_640` threshold sweep при `NMS IoU = 0.5` показал:

| Confidence | TP | FP | FN | Precision | Recall | Coverage@IoU0.3 | FP/image | Вывод |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 13 | 2 | 100 | 0.8667 | 0.1150 | 0.1150 | 0.04 | Стандартный threshold слишком консервативен. |
| 0.05 | 19 | 29 | 94 | 0.3958 | 0.1681 | 0.1858 | 0.53 | Лучший умеренный proposal-режим. |
| 0.03 | 20 | 47 | 93 | 0.2985 | 0.1770 | 0.2124 | 0.85 | Больше coverage, но FP уже заметно растет. |
| 0.01 | 21 | 113 | 92 | 0.1567 | 0.1858 | 0.2743 | 2.05 | Может быть полезно для aggressive proposal generation. |
| 0.005 | 27 | 211 | 86 | 0.1134 | 0.2389 | 0.3363 | 3.84 | Recall растет, но цена в FP высокая. |
| 0.001 | 34 | 1008 | 79 | 0.0326 | 0.3009 | 0.4690 | 18.33 | Почти diagnostic mode: показывает скрытые кандидаты, но непрактичен без сильного downstream-фильтра. |

Отдельный proposal coverage анализ на `conf=0.001`:

| IoU threshold | Covered GT | Total GT | Coverage rate |
|---:|---:|---:|---:|
| 0.10 | 73 | 113 | 0.6460 |
| 0.20 | 60 | 113 | 0.5310 |
| 0.30 | 53 | 113 | 0.4690 |
| 0.50 | 34 | 113 | 0.3009 |

Вывод по inference-only анализу: модель действительно слишком консервативна при стандартных threshold, но это не единственная проблема. При низких confidence она находит больше кандидатов, однако FP быстро растут. Для proposal pipeline наиболее реалистичный стартовый режим - `conf=0.03-0.05`; `conf=0.01` стоит проверять только если downstream segmentation/refinement способен дешево отсеивать ложные crop-кандидаты.

### Curated validation: Saratov effect

После ручной чистки был собран `v3h_li_manual_curated_val` с validation regions:

```text
008_СЕЛЯНЕ
019_ОСЕЧКИ_1
025_ШУМГОРА
028_САРАТОВ
037_КЧР
```

Полный curated validation run:

| Dataset | Val images | Val positive | Val bbox | Precision | Recall | mAP50 | mAP50-95 | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3h_li_manual_curated_val` | 46 | 38 | 112 | 0.42761 | 0.24107 | 0.27114 | 0.12092 | 60 |

Детальный FN-аудит этого run показал, что `028_САРАТОВ` доминирует среди ошибок:

| Region | False negatives |
|---|---:|
| `028_САРАТОВ` | 44 |
| `008_СЕЛЯНЕ` | 17 |
| `019_ОСЕЧКИ_1` | 12 |
| `025_ШУМГОРА` | 8 |
| `037_КЧР` | 7 |

Типы FN на полном v3h при рабочей метрике `conf >= 0.25`, `IoU >= 0.5`:

| FN type | Count | Интерпретация |
|---|---:|---|
| `metric_miss` | 43 | Предсказание близко к объекту, но не проходит рабочий threshold. |
| `hard_miss` | 28 | Модель не генерирует достаточно близкий bbox-кандидат. |
| `near_miss` | 17 | Есть приблизительный bbox-кандидат, но локализация недостаточна для IoU 0.5. |

Чтобы проверить, не ломает ли Саратов всю validation-картину, был запущен sanity check `v3h_no_saratov`: `028_САРАТОВ` перенесен из validation в train, остальные настройки обучения оставлены прежними.

| Dataset | Train images | Val images | Val bbox | Precision | Recall | mAP50 | mAP50-95 | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3h_li_manual_curated_val` | 212 | 46 | 112 | 0.42761 | 0.24107 | 0.27114 | 0.12092 | 60 |
| `v3h_no_saratov` | 227 | 31 | 66 | 0.68752 | 0.40909 | 0.46433 | 0.20339 | 82 |

No-Saratov результат сильно выше, но интерпретация осторожная: validation после удаления Саратова стала маленькой и менее репрезентативной. В ней осталось только 31 image, 27 positive images и 66 bbox; class balance в val смещен к `kurgany_povrezhdennye` (`58` bbox против `8` `kurgany_tselye`).

Региональный аудит no-Saratov validation при `conf = 0.25`, `IoU = 0.5`:

| Region | GT | TP | FN | Recall | `metric_miss` | `near_miss` | `hard_miss` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `019_ОСЕЧКИ_1` | 12 | 3 | 9 | 0.250 | 8 | 0 | 1 |
| `037_КЧР` | 12 | 3 | 9 | 0.250 | 6 | 1 | 2 |
| `008_СЕЛЯНЕ` | 30 | 11 | 19 | 0.367 | 11 | 4 | 4 |
| `025_ШУМГОРА` | 12 | 5 | 7 | 0.417 | 6 | 0 | 1 |

Threshold sweep для no-Saratov показывает, что значительная часть объектов существует среди low-confidence кандидатов:

| Confidence | TP | FP | FN | Precision | Recall | F1 | Coverage@IoU0.3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 12 | 2 | 54 | 0.857 | 0.182 | 0.300 | 0.879 |
| 0.25 | 22 | 9 | 44 | 0.710 | 0.333 | 0.454 | 0.879 |
| 0.10 | 29 | 29 | 37 | 0.500 | 0.439 | 0.468 | 0.879 |
| 0.05 | 37 | 54 | 29 | 0.407 | 0.561 | 0.471 | 0.879 |
| 0.03 | 39 | 79 | 27 | 0.331 | 0.591 | 0.424 | 0.879 |
| 0.01 | 43 | 244 | 23 | 0.150 | 0.652 | 0.244 | 0.879 |
| 0.005 | 47 | 529 | 19 | 0.082 | 0.712 | 0.146 | 0.879 |
| 0.003 | 51 | 917 | 15 | 0.053 | 0.773 | 0.099 | 0.879 |
| 0.001 | 53 | 2540 | 13 | 0.020 | 0.803 | 0.040 | 0.879 |

Вывод: `028_САРАТОВ` нужно анализировать отдельно как hard validation region или потенциально отдельный domain slice. На текущем этапе no-Saratov run доказывает, что модель может показывать заметно лучшую детекцию на более стабильных регионах, но финальный baseline должен оцениваться на validation split, где Саратов либо явно представлен как сложный регион, либо вынесен в отдельный stress-test.

### v3i: merged archaeological-object experiment

Следующая проверка была связана с другой гипотезой: возможно, bottleneck связан не только с качеством данных, но и со слишком узким определением positive-класса. Для этого был собран merged one-class dataset:

```text
dataset_yolo_bbox_v3i_li_archaeological_object_merged
```

В один YOLO-класс `0: archaeological_object` были сведены исходные классы:

- `kurgany_tselye`
- `kurgany_povrezhdennye`
- `gorodishcha`
- `fortifikatsii`
- `arkhitektury`

Фактически в Li-only split представлены курганы, городища и фортификации; `arkhitektury` в текущем датасете не дали bbox.

Состав v3i:

| Split | Images | Positive images | Negative images | BBox |
|---|---:|---:|---:|---:|
| train | 408 | 237 | 171 | 1069 |
| val | 68 | 48 | 20 | 108 |

Validation regions:

```text
004_ДЕМИДОВКА
005_ЛУБНО
006_МОСКОВИТЫ
011_РУНА
012_ЛИХУША
013_БЕРВЕНЕЦ
025_ШУМГОРА
037_КЧР
```

Leakage check: `region = 0`, `source_id = 0`, `raster_file = 0`.

Распределение bbox по исходным классам:

| Split | Source class | BBox |
|---|---|---:|
| train | `fortifikatsii` | 414 |
| train | `gorodishcha` | 54 |
| train | `kurgany_povrezhdennye` | 335 |
| train | `kurgany_tselye` | 266 |
| val | `fortifikatsii` | 47 |
| val | `gorodishcha` | 13 |
| val | `kurgany_povrezhdennye` | 28 |
| val | `kurgany_tselye` | 20 |

Обучение:

```text
model = yolov8n.pt
imgsz = 640
epochs = 300
batch = 16
seed = 42
single_cls = True
close_mosaic = 10
patience = 100
```

Фактический `results.csv` содержит 136 эпох: обучение было запущено с лимитом `epochs = 300`, но остановилось раньше. Лучший `mAP50` достигнут на эпохе 87.

| Dataset | Model | imgsz | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `v3i_archaeological_object` | YOLOv8n | 640 | 87 | 0.65580 | 0.32407 | 0.35723 | 0.10604 |

Сравнение с ближайшим kurgan-only sanity check:

| Dataset | Target | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| `v3h_no_saratov` | `kurgan` | 0.68752 | 0.40909 | 0.46433 | 0.20339 |
| `v3i_archaeological_object` | `archaeological_object` | 0.65580 | 0.32407 | 0.35723 | 0.10604 |

Вывод по стандартным detection-метрикам: простое расширение класса с `kurgan` до `archaeological_object` не улучшило baseline. Широкий класс добавил данных, но добавил и внутриклассовую неоднородность: округлые курганы, площадные городища и линейные фортификации имеют разные визуальные паттерны.

При этом corrected threshold sweep показал, что v3i интересен как proposal generator:

| Confidence | TP | FP | FN | Precision | Recall | Coverage@IoU0.3 | Predictions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 13 | 0 | 95 | 1.000 | 0.120 | 0.120 | 13 |
| 0.25 | 26 | 7 | 82 | 0.788 | 0.241 | 0.250 | 33 |
| 0.10 | 40 | 63 | 68 | 0.388 | 0.370 | 0.444 | 103 |
| 0.05 | 55 | 174 | 53 | 0.240 | 0.509 | 0.639 | 229 |
| 0.03 | 58 | 313 | 50 | 0.156 | 0.537 | 0.685 | 371 |
| 0.01 | 71 | 918 | 37 | 0.072 | 0.657 | 0.778 | 989 |
| 0.005 | 77 | 1745 | 31 | 0.042 | 0.713 | 0.824 | 1822 |
| 0.003 | 80 | 2718 | 28 | 0.029 | 0.741 | 0.861 | 2798 |
| 0.001 | 83 | 6643 | 25 | 0.012 | 0.769 | 0.926 | 6726 |

Интерпретация: как финальный detector v3i слабоват, но как первый этап `YOLO proposals -> crop -> segmentation/refinement` он может быть полезен. При `conf = 0.05-0.01` резко растет coverage, но вместе с ним растет и число false positives. Это делает v3i скорее proposal-моделью, чем финальной моделью детекции.

Региональный аудит при `conf = 0.25`:

| Region | GT | TP | FN | Recall | Основные классы |
|---|---:|---:|---:|---:|---|
| `004_ДЕМИДОВКА` | 3 | 0 | 3 | 0.000 | `gorodishcha` |
| `005_ЛУБНО` | 27 | 1 | 26 | 0.037 | `fortifikatsii; gorodishcha` |
| `037_КЧР` | 12 | 1 | 11 | 0.083 | `kurgany_povrezhdennye` |
| `006_МОСКОВИТЫ` | 22 | 3 | 19 | 0.136 | `fortifikatsii` |
| `025_ШУМГОРА` | 18 | 4 | 14 | 0.222 | `gorodishcha; kurgany_povrezhdennye; kurgany_tselye` |
| `012_ЛИХУША` | 12 | 4 | 8 | 0.333 | `gorodishcha; kurgany_tselye` |
| `013_БЕРВЕНЕЦ` | 12 | 11 | 1 | 0.917 | `kurgany_povrezhdennye; kurgany_tselye` |
| `011_РУНА` | 2 | 2 | 0 | 1.000 | `gorodishcha` |

Самые проблемные регионы для v3i: `005_ЛУБНО`, `006_МОСКОВИТЫ`, `037_КЧР`, `004_ДЕМИДОВКА`. Особенно заметна слабость на фортификациях в `005_ЛУБНО` и `006_МОСКОВИТЫ`.

### v3i model and image-size comparison

После базового v3i-run была проведена controlled ablation серия на том же dataset:

```text
dataset_yolo_bbox_v3i_li_archaeological_object_merged
```

Сравнивались четыре конфигурации:

- `YOLOv8n`, `imgsz = 640`
- `YOLOv8n`, `imgsz = 1024`
- `YOLO26n`, `imgsz = 640`
- `YOLO26n`, `imgsz = 1024`

Общие параметры:

```text
epochs = 300
patience = 100
batch = 16
seed = 42
single_cls = True
close_mosaic = 10
```

Результаты:

| Experiment | Model | imgsz | Epochs completed | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3i_yolov8n_img640` | YOLOv8n | 640 | 136 | 87 | 0.65580 | 0.32407 | 0.35723 | 0.10604 |
| `v3i_yolo26n_img640` | YOLO26n | 640 | 208 | 173 | 0.59923 | 0.34259 | 0.35024 | 0.15516 |
| `v3i_yolo26n_img1024` | YOLO26n | 1024 | 300 | 138 | 0.61251 | 0.26350 | 0.27636 | 0.11370 |
| `v3i_yolov8n_img1024` | YOLOv8n | 1024 | 258 | 141 | 0.58746 | 0.24074 | 0.27572 | 0.09563 |

Интерпретация:

- `YOLOv8n 640` остается лучшим вариантом по `mAP50` и самым полезным как proposal baseline.
- `YOLO26n 640` дает лучший `mAP50-95`, то есть немного лучше по строгой локализации, но не выигрывает по `mAP50`.
- `imgsz = 1024` не дал прироста ни для YOLOv8n, ни для YOLO26n.
- Визуально `YOLOv8n 1024` генерирует много теоретически подходящих археологических candidates, но по текущей разметке это превращается в большое число false positives.

Proposal-mode сравнение подтверждает это:

| Experiment | conf | TP | FP | FN | Precision | Recall | Coverage@IoU0.3 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3i_yolov8n_img640` | 0.05 | 55 | 174 | 53 | 0.240 | 0.509 | 0.639 | 229 |
| `v3i_yolov8n_img640` | 0.01 | 71 | 918 | 37 | 0.072 | 0.657 | 0.778 | 989 |
| `v3i_yolov8n_img1024` | 0.05 | 40 | 272 | 68 | 0.128 | 0.370 | 0.500 | 312 |
| `v3i_yolov8n_img1024` | 0.01 | 53 | 812 | 55 | 0.061 | 0.491 | 0.593 | 865 |
| `v3i_yolo26n_img640` | 0.05 | 40 | 99 | 68 | 0.288 | 0.370 | 0.398 | 139 |
| `v3i_yolo26n_img640` | 0.01 | 50 | 503 | 58 | 0.090 | 0.463 | 0.528 | 553 |
| `v3i_yolo26n_img1024` | 0.05 | 28 | 56 | 80 | 0.333 | 0.259 | 0.315 | 84 |
| `v3i_yolo26n_img1024` | 0.01 | 36 | 217 | 72 | 0.142 | 0.333 | 0.454 | 253 |

Практический вывод: если v3i использовать как первый этап `YOLO proposals -> crop -> segmentation/refinement`, наиболее полезная конфигурация сейчас `YOLOv8n 640` с low-confidence inference. `conf = 0.05` дает более управляемый поток candidates, `conf = 0.01` дает высокий coverage, но требует сильного downstream-фильтра.

### v3i low-confidence validation review

Чтобы проверить proposal-mode не только по метрикам, но и глазами, весь validation split `v3i` был прогнан через `YOLOv8n 640 best.pt` с низкими confidence thresholds.

Артефакты:

```text
runs/v3i_yolov8n640_low_conf_val_review_outputs/
├── analysis/
│   ├── threshold_sweep.csv
│   ├── per_image_threshold_summary.csv
│   ├── gallery_conf_0p05.html
│   ├── gallery_conf_0p03.html
│   ├── gallery_conf_0p01.html
│   ├── gallery_conf_0p005.html
│   ├── contact_sheets_conf_0p05/
│   ├── contact_sheets_conf_0p03/
│   ├── contact_sheets_conf_0p01/
│   └── contact_sheets_conf_0p005/
└── weights/
```

Цветовая маркировка review images:

- green - missed GT;
- cyan - matched GT / matched prediction;
- orange - unmatched prediction.

Threshold sweep:

| Confidence | TP | FP | FN | Precision | Recall | F1 | Coverage@IoU0.3 | Predictions | FP/image |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 26 | 7 | 82 | 0.788 | 0.241 | 0.369 | 0.250 | 33 | 0.10 |
| 0.10 | 40 | 63 | 68 | 0.388 | 0.370 | 0.379 | 0.444 | 103 | 0.93 |
| 0.05 | 55 | 174 | 53 | 0.240 | 0.509 | 0.326 | 0.639 | 229 | 2.56 |
| 0.03 | 58 | 313 | 50 | 0.156 | 0.537 | 0.242 | 0.685 | 371 | 4.60 |
| 0.01 | 71 | 918 | 37 | 0.072 | 0.657 | 0.129 | 0.778 | 989 | 13.50 |
| 0.005 | 77 | 1745 | 31 | 0.042 | 0.713 | 0.080 | 0.824 | 1822 | 25.66 |
| 0.003 | 80 | 2718 | 28 | 0.029 | 0.741 | 0.055 | 0.861 | 2798 | 39.97 |
| 0.001 | 83 | 6643 | 25 | 0.012 | 0.769 | 0.024 | 0.926 | 6726 | 97.69 |

Визуальный review contact sheets подтвердил количественную картину:

- `conf = 0.05` - лучший режим для ручной ревизии: модель находит много валидных candidates, но поток FP еще можно просматривать глазами;
- `conf = 0.03` - компромиссный aggressive review mode: coverage немного выше, но FP уже заметно мешают;
- `conf = 0.01` - почти режим разведки: модель покрывает больше объектов, но на большинстве сложных tiles появляется плотная сетка orange predictions;
- `conf <= 0.005` - diagnostic mode, полезный для понимания того, что модель вообще считает похожим на объект, но непрактичный без сильного downstream-фильтра.

Top FP-heavy images показывают, что ложные срабатывания концентрируются в регионах `005_ЛУБНО`, `006_МОСКОВИТЫ`, `004_ДЕМИДОВКА`, `011_РУНА` и `013_БЕРВЕНЕЦ`. На этих tiles модель часто реагирует на крупные рельефные формы, края склонов, тени, линейные структуры и площадные морфологические паттерны. Bbox overlay визуально корректный: проблема не в сдвиге координат, а в том, что low-confidence detector широко размечает геоморфологически похожие объекты.

Практический вывод: `YOLOv8n 640` действительно может работать как proposal generator, но не как самостоятельный high-quality detector. Рабочий диапазон для следующего этапа:

- `conf = 0.05`, если нужен ограниченный поток кандидатов для ручной проверки или слабого downstream-фильтра;
- `conf = 0.01`, если цель - максимальное покрытие и дальше есть сильный segmentation/refinement stage.

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

После ручного аудита был собран `v3g_li_manual_keep`: 258 images, 133 positive images, 125 negative images и 566 bbox. На новом region-aware split модель `YOLOv8n` получила `precision = 0.52174`, `recall = 0.20354`, `mAP50 = 0.20386`, `mAP50-95 = 0.08053`. Этот результат ниже v3b, но он важен как более честная clean-baseline точка после удаления битых tiles и неправильных bbox.

Следующий curated split `v3h_li_manual_curated_val` дал `precision = 0.42761`, `recall = 0.24107`, `mAP50 = 0.27114`, `mAP50-95 = 0.12092`. Дополнительный no-Saratov sanity check поднял метрики до `precision = 0.68752`, `recall = 0.40909`, `mAP50 = 0.46433`, `mAP50-95 = 0.20339`, но на меньшей validation выборке. Это главный свежий сигнал: проблема не только в модели, но и в сильной региональной неоднородности данных.

Эксперимент `v3i_archaeological_object` проверил более широкий target-класс. YOLOv8n на merged one-class dataset дал `precision = 0.65580`, `recall = 0.32407`, `mAP50 = 0.35723`, `mAP50-95 = 0.10604`. Это хуже, чем kurgan-only no-Saratov sanity check, поэтому расширение класса само по себе не является решением. Однако low-confidence sweep показал высокое proposal coverage: при `conf = 0.01` покрытие GT на `IoU >= 0.30` достигает `0.778`, а при `conf = 0.003` - `0.861`.

Дополнительное сравнение `YOLOv8n` и `YOLO26n` на v3i показало, что `YOLO26n 640` улучшает `mAP50-95` до `0.15516`, но не превосходит `YOLOv8n 640` по `mAP50` и proposal coverage. Увеличение `imgsz` до `1024` ухудшило стандартные метрики для обеих архитектур. Поэтому текущая рабочая конфигурация для proposal-mode остается `YOLOv8n + imgsz = 640`.

Полный визуальный review validation split для `YOLOv8n 640` подтвердил, что модель не просто "молчит": при снижении threshold она генерирует большое число археологически похожих candidates. Оптимальный практический режим сейчас - `conf = 0.05`: `coverage@IoU0.3 = 0.639`, `recall = 0.509`, примерно `2.56 FP/image`. `conf = 0.01` повышает coverage до `0.778`, но дает около `13.5 FP/image`, поэтому годится скорее для aggressive proposal mining, чем для финальной детекции.

## Key Findings

- Основной прирост качества достигнут не архитектурой, а переработкой dataset.
- `Li` дает более сильный структурный сигнал, чем aerial imagery.
- В прямом сравнении baseline-кандидатов `v3b_li_medium` оказался сильнее, чем более крупный `v3d_li_ae_medium`: `mAP50 0.33904` против `0.16164`.
- Добавление `Ae` без дополнительной чистки увеличивает число false positives и не повышает recall.
- Увеличение лимита обучения до 400 эпох и замена YOLOv8n на YOLO26n не улучшили Li-only baseline.
- `imgsz = 1024` не улучшил controlled v3b baseline как обычный detector.
- Ручная чистка v3g улучшила доверие к dataset/split, но не решила проблему качества detector.
- Curated validation v3h выявил региональный bottleneck: `028_САРАТОВ` дает непропорционально много false negatives.
- No-Saratov sanity check резко повышает метрики, но его нельзя читать как окончательную победу из-за маленького и смещенного validation split.
- Merged one-class постановка `archaeological_object` не превзошла kurgan-only baseline: широкий класс оказался визуально слишком неоднородным для YOLOv8n на текущем размере датасета.
- В v3i low-confidence режим дает высокий proposal coverage: `coverage@IoU0.3 = 0.778` при `conf = 0.01` и `0.861` при `conf = 0.003`, но с большим числом false positives.
- В v3i model/imgsz comparison `YOLO26n 640` улучшил строгую локализацию (`mAP50-95 = 0.15516`), но `YOLOv8n 640` остался лучше как practical proposal baseline.
- `imgsz = 1024` на v3i визуально находит много похожих объектов-кандидатов, но по GT-разметке это проявляется как FP-heavy режим, а не как улучшение detector quality.
- Full validation low-confidence review подтвердил рабочий диапазон для proposal generation: `conf = 0.05` как управляемый режим, `conf = 0.01` как aggressive mining mode.
- Low-confidence inference может быть полезен как proposal generation, но только вместе с downstream filtering/segmentation.
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

runs/yolo_v3i_archaeological_object_20260618_221705/
├── analysis/v3i_archaeological_object_yolov8n_640/
│   ├── metrics_summary.csv
│   ├── threshold_sweep.csv
│   ├── regional_validation_audit_conf_025.csv
│   └── v3i_archaeological_object_report.md
└── runs/archaeological_object_detection/
    ├── v3i_archaeological_object_yolov8n_640/
    │   ├── results.csv
    │   ├── results.png
    │   ├── BoxPR_curve.png
    │   ├── confusion_matrix.png
    │   └── weights/best.pt
    └── v3i_archaeological_object_yolov8n_640_val_best/

runs/v3i_yolov8n640_low_conf_val_review_outputs/
├── analysis/
│   ├── threshold_sweep.csv
│   ├── per_image_threshold_summary.csv
│   ├── gallery_conf_0p05.html
│   ├── gallery_conf_0p03.html
│   ├── gallery_conf_0p01.html
│   ├── gallery_conf_0p005.html
│   └── contact_sheets_conf_*/
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

## Текущее состояние проекта

На текущем этапе проект уже прошел путь от сырого YOLO bbox baseline к аккуратной experimental framework:

- собран и проаудирован исходный 5-class YOLO bbox dataset;
- проведены ablation-серии по модальностям, фильтрам, negative sampling и ручной чистке;
- сделан manual audit tool и clean dataset generation;
- проверены region-aware и curated validation splits;
- отдельно изучены `Li-only`, `Li + Ae`, `train Li+Ae -> val Li`, kurgan-only и merged `archaeological_object` постановки;
- проверены `YOLOv8n`, `YOLOv8s`, `YOLO26n`, `imgsz = 640/1024`, longer training и low-confidence inference.

Текущая инженерная оценка:

- как финальный detector YOLO пока недостаточно надежен: стандартный recall остается низким, а merged `archaeological_object` target добавляет внутриклассовую неоднородность;
- как proposal generator YOLO уже полезен: `YOLOv8n 640` при low confidence покрывает значительную часть GT и визуально находит много археологически правдоподобных candidates;
- `conf = 0.05` выглядит как лучший практический режим для ручной проверки и умеренного candidate generation;
- `conf = 0.01` полезен для aggressive mining, но требует сильного downstream-фильтра из-за большого FP потока;
- следующий качественный скачок, скорее всего, лежит не в новой YOLO-архитектуре, а в связке `YOLO proposals -> crop -> segmentation/refinement` и в доразметке/валидации сложных регионов.

Рекомендуемый следующий шаг:

```text
1. Зафиксировать YOLOv8n 640 + conf=0.05 как candidate proposal baseline.
2. Сформировать crop dataset из TP/FP/FN proposals.
3. Подключить segmentation/refinement stage для фильтрации candidates.
4. Отдельно разметить/проверить FP-heavy regions: 005_ЛУБНО, 006_МОСКОВИТЫ, 004_ДЕМИДОВКА, 011_РУНА, 013_БЕРВЕНЕЦ.
5. Держать conf=0.01 как mining mode для поиска потенциально пропущенных объектов, но не как production inference threshold.
```

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
