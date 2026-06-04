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
| v4 | ≈ 0.198 | ≈ 0.478 | ≈ 0.195 | ≈ 0.246 | ≈ 0.151 | Наиболее сбалансированный вариант; damaged-class AP вырос. |

### Threshold tuning

| Version | Confidence | mAP50 | Recall | AP `povrezhdennye` | Вывод |
|---|---:|---:|---:|---:|---|
| v5 | 0.10 | ≈ 0.187 | ≈ 0.195 | ≈ 0.130 | Recall не вырос относительно v4. |
| v5 | 0.15 | ≈ 0.178 | ≈ 0.195 | ≈ 0.111 | Повышение threshold ухудшило mAP50 и damaged-class AP. |

Главный вывод серии: bottleneck находится не в confidence threshold, а в данных, разметке и сложности damaged-object morphology.

Сохраненные локальные YOLO artifacts находятся в `runs/runs_2` и `runs/runs_4`.

Лучшие строки из локальных `results.csv`:

| Run | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `runs_2` | 51 | 0.43985 | 0.18478 | 0.18323 | 0.12095 |
| `runs_4` | 52 | 0.54479 | 0.18838 | 0.22443 | 0.12145 |

Эти значения могут отличаться от ручных summary-метрик, потому что в разных местах фиксировались best epoch, final epoch или отдельная threshold-specific validation.

## Результаты

Наиболее полезной текущей версией является v4:

```text
YOLOv8s + cleaned balanced Li/Ae kurgan dataset
```

Ключевой результат:

```text
mAP50 ≈ 0.198
precision ≈ 0.478
recall ≈ 0.195
AP kurgany_povrezhdennye ≈ 0.151
```

По сравнению с v2, качество damaged-class detection улучшилось:

```text
kurgany_povrezhdennye AP: ~0.07-0.08 -> ~0.15
```

При этом задача остается recall-limited: модель лучше находит хорошо выраженные целые курганы и часто отправляет поврежденные объекты в background.

## Key Findings

- Основной прирост качества достигнут не архитектурой, а переработкой dataset.
- `Li` дает более сильный структурный сигнал, чем aerial imagery.
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

runs/runs_4/
├── results.png
├── BoxF1_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
└── val_batch*_labels/pred.jpg
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

Существующие script prototypes:

```bash
python skripts/build_yolo_dataset_bbox.py
python skripts/make_dataset_v2_kurgans_li_ae.py
streamlit run skripts/visualize_yolo_labels.py
```

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

Перед полноценным воспроизведением нужно вынести hard-coded paths в configs:

```text
/content/dataset/...
/content/drive/...
/Volumes/Lexar/Датасет/
```

## Repository Structure

Текущая целевая структура модуля:

```text
04_detection_yolo/
├── README.md
├── requirements.txt
├── configs/
├── src/
├── app/
├── notebooks/
│   └── geo_li_ae_kurgan_detection.ipynb
├── reports/
│   ├── experiments.md
│   └── figures/
└── data_sample/
```

Сейчас часть кода еще находится в experimental folder `skripts/`. Следующий refactoring step - разделить reusable source code, command-line scripts и Streamlit app.

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
