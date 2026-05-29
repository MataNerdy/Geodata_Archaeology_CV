# DeepLab Segmentation для археологических геоданных

Третья часть проекта `Geodata_Archaeology_CV`: чистый research/portfolio модуль для DeepLab-сегментации археологических объектов. Код вынесен из exploratory notebook `all-class-best-eval.ipynb` в воспроизводимые скрипты.

## Situation

В `02_unet_segmentation` уже собран сильный UNet baseline для сегментации курганов:

| Model | Task | Modality | Input | Loss | Threshold | fg IoU |
|---|---|---|---:|---|---:|---:|
| UNetSmall | binary kurgan | Li | 256 | BCE | 0.60 | 0.6789 |

DeepLabV3+ нужен как следующий research-шаг: проверить, помогает ли более контекстная архитектура лучше отделять damaged/whole kurgans и переносится ли подход на все археологические классы.

## Task

Ожидаемая структура датасета:

```text
segmentation_dataset/
├── metadata.csv
├── images/
│   └── 000001.npy
└── masks/
    └── 000001.npy
```

Исходные классы маски:

| ID | Class |
|---:|---|
| 0 | background |
| 1 | kurgany_tselye |
| 2 | kurgany_povrezhdennye |
| 3 | gorodishcha |
| 4 | fortifikatsii |
| 5 | arkhitektury |

Поддерживаемые задачи:

| Task | Логика маски | Цель |
|---|---|---|
| `binary_kurgan` | `mask = np.isin(mask, [1, 2])` | Любой курган vs background |
| `kurgan_multiclass` | оставить `1/2`, классы `3/4/5 -> 0` | Разделить целые и поврежденные курганы |
| `archaeology_5class` | оставить исходные `0..5` | Полная археологическая сегментация |
| `all_classes` | alias для `archaeology_5class` | Обратная совместимость |

В `binary_kurgan` классы `3/4/5` считаются background / hard negatives.

## Action

Структура модуля:

```text
03_multiclass_segmentation_deeplab/
├── README.md
├── requirements.txt
├── configs/
├── arch_datasets/
├── models/
├── losses/
├── utils/
├── scripts/
├── notebooks/
├── assets/readme/
└── runs/
```

Модель: DeepLabV3+ из `segmentation_models_pytorch`.

Поддерживаемые encoders:

- `resnet34`
- `resnet50`
- `efficientnet-b0`

Все эксперименты используют `in_channels=1`, потому что Li/Ae/SpOr patches хранятся как одноканальные `.npy`.

## External Reference Checkpoint из Notebook

В notebook был зафиксирован внешний reference checkpoint `deeplab_5class_43_best.pth`. Это не reproduced old baseline, а отдельный reference result для ориентира.

Reported per-class IoU для external checkpoint:

| Class | IoU |
|---|---:|
| background | 0.949 |
| kurgany_tselye | 0.714 |
| kurgany_povrezhdennye | 0.767 |
| gorodishcha | 0.824 |
| fortifikatsii | 0.594 |
| arkhitektury | 0.671 |

Competition-like weighted F1 для external checkpoint: `0.7411`.

Веса классов для polygon F1:

| Class | Weight |
|---|---:|
| kurgany_povrezhdennye | 27.8 |
| kurgany_tselye | 22.2 |
| gorodishcha | 16.7 |
| arkhitektury | 11.1 |
| fortifikatsii | 5.6 |

## Reproduced Old Baseline

`all_class_baseline.ipynb` не воспроизводит `0.7411`, но дает важный old baseline `archaeology_5class_old_baseline_resnet34`.

Конфиг old baseline:

| Parameter | Value |
|---|---|
| task | `archaeology_5class` |
| encoder | `resnet34` |
| encoder weights | `None` |
| optimizer | Adam |
| lr | `1e-4` |
| batch size | `16` |
| epochs / patience | `80 / 12` |
| scheduler | ReduceLROnPlateau, factor `0.5`, patience `5` |
| grad clip | `1.0` |
| class weights | `0.2,1.0,1.0,1.4,1.8,1.8` |
| loss weights | CE `0.7` + Dice `0.3` |
| weighted sampler | off |
| metadata filtering | on |
| modalities | `Li,Ae,SpOr,Or` |
| val regions | `005_ЛУБНО,012_ЛИХУША,008_СЕЛЯНЕ,011_РУНА,014_СТРЕКАЛОВКА,016_ЗОЛОТАРЕВКА,044_ГОЧЕВО` |

Old baseline metrics:

| Metric | Value |
|---|---:|
| best val_mean_fg_iou | 0.1565 at epoch 36 |
| pixel mean_fg_iou | 0.1244 |
| object F1 | 0.5538 |
| weighted competition F1 | 0.4488 |

Comparison:

| Model | Split | Pixel mean_fg_iou | Object F1 | Weighted F1 |
|---|---|---:|---:|---:|
| current clean competition resnet50 Li | competition custom regions, Li only | 0.1517 | 0.4000 | 0.3346 |
| old_baseline_resnet34 | old baseline custom regions + metadata filtering | 0.1244 | 0.5538 | 0.4488 |
| external `deeplab_5class_43_best` checkpoint | external notebook reference split | high per-class IoU | not reported separately | 0.7411 |

Old baseline has worse pixel IoU than the clean competition ResNet50 run, but better object-level F1. This supports the main archaeology-aware finding: object metrics are highly sensitive to split, postprocessing and precision/recall balance, and can tell a different story than flat pixel segmentation.

Artifacts:

```text
runs/multiclass/archaeology_5class_old_baseline_resnet34/
├── summary.json
├── evaluation.csv
├── evaluation_object.csv
└── competition_metric.csv

runs/multiclass/archaeology_5class_old_baseline_comparison.csv
```

## Result

Первый порядок экспериментов:

1. `binary_kurgan` DeepLab Li only
2. threshold sweep
3. сравнение с UNet `fg_iou=0.6789`
4. `kurgan_multiclass` Li only
5. `archaeology_5class` DeepLab

Smoke test:

```bash
cd 03_multiclass_segmentation_deeplab

python scripts/train.py \
  --config configs/binary_kurgan.yaml \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/smoke_test \
  --epochs 2 \
  --batch-size 2 \
  --save-samples 2
```

Первый честный запуск:

```bash
python scripts/train.py \
  --config configs/binary_kurgan.yaml \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/binary/deeplab_binary_li
```

Оценка:

```bash
python scripts/evaluate.py \
  --checkpoint runs/binary/deeplab_binary_li/best_model.pth \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/binary/deeplab_binary_li
```

Threshold sweep:

```bash
python scripts/threshold_sweep.py \
  --checkpoint runs/binary/deeplab_binary_li/best_model.pth \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/binary/deeplab_binary_li
```

Визуализация:

```bash
python scripts/visualize_predictions.py \
  --checkpoint runs/binary/deeplab_binary_li/best_model.pth \
  --data-root ../datasets/segmentation_dataset \
  --output runs/binary/deeplab_binary_li/prediction_examples.png \
  --threshold 0.60
```

Kurgan multiclass:

```bash
python scripts/train.py \
  --config configs/kurgan_multiclass.yaml \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/multiclass/deeplab_kurgan_multiclass_li
```

## Multiclass Class-Weight Sweep

В первых `kurgan_multiclass` экспериментах DeepLab показывал перекос в сторону класса `2 kurgany_povrezhdennye`: foreground часто заливался как damaged, а качество класса `1 kurgany_tselye` было ниже и нестабильнее. Для этого добавлен отдельный Kaggle режим подбора class weights для задачи `0/1/2`.

Цель sweep не только максимизировать `mean_fg_iou`, но и найти более сбалансированную модель:

- повысить `iou_kurgany_tselye`;
- не обвалить `iou_kurgany_povrezhdennye`;
- уменьшить collapse foreground в damaged class.

Запуск на Kaggle:

```bash
cd /kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab

RUN_MODE=multiclass_weight_sweep \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Эксперименты:

| Experiment | Encoder | Class weights |
|---|---|---|
| `kurgan_multiclass_li_resnet50_w_02_2_1` | resnet50 | `0.2,2.0,1.0` |
| `kurgan_multiclass_li_resnet50_w_02_3_1` | resnet50 | `0.2,3.0,1.0` |
| `kurgan_multiclass_li_resnet50_w_01_3_1` | resnet50 | `0.1,3.0,1.0` |
| `kurgan_multiclass_li_resnet50_w_02_2_08` | resnet50 | `0.2,2.0,0.8` |
| `kurgan_multiclass_li_resnet34_w_02_3_1` | resnet34 | `0.2,3.0,1.0` |

Summary сохраняется в:

```text
runs/multiclass_weight_sweep_summary.csv
```

Ключевые колонки:

- `mean_fg_iou`
- `iou_kurgany_tselye`
- `iou_kurgany_povrezhdennye`
- `dice_kurgany_tselye`
- `dice_kurgany_povrezhdennye`
- `pixel_accuracy`
- `best_epoch`

## Full 5-Class DeepLab Pipeline

Следующий этап после binary и `kurgan_multiclass`: настоящая archaeological segmentation на всех исходных классах `0..5`.

Research questions:

1. Помогает ли DeepLabV3+ для полноценной multiclass archaeological segmentation?
2. Сохраняется ли преимущество LiDAR?
3. Какие классы самые сложные?
4. Улучшается ли separation между kurgans, gorodishcha и fortifikatsii?
5. Дает ли ResNet50 meaningful gain относительно ResNet34?

Стартовые class weights:

```text
0.1,3.0,1.5,1.0,1.0,1.0
```

Интерпретация:

- background сильно занижен;
- whole kurgan boosted;
- damaged moderate;
- other archaeology neutral.

Запуск всей серии на Kaggle:

```bash
cd /kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab

RUN_MODE=archaeology_5class \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Эксперименты:

| Experiment | Encoder | Modalities |
|---|---|---|
| `archaeology_5class_resnet34_li` | resnet34 | Li |
| `archaeology_5class_resnet50_li` | resnet50 | Li |
| `archaeology_5class_resnet34_all_modalities` | resnet34 | Li,Ae,SpOr/other present modalities |
| `archaeology_5class_resnet50_all_modalities` | resnet50 | Li,Ae,SpOr/other present modalities |

Summary сохраняется в:

```text
runs/archaeology_5class_summary.csv
```

Ключевые колонки:

- `mean_fg_iou`
- `iou_kurgany_tselye`
- `iou_kurgany_povrezhdennye`
- `iou_gorodishcha`
- `iou_fortifikatsii`
- `iou_arkhitektury`
- `pixel_accuracy`
- `best_epoch`

Comparison grid:

```text
runs/multiclass/archaeology_5class_comparison.png
```

## Archaeology-Aware Competition Pipeline

Raw pixel IoU treats the task as flat semantic segmentation: every pixel mismatch is penalized equally. The original notebook behaved more like archaeological object extraction: it filtered noisy patches, oversampled rare object classes, polygonized masks, and matched objects by polygon IoU or centroid hits.

This pipeline brings those notebook ideas back into the reproducible scripts:

- `WeightedRandomSampler` for rare archaeology classes;
- metadata filtering by crop size, object count, border touching, allowed classes and foreground pixels;
- connected-components postprocessing;
- polygon extraction;
- object-level precision/recall/F1;
- competition-like weighted F1 with class weights:
  - `kurgany_povrezhdennye`: `27.8`
  - `kurgany_tselye`: `22.2`
  - `gorodishcha`: `16.7`
  - `arkhitektury`: `11.1`
  - `fortifikatsii`: `5.6`

Run:

```bash
cd /kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab

RUN_MODE=archaeology_5class_competition \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Experiments:

| Experiment | Encoder | Modality | Evaluation |
|---|---|---|---|
| `archaeology_5class_resnet50_li_competition` | resnet50 | Li | pixel + object |
| `archaeology_5class_resnet34_li_competition` | resnet34 | Li | pixel + object |

Each run saves:

- `evaluation_pixel.json`
- `evaluation_object.json`
- `competition_metric.json`
- `polygons_preview.png`
- `prediction_examples.png`
- `matched_objects_visualization.png`

Summary:

```text
runs/competition_summary.csv
```

Columns:

- `experiment`
- `encoder`
- `mean_fg_iou`
- `object_precision`
- `object_recall`
- `object_f1`
- `weighted_competition_f1`

The core hypothesis is that raw pixel IoU can severely underestimate archaeological segmentation quality, while object-level evaluation is closer to the real target: finding and separating archaeological objects.

Single 5-class run:

```bash
python scripts/train.py \
  --config configs/all_5_classes.yaml \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/multiclass/deeplab_all_5_classes
```

Competition-like metric:

```bash
python scripts/build_competition_geojson.py \
  --checkpoint runs/multiclass/deeplab_all_5_classes/best_model.pth \
  --data-root ../datasets/segmentation_dataset \
  --out-dir runs/multiclass/deeplab_all_5_classes

python scripts/evaluate_competition_metric.py \
  --pred-geojson runs/multiclass/deeplab_all_5_classes/predictions_geojson.json \
  --gt-geojson runs/multiclass/deeplab_all_5_classes/ground_truth_geojson.json \
  --out-dir runs/multiclass/deeplab_all_5_classes
```

## Collected 5-Class Model Evaluation

Старые и новые 5-class checkpoints можно собрать в:

```text
runs/collect_models/
├── li/
└── all/
```

`li/` содержит модели, обученные только на LiDAR. `all/` содержит модели, обученные на всех доступных модальностях. Для честного сравнения скрипт прогоняет все checkpoints на едином validation split проекта и сохраняет pixel metrics, object/competition metrics, prediction grids и confidence/postprocessing sweep.

Primary metric для 5-class archaeology pipeline: `weighted_competition_f1`. Pixel IoU остается полезной диагностикой, но object F1 и weighted competition F1 могут расходиться, потому что archaeological task ближе к object extraction, чем к flat pixel-perfect segmentation.

Локальный запуск без обучения:

```bash
python scripts/evaluate_collected_models.py \
  --data-root ../datasets/segmentation_dataset \
  --models-root runs/collect_models \
  --out-dir runs/collect_models_eval \
  --task archaeology_5class \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --object-iou-threshold 0.3 \
  --min-area 8 \
  --eval-all-models-on-li-too

python scripts/collected_models_postprocess_sweep.py \
  --data-root ../datasets/segmentation_dataset \
  --models-root runs/collect_models \
  --eval-root runs/collect_models_eval \
  --task archaeology_5class \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --object-iou-threshold 0.3 \
  --eval-all-models-on-li-too
```

Kaggle run mode:

```bash
RUN_MODE=collect_models_eval \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Outputs:

```text
runs/collect_models_eval/
├── collected_models_summary.csv
├── skipped_models.csv
├── top_models.md
├── model_comparison_grid.png
└── <model_name>/
    ├── evaluation.csv
    ├── evaluation.json
    ├── evaluation_object.csv
    ├── competition_metric.csv
    ├── prediction_examples.png
    ├── postprocess_sweep.csv
    ├── postprocess_sweep.json
    └── postprocess_sweep.png
```

Итоговая цель этого блока: выбрать логичную финальную 5-class модель для research chain, сравнивая Li-only, all-modalities, old baseline и external checkpoint на одной validation выборке.

## Kaggle

1. Подключить `segmentation_dataset` как Kaggle Dataset.
2. Включить GPU.
3. Открыть `notebooks/deeplab_kaggle_runner.ipynb`.
4. При необходимости задать переменные:

```bash
REPO_URL=https://github.com/MataNerdy/Geodata_Archaeology_CV.git
BRANCH=main
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs
EXPERIMENT_CONFIG=configs/binary_kurgan.yaml
```

5. Для минимальной серии экспериментов можно запустить:

```bash
cd /kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab

DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

6. Запустить notebook и скачать `deeplab_runs.zip`.

## Артефакты

Training сохраняет:

- `best_model.pth`
- `history.csv`
- `config_used.yaml`
- `train_split.csv`
- `val_split.csv`
- `prediction_examples.png`

Evaluation сохраняет:

- `evaluation.json`
- `evaluation.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png` для multiclass/all-class задач

Threshold sweep сохраняет:

- `threshold_sweep.csv`
- `threshold_sweep.json`
- `threshold_sweep.png`

## Repository Hygiene

По умолчанию не трекаются:

- checkpoints в `runs/**/*.pth`
- logs
- smoke tests
- split CSV
- `history.csv`

Curated изображения для README можно хранить в `assets/readme/`. Сырые результаты экспериментов остаются в `runs/` и скачиваются из Kaggle архивом.

## Future Experiments

- Сравнить `resnet34`, `resnet50`, `efficientnet-b0`.
- Подобрать threshold для DeepLab binary и сравнить с UNet `0.6789`.
- Проверить, улучшает ли DeepLab разделение `whole/damaged`.
- Сравнить модальности `Li`, `Ae`, `SpOr` и их комбинации.
- Пересчитать competition-like weighted F1 для финальной 5-class модели.
