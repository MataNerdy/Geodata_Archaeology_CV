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

## Research Path and Results

Этот модуль развивался как продолжение `02_unet_segmentation`: сначала был проверен сильный binary UNet для курганов, затем DeepLabV3+ был перенесен на те же данные и постепенно расширен до полноценной 5-class archaeological segmentation.

### 1. Binary Kurgan Baseline

Первый контрольный вопрос: может ли DeepLabV3+ догнать лучший UNet на задаче `any kurgan vs background`?

| Model | Task | Modality | Input | Key setting | Metric |
|---|---|---|---:|---|---:|
| UNetSmall | binary kurgan | Li | 256 | BCE, threshold `0.60` | fg IoU `0.6789` |
| DeepLabV3+ ResNet50 | binary kurgan | Li | 256 | BCE/Dice experiments | fg IoU `0.6759` |

Вывод: DeepLab почти сравнялся с UNet на binary kurgan detection, но не дал явного преимущества. Поэтому дальнейшая ценность DeepLab проверялась не только на binary mask, а на разделении классов и object-level extraction.

### 2. Kurgan Multiclass: Whole vs Damaged

Следующий вопрос: сможет ли DeepLab лучше разделить `kurgany_tselye` и `kurgany_povrezhdennye`?

Первые `kurgan_multiclass` runs показали collapse в damaged class: foreground часто заливался как поврежденный курган. После class-weight sweep лучший reproducible config стал:

```text
DeepLabV3+ / ResNet50 / Li / class_weights = 0.2,3.0,1.0
```

| Run | whole IoU | damaged IoU | mean_fg_iou |
|---|---:|---:|---:|
| baseline ResNet50 | unstable / lower | damaged-dominant | 0.239 |
| weighted ResNet50 | 0.391 | 0.287 | 0.339 |

Вывод: class weights заметно улучшают balance между whole/damaged, но задача остается сложной: pixel segmentation alone плохо отражает качество поиска объектов.

### 3. Full 5-Class Segmentation

Затем pipeline был расширен до исходных классов `0..5`:

```text
0 background
1 kurgany_tselye
2 kurgany_povrezhdennye
3 gorodishcha
4 fortifikatsii
5 arkhitektury
```

Первые clean 5-class runs оказались слабыми по pixel IoU. Это привело к важному повороту: археологическая задача ближе к object extraction, чем к pixel-perfect segmentation. Поэтому в pipeline были возвращены notebook-style идеи:

- metadata filtering;
- object-level polygon extraction;
- connected component cleanup;
- competition-like weighted F1;
- confidence / postprocessing sweep.

### 4. Object-Aware Evaluation

Сравнение pixel metrics и object metrics показало, что они могут расходиться. На одном и том же семействе моделей более высокий pixel IoU не всегда означает лучший object-level score.

| Model / run | Split | Pixel mean_fg_iou | Object F1 | Weighted F1 |
|---|---|---:|---:|---:|
| clean competition ResNet50 Li | current Li validation | 0.1517 | 0.4000 | 0.3346 |
| reproduced old baseline ResNet34 | old-style split/filtering | 0.1244 | 0.5538 | 0.4488 |
| reproduced old baseline ResNet34 | current all-modality validation | 0.2864 | 0.6857 | 0.5251 |
| reproduced old baseline ResNet34 | current Li-only validation | 0.4012 | 0.7804 | 0.5186 |

Вывод: old baseline имеет не лучший flat pixel score, но заметно лучший object-level F1. Это подтверждает основную research-гипотезу проекта: для археологических объектов важна не только попиксельная маска, но и способность выделять корректные объекты с хорошим precision/recall balance.

### 5. Collected Model Evaluation

Для финального сравнения все собранные 5-class checkpoints были прогнаны на едином validation split через `scripts/evaluate_collected_models.py`. В итоговый portfolio narrative включены только reproducible project runs, без внешнего checkpoint как целевой модели.

Лучшие результаты среди project checkpoints:

| Model | Eval set | mean_fg_iou | Object F1 | Weighted F1 |
|---|---|---:|---:|---:|
| old baseline ResNet34 | all modalities | 0.2864 | 0.6857 | 0.5251 |
| old baseline ResNet34 | Li only | 0.4012 | 0.7804 | 0.5186 |
| ResNet34 Li competition | Li only | 0.1564 | 0.5361 | 0.4335 |
| ResNet50 Li | Li only | 0.1382 | 0.4122 | 0.4023 |
| ResNet34 all modalities | all modalities | 0.0737 | 0.5340 | 0.4172 |

Postprocessing sweep дополнительно показал, что слабые noisy-модели можно заметно улучшить object-aware cleanup:

| Model | Raw weighted F1 | Best postprocess weighted F1 | Best config |
|---|---:|---:|---|
| old baseline ResNet34, all | 0.5251 | 0.5438 | confidence `0.40`, min area `128`, no opening |
| old baseline ResNet34, Li | 0.5186 | 0.5249 | confidence `0.00`, min area `32`, no opening |
| ResNet50 Li competition | 0.3346 | 0.4985 | confidence `0.50`, min area `256`, no opening |
| ResNet34 Li | 0.3414 | 0.4963 | confidence `0.10`, min area `256`, no opening |

Итоговый вывод: лучшая воспроизводимая линия проекта для 5-class сейчас идет через old-style ResNet34 baseline + object-aware evaluation/postprocessing. DeepLab полезен не как простой replacement для UNet, а как часть pipeline для archaeological object extraction.

### Competition Weights

Object-level weighted F1 использует веса классов:

| Class | Weight |
|---|---:|
| kurgany_povrezhdennye | 27.8 |
| kurgany_tselye | 22.2 |
| gorodishcha | 16.7 |
| arkhitektury | 11.1 |
| fortifikatsii | 5.6 |

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

Итоговая цель этого блока: выбрать логичную финальную 5-class модель для research chain, сравнивая Li-only, all-modalities и old baseline на одной validation выборке.


## Frozen Research Split

Для финальной серии `archaeology_5class` split должен создаваться один раз и дальше только читаться из CSV. Это защищает от тихого подбора validation через повторный `n_trials=5000`.

Фиксированный порядок:

```text
raw metadata
↓
metadata filtering
↓
make_region_holdout_split
↓
save train_split.csv / val_split.csv
↓
all experiments use --split frozen
```

Создать split один раз:

```bash
python scripts/create_research_split.py \
  --data-root ../datasets/segmentation_dataset \
  --out-dir splits/archaeology_5class_research_split_v1
```

Скрипт использует notebook-style search:

```text
val_frac = 0.2
group_col = region
strat_cols = class_name,modality
min_val_per_class = 5
random_state = 42
n_trials = 5000
```

После создания split не пересчитывается. Для обучения:

```bash
python scripts/train.py \
  --config configs/archaeology_5class_research_split_v1.yaml
```

Или явно:

```bash
python scripts/train.py \
  --config configs/archaeology_5class_old_baseline.yaml \
  --split frozen \
  --train-split-csv splits/archaeology_5class_research_split_v1/train_split.csv \
  --val-split-csv splits/archaeology_5class_research_split_v1/val_split.csv
```

Для Li-only ablation используется тот же frozen split, но с фильтром модальности:

```bash
python scripts/train.py \
  --config configs/archaeology_5class_research_split_v1.yaml \
  --out-dir runs/multiclass/old_recipe_resnet34_li_research_split_v1 \
  --modalities Li
```

Важно: в `split=frozen` metadata filtering в train/evaluate выключается, потому что split CSV уже должен быть создан после filtering.


## Research Split v1 Benchmark

Следующий честный stage для `archaeology_5class` фиксирует один split и запускает только DeepLabV3+ ResNet34 old recipe. Новые архитектуры, ensemble, pseudo-labeling и подбор по test здесь не используются.

Серия экспериментов:

| Series | Experiment pattern | Modalities | Seeds | Purpose |
|---|---|---|---|---|
| A | `resnet34_all_seed_*` | `Li,Ae,SpOr` | `13,21,42,77,101` | основной all-modalities benchmark |
| B | `resnet34_li_seed_*` | `Li` | `13,21,42,77,101` | LiDAR-only ablation |
| C | best A/B + postprocessing sweep | выбранная best модель | validation best | confidence/min-area/opening tuning |
| D | sampler ablation | best modality setup | best seed | default vs weighted sampler |

Primary selection metric:

```text
weighted_competition_f1
```

Secondary metrics:

```text
object_f1
object_precision
object_recall
mean_fg_iou
per-class IoU
```

Запуск всей стадии после создания frozen split:

```bash
python scripts/run_research_split_v1.py \
  --data-root ../datasets/segmentation_dataset \
  --run-root runs/research_split_v1 \
  --train-split-csv splits/archaeology_5class_research_split_v1/train_split.csv \
  --val-split-csv splits/archaeology_5class_research_split_v1/val_split.csv \
  --run-training \
  --run-postprocess-sweep \
  --run-sampler-ablation
```

На Kaggle тот же stage запускается через:

```bash
RUN_MODE=research_split_v1 \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Если all-modalities серия уже сохранена, продолжить только `Li`-запуски можно без повторного обучения готовых моделей:

```bash
RUN_MODE=research_split_v1_li \
DATA_ROOT=/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset \
RUN_ROOT=/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs \
bash run_kaggle_experiments.sh
```

Resume-режим использует `num_workers=0`, запускает только `resnet34_li_seed_{13,21,42,77,101}` и пропускает уже завершенные runs. Postprocessing sweep и sampler ablation запускаются отдельным проходом после проверки обеих seed-серий.

Если split CSV еще не лежит в `SPLIT_DIR`, сначала создать его один раз:

```bash
python scripts/create_research_split.py \
  --data-root "$DATA_ROOT" \
  --out-dir splits/archaeology_5class_research_split_v1
```

Expected outputs:

```text
splits/archaeology_5class_research_split_v1/
├── train_split.csv
├── val_split.csv
├── split_config.json
└── split_stats.md

runs/research_split_v1/
├── resnet34_all_seed_13/
├── resnet34_all_seed_21/
├── resnet34_all_seed_42/
├── resnet34_all_seed_77/
├── resnet34_all_seed_101/
├── resnet34_li_seed_13/
├── resnet34_li_seed_21/
├── resnet34_li_seed_42/
├── resnet34_li_seed_77/
├── resnet34_li_seed_101/
├── research_split_v1_seed_summary.csv
├── research_split_v1_seed_summary.md
├── best_model_selection.md
├── postprocess_sweep/
└── sampler_ablation/
```

`test_split.csv` не создается автоматически. Если held-out test появится позже, он должен быть добавлен как отдельный protocol artifact, и model/postprocessing selection всё равно остается validation-only.

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
- run-local split CSV under `runs/**`; frozen research splits under `splits/` are protocol artifacts
- `history.csv`

Curated изображения для README можно хранить в `assets/readme/`. Сырые результаты экспериментов остаются в `runs/` и скачиваются из Kaggle архивом.

## Future Experiments

- Сравнить `resnet34`, `resnet50`, `efficientnet-b0`.
- Подобрать threshold для DeepLab binary и сравнить с UNet `0.6789`.
- Проверить, улучшает ли DeepLab разделение `whole/damaged`.
- Сравнить модальности `Li`, `Ae`, `SpOr` и их комбинации.
- Пересчитать competition-like weighted F1 для финальной 5-class модели.
