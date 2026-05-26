# 02_unet_segmentation

Экспериментальный пайплайн для semantic segmentation курганов на патчах геоданных.

## Структура датасета

Ожидаемый датасет:

```text
../datasets/segmentation_dataset/
├── images/
│   └── 000000.npy
├── masks/
│   └── 000000.npy
└── metadata.csv
```

`metadata.csv` должен содержать как минимум:

- `sample_id` - идентификатор патча, совпадает с именами `.npy`;
- `region` - регион для region-aware split;
- `modality` - модальность, например `Li`, `Ae`, `SpOr`.

Маска обучается как 3-классовая:

- `0` - background;
- `1` - whole kurgan;
- `2` - damaged kurgan.

Если в исходных масках встречаются другие археологические классы, например `3`, `4`, `5`, loader маппит их в background `0`. Для этого эксперимента модель учится только на классах курганов.

Скрипты проверяют наличие `metadata.csv`, папок `images/` и `masks/`, соответствие `sample_id` файлам, непустой train/val split и наличие выбранных validation regions.

## Окружение

Команды ниже нужно запускать из Python-окружения, где установлены `numpy`, `pandas`, `torch` и `matplotlib`.

```bash
python -c "import numpy, pandas, torch, matplotlib; print('env ok')"
```

## Multiclass Experiments

Минимальный smoke test перед обучением:

```bash
cd 02_unet_segmentation

python train.py \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/smoke_test" \
  --epochs 2 \
  --batch-size 2 \
  --split custom_regions \
  --val-regions "042_ИЗБОРСК"
```

Первый честный baseline:

```bash
python train.py \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/baseline_all_modalities_ce_dice" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА" \
  --modalities Li Ae SpOr \
  --class-weights "0.2,1.0,3.0"
```

Полезные параметры:

```bash
python train.py \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/unet_kurgans_Li_only" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА" \
  --modalities Li \
  --class-weights "0.2,1.0,3.0"
```

Baseline с early stopping на 12 эпох без улучшения:

```bash
bash run_early_stopping_experiment.sh
```

Для `custom_regions` скрипт проверяет, что список `--val-regions` не пустой, все регионы есть в `metadata.csv`, а train/val split не оказался пустым. В начале запуска он печатает найденные validation regions и количество samples по `region/modality`.

## Binary Experiments

В binary режиме маска становится:

- `0` - background;
- `1` - any kurgan, то есть `mask > 0` после ремапа исходных классов в `0/1/2`.

Binary loss: `BCEWithLogitsLoss + DiceLossBinary`. Для него используются `--pos-weight`, `--bce-weight`, `--dice-weight`; `--class-weights` не нужен.

Пример Li-only binary:

```bash
python train.py \
  --task binary \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/binary_li_only" \
  --epochs 50 \
  --patience 12 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --pos-weight 2.0 \
  --bce-weight 1.0 \
  --dice-weight 1.0
```

Binary Li+Ae:

```bash
python train.py \
  --task binary \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/binary_li_ae_only" \
  --epochs 50 \
  --patience 12 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li,Ae \
  --pos-weight 2.0 \
  --bce-weight 1.0 \
  --dice-weight 1.0
```

Binary evaluation:

```bash
python evaluate.py \
  --task binary \
  --data-root "../datasets/segmentation_dataset" \
  --checkpoint "runs/binary_li_only/best_model.pth" \
  --out-dir "runs/binary_li_only" \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --pos-weight 2.0
```

## План Экспериментов

| Эксперимент | Модальности | Loss | Class weights | Цель |
|---|---|---|---|---|
| baseline_all | Li,Ae,SpOr | CE + Dice | 0.2,1.0,3.0 | Общая модель |
| image_128 | Li,Ae,SpOr | CE + Dice | 0.2,1.0,3.0 | Проверить размер input |

Binary-план:

| Эксперимент | Модальности | Loss | Pos weight | Цель |
|---|---|---|---|---|
| binary_li_only | Li | BCE + Dice | 2.0 | Найти любой курган на LiDAR |
| binary_li_ae_only | Li,Ae | BCE + Dice | 2.0 | Проверить связку LiDAR + аэрофото |
| binary_all_modalities | Li,Ae,SpOr | BCE + Dice | 2.0 | Общая binary модель |
| binary_li_no_dice | Li | BCE | 2.0 | Проверить влияние Dice |
| binary_li_pos_weight_2 | Li | BCE + Dice | 2.0 | Базовый вес foreground |
| binary_li_pos_weight_4 | Li | BCE + Dice | 4.0 | Усилить foreground |

## Что сохраняется

В `--out-dir` сохраняются:

- `best_model.pth` - лучший чекпойнт по `val_mean_fg_iou` для multiclass и по `val_fg_iou` для binary;
- `history.csv` - loss и метрики по эпохам отдельно для train/val;
- `config.json` - параметры запуска и краткое описание split;
- `train_split.csv`, `val_split.csv` - использованные выборки;
- `prediction_examples.png` - несколько примеров image / GT / prediction / overlay.

В `history.csv` есть общие метрики и срезы по модальностям, например:

- `train_loss`, `val_loss`;
- `train_fg_iou`, `val_fg_iou`;
- `train_mean_fg_iou`, `val_mean_fg_iou`;
- `val_iou_whole_kurgan`, `val_iou_damaged_kurgan`;
- `val_Li_fg_iou`, `val_Ae_fg_iou`, `val_SpOr_fg_iou`, если такие модальности есть в split.

Как читать метрики:

- multiclass: основная метрика `val_mean_fg_iou`, это среднее IoU по `whole_kurgan` и `damaged_kurgan`; дополнительно смотреть `val_iou_whole_kurgan` и `val_iou_damaged_kurgan`;
- binary: основная метрика `val_fg_iou`, это IoU foreground для любого кургана; дополнительно смотреть `val_fg_dice` и `val_pixel_accuracy`.

## Оценка чекпойнта

```bash
python evaluate.py \
  --data-root "../datasets/segmentation_dataset" \
  --checkpoint "runs/baseline_all_modalities_ce_dice/best_model.pth" \
  --out-dir "runs/baseline_all_modalities_ce_dice" \
  --split custom_regions \
  --val-regions "042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА" \
  --modalities Li Ae SpOr \
  --class-weights "0.2,1.0,3.0"
```

Результат печатается в stdout и дополнительно сохраняется в `evaluation.csv` и `evaluation.json`, если указан `--out-dir`.

## Визуализация предсказаний

```bash
python visualize_predictions.py \
  --data-root "../datasets/segmentation_dataset" \
  --checkpoint "runs/baseline_all_modalities_ce_dice/best_model.pth" \
  --output "runs/baseline_all_modalities_ce_dice/prediction_examples_eval.png" \
  --split custom_regions \
  --val-regions "042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА" \
  --modalities Li Ae SpOr
```

## Запуск на Kaggle

1. Включить Internet в настройках Kaggle notebook, чтобы notebook мог клонировать репозиторий через GitHub.
2. Загрузить `segmentation_dataset` как отдельный Kaggle Dataset с путем `/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset`.
3. Включить GPU в настройках notebook.
4. Запустить [notebooks/kurgans_unet_kaggle.ipynb](../notebooks/kurgans_unet_kaggle.ipynb).
5. После выполнения скачать `/kaggle/working/kurgans_runs.zip`.

Внутри Kaggle notebook репозиторий клонируется или обновляется в `/kaggle/working/Geodata_Archaeology_CV`, а эксперименты запускаются через:

```bash
bash run_kaggle_experiments.sh
```

Скрипт принимает переменные окружения:

- `REPO_URL` - URL репозитория для Kaggle notebook, по умолчанию `https://github.com/MataNerdy/Geodata_Archaeology_CV.git`;
- `BRANCH` - ветка репозитория для Kaggle notebook, по умолчанию `main`;
- `DATA_ROOT` - путь к датасету, по умолчанию `/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset`;
- `RUN_ROOT` - путь для результатов, по умолчанию `/kaggle/working/Geodata_Archaeology_CV/02_unet_segmentation/runs`;
- `PYTHON_BIN` - Python executable, по умолчанию `python`.

`run_kaggle_experiments.sh` печатает версии Python/PyTorch, проверяет CUDA, запускает smoke test, затем вторую серию binary-экспериментов. Логи сохраняются в `RUN_ROOT/logs`.

Kaggle-скрипт сейчас запускает вторую серию с early stopping `--patience 12`:

- `binary_li_only`
- `binary_li_ae_only`
- `binary_all_modalities`
- `binary_li_no_dice`
- `binary_li_pos_weight_2`
- `binary_li_pos_weight_4`

После каждого обучения запускается `evaluate.py`; все `evaluation.json` под `RUN_ROOT`, включая сохранённые результаты первой серии, собираются в `RUN_ROOT/experiments_summary.csv`.

## Следующие эксперименты

1. Сравнить несколько наборов validation regions для binary mode.
2. Проверить `image-size 128` для лучшего binary-направления.
3. Подобрать threshold для binary inference вместо фиксированного `0.5`.
