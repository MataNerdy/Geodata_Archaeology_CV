# Binary Kurgan Segmentation with U-Net

## Обзор

Этот модуль посвящен построению базовой модели семантической сегментации курганов на археологических геоданных.

Цель этапа - проверить, насколько хорошо легкая U-Net архитектура решает более узкую задачу:

> можно ли надежно выделять курганы как единый foreground-класс?

Модуль стал baseline-этапом перед переходом к более сложному исследованию DeepLabV3+ в `03_multiclass_segmentation_deeplab`.

![Binary LiDAR segmentation](assets/readme/hero_binary_shumgora_medium.png)

*Binary LiDAR segmentation: input image, ground truth, prediction and overlay.*

## Исследовательский вопрос

Можно ли построить устойчивый binary segmentation baseline для курганов на многомодальных археологических данных?

В этом модуле проверялись:

- влияние модальности: `Li`, `Ae`, `SpOr`;
- multiclass vs binary formulation;
- BCE vs Dice loss;
- threshold calibration;
- влияние image size;
- роль hard negatives из других археологических классов.

## Dataset And Task

Исходный patch-based dataset был подготовлен в модуле `01_geodata_to_cv`.

```text
datasets/segmentation_dataset/
├── images/
├── masks/
└── metadata.csv
```

Поддерживаемые модальности:

| Модальность | Описание |
|---|---|
| `Li` | LiDAR-derived raster |
| `Ae` | aerial imagery |
| `SpOr` | satellite / orthophoto imagery |

В binary mode все курганы объединяются в один foreground-класс:

| Original class | Binary class |
|---|---|
| `kurgany_tselye` | foreground |
| `kurgany_povrezhdennye` | foreground |
| `background` | background |
| `gorodishcha` | background / hard negative |
| `fortifikatsii` | background / hard negative |
| `arkhitektury` | background / hard negative |

Это важно: другие археологические объекты остаются в данных как hard negatives, потому что визуально они могут быть похожи на курганы.

## Pipeline

```text
02_unet_segmentation/
├── datasets/        # dataset loading and binary/multiclass mapping
├── models/          # UNetSmall
├── losses/          # BCE, Dice and combined losses
├── scripts/         # training, evaluation and threshold sweep
├── utils/           # metrics and visualization
├── assets/readme/   # curated README figures
├── runs/            # local experiment outputs
└── notebooks/       # exploratory notebooks
```

Модуль поддерживает:

- filtering by modality;
- binary and multiclass training modes;
- custom validation regions;
- BCE / Dice / BCE + Dice losses;
- threshold sweep after training;
- visualization of predictions and failure cases.

## Experiment Summary

Основная серия экспериментов проверяла, что именно дает наибольший вклад в качество baseline.

| Experiment | Task | Modalities | Loss | Image size | Best metric |
|---|---|---|---|---:|---:|
| `baseline_all_modalities` | multiclass | `Li`, `Ae`, `SpOr` | CE + Dice | 256 | mean fg IoU = 0.137 |
| `li_only` | multiclass | `Li` | CE + Dice | 256 | mean fg IoU = 0.243 |
| `binary_li_only` | binary | `Li` | BCE + Dice | 256 | fg IoU = 0.647 |
| `binary_li_no_dice` | binary | `Li` | BCE | 256 | fg IoU = 0.665 |
| `binary_li_no_dice` + threshold sweep | binary | `Li` | BCE | 256 | **fg IoU = 0.6789** |
| `binary_li_512_no_dice` | binary | `Li` | BCE | 512 | fg IoU = 0.630 |

## Key Findings

- `Li` оказался самой информативной модальностью для выделения курганов.
- Binary formulation заметно стабильнее multiclass formulation для легкой U-Net.
- BCE-only оказался лучше BCE + Dice в binary mode.
- Threshold calibration улучшила результат без повторного обучения модели.
- Увеличение image size с `256` до `512` ухудшило качество для `UNetSmall`.
- Другие археологические структуры работают как важные hard negatives.

Модальность оказалась не второстепенной настройкой, а главным источником качества:

| Experiment | mean fg IoU |
|---|---:|
| all modalities | 0.137 |
| Li only | **0.243** |
| Ae only | 0.057 |
| SpOr only | 0.051 |

Этот результат стал одним из оснований для дальнейшего фокуса на LiDAR morphology в baseline и для отдельной проверки мультимодальности в DeepLabV3+.

## Best Result

Лучший результат был получен для LiDAR-only binary segmentation.

| Component | Value |
|---|---|
| Model | `UNetSmall` |
| Task | Binary kurgan segmentation |
| Modality | `Li` |
| Image size | `256` |
| Loss | BCE |
| Threshold | `0.60` |
| Foreground IoU | **0.6789** |

Threshold tuning дал прирост без retraining:

```text
fg IoU: 0.6651 -> 0.6789
```

## Visual Results

### Binary LiDAR Predictions

![Binary predictions](assets/readme/binary_li_shumgora_examples.png)

Примеры показывают good, medium и failure cases на validation patches.

### Threshold Sweep

![Threshold sweep](assets/readme/threshold_sweep_binary_li_no_dice.png)

Оптимальный threshold оказался выше стандартного `0.5`:

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.50 | 0.708 | 0.909 |
| 0.60 | 0.747 | 0.882 |
| 0.75 | 0.803 | 0.797 |

Это показывает, что модель склонна пере-предсказывать археологический foreground: повышение threshold уменьшает false positives и улучшает итоговый IoU.

### Failure Cases

![Failure cases](assets/readme/failure_cases_binary_li.png)

Типичные ошибки связаны с noisy terrain, merged objects, tiny kurgans и hard negatives, визуально похожими на курганы.

## Interpretation

Этот baseline показал, что главная сложность задачи не только в архитектуре модели.

Качество сильно зависит от постановки:

- multiclass segmentation на маленькой U-Net нестабильна;
- LiDAR дает наиболее читаемую морфологию;
- визуальные модальности могут добавлять domain noise;
- loss function и threshold calibration существенно влияют на результат;
- hard negatives нужны для реалистичной оценки.

Именно эти выводы определили следующий шаг проекта: перейти к более сильной архитектуре DeepLabV3+ и полноценной multiclass object-level evaluation.

## Reproducibility

### Training

```bash
python scripts/train.py \
  --task binary \
  --data-root "../datasets/segmentation_dataset" \
  --out-dir "runs/binary_li_no_dice" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --binary-positive-classes "1,2" \
  --dice-weight 0.0
```

### Threshold Sweep

```bash
python scripts/threshold_sweep.py \
  --data-root "../datasets/segmentation_dataset" \
  --checkpoint "runs/binary_li_no_dice/best_model.pth" \
  --out-dir "runs/binary_li_no_dice" \
  --task binary \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --binary-positive-classes "1,2"
```

## Role In The Full Project

Этот модуль является baseline-ступенью всей серии:

```text
Geodata preprocessing
        ↓
Binary U-Net segmentation baseline
        ↓
Multiclass DeepLabV3+ research
        ↓
YOLO-ready detection dataset
```

Главный вклад модуля — доказать, что LiDAR morphology действительно содержит сильный сигнал для археологической сегментации, а также зафиксировать baseline:

```text
UNetSmall + Li + binary segmentation + BCE + threshold 0.60
fg IoU = 0.6789
```

Дальнейшее развитие этой линии выполнено в `03_multiclass_segmentation_deeplab`, где задача расширена до пяти foreground-классов, region-aware benchmark split и object-level evaluation.
