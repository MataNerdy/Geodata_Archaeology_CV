# Binary Kurgan Segmentation with U-Net

## Overview

This module builds a semantic segmentation baseline for burial mounds on archaeological remote-sensing data.

The goal was to test whether a lightweight U-Net can solve a narrower version of the problem:

> Can burial mounds be segmented reliably as a single foreground class?

This module serves as the baseline stage before the more complex DeepLabV3+ study in `03_multiclass_segmentation_deeplab`.

![Binary LiDAR segmentation](assets/readme/hero_binary_shumgora_medium.png)

*Binary LiDAR segmentation: input image, ground truth, prediction, and overlay.*

## Research Question

Can a stable binary segmentation baseline be built for burial mounds on multi-modal archaeological data?

The experiments tested:

- modality effects: `Li`, `Ae`, `SpOr`;
- multiclass vs binary formulation;
- BCE vs Dice loss;
- threshold calibration;
- image size;
- hard negatives from other archaeological classes.

## Dataset and Task

The patch-based dataset was prepared in `01_geodata_to_cv`.

```text
datasets/segmentation_dataset/
├── images/
├── masks/
└── metadata.csv
```

Supported modalities:

| Modality | Description |
|---|---|
| `Li` | LiDAR-derived raster |
| `Ae` | aerial imagery |
| `SpOr` | satellite / orthophoto imagery |

In binary mode, both burial-mound classes are merged into one foreground class:

| Original class | Binary class |
|---|---|
| `kurgany_tselye` | foreground |
| `kurgany_povrezhdennye` | foreground |
| `background` | background |
| `gorodishcha` | background / hard negative |
| `fortifikatsii` | background / hard negative |
| `arkhitektury` | background / hard negative |

Other archaeological structures remain in the data as hard negatives because they can be visually similar to burial mounds.

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

The module supports:

- filtering by modality;
- binary and multiclass training modes;
- custom validation regions;
- BCE / Dice / BCE + Dice losses;
- post-training threshold sweep;
- prediction and failure-case visualization.

## Experiment Summary

The main experiment series tested which factors contributed most to the baseline quality.

| Experiment | Task | Modalities | Loss | Image size | Best metric |
|---|---|---|---|---:|---:|
| `baseline_all_modalities` | multiclass | `Li`, `Ae`, `SpOr` | CE + Dice | 256 | mean fg IoU = 0.137 |
| `li_only` | multiclass | `Li` | CE + Dice | 256 | mean fg IoU = 0.243 |
| `binary_li_only` | binary | `Li` | BCE + Dice | 256 | fg IoU = 0.647 |
| `binary_li_no_dice` | binary | `Li` | BCE | 256 | fg IoU = 0.665 |
| `binary_li_no_dice` + threshold sweep | binary | `Li` | BCE | 256 | **fg IoU = 0.6789** |
| `binary_li_512_no_dice` | binary | `Li` | BCE | 512 | fg IoU = 0.630 |

## Key Findings

- `Li` was the most informative modality for burial-mound segmentation.
- Binary formulation was more stable than multiclass formulation for the lightweight U-Net.
- BCE-only performed better than BCE + Dice in binary mode.
- Threshold calibration improved the result without retraining.
- Increasing image size from `256` to `512` reduced quality for `UNetSmall`.
- Other archaeological structures are important hard negatives.

Modality was not a minor configuration choice; it was one of the main drivers of quality:

| Experiment | mean fg IoU |
|---|---:|
| all modalities | 0.137 |
| Li only | **0.243** |
| Ae only | 0.057 |
| SpOr only | 0.051 |

This result motivated the later focus on LiDAR morphology in the baseline and the separate multimodality analysis in the DeepLabV3+ module.

## Best Result

The best result was obtained with LiDAR-only binary segmentation.

| Component | Value |
|---|---|
| Model | `UNetSmall` |
| Task | Binary kurgan segmentation |
| Modality | `Li` |
| Image size | `256` |
| Loss | BCE |
| Threshold | `0.60` |
| Foreground IoU | **0.6789** |

Threshold tuning improved the result without retraining:

```text
fg IoU: 0.6651 -> 0.6789
```

## Visual Results

### Binary LiDAR Predictions

![Binary predictions](assets/readme/binary_li_shumgora_examples.png)

Examples include good, medium, and failure cases on validation patches.

### Threshold Sweep

![Threshold sweep](assets/readme/threshold_sweep_binary_li_no_dice.png)

The best threshold was higher than the standard `0.5`:

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.50 | 0.708 | 0.909 |
| 0.60 | 0.747 | 0.882 |
| 0.75 | 0.803 | 0.797 |

The model tended to over-predict archaeological foreground. Raising the threshold reduced false positives and improved the final IoU.

### Failure Cases

![Failure cases](assets/readme/failure_cases_binary_li.png)

Typical errors involved noisy terrain, merged objects, tiny burial mounds, and hard negatives that resemble kurgans.

## Interpretation

This baseline showed that the difficulty was not only architectural. Quality depended strongly on the problem formulation:

- multiclass segmentation was unstable for the small U-Net;
- LiDAR provided the clearest morphology;
- visual modalities could introduce domain noise;
- loss function and threshold calibration had a measurable effect;
- hard negatives were necessary for realistic evaluation.

These observations led to the next stage of the project: a stronger DeepLabV3+ model and multiclass object-level evaluation.

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

## Role in the Full Project

This module is the baseline step in the full series:

```text
Geodata preprocessing
        ↓
Binary U-Net segmentation baseline
        ↓
Multiclass DeepLabV3+ research
        ↓
YOLO-ready detection dataset
```

The module established that LiDAR morphology contains a strong signal for archaeological segmentation and fixed the first segmentation baseline:

```text
UNetSmall + Li + binary segmentation + BCE + threshold 0.60
fg IoU = 0.6789
```

The next stage is `03_multiclass_segmentation_deeplab`, where the task is extended to five foreground classes, a region-aware benchmark split, and object-level evaluation.
