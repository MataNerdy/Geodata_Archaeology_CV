# Encoder / Modalities Diagnostic Ablation

This series is diagnostic only. The primary project result remains:

- protocol: `archaeology_5class_research_split_v1`
- model: DeepLabV3+ ResNet34
- modalities: all
- stage: Stage C
- weighted F1: `0.7457`

## Purpose

Compare encoder capacity and modality scope on the same region assignment:

1. ResNet34 Li
2. ResNet50 Li
3. ResNet34 all modalities
4. ResNet50 all modalities

## Raw Split

Materialize the separate raw split once:

```bash
python -u scripts/create_encoder_modality_ablation_split.py \
  --data-root ../datasets/segmentation_dataset
```

The command reuses the frozen train/val region assignment from
`archaeology_5class_research_split_v1`. It does not rerun region search and does
not apply metadata filtering. The split builder validates the expected
30-region validation holdout and stops if an older custom split is supplied.

## Run

```bash
python -u scripts/run_encoder_modality_ablation.py \
  --data-root ../datasets/segmentation_dataset \
  --num-workers 2
```

The runner trains and evaluates all four models, then writes:

- `runs/encoder_modality_ablation_summary.csv`
- `runs/encoder_modality_ablation_summary.md`
- `runs/encoder_modality_ablation_comparison_grid.png`

## Fixed Recipe

- task: `archaeology_5class`
- architecture: DeepLabV3+
- epochs: `50`
- batch size: `8`
- learning rate: `1e-3`
- image size: `256`
- loss: CE `0.7` + Dice `0.3`
- optimizer: Adam
- sampler: default
- checkpoint selection: validation `mean_fg_iou`
- metadata filtering: disabled

## Limitations

- There is no held-out test split. This series is validation-only.
- The series is intentionally separate from the publication benchmark.
