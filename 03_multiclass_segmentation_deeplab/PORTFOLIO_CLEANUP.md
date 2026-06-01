# Portfolio Cleanup Plan

This document proposes a clean publication structure. It does not delete or move files automatically.

## Keep In Git

| Path | Reason |
|---|---|
| `README.md` | Main portfolio narrative and verified results. |
| `requirements.txt` | Environment bootstrap. |
| `arch_datasets/` | Dataset loading, mask remapping and metadata filtering. |
| `models/` | DeepLabV3+ model factory. |
| `losses/` | CE, BCE and Dice combinations. |
| `utils/` | Metrics, splits, postprocessing and visualization helpers. |
| `scripts/` | Reproducible training, evaluation, sweep and audit workflows. |
| `configs/` | Versioned experiment recipes. |
| `splits/archaeology_5class_research_split_v1/` | Immutable benchmark protocol artifact. |
| `assets/` | Small curated figures used by README. |
| `notebooks/deeplab_kaggle_runner.ipynb` | Optional executable Kaggle entry point. |
| `notebooks/deeplab_old_baseline_reproduction.ipynb` | Optional reproduction notebook. |

## Keep Locally Or Archive Externally

| Path pattern | Reason |
|---|---|
| `runs/**/*.pth` | Large checkpoints; publish through Releases or external storage only when needed. |
| `runs/**/history.csv` | Useful locally, noisy in the portfolio repository. |
| `runs/**/train_split.csv`, `runs/**/val_split.csv` | Run-local duplicates; keep canonical frozen CSV under `splits/`. |
| `runs/kaggle/` | Raw Kaggle archives and exploratory runs. |
| `runs/collect_models/` | Checkpoint collection workspace. |
| `runs/different_seeds_experiments/` | Preserve locally; publish compact summaries and curated plots. |
| `.venv/` | Local environment. |
| `__pycache__/`, `.DS_Store` | Local generated files. |

## Optional Notebooks

| Path | Recommendation |
|---|---|
| `notebooks/deeplab_kaggle_runner.ipynb` | Keep. |
| `notebooks/deeplab_old_baseline_reproduction.ipynb` | Keep as an audit/reproduction supplement. |
| `notebooks/all-class-baseline-competition-metrics.ipynb` | Archive after confirming no unique narrative value. |
| `notebooks/all-class-best-eval.ipynb` | Archive after confirming no unique narrative value. |
| `notebooks/deeplab_old_baseline.ipynb` | Archive after confirming reproduction notebook supersedes it. |

## Recommended Final Structure

```text
03_multiclass_segmentation_deeplab/
├── assets/
│   ├── dataset/
│   ├── predictions/
│   ├── failures/
│   └── plots/
├── arch_datasets/
├── configs/
├── losses/
├── models/
├── notebooks/
├── scripts/
├── splits/
│   └── archaeology_5class_research_split_v1/
├── utils/
├── README.md
└── requirements.txt
```

## Recommended Ignore Additions

```gitignore
.venv/
runs/
*.zip
*.pth
*.pt
*.ckpt
__pycache__/
*.pyc
.DS_Store
```

If compact CSV summaries under `runs/` should be published, move selected summaries into `assets/` or a dedicated `reports/` directory before ignoring the whole `runs/` tree.
