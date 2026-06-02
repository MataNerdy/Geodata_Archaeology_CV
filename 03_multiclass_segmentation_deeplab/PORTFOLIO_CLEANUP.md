# Portfolio Cleanup Plan

This document records the safe publication cleanup plan for `03_multiclass_segmentation_deeplab`. It does not delete or move files automatically.

## Current Inventory

| Area | Approx. size | Publication decision |
|---|---:|---|
| `runs/` | 4.3G | Keep locally or archive externally; publish only compact summaries and curated figures. |
| `assets/` | 49M | Keep curated README figures; archive exploratory galleries if the repository becomes too large. |
| `.venv/` | 12M | Local environment only. |
| `notebooks/` | 9.1M | Keep only notebooks with clear reproduction value. |
| `reports/` | 1.4M | Keep compact CSV/MD summaries and README figures. |
| `splits/` | 968K | Keep frozen Research Split artifacts. |
| `scripts/` | 688K | Keep reproducible training, evaluation and visualization workflows. |

## Keep In Git

| Path | Reason |
|---|---|
| `README.md` | Main research narrative and verified final results. |
| `requirements.txt` | Environment bootstrap. |
| `arch_datasets/` | Dataset loading, mask remapping and metadata filtering. |
| `models/` | DeepLabV3+ model factory. |
| `losses/` | CE, BCE and Dice combinations. |
| `utils/` | Metrics, split helpers, postprocessing and visualization helpers. |
| `scripts/` | Reproducible training, evaluation, sweep, visualization and audit workflows. |
| `configs/` | Versioned experiment recipes. |
| `splits/archaeology_5class_research_split_v1/` | Immutable region-aware benchmark protocol. |
| `reports/*.md` | Research summaries and failure analysis. |
| `reports/*.csv` | Compact tables used by README or audit notes. |
| `reports/figures/` | Curated figures referenced by README. |
| `assets/dataset/` | Dataset profiling figures used by README. |
| `assets/plots/` | Curated plots used by README. |
| `assets/predictions/` | Curated prediction figures used by README. |
| `assets/readme/` | README-specific hero and model evolution figures. |
| `assets/failures/` | Curated failure examples, if kept for extended review. |
| `notebooks/deeplab_kaggle_runner.ipynb` | Optional executable Kaggle entry point. |
| `notebooks/deeplab_old_baseline_reproduction.ipynb` | Optional reproduction notebook for legacy baseline audit. |

## Keep Locally Or Archive Externally

| Path pattern | Reason |
|---|---|
| `runs/**/*.pth`, `runs/**/*.pt`, `runs/**/*.ckpt` | Large checkpoints; publish through Releases or external storage only when needed. |
| `runs/**/history.csv` | Useful locally, noisy in a portfolio repository. |
| `runs/**/train_split.csv`, `runs/**/val_split.csv` | Run-local duplicates; canonical frozen CSV files live under `splits/`. |
| `runs/archaeology_5class_encoder_modality_ablation_raw_v1/` | Diagnostic ablation artifacts; compact results are already summarized in README. |
| `runs/different_seeds_experiments/` | Preserve locally; publish compact summaries and curated plots. |
| `runs/kaggle/` | Kaggle working area, raw archives and exploratory runs. |
| `runs/collect_models/` | Checkpoint collection workspace. |
| `.venv/` | Local environment. |
| `__pycache__/`, `.DS_Store` | Local generated files. |

## Checkpoints Found

Checkpoints are present mainly under ignored `runs/` subdirectories:

- raw encoder/modality ablation checkpoints;
- `resnet34_li_seed_*` checkpoints from the seed study;
- `resnet34_all_seed_*` checkpoints from the seed study;
- Kaggle and local collection copies.

These files should not be deleted during portfolio cleanup unless their storage location and reproducibility role are confirmed.

## Optional Notebooks

| Path | Recommendation |
|---|---|
| `notebooks/deeplab_kaggle_runner.ipynb` | Keep. |
| `notebooks/deeplab_old_baseline_reproduction.ipynb` | Keep as an audit/reproduction supplement. |
| `notebooks/all-class-baseline-competition-metrics.ipynb` | Archive after confirming no unique narrative value. |
| `notebooks/all-class-best-eval.ipynb` | Archive after confirming no unique narrative value. |
| `notebooks/deeplab_old_baseline.ipynb` | Archive after confirming the reproduction notebook supersedes it. |

## Images And Figures

README currently depends on these key figures:

- `assets/predictions/final_resnet34_all_seed_101.png`
- `assets/readme/model_evolution_examples.png`
- `assets/plots/postprocess_sweep_resnet34_all_seed_101.png`
- `reports/figures/failure_cases.png`
- dataset profiling figures under `assets/dataset/`

Unused or exploratory images are concentrated in:

- `assets/archive/`
- `assets/predictions/top20_best_final/`
- `assets/predictions/top5_best/`
- `assets/failures/top20_worst_final/`
- `assets/failures/top5_worst/`
- `runs/**/prediction_examples.png`
- `runs/**/confusion_matrix.png`

Do not delete them automatically. If the repository becomes too heavy, move non-README galleries into an external archive or keep only representative curated figures.

## Recommended Final Structure

```text
03_multiclass_segmentation_deeplab/
├── assets/
│   ├── dataset/
│   ├── plots/
│   ├── predictions/
│   ├── readme/
│   └── failures/
├── arch_datasets/
├── configs/
├── losses/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── splits/
│   └── archaeology_5class_research_split_v1/
├── utils/
├── README.md
├── PORTFOLIO_CLEANUP.md
└── requirements.txt
```

## Recommended Ignore Rules

The repository should ignore local environments, generated caches, heavy runs, checkpoints and archives:

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
runs/
*.pth
*.pt
*.ckpt
*.zip
```

Before publishing, ensure compact summaries needed by README are available in `reports/` or `assets/`, not only inside `runs/`.

## Recommended Archive Or Delete Candidates

These paths are candidates for archive or deletion after explicit confirmation:

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- `runs/kaggle/`
- `runs/collect_models/`
- large checkpoint files under `runs/`
- run-local duplicate `train_split.csv` / `val_split.csv`
- exploratory notebooks listed in the Optional Notebooks section
- non-README exploratory image galleries

No cleanup action has been applied to these paths by this document.
