# Visual Assets Checklist

Curated README visuals belong in `assets/`. Raw figures remain under `runs/`.

| Visual | Status | Suggested path | Notes |
|---|---|---|---|
| Dataset examples | available | `assets/dataset/dataset_examples_collage.png` | One class-by-view collage plus source triplets in `assets/dataset/examples/`. `arkhitektury` uses `Or` because no Li sample exists. |
| Modality examples | available | `assets/dataset/modality_comparison.png` | Regional `Li`, `Ae`, `SpOr` comparison. Patches are not guaranteed pixel-aligned. |
| Class imbalance | available | `assets/dataset/class_imbalance.png` | Primary sample class distribution in the curated class palette. |
| Raw split results | available | `assets/plots/raw_ablation_resnet34_all_confusion_matrix.png` | Curated confusion matrix from raw ResNet34 all-modalities run. |
| Research split diagram | available in README | Mermaid diagram | Export to PNG later only if needed for external presentations. |
| Seed comparison plot | TODO | `assets/plots/seed_comparison.png` | Plot weighted F1 for Li and all-modalities seeds. |
| Postprocessing comparison | available | `assets/plots/postprocess_sweep_resnet34_all_seed_101.png` | Stage C validation sweep for the final model. |
| Best predictions | available | `assets/predictions/final_resnet34_all_seed_101.png` | Curated prediction grid from selected checkpoint. |
| Ranked best predictions | available | `assets/predictions/top5_best/` | Individual four-panel examples plus manifest. |
| Before/after postprocessing | available | `assets/postprocessing_examples/top10_before_after.png` | GT, raw checkpoint and Stage C comparison. |
| Failure cases | available | `assets/failures/final_failure_cases.png` | Diverse low-scoring validation patches. |
| Ranked failure cases | available | `assets/failures/top5_worst/` | Individual four-panel examples plus manifest. |
| Metric comparison tables | available in README | Markdown tables | Keep tables close to each research phase. |

## Curation Rule

Do not commit complete Kaggle archives, checkpoints or all intermediate images. Select a small set of readable figures that directly support the research narrative.
