# Visual Assets Checklist

Curated README visuals belong in `assets/`. Raw figures remain under `runs/`.

| Visual | Status | Suggested path | Notes |
|---|---|---|---|
| Dataset examples | TODO | `assets/dataset/dataset_examples.png` | Add representative image/mask pairs for each class. |
| Modality examples | TODO | `assets/dataset/modality_examples.png` | Show the same or similar region in `Li`, `Ae`, `SpOr`, `Or`. |
| Raw split results | available | `assets/plots/raw_ablation_resnet34_all_confusion_matrix.png` | Curated confusion matrix from raw ResNet34 all-modalities run. |
| Research split diagram | available in README | Mermaid diagram | Export to PNG later only if needed for external presentations. |
| Seed comparison plot | TODO | `assets/plots/seed_comparison.png` | Plot weighted F1 for Li and all-modalities seeds. |
| Postprocessing comparison | available | `assets/plots/postprocess_sweep_resnet34_all_seed_101.png` | Stage C validation sweep for the final model. |
| Best predictions | available | `assets/predictions/final_resnet34_all_seed_101.png` | Curated prediction grid from selected checkpoint. |
| Failure cases | TODO | `assets/failures/final_failure_cases.png` | Add false positives, false negatives and rare-class misses. |
| Metric comparison tables | available in README | Markdown tables | Keep tables close to each research phase. |

## Curation Rule

Do not commit complete Kaggle archives, checkpoints or all intermediate images. Select a small set of readable figures that directly support the research narrative.
