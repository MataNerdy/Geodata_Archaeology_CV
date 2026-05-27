# README Assets

Curated visual assets for `02_unet_segmentation/README.md`.

## Files

| File | Source | What It Shows | Notes |
|---|---|---|---|
| `hero_binary_lidar.png` | saved threshold sweep artifacts for `binary_li_no_dice` | Compact LiDAR binary segmentation example: image, ground truth, prediction, overlay | Binary UNet, Li, image size 256, threshold 0.60 |
| `threshold_sweep_binary_li_no_dice.png` | saved threshold sweep artifacts for `binary_li_no_dice` | Threshold vs `fg_iou`, `fg_dice`, precision and recall | Best threshold is 0.60 for `binary_li_no_dice` |
| `binary_li_best_predictions.png` | saved threshold sweep artifacts for `binary_li_no_dice` | Several LiDAR binary examples at threshold 0.60 | Cropped from saved best-threshold visualization |
| `multiclass_li_examples.png` | `runs/multiclass/li_only/prediction_examples.png` | Li-only multiclass predictions | Shows that multiclass mode detects foreground but class separation remains harder |
| `modality_comparison.png` | `runs/binary/binary_li_no_dice/prediction_examples.png`, `runs/binary/binary_li_ae_only/prediction_examples.png`, `runs/binary/binary_all_modalities/prediction_examples.png` | Representative saved rows for Li, Ae, and SpOr-related runs | Uses existing saved visualizations, not newly trained models |
| `failure_cases_binary_li.png` | `runs/binary/binary_li_no_dice/prediction_examples.png` | Lower rows from saved Li binary predictions | Placeholder failure-case panel until `scripts/select_readme_examples.py --mode failures` is run in a torch environment. If sample `000541` appears here, it is a `gorodishche` hard negative, not a missing kurgan label |

## Recommended README Inserts

Use these in the main README:

- Hero after the title: `assets/readme/hero_binary_lidar.png`
- Threshold tuning: `assets/readme/threshold_sweep_binary_li_no_dice.png`
- Best binary examples: `assets/readme/binary_li_best_predictions.png`
- Multiclass comparison: `assets/readme/multiclass_li_examples.png`
- Modality comparison: `assets/readme/modality_comparison.png`
- Failure cases: `assets/readme/failure_cases_binary_li.png`

## Reproducible Curated Selection

`scripts/select_readme_examples.py` was added for stricter IoU-based curation in an environment with `torch`, `numpy`, and `matplotlib`. The current `binary_li_best_predictions.png` and `failure_cases_binary_li.png` were assembled from saved artifacts because the local review environment did not have the PyTorch stack installed.

Local example:

```bash
python scripts/select_readme_examples.py \
  --data-root "../datasets/segmentation_dataset" \
  --checkpoint "runs/binary/binary_li_no_dice/binary_li_no_dice.pth" \
  --output "assets/readme/binary_li_best_predictions.png" \
  --task binary \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --binary-positive-classes "1,2" \
  --threshold 0.60 \
  --mode best
```

No model training is required for this asset generation step.

Kaggle commands for IoU-based README asset rebuild:

The same commands are also available as an optional `README Asset Rebuild`
section in `notebooks/kurgans_unet_kaggle.ipynb`. Set
`REBUILD_README_ASSETS=1` before running that notebook section.

```bash
cd /kaggle/working/Geodata_Archaeology_CV/02_unet_segmentation

python scripts/select_readme_examples.py \
  --data-root "/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset" \
  --checkpoint "/kaggle/input/datasets/matanerdy/kurgans-dataset/binary_li_no_dice.pth" \
  --output "assets/readme/binary_li_best_predictions.png" \
  --task binary \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --binary-positive-classes "1,2" \
  --threshold 0.60 \
  --mode best \
  --max-samples 5

python scripts/select_readme_examples.py \
  --data-root "/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset" \
  --checkpoint "/kaggle/input/datasets/matanerdy/kurgans-dataset/binary_li_no_dice.pth" \
  --output "assets/readme/failure_cases_binary_li.png" \
  --task binary \
  --image-size 256 \
  --split custom_regions \
  --val-regions "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км" \
  --modalities Li \
  --binary-positive-classes "1,2" \
  --threshold 0.60 \
  --mode failures \
  --max-samples 4
```
