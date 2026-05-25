#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../datasets/segmentation_dataset}"
RUN_ROOT="${RUN_ROOT:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PATIENCE="${PATIENCE:-12}"
VAL_REGIONS="${VAL_REGIONS:-042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА}"
OUT_DIR="$RUN_ROOT/baseline_all_modalities_ce_dice_early_stop"

mkdir -p "$RUN_ROOT/logs"

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "PATIENCE=$PATIENCE"

"$PYTHON_BIN" -u train.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "$VAL_REGIONS" \
  --modalities Li Ae SpOr \
  --class-weights "0.2,1.0,3.0" \
  --patience "$PATIENCE" \
  2>&1 | tee "$RUN_ROOT/logs/baseline_all_modalities_ce_dice_early_stop.train.log"

"$PYTHON_BIN" -u evaluate.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$OUT_DIR/best_model.pth" \
  --out-dir "$OUT_DIR" \
  --image-size 256 \
  --batch-size 8 \
  --split custom_regions \
  --val-regions "$VAL_REGIONS" \
  --modalities Li Ae SpOr \
  --class-weights "0.2,1.0,3.0" \
  2>&1 | tee "$RUN_ROOT/logs/baseline_all_modalities_ce_dice_early_stop.evaluate.log"

"$PYTHON_BIN" -u visualize_predictions.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$OUT_DIR/best_model.pth" \
  --output "$OUT_DIR/prediction_examples_eval.png" \
  --image-size 256 \
  --batch-size 8 \
  --split custom_regions \
  --val-regions "$VAL_REGIONS" \
  --modalities Li Ae SpOr \
  2>&1 | tee "$RUN_ROOT/logs/baseline_all_modalities_ce_dice_early_stop.visualize.log"
