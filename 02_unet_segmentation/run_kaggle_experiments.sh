#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/kurgans-dataset/kurgans_dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/02_unet_segmentation/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="$RUN_ROOT/logs"

SMOKE_VAL_REGIONS="042_ИЗБОРСК"
BASELINE_VAL_REGIONS="042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА"
BASELINE_DIR="$RUN_ROOT/baseline_all_modalities_ce_dice"

mkdir -p "$LOG_DIR"

echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import sys

print("python version:", sys.version.replace("\n", " "))
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Dataset directory not found: $DATA_ROOT" >&2
  exit 1
fi

echo "Running smoke test..."
"$PYTHON_BIN" -u train.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$RUN_ROOT/smoke_test" \
  --epochs 2 \
  --batch-size 2 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "$SMOKE_VAL_REGIONS" \
  2>&1 | tee "$LOG_DIR/smoke_test.log"

echo "Smoke test completed. Running baseline..."
"$PYTHON_BIN" -u train.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$BASELINE_DIR" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --image-size 256 \
  --split custom_regions \
  --val-regions "$BASELINE_VAL_REGIONS" \
  --class-weights "0.2,1.0,3.0" \
  2>&1 | tee "$LOG_DIR/baseline_all_modalities_ce_dice.log"

echo "Evaluating baseline checkpoint..."
"$PYTHON_BIN" -u evaluate.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$BASELINE_DIR/best_model.pth" \
  --out-dir "$BASELINE_DIR" \
  --image-size 256 \
  --batch-size 8 \
  --split custom_regions \
  --val-regions "$BASELINE_VAL_REGIONS" \
  --class-weights "0.2,1.0,3.0" \
  2>&1 | tee "$LOG_DIR/baseline_evaluate.log"

echo "Rendering baseline prediction examples..."
"$PYTHON_BIN" -u visualize_predictions.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$BASELINE_DIR/best_model.pth" \
  --output "$BASELINE_DIR/prediction_examples_eval.png" \
  --image-size 256 \
  --batch-size 8 \
  --split custom_regions \
  --val-regions "$BASELINE_VAL_REGIONS" \
  2>&1 | tee "$LOG_DIR/baseline_visualize.log"

echo "Done. Outputs are in $RUN_ROOT"
