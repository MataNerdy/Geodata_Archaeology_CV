#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$SEG_DIR:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/binary" "$RUN_ROOT/multiclass" "$RUN_ROOT/threshold_sweep" "$RUN_ROOT/logs"

echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "SEG_DIR=$SEG_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python version:", sys.version)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

run_train_eval() {
  local name="$1"
  local config="$2"
  local out_dir="$3"
  local encoder="$4"

  echo "Running $name..."
  "$PYTHON_BIN" scripts/train.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --encoder "$encoder" \
    --modalities Li \
    2>&1 | tee "$RUN_ROOT/logs/${name}_train.log"

  "$PYTHON_BIN" scripts/evaluate.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    2>&1 | tee "$RUN_ROOT/logs/${name}_evaluate.log"
}

run_train_eval \
  "binary_kurgan_li_resnet34" \
  "configs/binary_kurgan.yaml" \
  "$RUN_ROOT/binary/binary_kurgan_li_resnet34" \
  "resnet34"

run_train_eval \
  "binary_kurgan_li_resnet50" \
  "configs/binary_kurgan.yaml" \
  "$RUN_ROOT/binary/binary_kurgan_li_resnet50" \
  "resnet50"

echo "Running threshold sweep for binary_kurgan_li_resnet34..."
"$PYTHON_BIN" scripts/threshold_sweep.py \
  --checkpoint "$RUN_ROOT/binary/binary_kurgan_li_resnet34/best_model.pth" \
  --data-root "$DATA_ROOT" \
  --out-dir "$RUN_ROOT/threshold_sweep/binary_kurgan_li_resnet34" \
  2>&1 | tee "$RUN_ROOT/logs/binary_kurgan_li_resnet34_threshold_sweep.log"

run_train_eval \
  "kurgan_multiclass_li_resnet34" \
  "configs/kurgan_multiclass.yaml" \
  "$RUN_ROOT/multiclass/kurgan_multiclass_li_resnet34" \
  "resnet34"

run_train_eval \
  "kurgan_multiclass_li_resnet50" \
  "configs/kurgan_multiclass.yaml" \
  "$RUN_ROOT/multiclass/kurgan_multiclass_li_resnet50" \
  "resnet50"

echo "Done. Results are in $RUN_ROOT"

