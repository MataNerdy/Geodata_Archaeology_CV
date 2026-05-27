#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-full}"
SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_REGIONS="${VAL_REGIONS:-007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км}"

export PYTHONPATH="$SEG_DIR:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/binary" "$RUN_ROOT/multiclass" "$RUN_ROOT/threshold_sweep" "$RUN_ROOT/logs"

echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "SEG_DIR=$SEG_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RUN_MODE=$RUN_MODE"

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

append_multiclass_weight_summary() {
  local experiment="$1"
  local encoder="$2"
  local class_weights="$3"
  local out_dir="$4"
  local summary_path="$RUN_ROOT/multiclass_weight_sweep_summary.csv"

  EXPERIMENT="$experiment" \
  ENCODER="$encoder" \
  CLASS_WEIGHTS="$class_weights" \
  OUT_DIR="$out_dir" \
  SUMMARY_PATH="$summary_path" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

experiment = os.environ["EXPERIMENT"]
encoder = os.environ["ENCODER"]
class_weights = os.environ["CLASS_WEIGHTS"]
out_dir = Path(os.environ["OUT_DIR"])
summary_path = Path(os.environ["SUMMARY_PATH"])

evaluation_path = out_dir / "evaluation.json"
run_summary_path = out_dir / "summary.json"

if not evaluation_path.exists():
    raise FileNotFoundError(f"evaluation.json not found: {evaluation_path}")

evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
metrics = evaluation.get("metrics", evaluation)
best_epoch = None
if run_summary_path.exists():
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    best_epoch = run_summary.get("best_epoch")

row = {
    "experiment": experiment,
    "encoder": encoder,
    "class_weights": class_weights,
    "mean_fg_iou": metrics.get("mean_fg_iou"),
    "iou_kurgany_tselye": metrics.get("iou_kurgany_tselye"),
    "iou_kurgany_povrezhdennye": metrics.get("iou_kurgany_povrezhdennye"),
    "dice_kurgany_tselye": metrics.get("dice_kurgany_tselye"),
    "dice_kurgany_povrezhdennye": metrics.get("dice_kurgany_povrezhdennye"),
    "pixel_accuracy": metrics.get("pixel_accuracy"),
    "best_epoch": best_epoch,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
if summary_path.exists():
    df = pd.read_csv(summary_path)
    df = df[df["experiment"] != experiment]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
else:
    df = pd.DataFrame([row])
df.to_csv(summary_path, index=False)
print(f"Updated summary: {summary_path}")
print(pd.DataFrame([row]).to_string(index=False))
PY
}

run_multiclass_weight_experiment() {
  local name="$1"
  local encoder="$2"
  local class_weights="$3"
  local out_dir="$RUN_ROOT/multiclass/$name"

  echo "Running $name..."
  "$PYTHON_BIN" scripts/train.py \
    --config configs/kurgan_multiclass.yaml \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --task kurgan_multiclass \
    --modalities Li \
    --encoder "$encoder" \
    --class-weights "$class_weights" \
    --epochs 50 \
    --patience 12 \
    --image-size 256 \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --save-samples 6 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_train.log"

  "$PYTHON_BIN" scripts/evaluate.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    2>&1 | tee "$RUN_ROOT/logs/${name}_evaluate.log"

  "$PYTHON_BIN" scripts/visualize_predictions.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --output "$out_dir/prediction_examples.png" \
    --max-samples 6 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_visualize.log"

  append_multiclass_weight_summary "$name" "$encoder" "$class_weights" "$out_dir"
}

run_full_series() {
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
}

run_multiclass_weight_sweep() {
  rm -f "$RUN_ROOT/multiclass_weight_sweep_summary.csv"
  run_multiclass_weight_experiment "kurgan_multiclass_li_resnet50_w_02_2_1" "resnet50" "0.2,2.0,1.0"
  run_multiclass_weight_experiment "kurgan_multiclass_li_resnet50_w_02_3_1" "resnet50" "0.2,3.0,1.0"
  run_multiclass_weight_experiment "kurgan_multiclass_li_resnet50_w_01_3_1" "resnet50" "0.1,3.0,1.0"
  run_multiclass_weight_experiment "kurgan_multiclass_li_resnet50_w_02_2_08" "resnet50" "0.2,2.0,0.8"
  run_multiclass_weight_experiment "kurgan_multiclass_li_resnet34_w_02_3_1" "resnet34" "0.2,3.0,1.0"
}

case "$RUN_MODE" in
  full)
    run_full_series
    ;;
  multiclass_weight_sweep)
    run_multiclass_weight_sweep
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE. Use full or multiclass_weight_sweep." >&2
    exit 2
    ;;
esac

echo "Done. Results are in $RUN_ROOT"
