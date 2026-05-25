#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/02_unet_segmentation/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
LOG_DIR="$RUN_ROOT/logs"
export RUN_ROOT

SMOKE_VAL_REGIONS="042_ИЗБОРСК"
BASELINE_VAL_REGIONS="042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА"

mkdir -p "$LOG_DIR"

echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"

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

run_train_eval() {
  local name="$1"
  local epochs="$2"
  local batch_size="$3"
  local class_weights="$4"
  local dice_weight="$5"
  local modalities="$6"
  local val_regions="$7"
  local out_dir="$RUN_ROOT/$name"
  local log_prefix="$LOG_DIR/$name"

  echo "Running experiment: $name"

  local train_cmd=(
    "$PYTHON_BIN" -u train.py
    --data-root "$DATA_ROOT"
    --out-dir "$out_dir"
    --epochs "$epochs"
    --batch-size "$batch_size"
    --lr 1e-3
    --image-size 256
    --split custom_regions
    --val-regions "$val_regions"
    --patience "$PATIENCE"
  )
  if [[ -n "$class_weights" ]]; then
    train_cmd+=(--class-weights "$class_weights")
  fi
  if [[ -n "$dice_weight" ]]; then
    train_cmd+=(--dice-weight "$dice_weight")
  fi
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    train_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${train_cmd[@]}" 2>&1 | tee "$log_prefix.train.log"

  local eval_cmd=(
    "$PYTHON_BIN" -u evaluate.py
    --data-root "$DATA_ROOT"
    --checkpoint "$out_dir/best_model.pth"
    --out-dir "$out_dir"
    --image-size 256
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
  )
  if [[ -n "$class_weights" ]]; then
    eval_cmd+=(--class-weights "$class_weights")
  fi
  if [[ -n "$dice_weight" ]]; then
    eval_cmd+=(--dice-weight "$dice_weight")
  fi
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    eval_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${eval_cmd[@]}" 2>&1 | tee "$log_prefix.evaluate.log"
}

run_visualization() {
  local name="$1"
  local batch_size="$2"
  local modalities="$3"
  local val_regions="$4"
  local out_dir="$RUN_ROOT/$name"
  local log_prefix="$LOG_DIR/$name"

  echo "Rendering prediction examples: $name"
  local vis_cmd=(
    "$PYTHON_BIN" -u visualize_predictions.py
    --data-root "$DATA_ROOT"
    --checkpoint "$out_dir/best_model.pth"
    --output "$out_dir/prediction_examples_eval.png"
    --image-size 256
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
  )
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    vis_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${vis_cmd[@]}" 2>&1 | tee "$log_prefix.visualize.log"
}

build_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

run_root = Path(os.environ["RUN_ROOT"])

def _read_best_epoch(run_dir: Path):
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return None
    history = pd.read_csv(history_path)
    if "val_mean_fg_iou" not in history.columns or history.empty:
        return None
    idx = history["val_mean_fg_iou"].idxmax()
    return int(history.loc[idx, "epoch"])

rows = []
for path in sorted(run_root.glob("*/evaluation.json")):
    with path.open("r", encoding="utf-8") as file:
        row = json.load(file)
    run_dir = path.parent
    row = {"experiment": run_dir.name, **row}
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        row["epochs_requested"] = config.get("epochs")
        row["modalities"] = ",".join(config.get("modalities") or [])
        row["class_weights"] = config.get("class_weights")
        row["dice_weight"] = config.get("dice_weight")
        row["best_epoch"] = _read_best_epoch(run_dir)
    rows.append(row)

def _sort_key(row):
    order = [
        "smoke_test",
        "baseline_all_modalities_ce_dice",
        "li_only",
        "ae_only",
        "li_ae_only",
        "spor_only_diagnostic",
        "no_dice",
        "lower_damaged_weight",
    ]
    try:
        return order.index(row["experiment"])
    except ValueError:
        return len(order)

rows = sorted(rows, key=_sort_key)
summary_path = run_root / "experiments_summary.csv"
pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f"Saved summary to {summary_path}")
PY
}

echo "Running smoke test..."
run_train_eval "smoke_test" 2 2 "" "" "" "$SMOKE_VAL_REGIONS"

echo "Smoke test completed. Running full experiments with early stopping..."
run_train_eval "baseline_all_modalities_ce_dice" "$EPOCHS" 8 "0.2,1.0,3.0" "" "Li Ae SpOr" "$BASELINE_VAL_REGIONS"
run_visualization "baseline_all_modalities_ce_dice" 8 "Li Ae SpOr" "$BASELINE_VAL_REGIONS"

run_train_eval "li_only" "$EPOCHS" 8 "0.2,1.0,3.0" "" "Li" "$BASELINE_VAL_REGIONS"
run_train_eval "ae_only" "$EPOCHS" 8 "0.2,1.0,3.0" "" "Ae" "$BASELINE_VAL_REGIONS"
run_train_eval "li_ae_only" "$EPOCHS" 8 "0.2,1.0,3.0" "" "Li Ae" "$BASELINE_VAL_REGIONS"
run_train_eval "spor_only_diagnostic" "$EPOCHS" 8 "0.2,1.0,3.0" "" "SpOr" "$BASELINE_VAL_REGIONS"
run_train_eval "no_dice" "$EPOCHS" 8 "0.2,1.0,3.0" "0" "Li Ae SpOr" "$BASELINE_VAL_REGIONS"
run_train_eval "lower_damaged_weight" "$EPOCHS" 8 "0.2,1.0,2.0" "" "Li Ae SpOr" "$BASELINE_VAL_REGIONS"

build_summary

echo "Done. Outputs are in $RUN_ROOT"
