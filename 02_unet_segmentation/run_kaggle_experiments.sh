#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/02_unet_segmentation/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
RUN_MODE="${1:-${RUN_MODE:-train}}"
LOG_DIR="$RUN_ROOT/logs"
export RUN_ROOT

SMOKE_VAL_REGIONS="042_ИЗБОРСК"
BASELINE_VAL_REGIONS="042_ИЗБОРСК,044_ГОЧЕВО,033_МИЛОВИДОВО_0.1км,007_ЮШКОВО,047_КАЛМЫКИЯ_1,008_СЕЛЯНЕ,025_ШУМГОРА"
BINARY_VAL_REGIONS="007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км"
THRESHOLDS="${THRESHOLDS:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
BINARY_POSITIVE_CLASSES="${BINARY_POSITIVE_CLASSES:-1,2}"

mkdir -p "$LOG_DIR"

echo "DATA_ROOT=$DATA_ROOT"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "RUN_MODE=$RUN_MODE"
echo "BINARY_POSITIVE_CLASSES=$BINARY_POSITIVE_CLASSES"

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
  local task="$2"
  local epochs="$3"
  local batch_size="$4"
  local image_size="$5"
  local class_weights="$6"
  local dice_weight="$7"
  local modalities="$8"
  local val_regions="$9"
  local pos_weight="${10:-}"
  local bce_weight="${11:-}"
  local out_dir="$RUN_ROOT/$name"
  local log_prefix="$LOG_DIR/$name"

  echo "Running experiment: $name"

  local train_cmd=(
    "$PYTHON_BIN" -u scripts/train.py
    --task "$task"
    --data-root "$DATA_ROOT"
    --out-dir "$out_dir"
    --epochs "$epochs"
    --batch-size "$batch_size"
    --lr 1e-3
    --image-size "$image_size"
    --split custom_regions
    --val-regions "$val_regions"
    --patience "$PATIENCE"
  )
  if [[ "$task" == "binary" ]]; then
    train_cmd+=(--binary-positive-classes "$BINARY_POSITIVE_CLASSES")
  fi
  if [[ -n "$class_weights" ]]; then
    train_cmd+=(--class-weights "$class_weights")
  fi
  if [[ -n "$dice_weight" ]]; then
    train_cmd+=(--dice-weight "$dice_weight")
  fi
  if [[ -n "$pos_weight" ]]; then
    train_cmd+=(--pos-weight "$pos_weight")
  fi
  if [[ -n "$bce_weight" ]]; then
    train_cmd+=(--bce-weight "$bce_weight")
  fi
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    train_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${train_cmd[@]}" 2>&1 | tee "$log_prefix.train.log"

  local eval_cmd=(
    "$PYTHON_BIN" -u scripts/evaluate.py
    --task "$task"
    --data-root "$DATA_ROOT"
    --checkpoint "$out_dir/best_model.pth"
    --out-dir "$out_dir"
    --image-size "$image_size"
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
  )
  if [[ "$task" == "binary" ]]; then
    eval_cmd+=(--binary-positive-classes "$BINARY_POSITIVE_CLASSES")
  fi
  if [[ -n "$class_weights" ]]; then
    eval_cmd+=(--class-weights "$class_weights")
  fi
  if [[ -n "$dice_weight" ]]; then
    eval_cmd+=(--dice-weight "$dice_weight")
  fi
  if [[ -n "$pos_weight" ]]; then
    eval_cmd+=(--pos-weight "$pos_weight")
  fi
  if [[ -n "$bce_weight" ]]; then
    eval_cmd+=(--bce-weight "$bce_weight")
  fi
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    eval_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${eval_cmd[@]}" 2>&1 | tee "$log_prefix.evaluate.log"
}

run_visualization() {
  local name="$1"
  local task="$2"
  local batch_size="$3"
  local image_size="$4"
  local modalities="$5"
  local val_regions="$6"
  local out_dir="$RUN_ROOT/$name"
  local log_prefix="$LOG_DIR/$name"

  echo "Rendering prediction examples: $name"
  local vis_cmd=(
    "$PYTHON_BIN" -u scripts/visualize_predictions.py
    --task "$task"
    --data-root "$DATA_ROOT"
    --checkpoint "$out_dir/best_model.pth"
    --output "$out_dir/prediction_examples_eval.png"
    --image-size "$image_size"
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
  )
  if [[ "$task" == "binary" ]]; then
    vis_cmd+=(--binary-positive-classes "$BINARY_POSITIVE_CLASSES")
  fi
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    vis_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${vis_cmd[@]}" 2>&1 | tee "$log_prefix.visualize.log"
}

run_threshold_sweep() {
  local name="$1"
  local batch_size="$2"
  local image_size="$3"
  local modalities="$4"
  local val_regions="$5"
  local out_dir="$RUN_ROOT/$name"
  local log_prefix="$LOG_DIR/$name"
  local checkpoint="$out_dir/best_model.pth"

  if [[ ! -f "$checkpoint" ]]; then
    checkpoint="$CHECKPOINT_ROOT/$name.pth"
  fi

  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] checkpoint not found: $out_dir/best_model.pth or $CHECKPOINT_ROOT/$name.pth"
    return 0
  fi

  mkdir -p "$out_dir"

  echo "Running threshold sweep: $name"
  local sweep_cmd=(
    "$PYTHON_BIN" -u scripts/threshold_sweep.py
    --data-root "$DATA_ROOT"
    --checkpoint "$checkpoint"
    --out-dir "$out_dir"
    --task binary
    --image-size "$image_size"
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
    --thresholds "$THRESHOLDS"
    --binary-positive-classes "$BINARY_POSITIVE_CLASSES"
  )
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    sweep_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${sweep_cmd[@]}" 2>&1 | tee "$log_prefix.threshold_sweep.log"

  local best_threshold
  best_threshold=$("$PYTHON_BIN" - "$out_dir/threshold_sweep.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    summary = json.load(file)
print(summary["best_threshold"])
PY
)

  echo "Rendering best-threshold prediction examples: $name threshold=$best_threshold"
  local vis_cmd=(
    "$PYTHON_BIN" -u scripts/visualize_predictions.py
    --task binary
    --data-root "$DATA_ROOT"
    --checkpoint "$checkpoint"
    --output "$out_dir/prediction_examples_thr_best.png"
    --image-size "$image_size"
    --batch-size "$batch_size"
    --split custom_regions
    --val-regions "$val_regions"
    --threshold "$best_threshold"
    --binary-positive-classes "$BINARY_POSITIVE_CLASSES"
  )
  if [[ -n "$modalities" ]]; then
    read -r -a modality_args <<< "$modalities"
    vis_cmd+=(--modalities "${modality_args[@]}")
  fi
  "${vis_cmd[@]}" 2>&1 | tee "$log_prefix.visualize_thr_best.log"
}

build_threshold_sweeps_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

run_root = Path(os.environ["RUN_ROOT"])
columns = [
    "experiment",
    "checkpoint",
    "image_size",
    "best_threshold",
    "best_fg_iou",
    "best_fg_dice",
    "precision_at_best",
    "recall_at_best",
    "pixel_accuracy_at_best",
    "fg_iou_at_0_5",
    "delta_iou_vs_0_5",
]
rows = []
for path in sorted(run_root.glob("*/threshold_sweep.json")):
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    run_dir = path.parent
    rows.append(
        {
            "experiment": run_dir.name,
            "checkpoint": summary.get("checkpoint"),
            "image_size": summary.get("image_size"),
            "best_threshold": summary.get("best_threshold"),
            "best_fg_iou": summary.get("best_fg_iou"),
            "best_fg_dice": summary.get("best_fg_dice"),
            "precision_at_best": summary.get("precision_at_best"),
            "recall_at_best": summary.get("recall_at_best"),
            "pixel_accuracy_at_best": summary.get("pixel_accuracy_at_best"),
            "fg_iou_at_0_5": summary.get("fg_iou_at_0_5"),
            "delta_iou_vs_0_5": summary.get("delta_iou_vs_0_5"),
        }
    )

summary_path = run_root / "threshold_sweeps_summary.csv"
pd.DataFrame(rows, columns=columns).to_csv(summary_path, index=False)
print(f"Saved threshold sweeps summary to {summary_path}")
PY
}

run_threshold_sweeps() {
  echo "Running threshold sweeps for trained binary UNet models..."
  run_threshold_sweep "binary_li_no_dice" 8 256 "Li" "$BINARY_VAL_REGIONS"
  run_threshold_sweep "binary_li_pos_weight_2" 8 256 "Li" "$BINARY_VAL_REGIONS"
  run_threshold_sweep "binary_li_pos_weight_4" 8 256 "Li" "$BINARY_VAL_REGIONS"
  run_threshold_sweep "binary_li_only" 8 256 "Li" "$BINARY_VAL_REGIONS"
  run_threshold_sweep "binary_li_512_no_dice" 8 512 "Li" "$BINARY_VAL_REGIONS"
  build_threshold_sweeps_summary
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
    metric = "val_mean_fg_iou" if "val_mean_fg_iou" in history.columns else "val_fg_iou"
    if metric not in history.columns or history.empty:
        return None
    idx = history[metric].idxmax()
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
        row["task"] = config.get("task")
        row["modalities"] = ",".join(config.get("modalities") or [])
        row["class_weights"] = config.get("class_weights")
        row["pos_weight"] = config.get("pos_weight")
        row["bce_weight"] = config.get("bce_weight")
        row["dice_weight"] = config.get("dice_weight")
        row["best_epoch"] = _read_best_epoch(run_dir)
    rows.append(row)

def _sort_key(row):
    order = [
        "smoke_test",
        "baseline_all_modalities_ce_dice",
        "binary_li_only",
        "binary_li_ae_only",
        "binary_all_modalities",
        "binary_li_no_dice",
        "binary_li_pos_weight_2",
        "binary_li_pos_weight_4",
        "binary_li_512_no_dice",
        "binary_li_512_pos_weight_2",
        "binary_li_512_ce_dice",
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

if [[ "$RUN_MODE" == "threshold_sweeps" ]]; then
  run_threshold_sweeps
  echo "Done. Threshold sweep outputs are in $RUN_ROOT"
  exit 0
fi

if [[ "$RUN_MODE" != "train" ]]; then
  echo "Unknown RUN_MODE: $RUN_MODE. Use 'train' or 'threshold_sweeps'." >&2
  exit 1
fi

echo "Running smoke test..."
run_train_eval "smoke_test" "multiclass" 2 2 256 "" "" "" "$SMOKE_VAL_REGIONS"

echo "Smoke test completed. Running third-series final UNet binary experiments..."

run_train_eval "binary_li_512_no_dice" "binary" "$EPOCHS" 8 512 "" "0.0" "Li" "$BINARY_VAL_REGIONS" "1.0" "1.0"
run_visualization "binary_li_512_no_dice" "binary" 8 512 "Li" "$BINARY_VAL_REGIONS"

run_train_eval "binary_li_512_pos_weight_2" "binary" "$EPOCHS" 8 512 "" "0.0" "Li" "$BINARY_VAL_REGIONS" "2.0" "1.0"
run_visualization "binary_li_512_pos_weight_2" "binary" 8 512 "Li" "$BINARY_VAL_REGIONS"

run_train_eval "binary_li_512_ce_dice" "binary" "$EPOCHS" 8 512 "" "1.0" "Li" "$BINARY_VAL_REGIONS" "2.0" "1.0"
run_visualization "binary_li_512_ce_dice" "binary" 8 512 "Li" "$BINARY_VAL_REGIONS"

build_summary

echo "Done. Outputs are in $RUN_ROOT"
