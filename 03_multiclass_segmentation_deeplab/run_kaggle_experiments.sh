#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/matanerdy/kurgans-dataset/segmentation_dataset/segmentation_dataset}"
RUN_ROOT="${RUN_ROOT:-/kaggle/working/Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-full}"
SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_REGIONS="${VAL_REGIONS:-007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км}"
SPLIT_DIR="${SPLIT_DIR:-$SEG_DIR/splits/archaeology_5class_research_split_v1}"
RESEARCH_TRAINING_GROUPS="${RESEARCH_TRAINING_GROUPS:-both}"
RESEARCH_NUM_WORKERS="${RESEARCH_NUM_WORKERS:-0}"
RESEARCH_RUN_POSTPROCESS_SWEEP="${RESEARCH_RUN_POSTPROCESS_SWEEP:-1}"
RESEARCH_RUN_SAMPLER_ABLATION="${RESEARCH_RUN_SAMPLER_ABLATION:-1}"

export PYTHONPATH="$SEG_DIR:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/binary" "$RUN_ROOT/multiclass" "$RUN_ROOT/threshold_sweep" "$RUN_ROOT/logs"

echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "SEG_DIR=$SEG_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RUN_MODE=$RUN_MODE"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "RESEARCH_TRAINING_GROUPS=$RESEARCH_TRAINING_GROUPS"
echo "RESEARCH_NUM_WORKERS=$RESEARCH_NUM_WORKERS"

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

append_archaeology_5class_summary() {
  local experiment="$1"
  local encoder="$2"
  local modalities="$3"
  local out_dir="$4"
  local summary_path="$RUN_ROOT/archaeology_5class_summary.csv"

  EXPERIMENT="$experiment" \
  ENCODER="$encoder" \
  MODALITIES="$modalities" \
  OUT_DIR="$out_dir" \
  SUMMARY_PATH="$summary_path" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

experiment = os.environ["EXPERIMENT"]
encoder = os.environ["ENCODER"]
modalities = os.environ["MODALITIES"]
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
    "modalities": modalities,
    "mean_fg_iou": metrics.get("mean_fg_iou"),
    "iou_kurgany_tselye": metrics.get("iou_kurgany_tselye"),
    "iou_kurgany_povrezhdennye": metrics.get("iou_kurgany_povrezhdennye"),
    "iou_gorodishcha": metrics.get("iou_gorodishcha"),
    "iou_fortifikatsii": metrics.get("iou_fortifikatsii"),
    "iou_arkhitektury": metrics.get("iou_arkhitektury"),
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

append_competition_summary() {
  local experiment="$1"
  local encoder="$2"
  local out_dir="$3"
  local summary_path="$RUN_ROOT/competition_summary.csv"

  EXPERIMENT="$experiment" \
  ENCODER="$encoder" \
  OUT_DIR="$out_dir" \
  SUMMARY_PATH="$summary_path" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

experiment = os.environ["EXPERIMENT"]
encoder = os.environ["ENCODER"]
out_dir = Path(os.environ["OUT_DIR"])
summary_path = Path(os.environ["SUMMARY_PATH"])

pixel_path = out_dir / "evaluation_pixel.json"
object_path = out_dir / "evaluation_object.json"
if not pixel_path.exists():
    raise FileNotFoundError(f"evaluation_pixel.json not found: {pixel_path}")
if not object_path.exists():
    raise FileNotFoundError(f"evaluation_object.json not found: {object_path}")

pixel = json.loads(pixel_path.read_text(encoding="utf-8")).get("metrics", {})
obj = json.loads(object_path.read_text(encoding="utf-8")).get("metrics", {})
row = {
    "experiment": experiment,
    "encoder": encoder,
    "mean_fg_iou": pixel.get("mean_fg_iou"),
    "object_precision": obj.get("object_precision"),
    "object_recall": obj.get("object_recall"),
    "object_f1": obj.get("object_f1"),
    "weighted_competition_f1": obj.get("weighted_competition_f1"),
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

run_archaeology_5class_experiment() {
  local name="$1"
  local encoder="$2"
  local modalities="$3"
  local out_dir="$RUN_ROOT/multiclass/$name"

  echo "Running $name..."
  local modality_args=()
  if [[ "$modalities" != "all" ]]; then
    modality_args=(--modalities "$modalities")
  fi

  "$PYTHON_BIN" scripts/train.py \
    --config configs/all_5_classes.yaml \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --task archaeology_5class \
    "${modality_args[@]}" \
    --encoder "$encoder" \
    --class-weights "0.1,3.0,1.5,1.0,1.0,1.0" \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --patience 12 \
    --image-size 256 \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --save-samples 8 \
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
    --max-samples 8 \
    --use-postprocessing \
    --min-component-area 8 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_visualize.log"

  append_archaeology_5class_summary "$name" "$encoder" "$modalities" "$out_dir"
}

run_archaeology_competition_experiment() {
  local name="$1"
  local encoder="$2"
  local out_dir="$RUN_ROOT/multiclass/$name"

  echo "Running $name..."
  "$PYTHON_BIN" scripts/train.py \
    --config configs/all_5_classes.yaml \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --task archaeology_5class \
    --modalities Li \
    --encoder "$encoder" \
    --class-weights "0.1,3.0,1.5,1.0,1.0,1.0" \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --patience 12 \
    --image-size 256 \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --use-weighted-sampler \
    --sampler-mode class_name \
    --use-metadata-filtering \
    --max-crop-size 2048 \
    --max-objects-in-patch 40 \
    --allowed-classes "kurgany_tselye,kurgany_povrezhdennye,gorodishcha,fortifikatsii,arkhitektury" \
    --exclude-touches-border \
    --min-foreground-pixels 1 \
    --save-samples 8 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_train.log"

  "$PYTHON_BIN" scripts/evaluate.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --eval-mode pixel \
    --use-metadata-filtering \
    --max-crop-size 2048 \
    --max-objects-in-patch 40 \
    --allowed-classes "kurgany_tselye,kurgany_povrezhdennye,gorodishcha,fortifikatsii,arkhitektury" \
    --exclude-touches-border \
    --min-foreground-pixels 1 \
    --use-postprocessing \
    --min-component-area 8 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_evaluate_pixel.log"

  "$PYTHON_BIN" scripts/evaluate.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out_dir" \
    --eval-mode object \
    --object-iou-threshold 0.3 \
    --use-metadata-filtering \
    --max-crop-size 2048 \
    --max-objects-in-patch 40 \
    --allowed-classes "kurgany_tselye,kurgany_povrezhdennye,gorodishcha,fortifikatsii,arkhitektury" \
    --exclude-touches-border \
    --min-foreground-pixels 1 \
    --use-postprocessing \
    --min-component-area 8 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_evaluate_object.log"

  "$PYTHON_BIN" scripts/visualize_predictions.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --output "$out_dir/prediction_examples.png" \
    --max-samples 8 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_visualize.log"

  "$PYTHON_BIN" scripts/visualize_object_matches.py \
    --checkpoint "$out_dir/best_model.pth" \
    --data-root "$DATA_ROOT" \
    --output "$out_dir/matched_objects_visualization.png" \
    --object-iou-threshold 0.3 \
    --use-postprocessing \
    --min-component-area 8 \
    --max-samples 4 \
    2>&1 | tee "$RUN_ROOT/logs/${name}_visualize_matches.log"

  cp "$out_dir/matched_objects_visualization.png" "$out_dir/polygons_preview.png"
  append_competition_summary "$name" "$encoder" "$out_dir"
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

run_archaeology_5class() {
  rm -f "$RUN_ROOT/archaeology_5class_summary.csv"
  run_archaeology_5class_experiment "archaeology_5class_resnet34_li" "resnet34" "Li"
  run_archaeology_5class_experiment "archaeology_5class_resnet50_li" "resnet50" "Li"
  run_archaeology_5class_experiment "archaeology_5class_resnet34_all_modalities" "resnet34" "all"
  run_archaeology_5class_experiment "archaeology_5class_resnet50_all_modalities" "resnet50" "all"

  echo "Building 5-class model comparison grid..."
  "$PYTHON_BIN" scripts/compare_5class_models.py \
    --data-root "$DATA_ROOT" \
    --checkpoints "$RUN_ROOT/multiclass/archaeology_5class_resnet34_li/best_model.pth,$RUN_ROOT/multiclass/archaeology_5class_resnet50_li/best_model.pth,$RUN_ROOT/multiclass/archaeology_5class_resnet34_all_modalities/best_model.pth,$RUN_ROOT/multiclass/archaeology_5class_resnet50_all_modalities/best_model.pth" \
    --model-names "ResNet34 Li,ResNet50 Li,ResNet34 all,ResNet50 all" \
    --output "$RUN_ROOT/multiclass/archaeology_5class_comparison.png" \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --max-samples 6 \
    2>&1 | tee "$RUN_ROOT/logs/archaeology_5class_compare.log"
}

run_archaeology_5class_competition() {
  rm -f "$RUN_ROOT/competition_summary.csv"
  run_archaeology_competition_experiment "archaeology_5class_resnet50_li_competition" "resnet50"
  run_archaeology_competition_experiment "archaeology_5class_resnet34_li_competition" "resnet34"
}

run_collect_models_eval() {
  local eval_root="$RUN_ROOT/collect_models_eval"
  mkdir -p "$eval_root"

  echo "Evaluating collected 5-class models..."
  "$PYTHON_BIN" scripts/evaluate_collected_models.py \
    --data-root "$DATA_ROOT" \
    --models-root "$RUN_ROOT/collect_models" \
    --out-dir "$eval_root" \
    --task archaeology_5class \
    --image-size 256 \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --object-iou-threshold 0.3 \
    --min-area 8 \
    --eval-all-models-on-li-too \
    2>&1 | tee "$RUN_ROOT/logs/collect_models_evaluate.log"

  echo "Running collected model postprocessing sweeps..."
  "$PYTHON_BIN" scripts/collected_models_postprocess_sweep.py \
    --data-root "$DATA_ROOT" \
    --models-root "$RUN_ROOT/collect_models" \
    --eval-root "$eval_root" \
    --task archaeology_5class \
    --image-size 256 \
    --split custom_regions \
    --val-regions "$VAL_REGIONS" \
    --object-iou-threshold 0.3 \
    --eval-all-models-on-li-too \
    2>&1 | tee "$RUN_ROOT/logs/collect_models_postprocess_sweep.log"
}

run_research_split_v1() {
  local research_root="$RUN_ROOT/research_split_v1"
  mkdir -p "$research_root"
  echo "Running research_split_v1 stage..."
  echo "Expected frozen split: $SPLIT_DIR"
  if [[ ! -f "$SPLIT_DIR/train_split.csv" || ! -f "$SPLIT_DIR/val_split.csv" ]]; then
    echo "Frozen split CSV files not found. Create them once before training:" >&2
    echo "  $PYTHON_BIN scripts/create_research_split.py --data-root $DATA_ROOT --out-dir $SPLIT_DIR" >&2
    exit 3
  fi

  local optional_args=()
  if [[ "$RESEARCH_RUN_POSTPROCESS_SWEEP" == "1" ]]; then
    optional_args+=(--run-postprocess-sweep)
  fi
  if [[ "$RESEARCH_RUN_SAMPLER_ABLATION" == "1" ]]; then
    optional_args+=(--run-sampler-ablation)
  fi

  "$PYTHON_BIN" scripts/run_research_split_v1.py \
    --data-root "$DATA_ROOT" \
    --run-root "$research_root" \
    --python-bin "$PYTHON_BIN" \
    --train-split-csv "$SPLIT_DIR/train_split.csv" \
    --val-split-csv "$SPLIT_DIR/val_split.csv" \
    --training-groups "$RESEARCH_TRAINING_GROUPS" \
    --num-workers "$RESEARCH_NUM_WORKERS" \
    --skip-existing \
    --run-training \
    "${optional_args[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/research_split_v1.log"
}

run_research_split_v1_li() {
  RESEARCH_TRAINING_GROUPS=li
  RESEARCH_RUN_POSTPROCESS_SWEEP=0
  RESEARCH_RUN_SAMPLER_ABLATION=0
  run_research_split_v1
}

case "$RUN_MODE" in
  full)
    run_full_series
    ;;
  multiclass_weight_sweep)
    run_multiclass_weight_sweep
    ;;
  archaeology_5class)
    run_archaeology_5class
    ;;
  archaeology_5class_competition)
    run_archaeology_5class_competition
    ;;
  collect_models_eval)
    run_collect_models_eval
    ;;
  research_split_v1)
    run_research_split_v1
    ;;
  research_split_v1_li)
    run_research_split_v1_li
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE. Use full, multiclass_weight_sweep, archaeology_5class, archaeology_5class_competition, collect_models_eval, research_split_v1, or research_split_v1_li." >&2
    exit 2
    ;;
esac

echo "Done. Results are in $RUN_ROOT"
