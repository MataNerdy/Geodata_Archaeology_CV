"""Run the frozen-split ResNet34 research stage for 5-class archaeology segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 21, 42, 77, 101)
SPLIT_DIR = Path("splits/archaeology_5class_research_split_v1")
TRAIN_SPLIT = SPLIT_DIR / "train_split.csv"
VAL_SPLIT = SPLIT_DIR / "val_split.csv"
TEST_SPLIT = SPLIT_DIR / "test_split.csv"
BASE_CONFIG = "configs/archaeology_5class_research_split_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--run-root", default="runs/research_split_v1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", default=BASE_CONFIG)
    parser.add_argument("--train-split-csv", default=str(TRAIN_SPLIT))
    parser.add_argument("--val-split-csv", default=str(VAL_SPLIT))
    parser.add_argument("--run-training", action="store_true", help="Run seed-series training jobs.")
    parser.add_argument(
        "--training-groups",
        choices=("both", "all", "li"),
        default="both",
        help="Choose which seed series to train. Use li to resume after the all-modalities series.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a training run when best_model.pth and evaluation_object.json already exist.",
    )
    parser.add_argument("--run-postprocess-sweep", action="store_true", help="Run postprocessing sweep for the selected best model.")
    parser.add_argument("--run-sampler-ablation", action="store_true", help="Run default-vs-weighted sampler ablation after best selection.")
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = resolve_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    train_split = resolve_path(args.train_split_csv)
    val_split = resolve_path(args.val_split_csv)
    ensure_split_artifacts(train_split, val_split)

    if args.run_training:
        groups = {
            "both": (("all", "Li,Ae,SpOr"), ("li", "Li")),
            "all": (("all", "Li,Ae,SpOr"),),
            "li": (("li", "Li"),),
        }[args.training_groups]
        print(f"[research] Starting seed series: {args.training_groups}")
        for group, modalities in groups:
            for seed in SEEDS:
                exp_name = f"resnet34_{group}_seed_{seed}"
                run_dir = run_root / exp_name
                if args.skip_existing and is_completed_run(run_dir):
                    print(f"[research] Skipping completed run: {exp_name}")
                    continue
                run_one_training(args, run_dir, exp_name, seed, modalities, train_split, val_split)
                run_val_evaluations(args, run_dir, modalities, train_split, val_split)

    rows = collect_seed_summary(run_root)
    write_seed_summary(rows, run_root)
    best = select_best(rows, run_root)

    if args.run_postprocess_sweep and best:
        run_postprocess_sweep(args, run_root, best, train_split, val_split)
    if args.run_sampler_ablation and best:
        run_sampler_ablation(args, run_root, best, train_split, val_split)

    print(f"[research] Done. Run root: {run_root}")


def ensure_split_artifacts(train_split: Path, val_split: Path) -> None:
    print(f"[split] Train split file: {train_split}")
    print(f"[split] Val split file: {val_split}")
    if not train_split.exists():
        raise FileNotFoundError(f"Frozen train split is missing: {train_split}")
    if not val_split.exists():
        raise FileNotFoundError(f"Frozen val split is missing: {val_split}")
    split_dir = train_split.parent
    test_split = split_dir / "test_split.csv"
    if test_split.exists():
        print(f"[split] Test split file: {test_split}")
    else:
        print("[split] No test_split.csv found. TODO: add only when a real held-out test protocol exists.")
    for required in ("split_config.json", "split_stats.md"):
        path = split_dir / required
        if path.exists():
            print(f"[split] Found {required}: {path}")
        else:
            print(f"[split] Missing {required}. Re-run scripts/create_research_split.py once to materialize full split artifact.")


def is_completed_run(run_dir: Path) -> bool:
    return (run_dir / "best_model.pth").exists() and (run_dir / "evaluation_object.json").exists()


def run_one_training(args: argparse.Namespace, out_dir: Path, exp_name: str, seed: int, modalities: str, train_split: Path, val_split: Path) -> None:
    print(f"[train] Experiment name: {exp_name}")
    print(f"[train] Seed: {seed}")
    print(f"[train] Modalities: {modalities}")
    print(f"[train] Split files used: train={train_split} val={val_split}")
    print("[train] Model: DeepLabV3+ ResNet34")
    print("[train] Image size: 256")
    print(f"[train] Batch size: {args.batch_size}")
    print("[train] Loss: CE 0.7 + Dice 0.3")
    print("[train] Class weights: 0.2,1.0,1.0,1.4,1.8,1.8")
    print("[train] Optimizer / LR / scheduler: Adam / 1e-4 / ReduceLROnPlateau")
    cmd = [
        args.python_bin,
        "scripts/train.py",
        "--config",
        args.config,
        "--data-root",
        args.data_root,
        "--out-dir",
        str(out_dir),
        "--split",
        "frozen",
        "--train-split-csv",
        str(train_split),
        "--val-split-csv",
        str(val_split),
        "--modalities",
        modalities,
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--selection-metric",
        "weighted_competition_f1",
        "--object-iou-threshold",
        str(args.object_iou_threshold),
    ]
    run(cmd)


def run_val_evaluations(args: argparse.Namespace, run_dir: Path, modalities: str, train_split: Path, val_split: Path) -> None:
    checkpoint = run_dir / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint missing: {checkpoint}")
    common = [
        "--checkpoint", str(checkpoint),
        "--data-root", args.data_root,
        "--out-dir", str(run_dir),
        "--task", "archaeology_5class",
        "--encoder", "resnet34",
        "--image-size", "256",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--split", "frozen",
        "--train-split-csv", str(train_split),
        "--val-split-csv", str(val_split),
        "--modalities", modalities,
        "--object-iou-threshold", str(args.object_iou_threshold),
    ]
    print(f"[eval] Running val pixel evaluation: {checkpoint}")
    run([args.python_bin, "scripts/evaluate.py", *common, "--eval-mode", "pixel"])
    print(f"[eval] Running val object evaluation: {checkpoint}")
    run([args.python_bin, "scripts/evaluate.py", *common, "--eval-mode", "object"])
    copy_if_exists(run_dir / "evaluation.json", run_dir / "evaluation_val.json")
    copy_if_exists(run_dir / "evaluation.csv", run_dir / "evaluation_val.csv")
    test_split = train_split.parent / "test_split.csv"
    if test_split.exists():
        copy_if_exists(test_split, run_dir / "test_split.csv")


def collect_seed_summary(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(run_root.glob("resnet34_*_seed_*")):
        summary = read_json(run_dir / "summary.json")
        pixel = read_metrics_json(run_dir / "evaluation_pixel.json") or read_metrics_json(run_dir / "evaluation.json")
        obj = read_metrics_json(run_dir / "evaluation_object.json")
        config = summary.get("config", {}) if isinstance(summary, dict) else {}
        rows.append({
            "experiment": run_dir.name,
            "modalities": ",".join(config.get("modalities") or []),
            "seed": config.get("seed"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_weighted_f1": obj.get("weighted_competition_f1"),
            "best_val_object_f1": obj.get("object_f1"),
            "best_val_object_precision": obj.get("object_precision"),
            "best_val_object_recall": obj.get("object_recall"),
            "best_val_mean_fg_iou": pixel.get("mean_fg_iou"),
            "checkpoint_path": str(run_dir / "best_model.pth"),
        })
    return rows


def write_seed_summary(rows: list[dict[str, Any]], run_root: Path) -> None:
    csv_path = run_root / "research_split_v1_seed_summary.csv"
    fieldnames = [
        "experiment", "modalities", "seed", "best_epoch", "best_val_weighted_f1",
        "best_val_object_f1", "best_val_object_precision", "best_val_object_recall",
        "best_val_mean_fg_iou", "checkpoint_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = run_root / "research_split_v1_seed_summary.md"
    lines = ["# Research Split v1 Seed Summary", "", "| Experiment | Modalities | Seed | Weighted F1 | Object F1 | Precision | Recall | mean_fg_iou |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['modalities']} | {row['seed']} | {fmt(row['best_val_weighted_f1'])} | "
            f"{fmt(row['best_val_object_f1'])} | {fmt(row['best_val_object_precision'])} | "
            f"{fmt(row['best_val_object_recall'])} | {fmt(row['best_val_mean_fg_iou'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[research] Saved summary: {csv_path}")
    print(f"[research] Saved summary: {md_path}")


def select_best(rows: list[dict[str, Any]], run_root: Path) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("best_val_weighted_f1") not in (None, "")]
    if not valid:
        print("[research] No evaluated runs found for best selection yet.")
        return None
    best = max(valid, key=lambda row: float(row["best_val_weighted_f1"]))
    text = f"""# Best Model Selection

Selected model: `{best['experiment']}`

Primary metric: validation `weighted_competition_f1` = `{best['best_val_weighted_f1']}`.

Secondary metrics:

- object_f1: `{best['best_val_object_f1']}`
- object_precision: `{best['best_val_object_precision']}`
- object_recall: `{best['best_val_object_recall']}`
- mean_fg_iou: `{best['best_val_mean_fg_iou']}`

Checkpoint for postprocessing sweep:

`{best['checkpoint_path']}`

This is validation-only model selection. No test split is available unless `splits/archaeology_5class_research_split_v1/test_split.csv` is explicitly added as a held-out protocol artifact.
"""
    path = run_root / "best_model_selection.md"
    path.write_text(text, encoding="utf-8")
    print(f"[research] Best model selected: {best['experiment']} weighted_f1={best['best_val_weighted_f1']}")
    print(f"[research] Saved best selection: {path}")
    return best


def run_postprocess_sweep(args: argparse.Namespace, run_root: Path, best: dict[str, Any], train_split: Path, val_split: Path) -> None:
    sweep_root = run_root / "postprocess_sweep"
    models_root = sweep_root / "collect_models"
    group = "li" if best["modalities"] == "Li" else "all"
    model_dir = models_root / group
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(best["checkpoint_path"])
    staged = model_dir / checkpoint.name
    shutil.copy2(checkpoint, staged)
    print(f"[sweep] Starting postprocessing sweep for best checkpoint: {checkpoint}")
    cmd = [
        args.python_bin, "scripts/collected_models_postprocess_sweep.py",
        "--data-root", args.data_root,
        "--models-root", str(models_root),
        "--eval-root", str(sweep_root),
        "--task", "archaeology_5class",
        "--image-size", "256",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--split", "frozen",
        "--train-split-csv", str(train_split),
        "--val-split-csv", str(val_split),
        "--object-iou-threshold", str(args.object_iou_threshold),
        "--confidence-thresholds", "0.00,0.10,0.20,0.30,0.40,0.50",
        "--min-component-areas", "8,16,32,64,128,256",
    ]
    run(cmd)


def run_sampler_ablation(args: argparse.Namespace, run_root: Path, best: dict[str, Any], train_split: Path, val_split: Path) -> None:
    print("[sampler] Starting sampler ablation")
    sampler_root = run_root / "sampler_ablation"
    modalities = best["modalities"] or "Li,Ae,SpOr"
    seed = best.get("seed") or 42
    for sampler_name, flag in (("default", "false"), ("weighted", "true")):
        print(f"[sampler] Current sampler: {sampler_name}")
        print(f"[sampler] Seed: {seed}")
        out_dir = sampler_root / f"{best['experiment']}_{sampler_name}_sampler"
        cmd = [
            args.python_bin, "scripts/train.py",
            "--config", args.config,
            "--data-root", args.data_root,
            "--out-dir", str(out_dir),
            "--split", "frozen",
            "--train-split-csv", str(train_split),
            "--val-split-csv", str(val_split),
            "--modalities", modalities,
            "--seed", str(seed),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--selection-metric", "weighted_competition_f1",
            "--use-weighted-sampler" if flag == "true" else "--sampler-mode",
        ]
        if flag == "false":
            # There is no negative boolean CLI for argparse store_true. Use config default false from old recipe file.
            cmd.append("class_name")
        run(cmd)
        run_val_evaluations(args, out_dir, modalities, train_split, val_split)
    limitation = sampler_root / "sampler_ablation_notes.md"
    limitation.write_text(
        "# Sampler Ablation Notes\n\nImplemented default sampler vs weighted class-name sampler. "
        "A foreground-heavy sampler was not added because it would require a new sampling policy beyond the existing metadata-based infrastructure.\n",
        encoding="utf-8",
    )
    print(f"[sampler] Saved results: {sampler_root}")


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload.get("metrics", payload) if isinstance(payload, dict) else {}


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
