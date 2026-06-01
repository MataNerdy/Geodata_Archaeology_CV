"""Run Stage D default-vs-weighted sampler ablation on the frozen split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/archaeology_5class_research_split_v1.yaml"
DEFAULT_SPLIT_DIR = "splits/archaeology_5class_research_split_v1"


def parse_args() -> argparse.Namespace:
    """Parse Stage D CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", default="runs/research_split_v1/sampler_ablation")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-split-csv", default=f"{DEFAULT_SPLIT_DIR}/train_split.csv")
    parser.add_argument("--val-split-csv", default=f"{DEFAULT_SPLIT_DIR}/val_split.csv")
    parser.add_argument("--modalities", default="Li,Ae,SpOr")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Match Stage A by default. Override to 0 only when notebook worker startup is unstable.",
    )
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    """Run both sampler variants and save a validation-only comparison."""

    args = parse_args()
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_split = resolve_path(args.train_split_csv)
    val_split = resolve_path(args.val_split_csv)
    ensure_file(train_split)
    ensure_file(val_split)

    print("[sampler] Starting sampler ablation")
    print(f"[sampler] Seed: {args.seed}")
    print(f"[sampler] Modalities: {args.modalities}")
    print(f"[sampler] Frozen train split: {train_split}")
    print(f"[sampler] Frozen val split: {val_split}")
    for sampler in ("default", "weighted"):
        run_dir = out_root / f"resnet34_all_seed_{args.seed}_{sampler}_sampler"
        print(f"[sampler] Current sampler: {sampler}")
        run_training(args, run_dir, sampler, train_split, val_split)
        run_evaluation(args, run_dir, "pixel", train_split, val_split)
        run_evaluation(args, run_dir, "object", train_split, val_split)
        print(f"[sampler] Finished sampler: {sampler}")

    rows = collect_results(out_root)
    write_summary(rows, out_root)
    write_notes(out_root)
    print(f"[sampler] Saved results: {out_root}")


def run_training(args: argparse.Namespace, run_dir: Path, sampler: str, train_split: Path, val_split: Path) -> None:
    run(
        [
            args.python_bin,
            "-u",
            "scripts/train.py",
            "--config",
            args.config,
            "--data-root",
            args.data_root,
            "--out-dir",
            str(run_dir),
            "--split",
            "frozen",
            "--train-split-csv",
            str(train_split),
            "--val-split-csv",
            str(val_split),
            "--modalities",
            args.modalities,
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--selection-metric",
            "weighted_competition_f1",
            "--object-iou-threshold",
            str(args.object_iou_threshold),
            "--sampler",
            sampler,
        ]
    )


def run_evaluation(args: argparse.Namespace, run_dir: Path, eval_mode: str, train_split: Path, val_split: Path) -> None:
    checkpoint = run_dir / "best_model.pth"
    ensure_file(checkpoint)
    run(
        [
            args.python_bin,
            "-u",
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            args.data_root,
            "--out-dir",
            str(run_dir),
            "--task",
            "archaeology_5class",
            "--encoder",
            "resnet34",
            "--image-size",
            "256",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--split",
            "frozen",
            "--train-split-csv",
            str(train_split),
            "--val-split-csv",
            str(val_split),
            "--modalities",
            args.modalities,
            "--object-iou-threshold",
            str(args.object_iou_threshold),
            "--eval-mode",
            eval_mode,
        ]
    )


def collect_results(out_root: Path) -> list[dict[str, Any]]:
    """Collect the validation metrics for completed sampler runs."""

    rows = []
    for sampler in ("default", "weighted"):
        run_dir = next(out_root.glob(f"resnet34_all_seed_*_{sampler}_sampler"), None)
        if run_dir is None:
            continue
        summary = read_json(run_dir / "summary.json")
        pixel = unwrap_metrics(run_dir / "evaluation_pixel.json")
        obj = unwrap_metrics(run_dir / "evaluation_object.json")
        config = summary.get("config", {})
        rows.append(
            {
                "experiment": run_dir.name,
                "sampler": config.get("sampler", sampler),
                "seed": config.get("seed"),
                "modalities": ",".join(config.get("modalities") or []),
                "best_epoch": summary.get("best_epoch"),
                "best_val_weighted_f1": obj.get("weighted_competition_f1"),
                "best_val_object_f1": obj.get("object_f1"),
                "best_val_object_precision": obj.get("object_precision"),
                "best_val_object_recall": obj.get("object_recall"),
                "best_val_mean_fg_iou": pixel.get("mean_fg_iou"),
                "checkpoint_path": str(run_dir / "best_model.pth"),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    """Save Stage D CSV and Markdown summaries."""

    if not rows:
        raise RuntimeError("No completed sampler ablation runs found")
    csv_path = out_root / "sampler_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Stage D Sampler Ablation",
        "",
        "Only the train sampler changes. Architecture, loss, split, optimizer, scheduler and seed remain fixed.",
        "",
        "| Experiment | Sampler | Seed | Modalities | Best epoch | Weighted F1 | Object F1 | Precision | Recall | mean_fg_iou |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['experiment']} | {row['sampler']} | {row['seed']} | {row['modalities']} | {row['best_epoch']} | "
            f"{fmt(row['best_val_weighted_f1'])} | {fmt(row['best_val_object_f1'])} | "
            f"{fmt(row['best_val_object_precision'])} | {fmt(row['best_val_object_recall'])} | "
            f"{fmt(row['best_val_mean_fg_iou'])} |"
        )
    (out_root / "sampler_ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_notes(out_root: Path) -> None:
    """Document the deliberately narrow Stage D scope."""

    (out_root / "sampler_ablation_notes.md").write_text(
        "# Stage D Notes\n\n"
        "Implemented `default` vs inverse-frequency `weighted` sampling by metadata `class_name`.\n\n"
        "The default-sampler run intentionally matches Stage A CLI settings, including `num_workers=2`, "
        "unless the caller explicitly overrides them. It uses shuffled training batches, no explicit "
        "DataLoader generator, `drop_last=False`, the same frozen split, modalities, seed, selection metric "
        "and object IoU threshold as Stage A.\n\n"
        "Bit-for-bit reconstruction of a previously trained GPU checkpoint is not guaranteed because CUDA "
        "kernels were not run in deterministic-algorithms mode. The ablation controls the experiment recipe "
        "as closely as possible without changing the established Stage A training behavior.\n\n"
        "A foreground-heavy sampler was intentionally not added: it would introduce a new sample-scoring policy "
        "based on mask content and would no longer be a minimal sampler-only ablation.\n",
        encoding="utf-8",
    )


def run(cmd: list[str]) -> None:
    """Run a subprocess with unbuffered logs."""

    print("$ " + " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload.get("metrics", payload)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
