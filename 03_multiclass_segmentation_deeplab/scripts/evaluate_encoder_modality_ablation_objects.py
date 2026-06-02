"""Evaluate Phase 1 checkpoints with object-level metrics and build a summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTS = (
    ("resnet34_li", "resnet34", "Li", ("resnet34_li",)),
    ("resnet50_li", "resnet50", "Li", ("resnet50_li",)),
    ("resnet34_all", "resnet34", None, ("resnet34_all", "resnet34_all_modalities")),
    ("resnet50_all", "resnet50", None, ("resnet50_all", "resnet50_all_modalities")),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--run-root",
        default="runs/archaeology_5class_encoder_modality_ablation_raw_v1",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Evaluate checkpoints and save a comparison table."""

    args = parse_args()
    run_root = Path(args.run_root)
    print("[phase1-object-eval] Raw diagnostic series only. Main benchmark remains research_split_v1 Stage C.")

    for experiment, encoder, modalities, folder_candidates in EXPERIMENTS:
        run_dir = find_run_dir(run_root, folder_candidates)
        checkpoint = find_checkpoint(run_dir)
        train_split = require_file(run_dir / "train_split.csv")
        val_split = require_file(run_dir / "val_split.csv")

        print(f"[phase1-object-eval] Experiment: {experiment}")
        print(f"[phase1-object-eval] Checkpoint: {checkpoint}")
        print(f"[phase1-object-eval] Encoder: {encoder}")
        print(f"[phase1-object-eval] Modalities: {modalities or 'all (no modality filter)'}")
        print(f"[phase1-object-eval] Frozen split files: train={train_split} val={val_split}")

        if args.summary_only:
            print("[phase1-object-eval] Summary-only mode: inference skipped")
            continue

        cmd = [
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
            encoder,
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
            "--eval-mode",
            "object",
            "--object-iou-threshold",
            str(args.object_iou_threshold),
            "--min-component-area",
            str(args.min_component_area),
        ]
        if modalities:
            cmd.extend(["--modalities", modalities])
        run(cmd)

    summary = build_summary(run_root)
    csv_path = run_root / "encoder_modality_object_summary.csv"
    md_path = run_root / "encoder_modality_object_summary.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(summary), encoding="utf-8")
    print(f"[phase1-object-eval] Saved summary: {csv_path}")
    print(f"[phase1-object-eval] Saved summary: {md_path}")
    print(summary.to_string(index=False))


def build_summary(run_root: Path) -> pd.DataFrame:
    """Collect pixel and object metrics from four run folders."""

    rows = []
    for experiment, encoder, modalities, folder_candidates in EXPERIMENTS:
        run_dir = find_run_dir(run_root, folder_candidates)
        pixel = read_metrics(run_dir / "evaluation_pixel.json")
        objects = read_metrics(run_dir / "evaluation_object.json")
        rows.append(
            {
                "experiment": experiment,
                "encoder": encoder,
                "modalities": modalities or "all",
                "mean_fg_iou": pixel.get("mean_fg_iou"),
                "pixel_accuracy": pixel.get("pixel_accuracy"),
                "object_precision": objects.get("object_precision"),
                "object_recall": objects.get("object_recall"),
                "object_f1": objects.get("object_f1"),
                "weighted_competition_f1": objects.get("weighted_competition_f1"),
                "object_iou_threshold": objects.get("object_iou_threshold"),
            }
        )
    return pd.DataFrame(rows)


def read_metrics(path: Path) -> dict[str, float]:
    """Read metrics from an evaluation JSON artifact."""

    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["metrics"])


def to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def find_checkpoint(run_dir: Path) -> Path:
    """Find the downloaded best checkpoint without renaming it."""

    candidates = sorted(run_dir.glob("best_model*.pth"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one best_model*.pth in {run_dir}, found: {candidates}")
    return candidates[0]


def find_run_dir(run_root: Path, candidates: tuple[str, ...]) -> Path:
    """Resolve downloaded and canonical experiment folder names."""

    matches = [run_root / name for name in candidates if (run_root / name).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one run directory under {run_root}, found: {matches}")
    return matches[0]


def require_file(path: Path) -> Path:
    """Raise when an expected artifact is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def run(cmd: list[str]) -> None:
    """Run a subprocess with visible logs."""

    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
