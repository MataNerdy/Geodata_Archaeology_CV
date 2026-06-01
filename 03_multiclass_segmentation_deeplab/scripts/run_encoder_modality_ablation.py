"""Run the diagnostic DeepLabV3+ encoder/modalities ablation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    ("resnet34_li", "resnet34", "Li"),
    ("resnet50_li", "resnet50", "Li"),
    ("resnet34_all_modalities", "resnet34", None),
    ("resnet50_all_modalities", "resnet50", None),
)


def parse_args() -> argparse.Namespace:
    """Parse ablation runner CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split-dir",
        default="splits/archaeology_5class_encoder_modality_ablation_raw_v1",
    )
    parser.add_argument("--run-root", default="runs/encoder_modality_ablation")
    parser.add_argument(
        "--config",
        default="configs/archaeology_5class_encoder_modality_ablation_raw_v1.yaml",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train and evaluate the four diagnostic experiments."""

    args = parse_args()
    split_dir = Path(args.split_dir)
    train_split = split_dir / "train_split.csv"
    val_split = split_dir / "val_split.csv"
    require_file(train_split)
    require_file(val_split)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    print("[diagnostic] Encoder/modalities ablation only. Primary benchmark remains research_split_v1 Stage C.")
    print(f"[diagnostic] Frozen raw train split: {train_split}")
    print(f"[diagnostic] Frozen raw val split: {val_split}")

    for experiment, encoder, modalities in EXPERIMENTS:
        run_dir = run_root / experiment
        checkpoint = run_dir / "best_model.pth"
        if args.skip_existing and checkpoint.exists() and (run_dir / "evaluation.json").exists():
            print(f"[diagnostic] Skip existing: {experiment}")
            continue
        print(f"[diagnostic] Starting experiment: {experiment}")
        print(f"[diagnostic] Encoder: {encoder}")
        print(f"[diagnostic] Modalities: {modalities or 'all (no modality filter)'}")
        run(train_command(args, run_dir, encoder, modalities, train_split, val_split))
        run(evaluate_command(args, run_dir, encoder, modalities, train_split, val_split))

    compare_cmd = [
        args.python_bin,
        "-u",
        "scripts/compare_encoder_modality_ablation.py",
        "--run-root",
        str(run_root),
    ]
    run(compare_cmd)
    print(f"[diagnostic] Done. Results: {run_root}")


def train_command(
    args: argparse.Namespace,
    run_dir: Path,
    encoder: str,
    modalities: str | None,
    train_split: Path,
    val_split: Path,
) -> list[str]:
    """Build one training command."""

    cmd = [
        args.python_bin,
        "-u",
        "scripts/train.py",
        "--config",
        args.config,
        "--data-root",
        args.data_root,
        "--out-dir",
        str(run_dir),
        "--encoder",
        encoder,
        "--split",
        "frozen",
        "--train-split-csv",
        str(train_split),
        "--val-split-csv",
        str(val_split),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if modalities:
        cmd.extend(["--modalities", modalities])
    return cmd


def evaluate_command(
    args: argparse.Namespace,
    run_dir: Path,
    encoder: str,
    modalities: str | None,
    train_split: Path,
    val_split: Path,
) -> list[str]:
    """Build one pixel evaluation command."""

    cmd = [
        args.python_bin,
        "-u",
        "scripts/evaluate.py",
        "--checkpoint",
        str(run_dir / "best_model.pth"),
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
        "8",
        "--num-workers",
        str(args.num_workers),
        "--split",
        "frozen",
        "--train-split-csv",
        str(train_split),
        "--val-split-csv",
        str(val_split),
        "--eval-mode",
        "pixel",
    ]
    if modalities:
        cmd.extend(["--modalities", modalities])
    return cmd


def run(cmd: list[str]) -> None:
    """Run a subprocess with a visible command log."""

    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def require_file(path: Path) -> None:
    """Raise when a required split artifact is absent."""

    if not path.exists():
        raise FileNotFoundError(f"Required split artifact not found: {path}")


if __name__ == "__main__":
    main()
