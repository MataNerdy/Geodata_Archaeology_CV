"""Sweep probability thresholds for trained binary kurgan segmentation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from model import build_model
from train import get_device, normalize_modalities, parse_val_regions


DEFAULT_THRESHOLDS = ",".join(f"{value / 100:.2f}" for value in range(5, 100, 5))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--out-dir")
    parser.add_argument("--output", help="Optional explicit CSV path; kept for old calls")
    parser.add_argument("--task", choices=["binary", "multiclass"])
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=["region", "custom_regions", "random"],
        default="custom_regions",
    )
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run threshold sweep and save CSV, JSON and PNG artifacts."""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_run_config(checkpoint_path)

    task = resolve_value(args.task, checkpoint, config, "task", "binary")
    if task != "binary":
        raise ValueError("threshold_sweep.py currently supports only --task binary")

    image_size = int(resolve_value(args.image_size, checkpoint, config, "image_size", 256))
    modalities = normalize_modalities(args.modalities)
    if modalities is None:
        modalities = normalize_modalities(resolve_value(None, checkpoint, config, "modalities", None))

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output) if args.output else out_dir / "threshold_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "threshold_sweep.json"
    plot_path = out_dir / "threshold_sweep.png"

    thresholds = parse_thresholds(args)
    loader = build_validation_loader(args, image_size=image_size, modalities=modalities)
    probs, targets = collect_probabilities(checkpoint, loader, device)

    rows = [compute_threshold_metrics(probs, targets, threshold) for threshold in thresholds]
    result = pd.DataFrame(rows)
    result.to_csv(csv_path, index=False)

    best_row = result.sort_values(["fg_iou", "fg_dice"], ascending=False).iloc[0].to_dict()
    threshold_05 = result.loc[(result["threshold"] - 0.5).abs().idxmin()].to_dict()
    summary = {
        "checkpoint": str(checkpoint_path),
        "task": task,
        "image_size": image_size,
        "modalities": modalities,
        "best_threshold": float(best_row["threshold"]),
        "best_fg_iou": float(best_row["fg_iou"]),
        "best_fg_dice": float(best_row["fg_dice"]),
        "precision_at_best": float(best_row["precision"]),
        "recall_at_best": float(best_row["recall"]),
        "pixel_accuracy_at_best": float(best_row["pixel_accuracy"]),
        "fg_iou_at_0_5": float(threshold_05["fg_iou"]),
        "delta_iou_vs_0_5": float(best_row["fg_iou"] - threshold_05["fg_iou"]),
        "best_row": best_row,
        "threshold_0_5_row": threshold_05,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_threshold_plot(result, plot_path)
    print_top_thresholds(result, csv_path, json_path, plot_path)


def build_validation_loader(
    args: argparse.Namespace,
    image_size: int,
    modalities: list[str] | None,
) -> DataLoader:
    """Build the validation loader using the same split logic as evaluation."""

    _, val_df = make_experiment_split(
        load_metadata(args.data_root),
        split=args.split,
        val_region=args.val_region,
        val_regions=parse_val_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=modalities,
    )
    dataset = KurganSegmentationDataset(
        val_df,
        args.data_root,
        image_size,
        task="binary",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def collect_probabilities(
    checkpoint: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model once and keep binary probabilities and targets in memory."""

    model = build_model("unet_small", in_channels=1, num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    prob_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)[:, 0]
        prob_batches.append(torch.sigmoid(logits).cpu())
        target_batches.append(batch["mask"].cpu().long())
    if not prob_batches:
        raise ValueError("Validation loader is empty")
    return torch.cat(prob_batches, dim=0), torch.cat(target_batches, dim=0)


def compute_threshold_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Compute binary metrics and pixel counts for one threshold."""

    preds = probs > threshold
    truth = targets > 0
    tp = int((preds & truth).sum().item())
    fp = int((preds & ~truth).sum().item())
    fn = int((~preds & truth).sum().item())
    tn = int((~preds & ~truth).sum().item())

    fg_iou = safe_div(tp, tp + fp + fn)
    fg_dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    pixel_accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "threshold": threshold,
        "fg_iou": fg_iou,
        "fg_dice": fg_dice,
        "pixel_accuracy": pixel_accuracy,
        "precision": precision,
        "recall": recall,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
    }


def save_threshold_plot(result: pd.DataFrame, path: Path) -> None:
    """Save a three-panel threshold sweep plot."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(result["threshold"], result["fg_iou"], marker="o")
    axes[0].set_title("Threshold vs fg_iou")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("fg_iou")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["threshold"], result["fg_dice"], marker="o", color="#2ca02c")
    axes[1].set_title("Threshold vs fg_dice")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("fg_dice")
    axes[1].grid(alpha=0.3)

    axes[2].plot(result["threshold"], result["precision"], marker="o", label="precision")
    axes[2].plot(result["threshold"], result["recall"], marker="o", label="recall")
    axes[2].set_title("Threshold vs precision/recall")
    axes[2].set_xlabel("threshold")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_top_thresholds(
    result: pd.DataFrame,
    csv_path: Path,
    json_path: Path,
    plot_path: Path,
) -> None:
    """Print top thresholds by foreground IoU."""

    print(f"Saved threshold sweep CSV to {csv_path}")
    print(f"Saved threshold sweep JSON to {json_path}")
    print(f"Saved threshold sweep plot to {plot_path}")
    print("Top-5 thresholds by fg_iou:")
    top = result.sort_values(["fg_iou", "fg_dice"], ascending=False).head(5)
    print(top.to_string(index=False))


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    """Parse explicit threshold list or fallback to range arguments."""

    if args.thresholds:
        thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    else:
        thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("Threshold values must be between 0 and 1")
    return thresholds


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    """Create rounded threshold values including the end point."""

    values = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def load_run_config(checkpoint_path: Path) -> dict[str, Any]:
    """Load config.json next to the checkpoint when available."""

    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_value(
    cli_value: Any,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Resolve a value with CLI taking priority over checkpoint args and config."""

    if cli_value is not None:
        return cli_value
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if checkpoint_args.get(key) is not None:
        return checkpoint_args[key]
    if config.get(key) is not None:
        return config[key]
    return default


def safe_div(numerator: int, denominator: int) -> float:
    """Divide safely for metric computation."""

    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
