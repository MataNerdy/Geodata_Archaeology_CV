"""Sweep probability thresholds for binary kurgan segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from metrics import binary_metrics_from_confusion, confusion_matrix
from model import build_model
from train import get_device, normalize_modalities, parse_val_regions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-size", type=int, default=256)
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
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run threshold sweep and save CSV."""

    args = parse_args()
    args.modalities = normalize_modalities(args.modalities)
    device = get_device()

    _, val_df = make_experiment_split(
        load_metadata(args.data_root),
        split=args.split,
        val_region=args.val_region,
        val_regions=parse_val_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=args.modalities,
    )
    dataset = KurganSegmentationDataset(
        val_df,
        args.data_root,
        args.image_size,
        task="binary",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model("unet_small", in_channels=1, num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    thresholds = make_thresholds(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    matrices = {
        threshold: torch.zeros((2, 2), dtype=torch.int64)
        for threshold in thresholds
    }

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].cpu()
        probs = torch.sigmoid(model(images)[:, 0]).cpu()
        for threshold in thresholds:
            preds = (probs > threshold).long()
            matrices[threshold] += confusion_matrix(preds, targets, num_classes=2)

    rows = []
    for threshold in thresholds:
        metrics = binary_metrics_from_confusion(matrices[threshold])
        rows.append({"threshold": threshold, **metrics})

    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "threshold_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)

    best_idx = result["fg_iou"].idxmax()
    best = result.loc[best_idx]
    print(f"Saved threshold sweep to {output}")
    print(
        "Best threshold by fg_iou: "
        f"{best['threshold']:.2f} | "
        f"fg_iou={best['fg_iou']:.4f} | "
        f"fg_dice={best['fg_dice']:.4f} | "
        f"pixel_accuracy={best['pixel_accuracy']:.4f}"
    )


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    """Create rounded threshold values including the end point."""

    values = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


if __name__ == "__main__":
    main()
