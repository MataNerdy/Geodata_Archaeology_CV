"""Train UNetSmall for 3-class kurgan semantic segmentation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from losses import CombinedLoss
from metrics import (
    confusion_matrix,
    flatten_modality_metrics,
    metrics_from_confusion,
    update_modality_confusions,
)
from model import build_model
from visualize_predictions import save_prediction_grid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a training run."""

    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(defaults.data_root))
    parser.add_argument("--out-dir", default=str(defaults.out_dir))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument(
        "--split",
        choices=["region", "custom_regions", "random"],
        default="region",
    )
    parser.add_argument("--val-region")
    parser.add_argument(
        "--val-regions",
        help="Comma-separated validation regions for --split custom_regions",
    )
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--modalities", nargs="*", help="Optional modality filter: Li Ae SpOr")
    parser.add_argument("--ce-weight", type=float, default=defaults.ce_weight)
    parser.add_argument("--dice-weight", type=float, default=defaults.dice_weight)
    parser.add_argument(
        "--class-weights",
        help="Comma-separated CE class weights, for example: 0.2,1.0,3.0",
    )
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--save-samples", type=int, default=6)
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 3,
) -> dict[str, float]:
    """Train for one epoch and return aggregate metrics."""

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    modality_confusions: dict[str, torch.Tensor] = {}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, loss_parts = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.detach().argmax(dim=1).cpu()
        targets = masks.detach().cpu()
        matrix += confusion_matrix(preds, targets, num_classes)
        update_modality_confusions(
            modality_confusions,
            preds,
            targets,
            list(batch["modality"]),
            num_classes,
        )

        total_loss += float(loss.detach().cpu())
        total_ce += loss_parts["ce_loss"]
        total_dice_loss += loss_parts["dice_loss"]

    n_batches = max(len(loader), 1)
    metrics = {
        "train_loss": total_loss / n_batches,
        "train_ce_loss": total_ce / n_batches,
        "train_dice_loss": total_dice_loss / n_batches,
    }
    metrics.update(metrics_from_confusion(matrix, prefix="train_"))
    metrics.update(flatten_modality_metrics(modality_confusions, split="train"))
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    split_name: str = "val",
    num_classes: int = 3,
) -> dict[str, float]:
    """Evaluate a model on a loader and return aggregate metrics."""

    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    modality_confusions: dict[str, torch.Tensor] = {}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss, loss_parts = criterion(logits, masks)

        preds = logits.argmax(dim=1).cpu()
        targets = masks.cpu()
        matrix += confusion_matrix(preds, targets, num_classes)
        update_modality_confusions(
            modality_confusions,
            preds,
            targets,
            list(batch["modality"]),
            num_classes,
        )

        total_loss += float(loss.cpu())
        total_ce += loss_parts["ce_loss"]
        total_dice_loss += loss_parts["dice_loss"]

    n_batches = max(len(loader), 1)
    metrics = {
        f"{split_name}_loss": total_loss / n_batches,
        f"{split_name}_ce_loss": total_ce / n_batches,
        f"{split_name}_dice_loss": total_dice_loss / n_batches,
    }
    metrics.update(metrics_from_confusion(matrix, prefix=f"{split_name}_"))
    metrics.update(flatten_modality_metrics(modality_confusions, split=split_name))
    return metrics


def main() -> None:
    """Run the training experiment."""

    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_val_regions = parse_val_regions(args.val_regions)
    train_df, val_df = make_experiment_split(
        load_metadata(args.data_root),
        split=args.split,
        val_region=args.val_region,
        val_regions=parsed_val_regions,
        val_fraction=args.val_fraction,
        seed=args.seed,
        modalities=args.modalities,
    )
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_dataset = KurganSegmentationDataset(train_df, args.data_root, args.image_size)
    val_dataset = KurganSegmentationDataset(val_df, args.data_root, args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = build_model("unet_small", in_channels=1, num_classes=3).to(device)
    criterion = CombinedLoss(
        num_classes=3,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        class_weights=parse_class_weights(args.class_weights),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    save_config(args, out_dir, train_df, val_df, device)
    history: list[dict[str, float]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    if args.split == "custom_regions":
        print_custom_region_summary(val_df)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_loader(model, val_loader, criterion, device, split_name="val")
        lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        score = val_metrics["val_mean_fg_iou"]
        scheduler.step(score)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_fg_iou={val_metrics['val_fg_iou']:.4f} | "
            f"val_mean_fg_iou={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mean_fg_iou": best_score,
                "args": vars(args),
            }
            torch.save(checkpoint, out_dir / "best_model.pth")
            save_prediction_grid(
                model,
                val_loader,
                device,
                out_dir / "prediction_examples.png",
                max_samples=args.save_samples,
            )
            print(f"Saved new best model: {out_dir / 'best_model.pth'}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    print(f"Best epoch: {best_epoch} | best val mean foreground IoU: {best_score:.4f}")


def parse_class_weights(value: str | None) -> list[float] | None:
    """Parse comma-separated class weights."""

    if not value:
        return None
    weights = [float(item.strip()) for item in value.split(",")]
    if len(weights) != 3:
        raise ValueError("--class-weights must contain exactly 3 values")
    return weights


def parse_val_regions(value: str | None) -> list[str] | None:
    """Parse comma-separated validation regions."""

    if value is None:
        return None
    regions = [item.strip() for item in value.split(",") if item.strip()]
    if not regions:
        raise ValueError("--val-regions must contain at least one region")
    return regions


def print_custom_region_summary(val_df: pd.DataFrame) -> None:
    """Print selected validation regions and sample counts by region/modality."""

    val_regions = sorted(val_df["region"].astype(str).unique().tolist())
    print("Custom validation regions:")
    for region in val_regions:
        print(f"  - {region}")

    summary = (
        val_df.groupby(["region", "modality"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["region", "modality"])
    )
    print("Validation samples by region/modality:")
    print(summary.to_string(index=False))


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_config(
    args: argparse.Namespace,
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
) -> None:
    """Save run configuration and split summary."""

    payload = vars(args).copy()
    payload.update(
        {
            "device": str(device),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "train_regions": sorted(train_df["region"].astype(str).unique().tolist()),
            "val_regions": sorted(val_df["region"].astype(str).unique().tolist()),
            "modalities": sorted(
                set(train_df["modality"].astype(str)) | set(val_df["modality"].astype(str))
            ),
        }
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
