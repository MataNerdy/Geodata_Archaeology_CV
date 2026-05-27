"""Train DeepLabV3+ for archaeology segmentation tasks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _force_local_package(package_name: str) -> None:
    package_dir = PROJECT_ROOT / package_name
    init_file = package_dir / "__init__.py"
    if not package_dir.exists():
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

for _package_name in ("arch_datasets", "losses", "models", "utils"):
    _force_local_package(_package_name)

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    class_names_for_task,
    load_metadata,
    num_classes_for_task,
)
from losses import CombinedBinaryLoss, CombinedMulticlassLoss
from models.deeplab import build_model
from utils.metrics import (
    binary_metrics_from_confusion,
    confusion_matrix,
    logits_to_predictions,
    multiclass_metrics_from_confusion,
    to_jsonable,
)
from utils.splits import make_split, parse_regions
from utils.visualization import save_prediction_grid


TASKS = ("binary_kurgan", "kurgan_multiclass", "all_classes")


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--encoder", choices=["resnet34", "resnet50", "efficientnet-b0"])
    parser.add_argument("--encoder-weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--split", choices=["region", "custom_regions", "random", "stratified_region_holdout"])
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--class-weights")
    parser.add_argument("--ce-weight", type=float)
    parser.add_argument("--bce-weight", type=float)
    parser.add_argument("--dice-weight", type=float)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--save-samples", type=int)
    return parser.parse_args()


def main() -> None:
    """Run a DeepLab training experiment."""

    args = parse_args()
    config = resolve_config(args)
    set_seed(int(config["seed"]))

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(config["data_root"])
    train_df, val_df = make_split(
        meta,
        split=config["split"],
        val_region=config.get("val_region"),
        val_regions=parse_regions(config.get("val_regions")),
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
        modalities=config.get("modalities"),
    )
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    save_config(config, out_dir / "config_used.yaml")

    train_loader = build_loader(train_df, config, shuffle=True)
    val_loader = build_loader(val_df, config, shuffle=False)

    device = get_device()
    print(f"Device: {device}")
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print_summary_by_region_modality(val_df)

    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=num_classes_for_task(config["task"]),
    ).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_metric = -float("inf")
    best_epoch = 0
    patience = int(config["patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, config["task"], "train")
        val_metrics = run_epoch(model, val_loader, criterion, None, device, config["task"], "val")
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        selection_key = "val_fg_iou" if config["task"] == "binary_kurgan" else "val_mean_fg_iou"
        current_metric = float(val_metrics.get(selection_key, float("nan")))
        print(
            f"Epoch {epoch:03d}: train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} {selection_key}={current_metric:.4f}"
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "selection_metric": selection_key,
                    "selection_score": current_metric,
                },
                out_dir / "best_model.pth",
            )
            print(f"Saved best_model.pth ({selection_key}={current_metric:.4f})")

        if patience > 0 and epoch - best_epoch >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(
                {
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "task": config["task"],
                    "config": config,
                }
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if int(config["save_samples"]) > 0:
        save_prediction_grid(
            model,
            val_loader,
            device,
            out_dir / "prediction_examples.png",
            task=config["task"],
            max_samples=int(config["save_samples"]),
            threshold=float(config["threshold"]),
        )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
    split_name: str,
) -> dict[str, float]:
    """Run train or eval epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    loss_parts_total: dict[str, float] = {}
    num_classes = 2 if task == "binary_kurgan" else num_classes_for_task(task)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss, loss_parts = criterion(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()
        preds = logits_to_predictions(logits.detach(), task).cpu()
        matrix += confusion_matrix(preds, masks.detach().cpu(), num_classes)
        total_loss += float(loss.detach().cpu())
        for key, value in loss_parts.items():
            loss_parts_total[key] = loss_parts_total.get(key, 0.0) + float(value)

    n_batches = max(len(loader), 1)
    metrics = {f"{split_name}_loss": total_loss / n_batches}
    for key, value in loss_parts_total.items():
        metrics[f"{split_name}_{key}"] = value / n_batches
    if task == "binary_kurgan":
        metrics.update(binary_metrics_from_confusion(matrix, prefix=f"{split_name}_"))
    else:
        metrics.update(
            multiclass_metrics_from_confusion(
                matrix,
                class_names_for_task(task),
                prefix=f"{split_name}_",
            )
        )
    return metrics


def build_loader(meta: pd.DataFrame, config: dict[str, Any], shuffle: bool) -> DataLoader:
    """Build DataLoader from metadata and config."""

    dataset = ArchaeologySegmentationDataset(
        meta,
        config["data_root"],
        image_size=int(config["image_size"]),
        task=config["task"],
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    """Build loss for task."""

    if config["task"] == "binary_kurgan":
        return CombinedBinaryLoss(
            bce_weight=float(config["bce_weight"]),
            dice_weight=float(config["dice_weight"]),
            pos_weight=config.get("pos_weight"),
        )
    return CombinedMulticlassLoss(
        num_classes=num_classes_for_task(config["task"]),
        ce_weight=float(config["ce_weight"]),
        dice_weight=float(config["dice_weight"]),
        class_weights=parse_float_list(config.get("class_weights")),
    )


def default_config() -> dict[str, Any]:
    """Return default config values."""

    return {
        "data_root": "../datasets/segmentation_dataset",
        "out_dir": "runs/binary/deeplab_binary_li",
        "task": "binary_kurgan",
        "encoder": "resnet34",
        "encoder_weights": None,
        "epochs": 2,
        "patience": 0,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "image_size": 256,
        "split": "custom_regions",
        "val_region": None,
        "val_regions": "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км",
        "val_fraction": 0.2,
        "modalities": ["Li"],
        "class_weights": None,
        "ce_weight": 1.0,
        "bce_weight": 1.0,
        "dice_weight": 0.0,
        "pos_weight": None,
        "num_workers": 0,
        "seed": 42,
        "threshold": 0.5,
        "save_samples": 6,
    }


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load YAML config and apply CLI overrides."""

    config = default_config()
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    config["modalities"] = normalize_modalities(config.get("modalities"))
    if config["task"] == "binary_kurgan":
        config["class_weights"] = None
    else:
        config["pos_weight"] = None
    return config


def normalize_modalities(value: object) -> list[str] | None:
    """Normalize modality CLI/config value."""

    if not value:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    modalities = [part.strip() for part in parts if part.strip()]
    return modalities or None


def parse_float_list(value: object) -> list[float] | None:
    """Parse comma-separated floats."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def save_config(config: dict[str, Any], path: Path) -> None:
    """Save resolved YAML config."""

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(config), handle, allow_unicode=True, sort_keys=True)


def set_seed(seed: int) -> None:
    """Set random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return best available torch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_summary_by_region_modality(meta: pd.DataFrame) -> None:
    """Print validation split summary."""

    print("Validation samples by region/modality:")
    summary = meta.groupby(["region", "modality"]).size().reset_index(name="samples")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
