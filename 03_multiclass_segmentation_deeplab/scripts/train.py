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
from torch.utils.data import DataLoader, WeightedRandomSampler

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    class_names_for_task,
    filter_multiclass_metadata,
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


TASKS = ("binary_kurgan", "kurgan_multiclass", "all_classes", "archaeology_5class")


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
    parser.add_argument("--optimizer", choices=["adamw", "adam"])
    parser.add_argument("--scheduler", choices=["none", "plateau"])
    parser.add_argument("--scheduler-factor", type=float)
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--split", choices=["region", "custom_regions", "random", "stratified_region_holdout", "frozen"])
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--train-split-csv")
    parser.add_argument("--val-split-csv")
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
    parser.add_argument("--use-weighted-sampler", action="store_true", default=None)
    parser.add_argument("--sampler-mode", default=None, choices=["class_name"])
    parser.add_argument("--use-metadata-filtering", action="store_true", default=None)
    parser.add_argument("--max-crop-size", type=float)
    parser.add_argument("--max-objects-in-patch", type=int)
    parser.add_argument("--allowed-classes")
    parser.add_argument("--exclude-touches-border", action="store_true", default=None)
    parser.add_argument("--min-foreground-pixels", type=int)
    return parser.parse_args()


def main() -> None:
    """Run a DeepLab training experiment."""

    args = parse_args()
    config = resolve_config(args)
    set_seed(int(config["seed"]))

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(config["data_root"])
    if bool(config.get("use_metadata_filtering")):
        before_count = len(meta)
        meta = filter_multiclass_metadata(
            meta,
            allowed_classes=parse_str_list(config.get("allowed_classes")),
            max_crop_size=config.get("max_crop_size"),
            max_objects_in_patch=config.get("max_objects_in_patch"),
            exclude_touches_border=bool(config.get("exclude_touches_border")),
            min_foreground_pixels=config.get("min_foreground_pixels"),
        )
        print(f"Metadata filtering: {before_count} -> {len(meta)} samples")
    train_df, val_df = make_split(
        meta,
        split=config["split"],
        val_region=config.get("val_region"),
        val_regions=parse_regions(config.get("val_regions")),
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
        modalities=config.get("modalities"),
        train_split_csv=config.get("train_split_csv"),
        val_split_csv=config.get("val_split_csv"),
    )
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    save_config(config, out_dir / "config_used.yaml")

    train_loader = build_loader(train_df, config, shuffle=True, use_sampler=should_use_weighted_sampler(config))
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
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history: list[dict[str, float | int]] = []
    best_metric = -float("inf")
    best_epoch = 0
    patience = int(config["patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, config["task"], "train", config)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, config["task"], "val", config)
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        selection_key = "val_fg_iou" if config["task"] == "binary_kurgan" else "val_mean_fg_iou"
        current_metric = float(val_metrics.get(selection_key, float("nan")))
        print(
            f"Epoch {epoch:03d}: train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} {selection_key}={current_metric:.4f}"
        )
        if scheduler is not None:
            scheduler.step(current_metric)

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
    config: dict[str, Any] | None = None,
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
                grad_clip_norm = None if config is None else config.get("grad_clip_norm")
                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
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


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config."""

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build optional learning-rate scheduler."""

    scheduler_name = str(config.get("scheduler", "none")).lower()
    if scheduler_name in {"", "none", "null"}:
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_loader(
    meta: pd.DataFrame,
    config: dict[str, Any],
    shuffle: bool,
    use_sampler: bool = False,
) -> DataLoader:
    """Build DataLoader from metadata and config."""

    dataset = ArchaeologySegmentationDataset(
        meta,
        config["data_root"],
        image_size=int(config["image_size"]),
        task=config["task"],
    )
    sampler = build_weighted_sampler(dataset.meta, config) if use_sampler else None
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def should_use_weighted_sampler(config: dict[str, Any]) -> bool:
    """Return whether weighted sampler should be enabled."""

    if config.get("use_weighted_sampler") is not None:
        return bool(config.get("use_weighted_sampler"))
    return config["task"] == "archaeology_5class"


def build_weighted_sampler(
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> WeightedRandomSampler:
    """Build notebook-style inverse-frequency sampler for rare archaeology classes."""

    mode = config.get("sampler_mode") or "class_name"
    if mode != "class_name":
        raise ValueError(f"Unsupported sampler_mode: {mode}")
    if "class_name" in meta.columns:
        labels = meta["class_name"].astype(str)
    else:
        labels = infer_sample_labels_from_mask_pixels(meta)
    counts = labels.value_counts().to_dict()
    weights = labels.map(lambda label: 1.0 / max(counts[str(label)], 1)).astype(float).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def infer_sample_labels_from_mask_pixels(meta: pd.DataFrame) -> pd.Series:
    """Infer a dominant foreground class from mask pixel count columns."""

    pixel_cols = [col for col in meta.columns if col.startswith("mask_") and col.endswith("_pixels") and col != "mask_bg_pixels"]
    if not pixel_cols:
        return pd.Series(["unknown"] * len(meta), index=meta.index)
    dominant = meta[pixel_cols].idxmax(axis=1)
    return dominant.str.removeprefix("mask_").str.removesuffix("_pixels")


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
        "optimizer": "adamw",
        "scheduler": "none",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "grad_clip_norm": None,
        "image_size": 256,
        "split": "custom_regions",
        "val_region": None,
        "val_regions": "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км",
        "val_fraction": 0.2,
        "train_split_csv": None,
        "val_split_csv": None,
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
        "use_weighted_sampler": None,
        "sampler_mode": "class_name",
        "use_metadata_filtering": False,
        "max_crop_size": None,
        "max_objects_in_patch": None,
        "allowed_classes": None,
        "exclude_touches_border": False,
        "min_foreground_pixels": None,
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
    if config["task"] == "archaeology_5class" and config.get("use_weighted_sampler") is None:
        config["use_weighted_sampler"] = True
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


def parse_str_list(value: object) -> list[str] | None:
    """Parse comma-separated strings."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
