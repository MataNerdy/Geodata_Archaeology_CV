"""Compare 5-class DeepLab models on the same validation patches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import ArchaeologySegmentationDataset, load_metadata, num_classes_for_task
from models.deeplab import build_model
from utils.metrics import logits_to_predictions
from utils.splits import make_split, parse_regions
from utils.visualization import colorize_mask, mask_overlay, stretch


DEFAULT_MODEL_NAMES = [
    "resnet34_li",
    "resnet50_li",
    "resnet34_all",
    "resnet50_all",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated checkpoint paths in comparison order",
    )
    parser.add_argument(
        "--model-names",
        help="Comma-separated display names. Defaults to ResNet34/50 Li/all labels.",
    )
    parser.add_argument("--output", default="runs/multiclass/archaeology_5class_comparison.png")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--split")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Build a comparison grid."""

    args = parse_args()
    checkpoint_paths = [Path(item.strip()) for item in args.checkpoints.split(",") if item.strip()]
    if not checkpoint_paths:
        raise ValueError("--checkpoints must contain at least one path")
    model_names = parse_model_names(args.model_names, len(checkpoint_paths))
    checkpoints = [torch.load(path, map_location="cpu") for path in checkpoint_paths]
    base_config = dict(checkpoints[0].get("config", {}))
    base_config.update({key: value for key, value in vars(args).items() if value is not None})
    base_config["task"] = "archaeology_5class"
    base_config.setdefault("image_size", 256)
    base_config.setdefault("split", "custom_regions")
    base_config.setdefault("val_fraction", 0.2)

    _, val_df = make_split(
        load_metadata(base_config["data_root"]),
        split=base_config["split"],
        val_region=base_config.get("val_region"),
        val_regions=parse_regions(base_config.get("val_regions")),
        val_fraction=float(base_config["val_fraction"]),
        modalities=normalize_modalities(base_config.get("modalities")),
    )
    dataset = ArchaeologySegmentationDataset(
        val_df,
        base_config["data_root"],
        image_size=int(base_config["image_size"]),
        task="archaeology_5class",
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for checkpoint in checkpoints:
        config = dict(checkpoint.get("config", base_config))
        model = build_model(
            encoder_name=config.get("encoder", "resnet34"),
            encoder_weights=config.get("encoder_weights"),
            in_channels=1,
            classes=num_classes_for_task("archaeology_5class"),
        ).to(device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        models.append(model)

    records = collect_predictions(models, loader, device)
    selected = select_hard_examples(records, max_samples=int(args.max_samples))
    save_comparison_grid(selected, model_names, args.output)
    print(f"Saved comparison grid to {args.output}")


def collect_predictions(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    """Collect images, masks, predictions and per-sample scores."""

    records = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()
        preds_by_model = []
        for model in models:
            preds_by_model.append(logits_to_predictions(model(images), "archaeology_5class").cpu().numpy())

        for index in range(images.shape[0]):
            gt = masks[index]
            preds = [preds[index] for preds in preds_by_model]
            scores = [mean_foreground_iou(gt, pred) for pred in preds]
            records.append(
                {
                    "image": images[index, 0].cpu().numpy(),
                    "gt": gt,
                    "preds": preds,
                    "score": float(np.nanmean(scores)) if scores else np.nan,
                    "sample_id": batch["sample_id"][index],
                    "region": batch["region"][index],
                    "modality": batch["modality"][index],
                }
            )
    return records


def select_hard_examples(
    records: list[dict[str, object]],
    max_samples: int,
) -> list[dict[str, object]]:
    """Select low/mid quality examples where class confusion is visible."""

    valid = [record for record in records if not np.isnan(record["score"])]
    if not valid:
        return records[:max_samples]
    valid = sorted(valid, key=lambda item: item["score"])
    low_count = max(1, max_samples // 2)
    selected = valid[:low_count]
    selected_ids = {record["sample_id"] for record in selected}
    if len(selected) < max_samples:
        mid_start = max(0, len(valid) // 2 - (max_samples - len(selected)) // 2)
        for record in valid[mid_start:]:
            if record["sample_id"] not in selected_ids:
                selected.append(record)
                selected_ids.add(record["sample_id"])
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]


def save_comparison_grid(
    records: list[dict[str, object]],
    model_names: list[str],
    output: str | Path,
) -> None:
    """Save Image | GT | model predictions comparison grid."""

    n_rows = len(records)
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, record in enumerate(records):
        image = stretch(record["image"])
        gt = record["gt"]
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(
            f"{record['sample_id']} | {record['modality']}\n{record['region']}"
        )
        axes[row_index, 1].imshow(colorize_mask(gt, "archaeology_5class"))
        axes[row_index, 1].set_title("GT")

        for model_index, (name, pred) in enumerate(zip(model_names, record["preds"], strict=False)):
            col = model_index + 2
            axes[row_index, col].imshow(image, cmap="gray")
            axes[row_index, col].imshow(mask_overlay(pred, "archaeology_5class"), alpha=0.65)
            axes[row_index, col].set_title(f"{name}\nfg IoU={mean_foreground_iou(gt, pred):.3f}")

        for axis in axes[row_index]:
            axis.axis("off")

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mean_foreground_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute mean IoU over classes 1..5 for one sample."""

    values = []
    for class_id in range(1, 6):
        gt_mask = gt == class_id
        pred_mask = pred == class_id
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        values.append(intersection / union)
    return float(np.mean(values)) if values else np.nan


def parse_model_names(value: str | None, expected: int) -> list[str]:
    """Parse model display names."""

    if value:
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = DEFAULT_MODEL_NAMES[:expected]
    if len(names) != expected:
        raise ValueError(f"Expected {expected} model names, got {len(names)}")
    return names


def normalize_modalities(value: object) -> list[str] | None:
    """Normalize modality filter."""

    if not value:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return [part.strip() for part in parts if part.strip()] or None


if __name__ == "__main__":
    main()
