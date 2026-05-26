"""Select validation examples for README figures.

The script ranks validation samples by per-sample IoU and renders compact grids
for binary, multiclass, and modality-comparison README assets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from model import build_model
from train import normalize_modalities, parse_val_regions


MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 180, 80],
        [220, 40, 40],
    ],
    dtype=np.float32,
) / 255.0


@dataclass
class RankedSample:
    """One sample with its score and prediction."""

    dataset_index: int
    iou: float
    image: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    sample_id: str
    region: str
    modality: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--split", choices=["region", "custom_regions", "random"], default="custom_regions")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=["best", "failures", "modality"], default="best")
    parser.add_argument("--max-samples", type=int, default=5)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Rank validation samples and save a README-ready grid."""

    args = parse_args()
    args.modalities = normalize_modalities(args.modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        task=args.task,
    )
    model = load_model(args.checkpoint, args.task, device)
    ranked = rank_samples(model, dataset, args.task, args.threshold, device, args.num_workers)

    if args.mode == "failures":
        selected = ranked[: args.max_samples]
        title = "Low-IoU validation examples"
        include_iou = True
    elif args.mode == "modality":
        selected = select_one_per_modality(ranked, args.max_samples)
        title = "Representative examples by modality"
        include_iou = True
    else:
        selected = select_best_medium_low(ranked, args.max_samples)
        title = "Best / medium / low-IoU validation examples"
        include_iou = True

    save_grid(selected, args.output, title=title, include_iou=include_iou)


def load_model(checkpoint_path: str | Path, task: str, device: torch.device) -> torch.nn.Module:
    """Load a UNetSmall checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = 1 if task == "binary" else 3
    model = build_model("unet_small", in_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def rank_samples(
    model: torch.nn.Module,
    dataset: KurganSegmentationDataset,
    task: str,
    threshold: float,
    device: torch.device,
    num_workers: int,
) -> list[RankedSample]:
    """Return samples sorted by ascending IoU."""

    loader = DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    ranked: list[RankedSample] = []
    for index, batch in enumerate(loader):
        image_tensor = batch["image"].to(device)
        logits = model(image_tensor)
        if task == "binary":
            prediction = (torch.sigmoid(logits[:, 0]) > threshold).long().cpu()[0].numpy()
            target = (batch["mask"][0].cpu().numpy() > 0).astype(np.uint8)
        else:
            prediction = logits.argmax(dim=1).cpu()[0].numpy()
            target = batch["mask"][0].cpu().numpy()

        ranked.append(
            RankedSample(
                dataset_index=index,
                iou=foreground_iou(prediction, target, task),
                image=batch["image"][0, 0].cpu().numpy(),
                target=target,
                prediction=prediction,
                sample_id=str(batch["sample_id"][0]),
                region=str(batch["region"][0]),
                modality=str(batch["modality"][0]),
            )
        )
    return sorted(ranked, key=lambda sample: sample.iou)


def select_best_medium_low(ranked: list[RankedSample], max_samples: int) -> list[RankedSample]:
    """Pick top, middle and low-IoU samples for a curated grid."""

    if not ranked:
        return []
    n_good = min(3, max_samples)
    good = ranked[-n_good:]
    middle = [ranked[len(ranked) // 2]] if max_samples > n_good else []
    low = [ranked[0]] if max_samples > n_good + len(middle) else []
    return list(reversed(good)) + middle + low


def select_one_per_modality(ranked: list[RankedSample], max_samples: int) -> list[RankedSample]:
    """Pick representative samples close to the median IoU for each modality."""

    selected: list[RankedSample] = []
    for modality in ["Li", "Ae", "SpOr"]:
        candidates = [sample for sample in ranked if sample.modality == modality]
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    return selected[:max_samples]


def save_grid(
    samples: list[RankedSample],
    output: str | Path,
    title: str,
    include_iou: bool = True,
) -> None:
    """Save Image | GT | Prediction | Overlay panels."""

    if not samples:
        raise ValueError("No samples selected")
    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 3.2 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        image = stretch(sample.image)
        title_suffix = f"{sample.modality} | {sample.sample_id}"
        if include_iou:
            title_suffix += f" | IoU={sample.iou:.3f}"
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(title_suffix)
        axes[row, 1].imshow(colorize(sample.target))
        axes[row, 1].set_title("GT")
        axes[row, 2].imshow(colorize(sample.prediction))
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(overlay(sample.prediction), alpha=0.55)
        axes[row, 3].set_title(sample.region)
        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def foreground_iou(prediction: np.ndarray, target: np.ndarray, task: str) -> float:
    """Compute foreground IoU for one sample."""

    if task == "binary":
        pred_fg = prediction > 0
        target_fg = target > 0
    else:
        pred_fg = prediction > 0
        target_fg = target > 0
    intersection = np.logical_and(pred_fg, target_fg).sum()
    union = np.logical_or(pred_fg, target_fg).sum()
    return float(intersection / union) if union > 0 else 0.0


def stretch(image: np.ndarray) -> np.ndarray:
    """Robustly stretch one grayscale image to 0..1."""

    lo, hi = np.percentile(image, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Colorize binary or multiclass masks."""

    return MASK_COLORS[np.clip(mask.astype(np.int64), 0, 2)]


def overlay(mask: np.ndarray) -> np.ndarray:
    """Create RGBA overlay for prediction masks."""

    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask == 1] = [0.0, 0.8, 0.25, 0.65]
    rgba[mask == 2] = [1.0, 0.1, 0.1, 0.65]
    return rgba


if __name__ == "__main__":
    main()
