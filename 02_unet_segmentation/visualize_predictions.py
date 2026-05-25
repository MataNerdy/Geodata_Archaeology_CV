"""Save prediction grids for trained kurgan segmentation models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from model import build_model


MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 180, 80],
        [220, 40, 40],
    ],
    dtype=np.float32,
) / 255.0


def save_prediction_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str | Path,
    max_samples: int = 6,
) -> None:
    """Run the model on one loader batch and save image/GT/prediction panels."""

    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"]
    sample_ids = batch["sample_id"]
    regions = batch["region"]
    modalities = batch["modality"]

    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()

    n_samples = min(max_samples, images.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n_samples):
        image = images[idx, 0].detach().cpu().numpy()
        image = _stretch(image)
        gt = masks[idx].numpy()
        pred = preds[idx].numpy()

        axes[idx, 0].imshow(image, cmap="gray")
        axes[idx, 0].set_title(f"{sample_ids[idx]} | {modalities[idx]}")
        axes[idx, 1].imshow(_colorize(gt))
        axes[idx, 1].set_title("GT")
        axes[idx, 2].imshow(_colorize(pred))
        axes[idx, 2].set_title("Prediction")
        axes[idx, 3].imshow(image, cmap="gray")
        axes[idx, 3].imshow(_overlay(pred), alpha=0.55)
        axes[idx, 3].set_title(str(regions[idx]))

        for axis in axes[idx]:
            axis.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a UNetSmall model from a training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model("unet_small", in_channels=1, num_classes=3).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="prediction_examples.png")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=["region", "custom_regions", "random"],
        default="region",
    )
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--max-samples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    _, val_df = make_experiment_split(
        load_metadata(args.data_root),
        args.split,
        val_region=args.val_region,
        val_regions=parse_val_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=args.modalities,
    )
    dataset = KurganSegmentationDataset(val_df, args.data_root, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_checkpoint_model(args.checkpoint, device)
    save_prediction_grid(model, loader, device, args.output, args.max_samples)
    print(f"Saved predictions to {args.output}")


def _stretch(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def parse_val_regions(value: str | None) -> list[str] | None:
    if value is None:
        return None
    regions = [item.strip() for item in value.split(",") if item.strip()]
    if not regions:
        raise ValueError("--val-regions must contain at least one region")
    return regions


def _colorize(mask: np.ndarray) -> np.ndarray:
    return MASK_COLORS[np.clip(mask, 0, 2)]


def _overlay(mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask == 1] = [0.0, 0.8, 0.25, 0.65]
    rgba[mask == 2] = [1.0, 0.1, 0.1, 0.65]
    return rgba


if __name__ == "__main__":
    main()
