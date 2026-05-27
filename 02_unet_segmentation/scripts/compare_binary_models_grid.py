"""Render a comparison grid for binary models on the same validation patches."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.kurgan_dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from models.unet_small import build_model
from scripts.train import normalize_modalities, parse_int_list, parse_val_regions


DEFAULT_EXPERIMENTS = [
    "binary_li_no_dice",
    "binary_li_pos_weight_2",
    "binary_li_pos_weight_4",
    "binary_li_only",
    "binary_li_512_no_dice",
]


@dataclass
class ModelSpec:
    """Resolved model configuration for comparison rendering."""

    name: str
    checkpoint: Path
    threshold: float
    image_size: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--output", default="assets/readme/binary_models_comparison.png")
    parser.add_argument("--run-root", default="runs/binary")
    parser.add_argument("--threshold-root", default="runs/threshold_sweep")
    parser.add_argument(
        "--experiment",
        action="append",
        help=(
            "Experiment name to include. Defaults to the known binary UNet runs. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Explicit model spec: name=/path/model.pth,threshold=0.60,image_size=256. "
            "Can be repeated and overrides --experiment defaults when provided."
        ),
    )
    parser.add_argument("--split", choices=["region", "custom_regions", "random"], default="custom_regions")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="*", default=["Li"])
    parser.add_argument("--binary-positive-classes", default="1,2")
    parser.add_argument("--sample-ids", help="Comma-separated sample ids. If set, selection is exact.")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--selection", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-image-size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Render predictions from several binary models on the same sample ids."""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = normalize_modalities(args.modalities)
    binary_positive_classes = parse_int_list(args.binary_positive_classes)

    _, val_df = make_experiment_split(
        load_metadata(args.data_root),
        split=args.split,
        val_region=args.val_region,
        val_regions=parse_val_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=modalities,
    )
    sample_ids = choose_sample_ids(val_df, args)
    if not sample_ids:
        raise ValueError("No sample ids selected for comparison grid")

    specs = resolve_model_specs(args)
    if not specs:
        raise ValueError("No model checkpoints found")

    reference_df = val_df[val_df["sample_id"].astype(str).isin(sample_ids)].copy()
    reference_dataset = KurganSegmentationDataset(
        reference_df,
        args.data_root,
        args.reference_image_size,
        task="binary",
        binary_positive_classes=binary_positive_classes,
    )
    reference_by_id = {str(item["sample_id"]): item for item in reference_dataset}

    model_outputs = {}
    for spec in specs:
        model_outputs[spec.name] = predict_for_sample_ids(
            spec,
            val_df,
            sample_ids,
            args.data_root,
            binary_positive_classes,
            device,
        )

    save_grid(sample_ids, reference_by_id, model_outputs, specs, args.output)
    print(f"Saved model comparison grid to {args.output}")
    print("Models:")
    for spec in specs:
        print(f"  {spec.name}: threshold={spec.threshold:.2f} image_size={spec.image_size}")


def choose_sample_ids(val_df, args: argparse.Namespace) -> list[str]:
    """Choose shared sample ids for all models."""

    available = val_df["sample_id"].astype(str).tolist()
    if args.sample_ids:
        requested = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Sample ids are not in selected validation split: {missing}")
        return requested[: args.max_samples]

    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(available)
    return available[: args.max_samples]


def resolve_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    """Resolve model specs from explicit CLI specs or known run folders."""

    if args.model:
        specs = [parse_model_spec(value) for value in args.model]
    else:
        experiments = args.experiment or DEFAULT_EXPERIMENTS
        specs = [spec_from_experiment(name, args.run_root, args.threshold_root) for name in experiments]

    resolved = []
    for spec in specs:
        if spec.checkpoint.exists():
            resolved.append(spec)
        else:
            print(f"[SKIP] checkpoint not found: {spec.checkpoint}")
    return resolved


def spec_from_experiment(name: str, run_root: str, threshold_root: str) -> ModelSpec:
    """Create a model spec from project run conventions."""

    run_dir = Path(run_root) / name
    checkpoint = first_existing(
        [
            run_dir / "best_model.pth",
            run_dir / f"{name}.pth",
            Path(threshold_root) / name / "best_model.pth",
            Path(threshold_root) / name / f"{name}.pth",
        ]
    )
    threshold = load_best_threshold(Path(threshold_root) / name / "threshold_sweep.json", default=0.5)
    image_size = load_image_size(run_dir / "config.json", default=512 if "512" in name else 256)
    return ModelSpec(name=name, checkpoint=checkpoint, threshold=threshold, image_size=image_size)


def parse_model_spec(value: str) -> ModelSpec:
    """Parse explicit name/checkpoint/threshold/image_size model specs."""

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts or "=" not in parts[0]:
        raise ValueError("--model must start with name=/path/to/checkpoint.pth")
    name, checkpoint = parts[0].split("=", 1)
    values: dict[str, Any] = {"threshold": 0.5, "image_size": 256}
    for part in parts[1:]:
        key, val = part.split("=", 1)
        values[key.strip()] = val.strip()
    return ModelSpec(
        name=name.strip(),
        checkpoint=Path(checkpoint).expanduser(),
        threshold=float(values["threshold"]),
        image_size=int(values["image_size"]),
    )


def predict_for_sample_ids(
    spec: ModelSpec,
    val_df,
    sample_ids: list[str],
    data_root: str,
    binary_positive_classes: list[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Predict masks for the chosen sample ids with one model."""

    selected_df = val_df[val_df["sample_id"].astype(str).isin(sample_ids)].copy()
    dataset = KurganSegmentationDataset(
        selected_df,
        data_root,
        spec.image_size,
        task="binary",
        binary_positive_classes=binary_positive_classes,
    )
    by_id = {str(dataset[idx]["sample_id"]): dataset[idx] for idx in range(len(dataset))}

    checkpoint = torch.load(spec.checkpoint, map_location=device)
    model = build_model("unet_small", in_channels=1, num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    outputs = {}
    for sample_id in sample_ids:
        item = by_id[sample_id]
        image = item["image"].unsqueeze(0).to(device)
        probs = torch.sigmoid(model(image)[:, 0])
        outputs[sample_id] = (probs[0].cpu().numpy() > spec.threshold).astype(np.uint8)
    return outputs


def save_grid(
    sample_ids: list[str],
    reference_by_id: dict[str, dict[str, Any]],
    model_outputs: dict[str, dict[str, np.ndarray]],
    specs: list[ModelSpec],
    output: str | Path,
) -> None:
    """Save Image | GT | model predictions grid."""

    n_cols = 2 + len(specs)
    fig, axes = plt.subplots(len(sample_ids), n_cols, figsize=(3.2 * n_cols, 3.1 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample_id in enumerate(sample_ids):
        item = reference_by_id[sample_id]
        image = stretch(item["image"][0].numpy())
        target = item["mask"].numpy()
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{sample_id} | {item['region']}")
        axes[row, 1].imshow(colorize(target))
        axes[row, 1].set_title("GT")
        for col, spec in enumerate(specs, start=2):
            pred = model_outputs[spec.name][sample_id]
            axes[row, col].imshow(colorize(pred))
            axes[row, col].set_title(f"{spec.name}\nthr={spec.threshold:.2f}")
        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def first_existing(paths: list[Path]) -> Path:
    """Return the first existing path or the first candidate for skip reporting."""

    for path in paths:
        if path.exists():
            return path
    return paths[0]


def load_best_threshold(path: Path, default: float) -> float:
    """Read best threshold from threshold_sweep.json when available."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return float(data.get("best_threshold", default))


def load_image_size(path: Path, default: int) -> int:
    """Read image size from config.json when available."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return int(data.get("image_size", default))


def stretch(image: np.ndarray) -> np.ndarray:
    """Robust image stretch for display."""

    lo, hi = np.percentile(image, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Colorize a binary mask."""

    colors = np.array([[0, 0, 0], [0, 180, 80]], dtype=np.float32) / 255.0
    return colors[np.clip(mask.astype(np.int64), 0, 1)]


if __name__ == "__main__":
    main()
