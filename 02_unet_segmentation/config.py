"""Shared configuration defaults for kurgan segmentation experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


CLASS_NAMES = {
    0: "background",
    1: "whole_kurgan",
    2: "damaged_kurgan",
}

SUPPORTED_MODALITIES = ("Li", "Ae", "SpOr")


@dataclass(frozen=True)
class TrainConfig:
    """Default training parameters used by the CLI scripts."""

    data_root: Path = Path("../datasets/segmentation_dataset")
    out_dir: Path = Path("runs/unet_kurgans_baseline")
    image_size: int = 256
    num_classes: int = 3
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    num_workers: int = 0
    seed: int = 42
    val_fraction: float = 0.2
    class_weights: Tuple[float, float, float] | None = None
