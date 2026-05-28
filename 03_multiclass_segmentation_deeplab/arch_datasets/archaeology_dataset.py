"""Dataset utilities for archaeology semantic segmentation patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REQUIRED_METADATA_COLUMNS = {"sample_id", "region", "modality"}
BINARY_POSITIVE_CLASSES = (1, 2)
VALID_SOURCE_CLASSES = set(range(6))

TASK_CLASS_NAMES = {
    "binary_kurgan": {0: "background", 1: "any_kurgan"},
    "kurgan_multiclass": {
        0: "background",
        1: "kurgany_tselye",
        2: "kurgany_povrezhdennye",
    },
    "all_classes": {
        0: "background",
        1: "kurgany_tselye",
        2: "kurgany_povrezhdennye",
        3: "gorodishcha",
        4: "fortifikatsii",
        5: "arkhitektury",
    },
    "archaeology_5class": {
        0: "background",
        1: "kurgany_tselye",
        2: "kurgany_povrezhdennye",
        3: "gorodishcha",
        4: "fortifikatsii",
        5: "arkhitektury",
    },
}


def sample_id_to_name(sample_id: object) -> str:
    """Convert sample id values to six-digit file stems."""

    return str(sample_id).strip().zfill(6)


def validate_data_root(data_root: str | Path) -> tuple[Path, Path, Path]:
    """Validate dataset root and return metadata/images/masks paths."""

    root = Path(data_root)
    metadata_path = root / "metadata.csv"
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")
    return metadata_path, images_dir, masks_dir


def load_metadata(data_root: str | Path) -> pd.DataFrame:
    """Load metadata.csv and verify that sample ids match .npy files."""

    metadata_path, images_dir, masks_dir = validate_data_root(data_root)
    meta = pd.read_csv(metadata_path, dtype={"sample_id": str})
    missing_cols = REQUIRED_METADATA_COLUMNS - set(meta.columns)
    if missing_cols:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing_cols)}")
    if meta.empty:
        raise ValueError(f"metadata.csv has no rows: {metadata_path}")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].map(sample_id_to_name)
    if meta["sample_id"].duplicated().any():
        duplicated = sorted(meta.loc[meta["sample_id"].duplicated(), "sample_id"].unique())
        raise ValueError(f"Duplicated sample_id values: {duplicated[:10]}")

    image_ids = {path.stem for path in images_dir.glob("*.npy")}
    mask_ids = {path.stem for path in masks_dir.glob("*.npy")}
    meta_ids = set(meta["sample_id"])
    missing_images = sorted(meta_ids - image_ids)
    missing_masks = sorted(meta_ids - mask_ids)
    if missing_images:
        raise FileNotFoundError(f"Missing image .npy files for sample_id: {missing_images[:10]}")
    if missing_masks:
        raise FileNotFoundError(f"Missing mask .npy files for sample_id: {missing_masks[:10]}")
    return meta


def filter_modalities(meta: pd.DataFrame, modalities: Iterable[str] | None) -> pd.DataFrame:
    """Filter metadata by modality names."""

    if not modalities:
        return meta.reset_index(drop=True)
    wanted = {str(item) for item in modalities}
    present = set(meta["modality"].astype(str))
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(f"Requested modalities are absent from metadata: {missing}")
    return meta[meta["modality"].astype(str).isin(wanted)].reset_index(drop=True)


def filter_multiclass_metadata(
    meta: pd.DataFrame,
    allowed_classes: Iterable[str] | None = None,
    max_crop_size: float | None = None,
    max_objects_in_patch: int | None = None,
    exclude_touches_border: bool = False,
    min_foreground_pixels: int | None = None,
) -> pd.DataFrame:
    """Filter noisy or ambiguous multiclass patches using notebook-style metadata rules."""

    filtered = meta.copy()
    class_names = list(allowed_classes) if allowed_classes else [
        "kurgany_tselye",
        "kurgany_povrezhdennye",
        "gorodishcha",
        "fortifikatsii",
        "arkhitektury",
    ]
    pixel_cols = [
        f"mask_{class_name}_pixels"
        for class_name in class_names
        if f"mask_{class_name}_pixels" in filtered.columns
    ]

    if allowed_classes and "class_name" in filtered.columns:
        filtered = filtered[filtered["class_name"].isin(class_names)].copy()
    if max_crop_size is not None and "crop_size" in filtered.columns:
        filtered = filtered[filtered["crop_size"] <= float(max_crop_size)].copy()
    if max_objects_in_patch is not None and "n_objects_in_patch" in filtered.columns:
        filtered = filtered[filtered["n_objects_in_patch"] <= int(max_objects_in_patch)].copy()
    if exclude_touches_border and "touches_border" in filtered.columns:
        filtered = filtered[filtered["touches_border"] == False].copy()  # noqa: E712
    if min_foreground_pixels is not None and pixel_cols:
        filtered = filtered[filtered[pixel_cols].sum(axis=1) >= int(min_foreground_pixels)].copy()

    if filtered.empty:
        raise ValueError("Metadata filtering removed all samples. Relax filtering options.")
    return filtered.reset_index(drop=True)


def num_classes_for_task(task: str) -> int:
    """Return output class count for a task."""

    validate_task(task)
    return 1 if task == "binary_kurgan" else len(TASK_CLASS_NAMES[task])


def class_names_for_task(task: str) -> dict[int, str]:
    """Return class id to name mapping for a task."""

    validate_task(task)
    return TASK_CLASS_NAMES[task]


def validate_task(task: str) -> None:
    """Raise on unsupported task name."""

    if task not in TASK_CLASS_NAMES:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {sorted(TASK_CLASS_NAMES)}")


def remap_mask_for_task(mask: np.ndarray, task: str) -> np.ndarray:
    """Map source masks to task-specific target masks."""

    validate_task(task)
    source = mask.astype(np.int64)
    bad_values = sorted(set(np.unique(source).tolist()) - VALID_SOURCE_CLASSES)
    if bad_values:
        raise ValueError(f"Unexpected source mask values: {bad_values}. Expected 0..5")

    if task == "binary_kurgan":
        return np.isin(source, BINARY_POSITIVE_CLASSES).astype(np.int64)
    if task == "kurgan_multiclass":
        return np.where(np.isin(source, BINARY_POSITIVE_CLASSES), source, 0).astype(np.int64)
    return source


class ArchaeologySegmentationDataset(Dataset):
    """PyTorch dataset for .npy geodata patches and semantic masks."""

    def __init__(
        self,
        meta: pd.DataFrame,
        data_root: str | Path,
        image_size: int = 256,
        task: str = "all_classes",
        normalize: str = "zscore",
    ) -> None:
        validate_task(task)
        self.meta = meta.reset_index(drop=True).copy()
        _, self.images_dir, self.masks_dir = validate_data_root(data_root)
        self.image_size = int(image_size)
        self.task = task
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.meta.iloc[index]
        sample_id = sample_id_to_name(row["sample_id"])
        image = np.load(self.images_dir / f"{sample_id}.npy")
        mask = np.load(self.masks_dir / f"{sample_id}.npy")

        if image.ndim == 3:
            image = image[..., 0] if image.shape[-1] <= 4 else image[0]
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(f"Expected 2D image/mask for {sample_id}, got {image.shape}/{mask.shape}")

        if image.shape != (self.image_size, self.image_size):
            image = self._resize_image(image)
        if mask.shape != (self.image_size, self.image_size):
            mask = self._resize_mask(mask)

        mask = remap_mask_for_task(mask, self.task)
        image = self._normalize(image)
        return {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "mask": torch.from_numpy(mask).long(),
            "sample_id": sample_id,
            "region": str(row["region"]),
            "modality": str(row["modality"]),
        }

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(image.astype(np.float32))[None, None]
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor[0, 0].numpy()

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(mask.astype(np.int64))[None, None].float()
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="nearest")
        return tensor[0, 0].long().numpy()

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize == "zscore":
            mean = float(image.mean())
            std = max(float(image.std()), 1e-6)
            return (image - mean) / std
        if self.normalize == "minmax":
            min_value = float(image.min())
            max_value = float(image.max())
            return (image - min_value) / max(max_value - min_value, 1e-6)
        if self.normalize in {"none", None}:
            return image
        raise ValueError(f"Unknown normalize mode: {self.normalize}")
