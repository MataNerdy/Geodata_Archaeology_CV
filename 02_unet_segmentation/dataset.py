"""Dataset utilities for kurgan semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REQUIRED_METADATA_COLUMNS = {"sample_id", "region", "modality"}


def sample_id_to_name(sample_id: object) -> str:
    """Convert a metadata sample id to the canonical six-digit file stem."""

    return str(sample_id).strip().zfill(6)


def validate_data_root(data_root: str | Path) -> tuple[Path, Path, Path]:
    """Validate dataset directories and return metadata/images/masks paths."""

    root = Path(data_root)
    metadata_path = root / "metadata.csv"
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    return metadata_path, images_dir, masks_dir


def load_metadata(data_root: str | Path) -> pd.DataFrame:
    """Read metadata.csv and verify that image/mask files match sample_id values."""

    metadata_path, images_dir, masks_dir = validate_data_root(data_root)
    meta = pd.read_csv(metadata_path, dtype={"sample_id": str})

    missing_cols = REQUIRED_METADATA_COLUMNS - set(meta.columns)
    if missing_cols:
        raise ValueError(
            f"metadata.csv is missing required columns: {sorted(missing_cols)}"
        )
    if meta.empty:
        raise ValueError(f"metadata.csv has no rows: {metadata_path}")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].map(sample_id_to_name)
    duplicated_ids = sorted(meta.loc[meta["sample_id"].duplicated(), "sample_id"].unique())
    if duplicated_ids:
        preview = ", ".join(duplicated_ids[:10])
        raise ValueError(f"Duplicated sample_id values in metadata.csv: {preview}")

    image_ids = {p.stem for p in images_dir.glob("*.npy")}
    mask_ids = {p.stem for p in masks_dir.glob("*.npy")}
    meta_ids = set(meta["sample_id"])

    missing_images = sorted(meta_ids - image_ids)
    missing_masks = sorted(meta_ids - mask_ids)
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(f"Missing image .npy files for sample_id: {preview}")
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise FileNotFoundError(f"Missing mask .npy files for sample_id: {preview}")

    orphan_images = sorted(image_ids - meta_ids)
    orphan_masks = sorted(mask_ids - meta_ids)
    if orphan_images or orphan_masks:
        raise ValueError(
            "Found .npy files without matching metadata sample_id: "
            f"{len(orphan_images)} images, {len(orphan_masks)} masks"
        )

    return meta


def make_split(
    meta: pd.DataFrame,
    split: str,
    val_region: str | None = None,
    val_regions: list[str] | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation metadata splits."""

    split = split.lower()
    if split in {"region", "leave-one-region-out", "loro"}:
        if not val_region:
            raise ValueError("--val-region is required when --split region is used")
        regions = set(meta["region"].astype(str))
        if val_region not in regions:
            available = ", ".join(sorted(regions)[:25])
            raise ValueError(
                f"Validation region '{val_region}' was not found. "
                f"Available examples: {available}"
            )
        val_df = meta[meta["region"].astype(str) == val_region].copy()
        train_df = meta[meta["region"].astype(str) != val_region].copy()
    elif split == "custom_regions":
        if not val_regions:
            raise ValueError("--val-regions is required when --split custom_regions is used")
        regions = set(meta["region"].astype(str))
        missing_regions = sorted(set(val_regions) - regions)
        if missing_regions:
            available = ", ".join(sorted(regions)[:25])
            raise ValueError(
                f"Validation regions were not found: {missing_regions}. "
                f"Available examples: {available}"
            )
        val_df = meta[meta["region"].astype(str).isin(val_regions)].copy()
        train_df = meta[~meta["region"].astype(str).isin(val_regions)].copy()
    elif split == "random":
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("--val-fraction must be between 0 and 1")
        val_df = meta.sample(frac=val_fraction, random_state=seed)
        train_df = meta.drop(index=val_df.index).copy()
        val_df = val_df.copy()
    else:
        raise ValueError(
            f"Unknown split '{split}'. Use 'region', 'custom_regions', or 'random'."
        )

    if train_df.empty:
        raise ValueError("Train split is empty. Check split settings.")
    if val_df.empty:
        raise ValueError("Validation split is empty. Check split settings.")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_experiment_split(
    meta: pd.DataFrame,
    split: str,
    val_region: str | None = None,
    val_regions: list[str] | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    modalities: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation split and apply optional modality filtering."""

    if split.lower() == "custom_regions":
        train_df, val_df = make_split(
            meta,
            split=split,
            val_region=val_region,
            val_regions=val_regions,
            val_fraction=val_fraction,
            seed=seed,
        )
        train_df = filter_modalities(train_df, modalities)
        val_df = filter_modalities(val_df, modalities)
    else:
        filtered_meta = filter_modalities(meta, modalities)
        train_df, val_df = make_split(
            filtered_meta,
            split=split,
            val_region=val_region,
            val_regions=val_regions,
            val_fraction=val_fraction,
            seed=seed,
        )

    if train_df.empty:
        raise ValueError("Train split is empty after modality filtering.")
    if val_df.empty:
        raise ValueError("Validation split is empty after modality filtering.")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def filter_modalities(meta: pd.DataFrame, modalities: Iterable[str] | None) -> pd.DataFrame:
    """Filter metadata by modality names if requested."""

    if not modalities:
        return meta
    wanted = set(modalities)
    present = set(meta["modality"].astype(str))
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(f"Requested modalities are absent from metadata: {missing}")
    return meta[meta["modality"].astype(str).isin(wanted)].reset_index(drop=True)


class KurganSegmentationDataset(Dataset):
    """PyTorch Dataset for .npy image patches and 3-class segmentation masks."""

    def __init__(
        self,
        meta: pd.DataFrame,
        data_root: str | Path,
        image_size: int = 256,
        normalize: str = "zscore",
        task: str = "multiclass",
    ) -> None:
        if task not in {"multiclass", "binary"}:
            raise ValueError("task must be 'multiclass' or 'binary'")
        self.meta = meta.reset_index(drop=True).copy()
        _, self.images_dir, self.masks_dir = validate_data_root(data_root)
        self.image_size = image_size
        self.normalize = normalize
        self.task = task

    def __len__(self) -> int:
        return len(self.meta)

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
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        return tensor[0, 0].long().numpy()

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize == "zscore":
            mean = float(image.mean())
            std = float(image.std())
            return (image - mean) / max(std, 1e-6)
        if self.normalize == "minmax":
            min_value = float(image.min())
            max_value = float(image.max())
            denom = max(max_value - min_value, 1e-6)
            return (image - min_value) / denom
        if self.normalize in {"none", None}:
            return image
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.meta.iloc[index]
        sample_id = sample_id_to_name(row["sample_id"])
        image_path = self.images_dir / f"{sample_id}.npy"
        mask_path = self.masks_dir / f"{sample_id}.npy"

        image = np.load(image_path)
        mask = np.load(mask_path)

        if image.ndim == 3:
            image = image[..., 0] if image.shape[-1] <= 4 else image[0]
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected 2D image/mask for sample {sample_id}, "
                f"got image={image.shape}, mask={mask.shape}"
            )

        if image.shape != (self.image_size, self.image_size):
            image = self._resize_image(image)
        if mask.shape != (self.image_size, self.image_size):
            mask = self._resize_mask(mask)

        mask = remap_to_kurgan_classes(mask)
        if self.task == "binary":
            mask = (mask > 0).astype(np.int64)

        image = self._normalize(image)
        return {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "mask": torch.from_numpy(mask).long(),
            "sample_id": sample_id,
            "region": str(row["region"]),
            "modality": str(row["modality"]),
        }


def remap_to_kurgan_classes(mask: np.ndarray) -> np.ndarray:
    """Keep kurgan classes 1/2 and map all other labels to background."""

    mask = mask.astype(np.int64)
    return np.where(np.isin(mask, [1, 2]), mask, 0).astype(np.int64)
