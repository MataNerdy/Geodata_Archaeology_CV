"""Train/validation split helpers for region-aware experiments."""

from __future__ import annotations

import random
from typing import Iterable

import pandas as pd

from arch_datasets.archaeology_dataset import filter_modalities


def parse_regions(value: str | None) -> list[str] | None:
    """Parse comma-separated region names."""

    if value is None:
        return None
    regions = [item.strip() for item in value.split(",") if item.strip()]
    if not regions:
        raise ValueError("Region list must not be empty")
    return regions


def make_split(
    meta: pd.DataFrame,
    split: str,
    val_region: str | None = None,
    val_regions: list[str] | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    modalities: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation split and apply optional modality filtering."""

    split = split.lower()
    meta = filter_modalities(meta, modalities)
    if split == "custom_regions":
        train_df, val_df = _custom_regions_split(meta, val_regions)
    elif split in {"region", "leave-one-region-out", "loro"}:
        train_df, val_df = _single_region_split(meta, val_region)
    elif split == "random":
        train_df, val_df = _random_split(meta, val_fraction, seed)
    elif split == "stratified_region_holdout":
        train_df, val_df = stratified_region_holdout(meta, val_fraction, seed)
    else:
        raise ValueError(
            "Unknown split. Use region, custom_regions, random, or stratified_region_holdout."
        )

    if train_df.empty:
        raise ValueError("Train split is empty. Check split/modalities.")
    if val_df.empty:
        raise ValueError("Validation split is empty. Check split/modalities.")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def stratified_region_holdout(
    meta: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick validation regions greedily to approximate modality distribution."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = random.Random(seed)
    regions = list(meta["region"].astype(str).unique())
    rng.shuffle(regions)

    target = max(1, round(len(meta) * val_fraction))
    val_regions: list[str] = []
    total = 0
    counts = meta.groupby("region").size().to_dict()
    for region in sorted(regions, key=lambda item: counts.get(item, 0), reverse=True):
        if total >= target and val_regions:
            break
        val_regions.append(region)
        total += int(counts.get(region, 0))
    return _custom_regions_split(meta, val_regions)


def _single_region_split(
    meta: pd.DataFrame,
    val_region: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not val_region:
        raise ValueError("--val-region is required for split=region")
    regions = set(meta["region"].astype(str))
    if val_region not in regions:
        raise ValueError(f"Validation region '{val_region}' not found")
    val_df = meta[meta["region"].astype(str) == val_region].copy()
    train_df = meta[meta["region"].astype(str) != val_region].copy()
    return train_df, val_df


def _custom_regions_split(
    meta: pd.DataFrame,
    val_regions: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not val_regions:
        raise ValueError("--val-regions is required for split=custom_regions")
    regions = set(meta["region"].astype(str))
    missing = sorted(set(val_regions) - regions)
    if missing:
        raise ValueError(f"Validation regions not found in metadata.csv: {missing}")
    val_df = meta[meta["region"].astype(str).isin(val_regions)].copy()
    train_df = meta[~meta["region"].astype(str).isin(val_regions)].copy()
    return train_df, val_df


def _random_split(
    meta: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    val_df = meta.sample(frac=val_fraction, random_state=seed)
    train_df = meta.drop(index=val_df.index).copy()
    return train_df, val_df.copy()

