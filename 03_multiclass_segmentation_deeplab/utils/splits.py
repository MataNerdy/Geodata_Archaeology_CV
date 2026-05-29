"""Train/validation split helpers for region-aware experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
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
    train_split_csv: str | Path | None = None,
    val_split_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation split and apply optional modality filtering."""

    split = split.lower()
    if split in {"frozen", "frozen_csv", "csv"}:
        train_df, val_df = load_frozen_split(train_split_csv, val_split_csv, modalities)
    else:
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
                "Unknown split. Use region, custom_regions, random, stratified_region_holdout, or frozen."
            )

    if train_df.empty:
        raise ValueError("Train split is empty. Check split/modalities.")
    if val_df.empty:
        raise ValueError("Validation split is empty. Check split/modalities.")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)



def load_frozen_split(
    train_split_csv: str | Path | None,
    val_split_csv: str | Path | None,
    modalities: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a previously frozen train/validation split from CSV files.

    This is the preferred mode for final comparisons: create split CSVs once,
    commit/keep them as experiment protocol artifacts, and reuse them without
    running region search again.
    """

    if not train_split_csv or not val_split_csv:
        raise ValueError("split=frozen requires --train-split-csv and --val-split-csv")
    train_path = Path(train_split_csv)
    val_path = Path(val_split_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"Frozen train split not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Frozen validation split not found: {val_path}")

    train_df = pd.read_csv(train_path, dtype={"sample_id": str})
    val_df = pd.read_csv(val_path, dtype={"sample_id": str})
    if "sample_id" in train_df.columns:
        train_df["sample_id"] = train_df["sample_id"].astype(str).str.zfill(6)
    if "sample_id" in val_df.columns:
        val_df["sample_id"] = val_df["sample_id"].astype(str).str.zfill(6)

    train_df = filter_modalities(train_df, modalities)
    val_df = filter_modalities(val_df, modalities)
    overlap = set(train_df["sample_id"].astype(str)) & set(val_df["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"Frozen split has overlapping sample_id values: {sorted(overlap)[:10]}")
    return train_df, val_df


def make_region_holdout_split(
    meta: pd.DataFrame,
    val_frac: float = 0.2,
    group_col: str = "region",
    strat_cols: Sequence[str] = ("class_name", "modality"),
    min_val_per_class: int = 3,
    random_state: int = 42,
    n_trials: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """Notebook-style region holdout search balanced by class and modality.

    Run this once to create a frozen research split. Do not call it inside
    repeated training runs when comparing models.
    """

    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    missing = [col for col in (group_col, *strat_cols, "class_name") if col not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing split columns: {missing}")

    rng = np.random.default_rng(random_state)
    meta = meta.copy()
    target_n = int(len(meta) * val_frac)
    global_dist = meta.groupby(list(strat_cols)).size().div(len(meta))
    regions = meta[group_col].value_counts().index.to_list()
    region_sizes = meta[group_col].value_counts().to_dict()

    best_score = float("inf")
    best_regions: list[str] | None = None

    for _ in range(int(n_trials)):
        shuffled = regions.copy()
        rng.shuffle(shuffled)
        val_regions: list[str] = []
        val_n = 0

        for region in shuffled:
            if val_n < target_n:
                val_regions.append(region)
                val_n += int(region_sizes[region])

        val_df = meta[meta[group_col].isin(val_regions)]
        if val_df.empty:
            continue
        val_dist = val_df.groupby(list(strat_cols)).size().div(len(val_df))
        dist_diff = global_dist.sub(val_dist, fill_value=0).abs().sum()
        size_penalty = abs(len(val_df) - target_n) / len(meta)

        class_counts = val_df["class_name"].value_counts()
        missing_or_tiny_penalty = 0
        for class_name in meta["class_name"].unique():
            if class_counts.get(class_name, 0) < min_val_per_class:
                missing_or_tiny_penalty += 1

        region_domination = val_df[group_col].value_counts().max() / len(val_df)
        score = dist_diff + 2.0 * size_penalty + 0.5 * missing_or_tiny_penalty + 0.5 * region_domination

        if score < best_score:
            best_score = float(score)
            best_regions = list(val_regions)

    if not best_regions:
        raise ValueError("Could not build stratified region holdout split")

    train_df = meta[~meta[group_col].isin(best_regions)].copy()
    val_df = meta[meta[group_col].isin(best_regions)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), best_regions, best_score

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

