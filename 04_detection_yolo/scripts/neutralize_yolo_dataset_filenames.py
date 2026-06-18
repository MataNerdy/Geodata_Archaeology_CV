#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename YOLO dataset image/label files to neutral numeric filenames.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_dataset_path(dataset_dir: Path, path_value: Any, kind: str, split: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        Path.cwd() / path,
        dataset_dir.parent / path,
        dataset_dir / path,
        dataset_dir / kind / split / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return dataset_dir / kind / split / path.name


def path_like_original(dataset_dir: Path, original_value: Any, dst: Path) -> str:
    original = Path(str(original_value))
    if original.is_absolute():
        return str(dst.resolve())
    try:
        rel_to_parent = dst.resolve().relative_to(dataset_dir.parent.resolve())
    except ValueError:
        rel_to_parent = None
    if str(original).startswith(f"{dataset_dir.name}/") and rel_to_parent is not None:
        return rel_to_parent.as_posix()
    try:
        return dst.resolve().relative_to(dataset_dir.resolve()).as_posix()
    except ValueError:
        return str(dst)


def move_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path, dry_run: bool) -> None:
    if src_img.resolve() == dst_img.resolve() and src_lbl.resolve() == dst_lbl.resolve():
        return
    if dry_run:
        print(f"DRY image: {src_img} -> {dst_img}")
        print(f"DRY label: {src_lbl} -> {dst_lbl}")
        return

    tmp_img = src_img.with_name(f".__tmp_neutral__{src_img.name}")
    tmp_lbl = src_lbl.with_name(f".__tmp_neutral__{src_lbl.name}")
    if tmp_img.exists() or tmp_lbl.exists():
        raise FileExistsError(f"Temporary neutralization file already exists near {src_img}")

    shutil.move(str(src_img), str(tmp_img))
    shutil.move(str(src_lbl), str(tmp_lbl))
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_img), str(dst_img))
    shutil.move(str(tmp_lbl), str(dst_lbl))


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    meta = pd.read_csv(metadata_path)
    required = {"image", "label", "split"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    unique_images = meta.drop_duplicates("image").copy()
    sort_cols = [col for col in ["split", "region", "image_name", "image"] if col in unique_images.columns]
    unique_images = unique_images.sort_values(sort_cols).reset_index(drop=True)

    updates: dict[str, dict[str, str]] = {}
    for idx, (_, row) in enumerate(unique_images.iterrows(), start=1):
        split = str(row["split"])
        src_img = resolve_dataset_path(dataset_dir, row["image"], "images", split)
        src_lbl = resolve_dataset_path(dataset_dir, row["label"], "labels", split)
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        if not src_lbl.exists():
            raise FileNotFoundError(f"Missing label: {src_lbl}")

        neutral_stem = f"{idx:0{args.digits}d}"
        dst_img = dataset_dir / "images" / split / f"{neutral_stem}{src_img.suffix.lower()}"
        dst_lbl = dataset_dir / "labels" / split / f"{neutral_stem}.txt"
        updates[str(row["image"])] = {
            "neutral_image_id": neutral_stem,
            "image": path_like_original(dataset_dir, row["image"], dst_img),
            "label": path_like_original(dataset_dir, row["label"], dst_lbl),
            "image_name": dst_img.name,
            "label_name": dst_lbl.name,
            "source_image_name": str(row.get("source_image_name") or row.get("image_name") or src_img.name),
            "source_label_name": str(row.get("source_label_name") or row.get("label_name") or src_lbl.name),
            "source_image_path": str(row.get("source_image_path") or src_img.resolve()),
            "source_label_path": str(row.get("source_label_path") or src_lbl.resolve()),
            "_src_img": str(src_img),
            "_src_lbl": str(src_lbl),
            "_dst_img": str(dst_img),
            "_dst_lbl": str(dst_lbl),
        }

    if args.dry_run:
        print(f"Dataset: {dataset_dir}")
        print(f"Images to neutralize: {len(updates)}")
        for item in list(updates.values())[:10]:
            print(f"{item['_src_img']} -> {item['_dst_img']}")
        return

    # Move files after all paths were validated. Existing neutral files are
    # allowed only when they are the same source file for an already neutralized dataset.
    for item in updates.values():
        move_pair(Path(item["_src_img"]), Path(item["_src_lbl"]), Path(item["_dst_img"]), Path(item["_dst_lbl"]), dry_run=False)

    for old_image_value, item in updates.items():
        mask = meta["image"].astype(str).eq(old_image_value)
        for col in [
            "neutral_image_id",
            "image",
            "label",
            "image_name",
            "label_name",
            "source_image_name",
            "source_label_name",
            "source_image_path",
            "source_label_path",
        ]:
            meta.loc[mask, col] = item[col]

    metadata_path.write_text(meta.to_csv(index=False), encoding="utf-8")
    print(f"Neutralized dataset: {dataset_dir}")
    print(f"Images: {len(updates)}")
    print(f"metadata.csv: {metadata_path}")


if __name__ == "__main__":
    main()
