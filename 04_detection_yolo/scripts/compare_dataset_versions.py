#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_VERSIONS = {
    "v3a_minimal": "../datasets/dataset_yolo_bbox_v3a_li_binary_minimal",
    "v3b_medium": "../datasets/dataset_yolo_bbox_v3b_li_binary_medium",
    "v3c_strict": "../datasets/dataset_yolo_bbox_v3c_li_binary_strict",
    "v3d_li_ae_medium": "../datasets/dataset_yolo_bbox_v3d_li_ae_binary_medium",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated YOLO dataset versions.")
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_ablation/dataset_versions_comparison.csv"))
    return parser.parse_args()


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {path}")
    df = pd.read_csv(path)
    df["is_positive"] = df["is_positive"].astype(bool)
    df["n_objects"] = pd.to_numeric(df["n_objects"], errors="coerce").fillna(0).astype(int)
    return df


def metrics(version: str, dataset_dir: Path) -> dict[str, Any]:
    df = read_metadata(dataset_dir)
    images = df.drop_duplicates("image").copy()
    boxes = df[df["class_name"].notna()].copy()
    edge_bbox_ratio = (
        float(boxes["bbox_touches_tile_edge"].astype("boolean").fillna(False).mean())
        if len(boxes)
        else 0.0
    )
    return {
        "dataset_version": version,
        "images_total": int(len(images)),
        "train_images": int((images["split"] == "train").sum()),
        "val_images": int((images["split"] == "val").sum()),
        "positive_images": int(images["is_positive"].sum()),
        "negative_images": int((~images["is_positive"]).sum()),
        "bbox_total": int(len(boxes)),
        "bbox_per_positive_image_mean": float(images[images["is_positive"]]["n_objects"].mean()) if images["is_positive"].any() else 0.0,
        "median_bbox_area": float(pd.to_numeric(boxes["bbox_area_px"], errors="coerce").median()) if len(boxes) else 0.0,
        "edge_bbox_ratio": edge_bbox_ratio,
        "valid_fraction_mean": float(images["valid_fraction"].mean()),
        "modalities": ",".join(sorted(images["modality"].dropna().astype(str).unique())),
        "classes_used": ",".join(sorted(boxes["class_name"].dropna().astype(str).unique())),
    }


def markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    table.columns = [str(c) for c in table.columns]
    rows = ["| " + " | ".join(table.columns) + " |"]
    rows.append("|" + "|".join("---" for _ in table.columns) + "|")
    for _, row in table.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    rows = [metrics(name, Path(path)) for name, path in DEFAULT_VERSIONS.items()]
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_suffix(".md").write_text(
        "# Dataset Versions Comparison\n\n" + markdown_table(df) + "\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
