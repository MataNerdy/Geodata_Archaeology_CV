#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


OUTPUT_NAME = "dataset_yolo_bbox_v3h_li_manual_curated_val"
VALID_DECISIONS = {"val", "train", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset with manually curated validation regions.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../datasets/dataset_yolo_bbox_v3g_li_medium_manual_keep_only"),
    )
    parser.add_argument(
        "--regions-csv",
        type=Path,
        default=Path("manual_val_region_review/manual_val_regions.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_metadata(source_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    if "source_id" not in df.columns:
        source_cols = [col for col in ["region", "modality", "raster_file"] if col in df.columns]
        df["source_id"] = df[source_cols].astype(str).agg("|".join, axis=1)
    return df


def read_region_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path).fillna("")
    decisions["decision"] = decisions["decision"].astype(str).str.strip().str.lower()
    invalid = decisions[~decisions["decision"].isin(VALID_DECISIONS | {""})]
    if not invalid.empty:
        bad = invalid[["region", "decision"]].to_string(index=False)
        raise ValueError(f"Invalid decisions in {path}:\n{bad}")
    empty = decisions[decisions["decision"].eq("")]
    if not empty.empty:
        print("WARNING: empty region decisions will be excluded:")
        print(empty[["region", "images_total", "positive_images", "bbox_total"]].to_string(index=False))
    return decisions[["region", "decision", "comment"]].copy()


def resolve_dataset_path(source_dir: Path, path_value: Any, kind: str, split: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        source_dir / kind / split / path.name,
        source_dir / path,
        source_dir.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_dir / kind / split / path.name


def split_counts(meta: pd.DataFrame) -> pd.DataFrame:
    images = meta.drop_duplicates("image").copy()
    boxes = meta[meta["class_name"].notna()].copy()
    rows = []
    for split, split_images in images.groupby("split", sort=True):
        split_boxes = boxes[boxes["image"].isin(set(split_images["image"]))]
        positive = int(split_images["image"].isin(set(split_boxes["image"])).sum())
        rows.append(
            {
                "split": split,
                "images": int(len(split_images)),
                "positive_images": positive,
                "negative_images": int(len(split_images) - positive),
                "bbox": int(len(split_boxes)),
            }
        )
    return pd.DataFrame(rows)


def leakage_report(meta: pd.DataFrame) -> dict[str, list[str]]:
    images = meta.drop_duplicates("image").copy()
    train = images[images["split"].eq("train")]
    val = images[images["split"].eq("val")]
    report: dict[str, list[str]] = {}
    for col in ["region", "source_id", "raster_file"]:
        train_values = set(train[col].dropna().astype(str))
        val_values = set(val[col].dropna().astype(str))
        report[col] = sorted(train_values & val_values)
    return report


def validate_split(meta: pd.DataFrame) -> list[str]:
    warnings = []
    images = meta.drop_duplicates("image").copy()
    boxes = meta[meta["class_name"].notna()].copy()
    val_images = images[images["split"].eq("val")]
    val_boxes = boxes[boxes["image"].isin(set(val_images["image"]))]
    val_positive = int(val_images["image"].isin(set(val_boxes["image"])).sum())
    if val_positive < 30:
        warnings.append(f"WARNING: val positive images is {val_positive}, expected >= 30.")
    if len(val_boxes) < 80:
        warnings.append(f"WARNING: val bbox count is {len(val_boxes)}, expected >= 80.")
    source_classes = set(val_boxes["source_class_name"].dropna().astype(str))
    for cls in ["kurgany_tselye", "kurgany_povrezhdennye"]:
        if cls not in source_classes:
            warnings.append(f"WARNING: val split does not contain source class {cls}.")
    return warnings


def write_dataset_yaml(out_dir: Path) -> None:
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "kurgan"}}
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def materialize(source_dir: Path, meta: pd.DataFrame, out_dir: Path, overwrite: bool) -> pd.DataFrame:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = []
    unique_images = meta.drop_duplicates("image").copy()
    unique_images = unique_images.sort_values(["split", "region", "image_name", "image"]).reset_index(drop=True)
    for image_idx, (_, image_row) in enumerate(unique_images.iterrows(), start=1):
        split = str(image_row["split"])
        source_split = str(image_row.get("source_split") or image_row.get("old_split") or split)
        src_img = resolve_dataset_path(source_dir, image_row["image"], "images", source_split)
        src_lbl = resolve_dataset_path(source_dir, image_row["label"], "labels", source_split)
        neutral_stem = f"{image_idx:06d}"
        dst_img = out_dir / "images" / split / f"{neutral_stem}{src_img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / split / f"{neutral_stem}.txt"
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        group = meta[meta["image"].eq(image_row["image"])].copy()
        for _, row in group.iterrows():
            record = row.to_dict()
            record["neutral_image_id"] = neutral_stem
            record["source_image_name"] = Path(str(row.get("image_name") or src_img.name)).name
            record["source_label_name"] = Path(str(row.get("label_name") or src_lbl.name)).name
            record["source_image_path"] = str(src_img.resolve())
            record["source_label_path"] = str(src_lbl.resolve())
            record["image"] = str(dst_img.resolve())
            record["label"] = str(dst_lbl.resolve())
            record["image_name"] = dst_img.name
            record["label_name"] = dst_lbl.name
            rows.append(record)
    out_meta = pd.DataFrame(rows)
    out_meta.to_csv(out_dir / "metadata.csv", index=False)
    write_dataset_yaml(out_dir)
    return out_meta


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.fillna("")
    cols = list(formatted.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_reports(out_dir: Path, meta: pd.DataFrame, leak: dict[str, list[str]], warnings: list[str]) -> None:
    images = meta.drop_duplicates("image").copy()
    boxes = meta[meta["class_name"].notna()].copy()
    counts = split_counts(meta)
    val_regions = sorted(images[images["split"].eq("val")]["region"].dropna().astype(str).unique())
    train_regions = sorted(images[images["split"].eq("train")]["region"].dropna().astype(str).unique())
    class_balance = (
        boxes.groupby(["split", "source_class_name"])
        .size()
        .reset_index(name="bbox")
        .sort_values(["split", "source_class_name"])
    )

    summary_lines = [
        "# Curated Validation Split Summary",
        "",
        "## Split Counts",
        "",
        markdown_table(counts),
        "",
        "## Validation Regions",
        "",
        "\n".join(f"- `{region}`" for region in val_regions) or "_No validation regions._",
        "",
        f"Train regions count: `{len(train_regions)}`",
        "",
        "## BBox By Source Class",
        "",
        markdown_table(class_balance),
        "",
        "## Checks",
        "",
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- All validation size/class checks passed.",
        "",
    ]
    (out_dir / "split_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    leak_rows = [
        {"leakage_key": key, "overlap_count": len(values), "overlap_values": "; ".join(values[:30])}
        for key, values in leak.items()
    ]
    leak_df = pd.DataFrame(leak_rows)
    leak_lines = [
        "# Leakage Report",
        "",
        markdown_table(leak_df),
        "",
    ]
    (out_dir / "leakage_report.md").write_text("\n".join(leak_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = (args.output_root / args.output_name).resolve()
    meta = read_metadata(source_dir)
    meta["previous_split"] = meta["split"]
    decisions = read_region_decisions(args.regions_csv)
    meta = meta.merge(decisions, on="region", how="left")
    meta["decision"] = meta["decision"].fillna("")
    unresolved = sorted(meta.loc[meta["decision"].eq(""), "region"].dropna().astype(str).unique())
    if unresolved:
        print("WARNING: regions without explicit decision are excluded:")
        for region in unresolved:
            print(f"  - {region}")
    meta = meta[meta["decision"].isin(["train", "val"])].copy()
    meta["split"] = meta["decision"]
    if meta.empty:
        raise ValueError("No rows selected. Fill manual_val_regions.csv with train/val decisions first.")

    leak = leakage_report(meta)
    leak_failures = {key: values for key, values in leak.items() if values}
    warnings = validate_split(meta)
    if leak_failures:
        details = "\n".join(f"{key}: {len(values)} overlaps" for key, values in leak_failures.items())
        raise ValueError(f"Leakage detected; refusing to build dataset.\n{details}")

    out_meta = materialize(source_dir, meta, out_dir, args.overwrite)
    write_reports(out_dir, out_meta, leak, warnings)
    print("Dataset:", out_dir)
    print(split_counts(out_meta).to_string(index=False))
    if warnings:
        print("\n".join(warnings))
    print("split_summary.md:", out_dir / "split_summary.md")
    print("leakage_report.md:", out_dir / "leakage_report.md")


if __name__ == "__main__":
    main()
