#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


TARGET_CLASSES = [
    "kurgany_tselye",
    "kurgany_povrezhdennye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
]
CLASS_ID_TO_NAME = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}
TARGET_CLASS_IDS = set(CLASS_ID_TO_NAME)
VALID_REGION_DECISIONS = {"train", "val", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a one-class archaeological-object dataset by merging kurgan and other-class curated splits."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument(
        "--kurgan-dataset",
        type=Path,
        default=Path("../datasets/dataset_yolo_bbox_v3h_li_manual_curated_val_no_saratov"),
    )
    parser.add_argument(
        "--other-regions-csv",
        type=Path,
        default=Path("manual_val_region_review_other_classes/manual_val_regions.csv"),
    )
    parser.add_argument(
        "--kurgan-regions-csv",
        type=Path,
        default=Path("manual_val_region_review/manual_val_regions_no_saratov.csv"),
    )
    parser.add_argument(
        "--merged-regions-csv",
        type=Path,
        default=None,
        help="Optional single region decision CSV. If provided, it overrides kurgan/other region CSV merging.",
    )
    parser.add_argument("--decisions", type=Path, default=Path("manual_audit/audit_decisions.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--output-name", default="dataset_yolo_bbox_v3i_li_archaeological_object_merged")
    parser.add_argument("--modalities", nargs="+", default=["Li"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_audit_image_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "source_image_name" if "source_image_name" in out.columns else "image_name"
    out["image_id"] = out.apply(lambda row: f"{row['split']}_{Path(str(row.get(name_col) or row['image'])).stem}", axis=1)
    return out


def read_manual_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path).fillna("")
    decisions = decisions.drop_duplicates("image_id", keep="last")
    return decisions[["image_id", "decision", "reason", "comment", "updated_at"]].rename(
        columns={
            "decision": "manual_audit_decision",
            "reason": "manual_audit_reason",
            "comment": "manual_audit_comment",
            "updated_at": "manual_audit_updated_at",
        }
    )


def read_region_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    invalid = df[~df["decision"].isin(VALID_REGION_DECISIONS | {""})]
    if not invalid.empty:
        raise ValueError(f"Invalid region decisions in {path}:\n{invalid[['region', 'decision']].to_string(index=False)}")
    return df[df["decision"].isin(["train", "val"])][["region", "decision"]].copy()


def merged_region_split(*region_decision_frames: pd.DataFrame) -> dict[str, str]:
    decisions_by_region: dict[str, set[str]] = {}
    for frame in region_decision_frames:
        for _, row in frame.iterrows():
            decisions_by_region.setdefault(str(row["region"]), set()).add(str(row["decision"]))
    split_by_region = {}
    for region, decisions in decisions_by_region.items():
        split_by_region[region] = "val" if "val" in decisions else "train"
    return split_by_region


def split_from_single_regions_csv(region_decisions: pd.DataFrame) -> dict[str, str]:
    return {str(row["region"]): str(row["decision"]) for _, row in region_decisions.iterrows()}


def read_source_metadata(source_dir: Path, decisions_path: Path, modalities: set[str] | None) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    if "source_id" not in df.columns:
        source_cols = [col for col in ["region", "modality", "raster_file"] if col in df.columns]
        df["source_id"] = df[source_cols].astype(str).agg("|".join, axis=1)
    if modalities:
        df = df[df["modality"].astype(str).isin(modalities)].copy()
    df = add_audit_image_id(df)
    df = df.merge(read_manual_decisions(decisions_path), on="image_id", how="left")
    df["manual_audit_decision"] = df["manual_audit_decision"].fillna("")
    unresolved = df.drop_duplicates("image_id").query("manual_audit_decision == ''")
    if len(unresolved):
        print(f"WARNING: excluding {len(unresolved)} images without manual audit decision.")
    df = df[df["manual_audit_decision"].eq("keep")].copy()
    return df


def read_kurgan_image_ids(kurgan_dataset: Path) -> set[str]:
    df = pd.read_csv(kurgan_dataset / "metadata.csv")
    if "image_id" not in df.columns:
        raise ValueError(f"{kurgan_dataset / 'metadata.csv'} does not contain image_id.")
    return set(df["image_id"].dropna().astype(str))


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


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def split_counts(meta: pd.DataFrame) -> pd.DataFrame:
    images = meta.drop_duplicates("image_id").copy()
    boxes = meta[meta["is_target_object"]].copy()
    rows = []
    for split, split_images in images.groupby("split", sort=True):
        split_boxes = boxes[boxes["image_id"].isin(set(split_images["image_id"]))]
        positive = int(split_images["image_id"].isin(set(split_boxes["image_id"])).sum())
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
    images = meta.drop_duplicates("image_id").copy()
    train = images[images["split"].eq("train")]
    val = images[images["split"].eq("val")]
    report: dict[str, list[str]] = {}
    for col in ["region", "source_id", "raster_file"]:
        train_values = set(train[col].dropna().astype(str))
        val_values = set(val[col].dropna().astype(str))
        report[col] = sorted(train_values & val_values)
    return report


def write_dataset_yaml(out_dir: Path) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "archaeological_object"},
    }
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
    unique_images = meta.drop_duplicates("image_id").copy()
    unique_images = unique_images.sort_values(["split", "region", "image_id"]).reset_index(drop=True)
    for image_idx, (_, image_row) in enumerate(unique_images.iterrows(), start=1):
        split = str(image_row["split"])
        source_split = str(image_row.get("source_split") or image_row.get("old_split") or image_row.get("previous_split") or image_row["original_split"])
        src_img = resolve_dataset_path(source_dir, image_row["image"], "images", source_split)
        src_lbl = resolve_dataset_path(source_dir, image_row["label"], "labels", source_split)
        neutral_stem = f"{image_idx:06d}"
        dst_img = out_dir / "images" / split / f"{neutral_stem}{src_img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / split / f"{neutral_stem}.txt"
        shutil.copy2(src_img, dst_img)

        label_lines = []
        for cls_id, xc, yc, bw, bh in parse_label_file(src_lbl):
            if cls_id in TARGET_CLASS_IDS:
                label_lines.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        group = meta[meta["image_id"].eq(image_row["image_id"])].copy()
        for _, row in group.iterrows():
            record = row.to_dict()
            record["neutral_image_id"] = neutral_stem
            record["source_image_name"] = Path(str(row.get("source_image_name") or row.get("image_name") or src_img.name)).name
            record["source_label_name"] = Path(str(row.get("source_label_name") or row.get("label_name") or src_lbl.name)).name
            record["source_image_path"] = str(src_img.resolve())
            record["source_label_path"] = str(src_lbl.resolve())
            record["image"] = str(dst_img.resolve())
            record["label"] = str(dst_lbl.resolve())
            record["image_name"] = dst_img.name
            record["label_name"] = dst_lbl.name
            record["class_id"] = 0 if bool(record.get("is_target_object")) else record.get("class_id")
            record["class_name"] = "archaeological_object" if bool(record.get("is_target_object")) else record.get("class_name")
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


def region_decision_comparison(kurgan_regions: pd.DataFrame, other_regions: pd.DataFrame) -> pd.DataFrame:
    left = kurgan_regions.rename(columns={"decision": "kurgan_decision"})
    right = other_regions.rename(columns={"decision": "other_classes_decision"})
    comparison = left.merge(right, on="region", how="outer").fillna("")
    comparison["merged_decision"] = comparison.apply(
        lambda row: "val" if "val" in {row["kurgan_decision"], row["other_classes_decision"]} else "train",
        axis=1,
    )
    comparison["val_source"] = comparison.apply(
        lambda row: (
            "both"
            if row["kurgan_decision"] == "val" and row["other_classes_decision"] == "val"
            else (
                "kurgan_only"
                if row["kurgan_decision"] == "val"
                else ("other_classes_only" if row["other_classes_decision"] == "val" else "")
            )
        ),
        axis=1,
    )
    return comparison.sort_values(["merged_decision", "region"], ascending=[False, True])


def write_reports(
    out_dir: Path,
    meta: pd.DataFrame,
    leak: dict[str, list[str]],
    split_by_region: dict[str, str],
    kurgan_regions: pd.DataFrame | None,
    other_regions: pd.DataFrame | None,
) -> None:
    images = meta.drop_duplicates("image_id").copy()
    boxes = meta[meta["is_target_object"]].copy()
    source_class_balance = (
        boxes.groupby(["split", "source_class_name"]).size().reset_index(name="bbox").sort_values(["split", "source_class_name"])
    )
    val_regions = sorted(region for region, split in split_by_region.items() if split == "val")
    train_regions = sorted(region for region, split in split_by_region.items() if split == "train")
    merge_sources = (
        images.groupby(["split", "selection_source"])
        .size()
        .reset_index(name="images")
        .sort_values(["split", "selection_source"])
    )
    val_only = pd.DataFrame()
    if kurgan_regions is not None and other_regions is not None:
        region_comparison = region_decision_comparison(kurgan_regions, other_regions)
        val_only = region_comparison[region_comparison["val_source"].isin(["kurgan_only", "other_classes_only", "both"])][
            ["region", "kurgan_decision", "other_classes_decision", "val_source", "merged_decision"]
        ]
    lines = [
        "# Merged Archaeological Object Dataset",
        "",
        "One-class YOLO dataset: `0: archaeological_object`.",
        "",
        "Target source classes:",
        "",
        "\n".join(f"- `{cls}`" for cls in TARGET_CLASSES),
        "",
        "## Split Counts",
        "",
        markdown_table(split_counts(meta)),
        "",
        "## BBox By Source Class",
        "",
        markdown_table(source_class_balance),
        "",
        "## Image Selection Source",
        "",
        markdown_table(merge_sources),
        "",
        "## Validation Regions",
        "",
        "\n".join(f"- `{region}`" for region in val_regions) or "_No validation regions._",
        "",
        "## Validation Region Sources",
        "",
        markdown_table(val_only) if not val_only.empty else "_Single merged region decision file was used._",
        "",
        f"Train regions count: `{len(train_regions)}`",
        "",
    ]
    (out_dir / "split_summary.md").write_text("\n".join(lines), encoding="utf-8")

    leak_df = pd.DataFrame(
        [{"leakage_key": key, "overlap_count": len(values), "overlap_values": "; ".join(values[:30])} for key, values in leak.items()]
    )
    (out_dir / "leakage_report.md").write_text("# Leakage Report\n\n" + markdown_table(leak_df) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = (args.output_root / args.output_name).resolve()
    modalities = set(args.modalities) if args.modalities else None

    if args.merged_regions_csv is not None:
        merged_regions = read_region_decisions(args.merged_regions_csv)
        kurgan_regions = None
        other_regions = None
        split_by_region = split_from_single_regions_csv(merged_regions)
    else:
        kurgan_regions = read_region_decisions(args.kurgan_regions_csv)
        other_regions = read_region_decisions(args.other_regions_csv)
        split_by_region = merged_region_split(kurgan_regions, other_regions)

    source_meta = read_source_metadata(source_dir, args.decisions, modalities)
    source_meta["original_split"] = source_meta["split"]
    source_meta["region_split"] = source_meta["region"].map(split_by_region)

    kurgan_ids = read_kurgan_image_ids(args.kurgan_dataset)
    other_region_images = set(source_meta.loc[source_meta["region_split"].notna(), "image_id"].astype(str))
    selected_ids = kurgan_ids | other_region_images

    meta = source_meta[source_meta["image_id"].astype(str).isin(selected_ids)].copy()
    meta = meta[meta["region_split"].isin(["train", "val"])].copy()
    meta["split"] = meta["region_split"]
    meta["is_target_object"] = meta["source_class_name"].isin(TARGET_CLASSES)
    positive_images = set(meta.loc[meta["is_target_object"], "image_id"].astype(str))
    meta["is_target_positive"] = meta["image_id"].astype(str).isin(positive_images)
    kurgan_id_set = set(kurgan_ids)
    other_id_set = set(other_region_images)
    meta["selection_source"] = meta["image_id"].astype(str).map(
        lambda image_id: "kurgan+other" if image_id in kurgan_id_set and image_id in other_id_set else ("kurgan" if image_id in kurgan_id_set else "other")
    )

    if meta.empty:
        raise ValueError("No selected images.")

    leak = leakage_report(meta)
    leak_failures = {key: values for key, values in leak.items() if values}
    if leak_failures:
        details = "\n".join(f"{key}: {len(values)} overlaps" for key, values in leak_failures.items())
        raise ValueError(f"Leakage detected; refusing to build dataset.\n{details}")

    out_meta = materialize(source_dir, meta, out_dir, args.overwrite)
    write_reports(out_dir, out_meta, leak, split_by_region, kurgan_regions, other_regions)
    print("Dataset:", out_dir)
    print(split_counts(out_meta).to_string(index=False))
    class_counts = out_meta[out_meta["is_target_object"]].groupby(["split", "source_class_name"]).size().reset_index(name="bbox")
    print(class_counts.sort_values(["split", "source_class_name"]).to_string(index=False))
    print("split_summary.md:", out_dir / "split_summary.md")
    print("leakage_report.md:", out_dir / "leakage_report.md")


if __name__ == "__main__":
    main()
