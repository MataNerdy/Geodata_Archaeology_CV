#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


RANDOM_SEED = 42
TARGET_CLASSES = {"kurgany_tselye", "kurgany_povrezhdennye"}
SOURCE_ID_COLUMNS = ["region", "modality", "raster_file"]
OUTPUT_NAME = "dataset_yolo_bbox_v3g_li_medium_manual_keep_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3g Li kurgan dataset from manual keep-only audit.")
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--audit-index", type=Path, default=Path("manual_audit/audit_index.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("manual_audit/audit_decisions.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_source_metadata(source_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    df["n_objects"] = pd.to_numeric(df["n_objects"], errors="coerce").fillna(0).astype(int)
    df["valid_fraction"] = pd.to_numeric(df["valid_fraction"], errors="coerce")
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df["bbox_area_px"] = pd.to_numeric(df["bbox_area_px"], errors="coerce")
    df["bbox_x1_px"] = pd.to_numeric(df["bbox_x1_px"], errors="coerce")
    df["bbox_y1_px"] = pd.to_numeric(df["bbox_y1_px"], errors="coerce")
    df["bbox_x2_px"] = pd.to_numeric(df["bbox_x2_px"], errors="coerce")
    df["bbox_y2_px"] = pd.to_numeric(df["bbox_y2_px"], errors="coerce")
    df["bbox_touches_tile_edge"] = df["bbox_touches_tile_edge"].astype("boolean")
    df["tile_touches_raster_edge"] = df["tile_touches_raster_edge"].astype(bool)
    df["has_edge_object"] = df["has_edge_object"].astype(bool)
    df["source_id"] = df[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    return df


def image_table(meta: pd.DataFrame) -> pd.DataFrame:
    return meta.drop_duplicates("image").copy()


def read_decisions(audit_index_path: Path, decisions_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_index_path).fillna("")
    decisions = pd.read_csv(decisions_path).fillna("")
    decisions = decisions.drop_duplicates("image_id", keep="last")
    decisions = decisions.rename(
        columns={
            "decision": "manual_decision",
            "reason": "manual_reason",
            "comment": "manual_comment",
            "updated_at": "manual_updated_at",
        }
    )
    merged = audit.drop(columns=["decision", "reason", "comment"], errors="ignore").merge(
        decisions[["image_id", "manual_decision", "manual_reason", "manual_comment", "manual_updated_at"]],
        on="image_id",
        how="left",
    )
    for col in ["manual_decision", "manual_reason", "manual_comment", "manual_updated_at"]:
        merged[col] = merged[col].fillna("")
    return merged


def add_image_ids(images: pd.DataFrame) -> pd.DataFrame:
    out = images.copy()
    out["image_id"] = out.apply(
        lambda row: f"{row['split']}_{Path(str(row.get('image_name') or row['image'])).stem}",
        axis=1,
    )
    return out


def resolve_dataset_path(source_dir: Path, path_value: Any, kind: str, split: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        Path.cwd() / path,
        source_dir.parent / path,
        source_dir / kind / split / path.name,
        source_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_dir / kind / split / path.name


def build_v3b_keep_candidates(source_meta: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    images = add_image_ids(image_table(source_meta))
    source_objects = source_meta[source_meta["class_name"].isin(TARGET_CLASSES)].copy()
    audit_decisions = audit[["image_id", "manual_decision", "manual_reason", "manual_comment", "manual_updated_at"]].copy()
    images = images.merge(audit_decisions, on="image_id", how="left")
    for col in ["manual_decision", "manual_reason", "manual_comment", "manual_updated_at"]:
        images[col] = images[col].fillna("")

    stats: dict[str, Any] = {}
    stats["manual_decision_counts_all"] = images["manual_decision"].replace("", "EMPTY").value_counts().to_dict()

    # v3b image-level filters before manual keep-only.
    before_v3b = images.copy()
    images = images[images["modality"].eq("Li")].copy()
    images = images[images["valid_fraction"] >= 0.85].copy()
    images = images[images["n_objects"] <= 50].copy()

    objects = source_objects[source_objects["image"].isin(set(images["image"]))].copy()
    objects = objects[objects["bbox_area_px"] >= 100].copy()

    positive_ids = set(objects["image"])
    source_kurgan_ids = set(source_objects["image"])
    clean_negative_ids = set(images[~images["image"].isin(source_kurgan_ids)]["image"])
    images = images[images["image"].isin(positive_ids | clean_negative_ids)].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()

    positive_ids = set(objects["image"])
    positives = images[images["image"].isin(positive_ids)].copy()
    negatives = images[~images["image"].isin(positive_ids)].copy()
    n_negatives = min(len(negatives), len(positives))
    negatives = negatives.sample(n=n_negatives, random_state=RANDOM_SEED) if n_negatives else negatives.iloc[0:0].copy()
    images = pd.concat([positives, negatives], ignore_index=True).sort_values("image_id").reset_index(drop=True)
    objects = objects[objects["image"].isin(set(images["image"]))].copy()

    stats["images_before_v3b"] = int(len(before_v3b))
    stats["images_after_v3b_filters"] = int(len(images))
    stats["manual_decision_counts_after_v3b"] = images["manual_decision"].replace("", "EMPTY").value_counts().to_dict()

    empty_decisions = images[images["manual_decision"].eq("")].copy()
    if len(empty_decisions):
        stats["WARNING_empty_decisions_excluded"] = int(len(empty_decisions))

    removed = images[images["manual_decision"].isin(["remove_image", "fix_label", "uncertain"])].copy()
    keep = images[images["manual_decision"].eq("keep")].copy()
    objects_keep = objects[objects["image"].isin(set(keep["image"]))].copy()

    stats["removed_from_old_split"] = removed.groupby(["split", "manual_decision"]).size().to_dict()
    stats["kept_from_old_split"] = keep.groupby("split").size().to_dict()
    stats["excluded_bbox_by_decision"] = (
        objects.merge(images[["image", "manual_decision"]], on="image", how="left")
        .query("manual_decision != 'keep'")
        .groupby("manual_decision")
        .size()
        .to_dict()
    )
    return keep.reset_index(drop=True), objects_keep.reset_index(drop=True), stats


def split_score(images: pd.DataFrame, objects: pd.DataFrame, val_regions: set[str], target_val: int) -> float:
    val = images[images["region"].isin(val_regions)]
    train = images[~images["region"].isin(val_regions)]
    if val.empty or train.empty:
        return 1e9
    val_pos = int(val["image"].isin(set(objects["image"])).sum())
    train_pos = int(train["image"].isin(set(objects["image"])).sum())
    val_neg = len(val) - val_pos
    train_neg = len(train) - train_pos
    val_boxes = objects[objects["image"].isin(set(val["image"]))]
    train_boxes = objects[objects["image"].isin(set(train["image"]))]
    class_penalty = 0.0
    for cls in TARGET_CLASSES:
        val_count = int((val_boxes["class_name"] == cls).sum())
        train_count = int((train_boxes["class_name"] == cls).sum())
        total = val_count + train_count
        if total:
            class_penalty += abs((val_count / total) - 0.20) * 20
    return (
        abs(len(val) - target_val) * 2
        + abs((val_pos / max(1, len(val))) - (len(objects["image"].unique()) / max(1, len(images)))) * 80
        + abs(val_pos - max(1, round(len(set(objects["image"])) * 0.20))) * 3
        + abs(val_neg - max(1, round((len(images) - len(set(objects["image"]))) * 0.20))) * 2
        + class_penalty
        + (0 if val_pos >= 30 else (30 - val_pos) * 20)
    )


def make_region_split(images: pd.DataFrame, objects: pd.DataFrame, val_size: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    regions = sorted(images["region"].astype(str).unique())
    target_val = max(1, round(len(images) * val_size))
    rng = random.Random(RANDOM_SEED)
    best_regions: set[str] | None = None
    best_score = 1e18

    # Randomized region subsets around target size; deterministic seed.
    region_sizes = images.groupby("region").size().to_dict()
    for _ in range(20000):
        shuffled = regions[:]
        rng.shuffle(shuffled)
        val_regions: set[str] = set()
        count = 0
        for region in shuffled:
            if count < target_val or not val_regions:
                val_regions.add(region)
                count += int(region_sizes[region])
            if count >= target_val:
                break
        score = split_score(images, objects, val_regions, target_val)
        if score < best_score:
            best_score = score
            best_regions = val_regions

    assert best_regions is not None
    out = images.copy()
    out["split"] = out["region"].astype(str).map(lambda r: "val" if r in best_regions else "train")
    report = {
        "val_regions": sorted(best_regions),
        "target_val_images": target_val,
        "split_score": best_score,
    }
    return out, report


def yolo_box_from_metadata(row: pd.Series) -> tuple[int, float, float, float, float]:
    tile_size = float(row.get("tile_size", 1024))
    x1 = float(row["bbox_x1_px"])
    y1 = float(row["bbox_y1_px"])
    x2 = float(row["bbox_x2_px"])
    y2 = float(row["bbox_y2_px"])
    xc = ((x1 + x2) / 2.0) / tile_size
    yc = ((y1 + y2) / 2.0) / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return 0, xc, yc, w, h


def valid_yolo_box(box: tuple[int, float, float, float, float]) -> bool:
    _, xc, yc, w, h = box
    return 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1


def write_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes), encoding="utf-8")


def write_dataset_yaml(out_dir: Path) -> None:
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "kurgan"}}
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def materialize_dataset(source_dir: Path, images: pd.DataFrame, objects: pd.DataFrame, out_dir: Path, overwrite: bool) -> pd.DataFrame:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    object_groups = {image: group.copy() for image, group in objects.groupby("image")}
    rows: list[dict[str, Any]] = []
    for _, image_row in images.iterrows():
        split = str(image_row["split"])
        old_split = str(image_row["old_split"])
        src_img = resolve_dataset_path(source_dir, image_row["image"], "images", old_split)
        src_lbl = resolve_dataset_path(source_dir, image_row["label"], "labels", old_split)
        dst_img = out_dir / "images" / split / src_img.name
        dst_lbl = out_dir / "labels" / split / src_lbl.name
        shutil.copy2(src_img, dst_img)

        group = object_groups.get(image_row["image"], pd.DataFrame())
        box_records = []
        for _, obj in group.iterrows():
            box = yolo_box_from_metadata(obj)
            if valid_yolo_box(box):
                box_records.append((obj, box))
        write_label(dst_lbl, [box for _, box in box_records])

        base = image_row.to_dict()
        base["image"] = str(dst_img)
        base["label"] = str(dst_lbl)
        base["source_image"] = image_row["image"]
        base["source_label"] = image_row["label"]
        base["source_split"] = old_split
        base["is_positive"] = bool(box_records)
        base["n_objects"] = len(box_records)

        if box_records:
            for obj, box in box_records:
                _, xc, yc, w, h = box
                row = base.copy()
                row.update(
                    {
                        "class_id": 0,
                        "class_name": "kurgan",
                        "source_class_id": int(obj["class_id"]),
                        "source_class_name": obj["class_name"],
                        "bbox_x1_px": obj["bbox_x1_px"],
                        "bbox_y1_px": obj["bbox_y1_px"],
                        "bbox_x2_px": obj["bbox_x2_px"],
                        "bbox_y2_px": obj["bbox_y2_px"],
                        "bbox_area_px": obj["bbox_area_px"],
                        "bbox_touches_tile_edge": bool(obj["bbox_touches_tile_edge"]),
                        "yolo_xc": xc,
                        "yolo_yc": yc,
                        "yolo_w": w,
                        "yolo_h": h,
                    }
                )
                rows.append(row)
        else:
            row = base.copy()
            for col in [
                "class_id",
                "class_name",
                "source_class_id",
                "source_class_name",
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
                "bbox_area_px",
                "bbox_touches_tile_edge",
                "yolo_xc",
                "yolo_yc",
                "yolo_w",
                "yolo_h",
            ]:
                row[col] = None
            rows.append(row)

    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    write_dataset_yaml(out_dir)
    return meta


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    table.columns = [str(c) for c in table.columns]
    rows = ["| " + " | ".join(table.columns) + " |", "|" + "|".join("---" for _ in table.columns) + "|"]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(rows)


def save_bar(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, log_x: bool = False) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if log_x:
        values = values[values > 0]
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=50, edgecolor="black")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_width_height(meta: pd.DataFrame, out_path: Path) -> None:
    boxes = meta[meta["bbox_area_px"].notna()]
    widths = pd.to_numeric(boxes["bbox_x2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_x1_px"], errors="coerce")
    heights = pd.to_numeric(boxes["bbox_y2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_y1_px"], errors="coerce")
    plt.figure(figsize=(7, 7))
    plt.scatter(widths, heights, s=14, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("bbox width px")
    plt.ylabel("bbox height px")
    plt.title("BBox Width vs Height")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append((int(float(parts[0])), *map(float, parts[1:])))
    return boxes


def draw_thumb(image_path: Path, label_path: Path, thumb_size: int = 256) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    img.thumbnail((thumb_size, thumb_size))
    canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
    ox = (thumb_size - img.size[0]) // 2
    oy = (thumb_size - img.size[1]) // 2
    canvas.paste(img, (ox, oy))
    sx = img.size[0] / src_w
    sy = img.size[1] / src_h
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for cls_id, xc, yc, w, h in parse_label(label_path):
        x1 = (xc - w / 2) * src_w * sx + ox
        y1 = (yc - h / 2) * src_h * sy + oy
        x2 = (xc + w / 2) * src_w * sx + ox
        y2 = (yc + h / 2) * src_h * sy + oy
        draw.rectangle([x1, y1, x2, y2], outline="#00ff66", width=3)
        draw.rectangle([x1, max(0, y1 - 12), x1 + 14, max(12, y1)], fill="#00ff66")
        draw.text((x1 + 2, max(0, y1 - 12)), str(cls_id), fill="black", font=font)
    return canvas


def save_label_pages(meta: pd.DataFrame, out_dir: Path, split: str) -> None:
    images = image_table(meta)
    subset = images[images["split"].eq(split)].copy()
    positives = subset[subset["is_positive"]]
    source = positives if len(positives) else subset
    n = min(len(source), 5 * 25)
    sample = source.sample(n=n, random_state=RANDOM_SEED) if n else source
    for page in range(5):
        sheet = Image.new("RGB", (5 * 256, 5 * 256), "white")
        page_rows = sample.iloc[page * 25 : (page + 1) * 25]
        for idx, row in enumerate(page_rows.itertuples(index=False)):
            thumb = draw_thumb(Path(row.image), Path(row.label))
            sheet.paste(thumb, ((idx % 5) * 256, (idx // 5) * 256))
        sheet.save(out_dir / f"{split}_labels_page_{page + 1:02d}.jpg", quality=92)


def leakage_report(images: pd.DataFrame) -> dict[str, Any]:
    train = images[images["split"].eq("train")]
    val = images[images["split"].eq("val")]
    train_source = train[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    val_source = val[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    return {
        "region_overlap": sorted(set(train["region"].astype(str)) & set(val["region"].astype(str))),
        "source_id_overlap": sorted(set(train_source) & set(val_source)),
        "raster_file_overlap": sorted(set(train["raster_file"].astype(str)) & set(val["raster_file"].astype(str))),
    }


def describe_split(meta: pd.DataFrame) -> pd.DataFrame:
    images = image_table(meta)
    boxes = meta[meta["class_name"].notna()]
    rows = []
    for split in ["train", "val"]:
        split_images = images[images["split"].eq(split)]
        split_boxes = boxes[boxes["split"].eq(split)]
        rows.append(
            {
                "split": split,
                "images": len(split_images),
                "positive": int(split_images["is_positive"].sum()),
                "negative": int((~split_images["is_positive"]).sum()),
                "bbox": len(split_boxes),
            }
        )
    return pd.DataFrame(rows)


def write_reports(out_dir: Path, final_meta: pd.DataFrame, pre_images: pd.DataFrame, pre_objects: pd.DataFrame, stats: dict[str, Any], split_info: dict[str, Any]) -> None:
    images = image_table(final_meta)
    boxes = final_meta[final_meta["class_name"].notna()]
    leakage = leakage_report(images)
    split_df = describe_split(final_meta)
    old_removed_rows = [
        {"old_split": key[0], "decision": key[1], "images": value}
        for key, value in stats.get("removed_from_old_split", {}).items()
    ]
    old_removed = pd.DataFrame(old_removed_rows)

    bbox_by_source = boxes["source_class_name"].value_counts().rename_axis("source_class").reset_index(name="bbox")
    class_split = boxes.groupby(["split", "source_class_name"]).size().reset_index(name="bbox")
    val_positive = int(split_df.loc[split_df["split"].eq("val"), "positive"].iloc[0])

    warnings = []
    if "WARNING_empty_decisions_excluded" in stats:
        warnings.append(f"Empty manual decisions excluded: {stats['WARNING_empty_decisions_excluded']}")
    if val_positive < 30:
        warnings.append(f"val positive images < 30: {val_positive}")
    if leakage["region_overlap"] or leakage["source_id_overlap"] or leakage["raster_file_overlap"]:
        warnings.append("Leakage detected")

    lines = [
        "# v3g Manual Keep-Only Dataset Audit",
        "",
        "## Warnings",
        "",
        "\n".join(f"- WARNING: {w}" for w in warnings) if warnings else "- none",
        "",
        "## Manual Audit + v3b Filters",
        "",
        f"- total images after v3b filters: `{stats['images_after_v3b_filters']}`",
        f"- kept images: `{len(pre_images)}`",
        f"- positive kept: `{int(pre_images['image'].isin(set(pre_objects['image'])).sum())}`",
        f"- negative kept: `{int((~pre_images['image'].isin(set(pre_objects['image']))).sum())}`",
        f"- bbox kept: `{len(pre_objects)}`",
        f"- final dataset images: `{len(images)}`",
        f"- final bbox: `{len(boxes)}`",
        "",
        "Decision counts after v3b filters:",
        "",
        markdown_table(pd.Series(stats["manual_decision_counts_after_v3b"]).rename_axis("decision").reset_index(name="images")),
        "",
        "Removed from old train/val:",
        "",
        markdown_table(old_removed),
        "",
        "BBox removed by excluded decisions:",
        "",
        markdown_table(pd.Series(stats.get("excluded_bbox_by_decision", {})).rename_axis("decision").reset_index(name="bbox")),
        "",
        "## New Split",
        "",
        markdown_table(split_df),
        "",
        "Train/val source classes:",
        "",
        markdown_table(class_split),
        "",
        "BBox by source class:",
        "",
        markdown_table(bbox_by_source),
        "",
        "Val regions:",
        "",
        ", ".join(split_info["val_regions"]),
        "",
        "## Leakage",
        "",
        f"- region overlap: `{len(leakage['region_overlap'])}`",
        f"- source_id overlap: `{len(leakage['source_id_overlap'])}`",
        f"- raster_file overlap: `{len(leakage['raster_file_overlap'])}`",
        "",
        "## Answers",
        "",
        f"1. Data left after manual cleaning: `{len(images)}` images and `{len(boxes)}` bbox.",
        "2. Positive/negative balance is shown in the split table; negative sampling remains 1:1 before split where possible.",
        "3. BBox distribution is saved as `bbox_area_distribution.png` and `bbox_width_height.png`; median/source-class counts are listed above.",
        f"4. Objects lost due to excluded manual decisions: `{sum(stats.get('excluded_bbox_by_decision', {}).values())}` bbox.",
        f"5. New validation has `{val_positive}` positive images; see warning if below target.",
        "6. Leakage is acceptable only if all overlap counts are zero.",
        "7. This dataset can be used as a clean baseline if validation positive count is acceptable and leakage counts are zero.",
        "",
    ]
    (out_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")

    leakage_lines = [
        "# Split Leakage Report",
        "",
        f"- region overlap count: `{len(leakage['region_overlap'])}`",
        f"- source_id overlap count: `{len(leakage['source_id_overlap'])}`",
        f"- raster_file overlap count: `{len(leakage['raster_file_overlap'])}`",
        "",
        "## Region Overlap",
        "",
        "\n".join(leakage["region_overlap"]) if leakage["region_overlap"] else "none",
        "",
        "## Source ID Overlap",
        "",
        "\n".join(leakage["source_id_overlap"]) if leakage["source_id_overlap"] else "none",
        "",
        "## Raster File Overlap",
        "",
        "\n".join(leakage["raster_file_overlap"]) if leakage["raster_file_overlap"] else "none",
        "",
    ]
    (out_dir / "split_leakage_report.md").write_text("\n".join(leakage_lines), encoding="utf-8")


def write_figures(out_dir: Path, meta: pd.DataFrame) -> None:
    images = image_table(meta)
    boxes = meta[meta["class_name"].notna()]
    save_bar(boxes["source_class_name"].value_counts(), "Class Balance by Source BBox", "bbox count", out_dir / "class_balance.png")
    save_bar(images["is_positive"].map({True: "positive", False: "negative"}).value_counts(), "Positive / Negative Images", "image count", out_dir / "positive_negative_ratio.png")
    save_hist(boxes["bbox_area_px"], "BBox Area Distribution", "bbox area px", out_dir / "bbox_area_distribution.png", log_x=True)
    widths = pd.to_numeric(boxes["bbox_x2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_x1_px"], errors="coerce")
    heights = pd.to_numeric(boxes["bbox_y2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_y1_px"], errors="coerce")
    save_width_height(meta, out_dir / "bbox_width_height.png")
    save_hist(images["n_objects"], "Objects Per Image", "objects per image", out_dir / "objects_per_image.png")
    save_label_pages(meta, out_dir, "train")
    save_label_pages(meta, out_dir, "val")


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    source_dir = args.source_dir.resolve()
    out_dir = (args.output_root / OUTPUT_NAME).resolve()
    source_meta = read_source_metadata(source_dir)
    audit = read_decisions(args.audit_index.resolve(), args.decisions.resolve())

    keep_images, keep_objects, stats = build_v3b_keep_candidates(source_meta, audit)
    keep_images = keep_images.rename(columns={"split": "old_split"})
    split_images, split_info = make_region_split(keep_images, keep_objects, args.val_size)
    split_images["source_id"] = split_images[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)

    final_meta = materialize_dataset(source_dir, split_images, keep_objects, out_dir, args.overwrite)
    write_figures(out_dir, final_meta)
    write_reports(out_dir, final_meta, keep_images, keep_objects, stats, split_info)

    summary = {
        "output_dir": str(out_dir),
        "images": int(len(image_table(final_meta))),
        "bbox": int(final_meta["class_name"].notna().sum()),
        "split": describe_split(final_meta).to_dict(orient="records"),
        "leakage": leakage_report(image_table(final_meta)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
