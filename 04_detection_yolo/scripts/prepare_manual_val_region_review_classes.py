#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TARGET_CLASSES = ["gorodishcha", "fortifikatsii", "arkhitektury"]
CLASS_ID_TO_NAME = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}
CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}
CLASS_COLORS = {
    "gorodishcha": "#4cc9f0",
    "fortifikatsii": "#f72585",
    "arkhitektury": "#f9c74f",
    "kurgany_tselye": "#00ff66",
    "kurgany_povrezhdennye": "#ffcc00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare region-level review artifacts for selected YOLO classes.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--decisions", type=Path, default=Path("manual_audit/audit_decisions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("manual_val_region_review_other_classes"))
    parser.add_argument("--target-classes", nargs="+", default=DEFAULT_TARGET_CLASSES)
    parser.add_argument("--thumb-size", type=int, default=260)
    parser.add_argument("--max-images-per-region", type=int, default=80)
    parser.add_argument("--include-empty-regions", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("_") or "region"


def add_audit_image_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "source_image_name" if "source_image_name" in out.columns else "image_name"
    out["image_id"] = out.apply(lambda row: f"{row['split']}_{Path(str(row.get(name_col) or row['image'])).stem}", axis=1)
    return out


def read_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path).fillna("")
    decisions = decisions.drop_duplicates("image_id", keep="last")
    return decisions[["image_id", "decision", "reason", "comment", "updated_at"]].rename(
        columns={
            "decision": "manual_decision",
            "reason": "manual_reason",
            "comment": "manual_comment",
            "updated_at": "manual_updated_at",
        }
    )


def read_metadata(dataset_dir: Path, decisions_path: Path, target_classes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    for col in [
        "bbox_area_px",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
        "yolo_xc",
        "yolo_yc",
        "yolo_w",
        "yolo_h",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_audit_image_id(df)
    decisions = read_decisions(decisions_path)
    df = df.merge(decisions, on="image_id", how="left")
    df["manual_decision"] = df["manual_decision"].fillna("")
    unresolved = df.drop_duplicates("image").query("manual_decision == ''")
    if len(unresolved):
        print(f"WARNING: excluding {len(unresolved)} images without manual audit decision.")
    df = df[df["manual_decision"].eq("keep")].copy()
    df["is_target_object"] = df["source_class_name"].isin(target_classes)
    target_positive_images = set(df.loc[df["is_target_object"], "image"].astype(str))
    df["is_target_positive"] = df["image"].astype(str).isin(target_positive_images)
    return df


def resolve_image_path(dataset_dir: Path, row: pd.Series) -> Path:
    path = Path(str(row["image"]))
    if path.is_absolute() and path.exists():
        return path
    split = str(row["split"])
    image_name = str(row.get("image_name") or path.name)
    candidates = [
        dataset_dir / "images" / split / image_name,
        dataset_dir / "images" / split / path.name,
        dataset_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def yolo_to_xyxy(box: tuple[int, float, float, float, float], image_w: int, image_h: int) -> tuple[int, float, float, float, float]:
    cls_id, xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * image_w
    y1 = (yc - bh / 2) * image_h
    x2 = (xc + bw / 2) * image_w
    y2 = (yc + bh / 2) * image_h
    return cls_id, x1, y1, x2, y2


def resolve_label_path(dataset_dir: Path, row: pd.Series) -> Path:
    path = Path(str(row["label"]))
    if path.is_absolute() and path.exists():
        return path
    split = str(row["split"])
    label_name = str(row.get("label_name") or path.name)
    candidates = [
        dataset_dir / "labels" / split / label_name,
        dataset_dir / "labels" / split / path.name,
        dataset_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def draw_tile(image_path: Path, label_path: Path, label: str, thumb_size: int, target_class_ids: set[int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(thumb_size / original_w, thumb_size / original_h)
    preview = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for box in parse_label_file(label_path):
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, original_w, original_h)
        if cls_id not in target_class_ids:
            continue
        cls = CLASS_ID_TO_NAME.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls, "#00ff66")
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=color, width=2)
        short = cls[:12]
        text_w = max(32, len(short) * 6 + 5)
        y_text = max(0, box[1] - 13)
        draw.rectangle([box[0], y_text, box[0] + text_w, y_text + 12], fill=color)
        draw.text((box[0] + 2, y_text), short, fill="black", font=font)

    draw.rectangle([0, 0, preview.width, 16], fill="black")
    draw.text((3, 2), label[:42], fill="white", font=font)
    return preview


def write_contact_sheet(tiles: list[Image.Image], out_path: Path, columns: int = 5, pad: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not tiles:
        Image.new("RGB", (800, 160), "white").save(out_path, quality=90)
        return
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles)
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
    for idx, tile in enumerate(tiles):
        x = pad + (idx % columns) * (cell_w + pad)
        y = pad + (idx // columns) * (cell_h + pad)
        sheet.paste(tile, (x, y))
    sheet.save(out_path, quality=92, optimize=True)


def numeric_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def build_region_summary(df: pd.DataFrame, out_dir: Path, target_classes: list[str], include_empty_regions: bool) -> pd.DataFrame:
    image_rows = df.drop_duplicates("image").copy()
    object_rows = df[df["is_target_object"]].copy()
    positive_images = set(object_rows["image"].astype(str))
    rows = []
    for region, region_images in image_rows.groupby("region", sort=True):
        region_object_rows = object_rows[object_rows["region"].eq(region)].copy()
        region_positive = region_images[region_images["image"].astype(str).isin(positive_images)]
        bbox_by_source = region_object_rows["source_class_name"].value_counts()
        row = {
            "region": region,
            "images_total": int(len(region_images)),
            "positive_images": int(len(region_positive)),
            "negative_images": int(len(region_images) - len(region_positive)),
            "bbox_total": int(len(region_object_rows)),
            "bbox_area_median": numeric_quantile(region_object_rows["bbox_area_px"], 0.50),
            "bbox_area_p25": numeric_quantile(region_object_rows["bbox_area_px"], 0.25),
            "bbox_area_p75": numeric_quantile(region_object_rows["bbox_area_px"], 0.75),
            "objects_per_positive_image_mean": float(region_object_rows.groupby("image").size().mean()) if not region_object_rows.empty else 0.0,
            "preview_contact_sheet_path": str((out_dir / f"{safe_name(str(region))}.jpg").as_posix()),
        }
        for class_name in target_classes:
            row[f"{class_name}_bbox"] = int(bbox_by_source.get(class_name, 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not include_empty_regions:
        summary = summary[summary["bbox_total"].gt(0)].copy()
    return summary


def write_region_sheets(df: pd.DataFrame, dataset_dir: Path, out_dir: Path, thumb_size: int, max_images: int, overwrite: bool, include_empty_regions: bool) -> None:
    image_rows = df.drop_duplicates("image").copy()
    object_rows = df[df["is_target_object"]].copy()
    target_class_ids = set(pd.to_numeric(object_rows["class_id"], errors="coerce").dropna().astype(int).unique())
    for region, region_images in image_rows.groupby("region", sort=True):
        if not include_empty_regions and object_rows[object_rows["region"].eq(region)].empty:
            continue
        out_path = out_dir / f"{safe_name(str(region))}.jpg"
        if out_path.exists() and not overwrite:
            continue
        region_images = region_images.copy()
        region_images["sort_positive"] = region_images["is_target_positive"].astype(int)
        region_images = region_images.sort_values(["sort_positive", "image_name"], ascending=[False, True])
        if len(region_images) > max_images:
            positives = region_images[region_images["is_target_positive"]]
            negatives = region_images[~region_images["is_target_positive"]]
            pos_take = min(len(positives), max_images // 2 + max_images % 2)
            selected = pd.concat([positives.head(pos_take), negatives.head(max_images - pos_take)], ignore_index=True)
        else:
            selected = region_images
        tiles = []
        for _, row in selected.iterrows():
            image_path = resolve_image_path(dataset_dir, row)
            label_path = resolve_label_path(dataset_dir, row)
            objects = object_rows[object_rows["image"].astype(str).eq(str(row["image"]))].copy()
            label = f"{row.get('image_name', image_path.name)} | n={len(objects)}"
            tiles.append(draw_tile(image_path, label_path, label, thumb_size, target_class_ids))
        write_contact_sheet(tiles, out_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_classes = [str(c) for c in args.target_classes]
    unknown = sorted(set(target_classes) - set(CLASS_NAME_TO_ID))
    if unknown:
        raise ValueError(f"Unknown target classes: {unknown}")
    df = read_metadata(args.dataset_dir, args.decisions, set(target_classes))
    summary = build_region_summary(df, args.out_dir, target_classes, args.include_empty_regions)
    if not args.skip_contact_sheets:
        write_region_sheets(df, args.dataset_dir, args.out_dir, args.thumb_size, args.max_images_per_region, args.overwrite, args.include_empty_regions)
    summary_path = args.out_dir / "region_summary.csv"
    summary.to_csv(summary_path, index=False)

    template = summary[["region", "images_total", "positive_images", "negative_images", "bbox_total"]].copy()
    template["decision"] = ""
    template["comment"] = ""
    template_path = args.out_dir / "manual_val_regions.csv"
    if not template_path.exists() or args.overwrite:
        template.to_csv(template_path, index=False)

    print("Target classes:", ", ".join(target_classes))
    print("Region summary:", summary_path)
    print("Manual decision template:", template_path)
    print("Contact sheets:", args.out_dir)
    print("Regions:", len(summary))
    print("Regions with target objects:", int((summary["bbox_total"] > 0).sum()))


if __name__ == "__main__":
    main()
