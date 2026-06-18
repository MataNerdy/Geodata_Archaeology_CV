#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    "kurgany_tselye": "#00ff66",
    "kurgany_povrezhdennye": "#ffcc00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare region-level review artifacts for manual validation split.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../datasets/dataset_yolo_bbox_v3g_li_medium_manual_keep_only"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("manual_val_region_review"))
    parser.add_argument("--thumb-size", type=int, default=260)
    parser.add_argument("--max-images-per-region", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("_") or "region"


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    for col in ["bbox_area_px", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "yolo_xc", "yolo_yc", "yolo_w", "yolo_h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
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


def image_objects(df: pd.DataFrame, image_key: str) -> pd.DataFrame:
    return df[(df["image"].astype(str) == image_key) & df["class_name"].notna()].copy()


def draw_tile(image_path: Path, objects: pd.DataFrame, label: str, thumb_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(thumb_size / original_w, thumb_size / original_h)
    new_size = (max(1, int(original_w * scale)), max(1, int(original_h * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for _, obj in objects.iterrows():
        cls = str(obj.get("source_class_name") or obj.get("class_name") or "kurgan")
        color = CLASS_COLORS.get(cls, "#00ff66")
        if pd.notna(obj.get("yolo_xc")):
            x1 = (float(obj["yolo_xc"]) - float(obj["yolo_w"]) / 2) * original_w
            y1 = (float(obj["yolo_yc"]) - float(obj["yolo_h"]) / 2) * original_h
            x2 = (float(obj["yolo_xc"]) + float(obj["yolo_w"]) / 2) * original_w
            y2 = (float(obj["yolo_yc"]) + float(obj["yolo_h"]) / 2) * original_h
        else:
            x1, y1, x2, y2 = (float(obj["bbox_x1_px"]), float(obj["bbox_y1_px"]), float(obj["bbox_x2_px"]), float(obj["bbox_y2_px"]))
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=color, width=2)
        short = cls.replace("kurgany_", "")[:10]
        text_w = max(28, len(short) * 6 + 5)
        y_text = max(0, box[1] - 13)
        draw.rectangle([box[0], y_text, box[0] + text_w, y_text + 12], fill=color)
        draw.text((box[0] + 2, y_text), short, fill="black", font=font)

    draw.rectangle([0, 0, image.width, 16], fill="black")
    draw.text((3, 2), label[:42], fill="white", font=font)
    return image


def write_contact_sheet(tiles: list[Image.Image], out_path: Path, columns: int = 5, pad: int = 8) -> None:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92, optimize=True)


def numeric_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def build_region_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    image_rows = df.drop_duplicates("image").copy()
    object_rows = df[df["class_name"].notna()].copy()
    positive_images = set(object_rows["image"].astype(str))
    rows = []
    for region, region_images in image_rows.groupby("region", sort=True):
        region_object_rows = object_rows[object_rows["region"].eq(region)].copy()
        region_positive = region_images[region_images["image"].astype(str).isin(positive_images)]
        bbox_by_source = region_object_rows["source_class_name"].value_counts()
        rows.append(
            {
                "region": region,
                "images_total": int(len(region_images)),
                "positive_images": int(len(region_positive)),
                "negative_images": int(len(region_images) - len(region_positive)),
                "bbox_total": int(len(region_object_rows)),
                "kurgany_tselye_bbox": int(bbox_by_source.get("kurgany_tselye", 0)),
                "kurgany_povrezhdennye_bbox": int(bbox_by_source.get("kurgany_povrezhdennye", 0)),
                "bbox_area_median": numeric_quantile(region_object_rows["bbox_area_px"], 0.50),
                "bbox_area_p25": numeric_quantile(region_object_rows["bbox_area_px"], 0.25),
                "bbox_area_p75": numeric_quantile(region_object_rows["bbox_area_px"], 0.75),
                "objects_per_positive_image_mean": float(region_object_rows.groupby("image").size().mean()) if not region_object_rows.empty else 0.0,
                "preview_contact_sheet_path": str((out_dir / f"{safe_name(str(region))}.jpg").as_posix()),
            }
        )
    return pd.DataFrame(rows)


def write_region_sheets(df: pd.DataFrame, dataset_dir: Path, out_dir: Path, thumb_size: int, max_images: int, overwrite: bool) -> None:
    image_rows = df.drop_duplicates("image").copy()
    object_rows = df[df["class_name"].notna()].copy()
    for region, region_images in image_rows.groupby("region", sort=True):
        out_path = out_dir / f"{safe_name(str(region))}.jpg"
        if out_path.exists() and not overwrite:
            continue
        region_images = region_images.copy()
        region_images["sort_positive"] = region_images["is_positive"].astype(int)
        region_images = region_images.sort_values(["sort_positive", "image_name"], ascending=[False, True])
        if len(region_images) > max_images:
            positives = region_images[region_images["is_positive"]]
            negatives = region_images[~region_images["is_positive"]]
            pos_take = min(len(positives), max_images // 2 + max_images % 2)
            neg_take = max_images - pos_take
            selected = pd.concat([positives.head(pos_take), negatives.head(neg_take)], ignore_index=True)
        else:
            selected = region_images
        tiles = []
        for _, row in selected.iterrows():
            image_path = resolve_image_path(dataset_dir, row)
            objects = object_rows[object_rows["image"].astype(str).eq(str(row["image"]))].copy()
            label = f"{row.get('image_name', image_path.name)} | n={len(objects)}"
            tiles.append(draw_tile(image_path, objects, label, thumb_size))
        write_contact_sheet(tiles, out_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_metadata(args.dataset_dir)
    write_region_sheets(df, args.dataset_dir, args.out_dir, args.thumb_size, args.max_images_per_region, args.overwrite)
    summary = build_region_summary(df, args.out_dir)
    summary_path = args.out_dir / "region_summary.csv"
    summary.to_csv(summary_path, index=False)

    template = summary[["region", "images_total", "positive_images", "negative_images", "bbox_total"]].copy()
    template["decision"] = ""
    template["comment"] = ""
    template_path = args.out_dir / "manual_val_regions.csv"
    if not template_path.exists() or args.overwrite:
        template.to_csv(template_path, index=False)

    print("Region summary:", summary_path)
    print("Manual decision template:", template_path)
    print("Contact sheets:", args.out_dir)
    print("Regions:", len(summary))


if __name__ == "__main__":
    main()
