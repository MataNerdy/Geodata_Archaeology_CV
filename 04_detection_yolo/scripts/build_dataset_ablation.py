#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


RANDOM_SEED = 42
KURGAN_CLASSES = {"kurgany_tselye", "kurgany_povrezhdennye"}
KURGAN_CLASS_ID_MAP = {0: 0, 1: 0}
OUTPUT_NAMES = {0: "kurgan"}
SOURCE_ID_COLUMNS = ["region", "modality", "raster_file"]


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    output_name: str
    modalities: set[str]
    valid_fraction_min: float
    bbox_area_min: float | None
    bbox_area_max: float | None
    drop_edge_bboxes: bool
    max_source_objects: int | None
    negative_ratio: float


VERSIONS = [
    DatasetVersion(
        name="v3a_minimal",
        output_name="dataset_yolo_bbox_v3a_li_binary_minimal",
        modalities={"Li"},
        valid_fraction_min=0.80,
        bbox_area_min=None,
        bbox_area_max=None,
        drop_edge_bboxes=False,
        max_source_objects=None,
        negative_ratio=1.0,
    ),
    DatasetVersion(
        name="v3b_medium",
        output_name="dataset_yolo_bbox_v3b_li_binary_medium",
        modalities={"Li"},
        valid_fraction_min=0.85,
        bbox_area_min=100.0,
        bbox_area_max=None,
        drop_edge_bboxes=False,
        max_source_objects=50,
        negative_ratio=1.0,
    ),
    DatasetVersion(
        name="v3c_strict",
        output_name="dataset_yolo_bbox_v3c_li_binary_strict",
        modalities={"Li"},
        valid_fraction_min=0.90,
        bbox_area_min=240.0,
        bbox_area_max=435600.0,
        drop_edge_bboxes=True,
        max_source_objects=20,
        negative_ratio=1.0,
    ),
    DatasetVersion(
        name="v3d_li_ae_medium",
        output_name="dataset_yolo_bbox_v3d_li_ae_binary_medium",
        modalities={"Li", "Ae"},
        valid_fraction_min=0.85,
        bbox_area_min=100.0,
        bbox_area_max=None,
        drop_edge_bboxes=False,
        max_source_objects=50,
        negative_ratio=1.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset ablation versions and audits.")
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dataset_ablation"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_source_metadata(source_dir: Path) -> pd.DataFrame:
    meta_path = source_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    required = {
        "split",
        "region",
        "modality",
        "raster_file",
        "image",
        "label",
        "is_positive",
        "n_objects",
        "valid_fraction",
        "class_id",
        "class_name",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
        "bbox_area_px",
        "bbox_touches_tile_edge",
        "tile_touches_raster_edge",
        "has_edge_object",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Source metadata is missing columns: {missing}")

    df["is_positive"] = df["is_positive"].astype(bool)
    df["n_objects"] = pd.to_numeric(df["n_objects"], errors="coerce").fillna(0).astype(int)
    df["valid_fraction"] = pd.to_numeric(df["valid_fraction"], errors="coerce")
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df["bbox_area_px"] = pd.to_numeric(df["bbox_area_px"], errors="coerce")
    df["bbox_touches_tile_edge"] = df["bbox_touches_tile_edge"].astype("boolean")
    df["tile_touches_raster_edge"] = df["tile_touches_raster_edge"].astype(bool)
    df["has_edge_object"] = df["has_edge_object"].astype(bool)
    df["source_id"] = df[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    return df


def image_table(meta: pd.DataFrame) -> pd.DataFrame:
    return meta.drop_duplicates("image").copy()


def source_kurgan_objects(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[meta["class_name"].isin(KURGAN_CLASSES)].copy()


def resolve_dataset_path(source_dir: Path, path_value: Any, kind: str, split: str) -> Path:
    p = Path(str(path_value))
    if p.exists():
        return p
    candidate = source_dir / kind / split / p.name
    if candidate.exists():
        return candidate
    candidate = source_dir / p
    if candidate.exists():
        return candidate
    return p


def yolo_box_from_metadata(row: pd.Series) -> tuple[int, float, float, float, float]:
    old_class_id = int(row["class_id"])
    tile_size = float(row.get("tile_size", 1024))
    x1 = float(row["bbox_x1_px"])
    y1 = float(row["bbox_y1_px"])
    x2 = float(row["bbox_x2_px"])
    y2 = float(row["bbox_y2_px"])
    cls_id = KURGAN_CLASS_ID_MAP[old_class_id]
    xc = ((x1 + x2) / 2.0) / tile_size
    yc = ((y1 + y2) / 2.0) / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return cls_id, xc, yc, w, h


def valid_yolo_box(box: tuple[int, float, float, float, float]) -> bool:
    _, xc, yc, w, h = box
    return 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1


def write_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [
        f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for cls_id, xc, yc, w, h in boxes
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_yaml(out_dir: Path) -> None:
    text = f"""path: {out_dir.resolve()}

train: images/train
val: images/val

names:
  0: kurgan
"""
    (out_dir / "dataset.yaml").write_text(text, encoding="utf-8")


def state_counts(images: pd.DataFrame, objects: pd.DataFrame) -> dict[str, int]:
    positive_images = set(objects["image"]) if not objects.empty else set()
    return {
        "images": int(len(images)),
        "positive_images": int(images["image"].isin(positive_images).sum()),
        "bbox": int(len(objects)),
    }


def append_impact(
    rows: list[dict[str, Any]],
    version: str,
    step_name: str,
    before_images: pd.DataFrame,
    before_objects: pd.DataFrame,
    after_images: pd.DataFrame,
    after_objects: pd.DataFrame,
) -> None:
    before = state_counts(before_images, before_objects)
    after = state_counts(after_images, after_objects)
    rows.append(
        {
            "dataset_version": version,
            "step_name": step_name,
            "images_before": before["images"],
            "images_after": after["images"],
            "positive_before": before["positive_images"],
            "positive_after": after["positive_images"],
            "bbox_before": before["bbox"],
            "bbox_after": after["bbox"],
            "removed_images": before["images"] - after["images"],
            "removed_positive_images": before["positive_images"] - after["positive_images"],
            "removed_bbox": before["bbox"] - after["bbox"],
        }
    )


def apply_area_filter(objects: pd.DataFrame, version: DatasetVersion) -> pd.DataFrame:
    out = objects.copy()
    if version.bbox_area_min is not None:
        out = out[out["bbox_area_px"] >= version.bbox_area_min].copy()
    if version.bbox_area_max is not None:
        out = out[out["bbox_area_px"] <= version.bbox_area_max].copy()
    return out


def build_candidates(
    source_meta: pd.DataFrame,
    version: DatasetVersion,
    impact_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = image_table(source_meta)
    objects = source_kurgan_objects(source_meta)

    before_images, before_objects = images, objects
    images = images[images["modality"].isin(version.modalities)].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(impact_rows, version.name, "modality_filter", before_images, before_objects, images, objects)

    before_images, before_objects = images, objects
    images = images[images["valid_fraction"] >= version.valid_fraction_min].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        version.name,
        f"valid_fraction_gte_{version.valid_fraction_min}",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    objects = apply_area_filter(objects, version)
    positive_image_ids = set(objects["image"])
    source_kurgan_image_ids = set(source_kurgan_objects(source_meta)["image"])
    clean_negative_ids = set(images[~images["image"].isin(source_kurgan_image_ids)]["image"])
    images = images[images["image"].isin(positive_image_ids | clean_negative_ids)].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    area_label = "bbox_area_filter"
    if version.bbox_area_min is None and version.bbox_area_max is None:
        area_label = "bbox_area_filter_disabled"
    append_impact(impact_rows, version.name, area_label, before_images, before_objects, images, objects)

    before_images, before_objects = images, objects
    if version.drop_edge_bboxes:
        edge_mask = objects["bbox_touches_tile_edge"].fillna(False).astype(bool)
        objects = objects[~edge_mask].copy()
        positive_image_ids = set(objects["image"])
        images = images[images["image"].isin(positive_image_ids | clean_negative_ids)].copy()
        objects = objects[objects["image"].isin(set(images["image"]))].copy()
        step_name = "bbox_edge_filter"
    else:
        step_name = "bbox_edge_filter_disabled"
    append_impact(impact_rows, version.name, step_name, before_images, before_objects, images, objects)

    before_images, before_objects = images, objects
    if version.max_source_objects is not None:
        images = images[images["n_objects"] <= version.max_source_objects].copy()
        objects = objects[objects["image"].isin(set(images["image"]))].copy()
        step_name = f"n_objects_lte_{version.max_source_objects}"
    else:
        step_name = "n_objects_cutoff_disabled"
    append_impact(impact_rows, version.name, step_name, before_images, before_objects, images, objects)

    before_images, before_objects = images, objects
    positive_ids = set(objects["image"])
    positives = images[images["image"].isin(positive_ids)].copy()
    negatives = images[~images["image"].isin(positive_ids)].copy()
    n_negatives = min(len(negatives), int(round(len(positives) * version.negative_ratio)))
    negatives = negatives.sample(n=n_negatives, random_state=RANDOM_SEED) if n_negatives else negatives.iloc[0:0].copy()
    images = pd.concat([positives, negatives], ignore_index=True).sort_values(["split", "image"]).reset_index(drop=True)
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        version.name,
        f"negative_sampling_ratio_{version.negative_ratio:g}",
        before_images,
        before_objects,
        images,
        objects,
    )

    return images, objects


def build_dataset(
    source_dir: Path,
    source_meta: pd.DataFrame,
    version: DatasetVersion,
    output_root: Path,
    overwrite: bool,
    impact_rows: list[dict[str, Any]],
) -> Path:
    out_dir = output_root / version.output_name
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists. Pass --overwrite to replace it: {out_dir}")
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images, objects = build_candidates(source_meta, version, impact_rows)
    object_groups = {image: group.copy() for image, group in objects.groupby("image")}

    rows: list[dict[str, Any]] = []
    for _, image_row in images.iterrows():
        split = str(image_row["split"])
        old_img = resolve_dataset_path(source_dir, image_row["image"], "images", split)
        old_lbl = resolve_dataset_path(source_dir, image_row["label"], "labels", split)
        if not old_img.exists():
            continue

        new_img = out_dir / "images" / split / old_img.name
        new_lbl = out_dir / "labels" / split / old_lbl.name
        shutil.copy2(old_img, new_img)

        source_objects = object_groups.get(image_row["image"], pd.DataFrame())
        box_records: list[tuple[pd.Series, tuple[int, float, float, float, float]]] = []
        for _, obj in source_objects.iterrows():
            box = yolo_box_from_metadata(obj)
            if valid_yolo_box(box):
                box_records.append((obj, box))
        write_label(new_lbl, [box for _, box in box_records])

        base = image_row.to_dict()
        base["image"] = str(new_img)
        base["label"] = str(new_lbl)
        base["source_image"] = image_row["image"]
        base["source_label"] = image_row["label"]
        base["source_n_objects"] = int(image_row["n_objects"])
        base["is_positive"] = bool(box_records)
        base["n_objects"] = len(box_records)

        if box_records:
            for obj, box in box_records:
                cls_id, xc, yc, w, h = box
                row = base.copy()
                row.update(
                    {
                        "class_id": cls_id,
                        "class_name": OUTPUT_NAMES[cls_id],
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
            row.update(
                {
                    "class_id": None,
                    "class_name": None,
                    "source_class_id": None,
                    "source_class_name": None,
                    "bbox_x1_px": None,
                    "bbox_y1_px": None,
                    "bbox_x2_px": None,
                    "bbox_y2_px": None,
                    "bbox_area_px": None,
                    "bbox_touches_tile_edge": None,
                    "yolo_xc": None,
                    "yolo_yc": None,
                    "yolo_w": None,
                    "yolo_h": None,
                }
            )
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    write_dataset_yaml(out_dir)
    return out_dir


def save_bar(data: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    data.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    table = df.copy()
    if index:
        table = table.reset_index()
    table.columns = [str(c) for c in table.columns]
    rows = []
    rows.append("| " + " | ".join(table.columns) + " |")
    rows.append("|" + "|".join("---" for _ in table.columns) + "|")
    for _, row in table.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(rows)


def series_markdown(series: pd.Series, name: str = "value") -> str:
    df = series.rename(name).reset_index()
    df.columns = ["item", name]
    return markdown_table(df, index=False)


def save_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, log_x: bool = False) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0] if log_x else values
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=50, edgecolor="black")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_width_height_plot(meta: pd.DataFrame, out_path: Path) -> None:
    boxes = meta[meta["bbox_x1_px"].notna()].copy()
    widths = pd.to_numeric(boxes["bbox_x2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_x1_px"], errors="coerce")
    heights = pd.to_numeric(boxes["bbox_y2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_y1_px"], errors="coerce")
    plt.figure(figsize=(7, 7))
    plt.scatter(widths, heights, s=12, alpha=0.55)
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
        if len(parts) != 5:
            continue
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


def save_label_pages(images: pd.DataFrame, out_dir: Path, split: str, pages: int = 5) -> None:
    subset = images[images["split"] == split].copy()
    if subset.empty:
        sampled = subset
    else:
        n = min(len(subset), pages * 25)
        sampled = subset.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for page in range(pages):
        sheet = Image.new("RGB", (5 * 256, 5 * 256), "white")
        page_rows = sampled.iloc[page * 25 : (page + 1) * 25]
        for idx, row in enumerate(page_rows.itertuples(index=False)):
            thumb = draw_thumb(Path(row.image), Path(row.label))
            x = (idx % 5) * 256
            y = (idx // 5) * 256
            sheet.paste(thumb, (x, y))
        sheet.save(out_dir / f"{split}_labels_page_{page + 1:02d}.jpg", quality=92)


def leakage_counts(images: pd.DataFrame) -> dict[str, int]:
    train = images[images["split"] == "train"]
    val = images[images["split"] == "val"]
    train_source = train[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    val_source = val[SOURCE_ID_COLUMNS].astype(str).agg("|".join, axis=1)
    return {
        "region_overlap": len(set(train["region"].astype(str)) & set(val["region"].astype(str))),
        "source_id_overlap": len(set(train_source) & set(val_source)),
        "raster_file_overlap": len(set(train["raster_file"].astype(str)) & set(val["raster_file"].astype(str))),
    }


def dataset_metrics(meta: pd.DataFrame) -> dict[str, Any]:
    images = image_table(meta)
    boxes = meta[meta["class_name"].notna()].copy()
    positive_images = int(images["is_positive"].sum())
    negative_images = int((~images["is_positive"]).sum())
    width = pd.to_numeric(boxes["bbox_x2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_x1_px"], errors="coerce")
    height = pd.to_numeric(boxes["bbox_y2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_y1_px"], errors="coerce")
    edge_ratio = float(boxes["bbox_touches_tile_edge"].fillna(False).astype(bool).mean()) if len(boxes) else 0.0
    return {
        "images_total": int(len(images)),
        "train_images": int((images["split"] == "train").sum()),
        "val_images": int((images["split"] == "val").sum()),
        "positive_images": positive_images,
        "negative_images": negative_images,
        "bbox_total": int(len(boxes)),
        "bbox_per_positive_image_mean": float(images[images["is_positive"]]["n_objects"].mean()) if positive_images else 0.0,
        "median_bbox_area": float(pd.to_numeric(boxes["bbox_area_px"], errors="coerce").median()) if len(boxes) else 0.0,
        "edge_bbox_ratio": edge_ratio,
        "valid_fraction_mean": float(images["valid_fraction"].mean()),
        "modalities": ",".join(sorted(images["modality"].dropna().astype(str).unique())),
        "classes_used": ",".join(sorted(boxes["class_name"].dropna().astype(str).unique())),
        "bbox_width_median": float(width.median()) if len(width) else 0.0,
        "bbox_height_median": float(height.median()) if len(height) else 0.0,
        **leakage_counts(images),
    }


def write_audit(out_dir: Path, title: str, meta: pd.DataFrame, is_source: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = image_table(meta)
    boxes = meta[meta["class_name"].notna()].copy()
    metrics = dataset_metrics(meta)

    save_bar(boxes["class_name"].value_counts(), "Class Balance by BBox", "bbox count", out_dir / "class_balance.png")
    save_bar(images["is_positive"].map({True: "positive", False: "negative"}).value_counts(), "Positive / Negative Images", "image count", out_dir / "positive_negative_ratio.png")
    save_hist(boxes["bbox_area_px"], "BBox Area Distribution", "bbox_area_px", out_dir / "bbox_area_distribution.png", log_x=True)
    save_hist(images["n_objects"], "Objects Per Image", "objects per image", out_dir / "objects_per_image.png")
    save_width_height_plot(meta, out_dir / "bbox_width_height.png")
    if not is_source:
        save_label_pages(images, out_dir, "train", pages=5)
        save_label_pages(images, out_dir, "val", pages=5)

    class_images = (
        meta[meta["class_name"].notna()]
        .drop_duplicates(["image", "class_name"])
        .groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )
    width = pd.to_numeric(boxes["bbox_x2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_x1_px"], errors="coerce")
    height = pd.to_numeric(boxes["bbox_y2_px"], errors="coerce") - pd.to_numeric(boxes["bbox_y1_px"], errors="coerce")
    aspect = width / height.replace(0, pd.NA)
    leakage = leakage_counts(images)
    warning = ""
    if metrics["positive_images"] < 100:
        warning = "\n\n> WARNING: positive images < 100.\n"
    if leakage["region_overlap"] or leakage["source_id_overlap"]:
        warning += "\n\n> WARNING: train/val leakage detected.\n"

    lines = [
        f"# {title}",
        warning,
        "## Dataset Level",
        "",
        f"- Images: `{metrics['images_total']}`",
        f"- Positive images: `{metrics['positive_images']}`",
        f"- Negative images: `{metrics['negative_images']}`",
        f"- BBox count: `{metrics['bbox_total']}`",
        "",
        "## Split",
        "",
        series_markdown(images["split"].value_counts().sort_index(), "images"),
        "",
        "## Classes",
        "",
        "### BBox balance",
        "",
        series_markdown(boxes["class_name"].value_counts(), "bbox"),
        "",
        "### Image balance by class",
        "",
        series_markdown(class_images, "images"),
        "",
        "## Modalities",
        "",
        series_markdown(images["modality"].value_counts(), "images"),
        "",
        "## BBox Statistics",
        "",
        markdown_table(
            pd.DataFrame(
                {
                    "area": boxes["bbox_area_px"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
                    "width": width.describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
                    "height": height.describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
                    "aspect_ratio": aspect.describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
                }
            ),
            index=True,
        ),
        "",
        "## Image Statistics",
        "",
        "### Objects per image",
        "",
        series_markdown(images["n_objects"].describe(percentiles=[0.9, 0.95, 0.99]), "n_objects"),
        "",
        "### Objects per positive image",
        "",
        series_markdown(images[images["is_positive"]]["n_objects"].describe(percentiles=[0.9, 0.95, 0.99]), "n_objects"),
        "",
        "## Quality Metrics",
        "",
        f"- Edge bbox ratio: `{metrics['edge_bbox_ratio']:.4f}`",
        f"- valid_fraction mean: `{metrics['valid_fraction_mean']:.4f}`",
        "",
        "### valid_fraction",
        "",
        series_markdown(images["valid_fraction"].describe(percentiles=[0.1, 0.5, 0.9, 0.95, 0.99]), "valid_fraction"),
        "",
        "### tile_touches_raster_edge",
        "",
        series_markdown(images["tile_touches_raster_edge"].value_counts(dropna=False), "images"),
        "",
        "### has_edge_object",
        "",
        series_markdown(images["has_edge_object"].value_counts(dropna=False), "images"),
        "",
        "## Leakage",
        "",
        series_markdown(pd.Series(leakage), "count"),
        "",
        "## Figures",
        "",
        "- `class_balance.png`",
        "- `positive_negative_ratio.png`",
        "- `bbox_area_distribution.png`",
        "- `objects_per_image.png`",
        "- `bbox_width_height.png`",
    ]
    if not is_source:
        lines.extend(["- `train_labels_page_01..05.jpg`", "- `val_labels_page_01..05.jpg`"])
    lines.append("")
    (out_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def write_pipeline_report(report_root: Path) -> None:
    text = """# YOLO Dataset Pipeline Inspection

## Pipeline Scheme

```mermaid
flowchart TD
    A["Region folders"] --> B["GeoJSON markup + raster modalities"]
    B --> C["Choose raster per modality"]
    C --> D["Tile raster windows with overlap"]
    D --> E["Intersect GeoJSON objects with tile geometry"]
    E --> F["Clip polygons to tile bounds"]
    F --> G["Convert clipped polygons to YOLO bbox"]
    G --> H["Apply source quality filters"]
    H --> I["Write PNG tile + YOLO txt label"]
    I --> J["Write metadata.csv row per bbox / negative tile"]
    J --> K["Region-based train/val split"]
    K --> L["Ablation filters A-D"]
    L --> M["Binary kurgan datasets"]
    M --> N["Audit figures + filter impact + comparison"]
```

## Source Images

Images are generated by `scripts/build_yolo_dataset_bbox.py` from geospatial rasters discovered through `overlay_5_classes.find_regions(DATASET_ROOT)`. For each region and modality (`Li`, `Ae`, `SpOr`, `Or`), the script selects an available raster, tiles it with overlap, converts raster windows to RGB PNG tiles, and stores them under `images/train` or `images/val`.

## Labels and BBoxes

Vector markup is loaded from per-region GeoJSON files, reprojected to the raster CRS, intersected with each tile window, clipped to tile bounds, and converted to YOLO bbox format. The source class mapping is:

| Source class | YOLO id |
|---|---:|
| `kurgany_tselye` | 0 |
| `kurgany_povrezhdennye` | 1 |
| `gorodishcha` | 2 |
| `fortifikatsii` | 3 |
| `arkhitektury` | 4 |

The ablation datasets keep only `kurgany_tselye` and `kurgany_povrezhdennye`, remapped into a single `kurgan` class (`0`).

## Source Metadata

`metadata.csv` is written by the source generator with one row per bbox and one row for negative images. It stores split, region, modality, raster file, image/label paths, tile geometry, raster quality metrics, object count, bbox pixel coordinates, bbox area, and edge-touch flags.

## Split

The source split is region-based. The generator shuffles region names with `RANDOM_SEED = 42`, assigns `VAL_REGION_FRACTION = 0.2` to validation, and then all tiles from a region inherit that split. Ablation datasets preserve this split and do not create a new random image-level split.

## Negative Sampling

The source generator keeps negative tiles with `NEGATIVE_RATIO = 0.25`. In this ablation, `negative_ratio = 1.0` means at most one clean negative image per positive image, sampled with `RANDOM_SEED = 42`. Images where source kurgan objects exist but are removed by bbox filters are not converted into negatives.

## Existing Source Filters

The source generator already applies:

- minimum bbox area: `MIN_BBOX_AREA_PX = 80`;
- minimum valid raster fraction: `MIN_VALID_FRACTION = 0.35`;
- minimum raster standard deviation: `MIN_STD = 5`;
- minimum contrast: `MIN_P98_P2 = 10`;
- negative sampling;
- optional edge-object dropping is disabled in the current source generator.

## Leakage Risk

The intended leakage control is region-level splitting. The audit checks overlap by `region`, `source_id = region | modality | raster_file`, and `raster_file`.
"""
    (report_root / "pipeline_report.md").write_text(text, encoding="utf-8")


def write_filter_impact(report_root: Path, impact_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(impact_rows)
    csv_path = report_root / "filter_impact_table.csv"
    md_path = report_root / "filter_impact_table.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text("# Filter Impact Table\n\n" + markdown_table(df) + "\n", encoding="utf-8")
    return df


def write_comparison(report_root: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(report_root / "dataset_versions_comparison.csv", index=False)
    (report_root / "dataset_versions_comparison.md").write_text(
        "# Dataset Versions Comparison\n\n" + markdown_table(df) + "\n",
        encoding="utf-8",
    )
    return df


def write_final_report(report_root: Path, comparison: pd.DataFrame, impact: pd.DataFrame) -> None:
    positives_under_100 = comparison[comparison["positive_images"] < 100]["dataset_version"].tolist()
    strongest_positive = impact.sort_values("removed_positive_images", ascending=False).head(1)
    strongest_bbox = impact.sort_values("removed_bbox", ascending=False).head(1)
    non_modality = impact[impact["step_name"] != "modality_filter"].copy()
    strongest_positive_after_modality = non_modality.sort_values("removed_positive_images", ascending=False).head(1)
    strongest_bbox_after_modality = non_modality.sort_values("removed_bbox", ascending=False).head(1)
    medium_li = comparison[comparison["dataset_version"] == "v3b_medium"].iloc[0]
    medium_li_ae = comparison[comparison["dataset_version"] == "v3d_li_ae_medium"].iloc[0]

    lines = [
        "# Dataset Ablation Findings",
        "",
        "## Recommended First Baseline",
        "",
        (
            "`v3b_medium` is the safer first controlled baseline if the goal is to isolate Li-only behavior: "
            f"it keeps `{int(medium_li['positive_images'])}` positive images and `{int(medium_li['bbox_total'])}` boxes "
            "with moderate raster-quality and small-area filtering. "
            "`v3d_li_ae_medium` is the best scale-up candidate because it keeps "
            f"`{int(medium_li_ae['positive_images'])}` positive images and `{int(medium_li_ae['bbox_total'])}` boxes, "
            "but it changes the modality mix and should be compared against v3b rather than treated as a clean replacement."
        ),
        "",
        "## Strongest Positive-Image Filter Overall",
        "",
        markdown_table(strongest_positive),
        "",
        "## Strongest BBox Filter Overall",
        "",
        markdown_table(strongest_bbox),
        "",
        "## Strongest Positive-Image Filter After Modality Choice",
        "",
        markdown_table(strongest_positive_after_modality),
        "",
        "## Strongest BBox Filter After Modality Choice",
        "",
        markdown_table(strongest_bbox_after_modality),
        "",
        "## Over-Cleaning Signal",
        "",
        (
            "Versions with fewer than 100 positive images are likely over-cleaned for a stable YOLO baseline: "
            + (", ".join(f"`{x}`" for x in positives_under_100) if positives_under_100 else "none")
            + "."
        ),
        "",
        "## Remaining Label-Risk Areas",
        "",
        "- Very small boxes may represent ambiguous or barely visible damaged kurgans.",
        "- Edge-touching boxes can be valid partial objects, but they also encode tiling artifacts.",
        "- High-object-count tiles may contain dense archaeological zones; dropping them may remove useful hard cases.",
        "- Ae imagery may add useful context but also modality shift and weaker visual signal.",
        "- Strict filtering leaves only 73 positive images, so validation becomes fragile and recall estimates are noisy.",
        "",
        "## Manual Review Priority",
        "",
        "- Positive tiles near the lower bbox-area tail.",
        "- Tiles removed by the edge-bbox filter.",
        "- Images with high `n_objects` before cutoff.",
        "- Validation positives, because validation is small in strict settings.",
        "- False-negative candidates: source kurgan images where bbox filters removed all selected boxes.",
        "",
        "## Next Experiments",
        "",
        "1. Train YOLOv8n on `v3b_medium` at `imgsz=640` as the controlled Li-only baseline.",
        "2. Train the same config on `v3d_li_ae_medium` to test whether Ae helps or adds noise.",
        "3. Use `v3c_strict` only as a precision-oriented stress test, not as the first baseline.",
        "4. Inspect recall failures by bbox area bucket, especially damaged kurgans and very small boxes.",
        "5. Tune confidence only after dataset choice is fixed.",
        "",
    ]
    (report_root / "ablation_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    args.report_root.mkdir(parents=True, exist_ok=True)

    source_meta = read_source_metadata(args.source_dir)
    write_pipeline_report(args.report_root)
    source_metrics = write_audit(args.report_root / "source_dataset_audit", "Source YOLO Dataset Audit", source_meta, is_source=True)

    impact_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for version in VERSIONS:
        out_dir = build_dataset(args.source_dir, source_meta, version, args.output_root, args.overwrite, impact_rows)
        meta = read_source_metadata(out_dir)
        audit_dir = args.report_root / version.name
        metrics = write_audit(audit_dir, f"{version.name} Audit", meta, is_source=False)
        metrics["dataset_version"] = version.name
        metrics["dataset_path"] = str(out_dir)
        comparison_rows.append(metrics)
        if metrics["positive_images"] < 100:
            print(f"WARNING: {version.name} positive images < 100 ({metrics['positive_images']})")

    impact = write_filter_impact(args.report_root, impact_rows)
    comparison = write_comparison(args.report_root, comparison_rows)
    write_final_report(args.report_root, comparison, impact)

    print("=" * 80)
    print("Dataset ablation complete")
    print("source images:", source_metrics["images_total"])
    print("report root:", args.report_root)
    print("versions:")
    for row in comparison_rows:
        print(f"- {row['dataset_version']}: images={row['images_total']} positives={row['positive_images']} boxes={row['bbox_total']}")


if __name__ == "__main__":
    main()
