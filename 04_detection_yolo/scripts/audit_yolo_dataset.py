#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}

CLASS_COLORS = {
    0: "#00ff66",
    1: "#ff3b30",
    2: "#00b7ff",
    3: "#ffd60a",
    4: "#ff4dff",
}


@dataclass(frozen=True)
class YoloBox:
    cls_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO bbox dataset and generate figures/reports.")
    parser.add_argument("--data-root", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/dataset_audit/yolo_bbox"))
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--skip-label-sheets", action="store_true")
    return parser.parse_args()


def read_metadata(data_root: Path) -> pd.DataFrame:
    meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    required = {"split", "image", "label", "region", "modality", "raster_file", "is_positive", "n_objects"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    meta["is_positive"] = meta["is_positive"].astype(bool)
    meta["n_objects"] = pd.to_numeric(meta["n_objects"], errors="coerce").fillna(0).astype(int)
    meta["source_id"] = meta["region"].astype(str) + "|" + meta["modality"].astype(str) + "|" + meta["raster_file"].astype(str)
    return meta


def resolve_dataset_path(data_root: Path, path_value: str, kind: str, split: str) -> Path:
    p = Path(str(path_value))
    if p.exists():
        return p
    candidate = data_root / kind / split / p.name
    if candidate.exists():
        return candidate
    candidate = data_root / p
    if candidate.exists():
        return candidate
    return p


def image_level_table(meta: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    images = meta.drop_duplicates("image").copy()
    images["image_path"] = [
        resolve_dataset_path(data_root, row.image, "images", row.split)
        for row in images.itertuples(index=False)
    ]
    images["label_path"] = [
        resolve_dataset_path(data_root, row.label, "labels", row.split)
        for row in images.itertuples(index=False)
    ]
    return images


def parse_label(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append(YoloBox(cls_id, xc, yc, w, h))
    return boxes


def draw_labeled_thumb(image_path: Path, label_path: Path, thumb_size: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    img.thumbnail((thumb_size, thumb_size))

    canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
    offset_x = (thumb_size - img.size[0]) // 2
    offset_y = (thumb_size - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))

    scale_x = img.size[0] / src_w
    scale_y = img.size[1] / src_h
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in parse_label(label_path):
        x1 = (box.xc - box.w / 2) * src_w * scale_x + offset_x
        y1 = (box.yc - box.h / 2) * src_h * scale_y + offset_y
        x2 = (box.xc + box.w / 2) * src_w * scale_x + offset_x
        y2 = (box.yc + box.h / 2) * src_h * scale_y + offset_y
        color = CLASS_COLORS.get(box.cls_id, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(box.cls_id)
        label_x = max(0, min(thumb_size - 16, x1))
        label_y = max(0, y1 - 12)
        draw.rectangle([label_x, label_y, label_x + 14, label_y + 12], fill=color)
        draw.text((label_x + 2, label_y), label, fill="black", font=font)

    return canvas


def sample_rows(images: pd.DataFrame, split: str, n: int, seed: int, positive_only: bool) -> pd.DataFrame:
    subset = images[images["split"] == split].copy()
    if positive_only:
        subset = subset[subset["is_positive"]]
    if len(subset) <= n:
        return subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subset.sample(n=n, random_state=seed).reset_index(drop=True)


def chunks(rows: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(rows), size):
        yield rows.iloc[start : start + size]


def save_contact_sheets(
    images: pd.DataFrame,
    split: str,
    n: int,
    out_dir: Path,
    seed: int,
    thumb_size: int,
    grid_cols: int,
    grid_rows: int,
    positive_only: bool,
) -> list[Path]:
    sampled = sample_rows(images, split, n=n, seed=seed, positive_only=positive_only)
    page_size = grid_cols * grid_rows
    paths = []

    for page_idx, page in enumerate(chunks(sampled, page_size), start=1):
        sheet = Image.new("RGB", (grid_cols * thumb_size, grid_rows * thumb_size), "white")
        for cell_idx, row in enumerate(page.itertuples(index=False)):
            thumb = draw_labeled_thumb(Path(row.image_path), Path(row.label_path), thumb_size)
            x = (cell_idx % grid_cols) * thumb_size
            y = (cell_idx // grid_cols) * thumb_size
            sheet.paste(thumb, (x, y))

        out_path = out_dir / f"{split}_labels_page_{page_idx:02d}.jpg"
        sheet.save(out_path, quality=92)
        paths.append(out_path)

    return paths


def save_bar(counter: Counter, title: str, ylabel: str, out_path: Path) -> None:
    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_bbox_area_distribution(meta: pd.DataFrame, out_path: Path) -> None:
    boxes = meta[meta["bbox_area_px"].notna()].copy()
    boxes["bbox_area_px"] = pd.to_numeric(boxes["bbox_area_px"], errors="coerce")
    boxes = boxes[boxes["bbox_area_px"].notna() & (boxes["bbox_area_px"] > 0)]

    plt.figure(figsize=(10, 6))
    for class_name, group in boxes.groupby("class_name"):
        plt.hist(group["bbox_area_px"], bins=60, alpha=0.45, label=class_name)
    plt.xscale("log")
    plt.title("BBox Area Distribution")
    plt.xlabel("bbox_area_px, log scale")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_objects_per_image(images: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(images["n_objects"], bins=range(0, int(images["n_objects"].max()) + 2), edgecolor="black")
    plt.title("Objects Per Image")
    plt.xlabel("objects per image")
    plt.ylabel("image count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def leakage_report(meta: pd.DataFrame, images: pd.DataFrame, out_path: Path) -> None:
    train = images[images["split"] == "train"]
    val = images[images["split"] == "val"]

    train_regions = set(train["region"].dropna().astype(str))
    val_regions = set(val["region"].dropna().astype(str))
    region_overlap = sorted(train_regions & val_regions)

    train_sources = set(train["source_id"].dropna().astype(str))
    val_sources = set(val["source_id"].dropna().astype(str))
    source_overlap = sorted(train_sources & val_sources)

    lines = [
        "# Split Leakage Check",
        "",
        "Leakage is checked at two levels:",
        "",
        "- `region`",
        "- `source_id = region | modality | raster_file`",
        "",
        "| Check | Train unique | Val unique | Overlap |",
        "|---|---:|---:|---:|",
        f"| Region | {len(train_regions)} | {len(val_regions)} | {len(region_overlap)} |",
        f"| Source ID | {len(train_sources)} | {len(val_sources)} | {len(source_overlap)} |",
        "",
    ]

    if region_overlap:
        lines.extend(["## Region overlap", ""])
        lines.extend(f"- `{x}`" for x in region_overlap[:100])
        if len(region_overlap) > 100:
            lines.append(f"- ... {len(region_overlap) - 100} more")
        lines.append("")

    if source_overlap:
        lines.extend(["## Source ID overlap", ""])
        lines.extend(f"- `{x}`" for x in source_overlap[:100])
        if len(source_overlap) > 100:
            lines.append(f"- ... {len(source_overlap) - 100} more")
        lines.append("")

    if not region_overlap and not source_overlap:
        lines.append("No split leakage found by `region` or derived `source_id`.")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def dataset_summary(meta: pd.DataFrame, images: pd.DataFrame, out_path: Path, figure_paths: list[Path]) -> None:
    class_counts = Counter(meta[meta["class_name"].notna() & (meta["class_name"] != "")]["class_name"])
    split_counts = Counter(images["split"])
    modality_counts = Counter(images["modality"])
    pos_counts = Counter("positive" if x else "negative" for x in images["is_positive"])

    lines = [
        "# YOLO BBox Dataset Audit",
        "",
        f"Dataset images: `{len(images)}`",
        f"Metadata rows: `{len(meta)}`",
        f"Positive images: `{pos_counts.get('positive', 0)}`",
        f"Negative images: `{pos_counts.get('negative', 0)}`",
        "",
        "## Split Counts",
        "",
        "| Split | Images |",
        "|---|---:|",
    ]
    lines.extend(f"| `{k}` | {v} |" for k, v in sorted(split_counts.items()))

    lines.extend(["", "## Modality Counts", "", "| Modality | Images |", "|---|---:|"])
    lines.extend(f"| `{k}` | {v} |" for k, v in sorted(modality_counts.items()))

    lines.extend(["", "## Class Balance", "", "| Class | Boxes |", "|---|---:|"])
    lines.extend(f"| `{k}` | {v} |" for k, v in class_counts.most_common())

    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{p.name}`" for p in figure_paths)
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(args.data_root)
    images = image_level_table(meta, args.data_root)

    figure_paths: list[Path] = []
    if args.skip_label_sheets:
        figure_paths.extend(sorted(args.out_dir.glob("train_labels_page_*.jpg")))
        figure_paths.extend(sorted(args.out_dir.glob("val_labels_page_*.jpg")))
    else:
        figure_paths.extend(
            save_contact_sheets(
                images,
                split="train",
                n=args.num_train,
                out_dir=args.out_dir,
                seed=args.seed,
                thumb_size=args.thumb_size,
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
                positive_only=args.positive_only,
            )
        )
        figure_paths.extend(
            save_contact_sheets(
                images,
                split="val",
                n=args.num_val,
                out_dir=args.out_dir,
                seed=args.seed,
                thumb_size=args.thumb_size,
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
                positive_only=args.positive_only,
            )
        )

    class_balance_path = args.out_dir / "class_balance.png"
    save_bar(
        Counter(meta[meta["class_name"].notna() & (meta["class_name"] != "")]["class_name"]),
        "Class Balance by BBox Count",
        "bbox count",
        class_balance_path,
    )
    figure_paths.append(class_balance_path)

    pos_neg_path = args.out_dir / "positive_negative_ratio.png"
    save_bar(
        Counter("positive" if x else "negative" for x in images["is_positive"]),
        "Positive / Negative Image Ratio",
        "image count",
        pos_neg_path,
    )
    figure_paths.append(pos_neg_path)

    bbox_area_path = args.out_dir / "bbox_area_distribution.png"
    save_bbox_area_distribution(meta, bbox_area_path)
    figure_paths.append(bbox_area_path)

    objects_per_image_path = args.out_dir / "objects_per_image.png"
    save_objects_per_image(images, objects_per_image_path)
    figure_paths.append(objects_per_image_path)

    leakage_report(meta, images, args.out_dir / "split_leakage_report.md")
    dataset_summary(meta, images, args.out_dir / "dataset_audit_summary.md", figure_paths)

    print("=" * 80)
    print("Dataset audit complete")
    print("data root:", args.data_root)
    print("out dir:", args.out_dir)
    print("images:", len(images))
    print("metadata rows:", len(meta))
    print("figures:", len(figure_paths))


if __name__ == "__main__":
    main()
