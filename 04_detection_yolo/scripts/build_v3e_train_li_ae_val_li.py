#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from build_dataset_ablation import (
    OUTPUT_NAMES,
    RANDOM_SEED,
    DatasetVersion,
    append_impact,
    apply_area_filter,
    image_table,
    read_source_metadata,
    resolve_dataset_path,
    source_kurgan_objects,
    valid_yolo_box,
    write_audit,
    write_dataset_yaml,
    write_label,
    yolo_box_from_metadata,
)


VERSION = DatasetVersion(
    name="v3e_train_li_ae_val_li",
    output_name="dataset_yolo_bbox_v3e_train_li_ae_val_li",
    modalities={"Li", "Ae"},
    valid_fraction_min=0.85,
    bbox_area_min=100.0,
    bbox_area_max=None,
    drop_edge_bboxes=False,
    max_source_objects=50,
    negative_ratio=1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3e YOLO dataset: train on Li+Ae, validate on Li only."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/dataset_ablation/v3e_train_li_ae_val_li"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_modality_mask(images: pd.DataFrame) -> pd.Series:
    split = images["split"].astype(str)
    modality = images["modality"].astype(str)
    train_ok = (split == "train") & modality.isin({"Li", "Ae"})
    val_ok = (split == "val") & (modality == "Li")
    return train_ok | val_ok


def sample_negatives_by_split(images: pd.DataFrame, positive_ids: set[str]) -> pd.DataFrame:
    positives = images[images["image"].isin(positive_ids)].copy()
    negatives = images[~images["image"].isin(positive_ids)].copy()
    parts = [positives]

    for split_name, split_positives in positives.groupby("split", dropna=False):
        split_negatives = negatives[negatives["split"] == split_name].copy()
        n_negatives = min(
            len(split_negatives),
            int(round(len(split_positives) * VERSION.negative_ratio)),
        )
        if n_negatives:
            parts.append(split_negatives.sample(n=n_negatives, random_state=RANDOM_SEED))

    if len(parts) == 1:
        return positives.sort_values(["split", "image"]).reset_index(drop=True)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["split", "image"])
        .reset_index(drop=True)
    )


def build_candidates_v3e(
    source_meta: pd.DataFrame,
    impact_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = image_table(source_meta)
    objects = source_kurgan_objects(source_meta)

    before_images, before_objects = images, objects
    images = images[split_modality_mask(images)].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        VERSION.name,
        "split_modality_filter_train_li_ae_val_li",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    images = images[images["valid_fraction"] >= VERSION.valid_fraction_min].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        VERSION.name,
        f"valid_fraction_gte_{VERSION.valid_fraction_min}",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    objects = apply_area_filter(objects, VERSION)
    positive_image_ids = set(objects["image"])
    source_kurgan_image_ids = set(source_kurgan_objects(source_meta)["image"])
    clean_negative_ids = set(images[~images["image"].isin(source_kurgan_image_ids)]["image"])
    images = images[images["image"].isin(positive_image_ids | clean_negative_ids)].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        VERSION.name,
        "bbox_area_min_100_no_max",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    append_impact(
        impact_rows,
        VERSION.name,
        "bbox_edge_filter_disabled",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    images = images[images["n_objects"] <= VERSION.max_source_objects].copy()
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        VERSION.name,
        f"n_objects_lte_{VERSION.max_source_objects}",
        before_images,
        before_objects,
        images,
        objects,
    )

    before_images, before_objects = images, objects
    positive_ids = set(objects["image"])
    images = sample_negatives_by_split(images, positive_ids)
    objects = objects[objects["image"].isin(set(images["image"]))].copy()
    append_impact(
        impact_rows,
        VERSION.name,
        f"negative_sampling_ratio_{VERSION.negative_ratio:g}_by_split",
        before_images,
        before_objects,
        images,
        objects,
    )

    return images, objects


def materialize_dataset(
    source_dir: Path,
    source_meta: pd.DataFrame,
    output_root: Path,
    overwrite: bool,
    impact_rows: list[dict[str, Any]],
) -> Path:
    out_dir = output_root / VERSION.output_name
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists. Pass --overwrite to replace it: {out_dir}")
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images, objects = build_candidates_v3e(source_meta, impact_rows)
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
        box_records = []
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


def main() -> None:
    args = parse_args()
    source_meta = read_source_metadata(args.source_dir)
    impact_rows: list[dict[str, Any]] = []
    out_dir = materialize_dataset(
        source_dir=args.source_dir,
        source_meta=source_meta,
        output_root=args.output_root,
        overwrite=args.overwrite,
        impact_rows=impact_rows,
    )

    meta = pd.read_csv(out_dir / "metadata.csv")
    metrics = write_audit(args.report_root, "v3e Train Li+Ae / Val Li Audit", meta)
    pd.DataFrame(impact_rows).to_csv(args.report_root / "filter_impact_v3e.csv", index=False)

    print("Dataset:", out_dir)
    print("Audit:", args.report_root / "audit_summary.md")
    print(pd.Series(metrics).to_string())
    if metrics["positive_images"] < 100:
        print("WARNING: positive images < 100")
    if metrics["region_overlap"] or metrics["source_id_overlap"] or metrics["raster_file_overlap"]:
        print("WARNING: train/val leakage detected")


if __name__ == "__main__":
    main()
