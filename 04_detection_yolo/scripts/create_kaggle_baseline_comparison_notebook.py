#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3b_vs_v3d_baseline_comparison.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


cells = [
    md(
        """
        # YOLO Baseline Comparison: v3b Li-only vs v3d Li+Ae

        Goal: compare two fixed dataset candidates without changing labels, split, or filtering rules.

        Experiments:

        - `dataset_yolo_bbox_v3b_li_binary_medium`: cleaner Li-only medium dataset.
        - `dataset_yolo_bbox_v3d_li_ae_binary_medium`: larger Li+Ae medium dataset.

        The training config is identical for both runs except `dataset.yaml`.
        The notebook saves YOLO artifacts, metric comparison, FP/FN tables, object-size summaries, and a zip archive in `/kaggle/working`.
        """
    ),
    md("## 1. Configuration"),
    code(
        """
        from pathlib import Path

        REPO_URL = "https://github.com/MataNerdy/Geodata_Archaeology_CV.git"
        REPO_BRANCH = "main"
        REPO_DIR = Path("/kaggle/working/Geodata_Archaeology_CV")
        PROJECT_DIR = REPO_DIR / "04_detection_yolo"

        KAGGLE_INPUT_ROOT = Path("/kaggle/input")
        WORK_DATA_ROOT = Path("/kaggle/working/datasets")
        OUTPUT_ROOT = Path("/kaggle/working/yolo_baseline_comparison")
        RUN_PROJECT = OUTPUT_ROOT / "runs"
        ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

        RANDOM_SEED = 42
        MODEL_NAME = "yolov8n.pt"
        IMGSZ = 640
        EPOCHS = 100
        BATCH = -1  # Ultralytics AutoBatch: maximum feasible batch for the accelerator.
        FALLBACK_BATCH = 16
        SINGLE_CLS = True
        CLOSE_MOSAIC = 10
        PATIENCE = 25
        WORKERS = 2
        CACHE = "disk"

        CONF_FOR_ERROR_ANALYSIS = 0.25
        IOU_MATCH_THRESHOLD = 0.50

        DATASET_SPECS = {
            "v3b_li_medium": {
                "folder": "dataset_yolo_bbox_v3b_li_binary_medium",
                "builder_name": "v3b_medium",
                "run_name": "v3b_li_medium_yolov8n_img640",
            },
            "v3d_li_ae_medium": {
                "folder": "dataset_yolo_bbox_v3d_li_ae_binary_medium",
                "builder_name": "v3d_li_ae_medium",
                "run_name": "v3d_li_ae_medium_yolov8n_img640",
            },
        }

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        RUN_PROJECT.mkdir(parents=True, exist_ok=True)
        WORK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        """
    ),
    md("## 2. Install Dependencies and Clone Repository"),
    code(
        """
        import os
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pandas", "pyyaml", "pillow", "matplotlib"],
            check=True,
        )

        if REPO_DIR.exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_BRANCH], check=False)
            subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_BRANCH], check=False)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

        os.chdir(PROJECT_DIR)
        print("Project dir:", PROJECT_DIR)
        """
    ),
    md("## 3. Locate Source or Prebuilt Datasets"),
    code(
        """
        import shutil
        import zipfile
        import pandas as pd
        import yaml

        def find_dataset_dir(folder_name: str) -> Path | None:
            for meta in KAGGLE_INPUT_ROOT.rglob("metadata.csv"):
                parent = meta.parent
                if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                    return parent
            return None

        def find_source_dataset() -> Path:
            found = find_dataset_dir("dataset_yolo_bbox")
            if found is not None:
                return found
            zip_candidates = sorted(KAGGLE_INPUT_ROOT.rglob("dataset_yolo_bbox.zip"))
            if zip_candidates:
                unzip_root = WORK_DATA_ROOT / "_source_unzipped"
                if unzip_root.exists():
                    shutil.rmtree(unzip_root)
                unzip_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_candidates[0], "r") as zf:
                    zf.extractall(unzip_root)
                found = next(unzip_root.rglob("metadata.csv")).parent
                if found.name == "dataset_yolo_bbox":
                    return found
            raise FileNotFoundError("Attach Kaggle input containing dataset_yolo_bbox/ or dataset_yolo_bbox.zip")

        SOURCE_DATASET_DIR = find_source_dataset()
        print("Source dataset:", SOURCE_DATASET_DIR)
        print("Source metadata:", SOURCE_DATASET_DIR / "metadata.csv")
        """
    ),
    md("## 4. Materialize v3b and v3d Candidate Datasets"),
    code(
        """
        import importlib.util
        import random

        spec = importlib.util.spec_from_file_location("ablation", PROJECT_DIR / "scripts" / "build_dataset_ablation.py")
        ablation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ablation)

        source_meta = ablation.read_source_metadata(SOURCE_DATASET_DIR)
        version_by_name = {v.name: v for v in ablation.VERSIONS}

        DATASETS = {}
        for display_name, spec_info in DATASET_SPECS.items():
            prebuilt = find_dataset_dir(spec_info["folder"])
            work_dir = WORK_DATA_ROOT / spec_info["folder"]

            if work_dir.exists():
                shutil.rmtree(work_dir)

            if prebuilt is not None:
                print(f"{display_name}: copying prebuilt dataset from {prebuilt}")
                shutil.copytree(prebuilt, work_dir)
            else:
                print(f"{display_name}: building fixed local candidate from source metadata")
                impact_rows = []
                version = version_by_name[spec_info["builder_name"]]
                ablation.build_dataset(
                    source_dir=SOURCE_DATASET_DIR,
                    source_meta=source_meta,
                    version=version,
                    output_root=WORK_DATA_ROOT,
                    overwrite=True,
                    impact_rows=impact_rows,
                )

            dataset_yaml = work_dir / "dataset.yaml"
            dataset_yaml.write_text(
                f"path: {work_dir.resolve()}\\n\\ntrain: images/train\\nval: images/val\\n\\nnames:\\n  0: kurgan\\n",
                encoding="utf-8",
            )
            DATASETS[display_name] = {
                **spec_info,
                "path": work_dir,
                "yaml": dataset_yaml,
                "metadata": work_dir / "metadata.csv",
            }

            df = pd.read_csv(work_dir / "metadata.csv")
            imgs = df.drop_duplicates("image")
            print(display_name, "images:", len(imgs), "positive:", int(imgs["is_positive"].sum()), "bbox:", int(df["class_name"].notna().sum()))
        """
    ),
    md("## 5. Train Both YOLO Models with Identical Config"),
    code(
        """
        from ultralytics import YOLO
        import torch

        def train_one(dataset_key: str, dataset_info: dict) -> Path:
            model = YOLO(MODEL_NAME)
            train_kwargs = dict(
                data=str(dataset_info["yaml"]),
                imgsz=IMGSZ,
                epochs=EPOCHS,
                batch=BATCH,
                seed=RANDOM_SEED,
                deterministic=True,
                single_cls=SINGLE_CLS,
                close_mosaic=CLOSE_MOSAIC,
                patience=PATIENCE,
                workers=WORKERS,
                cache=CACHE,
                project=str(RUN_PROJECT),
                name=dataset_info["run_name"],
                exist_ok=True,
                plots=True,
            )
            try:
                result = model.train(**train_kwargs)
            except Exception as exc:
                if BATCH != -1:
                    raise
                print("AutoBatch failed, retrying with fallback batch:", FALLBACK_BATCH)
                train_kwargs["batch"] = FALLBACK_BATCH
                result = model.train(**train_kwargs)
            return Path(result.save_dir)

        RUN_DIRS = {}
        for key, info in DATASETS.items():
            print("=" * 80)
            print("Training:", key)
            RUN_DIRS[key] = train_one(key, info)
            print("Run dir:", RUN_DIRS[key])
        """
    ),
    md("## 6. Collect Best Metrics"),
    code(
        """
        def summarize_run(dataset_key: str) -> dict:
            info = DATASETS[dataset_key]
            meta = pd.read_csv(info["metadata"])
            images = meta.drop_duplicates("image")
            results_csv = RUN_DIRS[dataset_key] / "results.csv"
            results = pd.read_csv(results_csv)
            results.columns = [c.strip() for c in results.columns]
            best_idx = results["metrics/mAP50(B)"].idxmax()
            best = results.loc[best_idx]
            return {
                "Dataset": dataset_key,
                "Images": int(len(images)),
                "Positive": int(images["is_positive"].sum()),
                "BBox": int(meta["class_name"].notna().sum()),
                "Precision": float(best["metrics/precision(B)"]),
                "Recall": float(best["metrics/recall(B)"]),
                "mAP50": float(best["metrics/mAP50(B)"]),
                "mAP50-95": float(best["metrics/mAP50-95(B)"]),
                "best_epoch": int(best["epoch"]),
            }

        comparison = pd.DataFrame([summarize_run(key) for key in DATASETS])
        comparison_path = ANALYSIS_DIR / "metrics_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        display(comparison)
        print("Saved:", comparison_path)
        """
    ),
    md("## 7. Error Analysis Helpers"),
    code(
        """
        import math
        from PIL import Image, ImageDraw

        def xywhn_to_xyxy(row, image_size=1024):
            xc, yc, w, h = row["yolo_xc"], row["yolo_yc"], row["yolo_w"], row["yolo_h"]
            return [
                (xc - w / 2) * image_size,
                (yc - h / 2) * image_size,
                (xc + w / 2) * image_size,
                (yc + h / 2) * image_size,
            ]

        def box_iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def load_ground_truth(meta: pd.DataFrame):
            gt = meta[(meta["split"] == "val") & (meta["class_name"].notna())].copy()
            gt["gt_box"] = gt.apply(xywhn_to_xyxy, axis=1)
            gt["gt_width"] = gt["bbox_x2_px"] - gt["bbox_x1_px"]
            gt["gt_height"] = gt["bbox_y2_px"] - gt["bbox_y1_px"]
            return gt

        def predict_val(dataset_key: str):
            info = DATASETS[dataset_key]
            run_dir = RUN_DIRS[dataset_key]
            weights = run_dir / "weights" / "best.pt"
            model = YOLO(str(weights))
            meta = pd.read_csv(info["metadata"])
            val_images = meta.drop_duplicates("image")
            val_images = val_images[val_images["split"] == "val"].copy()
            image_paths = [str(Path(p)) for p in val_images["image"]]
            preds = model.predict(
                source=image_paths,
                imgsz=IMGSZ,
                conf=CONF_FOR_ERROR_ANALYSIS,
                iou=0.7,
                save=True,
                project=str(RUN_PROJECT),
                name=f"{info['run_name']}_pred_conf_{str(CONF_FOR_ERROR_ANALYSIS).replace('.', '_')}",
                exist_ok=True,
                verbose=False,
            )
            return meta, val_images, preds

        def match_predictions(dataset_key: str):
            meta, val_images, preds = predict_val(dataset_key)
            gt = load_ground_truth(meta)
            gt_by_image = {image: group.reset_index(drop=True) for image, group in gt.groupby("image")}
            image_lookup = {Path(row.image).name: row for row in val_images.itertuples(index=False)}
            matched_rows, fn_rows, fp_rows = [], [], []

            for result in preds:
                image_name = Path(result.path).name
                image_row = image_lookup[image_name]
                source_image = image_row.image
                gt_group = gt_by_image.get(source_image, pd.DataFrame()).copy()
                gt_used = set()
                pred_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for idx, box in enumerate(xyxy):
                        pred_boxes.append({"pred_idx": idx, "box": box.tolist(), "conf": float(confs[idx])})

                candidates = []
                for p in pred_boxes:
                    for gi, grow in gt_group.iterrows():
                        candidates.append((box_iou(p["box"], grow["gt_box"]), p["pred_idx"], gi, p))
                candidates.sort(reverse=True, key=lambda x: x[0])
                pred_used = set()
                for iou, pred_idx, gt_idx, pred in candidates:
                    if iou < IOU_MATCH_THRESHOLD or pred_idx in pred_used or gt_idx in gt_used:
                        continue
                    grow = gt_group.loc[gt_idx]
                    pred_used.add(pred_idx)
                    gt_used.add(gt_idx)
                    matched_rows.append({
                        "dataset": dataset_key,
                        "image": source_image,
                        "modality": grow["modality"],
                        "source_class_name": grow.get("source_class_name", grow.get("class_name")),
                        "bbox_area_px": grow["bbox_area_px"],
                        "bbox_width_px": grow["gt_width"],
                        "bbox_height_px": grow["gt_height"],
                        "gt_x1": grow["gt_box"][0],
                        "gt_y1": grow["gt_box"][1],
                        "gt_x2": grow["gt_box"][2],
                        "gt_y2": grow["gt_box"][3],
                        "confidence": pred["conf"],
                        "iou": iou,
                    })

                for gi, grow in gt_group.iterrows():
                    if gi not in gt_used:
                        fn_rows.append({
                            "dataset": dataset_key,
                            "image": source_image,
                            "modality": grow["modality"],
                        "source_class_name": grow.get("source_class_name", grow.get("class_name")),
                        "bbox_area_px": grow["bbox_area_px"],
                        "bbox_width_px": grow["gt_width"],
                        "bbox_height_px": grow["gt_height"],
                        "gt_x1": grow["gt_box"][0],
                        "gt_y1": grow["gt_box"][1],
                        "gt_x2": grow["gt_box"][2],
                        "gt_y2": grow["gt_box"][3],
                    })

                for pred in pred_boxes:
                    if pred["pred_idx"] not in pred_used:
                        fp_rows.append({
                            "dataset": dataset_key,
                            "image": source_image,
                            "modality": image_row.modality,
                            "confidence": pred["conf"],
                            "x1": pred["box"][0],
                            "y1": pred["box"][1],
                            "x2": pred["box"][2],
                            "y2": pred["box"][3],
                            "pred_width_px": pred["box"][2] - pred["box"][0],
                            "pred_height_px": pred["box"][3] - pred["box"][1],
                        })

            return pd.DataFrame(matched_rows), pd.DataFrame(fn_rows), pd.DataFrame(fp_rows)
        """
    ),
    md("## 8. Run FP/FN Analysis"),
    code(
        """
        ERROR_TABLES = {}
        for key in DATASETS:
            print("=" * 80)
            print("Error analysis:", key)
            found, fn, fp = match_predictions(key)
            out_dir = ANALYSIS_DIR / key
            out_dir.mkdir(parents=True, exist_ok=True)
            found.to_csv(out_dir / "found_objects.csv", index=False)
            fn.to_csv(out_dir / "false_negatives.csv", index=False)
            fp.to_csv(out_dir / "false_positives.csv", index=False)
            ERROR_TABLES[key] = {"found": found, "fn": fn, "fp": fp}
            print("found:", len(found), "FN:", len(fn), "FP:", len(fp))
        """
    ),
    md("## 9. FP/FN Visual Contact Sheets"),
    code(
        """
        def draw_error_thumb(row, kind: str, thumb_size: int = 256):
            img = Image.open(row["image"]).convert("RGB")
            src_w, src_h = img.size
            if kind == "fp":
                box = [row["x1"], row["y1"], row["x2"], row["y2"]]
                color = "red"
            else:
                box = [row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]]
                color = "lime"

            img.thumbnail((thumb_size, thumb_size))
            canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
            ox = (thumb_size - img.size[0]) // 2
            oy = (thumb_size - img.size[1]) // 2
            canvas.paste(img, (ox, oy))
            sx = img.size[0] / src_w
            sy = img.size[1] / src_h
            x1, y1, x2, y2 = box
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x1 * sx + ox, y1 * sy + oy, x2 * sx + ox, y2 * sy + oy], outline=color, width=3)
            return canvas

        def save_error_sheet(df: pd.DataFrame, out_path: Path, kind: str, max_items: int = 25):
            sheet = Image.new("RGB", (5 * 256, 5 * 256), "white")
            if df.empty:
                sheet.save(out_path, quality=92)
                return
            sample = df.head(max_items).reset_index(drop=True)
            for i, row in sample.iterrows():
                thumb = draw_error_thumb(row, kind=kind)
                x = (i % 5) * 256
                y = (i // 5) * 256
                sheet.paste(thumb, (x, y))
            sheet.save(out_path, quality=92)

        for key, tables in ERROR_TABLES.items():
            out_dir = ANALYSIS_DIR / key
            save_error_sheet(tables["fp"], out_dir / "false_positive_contact_sheet.jpg", kind="fp")
            save_error_sheet(tables["fn"], out_dir / "false_negative_contact_sheet.jpg", kind="fn")
            print(key, "visual sheets saved to", out_dir)
        """
    ),
    md("## 10. Object Size Analysis"),
    code(
        """
        size_rows = []
        for key, tables in ERROR_TABLES.items():
            for group_name, df in [("found", tables["found"]), ("missed", tables["fn"])]:
                if df.empty:
                    continue
                for col in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
                    desc = df[col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                    size_rows.append({
                        "dataset": key,
                        "group": group_name,
                        "metric": col,
                        "count": desc["count"],
                        "mean": desc["mean"],
                        "p10": desc["10%"],
                        "p25": desc["25%"],
                        "median": desc["50%"],
                        "p75": desc["75%"],
                        "p90": desc["90%"],
                        "max": desc["max"],
                    })

        size_summary = pd.DataFrame(size_rows)
        size_summary.to_csv(ANALYSIS_DIR / "object_size_found_vs_missed.csv", index=False)
        display(size_summary)

        class_rows = []
        for key, tables in ERROR_TABLES.items():
            for group_name, df in [("found", tables["found"]), ("missed", tables["fn"])]:
                if not df.empty and "source_class_name" in df.columns:
                    counts = df.groupby(["source_class_name", "modality"]).size().reset_index(name="count")
                    counts["dataset"] = key
                    counts["group"] = group_name
                    class_rows.append(counts)
        class_summary = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
        class_summary.to_csv(ANALYSIS_DIR / "found_vs_missed_by_class_modality.csv", index=False)
        display(class_summary)
        """
    ),
    md("## 11. Markdown Report"),
    code(
        """
        def md_table(df):
            if df.empty:
                return "_No rows._"
            cols = [str(c) for c in df.columns]
            lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
            return "\\n".join(lines)

        report_lines = [
            "# YOLO Baseline Comparison Report",
            "",
            "## Metric Comparison",
            "",
            md_table(comparison),
            "",
            "## Error Analysis Summary",
            "",
        ]

        for key, tables in ERROR_TABLES.items():
            found, fn, fp = tables["found"], tables["fn"], tables["fp"]
            report_lines.extend([
                f"### {key}",
                "",
                f"- Found GT objects: `{len(found)}`",
                f"- False negatives: `{len(fn)}`",
                f"- False positives: `{len(fp)}`",
                "",
            ])
            if not fn.empty:
                report_lines.extend([
                    "False negatives by source class and modality:",
                    "",
                    md_table(fn.groupby(["source_class_name", "modality"]).size().reset_index(name="count")),
                    "",
                ])
            if not fp.empty:
                report_lines.extend([
                    "False positives by modality:",
                    "",
                    md_table(fp.groupby("modality").size().reset_index(name="count")),
                    "",
                ])

        report_lines.extend([
            "## Object Size: Found vs Missed",
            "",
            md_table(size_summary),
            "",
            "## Visual Error Sheets",
            "",
            "Each dataset analysis folder contains:",
            "",
            "- `false_positive_contact_sheet.jpg`: model predictions not matched to a GT box.",
            "- `false_negative_contact_sheet.jpg`: GT boxes missed by the model.",
            "",
            "## Questions to Answer After Run",
            "",
            "1. Which dataset has better mAP50 and mAP50-95?",
            "2. Does Li+Ae improve recall enough to justify modality noise?",
            "3. Does precision drop on Li+Ae due to Ae false positives?",
            "4. Are missed objects smaller than found objects by median area/width/height?",
            "5. Are damaged kurgans missed more often than whole kurgans?",
            "6. Is the current bottleneck dataset coverage, label noise, object size, or model capacity?",
            "7. Next experiment: v3b/v3d with imgsz=1024, or YOLOv8s, depending on the error profile.",
            "",
        ])

        report_path = ANALYSIS_DIR / "baseline_comparison_report.md"
        report_path.write_text("\\n".join(report_lines), encoding="utf-8")
        print("Saved:", report_path)
        print("\\n".join(report_lines[:80]))
        """
    ),
    md("## 12. Archive Outputs"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path("/kaggle/working") / f"yolo_v3b_vs_v3d_baseline_comparison_{timestamp}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in [OUTPUT_ROOT]:
                for file in root.rglob("*"):
                    if file.is_file():
                        zf.write(file, arcname=file.relative_to(Path("/kaggle/working")))

        print("Archive:", archive_path)
        print("Archive size MB:", round(archive_path.stat().st_size / (1024 * 1024), 2))
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
