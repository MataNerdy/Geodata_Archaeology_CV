#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3f_archaeological_object.ipynb")


def md(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


cells = [
    md(
        """
        # YOLO v3f: Archaeological Object vs Kurgan Bottleneck Test

        Hypothesis: the current detector may be recall-limited because the target class is too narrow.

        This notebook compares:

        - `v3b_li_medium`: Li-only, binary `kurgan`.
        - `v3f_archaeological_object`: Li-only, all archaeological source classes merged into one `archaeological_object`.

        The YOLO training config is identical to the fixed v3b baseline.
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
        OUTPUT_ROOT = Path("/kaggle/working/yolo_v3f_archaeological_object")
        RUN_PROJECT = OUTPUT_ROOT / "runs"
        ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

        RANDOM_SEED = 42
        MODEL_NAME = "yolov8n.pt"
        IMGSZ = 640
        EPOCHS = 100
        BATCH = -1
        FALLBACK_BATCH = 16
        SINGLE_CLS = True
        CLOSE_MOSAIC = 10
        PATIENCE = 25
        WORKERS = 2
        CACHE = "disk"

        CONF_FOR_ERROR_ANALYSIS = 0.25
        IOU_MATCH_THRESHOLD = 0.50

        DATASET_FOLDER = "dataset_yolo_bbox_v3f_archaeological_object"
        RUN_NAME = "v3f_archaeological_object_yolov8n_img640"

        V3B_BASELINE = {
            "Dataset": "v3b_li_medium",
            "Target": "kurgan",
            "Images": 284,
            "Positive": 142,
            "BBox": 579,
            "Precision": 0.70544,
            "Recall": 0.28571,
            "mAP50": 0.33904,
            "mAP50-95": 0.11516,
            "best_epoch": 84,
        }

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_PROJECT.mkdir(parents=True, exist_ok=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
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

        os.chdir("/kaggle/working")
        if REPO_DIR.exists():
            print("Removing existing Kaggle working clone:", REPO_DIR)
            shutil.rmtree(REPO_DIR)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

        os.chdir(PROJECT_DIR)
        print("Project dir:", PROJECT_DIR)
        """
    ),
    md("## 3. Locate Source or Build v3f Dataset"),
    code(
        """
        import importlib.util
        import shutil
        import sys
        import zipfile

        import pandas as pd

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

        scripts_dir = PROJECT_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        prebuilt = find_dataset_dir(DATASET_FOLDER)
        work_dir = WORK_DATA_ROOT / DATASET_FOLDER
        if work_dir.exists():
            shutil.rmtree(work_dir)

        if prebuilt is not None:
            print("Copying prebuilt v3f dataset:", prebuilt)
            shutil.copytree(prebuilt, work_dir)
        else:
            source_dataset = find_source_dataset()
            print("Building v3f from source dataset:", source_dataset)
            spec = importlib.util.spec_from_file_location("v3f_builder", PROJECT_DIR / "scripts" / "build_v3f_archaeological_object.py")
            if spec is None or spec.loader is None:
                raise ImportError("Could not load build_v3f_archaeological_object.py")
            v3f_builder = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = v3f_builder
            spec.loader.exec_module(v3f_builder)
            source_meta = v3f_builder.read_source_metadata(source_dataset)
            v3f_builder.materialize_dataset(
                source_dir=source_dataset,
                source_meta=source_meta,
                output_root=WORK_DATA_ROOT,
                overwrite=True,
                impact_rows=[],
            )

        dataset_yaml = work_dir / "dataset.yaml"
        dataset_yaml.write_text(
            f"path: {work_dir.resolve()}\\n\\ntrain: images/train\\nval: images/val\\n\\nnames:\\n  0: archaeological_object\\n",
            encoding="utf-8",
        )

        metadata_path = work_dir / "metadata.csv"
        meta = pd.read_csv(metadata_path)
        images = meta.drop_duplicates("image")
        boxes = meta[meta["class_name"].notna()]
        print("Dataset:", work_dir)
        print("Images:", len(images))
        print("Positive:", int(images["is_positive"].sum()))
        print("Negative:", int((~images["is_positive"]).sum()))
        print("BBox:", int(len(boxes)))
        print("Train/val balance:")
        print(images.groupby(["split", "is_positive"]).size())
        print("BBox by source class:")
        print(boxes["source_class_name"].value_counts())
        """
    ),
    md("## 4. Train YOLOv8n with v3b Config"),
    code(
        """
        from ultralytics import YOLO

        model = YOLO(MODEL_NAME)
        train_kwargs = dict(
            data=str(dataset_yaml),
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
            name=RUN_NAME,
            exist_ok=True,
            plots=True,
        )

        try:
            result = model.train(**train_kwargs)
        except Exception:
            if BATCH != -1:
                raise
            print("AutoBatch failed, retrying with fallback batch:", FALLBACK_BATCH)
            train_kwargs["batch"] = FALLBACK_BATCH
            result = model.train(**train_kwargs)

        RUN_DIR = Path(result.save_dir)
        print("Run dir:", RUN_DIR)
        print("Best weights:", RUN_DIR / "weights" / "best.pt")
        """
    ),
    md("## 5. Collect Metrics"),
    code(
        """
        results = pd.read_csv(RUN_DIR / "results.csv")
        results.columns = [c.strip() for c in results.columns]
        best_idx = results["metrics/mAP50(B)"].idxmax()
        best = results.loc[best_idx]

        v3f_metrics = {
            "Dataset": "v3f_archaeological_object",
            "Target": "archaeological_object",
            "Images": int(len(images)),
            "Positive": int(images["is_positive"].sum()),
            "BBox": int(boxes["class_name"].notna().sum()),
            "Precision": float(best["metrics/precision(B)"]),
            "Recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50-95": float(best["metrics/mAP50-95(B)"]),
            "best_epoch": int(best["epoch"]),
        }

        comparison = pd.DataFrame([V3B_BASELINE, v3f_metrics])
        for metric in ["Precision", "Recall", "mAP50", "mAP50-95"]:
            comparison[f"delta_{metric}"] = comparison[metric] - comparison.loc[0, metric]

        comparison_path = ANALYSIS_DIR / "v3b_vs_v3f_metrics.csv"
        comparison.to_csv(comparison_path, index=False)
        display(comparison)
        print("Saved:", comparison_path)
        """
    ),
    md("## 6. Error Analysis"),
    code(
        """
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
            gt["bbox_width_px"] = gt["bbox_x2_px"] - gt["bbox_x1_px"]
            gt["bbox_height_px"] = gt["bbox_y2_px"] - gt["bbox_y1_px"]
            return gt

        best_model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
        val_images = images[images["split"] == "val"].copy()
        val_rows = list(val_images.itertuples(index=False))
        preds = best_model.predict(
            source=[str(Path(p)) for p in val_images["image"]],
            imgsz=IMGSZ,
            conf=CONF_FOR_ERROR_ANALYSIS,
            iou=0.7,
            save=True,
            project=str(RUN_PROJECT),
            name=f"{RUN_NAME}_pred_conf_{str(CONF_FOR_ERROR_ANALYSIS).replace('.', '_')}",
            exist_ok=True,
            verbose=False,
        )

        gt = load_ground_truth(meta)
        gt_by_image = {image: group.reset_index(drop=True) for image, group in gt.groupby("image")}
        matched_rows, fn_rows, fp_rows = [], [], []

        for result_idx, result in enumerate(preds):
            image_row = val_rows[result_idx]
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
                    "image": source_image,
                    "modality": grow["modality"],
                    "source_class_name": grow["source_class_name"],
                    "bbox_area_px": grow["bbox_area_px"],
                    "bbox_width_px": grow["bbox_width_px"],
                    "bbox_height_px": grow["bbox_height_px"],
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
                        "image": source_image,
                        "modality": grow["modality"],
                        "source_class_name": grow["source_class_name"],
                        "bbox_area_px": grow["bbox_area_px"],
                        "bbox_width_px": grow["bbox_width_px"],
                        "bbox_height_px": grow["bbox_height_px"],
                        "gt_x1": grow["gt_box"][0],
                        "gt_y1": grow["gt_box"][1],
                        "gt_x2": grow["gt_box"][2],
                        "gt_y2": grow["gt_box"][3],
                    })

            for pred in pred_boxes:
                if pred["pred_idx"] not in pred_used:
                    fp_rows.append({
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

        found = pd.DataFrame(matched_rows)
        fn = pd.DataFrame(fn_rows)
        fp = pd.DataFrame(fp_rows)

        found.to_csv(ANALYSIS_DIR / "found_objects.csv", index=False)
        fn.to_csv(ANALYSIS_DIR / "false_negatives.csv", index=False)
        fp.to_csv(ANALYSIS_DIR / "false_positives.csv", index=False)
        print("found:", len(found), "FN:", len(fn), "FP:", len(fp))
        if not fn.empty:
            print("FN by source class:")
            print(fn.groupby("source_class_name").size())
        if not fp.empty:
            print("FP by modality:")
            print(fp.groupby("modality").size())
        """
    ),
    md("## 7. Contact Sheets and Object Size Summary"),
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
            if not df.empty:
                sample = df.head(max_items).reset_index(drop=True)
                for i, row in sample.iterrows():
                    thumb = draw_error_thumb(row, kind=kind)
                    sheet.paste(thumb, ((i % 5) * 256, (i // 5) * 256))
            sheet.save(out_path, quality=92)

        save_error_sheet(fn, ANALYSIS_DIR / "false_negative_contact_sheet.jpg", kind="fn")
        save_error_sheet(fp, ANALYSIS_DIR / "false_positive_contact_sheet.jpg", kind="fp")

        size_rows = []
        for group_name, df in [("found", found), ("missed", fn)]:
            if df.empty:
                continue
            for col in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
                desc = df[col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                size_rows.append({
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
        """
    ),
    md("## 8. Markdown Report"),
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
            "# v3f Archaeological Object Report",
            "",
            "## Metric Comparison",
            "",
            md_table(comparison),
            "",
            "## Dataset Audit",
            "",
            f"- Images: `{len(images)}`",
            f"- Positive images: `{int(images['is_positive'].sum())}`",
            f"- Negative images: `{int((~images['is_positive']).sum())}`",
            f"- BBox total: `{len(boxes)}`",
            "",
            "### BBox by source class",
            "",
            md_table(boxes["source_class_name"].value_counts().rename("bbox").reset_index().rename(columns={"source_class_name": "source_class"})),
            "",
            "## Error Analysis",
            "",
            f"- Found GT objects: `{len(found)}`",
            f"- False negatives: `{len(fn)}`",
            f"- False positives: `{len(fp)}`",
            "",
            "### False negatives by source class",
            "",
            md_table(fn.groupby("source_class_name").size().reset_index(name="count") if not fn.empty else pd.DataFrame()),
            "",
            "### Object size: found vs missed",
            "",
            md_table(size_summary),
            "",
            "## Final Question",
            "",
            "Does moving from `kurgan` to `archaeological_object` improve recall and baseline quality?",
            "",
            "Answer this after inspecting the metric deltas and contact sheets.",
            "",
        ]

        report_path = ANALYSIS_DIR / "v3f_archaeological_object_report.md"
        report_path.write_text("\\n".join(report_lines), encoding="utf-8")
        print("Saved:", report_path)
        print("\\n".join(report_lines))
        """
    ),
    md("## 9. Archive Outputs"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path("/kaggle/working") / f"yolo_v3f_archaeological_object_{timestamp}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in OUTPUT_ROOT.rglob("*"):
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
