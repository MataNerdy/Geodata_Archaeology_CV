#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3h_no_saratov.ipynb")


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
        # YOLO v3h No-Saratov Validation Check

        One controlled experiment on the manually cleaned Li-only kurgan dataset with Saratov moved from validation to train.

        ```bash
        yolo detect train \\
          model=yolov8n.pt \\
          data=dataset_yolo_bbox_v3h_li_manual_curated_val_no_saratov/dataset.yaml \\
          imgsz=640 \\
          epochs=100 \\
          batch=16 \\
          seed=42 \\
          single_cls=True \\
          close_mosaic=10 \\
          project=runs/kurgan_detection \\
          name=v3h_no_saratov_yolov8n_640
        ```

        This notebook expects the prebuilt no-Saratov v3h dataset as a Kaggle input.
        """
    ),
    md("## 1. Configuration"),
    code(
        """
        from pathlib import Path

        KAGGLE_INPUT_ROOT = Path("/kaggle/input")
        PREBUILT_DATASET_DIR = Path(
            "/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3h_li_manual_curated_val_no_saratov"
        )
        WORK_ROOT = Path("/kaggle/working")
        WORK_DATASETS_DIR = WORK_ROOT / "datasets"

        DATASET_FOLDER = "dataset_yolo_bbox_v3h_li_manual_curated_val_no_saratov"
        DATASET_WORK_DIR = WORK_DATASETS_DIR / DATASET_FOLDER
        DATASET_YAML = DATASET_WORK_DIR / "dataset.yaml"

        MODEL_NAME = "yolov8n.pt"
        IMGSZ = 640
        EPOCHS = 100
        BATCH = 16
        RANDOM_SEED = 42
        SINGLE_CLS = True
        CLOSE_MOSAIC = 10

        RUN_PROJECT = WORK_ROOT / "runs" / "kurgan_detection"
        RUN_NAME = "v3h_no_saratov_yolov8n_640"
        RUN_DIR = RUN_PROJECT / RUN_NAME
        ANALYSIS_DIR = WORK_ROOT / "analysis" / RUN_NAME

        WORK_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        RUN_PROJECT.mkdir(parents=True, exist_ok=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        """
    ),
    md("## 2. Install Dependencies"),
    code(
        """
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pandas", "pyyaml", "pillow", "matplotlib"],
            check=True,
        )
        """
    ),
    md("## 3. Locate and Copy Dataset"),
    code(
        """
        import shutil
        import zipfile

        import pandas as pd
        import yaml

        def find_dataset_dir(folder_name: str) -> Path | None:
            if PREBUILT_DATASET_DIR.exists():
                return PREBUILT_DATASET_DIR
            for dataset_yaml in KAGGLE_INPUT_ROOT.rglob("dataset.yaml"):
                parent = dataset_yaml.parent
                if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                    return parent
            for metadata in KAGGLE_INPUT_ROOT.rglob("metadata.csv"):
                parent = metadata.parent
                if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                    return parent
            return None

        def find_dataset_zip(folder_name: str) -> Path | None:
            candidates = sorted(KAGGLE_INPUT_ROOT.rglob(f"{folder_name}.zip"))
            return candidates[0] if candidates else None

        source_dir = find_dataset_dir(DATASET_FOLDER)

        if DATASET_WORK_DIR.exists():
            shutil.rmtree(DATASET_WORK_DIR)

        if source_dir is not None:
            print("Copying prebuilt dataset from:", source_dir)
            shutil.copytree(source_dir, DATASET_WORK_DIR)
        else:
            zip_path = find_dataset_zip(DATASET_FOLDER)
            if zip_path is None:
                raise FileNotFoundError(
                    f"Attach a Kaggle input containing {DATASET_FOLDER}/ or {DATASET_FOLDER}.zip"
                )
            print("Extracting dataset zip:", zip_path)
            unzip_root = WORK_DATASETS_DIR / "_unzipped_v3h_no_saratov"
            if unzip_root.exists():
                shutil.rmtree(unzip_root)
            unzip_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(unzip_root)
            extracted_candidates = [
                p.parent for p in unzip_root.rglob("dataset.yaml") if p.parent.name == DATASET_FOLDER
            ]
            extracted = extracted_candidates[0] if extracted_candidates else next(unzip_root.rglob("dataset.yaml")).parent
            shutil.copytree(extracted, DATASET_WORK_DIR)

        DATASET_YAML.write_text(
            yaml.safe_dump(
                {
                    "path": str(DATASET_WORK_DIR.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "kurgan"},
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        metadata = pd.read_csv(DATASET_WORK_DIR / "metadata.csv")
        images = metadata.drop_duplicates("image").copy()
        boxes = metadata[metadata["class_name"].notna()].copy()

        print("Dataset work dir:", DATASET_WORK_DIR)
        print("Dataset yaml:", DATASET_YAML)
        print("Images:", len(images))
        print("Positive images:", int(images["is_positive"].astype(bool).sum()))
        print("Negative images:", int((~images["is_positive"].astype(bool)).sum()))
        print("BBox:", len(boxes))
        print("\\nSplit balance:")
        print(images.groupby(["split", "is_positive"]).size())
        print("\\nBBox by source class and split:")
        source_col = "source_class_name" if "source_class_name" in boxes.columns else "class_name"
        print(boxes.groupby(["split", source_col]).size())
        """
    ),
    md("## 4. Train YOLOv8n"),
    code(
        """
        from pathlib import Path

        from ultralytics import YOLO

        model = YOLO(MODEL_NAME)
        result = model.train(
            data=str(DATASET_YAML),
            imgsz=IMGSZ,
            epochs=EPOCHS,
            batch=BATCH,
            seed=RANDOM_SEED,
            deterministic=True,
            single_cls=SINGLE_CLS,
            close_mosaic=CLOSE_MOSAIC,
            project=str(RUN_PROJECT),
            name=RUN_NAME,
            exist_ok=True,
            plots=True,
        )

        RUN_DIR = Path(result.save_dir)
        print("Run dir:", RUN_DIR)
        print("Best weights:", RUN_DIR / "weights" / "best.pt")
        """
    ),
    md("## 5. Metrics Summary"),
    code(
        """
        import pandas as pd

        results_path = RUN_DIR / "results.csv"
        results = pd.read_csv(results_path)
        results.columns = [c.strip() for c in results.columns]

        best_idx = results["metrics/mAP50(B)"].idxmax()
        best = results.loc[best_idx]

        summary = pd.DataFrame(
            [
                {
                    "Dataset": "v3h_no_saratov",
                    "Model": MODEL_NAME,
                    "imgsz": IMGSZ,
                    "epochs": EPOCHS,
                    "batch": BATCH,
                    "best_epoch": int(best["epoch"]),
                    "Precision": float(best["metrics/precision(B)"]),
                    "Recall": float(best["metrics/recall(B)"]),
                    "mAP50": float(best["metrics/mAP50(B)"]),
                    "mAP50-95": float(best["metrics/mAP50-95(B)"]),
                }
            ]
        )

        summary_path = ANALYSIS_DIR / "metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        display(summary)
        print("Saved:", summary_path)
        """
    ),
    md("## 6. Validate Best Weights"),
    code(
        """
        best_model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
        val_result = best_model.val(
            data=str(DATASET_YAML),
            imgsz=IMGSZ,
            batch=BATCH,
            single_cls=SINGLE_CLS,
            project=str(RUN_PROJECT),
            name=f"{RUN_NAME}_val_best",
            exist_ok=True,
            plots=True,
        )

        print("Validation save dir:", val_result.save_dir)
        """
    ),
    md("## 7. Show Key Artifacts"),
    code(
        """
        from IPython.display import Image, display

        artifact_candidates = [
            RUN_DIR / "results.png",
            RUN_DIR / "confusion_matrix.png",
            RUN_DIR / "PR_curve.png",
            RUN_DIR / "F1_curve.png",
            RUN_DIR / "val_batch0_labels.jpg",
            RUN_DIR / "val_batch0_pred.jpg",
        ]

        for artifact in artifact_candidates:
            if artifact.exists():
                print(artifact)
                display(Image(filename=str(artifact)))
            else:
                print("Missing:", artifact)
        """
    ),
    md("## 8. Threshold Sweep and Regional Validation Audit"),
    code(
        """
        import numpy as np
        import pandas as pd
        from PIL import Image

        THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
        MATCH_IOU = 0.50
        COVERAGE_IOU = 0.30
        ANALYSIS_CONF = min(THRESHOLDS)
        WORKING_CONF = 0.25

        def resolve_image_path(row: pd.Series) -> Path:
            split = str(row["split"])
            image_name = str(row.get("image_name") or Path(str(row["image"])).name)
            candidates = [
                DATASET_WORK_DIR / "images" / split / image_name,
                Path(str(row["image"])),
                DATASET_WORK_DIR / "images" / split / Path(str(row["image"])).name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
            return candidates[0].resolve()

        def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if len(a) == 0 or len(b) == 0:
                return np.zeros((len(a), len(b)), dtype=float)
            x1 = np.maximum(a[:, None, 0], b[None, :, 0])
            y1 = np.maximum(a[:, None, 1], b[None, :, 1])
            x2 = np.minimum(a[:, None, 2], b[None, :, 2])
            y2 = np.minimum(a[:, None, 3], b[None, :, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
            area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
            union = area_a[:, None] + area_b[None, :] - inter
            return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        def load_val_gt(metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
            val = metadata[metadata["split"].astype(str).str.lower().eq("val")].copy()
            if "source_class_name" not in val.columns:
                val["source_class_name"] = val["class_name"]
            if "source_id" not in val.columns:
                source_cols = [col for col in ["region", "modality", "raster_file"] if col in val.columns]
                val["source_id"] = val[source_cols].astype(str).agg("|".join, axis=1)
            val["image_path"] = val.apply(resolve_image_path, axis=1)
            val["image_key"] = val["image_path"].map(lambda p: str(Path(p).resolve()))
            images_df = val.drop_duplicates("image_key").copy()
            image_paths = [Path(p).resolve() for p in images_df["image_path"]]

            gt = val[val["class_name"].notna()].copy().reset_index(drop=True)
            gt["gt_id"] = np.arange(len(gt))
            gt["objects_in_tile"] = pd.to_numeric(gt.get("n_objects", np.nan), errors="coerce")
            gt["bbox_width_px"] = pd.to_numeric(gt.get("bbox_width_px", np.nan), errors="coerce")
            gt["bbox_height_px"] = pd.to_numeric(gt.get("bbox_height_px", np.nan), errors="coerce")
            if gt["bbox_width_px"].isna().all():
                gt["bbox_width_px"] = pd.to_numeric(gt["bbox_x2_px"], errors="coerce") - pd.to_numeric(gt["bbox_x1_px"], errors="coerce")
            if gt["bbox_height_px"].isna().all():
                gt["bbox_height_px"] = pd.to_numeric(gt["bbox_y2_px"], errors="coerce") - pd.to_numeric(gt["bbox_y1_px"], errors="coerce")
            gt["bbox_area_px"] = pd.to_numeric(gt["bbox_area_px"], errors="coerce")

            boxes_xyxy = []
            for _, row in gt.iterrows():
                with Image.open(row["image_key"]) as image:
                    width, height = image.size
                if {"yolo_xc", "yolo_yc", "yolo_w", "yolo_h"}.issubset(gt.columns) and pd.notna(row.get("yolo_xc")):
                    xc = float(row["yolo_xc"]) * width
                    yc = float(row["yolo_yc"]) * height
                    bw = float(row["yolo_w"]) * width
                    bh = float(row["yolo_h"]) * height
                    boxes_xyxy.append((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
                else:
                    boxes_xyxy.append((float(row["bbox_x1_px"]), float(row["bbox_y1_px"]), float(row["bbox_x2_px"]), float(row["bbox_y2_px"])))
            gt["bbox_xyxy"] = boxes_xyxy
            return gt, image_paths

        def predict_dataframe(model: YOLO, image_paths: list[Path], conf: float) -> pd.DataFrame:
            results = model.predict(
                source=[str(path) for path in image_paths],
                imgsz=IMGSZ,
                conf=conf,
                iou=0.50,
                verbose=False,
                save=False,
                stream=False,
            )
            rows = []
            for result_idx, result in enumerate(results):
                image_key = str(image_paths[result_idx].resolve())
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for pred_idx, (box, score) in enumerate(zip(xyxy, confs)):
                    rows.append(
                        {
                            "image_key": image_key,
                            "pred_idx": pred_idx,
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": float(score),
                        }
                    )
            return pd.DataFrame(rows, columns=["image_key", "pred_idx", "x1", "y1", "x2", "y2", "confidence"])

        def match_predictions(gt: pd.DataFrame, predictions: pd.DataFrame, match_iou: float, best_predictions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
            audit = gt.copy()
            audit["is_found"] = False
            audit["matched_prediction_iou"] = 0.0
            audit["matched_prediction_confidence"] = np.nan
            audit["best_prediction_iou"] = 0.0
            audit["best_prediction_confidence"] = np.nan
            pred_by_image = {key: group.sort_values("confidence", ascending=False).reset_index(drop=True) for key, group in predictions.groupby("image_key")} if not predictions.empty else {}
            best_source = predictions if best_predictions is None else best_predictions
            best_by_image = {key: group.sort_values("confidence", ascending=False).reset_index(drop=True) for key, group in best_source.groupby("image_key")} if not best_source.empty else {}
            matched_prediction_keys: set[tuple[str, int]] = set()

            for image_key, gt_group in audit.groupby("image_key"):
                gt_indices = list(gt_group.index)
                gt_boxes = np.array(gt_group["bbox_xyxy"].tolist(), dtype=float)
                pred_group = pred_by_image.get(image_key, pd.DataFrame()).copy()
                best_group = best_by_image.get(image_key, pd.DataFrame()).copy()
                pred_boxes = pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not pred_group.empty else np.empty((0, 4))
                best_boxes = best_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not best_group.empty else np.empty((0, 4))
                ious = box_iou(pred_boxes, gt_boxes)
                best_ious = box_iou(best_boxes, gt_boxes)

                if len(best_boxes):
                    for local_gt_idx, global_gt_idx in enumerate(gt_indices):
                        best_pred_idx = int(np.argmax(best_ious[:, local_gt_idx]))
                        audit.loc[global_gt_idx, "best_prediction_iou"] = float(best_ious[best_pred_idx, local_gt_idx])
                        audit.loc[global_gt_idx, "best_prediction_confidence"] = float(best_group.iloc[best_pred_idx]["confidence"])

                candidates = []
                for pred_idx in range(len(pred_boxes)):
                    for local_gt_idx, global_gt_idx in enumerate(gt_indices):
                        iou_value = float(ious[pred_idx, local_gt_idx])
                        if iou_value >= match_iou:
                            candidates.append((iou_value, float(pred_group.iloc[pred_idx]["confidence"]), pred_idx, global_gt_idx))

                matched_pred = set()
                matched_gt = set()
                for iou_value, confidence, pred_idx, global_gt_idx in sorted(candidates, reverse=True):
                    if pred_idx in matched_pred or global_gt_idx in matched_gt:
                        continue
                    matched_pred.add(pred_idx)
                    matched_gt.add(global_gt_idx)
                    matched_prediction_keys.add((image_key, pred_idx))
                    audit.loc[global_gt_idx, "is_found"] = True
                    audit.loc[global_gt_idx, "matched_prediction_iou"] = iou_value
                    audit.loc[global_gt_idx, "matched_prediction_confidence"] = confidence

            fp_count = max(0, len(predictions) - len(matched_prediction_keys))
            return audit, fp_count

        def add_fn_types(audit: pd.DataFrame, conf_threshold: float) -> pd.DataFrame:
            audit = audit.copy()
            best_iou = pd.to_numeric(audit["best_prediction_iou"], errors="coerce").fillna(0.0)
            best_conf = pd.to_numeric(audit["best_prediction_confidence"], errors="coerce").fillna(0.0)
            audit["fn_type"] = "found"
            missed = ~audit["is_found"].astype(bool)
            metric_miss = missed & (best_iou >= MATCH_IOU) & (best_conf < conf_threshold)
            near_miss = missed & (~metric_miss) & (best_iou >= COVERAGE_IOU)
            hard_miss = missed & (~metric_miss) & (~near_miss)
            audit.loc[metric_miss, "fn_type"] = "metric_miss"
            audit.loc[near_miss, "fn_type"] = "near_miss"
            audit.loc[hard_miss, "fn_type"] = "hard_miss"
            return audit

        val_gt, val_image_paths = load_val_gt(metadata)
        print("Val images:", len(val_image_paths))
        print("Val GT objects:", len(val_gt))

        best_model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
        all_predictions = predict_dataframe(best_model, val_image_paths, ANALYSIS_CONF)
        all_predictions.to_csv(ANALYSIS_DIR / "predictions_all_conf.csv", index=False)
        print("Predictions at analysis conf:", len(all_predictions))

        sweep_rows = []
        audits_by_threshold = {}
        for threshold in THRESHOLDS:
            preds = all_predictions[all_predictions["confidence"].ge(threshold)].copy()
            audit_t, fp = match_predictions(val_gt, preds, MATCH_IOU, best_predictions=all_predictions)
            audit_t = add_fn_types(audit_t, threshold)
            audits_by_threshold[threshold] = audit_t
            tp = int(audit_t["is_found"].sum())
            fn = int((~audit_t["is_found"]).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            covered = int(pd.to_numeric(audit_t["best_prediction_iou"], errors="coerce").ge(COVERAGE_IOU).sum())
            sweep_rows.append(
                {
                    "conf": threshold,
                    "TP": tp,
                    "FP": int(fp),
                    "FN": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "covered_gt_iou_0.30": covered,
                    "total_gt": int(len(audit_t)),
                    "coverage_rate_iou_0.30": covered / len(audit_t) if len(audit_t) else 0.0,
                    "predictions": int(len(preds)),
                }
            )

        threshold_sweep = pd.DataFrame(sweep_rows)
        threshold_sweep.to_csv(ANALYSIS_DIR / "threshold_sweep.csv", index=False)
        display(threshold_sweep)
        print("Saved:", ANALYSIS_DIR / "threshold_sweep.csv")
        """
    ),
    md("## 9. Regional Validation Audit at conf=0.25"),
    code(
        """
        working_audit = audits_by_threshold[WORKING_CONF].copy()
        working_audit.to_csv(ANALYSIS_DIR / "gt_object_audit_conf_025.csv", index=False)

        regional_rows = []
        for region, group in working_audit.groupby("region", dropna=False):
            gt_count = int(len(group))
            tp = int(group["is_found"].sum())
            fn_group = group[~group["is_found"]].copy()
            regional_rows.append(
                {
                    "region": region,
                    "GT": gt_count,
                    "TP": tp,
                    "FN": int(len(fn_group)),
                    "recall": tp / gt_count if gt_count else 0.0,
                    "metric_miss": int(fn_group["fn_type"].eq("metric_miss").sum()),
                    "near_miss": int(fn_group["fn_type"].eq("near_miss").sum()),
                    "hard_miss": int(fn_group["fn_type"].eq("hard_miss").sum()),
                    "bbox_area_median": float(pd.to_numeric(group["bbox_area_px"], errors="coerce").median()),
                    "classes": "; ".join(sorted(group["source_class_name"].dropna().astype(str).unique())),
                }
            )

        regional_audit = pd.DataFrame(regional_rows).sort_values(["recall", "GT"], ascending=[True, False])
        regional_audit.to_csv(ANALYSIS_DIR / "regional_validation_audit_conf_025.csv", index=False)

        fn_type_by_region = (
            working_audit[~working_audit["is_found"]]
            .groupby(["region", "fn_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["region", "fn_type"])
        )
        fn_type_by_region.to_csv(ANALYSIS_DIR / "fn_type_by_region_conf_025.csv", index=False)

        display(regional_audit)
        display(fn_type_by_region)
        print("Saved:", ANALYSIS_DIR / "regional_validation_audit_conf_025.csv")
        print("Saved:", ANALYSIS_DIR / "fn_type_by_region_conf_025.csv")
        """
    ),
    md("## 10. Markdown Report"),
    code(
        """
        def md_table(df: pd.DataFrame) -> str:
            cols = [str(c) for c in df.columns]
            lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
            return "\\n".join(lines)

        report = [
            "# v3h No-Saratov Validation YOLOv8n 640 Report",
            "",
            "## Config",
            "",
            f"- dataset: `{DATASET_FOLDER}`",
            f"- model: `{MODEL_NAME}`",
            f"- imgsz: `{IMGSZ}`",
            f"- epochs: `{EPOCHS}`",
            f"- batch: `{BATCH}`",
            f"- seed: `{RANDOM_SEED}`",
            f"- single_cls: `{SINGLE_CLS}`",
            f"- close_mosaic: `{CLOSE_MOSAIC}`",
            f"- project: `{RUN_PROJECT}`",
            f"- name: `{RUN_NAME}`",
            "",
            "## Dataset",
            "",
            f"- images: `{len(images)}`",
            f"- positive images: `{int(images['is_positive'].astype(bool).sum())}`",
            f"- negative images: `{int((~images['is_positive'].astype(bool)).sum())}`",
            f"- bbox: `{len(boxes)}`",
            "",
            "## Best Metrics",
            "",
            md_table(summary),
            "",
            "## Threshold Sweep",
            "",
            md_table(threshold_sweep),
            "",
            "## Regional Validation Audit at conf=0.25",
            "",
            md_table(regional_audit),
            "",
            "## Artifacts",
            "",
            f"- run dir: `{RUN_DIR}`",
            f"- best weights: `{RUN_DIR / 'weights' / 'best.pt'}`",
            f"- results.csv: `{RUN_DIR / 'results.csv'}`",
            f"- results.png: `{RUN_DIR / 'results.png'}`",
            f"- validation dir: `{RUN_PROJECT / (RUN_NAME + '_val_best')}`",
            "",
        ]

        report_path = ANALYSIS_DIR / "v3h_no_saratov_report.md"
        report_path.write_text("\\n".join(report), encoding="utf-8")
        print("Saved:", report_path)
        print("\\n".join(report))
        """
    ),
    md("## 11. Archive Outputs"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = WORK_ROOT / f"yolo_v3h_no_saratov_{timestamp}.zip"
        roots_to_archive = [RUN_PROJECT, ANALYSIS_DIR]

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in roots_to_archive:
                if root.exists():
                    for file in root.rglob("*"):
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(WORK_ROOT))

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
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
