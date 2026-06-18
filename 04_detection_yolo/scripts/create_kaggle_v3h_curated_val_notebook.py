#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3h_curated_val.ipynb")


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
        # YOLO v3h Manual Curated Validation Baseline

        One controlled experiment on the manually cleaned Li-only kurgan dataset with curated validation regions:

        ```bash
        yolo detect train \\
          model=yolov8n.pt \\
          data=dataset_yolo_bbox_v3h_li_manual_curated_val/dataset.yaml \\
          imgsz=640 \\
          epochs=100 \\
          batch=16 \\
          seed=42 \\
          single_cls=True \\
          close_mosaic=10 \\
          project=runs/kurgan_detection \\
          name=v3h_li_manual_curated_val_yolov8n_640
        ```

        This notebook expects the prebuilt v3h dataset as a Kaggle input.
        """
    ),
    md("## 1. Configuration"),
    code(
        """
        from pathlib import Path

        KAGGLE_INPUT_ROOT = Path("/kaggle/input")
        PREBUILT_DATASET_DIR = Path(
            "/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3h_li_manual_curated_val"
        )
        WORK_ROOT = Path("/kaggle/working")
        WORK_DATASETS_DIR = WORK_ROOT / "datasets"

        DATASET_FOLDER = "dataset_yolo_bbox_v3h_li_manual_curated_val"
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
        RUN_NAME = "v3h_li_manual_curated_val_yolov8n_640"
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
            unzip_root = WORK_DATASETS_DIR / "_unzipped_v3h"
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
                    "Dataset": "v3h_li_manual_curated_val",
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
    md("## 8. Markdown Report"),
    code(
        """
        def md_table(df: pd.DataFrame) -> str:
            cols = [str(c) for c in df.columns]
            lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
            return "\\n".join(lines)

        report = [
            "# v3h Manual Curated Validation YOLOv8n 640 Report",
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
            "## Artifacts",
            "",
            f"- run dir: `{RUN_DIR}`",
            f"- best weights: `{RUN_DIR / 'weights' / 'best.pt'}`",
            f"- results.csv: `{RUN_DIR / 'results.csv'}`",
            f"- results.png: `{RUN_DIR / 'results.png'}`",
            f"- validation dir: `{RUN_PROJECT / (RUN_NAME + '_val_best')}`",
            "",
        ]

        report_path = ANALYSIS_DIR / "v3h_curated_val_report.md"
        report_path.write_text("\\n".join(report), encoding="utf-8")
        print("Saved:", report_path)
        print("\\n".join(report))
        """
    ),
    md("## 9. Archive Outputs"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = WORK_ROOT / f"yolo_v3h_curated_val_{timestamp}.zip"
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
