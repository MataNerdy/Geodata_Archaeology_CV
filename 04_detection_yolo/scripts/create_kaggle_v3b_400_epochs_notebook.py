#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3b_400_epochs.ipynb")


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
        # YOLO v3b Li-only Long Run: 400 Epoch Limit

        Controlled experiment:

        - Dataset: `dataset_yolo_bbox_v3b_li_binary_medium`
        - Train: `Li`
        - Val: `Li`
        - Model/config: same as the previous `v3b_li_medium` baseline
        - Only changed parameter: `epochs = 400`

        Note: `patience = 25` is intentionally unchanged. This means `400` is the maximum epoch limit; training may stop earlier if validation quality stops improving.
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
        OUTPUT_ROOT = Path("/kaggle/working/yolo_v3b_400_epochs")
        RUN_PROJECT = OUTPUT_ROOT / "runs"
        ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

        RANDOM_SEED = 42
        MODEL_NAME = "yolov8n.pt"
        IMGSZ = 640
        EPOCHS = 400
        BATCH = -1
        FALLBACK_BATCH = 16
        SINGLE_CLS = True
        CLOSE_MOSAIC = 10
        PATIENCE = 25
        WORKERS = 2
        CACHE = "disk"

        DATASET_FOLDER = "dataset_yolo_bbox_v3b_li_binary_medium"
        RUN_NAME = "v3b_li_medium_yolov8n_img640_epochs400"

        BASELINE_100_EPOCHS = {
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
    md("## 3. Locate Source or Prebuilt Dataset"),
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
            print("Copying prebuilt v3b dataset:", prebuilt)
            shutil.copytree(prebuilt, work_dir)
        else:
            source_dataset = find_source_dataset()
            print("Building v3b from source dataset:", source_dataset)
            spec = importlib.util.spec_from_file_location("ablation", PROJECT_DIR / "scripts" / "build_dataset_ablation.py")
            if spec is None or spec.loader is None:
                raise ImportError("Could not load build_dataset_ablation.py")
            ablation = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = ablation
            spec.loader.exec_module(ablation)
            source_meta = ablation.read_source_metadata(source_dataset)
            version_by_name = {v.name: v for v in ablation.VERSIONS}
            ablation.build_dataset(
                source_dir=source_dataset,
                source_meta=source_meta,
                version=version_by_name["v3b_medium"],
                output_root=WORK_DATA_ROOT,
                overwrite=True,
                impact_rows=[],
            )

        dataset_yaml = work_dir / "dataset.yaml"
        dataset_yaml.write_text(
            f"path: {work_dir.resolve()}\\n\\ntrain: images/train\\nval: images/val\\n\\nnames:\\n  0: kurgan\\n",
            encoding="utf-8",
        )

        metadata_path = work_dir / "metadata.csv"
        meta = pd.read_csv(metadata_path)
        images = meta.drop_duplicates("image")
        print("Dataset:", work_dir)
        print("Images:", len(images))
        print("Positive:", int(images["is_positive"].sum()))
        print("BBox:", int(meta["class_name"].notna().sum()))
        print(images.groupby(["split", "modality"]).size())
        """
    ),
    md("## 4. Train YOLOv8n for 400 Epoch Limit"),
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
    md("## 5. Collect Metrics and Compare with 100-Epoch Baseline"),
    code(
        """
        results = pd.read_csv(RUN_DIR / "results.csv")
        results.columns = [c.strip() for c in results.columns]
        best_idx = results["metrics/mAP50(B)"].idxmax()
        best = results.loc[best_idx]

        long_run = {
            "Experiment": "v3b_400_epoch_limit",
            "Precision": float(best["metrics/precision(B)"]),
            "Recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50-95": float(best["metrics/mAP50-95(B)"]),
            "best_epoch": int(best["epoch"]),
            "epochs_ran": int(results["epoch"].max()) + 1,
        }

        rows = [
            {"Experiment": "v3b_100_epoch_baseline", **BASELINE_100_EPOCHS, "epochs_ran": 100},
            long_run,
        ]
        comparison = pd.DataFrame(rows)
        for metric in ["Precision", "Recall", "mAP50", "mAP50-95"]:
            comparison[f"delta_{metric}"] = comparison[metric] - comparison.loc[0, metric]

        comparison_path = ANALYSIS_DIR / "v3b_400_vs_100_metrics.csv"
        comparison.to_csv(comparison_path, index=False)
        display(comparison)
        print("Saved:", comparison_path)
        """
    ),
    md("## 6. Save Validation Predictions"),
    code(
        """
        best_model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
        val_images = images[images["split"] == "val"].copy()
        pred_dir_name = f"{RUN_NAME}_pred_conf_0_25"
        preds = best_model.predict(
            source=[str(Path(p)) for p in val_images["image"]],
            imgsz=IMGSZ,
            conf=0.25,
            iou=0.7,
            save=True,
            project=str(RUN_PROJECT),
            name=pred_dir_name,
            exist_ok=True,
            verbose=False,
        )
        print("Predictions:", RUN_PROJECT / pred_dir_name)
        print("Predicted images:", len(preds))
        """
    ),
    md("## 7. Markdown Report"),
    code(
        """
        def md_table(df: pd.DataFrame) -> str:
            cols = [str(c) for c in df.columns]
            lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
            return "\\n".join(lines)

        report_lines = [
            "# v3b Li-only 400-Epoch-Limit Report",
            "",
            "## Config",
            "",
            f"- model: `{MODEL_NAME}`",
            f"- imgsz: `{IMGSZ}`",
            f"- epochs: `{EPOCHS}`",
            f"- patience: `{PATIENCE}`",
            f"- batch: `{BATCH}`",
            f"- seed: `{RANDOM_SEED}`",
            f"- close_mosaic: `{CLOSE_MOSAIC}`",
            f"- single_cls: `{SINGLE_CLS}`",
            "",
            "## Metrics",
            "",
            md_table(comparison),
            "",
            "## Artifacts",
            "",
            f"- run dir: `{RUN_DIR}`",
            f"- best weights: `{RUN_DIR / 'weights' / 'best.pt'}`",
            f"- results.csv: `{RUN_DIR / 'results.csv'}`",
            f"- results.png: `{RUN_DIR / 'results.png'}`",
            f"- predictions: `{RUN_PROJECT / pred_dir_name}`",
            "",
        ]

        report_path = ANALYSIS_DIR / "v3b_400_epoch_report.md"
        report_path.write_text("\\n".join(report_lines), encoding="utf-8")
        print("Saved:", report_path)
        print("\\n".join(report_lines))
        """
    ),
    md("## 8. Archive Outputs"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path("/kaggle/working") / f"yolo_v3b_400_epochs_{timestamp}.zip"
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
