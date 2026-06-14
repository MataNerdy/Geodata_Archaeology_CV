#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3b_img1024.ipynb")
SWEEP_SCRIPT_PATH = Path("scripts/sweep_v3b_thresholds.py")


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


def main() -> None:
    sweep_script = SWEEP_SCRIPT_PATH.read_text(encoding="utf-8")
    cells = [
        md(
            """
            # YOLO v3b Li-only Baseline: Image Size 1024

            Controlled experiment:

            - Dataset: `dataset_yolo_bbox_v3b_li_binary_medium`
            - Model: `yolov8n.pt`
            - Changed parameter: `imgsz = 1024`
            - Training config otherwise follows the fixed v3b baseline

            The comparison focuses on `mAP50` and proposal coverage:

            - `coverage_rate` at `conf=0.05`
            - `coverage_rate` at `conf=0.01`
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
            OUTPUT_ROOT = Path("/kaggle/working/yolo_v3b_img1024")
            RUN_PROJECT = OUTPUT_ROOT / "runs"
            ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

            SOURCE_DATASET_DIR = Path("/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox")

            DATASET_FOLDER = "dataset_yolo_bbox_v3b_li_binary_medium"
            RUN_NAME = "v3b_li_medium_yolov8n_img1024"

            RANDOM_SEED = 42
            MODEL_NAME = "yolov8n.pt"
            IMGSZ = 1024
            EPOCHS = 100
            BATCH = -1
            FALLBACK_BATCH = 8
            SINGLE_CLS = True
            CLOSE_MOSAIC = 10
            WORKERS = 2
            CACHE = "disk"
            DEVICE = 0

            BASELINE_640 = {
                "experiment": "v3b_yolov8n_img640",
                "imgsz": 640,
                "precision": 0.70544,
                "recall": 0.28571,
                "mAP50": 0.33904,
                "mAP50-95": 0.11516,
                "coverage_conf_0_05": 0.3673469388,
                "coverage_conf_0_01": 0.5306122449,
                "best_conf_by_f1": 0.05,
            }

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_PROJECT.mkdir(parents=True, exist_ok=True)
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            WORK_DATA_ROOT.mkdir(parents=True, exist_ok=True)

            if not str(WORK_DATA_ROOT.resolve()).startswith("/kaggle/working/"):
                raise ValueError(f"WORK_DATA_ROOT must be under /kaggle/working, got: {WORK_DATA_ROOT}")
            """
        ),
        md("## 2. Install Dependencies and Clone Repository"),
        code(
            """
            import os
            import shutil
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pandas", "pyyaml", "pillow", "matplotlib"],
                check=True,
            )

            os.chdir("/kaggle/working")
            if REPO_DIR.exists():
                print("Removing existing working clone:", REPO_DIR)
                shutil.rmtree(REPO_DIR)
            subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

            os.chdir(PROJECT_DIR)
            print("Project dir:", PROJECT_DIR)
            """
        ),
        md("## 3. Write Local Threshold Sweep Script"),
        code(
            f"""
            SWEEP_SCRIPT = {sweep_script!r}

            scripts_dir = PROJECT_DIR / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            sweep_script_path = scripts_dir / "sweep_v3b_thresholds.py"
            sweep_script_path.write_text(SWEEP_SCRIPT, encoding="utf-8")
            print("Sweep script:", sweep_script_path)
            """
        ),
        md("## 4. Locate or Build v3b Dataset"),
        code(
            """
            import importlib.util
            import shutil
            import sys
            import zipfile

            import pandas as pd

            def find_dataset_dir(folder_name: str) -> Path | None:
                for meta_path in KAGGLE_INPUT_ROOT.rglob("metadata.csv"):
                    parent = meta_path.parent
                    if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                        return parent
                return None

            def find_source_dataset() -> Path:
                if SOURCE_DATASET_DIR.exists() and (SOURCE_DATASET_DIR / "metadata.csv").exists():
                    return SOURCE_DATASET_DIR
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
                if not str(work_dir.resolve()).startswith("/kaggle/working/"):
                    raise ValueError(f"Refusing to delete unsafe path: {work_dir}")
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

            dataset_yaml = work_dir / "v3b.yaml"
            dataset_yaml.write_text(
                f"path: {work_dir.resolve()}\\n\\ntrain: images/train\\nval: images/val\\n\\nnames:\\n  0: kurgan\\n",
                encoding="utf-8",
            )

            metadata_path = work_dir / "metadata.csv"
            meta = pd.read_csv(metadata_path)
            images = meta.drop_duplicates("image")
            boxes = meta[meta["class_name"].notna()]
            print("Dataset:", work_dir)
            print("Images:", len(images))
            print("Positive images:", int(images["is_positive"].sum()))
            print("BBox:", int(len(boxes)))
            print(images.groupby(["split", "is_positive"]).size())
            """
        ),
        md("## 5. Train YOLOv8n at 1024 px"),
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
            best_weights = RUN_DIR / "weights" / "best.pt"
            print("Run dir:", RUN_DIR)
            print("Best weights:", best_weights)
            """
        ),
        md("## 6. Collect Training Metrics"),
        code(
            """
            results = pd.read_csv(RUN_DIR / "results.csv")
            results.columns = [c.strip() for c in results.columns]
            best_idx = results["metrics/mAP50(B)"].idxmax()
            best = results.loc[best_idx]

            img1024_metrics = {
                "experiment": "v3b_yolov8n_img1024",
                "imgsz": IMGSZ,
                "precision": float(best["metrics/precision(B)"]),
                "recall": float(best["metrics/recall(B)"]),
                "mAP50": float(best["metrics/mAP50(B)"]),
                "mAP50-95": float(best["metrics/mAP50-95(B)"]),
                "best_epoch": int(best["epoch"]),
            }

            print(img1024_metrics)
            """
        ),
        md("## 7. Run Threshold Coverage Sweep on 1024 Model"),
        code(
            """
            import subprocess
            import sys

            sweep_out_dir = ANALYSIS_DIR / "threshold_sweep_img1024"
            cmd = [
                sys.executable,
                str(PROJECT_DIR / "scripts" / "sweep_v3b_thresholds.py"),
                "--metadata",
                str(metadata_path),
                "--weights",
                str(best_weights),
                "--out-dir",
                str(sweep_out_dir),
                "--imgsz",
                str(IMGSZ),
                "--device",
                str(DEVICE),
                "--max-visual-images",
                "25",
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)
            """
        ),
        md("## 8. Compare 640 vs 1024"),
        code(
            """
            from IPython.display import Markdown, display

            sweep = pd.read_csv(sweep_out_dir / "confidence_sweep_metrics.csv")
            row_005 = sweep[sweep["conf"].eq(0.05)].iloc[0]
            row_001 = sweep[sweep["conf"].eq(0.01)].iloc[0]

            img1024_metrics["coverage_conf_0_05"] = float(row_005["coverage_rate"])
            img1024_metrics["coverage_conf_0_01"] = float(row_001["coverage_rate"])
            img1024_metrics["threshold_recall_conf_0_05"] = float(row_005["Recall"])
            img1024_metrics["threshold_recall_conf_0_01"] = float(row_001["Recall"])
            img1024_metrics["threshold_precision_conf_0_05"] = float(row_005["Precision"])
            img1024_metrics["threshold_precision_conf_0_01"] = float(row_001["Precision"])
            img1024_metrics["fp_conf_0_05"] = int(row_005["FP"])
            img1024_metrics["fp_conf_0_01"] = int(row_001["FP"])

            comparison = pd.DataFrame([BASELINE_640, img1024_metrics])
            for metric in ["mAP50", "mAP50-95", "coverage_conf_0_05", "coverage_conf_0_01"]:
                comparison[f"delta_{metric}_vs_640"] = comparison[metric] - comparison.loc[0, metric]

            comparison_path = ANALYSIS_DIR / "v3b_img640_vs_img1024_comparison.csv"
            comparison.to_csv(comparison_path, index=False)

            display(Markdown("### 640 vs 1024 comparison"))
            display(comparison)
            display(Markdown("### 1024 confidence sweep"))
            display(sweep)
            display(Markdown((sweep_out_dir / "threshold_sweep_report.md").read_text(encoding="utf-8")))
            """
        ),
        md("## 9. Visual Contact Sheets"),
        code(
            """
            from IPython.display import Image as IPImage, display

            for conf in ["0.05", "0.01"]:
                for kind in ["predictions", "false_positives", "false_negatives"]:
                    path = sweep_out_dir / f"conf_{conf}_{kind}.jpg"
                    if path.exists():
                        print(path.name)
                        display(IPImage(filename=str(path)))
            """
        ),
        md("## 10. Archive Results"),
        code(
            """
            import zipfile
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = Path("/kaggle/working") / f"yolo_v3b_img1024_{timestamp}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root in [RUN_DIR, ANALYSIS_DIR]:
                    for path in root.rglob("*"):
                        if path.is_file():
                            zf.write(path, path.relative_to(OUTPUT_ROOT))

            print("Archive:", archive_path)
            print("Archive size MB:", round(archive_path.stat().st_size / (1024 * 1024), 2))
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
