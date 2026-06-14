#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3b_threshold_sweep.ipynb")
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
            # YOLO v3b Threshold Sweep

            Inference-only experiment for the fixed baseline:

            - Dataset: `dataset_yolo_bbox_v3b_li_binary_medium`
            - Model: YOLOv8n `best.pt`
            - No training
            - No dataset modification

            The notebook evaluates confidence thresholds, NMS IoU thresholds, proposal coverage, TP/FN object sizes, and visual contact sheets.
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
            OUTPUT_ROOT = Path("/kaggle/working/yolo_v3b_threshold_sweep")
            ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
            WEIGHTS_WORK_DIR = OUTPUT_ROOT / "weights"

            SOURCE_DATASET_DIR = Path("/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox")
            WEIGHTS_INPUT_PATH = Path("/kaggle/input/datasets/matanerdy/detection-dataset/best.pt")

            DATASET_FOLDER = "dataset_yolo_bbox_v3b_li_binary_medium"
            IMGSZ = 640
            DEVICE = 0

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            WEIGHTS_WORK_DIR.mkdir(parents=True, exist_ok=True)
            WORK_DATA_ROOT.mkdir(parents=True, exist_ok=True)

            if not str(WORK_DATA_ROOT.resolve()).startswith("/kaggle/working/"):
                raise ValueError(
                    f"WORK_DATA_ROOT must be under /kaggle/working, got: {WORK_DATA_ROOT}. "
                    "Kaggle input folders are read-only and must never be deleted or modified."
                )
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
        md("## 3. Write Local Sweep Script"),
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
        md("## 4. Locate Dataset"),
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
                raise FileNotFoundError("Attach a Kaggle input containing dataset_yolo_bbox/ or dataset_yolo_bbox.zip")

            scripts_dir = PROJECT_DIR / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            prebuilt = find_dataset_dir(DATASET_FOLDER)
            work_dir = WORK_DATA_ROOT / DATASET_FOLDER
            if work_dir.exists():
                if not str(work_dir.resolve()).startswith("/kaggle/working/"):
                    raise ValueError(f"Refusing to delete read-only or unsafe path: {work_dir}")
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
            boxes = meta[meta["class_name"].notna()]

            print("Dataset:", work_dir)
            print("Images:", len(images))
            print("Positive images:", int(images["is_positive"].sum()))
            print("BBox:", int(len(boxes)))
            print(images.groupby(["split", "is_positive"]).size())
            """
        ),
        md("## 5. Locate best.pt"),
        code(
            """
            def find_best_weights() -> Path:
                if WEIGHTS_INPUT_PATH.exists():
                    return WEIGHTS_INPUT_PATH
                candidates = sorted(KAGGLE_INPUT_ROOT.rglob("best.pt"))
                if not candidates:
                    raise FileNotFoundError("Attach the trained YOLOv8n v3b best.pt as a Kaggle input.")
                preferred = [
                    p for p in candidates
                    if "v3b" in str(p).lower() or "baseline" in str(p).lower() or "yolov8n" in str(p).lower()
                ]
                selected = preferred[0] if preferred else candidates[0]
                return selected

            source_weights = find_best_weights()
            weights_path = WEIGHTS_WORK_DIR / "v3b_yolov8n_best.pt"
            shutil.copy2(source_weights, weights_path)
            print("Source weights:", source_weights)
            print("Working weights:", weights_path)
            print("Size MB:", round(weights_path.stat().st_size / (1024 * 1024), 2))
            """
        ),
        md("## 6. Run Threshold Sweep"),
        code(
            """
            import subprocess
            import sys

            cmd = [
                sys.executable,
                str(PROJECT_DIR / "scripts" / "sweep_v3b_thresholds.py"),
                "--metadata",
                str(metadata_path),
                "--weights",
                str(weights_path),
                "--out-dir",
                str(ANALYSIS_DIR),
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
        md("## 7. Inspect Metrics"),
        code(
            """
            import pandas as pd
            from IPython.display import Markdown, display

            confidence = pd.read_csv(ANALYSIS_DIR / "confidence_sweep_metrics.csv")
            nms = pd.read_csv(ANALYSIS_DIR / "nms_sweep_metrics.csv")

            display(Markdown("### Confidence sweep"))
            display(confidence)

            display(Markdown("### NMS sweep"))
            display(nms)

            report_path = ANALYSIS_DIR / "threshold_sweep_report.md"
            display(Markdown(report_path.read_text(encoding="utf-8")))
            """
        ),
        md("## 8. Visual Contact Sheets"),
        code(
            """
            from IPython.display import Image as IPImage, display

            for conf in ["0.25", "0.10", "0.05", "0.01"]:
                for kind in ["predictions", "false_positives", "false_negatives"]:
                    path = ANALYSIS_DIR / f"conf_{conf}_{kind}.jpg"
                    if path.exists():
                        print(path.name)
                        display(IPImage(filename=str(path)))
            """
        ),
        md("## 9. Archive Results"),
        code(
            """
            import zipfile
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = Path("/kaggle/working") / f"yolo_v3b_threshold_sweep_{timestamp}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in ANALYSIS_DIR.rglob("*"):
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
