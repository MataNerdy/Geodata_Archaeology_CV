#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3g_inference_analysis.ipynb")
SWEEP_SCRIPT_PATH = Path("scripts/sweep_v3g_inference.py")


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
            # v3g YOLO Inference-Only Proposal Analysis

            This notebook does not train a model.

            It evaluates the trained YOLOv8n `v3g_li_manual_keep_yolov8n_640` model as a proposal generator for:

            ```text
            LiDAR tile -> YOLO candidate bbox -> segmentation refinement
            ```

            Outputs:

            - `threshold_sweep.csv`
            - `proposal_coverage.csv`
            - `false_negative_analysis.csv`
            - `summary.md`
            """
        ),
        md("## 1. Configuration"),
        code(
            """
            from pathlib import Path

            KAGGLE_INPUT_ROOT = Path("/kaggle/input")
            WORK_ROOT = Path("/kaggle/working")
            WORK_DATASETS_DIR = WORK_ROOT / "datasets"
            OUTPUT_ROOT = WORK_ROOT / "v3g_inference_analysis"
            ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
            SCRIPT_DIR = OUTPUT_ROOT / "scripts"
            WEIGHTS_WORK_DIR = OUTPUT_ROOT / "weights"

            DATASET_FOLDER = "dataset_yolo_bbox_v3g_li_medium_manual_keep_only"
            PREBUILT_DATASET_DIR = Path(
                "/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3g_li_medium_manual_keep_only"
            )
            DATASET_WORK_DIR = WORK_DATASETS_DIR / DATASET_FOLDER
            METADATA_PATH = DATASET_WORK_DIR / "metadata.csv"

            # Update this if the trained v3g weights are uploaded under a different Kaggle input path.
            PREFERRED_WEIGHTS_PATHS = [
                Path("/kaggle/input/datasets/matanerdy/detection-dataset/v3g_li_manual_keep_yolov8n_640_best.pt"),
                Path("/kaggle/input/datasets/matanerdy/detection-dataset/best.pt"),
            ]

            IMGSZ = 640
            NMS_IOU = 0.50
            DEVICE = 0

            WORK_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            WEIGHTS_WORK_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        md("## 2. Install Dependencies"),
        code(
            """
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pandas", "pillow", "matplotlib"],
                check=True,
            )
            """
        ),
        md("## 3. Copy Dataset to Working Directory"),
        code(
            """
            import shutil
            import zipfile

            import pandas as pd

            def find_dataset_dir(folder_name: str) -> Path | None:
                if PREBUILT_DATASET_DIR.exists():
                    return PREBUILT_DATASET_DIR
                for metadata in KAGGLE_INPUT_ROOT.rglob("metadata.csv"):
                    parent = metadata.parent
                    if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                        return parent
                return None

            def find_dataset_zip(folder_name: str) -> Path | None:
                candidates = sorted(KAGGLE_INPUT_ROOT.rglob(f"{folder_name}.zip"))
                return candidates[0] if candidates else None

            source_dataset = find_dataset_dir(DATASET_FOLDER)
            if DATASET_WORK_DIR.exists():
                shutil.rmtree(DATASET_WORK_DIR)

            if source_dataset is not None:
                print("Copying dataset:", source_dataset)
                shutil.copytree(source_dataset, DATASET_WORK_DIR)
            else:
                zip_path = find_dataset_zip(DATASET_FOLDER)
                if zip_path is None:
                    raise FileNotFoundError(
                        f"Attach Kaggle input containing {DATASET_FOLDER}/ or {DATASET_FOLDER}.zip"
                    )
                print("Extracting dataset:", zip_path)
                unzip_root = WORK_DATASETS_DIR / "_unzipped_v3g"
                if unzip_root.exists():
                    shutil.rmtree(unzip_root)
                unzip_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(unzip_root)
                candidates = [p.parent for p in unzip_root.rglob("metadata.csv") if p.parent.name == DATASET_FOLDER]
                extracted = candidates[0] if candidates else next(unzip_root.rglob("metadata.csv")).parent
                shutil.copytree(extracted, DATASET_WORK_DIR)

            meta = pd.read_csv(METADATA_PATH)
            images = meta.drop_duplicates("image").copy()
            boxes = meta[meta["class_name"].notna()].copy()
            print("Dataset:", DATASET_WORK_DIR)
            print("Images:", len(images))
            print("Positive:", int(images["is_positive"].astype(bool).sum()))
            print("Negative:", int((~images["is_positive"].astype(bool)).sum()))
            print("BBox:", len(boxes))
            print("\\nVal split:")
            print(images[images["split"].eq("val")].groupby("is_positive").size())
            """
        ),
        md("## 4. Locate Weights"),
        code(
            """
            import shutil

            def find_weights() -> Path:
                for path in PREFERRED_WEIGHTS_PATHS:
                    if path.exists():
                        return path
                candidates = sorted(KAGGLE_INPUT_ROOT.rglob("*.pt"))
                if not candidates:
                    raise FileNotFoundError(
                        "Attach the trained v3g YOLO best.pt as a Kaggle input. "
                        "If needed, update PREFERRED_WEIGHTS_PATHS in the config cell."
                    )
                preferred = [
                    p for p in candidates
                    if "v3g" in str(p).lower()
                    or "manual" in str(p).lower()
                    or "keep" in str(p).lower()
                    or "kurgan" in str(p).lower()
                ]
                return preferred[0] if preferred else candidates[0]

            source_weights = find_weights()
            weights_path = WEIGHTS_WORK_DIR / "v3g_li_manual_keep_yolov8n_640_best.pt"
            shutil.copy2(source_weights, weights_path)
            print("Source weights:", source_weights)
            print("Working weights:", weights_path)
            print("Size MB:", round(weights_path.stat().st_size / (1024 * 1024), 2))
            """
        ),
        md("## 5. Write Analysis Script"),
        code(
            f"""
            SWEEP_SCRIPT = {sweep_script!r}

            sweep_script_path = SCRIPT_DIR / "sweep_v3g_inference.py"
            sweep_script_path.write_text(SWEEP_SCRIPT, encoding="utf-8")
            print("Sweep script:", sweep_script_path)
            """
        ),
        md("## 6. Run Inference-Only Analysis"),
        code(
            """
            import subprocess
            import sys

            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "sweep_v3g_inference.py"),
                "--metadata",
                str(METADATA_PATH),
                "--weights",
                str(weights_path),
                "--out-dir",
                str(ANALYSIS_DIR),
                "--imgsz",
                str(IMGSZ),
                "--nms-iou",
                str(NMS_IOU),
                "--device",
                str(DEVICE),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        md("## 7. Inspect Tables"),
        code(
            """
            import pandas as pd
            from IPython.display import Markdown, display

            threshold_sweep = pd.read_csv(ANALYSIS_DIR / "threshold_sweep.csv")
            proposal_coverage = pd.read_csv(ANALYSIS_DIR / "proposal_coverage.csv")
            fn_analysis = pd.read_csv(ANALYSIS_DIR / "false_negative_analysis.csv")
            size_stats = pd.read_csv(ANALYSIS_DIR / "object_size_found_vs_missed.csv")

            display(Markdown("### Threshold Sweep"))
            display(threshold_sweep)
            display(Markdown("### Proposal Coverage"))
            display(proposal_coverage)
            display(Markdown("### Found vs Missed Object Size"))
            display(size_stats)
            display(Markdown(f"False negatives saved: `{len(fn_analysis)}` rows"))
            """
        ),
        md("## 8. Show Plots"),
        code(
            """
            from IPython.display import Image, display

            for name in [
                "threshold_sweep.png",
                "max_iou_per_gt_distribution.png",
                "bbox_area_px_found_vs_missed.png",
                "bbox_width_px_found_vs_missed.png",
                "bbox_height_px_found_vs_missed.png",
            ]:
                path = ANALYSIS_DIR / name
                if path.exists():
                    print(path)
                    display(Image(filename=str(path)))
                else:
                    print("Missing:", path)
            """
        ),
        md("## 9. Summary"),
        code(
            """
            from IPython.display import Markdown, display

            summary_path = ANALYSIS_DIR / "summary.md"
            summary_text = summary_path.read_text(encoding="utf-8")
            display(Markdown(summary_text))
            """
        ),
        md("## 10. Archive Outputs"),
        code(
            """
            import zipfile
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = WORK_ROOT / f"v3g_inference_analysis_{timestamp}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file in OUTPUT_ROOT.rglob("*"):
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


if __name__ == "__main__":
    main()
