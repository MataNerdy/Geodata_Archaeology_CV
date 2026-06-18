#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3h_false_negative_audit.ipynb")
AUDIT_SCRIPT_PATH = Path("scripts/audit_v3h_false_negatives.py")


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
    audit_script = AUDIT_SCRIPT_PATH.read_text(encoding="utf-8")
    cells = [
        md(
            """
            # v3h False Negative Audit

            Inference-only analysis for the current v3h YOLO baseline.

            No training is performed.

            Inputs:

            - Dataset: `/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3h_li_manual_curated_val/dataset_yolo_bbox_v3h_li_manual_curated_val`
            - Weights: `/kaggle/input/datasets/matanerdy/detection-dataset/best.pt`

            Outputs:

            - `gt_object_audit.csv`
            - `false_negative_audit.csv`
            - `false_negative_types.csv`
            - `found_vs_missed_size_stats.csv`
            - `reason_candidates.csv`
            - `manual_review_priority_images.csv`
            - `outputs/fn_crops/`
            - `outputs/fn_contact_sheets/fn_page_*.jpg`
            - `outputs/suspicious_fn/`
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
            OUTPUT_ROOT = WORK_ROOT / "v3h_false_negative_audit"
            SCRIPT_DIR = OUTPUT_ROOT / "scripts"

            DATASET_FOLDER = "dataset_yolo_bbox_v3h_li_manual_curated_val"
            PREBUILT_DATASET_DIR = Path(
                "/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3h_li_manual_curated_val/dataset_yolo_bbox_v3h_li_manual_curated_val"
            )
            WEIGHTS_INPUT_PATH = Path("/kaggle/input/datasets/matanerdy/detection-dataset/best.pt")

            DATASET_WORK_DIR = WORK_DATASETS_DIR / DATASET_FOLDER
            METADATA_PATH = DATASET_WORK_DIR / "metadata.csv"

            IMGSZ = 640
            CONF = 0.25
            ANALYSIS_CONF = 0.001
            NMS_IOU = 0.50
            MATCH_IOU = 0.50
            NEAR_MISS_IOU = 0.30
            DEVICE = 0

            WORK_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
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
        md("## 3. Copy Dataset and Weights"),
        code(
            """
            import shutil
            import zipfile

            import pandas as pd
            import yaml

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
                unzip_root = WORK_DATASETS_DIR / "_unzipped_v3h"
                if unzip_root.exists():
                    shutil.rmtree(unzip_root)
                unzip_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(unzip_root)
                candidates = [p.parent for p in unzip_root.rglob("metadata.csv") if p.parent.name == DATASET_FOLDER]
                extracted = candidates[0] if candidates else next(unzip_root.rglob("metadata.csv")).parent
                shutil.copytree(extracted, DATASET_WORK_DIR)

            dataset_yaml = DATASET_WORK_DIR / "dataset.yaml"
            dataset_yaml.write_text(
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

            if not WEIGHTS_INPUT_PATH.exists():
                candidates = sorted(KAGGLE_INPUT_ROOT.rglob("best.pt"))
                if not candidates:
                    raise FileNotFoundError("Attach best.pt as a Kaggle input.")
                weights_path = candidates[0]
            else:
                weights_path = WEIGHTS_INPUT_PATH

            metadata = pd.read_csv(METADATA_PATH)
            images = metadata.drop_duplicates("image")
            boxes = metadata[metadata["class_name"].notna()]
            print("Dataset:", DATASET_WORK_DIR)
            print("Weights:", weights_path)
            print("Images:", len(images))
            print("Val images:", len(images[images["split"].eq("val")]))
            print("Val bbox:", len(boxes[boxes["split"].eq("val")]))
            print(boxes[boxes["split"].eq("val")].groupby(["region", "source_class_name"]).size())
            """
        ),
        md("## 4. Write Audit Script"),
        code(
            f"""
            AUDIT_SCRIPT = {audit_script!r}

            audit_script_path = SCRIPT_DIR / "audit_v3h_false_negatives.py"
            audit_script_path.write_text(AUDIT_SCRIPT, encoding="utf-8")
            print("Audit script:", audit_script_path)
            """
        ),
        md("## 5. Run False Negative Audit"),
        code(
            """
            import subprocess
            import sys

            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "audit_v3h_false_negatives.py"),
                "--metadata",
                str(METADATA_PATH),
                "--weights",
                str(weights_path),
                "--out-dir",
                str(OUTPUT_ROOT),
                "--imgsz",
                str(IMGSZ),
                "--conf",
                str(CONF),
                "--analysis-conf",
                str(ANALYSIS_CONF),
                "--nms-iou",
                str(NMS_IOU),
                "--match-iou",
                str(MATCH_IOU),
                "--near-miss-iou",
                str(NEAR_MISS_IOU),
                "--device",
                str(DEVICE),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        md("## 6. Inspect Tables"),
        code(
            """
            import pandas as pd
            from IPython.display import Markdown, display

            audit = pd.read_csv(OUTPUT_ROOT / "gt_object_audit.csv")
            fn = pd.read_csv(OUTPUT_ROOT / "false_negative_audit.csv")
            size_stats = pd.read_csv(OUTPUT_ROOT / "found_vs_missed_size_stats.csv")
            reasons = pd.read_csv(OUTPUT_ROOT / "reason_candidates.csv")
            fn_types = pd.read_csv(OUTPUT_ROOT / "false_negative_types.csv")
            priority = pd.read_csv(OUTPUT_ROOT / "manual_review_priority_images.csv")

            display(Markdown("### GT Object Audit"))
            display(audit.head())
            display(Markdown("### FOUND vs MISSED size stats"))
            display(size_stats)
            display(Markdown("### Reason candidates"))
            display(reasons)
            display(Markdown("### False negative types"))
            display(fn_types)
            display(Markdown("### False negative audit with best low-confidence prediction"))
            display(fn[[
                "image_id",
                "region",
                "source_class_name",
                "bbox_area_px",
                "best_prediction_iou",
                "best_prediction_confidence",
                "fn_type",
            ]].head(25))
            display(Markdown("### 20 images to review first"))
            display(priority)
            display(Markdown(f"False negatives: `{len(fn)}`"))
            """
        ),
        md("## 7. Show Plots"),
        code(
            """
            from IPython.display import Image, display

            for path in sorted((OUTPUT_ROOT / "plots").glob("*.png")):
                print(path)
                display(Image(filename=str(path)))
            """
        ),
        md("## 8. Show Contact Sheets"),
        code(
            """
            from IPython.display import Image, display

            sheets = sorted((OUTPUT_ROOT / "outputs" / "fn_contact_sheets").glob("fn_page_*.jpg"))
            for sheet in sheets[:6]:
                print(sheet)
                display(Image(filename=str(sheet)))
            print("Total contact sheets:", len(sheets))
            """
        ),
        md("## 9. Show Suspicious FN Examples"),
        code(
            """
            from IPython.display import Image, display

            suspicious = sorted((OUTPUT_ROOT / "outputs" / "suspicious_fn").glob("*.jpg"))
            for crop in suspicious[:20]:
                print(crop)
                display(Image(filename=str(crop)))
            print("Suspicious FN crops:", len(suspicious))
            """
        ),
        md("## 10. Summary"),
        code(
            """
            from IPython.display import Markdown, display

            summary_text = (OUTPUT_ROOT / "summary.md").read_text(encoding="utf-8")
            display(Markdown(summary_text))
            """
        ),
        md("## 11. Archive Outputs"),
        code(
            """
            import zipfile
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = WORK_ROOT / f"v3h_false_negative_audit_{timestamp}.zip"
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
