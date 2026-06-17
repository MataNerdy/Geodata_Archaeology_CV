#!/usr/bin/env python
from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/kaggle_yolo_v3g_class_difficulty.ipynb")
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
            # v3g Class Difficulty Experiment

            Goal: test whether `kurgany_povrezhdennye` are substantially harder to detect than `kurgany_tselye`.

            Starting point:

            - Source dataset: `dataset_yolo_bbox_v3g_li_medium_manual_keep_only`
            - Modality: Li
            - Existing v3g train/val split is preserved

            This notebook creates two derived single-class datasets:

            1. `dataset_yolo_tselye_only`: labels only `kurgany_tselye`
            2. `dataset_yolo_povrezhdennye_only`: labels only `kurgany_povrezhdennye`

            Images without the target class are treated as negative images for that specific task.
            """
        ),
        md("## 1. Configuration"),
        code(
            """
            from pathlib import Path

            KAGGLE_INPUT_ROOT = Path("/kaggle/input")
            WORK_ROOT = Path("/kaggle/working")
            WORK_DATASETS_DIR = WORK_ROOT / "datasets"
            OUTPUT_ROOT = WORK_ROOT / "v3g_class_difficulty"
            RUN_PROJECT = WORK_ROOT / "runs" / "kurgan_class_difficulty"
            ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
            SCRIPT_DIR = OUTPUT_ROOT / "scripts"

            SOURCE_DATASET_FOLDER = "dataset_yolo_bbox_v3g_li_medium_manual_keep_only"
            PREBUILT_SOURCE_DATASET_DIR = Path(
                "/kaggle/input/datasets/matanerdy/detection-dataset/dataset_yolo_bbox_v3g_li_medium_manual_keep_only"
            )
            SOURCE_WORK_DIR = WORK_DATASETS_DIR / SOURCE_DATASET_FOLDER

            TARGET_DATASETS = {
                "tselye_only": {
                    "dataset_folder": "dataset_yolo_tselye_only",
                    "source_class_name": "kurgany_tselye",
                    "class_name": "kurgan_tselye",
                    "run_name": "tselye_only_yolov8n_640",
                },
                "povrezhdennye_only": {
                    "dataset_folder": "dataset_yolo_povrezhdennye_only",
                    "source_class_name": "kurgany_povrezhdennye",
                    "class_name": "kurgan_povrezhdennye",
                    "run_name": "povrezhdennye_only_yolov8n_640",
                },
            }

            MODEL_NAME = "yolov8n.pt"
            IMGSZ = 640
            EPOCHS = 100
            BATCH = 16
            RANDOM_SEED = 42
            SINGLE_CLS = True
            CLOSE_MOSAIC = 10
            COVERAGE_CONFS = [0.05, 0.01]

            WORK_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_PROJECT.mkdir(parents=True, exist_ok=True)
            ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
            SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
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
        md("## 3. Copy Source v3g Dataset"),
        code(
            """
            import shutil
            import zipfile

            import pandas as pd
            import yaml

            def find_dataset_dir(folder_name: str) -> Path | None:
                if PREBUILT_SOURCE_DATASET_DIR.exists():
                    return PREBUILT_SOURCE_DATASET_DIR
                for metadata in KAGGLE_INPUT_ROOT.rglob("metadata.csv"):
                    parent = metadata.parent
                    if parent.name == folder_name and (parent / "images").exists() and (parent / "labels").exists():
                        return parent
                return None

            def find_dataset_zip(folder_name: str) -> Path | None:
                candidates = sorted(KAGGLE_INPUT_ROOT.rglob(f"{folder_name}.zip"))
                return candidates[0] if candidates else None

            source_dataset = find_dataset_dir(SOURCE_DATASET_FOLDER)
            if SOURCE_WORK_DIR.exists():
                shutil.rmtree(SOURCE_WORK_DIR)

            if source_dataset is not None:
                print("Copying v3g source dataset:", source_dataset)
                shutil.copytree(source_dataset, SOURCE_WORK_DIR)
            else:
                zip_path = find_dataset_zip(SOURCE_DATASET_FOLDER)
                if zip_path is None:
                    raise FileNotFoundError(
                        f"Attach Kaggle input containing {SOURCE_DATASET_FOLDER}/ or {SOURCE_DATASET_FOLDER}.zip"
                    )
                unzip_root = WORK_DATASETS_DIR / "_unzipped_v3g_source"
                if unzip_root.exists():
                    shutil.rmtree(unzip_root)
                unzip_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(unzip_root)
                candidates = [p.parent for p in unzip_root.rglob("metadata.csv") if p.parent.name == SOURCE_DATASET_FOLDER]
                extracted = candidates[0] if candidates else next(unzip_root.rglob("metadata.csv")).parent
                shutil.copytree(extracted, SOURCE_WORK_DIR)

            source_meta = pd.read_csv(SOURCE_WORK_DIR / "metadata.csv")
            source_meta["image_uid"] = source_meta["split"].astype(str) + "/" + source_meta["image_name"].astype(str)
            source_images = source_meta.drop_duplicates("image_uid").copy()
            source_boxes = source_meta[source_meta["source_class_name"].notna()].copy()
            print("Source dataset:", SOURCE_WORK_DIR)
            print("Images:", len(source_images))
            print("Positive:", int(source_images["is_positive"].astype(bool).sum()))
            print("BBox:", len(source_boxes))
            print(source_boxes["source_class_name"].value_counts())
            """
        ),
        md("## 4. Build Class-Specific Datasets"),
        code(
            """
            import math
            from PIL import Image

            def resolve_image_path(row: pd.Series, dataset_dir: Path) -> Path:
                split = str(row["split"])
                image_name = str(row["image_name"])
                candidates = [
                    dataset_dir / "images" / split / image_name,
                    Path(str(row["image"])),
                    dataset_dir / "images" / split / Path(str(row["image"])).name,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                return candidates[0]

            def yolo_line(row: pd.Series) -> str:
                return (
                    f"0 {float(row['yolo_xc']):.6f} {float(row['yolo_yc']):.6f} "
                    f"{float(row['yolo_w']):.6f} {float(row['yolo_h']):.6f}"
                )

            def build_target_dataset(key: str, cfg: dict) -> dict:
                out_dir = WORK_DATASETS_DIR / cfg["dataset_folder"]
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                for split in ["train", "val"]:
                    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

                target_class = cfg["source_class_name"]
                target_boxes = source_meta[source_meta["source_class_name"].eq(target_class)].copy()
                target_by_image = {name: group.copy() for name, group in target_boxes.groupby("image_uid")}

                rows = []
                for _, image_row in source_images.iterrows():
                    split = str(image_row["split"])
                    image_name = str(image_row["image_name"])
                    image_uid = str(image_row["image_uid"])
                    src_img = resolve_image_path(image_row, SOURCE_WORK_DIR)
                    dst_img = out_dir / "images" / split / image_name
                    dst_lbl = out_dir / "labels" / split / str(image_row["label_name"])
                    shutil.copy2(src_img, dst_img)

                    group = target_by_image.get(image_uid, pd.DataFrame()).copy()
                    if group.empty:
                        dst_lbl.write_text("", encoding="utf-8")
                        row = image_row.to_dict()
                        row.update(
                            {
                                "image": str(dst_img.resolve()),
                                "label": str(dst_lbl.resolve()),
                                "class_id": math.nan,
                                "class_name": math.nan,
                                "source_class_name": math.nan,
                                "bbox_x1_px": math.nan,
                                "bbox_y1_px": math.nan,
                                "bbox_x2_px": math.nan,
                                "bbox_y2_px": math.nan,
                                "bbox_area_px": math.nan,
                                "bbox_width_px": math.nan,
                                "bbox_height_px": math.nan,
                                "yolo_xc": math.nan,
                                "yolo_yc": math.nan,
                                "yolo_w": math.nan,
                                "yolo_h": math.nan,
                                "is_positive": False,
                                "n_objects": 0,
                            }
                        )
                        rows.append(row)
                    else:
                        dst_lbl.write_text("\\n".join(yolo_line(row) for _, row in group.iterrows()), encoding="utf-8")
                        for _, box_row in group.iterrows():
                            row = box_row.to_dict()
                            row.update(
                                {
                                    "image": str(dst_img.resolve()),
                                    "label": str(dst_lbl.resolve()),
                                    "class_id": 0,
                                    "class_name": cfg["class_name"],
                                    "is_positive": True,
                                    "n_objects": int(len(group)),
                                }
                            )
                            if "bbox_width_px" not in row or pd.isna(row.get("bbox_width_px")):
                                row["bbox_width_px"] = float(row["bbox_x2_px"]) - float(row["bbox_x1_px"])
                            if "bbox_height_px" not in row or pd.isna(row.get("bbox_height_px")):
                                row["bbox_height_px"] = float(row["bbox_y2_px"]) - float(row["bbox_y1_px"])
                            rows.append(row)

                metadata = pd.DataFrame(rows)
                metadata.to_csv(out_dir / "metadata.csv", index=False)
                (out_dir / "dataset.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "path": str(out_dir.resolve()),
                            "train": "images/train",
                            "val": "images/val",
                            "names": {0: cfg["class_name"]},
                        },
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
                return summarize_dataset(key, cfg, out_dir, metadata)

            def summarize_distribution(values: pd.Series, prefix: str) -> dict:
                values = pd.to_numeric(values, errors="coerce").dropna()
                if values.empty:
                    return {
                        f"{prefix}_mean": 0.0,
                        f"{prefix}_median": 0.0,
                        f"{prefix}_p10": 0.0,
                        f"{prefix}_p90": 0.0,
                    }
                return {
                    f"{prefix}_mean": float(values.mean()),
                    f"{prefix}_median": float(values.median()),
                    f"{prefix}_p10": float(values.quantile(0.10)),
                    f"{prefix}_p90": float(values.quantile(0.90)),
                }

            def summarize_dataset(key: str, cfg: dict, out_dir: Path, metadata: pd.DataFrame) -> dict:
                images = metadata.drop_duplicates("image_uid").copy()
                boxes = metadata[metadata["class_name"].notna()].copy()
                objects_per_image = boxes.groupby("image_uid").size()
                summary = {
                    "dataset": key,
                    "dataset_folder": cfg["dataset_folder"],
                    "target_source_class": cfg["source_class_name"],
                    "images": int(len(images)),
                    "positive_images": int(images["is_positive"].astype(bool).sum()),
                    "negative_images": int((~images["is_positive"].astype(bool)).sum()),
                    "bbox_count": int(len(boxes)),
                    "train_images": int((images["split"] == "train").sum()),
                    "val_images": int((images["split"] == "val").sum()),
                }
                summary.update(summarize_distribution(boxes["bbox_area_px"], "bbox_area"))
                summary.update(summarize_distribution(boxes["bbox_width_px"], "bbox_width"))
                summary.update(summarize_distribution(boxes["bbox_height_px"], "bbox_height"))
                summary.update(summarize_distribution(objects_per_image, "objects_per_positive_image"))
                return summary

            dataset_summaries = []
            for key, cfg in TARGET_DATASETS.items():
                summary = build_target_dataset(key, cfg)
                dataset_summaries.append(summary)

            dataset_summary = pd.DataFrame(dataset_summaries)
            dataset_summary.to_csv(ANALYSIS_ROOT / "class_dataset_summary.csv", index=False)
            display(dataset_summary)
            """
        ),
        md("## 5. Plot Dataset Distributions"),
        code(
            """
            import matplotlib.pyplot as plt

            def plot_dataset_distribution(dataset_key: str, cfg: dict) -> None:
                out_dir = WORK_DATASETS_DIR / cfg["dataset_folder"]
                metadata = pd.read_csv(out_dir / "metadata.csv")
                boxes = metadata[metadata["class_name"].notna()].copy()
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, metric, title in [
                    (axes[0], "bbox_area_px", "BBox area"),
                    (axes[1], "bbox_width_px", "BBox width"),
                    (axes[2], "bbox_height_px", "BBox height"),
                ]:
                    values = pd.to_numeric(boxes[metric], errors="coerce").dropna()
                    ax.hist(values, bins=24, alpha=0.8)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.25)
                fig.suptitle(dataset_key)
                fig.tight_layout()
                fig.savefig(ANALYSIS_ROOT / f"{dataset_key}_bbox_distributions.png", dpi=180)
                plt.show()

                objects_per_image = boxes.groupby("image_uid").size()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(objects_per_image, bins=range(1, int(objects_per_image.max()) + 2) if len(objects_per_image) else 1)
                ax.set_title(f"{dataset_key}: objects per positive image")
                ax.set_xlabel("objects")
                ax.set_ylabel("positive images")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(ANALYSIS_ROOT / f"{dataset_key}_objects_per_image.png", dpi=180)
                plt.show()

            for key, cfg in TARGET_DATASETS.items():
                plot_dataset_distribution(key, cfg)
            """
        ),
        md("## 6. Train Identical YOLO Baselines"),
        code(
            """
            from ultralytics import YOLO

            train_rows = []
            run_dirs = {}

            for key, cfg in TARGET_DATASETS.items():
                dataset_dir = WORK_DATASETS_DIR / cfg["dataset_folder"]
                dataset_yaml = dataset_dir / "dataset.yaml"
                print("=" * 80)
                print("Training:", key)
                model = YOLO(MODEL_NAME)
                result = model.train(
                    data=str(dataset_yaml),
                    imgsz=IMGSZ,
                    epochs=EPOCHS,
                    batch=BATCH,
                    seed=RANDOM_SEED,
                    deterministic=True,
                    single_cls=SINGLE_CLS,
                    close_mosaic=CLOSE_MOSAIC,
                    project=str(RUN_PROJECT),
                    name=cfg["run_name"],
                    exist_ok=True,
                    plots=True,
                )
                run_dir = Path(result.save_dir)
                run_dirs[key] = run_dir
                results = pd.read_csv(run_dir / "results.csv")
                results.columns = [c.strip() for c in results.columns]
                best = results.loc[results["metrics/mAP50(B)"].idxmax()]
                train_rows.append(
                    {
                        "dataset": key,
                        "source_class_name": cfg["source_class_name"],
                        "run_dir": str(run_dir),
                        "best_epoch": int(best["epoch"]),
                        "Precision": float(best["metrics/precision(B)"]),
                        "Recall": float(best["metrics/recall(B)"]),
                        "mAP50": float(best["metrics/mAP50(B)"]),
                        "mAP50-95": float(best["metrics/mAP50-95(B)"]),
                    }
                )

            train_metrics = pd.DataFrame(train_rows)
            train_metrics.to_csv(ANALYSIS_ROOT / "class_training_metrics.csv", index=False)
            display(train_metrics)
            """
        ),
        md("## 7. Run Proposal Coverage Analysis"),
        code(
            f"""
            SWEEP_SCRIPT = {sweep_script!r}

            sweep_script_path = SCRIPT_DIR / "sweep_v3g_inference.py"
            sweep_script_path.write_text(SWEEP_SCRIPT, encoding="utf-8")
            print("Sweep script:", sweep_script_path)
            """
        ),
        code(
            """
            import subprocess
            import sys

            coverage_rows = []
            for key, cfg in TARGET_DATASETS.items():
                dataset_dir = WORK_DATASETS_DIR / cfg["dataset_folder"]
                run_dir = run_dirs[key]
                out_dir = ANALYSIS_ROOT / key / "inference"
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(sweep_script_path),
                    "--metadata",
                    str(dataset_dir / "metadata.csv"),
                    "--weights",
                    str(run_dir / "weights" / "best.pt"),
                    "--out-dir",
                    str(out_dir),
                    "--imgsz",
                    str(IMGSZ),
                    "--nms-iou",
                    "0.50",
                    "--device",
                    "0",
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
                threshold = pd.read_csv(out_dir / "threshold_sweep.csv")
                for conf in COVERAGE_CONFS:
                    row = threshold[threshold["conf"].eq(conf)].iloc[0].to_dict()
                    coverage_rows.append(
                        {
                            "dataset": key,
                            "conf": conf,
                            "coverage_rate": row["coverage_rate"],
                            "Recall": row["Recall"],
                            "Precision": row["Precision"],
                            "F1": row["F1"],
                            "TP": row["TP"],
                            "FP": row["FP"],
                            "FN": row["FN"],
                            "FP_per_image": row["FP_per_image"],
                        }
                    )

            coverage_metrics = pd.DataFrame(coverage_rows)
            coverage_metrics.to_csv(ANALYSIS_ROOT / "class_coverage_metrics.csv", index=False)
            display(coverage_metrics)
            """
        ),
        md("## 8. Final Comparison"),
        code(
            """
            comparison = train_metrics.merge(
                coverage_metrics[coverage_metrics["conf"].eq(0.05)][["dataset", "coverage_rate", "FP_per_image"]].rename(
                    columns={"coverage_rate": "coverage@conf=0.05", "FP_per_image": "FP/image@conf=0.05"}
                ),
                on="dataset",
                how="left",
            ).merge(
                coverage_metrics[coverage_metrics["conf"].eq(0.01)][["dataset", "coverage_rate", "FP_per_image"]].rename(
                    columns={"coverage_rate": "coverage@conf=0.01", "FP_per_image": "FP/image@conf=0.01"}
                ),
                on="dataset",
                how="left",
            ).merge(
                dataset_summary[["dataset", "images", "positive_images", "negative_images", "bbox_count", "bbox_area_median"]],
                on="dataset",
                how="left",
            )

            comparison.to_csv(ANALYSIS_ROOT / "class_difficulty_comparison.csv", index=False)
            display(comparison)
            """
        ),
        md("## 9. Write Analytical Summary"),
        code(
            """
            def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
                formatted = df.copy()
                for col in formatted.columns:
                    if pd.api.types.is_float_dtype(formatted[col]):
                        formatted[col] = formatted[col].map(lambda value: format(value, floatfmt))
                formatted = formatted.fillna("")
                cols = [str(c) for c in formatted.columns]
                lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                for _, row in formatted.iterrows():
                    lines.append("| " + " | ".join(str(row[c]) for c in formatted.columns) + " |")
                return "\\n".join(lines)

            ts = comparison[comparison["dataset"].eq("tselye_only")].iloc[0]
            pv = comparison[comparison["dataset"].eq("povrezhdennye_only")].iloc[0]

            recall_delta = float(pv["Recall"] - ts["Recall"])
            map_delta = float(pv["mAP50"] - ts["mAP50"])
            cov05_delta = float(pv["coverage@conf=0.05"] - ts["coverage@conf=0.05"])
            cov01_delta = float(pv["coverage@conf=0.01"] - ts["coverage@conf=0.01"])

            if pv["mAP50"] < ts["mAP50"] and pv["Recall"] < ts["Recall"] and pv["coverage@conf=0.01"] < ts["coverage@conf=0.01"]:
                verdict = "Поврежденные курганы выглядят существенно сложнее: хуже и detection metrics, и proposal coverage."
            elif pv["coverage@conf=0.01"] >= ts["coverage@conf=0.01"] and pv["mAP50"] < ts["mAP50"]:
                verdict = (
                    "Поврежденные курганы хуже как финальная детекция, но в low-confidence proposal mode модель все же может находить кандидаты. "
                    "Проблема может быть не только в видимости объектов, но и в threshold/calibration."
                )
            else:
                verdict = (
                    "Разница между классами не выглядит однозначным объяснением bottleneck. "
                    "Нужно смотреть размер объектов, число примеров и качество разметки."
                )

            report = [
                "# v3g Class Difficulty Analysis",
                "",
                "## Dataset Summary",
                "",
                md_table(dataset_summary),
                "",
                "## Training Metrics",
                "",
                md_table(train_metrics),
                "",
                "## Coverage Metrics",
                "",
                md_table(coverage_metrics),
                "",
                "## Final Comparison",
                "",
                md_table(comparison),
                "",
                "## Interpretation",
                "",
                verdict,
                "",
                f"- Recall delta `povrezhdennye - tselye`: `{recall_delta:+.4f}`",
                f"- mAP50 delta `povrezhdennye - tselye`: `{map_delta:+.4f}`",
                f"- coverage@0.05 delta `povrezhdennye - tselye`: `{cov05_delta:+.4f}`",
                f"- coverage@0.01 delta `povrezhdennye - tselye`: `{cov01_delta:+.4f}`",
                "",
                "Если поврежденные хуже по mAP/Recall, но low-confidence coverage остается сопоставимым, то они не обязательно полностью невидимы для YOLO: проблема может быть в уверенности и локализации. "
                "Если же coverage тоже сильно ниже, класс действительно является более тяжелым визуальным target для детекции.",
                "",
            ]

            summary_path = ANALYSIS_ROOT / "class_difficulty_summary.md"
            summary_path.write_text("\\n".join(report), encoding="utf-8")
            print(summary_path)
            print("\\n".join(report))
            """
        ),
        md("## 10. Archive Outputs"),
        code(
            """
            import zipfile
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = WORK_ROOT / f"v3g_class_difficulty_{timestamp}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root in [OUTPUT_ROOT, RUN_PROJECT]:
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


if __name__ == "__main__":
    main()
