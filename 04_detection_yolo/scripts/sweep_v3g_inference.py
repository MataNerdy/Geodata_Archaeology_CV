from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


CONF_SWEEP = [0.50, 0.25, 0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
MATCH_IOU = 0.50
COVERAGE_IOU = 0.30
PROPOSAL_IOUS = [0.10, 0.20, 0.30, 0.50]
RECOMMENDATION_CONFS = [0.05, 0.03, 0.01, 0.005]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-only threshold/proposal analysis for v3g YOLO baseline.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_image_path(row: pd.Series, dataset_dir: Path) -> Path:
    split = str(row["split"])
    image_name = str(row.get("image_name") or Path(str(row["image"])).name)
    candidates = [
        dataset_dir / "images" / split / image_name,
        Path(str(row["image"])),
        dataset_dir / "images" / split / Path(str(row["image"])).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_validation_gt(metadata_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    metadata_path = metadata_path.resolve()
    dataset_dir = metadata_path.parent
    meta = pd.read_csv(metadata_path)
    val = meta[meta["split"].astype(str).str.lower().eq("val")].copy()
    if val.empty:
        raise ValueError("No validation rows in metadata.csv")

    val["image_path"] = val.apply(lambda row: resolve_image_path(row, dataset_dir), axis=1)
    missing = sorted({str(path) for path in val["image_path"] if not path.exists()})
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Validation images are missing. First paths:\n{preview}")
    val["image_key"] = val["image_path"].map(lambda p: str(p.resolve()))
    val_images = val.drop_duplicates("image_key").copy()

    gt = val[val["class_name"].notna()].copy()
    gt["gt_id"] = np.arange(len(gt))
    gt["bbox_width_px"] = pd.to_numeric(gt["bbox_width_px"], errors="coerce") if "bbox_width_px" in gt else np.nan
    gt["bbox_height_px"] = pd.to_numeric(gt["bbox_height_px"], errors="coerce") if "bbox_height_px" in gt else np.nan
    if gt["bbox_width_px"].isna().all():
        gt["bbox_width_px"] = pd.to_numeric(gt["bbox_x2_px"], errors="coerce") - pd.to_numeric(gt["bbox_x1_px"], errors="coerce")
    if gt["bbox_height_px"].isna().all():
        gt["bbox_height_px"] = pd.to_numeric(gt["bbox_y2_px"], errors="coerce") - pd.to_numeric(gt["bbox_y1_px"], errors="coerce")

    xyxy = []
    for _, row in gt.iterrows():
        image_path = Path(row["image_key"])
        width, height = Image.open(image_path).size
        if {"yolo_xc", "yolo_yc", "yolo_w", "yolo_h"}.issubset(gt.columns) and pd.notna(row.get("yolo_xc")):
            xc = float(row["yolo_xc"]) * width
            yc = float(row["yolo_yc"]) * height
            bw = float(row["yolo_w"]) * width
            bh = float(row["yolo_h"]) * height
            xyxy.append((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
        else:
            xyxy.append((float(row["bbox_x1_px"]), float(row["bbox_y1_px"]), float(row["bbox_x2_px"]), float(row["bbox_y2_px"])))
    gt["bbox_xyxy"] = xyxy

    gt_by_image = {key: group.sort_values("gt_id").reset_index(drop=True) for key, group in gt.groupby("image_key")}
    image_paths = [Path(p).resolve() for p in val_images["image_path"]]
    return val, gt_by_image, image_paths


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


def predict(model: YOLO, image_paths: list[Path], conf: float, nms_iou: float, imgsz: int, device: str | None) -> pd.DataFrame:
    kwargs = {
        "source": [str(p) for p in image_paths],
        "imgsz": imgsz,
        "conf": conf,
        "iou": nms_iou,
        "verbose": False,
        "save": False,
        "stream": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
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
                    "pred_id": f"{Path(image_key).name}:{pred_idx}",
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "confidence": float(score),
                    "bbox_width_px": float(box[2] - box[0]),
                    "bbox_height_px": float(box[3] - box[1]),
                    "bbox_area_px": float((box[2] - box[0]) * (box[3] - box[1])),
                }
            )
    return pd.DataFrame(rows)


def object_fields(row: pd.Series | dict) -> dict:
    get = row.get
    return {
        "gt_id": get("gt_id"),
        "image_id": get("image_id") or Path(str(get("image_key"))).stem,
        "image_key": get("image_key"),
        "source_class_name": get("source_class_name"),
        "bbox_area_px": get("bbox_area_px"),
        "bbox_width_px": get("bbox_width_px"),
        "bbox_height_px": get("bbox_height_px"),
        "bbox_x1_px": get("bbox_x1_px"),
        "bbox_y1_px": get("bbox_y1_px"),
        "bbox_x2_px": get("bbox_x2_px"),
        "bbox_y2_px": get("bbox_y2_px"),
        "region": get("region"),
        "modality": get("modality"),
    }


def evaluate(
    predictions: pd.DataFrame,
    gt_by_image: dict[str, pd.DataFrame],
    image_paths: list[Path],
    conf: float,
    nms_iou: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tp_rows = []
    fp_rows = []
    fn_rows = []
    covered_gt = 0
    total_gt = sum(len(group) for group in gt_by_image.values())
    pred_by_image = {key: group.sort_values("confidence", ascending=False).reset_index(drop=True) for key, group in predictions.groupby("image_key")} if not predictions.empty else {}

    for image_path in image_paths:
        image_key = str(image_path.resolve())
        gt_group = gt_by_image.get(image_key, pd.DataFrame()).copy()
        pred_group = pred_by_image.get(image_key, pd.DataFrame()).copy()
        gt_boxes = np.array(gt_group["bbox_xyxy"].tolist(), dtype=float) if not gt_group.empty else np.empty((0, 4))
        pred_boxes = pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not pred_group.empty else np.empty((0, 4))
        ious = box_iou(pred_boxes, gt_boxes)

        if len(pred_boxes) and len(gt_boxes):
            covered_gt += int((ious.max(axis=0) >= COVERAGE_IOU).sum())

        candidates = []
        if len(pred_boxes) and len(gt_boxes):
            pred_order = pred_group["confidence"].to_numpy()
            for pred_idx in range(len(pred_boxes)):
                for gt_idx in range(len(gt_boxes)):
                    if ious[pred_idx, gt_idx] >= MATCH_IOU:
                        candidates.append((float(ious[pred_idx, gt_idx]), float(pred_order[pred_idx]), pred_idx, gt_idx))
        matched_pred = set()
        matched_gt = set()
        for iou_value, score, pred_idx, gt_idx in sorted(candidates, reverse=True):
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            gt_row = gt_group.iloc[gt_idx]
            pred_row = pred_group.iloc[pred_idx]
            tp_rows.append(
                {
                    **object_fields(gt_row),
                    "conf": conf,
                    "match_iou": iou_value,
                    "prediction_confidence": float(pred_row["confidence"]),
                    "pred_bbox_area_px": float(pred_row["bbox_area_px"]),
                }
            )

        for pred_idx, pred_row in pred_group.iterrows():
            if pred_idx in matched_pred:
                continue
            best_gt_iou = float(ious[pred_idx].max()) if len(gt_boxes) else 0.0
            fp_rows.append(
                {
                    "conf": conf,
                    "image_key": image_key,
                    "prediction_confidence": float(pred_row["confidence"]),
                    "best_gt_iou": best_gt_iou,
                    "bbox_area_px": float(pred_row["bbox_area_px"]),
                    "bbox_width_px": float(pred_row["bbox_width_px"]),
                    "bbox_height_px": float(pred_row["bbox_height_px"]),
                }
            )

        for gt_idx, gt_row in gt_group.iterrows():
            if gt_idx in matched_gt:
                continue
            best_pred_iou = float(ious[:, gt_idx].max()) if len(pred_boxes) else 0.0
            fn_rows.append({**object_fields(gt_row), "conf": conf, "best_pred_iou_at_conf": best_pred_iou})

    tp = len(tp_rows)
    fp = len(fp_rows)
    fn = len(fn_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "conf": conf,
        "nms_iou": nms_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "covered_gt": covered_gt,
        "total_gt": total_gt,
        "coverage_rate": covered_gt / total_gt if total_gt else 0.0,
        "FP_per_image": fp / len(image_paths) if image_paths else 0.0,
    }
    return metrics, pd.DataFrame(tp_rows), pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)


def max_iou_per_gt(predictions: pd.DataFrame, gt_by_image: dict[str, pd.DataFrame], image_paths: list[Path]) -> pd.DataFrame:
    rows = []
    pred_by_image = {key: group.reset_index(drop=True) for key, group in predictions.groupby("image_key")} if not predictions.empty else {}
    for image_path in image_paths:
        image_key = str(image_path.resolve())
        gt_group = gt_by_image.get(image_key, pd.DataFrame()).copy()
        pred_group = pred_by_image.get(image_key, pd.DataFrame()).copy()
        gt_boxes = np.array(gt_group["bbox_xyxy"].tolist(), dtype=float) if not gt_group.empty else np.empty((0, 4))
        pred_boxes = pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not pred_group.empty else np.empty((0, 4))
        ious = box_iou(pred_boxes, gt_boxes)
        for gt_idx, gt_row in gt_group.iterrows():
            max_iou = float(ious[:, gt_idx].max()) if len(pred_boxes) else 0.0
            best_conf = float(pred_group.iloc[int(np.argmax(ious[:, gt_idx]))]["confidence"]) if len(pred_boxes) else np.nan
            rows.append({**object_fields(gt_row), "max_iou_any_prediction": max_iou, "best_prediction_confidence": best_conf})
    return pd.DataFrame(rows)


def describe_objects(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna() if metric in df else pd.Series(dtype=float)
        if values.empty:
            continue
        rows.append(
            {
                "group": group_name,
                "metric": metric,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def write_plots(threshold_df: pd.DataFrame, proposal_gt: pd.DataFrame, found: pd.DataFrame, missed: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["Precision", "Recall", "F1", "coverage_rate"]:
        ax.plot(threshold_df["conf"], threshold_df[col], marker="o", label=col)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_title("v3g threshold sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(proposal_gt["max_iou_any_prediction"], bins=24, alpha=0.85)
    ax.set_xlabel("max IoU per GT")
    ax.set_ylabel("GT objects")
    ax.set_title("Proposal max IoU distribution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "max_iou_per_gt_distribution.png", dpi=180)
    plt.close(fig)

    combined = []
    if not found.empty:
        tmp = found.copy()
        tmp["group"] = "found"
        combined.append(tmp)
    if not missed.empty:
        tmp = missed.copy()
        tmp["group"] = "missed"
        combined.append(tmp)
    if combined:
        both = pd.concat(combined, ignore_index=True)
        for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for group, group_df in both.groupby("group"):
                values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
                if not values.empty:
                    ax.hist(values, bins=24, alpha=0.55, label=group)
            ax.set_title(f"{metric}: found vs missed")
            ax.set_xlabel(metric)
            ax.set_ylabel("GT objects")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{metric}_found_vs_missed.png", dpi=180)
            plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: format(x, floatfmt))
    formatted = formatted.fillna("")
    cols = [str(c) for c in formatted.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in formatted.columns) + " |")
    return "\n".join(lines)


def write_summary(
    out_path: Path,
    threshold_df: pd.DataFrame,
    proposal_cov: pd.DataFrame,
    size_stats: pd.DataFrame,
    rec_df: pd.DataFrame,
    selected_conf: float,
) -> None:
    selected = threshold_df[threshold_df["conf"].eq(selected_conf)].iloc[0]
    high = threshold_df[threshold_df["conf"].eq(0.25)].iloc[0]
    low = threshold_df[threshold_df["conf"].eq(0.001)].iloc[0]

    if float(low["coverage_rate"]) > float(high["coverage_rate"]) + 0.10:
        verdict = "B. Модель умеет предлагать часть кандидатов, но стандартный threshold слишком высокий."
    else:
        verdict = "A. Основное ограничение похоже не только на threshold: модель часто не генерирует достаточно близкие кандидаты."

    lines = [
        "# v3g Inference-Only Proposal Analysis",
        "",
        "## Threshold Sweep",
        "",
        markdown_table(threshold_df),
        "",
        "## Proposal Coverage",
        "",
        markdown_table(proposal_cov),
        "",
        "## Proposal Mode Candidates",
        "",
        markdown_table(rec_df),
        "",
        "## Found vs Missed Object Size",
        "",
        markdown_table(size_stats, floatfmt=".2f"),
        "",
        "## Итоговый вывод",
        "",
        verdict,
        "",
        f"При стандартном `conf=0.25` модель дает Recall `{high['Recall']:.3f}` и coverage `{high['coverage_rate']:.3f}`. "
        f"При экстремально низком `conf=0.001` Recall становится `{low['Recall']:.3f}`, coverage `{low['coverage_rate']:.3f}`, "
        f"но FP/image растет до `{low['FP_per_image']:.2f}`.",
        "",
        f"Для proposal режима выбран рабочий кандидат `conf={selected_conf}`: "
        f"Recall `{selected['Recall']:.3f}`, coverage `{selected['coverage_rate']:.3f}`, FP/image `{selected['FP_per_image']:.2f}`. "
        "Это значение стоит сравнивать визуально с соседними `0.03`, `0.01`, `0.005`, потому что итоговый выбор зависит от того, сколько ложных crop-кандидатов выдержит следующий segmentation/refinement этап.",
        "",
        "Если coverage на низких threshold заметно выше обычного Recall, то связка `LiDAR -> YOLO proposal generation -> segmentation refinement` выглядит более реалистичной, чем попытка сразу получить высокий mAP на детекторе. "
        "Если же coverage почти не растет, bottleneck остается в данных/разметке/визуальной различимости объектов, а не только в threshold.",
        "",
        "## Saved Files",
        "",
        "- `threshold_sweep.csv`",
        "- `proposal_coverage.csv`",
        "- `false_negative_analysis.csv`",
        "- `found_object_analysis.csv`",
        "- `object_size_found_vs_missed.csv`",
        "- `max_iou_per_gt.csv`",
        "- `summary.md`",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val, gt_by_image, image_paths = load_validation_gt(args.metadata)
    model = YOLO(str(args.weights))

    all_metrics = []
    tp_by_conf: dict[float, pd.DataFrame] = {}
    fn_by_conf: dict[float, pd.DataFrame] = {}
    fp_by_conf: dict[float, pd.DataFrame] = {}
    pred_by_conf: dict[float, pd.DataFrame] = {}
    for conf in CONF_SWEEP:
        print(f"Predict/evaluate conf={conf}")
        predictions = predict(model, image_paths, conf=conf, nms_iou=args.nms_iou, imgsz=args.imgsz, device=args.device)
        metrics, tp, fp, fn = evaluate(predictions, gt_by_image, image_paths, conf=conf, nms_iou=args.nms_iou)
        all_metrics.append(metrics)
        tp_by_conf[conf] = tp
        fp_by_conf[conf] = fp
        fn_by_conf[conf] = fn
        pred_by_conf[conf] = predictions

    threshold_df = pd.DataFrame(all_metrics)
    threshold_df.to_csv(args.out_dir / "threshold_sweep.csv", index=False)

    low_conf = min(CONF_SWEEP)
    proposal_gt = max_iou_per_gt(pred_by_conf[low_conf], gt_by_image, image_paths)
    proposal_gt.to_csv(args.out_dir / "max_iou_per_gt.csv", index=False)
    proposal_rows = []
    total_gt = len(proposal_gt)
    for iou_threshold in PROPOSAL_IOUS:
        covered = int((proposal_gt["max_iou_any_prediction"] >= iou_threshold).sum())
        proposal_rows.append(
            {
                "prediction_conf": low_conf,
                "nms_iou": args.nms_iou,
                "coverage_iou_threshold": iou_threshold,
                "covered_gt": covered,
                "total_gt": total_gt,
                "coverage_rate": covered / total_gt if total_gt else 0.0,
            }
        )
    proposal_cov = pd.DataFrame(proposal_rows)
    proposal_cov.to_csv(args.out_dir / "proposal_coverage.csv", index=False)

    rec_df = threshold_df[threshold_df["conf"].isin(RECOMMENDATION_CONFS)].copy()
    rec_df = rec_df[["conf", "TP", "FP", "FN", "Precision", "Recall", "F1", "coverage_rate", "FP_per_image"]]
    rec_df.to_csv(args.out_dir / "proposal_mode_candidates.csv", index=False)

    # Use the best F1 among the proposal-relevant thresholds as the main FN table.
    selected_row = rec_df.sort_values(["F1", "coverage_rate"], ascending=False).iloc[0]
    selected_conf = float(selected_row["conf"])
    false_negative = fn_by_conf[selected_conf].copy()
    found = tp_by_conf[selected_conf].copy()
    false_negative[["image_id", "bbox_area_px", "bbox_width_px", "bbox_height_px", "source_class_name"]].to_csv(
        args.out_dir / "false_negative_analysis.csv", index=False
    )
    found.to_csv(args.out_dir / "found_object_analysis.csv", index=False)
    fp_by_conf[selected_conf].to_csv(args.out_dir / "false_positive_analysis.csv", index=False)

    all_fn = pd.concat([df for df in fn_by_conf.values() if not df.empty], ignore_index=True) if any(not df.empty for df in fn_by_conf.values()) else pd.DataFrame()
    all_fn.to_csv(args.out_dir / "false_negative_analysis_all_thresholds.csv", index=False)

    size_stats = pd.concat(
        [describe_objects(found, "found"), describe_objects(false_negative, "missed")],
        ignore_index=True,
    )
    size_stats.to_csv(args.out_dir / "object_size_found_vs_missed.csv", index=False)

    write_plots(threshold_df, proposal_gt, found, false_negative, args.out_dir)
    write_summary(args.out_dir / "summary.md", threshold_df, proposal_cov, size_stats, rec_df, selected_conf)
    print("Saved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
