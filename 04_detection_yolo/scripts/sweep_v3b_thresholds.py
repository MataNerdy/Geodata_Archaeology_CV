from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CONF_SWEEP = [0.50, 0.25, 0.10, 0.05, 0.03, 0.01]
NMS_SWEEP = [0.40, 0.50, 0.60, 0.70, 0.80]
VISUAL_CONFS = [0.25, 0.10, 0.05, 0.01]
MATCH_IOU = 0.50
COVERAGE_IOU = 0.30


@dataclass(frozen=True)
class RunConfig:
    name: str
    conf: float
    nms_iou: float
    stage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confidence/NMS threshold sweep for the v3b YOLO baseline."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../datasets/dataset_yolo_bbox_v3b_li_binary_medium/metadata.csv"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "runs/yolo_baseline_comparison/runs/v3b_li_medium_yolov8n_img640/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/threshold_sweep_v3b"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-visual-images", type=int, default=25)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def xywhn_to_xyxy(row: pd.Series, image_size: int) -> tuple[float, float, float, float]:
    xc = float(row["yolo_xc"]) * image_size
    yc = float(row["yolo_yc"]) * image_size
    w = float(row["yolo_w"]) * image_size
    h = float(row["yolo_h"]) * image_size
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


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


def load_validation_data(metadata_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    metadata_path = metadata_path.resolve()
    project_dir = Path.cwd()
    repo_root = project_dir.parent
    df = pd.read_csv(metadata_path)
    val_df = df[df["split"].astype(str).str.lower().eq("val")].copy()
    if val_df.empty:
        raise ValueError("No validation rows found in metadata.csv")

    val_df["image_path"] = val_df["image"].apply(lambda p: resolve_path(p, project_dir))
    val_df["image_key"] = val_df["image_path"].apply(lambda p: str(p.resolve()))

    missing = sorted({str(p) for p in val_df["image_path"] if not Path(p).exists()})
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Validation images are missing. First paths:\n{preview}")

    gt_df = val_df[val_df["class_name"].notna()].copy()
    gt_df["gt_id"] = np.arange(len(gt_df))
    gt_df["bbox_xyxy"] = gt_df.apply(
        lambda row: xywhn_to_xyxy(row, int(row.get("resize_to", 1024))), axis=1
    )
    gt_df["bbox_width_px"] = gt_df["bbox_x2_px"] - gt_df["bbox_x1_px"]
    gt_df["bbox_height_px"] = gt_df["bbox_y2_px"] - gt_df["bbox_y1_px"]

    gt_by_image = {
        image_key: group.sort_values("gt_id").reset_index(drop=True)
        for image_key, group in gt_df.groupby("image_key")
    }
    image_paths = [Path(p).resolve() for p in val_df.drop_duplicates("image_key")["image_path"]]
    return val_df, gt_by_image, image_paths


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
    rows: list[dict] = []
    for result in results:
        image_key = str(Path(result.path).resolve())
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for idx, (box, score) in enumerate(zip(xyxy, confs)):
            rows.append(
                {
                    "image_key": image_key,
                    "pred_id": f"{Path(image_key).name}:{idx}",
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "confidence": float(score),
                }
            )
    return pd.DataFrame(rows)


def match_predictions(
    predictions: pd.DataFrame,
    gt_by_image: dict[str, pd.DataFrame],
    image_paths: list[Path],
    low_conf_predictions: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tp_rows: list[dict] = []
    fp_rows: list[dict] = []
    fn_rows: list[dict] = []
    covered_gt = 0
    total_gt = sum(len(group) for group in gt_by_image.values())

    pred_by_image = {
        image_key: group.sort_values("confidence", ascending=False).reset_index(drop=True)
        for image_key, group in predictions.groupby("image_key")
    }
    low_pred_by_image = {}
    if low_conf_predictions is not None and not low_conf_predictions.empty:
        low_pred_by_image = {
            image_key: group.sort_values("confidence", ascending=False).reset_index(drop=True)
            for image_key, group in low_conf_predictions.groupby("image_key")
        }

    for image_path in image_paths:
        image_key = str(image_path.resolve())
        gt_group = gt_by_image.get(image_key, pd.DataFrame()).copy()
        pred_group = pred_by_image.get(image_key, pd.DataFrame()).copy()

        gt_boxes = np.array(gt_group["bbox_xyxy"].tolist(), dtype=float) if not gt_group.empty else np.empty((0, 4))
        pred_boxes = (
            pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if not pred_group.empty
            else np.empty((0, 4))
        )
        ious = box_iou(pred_boxes, gt_boxes)

        covered = set()
        if len(pred_boxes) and len(gt_boxes):
            for gt_idx in range(len(gt_boxes)):
                if np.max(ious[:, gt_idx]) >= COVERAGE_IOU:
                    covered.add(gt_idx)
            covered_gt += len(covered)

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        if len(pred_boxes) and len(gt_boxes):
            for pred_idx in range(len(pred_boxes)):
                for gt_idx in range(len(gt_boxes)):
                    if ious[pred_idx, gt_idx] >= MATCH_IOU:
                        candidates.append((float(ious[pred_idx, gt_idx]), pred_idx, gt_idx))
        for iou_value, pred_idx, gt_idx in sorted(candidates, reverse=True):
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            gt_row = gt_group.iloc[gt_idx].to_dict()
            pred_row = pred_group.iloc[pred_idx].to_dict()
            tp_rows.append(
                {
                    **object_fields(gt_row),
                    "image_key": image_key,
                    "prediction_confidence": pred_row["confidence"],
                    "prediction_iou": iou_value,
                    "pred_x1": pred_row["x1"],
                    "pred_y1": pred_row["y1"],
                    "pred_x2": pred_row["x2"],
                    "pred_y2": pred_row["y2"],
                }
            )

        for pred_idx, pred_row in pred_group.iterrows():
            if pred_idx in matched_pred:
                continue
            best_iou = float(np.max(ious[pred_idx])) if len(gt_boxes) else 0.0
            fp_rows.append(
                {
                    "image_key": image_key,
                    "confidence": pred_row["confidence"],
                    "best_gt_iou": best_iou,
                    "x1": pred_row["x1"],
                    "y1": pred_row["y1"],
                    "x2": pred_row["x2"],
                    "y2": pred_row["y2"],
                    "bbox_width_px": pred_row["x2"] - pred_row["x1"],
                    "bbox_height_px": pred_row["y2"] - pred_row["y1"],
                    "bbox_area_px": (pred_row["x2"] - pred_row["x1"]) * (pred_row["y2"] - pred_row["y1"]),
                }
            )

        low_group = low_pred_by_image.get(image_key, pd.DataFrame()).copy()
        low_boxes = (
            low_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if not low_group.empty
            else np.empty((0, 4))
        )
        low_ious = box_iou(low_boxes, gt_boxes)
        for gt_idx, gt_row in gt_group.iterrows():
            if gt_idx in matched_gt:
                continue
            base = object_fields(gt_row.to_dict())
            best_low_iou = 0.0
            best_low_conf = np.nan
            if len(low_boxes):
                best_idx = int(np.argmax(low_ious[:, gt_idx]))
                best_low_iou = float(low_ious[best_idx, gt_idx])
                best_low_conf = float(low_group.iloc[best_idx]["confidence"])
            fn_rows.append(
                {
                    **base,
                    "image_key": image_key,
                    "best_low_conf_iou": best_low_iou,
                    "best_low_confidence": best_low_conf,
                }
            )

    tp = len(tp_rows)
    fp = len(fp_rows)
    fn = len(fn_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "covered_gt": covered_gt,
        "total_gt": total_gt,
        "coverage_rate": covered_gt / total_gt if total_gt else 0.0,
    }
    return metrics, pd.DataFrame(tp_rows), pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)


def object_fields(row: dict) -> dict:
    return {
        "gt_id": row.get("gt_id"),
        "source_class_name": row.get("source_class_name"),
        "class_name": row.get("class_name"),
        "region": row.get("region"),
        "modality": row.get("modality"),
        "bbox_area_px": row.get("bbox_area_px"),
        "bbox_width_px": row.get("bbox_width_px"),
        "bbox_height_px": row.get("bbox_height_px"),
        "bbox_x1_px": row.get("bbox_x1_px"),
        "bbox_y1_px": row.get("bbox_y1_px"),
        "bbox_x2_px": row.get("bbox_x2_px"),
        "bbox_y2_px": row.get("bbox_y2_px"),
    }


def write_metrics_plot(df: pd.DataFrame, x_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["Precision", "Recall", "F1", "coverage_rate"]:
        ax.plot(df[x_col], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if x_col == "conf":
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_distribution_plots(tp: pd.DataFrame, fn: pd.DataFrame, out_dir: Path) -> None:
    dist = []
    if not tp.empty:
        t = tp.copy()
        t["group"] = "TP"
        dist.append(t)
    if not fn.empty:
        f = fn.copy()
        f["group"] = "FN"
        dist.append(f)
    if not dist:
        return
    combined = pd.concat(dist, ignore_index=True)
    for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for group, group_df in combined.groupby("group"):
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if values.empty:
                continue
            ax.hist(values, bins=24, alpha=0.55, label=group)
        ax.set_title(f"{metric}: TP vs FN")
        ax.set_xlabel(metric)
        ax.set_ylabel("objects")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"tp_fn_{metric}_distribution.png", dpi=180)
        plt.close(fig)

    if not tp.empty and "prediction_confidence" in tp.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(tp["prediction_confidence"].dropna(), bins=20, alpha=0.8)
        ax.set_title("TP prediction confidence distribution")
        ax.set_xlabel("confidence")
        ax.set_ylabel("TP objects")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "tp_prediction_confidence_distribution.png", dpi=180)
        plt.close(fig)

    if not fn.empty and "best_low_confidence" in fn.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = fn["best_low_confidence"].dropna()
        if not values.empty:
            ax.hist(values, bins=20, alpha=0.8)
        ax.set_title("FN best low-confidence candidate distribution")
        ax.set_xlabel("best candidate confidence at conf=0.01")
        ax.set_ylabel("FN objects")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "fn_best_low_confidence_distribution.png", dpi=180)
        plt.close(fig)


def draw_boxes(
    image_path: Path,
    gt_rows: pd.DataFrame,
    pred_rows: pd.DataFrame,
    label: str,
    max_size: int = 320,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(max_size / original_w, max_size / original_h)
    new_size = (max(1, int(original_w * scale)), max(1, int(original_h * scale)))
    image = image.resize(new_size)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for _, row in gt_rows.iterrows():
        box = row["bbox_xyxy"] if "bbox_xyxy" in row else xywhn_to_xyxy(row, int(row.get("resize_to", original_w)))
        scaled = tuple(float(v) * scale for v in box)
        draw.rectangle(scaled, outline=(35, 220, 70), width=2)
    for _, row in pred_rows.iterrows():
        scaled = (
            float(row["x1"]) * scale,
            float(row["y1"]) * scale,
            float(row["x2"]) * scale,
            float(row["y2"]) * scale,
        )
        draw.rectangle(scaled, outline=(230, 45, 45), width=2)
        draw.text((scaled[0] + 2, scaled[1] + 2), f"{row['confidence']:.2f}", fill=(255, 235, 235), font=font)
    draw.rectangle((0, 0, image.width, 16), fill=(0, 0, 0))
    draw.text((3, 2), label[:48], fill=(255, 255, 255), font=font)
    return image


def write_contact_sheet(images: list[Image.Image], out_path: Path, columns: int = 5, pad: int = 8) -> None:
    if not images:
        Image.new("RGB", (640, 120), "white").save(out_path)
        return
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
    for idx, img in enumerate(images):
        x = pad + (idx % columns) * (cell_w + pad)
        y = pad + (idx // columns) * (cell_h + pad)
        sheet.paste(img, (x, y))
    sheet.save(out_path, quality=92)


def write_visuals(
    conf: float,
    predictions: pd.DataFrame,
    fp: pd.DataFrame,
    fn: pd.DataFrame,
    gt_by_image: dict[str, pd.DataFrame],
    image_paths: list[Path],
    out_dir: Path,
    max_images: int,
) -> None:
    pred_by_image = {k: g for k, g in predictions.groupby("image_key")}

    pred_images = []
    for image_path in image_paths[:max_images]:
        image_key = str(image_path.resolve())
        gt = gt_by_image.get(image_key, pd.DataFrame())
        pred = pred_by_image.get(image_key, pd.DataFrame())
        pred_images.append(draw_boxes(image_path, gt, pred, f"conf={conf:.2f} {image_path.name}"))
    write_contact_sheet(pred_images, out_dir / f"conf_{conf:.2f}_predictions.jpg")

    fp_images = []
    for _, row in fp.head(max_images).iterrows():
        image_path = Path(row["image_key"])
        gt = gt_by_image.get(str(image_path.resolve()), pd.DataFrame())
        pred = pd.DataFrame([row])
        fp_images.append(draw_boxes(image_path, gt, pred, f"FP {row['confidence']:.2f} {image_path.name}"))
    write_contact_sheet(fp_images, out_dir / f"conf_{conf:.2f}_false_positives.jpg")

    fn_images = []
    for _, row in fn.head(max_images).iterrows():
        image_path = Path(row["image_key"])
        gt = gt_by_image.get(str(image_path.resolve()), pd.DataFrame())
        one_gt = gt[gt["gt_id"].eq(row["gt_id"])] if not gt.empty and "gt_id" in gt else pd.DataFrame([row])
        pred = pred_by_image.get(str(image_path.resolve()), pd.DataFrame())
        fn_images.append(draw_boxes(image_path, one_gt, pred, f"FN {image_path.name}"))
    write_contact_sheet(fn_images, out_dir / f"conf_{conf:.2f}_false_negatives.jpg")


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    out_path: Path,
    confidence_df: pd.DataFrame,
    nms_df: pd.DataFrame,
    best_conf: float,
    best_row: pd.Series,
    best_nms_row: pd.Series,
    tp: pd.DataFrame,
    fp: pd.DataFrame,
    fn: pd.DataFrame,
) -> None:
    low = confidence_df.sort_values("conf").iloc[0]
    baseline = confidence_df[confidence_df["conf"].eq(0.25)]
    baseline_row = baseline.iloc[0] if not baseline.empty else confidence_df.iloc[0]

    recall_gain = float(low["Recall"] - baseline_row["Recall"])
    fp_gain = int(low["FP"] - baseline_row["FP"])
    covered_with_candidate = 0
    if not fn.empty and "best_low_confidence" in fn.columns:
        covered_with_candidate = int((fn["best_low_conf_iou"] >= COVERAGE_IOU).sum())

    lines = [
        "# v3b YOLO Threshold Sweep",
        "",
        "Inference-only analysis for `dataset_yolo_bbox_v3b_li_medium` using the existing YOLOv8n `best.pt`.",
        "",
        "## Confidence Sweep",
        "",
        confidence_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## NMS Sweep",
        "",
        nms_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selected Configurations",
        "",
        f"- Best confidence by F1 at NMS IoU 0.50: `conf={best_conf:.2f}`.",
        f"- Best NMS setting for that confidence by F1: `iou={float(best_nms_row['nms_iou']):.2f}`.",
        f"- Best confidence row: Precision `{format_float(float(best_row['Precision']))}`, Recall `{format_float(float(best_row['Recall']))}`, F1 `{format_float(float(best_row['F1']))}`, FP `{int(best_row['FP'])}`, FN `{int(best_row['FN'])}`.",
        f"- Low-confidence `conf={float(low['conf']):.2f}` row: Precision `{format_float(float(low['Precision']))}`, Recall `{format_float(float(low['Recall']))}`, coverage `{format_float(float(low['coverage_rate']))}`, FP `{int(low['FP'])}`.",
        "",
        "## TP / FN Object Size Summary",
        "",
        size_summary(tp, fn).to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Interpretation",
        "",
        f"1. Recall change from `conf={float(baseline_row['conf']):.2f}` to `conf={float(low['conf']):.2f}` is `{recall_gain:+.4f}`, while FP changes by `{fp_gain:+d}`.",
        f"2. Among FNs at the selected best-F1 configuration, `{covered_with_candidate}` have a low-confidence candidate with IoU >= {COVERAGE_IOU:.2f}.",
        "3. If recall grows only slightly as confidence drops, the model is not just threshold-conservative: many GT objects are not proposed with sufficient spatial overlap.",
        "4. If coverage is materially higher than strict recall, low-confidence inference can still be useful as a proposal generator before segmentation/manual review.",
        "",
        "## Generated Artifacts",
        "",
        "- `confidence_sweep_metrics.csv`",
        "- `nms_sweep_metrics.csv`",
        "- `all_sweep_metrics.csv`",
        "- `tp_objects_best_f1.csv`",
        "- `false_positives_best_f1.csv`",
        "- `false_negatives_best_f1.csv`",
        "- metric/distribution plots as PNG",
        "- contact sheets for `conf=0.25`, `0.10`, `0.05`, `0.01`",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def size_summary(tp: pd.DataFrame, fn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, df in [("TP", tp), ("FN", fn)]:
        if df.empty:
            continue
        for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "count": len(values),
                    "mean": values.mean(),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_path(args.metadata, Path.cwd())
    weights_path = resolve_path(args.weights, Path.cwd())
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    _, gt_by_image, image_paths = load_validation_data(metadata_path)
    model = YOLO(str(weights_path))

    prediction_cache: dict[tuple[float, float], pd.DataFrame] = {}

    def get_predictions(conf: float, nms_iou: float) -> pd.DataFrame:
        key = (conf, nms_iou)
        if key not in prediction_cache:
            print(f"Predicting conf={conf:.2f}, nms_iou={nms_iou:.2f}")
            prediction_cache[key] = predict(model, image_paths, conf, nms_iou, args.imgsz, args.device)
        return prediction_cache[key]

    low_conf_predictions = get_predictions(min(CONF_SWEEP), 0.50)

    all_rows = []
    details: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for conf in CONF_SWEEP:
        preds = get_predictions(conf, 0.50)
        metrics, tp, fp, fn = match_predictions(preds, gt_by_image, image_paths, low_conf_predictions)
        row = {"stage": "confidence", "conf": conf, "nms_iou": 0.50, **metrics}
        all_rows.append(row)
        details[f"conf_{conf:.2f}_iou_0.50"] = (preds, tp, fp, fn)

    confidence_df = pd.DataFrame(all_rows).query("stage == 'confidence'").sort_values("conf", ascending=False)
    best_row = confidence_df.sort_values(["F1", "Recall", "Precision"], ascending=False).iloc[0]
    best_conf = float(best_row["conf"])

    nms_rows = []
    for nms_iou in NMS_SWEEP:
        preds = get_predictions(best_conf, nms_iou)
        metrics, tp, fp, fn = match_predictions(preds, gt_by_image, image_paths, low_conf_predictions)
        row = {"stage": "nms", "conf": best_conf, "nms_iou": nms_iou, **metrics}
        nms_rows.append(row)
        details[f"conf_{best_conf:.2f}_iou_{nms_iou:.2f}"] = (preds, tp, fp, fn)

    nms_df = pd.DataFrame(nms_rows).sort_values("nms_iou")
    all_df = pd.concat([confidence_df, nms_df], ignore_index=True)
    best_nms_row = nms_df.sort_values(["F1", "Recall", "Precision"], ascending=False).iloc[0]

    best_key = f"conf_{best_conf:.2f}_iou_{float(best_nms_row['nms_iou']):.2f}"
    _, best_tp, best_fp, best_fn = details[best_key]

    confidence_df.to_csv(out_dir / "confidence_sweep_metrics.csv", index=False)
    nms_df.to_csv(out_dir / "nms_sweep_metrics.csv", index=False)
    all_df.to_csv(out_dir / "all_sweep_metrics.csv", index=False)
    best_tp.to_csv(out_dir / "tp_objects_best_f1.csv", index=False)
    best_fp.to_csv(out_dir / "false_positives_best_f1.csv", index=False)
    best_fn.to_csv(out_dir / "false_negatives_best_f1.csv", index=False)

    write_metrics_plot(confidence_df, "conf", out_dir / "confidence_sweep_metrics.png", "Confidence sweep at NMS IoU 0.50")
    write_metrics_plot(nms_df, "nms_iou", out_dir / "nms_sweep_metrics.png", f"NMS sweep at conf={best_conf:.2f}")
    write_distribution_plots(best_tp, best_fn, out_dir)

    for conf in VISUAL_CONFS:
        key = f"conf_{conf:.2f}_iou_0.50"
        preds, _, fp, fn = details[key]
        write_visuals(conf, preds, fp, fn, gt_by_image, image_paths, out_dir, args.max_visual_images)

    write_report(
        out_dir / "threshold_sweep_report.md",
        confidence_df,
        nms_df,
        best_conf,
        best_row,
        best_nms_row,
        best_tp,
        best_fp,
        best_fn,
    )

    print(f"Done. Report: {out_dir / 'threshold_sweep_report.md'}")


if __name__ == "__main__":
    main()
