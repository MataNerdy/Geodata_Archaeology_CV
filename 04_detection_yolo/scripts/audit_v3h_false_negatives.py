from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


MATCH_IOU = 0.50
NEAR_MISS_IOU = 0.30
ANALYSIS_CONF = 0.001
EDGE_MARGIN_PX = 2.0
PAGE_SIZE = 25
CLASS_COLORS = {
    "kurgany_tselye": "#00ff66",
    "kurgany_povrezhdennye": "#ffcc00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed false negative audit for v3h YOLO baseline.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--analysis-conf", type=float, default=ANALYSIS_CONF)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=MATCH_IOU)
    parser.add_argument("--near-miss-iou", type=float, default=NEAR_MISS_IOU)
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


def load_val_metadata(metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    metadata_path = metadata_path.resolve()
    dataset_dir = metadata_path.parent
    meta = pd.read_csv(metadata_path)
    val = meta[meta["split"].astype(str).str.lower().eq("val")].copy()
    if val.empty:
        raise ValueError("No validation rows in metadata.csv")
    if "source_class_name" not in val.columns:
        val["source_class_name"] = val["class_name"]
    if "source_id" not in val.columns:
        source_cols = [col for col in ["region", "modality", "raster_file"] if col in val.columns]
        val["source_id"] = val[source_cols].astype(str).agg("|".join, axis=1)

    val["image_path"] = val.apply(lambda row: resolve_image_path(row, dataset_dir), axis=1)
    missing = sorted({str(path) for path in val["image_path"] if not Path(path).exists()})
    if missing:
        raise FileNotFoundError("Missing validation images:\n" + "\n".join(missing[:10]))
    val["image_key"] = val["image_path"].map(lambda p: str(Path(p).resolve()))

    images = val.drop_duplicates("image_key").copy()
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

    boxes = []
    edge_flags = []
    for _, row in gt.iterrows():
        image = Image.open(row["image_key"])
        width, height = image.size
        if {"yolo_xc", "yolo_yc", "yolo_w", "yolo_h"}.issubset(gt.columns) and pd.notna(row.get("yolo_xc")):
            xc = float(row["yolo_xc"]) * width
            yc = float(row["yolo_yc"]) * height
            bw = float(row["yolo_w"]) * width
            bh = float(row["yolo_h"]) * height
            x1, y1, x2, y2 = (xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2)
        else:
            x1, y1, x2, y2 = (float(row["bbox_x1_px"]), float(row["bbox_y1_px"]), float(row["bbox_x2_px"]), float(row["bbox_y2_px"]))
        boxes.append((x1, y1, x2, y2))
        metadata_edge = str(row.get("bbox_touches_tile_edge", "")).lower() in {"true", "1", "yes"}
        geometric_edge = x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX or x2 >= width - EDGE_MARGIN_PX or y2 >= height - EDGE_MARGIN_PX
        edge_flags.append(bool(metadata_edge or geometric_edge))
    gt["bbox_xyxy"] = boxes
    gt["touches_tile_edge"] = edge_flags
    return images, gt, [Path(p).resolve() for p in images["image_path"]]


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
                }
            )
    columns = ["image_key", "pred_id", "x1", "y1", "x2", "y2", "confidence"]
    return pd.DataFrame(rows, columns=columns)


def match_gt(gt: pd.DataFrame, predictions: pd.DataFrame, best_predictions: pd.DataFrame, match_iou: float) -> pd.DataFrame:
    gt = gt.copy()
    gt["is_found"] = False
    gt["matched_prediction_confidence"] = np.nan
    gt["matched_prediction_iou"] = 0.0
    gt["best_prediction_confidence"] = np.nan
    gt["best_prediction_iou"] = 0.0
    pred_by_image = {key: group.sort_values("confidence", ascending=False).reset_index(drop=True) for key, group in predictions.groupby("image_key")} if not predictions.empty else {}
    best_pred_by_image = {key: group.sort_values("confidence", ascending=False).reset_index(drop=True) for key, group in best_predictions.groupby("image_key")} if not best_predictions.empty else {}

    for image_key, gt_group in gt.groupby("image_key"):
        pred_group = pred_by_image.get(image_key, pd.DataFrame()).copy()
        best_pred_group = best_pred_by_image.get(image_key, pd.DataFrame()).copy()
        gt_indices = list(gt_group.index)
        gt_boxes = np.array(gt_group["bbox_xyxy"].tolist(), dtype=float)
        pred_boxes = pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not pred_group.empty else np.empty((0, 4))
        best_pred_boxes = best_pred_group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not best_pred_group.empty else np.empty((0, 4))
        ious = box_iou(pred_boxes, gt_boxes)
        best_ious = box_iou(best_pred_boxes, gt_boxes)

        if len(best_pred_boxes):
            for local_gt_idx, global_gt_idx in enumerate(gt_indices):
                best_pred_idx = int(np.argmax(best_ious[:, local_gt_idx]))
                gt.loc[global_gt_idx, "best_prediction_iou"] = float(best_ious[best_pred_idx, local_gt_idx])
                gt.loc[global_gt_idx, "best_prediction_confidence"] = float(best_pred_group.iloc[best_pred_idx]["confidence"])

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
            gt.loc[global_gt_idx, "is_found"] = True
            gt.loc[global_gt_idx, "matched_prediction_iou"] = iou_value
            gt.loc[global_gt_idx, "matched_prediction_confidence"] = confidence
    return gt


def add_fn_types(audit: pd.DataFrame, conf_threshold: float, match_iou: float, near_miss_iou: float) -> pd.DataFrame:
    audit = audit.copy()
    best_iou = pd.to_numeric(audit["best_prediction_iou"], errors="coerce").fillna(0.0)
    best_conf = pd.to_numeric(audit["best_prediction_confidence"], errors="coerce").fillna(0.0)
    audit["fn_type"] = "found"
    missed = ~audit["is_found"].astype(bool)
    metric_miss = missed & (best_iou >= match_iou) & (best_conf < conf_threshold)
    near_miss = missed & (~metric_miss) & (best_iou >= near_miss_iou)
    hard_miss = missed & (~metric_miss) & (~near_miss)
    audit.loc[metric_miss, "fn_type"] = "metric_miss"
    audit.loc[near_miss, "fn_type"] = "near_miss"
    audit.loc[hard_miss, "fn_type"] = "hard_miss"
    return audit


def add_features(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = audit.copy()
    area = pd.to_numeric(audit["bbox_area_px"], errors="coerce")
    width = pd.to_numeric(audit["bbox_width_px"], errors="coerce")
    height = pd.to_numeric(audit["bbox_height_px"], errors="coerce")
    objects = pd.to_numeric(audit["objects_in_tile"], errors="coerce")
    median_area = float(area.median())
    dense_threshold = max(3.0, float(objects.quantile(0.75)))

    q = area.quantile([0.2, 0.4, 0.6, 0.8]).to_list()
    # Duplicate quantiles can happen on tiny samples; rank fallback keeps buckets usable.
    if len(set(round(x, 6) for x in q if pd.notna(x))) < 4:
        audit["size_bucket"] = pd.qcut(area.rank(method="first"), 5, labels=["tiny", "small", "medium", "large", "huge"])
    else:
        audit["size_bucket"] = pd.cut(area, bins=[-np.inf, *q, np.inf], labels=["tiny", "small", "medium", "large", "huge"])

    audit["edge_object"] = audit["touches_tile_edge"].astype(bool)
    audit["dense_cluster"] = objects >= dense_threshold
    audit["small_object"] = area < median_area
    audit["large_object"] = area > median_area
    audit["isolated_object"] = objects <= 1
    audit["suspicious_fn"] = (~audit["is_found"]) & (area > median_area) & (~audit["edge_object"]) & (~audit["dense_cluster"])

    stats = pd.DataFrame(
        [
            {"metric": "bbox_area_median", "value": median_area},
            {"metric": "bbox_width_median", "value": float(width.median())},
            {"metric": "bbox_height_median", "value": float(height.median())},
            {"metric": "dense_cluster_objects_threshold", "value": dense_threshold},
            {"metric": "size_bucket_q20", "value": float(area.quantile(0.2))},
            {"metric": "size_bucket_q40", "value": float(area.quantile(0.4))},
            {"metric": "size_bucket_q60", "value": float(area.quantile(0.6))},
            {"metric": "size_bucket_q80", "value": float(area.quantile(0.8))},
        ]
    )
    return audit, stats


def crop_with_header(row: pd.Series, out_path: Path, pad_ratio: float = 0.25, min_size: int = 180) -> Image.Image:
    image = Image.open(row["image_key"]).convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
    bw = x2 - x1
    bh = y2 - y1
    pad = max(bw, bh) * pad_ratio
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(w, int(x2 + pad))
    cy2 = min(h, int(y2 + pad))
    crop = image.crop((cx1, cy1, cx2, cy2))
    scale = max(1.0, min_size / max(crop.width, crop.height))
    if scale > 1.0:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
    header_h = 46
    canvas = Image.new("RGB", (crop.width, crop.height + header_h), "white")
    canvas.paste(crop, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    label = f"{row['image_id']} | {row['region']} | {row['source_class_name']} | area={row['bbox_area_px']:.0f}"
    draw.rectangle([0, 0, canvas.width, header_h], fill="black")
    draw.text((4, 4), label[:120], fill="white", font=font)
    conf_text = f"{row['best_prediction_confidence']:.3f}" if pd.notna(row["best_prediction_confidence"]) else "NA"
    draw.text((4, 22), f"{row['fn_type']} | best IoU={row['best_prediction_iou']:.3f} conf={conf_text}", fill="white", font=font)
    # Draw GT bbox position inside crop after scaling.
    sx = crop.width / max(1, cx2 - cx1)
    sy = crop.height / max(1, cy2 - cy1)
    box = [(x1 - cx1) * sx, header_h + (y1 - cy1) * sy, (x2 - cx1) * sx, header_h + (y2 - cy1) * sy]
    color = CLASS_COLORS.get(str(row["source_class_name"]), "#00ff66")
    draw.rectangle(box, outline=color, width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return canvas


def write_contact_sheets(crop_paths: list[Path], out_dir: Path, prefix: str = "fn_page", page_size: int = PAGE_SIZE, columns: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_idx in range(math.ceil(len(crop_paths) / page_size)):
        paths = crop_paths[page_idx * page_size : (page_idx + 1) * page_size]
        images = [Image.open(path).convert("RGB") for path in paths]
        if not images:
            continue
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)
        rows = math.ceil(len(images) / columns)
        pad = 10
        sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
        for idx, img in enumerate(images):
            x = pad + (idx % columns) * (cell_w + pad)
            y = pad + (idx // columns) * (cell_h + pad)
            sheet.paste(img, (x, y))
        sheet.save(out_dir / f"{prefix}_{page_idx + 1:02d}.jpg", quality=92)


def write_plots(audit: pd.DataFrame, fn: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    found = audit[audit["is_found"]].copy()
    missed = audit[~audit["is_found"]].copy()
    for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, df in [("FOUND", found), ("MISSED", missed)]:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not values.empty:
                ax.hist(values, bins=24, alpha=0.55, label=label)
        ax.set_title(f"{metric}: FOUND vs MISSED")
        ax.set_xlabel(metric)
        ax.set_ylabel("GT objects")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"found_vs_missed_{metric}.png", dpi=180)
        plt.close(fig)

    for col in ["source_class_name", "region", "size_bucket"]:
        counts = fn[col].astype(str).value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.35)))
        counts.plot(kind="barh", ax=ax)
        ax.set_title(f"False negatives by {col}")
        ax.set_xlabel("count")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"fn_by_{col}.png", dpi=180)
        plt.close(fig)


def describe_found_vs_missed(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("FOUND", audit[audit["is_found"]]), ("MISSED", audit[~audit["is_found"]])]:
        for metric in ["bbox_area_px", "bbox_width_px", "bbox_height_px"]:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "group": label,
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


def reason_candidates(fn: pd.DataFrame) -> pd.DataFrame:
    reason_cols = ["small_object", "edge_object", "dense_cluster", "large_object", "isolated_object"]
    rows = [{"reason_candidate": col, "count": int(fn[col].sum())} for col in reason_cols]
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def fn_type_summary(fn: pd.DataFrame) -> pd.DataFrame:
    if fn.empty:
        return pd.DataFrame(columns=["fn_type", "count"])
    return fn["fn_type"].astype(str).value_counts().rename_axis("fn_type").reset_index(name="count")


def priority_images(fn: pd.DataFrame) -> pd.DataFrame:
    by_image = (
        fn.groupby(["image_id", "image_key", "region"], dropna=False)
        .agg(
            fn_count=("gt_id", "count"),
            suspicious_fn_count=("suspicious_fn", "sum"),
            median_fn_area=("bbox_area_px", "median"),
            max_fn_area=("bbox_area_px", "max"),
            source_classes=("source_class_name", lambda s: "; ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    return by_image.sort_values(["suspicious_fn_count", "fn_count", "max_fn_area"], ascending=False).head(20)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: format(x, floatfmt))
    formatted = formatted.fillna("")
    cols = list(formatted.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_summary(
    out_path: Path,
    audit: pd.DataFrame,
    fn: pd.DataFrame,
    size_stats: pd.DataFrame,
    reasons: pd.DataFrame,
    fn_types: pd.DataFrame,
    priority: pd.DataFrame,
    conf_threshold: float,
    match_iou: float,
    near_miss_iou: float,
) -> None:
    found = audit[audit["is_found"]]
    recall = len(found) / len(audit) if len(audit) else 0.0
    top_reason = reasons.iloc[0]["reason_candidate"] if not reasons.empty else "NA"
    fn_by_region = fn["region"].astype(str).value_counts().reset_index()
    fn_by_region.columns = ["region", "fn_count"]
    fn_by_class = fn["source_class_name"].astype(str).value_counts().reset_index()
    fn_by_class.columns = ["source_class_name", "fn_count"]
    fn_by_size = fn["size_bucket"].astype(str).value_counts().reset_index()
    fn_by_size.columns = ["size_bucket", "fn_count"]
    found_median = size_stats[(size_stats["group"].eq("FOUND")) & (size_stats["metric"].eq("bbox_area_px"))]["median"]
    missed_median = size_stats[(size_stats["group"].eq("MISSED")) & (size_stats["metric"].eq("bbox_area_px"))]["median"]

    if not found_median.empty and not missed_median.empty:
        size_answer = "маленькими объектами" if float(missed_median.iloc[0]) < float(found_median.iloc[0]) else "не только маленькими объектами"
    else:
        size_answer = "недостаточно данных для вывода"
    dense_share = float(fn["dense_cluster"].mean()) if len(fn) else 0.0
    dense_answer = "да" if dense_share >= 0.5 else "нет"

    lines = [
        "# v3h False Negative Audit",
        "",
        f"GT objects: `{len(audit)}`",
        f"Found: `{len(found)}`",
        f"Missed: `{len(fn)}`",
        f"Object-level recall at audit threshold: `{recall:.3f}`",
        f"FOUND/MISSED rule: `confidence >= {conf_threshold}` and `IoU >= {match_iou}`.",
        f"FN typing rule: `metric_miss` = IoU >= {match_iou} but confidence < {conf_threshold}; `near_miss` = IoU >= {near_miss_iou} but not FOUND; `hard_miss` = IoU < {near_miss_iou}.",
        "",
        "## FOUND vs MISSED Size Statistics",
        "",
        markdown_table(size_stats),
        "",
        "## False Negatives By Source Class",
        "",
        markdown_table(fn_by_class),
        "",
        "## False Negatives By Region",
        "",
        markdown_table(fn_by_region.head(20)),
        "",
        "## False Negatives By Size Bucket",
        "",
        markdown_table(fn_by_size),
        "",
        "## Reason Candidates",
        "",
        markdown_table(reasons),
        "",
        "## False Negative Types",
        "",
        markdown_table(fn_types),
        "",
        "## 20 Images For Manual Review",
        "",
        markdown_table(priority[["image_id", "region", "fn_count", "suspicious_fn_count", "median_fn_area", "max_fn_area", "source_classes"]]),
        "",
        "## Answers",
        "",
        f"- Самая массовая группа FN: `{top_reason}`.",
        f"- Recall ограничивается {size_answer}; см. медианы FOUND/MISSED выше.",
        f"- Recall ограничивается плотными кластерами: `{dense_answer}` (`dense_cluster` share among FN = `{dense_share:.3f}`).",
        f"- Регионы с наибольшим числом FN: {', '.join(fn_by_region.head(5)['region'].astype(str).tolist())}.",
        "- Подозрительные FN сохранены в `outputs/suspicious_fn/`: это крупные, не edge, не dense-cluster объекты, которые модель должна была увидеть.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = args.out_dir / "outputs"
    crops_dir = outputs_dir / "fn_crops"
    suspicious_dir = outputs_dir / "suspicious_fn"
    sheets_dir = outputs_dir / "fn_contact_sheets"
    plots_dir = args.out_dir / "plots"
    for path in [crops_dir, suspicious_dir, sheets_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    _, gt, image_paths = load_val_metadata(args.metadata)
    model = YOLO(str(args.weights))
    all_predictions = predict(model, image_paths, args.analysis_conf, args.nms_iou, args.imgsz, args.device)
    predictions = all_predictions[pd.to_numeric(all_predictions["confidence"], errors="coerce") >= args.conf].copy() if not all_predictions.empty else all_predictions.copy()
    all_predictions.to_csv(args.out_dir / "predictions_all_conf.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    audit = match_gt(gt, predictions, all_predictions, args.match_iou)
    audit = add_fn_types(audit, args.conf, args.match_iou, args.near_miss_iou)
    audit, feature_stats = add_features(audit)

    export_cols = [
        "gt_id",
        "image_id",
        "image_key",
        "is_found",
        "matched_prediction_confidence",
        "matched_prediction_iou",
        "best_prediction_confidence",
        "best_prediction_iou",
        "fn_type",
        "bbox_area_px",
        "bbox_width_px",
        "bbox_height_px",
        "source_class_name",
        "region",
        "source_id",
        "touches_tile_edge",
        "objects_in_tile",
        "size_bucket",
        "edge_object",
        "dense_cluster",
        "small_object",
        "large_object",
        "isolated_object",
        "suspicious_fn",
    ]
    audit[export_cols].to_csv(args.out_dir / "gt_object_audit.csv", index=False)
    fn = audit[~audit["is_found"]].copy()
    found = audit[audit["is_found"]].copy()
    fn[export_cols].to_csv(args.out_dir / "false_negative_audit.csv", index=False)
    found[export_cols].to_csv(args.out_dir / "found_object_audit.csv", index=False)

    size_stats = describe_found_vs_missed(audit)
    reasons = reason_candidates(fn)
    fn_types = fn_type_summary(fn)
    priority = priority_images(fn)
    size_stats.to_csv(args.out_dir / "found_vs_missed_size_stats.csv", index=False)
    feature_stats.to_csv(args.out_dir / "feature_thresholds.csv", index=False)
    reasons.to_csv(args.out_dir / "reason_candidates.csv", index=False)
    fn_types.to_csv(args.out_dir / "false_negative_types.csv", index=False)
    priority.to_csv(args.out_dir / "manual_review_priority_images.csv", index=False)

    fn_crop_paths: list[Path] = []
    for _, row in fn.sort_values(["suspicious_fn", "bbox_area_px"], ascending=False).iterrows():
        crop_name = f"fn_gt_{int(row['gt_id']):04d}_{Path(str(row['image_id'])).stem}.jpg"
        crop_path = crops_dir / crop_name
        crop_with_header(row, crop_path)
        fn_crop_paths.append(crop_path)
        if bool(row["suspicious_fn"]):
            crop_with_header(row, suspicious_dir / crop_name)

    write_contact_sheets(fn_crop_paths, sheets_dir)
    write_plots(audit, fn, plots_dir)
    write_summary(args.out_dir / "summary.md", audit, fn, size_stats, reasons, fn_types, priority, args.conf, args.match_iou, args.near_miss_iou)

    print("Saved audit to:", args.out_dir)
    print("GT:", len(audit), "found:", len(found), "missed:", len(fn))
    print("FN crops:", crops_dir)
    print("Suspicious FN:", suspicious_dir)
    print("Summary:", args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
