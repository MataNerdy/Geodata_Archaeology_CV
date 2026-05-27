"""Pixel-level metrics for DeepLab segmentation experiments."""

from __future__ import annotations

import math

import numpy as np
import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build target-by-prediction confusion matrix."""

    preds = preds.reshape(-1).to(torch.int64)
    targets = targets.reshape(-1).to(torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    encoded = targets[valid] * num_classes + preds[valid].clamp(0, num_classes - 1)
    matrix = torch.bincount(encoded, minlength=num_classes**2)
    return matrix.reshape(num_classes, num_classes).cpu()


def binary_metrics_from_confusion(
    matrix: torch.Tensor,
    prefix: str = "",
) -> dict[str, float]:
    """Compute binary foreground metrics."""

    matrix = matrix.float()
    tp = matrix[1, 1]
    tn = matrix[0, 0]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    return {
        f"{prefix}fg_iou": _safe_div(tp, tp + fp + fn),
        f"{prefix}fg_dice": _safe_div(2.0 * tp, 2.0 * tp + fp + fn),
        f"{prefix}precision": _safe_div(tp, tp + fp),
        f"{prefix}recall": _safe_div(tp, tp + fn),
        f"{prefix}pixel_accuracy": _safe_div(tp + tn, matrix.sum()),
    }


def multiclass_metrics_from_confusion(
    matrix: torch.Tensor,
    class_names: dict[int, str],
    prefix: str = "",
) -> dict[str, float]:
    """Compute per-class IoU/Dice and foreground aggregates."""

    matrix = matrix.float()
    tp = matrix.diag()
    target_pixels = matrix.sum(dim=1)
    pred_pixels = matrix.sum(dim=0)
    iou_den = target_pixels + pred_pixels - tp
    dice_den = target_pixels + pred_pixels
    iou = torch.where(iou_den > 0, tp / iou_den.clamp_min(1.0), torch.nan)
    dice = torch.where(dice_den > 0, 2.0 * tp / dice_den.clamp_min(1.0), torch.nan)

    result: dict[str, float] = {}
    for class_id, name in class_names.items():
        result[f"{prefix}iou_{name}"] = _to_float(iou[class_id])
        result[f"{prefix}dice_{name}"] = _to_float(dice[class_id])

    fg_ids = torch.tensor([idx for idx in class_names if idx != 0], dtype=torch.long)
    result[f"{prefix}mean_fg_iou"] = _nanmean(iou[fg_ids])
    result[f"{prefix}mean_fg_dice"] = _nanmean(dice[fg_ids])
    result[f"{prefix}pixel_accuracy"] = _safe_div(tp.sum(), matrix.sum())
    return result


def logits_to_predictions(
    logits: torch.Tensor,
    task: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert logits to integer prediction masks."""

    if task == "binary_kurgan":
        return (torch.sigmoid(logits[:, 0]) > threshold).long()
    return logits.argmax(dim=1)


def confusion_matrix_to_csv_rows(
    matrix: torch.Tensor,
    class_names: dict[int, str],
) -> list[dict[str, int | str]]:
    """Flatten confusion matrix into CSV-friendly rows."""

    rows = []
    values = matrix.cpu().numpy()
    for target_id, target_name in class_names.items():
        for pred_id, pred_name in class_names.items():
            rows.append(
                {
                    "target_id": target_id,
                    "target_name": target_name,
                    "pred_id": pred_id,
                    "pred_name": pred_name,
                    "pixels": int(values[target_id, pred_id]),
                }
            )
    return rows


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if float(denominator) <= 0:
        return math.nan
    return float((numerator / denominator).cpu())


def _nanmean(values: torch.Tensor) -> float:
    values = values[~torch.isnan(values)]
    if values.numel() == 0:
        return math.nan
    return float(values.mean().cpu())


def _to_float(value: torch.Tensor) -> float:
    if torch.isnan(value):
        return math.nan
    return float(value.cpu())


def to_jsonable(value: object) -> object:
    """Convert numpy/torch scalar values into JSON-safe objects."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value

