"""Metrics for multiclass semantic segmentation."""

from __future__ import annotations

import math

import torch

from config import CLASS_NAMES


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Build a target-by-prediction confusion matrix."""

    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    encoded = targets[valid] * num_classes + preds[valid].clamp(0, num_classes - 1)
    matrix = torch.bincount(encoded, minlength=num_classes**2)
    return matrix.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(matrix: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Compute IoU and Dice metrics from a confusion matrix."""

    matrix = matrix.float()
    tp = matrix.diag()
    target_pixels = matrix.sum(dim=1)
    pred_pixels = matrix.sum(dim=0)

    iou_den = target_pixels + pred_pixels - tp
    dice_den = target_pixels + pred_pixels
    iou = torch.where(iou_den > 0, tp / iou_den.clamp_min(1.0), torch.nan)
    dice = torch.where(dice_den > 0, 2.0 * tp / dice_den.clamp_min(1.0), torch.nan)

    result: dict[str, float] = {}
    for class_id, name in CLASS_NAMES.items():
        result[f"{prefix}iou_{name}"] = _to_float(iou[class_id])
        result[f"{prefix}dice_{name}"] = _to_float(dice[class_id])

    fg_classes = torch.tensor([1, 2])
    result[f"{prefix}mean_fg_iou"] = _nanmean(iou[fg_classes])
    result[f"{prefix}mean_fg_dice"] = _nanmean(dice[fg_classes])

    fg_intersection = matrix[1:, 1:].sum()
    fg_target = matrix[1:, :].sum()
    fg_pred = matrix[:, 1:].sum()
    fg_union = fg_target + fg_pred - fg_intersection
    result[f"{prefix}fg_iou"] = _safe_div(fg_intersection, fg_union)
    result[f"{prefix}fg_dice"] = _safe_div(2.0 * fg_intersection, fg_target + fg_pred)

    correct = tp.sum()
    total = matrix.sum()
    result[f"{prefix}pixel_accuracy"] = _safe_div(correct, total)
    return result


def update_modality_confusions(
    store: dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
    modalities: list[str],
    num_classes: int = 3,
) -> None:
    """Accumulate confusion matrices separately for each modality."""

    for index, modality in enumerate(modalities):
        if modality not in store:
            store[modality] = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        store[modality] += confusion_matrix(preds[index], targets[index], num_classes)


def flatten_modality_metrics(
    modality_confusions: dict[str, torch.Tensor],
    split: str,
) -> dict[str, float]:
    """Convert per-modality confusion matrices to flat CSV-friendly metrics."""

    output: dict[str, float] = {}
    for modality, matrix in sorted(modality_confusions.items()):
        output.update(metrics_from_confusion(matrix, prefix=f"{split}_{modality}_"))
    return output


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
