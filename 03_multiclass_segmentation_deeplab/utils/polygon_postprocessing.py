"""Polygon extraction and mask postprocessing for archaeology-aware evaluation."""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from utils.polygon_metrics import mask_to_polygons, soft_match


def remove_small_components(
    mask: np.ndarray,
    min_component_area: int = 8,
    class_ids: range | list[int] = range(1, 6),
) -> np.ndarray:
    """Remove tiny connected components class-by-class."""

    if min_component_area <= 0:
        return mask
    cleaned = mask.copy()
    for class_id in class_ids:
        binary = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, num_labels):
            if int(stats[label_id, cv2.CC_STAT_AREA]) < min_component_area:
                cleaned[labels == label_id] = 0
    return cleaned


def morphology_opening(
    mask: np.ndarray,
    kernel_size: int = 3,
    class_ids: range | list[int] = range(1, 6),
) -> np.ndarray:
    """Apply class-wise morphology opening to suppress salt-and-pepper predictions."""

    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = np.zeros_like(mask)
    for class_id in class_ids:
        binary = (mask == class_id).astype(np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        opened[cleaned > 0] = class_id
    return opened


def postprocess_prediction(
    mask: np.ndarray,
    min_component_area: int = 8,
    use_postprocessing: bool = False,
    use_morphology_opening: bool = False,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Apply optional archaeology-aware cleanup to a predicted mask."""

    if not use_postprocessing:
        return mask
    processed = remove_small_components(mask, min_component_area=min_component_area)
    if use_morphology_opening:
        processed = morphology_opening(processed, kernel_size=morphology_kernel_size)
        processed = remove_small_components(processed, min_component_area=min_component_area)
    return processed


def polygonize_prediction(
    mask: np.ndarray,
    min_area: float = 8,
    class_ids: range | list[int] = range(1, 6),
) -> dict[int, list[Polygon]]:
    """Convert a multiclass mask to polygons keyed by class id."""

    return {class_id: mask_to_polygons(mask, class_id, min_area=min_area) for class_id in class_ids}


def polygon_iou_matching(
    pred_polygons: list[Polygon],
    gt_polygons: list[Polygon],
    iou_threshold: float = 0.3,
) -> tuple[int, int, int]:
    """Match predicted and GT polygons using notebook-style soft matching."""

    preds = [(polygon, 1.0) for polygon in pred_polygons]
    gts = [(polygon, 1.0) for polygon in gt_polygons]
    return soft_match(preds, gts, iou_threshold=iou_threshold)

