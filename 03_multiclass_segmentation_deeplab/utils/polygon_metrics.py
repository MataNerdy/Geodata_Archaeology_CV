"""Competition-like polygon metrics extracted from the evaluation notebook."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.validation import make_valid


CLASS_NAMES = {
    1: "kurgany_tselye",
    2: "kurgany_povrezhdennye",
    3: "gorodishcha",
    4: "fortifikatsii",
    5: "arkhitektury",
}

CLASS_WEIGHTS = {
    "kurgany_povrezhdennye": 27.8,
    "kurgany_tselye": 22.2,
    "gorodishcha": 16.7,
    "arkhitektury": 11.1,
    "fortifikatsii": 5.6,
}


def mask_to_polygons(mask: np.ndarray, class_id: int, min_area: float = 8) -> list[Polygon]:
    """Convert a class mask into valid polygons."""

    binary = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = Polygon(contour[:, 0, :].astype(float))
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        polygons.extend([geom for geom in geoms if geom.area >= min_area])
    return polygons


def masks_to_geojson_features(
    masks: np.ndarray,
    sample_ids: list[str],
    min_area: float = 8,
) -> dict[str, object]:
    """Build patch-level GeoJSON-like features from prediction/GT masks."""

    features = []
    for mask, sample_id in zip(masks, sample_ids, strict=False):
        for class_id, class_name in CLASS_NAMES.items():
            for polygon in mask_to_polygons(mask, class_id, min_area=min_area):
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(polygon),
                        "properties": {
                            "region_name": str(sample_id),
                            "class_name": class_name,
                            "confidence": 1.0,
                        },
                    }
                )
    return {"type": "FeatureCollection", "features": features}


def competition_like_f1(
    pred_geojson: dict[str, object],
    gt_geojson: dict[str, object],
    iou_threshold: float = 0.3,
) -> tuple[float, pd.DataFrame]:
    """Compute weighted competition-like F1 from prediction and GT GeoJSONs."""

    pred_by_class = collect_by_class(pred_geojson)
    gt_by_class = collect_by_class(gt_geojson)
    rows = []
    weighted_sum = 0.0
    weight_sum = 0.0

    for class_name, weight in CLASS_WEIGHTS.items():
        preds = pred_by_class.get(class_name, [])
        gts = gt_by_class.get(class_name, [])
        tp, fp, fn = soft_match(preds, gts, iou_threshold=iou_threshold)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        weighted_sum += f1 * weight
        weight_sum += weight
        rows.append(
            {
                "class_name": class_name,
                "weight": weight,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "num_predictions": len(preds),
                "num_ground_truth": len(gts),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return weighted_sum / weight_sum, pd.DataFrame(rows)


def collect_by_class(geojson_data: dict[str, object]) -> dict[str, list[tuple[Polygon, float]]]:
    """Collect valid polygons by class name."""

    output = {class_name: [] for class_name in CLASS_WEIGHTS}
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {}) or {}
        class_name = props.get("class_name")
        if class_name not in output:
            continue
        polygon = _extract_polygon(feature)
        if polygon is None:
            continue
        output[class_name].append((polygon, _confidence(feature)))
    return output


def soft_match(
    preds: list[tuple[Polygon, float]],
    gts: list[tuple[Polygon, float]],
    iou_threshold: float = 0.3,
) -> tuple[int, int, int]:
    """Greedy polygon matching with IoU or centroid-hit criterion."""

    if not preds and not gts:
        return 0, 0, 0
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0

    preds = sorted(preds, key=lambda item: item[1], reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    for pred_polygon, _ in preds:
        best_iou = -1.0
        best_index = -1
        for index, (gt_polygon, _) in enumerate(gts):
            if index in matched_gt:
                continue
            iou = _polygon_iou(pred_polygon, gt_polygon)
            if (iou > iou_threshold or _centroid_hit(pred_polygon, gt_polygon)) and iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            matched_gt.add(best_index)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def _extract_polygon(feature: dict[str, object]) -> Polygon | None:
    try:
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda polygon: polygon.area)
        if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
            return geom
    except Exception:
        return None
    return None


def _confidence(feature: dict[str, object]) -> float:
    try:
        return float(feature.get("properties", {}).get("confidence", 1.0))
    except Exception:
        return 1.0


def _polygon_iou(a: Polygon, b: Polygon) -> float:
    try:
        union = a.union(b).area
        return a.intersection(b).area / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _centroid_hit(pred_polygon: Polygon, gt_polygon: Polygon) -> bool:
    try:
        centroid = pred_polygon.centroid
        return gt_polygon.contains(centroid) or gt_polygon.boundary.distance(centroid) < 1e-10
    except Exception:
        return False

