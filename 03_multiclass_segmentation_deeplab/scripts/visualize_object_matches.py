"""Visualize GT/predicted polygon matches for archaeology-aware evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _force_local_package(package_name: str) -> None:
    package_dir = PROJECT_ROOT / package_name
    init_file = package_dir / "__init__.py"
    if not package_dir.exists():
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location(package_name, init_file, submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


for _package_name in ("arch_datasets", "losses", "models", "utils"):
    _force_local_package(_package_name)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import ArchaeologySegmentationDataset, load_metadata, num_classes_for_task
from models.deeplab import build_model
from utils.metrics import logits_to_predictions
from utils.polygon_metrics import mask_to_polygons, soft_match
from utils.polygon_postprocessing import postprocess_prediction
from utils.splits import make_split, parse_regions
from utils.visualization import stretch


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="matched_objects_visualization.png")
    parser.add_argument("--task", default="archaeology_5class")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--use-postprocessing", action="store_true")
    parser.add_argument("--use-morphology-opening", action="store_true")
    parser.add_argument("--max-samples", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Save polygon match visualization."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.update({key: value for key, value in vars(args).items() if value is not None})
    config.setdefault("task", "archaeology_5class")
    config.setdefault("image_size", 256)
    config.setdefault("split", "custom_regions")
    config.setdefault("val_fraction", 0.2)

    _, val_df = make_split(
        load_metadata(config["data_root"]),
        split=config["split"],
        val_region=config.get("val_region"),
        val_regions=parse_regions(config.get("val_regions")),
        val_fraction=float(config["val_fraction"]),
        modalities=normalize_modalities(config.get("modalities")),
    )
    dataset = ArchaeologySegmentationDataset(val_df, config["data_root"], int(config["image_size"]), task=config["task"])
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=num_classes_for_task(config["task"]),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    preds = logits_to_predictions(model(images), config["task"]).cpu().numpy()
    masks = batch["mask"].numpy()
    images_np = images[:, 0].cpu().numpy()

    n = min(int(args.max_samples), len(preds))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for index in range(n):
        pred = postprocess_prediction(
            preds[index],
            min_component_area=int(args.min_component_area),
            use_postprocessing=bool(args.use_postprocessing),
            use_morphology_opening=bool(args.use_morphology_opening),
        )
        gt = masks[index]
        image = stretch(images_np[index])
        for axis in axes[index]:
            axis.imshow(image, cmap="gray")
            axis.axis("off")
        axes[index, 0].set_title(f"{batch['sample_id'][index]} GT polygons")
        axes[index, 1].set_title("Pred polygons")
        axes[index, 2].set_title("Matches: TP blue, FP red, FN yellow")
        draw_all_polygons(axes[index, 0], gt, "lime")
        draw_all_polygons(axes[index, 1], pred, "orange")
        draw_match_panel(axes[index, 2], gt, pred, float(args.object_iou_threshold), int(args.min_component_area))
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved object match visualization to {output}")


def draw_all_polygons(axis, mask: np.ndarray, color: str) -> None:
    """Draw all foreground polygons."""

    for class_id in range(1, 6):
        for polygon in mask_to_polygons(mask, class_id, min_area=8):
            draw_polygon(axis, polygon, color)


def draw_match_panel(axis, gt: np.ndarray, pred: np.ndarray, iou_threshold: float, min_area: int) -> None:
    """Draw TP/FP/FN approximation by class."""

    for class_id in range(1, 6):
        gt_polys = mask_to_polygons(gt, class_id, min_area=min_area)
        pred_polys = mask_to_polygons(pred, class_id, min_area=min_area)
        matched_pred, matched_gt = greedy_match_indices(pred_polys, gt_polys, iou_threshold)
        for idx, polygon in enumerate(pred_polys):
            draw_polygon(axis, polygon, "deepskyblue" if idx in matched_pred else "red")
        for idx, polygon in enumerate(gt_polys):
            if idx not in matched_gt:
                draw_polygon(axis, polygon, "yellow")


def greedy_match_indices(pred_polys, gt_polys, iou_threshold: float):
    """Return matched prediction and GT indices using the same soft-match idea."""

    matched_pred = set()
    matched_gt = set()
    for pred_idx, pred_poly in enumerate(pred_polys):
        best_gt = -1
        best_iou = -1.0
        for gt_idx, gt_poly in enumerate(gt_polys):
            if gt_idx in matched_gt:
                continue
            tp, _, _ = soft_match([(pred_poly, 1.0)], [(gt_poly, 1.0)], iou_threshold=iou_threshold)
            if tp:
                iou = safe_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
        if best_gt >= 0:
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt)
    return matched_pred, matched_gt


def safe_iou(a, b) -> float:
    """Compute polygon IoU defensively."""

    try:
        union = a.union(b).area
        return a.intersection(b).area / union if union > 0 else 0.0
    except Exception:
        return 0.0


def draw_polygon(axis, polygon, color: str) -> None:
    """Draw one shapely polygon outline."""

    if polygon.is_empty:
        return
    if polygon.geom_type == "Polygon":
        coords = np.asarray(polygon.exterior.coords)
        axis.add_patch(MplPolygon(coords, fill=False, edgecolor=color, linewidth=1.6))
        return
    if polygon.geom_type == "MultiPolygon":
        for geom in polygon.geoms:
            draw_polygon(axis, geom, color)
        return
    if polygon.geom_type == "GeometryCollection":
        for geom in polygon.geoms:
            if geom.geom_type in {"Polygon", "MultiPolygon", "GeometryCollection"}:
                draw_polygon(axis, geom, color)


def normalize_modalities(value: object) -> list[str] | None:
    """Normalize modality value."""

    if not value:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return [part.strip() for part in parts if part.strip()] or None


if __name__ == "__main__":
    main()
