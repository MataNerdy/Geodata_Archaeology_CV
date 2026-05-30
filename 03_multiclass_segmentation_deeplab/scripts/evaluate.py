"""Evaluate a trained DeepLab checkpoint."""

from __future__ import annotations

import argparse
import json
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

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

for _package_name in ("arch_datasets", "losses", "models", "utils"):
    _force_local_package(_package_name)

import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    class_names_for_task,
    filter_multiclass_metadata,
    load_metadata,
    num_classes_for_task,
)
from models.deeplab import build_model
from utils.metrics import (
    binary_metrics_from_confusion,
    confusion_matrix,
    confusion_matrix_to_csv_rows,
    logits_to_predictions,
    multiclass_metrics_from_confusion,
    to_jsonable,
)
from utils.splits import make_split, parse_regions
from utils.visualization import plot_confusion_matrix
from utils.polygon_metrics import competition_like_f1, masks_to_geojson_features
from utils.polygon_postprocessing import postprocess_prediction


def parse_args() -> argparse.Namespace:
    """Parse evaluate CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--task", choices=["binary_kurgan", "kurgan_multiclass", "all_classes", "archaeology_5class"])
    parser.add_argument("--encoder")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--split")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--train-split-csv")
    parser.add_argument("--val-split-csv")
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--eval-mode", choices=["pixel", "object"], default="pixel")
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--use-postprocessing", action="store_true")
    parser.add_argument("--use-morphology-opening", action="store_true")
    parser.add_argument("--morphology-kernel-size", type=int, default=3)
    parser.add_argument("--use-metadata-filtering", action="store_true")
    parser.add_argument("--max-crop-size", type=float)
    parser.add_argument("--max-objects-in-patch", type=int)
    parser.add_argument("--allowed-classes")
    parser.add_argument("--exclude-touches-border", action="store_true")
    parser.add_argument("--min-foreground-pixels", type=int)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run evaluation."""

    args = parse_args()
    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.update({key: value for key, value in vars(args).items() if value is not None})
    config.setdefault("out_dir", str(Path(args.checkpoint).parent))
    config.setdefault("threshold", 0.5)
    config.setdefault("num_workers", 0)
    config.setdefault("batch_size", 8)
    config.setdefault("val_fraction", 0.2)

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_metadata(config["data_root"])
    if bool(config.get("use_metadata_filtering")):
        before_count = len(meta)
        meta = filter_multiclass_metadata(
            meta,
            allowed_classes=parse_str_list(config.get("allowed_classes")),
            max_crop_size=config.get("max_crop_size"),
            max_objects_in_patch=config.get("max_objects_in_patch"),
            exclude_touches_border=bool(config.get("exclude_touches_border")),
            min_foreground_pixels=config.get("min_foreground_pixels"),
        )
        print(f"Metadata filtering: {before_count} -> {len(meta)} samples")

    print(f"[eval] Split used: {config.get('split')}")
    print(f"[eval] Split files used: train={config.get('train_split_csv')} val={config.get('val_split_csv')}")
    print(f"[eval] Modalities: {normalize_modalities(config.get('modalities')) or 'all'}")
    print(f"[eval] Object IoU threshold: {config.get('object_iou_threshold', 0.3)}")
    _, val_df = make_split(
        meta,
        split=config["split"],
        val_region=config.get("val_region"),
        val_regions=parse_regions(config.get("val_regions")),
        val_fraction=float(config["val_fraction"]),
        modalities=normalize_modalities(config.get("modalities")),
        train_split_csv=config.get("train_split_csv"),
        val_split_csv=config.get("val_split_csv"),
    )
    print(f"[eval] Number of samples: {len(val_df)}")
    dataset = ArchaeologySegmentationDataset(
        val_df,
        config["data_root"],
        image_size=int(config["image_size"]),
        task=config["task"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=num_classes_for_task(config["task"]),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    print("[eval] Computing pixel metrics...")
    pred_masks, gt_masks, sample_ids, matrix = collect_predictions(model, loader, device, config)
    class_names = {0: "background", 1: "any_kurgan"} if config["task"] == "binary_kurgan" else class_names_for_task(config["task"])

    if config["eval_mode"] == "pixel":
        metrics = save_pixel_evaluation(matrix, class_names, config, out_dir)
    else:
        print("[eval] Computing object metrics...")
        metrics = save_object_evaluation(pred_masks, gt_masks, sample_ids, config, out_dir)
    print("[eval] Saving evaluation.csv/json")
    print(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False))
    print("[eval] Done")


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, object],
) -> tuple[list, list, list[str], torch.Tensor]:
    """Collect predicted masks, GT masks and pixel confusion matrix."""

    num_classes = 2 if config["task"] == "binary_kurgan" else num_classes_for_task(config["task"])
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    pred_masks = []
    gt_masks = []
    sample_ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"]
        preds = logits_to_predictions(
            model(images),
            config["task"],
            threshold=float(config["threshold"]),
        ).cpu()
        if config["task"] != "binary_kurgan":
            processed = []
            for pred in preds.numpy():
                processed.append(
                    postprocess_prediction(
                        pred,
                        min_component_area=int(config.get("min_component_area", 8)),
                        use_postprocessing=bool(config.get("use_postprocessing")),
                        use_morphology_opening=bool(config.get("use_morphology_opening")),
                        morphology_kernel_size=int(config.get("morphology_kernel_size", 3)),
                    )
                )
            preds = torch.from_numpy(np.stack(processed)).long()
        matrix += confusion_matrix(preds, targets, num_classes)
        pred_masks.extend(list(preds.numpy()))
        gt_masks.extend(list(targets.numpy()))
        sample_ids.extend([str(item) for item in batch["sample_id"]])
    return pred_masks, gt_masks, sample_ids, matrix


def save_pixel_evaluation(
    matrix: torch.Tensor,
    class_names: dict[int, str],
    config: dict[str, object],
    out_dir: Path,
) -> dict[str, float]:
    """Save pixel-level evaluation artifacts."""

    if config["task"] == "binary_kurgan":
        metrics = binary_metrics_from_confusion(matrix)
    else:
        metrics = multiclass_metrics_from_confusion(matrix, class_names)
        plot_confusion_matrix(matrix, class_names, out_dir / "confusion_matrix.png")

    payload = {"metrics": metrics, "config": config, "confusion_matrix": matrix.tolist()}
    for filename in ("evaluation.json", "evaluation_pixel.json"):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_pixel.csv", index=False)
    if config["task"] != "binary_kurgan":
        pd.DataFrame(per_class_rows(metrics, class_names)).to_csv(out_dir / "per_class_iou.csv", index=False)
    pd.DataFrame(confusion_matrix_to_csv_rows(matrix, class_names)).to_csv(out_dir / "confusion_matrix.csv", index=False)
    return metrics


def save_object_evaluation(
    pred_masks: list,
    gt_masks: list,
    sample_ids: list[str],
    config: dict[str, object],
    out_dir: Path,
) -> dict[str, float]:
    """Save object-level competition-like evaluation artifacts."""

    pred_geojson = masks_to_geojson_features(pred_masks, sample_ids, min_area=float(config.get("min_component_area", 8)))
    gt_geojson = masks_to_geojson_features(gt_masks, sample_ids, min_area=float(config.get("min_component_area", 8)))
    weighted_f1, rows = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=float(config.get("object_iou_threshold", 0.3)),
    )
    tp = float(rows["tp"].sum())
    fp = float(rows["fp"].sum())
    fn = float(rows["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    metrics = {
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "weighted_competition_f1": weighted_f1,
        "object_iou_threshold": float(config.get("object_iou_threshold", 0.3)),
    }
    payload = {"metrics": metrics, "config": config, "per_class": rows.to_dict(orient="records")}
    (out_dir / "evaluation_object.json").write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "competition_metric.json").write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_object.csv", index=False)
    rows.to_csv(out_dir / "competition_metric.csv", index=False)
    (out_dir / "predictions_geojson.json").write_text(json.dumps(pred_geojson, ensure_ascii=False), encoding="utf-8")
    (out_dir / "ground_truth_geojson.json").write_text(json.dumps(gt_geojson, ensure_ascii=False), encoding="utf-8")
    return metrics


def per_class_rows(
    metrics: dict[str, float],
    class_names: dict[int, str],
) -> list[dict[str, float | int | str]]:
    """Convert per-class IoU/Dice metrics to a compact table."""

    rows = []
    for class_id, class_name in class_names.items():
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": metrics.get(f"iou_{class_name}"),
                "dice": metrics.get(f"dice_{class_name}"),
            }
        )
    return rows


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


def parse_str_list(value: object) -> list[str] | None:
    """Parse comma-separated strings."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
