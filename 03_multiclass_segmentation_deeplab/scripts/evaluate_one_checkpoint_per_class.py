"""Evaluate one archaeology_5class checkpoint with per-class pixel and object metrics."""

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


for _package_name in ("arch_datasets", "models", "utils"):
    _force_local_package(_package_name)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    class_names_for_task,
    load_metadata,
    num_classes_for_task,
)
from models.deeplab import build_model
from utils.metrics import (
    confusion_matrix,
    multiclass_metrics_from_confusion,
    to_jsonable,
)
from utils.polygon_metrics import competition_like_f1, masks_to_geojson_features
from utils.splits import make_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-split-csv", required=True)
    parser.add_argument("--val-split-csv", required=True)
    parser.add_argument("--modalities", default="Li,Ae,SpOr")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=float, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[eval] Loading metadata")
    meta = load_metadata(args.data_root)

    print("[eval] Loading frozen validation split")
    _, val_df = make_split(
        meta,
        split="frozen",
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        modalities=modalities,
    )

    print(f"[eval] Validation samples: {len(val_df)}")
    print(f"[eval] Modalities: {modalities}")
    print("[eval] Val class distribution:")
    print(val_df["class_name"].value_counts().to_string())
    print("[eval] Val modality distribution:")
    print(val_df["modality"].value_counts().to_string())

    dataset = ArchaeologySegmentationDataset(
        val_df,
        args.data_root,
        image_size=args.image_size,
        task="archaeology_5class",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("[eval] Loading checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    print("[eval] checkpoint epoch:", checkpoint.get("epoch") if isinstance(checkpoint, dict) else None)
    print("[eval] checkpoint selection_metric:", checkpoint.get("selection_metric") if isinstance(checkpoint, dict) else None)
    print("[eval] checkpoint selection_score:", checkpoint.get("selection_score") if isinstance(checkpoint, dict) else None)
    print("[eval] checkpoint config seed:", config.get("seed"))
    print("[eval] checkpoint config modalities:", config.get("modalities"))

    model = build_model(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=1,
        classes=num_classes_for_task("archaeology_5class"),
    ).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    class_names = class_names_for_task("archaeology_5class")
    matrix = torch.zeros((6, 6), dtype=torch.int64)
    pred_masks = []
    gt_masks = []
    sample_ids = []

    print("[eval] Running inference")
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"]

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()

        matrix += confusion_matrix(preds, masks, 6)

        pred_masks.extend([x.numpy() for x in preds])
        gt_masks.extend([x.numpy() for x in masks])
        sample_ids.extend([str(x) for x in batch["sample_id"]])

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"[eval] batch {batch_idx}/{len(loader)}")

    print("[eval] Computing pixel metrics")
    pixel_metrics = multiclass_metrics_from_confusion(matrix, class_names)

    per_class_pixel_rows = []
    for class_id, class_name in class_names.items():
        per_class_pixel_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": pixel_metrics.get(f"iou_{class_name}"),
                "dice": pixel_metrics.get(f"dice_{class_name}"),
            }
        )

    print("[eval] Computing object metrics")
    pred_geojson = masks_to_geojson_features(
        pred_masks,
        sample_ids,
        min_area=args.min_area,
    )
    gt_geojson = masks_to_geojson_features(
        gt_masks,
        sample_ids,
        min_area=args.min_area,
    )

    weighted_f1, object_df = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=args.object_iou_threshold,
    )

    object_df = object_df.copy()
    object_df["object_iou_threshold"] = args.object_iou_threshold
    object_df["min_area"] = args.min_area

    tp = float(object_df["tp"].sum())
    fp = float(object_df["fp"].sum())
    fn = float(object_df["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    object_f1 = 2 * precision * recall / (precision + recall + 1e-6)

    summary = {
        "checkpoint": str(args.checkpoint),
        "encoder": args.encoder,
        "modalities": modalities,
        "num_val_samples": len(val_df),
        "mean_fg_iou": pixel_metrics.get("mean_fg_iou"),
        "mean_fg_dice": pixel_metrics.get("mean_fg_dice"),
        "pixel_accuracy": pixel_metrics.get("pixel_accuracy"),
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": object_f1,
        "weighted_competition_f1": weighted_f1,
        "object_iou_threshold": args.object_iou_threshold,
        "min_area": args.min_area,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_selection_metric": checkpoint.get("selection_metric") if isinstance(checkpoint, dict) else None,
        "checkpoint_selection_score": checkpoint.get("selection_score") if isinstance(checkpoint, dict) else None,
        "checkpoint_config": config,
    }

    print("\n=== Pixel per-class IoU / Dice ===")
    print(pd.DataFrame(per_class_pixel_rows).to_string(index=False))

    print("\n=== Object per-class Precision / Recall / F1 ===")
    print(object_df.to_string(index=False))

    print("\n=== Summary ===")
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))

    pd.DataFrame(per_class_pixel_rows).to_csv(out_dir / "per_class_pixel_metrics.csv", index=False)
    object_df.to_csv(out_dir / "per_class_object_metrics.csv", index=False)
    pd.DataFrame([summary]).drop(columns=["checkpoint_config"]).to_csv(out_dir / "evaluation_summary.csv", index=False)

    with (out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)

    with (out_dir / "checkpoint_config.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(config), f, indent=2, ensure_ascii=False)

    print(f"\n[eval] Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()