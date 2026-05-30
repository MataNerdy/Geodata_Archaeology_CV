"""Visualize top/bottom validation examples for one archaeology_5class checkpoint."""

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

import matplotlib.pyplot as plt
import numpy as np
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
from utils.metrics import confusion_matrix, multiclass_metrics_from_confusion
from utils.polygon_metrics import competition_like_f1, masks_to_geojson_features
from utils.splits import make_split


PALETTE = {
    0: (0, 0, 0),
    1: (230, 80, 70),
    2: (80, 160, 240),
    3: (80, 200, 120),
    4: (240, 190, 70),
    5: (180, 100, 220),
}


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
    parser.add_argument("--top-k", type=int, default=10)
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

    meta = load_metadata(args.data_root)
    _, val_df = make_split(
        meta,
        split="frozen",
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        modalities=modalities,
    )

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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=1,
        classes=num_classes_for_task("archaeology_5class"),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    rows = []
    examples = {}

    print("[viz] Running inference")
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()

        for i in range(images.shape[0]):
            sample_id = str(batch["sample_id"][i])
            image_np = images[i, 0].detach().cpu().numpy()
            gt = masks[i].numpy()
            pred = preds[i].numpy()

            metrics = score_one_sample(
                pred,
                gt,
                sample_id,
                args.object_iou_threshold,
                args.min_area,
            )
            row = {
                "sample_id": sample_id,
                "region": str(batch.get("region", [""] * images.shape[0])[i]) if "region" in batch else "",
                "modality": str(batch.get("modality", [""] * images.shape[0])[i]) if "modality" in batch else "",
                **metrics,
            }
            rows.append(row)
            examples[sample_id] = {
                "image": image_np,
                "gt": gt,
                "pred": pred,
                "row": row,
            }

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"[viz] batch {batch_idx}/{len(loader)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_scores.csv", index=False)

    best = df.sort_values("weighted_competition_f1", ascending=False).head(args.top_k)
    worst = df.sort_values("weighted_competition_f1", ascending=True).head(args.top_k)

    best.to_csv(out_dir / "top_best_samples.csv", index=False)
    worst.to_csv(out_dir / "top_worst_samples.csv", index=False)

    save_grid(best, examples, out_dir / "top_10_best.png", title="Top 10 best by per-sample weighted object F1")
    save_grid(worst, examples, out_dir / "top_10_worst.png", title="Top 10 worst by per-sample weighted object F1")

    summary = {
        "checkpoint": str(args.checkpoint),
        "modalities": modalities,
        "top_k": args.top_k,
        "ranking_metric": "weighted_competition_f1",
        "best_samples": best.to_dict(orient="records"),
        "worst_samples": worst.to_dict(orient="records"),
    }
    with (out_dir / "best_worst_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[viz] Saved:")
    print(out_dir / "sample_scores.csv")
    print(out_dir / "top_10_best.png")
    print(out_dir / "top_10_worst.png")


def score_one_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_id: str,
    object_iou_threshold: float,
    min_area: float,
) -> dict[str, float]:
    class_names = class_names_for_task("archaeology_5class")
    matrix = confusion_matrix(torch.from_numpy(pred), torch.from_numpy(gt), 6)
    pixel = multiclass_metrics_from_confusion(matrix, class_names)

    pred_geojson = masks_to_geojson_features([pred], [sample_id], min_area=min_area)
    gt_geojson = masks_to_geojson_features([gt], [sample_id], min_area=min_area)
    weighted_f1, object_df = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=object_iou_threshold,
    )

    tp = float(object_df["tp"].sum())
    fp = float(object_df["fp"].sum())
    fn = float(object_df["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    object_f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "mean_fg_iou": pixel.get("mean_fg_iou"),
        "pixel_accuracy": pixel.get("pixel_accuracy"),
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": object_f1,
        "weighted_competition_f1": weighted_f1,
        "num_pred_objects": int(object_df["num_predictions"].sum()),
        "num_gt_objects": int(object_df["num_ground_truth"].sum()),
    }


def save_grid(df: pd.DataFrame, examples: dict, save_path: Path, title: str) -> None:
    n = len(df)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(14, max(3, n * 2.8)))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(df.to_dict(orient="records")):
        sample_id = str(row["sample_id"])
        ex = examples[sample_id]

        image = normalize_image(ex["image"])
        gt_rgb = colorize_mask(ex["gt"])
        pred_rgb = colorize_mask(ex["pred"])
        overlay = make_overlay(image, ex["pred"])

        panels = [
            (image, "Image", "gray"),
            (gt_rgb, "GT", None),
            (pred_rgb, "Prediction", None),
            (overlay, "Overlay", None),
        ]

        for col_idx, (arr, label, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(arr, cmap=cmap)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(label)

        axes[row_idx, 0].set_ylabel(
            f"{sample_id}\nWF1={row['weighted_competition_f1']:.3f}\n"
            f"OF1={row['object_f1']:.3f}\nIoU={row['mean_fg_iou']:.3f}",
            fontsize=8,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = np.array(color, dtype=np.float32) / 255.0
    return rgb


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = normalize_image(image)
    rgb = np.stack([gray, gray, gray], axis=-1)
    color = colorize_mask(mask)
    alpha = (mask > 0)[..., None] * 0.45
    return rgb * (1 - alpha) + color * alpha


if __name__ == "__main__":
    main()