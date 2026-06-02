"""Build a README figure comparing model evolution on fixed validation patches."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    num_classes_for_task,
    sample_id_to_name,
)
from models.deeplab import build_model
from scripts.generate_final_readme_visualizations import mean_matched_iou, sample_metrics
from utils.polygon_postprocessing import postprocess_prediction
from utils.visualization import colorize_mask, mask_overlay, stretch


DEFAULT_SAMPLE_IDS = ("000546", "000443", "000065", "000007", "000180")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder: str
    checkpoint: str
    final_stage_c: bool = False


MODEL_SPECS = (
    ModelSpec(
        "ResNet34 Li",
        "resnet34",
        "runs/archaeology_5class_encoder_modality_ablation_raw_v1/resnet34_li/best_model (1).pth",
    ),
    ModelSpec(
        "ResNet50 Li",
        "resnet50",
        "runs/archaeology_5class_encoder_modality_ablation_raw_v1/resnet50_li/best_model (1).pth",
    ),
    ModelSpec(
        "ResNet34 All",
        "resnet34",
        "runs/archaeology_5class_encoder_modality_ablation_raw_v1/resnet34_all/best_model (1).pth",
    ),
    ModelSpec(
        "ResNet50 All",
        "resnet50",
        "runs/archaeology_5class_encoder_modality_ablation_raw_v1/resnet50_all/best_model (1).pth",
    ),
    ModelSpec(
        "Final Stage C",
        "resnet34",
        "runs/different_seeds_experiments/all_resnet34_seeds/resnet34_all_seed_101/resnet34_all_seed_101.pth",
        final_stage_c=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument(
        "--val-split-csv",
        default="splits/archaeology_5class_research_split_v1/val_split.csv",
    )
    parser.add_argument("--output", default="assets/readme/model_evolution_examples.png")
    parser.add_argument("--task", default="archaeology_5class")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--morphology-kernel-size", type=int, default=3)
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--sample-ids", nargs="+", default=list(DEFAULT_SAMPLE_IDS))
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    sample_ids = [sample_id_to_name(Path(str(sample_id)).stem) for sample_id in args.sample_ids]
    print(f"[model-evolution] loading samples: {', '.join(sample_ids)}", flush=True)
    val_df = load_selected_validation_rows(args.val_split_csv, sample_ids)
    dataset = ArchaeologySegmentationDataset(
        val_df,
        args.data_root,
        image_size=args.image_size,
        task=args.task,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model-evolution] device={device}", flush=True)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    gt_masks = batch["mask"].cpu().numpy()
    metadata = val_df.set_index("sample_id", drop=False)

    predictions: dict[str, np.ndarray] = {}
    for spec in MODEL_SPECS:
        print(f"[model-evolution] loading model: {spec.name}", flush=True)
        model = load_model(spec, args, device)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        pred = probs.argmax(axis=1).astype(np.int64)
        if spec.final_stage_c:
            max_prob = probs.max(axis=1)
            pred = apply_stage_c(pred, max_prob, args)
        predictions[spec.name] = pred

    metric_args = SimpleNamespace(
        min_component_area=args.min_component_area,
        object_iou_threshold=args.object_iou_threshold,
    )
    rows = []
    for index, sample_id in enumerate(sample_ids):
        final_pred = predictions["Final Stage C"][index]
        gt = gt_masks[index]
        metrics = sample_metrics(final_pred, gt, sample_id, metric_args)
        rows.append(
            {
                "sample_id": sample_id,
                "region": str(metadata.loc[sample_id, "region"]),
                "modality": str(metadata.loc[sample_id, "modality"]),
                "final_iou": mean_matched_iou(final_pred, gt, metric_args),
                "final_object_f1": float(metrics["object_f1"]),
            }
        )

    output = Path(args.output)
    print(f"[model-evolution] saving figure: {output}", flush=True)
    save_figure(images.cpu().numpy()[:, 0], gt_masks, predictions, rows, output, args)
    output.with_suffix(".csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    print("[model-evolution] done", flush=True)


def load_selected_validation_rows(val_split_csv: str, sample_ids: list[str]) -> pd.DataFrame:
    val_df = pd.read_csv(val_split_csv, dtype={"sample_id": str})
    val_df["sample_id"] = val_df["sample_id"].map(sample_id_to_name)
    selected = val_df[val_df["sample_id"].isin(sample_ids)].copy()
    missing = sorted(set(sample_ids) - set(selected["sample_id"]))
    if missing:
        raise ValueError(f"Sample IDs are not in validation split: {missing}")
    selected = selected.set_index("sample_id").loc[sample_ids].reset_index()
    return selected


def load_model(spec: ModelSpec, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(spec.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {spec.name}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        encoder_name=spec.encoder,
        encoder_weights=None,
        in_channels=1,
        classes=num_classes_for_task(args.task),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def apply_stage_c(predictions: np.ndarray, max_probs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    output = []
    for pred, max_prob in zip(predictions, max_probs, strict=True):
        final = pred.copy()
        final[max_prob < args.confidence_threshold] = 0
        final = postprocess_prediction(
            final,
            min_component_area=args.min_component_area,
            use_postprocessing=True,
            use_morphology_opening=True,
            morphology_kernel_size=args.morphology_kernel_size,
        )
        output.append(final)
    return np.stack(output, axis=0)


def save_figure(
    images: np.ndarray,
    gt_masks: np.ndarray,
    predictions: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    output: Path,
    args: argparse.Namespace,
) -> None:
    column_titles = [
        "Input raster",
        "GT mask",
        *[spec.name for spec in MODEL_SPECS],
        "Final overlay",
    ]
    figure, axes = plt.subplots(len(rows), len(column_titles), figsize=(24, 3.1 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, row in enumerate(rows):
        image = stretch(images[row_index])
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(
            f"{row['sample_id']} | {row['region']} | {row['modality']}\n"
            f"Final IoU={row['final_iou']:.3f} | object F1={row['final_object_f1']:.3f}",
            fontsize=9,
        )
        axes[row_index, 1].imshow(colorize_mask(gt_masks[row_index], args.task))
        axes[row_index, 1].set_title(column_titles[1], fontsize=10)
        for model_index, spec in enumerate(MODEL_SPECS, start=2):
            axes[row_index, model_index].imshow(colorize_mask(predictions[spec.name][row_index], args.task))
            axes[row_index, model_index].set_title(spec.name, fontsize=10)
        overlay_axis = axes[row_index, -1]
        overlay_axis.imshow(image, cmap="gray")
        overlay_axis.imshow(mask_overlay(predictions["Final Stage C"][row_index], args.task))
        overlay_axis.set_title("Final overlay", fontsize=10)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle("Model evolution on the same validation samples", fontsize=17, y=0.996)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.08, hspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
