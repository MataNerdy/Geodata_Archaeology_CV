"""Generate ranked README galleries for the selected archaeology pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_CURATED_SAMPLE_IDS = (
    "000546",
    "000443",
    "000065",
    "000007",
    "000066",
    "000040",
    "002109",
    "000436",
    "000045",
    "000545",
    "000180",
    "000501",
    "000527",
    "000510",
)


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
from utils.polygon_metrics import (
    CLASS_NAMES,
    competition_like_f1,
    mask_to_polygons,
    masks_to_geojson_features,
)
from utils.polygon_postprocessing import postprocess_prediction
from utils.visualization import colorize_mask, mask_overlay, stretch


@dataclass
class SampleResult:
    """Cached masks and patch-level object metrics for one validation sample."""

    sample_id: str
    region: str
    modality: str
    main_class: str
    image: np.ndarray
    gt: np.ndarray
    raw_pred: np.ndarray
    final_pred: np.ndarray
    object_f1: float
    mean_object_iou: float
    weighted_contribution: float
    raw_object_f1: float
    raw_weighted_contribution: float
    gt_objects: int
    pred_objects: int

    @property
    def delta_weighted_contribution(self) -> float:
        return self.weighted_contribution - self.raw_weighted_contribution


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="runs/different_seeds_experiments/all_resnet34_seeds/resnet34_all_seed_101/resnet34_all_seed_101.pth",
    )
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument(
        "--val-split-csv",
        default="splits/archaeology_5class_research_split_v1/val_split.csv",
    )
    parser.add_argument("--assets-root", default="assets")
    parser.add_argument("--task", default="archaeology_5class")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--morphology-kernel-size", type=int, default=3)
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Save only individual top-N best final Stage C prediction panels.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of best panels to save in --best-only mode.")
    parser.add_argument(
        "--best-output-dir",
        default=None,
        help="Optional output directory for --best-only mode.",
    )
    parser.add_argument(
        "--curated-only",
        action="store_true",
        help="Save one compact README collage for an explicit ordered list of sample IDs.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Ordered sample IDs for --curated-only mode. Defaults to the README curated sample pool.",
    )
    parser.add_argument(
        "--curated-output",
        default="assets/predictions/final_resnet34_all_seed_101.png",
        help="Output PNG for --curated-only mode.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run inference, rank samples and save README galleries."""

    args = parse_args()
    print("[readme-viz] loading predictions: preparing frozen validation dataset", flush=True)
    val_df = pd.read_csv(args.val_split_csv, dtype={"sample_id": str})
    val_df["sample_id"] = val_df["sample_id"].map(sample_id_to_name)
    curated_ids = requested_sample_ids(args)
    if curated_ids:
        selected_ids = set(curated_ids)
        val_df = val_df[val_df["sample_id"].isin(selected_ids)].copy()
        missing_ids = sorted(selected_ids - set(val_df["sample_id"]))
        if missing_ids:
            raise ValueError(f"Unknown validation sample IDs: {', '.join(missing_ids)}")
    print(f"[readme-viz] loading predictions: validation samples={len(val_df)}", flush=True)
    print("[readme-viz] loading predictions: modalities=Li,Ae,SpOr", flush=True)

    dataset = ArchaeologySegmentationDataset(
        val_df,
        args.data_root,
        image_size=args.image_size,
        task=args.task,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[readme-viz] loading predictions: device={device}", flush=True)
    model = load_model(args, device)

    print("[readme-viz] loading predictions: running checkpoint inference", flush=True)
    cached = collect_predictions(model, loader, dataset.meta, device, args)
    print(f"[readme-viz] loading predictions: cached={len(cached)}", flush=True)

    print("[readme-viz] computing metrics: patch-level polygon metrics", flush=True)
    results = []
    for index, cached_sample in enumerate(cached, start=1):
        results.append(build_sample_result(cached_sample, args))
        if index % args.progress_every == 0 or index == len(cached):
            print(f"[readme-viz] computing metrics: {index}/{len(cached)} samples", flush=True)

    if args.best_only:
        save_best_only_gallery(results, args)
        return
    if args.curated_only:
        save_curated_gallery(results, curated_ids, args)
        return

    print("[readme-viz] sorting samples: selecting top-5 best validation patches", flush=True)
    foreground = [item for item in results if item.gt_objects > 0]
    best = sorted(
        foreground,
        key=lambda item: (round(item.object_f1, 4), item.mean_object_iou, item.weighted_contribution),
        reverse=True,
    )[:5]
    print("[readme-viz] sorting samples: selecting top-5 worst validation patches", flush=True)
    worst = select_diverse_worst(foreground, limit=5)
    print("[readme-viz] sorting samples: selecting top-10 postprocessing changes", flush=True)
    before_after = sorted(
        foreground,
        key=lambda item: abs(item.delta_weighted_contribution),
        reverse=True,
    )[:10]

    assets_root = Path(args.assets_root)
    predictions_dir = assets_root / "predictions" / "top5_best"
    failures_dir = assets_root / "failures" / "top5_worst"
    postprocessing_dir = assets_root / "postprocessing_examples"
    archive_root = assets_root / "archive" / f"readme_viz_{datetime.now():%Y%m%d_%H%M%S}"
    for directory in (predictions_dir, failures_dir, postprocessing_dir):
        directory.mkdir(parents=True, exist_ok=True)
        archive_previous_gallery(directory, archive_root)

    print("[readme-viz] saving images: top-5 best validation patches", flush=True)
    save_ranked_examples(best, predictions_dir, "best", args)
    print("[readme-viz] saving images: top-5 worst validation patches", flush=True)
    save_ranked_examples(worst, failures_dir, "worst", args)
    print("[readme-viz] saving images: top-10 before/after postprocessing", flush=True)
    save_before_after_examples(before_after, postprocessing_dir, args)

    final_predictions = assets_root / "predictions" / "final_resnet34_all_seed_101.png"
    final_failures = assets_root / "failures" / "final_failure_cases.png"
    postprocessing_collage = postprocessing_dir / "top10_before_after.png"
    archive_existing_files(
        (final_predictions, final_failures, postprocessing_collage),
        archive_root / "collages",
    )
    print(f"[readme-viz] saving images: {final_predictions}", flush=True)
    save_ranked_collage(best, final_predictions, "Top validation predictions", args)
    print(f"[readme-viz] saving images: {final_failures}", flush=True)
    save_ranked_collage(worst, final_failures, "Representative failure cases", args)
    print(f"[readme-viz] saving images: {postprocessing_collage}", flush=True)
    save_before_after_collage(before_after, postprocessing_collage, args)

    print("[readme-viz] saving images: CSV manifests", flush=True)
    save_manifest(best, predictions_dir / "manifest.csv")
    save_manifest(worst, failures_dir / "manifest.csv")
    save_manifest(before_after, postprocessing_dir / "manifest.csv")
    print("[readme-viz] saving images: done", flush=True)


def requested_sample_ids(args: argparse.Namespace) -> list[str]:
    """Normalize an optional ordered curated sample list."""

    if not args.curated_only:
        return []
    sample_ids = args.sample_ids or DEFAULT_CURATED_SAMPLE_IDS
    return [sample_id_to_name(Path(str(sample_id)).stem) for sample_id in sample_ids]


def save_curated_gallery(
    results: list[SampleResult],
    sample_ids: list[str],
    args: argparse.Namespace,
) -> None:
    """Save a compact manually curated README collage in the requested order."""

    by_id = {sample.sample_id: sample for sample in results}
    samples = [by_id[sample_id] for sample_id in sample_ids]
    output = Path(args.curated_output)
    archive_root = Path(args.assets_root) / "archive" / f"curated_{datetime.now():%Y%m%d_%H%M%S}"
    archive_existing_files((output, output.with_suffix(".csv")), archive_root / "collages")
    print(f"[readme-viz] saving images: curated collage -> {output}", flush=True)
    save_compact_curated_collage(samples, output, args)
    save_manifest(samples, output.with_suffix(".csv"))
    print(f"[readme-viz] saving images: done ({output})", flush=True)


def save_best_only_gallery(results: list[SampleResult], args: argparse.Namespace) -> None:
    """Save only top-N individual final Stage C prediction panels."""

    if args.top_n < 1:
        raise ValueError("--top-n must be greater than zero")
    print(f"[readme-viz] sorting samples: selecting top-{args.top_n} best validation patches", flush=True)
    foreground = [item for item in results if item.gt_objects > 0]
    best = sorted(
        foreground,
        key=lambda item: (round(item.object_f1, 4), item.mean_object_iou, item.weighted_contribution),
        reverse=True,
    )[: args.top_n]
    output_dir = (
        Path(args.best_output_dir)
        if args.best_output_dir
        else Path(args.assets_root) / "predictions" / f"top{args.top_n}_best_final"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = Path(args.assets_root) / "archive" / f"best_only_{datetime.now():%Y%m%d_%H%M%S}"
    archive_previous_gallery(output_dir, archive_root)
    print(f"[readme-viz] saving images: top-{len(best)} best final Stage C patches", flush=True)
    save_ranked_examples(best, output_dir, "best", args)
    save_manifest(best, output_dir / "manifest.csv")
    print(f"[readme-viz] saving images: done ({output_dir})", flush=True)


def select_diverse_worst(samples: list[SampleResult], limit: int) -> list[SampleResult]:
    """Select low-scoring errors while avoiding a collage of near-duplicates."""

    ranked = sorted(
        samples,
        key=lambda item: (
            round(item.weighted_contribution, 4),
            round(item.object_f1, 4),
            item.mean_object_iou,
        ),
    )
    selected: list[SampleResult] = []
    used_classes: set[str] = set()
    used_regions: set[str] = set()

    for sample in ranked:
        if sample.main_class in used_classes:
            continue
        selected.append(sample)
        used_classes.add(sample.main_class)
        used_regions.add(sample.region)
        if len(selected) == limit:
            return selected

    for sample in ranked:
        if any(item.sample_id == sample.sample_id for item in selected) or sample.region in used_regions:
            continue
        selected.append(sample)
        used_regions.add(sample.region)
        if len(selected) == limit:
            return selected

    for sample in ranked:
        if not any(item.sample_id == sample.sample_id for item in selected):
            selected.append(sample)
        if len(selected) == limit:
            return selected
    return selected


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load selected final checkpoint."""

    print(f"[readme-viz] loading predictions: checkpoint={args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=1,
        classes=num_classes_for_task(args.task),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    meta: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Cache images and masks from one validation inference pass."""

    cached = []
    processed = 0
    next_progress = args.progress_every
    metadata = meta.set_index("sample_id", drop=False)
    for batch in loader:
        images = batch["image"].to(device)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        for batch_index, sample_id in enumerate(batch["sample_id"]):
            sample_id = str(sample_id)
            gt = batch["mask"][batch_index].numpy()
            max_prob = probs[batch_index].max(axis=0)
            raw_pred = probs[batch_index].argmax(axis=0).astype(np.int64)
            final_pred = raw_pred.copy()
            final_pred[max_prob < args.confidence_threshold] = 0
            final_pred = postprocess_prediction(
                final_pred,
                min_component_area=args.min_component_area,
                use_postprocessing=True,
                use_morphology_opening=True,
                morphology_kernel_size=args.morphology_kernel_size,
            )
            row = metadata.loc[sample_id]
            cached.append(
                {
                    "sample_id": sample_id,
                    "region": str(row["region"]),
                    "modality": str(row["modality"]),
                    "main_class": str(row.get("class_name", "unknown")),
                    "image": images[batch_index, 0].cpu().numpy(),
                    "gt": gt,
                    "raw_pred": raw_pred,
                    "final_pred": final_pred,
                }
            )
            processed += 1
            if processed >= next_progress or processed == len(meta):
                print(f"[readme-viz] loading predictions: {processed}/{len(meta)} samples", flush=True)
                next_progress += args.progress_every
    return cached


def build_sample_result(sample: dict[str, object], args: argparse.Namespace) -> SampleResult:
    """Compute patch-level raw and final polygon metrics."""

    sample_id = str(sample["sample_id"])
    raw_metrics = sample_metrics(sample["raw_pred"], sample["gt"], sample_id, args)
    final_metrics = sample_metrics(sample["final_pred"], sample["gt"], sample_id, args)
    return SampleResult(
        sample_id=sample_id,
        region=str(sample["region"]),
        modality=str(sample["modality"]),
        main_class=str(sample["main_class"]),
        image=np.asarray(sample["image"]),
        gt=np.asarray(sample["gt"]),
        raw_pred=np.asarray(sample["raw_pred"]),
        final_pred=np.asarray(sample["final_pred"]),
        object_f1=final_metrics["object_f1"],
        mean_object_iou=mean_matched_iou(sample["final_pred"], sample["gt"], args),
        weighted_contribution=final_metrics["weighted_competition_f1"],
        raw_object_f1=raw_metrics["object_f1"],
        raw_weighted_contribution=raw_metrics["weighted_competition_f1"],
        gt_objects=final_metrics["gt_objects"],
        pred_objects=final_metrics["pred_objects"],
    )


def sample_metrics(pred: np.ndarray, gt: np.ndarray, sample_id: str, args: argparse.Namespace) -> dict[str, float | int]:
    """Compute object metrics for one validation patch."""

    pred_geojson = masks_to_geojson_features([pred], [sample_id], min_area=args.min_component_area)
    gt_geojson = masks_to_geojson_features([gt], [sample_id], min_area=args.min_component_area)
    weighted_f1, rows = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=args.object_iou_threshold,
    )
    tp = float(rows["tp"].sum())
    fp = float(rows["fp"].sum())
    fn = float(rows["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    object_f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "object_f1": object_f1,
        "weighted_competition_f1": weighted_f1,
        "gt_objects": int(rows["num_ground_truth"].sum()),
        "pred_objects": int(rows["num_predictions"].sum()),
    }


def mean_matched_iou(pred: np.ndarray, gt: np.ndarray, args: argparse.Namespace) -> float:
    """Return average IoU of greedy polygon matches in one patch."""

    matched_ious = []
    for class_id in CLASS_NAMES:
        pred_polygons = mask_to_polygons(pred, class_id, min_area=args.min_component_area)
        gt_polygons = mask_to_polygons(gt, class_id, min_area=args.min_component_area)
        used_gt: set[int] = set()
        for pred_polygon in pred_polygons:
            candidates = []
            for index, gt_polygon in enumerate(gt_polygons):
                if index in used_gt:
                    continue
                iou = polygon_iou(pred_polygon, gt_polygon)
                if iou > args.object_iou_threshold or centroid_hit(pred_polygon, gt_polygon):
                    candidates.append((iou, index))
            if candidates:
                best_iou, best_index = max(candidates)
                used_gt.add(best_index)
                matched_ious.append(best_iou)
    return float(np.mean(matched_ious)) if matched_ious else 0.0


def polygon_iou(pred_polygon, gt_polygon) -> float:
    """Compute polygon IoU defensively."""

    try:
        union = pred_polygon.union(gt_polygon).area
        return pred_polygon.intersection(gt_polygon).area / union if union > 0 else 0.0
    except Exception:
        return 0.0


def centroid_hit(pred_polygon, gt_polygon) -> bool:
    """Match the evaluator's centroid-hit fallback."""

    try:
        centroid = pred_polygon.centroid
        return gt_polygon.contains(centroid) or gt_polygon.boundary.distance(centroid) < 1e-10
    except Exception:
        return False


def save_ranked_examples(samples: list[SampleResult], output_dir: Path, prefix: str, args: argparse.Namespace) -> None:
    """Save individual Image | GT | Prediction | Overlay panels."""

    for rank, sample in enumerate(samples, start=1):
        output = output_dir / f"{rank:02d}_{sample.sample_id}.png"
        print(f"[readme-viz] saving images: {output}", flush=True)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        draw_prediction_row(axes, sample, args)
        fig.suptitle(sample_caption(sample), fontsize=10)
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_ranked_collage(
    samples: list[SampleResult],
    output: Path,
    title: str,
    args: argparse.Namespace,
) -> None:
    """Save README collage for selected validation patches."""

    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 3.6 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, sample in enumerate(samples):
        draw_prediction_row(axes[row_index], sample, args)
        add_row_caption(axes[row_index, 0], sample_caption(sample))
    fig.suptitle(title, fontsize=14, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_compact_curated_collage(
    samples: list[SampleResult],
    output: Path,
    args: argparse.Namespace,
) -> None:
    """Save a dense curated collage without long row captions."""

    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 3.15 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, sample in enumerate(samples):
        image = stretch(sample.image)
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(
            f"{sample.modality} | {sample.sample_id} | IoU={sample.mean_object_iou:.3f}",
            fontsize=11,
        )
        axes[row_index, 1].imshow(colorize_mask(sample.gt, args.task))
        axes[row_index, 1].set_title("GT", fontsize=11)
        axes[row_index, 2].imshow(colorize_mask(sample.final_pred, args.task))
        axes[row_index, 2].set_title("Prediction", fontsize=11)
        axes[row_index, 3].imshow(image, cmap="gray")
        axes[row_index, 3].imshow(mask_overlay(sample.final_pred, args.task))
        axes[row_index, 3].set_title(sample.region, fontsize=11)
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle("Selected final Stage C validation examples", fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.01, top=0.945, wspace=0.12, hspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_prediction_row(axes, sample: SampleResult, args: argparse.Namespace) -> None:
    """Draw Image | GT | Prediction | Overlay for one sample."""

    image = stretch(sample.image)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Input raster ({sample.modality})")
    axes[1].imshow(colorize_mask(sample.gt, args.task))
    axes[1].set_title("GT mask")
    axes[2].imshow(colorize_mask(sample.final_pred, args.task))
    axes[2].set_title("Final prediction")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(mask_overlay(sample.final_pred, args.task))
    axes[3].set_title("Prediction overlay")
    for axis in axes:
        axis.axis("off")


def save_before_after_examples(samples: list[SampleResult], output_dir: Path, args: argparse.Namespace) -> None:
    """Save individual zoomed GT | Raw | Stage C | Change map panels."""

    for rank, sample in enumerate(samples, start=1):
        output = output_dir / f"{rank:02d}_{sample.sample_id}.png"
        print(f"[readme-viz] saving images: {output}", flush=True)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        draw_before_after_row(axes, sample, args)
        fig.suptitle(before_after_caption(sample), fontsize=10)
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_before_after_collage(samples: list[SampleResult], output: Path, args: argparse.Namespace) -> None:
    """Save README collage for raw vs Stage C predictions."""

    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 3.4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, sample in enumerate(samples):
        draw_before_after_row(axes[row_index], sample, args)
        add_row_caption(axes[row_index, 0], before_after_caption(sample))
    fig.suptitle("Before and after Stage C postprocessing", fontsize=14, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_before_after_row(axes, sample: SampleResult, args: argparse.Namespace) -> None:
    """Draw a zoomed GT | Raw checkpoint | Stage C | Change map comparison."""

    crop = change_crop(sample.raw_pred, sample.final_pred)
    axes[0].imshow(colorize_mask(sample.gt[crop], args.task))
    axes[0].set_title("GT mask")
    axes[1].imshow(colorize_mask(sample.raw_pred[crop], args.task))
    axes[1].set_title("Raw checkpoint")
    axes[2].imshow(colorize_mask(sample.final_pred[crop], args.task))
    axes[2].set_title("After Stage C")
    axes[3].imshow(change_map(sample.raw_pred[crop], sample.final_pred[crop]))
    axes[3].set_title("Change map")
    for axis in axes:
        axis.axis("off")


def change_crop(raw_pred: np.ndarray, final_pred: np.ndarray, padding: int = 20) -> tuple[slice, slice]:
    """Return a padded crop around pixels affected by Stage C."""

    changed = raw_pred != final_pred
    if not changed.any():
        return slice(0, raw_pred.shape[0]), slice(0, raw_pred.shape[1])
    rows, columns = np.where(changed)
    row_start = max(0, int(rows.min()) - padding)
    row_end = min(raw_pred.shape[0], int(rows.max()) + padding + 1)
    column_start = max(0, int(columns.min()) - padding)
    column_end = min(raw_pred.shape[1], int(columns.max()) + padding + 1)
    return slice(row_start, row_end), slice(column_start, column_end)


def change_map(raw_pred: np.ndarray, final_pred: np.ndarray) -> np.ndarray:
    """Render removed, added and class-changed pixels with high contrast."""

    result = np.zeros((*raw_pred.shape, 3), dtype=np.uint8)
    unchanged_foreground = (raw_pred == final_pred) & (raw_pred != 0)
    removed = (raw_pred != 0) & (final_pred == 0)
    added = (raw_pred == 0) & (final_pred != 0)
    changed_class = (raw_pred != 0) & (final_pred != 0) & (raw_pred != final_pred)
    result[unchanged_foreground] = (80, 80, 80)
    result[removed] = (255, 120, 0)
    result[added] = (0, 220, 255)
    result[changed_class] = (255, 0, 220)
    return result


def sample_caption(sample: SampleResult) -> str:
    """Format required patch metadata for image labels."""

    return (
        f"id={sample.sample_id} | region={sample.region} | class={sample.main_class} | "
        f"object_f1={sample.object_f1:.3f} | weighted={sample.weighted_contribution:.3f}"
    )


def before_after_caption(sample: SampleResult) -> str:
    """Format metadata and score change for postprocessing panels."""

    return (
        f"id={sample.sample_id} | region={sample.region} | class={sample.main_class} | "
        f"object_f1={sample.object_f1:.3f} | weighted={sample.weighted_contribution:.3f} | "
        f"delta={sample.delta_weighted_contribution:+.3f} | "
        f"changed_px={int(np.count_nonzero(sample.raw_pred != sample.final_pred))}"
    )


def add_row_caption(axis, caption: str) -> None:
    """Add metadata below a collage row even when plot axes are hidden."""

    axis.text(
        0.0,
        -0.09,
        caption,
        transform=axis.transAxes,
        fontsize=7,
        ha="left",
        va="top",
        clip_on=False,
    )


def save_manifest(samples: list[SampleResult], output: Path) -> None:
    """Save selected samples and metrics next to gallery PNGs."""

    rows = [
        {
            "sample_id": sample.sample_id,
            "region": sample.region,
            "modality": sample.modality,
            "main_class": sample.main_class,
            "object_f1": sample.object_f1,
            "mean_object_iou": sample.mean_object_iou,
            "weighted_contribution": sample.weighted_contribution,
            "raw_object_f1": sample.raw_object_f1,
            "raw_weighted_contribution": sample.raw_weighted_contribution,
            "delta_weighted_contribution": sample.delta_weighted_contribution,
            "gt_objects": sample.gt_objects,
            "pred_objects": sample.pred_objects,
        }
        for sample in samples
    ]
    pd.DataFrame(rows).to_csv(output, index=False)


def archive_previous_gallery(directory: Path, archive_root: Path) -> None:
    """Archive the previous gallery before rebuilding ranked images."""

    files = [*sorted(directory.glob("[0-9][0-9]_*.png"))]
    manifest = directory / "manifest.csv"
    if manifest.exists():
        files.append(manifest)
    archive_existing_files(files, archive_root / directory.name)


def archive_existing_files(files: tuple[Path, ...] | list[Path], destination: Path) -> None:
    """Move prior generated files into a timestamped archive."""

    existing = [path for path in files if path.exists()]
    if not existing:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in existing:
        archived_path = destination / path.name
        print(f"[readme-viz] archiving previous image: {path} -> {archived_path}", flush=True)
        shutil.move(str(path), str(archived_path))


if __name__ == "__main__":
    main()
