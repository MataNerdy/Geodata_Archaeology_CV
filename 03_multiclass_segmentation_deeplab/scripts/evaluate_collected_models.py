"""Evaluate collected 5-class DeepLab checkpoints on a common validation split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    confusion_matrix,
    confusion_matrix_to_csv_rows,
    logits_to_predictions,
    multiclass_metrics_from_confusion,
    to_jsonable,
)
from utils.polygon_metrics import competition_like_f1, masks_to_geojson_features
from utils.splits import make_split, parse_regions
from utils.visualization import plot_confusion_matrix, save_prediction_grid

TASK = "archaeology_5class"
DEFAULT_VAL_REGIONS = "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
REFERENCE_ROWS = [
    {
        "model_name": "reference_current_clean_competition_resnet50_li",
        "model_path": "reference",
        "group": "reference",
        "eval_modalities": "Li",
        "encoder": "resnet50",
        "loaded_ok": "reference",
        "mean_fg_iou": 0.1517,
        "pixel_accuracy": math.nan,
        "object_precision": math.nan,
        "object_recall": math.nan,
        "object_f1": 0.4000,
        "weighted_competition_f1": 0.3346,
        "best_postprocess_weighted_f1": math.nan,
        "best_confidence_threshold": math.nan,
        "best_min_component_area": math.nan,
        "best_morphology_opening": math.nan,
    },
    {
        "model_name": "reference_old_baseline_resnet34",
        "model_path": "reference",
        "group": "reference",
        "eval_modalities": "Li,Ae,SpOr,Or",
        "encoder": "resnet34",
        "loaded_ok": "reference",
        "mean_fg_iou": 0.1244,
        "pixel_accuracy": 0.7174,
        "object_precision": 0.9136,
        "object_recall": 0.3974,
        "object_f1": 0.5538,
        "weighted_competition_f1": 0.4488,
        "best_postprocess_weighted_f1": math.nan,
        "best_confidence_threshold": math.nan,
        "best_min_component_area": math.nan,
        "best_morphology_opening": math.nan,
    },
]


@dataclass
class ModelSpec:
    path: Path
    group: str
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models-root", default="runs/collect_models")
    parser.add_argument("--out-dir", default="runs/collect_models_eval")
    parser.add_argument("--task", default=TASK, choices=["archaeology_5class", "all_classes"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="custom_regions")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions", default=DEFAULT_VAL_REGIONS)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--eval-all-models-on-li-too", action="store_true")
    parser.add_argument("--use-metadata-filtering", action="store_true")
    parser.add_argument("--max-crop-size", type=float)
    parser.add_argument("--max-objects-in-patch", type=int)
    parser.add_argument("--allowed-classes")
    parser.add_argument("--exclude-touches-border", action="store_true")
    parser.add_argument("--min-foreground-pixels", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models(resolve_path(args.models_root))
    print(f"Found checkpoints: {len(models)}")

    meta = prepare_metadata(args)
    dataset_cache: dict[str, tuple[ArchaeologySegmentationDataset, DataLoader]] = {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    for spec in models:
        evals = [(default_eval_tag(spec.group), default_modalities(spec.group))]
        if args.eval_all_models_on_li_too and spec.group == "all":
            evals.append(("li", ["Li"]))
        for eval_tag, modalities in evals:
            row, skip = evaluate_one(spec, eval_tag, modalities, meta, args, out_dir, device, dataset_cache)
            if skip is not None:
                skipped.append(skip)
            rows.append(row)

    rows.extend(REFERENCE_ROWS)
    summary_path = out_dir / "collected_models_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_models.csv", index=False)
    write_top_models(rows, out_dir / "top_models.md")
    save_model_comparison_grid(rows, out_dir / "model_comparison_grid.png")
    print(f"Loaded OK: {sum(str(row.get('loaded_ok')) == 'True' for row in rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Summary: {summary_path}")
    print(f"Comparison grid: {out_dir / 'model_comparison_grid.png'}")


def evaluate_one(
    spec: ModelSpec,
    eval_tag: str,
    modalities: list[str] | None,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    dataset_cache: dict[str, tuple[ArchaeologySegmentationDataset, DataLoader]],
) -> tuple[dict[str, Any], dict[str, str] | None]:
    eval_modalities = ",".join(modalities) if modalities else "all"
    run_name = safe_name(f"{spec.group}__{spec.name}__eval_{eval_tag}")
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    base_row = {
        "model_name": run_name,
        "model_path": str(spec.path),
        "group": spec.group,
        "eval_modalities": eval_modalities,
        "encoder": "",
        "loaded_ok": False,
        "mean_fg_iou": math.nan,
        "pixel_accuracy": math.nan,
        "object_precision": math.nan,
        "object_recall": math.nan,
        "object_f1": math.nan,
        "weighted_competition_f1": math.nan,
        "best_postprocess_weighted_f1": math.nan,
        "best_confidence_threshold": math.nan,
        "best_min_component_area": math.nan,
        "best_morphology_opening": math.nan,
        "worst_samples": "",
    }

    try:
        checkpoint = torch.load(spec.path, map_location="cpu")
        model, encoder = load_model_from_checkpoint(checkpoint, spec.path, args.task, device)
        dataset, loader = get_dataset_loader(meta, modalities, args, dataset_cache)
        matrix, preds, targets, sample_ids = collect_predictions(model, loader, device, args.task)
        sample_metrics = sample_iou_rows(preds, targets, sample_ids)
        pd.DataFrame(sample_metrics).to_csv(run_dir / "sample_metrics.csv", index=False)
        pd.DataFrame(sample_metrics).sort_values("sample_mean_fg_iou", ascending=True).head(12).to_csv(
            run_dir / "failure_cases.csv", index=False
        )
        class_names = class_names_for_task(args.task)
        pixel_metrics = save_pixel_outputs(matrix, class_names, run_dir)
        object_metrics = save_object_outputs(preds, targets, sample_ids, args, run_dir)
        save_prediction_grid(
            model,
            loader,
            device,
            run_dir / "prediction_examples.png",
            args.task,
            max_samples=args.max_samples,
        )
        payload = {
            "model_path": str(spec.path),
            "group": spec.group,
            "eval_modalities": eval_modalities,
            "encoder": encoder,
            "pixel_metrics": pixel_metrics,
            "object_metrics": object_metrics,
        }
        (run_dir / "evaluation_summary.json").write_text(
            json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        base_row.update(
            {
                "encoder": encoder,
                "loaded_ok": True,
                "mean_fg_iou": pixel_metrics.get("mean_fg_iou"),
                "pixel_accuracy": pixel_metrics.get("pixel_accuracy"),
                "object_precision": object_metrics.get("object_precision"),
                "object_recall": object_metrics.get("object_recall"),
                "object_f1": object_metrics.get("object_f1"),
                "weighted_competition_f1": object_metrics.get("weighted_competition_f1"),
                "worst_samples": ",".join(
                    str(row["sample_id"]) for row in sorted(sample_metrics, key=lambda item: item["sample_mean_fg_iou"])[:2]
                ),
            }
        )
        print(f"[OK] {run_name}: encoder={encoder} weighted_f1={base_row['weighted_competition_f1']}")
        return base_row, None
    except Exception as exc:  # noqa: BLE001 - keep batch evaluation going
        reason = f"{type(exc).__name__}: {exc}"
        base_row["loaded_ok"] = False
        print(f"[SKIP] {spec.path}: {reason}")
        return base_row, {"model_path": str(spec.path), "group": spec.group, "eval_modalities": eval_modalities, "reason": reason}


def discover_models(models_root: Path) -> list[ModelSpec]:
    specs = []
    for group in ("li", "all"):
        group_dir = models_root / group
        if not group_dir.exists():
            continue
        for path in sorted(group_dir.rglob("*")):
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                specs.append(ModelSpec(path=path, group=group, name=path.stem))
    return specs


def prepare_metadata(args: argparse.Namespace) -> pd.DataFrame:
    meta = load_metadata(args.data_root)
    if args.use_metadata_filtering:
        before_count = len(meta)
        meta = filter_multiclass_metadata(
            meta,
            allowed_classes=parse_csv_list(args.allowed_classes),
            max_crop_size=args.max_crop_size,
            max_objects_in_patch=args.max_objects_in_patch,
            exclude_touches_border=args.exclude_touches_border,
            min_foreground_pixels=args.min_foreground_pixels,
        )
        print(f"Metadata filtering: {before_count} -> {len(meta)} samples")
    return meta


def get_dataset_loader(
    meta: pd.DataFrame,
    modalities: list[str] | None,
    args: argparse.Namespace,
    cache: dict[str, tuple[ArchaeologySegmentationDataset, DataLoader]],
) -> tuple[ArchaeologySegmentationDataset, DataLoader]:
    key = ",".join(modalities) if modalities else "all"
    if key in cache:
        return cache[key]
    _, val_df = make_split(
        meta,
        split=args.split,
        val_region=args.val_region,
        val_regions=parse_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=modalities,
    )
    dataset = ArchaeologySegmentationDataset(
        val_df,
        args.data_root,
        image_size=args.image_size,
        task=args.task,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cache[key] = (dataset, loader)
    print(f"Validation samples ({key}): {len(dataset)}")
    return dataset, loader


def load_model_from_checkpoint(
    checkpoint: Any,
    checkpoint_path: Path,
    task: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    candidates = encoder_candidates(checkpoint_path.name, config)
    state = extract_state_dict(checkpoint)
    errors = []
    for encoder in candidates:
        try:
            model = build_model(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=1,
                classes=num_classes_for_task(task),
            )
            load_state_dict_flexibly(model, state)
            model.to(device)
            model.eval()
            return model, encoder
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoder}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load checkpoint with candidate encoders: " + " | ".join(errors))


def encoder_candidates(filename: str, config: dict[str, Any]) -> list[str]:
    candidates = []
    for value in (config.get("encoder"), config.get("encoder_name")):
        if value:
            candidates.append(str(value))
    lowered = filename.lower()
    for encoder in ("resnet34", "resnet50", "efficientnet-b0"):
        if encoder.replace("-", "") in lowered.replace("-", ""):
            candidates.append(encoder)
    candidates.extend(["resnet34", "resnet50"])
    return list(dict.fromkeys(candidates))


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable state dict")


def load_state_dict_flexibly(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    variants = [state]
    if not all(key.startswith("model.") for key in state):
        variants.append({f"model.{key}": value for key, value in state.items()})
    if any(key.startswith("module.") for key in state):
        variants.append({key.removeprefix("module."): value for key, value in state.items()})
    last_error = None
    for variant in variants:
        try:
            model.load_state_dict(variant, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    raise last_error or RuntimeError("state_dict load failed")


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> tuple[torch.Tensor, list[np.ndarray], list[np.ndarray], list[str]]:
    num_classes = num_classes_for_task(task)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"]
        preds = logits_to_predictions(model(images), task).cpu()
        matrix += confusion_matrix(preds, targets, num_classes)
        preds_all.extend([pred.numpy() for pred in preds])
        targets_all.extend([target.numpy() for target in targets])
        sample_ids.extend([str(item) for item in batch["sample_id"]])
    return matrix, preds_all, targets_all, sample_ids



def sample_iou_rows(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    """Compute coarse per-sample foreground IoU for failure-case discovery."""

    rows = []
    for sample_id, pred, target in zip(sample_ids, preds, targets, strict=False):
        class_ious = []
        for class_id in range(1, 6):
            pred_fg = pred == class_id
            target_fg = target == class_id
            union = np.logical_or(pred_fg, target_fg).sum()
            if union == 0:
                continue
            intersection = np.logical_and(pred_fg, target_fg).sum()
            class_ious.append(float(intersection / union))
        rows.append(
            {
                "sample_id": sample_id,
                "sample_mean_fg_iou": float(np.mean(class_ious)) if class_ious else math.nan,
                "gt_foreground_pixels": int((target > 0).sum()),
                "pred_foreground_pixels": int((pred > 0).sum()),
            }
        )
    return rows

def save_pixel_outputs(matrix: torch.Tensor, class_names: dict[int, str], out_dir: Path) -> dict[str, float]:
    metrics = multiclass_metrics_from_confusion(matrix, class_names)
    payload = {"metrics": metrics, "confusion_matrix": matrix.tolist()}
    (out_dir / "evaluation.json").write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evaluation_pixel.json").write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_pixel.csv", index=False)
    pd.DataFrame(confusion_matrix_to_csv_rows(matrix, class_names)).to_csv(out_dir / "confusion_matrix.csv", index=False)
    plot_confusion_matrix(matrix, class_names, out_dir / "confusion_matrix.png")
    return metrics


def save_object_outputs(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sample_ids: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, float]:
    pred_geojson = masks_to_geojson_features(preds, sample_ids, min_area=float(args.min_area))
    gt_geojson = masks_to_geojson_features(targets, sample_ids, min_area=float(args.min_area))
    weighted_f1, rows = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=float(args.object_iou_threshold),
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
        "object_iou_threshold": float(args.object_iou_threshold),
        "min_area": float(args.min_area),
    }
    payload = {"metrics": metrics, "per_class": rows.to_dict(orient="records")}
    (out_dir / "evaluation_object.json").write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "competition_metric.json").write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_object.csv", index=False)
    rows.to_csv(out_dir / "competition_metric.csv", index=False)
    return metrics


def write_top_models(rows: list[dict[str, Any]], save_path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df["loaded_ok"].astype(str) == "True"].copy()
    lines = ["# Collected Models: Top Results", ""]
    sections = [
        ("Top by weighted_competition_f1", "weighted_competition_f1", False),
        ("Top by mean_fg_iou", "mean_fg_iou", False),
        ("Top by object_precision", "object_precision", False),
        ("Top by object_recall", "object_recall", False),
    ]
    for title, column, ascending in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(markdown_table(df.sort_values(column, ascending=ascending).head(5), ["model_name", "group", "eval_modalities", "encoder", column]))
        lines.append("")
    for group, title in (("li", "Best Li-only model"), ("all", "Best all-modalities model")):
        subset = df[df["group"] == group].sort_values("weighted_competition_f1", ascending=False).head(1)
        lines.extend([f"## {title}", ""])
        lines.extend(markdown_table(subset, ["model_name", "encoder", "mean_fg_iou", "object_f1", "weighted_competition_f1"]))
        lines.append("")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_model_comparison_grid(rows: list[dict[str, Any]], save_path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df["loaded_ok"].astype(str) == "True"].copy()
    if df.empty:
        return
    top_weighted = df.sort_values("weighted_competition_f1", ascending=False).head(3)
    top_pixel = df.sort_values("mean_fg_iou", ascending=False).head(3)
    best_row = top_weighted.iloc[0] if not top_weighted.empty else None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, subset, column, title in [
        (axes[0], top_weighted, "weighted_competition_f1", "Top weighted F1"),
        (axes[1], top_pixel, "mean_fg_iou", "Top pixel mean fg IoU"),
    ]:
        labels = [short_label(name) for name in subset["model_name"]]
        axis.barh(labels, subset[column].astype(float))
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
    axes[2].axis("off")
    if best_row is not None:
        axes[2].set_title("Failure cases from best weighted model")
        axes[2].text(
            0.0,
            0.9,
            f"Model:\n{best_row['model_name']}\n\nWorst sample ids:\n{best_row.get('worst_samples', '')}",
            va="top",
            fontsize=10,
        )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No models."]
    shown = df[columns].copy()
    for column in shown.columns:
        shown[column] = shown[column].map(format_value)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def default_modalities(group: str) -> list[str] | None:
    return ["Li"] if group == "li" else None


def default_eval_tag(group: str) -> str:
    return "li" if group == "li" else "all"


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def short_label(value: str) -> str:
    value = value.replace("archaeology_5class_", "")
    value = value.replace("__eval_", "\n")
    return value[:42]


if __name__ == "__main__":
    main()
