"""Confidence and postprocessing sweep for collected 5-class checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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

from arch_datasets.archaeology_dataset import ArchaeologySegmentationDataset, filter_multiclass_metadata, load_metadata, num_classes_for_task
from models.deeplab import build_model
from utils.metrics import confusion_matrix, multiclass_metrics_from_confusion, to_jsonable
from utils.polygon_metrics import competition_like_f1, masks_to_geojson_features
from utils.polygon_postprocessing import postprocess_prediction
from utils.splits import make_split, parse_regions

TASK = "archaeology_5class"
DEFAULT_VAL_REGIONS = "007_ЮШКОВО,008_СЕЛЯНЕ,025_ШУМГОРА,033_МИЛОВИДОВО_0.1км"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
CLASS_NAMES = {
    0: "background",
    1: "kurgany_tselye",
    2: "kurgany_povrezhdennye",
    3: "gorodishcha",
    4: "fortifikatsii",
    5: "arkhitektury",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models-root", default="runs/collect_models")
    parser.add_argument("--eval-root", default="runs/collect_models_eval")
    parser.add_argument("--task", default=TASK, choices=["archaeology_5class", "all_classes"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="custom_regions")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions", default=DEFAULT_VAL_REGIONS)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-split-csv")
    parser.add_argument("--val-split-csv")
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--polygon-min-area", type=int, default=8)
    parser.add_argument("--confidence-thresholds", default="0.00,0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--min-component-areas", default="8,16,32,64,128,256")
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
    eval_root = resolve_path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    models = discover_models(resolve_path(args.models_root))
    meta = prepare_metadata(args)
    dataset_cache: dict[str, tuple[ArchaeologySegmentationDataset, DataLoader]] = {}
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    thresholds = parse_float_list(args.confidence_thresholds)
    min_areas = parse_int_list(args.min_component_areas)
    summary_updates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    total_configs = len(thresholds) * len(min_areas) * 2
    print("[sweep] Starting postprocessing sweep")
    print(f"[sweep] Found checkpoints: {len(models)}")
    print(f"[sweep] Number of configs per eval: {total_configs}")
    for spec in models:
        evals = [(default_eval_tag(spec["group"]), default_modalities(spec["group"]))]
        if args.eval_all_models_on_li_too and spec["group"] == "all":
            evals.append(("li", ["Li"]))
        for eval_tag, modalities in evals:
            try:
                update = sweep_one(spec, eval_tag, modalities, meta, args, eval_root, device, dataset_cache, thresholds, min_areas)
                summary_updates.append(update)
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                skipped.append({"model_path": str(spec["path"]), "group": spec["group"], "eval_tag": eval_tag, "reason": reason})
                print(f"[SKIP] {spec['path']}: {reason}")

    update_summary(eval_root / "collected_models_summary.csv", summary_updates)
    pd.DataFrame(skipped).to_csv(eval_root / "postprocess_sweep_skipped.csv", index=False)
    print(f"Updated summary: {eval_root / 'collected_models_summary.csv'}")


def sweep_one(
    spec: dict[str, Any],
    eval_tag: str,
    modalities: list[str] | None,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    eval_root: Path,
    device: torch.device,
    dataset_cache: dict[str, tuple[ArchaeologySegmentationDataset, DataLoader]],
    thresholds: list[float],
    min_areas: list[int],
) -> dict[str, Any]:
    run_name = safe_name(f"{spec['group']}__{spec['path'].stem}__eval_{eval_tag}")
    run_dir = eval_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(spec["path"], map_location="cpu")
    model, encoder = load_model_from_checkpoint(checkpoint, spec["path"], args.task, device)
    dataset, loader = get_dataset_loader(meta, modalities, args, dataset_cache)
    print(f"[sweep] Model: {spec['path']}")
    print(f"[sweep] Eval modalities: {modalities or 'all'}")
    print(f"[sweep] Validation samples: {len(dataset)}")
    print("[sweep] Collecting softmax probabilities...")
    probabilities, targets, sample_ids = collect_probabilities(model, loader, device)
    print(f"[sweep] Probability maps cached: {len(probabilities)}")
    rows = []
    gt_objects = None
    best_seen = -float("inf")
    for threshold in thresholds:
        for min_area in min_areas:
            for opening in (False, True):
                print(f"[sweep] Current config: confidence={threshold}, min_area={min_area}, opening={opening}")
                preds = predictions_from_probabilities(probabilities, threshold, min_area, opening)
                row = evaluate_predictions(preds, targets, sample_ids, args, threshold, min_area, opening)
                rows.append(row)
                gt_objects = row["num_gt_objects"]
                print(
                    f"[sweep] weighted_f1={row['weighted_competition_f1']:.4f}, "
                    f"object_f1={row['object_f1']:.4f}, precision={row['object_precision']:.4f}, recall={row['object_recall']:.4f}"
                )
                if row["weighted_competition_f1"] > best_seen:
                    best_seen = row["weighted_competition_f1"]
                    print("[sweep] New best config found")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "postprocess_sweep.csv", index=False)
    best = df.sort_values("weighted_competition_f1", ascending=False).iloc[0].to_dict()
    payload = {
        "model_path": str(spec["path"]),
        "group": spec["group"],
        "eval_modalities": ",".join(modalities) if modalities else "all",
        "encoder": encoder,
        "best_config": best,
    }
    (run_dir / "postprocess_sweep.json").write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    plot_sweep(df, run_dir / "postprocess_sweep.png")
    print(f"[sweep] Saved sweep results: {run_dir}")
    print(f"[SWEEP] {run_name}: best weighted_f1={best['weighted_competition_f1']:.4f}")
    return {
        "model_name": run_name,
        "best_postprocess_weighted_f1": best.get("weighted_competition_f1"),
        "best_confidence_threshold": best.get("confidence_threshold"),
        "best_min_component_area": best.get("min_component_area"),
        "best_morphology_opening": best.get("morphology_opening"),
    }


@torch.no_grad()
def collect_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        logits = model(batch["image"].to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probabilities.extend([item for item in probs])
        targets.extend([item.numpy() for item in batch["mask"]])
        sample_ids.extend([str(item) for item in batch["sample_id"]])
        if batch_index % 10 == 0 or batch_index == total_batches:
            print(f"[sweep] Inference batch {batch_index}/{total_batches}")
    return probabilities, targets, sample_ids


def predictions_from_probabilities(
    probabilities: list[np.ndarray],
    confidence_threshold: float,
    min_component_area: int,
    morphology_opening: bool,
) -> list[np.ndarray]:
    predictions = []
    for probs in probabilities:
        max_prob = probs.max(axis=0)
        pred = probs.argmax(axis=0).astype(np.int64)
        pred[max_prob < confidence_threshold] = 0
        pred = postprocess_prediction(
            pred,
            min_component_area=min_component_area,
            use_postprocessing=True,
            use_morphology_opening=morphology_opening,
        )
        predictions.append(pred)
    return predictions


def evaluate_predictions(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sample_ids: list[str],
    args: argparse.Namespace,
    threshold: float,
    min_area: int,
    opening: bool,
) -> dict[str, Any]:
    matrix = torch.zeros((num_classes_for_task(args.task), num_classes_for_task(args.task)), dtype=torch.int64)
    for pred, target in zip(preds, targets, strict=False):
        matrix += confusion_matrix(torch.from_numpy(pred), torch.from_numpy(target), num_classes_for_task(args.task))
    pixel = multiclass_metrics_from_confusion(matrix, CLASS_NAMES)
    pred_geojson = masks_to_geojson_features(preds, sample_ids, min_area=float(args.polygon_min_area))
    gt_geojson = masks_to_geojson_features(targets, sample_ids, min_area=float(args.polygon_min_area))
    weighted_f1, per_class = competition_like_f1(pred_geojson, gt_geojson, iou_threshold=args.object_iou_threshold)
    tp = float(per_class["tp"].sum())
    fp = float(per_class["fp"].sum())
    fn = float(per_class["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    object_f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "confidence_threshold": threshold,
        "min_component_area": min_area,
        "morphology_opening": opening,
        "mean_fg_iou": pixel.get("mean_fg_iou"),
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": object_f1,
        "weighted_competition_f1": weighted_f1,
        "num_pred_objects": int(per_class["num_predictions"].sum()),
        "num_gt_objects": int(per_class["num_ground_truth"].sum()),
    }


def discover_models(models_root: Path) -> list[dict[str, Any]]:
    specs = []
    for group in ("li", "all"):
        group_dir = models_root / group
        if not group_dir.exists():
            continue
        for path in sorted(group_dir.rglob("*")):
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                specs.append({"path": path, "group": group})
    return specs


def prepare_metadata(args: argparse.Namespace) -> pd.DataFrame:
    meta = load_metadata(args.data_root)
    if args.use_metadata_filtering:
        meta = filter_multiclass_metadata(
            meta,
            allowed_classes=parse_csv_list(args.allowed_classes),
            max_crop_size=args.max_crop_size,
            max_objects_in_patch=args.max_objects_in_patch,
            exclude_touches_border=args.exclude_touches_border,
            min_foreground_pixels=args.min_foreground_pixels,
        )
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
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
    )
    dataset = ArchaeologySegmentationDataset(val_df, args.data_root, image_size=args.image_size, task=args.task)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cache[key] = (dataset, loader)
    return dataset, loader


def load_model_from_checkpoint(checkpoint: Any, checkpoint_path: Path, task: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state = extract_state_dict(checkpoint)
    errors = []
    for encoder in encoder_candidates(checkpoint_path.name, config):
        try:
            model = build_model(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=num_classes_for_task(task))
            load_state_dict_flexibly(model, state)
            model.to(device)
            model.eval()
            return model, encoder
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoder}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load checkpoint: " + " | ".join(errors))


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
    last_error = None
    for variant in variants:
        try:
            model.load_state_dict(variant, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    raise last_error or RuntimeError("state_dict load failed")


def update_summary(summary_path: Path, updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame([{"model_name": item["model_name"]} for item in updates])
    for update in updates:
        if "model_name" not in summary.columns:
            summary["model_name"] = ""
        if not (summary["model_name"] == update["model_name"]).any():
            summary = pd.concat([summary, pd.DataFrame([{"model_name": update["model_name"]}])], ignore_index=True)
        for key, value in update.items():
            summary.loc[summary["model_name"] == update["model_name"], key] = value
    summary.to_csv(summary_path, index=False)


def plot_sweep(df: pd.DataFrame, save_path: Path) -> None:
    best_by_threshold = df.groupby("confidence_threshold", as_index=False)["weighted_competition_f1"].max()
    best_by_area = df.groupby("min_component_area", as_index=False)["weighted_competition_f1"].max()
    opening = df.groupby("morphology_opening", as_index=False)["weighted_competition_f1"].max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(best_by_threshold["confidence_threshold"], best_by_threshold["weighted_competition_f1"], marker="o")
    axes[0].set_title("Best weighted F1 by confidence")
    axes[1].plot(best_by_area["min_component_area"], best_by_area["weighted_competition_f1"], marker="o")
    axes[1].set_title("Best weighted F1 by min area")
    axes[2].bar(opening["morphology_opening"].astype(str), opening["weighted_competition_f1"])
    axes[2].set_title("Best weighted F1 by opening")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_modalities(group: str) -> list[str] | None:
    return ["Li"] if group == "li" else None


def default_eval_tag(group: str) -> str:
    return "li" if group == "li" else "all"


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
