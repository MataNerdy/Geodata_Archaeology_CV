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
import torch
import yaml
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import (
    ArchaeologySegmentationDataset,
    class_names_for_task,
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
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run evaluation."""

    args = parse_args()
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

    _, val_df = make_split(
        load_metadata(config["data_root"]),
        split=config["split"],
        val_region=config.get("val_region"),
        val_regions=parse_regions(config.get("val_regions")),
        val_fraction=float(config["val_fraction"]),
        modalities=normalize_modalities(config.get("modalities")),
    )
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

    num_classes = 2 if config["task"] == "binary_kurgan" else num_classes_for_task(config["task"])
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"]
        preds = logits_to_predictions(
            model(images),
            config["task"],
            threshold=float(config["threshold"]),
        ).cpu()
        matrix += confusion_matrix(preds, targets, num_classes)

    if config["task"] == "binary_kurgan":
        metrics = binary_metrics_from_confusion(matrix)
        class_names = {0: "background", 1: "any_kurgan"}
    else:
        class_names = class_names_for_task(config["task"])
        metrics = multiclass_metrics_from_confusion(matrix, class_names)
        plot_confusion_matrix(matrix, class_names, out_dir / "confusion_matrix.png")

    payload = {"metrics": metrics, "config": config, "confusion_matrix": matrix.tolist()}
    with (out_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation.csv", index=False)
    if config["task"] != "binary_kurgan":
        pd.DataFrame(per_class_rows(metrics, class_names)).to_csv(
            out_dir / "per_class_iou.csv",
            index=False,
        )
    pd.DataFrame(confusion_matrix_to_csv_rows(matrix, class_names)).to_csv(
        out_dir / "confusion_matrix.csv",
        index=False,
    )
    print(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False))


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


if __name__ == "__main__":
    main()
