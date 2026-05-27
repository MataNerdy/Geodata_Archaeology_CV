"""Run threshold sweep for binary_kurgan checkpoints."""

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
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import ArchaeologySegmentationDataset, load_metadata
from models.deeplab import build_model
from utils.metrics import binary_metrics_from_confusion, confusion_matrix, to_jsonable
from utils.splits import make_split, parse_regions
from utils.visualization import plot_threshold_sweep


def parse_args() -> argparse.Namespace:
    """Parse threshold sweep CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--encoder")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--split")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run binary threshold sweep."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.update({key: value for key, value in vars(args).items() if value is not None})
    config["task"] = "binary_kurgan"
    config.setdefault("out_dir", str(Path(args.checkpoint).parent))
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
        task="binary_kurgan",
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=1,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs.append(torch.sigmoid(logits[:, 0]).cpu())
        targets.append(batch["mask"].cpu())
    prob_tensor = torch.cat(probs, dim=0)
    target_tensor = torch.cat(targets, dim=0)

    rows = []
    for threshold in parse_thresholds(args.thresholds):
        preds = (prob_tensor > threshold).long()
        matrix = confusion_matrix(preds, target_tensor, 2)
        metrics = binary_metrics_from_confusion(matrix)
        rows.append({"threshold": threshold, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    best = df.sort_values("fg_iou", ascending=False).iloc[0].to_dict()
    with (out_dir / "threshold_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable({"best": best, "config": config}), handle, indent=2, ensure_ascii=False)
    plot_threshold_sweep(rows, out_dir / "threshold_sweep.png")
    print("Top-5 thresholds by fg_iou:")
    print(df.sort_values("fg_iou", ascending=False).head(5).to_string(index=False))


def parse_thresholds(value: str) -> list[float]:
    """Parse comma-separated thresholds."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
