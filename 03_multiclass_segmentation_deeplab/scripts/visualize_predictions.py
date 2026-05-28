"""Save DeepLab prediction examples."""

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

import torch
from torch.utils.data import DataLoader

from arch_datasets.archaeology_dataset import ArchaeologySegmentationDataset, load_metadata, num_classes_for_task
from models.deeplab import build_model
from utils.splits import make_split, parse_regions
from utils.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    """Parse visualization CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="prediction_examples.png")
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
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--use-postprocessing", action="store_true")
    parser.add_argument("--min-component-area", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Run visualization."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.update({key: value for key, value in vars(args).items() if value is not None})
    config.setdefault("threshold", 0.5)
    config.setdefault("batch_size", 8)
    config.setdefault("val_fraction", 0.2)

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
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=num_classes_for_task(config["task"]),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    save_prediction_grid(
        model,
        loader,
        device,
        args.output,
        task=config["task"],
        max_samples=int(args.max_samples),
        threshold=float(config["threshold"]),
        use_postprocessing=bool(args.use_postprocessing),
        min_component_area=int(args.min_component_area),
    )
    print(f"Saved predictions to {args.output}")


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
