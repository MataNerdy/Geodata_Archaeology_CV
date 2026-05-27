"""Build patch-level GeoJSON-like prediction and GT files for competition metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.archaeology_dataset import ArchaeologySegmentationDataset, load_metadata, num_classes_for_task
from models.deeplab import build_model
from utils.metrics import logits_to_predictions
from utils.polygon_metrics import masks_to_geojson_features
from utils.splits import make_split, parse_regions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--task", choices=["all_classes"], default="all_classes")
    parser.add_argument("--encoder")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--split")
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--min-area", type=float, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Build prediction and GT GeoJSON files."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.update({key: value for key, value in vars(args).items() if value is not None})
    config["task"] = "all_classes"
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
        task="all_classes",
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(
        encoder_name=config["encoder"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=1,
        classes=num_classes_for_task("all_classes"),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    pred_masks = []
    gt_masks = []
    sample_ids = []
    for batch in loader:
        preds = logits_to_predictions(model(batch["image"].to(device)), "all_classes").cpu().numpy()
        pred_masks.extend(list(preds))
        gt_masks.extend(list(batch["mask"].numpy()))
        sample_ids.extend([str(item) for item in batch["sample_id"]])

    pred_geojson = masks_to_geojson_features(pred_masks, sample_ids, min_area=float(args.min_area))
    gt_geojson = masks_to_geojson_features(gt_masks, sample_ids, min_area=float(args.min_area))
    (out_dir / "predictions_geojson.json").write_text(
        json.dumps(pred_geojson, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "ground_truth_geojson.json").write_text(
        json.dumps(gt_geojson, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved GeoJSON files to {out_dir}")


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

