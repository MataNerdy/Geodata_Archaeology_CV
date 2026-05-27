"""Evaluate a trained kurgan segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.kurgan_dataset import KurganSegmentationDataset, load_metadata, make_experiment_split
from models.unet_small import build_model
from scripts.train import (
    build_criterion,
    evaluate_loader,
    get_device,
    normalize_modalities,
    parse_int_list,
    parse_val_regions,
    validate_task_args,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--binary-positive-classes", default="1,2")
    parser.add_argument(
        "--split",
        choices=["region", "custom_regions", "random"],
        default="region",
    )
    parser.add_argument("--val-region")
    parser.add_argument("--val-regions")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--class-weights")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run checkpoint evaluation."""

    args = parse_args()
    args.modalities = normalize_modalities(args.modalities)
    args.binary_positive_classes = parse_int_list(args.binary_positive_classes)
    validate_task_args(args)
    device = get_device()
    _, val_df = make_experiment_split(
        load_metadata(args.data_root),
        split=args.split,
        val_region=args.val_region,
        val_regions=parse_val_regions(args.val_regions),
        val_fraction=args.val_fraction,
        modalities=args.modalities,
    )

    dataset = KurganSegmentationDataset(
        val_df,
        args.data_root,
        args.image_size,
        task=args.task,
        binary_positive_classes=args.binary_positive_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_outputs = 1 if args.task == "binary" else 3
    model = build_model("unet_small", in_channels=1, num_classes=num_outputs).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    criterion = build_criterion(args).to(device)

    metrics = evaluate_loader(
        model,
        loader,
        criterion,
        device,
        split_name="val",
        task=args.task,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(out_dir / "evaluation.csv", index=False)
        with (out_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
