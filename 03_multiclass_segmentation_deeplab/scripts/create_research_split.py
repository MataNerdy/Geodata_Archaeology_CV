"""Create one frozen archaeology research split from filtered metadata.

This script intentionally runs the expensive region holdout search once and
saves train_split.csv / val_split.csv. Training and evaluation should then use
--split frozen instead of recomputing candidate validation regions.
"""

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

for _package_name in ("arch_datasets", "utils"):
    _force_local_package(_package_name)

from arch_datasets.archaeology_dataset import filter_multiclass_metadata, load_metadata
from utils.metrics import to_jsonable
from utils.splits import make_region_holdout_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/segmentation_dataset")
    parser.add_argument("--out-dir", default="splits/archaeology_5class_research_split_v1")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--group-col", default="region")
    parser.add_argument("--strat-cols", default="class_name,modality")
    parser.add_argument("--min-val-per-class", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--use-metadata-filtering", action="store_true", default=True)
    parser.add_argument("--no-metadata-filtering", action="store_false", dest="use_metadata_filtering")
    parser.add_argument("--max-crop-size", type=float, default=2048)
    parser.add_argument("--max-objects-in-patch", type=int, default=40)
    parser.add_argument(
        "--allowed-classes",
        default="kurgany_tselye,kurgany_povrezhdennye,gorodishcha,fortifikatsii,arkhitektury",
    )
    parser.add_argument("--exclude-touches-border", action="store_true", default=True)
    parser.add_argument("--include-touches-border", action="store_false", dest="exclude_touches_border")
    parser.add_argument("--min-foreground-pixels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    train_path = out_dir / "train_split.csv"
    val_path = out_dir / "val_split.csv"
    if (train_path.exists() or val_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Frozen split already exists in {out_dir}. Use --overwrite only when intentionally creating a new split version."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_meta = load_metadata(args.data_root)
    print(f"[split] Loaded metadata: {len(raw_meta)} rows")
    meta = raw_meta.copy()
    if args.use_metadata_filtering:
        meta = filter_multiclass_metadata(
            meta,
            allowed_classes=parse_list(args.allowed_classes),
            max_crop_size=args.max_crop_size,
            max_objects_in_patch=args.max_objects_in_patch,
            exclude_touches_border=args.exclude_touches_border,
            min_foreground_pixels=args.min_foreground_pixels,
        )
    print(f"[split] After filtering: {len(meta)} rows")

    strat_cols = tuple(parse_list(args.strat_cols))
    train_df, val_df, val_regions, score = make_region_holdout_split(
        meta,
        val_frac=args.val_frac,
        group_col=args.group_col,
        strat_cols=strat_cols,
        min_val_per_class=args.min_val_per_class,
        random_state=args.random_state,
        n_trials=args.n_trials,
    )

    print(f"[split] Train/val sizes: train={len(train_df)} val={len(val_df)}")
    print("[split] Class distribution per split")
    print("[split] train classes:\n" + train_df["class_name"].value_counts().to_string())
    print("[split] val classes:\n" + val_df["class_name"].value_counts().to_string())
    print("[split] Modality distribution per split")
    print("[split] train modalities:\n" + train_df["modality"].value_counts().to_string())
    print("[split] val modalities:\n" + val_df["modality"].value_counts().to_string())
    print("[split] Region count per split")
    print(f"[split] train regions={train_df['region'].nunique()} val regions={val_df['region'].nunique()}")

    train_df.to_csv(train_path, index=False)
    print(f"[split] Saved train_split.csv: {train_path}")
    val_df.to_csv(val_path, index=False)
    print(f"[split] Saved val_split.csv: {val_path}")
    print("[split] No test split was provided or created. TODO: add test_split.csv only when a real held-out test protocol exists.")

    summary = {
        "split_name": out_dir.name,
        "data_root": str(args.data_root),
        "raw_samples": len(raw_meta),
        "filtered_samples": len(meta),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "score": score,
        "val_regions": val_regions,
        "params": vars(args),
        "train_class_counts": train_df["class_name"].value_counts().to_dict() if "class_name" in train_df else {},
        "val_class_counts": val_df["class_name"].value_counts().to_dict() if "class_name" in val_df else {},
        "train_modality_counts": train_df["modality"].value_counts().to_dict() if "modality" in train_df else {},
        "val_modality_counts": val_df["modality"].value_counts().to_dict() if "modality" in val_df else {},
        "val_region_counts": val_df["region"].value_counts().to_dict() if "region" in val_df else {},
    }
    with (out_dir / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, ensure_ascii=False)
    print(f"[split] Saved split_config.json: {out_dir / 'split_config.json'}")
    write_split_stats(summary, out_dir / "split_stats.md")
    print(f"[split] Saved split_stats.md: {out_dir / 'split_stats.md'}")

    print(f"Saved frozen split to: {out_dir}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Score: {score:.6f}")
    print("Val regions:")
    for region in val_regions:
        print(f"  - {region}")
    print("Val class counts:")
    print(val_df["class_name"].value_counts().to_string())
    print("Val modality counts:")
    print(val_df["modality"].value_counts().to_string())


def write_split_stats(summary: dict, path: Path) -> None:
    """Write human-readable split statistics."""

    lines = [
        "# archaeology_5class_research_split_v1",
        "",
        "Frozen split protocol artifact. This split is created once and reused with `--split frozen`.",
        "",
        "## Counts",
        "",
        f"- raw samples: {summary['raw_samples']}",
        f"- filtered samples: {summary['filtered_samples']}",
        f"- train samples: {summary['train_samples']}",
        f"- val samples: {summary['val_samples']}",
        f"- test samples: not available; no `test_split.csv` was created",
        f"- split score: {summary['score']}",
        "",
        "## Validation Regions",
        "",
    ]
    lines.extend(f"- {region}" for region in summary["val_regions"])
    lines.extend(["", "## Train Class Counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["train_class_counts"].items())
    lines.extend(["", "## Val Class Counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["val_class_counts"].items())
    lines.extend(["", "## Train Modality Counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["train_modality_counts"].items())
    lines.extend(["", "## Val Modality Counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["val_modality_counts"].items())
    lines.extend(["", "## Limitation", "", "No test split is available yet. Model selection and postprocessing are validation-only."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        parts = str(value).split(",")
    return [part.strip() for part in parts if part.strip()]


if __name__ == "__main__":
    main()
