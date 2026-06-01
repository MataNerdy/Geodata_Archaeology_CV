"""Create the raw diagnostic split using research_split_v1 train/val regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from arch_datasets.archaeology_dataset import load_metadata
from utils.metrics import to_jsonable


EXPECTED_VAL_REGIONS = {
    "005_ЛУБНО",
    "006_МОСКОВИТЫ",
    "008_СЕЛЯНЕ",
    "010_НОВЕНЬКОЕ",
    "018_СЕМИБРАТНЕЕ_1",
    "020_ОСЕЧКИ_2",
    "021_НОВОТИТАРОВСКАЯ",
    "022_КРАСНОСЕЛЬСКАЯ_0.5км",
    "024_УСТЬ-РЕКА",
    "025_ШУМГОРА",
    "028_САРАТОВ",
    "030_КОПАНСКОЕ",
    "032_ПРИМАКИ_1.3км",
    "037_КЧР",
    "038_ПЕТРОВСКОЕ",
    "044_ГОЧЕВО",
    "046_ТЫВА_2",
    "054_КУРМЕНТУ",
    "072_Каменка",
    "075_Сары_Булун",
    "080_Белая_Гора",
    "084_Маяцкая_крепость",
    "087_Городище",
    "088_Верхний_Карабут",
    "122_Курганы_3",
    "123_Курганы_4",
    "141_Каменные_выкладки_2",
    "152_Постройки_3",
    "153_Постройки_4",
    "154_Постройки_5",
}


def parse_args() -> argparse.Namespace:
    """Parse split-builder CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--source-split-dir",
        default="splits/archaeology_5class_research_split_v1",
        help="Frozen filtered split whose train/val region assignment must be reused.",
    )
    parser.add_argument(
        "--out-dir",
        default="splits/archaeology_5class_encoder_modality_ablation_raw_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Save a separate raw split artifact without recalculating regions."""

    args = parse_args()
    source_dir = Path(args.source_split_dir)
    out_dir = Path(args.out_dir)
    train_path = out_dir / "train_split.csv"
    val_path = out_dir / "val_split.csv"
    if (train_path.exists() or val_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Diagnostic split already exists in {out_dir}. "
            "Use --overwrite only when intentionally rematerializing the same protocol."
        )

    source_train = read_split(source_dir / "train_split.csv")
    source_val = read_split(source_dir / "val_split.csv")
    train_regions = set(source_train["region"].astype(str))
    val_regions = set(source_val["region"].astype(str))
    validate_val_regions(val_regions)
    overlap = sorted(train_regions & val_regions)
    if overlap:
        raise ValueError(f"Source split contains train/val region overlap: {overlap[:10]}")

    raw_meta = load_metadata(args.data_root)
    print(f"[split] Loaded metadata: {len(raw_meta)} rows")
    print("[split] Metadata filtering: disabled")
    print(f"[split] After filtering: {len(raw_meta)} rows")

    known_regions = train_regions | val_regions
    excluded = raw_meta[~raw_meta["region"].astype(str).isin(known_regions)].copy()
    train_df = raw_meta[raw_meta["region"].astype(str).isin(train_regions)].copy()
    val_df = raw_meta[raw_meta["region"].astype(str).isin(val_regions)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Raw diagnostic train/val split is empty. Check source region assignment.")
    assert_no_overlap(train_df, val_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[split] Train/val sizes: train={len(train_df)} val={len(val_df)}")
    log_distribution("train", train_df)
    log_distribution("val", val_df)
    print(f"[split] Excluded rows outside research_split_v1 regions: {len(excluded)}")
    print(f"[split] Saved train_split.csv: {train_path}")
    print(f"[split] Saved val_split.csv: {val_path}")
    print("[split] No test split exists. This diagnostic series remains validation-only.")

    summary = {
        "split_name": out_dir.name,
        "purpose": "diagnostic encoder/modalities ablation; not the primary benchmark",
        "protocol": "reuse research_split_v1 train/val regions on raw metadata without metadata filtering",
        "source_split_dir": str(source_dir),
        "data_root": str(args.data_root),
        "metadata_filtering": False,
        "raw_samples": len(raw_meta),
        "included_samples": len(train_df) + len(val_df),
        "excluded_rows_outside_source_regions": len(excluded),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_region_count": int(train_df["region"].nunique()),
        "val_region_count": int(val_df["region"].nunique()),
        "train_regions": sorted(train_regions),
        "val_regions": sorted(val_regions),
        "train_class_counts": counts(train_df, "class_name"),
        "val_class_counts": counts(val_df, "class_name"),
        "train_modality_counts": counts(train_df, "modality"),
        "val_modality_counts": counts(val_df, "modality"),
    }
    (out_dir / "split_config.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_stats(summary, out_dir / "split_stats.md")
    print(f"[split] Saved split_config.json: {out_dir / 'split_config.json'}")
    print(f"[split] Saved split_stats.md: {out_dir / 'split_stats.md'}")


def read_split(path: Path) -> pd.DataFrame:
    """Read an existing frozen split."""

    if not path.exists():
        raise FileNotFoundError(f"Frozen split file not found: {path}")
    return pd.read_csv(path, dtype={"sample_id": str})


def assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Raise if materialized split has overlapping samples."""

    overlap = set(train_df["sample_id"].astype(str)) & set(val_df["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"Raw diagnostic split has overlapping sample ids: {sorted(overlap)[:10]}")


def validate_val_regions(val_regions: set[str]) -> None:
    """Require the frozen research_split_v1 validation region assignment."""

    if val_regions == EXPECTED_VAL_REGIONS:
        return
    missing = sorted(EXPECTED_VAL_REGIONS - val_regions)
    unexpected = sorted(val_regions - EXPECTED_VAL_REGIONS)
    raise ValueError(
        "Source validation regions do not match research_split_v1. "
        f"Missing={missing}; unexpected={unexpected}"
    )


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return JSON-friendly value counts."""

    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def log_distribution(name: str, frame: pd.DataFrame) -> None:
    """Print split diagnostics."""

    print(f"[split] {name} classes:\n{frame['class_name'].value_counts().to_string()}")
    print(f"[split] {name} modalities:\n{frame['modality'].value_counts().to_string()}")
    print(f"[split] {name} regions: {frame['region'].nunique()}")


def write_stats(summary: dict[str, object], path: Path) -> None:
    """Write human-readable diagnostic split statistics."""

    lines = [
        "# archaeology_5class_encoder_modality_ablation_raw_v1",
        "",
        "Diagnostic-only frozen split. This does not replace `archaeology_5class_research_split_v1`.",
        "",
        "## Protocol",
        "",
        "- train/val regions: reused from `archaeology_5class_research_split_v1`",
        "- metadata filtering: disabled",
        "- region search: not rerun",
        "- test split: not available",
        "",
        "## Counts",
        "",
        f"- raw metadata rows: {summary['raw_samples']}",
        f"- included rows: {summary['included_samples']}",
        f"- excluded rows outside source regions: {summary['excluded_rows_outside_source_regions']}",
        f"- train samples: {summary['train_samples']}",
        f"- val samples: {summary['val_samples']}",
        f"- train regions: {summary['train_region_count']}",
        f"- val regions: {summary['val_region_count']}",
    ]
    for title, key in (
        ("Train Class Counts", "train_class_counts"),
        ("Val Class Counts", "val_class_counts"),
        ("Train Modality Counts", "train_modality_counts"),
        ("Val Modality Counts", "val_modality_counts"),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {name}: {value}" for name, value in summary[key].items())
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This is a raw-data diagnostic comparison. It must not be reported as the primary benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
