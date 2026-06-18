#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_TARGET_CLASSES = ["gorodishcha", "fortifikatsii", "arkhitektury"]
CLASS_ID_TO_NAME = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}
CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}
VALID_DECISIONS = {"val", "train", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated-val YOLO dataset for selected classes.")
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--decisions", type=Path, default=Path("manual_audit/audit_decisions.csv"))
    parser.add_argument("--regions-csv", type=Path, default=Path("manual_val_region_review_other_classes/manual_val_regions.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--output-name", default="dataset_yolo_bbox_other_classes_manual_curated_val")
    parser.add_argument("--target-classes", nargs="+", default=DEFAULT_TARGET_CLASSES)
    parser.add_argument("--modalities", nargs="+", default=["Li"])
    parser.add_argument("--negative-ratio", type=float, default=None, help="Optional per-split negative/positive image cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_audit_image_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "source_image_name" if "source_image_name" in out.columns else "image_name"
    out["image_id"] = out.apply(lambda row: f"{row['split']}_{Path(str(row.get(name_col) or row['image'])).stem}", axis=1)
    return out


def read_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path).fillna("")
    decisions = decisions.drop_duplicates("image_id", keep="last")
    return decisions[["image_id", "decision", "reason", "comment", "updated_at"]].rename(
        columns={
            "decision": "manual_audit_decision",
            "reason": "manual_audit_reason",
            "comment": "manual_audit_comment",
            "updated_at": "manual_audit_updated_at",
        }
    )


def read_metadata(source_dir: Path, decisions_path: Path, target_classes: set[str], modalities: set[str] | None) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    if modalities:
        df = df[df["modality"].astype(str).isin(modalities)].copy()
    if "source_id" not in df.columns:
        source_cols = [col for col in ["region", "modality", "raster_file"] if col in df.columns]
        df["source_id"] = df[source_cols].astype(str).agg("|".join, axis=1)
    df = add_audit_image_id(df)
    decisions = read_decisions(decisions_path)
    df = df.merge(decisions, on="image_id", how="left")
    df["manual_audit_decision"] = df["manual_audit_decision"].fillna("")
    unresolved = df.drop_duplicates("image").query("manual_audit_decision == ''")
    if len(unresolved):
        print(f"WARNING: excluding {len(unresolved)} images without manual audit decision.")
    df = df[df["manual_audit_decision"].eq("keep")].copy()
    df["is_target_object"] = df["source_class_name"].isin(target_classes)
    positive_images = set(df.loc[df["is_target_object"], "image"].astype(str))
    df["is_target_positive"] = df["image"].astype(str).isin(positive_images)
    return df


def read_region_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path).fillna("")
    decisions["decision"] = decisions["decision"].astype(str).str.strip().str.lower()
    invalid = decisions[~decisions["decision"].isin(VALID_DECISIONS | {""})]
    if not invalid.empty:
        bad = invalid[["region", "decision"]].to_string(index=False)
        raise ValueError(f"Invalid decisions in {path}:\n{bad}")
    empty = decisions[decisions["decision"].eq("")]
    if not empty.empty:
        print("WARNING: empty region decisions will be excluded:")
        print(empty[["region", "images_total", "positive_images", "bbox_total"]].to_string(index=False))
    return decisions[["region", "decision", "comment"]].copy()


def resolve_dataset_path(source_dir: Path, path_value: Any, kind: str, split: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        source_dir / kind / split / path.name,
        source_dir / path,
        source_dir.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_dir / kind / split / path.name


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def apply_negative_ratio(meta: pd.DataFrame, negative_ratio: float | None, seed: int) -> pd.DataFrame:
    if negative_ratio is None:
        return meta
    keep_images = []
    image_rows = meta.drop_duplicates("image").copy()
    for split, split_images in image_rows.groupby("split", sort=True):
        positives = split_images[split_images["is_target_positive"]]
        negatives = split_images[~split_images["is_target_positive"]]
        max_negatives = int(round(len(positives) * negative_ratio))
        sampled_negatives = negatives.sample(n=min(len(negatives), max_negatives), random_state=seed) if max_negatives > 0 else negatives.iloc[0:0]
        keep_images.extend(positives["image"].astype(str).tolist())
        keep_images.extend(sampled_negatives["image"].astype(str).tolist())
    return meta[meta["image"].astype(str).isin(set(keep_images))].copy()


def split_counts(meta: pd.DataFrame) -> pd.DataFrame:
    images = meta.drop_duplicates("image").copy()
    boxes = meta[meta["is_target_object"]].copy()
    rows = []
    for split, split_images in images.groupby("split", sort=True):
        split_boxes = boxes[boxes["image"].isin(set(split_images["image"]))]
        positive = int(split_images["image"].isin(set(split_boxes["image"])).sum())
        rows.append(
            {
                "split": split,
                "images": int(len(split_images)),
                "positive_images": positive,
                "negative_images": int(len(split_images) - positive),
                "bbox": int(len(split_boxes)),
            }
        )
    return pd.DataFrame(rows)


def leakage_report(meta: pd.DataFrame) -> dict[str, list[str]]:
    images = meta.drop_duplicates("image").copy()
    train = images[images["split"].eq("train")]
    val = images[images["split"].eq("val")]
    report: dict[str, list[str]] = {}
    for col in ["region", "source_id", "raster_file"]:
        train_values = set(train[col].dropna().astype(str))
        val_values = set(val[col].dropna().astype(str))
        report[col] = sorted(train_values & val_values)
    return report


def write_dataset_yaml(out_dir: Path, target_classes: list[str]) -> None:
    names = {idx: class_name for idx, class_name in enumerate(target_classes)}
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": names}
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def materialize(source_dir: Path, meta: pd.DataFrame, out_dir: Path, target_classes: list[str], overwrite: bool) -> pd.DataFrame:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_class_id_to_target_id = {CLASS_NAME_TO_ID[class_name]: idx for idx, class_name in enumerate(target_classes)}
    rows = []
    unique_images = meta.drop_duplicates("image").copy()
    unique_images = unique_images.sort_values(["split", "region", "image_name", "image"]).reset_index(drop=True)
    for image_idx, (_, image_row) in enumerate(unique_images.iterrows(), start=1):
        split = str(image_row["split"])
        source_split = str(image_row.get("source_split") or image_row.get("old_split") or split)
        src_img = resolve_dataset_path(source_dir, image_row["image"], "images", source_split)
        src_lbl = resolve_dataset_path(source_dir, image_row["label"], "labels", source_split)
        neutral_stem = f"{image_idx:06d}"
        dst_img = out_dir / "images" / split / f"{neutral_stem}{src_img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / split / f"{neutral_stem}.txt"
        shutil.copy2(src_img, dst_img)

        group = meta[meta["image"].eq(image_row["image"])].copy()
        label_lines = []
        for cls_id, xc, yc, bw, bh in parse_label_file(src_lbl):
            if cls_id not in source_class_id_to_target_id:
                continue
            label_lines.append(f"{source_class_id_to_target_id[cls_id]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        for _, row in group.iterrows():
            record = row.to_dict()
            record["neutral_image_id"] = neutral_stem
            record["source_image_name"] = Path(str(row.get("image_name") or src_img.name)).name
            record["source_label_name"] = Path(str(row.get("label_name") or src_lbl.name)).name
            record["source_image_path"] = str(src_img.resolve())
            record["source_label_path"] = str(src_lbl.resolve())
            record["image"] = str(dst_img.resolve())
            record["label"] = str(dst_lbl.resolve())
            record["image_name"] = dst_img.name
            record["label_name"] = dst_lbl.name
            rows.append(record)
    out_meta = pd.DataFrame(rows)
    out_meta.to_csv(out_dir / "metadata.csv", index=False)
    write_dataset_yaml(out_dir, target_classes)
    return out_meta


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.fillna("")
    cols = list(formatted.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_reports(out_dir: Path, meta: pd.DataFrame, leak: dict[str, list[str]], target_classes: list[str], modalities: list[str]) -> None:
    images = meta.drop_duplicates("image").copy()
    boxes = meta[meta["is_target_object"]].copy()
    val_regions = sorted(images[images["split"].eq("val")]["region"].dropna().astype(str).unique())
    train_regions = sorted(images[images["split"].eq("train")]["region"].dropna().astype(str).unique())
    class_balance = boxes.groupby(["split", "source_class_name"]).size().reset_index(name="bbox").sort_values(["split", "source_class_name"])

    summary_lines = [
        "# Curated Validation Split Summary",
        "",
        f"Target classes: `{', '.join(target_classes)}`",
        f"Modalities: `{', '.join(modalities) if modalities else 'ALL'}`",
        "",
        "## Split Counts",
        "",
        markdown_table(split_counts(meta)),
        "",
        "## Validation Regions",
        "",
        "\n".join(f"- `{region}`" for region in val_regions) or "_No validation regions._",
        "",
        f"Train regions count: `{len(train_regions)}`",
        "",
        "## BBox By Source Class",
        "",
        markdown_table(class_balance),
        "",
    ]
    (out_dir / "split_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    leak_df = pd.DataFrame(
        [{"leakage_key": key, "overlap_count": len(values), "overlap_values": "; ".join(values[:30])} for key, values in leak.items()]
    )
    (out_dir / "leakage_report.md").write_text("# Leakage Report\n\n" + markdown_table(leak_df) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = (args.output_root / args.output_name).resolve()
    target_classes = [str(c) for c in args.target_classes]
    modalities = [str(m) for m in args.modalities] if args.modalities else []
    unknown = sorted(set(target_classes) - set(CLASS_NAME_TO_ID))
    if unknown:
        raise ValueError(f"Unknown target classes: {unknown}")
    meta = read_metadata(source_dir, args.decisions, set(target_classes), set(modalities) if modalities else None)
    meta["previous_split"] = meta["split"]
    decisions = read_region_decisions(args.regions_csv)
    meta = meta.merge(decisions, on="region", how="left")
    meta["decision"] = meta["decision"].fillna("")
    unresolved = sorted(meta.loc[meta["decision"].eq(""), "region"].dropna().astype(str).unique())
    if unresolved:
        print("WARNING: regions without explicit decision are excluded:")
        for region in unresolved:
            print(f"  - {region}")
    meta = meta[meta["decision"].isin(["train", "val"])].copy()
    meta["split"] = meta["decision"]
    meta = apply_negative_ratio(meta, args.negative_ratio, args.seed)
    if meta.empty:
        raise ValueError("No rows selected. Fill manual_val_regions.csv with train/val decisions first.")

    leak = leakage_report(meta)
    leak_failures = {key: values for key, values in leak.items() if values}
    if leak_failures:
        details = "\n".join(f"{key}: {len(values)} overlaps" for key, values in leak_failures.items())
        raise ValueError(f"Leakage detected; refusing to build dataset.\n{details}")

    out_meta = materialize(source_dir, meta, out_dir, target_classes, args.overwrite)
    write_reports(out_dir, out_meta, leak, target_classes, modalities)
    print("Dataset:", out_dir)
    print(split_counts(out_meta).to_string(index=False))
    print("split_summary.md:", out_dir / "split_summary.md")
    print("leakage_report.md:", out_dir / "leakage_report.md")


if __name__ == "__main__":
    main()
