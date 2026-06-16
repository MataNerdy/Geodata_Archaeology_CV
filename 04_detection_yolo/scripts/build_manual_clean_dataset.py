from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml


KEEP_DECISIONS = {"", "keep", "uncertain", "fix_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from manual audit decisions.")
    parser.add_argument("--source-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox"))
    parser.add_argument("--audit-index", type=Path, default=Path("manual_audit/audit_index.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("manual_audit/audit_decisions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("../datasets/dataset_yolo_bbox_manual_clean"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_decisions(path: Path) -> pd.DataFrame:
    columns = ["image_id", "decision", "reason", "comment", "updated_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns].fillna("")
    return df.drop_duplicates("image_id", keep="last")


def resolve_source_path(source_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        source_dir.parent / path,
        source_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_dir.parent / path


def write_dataset_yaml(output_dir: Path, names: dict | list | None) -> None:
    if names is None:
        names = {
            0: "kurgany_tselye",
            1: "kurgany_povrezhdennye",
            2: "gorodishcha",
            3: "fortifikatsii",
            4: "arkhitektury",
        }
    text = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(text, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def copy_selected_files(selected: pd.DataFrame, source_dir: Path, output_dir: Path) -> dict[str, int]:
    counts = {"images": 0, "labels": 0}
    unique_images = selected.drop_duplicates("image_id")
    for _, row in unique_images.iterrows():
        split = str(row["split"])
        image_src = Path(str(row["image_path"]))
        label_src = Path(str(row["label_path"]))
        if not image_src.exists():
            image_src = resolve_source_path(source_dir, row["image_path"])
        if not label_src.exists():
            label_src = resolve_source_path(source_dir, row["label_path"])

        image_dst = output_dir / "images" / split / image_src.name
        label_dst = output_dir / "labels" / split / label_src.name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)

        if not image_src.exists():
            raise FileNotFoundError(f"Missing source image: {image_src}")
        shutil.copy2(image_src, image_dst)
        counts["images"] += 1

        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.write_text("", encoding="utf-8")
        counts["labels"] += 1
    return counts


def update_metadata_paths(meta: pd.DataFrame, source_dir: Path, output_dir: Path) -> pd.DataFrame:
    updated = meta.copy()

    def image_out(row: pd.Series) -> str:
        src = resolve_source_path(source_dir, row["image"])
        return str((output_dir / "images" / str(row["split"]) / src.name).resolve())

    def label_out(row: pd.Series) -> str:
        src = resolve_source_path(source_dir, row["label"])
        return str((output_dir / "labels" / str(row["split"]) / src.name).resolve())

    updated["image"] = updated.apply(image_out, axis=1)
    updated["label"] = updated.apply(label_out, axis=1)
    return updated


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    audit_index_path = args.audit_index.resolve()
    decisions_path = args.decisions.resolve()
    output_dir = args.output_dir.resolve()

    if not (source_dir / "metadata.csv").exists():
        raise FileNotFoundError(source_dir / "metadata.csv")
    if not audit_index_path.exists():
        raise FileNotFoundError(audit_index_path)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meta = pd.read_csv(source_dir / "metadata.csv")
    audit_index = pd.read_csv(audit_index_path).fillna("")
    decisions = read_decisions(decisions_path)
    audit_base = audit_index.drop(columns=["decision", "reason", "comment"], errors="ignore")
    decisions = decisions.rename(
        columns={
            "decision": "audit_decision",
            "reason": "audit_reason",
            "comment": "audit_comment",
            "updated_at": "audit_updated_at",
        }
    )
    audit = audit_base.merge(decisions, on="image_id", how="left")
    for col in ["audit_decision", "audit_reason", "audit_comment", "audit_updated_at"]:
        audit[col] = audit[col].fillna("")
    audit["effective_decision"] = audit["audit_decision"].replace("", "keep")

    unknown = sorted(set(audit["effective_decision"]) - (KEEP_DECISIONS | {"remove_image"}))
    if unknown:
        raise ValueError(f"Unknown decisions: {unknown}")

    remove_ids = set(audit.loc[audit["effective_decision"].eq("remove_image"), "image_id"])
    fix_label = audit[audit["effective_decision"].eq("fix_label")].copy()
    uncertain = audit[audit["effective_decision"].eq("uncertain")].copy()
    kept_audit = audit[~audit["image_id"].isin(remove_ids)].copy()

    kept_keys = set(zip(kept_audit["split"].astype(str), kept_audit["image_path"].map(lambda p: Path(str(p)).name)))
    source_keys = source_meta.apply(lambda row: (str(row["split"]), Path(str(row["image"])).name), axis=1)
    filtered_meta = source_meta[source_keys.isin(kept_keys)].copy()

    counts = copy_selected_files(kept_audit, source_dir, output_dir)
    filtered_meta = update_metadata_paths(filtered_meta, source_dir, output_dir)
    filtered_meta.to_csv(output_dir / "metadata.csv", index=False)

    if (source_dir / "rename_mapping.csv").exists():
        shutil.copy2(source_dir / "rename_mapping.csv", output_dir / "rename_mapping.csv")

    names = None
    dataset_yaml = source_dir / "dataset.yaml"
    if dataset_yaml.exists():
        data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names")
    write_dataset_yaml(output_dir, names)

    audit.to_csv(output_dir / "manual_audit_decisions_resolved.csv", index=False)
    if not fix_label.empty:
        fix_label.to_csv(output_dir / "manual_audit_fix_label_warnings.csv", index=False)
    if not uncertain.empty:
        uncertain.to_csv(output_dir / "manual_audit_uncertain_kept.csv", index=False)

    image_rows = filtered_meta.drop_duplicates("image")
    box_count = int(filtered_meta["class_name"].notna().sum()) if "class_name" in filtered_meta.columns else 0
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "decisions_path": str(decisions_path),
        "source_images": int(audit_index["image_id"].nunique()),
        "removed_images": int(len(remove_ids)),
        "kept_images": int(len(image_rows)),
        "fix_label_kept_with_warning": int(len(fix_label)),
        "uncertain_kept": int(len(uncertain)),
        "bbox_rows": box_count,
        **counts,
    }
    (output_dir / "manual_clean_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not fix_label.empty:
        print("WARNING: fix_label images were kept. See manual_audit_fix_label_warnings.csv")


if __name__ == "__main__":
    main()
