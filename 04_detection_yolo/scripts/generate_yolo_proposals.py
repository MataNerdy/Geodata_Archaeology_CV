#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_FIELDS = [
    "candidate_id",
    "image_path",
    "image_id",
    "region",
    "source_id",
    "original_width",
    "original_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "x1_pad",
    "y1_pad",
    "x2_pad",
    "y2_pad",
    "yolo_conf",
    "yolo_class",
    "crop_path",
    "overlay_path",
    "split",
    "matched_gt_id",
    "max_iou_with_gt",
    "is_tp_iou03",
    "is_tp_iou05",
]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO proposal crops, overlays, and candidate metadata."
    )
    parser.add_argument("--config", type=Path, help="YAML or JSON config file.")
    parser.add_argument("--weights", type=Path, help="Trained YOLO weights.")
    parser.add_argument("--dataset-yaml", type=Path, help="YOLO dataset.yaml.")
    parser.add_argument("--dataset-dir", type=Path, help="Dataset root with images/ and labels/.")
    parser.add_argument("--images-dir", type=Path, help="Direct image directory; overrides dataset split lookup.")
    parser.add_argument("--metadata", type=Path, help="Optional metadata.csv.")
    parser.add_argument("--split", help="Dataset split to use when dataset root is provided.")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--padding-factor", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidates-csv", type=Path, help="Optional explicit candidates CSV path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and write no predictions/crops.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise SystemExit("PyYAML is required for YAML configs. Install pyyaml or use JSON config.") from exc
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_path(value: str | Path | None, *, config_dir: Path, cwd: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in [cwd, config_dir, config_dir.parent]:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (cwd / path).resolve()


def read_dataset_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit("PyYAML is required to read dataset.yaml. Install pyyaml.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dataset_root_from_yaml(dataset_yaml: Path | None) -> Path | None:
    if dataset_yaml is None:
        return None
    data = read_dataset_yaml(dataset_yaml)
    root = data.get("path")
    if root:
        root_path = Path(str(root)).expanduser()
        if root_path.is_absolute():
            return root_path
        return (dataset_yaml.parent / root_path).resolve()
    return dataset_yaml.parent.resolve()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(p.resolve() for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = set()
            for col in ["image", "image_path", "image_name", "filename", "file_name"]:
                value = row.get(col)
                if value:
                    keys.add(Path(value).name)
                    keys.add(Path(value).stem)
            if not keys:
                continue
            for key in keys:
                mapping.setdefault(key, row)
    return mapping


def yolo_label_to_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for idx, line in enumerate(label_path.read_text().splitlines()):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append({"gt_id": f"{label_path.stem}:{idx}", "class_id": int(cls), "box": Box(x1, y1, x2, y2)})
    return boxes


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def padded_box(box: Box, width: int, height: int, factor: float) -> Box:
    if factor <= 1.0:
        factor = 1.0
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    bw = (box.x2 - box.x1) * factor
    bh = (box.y2 - box.y1) * factor
    return Box(
        max(0.0, cx - bw / 2.0),
        max(0.0, cy - bh / 2.0),
        min(float(width), cx + bw / 2.0),
        min(float(height), cy + bh / 2.0),
    )


def int_box(box: Box) -> tuple[int, int, int, int]:
    return (int(math.floor(box.x1)), int(math.floor(box.y1)), int(math.ceil(box.x2)), int(math.ceil(box.y2)))


def draw_box(draw: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int], width: int = 3) -> None:
    coords = [box.x1, box.y1, box.x2, box.y2]
    for offset in range(width):
        draw.rectangle(
            [coords[0] - offset, coords[1] - offset, coords[2] + offset, coords[3] + offset],
            outline=color,
        )


def load_font() -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, 16)
        except OSError:
            pass
    return ImageFont.load_default()


def save_overlay(
    image: Image.Image,
    pred_box: Box,
    pad_box: Box,
    gt_boxes: list[dict[str, Any]],
    out_path: Path,
    label: str,
) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font()
    for gt in gt_boxes:
        draw_box(draw, gt["box"], (0, 220, 80), width=3)
    draw_box(draw, pad_box, (255, 180, 0), width=2)
    draw_box(draw, pred_box, (0, 170, 255), width=3)
    x = max(2, int(pred_box.x1))
    y = max(2, int(pred_box.y1) - 22)
    text_box = draw.textbbox((x, y), label, font=font)
    draw.rectangle([text_box[0] - 3, text_box[1] - 3, text_box[2] + 3, text_box[3] + 3], fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path, quality=92)


def greedy_metrics(candidates: list[dict[str, Any]], gt_by_image: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, float]:
    tp = 0
    fp = 0
    matched_by_image: dict[str, set[str]] = {key: set() for key in gt_by_image}
    for row in sorted(candidates, key=lambda x: float(x["yolo_conf"]), reverse=True):
        image_id = str(row["image_id"])
        gt_id = str(row.get("matched_gt_id") or "")
        iou = float(row.get("max_iou_with_gt") or 0.0)
        if gt_id and iou >= threshold and gt_id not in matched_by_image.setdefault(image_id, set()):
            matched_by_image[image_id].add(gt_id)
            tp += 1
        else:
            fp += 1
    total_gt = sum(len(v) for v in gt_by_image.values())
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall}


def coverage_metrics(candidates: list[dict[str, Any]], gt_by_image: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, float]:
    candidates_by_image: dict[str, list[Box]] = {}
    for row in candidates:
        image_id = str(row["image_id"])
        candidates_by_image.setdefault(image_id, []).append(
            Box(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        )
    covered: dict[str, set[str]] = {key: set() for key in gt_by_image}
    for image_id, gt_boxes in gt_by_image.items():
        pred_boxes = candidates_by_image.get(image_id, [])
        for gt in gt_boxes:
            if any(box_iou(pred, gt["box"]) >= threshold for pred in pred_boxes):
                covered.setdefault(image_id, set()).add(str(gt["gt_id"]))
    total_gt = sum(len(v) for v in gt_by_image.values())
    covered_gt = sum(len(v) for v in covered.values())
    return {
        "covered_gt": covered_gt,
        "total_gt": total_gt,
        "coverage_rate": covered_gt / total_gt if total_gt else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def path_for_csv(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    cwd = Path.cwd().resolve()
    config = load_config(args.config)
    config_dir = args.config.resolve().parent if args.config else cwd

    weights = resolve_path(coalesce(args.weights, config.get("weights")), config_dir=config_dir, cwd=cwd)
    dataset_yaml = resolve_path(coalesce(args.dataset_yaml, config.get("dataset_yaml"), config.get("data")), config_dir=config_dir, cwd=cwd)
    dataset_dir = resolve_path(coalesce(args.dataset_dir, config.get("dataset_dir"), config.get("dataset_path")), config_dir=config_dir, cwd=cwd)
    images_dir = resolve_path(coalesce(args.images_dir, config.get("images_dir")), config_dir=config_dir, cwd=cwd)
    metadata_path = resolve_path(coalesce(args.metadata, config.get("metadata")), config_dir=config_dir, cwd=cwd)
    split = str(coalesce(args.split, config.get("split"), "val"))
    conf = float(coalesce(args.conf, config.get("conf"), 0.05))
    nms_iou = float(coalesce(args.iou, config.get("iou"), 0.5))
    imgsz = int(coalesce(args.imgsz, config.get("imgsz"), 640))
    padding_factor = float(coalesce(args.padding_factor, config.get("padding_factor"), 1.5))
    max_det = int(coalesce(args.max_det, config.get("max_det"), 300))
    output_dir = resolve_path(coalesce(args.output_dir, config.get("output_dir")), config_dir=config_dir, cwd=cwd) or (cwd / "reports/proposals/run")
    candidates_csv = resolve_path(coalesce(args.candidates_csv, config.get("candidates_csv")), config_dir=config_dir, cwd=cwd)
    dry_run = bool(args.dry_run or config.get("dry_run", False))

    if dataset_dir is None:
        dataset_dir = dataset_root_from_yaml(dataset_yaml)
    if images_dir is None and dataset_dir is not None:
        images_dir = dataset_dir / "images" / split
    if metadata_path is None and dataset_dir is not None and (dataset_dir / "metadata.csv").exists():
        metadata_path = dataset_dir / "metadata.csv"
    if candidates_csv is None:
        candidates_csv = output_dir.parent / f"{output_dir.name}_candidates.csv"

    if images_dir is None:
        raise SystemExit("Provide --images-dir or --dataset-dir/--dataset-yaml.")

    image_paths = list_images(images_dir)
    print(f"Images: {len(image_paths)}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Candidates CSV: {candidates_csv}")
    if dry_run:
        print("Dry run: model inference and file generation skipped.")
        return
    if weights is None or not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit("ultralytics is required. Install it with `pip install ultralytics`.") from exc

    crop_dir = output_dir / "crops"
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(metadata_path)
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=nms_iou,
        max_det=max_det,
        device=args.device,
        save=False,
        verbose=False,
        stream=False,
    )

    candidate_rows: list[dict[str, Any]] = []
    gt_by_image: dict[str, list[dict[str, Any]]] = {}

    image_path_by_name = {path.name: path for path in image_paths}
    image_path_by_stem = {path.stem: path for path in image_paths}

    for result_idx, result in enumerate(results):
        result_path = Path(result.path)
        image_path = image_path_by_name.get(result_path.name) or image_path_by_stem.get(result_path.stem)
        if image_path is None:
            image_path = image_paths[result_idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_id = image_path.stem
        label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt" if dataset_dir else Path("__missing__")
        gt_boxes = yolo_label_to_boxes(label_path, width, height)
        gt_by_image[image_id] = gt_boxes
        meta = metadata.get(image_path.name) or metadata.get(image_path.stem) or {}

        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for pred_idx, (coords, score, cls_id) in enumerate(zip(xyxy, confs, classes), start=1):
            pred_box = Box(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
            pad_box = padded_box(pred_box, width, height, padding_factor)
            ious = [(gt["gt_id"], box_iou(pred_box, gt["box"])) for gt in gt_boxes]
            matched_gt_id, max_iou = max(ious, key=lambda x: x[1]) if ious else ("", 0.0)
            candidate_id = f"{image_id}_p{pred_idx:04d}"
            crop_path = crop_dir / f"{candidate_id}.jpg"
            overlay_path = overlay_dir / f"{candidate_id}.jpg"

            if args.save_crops:
                image.crop(int_box(pad_box)).save(crop_path, quality=92)
            if args.save_overlays:
                save_overlay(image, pred_box, pad_box, gt_boxes, overlay_path, f"{score:.3f}")

            row = {
                "candidate_id": candidate_id,
                "image_path": path_for_csv(image_path, cwd),
                "image_id": image_id,
                "region": meta.get("region", ""),
                "source_id": meta.get("source_id", ""),
                "original_width": width,
                "original_height": height,
                "x1": pred_box.x1,
                "y1": pred_box.y1,
                "x2": pred_box.x2,
                "y2": pred_box.y2,
                "x1_pad": pad_box.x1,
                "y1_pad": pad_box.y1,
                "x2_pad": pad_box.x2,
                "y2_pad": pad_box.y2,
                "yolo_conf": float(score),
                "yolo_class": int(cls_id),
                "crop_path": path_for_csv(crop_path, cwd) if args.save_crops else "",
                "overlay_path": path_for_csv(overlay_path, cwd) if args.save_overlays else "",
                "split": split,
                "matched_gt_id": matched_gt_id,
                "max_iou_with_gt": max_iou,
                "is_tp_iou03": max_iou >= 0.30,
                "is_tp_iou05": max_iou >= 0.50,
            }
            candidate_rows.append(row)

    write_csv(candidates_csv, candidate_rows, DEFAULT_FIELDS)
    write_csv(output_dir / "predictions.csv", candidate_rows, DEFAULT_FIELDS)

    metrics03 = greedy_metrics(candidate_rows, gt_by_image, 0.30)
    metrics05 = greedy_metrics(candidate_rows, gt_by_image, 0.50)
    coverage03 = coverage_metrics(candidate_rows, gt_by_image, 0.30)
    coverage05 = coverage_metrics(candidate_rows, gt_by_image, 0.50)
    total_gt = sum(len(v) for v in gt_by_image.values())
    summary = {
        "weights": str(weights),
        "dataset_dir": str(dataset_dir) if dataset_dir else "",
        "images_dir": str(images_dir),
        "split": split,
        "conf": conf,
        "nms_iou": nms_iou,
        "imgsz": imgsz,
        "padding_factor": padding_factor,
        "images": len(image_paths),
        "total_proposals": len(candidate_rows),
        "proposals_per_image": len(candidate_rows) / len(image_paths) if image_paths else 0.0,
        "total_gt": total_gt,
        "iou03": metrics03,
        "iou05": metrics05,
        "coverage_iou03": coverage03["coverage_rate"],
        "coverage_iou05": coverage05["coverage_rate"],
        "covered_gt_iou03": coverage03["covered_gt"],
        "covered_gt_iou05": coverage05["covered_gt"],
        "fp_per_image_iou03": metrics03["FP"] / len(image_paths) if image_paths else 0.0,
        "fp_per_image_iou05": metrics05["FP"] / len(image_paths) if image_paths else 0.0,
        "candidates_per_found_object_iou03": len(candidate_rows) / coverage03["covered_gt"] if coverage03["covered_gt"] else None,
        "candidates_csv": str(candidates_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# YOLO Proposal Generation Summary\n\n")
        for key in [
            "images",
            "total_proposals",
            "proposals_per_image",
            "total_gt",
            "covered_gt_iou03",
            "coverage_iou03",
            "covered_gt_iou05",
            "coverage_iou05",
        ]:
            f.write(f"- {key}: `{summary[key]}`\n")
        f.write("\n## Metrics\n\n")
        f.write("| IoU | TP | FP | FN | Precision | Recall |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for label, metrics in [("0.30", metrics03), ("0.50", metrics05)]:
            f.write(
                f"| {label} | {metrics['TP']} | {metrics['FP']} | {metrics['FN']} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |\n"
            )
    print(f"Saved candidates: {candidates_csv}")
    print(f"Saved summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
