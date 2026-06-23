#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SEGMENTATION_ROOT = REPO_ROOT / "03_multiclass_segmentation_deeplab"


@dataclass(frozen=True)
class RuntimeConfig:
    candidates_csv: Path
    checkpoint: Path
    output_dir: Path
    segmentation_root: Path
    task: str
    encoder: str | None
    image_size: int
    device: str | None
    mask_threshold: float
    total_gt: int | None
    baseline_covered_gt_iou03: int | None
    baseline_images: int | None
    max_candidates: int | None
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an existing segmentation checkpoint as a refinement stage for YOLO proposal crops."
    )
    parser.add_argument("--config", type=Path, help="YAML or JSON config file.")
    parser.add_argument("--candidates-csv", type=Path, help="YOLO proposal candidate CSV.")
    parser.add_argument("--checkpoint", type=Path, help="Segmentation checkpoint .pth.")
    parser.add_argument("--output-dir", type=Path, help="Output directory.")
    parser.add_argument("--segmentation-root", type=Path, help="Path to 03_multiclass_segmentation_deeplab.")
    parser.add_argument("--task", choices=["binary_kurgan", "kurgan_multiclass", "all_classes", "archaeology_5class"])
    parser.add_argument("--encoder", help="Override encoder name. If omitted, infer from checkpoint/config/name.")
    parser.add_argument("--image-size", type=int, help="Segmentation model input size.")
    parser.add_argument("--device", help="cpu, cuda, mps. If omitted, auto-detect.")
    parser.add_argument("--mask-threshold", type=float, help="Foreground probability threshold.")
    parser.add_argument("--max-candidates", type=int, help="Optional limit for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and inspect inputs without inference.")
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
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs.") from exc
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_path(value: str | Path | None, *, cwd: Path, config_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in [cwd, config_dir, ROOT, REPO_ROOT]:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (cwd / path).resolve()


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config = load_config(args.config)
    cwd = Path.cwd()
    config_dir = args.config.resolve().parent if args.config else cwd

    candidates_csv = resolve_path(
        coalesce(args.candidates_csv, config.get("candidates_csv")),
        cwd=cwd,
        config_dir=config_dir,
    )
    checkpoint = resolve_path(coalesce(args.checkpoint, config.get("checkpoint")), cwd=cwd, config_dir=config_dir)
    output_dir = resolve_path(
        coalesce(args.output_dir, config.get("output_dir"), ROOT / "reports" / "segmentation_refinement" / "v3i_conf005"),
        cwd=cwd,
        config_dir=config_dir,
    )
    segmentation_root = resolve_path(
        coalesce(args.segmentation_root, config.get("segmentation_root"), SEGMENTATION_ROOT),
        cwd=cwd,
        config_dir=config_dir,
    )

    missing = {
        "candidates_csv": candidates_csv,
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "segmentation_root": segmentation_root,
    }
    unresolved = [name for name, value in missing.items() if value is None]
    if unresolved:
        raise ValueError(f"Missing required paths: {unresolved}")

    return RuntimeConfig(
        candidates_csv=candidates_csv,
        checkpoint=checkpoint,
        output_dir=output_dir,
        segmentation_root=segmentation_root,
        task=str(coalesce(args.task, config.get("task"), "archaeology_5class")),
        encoder=coalesce(args.encoder, config.get("encoder")),
        image_size=int(coalesce(args.image_size, config.get("image_size"), 256)),
        device=coalesce(args.device, config.get("device")),
        mask_threshold=float(coalesce(args.mask_threshold, config.get("mask_threshold"), 0.5)),
        total_gt=optional_int(config.get("total_gt")),
        baseline_covered_gt_iou03=optional_int(config.get("baseline_covered_gt_iou03")),
        baseline_images=optional_int(config.get("baseline_images")),
        max_candidates=coalesce(args.max_candidates, config.get("max_candidates")),
        dry_run=bool(args.dry_run or config.get("dry_run", False)),
    )


def optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def read_candidates(path: Path, max_candidates: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"candidate_{idx:06d}" for idx in range(len(df))]
    if "max_iou_with_gt" not in df.columns:
        df["max_iou_with_gt"] = 0.0
    if "is_tp_iou03" not in df.columns:
        df["is_tp_iou03"] = pd.to_numeric(df["max_iou_with_gt"], errors="coerce").fillna(0) >= 0.3
    if "is_tp_iou05" not in df.columns:
        df["is_tp_iou05"] = pd.to_numeric(df["max_iou_with_gt"], errors="coerce").fillna(0) >= 0.5
    if "group" not in df.columns:
        df["group"] = df.apply(infer_candidate_group, axis=1)
    if max_candidates is not None:
        df = df.head(int(max_candidates)).copy()
    return df.reset_index(drop=True)


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_candidate_group(row: pd.Series) -> str:
    if boolish(row.get("is_tp_iou03", False)):
        return "tp"
    max_iou = float_or_nan(row.get("max_iou_with_gt"))
    conf = float_or_nan(row.get("yolo_conf"))
    area_norm = float_or_nan(row.get("bbox_area_norm"))
    aspect_ratio = float_or_nan(row.get("aspect_ratio"))
    edge = boolish(row.get("bbox_touches_image_edge", False)) or boolish(row.get("pad_touches_image_edge", False))
    if (
        (not math.isnan(max_iou) and max_iou >= 0.3)
        or boolish(row.get("is_tp_iou03", False))
    ):
        return "tp"
    if (
        (not math.isnan(aspect_ratio) and aspect_ratio > 3.0)
        or (not math.isnan(area_norm) and not math.isnan(conf) and area_norm > 0.1 and conf < 0.15)
        or (edge and not math.isnan(conf) and conf < 0.1)
    ):
        return "obvious_fp"
    return "plausible_fp"


def float_or_nan(value: object) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def import_segmentation_model(segmentation_root: Path):
    if str(segmentation_root) not in sys.path:
        sys.path.insert(0, str(segmentation_root))
    from models.deeplab import build_model  # type: ignore

    return build_model


def num_classes_for_task(task: str) -> int:
    if task == "binary_kurgan":
        return 1
    if task == "kurgan_multiclass":
        return 3
    if task in {"all_classes", "archaeology_5class"}:
        return 6
    raise ValueError(f"Unsupported task: {task}")


def encoder_candidates(filename: str, checkpoint_config: dict[str, Any], override: str | None) -> list[str]:
    candidates: list[str] = []
    if override:
        candidates.append(str(override))
    for value in (checkpoint_config.get("encoder"), checkpoint_config.get("encoder_name")):
        if value:
            candidates.append(str(value))
    lowered = filename.lower()
    for encoder in ("resnet34", "resnet50", "efficientnet-b0"):
        if encoder.replace("-", "") in lowered.replace("-", ""):
            candidates.append(encoder)
    candidates.extend(["resnet34", "resnet50"])
    return list(dict.fromkeys(candidates))


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        try:
            import torch

            if all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        except ImportError:  # pragma: no cover
            pass
    raise ValueError("Checkpoint does not contain a recognizable state dict")


def load_state_dict_flexibly(model: Any, state: dict[str, Any]) -> None:
    variants = [state]
    if not all(key.startswith("model.") for key in state):
        variants.append({f"model.{key}": value for key, value in state.items()})
    if any(key.startswith("module.") for key in state):
        variants.append({key.removeprefix("module."): value for key, value in state.items()})
    last_error = None
    for variant in variants:
        try:
            model.load_state_dict(variant, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    raise last_error or RuntimeError("state_dict load failed")


def load_model(config: RuntimeConfig):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch is required to run segmentation refinement inference.") from exc

    build_model = import_segmentation_model(config.segmentation_root)
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state = extract_state_dict(checkpoint)
    errors = []
    for encoder in encoder_candidates(config.checkpoint.name, checkpoint_config, config.encoder):
        try:
            model = build_model(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=1,
                classes=num_classes_for_task(config.task),
            )
            load_state_dict_flexibly(model, state)
            device = resolve_device(config.device)
            model.to(device)
            model.eval()
            return model, device, encoder, checkpoint_config
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoder}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load segmentation checkpoint: " + " | ".join(errors))


def resolve_device(device_arg: str | None):
    import torch

    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_crop_path(value: object, candidates_csv: Path) -> Path | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    for base in [Path.cwd(), candidates_csv.parent, ROOT, REPO_ROOT]:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def preprocess_crop(path: Path, image_size: int):
    import torch
    import torch.nn.functional as F

    image = Image.open(path).convert("L")
    original_size = image.size
    array = np.asarray(image, dtype=np.float32)
    mean = float(array.mean())
    std = max(float(array.std()), 1e-6)
    array = (array - mean) / std
    tensor = torch.from_numpy(array)[None, None]
    if array.shape != (image_size, image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.float(), original_size


def predict_mask(model: Any, device: Any, crop_path: Path, config: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    tensor, original_size = preprocess_crop(crop_path, config.image_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        if config.task == "binary_kurgan":
            prob = torch.sigmoid(logits[:, 0:1])
            foreground = prob[:, 0] >= config.mask_threshold
            prob_map = prob[:, 0]
        else:
            probs = torch.softmax(logits, dim=1)
            foreground_probs = probs[:, 1:].max(dim=1).values
            pred_class = probs.argmax(dim=1)
            foreground = (pred_class != 0) & (foreground_probs >= config.mask_threshold)
            prob_map = foreground_probs
        foreground = foreground[:, None].float()
        prob_map = prob_map[:, None]
        if foreground.shape[-2:] != (original_size[1], original_size[0]):
            foreground = F.interpolate(foreground, size=(original_size[1], original_size[0]), mode="nearest")
            prob_map = F.interpolate(prob_map, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
    mask = foreground[0, 0].cpu().numpy().astype(bool)
    probs_np = prob_map[0, 0].cpu().numpy().astype(np.float32)
    return mask, probs_np


def connected_components(mask: np.ndarray) -> tuple[int, int]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    num_components = 0
    largest = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            num_components += 1
            stack = [(y, x)]
            seen[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            largest = max(largest, area)
    return num_components, largest


def mask_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(np.uint8), 1)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    boundary = (center == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    return int(boundary.sum())


def mask_bbox_area(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def mask_touches_edge(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def compute_mask_features(mask: np.ndarray, prob_map: np.ndarray) -> dict[str, Any]:
    total_pixels = int(mask.size)
    mask_area = int(mask.sum())
    num_components, largest_area = connected_components(mask)
    bbox_area = mask_bbox_area(mask)
    perimeter = mask_perimeter(mask)
    compactness = float((4.0 * math.pi * mask_area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
    return {
        "mask_area": mask_area,
        "foreground_fraction": mask_area / total_pixels if total_pixels else 0.0,
        "largest_component_area": int(largest_area),
        "largest_component_fraction": largest_area / mask_area if mask_area else 0.0,
        "mask_bbox_area": bbox_area,
        "mask_bbox_fraction": bbox_area / total_pixels if total_pixels else 0.0,
        "compactness": compactness,
        "touches_crop_edge": mask_touches_edge(mask),
        "num_components": int(num_components),
        "mean_foreground_prob": float(prob_map[mask].mean()) if mask_area else 0.0,
        "max_foreground_prob": float(prob_map.max()) if prob_map.size else 0.0,
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_mask_overlay(crop_path: Path, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(crop_path).convert("RGB")
    if mask.shape != (image.height, image.width):
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_arr = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay_arr[mask] = [0, 255, 80, 115]
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    composed.save(path, quality=92)


def feature_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    features = [
        "mask_area",
        "foreground_fraction",
        "largest_component_area",
        "largest_component_fraction",
        "mask_bbox_area",
        "mask_bbox_fraction",
        "compactness",
        "num_components",
        "mean_foreground_prob",
        "max_foreground_prob",
    ]
    rows = []
    for group, part in df.groupby(group_col, dropna=False):
        for feature in features:
            values = pd.to_numeric(part[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    group_col: group,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def covered_gt_count(df: pd.DataFrame, mask: pd.Series, iou_col: str = "is_tp_iou03") -> int:
    kept = df[mask & df[iou_col].map(boolish)]
    if "matched_gt_id" in kept.columns:
        ids = kept["matched_gt_id"].dropna().astype(str)
        ids = ids[ids.ne("")]
        return int(ids.nunique()) if not ids.empty else int(len(kept))
    return int(len(kept))


def simulate_rules(df: pd.DataFrame, config: RuntimeConfig) -> pd.DataFrame:
    total_images = config.baseline_images or max(1, int(df.get("image_id", pd.Series(range(len(df)))).nunique()))
    total_gt = config.total_gt or infer_total_gt(df)
    baseline_mask = pd.Series(True, index=df.index)
    measured_covered = covered_gt_count(df, baseline_mask, "is_tp_iou03")
    baseline_covered = config.baseline_covered_gt_iou03 or measured_covered
    baseline_fp = int((~df["is_tp_iou03"].map(boolish)).sum())

    rule_masks: dict[str, pd.Series] = {}
    for fg in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]:
        rule_masks[f"foreground_fraction >= {fg}"] = df["foreground_fraction"] >= fg
    for area in [8, 16, 32, 64, 128, 256]:
        rule_masks[f"largest_component_area >= {area}"] = df["largest_component_area"] >= area
    for comp in [0.02, 0.05, 0.1, 0.2]:
        rule_masks[f"compactness >= {comp}"] = df["compactness"] >= comp
    for max_components in [1, 3, 5, 10]:
        rule_masks[f"1 <= num_components <= {max_components}"] = (df["num_components"] >= 1) & (df["num_components"] <= max_components)
    rule_masks["foreground_fraction >= 0.005 AND largest_component_area >= 32"] = (
        (df["foreground_fraction"] >= 0.005) & (df["largest_component_area"] >= 32)
    )
    rule_masks["foreground_fraction >= 0.01 AND compactness >= 0.05"] = (
        (df["foreground_fraction"] >= 0.01) & (df["compactness"] >= 0.05)
    )
    rule_masks["largest_component_area >= 64 AND NOT touches_crop_edge"] = (
        (df["largest_component_area"] >= 64) & (~df["touches_crop_edge"].map(boolish))
    )

    rows = []
    for name, keep_mask in rule_masks.items():
        proposals_kept = int(keep_mask.sum())
        proposals_rejected = int((~keep_mask).sum())
        covered = covered_gt_count(df, keep_mask, "is_tp_iou03")
        if config.baseline_covered_gt_iou03 and measured_covered:
            # Candidate CSV only contains matched GT ids for proposals. If the
            # known baseline coverage is larger than the measured unique matched
            # IDs, carry the offset through rule simulations so the report stays
            # comparable with the proposal-stage threshold sweep.
            covered += config.baseline_covered_gt_iou03 - measured_covered
        fp_kept = int((keep_mask & ~df["is_tp_iou03"].map(boolish)).sum())
        fp_removed = baseline_fp - fp_kept
        rows.append(
            {
                "rule": name,
                "proposals_kept": proposals_kept,
                "proposals_rejected": proposals_rejected,
                "coverage_iou03_kept": covered,
                "coverage_iou03_rate": covered / total_gt if total_gt else 0.0,
                "coverage_loss": baseline_covered - covered,
                "coverage_loss_fraction_of_baseline": (baseline_covered - covered) / baseline_covered if baseline_covered else 0.0,
                "fp_kept_iou03": fp_kept,
                "fp_removed_iou03": fp_removed,
                "fp_reduction_fraction": fp_removed / baseline_fp if baseline_fp else 0.0,
                "fp_per_image_iou03": fp_kept / total_images,
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_loss_fraction_of_baseline", "fp_reduction_fraction"], ascending=[True, False])


def infer_total_gt(df: pd.DataFrame) -> int:
    if "total_gt" in df.columns and pd.to_numeric(df["total_gt"], errors="coerce").notna().any():
        return int(pd.to_numeric(df["total_gt"], errors="coerce").max())
    if "matched_gt_id" in df.columns:
        matched = df.loc[df["is_tp_iou03"].map(boolish), "matched_gt_id"].dropna().astype(str)
        # This is only covered GT, not total GT. The current v3i validation has 108 GT; use it when known.
        if int(df.get("image_id", pd.Series()).nunique()) == 68:
            return 108
        return int(matched.nunique())
    return int(df["is_tp_iou03"].map(boolish).sum())


def make_contact_sheet(df: pd.DataFrame, image_col: str, output_path: Path, title_col: str = "candidate_id", max_items: int = 25) -> None:
    rows = df.head(max_items)
    if rows.empty:
        return
    thumbs = []
    font = load_font()
    for _, row in rows.iterrows():
        path = Path(str(row[image_col]))
        if not path.is_absolute():
            path = resolve_existing_relative_path(path)
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((220, 180), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (220, 215), "white")
        canvas.paste(img, ((220 - img.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        label = str(row.get(title_col, ""))[:34]
        draw.text((5, 184), label, fill="black", font=font)
        draw.text((5, 199), f"fg={float_or_nan(row.get('foreground_fraction')):.3f}", fill="black", font=font)
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 5
    grid_rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 220, grid_rows * 215), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 220, (idx // cols) * 215))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def resolve_existing_relative_path(path: Path) -> Path:
    for base in [Path.cwd(), REPO_ROOT, ROOT]:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def load_font() -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, 13)
        except OSError:
            pass
    return ImageFont.load_default()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional tabulate dependency."""

    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(format_markdown_value)
    headers = [str(col) for col in text_df.columns]
    rows = text_df.astype(str).values.tolist()
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        render_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(config: RuntimeConfig, refined: pd.DataFrame, rule_sim: pd.DataFrame, output_path: Path, encoder: str | None) -> None:
    total_images = config.baseline_images or int(refined.get("image_id", pd.Series(range(len(refined)))).nunique())
    total_gt = config.total_gt or infer_total_gt(refined)
    baseline_covered = covered_gt_count(refined, pd.Series(True, index=refined.index), "is_tp_iou03")
    if config.baseline_covered_gt_iou03:
        baseline_covered = config.baseline_covered_gt_iou03
    baseline_fp = int((~refined["is_tp_iou03"].map(boolish)).sum())
    promising = rule_sim[
        (rule_sim["fp_reduction_fraction"] >= 0.5)
        & (rule_sim["coverage_loss_fraction_of_baseline"] <= 0.10)
    ]

    lines = [
        "# Segmentation Refinement Audit for v3i Proposals",
        "",
        "## Scope",
        "",
        "No new YOLO or segmentation training was run. This report applies an existing segmentation checkpoint to YOLO proposal crops.",
        "",
        "Important caveat: the DeepLab segmentation model was trained on segmentation `.npy` geodata patches, while current YOLO proposal crops are image crops saved from the detection pipeline. Treat this as an exploratory refinement check unless the crop source is aligned with the original segmentation input distribution.",
        "",
        "## Inputs",
        "",
        f"- candidates: `{config.candidates_csv}`",
        f"- checkpoint: `{config.checkpoint}`",
        f"- segmentation_root: `{config.segmentation_root}`",
        f"- task: `{config.task}`",
        f"- encoder: `{encoder or config.encoder or 'auto'}`",
        f"- image_size: `{config.image_size}`",
        f"- mask_threshold: `{config.mask_threshold}`",
        "",
        "## Baseline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Images | {total_images} |",
        f"| Proposals | {len(refined)} |",
        f"| GT objects | {total_gt} |",
        f"| Covered GT @ IoU0.3 | {baseline_covered} |",
        f"| Coverage @ IoU0.3 | {baseline_covered / total_gt if total_gt else 0:.3f} |",
        f"| FP candidates @ IoU0.3 | {baseline_fp} |",
        f"| FP/image @ IoU0.3 | {baseline_fp / max(total_images, 1):.3f} |",
        "",
        "## Rule Simulation",
        "",
        dataframe_to_markdown(rule_sim.head(12)),
        "",
        "## Target Check",
        "",
    ]
    if promising.empty:
        lines.extend(
            [
                "No tested mask-based rule reached the target of `>= 50% FP reduction` with `<= 10% covered-GT loss`.",
                "",
                "This does not mean segmentation refinement is useless; it means the simple rules tested here are not sufficient under the current crop/domain setup.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one tested mask-based rule reached the target of `>= 50% FP reduction` with `<= 10% covered-GT loss`:",
                "",
                dataframe_to_markdown(promising.head(10)),
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    candidates = read_candidates(config.candidates_csv, config.max_candidates)
    print(f"Candidates: {len(candidates)}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Output: {config.output_dir}")

    if config.dry_run:
        print("Dry run: inference skipped.")
        print("Candidate groups:")
        print(candidates["group"].value_counts(dropna=False).to_string())
        return

    model, device, encoder, _ = load_model(config)
    masks_dir = config.output_dir / "masks"
    overlays_dir = config.output_dir / "overlays"

    rows = []
    for idx, row in candidates.iterrows():
        crop_path = resolve_crop_path(row.get("crop_path"), config.candidates_csv)
        if crop_path is None or not crop_path.exists():
            print(f"[WARN] missing crop for candidate {row.get('candidate_id')}: {crop_path}")
            continue
        mask, prob_map = predict_mask(model, device, crop_path, config)
        candidate_id = str(row["candidate_id"])
        mask_path = masks_dir / f"{candidate_id}.png"
        overlay_path = overlays_dir / f"{candidate_id}.jpg"
        save_mask(mask, mask_path)
        save_mask_overlay(crop_path, mask, overlay_path)
        features = compute_mask_features(mask, prob_map)
        out = row.to_dict()
        out.update(features)
        out["segmentation_mask_path"] = str(mask_path.relative_to(REPO_ROOT) if mask_path.is_relative_to(REPO_ROOT) else mask_path)
        out["segmentation_overlay_path"] = str(overlay_path.relative_to(REPO_ROOT) if overlay_path.is_relative_to(REPO_ROOT) else overlay_path)
        rows.append(out)
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(candidates)}")

    refined = pd.DataFrame(rows)
    if refined.empty:
        raise RuntimeError("No candidates were processed. Check crop paths.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = config.output_dir / "refined_candidates.csv"
    refined.to_csv(refined_path, index=False)

    refined.to_csv(ROOT / "reports" / "refined_candidates.csv", index=False)

    refined["tp_fp_iou03"] = np.where(refined["is_tp_iou03"].map(boolish), "TP", "FP")
    summary_tp_fp = feature_summary(refined, "tp_fp_iou03")
    summary_group = feature_summary(refined, "group")
    summary_tp_fp.to_csv(config.output_dir / "mask_feature_summary_tp_fp.csv", index=False)
    summary_group.to_csv(config.output_dir / "mask_feature_summary_by_group.csv", index=False)
    summary_tp_fp.to_csv(ROOT / "reports" / "mask_feature_summary_tp_fp.csv", index=False)

    rule_sim = simulate_rules(refined, config)
    rule_sim.to_csv(config.output_dir / "rule_simulation_segmentation.csv", index=False)
    rule_sim.to_csv(ROOT / "reports" / "rule_simulation_segmentation.csv", index=False)

    summary_rows = [
        {
            "metric": "proposals",
            "value": len(refined),
        },
        {
            "metric": "covered_gt_iou03",
            "value": covered_gt_count(refined, pd.Series(True, index=refined.index), "is_tp_iou03"),
        },
        {
            "metric": "fp_iou03",
            "value": int((~refined["is_tp_iou03"].map(boolish)).sum()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(config.output_dir / "segmentation_refinement_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ROOT / "reports" / "segmentation_refinement_summary.csv", index=False)

    contact_dir = config.output_dir / "contact_sheets"
    kept_rule = refined["foreground_fraction"] >= 0.005
    make_contact_sheet(refined[refined["is_tp_iou03"].map(boolish) & kept_rule], "segmentation_overlay_path", contact_dir / "tp_kept.jpg")
    make_contact_sheet(refined[refined["is_tp_iou03"].map(boolish) & ~kept_rule], "segmentation_overlay_path", contact_dir / "tp_rejected.jpg")
    make_contact_sheet(refined[(~refined["is_tp_iou03"].map(boolish)) & kept_rule], "segmentation_overlay_path", contact_dir / "fp_kept.jpg")
    make_contact_sheet(refined[(~refined["is_tp_iou03"].map(boolish)) & ~kept_rule], "segmentation_overlay_path", contact_dir / "fp_rejected.jpg")

    report_path = ROOT / "reports" / "segmentation_refinement_v3i.md"
    write_report(config, refined, rule_sim, report_path, encoder)
    print(f"Saved: {refined_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
