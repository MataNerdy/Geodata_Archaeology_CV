"""Build a dataset profile and curated README visuals for archaeology segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import ndimage


CLASS_NAMES = {
    1: "kurgany_tselye",
    2: "kurgany_povrezhdennye",
    3: "gorodishcha",
    4: "fortifikatsii",
    5: "arkhitektury",
}
MODALITY_ORDER = ["Li", "Ae", "SpOr", "Or"]
MIN_COMPONENT_AREA = 8
MAX_VISUAL_SIDE = 1600
CURATED_EXAMPLE_SAMPLE_IDS = {
    "kurgany_povrezhdennye": "000507",
    "gorodishcha": "000362",
    "arkhitektury": "002580",
}
PALETTE = np.array(
    [
        [0, 0, 0],
        [90, 200, 120],
        [255, 99, 71],
        [255, 214, 10],
        [50, 160, 255],
        [190, 110, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="assets/dataset")
    return parser.parse_args()


def main() -> None:
    """Compute tables and render curated dataset visuals."""

    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_root / "metadata.csv", dtype={"sample_id": str})
    meta["sample_id"] = meta["sample_id"].map(lambda value: str(value).strip().zfill(6))
    print(f"[profile] Loaded metadata: {len(meta)} samples")

    class_sample_counts = (
        meta["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES.values(), fill_value=0)
        .rename_axis("class_name")
        .reset_index(name="samples")
    )
    modality_sample_counts = (
        meta["modality"]
        .value_counts()
        .reindex([item for item in MODALITY_ORDER if item in set(meta["modality"])], fill_value=0)
        .rename_axis("modality")
        .reset_index(name="samples")
    )
    heatmap = pd.crosstab(meta["class_name"], meta["modality"]).reindex(
        index=list(CLASS_NAMES.values()),
        columns=[item for item in MODALITY_ORDER if item in set(meta["modality"])],
        fill_value=0,
    )
    top_regions = (
        meta["region"]
        .value_counts()
        .head(20)
        .rename_axis("region")
        .reset_index(name="samples")
    )

    print("[profile] Counting connected components in stored masks...")
    component_areas = {class_name: [] for class_name in CLASS_NAMES.values()}
    for row_idx, sample_id in enumerate(meta["sample_id"], start=1):
        mask = load_2d(data_root / "masks" / f"{sample_id}.npy")
        for class_id, class_name in CLASS_NAMES.items():
            labels, count = ndimage.label(mask == class_id)
            if count:
                areas = ndimage.sum(mask == class_id, labels=labels, index=range(1, count + 1))
                component_areas[class_name].extend(float(area) for area in areas)
        if row_idx % 250 == 0 or row_idx == len(meta):
            print(f"[profile] Processed masks: {row_idx}/{len(meta)}")

    class_object_counts = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "raw_object_occurrences": len(component_areas[class_name]),
                "object_occurrences_min_area_8": sum(
                    area >= MIN_COMPONENT_AREA for area in component_areas[class_name]
                ),
            }
            for class_name in CLASS_NAMES.values()
        ]
    )
    avg_object_areas = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "raw_object_occurrences": len(areas),
                "raw_mean_area_px": float(np.mean(areas)) if areas else np.nan,
                "raw_median_area_px": float(np.median(areas)) if areas else np.nan,
                "mean_area_px_min_area_8": float(np.mean([area for area in areas if area >= MIN_COMPONENT_AREA]))
                if any(area >= MIN_COMPONENT_AREA for area in areas)
                else np.nan,
                "median_area_px_min_area_8": float(np.median([area for area in areas if area >= MIN_COMPONENT_AREA]))
                if any(area >= MIN_COMPONENT_AREA for area in areas)
                else np.nan,
            }
            for class_name, areas in component_areas.items()
        ]
    )

    class_object_counts.to_csv(out_dir / "class_object_counts.csv", index=False)
    class_sample_counts.to_csv(out_dir / "class_sample_counts.csv", index=False)
    modality_sample_counts.to_csv(out_dir / "modality_sample_counts.csv", index=False)
    heatmap.to_csv(out_dir / "class_modality_heatmap.csv")
    top_regions.to_csv(out_dir / "top20_regions.csv", index=False)
    avg_object_areas.to_csv(out_dir / "avg_object_area_by_class.csv", index=False)
    print(f"[profile] Saved statistics: {out_dir}")

    plot_heatmap(heatmap, out_dir / "class_modality_heatmap.png")
    plot_class_imbalance(class_sample_counts, out_dir / "class_imbalance.png")
    selected_examples = render_class_examples(meta, data_root, examples_dir)
    selected_examples.to_csv(out_dir / "selected_class_examples.csv", index=False)
    render_examples_collage(selected_examples, examples_dir, out_dir / "dataset_examples_collage.png")
    modality_selection = render_modality_comparison(meta, data_root, out_dir / "modality_comparison.png")
    modality_selection.to_csv(out_dir / "modality_comparison_selection.csv", index=False)

    summary_path = out_dir / "dataset_profile_summary.md"
    summary_path.write_text(
        build_summary(
            meta=meta,
            class_object_counts=class_object_counts,
            class_sample_counts=class_sample_counts,
            modality_sample_counts=modality_sample_counts,
            heatmap=heatmap,
            top_regions=top_regions,
            avg_object_areas=avg_object_areas,
            modality_selection=modality_selection,
        ),
        encoding="utf-8",
    )
    print(f"[profile] Saved summary: {summary_path}")
    print("[profile] Done")


def load_2d(path: Path) -> np.ndarray:
    """Load an image or mask as a two-dimensional array."""

    array = np.load(path)
    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] <= 4 else array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array: {path}, got {array.shape}")
    return array


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize raster values for display using robust percentiles."""

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(image, [2, 98])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)


def plot_heatmap(heatmap: pd.DataFrame, path: Path) -> None:
    """Plot class by modality sample counts."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = heatmap.to_numpy()
    image = ax.imshow(values, cmap="YlGnBu")
    ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_xlabel("Modality")
    ax.set_ylabel("Primary sample class")
    ax.set_title("Samples by class and modality")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(values[row, col]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[profile] Saved heatmap: {path}")


def plot_class_imbalance(class_sample_counts: pd.DataFrame, path: Path) -> None:
    """Plot primary sample class imbalance."""

    class_to_id = {class_name: class_id for class_id, class_name in CLASS_NAMES.items()}
    frame = class_sample_counts.sort_values("samples", ascending=True)
    colors = [PALETTE[class_to_id[class_name]] / 255.0 for class_name in frame["class_name"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(frame["class_name"], frame["samples"], color=colors)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Primary sample class")
    ax.set_title("Class imbalance")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, frame["samples"], strict=True):
        ax.text(value + max(frame["samples"]) * 0.015, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.set_xlim(0, max(frame["samples"]) * 1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[profile] Saved class imbalance: {path}")


def render_class_examples(meta: pd.DataFrame, data_root: Path, out_dir: Path) -> pd.DataFrame:
    """Save Li raster, mask and overlay examples for every foreground class."""

    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        curated_sample_id = CURATED_EXAMPLE_SAMPLE_IDS.get(class_name)
        if curated_sample_id:
            subset = meta[meta["sample_id"] == curated_sample_id].copy()
            if subset.empty:
                raise ValueError(f"Curated sample_id not found: {curated_sample_id}")
        else:
            subset = meta[(meta["modality"] == "Li") & (meta[f"has_{class_name}"] == True)].copy()  # noqa: E712
        if subset.empty:
            subset = meta[meta[f"has_{class_name}"] == True].copy()  # noqa: E712
            print(f"[profile] No Li example found: {class_name}; using available modality fallback")
        subset["display_score"] = subset[f"mask_{class_name}_pixels"].astype(float)
        if "touches_border" in subset:
            clean = subset[subset["touches_border"] == False]  # noqa: E712
            if not clean.empty:
                subset = clean
        row = subset.sort_values("display_score", ascending=False).iloc[0]
        sample_id = row["sample_id"]
        image = normalize_image(load_2d(data_root / "images" / f"{sample_id}.npy"))
        mask = load_2d(data_root / "masks" / f"{sample_id}.npy").astype(np.int64)
        save_example_triplet(image, mask, class_name, str(row["modality"]), out_dir)
        rows.append(
            {
                "class_name": class_name,
                "sample_id": sample_id,
                "region": row["region"],
                "modality": row["modality"],
                "foreground_pixels": int(row[f"mask_{class_name}_pixels"]),
            }
        )
    print(f"[profile] Saved class examples: {out_dir}")
    return pd.DataFrame(rows)


def save_example_triplet(
    image: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    modality: str,
    out_dir: Path,
) -> None:
    """Save raster, full multiclass mask and overlay PNG files."""

    mask_rgb = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
    base_rgb = np.repeat(image[..., None], 3, axis=2)
    overlay = base_rgb.copy()
    foreground = mask > 0
    overlay[foreground] = 0.55 * base_rgb[foreground] + 0.45 * (mask_rgb[foreground] / 255.0)

    save_resized_png(out_dir / f"{class_name}_{modality.lower()}.png", image, is_mask=False)
    save_resized_png(out_dir / f"{class_name}_mask.png", mask_rgb, is_mask=True)
    save_resized_png(out_dir / f"{class_name}_overlay.png", overlay, is_mask=False)


def save_resized_png(path: Path, image: np.ndarray, *, is_mask: bool) -> None:
    """Save a portfolio PNG with bounded dimensions."""

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    rendered = Image.fromarray(image)
    if max(rendered.size) > MAX_VISUAL_SIDE:
        scale = MAX_VISUAL_SIDE / max(rendered.size)
        size = tuple(max(1, round(value * scale)) for value in rendered.size)
        rendered = rendered.resize(size, resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS)
    rendered.save(path, optimize=True)


def render_examples_collage(selected_examples: pd.DataFrame, examples_dir: Path, path: Path) -> None:
    """Render one publication-ready class example grid."""

    rows = selected_examples.to_dict(orient="records")
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 18))
    column_names = ["Patch", "Mask", "Overlay"]
    for col, column_name in enumerate(column_names):
        axes[0, col].set_title(column_name, fontsize=14, pad=10)

    for row_idx, row in enumerate(rows):
        class_name = str(row["class_name"])
        modality = str(row["modality"])
        patch_path = examples_dir / f"{class_name}_{modality.lower()}.png"
        mask_path = examples_dir / f"{class_name}_mask.png"
        overlay_path = examples_dir / f"{class_name}_overlay.png"
        for col_idx, image_path in enumerate([patch_path, mask_path, overlay_path]):
            axes[row_idx, col_idx].imshow(Image.open(image_path), cmap="gray" if col_idx == 0 else None)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            f"{class_name}\n{row['region']}\n{modality} / {row['sample_id']}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=10,
            ha="right",
            va="center",
        )

    fig.suptitle("Archaeological segmentation dataset examples", fontsize=16, y=0.995)
    legend_items = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE[class_id] / 255.0, markersize=11)
        for class_id in CLASS_NAMES
    ]
    fig.legend(
        legend_items,
        list(CLASS_NAMES.values()),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.12, 0.045, 1, 0.985))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[profile] Saved dataset examples collage: {path}")


def render_modality_comparison(meta: pd.DataFrame, data_root: Path, path: Path) -> pd.DataFrame:
    """Render one regional Li/Ae/SpOr comparison for a shared primary class."""

    required = {"Li", "Ae", "SpOr"}
    candidates = []
    for (region, class_name), group in meta.groupby(["region", "class_name"]):
        if required.issubset(set(group["modality"])):
            candidates.append((len(group), region, class_name, group))
    if not candidates:
        raise RuntimeError("No region/class combination contains Li, Ae and SpOr")

    _, region, class_name, group = max(candidates, key=lambda item: item[0])
    selected = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, modality in zip(axes, ["Li", "Ae", "SpOr"], strict=True):
        subset = group[group["modality"] == modality].copy()
        pixel_col = f"mask_{class_name}_pixels"
        row = subset.sort_values(pixel_col, ascending=False).iloc[0]
        sample_id = row["sample_id"]
        image = normalize_image(load_2d(data_root / "images" / f"{sample_id}.npy"))
        mask = load_2d(data_root / "masks" / f"{sample_id}.npy").astype(np.int64)
        mask_rgb = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)] / 255.0
        base_rgb = np.repeat(image[..., None], 3, axis=2)
        foreground = mask > 0
        base_rgb[foreground] = 0.65 * base_rgb[foreground] + 0.35 * mask_rgb[foreground]
        ax.imshow(base_rgb)
        ax.set_title(f"{modality}: {sample_id}")
        ax.axis("off")
        selected.append(
            {
                "region": region,
                "class_name": class_name,
                "modality": modality,
                "sample_id": sample_id,
            }
        )
    fig.suptitle(f"Regional modality comparison: {region} / {class_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[profile] Saved modality comparison: {path}")
    return pd.DataFrame(selected)


def build_summary(
    *,
    meta: pd.DataFrame,
    class_object_counts: pd.DataFrame,
    class_sample_counts: pd.DataFrame,
    modality_sample_counts: pd.DataFrame,
    heatmap: pd.DataFrame,
    top_regions: pd.DataFrame,
    avg_object_areas: pd.DataFrame,
    modality_selection: pd.DataFrame,
) -> str:
    """Build a Markdown dataset profile."""

    return f"""# Archaeology Dataset Profile

## Scope

- Metadata rows: {len(meta)}
- Regions: {meta["region"].nunique()}
- Modalities: {", ".join(sorted(meta["modality"].astype(str).unique()))}
- Object counts are connected-component occurrences inside stored patch masks.
- The dataset has no global object identifier, so repeated crops of one source polygon cannot be deduplicated reliably.
- Object areas are native-mask pixel areas.
- The profile reports raw components and components with `min_area >= {MIN_COMPONENT_AREA}`, matching the polygon evaluator cleanup threshold.

## Object Occurrences By Class

{to_markdown(class_object_counts)}

## Samples By Primary Class

{to_markdown(class_sample_counts)}

## Samples By Modality

{to_markdown(modality_sample_counts)}

## Class x Modality

{to_markdown(heatmap.reset_index())}

## Top 20 Regions

{to_markdown(top_regions)}

## Object Area By Class

{to_markdown(avg_object_areas.round(2))}

## Regional Modality Comparison

The collage uses one region and one primary class represented in `Li`, `Ae` and `SpOr`.
The metadata does not contain a shared crop coordinate identifier, so the selected patches are regional examples and are not guaranteed to be pixel-aligned.

{to_markdown(modality_selection)}
"""


def to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    main()
