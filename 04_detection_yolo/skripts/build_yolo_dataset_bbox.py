from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from PIL import Image
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

from overlay_5_classes import (
    DATASET_ROOT,
    find_regions,
    read_target_crs_from_utm,
    load_geojsons,
    choose_raster_for_modality,
)

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

OUT_DIR = Path("dataset_yolo_bbox")

# Kaggle-friendly filenames: short image/label names instead of huge region/raster stems.
# The full descriptive stem is still stored in metadata.csv and rename_mapping.csv.
USE_SHORT_FILENAMES = True
SHORT_NAME_COUNTER_START = 0


TARGET_CONTEXT_M_BY_MODALITY = {
    "Li": 250,
    "Ae": 600,
    "SpOr": 1200,
    "Or": 250,
}

SOURCE_CLASS_TO_YOLO_ID = {
    "kurgany_tselye": 0,
    "kurgany_povrezhdennye": 1,
    "gorodishcha": 2,
    "fortifikatsii": 3,
    "arkhitektury": 4,
}

YOLO_NAMES = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}

TILE_CANDIDATES = [1024, 1536, 2048, 3072, 4096]
OVERLAP_FRACTION = 0.25
RESIZE_TO = 1024

POSITIVE_ONLY_FOR_DEBUG = False
NEGATIVE_RATIO = 0.25
MODALITIES_TO_USE = {"Li", "Ae", "SpOr", "Or"}

MIN_BBOX_AREA_PX = 80
MIN_VALID_FRACTION = 0.35
MIN_STD = 5
MIN_P98_P2 = 10

VAL_REGION_FRACTION = 0.2
RANDOM_SEED = 42

# If True, drops positive tiles where at least one bbox touches tile edge.
DROP_POSITIVE_TILES_WITH_EDGE_OBJECTS = False

# If True, drops only individual edge-touching bboxes and keeps other objects in tile.
DROP_EDGE_BBOXES = False

EDGE_EPS = 1e-6


# =========================
# UTILS
# =========================

def make_dirs(out_dir: Path):
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def make_sample_stem(split: str, counter: int, long_stem: str):
    """
    Returns the real file stem and the human-readable/original stem.

    With USE_SHORT_FILENAMES=True files are saved as e.g. train_000123.png,
    while long_stem is preserved in metadata/mapping for debugging.
    """
    if USE_SHORT_FILENAMES:
        return f"{split}_{counter:06d}", long_stem
    return long_stem, long_stem


def choose_tile_size(src, modality):
    px = max(abs(src.transform.a), abs(src.transform.e))
    target_context = TARGET_CONTEXT_M_BY_MODALITY.get(modality, 300)

    raw_tile = target_context / px
    candidates = np.array(TILE_CANDIDATES)

    tile_size = int(candidates[np.argmin(np.abs(candidates - raw_tile))])
    return tile_size, px, raw_tile, target_context


def iter_windows(width, height, tile_size, stride):
    xs = list(range(0, max(width - tile_size + 1, 1), stride))
    ys = list(range(0, max(height - tile_size + 1, 1), stride))

    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))

    seen = set()
    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield Window(x, y, tile_size, tile_size)


def tile_to_rgb(tile):
    """
    Input: rasterio masked array CxHxW or HxW.
    Output: uint8 RGB HxWx3.
    """
    tile = tile.astype(np.float32)
    arr = tile.filled(np.nan)

    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.shape[2] > 3:
        arr = arr[:, :, :3]

    out_channels = []

    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        valid = ch[np.isfinite(ch)]

        if valid.size == 0:
            out = np.zeros(ch.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(valid, [2, 98])
            if hi <= lo:
                out = np.zeros(ch.shape, dtype=np.uint8)
            else:
                norm = np.clip((ch - lo) / (hi - lo), 0, 1)
                norm = np.nan_to_num(norm, nan=0.0)
                out = (norm * 255).astype(np.uint8)

        out_channels.append(out)

    rgb = np.stack(out_channels, axis=-1)

    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)

    if rgb.shape[2] == 2:
        zero = np.zeros(rgb.shape[:2] + (1,), dtype=np.uint8)
        rgb = np.concatenate([rgb, zero], axis=2)

    return rgb[:, :, :3]


def extract_polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(extract_polygons(g))
        return out
    return []


def polygon_to_bbox_yolo(poly, transform, window, tile_size):
    """
    Polygon in raster CRS -> YOLO bbox:
    cls x_center y_center width height, all normalized to [0, 1].
    """
    minx, miny, maxx, maxy = poly.bounds

    xs = np.array([minx, maxx, minx, maxx], dtype=np.float64)
    ys = np.array([miny, miny, maxy, maxy], dtype=np.float64)

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.float32) - float(window.row_off)
    cols = np.asarray(cols, dtype=np.float32) - float(window.col_off)

    x1 = float(np.clip(cols.min(), 0, tile_size - 1))
    x2 = float(np.clip(cols.max(), 0, tile_size - 1))
    y1 = float(np.clip(rows.min(), 0, tile_size - 1))
    y2 = float(np.clip(rows.max(), 0, tile_size - 1))

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh

    if bw <= 1 or bh <= 1:
        return None

    xc = (x1 + x2) / 2.0 / tile_size
    yc = (y1 + y2) / 2.0 / tile_size
    wn = bw / tile_size
    hn = bh / tile_size

    touches_tile_edge = (
        x1 <= EDGE_EPS
        or y1 <= EDGE_EPS
        or x2 >= tile_size - 1 - EDGE_EPS
        or y2 >= tile_size - 1 - EDGE_EPS
    )

    return {
        "bbox": [xc, yc, wn, hn],
        "bbox_px": [x1, y1, x2, y2],
        "bbox_area_px": float(area),
        "touches_tile_edge": bool(touches_tile_edge),
    }


def tile_quality(tile):
    tile_float = tile.astype(np.float32)
    tile_filled = tile_float.filled(np.nan)

    valid_mask = ~np.ma.getmaskarray(tile)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.any(axis=0)

    valid_fraction = float(valid_mask.mean())
    base = {
        "valid_fraction": valid_fraction,
        "tile_std": np.nan,
        "tile_p2": np.nan,
        "tile_p98": np.nan,
        "tile_p98_minus_p2": np.nan,
    }

    if valid_fraction < MIN_VALID_FRACTION:
        return False, base

    finite_vals = tile_filled[np.isfinite(tile_filled)]
    if finite_vals.size == 0:
        return False, base

    tile_std = float(np.nanstd(finite_vals))
    p2, p98 = np.nanpercentile(finite_vals, [2, 98])
    p2 = float(p2)
    p98 = float(p98)
    contrast = float(p98 - p2)

    ok = tile_std >= MIN_STD and contrast >= MIN_P98_P2

    return ok, {
        "valid_fraction": valid_fraction,
        "tile_std": tile_std,
        "tile_p2": p2,
        "tile_p98": p98,
        "tile_p98_minus_p2": contrast,
    }


def write_dataset_yaml(out_dir: Path):
    names = "\n".join(f"  {i}: {name}" for i, name in YOLO_NAMES.items())
    text = f"""path: {out_dir.resolve()}

train: images/train
val: images/val

names:
{names}
"""
    (out_dir / "dataset.yaml").write_text(text, encoding="utf-8")


# =========================
# MAIN
# =========================

def main():
    random.seed(RANDOM_SEED)
    make_dirs(OUT_DIR)

    regions = find_regions(DATASET_ROOT)
    region_names = sorted([r["region_dir"].name for r in regions])
    random.shuffle(region_names)

    n_val = max(1, int(len(region_names) * VAL_REGION_FRACTION))
    val_regions = set(region_names[:n_val])

    print(f"Found regions: {len(regions)}")
    print(f"Val regions: {len(val_regions)}")
    print(sorted(val_regions))

    metadata_rows = []
    rename_mapping_rows = []
    short_name_counter = SHORT_NAME_COUNTER_START
    image_count = 0
    positive_count = 0
    negative_count = 0
    bbox_count = 0

    for region_info in regions:
        region = region_info["region_dir"].name
        split = "val" if region in val_regions else "train"

        target_crs = read_target_crs_from_utm(region_info["utm_path"])
        gdf = load_geojsons(region_info["markup_dir"], target_crs)

        if gdf is None or gdf.empty:
            continue

        for modality, raster_dir in region_info["raster_dirs"]:
            if modality not in MODALITIES_TO_USE:
                continue

            raster_path = choose_raster_for_modality(modality, raster_dir)
            if raster_path is None:
                continue

            gdf_mod = gdf[gdf["modality"] == modality].copy()
            if gdf_mod.empty:
                continue

            with rasterio.open(raster_path) as src:
                tile_size, px, raw_tile, target_context = choose_tile_size(src, modality)
                overlap = int(tile_size * OVERLAP_FRACTION)
                stride = tile_size - overlap
                raster_width = src.width
                raster_height = src.height
                raster_bounds = box(*src.bounds)

                gdf_in_raster = gdf_mod[gdf_mod.geometry.intersects(raster_bounds)].copy()
                used_crs_fallback = False

                if gdf_in_raster.empty and src.crs is not None:
                    try:
                        gdf_alt = gdf_mod.to_crs(src.crs)
                        gdf_in_raster = gdf_alt[gdf_alt.geometry.intersects(raster_bounds)].copy()
                        used_crs_fallback = not gdf_in_raster.empty
                    except Exception as e:
                        print(f"{region} | {modality}: CRS fallback failed: {e}")

                if gdf_in_raster.empty:
                    print(f"{region} | {modality}: no objects intersect raster")
                    continue

                print(
                    f"{region} | {modality} | px={px:.3f} | "
                    f"context≈{tile_size * px:.1f}m | tile={tile_size} | "
                    f"objects={len(gdf_in_raster)}"
                )

                for window in iter_windows(raster_width, raster_height, tile_size, stride):
                    tile_geom = box(*src.window_bounds(window))
                    tile_touches_raster_edge = (
                        window.col_off <= 0
                        or window.row_off <= 0
                        or window.col_off + window.width >= raster_width
                        or window.row_off + window.height >= raster_height
                    )

                    gdf_tile = gdf_in_raster[gdf_in_raster.geometry.intersects(tile_geom)]

                    label_lines = []
                    object_records = []
                    has_edge_object = False

                    for _, row in gdf_tile.iterrows():
                        source_class = row["class_name"]
                        if source_class not in SOURCE_CLASS_TO_YOLO_ID:
                            continue

                        cls_id = SOURCE_CLASS_TO_YOLO_ID[source_class]
                        clipped = row.geometry.intersection(tile_geom)

                        for poly in extract_polygons(clipped):
                            if poly.is_empty:
                                continue

                            bbox_info = polygon_to_bbox_yolo(poly, src.transform, window, tile_size)
                            if bbox_info is None:
                                continue

                            if bbox_info["bbox_area_px"] < MIN_BBOX_AREA_PX:
                                continue

                            if bbox_info["touches_tile_edge"]:
                                has_edge_object = True
                                if DROP_EDGE_BBOXES:
                                    continue

                            xc, yc, w, h = bbox_info["bbox"]
                            label_lines.append(
                                f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )

                            x1, y1, x2, y2 = bbox_info["bbox_px"]
                            object_records.append({
                                "class_id": cls_id,
                                "class_name": source_class,
                                "bbox_x1_px": x1,
                                "bbox_y1_px": y1,
                                "bbox_x2_px": x2,
                                "bbox_y2_px": y2,
                                "bbox_area_px": bbox_info["bbox_area_px"],
                                "bbox_touches_tile_edge": bbox_info["touches_tile_edge"],
                            })

                    is_positive = len(label_lines) > 0

                    if DROP_POSITIVE_TILES_WITH_EDGE_OBJECTS and is_positive and has_edge_object:
                        continue

                    if POSITIVE_ONLY_FOR_DEBUG and not is_positive:
                        continue

                    if not is_positive and random.random() > NEGATIVE_RATIO:
                        continue

                    tile = src.read(window=window, masked=True)
                    ok_quality, q = tile_quality(tile)
                    if not ok_quality:
                        continue

                    long_stem = safe_name(
                        f"{region}_{modality}_{Path(raster_path).stem}_"
                        f"x{int(window.col_off)}_y{int(window.row_off)}"
                    )
                    stem, original_stem = make_sample_stem(split, short_name_counter, long_stem)
                    short_name_counter += 1

                    img_path = OUT_DIR / "images" / split / f"{stem}.png"
                    lbl_path = OUT_DIR / "labels" / split / f"{stem}.txt"

                    rename_mapping_rows.append({
                        "split": split,
                        "old_stem": original_stem,
                        "old_image_name": f"{original_stem}.png",
                        "old_label_name": f"{original_stem}.txt",
                        "new_stem": stem,
                        "new_image_name": f"{stem}.png",
                        "new_label_name": f"{stem}.txt",
                        "image": str(img_path),
                        "label": str(lbl_path),
                    })

                    rgb = tile_to_rgb(tile)
                    img = Image.fromarray(rgb)
                    if RESIZE_TO is not None and RESIZE_TO != tile_size:
                        img = img.resize((RESIZE_TO, RESIZE_TO), Image.BILINEAR)

                    img.save(img_path)
                    lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

                    image_count += 1
                    if is_positive:
                        positive_count += 1
                    else:
                        negative_count += 1
                    bbox_count += len(label_lines)

                    base_row = {
                        "split": split,
                        "region": region,
                        "modality": modality,
                        "raster_file": Path(raster_path).name,
                        "sample_stem": stem,
                        "original_stem": original_stem,
                        "image_name": img_path.name,
                        "label_name": lbl_path.name,
                        "image": str(img_path),
                        "label": str(lbl_path),
                        "x": int(window.col_off),
                        "y": int(window.row_off),
                        "tile_size": int(tile_size),
                        "resize_to": int(RESIZE_TO) if RESIZE_TO is not None else None,
                        "pixel_size_m": float(px),
                        "used_crs_fallback": bool(used_crs_fallback),
                        "context_m": float(tile_size * px),
                        "target_context_m": float(target_context),
                        "raw_tile": float(raw_tile),
                        "overlap": int(overlap),
                        "stride": int(stride),
                        "raster_width": int(raster_width),
                        "raster_height": int(raster_height),
                        "tile_touches_raster_edge": bool(tile_touches_raster_edge),
                        "is_positive": bool(is_positive),
                        "n_objects": int(len(label_lines)),
                        "has_edge_object": bool(has_edge_object),
                        **q,
                    }

                    if object_records:
                        for obj in object_records:
                            metadata_rows.append({**base_row, **obj})
                    else:
                        metadata_rows.append({
                            **base_row,
                            "class_id": None,
                            "class_name": None,
                            "bbox_x1_px": None,
                            "bbox_y1_px": None,
                            "bbox_x2_px": None,
                            "bbox_y2_px": None,
                            "bbox_area_px": None,
                            "bbox_touches_tile_edge": None,
                        })

    meta = pd.DataFrame(metadata_rows)
    meta.to_csv(OUT_DIR / "metadata.csv", index=False)

    rename_mapping = pd.DataFrame(rename_mapping_rows)
    rename_mapping.to_csv(OUT_DIR / "rename_mapping.csv", index=False)

    write_dataset_yaml(OUT_DIR)

    print("=" * 80)
    print("DONE")
    print("images:", image_count)
    print("positive images:", positive_count)
    print("negative images:", negative_count)
    print("bboxes:", bbox_count)
    if not meta.empty:
        print("\nImages by split/modality/positive:")
        print(
            meta.drop_duplicates("image")
            .groupby(["split", "modality", "is_positive"])
            .size()
        )
        print("\nBBoxes by class:")
        print(meta[meta["is_positive"]].groupby("class_name").size())
    print("yaml:", OUT_DIR / "dataset.yaml")
    print("metadata:", OUT_DIR / "metadata.csv")
    print("rename mapping:", OUT_DIR / "rename_mapping.csv")


if __name__ == "__main__":
    main()
