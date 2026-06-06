from pathlib import Path
import random
import warnings
import numpy as np
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

OUT_DIR = Path("dataset_yolo")
AUTO_TILE_SIZE = True

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

TILE_CANDIDATES = [1024, 1536, 2048, 3072, 4096]
OVERLAP_FRACTION = 0.25

RESIZE_TO = 1024
POSITIVE_ONLY_FOR_DEBUG = False
NEGATIVE_RATIO = 0.25
MODALITIES_TO_USE = {"Li", "Ae", "SpOr", "Or"}

MIN_POLYGON_AREA_PX = 80
VAL_REGION_FRACTION = 0.2

# =========================

def choose_tile_size(src, modality):
    px = max(abs(src.transform.a), abs(src.transform.e))
    target_context = TARGET_CONTEXT_M_BY_MODALITY.get(modality, 300)

    raw_tile = target_context / px
    candidates = np.array(TILE_CANDIDATES)

    tile_size = int(candidates[np.argmin(np.abs(candidates - raw_tile))])
    return tile_size, px, raw_tile


def iter_windows(width, height, tile_size, stride):
    xs = list(range(0, max(width - tile_size + 1, 1), stride))
    ys = list(range(0, max(height - tile_size + 1, 1), stride))

    if xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))

    for y in ys:
        for x in xs:
            yield Window(x, y, tile_size, tile_size)


def tile_to_rgb(tile):
    """
    На вход: masked array из rasterio: CxHxW или HxW.
    На выход: обычный uint8 RGB: HxWx3.
    """
    tile = tile.astype(np.float32)
    arr = tile.filled(np.nan)

    # CxHxW -> HxWxC
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)

    # HxW -> HxWx1
    if arr.ndim == 2:
        arr = arr[..., None]

    # если каналов больше 3 — берём первые 3
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

    # 1 канал -> RGB
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)

    # 2 канала -> добавим третий пустой
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


def polygon_to_yolo(poly, transform, window, tile_size):
    coords = np.array(poly.exterior.coords)

    rows, cols = rowcol(transform, coords[:, 0], coords[:, 1])

    rows = np.asarray(rows, dtype=np.float32) - float(window.row_off)
    cols = np.asarray(cols, dtype=np.float32) - float(window.col_off)

    pts = np.stack([cols, rows], axis=1).astype(np.float32)

    pts[:, 0] /= float(tile_size)
    pts[:, 1] /= float(tile_size)

    pts = np.clip(pts, 0.0, 1.0)

    return pts.reshape(-1)


# =========================
# MAIN
# =========================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "images/train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images/val").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels/val").mkdir(parents=True, exist_ok=True)

    regions = find_regions(DATASET_ROOT)

    region_names = [r["region_dir"].name for r in regions]
    random.shuffle(region_names)

    val_regions = set(region_names[:int(len(region_names)*VAL_REGION_FRACTION)])

    print("Val regions:", val_regions)

    total = 0

    for region_info in regions:
        region = region_info["region_dir"].name
        split = "val" if region in val_regions else "train"

        gdf = load_geojsons(region_info["markup_dir"],
                            read_target_crs_from_utm(region_info["utm_path"]))

        if gdf is None:
            continue

        for modality, raster_dir in region_info["raster_dirs"]:
            if modality not in MODALITIES_TO_USE:
                continue

            raster_path = choose_raster_for_modality(modality, raster_dir)
            if raster_path is None:
                continue

            gdf_mod = gdf[gdf["modality"] == modality]
            if gdf_mod.empty:
                continue

            with rasterio.open(raster_path) as src:

                tile_size, px, raw = choose_tile_size(src, modality)
                overlap = int(tile_size * OVERLAP_FRACTION)
                stride = tile_size - overlap

                print(f"{region} | {modality} | px={px:.3f} | tile={tile_size}")

                for window in iter_windows(src.width, src.height, tile_size, stride):

                    tile_geom = box(*src.window_bounds(window))

                    gdf_tile = gdf_mod[gdf_mod.geometry.intersects(tile_geom)]

                    labels = []

                    for _, row in gdf_tile.iterrows():
                        for poly in extract_polygons(row.geometry.intersection(tile_geom)):
                            if poly.area < MIN_POLYGON_AREA_PX:
                                continue

                            coords = polygon_to_yolo(poly, src.transform, window, tile_size)
                            cls_id = SOURCE_CLASS_TO_YOLO_ID[row["class_name"]]
                            labels.append(f"{cls_id} " + " ".join(map(str, coords)))

                    is_positive = len(labels) > 0

                    if POSITIVE_ONLY_FOR_DEBUG and not is_positive:
                        continue

                    if not is_positive:
                        if random.random() > NEGATIVE_RATIO:
                            continue

                    tile = src.read(window=window, masked=True)

                    tile_float = tile.astype(np.float32)
                    tile_filled = tile_float.filled(np.nan)

                    valid_mask = ~np.ma.getmaskarray(tile)
                    if valid_mask.ndim == 3:
                        valid_mask = valid_mask.any(axis=0)

                    valid_fraction = valid_mask.mean()
                    if valid_fraction < 0.35:
                        continue

                    finite_vals = tile_filled[np.isfinite(tile_filled)]
                    if finite_vals.size == 0:
                        continue

                    # 🔥 ключевые фильтры
                    if np.nanstd(finite_vals) < 5:
                        continue

                    p2, p98 = np.nanpercentile(finite_vals, [2, 98])
                    if (p98 - p2) < 10:
                        continue

                    rgb = tile_to_rgb(tile)

                    if RESIZE_TO:
                        rgb = Image.fromarray(rgb).resize((RESIZE_TO, RESIZE_TO))
                    else:
                        rgb = Image.fromarray(rgb)

                    name = f"{region}_{modality}_{window.col_off}_{window.row_off}"

                    img_path = OUT_DIR / f"images/{split}/{name}.png"
                    lbl_path = OUT_DIR / f"labels/{split}/{name}.txt"

                    rgb.save(img_path)
                    lbl_path.write_text("\n".join(labels))

                    total += 1

    print("DONE:", total)


if __name__ == "__main__":
    main()