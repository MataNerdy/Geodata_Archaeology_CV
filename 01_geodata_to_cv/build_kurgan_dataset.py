import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import box
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from contextlib import ExitStack
from overlay import (
    read_target_crs_from_utm,
    find_li_kurgan_regions,
    find_ae_kurgan_regions,
    find_spor_kurgan_regions,
)
import warnings
from affine import Affine

warnings.filterwarnings("ignore", message="Several features with id", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)

DATASET_ROOT = Path("/Volumes/Lexar/Датасет/")

OUT_DIR = Path("dataset_multi_full_non_binary")
IMG_DIR = OUT_DIR / "images"
MASK_DIR = OUT_DIR / "masks"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = {
    "Li": find_li_kurgan_regions,
    "Ae": find_ae_kurgan_regions,
    "SpOr": find_spor_kurgan_regions,
}

MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 4096
CONTEXT_SCALE = 2.0
USE_TYPES = ["whole", "damaged"]

CLASS_TO_ID = {
    "background": 0,
    "whole": 1,
    "damaged": 2,
}

# --- utils ---

def polygon_intersects_raster(src, polygon):
    return polygon.intersects(box(*src.bounds))


def try_reproject_gdf(gdf, dst_crs):
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None or dst_crs is None:
        return gdf
    if gdf.crs != dst_crs:
        return gdf.to_crs(dst_crs)
    return gdf

PATCH_SIZE = 256


def extract_adaptive_patch_and_multi_mask(
    src,
    target_polygon,
    all_polygons_gdf,
    min_crop_size=256,
    max_crop_size=512,
    context_scale=2.0,
):
    # --- 1. размеры объекта в координатах карты ---
    minx, miny, maxx, maxy = target_polygon.bounds
    obj_w_map = maxx - minx
    obj_h_map = maxy - miny

    # --- 2. переводим размер объекта в пиксели ---
    # предполагаем обычный north-up raster без сильного skew
    px_w = abs(src.transform.a)
    px_h = abs(src.transform.e)

    if px_w <= 0 or px_h <= 0:
        raise ValueError("Invalid raster pixel size")

    obj_w_px = obj_w_map / px_w
    obj_h_px = obj_h_map / px_h

    # --- 3. считаем адаптивный размер crop ---
    crop_size = int(np.ceil(max(obj_w_px, obj_h_px) * context_scale))
    if crop_size > MAX_CROP_SIZE:
        print(f"[LARGE CROP] crop_size={crop_size}")
    crop_size = max(min_crop_size, crop_size)
    crop_size = min(max_crop_size, crop_size)

    # делаем crop_size четным
    if crop_size % 2 != 0:
        crop_size += 1



    # --- 4. центр объекта ---
    c = target_polygon.centroid
    cx, cy = c.x, c.y

    row_c, col_c = src.index(cx, cy)

    half = crop_size // 2
    row0 = row_c - half
    col0 = col_c - half
    row1 = row0 + crop_size
    col1 = col0 + crop_size

    # --- 5. пересечение с растром ---
    read_row0 = max(0, row0)
    read_col0 = max(0, col0)
    read_row1 = min(src.height, row1)
    read_col1 = min(src.width, col1)

    read_h = read_row1 - read_row0
    read_w = read_col1 - read_col0

    if read_h <= 0 or read_w <= 0:
        raise ValueError("Patch is completely outside raster")

    read_window = Window(
        col_off=read_col0,
        row_off=read_row0,
        width=read_w,
        height=read_h,
    )

    patch_raw = src.read(1, window=read_window)
    if patch_raw.size == 0:
        raise ValueError("Empty patch")

    # --- 6. паддинг до полного crop_size x crop_size ---
    patch = np.zeros((crop_size, crop_size), dtype=patch_raw.dtype)

    dst_row0 = read_row0 - row0
    dst_col0 = read_col0 - col0
    dst_row1 = dst_row0 + read_h
    dst_col1 = dst_col0 + read_w

    patch[dst_row0:dst_row1, dst_col0:dst_col1] = patch_raw

    # --- 7. affine transform полного окна ---
    x_left, y_top = src.xy(row0, col0, offset="ul")

    transform = Affine(
        src.transform.a, src.transform.b, x_left,
        src.transform.d, src.transform.e, y_top
    )

    # --- 8. геометрия окна ---
    x_right, y_bottom = rasterio.transform.xy(
        transform, crop_size, crop_size, offset="ul"
    )
    window_geom = box(x_left, y_bottom, x_right, y_top)

    intersecting = all_polygons_gdf[
        all_polygons_gdf.geometry.intersects(window_geom)
    ].copy()

    if len(intersecting) == 0:
        raise ValueError("No polygons in adaptive patch")

    items = []

    for _, r in intersecting.iterrows():
        value = CLASS_TO_ID.get(r["kurgan_type"], 0)
        if value == 0:
            continue

        geom = r.geometry
        if geom is None or geom.is_empty:
            continue

        items.append({
            "geometry": geom,
            "value": value,
            "area": geom.area,
            "kurgan_type": r["kurgan_type"],
        })

    # Большие объекты рисуем первыми, маленькие — последними.
    # Так маленькие курганы не затираются большими поврежденными областями.
    items = sorted(items, key=lambda x: x["area"], reverse=True)

    shapes = [
        (item["geometry"], item["value"])
        for item in items
    ]

    if not shapes:
        raise ValueError("No valid labeled polygons")

    mask = rasterize(
        shapes,
        out_shape=(crop_size, crop_size),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    if not np.any(mask > 0):
        raise ValueError("Empty mask")

    touches_border = (
        row0 < 0 or col0 < 0 or row1 > src.height or col1 > src.width
    )

    target_fits_inside = (
        obj_w_px <= crop_size and obj_h_px <= crop_size
    )

    return patch, mask, len(intersecting), touches_border, crop_size, obj_w_px, obj_h_px, target_fits_inside


# --- main loop ---

metadata_rows = []
global_idx = 0

for modality, find_fn in MODALITIES.items():

    print(f"\n=== {modality} ===")

    regions = find_fn(DATASET_ROOT)

    for region_info in regions:

        utm_path = region_info["utm_path"]
        target_crs = read_target_crs_from_utm(utm_path)

        # --- geojson ---
        geojson_files = []

        if "whole" in USE_TYPES:
            geojson_files += region_info.get("geojson_files_whole", [])
        if "damaged" in USE_TYPES:
            geojson_files += region_info.get("geojson_files_damaged", [])

        if not geojson_files:
            continue

        gdfs = []
        for f in geojson_files:
            gdf = gpd.read_file(f).to_crs(target_crs)
            gdf = gdf[["geometry"]].copy()
            gdf["source_file"] = f.name

            stem = f.stem.lower()
            if "целые" in stem:
                gdf["kurgan_type"] = "whole"
            elif "поврежденные" in stem:
                gdf["kurgan_type"] = "damaged"
            else:
                gdf["kurgan_type"] = "unknown"

            gdfs.append(gdf)

        all_gdf = pd.concat(gdfs, ignore_index=True)
        all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs=target_crs)

        all_gdf = all_gdf[
            (~all_gdf.geometry.is_empty) &
            (all_gdf.geometry.notna()) &
            all_gdf.geometry.is_valid
        ]

        # --- raster handling ---
        if modality == "SpOr":
            raster_list = region_info["raster_paths"]
        else:
            raster_list = [region_info["raster_path"]]

        with ExitStack() as stack:
            opened = [(rp, stack.enter_context(rasterio.open(rp))) for rp in raster_list]

            for idx, row in all_gdf.iterrows():
                polygon = row.geometry

                matched = None
                matched_gdf = all_gdf
                matched_row = row
                used_fallback = False

                # --- основной сценарий: polygon уже в target_crs / UTM ---
                for rp, src in opened:
                    if polygon_intersects_raster(src, polygon):
                        matched = (rp, src)
                        break

                # --- fallback: если не нашли, пробуем привести все к CRS конкретного растра ---
                if matched is None:
                    for rp, src in opened:
                        try:
                            all_gdf_alt = try_reproject_gdf(all_gdf, src.crs)
                            polygon_alt = all_gdf_alt.loc[idx, "geometry"]

                            if polygon_intersects_raster(src, polygon_alt):
                                matched = (rp, src)
                                matched_gdf = all_gdf_alt
                                matched_row = all_gdf_alt.loc[idx]
                                used_fallback = True
                                print(f"[FALLBACK CRS OK] {region_info['region_dir'].name} | {modality} | {rp.name} | polygon idx {idx}")
                                break
                        except Exception as e:
                            print(f"[FALLBACK CRS FAILED] {region_info['region_dir'].name} | {modality} | {rp.name} | polygon idx {idx}: {e}")

                if matched is None:
                    continue

                rp, src = matched

                try:
                    patch, mask, n_objs, touches_border, crop_size, obj_w_px, obj_h_px, target_fits_inside = extract_adaptive_patch_and_multi_mask(
                        src,
                        matched_row.geometry,
                        matched_gdf,
                        min_crop_size=MIN_CROP_SIZE,
                        max_crop_size=MAX_CROP_SIZE,
                        context_scale=CONTEXT_SCALE,
                    )

                    sample_id = f"{global_idx:06d}"

                    np.save(IMG_DIR / f"{sample_id}.npy", patch)
                    np.save(MASK_DIR / f"{sample_id}.npy", mask)

                    metadata_rows.append({
                        "sample_id": sample_id,
                        "region": region_info["region_dir"].name,
                        "modality": modality,
                        "raster_file": rp.name,
                        "kurgan_type": matched_row["kurgan_type"],
                        "n_objects_in_patch": n_objs,
                        "height": patch.shape[0],
                        "width": patch.shape[1],
                        "crop_size": crop_size,
                        "obj_w_px": float(obj_w_px),
                        "obj_h_px": float(obj_h_px),
                        "target_fits_inside": bool(target_fits_inside),
                        "touches_border": touches_border,
                        "used_crs_fallback": used_fallback,
                        "mask_bg_pixels": int((mask == 0).sum()),
                        "mask_whole_pixels": int((mask == 1).sum()),
                        "mask_damaged_pixels": int((mask == 2).sum()),
                        "has_whole": bool((mask == 1).any()),
                        "has_damaged": bool((mask == 2).any()),
                    })

                    global_idx += 1

                except Exception:
                    continue


# --- save ---
meta = pd.DataFrame(metadata_rows)
meta.to_csv(OUT_DIR / "metadata.csv", index=False)

print("DONE:", len(meta))
print(meta["modality"].value_counts())
print(meta["kurgan_type"].value_counts())

'''
DONE: 2491

modality
Ae      1004
SpOr     760
Li       727

kurgan_type
damaged    1822
whole       669
'''
