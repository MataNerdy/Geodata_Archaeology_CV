import rasterio
from rasterio.windows import from_bounds, Window
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

PADDING = 5
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


def extract_patch_and_multi_mask(src, target_polygon, all_polygons_gdf):
    minx, miny, maxx, maxy = target_polygon.bounds

    minx -= PADDING
    miny -= PADDING
    maxx += PADDING
    maxy += PADDING

    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    window = window.round_offsets().round_lengths()

    window = window.intersection(
        Window(0, 0, src.width, src.height)
    )

    patch = src.read(1, window=window)
    transform = src.window_transform(window)

    if patch.size == 0:
        raise ValueError("Empty patch")

    window_geom = box(*rasterio.windows.bounds(window, src.transform))

    intersecting = all_polygons_gdf[
        all_polygons_gdf.geometry.intersects(window_geom)
    ]

    if len(intersecting) == 0:
        raise ValueError("No polygons")

    shapes = []
    for _, r in intersecting.iterrows():
        value = CLASS_TO_ID.get(r["kurgan_type"], 0)
        shapes.append((r.geometry, value))

    mask = rasterize(
        shapes,
        out_shape=patch.shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    if not np.any(mask > 0):
        raise ValueError("Empty mask")

    return patch, mask, len(intersecting)


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
            all_gdf.geometry.notna() &
            (~all_gdf.geometry.is_empty) &
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
                    patch, mask, n_objs = extract_patch_and_multi_mask(
                        src, matched_row.geometry, matched_gdf
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
