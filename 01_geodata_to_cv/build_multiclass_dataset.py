import warnings
from pathlib import Path
from contextlib import ExitStack

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

from overlay_5_classes import (
    CLASS_TO_ID,
    read_target_crs_from_utm,
    find_regions,
    choose_raster_for_modality,
    load_geojsons,
)

warnings.filterwarnings("ignore", message="Several features with id", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)

DATASET_ROOT = Path("/Volumes/Lexar/Датасет/")

OUT_DIR = Path("dataset_5_classes_multiclass")
IMG_DIR = OUT_DIR / "images"
MASK_DIR = OUT_DIR / "masks"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 4096
CONTEXT_SCALE = 2.0

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
TARGET_CLASSES = [c for c in CLASS_TO_ID if c != "background"]


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


def get_raster_paths_for_modality(modality: str, raster_dir: Path):
    """
    Для Li берём только канал _g.
    Для SpOr оставляем все tif/tiff, потому что в регионе их может быть несколько.
    Для Ae/Or берём выбранный overlay-логикой первый tif/tiff.
    """
    raster_files = sorted(list(raster_dir.glob("*.tif")) + list(raster_dir.glob("*.tiff")))
    if not raster_files:
        return []

    if modality == "Li":
        g_files = [
            p for p in raster_files
            if p.stem.lower().endswith("_g") or "_g_" in p.stem.lower()
        ]
        return [g_files[0] if g_files else raster_files[0]]

    if modality == "SpOr":
        return raster_files

    chosen = choose_raster_for_modality(modality, raster_dir)
    return [chosen] if chosen is not None else []


def extract_adaptive_patch_and_multi_mask(
    src,
    target_polygon,
    all_polygons_gdf,
    min_crop_size=256,
    max_crop_size=4096,
    context_scale=2.0,
    sample_idx=None,
):
    # --- 1. размеры объекта в координатах карты ---
    minx, miny, maxx, maxy = target_polygon.bounds
    obj_w_map = maxx - minx
    obj_h_map = maxy - miny

    # --- 2. переводим размер объекта в пиксели ---
    px_w = abs(src.transform.a)
    px_h = abs(src.transform.e)

    if px_w <= 0 or px_h <= 0:
        raise ValueError("Invalid raster pixel size")

    obj_w_px = obj_w_map / px_w
    obj_h_px = obj_h_map / px_h

    # --- 3. считаем adaptive crop ---
    crop_size_raw = int(np.ceil(max(obj_w_px, obj_h_px) * context_scale))
    if crop_size_raw > max_crop_size:
        print(f"[LARGE CROP] sample_idx={sample_idx} crop_size_raw={crop_size_raw}")

    crop_size = max(min_crop_size, crop_size_raw)
    crop_size = min(max_crop_size, crop_size)

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

    # --- 5. пересечение окна с растром ---
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

    # --- 6. паддинг до crop_size x crop_size ---
    patch = np.zeros((crop_size, crop_size), dtype=patch_raw.dtype)

    dst_row0 = read_row0 - row0
    dst_col0 = read_col0 - col0
    dst_row1 = dst_row0 + read_h
    dst_col1 = dst_col0 + read_w

    patch[dst_row0:dst_row1, dst_col0:dst_col1] = patch_raw

    # --- 7. affine transform полного окна, включая паддинг ---
    x_left, y_top = src.xy(row0, col0, offset="ul")

    transform = Affine(
        src.transform.a, src.transform.b, x_left,
        src.transform.d, src.transform.e, y_top,
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

    shapes = []
    class_names_in_patch = set()

    for _, r in intersecting.iterrows():
        class_name = r.get("class_name")
        value = CLASS_TO_ID.get(class_name, 0)
        if value == 0:
            continue
        shapes.append((r.geometry, value))
        class_names_in_patch.add(class_name)

    if not shapes:
        raise ValueError("No valid labeled polygons")

    mask = rasterize(
        shapes,
        out_shape=(crop_size, crop_size),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    if not np.any(mask > 0):
        raise ValueError("Empty mask")

    touches_border = row0 < 0 or col0 < 0 or row1 > src.height or col1 > src.width
    target_fits_inside = obj_w_px <= crop_size and obj_h_px <= crop_size

    return (
        patch,
        mask,
        len(intersecting),
        sorted(class_names_in_patch),
        touches_border,
        crop_size,
        obj_w_px,
        obj_h_px,
        target_fits_inside,
    )


def main():
    metadata_rows = []
    global_idx = 0

    regions = find_regions(DATASET_ROOT)
    print(f"Found {len(regions)} regions")

    for region_info in regions:
        region_name = region_info["region_dir"].name
        utm_path = region_info["utm_path"]
        markup_dir = region_info["markup_dir"]
        raster_dirs = region_info["raster_dirs"]

        target_crs = read_target_crs_from_utm(utm_path)
        all_gdf = load_geojsons(markup_dir, target_crs)

        if all_gdf is None or all_gdf.empty:
            continue

        all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs=target_crs)
        all_gdf = all_gdf[
            all_gdf.geometry.notna()
            & (~all_gdf.geometry.is_empty)
            & all_gdf.geometry.is_valid
            & all_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            & all_gdf["class_name"].isin(TARGET_CLASSES)
        ].copy()

        if all_gdf.empty:
            continue

        print("=" * 80)
        print(region_name)
        print(all_gdf.groupby(["modality", "class_name"]).size())

        for modality, raster_dir in raster_dirs:
            gdf_modality = all_gdf[all_gdf["modality"] == modality].copy()
            if gdf_modality.empty:
                continue

            raster_list = get_raster_paths_for_modality(modality, raster_dir)
            if not raster_list:
                continue

            print(f"\n--- {region_name} | {modality} | rasters={len(raster_list)} | objects={len(gdf_modality)} ---")

            with ExitStack() as stack:
                opened = [(rp, stack.enter_context(rasterio.open(rp))) for rp in raster_list]

                for idx, row in gdf_modality.iterrows():
                    polygon = row.geometry

                    matched = None
                    matched_gdf = gdf_modality
                    matched_row = row
                    used_fallback = False

                    # основной сценарий: polygon уже в target_crs / UTM
                    for rp, src in opened:
                        if polygon_intersects_raster(src, polygon):
                            matched = (rp, src)
                            break

                    # fallback: если не нашли, пробуем привести все geojson этой modality к CRS конкретного растра
                    if matched is None:
                        for rp, src in opened:
                            try:
                                gdf_alt = try_reproject_gdf(gdf_modality, src.crs)
                                polygon_alt = gdf_alt.loc[idx, "geometry"]

                                if polygon_intersects_raster(src, polygon_alt):
                                    matched = (rp, src)
                                    matched_gdf = gdf_alt
                                    matched_row = gdf_alt.loc[idx]
                                    used_fallback = True
                                    print(
                                        f"[FALLBACK CRS OK] {region_name} | {modality} | "
                                        f"{rp.name} | polygon idx {idx}"
                                    )
                                    break
                            except Exception as e:
                                print(
                                    f"[FALLBACK CRS FAILED] {region_name} | {modality} | "
                                    f"{rp.name} | polygon idx {idx}: {e}"
                                )

                    if matched is None:
                        continue

                    rp, src = matched

                    try:
                        (
                            patch,
                            mask,
                            n_objs,
                            class_names_in_patch,
                            touches_border,
                            crop_size,
                            obj_w_px,
                            obj_h_px,
                            target_fits_inside,
                        ) = extract_adaptive_patch_and_multi_mask(
                            src,
                            matched_row.geometry,
                            matched_gdf,
                            min_crop_size=MIN_CROP_SIZE,
                            max_crop_size=MAX_CROP_SIZE,
                            context_scale=CONTEXT_SCALE,
                            sample_idx=global_idx,
                        )

                        sample_id = f"{global_idx:06d}"

                        np.save(IMG_DIR / f"{sample_id}.npy", patch)
                        np.save(MASK_DIR / f"{sample_id}.npy", mask)

                        row_meta = {
                            "sample_id": sample_id,
                            "region": region_name,
                            "modality": modality,
                            "raster_file": rp.name,
                            "source_file": matched_row.get("source_file", None),
                            "class_name": matched_row["class_name"],
                            "class_id": int(matched_row["class_id"]),
                            "class_label_ru": matched_row.get("class_label_ru", None),
                            "n_objects_in_patch": int(n_objs),
                            "classes_in_patch": ";".join(class_names_in_patch),
                            "height": int(patch.shape[0]),
                            "width": int(patch.shape[1]),
                            "crop_size": int(crop_size),
                            "obj_w_px": float(obj_w_px),
                            "obj_h_px": float(obj_h_px),
                            "target_fits_inside": bool(target_fits_inside),
                            "touches_border": bool(touches_border),
                            "used_crs_fallback": bool(used_fallback),
                            "mask_bg_pixels": int((mask == 0).sum()),
                        }

                        for class_name in TARGET_CLASSES:
                            class_id = CLASS_TO_ID[class_name]
                            row_meta[f"mask_{class_name}_pixels"] = int((mask == class_id).sum())
                            row_meta[f"has_{class_name}"] = bool((mask == class_id).any())

                        metadata_rows.append(row_meta)
                        global_idx += 1

                    except Exception as e:
                        # Для массовой сборки датасета лучше не падать на одном объекте.
                        # При отладке можно раскомментировать print ниже.
                        print(f"[SKIP] {region_name} | {modality} | idx={idx}: {e}")
                        continue

    meta = pd.DataFrame(metadata_rows)
    meta.to_csv(OUT_DIR / "metadata.csv", index=False)

    print("\nDONE:", len(meta))

    if len(meta) > 0:
        print("\nBy modality:")
        print(meta["modality"].value_counts())

        print("\nBy target class:")
        print(meta["class_name"].value_counts())

        print("\nMask pixels by class:")
        pixel_cols = ["mask_bg_pixels"] + [f"mask_{c}_pixels" for c in TARGET_CLASSES]
        print(meta[pixel_cols].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
