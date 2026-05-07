import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from overlay import read_target_crs_from_utm, find_ae_kurgan_regions
import matplotlib.pyplot as plt

DATASET_ROOT = Path("/Volumes/Lexar/Датасет/")


def extract_patch_and_mask(src, polygon, padding=5):
    """
    src: rasterio dataset (open)
    polygon: shapely geometry in raster CRS
    padding: padding in meters (UTM)
    """
    minx, miny, maxx, maxy = polygon.bounds

    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding

    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    window = window.round_offsets().round_lengths()

    full_window = Window(
        col_off=0,
        row_off=0,
        width=src.width,
        height=src.height
    )
    window = window.intersection(full_window)

    patch = src.read(1, window=window)
    transform = src.window_transform(window)

    mask = rasterize(
        [(polygon, 1)],
        out_shape=patch.shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    if patch.size == 0 or mask.size == 0:
        raise ValueError("Empty patch or mask")

    if mask.sum() == 0:
        raise ValueError("Empty mask")

    return patch, mask


regions = find_ae_kurgan_regions(DATASET_ROOT)

OUT_DIR = Path("dataset")
IMG_DIR = OUT_DIR / "images"
MASK_DIR = OUT_DIR / "masks"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

metadata_rows = []
global_idx = 0

for region_info in regions:
    raster_path = region_info["raster_path"]
    utm_path = region_info["utm_path"]
    # можно переключать источник:
    # geojson_files = region_info["geojson_files"]            # все
    # geojson_files = region_info["geojson_files_whole"]      # только целые
    # geojson_files = region_info["geojson_files_damaged"]    # только поврежденные

    geojson_files = region_info["geojson_files_whole"]

    if not geojson_files:
        print(f"{region_info['region_dir'].name}: no whole kurgans, skip")
        continue

    target_crs = read_target_crs_from_utm(utm_path)

    gdfs = []
    for f in geojson_files:
        gdf = gpd.read_file(f).to_crs(target_crs)
        gdf = gdf[["geometry"]].copy()
        gdf["source_file"] = f.name

        stem = f.stem.lower()
        if "курганы_целые" in stem:
            gdf["kurgan_type"] = "whole"
        elif "курганы_поврежденные" in stem:
            gdf["kurgan_type"] = "damaged"
        else:
            gdf["kurgan_type"] = "unknown"

        gdfs.append(gdf)

    all_gdf = pd.concat(gdfs, ignore_index=True)
    all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs=target_crs)

    # чтобы не было warning
    geom = all_gdf.geometry
    all_gdf = all_gdf[
        geom.notna() &
        (~geom.is_empty) &
        geom.is_valid
    ].copy()

    print(region_info["region_dir"].name)
    print(all_gdf["kurgan_type"].value_counts())

    patches = []
    masks = []

    with rasterio.open(raster_path) as src:
        for idx, row in all_gdf.iterrows():
            polygon = row.geometry
            kurgan_type = row["kurgan_type"]
            source_file = row["source_file"]

            try:
                patch, mask = extract_patch_and_mask(src, polygon, padding=5)

                patches.append(patch)
                masks.append(mask)

                sample_id = f"{global_idx:06d}"

                np.save(IMG_DIR / f"{sample_id}.npy", patch)
                np.save(MASK_DIR / f"{sample_id}.npy", mask)

                metadata_rows.append({
                    "sample_id": sample_id,
                    "region": region_info["region_dir"].name,
                    "raster_file": raster_path.name,
                    "source_file": source_file,
                    "polygon_idx": idx,
                    "kurgan_type": kurgan_type,
                    "height": patch.shape[0],
                    "width": patch.shape[1],
                    "mask_sum": int(mask.sum()),
                    "polygon_area": float(polygon.area),
                })

                global_idx += 1

            except Exception as e:
                print(
                    f"{region_info['region_dir'].name} | "
                    f"{kurgan_type} | {source_file} | "
                    f"skip polygon {idx}: {e}"
                )

    print(f"{region_info['region_dir'].name}: {len(patches)} patches")
    print("Saved samples:", global_idx)

    if len(patches) == 0:
        continue

    i = 0

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Patch")
    plt.imshow(patches[i], cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(masks[i], cmap="gray")

    plt.tight_layout()
    plt.savefig(f"{region_info['region_dir'].name}_patch_mask.png", dpi=150, bbox_inches="tight")
    plt.close()

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(OUT_DIR / "metadata.csv", index=False, encoding="utf-8")

print("Saved metadata:", OUT_DIR / "metadata.csv")
print("Total samples:", len(metadata_df))
print(metadata_df["kurgan_type"].value_counts())

meta = pd.read_csv("dataset/metadata.csv")

print(meta["kurgan_type"].value_counts())
print(meta.groupby("kurgan_type")[["height", "width", "mask_sum", "polygon_area"]].describe())
