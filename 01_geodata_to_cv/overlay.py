from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

DATASET_ROOT = Path("/Volumes/Lexar/Датасет/")

MODALITY_ALIASES = {
    "li": "Li",
    "ae": "Ae",
    "ае": "Ae",
    "spor": "SpOr",
    "sp": "SpOr",
    "or": "Or",
}

COLOR_MAP = {
    ("Li", "whole"): "lime",
    ("Li", "damaged"): "green",
    ("Ae", "whole"): "magenta",
    ("Ae", "damaged"): "purple",
    ("SpOr", "whole"): "cyan",
    ("SpOr", "damaged"): "blue",
    ("Or", "whole"): "orange",
    ("Or", "damaged"): "red",
}

def reproject_to_raster_crs_if_needed(gdf, raster_crs):
    if gdf is None or gdf.empty:
        return gdf

    if gdf.crs is None or raster_crs is None:
        return gdf

    if gdf.crs != raster_crs:
        return gdf.to_crs(raster_crs)

    return gdf

def read_target_crs_from_utm(utm_json_path: Path) -> str:
    with open(utm_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")

def normalize_modality_name(text: str) -> str:
    text = text.lower()
    return (
        text.replace("с", "c")
            .replace("о", "o")
            .replace("р", "p")
    )


def is_spor_folder_name(name: str) -> bool:
    name_norm = normalize_modality_name(name)
    return ("_spor_" in name_norm) or name_norm.endswith("_spor")


def is_spor_geojson_name(stem: str) -> bool:
    stem_norm = normalize_modality_name(stem)
    return ("_spor_" in stem_norm)

def normalize_text(text: str) -> str:
    return text.lower().replace("с", "c").replace("о", "o").replace("р", "p")

def detect_modality(name: str) -> str:
    name = normalize_text(name)
    for key, val in MODALITY_ALIASES.items():
        if key in name:
            return val
    return "Unknown"

def find_regions(dataset_root: Path):
    results = []

    for region_dir in sorted(dataset_root.iterdir()):
        if not region_dir.is_dir():
            continue

        utm_path = region_dir / "UTM.json"
        if not utm_path.exists():
            continue
        raster_dirs = []
        markup_dir = None

        for sub in region_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "разметка" in name:
                markup_dir = sub
            else:
                modality = detect_modality(name)
                if modality != "Unknown":
                    raster_dirs.append((modality, sub))
        if markup_dir is None or not raster_dirs:
            continue
        results.append({
            "region_dir": region_dir,
            "utm_path": utm_path,
            "raster_dirs": raster_dirs,
            "markup_dir": markup_dir,
        })
    return results

def find_spor_kurgan_regions(dataset_root: Path):
    results = []

    for region_dir in sorted(dataset_root.iterdir()):
        if not region_dir.is_dir():
            continue

        utm_path = region_dir / "UTM.json"
        if not utm_path.exists():
            continue

        spor_dir = None
        markup_dir = None

        for sub in region_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if is_spor_folder_name(name):
                spor_dir = sub
            elif "разметка" in name:
                markup_dir = sub

        if spor_dir is None or markup_dir is None:
            continue

        spor_tifs = sorted(list(spor_dir.glob("*.tif")) + list(spor_dir.glob("*.tiff")))
        if not spor_tifs:
            continue

        spor_kurgan_geojsons = []
        spor_kurgan_whole_geojsons = []
        spor_kurgan_damaged_geojsons = []

        for gj in sorted(markup_dir.glob("*.geojson")):
            stem = gj.stem.lower()

            is_spor = is_spor_geojson_name(stem)
            is_whole = "курганы_целые" in stem
            is_damaged = "курганы_поврежденные" in stem
            is_kurgan = is_whole or is_damaged

            if is_spor and is_kurgan:
                spor_kurgan_geojsons.append(gj)

                if is_whole:
                    spor_kurgan_whole_geojsons.append(gj)
                elif is_damaged:
                    spor_kurgan_damaged_geojsons.append(gj)

        if spor_kurgan_geojsons:
            results.append({
                "region_dir": region_dir,
                "utm_path": utm_path,
                "raster_paths": spor_tifs,
                "geojson_files": spor_kurgan_geojsons,
                "geojson_files_whole": spor_kurgan_whole_geojsons,
                "geojson_files_damaged": spor_kurgan_damaged_geojsons,
                "n_geojson_whole": len(spor_kurgan_whole_geojsons),
                "n_geojson_damaged": len(spor_kurgan_damaged_geojsons),
            })

    return results

def find_li_kurgan_regions(dataset_root: Path):
    results = []

    for region_dir in sorted(dataset_root.iterdir()):
        if not region_dir.is_dir():
            continue

        utm_path = region_dir / "UTM.json"
        if not utm_path.exists():
            continue

        lidar_dir = None
        markup_dir = None

        for sub in region_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "li_карты" in name:
                lidar_dir = sub
            elif "разметка" in name:
                markup_dir = sub

        if lidar_dir is None or markup_dir is None:
            continue

        g_tif = None
        for tif in sorted(list(lidar_dir.glob("*.tif")) + list(lidar_dir.glob("*.tiff"))):
            stem = tif.stem.lower()
            if stem.endswith("_g") or "_g_" in stem:
                g_tif = tif
                break

        if g_tif is None:
            continue

        li_kurgan_geojsons = []
        li_kurgan_whole_geojsons = []
        li_kurgan_damaged_geojsons = []

        for gj in sorted(markup_dir.glob("*.geojson")):
            stem = gj.stem.lower()

            is_li = "_li_" in stem
            is_whole = "курганы_целые" in stem
            is_damaged = "курганы_поврежденные" in stem
            is_kurgan = is_whole or is_damaged

            if is_li and is_kurgan:
                li_kurgan_geojsons.append(gj)

                if is_whole:
                    li_kurgan_whole_geojsons.append(gj)
                elif is_damaged:
                    li_kurgan_damaged_geojsons.append(gj)

        if li_kurgan_geojsons:
            results.append({
                "region_dir": region_dir,
                "utm_path": utm_path,
                "raster_path": g_tif,
                "geojson_files": li_kurgan_geojsons,
                "geojson_files_whole": li_kurgan_whole_geojsons,
                "geojson_files_damaged": li_kurgan_damaged_geojsons,
                "n_geojson_whole": len(li_kurgan_whole_geojsons),
                "n_geojson_damaged": len(li_kurgan_damaged_geojsons),
            })

    return results

def is_ae_folder_name(name: str) -> bool:
    name = name.lower()
    return ("_ae_" in name) or ("_ае_" in name) or name.endswith("_ae") or name.endswith("_ае")


def is_ae_geojson_name(stem: str) -> bool:
    stem = stem.lower()
    return ("_ae_" in stem) or ("_ае_" in stem)


def find_ae_kurgan_regions(dataset_root: Path):
    results = []

    for region_dir in sorted(dataset_root.iterdir()):
        if not region_dir.is_dir():
            continue

        utm_path = region_dir / "UTM.json"
        if not utm_path.exists():
            continue

        ae_dir = None
        markup_dir = None

        for sub in region_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if is_ae_folder_name(name):
                ae_dir = sub
            elif "разметка" in name:
                markup_dir = sub

        if ae_dir is None or markup_dir is None:
            continue

        ae_tif = None
        ae_tifs = sorted(list(ae_dir.glob("*.tif")) + list(ae_dir.glob("*.tiff")))
        if ae_tifs:
            ae_tif = ae_tifs[0]

        if ae_tif is None:
            continue

        ae_kurgan_geojsons = []
        ae_kurgan_whole_geojsons = []
        ae_kurgan_damaged_geojsons = []

        for gj in sorted(markup_dir.glob("*.geojson")):
            stem = gj.stem.lower()

            is_ae = is_ae_geojson_name(stem)
            is_whole = "курганы_целые" in stem
            is_damaged = "курганы_поврежденные" in stem
            is_kurgan = is_whole or is_damaged

            if is_ae and is_kurgan:
                ae_kurgan_geojsons.append(gj)

                if is_whole:
                    ae_kurgan_whole_geojsons.append(gj)
                elif is_damaged:
                    ae_kurgan_damaged_geojsons.append(gj)

        if ae_kurgan_geojsons:
            results.append({
                "region_dir": region_dir,
                "utm_path": utm_path,
                "raster_path": ae_tif,
                "geojson_files": ae_kurgan_geojsons,
                "geojson_files_whole": ae_kurgan_whole_geojsons,
                "geojson_files_damaged": ae_kurgan_damaged_geojsons,
                "n_geojson_whole": len(ae_kurgan_whole_geojsons),
                "n_geojson_damaged": len(ae_kurgan_damaged_geojsons),
            })

    return results

def load_geojsons(markup_dir, target_crs):
    gdfs = []
    for f in markup_dir.glob("*.geojson"):
        stem = f.stem.lower()
        is_whole = "курганы_целые" in stem
        is_damaged = "курганы_поврежденные" in stem
        if not (is_whole or is_damaged):
            continue
        gdf = gpd.read_file(f).to_crs(target_crs)
        gdf = gdf[["geometry"]].copy()
        gdf["type"] = "whole" if is_whole else "damaged"
        gdfs.append(gdf)
    if not gdfs:
        return None
    return pd.concat(gdfs, ignore_index=True)

def test_region_overlay(region_info, save_dir="overlay_test"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    region_dir = region_info["region_dir"]
    utm_path = region_info["utm_path"]
    markup_dir = region_info["markup_dir"]
    raster_dirs = region_info["raster_dirs"]

    target_crs = read_target_crs_from_utm(utm_path)
    gdf = load_geojsons(markup_dir, target_crs)

    if gdf is None:
        return
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=target_crs)

    print("=" * 80)
    print(region_dir.name)
    print("Objects:", len(gdf))
    print(gdf["type"].value_counts())

    for modality, raster_dir in raster_dirs:
        raster_files = sorted(list(raster_dir.glob("*.tif")) + list(raster_dir.glob("*.tiff")))
        if not raster_files:
            continue

        if modality == "Li":
            g_files = [p for p in raster_files if p.stem.lower().endswith("_g")]
            raster_path = g_files[0] if g_files else raster_files[0]
        else:
            raster_path = raster_files[0]

        with rasterio.open(raster_path) as src:
            img = src.read(1, masked=True)
            bounds = src.bounds
            raster_bounds_geom = box(*src.bounds)
            raster_crs = src.crs

        # --- основной сценарий: как раньше, через UTM.json ---
        gdf_in_raster = gdf[gdf.geometry.intersects(raster_bounds_geom)].copy()

        # --- fallback: если не сработало, пробуем привести к CRS растра ---
        used_fallback = False
        if len(gdf_in_raster) == 0 and raster_crs is not None:
            try:
                gdf_fallback = reproject_to_raster_crs_if_needed(gdf, raster_crs)
                gdf_in_raster = gdf_fallback[gdf_fallback.geometry.intersects(raster_bounds_geom)].copy()

                if len(gdf_in_raster) > 0:
                    used_fallback = True
                    print(f"[FALLBACK CRS OK] {region_dir.name} | {modality} | {raster_path.name}")
                    print(f"  UTM CRS: {target_crs}")
                    print(f"  Raster CRS: {raster_crs}")
            except Exception as e:
                print(f"[FALLBACK CRS FAILED] {region_dir.name} | {modality} | {raster_path.name}: {e}")

        if len(gdf_in_raster) == 0:
            print(f"{region_dir.name} | {modality}: no objects intersect raster {raster_path.name}")
            print(f"  gdf.crs = {gdf.crs}")
            print(f"  raster.crs = {raster_crs}")
            print(f"  gdf bounds = {gdf.total_bounds}")
            print(f"  raster bounds = {bounds}")
            continue

        valid = img.compressed()
        if len(valid) == 0:
            print(f"{region_dir.name} | {modality}: raster {raster_path.name} is empty/masked")
            continue

        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(
            img,
            cmap="gray",
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            vmin=vmin,
            vmax=vmax
        )

        for idx, row in gdf_in_raster.iterrows():
            geom = row.geometry
            obj_type = row["type"]
            color = COLOR_MAP.get((modality, obj_type), "white")

            gpd.GeoSeries([geom], crs=gdf_in_raster.crs).plot(
                ax=ax,
                facecolor=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=2.5
            )

            # c = geom.centroid
            # ax.text(c.x, c.y, str(idx), fontsize=6, color="yellow")

        fallback_note = " | fallback to raster CRS" if used_fallback else ""
        ax.set_title(f"{region_dir.name} | {modality} | {raster_path.name}{fallback_note}")
        plt.tight_layout()

        out_path = save_dir / f"{region_dir.name}_{modality}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print("Saved:", out_path)


if __name__ == "__main__":
    regions = find_regions(DATASET_ROOT)

    print(f"Found {len(regions)} regions")

    TARGET_REGIONS = {
        "027_ТИМЕРЕВО",
        "005_ЛУБНО",
        "024_УСТЬ-РЕКА",
        "039_САРСКОЕ",
        "012_ЛИХУША",
    }

    selected = [
        r for r in regions
        if r["region_dir"].name in TARGET_REGIONS
    ]

    print("\nSelected regions:")
    for r in selected:
        print("-", r["region_dir"].name)

    for r in selected:
        test_region_overlay(
            r,
            save_dir="overlay_portfolio_examples"
        )