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

# 5 целевых классов + фон для будущей segmentation mask
CLASS_TO_ID = {
    "background": 0,
    "kurgany_tselye": 1,
    "kurgany_povrezhdennye": 2,
    "gorodishcha": 3,
    "fortifikatsii": 4,
    "arkhitektury": 5,
}

CLASS_LABELS_RU = {
    "kurgany_tselye": "курганы_целые",
    "kurgany_povrezhdennye": "курганы_поврежденные",
    "gorodishcha": "городища",
    "fortifikatsii": "фортификации",
    "arkhitektury": "архитектуры",
}

COLOR_MAP = {
    "kurgany_tselye": "lime",
    "kurgany_povrezhdennye": "green",
    "gorodishcha": "yellow",
    "fortifikatsii": "cyan",
    "arkhitektury": "magenta",
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


def normalize_text(text: str) -> str:
    """
    Чуть-чуть защищаемся от смешения кириллицы/латиницы в названиях.
    Особенно важно для SpOr: там может быть кириллическая 'о/р/с'.
    """
    return (
        text.lower()
        .replace("с", "c")
        .replace("о", "o")
        .replace("р", "p")
    )


def detect_modality(name: str) -> str:
    name_norm = normalize_text(name)
    for key, val in MODALITY_ALIASES.items():
        if key in name_norm:
            return val
    return "Unknown"


def is_spor_folder_name(name: str) -> bool:
    name_norm = normalize_text(name)
    return ("_spor_" in name_norm) or name_norm.endswith("_spor")


def is_spor_geojson_name(stem: str) -> bool:
    stem_norm = normalize_text(stem)
    return "_spor_" in stem_norm


def is_ae_folder_name(name: str) -> bool:
    name = name.lower()
    return ("_ae_" in name) or ("_ае_" in name) or name.endswith("_ae") or name.endswith("_ае")


def is_ae_geojson_name(stem: str) -> bool:
    stem = stem.lower()
    return ("_ae_" in stem) or ("_ае_" in stem)


def is_or_name(name: str) -> bool:
    name_norm = normalize_text(name)
    return ("_or_" in name_norm) or name_norm.endswith("_or")


def is_li_name(name: str) -> bool:
    name_norm = normalize_text(name)
    return "_li_" in name_norm or name_norm.startswith("li_") or "li_карты" in name_norm


def detect_modality_from_geojson_name(stem: str) -> str:
    stem_norm = normalize_text(stem)
    stem_low = stem.lower()

    if "_li_" in stem_norm:
        return "Li"
    if is_ae_geojson_name(stem_low):
        return "Ae"
    if is_spor_geojson_name(stem):
        return "SpOr"
    if "_or_" in stem_norm:
        return "Or"

    return "Unknown"


def detect_class_from_geojson_name(stem: str):
    stem = stem.lower()

    if "курганы_целые" in stem:
        return "kurgany_tselye"
    if "курганы_поврежденные" in stem:
        return "kurgany_povrezhdennye"
    if "городища" in stem:
        return "gorodishcha"
    if "фортификации" in stem:
        return "fortifikatsii"
    if "архитектуры" in stem:
        return "arkhitektury"

    # object_poly / finds_points / прочее тут намеренно игнорируем
    return None


def find_regions(dataset_root: Path):
    """
    Универсальный поиск регионов:
    - регион должен иметь UTM.json
    - регион должен иметь папку Разметка
    - регион должен иметь хотя бы одну папку с растровыми данными Li/Ae/SpOr/Or
    """
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
                continue

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


def choose_raster_for_modality(modality: str, raster_dir: Path):
    raster_files = sorted(list(raster_dir.glob("*.tif")) + list(raster_dir.glob("*.tiff")))
    if not raster_files:
        return None

    # Для Li берём канал _g, как в курганном бейзлайне
    if modality == "Li":
        g_files = [p for p in raster_files if p.stem.lower().endswith("_g") or "_g_" in p.stem.lower()]
        return g_files[0] if g_files else raster_files[0]

    # Для остальных пока берём первый tif.
    # Для SpOr в некоторых регионах может быть несколько снимков; этот overlay — быстрая визуальная проверка.
    return raster_files[0]


def load_geojsons(markup_dir: Path, target_crs: str):
    """
    Загружаем только целевые polygon-классы:
    - kurgany_tselye
    - kurgany_povrezhdennye
    - gorodishcha
    - fortifikatsii
    - arkhitektury

    object_poly и finds_points не берём.
    """
    gdfs = []

    for f in sorted(markup_dir.glob("*.geojson")):
        stem = f.stem
        class_name = detect_class_from_geojson_name(stem)
        modality = detect_modality_from_geojson_name(stem)

        if class_name is None:
            continue
        if modality == "Unknown":
            continue

        gdf = gpd.read_file(f)
        if gdf.empty:
            continue

        gdf = gdf.to_crs(target_crs)
        gdf = gdf[["geometry"]].copy()
        gdf["class_name"] = class_name
        gdf["class_id"] = CLASS_TO_ID[class_name]
        gdf["class_label_ru"] = CLASS_LABELS_RU[class_name]
        gdf["modality"] = modality
        gdf["source_file"] = f.name

        # Берём только полигональные геометрии.
        # Points/LineString тут не нужны для segmentation baseline.
        gdf = gdf[
            gdf.geometry.notna()
            & (~gdf.geometry.is_empty)
            & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        if not gdf.empty:
            gdfs.append(gdf)

    if not gdfs:
        return None

    merged = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)


def test_region_overlay(region_info, save_dir="overlay_5_classes_test"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    region_dir = region_info["region_dir"]
    utm_path = region_info["utm_path"]
    markup_dir = region_info["markup_dir"]
    raster_dirs = region_info["raster_dirs"]

    target_crs = read_target_crs_from_utm(utm_path)
    gdf = load_geojsons(markup_dir, target_crs)

    if gdf is None or gdf.empty:
        print(f"{region_dir.name}: no target polygon objects")
        return

    print("=" * 80)
    print(region_dir.name)
    print("Objects total:", len(gdf))
    print(gdf.groupby(["modality", "class_name"]).size())

    for modality, raster_dir in raster_dirs:
        raster_path = choose_raster_for_modality(modality, raster_dir)
        if raster_path is None:
            continue

        # Важно: на растр конкретной modality кладём только geojson этой же modality
        gdf_modality = gdf[gdf["modality"] == modality].copy()
        if gdf_modality.empty:
            continue

        with rasterio.open(raster_path) as src:
            img = src.read(1, masked=True)
            bounds = src.bounds
            raster_bounds_geom = box(*src.bounds)
            raster_crs = src.crs

        # основной сценарий: geojson уже в UTM из UTM.json
        gdf_in_raster = gdf_modality[gdf_modality.geometry.intersects(raster_bounds_geom)].copy()

        # fallback: если не сработало, пробуем привести geojson к CRS конкретного растра
        used_fallback = False
        if len(gdf_in_raster) == 0 and raster_crs is not None:
            try:
                gdf_fallback = reproject_to_raster_crs_if_needed(gdf_modality, raster_crs)
                gdf_in_raster = gdf_fallback[
                    gdf_fallback.geometry.intersects(raster_bounds_geom)
                ].copy()

                if len(gdf_in_raster) > 0:
                    used_fallback = True
                    print(f"[FALLBACK CRS OK] {region_dir.name} | {modality} | {raster_path.name}")
                    print(f"  UTM CRS: {target_crs}")
                    print(f"  Raster CRS: {raster_crs}")

            except Exception as e:
                print(f"[FALLBACK CRS FAILED] {region_dir.name} | {modality} | {raster_path.name}: {e}")

        if len(gdf_in_raster) == 0:
            print(f"{region_dir.name} | {modality}: no target objects intersect raster {raster_path.name}")
            print(f"  gdf.crs = {gdf_modality.crs}")
            print(f"  raster.crs = {raster_crs}")
            print(f"  gdf bounds = {gdf_modality.total_bounds}")
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
            vmax=vmax,
        )

        for idx, row in gdf_in_raster.iterrows():
            geom = row.geometry
            class_name = row["class_name"]
            color = COLOR_MAP.get(class_name, "white")

            gpd.GeoSeries([geom], crs=gdf_in_raster.crs).plot(
                ax=ax,
                facecolor="none",
                edgecolor=color,
                linewidth=2,
            )

            c = geom.centroid
            ax.text(
                c.x,
                c.y,
                f"{row['class_id']}",
                fontsize=7,
                color="white",
            )

        fallback_note = " | fallback to raster CRS" if used_fallback else ""
        ax.set_title(
            f"{region_dir.name} | {modality} | {raster_path.name}{fallback_note}\n"
            f"{dict(gdf_in_raster['class_name'].value_counts())}"
        )
        plt.tight_layout()

        out_path = save_dir / f"{region_dir.name}_{modality}_{raster_path.stem}_5classes.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print("Saved:", out_path)


if __name__ == "__main__":
    regions = find_regions(DATASET_ROOT)

    print(f"Found {len(regions)} regions")

    for r in regions[:5]:
        test_region_overlay(r)
