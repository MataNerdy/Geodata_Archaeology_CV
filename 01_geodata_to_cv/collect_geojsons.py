import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.validation import make_valid


# корневая папка, где лежат данные
GEOJSON_ROOT = Path("/Volumes/Lexar/Датасет")
OUT_CSV = "objects_registry.csv"


CLASS_MAPPING = {
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_целые": "kurgany_tselye",
    "городища": "gorodishcha",
    "городище": "gorodishcha",
    "фортификации": "fortifikatsii",
    "фортификация": "fortifikatsii",
    "архитектуры": "arkhitektury",
    "архитектура": "arkhitektury",
    "FindsPoints": "finds_points",
    "ObjectPoly": "object_poly",
}

TARGET_CLASSES = {
    "kurgany_povrezhdennye",
    "kurgany_tselye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
    "finds_points",
    "object_poly",
}

MODALITY_ALIASES = {
    "Li": "Li",
    "Ae": "Ae",
    "Ае": "Ae",   # кириллица
    "SpOr": "SpOr",
    "Sp": "SpOr",
    "Or": "Or",
    "MagGr": "MagGr",
    "GeoGr": "GeoGr",
}


def extract_class_from_filename(name: str) -> str | None:
    base = Path(name).stem

    if "курганы_поврежденные" in base:
        return "курганы_поврежденные"
    if "курганы_целые" in base:
        return "курганы_целые"

    return base.rsplit("_", 1)[-1]


def normalize_class(raw: str | None) -> str | None:
    if raw is None:
        return None
    return CLASS_MAPPING.get(raw, None)


def extract_region_and_modality(file_path: Path) -> tuple[str | None, str | None]:
    name = file_path.stem
    parts = name.split("_")

    modality = None
    modality_idx = None

    for i, part in enumerate(parts):
        if part in MODALITY_ALIASES:
            modality = MODALITY_ALIASES[part]
            modality_idx = i
            break

    if modality is None:
        return None, None

    region = "_".join(parts[:modality_idx]) if modality_idx > 0 else None
    return region, modality


def split_geometry(geom):
    if geom is None or geom.is_empty:
        return []

    try:
        geom = make_valid(geom)
    except Exception:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, Point):
        return [geom]

    # если потом встретятся другие типы, пока пропускаем
    return []


rows = []

if not GEOJSON_ROOT.exists():
    raise FileNotFoundError(f"Root folder does not exist: {GEOJSON_ROOT}")

files = sorted(GEOJSON_ROOT.rglob("*.geojson"))
print(f"Found geojson files: {len(files)}")

for file_path in files:
    file_name = file_path.name.lower()

    # пропускаем служебные/нецелевые файлы
    if "границы" in file_name:
        continue

    class_raw = extract_class_from_filename(file_path.name)
    class_norm = normalize_class(class_raw)

    if class_norm not in TARGET_CLASSES:
        continue

    region, modality = extract_region_and_modality(file_path)

    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    if gdf.empty:
        continue

    for obj_idx, obj_row in gdf.iterrows():
        geom = obj_row.geometry

        if geom is None or geom.is_empty:
            continue

        parts = split_geometry(geom)

        for part_id, part in enumerate(parts):
            if part.is_empty:
                continue

            minx, miny, maxx, maxy = part.bounds
            centroid = part.centroid

            rows.append({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "region": region,
                "modality": modality,
                "class_raw": class_raw,
                "class_norm": class_norm,
                "geometry_type": part.geom_type,
                "object_id_in_file": obj_idx,
                "part_id": part_id,
                "area_m2": part.area,
                "perimeter_m": part.length,
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
            })

registry = pd.DataFrame(rows)

if registry.empty:
    print("Registry is empty. Check GEOJSON_ROOT and filename parsing.")
else:
    registry["modality"] = registry["modality"].fillna("unknown")
    registry["region"] = registry["region"].fillna("unknown")
    registry.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_CSV}")
    print(f"Total objects: {len(registry)}")
    print()
    print("Counts by class:")
    print(registry["class_norm"].value_counts())
    print()
    print("Counts by modality:")
    print(registry["modality"].value_counts(dropna=False))
    print()
    print("Counts by region:")
    print(registry["region"].value_counts().head(20))