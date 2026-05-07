from pathlib import Path
import pandas as pd


ROOT = Path("/Volumes/Lexar/Датасет")
OUT_FILES_CSV = "tif_registry.csv"
OUT_SUMMARY_CSV = "tif_region_summary.csv"

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

TARGET_MODALITIES = {"MagGr", "Ae", "SpOr", "Li", "Or"}


def normalize_modality_token(token: str | None) -> str | None:
    if token is None:
        return None
    return MODALITY_ALIASES.get(token, None)


def extract_region_from_top_folder(folder_name: str) -> str:
    """
    Пример:
    '003_ЛУБНО_FINAL' -> 'ЛУБНО'
    '004_ДЕМИДОВКА' -> 'ДЕМИДОВКА'
    """
    parts = folder_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return folder_name


def detect_modality_from_path(file_path: Path) -> str | None:
    """
    Ищем модальность по имени файла и по именам родительских папок.
    Предпочитаем точные токены, разделенные '_' .
    """
    candidates = []

    # имя файла без расширения
    candidates.extend(file_path.stem.split("_"))

    # все папки по пути
    for part in file_path.parts:
        candidates.extend(str(part).split("_"))

    for token in candidates:
        norm = normalize_modality_token(token)
        if norm in TARGET_MODALITIES:
            return norm

    return None


def extract_map_type_full(file_path: Path, region_name: str) -> str:
    """
    Хотим фрагмент названия от региона до .tif без расширения.

    Пример:
    03_Лубно_Lidar_g.tif -> Лубно_Lidar_g
    01_Демидовка_SpOr.tif -> Демидовка_SpOr
    """
    stem = file_path.stem
    parts = stem.split("_")

    # убираем ведущий числовой префикс вроде 01_, 02_, 03_
    if parts and parts[0].isdigit():
        parts = parts[1:]

    return "_".join(parts)


def extract_map_suffix(file_path: Path, modality: str | None) -> str | None:
    """
    Для Li-карт полезно отдельно вытащить c / ch / g / i.
    Для остальных модальностей обычно suffix не нужен.
    """
    stem = file_path.stem
    parts = stem.split("_")

    if modality == "Li":
        last = parts[-1]
        if last in {"c", "ch", "g", "i"}:
            return last

    return None


records = []

for region_dir in sorted(ROOT.iterdir()):
    if not region_dir.is_dir():
        continue

    region_folder_name = region_dir.name
    region_name = extract_region_from_top_folder(region_folder_name)

    tif_files = sorted(region_dir.rglob("*.tif"))

    for tif_path in tif_files:
        modality = detect_modality_from_path(tif_path)
        map_type_full = extract_map_type_full(tif_path, region_name)
        map_suffix = extract_map_suffix(tif_path, modality)

        records.append({
            "region_folder": region_folder_name,
            "region": region_name,
            "file_name": tif_path.name,
            "file_path": str(tif_path),
            "parent_folder": str(tif_path.parent),
            "modality": modality,
            "map_type_full": map_type_full,
            "map_type_suffix": map_suffix,
            "exists": tif_path.exists(),
        })

tif_registry = pd.DataFrame(records)

if tif_registry.empty:
    print("No tif files found.")
else:
    tif_registry["modality"] = tif_registry["modality"].fillna("unknown")
    tif_registry.to_csv(OUT_FILES_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved file-level registry: {OUT_FILES_CSV}")
    print(f"Total tif files: {len(tif_registry)}")
    print()
    print("Counts by modality:")
    print(tif_registry["modality"].value_counts(dropna=False))
    print()
    print("Counts by region:")
    print(tif_registry["region"].value_counts())

    # --- summary по регионам ---
    summary_rows = []

    for region, group in tif_registry.groupby("region", dropna=False):
        modalities_present = sorted(
            m for m in group["modality"].dropna().unique()
            if m in TARGET_MODALITIES
        )

        li_variants = sorted(
            x for x in group.loc[group["modality"] == "Li", "map_type_suffix"].dropna().unique()
        )

        summary_rows.append({
            "region": region,
            "num_tif_total": len(group),
            "num_modalities_present": len(modalities_present),
            "modalities_present": ", ".join(modalities_present),
            "has_Li": "Li" in modalities_present,
            "has_Ae": "Ae" in modalities_present,
            "has_SpOr": "SpOr" in modalities_present,
            "has_Or": "Or" in modalities_present,
            "has_MagGr": "MagGr" in modalities_present,
            "num_Li_files": int((group["modality"] == "Li").sum()),
            "Li_suffixes": ", ".join(li_variants),
        })

    tif_summary = pd.DataFrame(summary_rows).sort_values("region")
    tif_summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved region summary: {OUT_SUMMARY_CSV}")
    print()
    print("Summary preview:")
    print(tif_summary.head(20))