import geopandas as gpd
import rasterio
import json
from pathlib import Path

path = "/Volumes/Lexar/Датасет/024_УСТЬ-РЕКА/04_Усть-река_SpOr_спутник/01_Усть-река_SpOr_спутник.tiff"
geojson_path = "/Volumes/Lexar/Датасет/024_УСТЬ-РЕКА/06_Усть-река_разметка/Усть-река_SpOr_курганы_поврежденные.geojson"
utm_json_path = "/Volumes/Lexar/Датасет/024_УСТЬ-РЕКА/UTM.json"

# 1. Читаем разметку
gdf = gpd.read_file(geojson_path)
print("Original gdf CRS:", gdf.crs)

def read_target_crs_from_utm(utm_json_path: Path) -> str:
    with open(utm_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")

target_crs = read_target_crs_from_utm(utm_json_path)

gdf_utm = gdf.to_crs(target_crs)
print("Reprojected gdf CRS:", gdf_utm.crs)
print("gdf_utm bounds:", gdf_utm.total_bounds)

# 3. Читаем raster
with rasterio.open(path) as src:
    img = src.read(1)
    bounds = src.bounds
    transform = src.transform
    print("raster crs =", src.crs)
    print("raster bounds =", src.bounds)

print("Raster bounds:", bounds)
print("Raster transform:", transform)
