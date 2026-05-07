Computer Vision, Geospatial Data Processing, PyTorch, Raster/Vector Alignment,
Semantic Segmentation, Object Detection, YOLOv8, U-Net, DeepLabV3+,
GeoTIFF, GeoJSON, Rasterio, Streamlit, Dataset Engineering,
Threshold Optimization, Region-aware Validation



# Geodata → Computer Vision Pipeline

## Goal

Convert archaeological geospatial data into CV-ready datasets for segmentation and detection tasks.

## Input Data

- GeoTIFF rasters
- GeoJSON polygon annotations
- UTM-based CRS metadata

Modalities:
- LiDAR
- aerial imagery
- satellite imagery
- orthophoto

## Main Challenges

- inconsistent CRS
- raster/vector misalignment
- heterogeneous modality naming
- extremely different object scales
- edge truncation
- noisy annotations

## Implemented Pipeline

1. Raster inventory generation
2. GeoJSON parsing and validation
3. CRS-aware reprojection
4. Polygon overlay verification
5. Adaptive patch extraction
6. Polygon rasterization
7. Multiclass mask generation
8. Metadata registry creation

## Key Features

- CRS fallback strategy
- adaptive crop scaling
- multimask generation
- multi-object patches
- modality-aware raster selection

## Output

Generated:
- segmentation datasets
- multiclass masks
- metadata registries
- visualization overlays



# Geodata → Computer Vision Pipeline

Pipeline for converting archaeological geospatial data into computer-vision-ready datasets for segmentation and object detection tasks.

This stage focuses on:
- raster/vector alignment,
- CRS-aware preprocessing,
- adaptive patch extraction,
- polygon rasterization,
- multimodal dataset generation.

The resulting datasets were later used for:
- binary segmentation (U-Net),
- multiclass segmentation (DeepLabV3+),
- YOLO object detection.

---

# Task

The original dataset consists of:

- GeoTIFF raster maps
- GeoJSON polygon annotations
- UTM-based CRS metadata

The goal was to transform heterogeneous geospatial data into ML-ready datasets while preserving spatial consistency.

Target archaeological classes:

- kurgany_tselye
- kurgany_povrezhdennye
- gorodishcha
- fortifikatsii
- arkhitektury

Supported modalities:

- LiDAR (`Li`)
- aerial imagery (`Ae`)
- satellite imagery (`SpOr`)
- orthophoto (`Or`)

---

# Main Challenges

The dataset contains multiple real-world geospatial issues:

- inconsistent CRS between rasters and GeoJSON
- mixed UTM / EPSG projections
- inconsistent modality naming
- different raster resolutions
- extremely different object scales
- edge truncation during tiling
- noisy annotations
- heterogeneous modalities

Additionally:
- some rasters use local CRS,
- some GeoJSON files contain invalid geometries,
- modality names sometimes mix Cyrillic and Latin characters.

---

# Implemented Pipeline

## 1. Raster inventory generation

Scripts:
- `collect_tifs.py`

Implemented:
- recursive raster discovery
- modality detection
- region extraction
- raster metadata registry
- LiDAR channel parsing (`_g`, `_c`, `_ch`, `_i`)

Generated:
- `tif_registry.csv`
- `tif_region_summary.csv`

---

## 2. GeoJSON parsing and object registry

Scripts:
- `collect_geojsons.py`

Implemented:
- GeoJSON parsing
- geometry validation
- MultiPolygon splitting
- class normalization
- modality extraction
- geometric statistics

Generated metadata:
- object area
- perimeter
- centroid
- bounding box
- modality
- region
- class labels

Generated:
- `objects_registry.csv`

---

# CRS Handling

Scripts:
- `research_utm.py`
- `research_tif.py`

Implemented:
- reading CRS from `UTM.json`
- GeoJSON reprojection
- raster CRS inspection
- fallback reprojection to raster CRS

Main idea:

```python
GeoJSON -> target UTM CRS
if failed:
    GeoJSON -> raster CRS
```

This fallback strategy allowed processing regions with broken or inconsistent metadata.

---

# Overlay Validation

Scripts:
- `overlay.py`
- `overlay_5_classes.py`

Implemented:
- polygon overlay on raster maps
- modality-aware raster selection
- multiclass visualization
- raster intersection validation
- CRS fallback debugging

The overlay stage was used for:
- validating spatial alignment,
- debugging CRS issues,
- inspecting annotation quality,
- checking modality consistency.

---

# Dataset Generation

## Binary segmentation dataset

Script:
- `build_binary_dataset.py`

Implemented:
- adaptive patch extraction
- polygon-centered crops
- multimask generation
- rasterization
- border handling
- metadata generation

Classes:
- background
- whole
- damaged

Output:
- `.npy` image patches
- `.npy` masks
- `metadata.csv`

---

## Multiclass segmentation dataset

Script:
- `build_multiclass_dataset.py`

Implemented:
- multiclass rasterization
- adaptive crop scaling
- multi-object masks
- patch metadata tracking

Classes:
- background
- kurgany_tselye
- kurgany_povrezhdennye
- gorodishcha
- fortifikatsii
- arkhitektury

Key feature:
multiple neighboring objects are preserved inside a patch instead of using single-object masks.

---

## YOLO detection dataset

Script:
- `build_yolo_detection_dataset.py`

Implemented:
- adaptive raster tiling
- polygon clipping
- polygon → bbox conversion
- bbox normalization
- tile quality filtering
- negative sampling
- edge-object filtering
- region-aware train/val split

Generated:
- YOLO labels
- PNG tiles
- `dataset.yaml`
- metadata tables

Additional features:
- modality-aware context scaling
- automatic tile size selection
- tile quality heuristics
- metadata tracking for debugging

---

# Key Technical Features

## CRS-aware preprocessing

The pipeline automatically handles:
- UTM projections,
- raster CRS mismatch,
- reprojection fallback strategies.

---

## Adaptive crop extraction

Patch size is dynamically selected based on object scale:

```python
crop_size ≈ object_size × context_scale
```

This allows handling:
- very small mounds,
- large archaeological structures,
- multimodal resolution differences.

---

## Multi-object masks

Instead of:
```text
one polygon -> one isolated mask
```

the pipeline preserves neighboring objects inside the crop.

This reduces false penalties during segmentation training and better reflects real archaeological scenes.

---

## Modality-aware preprocessing

Different modalities require different handling:

- LiDAR → relief-oriented processing
- aerial imagery → noisy textures
- satellite imagery → large-scale context
- orthophoto → high-detail local structures

The pipeline automatically adapts:
- raster selection,
- tile size,
- context scaling.

---

# Result

The pipeline successfully converts raw geospatial archaeological data into:

- segmentation datasets,
- multiclass masks,
- YOLO detection datasets,
- object registries,
- overlay validation tools,
- metadata tables.

This preprocessing stage became the foundation for all subsequent CV experiments in the project.

---

# Example Pipeline

```text
GeoTIFF + GeoJSON
        ↓
CRS alignment
        ↓
Overlay validation
        ↓
Adaptive crop extraction
        ↓
Rasterization / bbox generation
        ↓
CV-ready dataset
```

---

# Technologies

- Python
- GeoPandas
- Rasterio
- Shapely
- NumPy
- Pandas
- PIL
- Matplotlib

---

# Related Stages

This preprocessing pipeline was later used in:

- `02_binary_segmentation_unet`
- `03_multiclass_segmentation_deeplab`
- `04_detection_yolo`
