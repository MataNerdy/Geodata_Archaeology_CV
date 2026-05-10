#  # Geospatial CV Pipeline for Archaeological Object Detection

Computer vision pipeline for archaeological object detection and segmentation from multimodal geodata.

## Что вообще происходит

Проект начался с археологических геоданных:

    LiDAR
    аэрофотосъемки
    спутниковых изображений
    .geojson разметки

Главная проблема:
    датасет изначально вообще не был CV-ready.

Нужно было:

    разбираться с CRS,
    совмещать растры и полигоны,
    строить overlay,
    генерировать patch/mask датасеты,
    собирать detection dataset.

## Overlay и проверка CRS


Одной из первых задач была проверка:
    совпадает ли геометрия объектов с растрами.

Использовались:

    rasterio
    geopandas
    shapely

<p align="center">
    <img src="assets/overlay_assets/img1.png" width="700">
    <img src="assets/overlay_assets/img2.png" width="700">
    <img src="assets/overlay_assets/img3.png" width="700">
    <img src="assets/overlay_assets/img4.png" width="700">
</p>

Некоторые растры имели несовпадающий CRS, поэтому пришлось реализовать fallback reprojection.

## Генерация segmentation dataset

Следующим этапом стала генерация patch/mask датасетов.

### Early baseline

Сначала использовался простой crop вокруг объекта.

```python
    patch, mask = extract_patch_and_mask(src, polygon, padding=5)
```

<p align="center">
    <img src="assets/patch.png" width="700">
</p>

### Adaptive crop extraction

Позже появился adaptive crop pipeline.

Идея:

    размер crop зависит от размера объекта.

```python
    crop_size = max(object_size * context_scale, min_crop_size)
```
<p align="center">
    <img src="assets/mask_assets/mask3.png" width="700">
    <img src="assets/mask_assets/mask1.png" width="700">
    <img src="assets/mask_assets/mask2.png" width="700">
    <img src="assets/mask_assets/mask4.png" width="700">
    <img src="assets/mask_assets/mask5.png" width="700">
    <img src="assets/mask_assets/mask6.png" width="700">
</p>

Это позволило:

    не терять маленькие объекты,
    сохранять контекст,
    избежать сильного ресайза.

### YOLO dataset generation

После segmentation pipeline был собран detection pipeline.

<p align="center">
    <img src="assets/bbox_assets/bbox1.png" width="700">
    <img src="assets/bbox_assets/bbox2.png" width="700">
    <img src="assets/bbox_assets/bbox3.png" width="700">
    <img src="assets/bbox_assets/bbox4.png" width="700">
    <img src="assets/bbox_assets/bbox5.png" width="700">
</p>

Одна из сложностей:

    огромное количество маленьких объектов на одном изображении.

### Используемые инструменты

    Python
    rasterio
    geopandas
    shapely
    numpy
    pandas
    matplotlib
    PyTorch
    YOLOv8
    Streamlit