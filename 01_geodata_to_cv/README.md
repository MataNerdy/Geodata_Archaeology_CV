# Geodata Archaeology CV

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

![LiDAR fortifications](assets/overlay_assets/img7.png)

![Пример CRS fallback](assets/overlay_assets/img2.png)

Некоторые растры имели несовпадающий CRS, поэтому пришлось реализовать fallback reprojection.

## Генерация segmentation dataset

Следующим этапом стала генерация patch/mask датасетов.

### Early baseline

Сначала использовался простой crop вокруг объекта.

```python
    patch, mask = extract_patch_and_mask(src, polygon, padding=5)
```

![Пример patch + mask](assets/patch.png)

### Adaptive crop extraction

Позже появился adaptive crop pipeline.

Идея:

    размер crop зависит от размера объекта.

```python
    crop_size = max(object_size * context_scale, min_crop_size)
```

![LiDAR fortifications](assets/mask_assets/mask4.png)

![Aerial imagery kurgans and fortifications](assets/mask_assets/mask1.png)

![multiclass_neighbors](assets/mask_assets/mask5.png)

Это позволило:

    не терять маленькие объекты,
    сохранять контекст,
    избежать сильного ресайза.

### YOLO dataset generation

После segmentation pipeline был собран detection pipeline.

![Dense bbox scene](assets/bbox_assets/bbox2.png)
![Multi-object scenes](assets/bbox_assets/bbox3.png)

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