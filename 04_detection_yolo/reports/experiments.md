# Notebook Audit

Source notebook: `notebooks/geo_li_ae_kurgan_detection.ipynb`

Additional local notes inspected: `results.txt`, `remarks.txt`

Saved YOLO artifacts inspected: `runs/runs_2`, `runs/runs_4`

## Реализовано в ноутбуке

### Подготовка датасета

- Mount Google Drive in Colab.
- Unzip `/content/drive/MyDrive/Share/Geodata/dataset_yolo_bbox.zip` into `/content/dataset`.
- Verify `images/train`, `images/val`, `labels/train`, `labels/val`, and `dataset.yaml`.
- Rewrite the initial `dataset.yaml` for five classes:
  - `kurgany_tselye`
  - `kurgany_povrezhdennye`
  - `gorodishcha`
  - `fortifikatsii`
  - `arkhitektury`
- Load `metadata.csv`; notebook output shows initial metadata shape `(7418, 36)`.

### Фильтрация v2

- Input: `/content/dataset/dataset_yolo_bbox`
- Output: `/content/dataset/dataset_yolo_bbox_v2_kurgans_li_ae`
- Keeps modalities: `Li`, `Ae`
- Keeps old class ids: `0`, `1`
- Class mapping:
  - `0 -> kurgany_tselye`
  - `1 -> kurgany_povrezhdennye`
- Validates YOLO boxes with normalized coordinates in `[0, 1]`.
- Deduplicates metadata to one row per image before copying.
- Negative sampling: `NEGATIVE_RATIO = 0.25`
- Random seed: `42`
- Writes filtered labels, metadata, and `dataset.yaml`.

v2 output from notebook:

- images copied: `635`
- positive images: `368`
- negative images: `267`
- total boxes: `4288`
- bad boxes skipped: `0`
- missing images skipped: `0`
- boxes by class:
  - `kurgany_povrezhdennye`: `3275`
  - `kurgany_tselye`: `1013`

### Обучение YOLO v2

- Model: `yolov8n.pt`
- Data: `/content/dataset/dataset_yolo_bbox_v2_kurgans_li_ae/dataset.yaml`
- `imgsz = 1024`
- `epochs = 60`
- `batch = 8`
- `patience = 20`
- `cos_lr = True`
- `workers = 2`
- `cache = True`
- `close_mosaic = 10`
- Saved local artifact equivalent: `runs/runs_2`

### Фильтрация v3 / v4

The notebook contains the v4 filtering code. `remarks.txt` describes v3 as an intermediate over-cleaned dataset.

v4 notebook parameters:

- Input: `/content/dataset/dataset_yolo_bbox`
- Output: `/content/dataset/dataset_yolo_bbox_v4_kurgans_li_ae_balanced`
- Keeps modalities: `Li`, `Ae`
- Keeps classes: kurgan classes only
- Random seed: `42`
- Negative sampling: `NEGATIVE_RATIO = 0.15`
- `DROP_EDGE_OBJECTS = False` (declared but not directly applied as a drop switch)
- Edge-ratio filter: skip image if mean `bbox_touches_tile_edge > 0.8`
- `MAX_OBJECTS = 20`
- `MIN_VALID_FRACTION = 0.25`
- `MIN_CONTRAST = 3` using `tile_p98_minus_p2`

v4 output from notebook:

- images copied: `347`
- positive images: `226`
- negative images: `121`
- total boxes: `772`
- bad boxes skipped: `0`
- missing images skipped: `0`
- boxes by class:
  - `kurgany_povrezhdennye`: `430`
  - `kurgany_tselye`: `342`

### Обучение YOLO v4

- Model: `yolov8s.pt`
- Data: `/content/dataset/dataset_yolo_bbox_v4_kurgans_li_ae_balanced/dataset.yaml`
- `imgsz = 1024`
- `epochs = 80`
- `batch = 8`
- `patience = 25`
- `cos_lr = True`
- `workers = 2`
- `cache = True`
- `close_mosaic = 15`
- Run name: `kurgans_li_ae_v4_yolov8s_balanced`
- Saved local artifact equivalent: `runs/runs_4`

### Валидация

- Validation is handled by Ultralytics during training with `val=True` and `split=val`.
- Saved artifacts include `results.csv`, `results.png`, confusion matrices, confidence curves, and validation batch predictions.
- The notebook does not contain a clean standalone validation script yet.

### Визуализация

- Notebook relies mainly on Ultralytics-generated plots and validation batches.
- Existing local visualization code is in `skripts/visualize_yolo_labels.py`, a Streamlit bbox dataset viewer.
- Existing overlay/debug code is in `skripts/overlay_5_classes.py`.

### Анализ результатов

The strongest analysis is currently in `results.txt` and `remarks.txt`:

- v2 works but misses many objects.
- The detector is conservative: moderate precision, low recall.
- LiDAR provides stronger structural signal than aerial imagery.
- `kurgany_tselye` is easier than `kurgany_povrezhdennye`.
- Aggressive cleaning improves precision but can hurt recall.
- Threshold tuning did not recover recall; the bottleneck is data quality, annotation ambiguity, and object difficulty.

## Эксперименты

| Version | Dataset idea | Model | Key filters | Metrics | Main conclusion |
|---|---|---|---|---|---|
| v1 | Initial 5-class multimodal bbox dataset | TODO: confirm | Five classes; Li/Ae/SpOr/Or in generation scripts | TODO: recover if needed | Too noisy and imbalanced for stable training. |
| v2 | Kurgan-only Li + Ae baseline | YOLOv8n | Keep classes 0/1; keep Li/Ae; negative ratio 0.25 | Notes: mAP50 `0.182`, precision approx. `0.32`, recall approx. `0.23`; per-class AP `tselye` `0.286`, `povrezhdennye` `0.078`. `runs_2` best row: P `0.43985`, R `0.18478`, mAP50 `0.18323`, mAP50-95 `0.12095`. | Model works but is recall-limited and weak on damaged kurgans. |
| v3 | Cleaned dataset | YOLOv8s | Partial edge filtering, `MIN_VALID_FRACTION`, `MIN_CONTRAST`, `MAX_OBJECTS`, negative downsampling | Notes: images `313`, boxes `539`; mAP50 approx. `0.17`, precision approx. `0.455`, recall approx. `0.162` | Over-cleaning increases precision and reduces recall. |
| v4 | Balanced cleaned Li + Ae kurgan dataset | YOLOv8s | Edge-ratio filter, `MIN_VALID_FRACTION = 0.25`, `MIN_CONTRAST = 3`, `MAX_OBJECTS = 20`, negative ratio `0.15`, class balancing | Notes: images `347`, boxes `772`; mAP50 approx. `0.198`, precision approx. `0.478`, recall approx. `0.195`; AP `tselye` `0.246`, AP `povrezhdennye` `0.151`. `runs_4` best row: P `0.54479`, R `0.18838`, mAP50 `0.22443`, mAP50-95 `0.12145`. | Best balanced variant; damaged-kurgan AP improved. |
| v5 | Threshold tuning on v4-style model | YOLOv8s / v4 | confidence threshold tests at `0.10` and `0.15` | conf `0.10`: mAP50 approx. `0.187`, R approx. `0.195`, AP `povrezhdennye` approx. `0.130`; conf `0.15`: mAP50 approx. `0.178`, R approx. `0.195`, AP `povrezhdennye` approx. `0.111` | Lower thresholds did not improve recall; threshold is not the main bottleneck. |

## Зафиксированный baseline после dataset ablation

После абляции датасета текущий controlled baseline:

```text
dataset = dataset_yolo_bbox_v3b_li_binary_medium
model = yolov8n.pt
imgsz = 640
epochs = 100
single_cls = True
seed = 42
close_mosaic = 10
patience = 25
```

| Experiment | Model | Train / Val | Precision | Recall | mAP50 | mAP50-95 | Best epoch | Conclusion |
|---|---|---|---:|---:|---:|---:|---:|---|
| `v3b_100_epoch_baseline` | YOLOv8n | Li / Li | 0.70544 | 0.28571 | 0.33904 | 0.11516 | 84 | Current best controlled Li-only baseline. |
| `v3d_li_ae_medium` | YOLOv8n | Li+Ae / Li+Ae | 0.35951 | 0.21348 | 0.16164 | 0.06106 | 76 | Larger mixed-modality dataset is noisier and worse. |
| `v3e_train_li_ae_val_li` | YOLOv8n | Li+Ae / Li | 0.46538 | 0.26531 | 0.25010 | 0.07726 | 60 | Ae in train does not improve Li validation quality. |
| `v3b_400_epoch_limit` | YOLOv8n | Li / Li | 0.51425 | 0.26531 | 0.27816 | 0.09593 | 74 | Longer epoch limit did not help; early stopped at 105 epochs. |
| `v3b_yolo26n_400_epoch_limit` | YOLO26n | Li / Li | 0.52136 | 0.20408 | 0.22218 | 0.09721 | 37 | YOLO26n did not improve the baseline; early stopped at 63 epochs. |

Final decision: keep `v3b_li_medium + YOLOv8n + 100 epochs` as the baseline. The next useful experiments should target image scale (`imgsz = 1024`), a slightly larger model, or dataset/label improvements rather than more epochs.

## Код, который надо вынести в scripts/src

- Dataset filtering from notebook cell 6 into `src/dataset/filter_kurgans.py` or `scripts/make_dataset_v2.py`.
- v4 filtering from notebook cell 9 into `src/dataset/filter_balanced.py` or a parameterized filter script.
- YOLO training calls from cells 7 and 11 into `scripts/train_yolo.py`.
- Validation and threshold tuning into `scripts/validate_yolo.py`.
- Results extraction from `results.csv` into `src/evaluation/summarize_runs.py`.
- Figure copying/selection into `scripts/export_report_figures.py`.
- Streamlit viewer from `skripts/visualize_yolo_labels.py` into `app/streamlit_app.py`.
- Existing raster-to-YOLO conversion from `skripts/build_yolo_dataset_bbox.py` should become reusable source code under `src/dataset/`.
- Existing overlay utilities from `skripts/overlay_5_classes.py` should become `src/geodata/` or `src/visualization/`.

## Артефакты, которые стоит сохранить в reports/figures

Recommended lightweight figures:

- `runs/runs_4/results.png`
- `runs/runs_4/confusion_matrix.png`
- `runs/runs_4/confusion_matrix_normalized.png`
- `runs/runs_4/BoxF1_curve.png`
- `runs/runs_4/BoxP_curve.png`
- `runs/runs_4/BoxR_curve.png`
- `runs/runs_4/val_batch0_labels.jpg`
- `runs/runs_4/val_batch0_pred.jpg`
- `runs/runs_4/val_batch1_labels.jpg`
- `runs/runs_4/val_batch1_pred.jpg`
- `runs/runs_2/results.png`
- `runs/runs_2/BoxPR_curve.png`
- `runs/runs_2/confusion_matrix.png`

Do not commit:

- `runs/**/best.pt`
- `runs/**/last.pt`
- full run directories
- full datasets

## Что не хватает для воспроизводимости

- No `requirements.txt` yet.
- No stable config files for v2/v3/v4/v5.
- Hard-coded local paths:
  - `/content/dataset/...`
  - `/content/drive/...`
  - `/Volumes/Lexar/Датасет/`
- Folder `skripts/` should be normalized to `scripts/` or split into `src/` and `scripts/`.
- No CLI arguments for input/output dataset roots.
- v3 and v5 metrics are preserved in notes, but local run folders are not present.
- No small public `data_sample/`.
- No `.gitignore` rules yet for datasets, model weights, and run artifacts.
- No automated smoke test for dataset generation.

# Repository Integration Plan

Target structure:

```text
04_detection_yolo/
  README.md
  requirements.txt
  configs/
  src/
  app/
  notebooks/
  reports/
  reports/figures/
  data_sample/
```

## 1. Какие файлы создать

- `requirements.txt`
- `configs/dataset_v2.yaml`
- `configs/dataset_v4.yaml`
- `configs/train_v2_yolov8n.yaml`
- `configs/train_v4_yolov8s.yaml`
- `configs/validate_thresholds.yaml`
- `src/dataset/build_bbox_dataset.py`
- `src/dataset/filter_dataset.py`
- `src/training/train_yolo.py`
- `src/evaluation/validate_yolo.py`
- `src/evaluation/summarize_runs.py`
- `src/visualization/export_figures.py`
- `app/streamlit_app.py`
- `reports/figures/`
- `data_sample/README.md`
- `.gitignore`

## 2. Какие файлы переименовать

Recommended later, not done in this phase:

- `skripts/` -> `scripts/` or split into `src/` + `scripts/`
- `skripts/visualize_yolo_labels.py` -> `app/streamlit_app.py`
- `skripts/build_yolo_dataset_bbox.py` -> `src/dataset/build_bbox_dataset.py`
- `skripts/make_dataset_v2_kurgans_li_ae.py` -> `src/dataset/filter_dataset.py` or `scripts/make_dataset_v2.py`
- `remarks.txt` and `results.txt` -> fold into `reports/experiments.md`

## 3. Какие notebook sections превратить в scripts

- Colab unzip/setup should become documented data-preparation instructions, not core code.
- Dataset YAML creation should become config generation or static config templates.
- v2 filtering cell should become a parameterized filter script.
- v4 filtering cell should become the same filter script with stricter config.
- YOLO train cells should become `train_yolo.py`.
- Threshold tuning should become `validate_yolo.py`.
- Results analysis should become `summarize_runs.py` plus `reports/experiments.md`.

## 4. Какие figures скопировать в reports/figures

Copy selected, lightweight, portfolio-useful figures:

- v4 training curves: `results.png`
- v4 confusion matrix and normalized confusion matrix
- v4 confidence curves: F1, precision, recall
- 2-4 validation label/prediction pairs
- v2 PR/results plot for comparison

## 5. Что оставить в notebook как demo/experiment log

- A short Colab-compatible walkthrough.
- Dataset archive loading.
- A minimal demo of filtering and training.
- Explanatory markdown about experiment motivation.
- Historical outputs that document the experiment process.

The notebook should not be the primary production entry point after extraction.

## 6. Что добавить в .gitignore

Recommended rules:

```gitignore
runs/
**/runs/
*.pt
*.onnx
*.engine
dataset_yolo*/
data/
data_raw/
data_processed/
data_sample/images_full/
__pycache__/
.ipynb_checkpoints/
.DS_Store
```
