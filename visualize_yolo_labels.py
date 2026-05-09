from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, Patch

DATASET_DIR = Path("/Users/Di/Documents/GitHub/My projects/Geodata_Archaeology_CV/datasets/dataset_yolo_bbox")
IMG_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
META_PATH = DATASET_DIR / "metadata.csv"

CLASS_LABELS = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}

CLASS_COLORS = {
    0: "lime",
    1: "green",
    2: "yellow",
    3: "cyan",
    4: "magenta",
}


st.set_page_config(page_title="YOLO BBox Viewer", layout="wide")
st.title("YOLO BBox Dataset Viewer")


@st.cache_data
def load_metadata(path):
    meta = pd.read_csv(path)
    if "class_id" in meta.columns:
        meta["class_id"] = meta["class_id"].astype("Int64")
    return meta


def load_image(path):
    return np.array(Image.open(path))


def parse_label_file(path):
    bboxes = []
    if not path.exists():
        return bboxes

    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        bboxes.append((cls_id, xc, yc, w, h))

    return bboxes


def yolo_to_xyxy(box, img_w, img_h):
    cls_id, xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return cls_id, x1, y1, x2, y2


def draw_figure(img, bboxes, show_labels=False, show_legend=True, zoom_mode=False):
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)

    xyxy_boxes = []

    for box in bboxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        xyxy_boxes.append((cls_id, x1, y1, x2, y2))

        color = CLASS_COLORS.get(cls_id, "white")

        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if show_labels:
            ax.text(
                x1,
                y1,
                CLASS_LABELS.get(cls_id, str(cls_id)),
                color=color,
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1),
            )

    if zoom_mode and xyxy_boxes:
        all_x1 = min(b[1] for b in xyxy_boxes)
        all_y1 = min(b[2] for b in xyxy_boxes)
        all_x2 = max(b[3] for b in xyxy_boxes)
        all_y2 = max(b[4] for b in xyxy_boxes)

        pad_x = max(30, 0.15 * (all_x2 - all_x1))
        pad_y = max(30, 0.15 * (all_y2 - all_y1))

        ax.set_xlim(max(0, all_x1 - pad_x), min(w, all_x2 + pad_x))
        ax.set_ylim(min(h, all_y2 + pad_y), max(0, all_y1 - pad_y))

    if show_legend:
        handles = [
            Patch(facecolor="none", edgecolor=color, label=CLASS_LABELS[cls_id])
            for cls_id, color in CLASS_COLORS.items()
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.75)

    ax.axis("off")
    return fig


def bool_filter(df, col, selected):
    if col not in df.columns or selected == "ALL":
        return df
    want = selected == "True"
    vals = df[col].astype(str).str.lower().map({"true": True, "false": False})
    return df[vals == want].copy()


if not META_PATH.exists():
    st.error(f"metadata.csv not found: {META_PATH}")
    st.stop()

meta = load_metadata(META_PATH)

if meta.empty:
    st.error("metadata.csv is empty")
    st.stop()

# one row per image for sample selection
image_meta = meta.drop_duplicates("image").copy()

st.sidebar.header("Filters")

regions = ["ALL"] + sorted(image_meta["region"].dropna().unique().tolist())
modalities = ["ALL"] + sorted(image_meta["modality"].dropna().unique().tolist())

selected_region = st.sidebar.selectbox("Region", regions)
selected_modality = st.sidebar.selectbox("Modality", modalities)

class_options = ["ALL"] + [
    CLASS_LABELS[i] for i in sorted(CLASS_LABELS)
    if CLASS_LABELS[i] in meta["class_name"].dropna().unique()
]
selected_class = st.sidebar.selectbox("Class", class_options)

only_positive = st.sidebar.checkbox("Only positive images", value=True)

min_area = int(meta["bbox_area_px"].dropna().min()) if meta["bbox_area_px"].notna().any() else 0
max_area = int(meta["bbox_area_px"].dropna().max()) if meta["bbox_area_px"].notna().any() else 1
bbox_area_range = st.sidebar.slider(
    "BBox area px",
    min_area,
    max_area,
    (min_area, max_area),
)

edge_filter = st.sidebar.selectbox("BBox touches tile edge", ["ALL", "False", "True"])

show_labels = st.sidebar.checkbox("Show text labels", value=False)
show_legend = st.sidebar.checkbox("Show legend", value=True)
zoom_mode = st.sidebar.checkbox("Zoom to bboxes", value=False)

filtered = meta.copy()

if selected_region != "ALL":
    filtered = filtered[filtered["region"] == selected_region]

if selected_modality != "ALL":
    filtered = filtered[filtered["modality"] == selected_modality]

if only_positive:
    filtered = filtered[filtered["is_positive"].astype(bool)]

if selected_class != "ALL":
    filtered = filtered[filtered["class_name"] == selected_class]

filtered = filtered[
    filtered["bbox_area_px"].isna()
    | filtered["bbox_area_px"].between(bbox_area_range[0], bbox_area_range[1])
].copy()

filtered = bool_filter(filtered, "bbox_touches_tile_edge", edge_filter)

if filtered.empty:
    st.warning("No samples match filters.")
    st.stop()

filtered_images = filtered.drop_duplicates("image").reset_index(drop=True)

st.write(f"Images: {len(filtered_images)} | rows: {len(filtered)}")

sort_candidates = [
    c for c in [
        "region", "modality", "is_positive", "n_objects",
        "tile_size", "context_m", "pixel_size_m",
        "valid_fraction", "tile_std", "tile_p98_minus_p2",
    ]
    if c in filtered_images.columns
]

sort_col = st.selectbox("Sort by", sort_candidates, index=0)
ascending = st.checkbox("Ascending", value=True)
filtered_images = filtered_images.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

if "bbox_viewer_idx" not in st.session_state:
    st.session_state.bbox_viewer_idx = 0

st.session_state.bbox_viewer_idx = min(st.session_state.bbox_viewer_idx, len(filtered_images) - 1)

sample_labels = [
    f"{i} | {r.region} | {r.modality} | objs={r.n_objects} | tile={r.tile_size} | {Path(r.image).name}"
    for i, r in filtered_images.iterrows()
]

idx = st.selectbox(
    "Choose image",
    range(len(filtered_images)),
    index=st.session_state.bbox_viewer_idx,
    format_func=lambda i: sample_labels[i],
)

nav1, nav2, nav3 = st.columns([1, 1, 6])
with nav1:
    if st.button("⬅ Prev"):
        st.session_state.bbox_viewer_idx = max(0, idx - 1)
        st.rerun()
with nav2:
    if st.button("Next ➡"):
        st.session_state.bbox_viewer_idx = min(len(filtered_images) - 1, idx + 1)
        st.rerun()

st.session_state.bbox_viewer_idx = idx
row = filtered_images.iloc[idx]

img_path = Path(row["image"])
lbl_path = Path(row["label"])

if not img_path.is_absolute():
    img_path = DATASET_DIR.parent / img_path

if not lbl_path.is_absolute():
    lbl_path = DATASET_DIR.parent / lbl_path

img = load_image(img_path)
bboxes_all = parse_label_file(lbl_path)

# apply class filter also to drawn boxes
if selected_class != "ALL":
    class_id = {v: k for k, v in CLASS_LABELS.items()}[selected_class]
    bboxes = [b for b in bboxes_all if b[0] == class_id]
else:
    bboxes = bboxes_all

left, right = st.columns([2, 1])

with left:
    st.subheader("Image + BBoxes")
    fig = draw_figure(
        img,
        bboxes,
        show_labels=show_labels,
        show_legend=show_legend,
        zoom_mode=zoom_mode,
    )
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Metadata")

    metadata_fields = st.multiselect(
        "Fields to show",
        options=list(row.index),
        default=[
            c for c in [
                "split", "region", "modality", "raster_file",
                "tile_size", "resize_to", "context_m",
                "pixel_size_m", "n_objects",
                "is_positive", "used_crs_fallback",
                "valid_fraction", "tile_std", "tile_p98_minus_p2",
            ]
            if c in row.index
        ],
    )

    st.dataframe(
        pd.DataFrame({"field": metadata_fields, "value": [row[c] for c in metadata_fields]}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("BBoxes in label")
    bbox_rows = []
    h, w = img.shape[:2]
    for b in bboxes_all:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(b, w, h)
        bbox_rows.append({
            "class_id": cls_id,
            "class_name": CLASS_LABELS.get(cls_id, str(cls_id)),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "w": round(x2 - x1, 1),
            "h": round(y2 - y1, 1),
            "area": round((x2 - x1) * (y2 - y1), 1),
        })
    st.dataframe(pd.DataFrame(bbox_rows), use_container_width=True, hide_index=True)

st.subheader("BBox vs metadata rows for this image")
same_image_rows = filtered[filtered["image"] == row["image"]].copy()
cols = [
    c for c in [
        "class_id", "class_name",
        "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
        "bbox_area_px", "bbox_touches_tile_edge",
    ]
    if c in same_image_rows.columns
]
st.dataframe(same_image_rows[cols], use_container_width=True, hide_index=True)

with st.expander("Filtered image table"):
    st.dataframe(filtered_images, use_container_width=True)