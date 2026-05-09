from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# =========================
# CONFIG
# =========================

DATASET_DIR = Path(
    "/Users/Di/Documents/GitHub/My projects/Geodata_Archaeology_CV/dataset_5_classes_multiclass"
)

IMG_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"
META_PATH = DATASET_DIR / "metadata.csv"

CLASS_TO_ID = {
    "background": 0,
    "kurgany_tselye": 1,
    "kurgany_povrezhdennye": 2,
    "gorodishcha": 3,
    "fortifikatsii": 4,
    "arkhitektury": 5,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

CLASS_LABEL_RU = {
    0: "background",
    1: "курганы целые",
    2: "курганы поврежденные",
    3: "городища",
    4: "фортификации",
    5: "архитектуры",
}

TARGET_CLASSES = [
    "kurgany_tselye",
    "kurgany_povrezhdennye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
]

# Цвета только для viewer-а. На обучение никак не влияют.
MASK_COLORS = [
    "#000000",  # background
    "#7CFC00",  # kurgany_tselye
    "#228B22",  # kurgany_povrezhdennye
    "#FFD84D",  # gorodishcha
    "#00B7C7",  # fortifikatsii
    "#FF4FD8",  # arkhitektury
]

MASK_CMAP = ListedColormap(MASK_COLORS)
MASK_NORM = BoundaryNorm(np.arange(-0.5, len(MASK_COLORS) + 0.5, 1), MASK_CMAP.N)

OVERLAY_RGBA = {
    1: [0.49, 0.99, 0.00, 0.35],  # light green
    2: [0.13, 0.55, 0.13, 0.35],  # dark green
    3: [1.00, 0.85, 0.30, 0.35],  # warm yellow
    4: [0.00, 0.72, 0.78, 0.35],  # cyan
    5: [1.00, 0.31, 0.85, 0.35],  # magenta
}

st.set_page_config(page_title="5-class Dataset Viewer", layout="wide")
st.title("5-class Archaeology Dataset Viewer")


@st.cache_data
def load_metadata(meta_path: Path):
    return pd.read_csv(meta_path)


def load_sample(sample_id: str):
    patch = np.load(IMG_DIR / f"{sample_id}.npy")
    mask = np.load(MASK_DIR / f"{sample_id}.npy")
    return patch, mask


def stretch_for_display(img: np.ndarray) -> np.ndarray:
    """Аккуратный contrast stretch для визуализации patch."""
    img = np.asarray(img)
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def make_overlay(mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for class_id, rgba in OVERLAY_RGBA.items():
        overlay[mask == class_id] = rgba
    return overlay


def make_overlay_figure(patch, mask, meta_row):
    patch_vis = stretch_for_display(patch)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(patch_vis, cmap="gray")
    axes[0].set_title("Patch")
    axes[0].axis("off")

    im = axes[1].imshow(mask, cmap=MASK_CMAP, norm=MASK_NORM)
    axes[1].set_title("Mask: 0 bg / 1 whole / 2 damaged / 3 gorod / 4 fort / 5 arch")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, ticks=list(ID_TO_CLASS.keys()))
    cbar.ax.set_yticklabels([CLASS_LABEL_RU[i] for i in ID_TO_CLASS.keys()])

    overlay = make_overlay(mask)
    axes[2].imshow(patch_vis, cmap="gray")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    class_title = meta_row.get("class_name", "unknown")
    if "class_label_ru" in meta_row.index and pd.notna(meta_row["class_label_ru"]):
        class_title = f"{class_title} / {meta_row['class_label_ru']}"

    fig.suptitle(
        f"sample_id={meta_row['sample_id']} | "
        f"region={meta_row['region']} | "
        f"modality={meta_row['modality']} | "
        f"class={class_title}",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def bool_filter_sidebar(df: pd.DataFrame, col: str, label: str):
    if col not in df.columns:
        return "ALL"
    return st.sidebar.selectbox(label, ["ALL", "False", "True"], index=0)


def apply_bool_filter(df: pd.DataFrame, col: str, selected: str) -> pd.DataFrame:
    if col not in df.columns or selected == "ALL":
        return df

    # metadata может прийти как bool или как строки True/False.
    if df[col].dtype == object:
        values = df[col].astype(str).str.lower().map({"true": True, "false": False})
    else:
        values = df[col].astype(bool)

    want = selected == "True"
    return df[values == want].copy()


def normalize_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return df


# =========================
# LOAD
# =========================

if not META_PATH.exists():
    st.error(f"metadata.csv not found: {META_PATH}")
    st.stop()

meta = load_metadata(META_PATH)

if meta.empty:
    st.error("metadata.csv is empty")
    st.stop()

meta = normalize_sample_id_column(meta)

# Backward compatibility: если вдруг где-то остался старый kurgan_type.
if "class_name" not in meta.columns and "kurgan_type" in meta.columns:
    meta["class_name"] = meta["kurgan_type"].map({
        "whole": "kurgany_tselye",
        "damaged": "kurgany_povrezhdennye",
    }).fillna(meta["kurgan_type"])

if "class_id" not in meta.columns and "class_name" in meta.columns:
    meta["class_id"] = meta["class_name"].map(CLASS_TO_ID)

if "class_label_ru" not in meta.columns and "class_id" in meta.columns:
    meta["class_label_ru"] = meta["class_id"].map(CLASS_LABEL_RU)

required_columns = ["sample_id", "region", "modality", "class_name", "height", "width", "n_objects_in_patch"]
missing_columns = [c for c in required_columns if c not in meta.columns]
if missing_columns:
    st.error(f"metadata.csv is missing required columns: {missing_columns}")
    st.stop()

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Filters")

regions = ["ALL"] + sorted(meta["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions, index=0)

modalities = ["ALL"] + sorted(meta["modality"].dropna().unique().tolist())
selected_modality = st.sidebar.selectbox("Modality", modalities, index=0)

classes = ["ALL"] + sorted(meta["class_name"].dropna().unique().tolist())
selected_class = st.sidebar.selectbox("Target class", classes, index=0)

min_h, max_h = int(meta["height"].min()), int(meta["height"].max())
if min_h == max_h:
    st.sidebar.write(f"Height: {min_h} (fixed)")
    height_range = (min_h, max_h)
else:
    height_range = st.sidebar.slider("Height range", min_h, max_h, (min_h, max_h))

min_w, max_w = int(meta["width"].min()), int(meta["width"].max())
if min_w == max_w:
    st.sidebar.write(f"Width: {min_w} (fixed)")
    width_range = (min_w, max_w)
else:
    width_range = st.sidebar.slider("Width range", min_w, max_w, (min_w, max_w))

min_n, max_n = int(meta["n_objects_in_patch"].min()), int(meta["n_objects_in_patch"].max())
if min_n == max_n:
    st.sidebar.write(f"Objects in patch: {min_n} (fixed)")
    n_objects_range = (min_n, max_n)
else:
    n_objects_range = st.sidebar.slider("Objects in patch", min_n, max_n, (min_n, max_n))

selected_fallback = bool_filter_sidebar(meta, "used_crs_fallback", "Used CRS fallback")
touches_border = bool_filter_sidebar(meta, "touches_border", "Touches border")
target_fits_inside = bool_filter_sidebar(meta, "target_fits_inside", "Target fits inside")

has_filter_values = {}
for class_name in TARGET_CLASSES:
    col = f"has_{class_name}"
    has_filter_values[col] = bool_filter_sidebar(meta, col, f"Has {class_name}")

# =========================
# FILTERING
# =========================

filtered = meta.copy()

if selected_region != "ALL":
    filtered = filtered[filtered["region"] == selected_region]

if selected_modality != "ALL":
    filtered = filtered[filtered["modality"] == selected_modality]

if selected_class != "ALL":
    filtered = filtered[filtered["class_name"] == selected_class]

filtered = filtered[
    filtered["height"].between(height_range[0], height_range[1])
    & filtered["width"].between(width_range[0], width_range[1])
    & filtered["n_objects_in_patch"].between(n_objects_range[0], n_objects_range[1])
].copy()

filtered = apply_bool_filter(filtered, "used_crs_fallback", selected_fallback)
filtered = apply_bool_filter(filtered, "touches_border", touches_border)
filtered = apply_bool_filter(filtered, "target_fits_inside", target_fits_inside)

for col, selected_value in has_filter_values.items():
    filtered = apply_bool_filter(filtered, col, selected_value)

st.subheader("Filtered dataset")
st.write(f"Samples: {len(filtered)}")

if filtered.empty:
    st.warning("No samples match the filters.")
    st.stop()

# =========================
# SORT + SELECT
# =========================

sort_candidates = [
    "sample_id",
    "region",
    "modality",
    "class_name",
    "n_objects_in_patch",
    "height",
    "width",
]

for optional_col in [
    "class_id",
    "used_crs_fallback",
    "touches_border",
    "target_fits_inside",
    "crop_size",
    "obj_w_px",
    "obj_h_px",
]:
    if optional_col in filtered.columns and optional_col not in sort_candidates:
        sort_candidates.append(optional_col)

sort_col = st.selectbox("Sort by", sort_candidates)
ascending = st.checkbox("Ascending", value=True)
filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

if "viewer_idx" not in st.session_state:
    st.session_state.viewer_idx = 0

st.session_state.viewer_idx = min(st.session_state.viewer_idx, len(filtered) - 1)

sample_labels = []
for row_tuple in filtered.itertuples(index=False):
    label = (
        f"{row_tuple.sample_id} | {row_tuple.region} | {row_tuple.modality} | "
        f"{row_tuple.class_name} | objs={row_tuple.n_objects_in_patch} | {row_tuple.height}x{row_tuple.width}"
    )
    sample_labels.append(label)

selected_idx = st.selectbox(
    "Choose sample",
    range(len(filtered)),
    index=st.session_state.viewer_idx,
    format_func=lambda i: sample_labels[i],
)

# Важно: сначала синхронизируем index, потом берём row.
st.session_state.viewer_idx = selected_idx
row = filtered.iloc[selected_idx]
sample_id = str(row["sample_id"]).zfill(6)

try:
    patch, mask = load_sample(sample_id)
except FileNotFoundError as e:
    st.error(f"Could not load sample {sample_id}: {e}")
    st.stop()

# =========================
# METADATA DISPLAY
# =========================

st.subheader("Metadata")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**sample_id:** {sample_id}")
    st.write(f"**region:** {row['region']}")
    st.write(f"**modality:** {row['modality']}")
    st.write(f"**class_name:** {row['class_name']}")
    if "class_label_ru" in row.index:
        st.write(f"**class_label_ru:** {row['class_label_ru']}")
    if "class_id" in row.index:
        st.write(f"**class_id:** {row['class_id']}")

with col2:

    st.write(f"**n_objects_in_patch:** {row['n_objects_in_patch']}")
    st.write(f"**height:** {row['height']}")
    st.write(f"**width:** {row['width']}")
    for optional_col in ["crop_size", "used_crs_fallback", "touches_border", "target_fits_inside"]:
        if optional_col in row.index:
            st.write(f"**{optional_col}:** {row[optional_col]}")

with col3:
    for class_name in TARGET_CLASSES:
        pix_col = f"mask_{class_name}_pixels"
        has_col = f"has_{class_name}"
        if pix_col in row.index:
            has_value = row[has_col] if has_col in row.index else "?"
            st.write(f"**{pix_col}:** {row[pix_col]}")

unique_ids, counts = np.unique(mask, return_counts=True)
mask_stats = pd.DataFrame({
    "class_id": unique_ids.astype(int),
    "class_name": [ID_TO_CLASS.get(int(i), "unknown") for i in unique_ids],
    "class_label_ru": [CLASS_LABEL_RU.get(int(i), "unknown") for i in unique_ids],
    "pixels": counts.astype(int),
})
st.write("**Mask actual classes:**")
st.dataframe(mask_stats, use_container_width=True)

fig = make_overlay_figure(patch, mask, row)
st.pyplot(fig)
plt.close(fig)

col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 6])

with col_nav1:
    prev_clicked = st.button("⬅ Prev")

with col_nav2:
    next_clicked = st.button("Next ➡")

if prev_clicked:
    st.session_state.viewer_idx = max(0, selected_idx - 1)
    st.rerun()
elif next_clicked:
    st.session_state.viewer_idx = min(len(filtered) - 1, selected_idx + 1)
    st.rerun()

with st.expander("Show filtered metadata table"):
    st.dataframe(filtered, use_container_width=True)
