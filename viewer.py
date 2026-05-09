from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MASK_CMAP = ListedColormap([
    "black",   # 0 background
    "lime",    # 1 whole
    "red",     # 2 damaged
])


DATASET_DIR = Path("/Users/Di/Documents/Новая папка/Geodata/main/single_classes/baseline_kurgan/dataset_multi_full_non_binary_сrop")
IMG_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"
META_PATH = DATASET_DIR / "metadata.csv"


st.set_page_config(page_title="Kurgan Dataset Viewer", layout="wide")
st.title("Kurgan Dataset Viewer")


@st.cache_data
def load_metadata():
    return pd.read_csv(META_PATH)


def load_sample(sample_id: str):
    patch = np.load(IMG_DIR / f"{sample_id}.npy")
    mask = np.load(MASK_DIR / f"{sample_id}.npy")
    return patch, mask


def make_overlay_figure(patch, mask, meta_row):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(patch, cmap="gray")
    axes[0].set_title("Patch")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=2)
    axes[1].set_title("Mask (bg / whole / damaged)")
    axes[1].axis("off")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 1] = [0.0, 1.0, 0.0, 0.35]  # whole = green
    overlay[mask == 2] = [1.0, 0.0, 0.0, 0.35]  # damaged = red

    axes[2].imshow(patch, cmap="gray")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"sample_id={meta_row['sample_id']} | "
        f"region={meta_row['region']} | "
        f"modality={meta_row['modality']} | "
        f"type={meta_row['kurgan_type']}",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

meta = load_metadata()

if meta.empty:
    st.error("metadata.csv is empty")
    st.stop()

# приведение типов на всякий случай
meta["sample_id"] = meta["sample_id"].astype(str).str.zfill(6)

st.sidebar.header("Filters")

regions = ["ALL"] + sorted(meta["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions, index=0)

modalities = ["ALL"] + sorted(meta["modality"].dropna().unique().tolist())
selected_modality = st.sidebar.selectbox("Modality", modalities, index=0)

types = ["ALL"] + sorted(meta["kurgan_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Kurgan type", types, index=0)

min_h, max_h = int(meta["height"].min()), int(meta["height"].max())

if min_h == max_h:
    st.sidebar.write(f"Height: {min_h} (fixed)")
    height_range = (min_h, max_h)
else:
    height_range = st.sidebar.slider(
        "Height range",
        min_value=min_h,
        max_value=max_h,
        value=(min_h, max_h),
    )
min_w, max_w = int(meta["width"].min()), int(meta["width"].max())
if min_w == max_w:
    st.sidebar.write(f"Width: {min_w} (fixed)")
    width_range = (min_w, max_w)
else:
    width_range = st.sidebar.slider(
        "Width range",
        min_value=min_w,
        max_value=max_w,
    value=(min_w, max_w),
)

min_n, max_n = int(meta["n_objects_in_patch"].min()), int(meta["n_objects_in_patch"].max())
if min_n == max_n:
    st.sidebar.write(f"Objects in patch: {min_n} (fixed)")
    n_objects_range = (min_n, max_n)
else:
    n_objects_range = st.sidebar.slider(
        "Objects in patch",
        min_value=min_n,
        max_value=max_n,
        value=(min_n, max_n),
    )

if "used_crs_fallback" in meta.columns:
    fallback_options = ["ALL", "False", "True"]
    selected_fallback = st.sidebar.selectbox("Used CRS fallback", fallback_options, index=0)
else:
    selected_fallback = "ALL"

if "has_whole" in meta.columns:
    whole_options = ["ALL", "False", "True"]
    selected_has_whole = st.sidebar.selectbox("Has whole", whole_options, index=0)
else:
    selected_has_whole = "ALL"

if "has_damaged" in meta.columns:
    damaged_options = ["ALL", "False", "True"]
    selected_has_damaged = st.sidebar.selectbox("Has damaged", damaged_options, index=0)
else:
    selected_has_damaged = "ALL"


# --- filtering ---
filtered = meta.copy()

if selected_region != "ALL":
    filtered = filtered[filtered["region"] == selected_region]

if selected_modality != "ALL":
    filtered = filtered[filtered["modality"] == selected_modality]

if selected_type != "ALL":
    filtered = filtered[filtered["kurgan_type"] == selected_type]

filtered = filtered[
    filtered["height"].between(height_range[0], height_range[1]) &
    filtered["width"].between(width_range[0], width_range[1]) &
    filtered["n_objects_in_patch"].between(n_objects_range[0], n_objects_range[1])
].copy()

if "used_crs_fallback" in filtered.columns and selected_fallback != "ALL":
    want_fallback = selected_fallback == "True"
    filtered = filtered[filtered["used_crs_fallback"] == want_fallback].copy()

if "has_whole" in filtered.columns and selected_has_whole != "ALL":
    want_whole = selected_has_whole == "True"
    filtered = filtered[filtered["has_whole"] == want_whole].copy()

if "has_damaged" in filtered.columns and selected_has_damaged != "ALL":
    want_damaged = selected_has_damaged == "True"
    filtered = filtered[filtered["has_damaged"] == want_damaged].copy()

st.subheader("Filtered dataset")
st.write(f"Samples: {len(filtered)}")

if filtered.empty:
    st.warning("No samples match the filters.")
    st.stop()

# --- sorting ---
sort_candidates = ["sample_id", "region", "modality", "kurgan_type", "n_objects_in_patch", "height", "width"]
if "used_crs_fallback" in filtered.columns:
    sort_candidates.append("used_crs_fallback")

sort_col = st.selectbox("Sort by", sort_candidates)
ascending = st.checkbox("Ascending", value=True)

filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

# --- session state sync ---
if "viewer_idx" not in st.session_state:
    st.session_state.viewer_idx = 0

sample_labels = [
    f"{row.sample_id} | {row.region} | {row.modality} | "
    f"{row.kurgan_type} | objs={row.n_objects_in_patch} | {row.height}x{row.width}"
    for row in filtered.itertuples(index=False)
]

selected_idx = st.selectbox(
    "Choose sample",
    range(len(filtered)),
    index=min(st.session_state.viewer_idx, len(filtered) - 1),
    format_func=lambda i: sample_labels[i]
)

col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 6])

with col_nav1:
    prev_clicked = st.button("⬅ Prev")

with col_nav2:
    next_clicked = st.button("Next ➡")

if prev_clicked:
    st.session_state.viewer_idx = max(0, selected_idx - 1)
elif next_clicked:
    st.session_state.viewer_idx = min(len(filtered) - 1, selected_idx + 1)
else:
    st.session_state.viewer_idx = selected_idx

row = filtered.iloc[st.session_state.viewer_idx]
sample_id = str(row["sample_id"]).zfill(6)

patch, mask = load_sample(sample_id)

# --- metadata display ---
st.subheader("Metadata")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**sample_id:** {sample_id}")
    st.write(f"**region:** {row['region']}")
    st.write(f"**modality:** {row['modality']}")
    st.write(f"**kurgan_type:** {row['kurgan_type']}")
    if "has_whole" in row.index:
        st.write(f"**has_whole:** {row['has_whole']}")
    if "has_damaged" in row.index:
        st.write(f"**has_damaged:** {row['has_damaged']}")

with col2:
    st.write(f"**raster_file:** {row['raster_file']}")
    st.write(f"**n_objects_in_patch:** {row['n_objects_in_patch']}")
    st.write(f"**height:** {row['height']}")
    st.write(f"**width:** {row['width']}")

    if "mask_whole_pixels" in row.index:
        st.write(f"**mask_whole_pixels:** {row['mask_whole_pixels']}")
    if "mask_damaged_pixels" in row.index:
        st.write(f"**mask_damaged_pixels:** {row['mask_damaged_pixels']}")

fig = make_overlay_figure(patch, mask, row)
st.pyplot(fig)

with st.expander("Show filtered metadata table"):
    st.dataframe(filtered, use_container_width=True)