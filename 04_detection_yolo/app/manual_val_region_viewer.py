from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT.parent / "datasets" / "dataset_yolo_bbox_v3g_li_medium_manual_keep_only"
REVIEW_DIR = ROOT / "manual_val_region_review"
SUMMARY_PATH = REVIEW_DIR / "region_summary.csv"
DECISIONS_PATH = REVIEW_DIR / "manual_val_regions.csv"

DECISIONS = ["", "val", "train", "exclude"]
CLASS_COLORS = {
    "kurgany_tselye": "#00ff66",
    "kurgany_povrezhdennye": "#ffcc00",
}


st.set_page_config(page_title="Validation Region Viewer", layout="wide")


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("_") or "region"


@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "images_total",
        "positive_images",
        "negative_images",
        "bbox_total",
        "kurgany_tselye_bbox",
        "kurgany_povrezhdennye_bbox",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "bbox_area_median",
        "bbox_area_p25",
        "bbox_area_p75",
        "objects_per_positive_image_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_metadata(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    for col in ["bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "yolo_xc", "yolo_yc", "yolo_w", "yolo_h", "bbox_area_px"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    return df


def load_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["decision", "comment"]:
        if col not in df.columns:
            df[col] = ""
    return df


def save_region_decision(region: str, decision: str, comment: str) -> None:
    decisions = load_decisions(DECISIONS_PATH)
    mask = decisions["region"].astype(str).eq(region)
    if not mask.any():
        decisions = pd.concat(
            [
                decisions,
                pd.DataFrame(
                    [
                        {
                            "region": region,
                            "images_total": "",
                            "positive_images": "",
                            "negative_images": "",
                            "bbox_total": "",
                            "decision": decision,
                            "comment": comment,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        decisions.loc[mask, "decision"] = decision
        decisions.loc[mask, "comment"] = comment
    decisions.to_csv(DECISIONS_PATH, index=False)
    st.cache_data.clear()


def resolve_image_path(row: pd.Series) -> Path:
    path = Path(str(row["image"]))
    if path.is_absolute() and path.exists():
        return path
    split = str(row["split"])
    image_name = str(row.get("image_name") or path.name)
    candidates = [
        DATASET_DIR / "images" / split / image_name,
        DATASET_DIR / "images" / split / path.name,
        DATASET_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def draw_image(image_path: Path, objects: pd.DataFrame, max_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(max_size / original_w, max_size / original_h)
    new_size = (max(1, int(original_w * scale)), max(1, int(original_h * scale)))
    preview = image.resize(new_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for _, obj in objects.iterrows():
        cls = str(obj.get("source_class_name") or obj.get("class_name") or "kurgan")
        color = CLASS_COLORS.get(cls, "#00ff66")
        if pd.notna(obj.get("yolo_xc")):
            x1 = (float(obj["yolo_xc"]) - float(obj["yolo_w"]) / 2) * original_w
            y1 = (float(obj["yolo_yc"]) - float(obj["yolo_h"]) / 2) * original_h
            x2 = (float(obj["yolo_xc"]) + float(obj["yolo_w"]) / 2) * original_w
            y2 = (float(obj["yolo_yc"]) + float(obj["yolo_h"]) / 2) * original_h
        else:
            x1 = float(obj["bbox_x1_px"])
            y1 = float(obj["bbox_y1_px"])
            x2 = float(obj["bbox_x2_px"])
            y2 = float(obj["bbox_y2_px"])
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=color, width=3)
        label = cls.replace("kurgany_", "")[:14]
        text_w = max(30, len(label) * 6 + 6)
        y_text = max(0, box[1] - 14)
        draw.rectangle([box[0], y_text, box[0] + text_w, y_text + 13], fill=color)
        draw.text((box[0] + 3, y_text + 1), label, fill="black", font=font)
    return preview


def region_options(summary: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    decisions_small = decisions[["region", "decision", "comment"]].copy()
    out = summary.merge(decisions_small, on="region", how="left")
    out["decision"] = out["decision"].fillna("")
    out["comment"] = out["comment"].fillna("")
    return out


def fmt_region(row: pd.Series) -> str:
    decision = row["decision"] or "undecided"
    return (
        f"{row['region']} | {decision} | "
        f"img={row['images_total']} pos={row['positive_images']} neg={row['negative_images']} bbox={row['bbox_total']}"
    )


st.title("Curated Validation Region Viewer")

if not SUMMARY_PATH.exists():
    st.error(f"Missing region summary: {SUMMARY_PATH}")
    st.stop()
if not DECISIONS_PATH.exists():
    st.error(f"Missing manual decision template: {DECISIONS_PATH}")
    st.stop()
if not (DATASET_DIR / "metadata.csv").exists():
    st.error(f"Missing dataset metadata: {DATASET_DIR / 'metadata.csv'}")
    st.stop()

summary = load_summary(SUMMARY_PATH)
metadata = load_metadata(DATASET_DIR)
decisions = load_decisions(DECISIONS_PATH)
regions = region_options(summary, decisions)

with st.sidebar:
    st.header("Region")
    decision_filter = st.selectbox("Decision filter", ["all", "undecided", "val", "train", "exclude"])
    sort_mode = st.selectbox(
        "Sort",
        [
            "bbox desc",
            "positive images desc",
            "images desc",
            "region name",
            "bbox area median desc",
            "bbox area median asc",
        ],
    )
    show_only_positive = st.checkbox("Show positive images only", value=False)
    show_only_negative = st.checkbox("Show negative images only", value=False)
    max_images = st.slider("Max images to show", 5, 100, 40, step=5)
    thumb_size = st.slider("Image preview size", 220, 640, 360, step=20)

filtered_regions = regions.copy()
if decision_filter == "undecided":
    filtered_regions = filtered_regions[filtered_regions["decision"].eq("")]
elif decision_filter != "all":
    filtered_regions = filtered_regions[filtered_regions["decision"].eq(decision_filter)]

if sort_mode == "bbox desc":
    filtered_regions = filtered_regions.sort_values("bbox_total", ascending=False)
elif sort_mode == "positive images desc":
    filtered_regions = filtered_regions.sort_values("positive_images", ascending=False)
elif sort_mode == "images desc":
    filtered_regions = filtered_regions.sort_values("images_total", ascending=False)
elif sort_mode == "bbox area median desc":
    filtered_regions = filtered_regions.sort_values("bbox_area_median", ascending=False, na_position="last")
elif sort_mode == "bbox area median asc":
    filtered_regions = filtered_regions.sort_values("bbox_area_median", ascending=True, na_position="last")
else:
    filtered_regions = filtered_regions.sort_values("region")

if filtered_regions.empty:
    st.warning("No regions match current filters.")
    st.stop()

selected_region = st.selectbox(
    "Select region",
    filtered_regions["region"].tolist(),
    format_func=lambda region: fmt_region(filtered_regions[filtered_regions["region"].eq(region)].iloc[0]),
)

row = regions[regions["region"].eq(selected_region)].iloc[0]
region_images = metadata[metadata["region"].astype(str).eq(selected_region)].drop_duplicates("image").copy()
region_objects = metadata[(metadata["region"].astype(str).eq(selected_region)) & metadata["class_name"].notna()].copy()
positive_image_keys = set(region_objects["image"].astype(str))
region_images["has_objects"] = region_images["image"].astype(str).isin(positive_image_keys)

if show_only_positive and not show_only_negative:
    region_images = region_images[region_images["has_objects"]]
elif show_only_negative and not show_only_positive:
    region_images = region_images[~region_images["has_objects"]]

region_images = region_images.sort_values(["has_objects", "image_name"], ascending=[False, True]).head(max_images)

metrics = st.columns(6)
metrics[0].metric("images", int(row["images_total"]))
metrics[1].metric("positive", int(row["positive_images"]))
metrics[2].metric("negative", int(row["negative_images"]))
metrics[3].metric("bbox", int(row["bbox_total"]))
metrics[4].metric("tselye", int(row["kurgany_tselye_bbox"]))
metrics[5].metric("povrezhdennye", int(row["kurgany_povrezhdennye_bbox"]))

left, right = st.columns([2, 1])
with left:
    st.subheader(selected_region)
    sheet_path = REVIEW_DIR / f"{safe_name(selected_region)}.jpg"
    if sheet_path.exists():
        st.image(str(sheet_path), caption="Region contact sheet", use_container_width=True)
    else:
        st.warning(f"Missing contact sheet: {sheet_path}")

with right:
    st.subheader("Split decision")
    current_decision = row["decision"] if row["decision"] in DECISIONS else ""
    current_comment = row["comment"] if isinstance(row["comment"], str) else ""
    decision = st.selectbox("decision", DECISIONS, index=DECISIONS.index(current_decision))
    comment = st.text_area("comment", value=current_comment, height=120)
    if st.button("Save decision", use_container_width=True):
        save_region_decision(selected_region, decision, comment)
        st.success("Saved")
        st.rerun()

    st.dataframe(
        pd.DataFrame(
            [
                ("bbox_area_p25", row["bbox_area_p25"]),
                ("bbox_area_median", row["bbox_area_median"]),
                ("bbox_area_p75", row["bbox_area_p75"]),
                ("objects_per_positive_image_mean", row["objects_per_positive_image_mean"]),
            ],
            columns=["field", "value"],
        ),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("Images With GT BBox")

legend_cols = st.columns(2)
legend_cols[0].markdown("<span style='color:#00aa44'>■</span> `kurgany_tselye`", unsafe_allow_html=True)
legend_cols[1].markdown("<span style='color:#cc9900'>■</span> `kurgany_povrezhdennye`", unsafe_allow_html=True)

if region_images.empty:
    st.info("No images match image filters.")
else:
    cols = st.columns(3)
    for idx, (_, image_row) in enumerate(region_images.iterrows()):
        image_path = resolve_image_path(image_row)
        objects = region_objects[region_objects["image"].astype(str).eq(str(image_row["image"]))].copy()
        preview = draw_image(image_path, objects, thumb_size)
        with cols[idx % 3]:
            st.image(preview, use_container_width=True)
            st.caption(
                f"{image_row.get('image_name', image_path.name)} | "
                f"split={image_row.get('split', '')} | objects={len(objects)}"
            )

st.divider()
st.caption(f"Dataset: {DATASET_DIR}")
st.caption(f"Region summary: {SUMMARY_PATH}")
st.caption(f"Manual decisions: {DECISIONS_PATH}")
