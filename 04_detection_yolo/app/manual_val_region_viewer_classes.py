from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT.parent / "datasets" / "dataset_yolo_bbox"
REVIEW_DIR = ROOT / "manual_val_region_review_other_classes"
SUMMARY_PATH = REVIEW_DIR / "region_summary.csv"
DECISIONS_PATH = REVIEW_DIR / "manual_val_regions.csv"
AUDIT_DECISIONS_PATH = ROOT / "manual_audit" / "audit_decisions.csv"
TARGET_CLASSES = ["gorodishcha", "fortifikatsii", "arkhitektury"]
MODALITIES = ["Li"]
CLASS_ID_TO_NAME = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}
CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}
TARGET_CLASS_IDS = {CLASS_NAME_TO_ID[name] for name in TARGET_CLASSES}
DECISIONS = ["", "val", "train", "exclude"]
CLASS_COLORS = {
    "gorodishcha": "#4cc9f0",
    "fortifikatsii": "#f72585",
    "arkhitektury": "#f9c74f",
}


st.set_page_config(page_title="Validation Region Viewer: Other Classes", layout="wide")


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("_") or "region"


@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    int_cols = ["images_total", "positive_images", "negative_images", "bbox_total"] + [f"{cls}_bbox" for cls in TARGET_CLASSES]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["bbox_area_median", "bbox_area_p25", "bbox_area_p75", "objects_per_positive_image_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_metadata(dataset_dir: Path, audit_decisions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    df = df[df["modality"].astype(str).isin(MODALITIES)].copy()
    name_col = "source_image_name" if "source_image_name" in df.columns else "image_name"
    df["image_id"] = df.apply(lambda row: f"{row['split']}_{Path(str(row.get(name_col) or row['image'])).stem}", axis=1)
    decisions = pd.read_csv(audit_decisions_path).fillna("")
    decisions = decisions.drop_duplicates("image_id", keep="last")
    decisions = decisions.rename(columns={"decision": "manual_audit_decision"})
    df = df.merge(decisions[["image_id", "manual_audit_decision"]], on="image_id", how="left")
    df["manual_audit_decision"] = df["manual_audit_decision"].fillna("")
    df = df[df["manual_audit_decision"].eq("keep")].copy()
    for col in ["bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "yolo_xc", "yolo_yc", "yolo_w", "yolo_h", "bbox_area_px"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_target_object"] = df["source_class_name"].isin(TARGET_CLASSES)
    target_positive_images = set(df.loc[df["is_target_object"], "image"].astype(str))
    df["is_target_positive"] = df["image"].astype(str).isin(target_positive_images)
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
    candidates = [DATASET_DIR / "images" / split / image_name, DATASET_DIR / "images" / split / path.name, DATASET_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_label_path(row: pd.Series) -> Path:
    path = Path(str(row["label"]))
    if path.is_absolute() and path.exists():
        return path
    split = str(row["split"])
    label_name = str(row.get("label_name") or path.name)
    candidates = [DATASET_DIR / "labels" / split / label_name, DATASET_DIR / "labels" / split / path.name, DATASET_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def yolo_to_xyxy(box: tuple[int, float, float, float, float], image_w: int, image_h: int) -> tuple[int, float, float, float, float]:
    cls_id, xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * image_w
    y1 = (yc - bh / 2) * image_h
    x2 = (xc + bw / 2) * image_w
    y2 = (yc + bh / 2) * image_h
    return cls_id, x1, y1, x2, y2


def draw_image(image_path: Path, label_path: Path, max_size: int, target_class_ids: set[int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(max_size / original_w, max_size / original_h)
    preview = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for box in parse_label_file(label_path):
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, original_w, original_h)
        if cls_id not in target_class_ids:
            continue
        cls = CLASS_ID_TO_NAME.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls, "#00ff66")
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=color, width=3)
        label = cls[:14]
        text_w = max(30, len(label) * 6 + 6)
        y_text = max(0, box[1] - 14)
        draw.rectangle([box[0], y_text, box[0] + text_w, y_text + 13], fill=color)
        draw.text((box[0] + 3, y_text + 1), label, fill="black", font=font)
    return preview


def region_options(summary: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(decisions[["region", "decision", "comment"]], on="region", how="left")
    out["decision"] = out["decision"].fillna("")
    out["comment"] = out["comment"].fillna("")
    return out


def fmt_region(row: pd.Series) -> str:
    decision = row["decision"] or "undecided"
    return f"{row['region']} | {decision} | img={row['images_total']} pos={row['positive_images']} bbox={row['bbox_total']}"


st.title("Curated Validation Region Viewer: Li Other Classes")
st.caption("Target classes: gorodishcha / fortifikatsii / arkhitektury. Modality filter: Li only.")

if not SUMMARY_PATH.exists():
    st.error(f"Missing region summary: {SUMMARY_PATH}")
    st.stop()
if not DECISIONS_PATH.exists():
    st.error(f"Missing manual decision template: {DECISIONS_PATH}")
    st.stop()
if not (DATASET_DIR / "metadata.csv").exists():
    st.error(f"Missing dataset metadata: {DATASET_DIR / 'metadata.csv'}")
    st.stop()
if not AUDIT_DECISIONS_PATH.exists():
    st.error(f"Missing manual audit decisions: {AUDIT_DECISIONS_PATH}")
    st.stop()

summary = load_summary(SUMMARY_PATH)
metadata = load_metadata(DATASET_DIR, AUDIT_DECISIONS_PATH)
decisions = load_decisions(DECISIONS_PATH)
regions = region_options(summary, decisions)

with st.sidebar:
    st.header("Region")
    decision_filter = st.selectbox("Decision filter", ["all", "undecided", "val", "train", "exclude"])
    class_filter = st.selectbox("Class filter", ["any target", *TARGET_CLASSES])
    sort_mode = st.selectbox("Sort", ["bbox desc", "positive images desc", "images desc", "region name", "bbox area median desc", "bbox area median asc"])
    show_only_positive = st.checkbox("Show target-positive images only", value=True)
    show_only_negative = st.checkbox("Show target-negative images only", value=False)
    max_images = st.slider("Max images to show", 5, 100, 40, step=5)
    thumb_size = st.slider("Image preview size", 220, 640, 360, step=20)

filtered_regions = regions.copy()
if decision_filter == "undecided":
    filtered_regions = filtered_regions[filtered_regions["decision"].eq("")]
elif decision_filter != "all":
    filtered_regions = filtered_regions[filtered_regions["decision"].eq(decision_filter)]
if class_filter != "any target":
    filtered_regions = filtered_regions[filtered_regions[f"{class_filter}_bbox"].gt(0)]

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
region_objects = metadata[(metadata["region"].astype(str).eq(selected_region)) & metadata["is_target_object"]].copy()
positive_image_keys = set(region_objects["image"].astype(str))
region_images["has_target_objects"] = region_images["image"].astype(str).isin(positive_image_keys)

if show_only_positive and not show_only_negative:
    region_images = region_images[region_images["has_target_objects"]]
elif show_only_negative and not show_only_positive:
    region_images = region_images[~region_images["has_target_objects"]]

region_images = region_images.sort_values(["has_target_objects", "image_name"], ascending=[False, True]).head(max_images)

metric_cols = st.columns(7)
metric_cols[0].metric("images", int(row["images_total"]))
metric_cols[1].metric("positive", int(row["positive_images"]))
metric_cols[2].metric("negative", int(row["negative_images"]))
metric_cols[3].metric("bbox", int(row["bbox_total"]))
for idx, cls in enumerate(TARGET_CLASSES, start=4):
    metric_cols[idx].metric(cls, int(row.get(f"{cls}_bbox", 0)))

left, right = st.columns([2, 1])
with left:
    st.subheader(selected_region)
    st.info("Pre-rendered contact sheets are disabled here. BBoxes below are drawn live from YOLO label files.")

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
st.subheader("Images With Target GT BBox")
legend_cols = st.columns(3)
for col, cls in zip(legend_cols, TARGET_CLASSES):
    col.markdown(f"<span style='color:{CLASS_COLORS[cls]}'>■</span> `{cls}`", unsafe_allow_html=True)

if region_images.empty:
    st.info("No images match image filters.")
else:
    cols = st.columns(3)
    for idx, (_, image_row) in enumerate(region_images.iterrows()):
        image_path = resolve_image_path(image_row)
        label_path = resolve_label_path(image_row)
        objects = region_objects[region_objects["image"].astype(str).eq(str(image_row["image"]))].copy()
        preview = draw_image(image_path, label_path, thumb_size, TARGET_CLASS_IDS)
        with cols[idx % 3]:
            st.image(preview, use_container_width=True)
            st.caption(f"{image_row.get('image_name', image_path.name)} | split={image_row.get('split', '')} | objects={len(objects)}")

st.divider()
st.caption(f"Dataset: {DATASET_DIR}")
st.caption(f"Modalities: {', '.join(MODALITIES)}")
st.caption(f"Region summary: {SUMMARY_PATH}")
st.caption(f"Manual decisions: {DECISIONS_PATH}")
