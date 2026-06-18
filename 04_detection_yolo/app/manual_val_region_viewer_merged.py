from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT.parent / "datasets" / "dataset_yolo_bbox_v3i_li_archaeological_object_merged"
REVIEW_DIR = ROOT / "manual_val_region_review_merged"
DECISIONS_PATH = REVIEW_DIR / "manual_val_regions.csv"
SUMMARY_PATH = REVIEW_DIR / "region_summary.csv"

DECISIONS = ["train", "val", "exclude"]
CLASS_COLOR = "#00ff66"


st.set_page_config(page_title="Merged Archaeological Object Split Viewer", layout="wide")


@st.cache_data(show_spinner=False)
def load_metadata(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    df["is_positive"] = df["is_positive"].astype(bool)
    if "source_class_name" not in df.columns:
        df["source_class_name"] = df["class_name"]
    for col in ["bbox_area_px", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "yolo_xc", "yolo_yc", "yolo_w", "yolo_h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_region_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    images = metadata.drop_duplicates("image_id").copy()
    boxes = metadata[metadata["is_target_object"].astype(bool)].copy()
    positive_ids = set(boxes["image_id"].astype(str))
    rows = []
    for region, region_images in images.groupby("region", sort=True):
        region_boxes = boxes[boxes["region"].astype(str).eq(str(region))].copy()
        region_positive = region_images[region_images["image_id"].astype(str).isin(positive_ids)]
        bbox_by_source = region_boxes["source_class_name"].value_counts()
        current_split = sorted(region_images["split"].dropna().astype(str).unique())
        current_decision = current_split[0] if len(current_split) == 1 and current_split[0] in DECISIONS else ""
        rows.append(
            {
                "region": region,
                "images_total": int(len(region_images)),
                "positive_images": int(len(region_positive)),
                "negative_images": int(len(region_images) - len(region_positive)),
                "bbox_total": int(len(region_boxes)),
                "kurgany_tselye_bbox": int(bbox_by_source.get("kurgany_tselye", 0)),
                "kurgany_povrezhdennye_bbox": int(bbox_by_source.get("kurgany_povrezhdennye", 0)),
                "gorodishcha_bbox": int(bbox_by_source.get("gorodishcha", 0)),
                "fortifikatsii_bbox": int(bbox_by_source.get("fortifikatsii", 0)),
                "arkhitektury_bbox": int(bbox_by_source.get("arkhitektury", 0)),
                "bbox_area_median": float(region_boxes["bbox_area_px"].median()) if not region_boxes.empty else 0.0,
                "objects_per_positive_image_mean": float(region_boxes.groupby("image_id").size().mean()) if not region_boxes.empty else 0.0,
                "decision": current_decision,
                "comment": "",
            }
        )
    return pd.DataFrame(rows).sort_values("region")


def ensure_review_files(metadata: pd.DataFrame) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_region_summary(metadata)
    if not SUMMARY_PATH.exists():
        summary.to_csv(SUMMARY_PATH, index=False)
    if not DECISIONS_PATH.exists():
        summary[
            [
                "region",
                "images_total",
                "positive_images",
                "negative_images",
                "bbox_total",
                "decision",
                "comment",
            ]
        ].to_csv(DECISIONS_PATH, index=False)


def load_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["decision", "comment"]:
        if col not in df.columns:
            df[col] = ""
    return df


def load_regions(summary_path: Path, decisions_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path).fillna("")
    decisions = load_decisions(decisions_path)[["region", "decision", "comment"]]
    for col in [
        "images_total",
        "positive_images",
        "negative_images",
        "bbox_total",
        "kurgany_tselye_bbox",
        "kurgany_povrezhdennye_bbox",
        "gorodishcha_bbox",
        "fortifikatsii_bbox",
        "arkhitektury_bbox",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).astype(int)
    for col in ["bbox_area_median", "objects_per_positive_image_mean"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    merged = summary.drop(columns=[c for c in ["decision", "comment"] if c in summary.columns]).merge(decisions, on="region", how="left")
    merged["decision"] = merged["decision"].fillna("")
    merged["comment"] = merged["comment"].fillna("")
    return merged


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


def resolve_path(path_value: str, kind: str, split: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        DATASET_DIR / kind / split / path.name,
        DATASET_DIR / path,
        DATASET_DIR.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def draw_image(image_path: Path, label_path: Path, max_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(max_size / original_w, max_size / original_h)
    preview = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for _, xc, yc, bw, bh in parse_label_file(label_path):
        x1 = (xc - bw / 2) * original_w
        y1 = (yc - bh / 2) * original_h
        x2 = (xc + bw / 2) * original_w
        y2 = (yc + bh / 2) * original_h
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=CLASS_COLOR, width=3)
        label = "arch_object"
        text_w = max(62, len(label) * 6 + 6)
        y_text = max(0, box[1] - 14)
        draw.rectangle([box[0], y_text, box[0] + text_w, y_text + 13], fill=CLASS_COLOR)
        draw.text((box[0] + 3, y_text + 1), label, fill="black", font=font)
    return preview


def fmt_region(row: pd.Series) -> str:
    decision = row["decision"] or "undecided"
    return f"{row['region']} | {decision} | img={row['images_total']} pos={row['positive_images']} bbox={row['bbox_total']}"


st.title("Merged Archaeological Object Validation Viewer")
st.caption("Dataset: dataset_yolo_bbox_v3i_li_archaeological_object_merged. One YOLO class: archaeological_object.")

if not (DATASET_DIR / "metadata.csv").exists():
    st.error(f"Missing dataset metadata: {DATASET_DIR / 'metadata.csv'}")
    st.stop()

metadata = load_metadata(DATASET_DIR)
ensure_review_files(metadata)
regions = load_regions(SUMMARY_PATH, DECISIONS_PATH)

with st.sidebar:
    st.header("Region")
    decision_filter = st.selectbox("Decision filter", ["all", "val", "train", "exclude"])
    class_filter = st.selectbox(
        "Source class filter",
        [
            "any",
            "kurgany_tselye",
            "kurgany_povrezhdennye",
            "gorodishcha",
            "fortifikatsii",
            "arkhitektury",
        ],
    )
    sort_mode = st.selectbox("Sort", ["bbox desc", "positive desc", "images desc", "region", "bbox area median desc", "bbox area median asc"])
    show_only_positive = st.checkbox("Show positive images only", value=True)
    show_only_negative = st.checkbox("Show negative images only", value=False)
    max_images = st.slider("Max images to show", 5, 120, 45, step=5)
    thumb_size = st.slider("Image preview size", 220, 700, 360, step=20)

filtered_regions = regions.copy()
if decision_filter != "all":
    filtered_regions = filtered_regions[filtered_regions["decision"].eq(decision_filter)]
if class_filter != "any":
    filtered_regions = filtered_regions[filtered_regions[f"{class_filter}_bbox"].gt(0)]

if sort_mode == "bbox desc":
    filtered_regions = filtered_regions.sort_values("bbox_total", ascending=False)
elif sort_mode == "positive desc":
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
region_images = metadata[metadata["region"].astype(str).eq(selected_region)].drop_duplicates("image_id").copy()
region_objects = metadata[(metadata["region"].astype(str).eq(selected_region)) & metadata["is_target_object"].astype(bool)].copy()
positive_ids = set(region_objects["image_id"].astype(str))
region_images["has_target_objects"] = region_images["image_id"].astype(str).isin(positive_ids)

if show_only_positive and not show_only_negative:
    region_images = region_images[region_images["has_target_objects"]]
elif show_only_negative and not show_only_positive:
    region_images = region_images[~region_images["has_target_objects"]]

region_images = region_images.sort_values(["has_target_objects", "image_name"], ascending=[False, True]).head(max_images)

metric_cols = st.columns(9)
metric_cols[0].metric("images", int(row["images_total"]))
metric_cols[1].metric("positive", int(row["positive_images"]))
metric_cols[2].metric("negative", int(row["negative_images"]))
metric_cols[3].metric("bbox", int(row["bbox_total"]))
metric_cols[4].metric("whole kurgans", int(row["kurgany_tselye_bbox"]))
metric_cols[5].metric("damaged kurgans", int(row["kurgany_povrezhdennye_bbox"]))
metric_cols[6].metric("gorodishcha", int(row["gorodishcha_bbox"]))
metric_cols[7].metric("fortifikatsii", int(row["fortifikatsii_bbox"]))
metric_cols[8].metric("arkhitektury", int(row["arkhitektury_bbox"]))

left, right = st.columns([2, 1])
with left:
    st.subheader(selected_region)
    source_counts = (
        region_objects["source_class_name"]
        .value_counts()
        .rename_axis("source_class")
        .reset_index(name="bbox")
        .sort_values("source_class")
    )
    st.dataframe(source_counts, hide_index=True, use_container_width=True)

with right:
    st.subheader("Split decision")
    current_decision = row["decision"] if row["decision"] in DECISIONS else "train"
    current_comment = row["comment"] if isinstance(row["comment"], str) else ""
    decision = st.selectbox("decision", DECISIONS, index=DECISIONS.index(current_decision))
    comment = st.text_area("comment", value=current_comment, height=120)
    if st.button("Save decision", use_container_width=True):
        save_region_decision(selected_region, decision, comment)
        st.success("Saved")
        st.rerun()

    st.code(
        "python scripts/build_merged_archaeological_object_dataset.py "
        "--source-dir ../datasets/dataset_yolo_bbox "
        "--kurgan-dataset ../datasets/dataset_yolo_bbox_v3h_li_manual_curated_val_no_saratov "
        "--merged-regions-csv manual_val_region_review_merged/manual_val_regions.csv "
        "--output-root ../datasets "
        "--output-name dataset_yolo_bbox_v3i_li_archaeological_object_merged "
        "--overwrite",
        language="bash",
    )

st.divider()
st.subheader("Images")

if region_images.empty:
    st.info("No images match current image filters.")
else:
    cols = st.columns(3)
    for idx, (_, image_row) in enumerate(region_images.iterrows()):
        split = str(image_row["split"])
        image_path = resolve_path(str(image_row["image"]), "images", split)
        label_path = resolve_path(str(image_row["label"]), "labels", split)
        objects = region_objects[region_objects["image_id"].astype(str).eq(str(image_row["image_id"]))]
        with cols[idx % 3]:
            st.image(draw_image(image_path, label_path, thumb_size), use_container_width=True)
            st.caption(
                f"{image_row['image_name']} | {image_row['split']} | "
                f"objects={len(objects)} | {', '.join(sorted(objects['source_class_name'].dropna().astype(str).unique())) or 'negative'}"
            )

st.divider()
st.subheader("Current Region Decisions")
overview = regions.groupby("decision", dropna=False).agg(
    regions=("region", "count"),
    images=("images_total", "sum"),
    positive=("positive_images", "sum"),
    bbox=("bbox_total", "sum"),
).reset_index()
st.dataframe(overview, hide_index=True, use_container_width=True)
