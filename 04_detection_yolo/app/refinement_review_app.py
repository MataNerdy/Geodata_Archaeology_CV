from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "reports" / "proposals" / "v3i_conf005" / "filter_audit" / "candidate_features.csv"
DEFAULT_DECISIONS = ROOT / "reports" / "refinement_manual_review.csv"

REVIEW_LABELS = [
    "trash",
    "object",
    "plausible_object",
    "terrain_like",
    "uncertain",
    "bad_crop",
    "skip",
]

NEGATIVE_SAFE_LABELS = {"trash", "terrain_like", "bad_crop"}
NOT_NEGATIVE_LABELS = {"object", "plausible_object", "uncertain", "skip"}


st.set_page_config(page_title="Refinement Crop Review", layout="wide")


def as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


@st.cache_data(show_spinner=False)
def load_metadata(path_text: str) -> pd.DataFrame:
    path = as_path(path_text)
    df = pd.read_csv(path)
    df = df.copy()
    if "review_id" not in df.columns:
        if "candidate_id" in df.columns:
            df["review_id"] = df["candidate_id"].astype(str)
        elif "crop_id" in df.columns:
            df["review_id"] = df["crop_id"].astype(str)
        else:
            df["review_id"] = [f"row_{idx:06d}" for idx in range(len(df))]

    if "group" not in df.columns:
        df["group"] = df.apply(infer_group, axis=1)

    for col in ["yolo_conf", "max_iou_with_gt"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["region", "source_class_name", "crop_path", "overlay_path", "image_path"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in ["bbox_touches_image_edge", "pad_touches_image_edge", "tile_touches_raster_edge", "has_edge_object"]:
        if col not in df.columns:
            df[col] = False
        df[col] = to_bool_series(df[col])

    return df


def infer_group(row: pd.Series) -> str:
    if boolish(row.get("is_tp_iou03", False)):
        return "tp"
    max_iou = pd.to_numeric(pd.Series([row.get("max_iou_with_gt", pd.NA)]), errors="coerce").iloc[0]
    conf = pd.to_numeric(pd.Series([row.get("yolo_conf", pd.NA)]), errors="coerce").iloc[0]
    area_norm = pd.to_numeric(pd.Series([row.get("bbox_area_norm", pd.NA)]), errors="coerce").iloc[0]
    aspect_ratio = pd.to_numeric(pd.Series([row.get("aspect_ratio", pd.NA)]), errors="coerce").iloc[0]

    if pd.notna(max_iou) and max_iou >= 0.3:
        return "tp"
    if (
        (pd.notna(aspect_ratio) and aspect_ratio > 3)
        or (pd.notna(area_norm) and pd.notna(conf) and area_norm > 0.1 and conf < 0.15)
        or boolish(row.get("bbox_touches_image_edge", False))
    ):
        return "obvious_fp"
    return "plausible_fp"


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_bool_series(series: pd.Series) -> pd.Series:
    return series.map(boolish).fillna(False).astype(bool)


def load_decisions(path: Path) -> pd.DataFrame:
    columns = ["review_id", "manual_label", "comment", "updated_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].fillna("")


def save_decision(path: Path, review_id: str, label: str, comment: str) -> None:
    decisions = load_decisions(path)
    decisions = decisions[decisions["review_id"].astype(str) != str(review_id)]
    row = {
        "review_id": str(review_id),
        "manual_label": label,
        "comment": comment,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    decisions = pd.concat([decisions, pd.DataFrame([row])], ignore_index=True)
    decisions = decisions.sort_values("review_id").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(path, index=False)


def merge_decisions(metadata: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    merged = metadata.merge(decisions, on="review_id", how="left")
    for col in ["manual_label", "comment", "updated_at"]:
        merged[col] = merged[col].fillna("")
    return merged


def resolve_file(path_value: object, metadata_path: Path, roots: list[Path]) -> Path | None:
    if pd.isna(path_value) or str(path_value).strip() == "":
        return None
    raw = Path(str(path_value))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            metadata_path.parent / raw,
            ROOT / raw,
            ROOT.parent / raw,
        ]
    )
    for root in roots:
        if root:
            candidates.extend([root / raw, root / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def numeric_range_filter(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    finite = values.dropna()
    if finite.empty:
        st.sidebar.caption(f"{label}: no numeric values")
        return pd.Series(True, index=df.index)
    low, high = float(finite.min()), float(finite.max())
    selected = st.sidebar.slider(label, min_value=low, max_value=high, value=(low, high))
    return values.between(selected[0], selected[1], inclusive="both") | values.isna()


def filter_table(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar:
        st.header("Filters")
        only_unreviewed = st.checkbox("Only unreviewed", value=False)

        groups = sorted([v for v in filtered["group"].dropna().astype(str).unique() if v])
        group_filter = st.multiselect("group", groups, default=groups)
        if group_filter:
            filtered = filtered[filtered["group"].isin(group_filter)]

        regions = sorted([v for v in filtered["region"].dropna().astype(str).unique() if v])
        region_filter = st.multiselect("region", regions, default=[])
        if region_filter:
            filtered = filtered[filtered["region"].isin(region_filter)]

        classes = sorted([v for v in filtered["source_class_name"].dropna().astype(str).unique() if v])
        class_filter = st.multiselect("source_class_name", classes, default=[])
        if class_filter:
            filtered = filtered[filtered["source_class_name"].isin(class_filter)]

        filtered = filtered[numeric_range_filter(filtered, "yolo_conf", "yolo_conf")]
        filtered = filtered[numeric_range_filter(filtered, "max_iou_with_gt", "max_iou_with_gt")]

        edge_options = {
            "bbox_touches_image_edge": "bbox edge",
            "pad_touches_image_edge": "padded crop edge",
            "tile_touches_raster_edge": "tile raster edge",
            "has_edge_object": "has edge object",
        }
        selected_edges = st.multiselect("edge flags must be true", list(edge_options.keys()), format_func=edge_options.get)
        for col in selected_edges:
            filtered = filtered[filtered[col]]

        label_filter = st.selectbox("manual label", ["all", "unreviewed", *REVIEW_LABELS])
        if only_unreviewed and label_filter == "all":
            filtered = filtered[filtered["manual_label"].eq("")]
        elif label_filter == "unreviewed":
            filtered = filtered[filtered["manual_label"].eq("")]
        elif label_filter != "all":
            filtered = filtered[filtered["manual_label"].eq(label_filter)]

        sort_mode = st.selectbox(
            "sort",
            [
                "metadata order",
                "highest confidence",
                "lowest confidence",
                "highest IoU",
                "lowest IoU",
            ],
        )

    if sort_mode == "highest confidence":
        filtered = filtered.sort_values("yolo_conf", ascending=False, na_position="last")
    elif sort_mode == "lowest confidence":
        filtered = filtered.sort_values("yolo_conf", ascending=True, na_position="last")
    elif sort_mode == "highest IoU":
        filtered = filtered.sort_values("max_iou_with_gt", ascending=False, na_position="last")
    elif sort_mode == "lowest IoU":
        filtered = filtered.sort_values("max_iou_with_gt", ascending=True, na_position="last")

    return filtered.reset_index(drop=True)


def progress(filtered: pd.DataFrame) -> tuple[int, int]:
    reviewed = int(filtered["manual_label"].ne("").sum())
    total = int(len(filtered))
    return reviewed, total


def show_image(path: Path | None, caption: str) -> None:
    if path is None:
        st.info(f"No {caption} path.")
    elif path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing {caption}: {path}")


def metadata_rows(row: pd.Series) -> pd.DataFrame:
    fields = [
        "review_id",
        "group",
        "manual_label",
        "region",
        "source_id",
        "source_class_name",
        "image_id",
        "split",
        "yolo_conf",
        "max_iou_with_gt",
        "bbox_area_norm",
        "aspect_ratio",
        "bbox_touches_image_edge",
        "pad_touches_image_edge",
        "tile_touches_raster_edge",
        "has_edge_object",
        "crop_path",
        "overlay_path",
        "image_path",
    ]
    rows = []
    for field in fields:
        if field in row.index:
            rows.append({"field": field, "value": row[field]})
    return pd.DataFrame(rows)


st.title("Crop-Level Refinement Review")
st.caption("YOLO proposals -> padded crops -> safe manual typology for refinement datasets.")

with st.sidebar:
    st.header("Inputs")
    metadata_text = st.text_input("metadata.csv", value=str(DEFAULT_METADATA))
    crops_root_text = st.text_input("crops directory", value=str(ROOT / "reports" / "proposals" / "v3i_conf005" / "crops"))
    overlays_root_text = st.text_input("overlays directory", value=str(ROOT / "reports" / "proposals" / "v3i_conf005" / "overlays"))
    tiles_root_text = st.text_input("optional full tile root", value="")
    decisions_text = st.text_input("review output CSV", value=str(DEFAULT_DECISIONS))
    st.divider()
    st.warning("Do not use `plausible_object`, `uncertain`, or `skip` as negative labels.")

metadata_path = as_path(metadata_text)
decisions_path = as_path(decisions_text)
if not metadata_path.exists():
    st.error(f"Missing metadata file: {metadata_path}")
    st.stop()

metadata = load_metadata(str(metadata_path))
decisions = load_decisions(decisions_path)
table = merge_decisions(metadata, decisions)
filtered = filter_table(table)

if filtered.empty:
    st.success("No rows match current filters.")
    st.stop()

reviewed, total = progress(filtered)
global_reviewed = int(table["manual_label"].ne("").sum())
global_total = int(len(table))

metric_cols = st.columns(4)
metric_cols[0].metric("filtered reviewed", reviewed)
metric_cols[1].metric("filtered total", total)
metric_cols[2].metric("global reviewed", global_reviewed)
metric_cols[3].metric("global total", global_total)
st.progress(reviewed / total if total else 0.0)

if "refinement_review_idx" not in st.session_state:
    st.session_state.refinement_review_idx = 0
st.session_state.refinement_review_idx = min(st.session_state.refinement_review_idx, len(filtered) - 1)

nav_left, nav_mid, nav_right = st.columns([1, 4, 1])
with nav_left:
    if st.button("Prev", use_container_width=True):
        st.session_state.refinement_review_idx = max(0, st.session_state.refinement_review_idx - 1)
        st.rerun()
with nav_mid:
    idx = st.selectbox(
        "Candidate",
        range(len(filtered)),
        index=st.session_state.refinement_review_idx,
        format_func=lambda i: (
            f"{i + 1}/{len(filtered)} | {filtered.iloc[i].review_id} | "
            f"{filtered.iloc[i].group} | conf={filtered.iloc[i].yolo_conf:.3f}"
        ),
        label_visibility="collapsed",
    )
    st.session_state.refinement_review_idx = idx
with nav_right:
    if st.button("Next", use_container_width=True):
        st.session_state.refinement_review_idx = min(len(filtered) - 1, st.session_state.refinement_review_idx + 1)
        st.rerun()

row = filtered.iloc[st.session_state.refinement_review_idx]
roots = [
    as_path(crops_root_text) if crops_root_text else Path(),
    as_path(overlays_root_text) if overlays_root_text else Path(),
    as_path(tiles_root_text) if tiles_root_text else Path(),
]
crop_path = resolve_file(row.get("crop_path", ""), metadata_path, roots)
overlay_path = resolve_file(row.get("overlay_path", ""), metadata_path, roots)
tile_path = resolve_file(row.get("image_path", ""), metadata_path, roots)

image_col, info_col = st.columns([2.2, 1])
with image_col:
    tab_crop, tab_overlay, tab_tile = st.tabs(["Crop", "Overlay", "Full tile"])
    with tab_crop:
        show_image(crop_path, "crop")
    with tab_overlay:
        show_image(overlay_path, "overlay")
    with tab_tile:
        show_image(tile_path, "full tile")

with info_col:
    st.subheader(str(row["review_id"]))
    current_label = str(row.get("manual_label", ""))
    if current_label:
        st.success(f"Current label: {current_label}")
    else:
        st.info("Not reviewed yet.")

    st.dataframe(metadata_rows(row), hide_index=True, use_container_width=True, height=440)

    comment = st.text_area("comment", value=str(row.get("comment", "")), height=120)

    st.caption("Manual labels")
    cols = st.columns(2)
    for i, label in enumerate(REVIEW_LABELS):
        button_type = "primary" if label in {"trash", "object", "plausible_object"} else "secondary"
        if cols[i % 2].button(label, key=f"label_{label}", use_container_width=True, type=button_type):
            save_decision(decisions_path, str(row["review_id"]), label, comment)
            st.session_state.refinement_review_idx = min(st.session_state.refinement_review_idx, max(0, len(filtered) - 2))
            st.rerun()

    st.divider()
    st.markdown(
        "**Negative-safe labels:** "
        + ", ".join(f"`{label}`" for label in sorted(NEGATIVE_SAFE_LABELS))
        + "\n\n"
        "**Never auto-negative:** "
        + ", ".join(f"`{label}`" for label in sorted(NOT_NEGATIVE_LABELS))
    )

st.caption(f"Metadata: {metadata_path}")
st.caption(f"Decisions: {decisions_path}")
