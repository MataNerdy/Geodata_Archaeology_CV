from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "manual_audit"
INDEX_PATH = AUDIT_DIR / "audit_index.csv"
DECISIONS_PATH = AUDIT_DIR / "audit_decisions.csv"

DECISIONS = ["keep", "remove_image", "fix_label", "uncertain"]
REASONS = [
    "",
    "broken_tile",
    "bad_contrast",
    "edge_artifact",
    "wrong_bbox",
    "object_not_visible",
    "huge_bbox",
    "tiny_uncertain_object",
    "duplicate_or_near_duplicate",
    "other",
]


st.set_page_config(page_title="Manual YOLO Dataset Audit", layout="wide")


@st.cache_data(show_spinner=False)
def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["decision", "reason", "comment"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["n_objects"] = pd.to_numeric(df["n_objects"], errors="coerce").fillna(0).astype(int)
    for col in [
        "bbox_area_min",
        "bbox_area_median",
        "bbox_area_max",
        "valid_fraction",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tile_touches_raster_edge", "has_edge_object"]:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return df


def load_decisions(path: Path) -> pd.DataFrame:
    columns = ["image_id", "decision", "reason", "comment", "updated_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].fillna("")


def save_decision(image_id: str, decision: str, reason: str, comment: str) -> None:
    decisions = load_decisions(DECISIONS_PATH)
    row = {
        "image_id": image_id,
        "decision": decision,
        "reason": reason,
        "comment": comment,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    decisions = decisions[decisions["image_id"] != image_id]
    decisions = pd.concat([decisions, pd.DataFrame([row])], ignore_index=True)
    decisions = decisions.sort_values("image_id").reset_index(drop=True)
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(DECISIONS_PATH, index=False)
    st.cache_data.clear()


def merged_table(index: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    decision_cols = decisions.rename(
        columns={
            "decision": "saved_decision",
            "reason": "saved_reason",
            "comment": "saved_comment",
            "updated_at": "saved_updated_at",
        }
    )
    df = index.merge(decision_cols, on="image_id", how="left")
    for col in ["saved_decision", "saved_reason", "saved_comment", "saved_updated_at"]:
        df[col] = df[col].fillna("")
    return df


def filter_table(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        only_undecided = st.checkbox("Only undecided", value=True)
        decision_filter = st.selectbox(
            "Decision",
            ["all", "keep", "remove_image", "fix_label", "uncertain", "undecided"],
            index=0,
        )
        split = st.selectbox("Split", ["all", "train", "val"])
        polarity = st.radio("Objects", ["all", "positives only", "negatives only"], horizontal=False)
        edge_only = st.checkbox("Edge objects only", value=False)
        sort_mode = st.selectbox("Sort", ["original order", "largest bbox first", "smallest bbox first"])

    filtered = df.copy()
    if only_undecided and decision_filter == "all":
        filtered = filtered[filtered["saved_decision"].eq("")]
    if decision_filter == "undecided":
        filtered = filtered[filtered["saved_decision"].eq("")]
    elif decision_filter != "all":
        filtered = filtered[filtered["saved_decision"].eq(decision_filter)]
    if split != "all":
        filtered = filtered[filtered["split"].eq(split)]
    if polarity == "positives only":
        filtered = filtered[filtered["n_objects"] > 0]
    elif polarity == "negatives only":
        filtered = filtered[filtered["n_objects"] == 0]
    if edge_only:
        filtered = filtered[filtered["has_edge_object"]]
    if sort_mode == "largest bbox first":
        filtered = filtered.sort_values("bbox_area_median", ascending=False, na_position="last")
    elif sort_mode == "smallest bbox first":
        filtered = filtered.sort_values("bbox_area_median", ascending=True, na_position="last")
    return filtered.reset_index(drop=True)


def progress_stats(df: pd.DataFrame) -> dict[str, int]:
    saved = df["saved_decision"].fillna("")
    reviewed_mask = saved.ne("")
    return {
        "total": int(len(df)),
        "reviewed": int(reviewed_mask.sum()),
        "remaining": int((~reviewed_mask).sum()),
        "removed": int(saved.eq("remove_image").sum()),
        "uncertain": int(saved.eq("uncertain").sum()),
        "fix_label": int(saved.eq("fix_label").sum()),
    }


def fmt(value: object, ndigits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


st.title("Manual YOLO Dataset Audit")

if not INDEX_PATH.exists():
    st.error(f"Missing audit index: {INDEX_PATH}")
    st.stop()

index = load_index(INDEX_PATH)
decisions = load_decisions(DECISIONS_PATH)
table = merged_table(index, decisions)
stats = progress_stats(table)

metric_cols = st.columns(6)
metric_cols[0].metric("total", stats["total"])
metric_cols[1].metric("reviewed", stats["reviewed"])
metric_cols[2].metric("remaining", stats["remaining"])
metric_cols[3].metric("removed", stats["removed"])
metric_cols[4].metric("uncertain", stats["uncertain"])
metric_cols[5].metric("fix_label", stats["fix_label"])

filtered = filter_table(table)
if filtered.empty:
    st.success("No images match current filters.")
    st.stop()

if "audit_idx" not in st.session_state:
    st.session_state.audit_idx = 0
st.session_state.audit_idx = min(st.session_state.audit_idx, len(filtered) - 1)

top_left, top_mid, top_right = st.columns([1, 3, 1])
with top_left:
    if st.button("Prev", use_container_width=True):
        st.session_state.audit_idx = max(0, st.session_state.audit_idx - 1)
        st.rerun()
with top_mid:
    selected_idx = st.selectbox(
        "Image",
        range(len(filtered)),
        index=st.session_state.audit_idx,
        format_func=lambda i: f"{i + 1}/{len(filtered)} | {filtered.iloc[i].image_id} | objs={filtered.iloc[i].n_objects}",
        label_visibility="collapsed",
    )
    st.session_state.audit_idx = selected_idx
with top_right:
    if st.button("Next", use_container_width=True):
        st.session_state.audit_idx = min(len(filtered) - 1, st.session_state.audit_idx + 1)
        st.rerun()

row = filtered.iloc[st.session_state.audit_idx]
preview_path = AUDIT_DIR / str(row["preview_path"])

left, right = st.columns([2, 1])
with left:
    st.subheader(row["image_id"])
    if preview_path.exists():
        st.image(str(preview_path), use_container_width=True)
    else:
        st.error(f"Missing preview: {preview_path}")

with right:
    st.subheader("Metadata")
    metadata_rows = [
        ("filename", Path(str(row["image_path"])).name),
        ("split", row["split"]),
        ("n_objects", row["n_objects"]),
        ("source_class_names", row["source_class_names"]),
        ("bbox_area_min", fmt(row["bbox_area_min"], 1)),
        ("bbox_area_median", fmt(row["bbox_area_median"], 1)),
        ("bbox_area_max", fmt(row["bbox_area_max"], 1)),
        ("valid_fraction", fmt(row["valid_fraction"], 3)),
        ("tile_touches_raster_edge", row["tile_touches_raster_edge"]),
        ("has_edge_object", row["has_edge_object"]),
        ("saved_decision", row["saved_decision"]),
        ("saved_reason", row["saved_reason"]),
        ("saved_comment", row["saved_comment"]),
    ]
    st.dataframe(pd.DataFrame(metadata_rows, columns=["field", "value"]), hide_index=True, use_container_width=True)

    current_reason = row["saved_reason"] if row["saved_reason"] in REASONS else ""
    reason = st.selectbox("reason", REASONS, index=REASONS.index(current_reason))
    comment = st.text_area("comment", value=row["saved_comment"], height=110)

    st.caption("Decision is saved immediately to `manual_audit/audit_decisions.csv`.")
    b1, b2 = st.columns(2)
    b3, b4 = st.columns(2)
    if b1.button("keep", use_container_width=True):
        save_decision(row["image_id"], "keep", reason, comment)
        st.session_state.audit_idx = min(st.session_state.audit_idx, max(0, len(filtered) - 2))
        st.rerun()
    if b2.button("remove_image", use_container_width=True):
        save_decision(row["image_id"], "remove_image", reason, comment)
        st.session_state.audit_idx = min(st.session_state.audit_idx, max(0, len(filtered) - 2))
        st.rerun()
    if b3.button("fix_label", use_container_width=True):
        save_decision(row["image_id"], "fix_label", reason, comment)
        st.session_state.audit_idx = min(st.session_state.audit_idx, max(0, len(filtered) - 2))
        st.rerun()
    if b4.button("uncertain", use_container_width=True):
        save_decision(row["image_id"], "uncertain", reason, comment)
        st.session_state.audit_idx = min(st.session_state.audit_idx, max(0, len(filtered) - 2))
        st.rerun()

st.divider()
st.caption(f"Audit index: {INDEX_PATH}")
st.caption(f"Decisions: {DECISIONS_PATH}")
