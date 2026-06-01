"""Collect and visualize diagnostic encoder/modalities ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = (
    ("resnet34_li", "resnet34", "Li"),
    ("resnet50_li", "resnet50", "Li"),
    ("resnet34_all_modalities", "resnet34", "all"),
    ("resnet50_all_modalities", "resnet50", "all"),
)
METRIC_COLUMNS = (
    "mean_fg_iou",
    "pixel_accuracy",
    "iou_kurgany_tselye",
    "iou_kurgany_povrezhdennye",
    "iou_gorodishcha",
    "iou_fortifikatsii",
    "iou_arkhitektury",
)
SUMMARY_COLUMNS = ("experiment", "encoder", "modalities", *METRIC_COLUMNS, "best_epoch")


def parse_args() -> argparse.Namespace:
    """Parse comparison CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/encoder_modality_ablation")
    parser.add_argument("--summary-csv", default="runs/encoder_modality_ablation_summary.csv")
    parser.add_argument("--summary-md", default="runs/encoder_modality_ablation_summary.md")
    parser.add_argument(
        "--comparison-grid",
        default="runs/encoder_modality_ablation_comparison_grid.png",
    )
    return parser.parse_args()


def main() -> None:
    """Save the diagnostic summary table, markdown notes and metric grid."""

    args = parse_args()
    rows = collect_rows(Path(args.run_root))
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary_csv, index=False)
    save_comparison_grid(frame, Path(args.comparison_grid))
    save_markdown(frame, Path(args.summary_md))
    print("[diagnostic] Saved summary:", args.summary_csv)
    print("[diagnostic] Saved comparison grid:", args.comparison_grid)
    print("[diagnostic] Saved markdown summary:", args.summary_md)
    print(frame.to_string(index=False))


def collect_rows(run_root: Path) -> list[dict[str, object]]:
    """Load the required metrics from all four run folders."""

    rows = []
    missing = []
    for experiment, encoder, modalities in EXPERIMENTS:
        run_dir = run_root / experiment
        evaluation_path = run_dir / "evaluation.json"
        summary_path = run_dir / "summary.json"
        if not evaluation_path.exists() or not summary_path.exists():
            missing.append(str(run_dir))
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = evaluation["metrics"]
        row = {
            "experiment": experiment,
            "encoder": encoder,
            "modalities": modalities,
            "best_epoch": summary.get("best_epoch"),
        }
        row.update({column: metrics.get(column) for column in METRIC_COLUMNS})
        rows.append(row)
    if missing:
        raise FileNotFoundError("Missing diagnostic run artifacts:\n- " + "\n- ".join(missing))
    return rows


def save_comparison_grid(frame: pd.DataFrame, path: Path) -> None:
    """Plot encoder and modality effects across core pixel metrics."""

    labels = frame["experiment"].str.replace("_modalities", "", regex=False)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    charts = (
        ("mean_fg_iou", "Mean foreground IoU"),
        ("pixel_accuracy", "Pixel accuracy"),
        ("iou_kurgany_tselye", "IoU: intact kurgans"),
        ("iou_kurgany_povrezhdennye", "IoU: damaged kurgans"),
    )
    colors = ["#26734d", "#4f86c6", "#d98c2b", "#a34f7a"]
    for axis, (column, title) in zip(axes.flat, charts, strict=False):
        values = frame[column].astype(float)
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.3)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Diagnostic encoder/modalities ablation: raw metadata, fixed regions")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_markdown(frame: pd.DataFrame, path: Path) -> None:
    """Write a concise interpretation table with pairwise deltas."""

    indexed = frame.set_index("experiment")
    lines = [
        "# Encoder / Modalities Diagnostic Ablation",
        "",
        "This diagnostic raw-data series does not replace the primary `research_split_v1` Stage C benchmark.",
        "",
        "## Results",
        "",
        markdown_table(frame),
        "",
        "## Pairwise Mean FG IoU Deltas",
        "",
        f"- ResNet50 - ResNet34 on Li: {delta(indexed, 'resnet50_li', 'resnet34_li'):+.4f}",
        f"- ResNet50 - ResNet34 on all modalities: {delta(indexed, 'resnet50_all_modalities', 'resnet34_all_modalities'):+.4f}",
        f"- all modalities - Li for ResNet34: {delta(indexed, 'resnet34_all_modalities', 'resnet34_li'):+.4f}",
        f"- all modalities - Li for ResNet50: {delta(indexed, 'resnet50_all_modalities', 'resnet50_li'):+.4f}",
        "",
        "## Protocol",
        "",
        "- fixed train/val region assignment reused from `archaeology_5class_research_split_v1`",
        "- raw metadata without filtering",
        "- diagnostic-only comparison",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delta(frame: pd.DataFrame, left: str, right: str) -> float:
    """Return mean foreground IoU difference."""

    return float(frame.loc[left, "mean_fg_iou"]) - float(frame.loc[right, "mean_fg_iou"])


def markdown_table(frame: pd.DataFrame) -> str:
    """Render selected results as a markdown table."""

    columns = ["experiment", "encoder", "modalities", "mean_fg_iou", "pixel_accuracy", "best_epoch"]
    shown = frame[columns].copy()
    for column in ("mean_fg_iou", "pixel_accuracy"):
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in shown.iterrows()]
    return "\n".join([header, divider, *rows])


if __name__ == "__main__":
    main()
