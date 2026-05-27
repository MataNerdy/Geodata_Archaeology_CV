"""Evaluate notebook-style weighted polygon F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.polygon_metrics import competition_like_f1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-geojson", required=True)
    parser.add_argument("--gt-geojson", required=True)
    parser.add_argument("--out-dir", default="runs/multiclass/deeplab_all_5_classes")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    """Compute competition-like weighted F1."""

    args = parse_args()
    pred_geojson = json.loads(Path(args.pred_geojson).read_text(encoding="utf-8"))
    gt_geojson = json.loads(Path(args.gt_geojson).read_text(encoding="utf-8"))
    score, rows = competition_like_f1(
        pred_geojson,
        gt_geojson,
        iou_threshold=float(args.iou_threshold),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "competition_metric.csv", index=False)
    payload = {
        "competition_like_weighted_f1": score,
        "iou_threshold": args.iou_threshold,
        "rows": rows.to_dict(orient="records"),
    }
    (out_dir / "competition_metric.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"competition-like weighted F1: {score:.4f}")


if __name__ == "__main__":
    main()
