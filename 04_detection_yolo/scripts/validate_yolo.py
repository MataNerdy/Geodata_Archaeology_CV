#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO weights at one or more confidence thresholds.")
    parser.add_argument("--config", type=Path, help="Validation YAML config.")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25])
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name-prefix", default="threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}

    if not args.config and (args.weights is None or args.data is None):
        raise SystemExit("--weights and --data are required when --config is not provided")

    weights = args.weights or Path(cfg["weights"])
    data = args.data or Path(cfg["data"])
    imgsz = int(cfg.get("imgsz", args.imgsz))
    conf_values = args.conf if args.conf != [0.25] or "conf" not in cfg else cfg["conf"]
    project = str(cfg.get("project", args.project))
    name_prefix = str(cfg.get("name_prefix", args.name_prefix))

    model = YOLO(str(weights))
    for conf in conf_values:
        name = f"{name_prefix}_conf_{str(conf).replace('.', '_')}"
        print("=" * 80)
        print(f"Validating conf={conf} -> {name}")
        model.val(
            data=str(data),
            imgsz=imgsz,
            conf=conf,
            project=project,
            name=name,
            exist_ok=True,
            plots=True,
        )


if __name__ == "__main__":
    main()
