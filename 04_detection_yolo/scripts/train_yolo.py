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
    parser = argparse.ArgumentParser(description="Train YOLO on a configured detection dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.pop("model")
    data = cfg.pop("data")
    model = YOLO(model_name)
    model.train(data=data, **cfg)


if __name__ == "__main__":
    main()
