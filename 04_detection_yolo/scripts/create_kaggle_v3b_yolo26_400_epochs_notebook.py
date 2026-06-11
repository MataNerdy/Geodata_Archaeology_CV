#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


BASE = Path("notebooks/kaggle_yolo_v3b_400_epochs.ipynb")
OUT = Path("notebooks/kaggle_yolo_v3b_yolo26_400_epochs.ipynb")


def replace_all(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    if not BASE.exists():
        raise FileNotFoundError(f"Base notebook not found: {BASE}")

    nb = json.loads(BASE.read_text(encoding="utf-8"))
    replacements = {
        "# YOLO v3b Li-only Long Run: 400 Epoch Limit": "# YOLO26 v3b Li-only Long Run: 400 Epoch Limit",
        "YOLO v3b Li-only Long Run": "YOLO26 v3b Li-only Long Run",
        "## 4. Train YOLOv8n for 400 Epoch Limit": "## 4. Train YOLO26n for 400 Epoch Limit",
        "Model/config: same as the previous `v3b_li_medium` baseline": "Model/config: same as the previous `v3b_li_medium` baseline except the model family is changed from YOLOv8n to YOLO26n",
        "Only changed parameter: `epochs = 400`": "Changed parameters relative to the original v3b baseline: `model = yolo26n.pt` and `epochs = 400`",
        'MODEL_NAME = "yolov8n.pt"': 'MODEL_NAME = "yolo26n.pt"',
        'OUTPUT_ROOT = Path("/kaggle/working/yolo_v3b_400_epochs")': 'OUTPUT_ROOT = Path("/kaggle/working/yolo_v3b_yolo26_400_epochs")',
        'RUN_NAME = "v3b_li_medium_yolov8n_img640_epochs400"': 'RUN_NAME = "v3b_li_medium_yolo26n_img640_epochs400"',
        '"Experiment": "v3b_400_epoch_limit"': '"Experiment": "v3b_yolo26n_400_epoch_limit"',
        'comparison_path = ANALYSIS_DIR / "v3b_400_vs_100_metrics.csv"': 'comparison_path = ANALYSIS_DIR / "v3b_yolo26_400_vs_yolov8_100_metrics.csv"',
        '"# v3b Li-only 400-Epoch-Limit Report"': '"# v3b Li-only YOLO26n 400-Epoch-Limit Report"',
        'report_path = ANALYSIS_DIR / "v3b_400_epoch_report.md"': 'report_path = ANALYSIS_DIR / "v3b_yolo26_400_epoch_report.md"',
        'archive_path = Path("/kaggle/working") / f"yolo_v3b_400_epochs_{timestamp}.zip"': 'archive_path = Path("/kaggle/working") / f"yolo_v3b_yolo26_400_epochs_{timestamp}.zip"',
    }

    for cell in nb["cells"]:
        cell["source"] = [
            replace_all(line, replacements)
            for line in cell.get("source", [])
        ]

    install_cell = "".join(nb["cells"][4]["source"])
    install_cell = install_cell.replace(
        '[sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pandas", "pyyaml", "pillow", "matplotlib"],',
        '[sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ultralytics/ultralytics.git", "pandas", "pyyaml", "pillow", "matplotlib"],',
    )
    nb["cells"][4]["source"] = [line + "\n" for line in install_cell.splitlines()]

    train_cell = "".join(nb["cells"][8]["source"])
    train_cell = train_cell.replace(
        "model = YOLO(MODEL_NAME)\n",
        "print('Using model:', MODEL_NAME)\nmodel = YOLO(MODEL_NAME)\n",
    )
    nb["cells"][8]["source"] = [line + "\n" for line in train_cell.splitlines()]

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
