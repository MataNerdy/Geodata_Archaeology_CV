"""Streamlit entrypoint for the YOLO bbox dataset viewer.

The current viewer implementation lives in `skripts/visualize_yolo_labels.py`.
This wrapper gives the portfolio project the expected `app/` entrypoint without
deleting the original exploratory script.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "skripts" / "visualize_yolo_labels.py"

runpy.run_path(str(VIEWER), run_name="__main__")
