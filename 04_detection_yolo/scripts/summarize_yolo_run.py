#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print compact summary from Ultralytics results.csv.")
    parser.add_argument("results_csv", type=Path)
    return parser.parse_args()


def print_row(label: str, row: dict[str, str]) -> None:
    print(label)
    for col in METRIC_COLUMNS:
        if col in row:
            print(f"  {col}: {row[col]}")


def main() -> None:
    args = parse_args()
    with args.results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]

    if not rows:
        raise SystemExit(f"No rows in {args.results_csv}")

    print("rows:", len(rows))
    print_row("last", rows[-1])

    for metric in ["metrics/mAP50(B)", "metrics/recall(B)", "metrics/precision(B)", "metrics/mAP50-95(B)"]:
        if metric not in rows[0]:
            continue
        best = max(rows, key=lambda row: float(row[metric]))
        print()
        print_row(f"best by {metric}", best)


if __name__ == "__main__":
    main()
