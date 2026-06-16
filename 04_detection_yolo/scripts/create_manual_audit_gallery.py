from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DECISIONS = ["keep", "remove_image", "fix_label", "uncertain"]
REASONS = [
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

CLASS_COLORS = {
    "kurgany_tselye": "#00ff66",
    "kurgany_povrezhdennye": "#ffcc00",
    "gorodishcha": "#00c8ff",
    "fortifikatsii": "#ff4f81",
    "arkhitektury": "#c77dff",
}
DEFAULT_COLOR = "#00ff66"
CLASS_LABELS = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
    2: "gorodishcha",
    3: "fortifikatsii",
    4: "arkhitektury",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manual audit gallery for YOLO bbox dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../datasets/dataset_yolo_bbox"),
        help="Path to dataset_yolo_bbox directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("manual_audit"),
        help="Output folder with audit_index.csv, audit_gallery.html and previews/.",
    )
    parser.add_argument("--preview-size", type=int, default=420)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument(
        "--overwrite-previews",
        action="store_true",
        help="Regenerate preview images even if they already exist.",
    )
    return parser.parse_args()


def resolve_dataset_path(path_value: str | Path, dataset_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        dataset_dir.parent / path,
        dataset_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return dataset_dir.parent / path


def numeric_summary(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return (np.nan, np.nan, np.nan)
    return (float(clean.min()), float(clean.median()), float(clean.max()))


def bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def make_image_id(row: pd.Series) -> str:
    image_name = str(row.get("image_name") or Path(str(row["image"])).name)
    return f"{row['split']}_{Path(image_name).stem}"


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def yolo_to_xyxy(
    box: tuple[int, float, float, float, float],
    image_w: int,
    image_h: int,
) -> tuple[int, float, float, float, float]:
    cls_id, xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * image_w
    y1 = (yc - bh / 2) * image_h
    x2 = (xc + bw / 2) * image_w
    y2 = (yc + bh / 2) * image_h
    return cls_id, x1, y1, x2, y2


def draw_preview(
    image_path: Path,
    label_path: Path,
    out_path: Path,
    preview_size: int,
    jpeg_quality: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    scale = min(preview_size / original_w, preview_size / original_h)
    new_w = max(1, int(original_w * scale))
    new_h = max(1, int(original_h * scale))
    preview = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for box in parse_label_file(label_path):
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, original_w, original_h)
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale
        class_name = CLASS_LABELS.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = class_name.replace("kurgany_", "")[:14] if class_name else "obj"
        label_w = max(28, len(label) * 6 + 6)
        label_y = max(0, y1 - 14)
        draw.rectangle([x1, label_y, x1 + label_w, label_y + 13], fill=color)
        draw.text((x1 + 3, label_y + 1), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=jpeg_quality, optimize=True)


def build_index(
    metadata: pd.DataFrame,
    dataset_dir: Path,
    out_dir: Path,
    preview_size: int,
    jpeg_quality: int,
    overwrite_previews: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    previews_dir = out_dir / "previews"
    grouped = metadata.groupby("image", sort=False)
    total = len(grouped)

    for idx, (image_value, group) in enumerate(grouped, start=1):
        first = group.iloc[0]
        image_id = make_image_id(first)
        image_path = resolve_dataset_path(image_value, dataset_dir)
        label_path = resolve_dataset_path(first["label"], dataset_dir)
        preview_rel = Path("previews") / f"{image_id}.jpg"
        preview_abs = out_dir / preview_rel

        if not image_path.exists():
            print(f"WARNING: missing image: {image_path}")
        elif overwrite_previews or not preview_abs.exists():
            draw_preview(image_path, label_path, preview_abs, preview_size, jpeg_quality)

        object_rows = group[group["class_name"].notna()].copy()
        area_min, area_median, area_max = numeric_summary(object_rows["bbox_area_px"])
        width_min, width_median, width_max = numeric_summary(object_rows["bbox_x2_px"] - object_rows["bbox_x1_px"])
        height_min, height_median, height_max = numeric_summary(object_rows["bbox_y2_px"] - object_rows["bbox_y1_px"])
        source_classes = sorted(str(v) for v in object_rows["class_name"].dropna().unique())

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "split": first.get("split", ""),
                "modality": first.get("modality", ""),
                "source_class_names": "; ".join(source_classes),
                "n_objects": int(pd.to_numeric(first.get("n_objects", len(object_rows)), errors="coerce") or 0),
                "bbox_area_min": area_min,
                "bbox_area_median": area_median,
                "bbox_area_max": area_max,
                "bbox_width_min": width_min,
                "bbox_width_median": width_median,
                "bbox_width_max": width_max,
                "bbox_height_min": height_min,
                "bbox_height_median": height_median,
                "bbox_height_max": height_max,
                "valid_fraction": first.get("valid_fraction", np.nan),
                "tile_touches_raster_edge": bool_value(first.get("tile_touches_raster_edge", False)),
                "has_edge_object": bool_value(first.get("has_edge_object", False)),
                "preview_path": preview_rel.as_posix(),
                "decision": "",
                "reason": "",
                "comment": "",
            }
        )

        if idx % 500 == 0 or idx == total:
            print(f"Processed {idx}/{total} images")

    return pd.DataFrame(rows)


def fmt_value(value: object, ndigits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def write_html(index: pd.DataFrame, out_path: Path) -> None:
    cards = []
    for _, row in index.iterrows():
        classes = html.escape(str(row["source_class_names"]) or "negative")
        edge_flags = []
        if bool_value(row["tile_touches_raster_edge"]):
            edge_flags.append("tile edge")
        if bool_value(row["has_edge_object"]):
            edge_flags.append("edge object")
        flag_html = "".join(f'<span class="flag">{html.escape(flag)}</span>' for flag in edge_flags)
        cards.append(
            f"""
            <article class="card" data-split="{html.escape(str(row['split']))}" data-modality="{html.escape(str(row['modality']))}">
              <a href="{html.escape(row['preview_path'])}" target="_blank">
                <img loading="lazy" src="{html.escape(row['preview_path'])}" alt="{html.escape(row['image_id'])}">
              </a>
              <div class="meta">
                <h2>{html.escape(row['image_id'])}</h2>
                <p><b>split</b> {html.escape(str(row['split']))} <b>modality</b> {html.escape(str(row['modality']))}</p>
                <p><b>objects</b> {row['n_objects']} <b>bbox area median</b> {fmt_value(row['bbox_area_median'], 1)}</p>
                <p><b>valid_fraction</b> {fmt_value(row['valid_fraction'], 3)}</p>
                <p>{flag_html}</p>
                <p class="classes">{classes}</p>
              </div>
            </article>
            """
        )

    decision_hint = " ".join(f"<code>{item}</code>" for item in DECISIONS)
    reason_hint = " ".join(f"<code>{item}</code>" for item in REASONS)
    class_hint = " ".join(
        f'<span class="class-chip"><span style="background:{color}"></span>{html.escape(name)}</span>'
        for name, color in CLASS_COLORS.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Dataset Manual Audit</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f5f2;
      --ink: #1f2522;
      --muted: #69706b;
      --line: #d6d8d2;
      --card: #ffffff;
      --accent: #0f766e;
      --flag: #f5d547;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(245, 245, 242, 0.96);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .hint {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
      gap: 8px 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    code {{
      display: inline-block;
      margin: 0 4px 4px 0;
      padding: 2px 6px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font-size: 12px;
    }}
    main {{
      padding: 20px 24px 48px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 16px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--card);
    }}
    .card img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #111;
      border-bottom: 1px solid var(--line);
    }}
    .meta {{
      padding: 10px 12px 12px;
      font-size: 13px;
    }}
    .meta h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .meta p {{
      margin: 5px 0;
      color: var(--muted);
    }}
    .meta b {{
      color: var(--ink);
      font-weight: 650;
    }}
    .classes {{
      min-height: 18px;
      color: var(--accent) !important;
      overflow-wrap: anywhere;
    }}
    .flag {{
      display: inline-block;
      margin: 0 6px 4px 0;
      padding: 2px 7px;
      border-radius: 4px;
      background: var(--flag);
      color: #2b2600;
      font-size: 12px;
      font-weight: 650;
    }}
    .class-legend {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .class-chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin: 0 10px 4px 0;
      white-space: nowrap;
    }}
    .class-chip span {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgba(0, 0, 0, 0.25);
    }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO Dataset Manual Audit</h1>
    <div class="hint">
      <div><b>decision</b><br>{decision_hint}</div>
      <div><b>reason</b><br>{reason_hint}</div>
    </div>
    <div class="class-legend"><b>bbox colors</b><br>{class_hint}</div>
  </header>
  <main>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    index = build_index(
        metadata,
        dataset_dir,
        out_dir,
        args.preview_size,
        args.jpeg_quality,
        args.overwrite_previews,
    )
    index_path = out_dir / "audit_index.csv"
    html_path = out_dir / "audit_gallery.html"
    index.to_csv(index_path, index=False)
    write_html(index, html_path)

    print(f"Images: {len(index)}")
    print(f"Index: {index_path}")
    print(f"Gallery: {html_path}")
    print(f"Previews: {out_dir / 'previews'}")


if __name__ == "__main__":
    main()
