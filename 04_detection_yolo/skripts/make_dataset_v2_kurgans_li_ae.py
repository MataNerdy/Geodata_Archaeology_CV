from pathlib import Path
import shutil
import random

import pandas as pd


OLD_DIR = Path("dataset_yolo_bbox")
NEW_DIR = Path("dataset_yolo_bbox_v2_kurgans_li_ae")

META_PATH = OLD_DIR / "metadata.csv"

KEEP_MODALITIES = {"Li", "Ae"}
KEEP_OLD_CLASS_IDS = {0, 1}

CLASS_ID_MAP = {
    0: 0,  # kurgany_tselye
    1: 1,  # kurgany_povrezhdennye
}

NAMES = {
    0: "kurgany_tselye",
    1: "kurgany_povrezhdennye",
}

NEGATIVE_RATIO = 0.25
RANDOM_SEED = 42


def make_dirs():
    for split in ["train", "val"]:
        (NEW_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (NEW_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_yolo_label(path: Path):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        if cls_id not in KEEP_OLD_CLASS_IDS:
            continue

        new_cls = CLASS_ID_MAP[cls_id]
        boxes.append((new_cls, *coords))

    return boxes


def valid_box(box):
    _, xc, yc, w, h = box
    return (
        0 <= xc <= 1
        and 0 <= yc <= 1
        and 0 < w <= 1
        and 0 < h <= 1
    )


def write_label(path: Path, boxes):
    lines = [
        f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for cls_id, xc, yc, w, h in boxes
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_yaml():
    text = f"""path: {NEW_DIR.resolve()}

train: images/train
val: images/val

names:
  0: kurgany_tselye
  1: kurgany_povrezhdennye
"""
    (NEW_DIR / "dataset.yaml").write_text(text, encoding="utf-8")


def main():
    random.seed(RANDOM_SEED)
    make_dirs()

    meta = pd.read_csv(META_PATH)

    # один ряд на изображение — иначе одно изображение может копироваться много раз
    images = meta.drop_duplicates("image").copy()

    # сначала режем по модальности
    images = images[images["modality"].isin(KEEP_MODALITIES)].copy()

    new_rows = []
    copied_images = 0
    positive_images = 0
    negative_images = 0
    total_boxes = 0
    bad_boxes = 0
    skipped_missing = 0

    for _, row in images.iterrows():
        old_img = Path(row["image"])
        old_lbl = Path(row["label"])

        if not old_img.exists():
            skipped_missing += 1
            continue

        boxes = parse_yolo_label(old_lbl)
        boxes_ok = []

        for b in boxes:
            if valid_box(b):
                boxes_ok.append(b)
            else:
                bad_boxes += 1

        is_positive = len(boxes_ok) > 0

        # negative оставляем не все, чтобы фон не задавил курганы
        if not is_positive and random.random() > NEGATIVE_RATIO:
            continue

        split = row["split"]
        new_img = NEW_DIR / "images" / split / old_img.name
        new_lbl = NEW_DIR / "labels" / split / old_lbl.name

        shutil.copy2(old_img, new_img)
        write_label(new_lbl, boxes_ok)

        copied_images += 1
        total_boxes += len(boxes_ok)

        if is_positive:
            positive_images += 1
        else:
            negative_images += 1

        base = row.to_dict()
        base["image"] = str(new_img)
        base["label"] = str(new_lbl)
        base["is_positive"] = bool(is_positive)
        base["n_objects"] = len(boxes_ok)

        if boxes_ok:
            for cls_id, xc, yc, w, h in boxes_ok:
                obj = base.copy()
                obj["class_id"] = cls_id
                obj["class_name"] = NAMES[cls_id]
                obj["yolo_xc"] = xc
                obj["yolo_yc"] = yc
                obj["yolo_w"] = w
                obj["yolo_h"] = h
                new_rows.append(obj)
        else:
            obj = base.copy()
            obj["class_id"] = None
            obj["class_name"] = None
            obj["yolo_xc"] = None
            obj["yolo_yc"] = None
            obj["yolo_w"] = None
            obj["yolo_h"] = None
            new_rows.append(obj)

    new_meta = pd.DataFrame(new_rows)
    new_meta.to_csv(NEW_DIR / "metadata.csv", index=False)
    write_yaml()

    print("=" * 80)
    print("DONE: v2 kurgans Li+Ae")
    print("dataset:", NEW_DIR)
    print("images copied:", copied_images)
    print("positive images:", positive_images)
    print("negative images:", negative_images)
    print("total boxes:", total_boxes)
    print("bad boxes skipped:", bad_boxes)
    print("missing images skipped:", skipped_missing)

    if not new_meta.empty:
        print("\nImages by split/modality/positive:")
        print(
            new_meta.drop_duplicates("image")
            .groupby(["split", "modality", "is_positive"])
            .size()
        )

        print("\nBBoxes by class:")
        print(
            new_meta[new_meta["is_positive"]]
            .groupby("class_name")
            .size()
        )

    print("\nyaml:", NEW_DIR / "dataset.yaml")
    print("metadata:", NEW_DIR / "metadata.csv")


if __name__ == "__main__":
    main()