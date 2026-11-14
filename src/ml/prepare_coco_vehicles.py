"""Trích tập con "phương tiện" từ COCO 2017 và chuyển sang định dạng YOLO.

    python -m ml.prepare_coco_vehicles --coco /path/to/coco --out data/detection --max-images 5000

Yêu cầu: annotations/instances_{train,val}2017.json và thư mục ảnh tương ứng.
Lớp COCO → lớp đề tài: bicycle(2)→0, bus(6)→1, car(3)→2, motorcycle(4)→3, truck(8)→4
(category_id gốc của COCO, khác với chỉ số 0..79 của YOLO).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

COCO_CAT_TO_OURS = {2: 0, 6: 1, 3: 2, 4: 3, 8: 4}


def convert(coco_root: Path, split: str, out: Path, max_images: int | None, seed: int = 0):
    ann = json.loads((coco_root / "annotations" / f"instances_{split}2017.json").read_text())
    imgs = {im["id"]: im for im in ann["images"]}
    per_img = defaultdict(list)
    for a in ann["annotations"]:
        if a["category_id"] in COCO_CAT_TO_OURS and not a.get("iscrowd", 0):
            per_img[a["image_id"]].append(a)

    ids = list(per_img)
    random.Random(seed).shuffle(ids)
    if max_images:
        ids = ids[:max_images]

    dst_split = "train" if split == "train" else "val"
    (out / "images" / dst_split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
    for iid in ids:
        im = imgs[iid]
        W, H = im["width"], im["height"]
        lines = []
        for a in per_img[iid]:
            x, y, w, h = a["bbox"]                       # COCO: góc trên-trái + kích thước
            if w < 4 or h < 4:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H    # YOLO: tâm + kích thước, chuẩn hoá
            lines.append(f"{COCO_CAT_TO_OURS[a['category_id']]} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        if not lines:
            continue
        shutil.copy(coco_root / f"{split}2017" / im["file_name"], out / "images" / dst_split / im["file_name"])
        (out / "labels" / dst_split / Path(im["file_name"]).with_suffix(".txt").name).write_text("\n".join(lines))
    print(f"{split}: {len(ids)} ảnh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", default="data/detection")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()
    for split in ("train", "val"):
        convert(Path(args.coco), split, Path(args.out), args.max_images)


if __name__ == "__main__":
    main()
