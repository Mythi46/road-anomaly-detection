"""Create a crop-level dataset for a second-stage damage suppressor.

The suppressor is a binary classifier:

- damage: crops from pothole/crack boxes
- suppress: crops from repair_negative/neutral_negative boxes

It is intentionally trained from public converted labels only. Customer PoC
before/after images should remain held out for evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DAMAGE_IDS = {0, 1}
SUPPRESS_IDS = {2, 3}
CLASS_NAMES = {
    "damage": 0,
    "suppress": 1,
}


def image_paths_from_entry(entry: str) -> list[Path]:
    path = Path(entry)
    if path.is_file() and path.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(entry)


def collect_images(entries: list[str]) -> list[Path]:
    images: list[Path] = []
    for entry in entries:
        images.extend(image_paths_from_entry(entry))
    return images


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Cannot infer label path for {image_path}")


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        bits = line.split()
        if len(bits) != 5:
            continue
        cls_id = int(float(bits[0]))
        x, y, w, h = (float(v) for v in bits[1:])
        boxes.append((cls_id, x, y, w, h))
    return boxes


def crop_bounds(
    box: tuple[int, float, float, float, float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    _cls_id, x, y, w, h = box
    cx = x * image_width
    cy = y * image_height
    bw = w * image_width
    bh = h * image_height
    pad_x = bw * padding
    pad_y = bh * padding
    x1 = max(0, int(round(cx - bw / 2 - pad_x)))
    y1 = max(0, int(round(cy - bh / 2 - pad_y)))
    x2 = min(image_width, int(round(cx + bw / 2 + pad_x)))
    y2 = min(image_height, int(round(cy + bh / 2 + pad_y)))
    return x1, y1, x2, y2


def bucket_for_class(cls_id: int) -> str | None:
    if cls_id in DAMAGE_IDS:
        return "damage"
    if cls_id in SUPPRESS_IDS:
        return "suppress"
    return None


def gather_candidates(
    images: list[Path],
    padding: float,
    min_side: int,
    min_area: int,
) -> dict[str, list[dict]]:
    candidates: dict[str, list[dict]] = {"damage": [], "suppress": []}
    skipped = Counter()

    for image_path in images:
        label_path = label_path_for(image_path)
        boxes = read_yolo_boxes(label_path)
        if not boxes:
            skipped["no_boxes"] += 1
            continue
        img = read_image(image_path)
        if img is None:
            skipped["decode_failed"] += 1
            continue
        image_height, image_width = img.shape[:2]
        for box_index, box in enumerate(boxes):
            cls_id = box[0]
            bucket = bucket_for_class(cls_id)
            if bucket is None:
                continue
            x1, y1, x2, y2 = crop_bounds(box, image_width, image_height, padding)
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_side or crop_h < min_side or crop_w * crop_h < min_area:
                skipped[f"small_{bucket}"] += 1
                continue
            candidates[bucket].append(
                {
                    "image": image_path,
                    "box_index": box_index,
                    "cls_id": cls_id,
                    "xyxy": (x1, y1, x2, y2),
                }
            )

    candidates["_skipped"] = [dict(skipped)]  # type: ignore[assignment]
    return candidates


def write_crops(
    candidates: dict[str, list[dict]],
    split: str,
    output_dir: Path,
    max_per_class: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    summary = {"split": split, "classes": {}}
    for bucket in ("damage", "suppress"):
        items = list(candidates[bucket])
        rng.shuffle(items)
        selected = items[:max_per_class] if max_per_class > 0 else items
        class_dir = output_dir / split / bucket
        class_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        by_source_class = Counter()
        for index, item in enumerate(selected):
            img = read_image(item["image"])
            if img is None:
                continue
            x1, y1, x2, y2 = item["xyxy"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            source_stem = item["image"].stem
            out_name = f"{index:06d}_c{item['cls_id']}_{source_stem}_{item['box_index']}.jpg"
            out_path = class_dir / out_name
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                continue
            buf.tofile(str(out_path))
            written += 1
            by_source_class[item["cls_id"]] += 1
        summary["classes"][bucket] = {
            "available": len(items),
            "written": written,
            "source_class_counts": {str(k): v for k, v in sorted(by_source_class.items())},
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-max-per-class", type=int, default=10000)
    parser.add_argument("--val-max-per-class", type=int, default=2000)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--min-side", type=int, default=24)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    cfg = yaml.safe_load(args.input_yaml.read_text(encoding="utf-8"))
    train_images = collect_images(cfg["train"])
    val_images = collect_images(cfg["val"])

    train_candidates = gather_candidates(
        train_images,
        padding=args.padding,
        min_side=args.min_side,
        min_area=args.min_area,
    )
    val_candidates = gather_candidates(
        val_images,
        padding=args.padding,
        min_side=args.min_side,
        min_area=args.min_area,
    )

    train_summary = write_crops(
        train_candidates,
        split="train",
        output_dir=args.output_dir,
        max_per_class=args.train_max_per_class,
        seed=args.seed,
    )
    val_summary = write_crops(
        val_candidates,
        split="val",
        output_dir=args.output_dir,
        max_per_class=args.val_max_per_class,
        seed=args.seed + 1,
    )

    summary = {
        "input_yaml": str(args.input_yaml),
        "output_dir": str(args.output_dir),
        "class_names": CLASS_NAMES,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train": train_summary,
        "val": val_summary,
        "train_skipped": train_candidates.get("_skipped", [{}])[0],
        "val_skipped": val_candidates.get("_skipped", [{}])[0],
        "settings": {
            "padding": args.padding,
            "min_side": args.min_side,
            "min_area": args.min_area,
            "train_max_per_class": args.train_max_per_class,
            "val_max_per_class": args.val_max_per_class,
            "seed": args.seed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
