"""Convert RDD-style Pascal VOC XML annotations to YOLO labels."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_SETS = {
    "v0": ["pothole", "crack"],
    "multi-bucket": ["pothole", "crack", "repair_negative", "neutral_negative", "deferred_damage"],
}

LABEL_MAP = {
    "D00": "crack",
    "D10": "crack",
    "D20": "crack",
    "D30": "repair_negative",
    "D40": "pothole",
    "D50": "neutral_negative",
    "D60": "neutral_negative",
    "D70": "neutral_negative",
    "D80": "repair_negative",
    "D90": "deferred_damage",
    "LONGITUDINAL_CRACK": "crack",
    "TRANSVERSE_CRACK": "crack",
    "ALLIGATOR_CRACK": "crack",
    "POTHOLE": "pothole",
    "REPAIR": "repair_negative",
    "PATCH": "repair_negative",
    "PATCHY_ROAD": "repair_negative",
    "MANHOLE": "neutral_negative",
    "MAINTENANCE_HOLE": "neutral_negative",
    "LANE_MARKING": "neutral_negative",
    "FADED_MARKING": "neutral_negative",
    "ALLIGATOR_CRACK": "crack",
    "ALLIGATOR_CRACK_HIGH": "crack",
    "ALLIGATOR_CRACK_LOW": "crack",
    "LINEAR_CRACK": "crack",
    "LINEAR_CRACK_HIGH": "crack",
    "LINEAR_CRACK_LOW": "crack",
    "BLOCK_CRACK_HIGH": "deferred_damage",
    "BLOCK_CRACK_LOW": "deferred_damage",
    "FADED_MARKING_HIGH": "neutral_negative",
    "FADED_MARKING_LOW": "neutral_negative",
    "LANE_SHOULDER_DROP_OFF_HIGH": "deferred_damage",
    "MANHOLE_HIGH": "neutral_negative",
    "MANHOLE_LOW": "neutral_negative",
    "PATCH_LOW": "repair_negative",
    "PATCH_AND_UTILITY_CUT_HIGH": "repair_negative",
    "PATCH_AND_UTILITY_CUT_LOW": "repair_negative",
    "POTHOLE_HIGH": "pothole",
    "POTHOLE_LOW": "pothole",
    "RAVELING_HIGH": "deferred_damage",
    "RAVELING_LOW": "deferred_damage",
    "WEATHERING_HIGH": "deferred_damage",
    "WEATHERING_LOW": "deferred_damage",
}

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", normalized).strip("_")


def text_or_empty(node: ET.Element, path: str) -> str:
    value = node.findtext(path)
    return value.strip() if value else ""


def parse_size(root: ET.Element) -> tuple[int, int]:
    width = int(float(text_or_empty(root, "size/width")))
    height = int(float(text_or_empty(root, "size/height")))
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height must be positive")
    return width, height


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def voc_box_to_yolo(box: ET.Element, width: int, height: int) -> tuple[float, float, float, float]:
    xmin = clamp(float(text_or_empty(box, "xmin")), 0.0, float(width))
    ymin = clamp(float(text_or_empty(box, "ymin")), 0.0, float(height))
    xmax = clamp(float(text_or_empty(box, "xmax")), 0.0, float(width))
    ymax = clamp(float(text_or_empty(box, "ymax")), 0.0, float(height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox: {(xmin, ymin, xmax, ymax)}")

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def build_image_index(image_root: Path | None) -> dict[str, Path]:
    if not image_root:
        return {}
    index: dict[str, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for image_path in image_root.rglob(f"*{suffix}"):
            index.setdefault(image_path.name, image_path)
    return index


def find_image(
    xml_path: Path,
    filename: str,
    image_root: Path | None,
    image_index: dict[str, Path],
) -> Path | None:
    if filename and filename in image_index:
        return image_index[filename]

    candidates: list[Path] = []
    if filename:
        if image_root:
            candidates.append(image_root / filename)
        candidates.append(xml_path.parent / filename)

    stem = Path(filename).stem if filename else xml_path.stem
    for suffix in IMAGE_SUFFIXES:
        if image_root:
            candidates.append(image_root / f"{stem}{suffix}")
        candidates.append(xml_path.parent / f"{stem}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def convert_one(
    xml_path: Path,
    output_root: Path,
    class_names: list[str],
    split: str,
    image_root: Path | None,
    image_index: dict[str, Path],
    copy_images: bool,
    write_empty_labels: bool,
) -> dict[str, object]:
    root = ET.parse(xml_path).getroot()
    width, height = parse_size(root)
    filename = text_or_empty(root, "filename")
    image_path = find_image(xml_path, filename, image_root, image_index)

    label_lines = []
    skipped_labels: list[str] = []
    invalid_boxes = 0
    for obj in root.findall("object"):
        source_label = text_or_empty(obj, "name")
        mapped_label = LABEL_MAP.get(normalize_label(source_label))
        if mapped_label not in class_names:
            skipped_labels.append(source_label)
            continue
        box = obj.find("bndbox")
        if box is None:
            invalid_boxes += 1
            continue
        try:
            x_center, y_center, box_width, box_height = voc_box_to_yolo(box, width, height)
        except ValueError:
            invalid_boxes += 1
            continue
        class_id = class_names.index(mapped_label)
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    if not label_lines and not write_empty_labels:
        return {
            "xml": str(xml_path),
            "status": "skipped_empty",
            "skipped_labels": skipped_labels,
            "invalid_boxes": invalid_boxes,
        }

    label_dir = output_root / "labels" / split
    image_dir = output_root / "images" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    target_stem = image_path.stem if image_path else xml_path.stem
    label_path = label_dir / f"{target_stem}.txt"
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    copied_image = None
    if copy_images and image_path:
        target_image = image_dir / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        copied_image = str(target_image)

    return {
        "xml": str(xml_path),
        "status": "converted",
        "label_path": str(label_path),
        "image_path": str(image_path) if image_path else None,
        "copied_image": copied_image,
        "boxes": len(label_lines),
        "skipped_labels": skipped_labels,
        "invalid_boxes": invalid_boxes,
    }


def write_data_yaml(output_root: Path, class_names: list[str], split: str) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    content = (
        f"path: {output_root.as_posix()}\n"
        f"train: images/{split}\n"
        f"val: images/{split}\n"
        "names:\n"
        f"{names}\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Folder containing VOC XML files.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Optional folder containing source images.")
    parser.add_argument("--class-mode", choices=sorted(CLASS_SETS), default="v0")
    parser.add_argument("--split", default="all")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--write-empty-labels", action="store_true")
    args = parser.parse_args()

    class_names = CLASS_SETS[args.class_mode]
    xml_files = sorted(args.input_root.rglob("*.xml"))
    image_index = build_image_index(args.image_root)
    results = [
        convert_one(
            xml_path=xml_path,
            output_root=args.output_root,
            class_names=class_names,
            split=args.split,
            image_root=args.image_root,
            image_index=image_index,
            copy_images=args.copy_images,
            write_empty_labels=args.write_empty_labels,
        )
        for xml_path in xml_files
    ]
    write_data_yaml(args.output_root, class_names, args.split)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "class_mode": args.class_mode,
        "classes": class_names,
        "xml_files": len(xml_files),
        "indexed_images": len(image_index),
        "converted_files": sum(1 for result in results if result["status"] == "converted"),
        "skipped_empty": sum(1 for result in results if result["status"] == "skipped_empty"),
        "boxes": sum(int(result.get("boxes", 0)) for result in results),
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
