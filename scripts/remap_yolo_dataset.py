"""Remap an existing YOLO dataset to the project's internal class taxonomy."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "PyYAML is required. Install with: pip install -r RoadAnomalyDetection/requirements.txt"
    ) from exc


TARGET_CLASSES = {
    "v0": ["pothole", "crack"],
    "multi-bucket": ["pothole", "crack", "repair_negative", "neutral_negative", "deferred_damage"],
}

SOURCE_TO_INTERNAL = {
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
    "pothole": "pothole",
    "potholes": "pothole",
    "Potholes": "pothole",
    "crack": "crack",
    "cracks": "crack",
    "Cracks": "crack",
    "maintenance_hole": "neutral_negative",
    "maintenance_holes": "neutral_negative",
    "manhole_cover": "neutral_negative",
    "manhole": "neutral_negative",
    "manholes": "neutral_negative",
    "Manholes": "neutral_negative",
    "patch": "repair_negative",
    "repair": "repair_negative",
    "Alligator_crack": "crack",
    "Alligator_crack_-_High": "crack",
    "Alligator_crack_-_Low": "crack",
    "Alligator_crack_-_low": "crack",
    "Linear_crack": "crack",
    "Linear_crack_-_High": "crack",
    "Linear_crack_-_Low": "crack",
    "Block_crack_-_High": "deferred_damage",
    "Block_crack_-_Low": "deferred_damage",
    "Faded_marking_-_High": "neutral_negative",
    "Faded_marking_-_Low": "neutral_negative",
    "Lane_shoulder_drop-off_-_High": "deferred_damage",
    "Manhole_-_High": "neutral_negative",
    "Manhole_-_Low": "neutral_negative",
    "Patch_-_Low": "repair_negative",
    "Patch_and_utility_cut-_High": "repair_negative",
    "Patch_and_utility_cut-_Low": "repair_negative",
    "Pothole_-_High": "pothole",
    "Pothole_-_Low": "pothole",
    "Raveling_-_High": "deferred_damage",
    "Raveling_-_Low": "deferred_damage",
    "Weathering_-_High": "deferred_damage",
    "Weathering_-_Low": "deferred_damage",
}

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
FLAT_SPLIT = "__flat__"


def load_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data.yaml: {path}")
    return data


def parse_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    raise ValueError("data.yaml must contain names as a list or dictionary")


def normalize_source_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def source_name_for_id(source_names: list[str], source_id: int) -> str:
    if 0 <= source_id < len(source_names):
        return normalize_source_name(source_names[source_id])
    return f"unknown_{source_id}"


def find_split_dirs(input_root: Path) -> list[str]:
    if (input_root / "images").exists() or (input_root / "Images").exists():
        return [FLAT_SPLIT]
    splits = []
    for split in ["train", "valid", "val", "test"]:
        if (input_root / split / "images").exists() or (input_root / split / "Images").exists():
            splits.append(split)
    if not splits:
        raise ValueError(f"No YOLO split directories found under {input_root}")
    return splits


def input_image_dir(input_root: Path, split: str) -> Path:
    base = input_root if split == FLAT_SPLIT else input_root / split
    return base / "images" if (base / "images").exists() else base / "Images"


def input_label_dir(input_root: Path, split: str) -> Path:
    base = input_root if split == FLAT_SPLIT else input_root / split
    return base / "labels" if (base / "labels").exists() else base / "Labels"


def output_split_name(split: str) -> str:
    return "train" if split == FLAT_SPLIT else split


def image_to_label_path(input_root: Path, split: str, image_path: Path) -> Path:
    return input_label_dir(input_root, split) / f"{image_path.stem}.txt"


def coords_to_bbox(coords: list[str]) -> list[str] | None:
    values = [float(value) for value in coords]
    if len(values) == 4:
        return [f"{value:.6f}" for value in values]
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        return [f"{value:.6f}" for value in [x_center, y_center, width, height]]
    return None


def remap_label_file(
    label_path: Path,
    source_names: list[str],
    target_classes: list[str],
    unknown_policy: str,
) -> tuple[list[str], Counter[str], Counter[str]]:
    output_lines = []
    source_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()

    if not label_path.exists():
        return output_lines, source_counter, skipped_counter

    for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            skipped_counter["malformed_line"] += 1
            continue
        source_id = int(float(parts[0]))
        source_name = source_name_for_id(source_names, source_id)
        source_counter[source_name] += 1
        mapped_name = SOURCE_TO_INTERNAL.get(source_name)

        if mapped_name is None:
            if unknown_policy == "error":
                raise ValueError(f"Unknown source class {source_name} in {label_path}")
            skipped_counter[source_name] += 1
            continue
        if mapped_name not in target_classes:
            skipped_counter[mapped_name] += 1
            continue

        bbox = coords_to_bbox(parts[1:])
        if bbox is None:
            skipped_counter["malformed_coords"] += 1
            continue
        target_id = target_classes.index(mapped_name)
        output_lines.append(" ".join([str(target_id), *bbox]))

    return output_lines, source_counter, skipped_counter


def copy_or_link_image(image_path: Path, target_path: Path, mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if mode == "copy":
        shutil.copy2(image_path, target_path)
    elif mode == "hardlink":
        try:
            target_path.hardlink_to(image_path)
        except OSError:
            shutil.copy2(image_path, target_path)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def write_data_yaml(output_root: Path, target_classes: list[str], splits: list[str]) -> None:
    output_splits = [output_split_name(split) for split in splits]
    train_split = "train" if "train" in output_splits else output_splits[0]
    val_split = "valid" if "valid" in output_splits else ("val" if "val" in output_splits else train_split)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(target_classes))
    content = (
        f"path: {output_root.as_posix()}\n"
        f"train: {train_split}/images\n"
        f"val: {val_split}/images\n"
        "names:\n"
        f"{names}\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument(
        "--source-name",
        action="append",
        help="Source class name by class id. Repeat in id order if data.yaml is absent.",
    )
    parser.add_argument("--target-mode", choices=sorted(TARGET_CLASSES), default="multi-bucket")
    parser.add_argument("--image-mode", choices=["copy", "hardlink"], default="hardlink")
    parser.add_argument("--unknown-policy", choices=["skip", "error"], default="skip")
    args = parser.parse_args()

    data_yaml = args.data_yaml or args.input_root / "data.yaml"
    if data_yaml.exists():
        source_data = load_data_yaml(data_yaml)
        source_names = parse_names(source_data)
    elif args.source_name:
        source_names = args.source_name
    else:
        raise ValueError("Provide --data-yaml or repeated --source-name values.")
    target_classes = TARGET_CLASSES[args.target_mode]
    splits = find_split_dirs(args.input_root)

    total_images = 0
    total_boxes = 0
    total_output_boxes = 0
    source_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for split in splits:
        source_image_dir = input_image_dir(args.input_root, split)
        target_split = output_split_name(split)
        image_paths = []
        for suffix in IMAGE_SUFFIXES:
            image_paths.extend(source_image_dir.glob(f"*{suffix}"))

        for image_path in sorted(image_paths):
            total_images += 1
            label_path = image_to_label_path(args.input_root, split, image_path)
            output_lines, file_source_counts, file_skipped_counts = remap_label_file(
                label_path=label_path,
                source_names=source_names,
                target_classes=target_classes,
                unknown_policy=args.unknown_policy,
            )
            source_counts.update(file_source_counts)
            skipped_counts.update(file_skipped_counts)
            total_boxes += sum(file_source_counts.values())
            total_output_boxes += len(output_lines)

            target_image = args.output_root / target_split / "images" / image_path.name
            target_label = args.output_root / target_split / "labels" / f"{image_path.stem}.txt"
            copy_or_link_image(image_path, target_image, args.image_mode)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    write_data_yaml(args.output_root, target_classes, splits)
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "source_names": source_names,
        "target_mode": args.target_mode,
        "target_classes": target_classes,
        "splits": [output_split_name(split) for split in splits],
        "images": total_images,
        "source_boxes": total_boxes,
        "output_boxes": total_output_boxes,
        "source_counts": dict(sorted(source_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "remap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
