"""Build YOLO train/val manifests with hard-negative oversampling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANOMALY_IDS = {0, 1}
REPAIR_ID = 2
NEUTRAL_ID = 3
DEFERRED_ID = 4


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Cannot infer label path for image outside an images directory: {image_path}")


def read_class_ids(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(int(float(line.split()[0])))
    return ids


def categorize(ids: set[int]) -> str:
    has_anomaly = bool(ids & ANOMALY_IDS)
    has_repair = REPAIR_ID in ids
    has_neutral = NEUTRAL_ID in ids
    has_deferred = DEFERRED_ID in ids
    if has_repair and not has_anomaly:
        return "repair_no_anomaly"
    if has_neutral and not has_anomaly:
        return "neutral_no_anomaly"
    if (has_repair or has_neutral) and has_anomaly:
        return "mixed_negative_anomaly"
    if has_anomaly:
        return "anomaly_only_or_deferred"
    if has_deferred:
        return "deferred_only"
    return "empty"


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


def repeat_count(
    category: str,
    repair_extra: int,
    neutral_extra: int,
    mixed_extra: int,
    deferred_extra: int,
    empty_extra: int,
) -> int:
    base = 1
    if category == "repair_no_anomaly":
        return base + repair_extra
    if category == "neutral_no_anomaly":
        return base + neutral_extra
    if category == "mixed_negative_anomaly":
        return base + mixed_extra
    if category == "deferred_only":
        return base + deferred_extra
    if category == "empty":
        return base + empty_extra
    return base


def write_lines(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(p.resolve().as_posix() for p in images) + "\n", encoding="utf-8")


def build_train_list(args: argparse.Namespace, images: list[Path]) -> tuple[list[Path], dict[str, Any]]:
    output: list[Path] = []
    categories = Counter()
    emitted = Counter()
    class_images = Counter()
    missing_labels: list[str] = []

    for image_path in images:
        label_path = label_path_for(image_path)
        if not label_path.exists():
            missing_labels.append(str(label_path))
        ids = read_class_ids(label_path)
        for class_id in ids:
            class_images[class_id] += 1
        category = categorize(ids)
        categories[category] += 1
        repeats = repeat_count(
            category=category,
            repair_extra=args.repair_extra,
            neutral_extra=args.neutral_extra,
            mixed_extra=args.mixed_extra,
            deferred_extra=args.deferred_extra,
            empty_extra=args.empty_extra,
        )
        emitted[category] += repeats
        output.extend([image_path] * repeats)

    summary = {
        "input_images": len(images),
        "output_entries": len(output),
        "category_images": dict(categories),
        "category_entries_after_oversampling": dict(emitted),
        "class_image_presence": {str(k): v for k, v in sorted(class_images.items())},
        "missing_label_count": len(missing_labels),
        "missing_label_examples": missing_labels[:20],
        "oversampling": {
            "repair_extra": args.repair_extra,
            "neutral_extra": args.neutral_extra,
            "mixed_extra": args.mixed_extra,
            "deferred_extra": args.deferred_extra,
            "empty_extra": args.empty_extra,
        },
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repair-extra", type=int, default=8)
    parser.add_argument("--neutral-extra", type=int, default=4)
    parser.add_argument("--mixed-extra", type=int, default=1)
    parser.add_argument("--deferred-extra", type=int, default=1)
    parser.add_argument("--empty-extra", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.input_yaml.read_text(encoding="utf-8"))
    train_images = collect_images(cfg["train"])
    val_images = collect_images(cfg["val"])

    train_manifest, summary = build_train_list(args, train_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    val_path = args.output_dir / "val.txt"
    data_path = args.output_dir / "data.yaml"
    summary_path = args.output_dir / "summary.json"

    write_lines(train_path, train_manifest)
    write_lines(val_path, val_images)

    data_yaml = {
        "path": ".",
        "train": train_path.resolve().as_posix(),
        "val": val_path.resolve().as_posix(),
        "names": cfg["names"],
    }
    data_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary["validation_images"] = len(val_images)
    summary["input_yaml"] = str(args.input_yaml)
    summary["output_yaml"] = str(data_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
