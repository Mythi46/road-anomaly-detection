"""Write a combined Ultralytics data.yaml from converted YOLO datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - CLI dependency guard
    raise SystemExit(
        "PyYAML is required. Install with: pip install -r RoadAnomalyDetection/requirements.txt"
    ) from exc


DEFAULT_SOURCES = [
    "rdd2022_v0",
    "rdd2020_v0",
    "n_rdd2024_multi",
    "mwpd_v0",
    "road_damage_potholes_cracks_manholes_multi",
    "water_filled_dry_potholes_v0",
    "attain_os_v1_multi",
    "attain_ws_v1_multi",
    "attain_ws_v2_multi",
]

CLASS_NAMES = [
    "pothole",
    "crack",
    "repair_negative",
    "neutral_negative",
    "deferred_damage",
]


def candidate_image_dirs(dataset_dir: Path, split: str) -> list[Path]:
    candidates = [
        dataset_dir / split / "images",
        dataset_dir / "images" / split,
    ]
    return [path for path in candidates if path.exists()]


def count_images(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def discover_sources(converted_root: Path, source_names: list[str]) -> dict[str, Any]:
    train_dirs: list[str] = []
    val_dirs: list[str] = []
    source_report: dict[str, Any] = {}

    for source_name in source_names:
        dataset_dir = converted_root / source_name
        if not dataset_dir.exists():
            source_report[source_name] = {"status": "missing", "path": str(dataset_dir)}
            continue

        train_candidates = candidate_image_dirs(dataset_dir, "train")
        val_candidates = candidate_image_dirs(dataset_dir, "valid") + candidate_image_dirs(dataset_dir, "val")
        test_candidates = candidate_image_dirs(dataset_dir, "test")

        train_counts = {str(path): count_images(path) for path in train_candidates}
        val_counts = {str(path): count_images(path) for path in val_candidates}
        test_counts = {str(path): count_images(path) for path in test_candidates}

        for path in train_candidates:
            if train_counts[str(path)]:
                train_dirs.append(str(path.resolve()))
        for path in val_candidates:
            if val_counts[str(path)]:
                val_dirs.append(str(path.resolve()))

        source_report[source_name] = {
            "status": "included" if train_candidates or val_candidates or test_candidates else "no_split_dirs",
            "path": str(dataset_dir),
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        }

    if not train_dirs:
        raise ValueError(f"No training image directories found under {converted_root}")
    if not val_dirs:
        val_dirs = train_dirs[:1]

    return {
        "train_dirs": train_dirs,
        "val_dirs": val_dirs,
        "source_report": source_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", help="Converted dataset directory name. Repeatable.")
    parser.add_argument("--name", default="road_public_combined")
    args = parser.parse_args()

    source_names = args.source or DEFAULT_SOURCES
    discovered = discover_sources(args.converted_root, source_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data_yaml = {
        "path": ".",
        "train": discovered["train_dirs"],
        "val": discovered["val_dirs"],
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    args.output.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "name": args.name,
        "output": str(args.output),
        "classes": CLASS_NAMES,
        "train_dirs": discovered["train_dirs"],
        "val_dirs": discovered["val_dirs"],
        "sources": discovered["source_report"],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
