# coding: utf-8
"""CLI: analyze a single image (abnormal / normal verdict) and optionally
save an annotated visualisation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import AnomalyResult, RoadAnomalyDetector


_COLOR_ANOMALY = (0, 0, 255)
_COLOR_REPAIR = (0, 200, 0)
_COLOR_NEUTRAL = (180, 180, 180)


def _color_for(cls_id: int) -> tuple:
    if cls_id in config.ANOMALY_CLASSES:
        return _COLOR_ANOMALY
    if cls_id in config.REPAIR_CLASSES:
        return _COLOR_REPAIR
    return _COLOR_NEUTRAL


def annotate(image_path: Path, result: AnomalyResult) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not decode image: {image_path}")
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = _color_for(det.cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.confidence:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    banner_color = _COLOR_ANOMALY if result.verdict == config.LABEL_ABNORMAL else _COLOR_REPAIR
    banner = f"{result.verdict.upper()}  anom={result.anomaly_score:.2f}  rep={result.repair_score:.2f}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), banner_color, -1)
    cv2.putText(img, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a single road image for anomalies.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional output path for the annotated image. "
                             "Defaults to RoadAnomalyDetection/outputs/<name>_annotated.jpg")
    parser.add_argument("--no-save", action="store_true", help="Do not save an annotated image.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 1

    detector = RoadAnomalyDetector()
    result = detector.analyze(args.image)

    print()
    print(f"Image           : {result.image_path}")
    print(f"Verdict         : {result.verdict.upper()}")
    print(f"Anomaly score   : {result.anomaly_score:.4f}")
    print(f"Repair score    : {result.repair_score:.4f}")
    print(f"Detections      : {len(result.detections)}")
    for det in result.detections:
        bucket = (
            "ANOMALY" if det.cls_id in config.ANOMALY_CLASSES
            else "REPAIR" if det.cls_id in config.REPAIR_CLASSES
            else "NEUTRAL"
        )
        print(f"  - {det.cls_name:<14} conf={det.confidence:.2f}  [{bucket}]")

    if not args.no_save:
        save_path = args.save
        if save_path is None:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = config.OUTPUT_DIR / f"{args.image.stem}_annotated.jpg"
        annotated = annotate(args.image, result)
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            print("Failed to encode annotated image.")
            return 2
        save_path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(save_path))
        print(f"Annotated image : {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
