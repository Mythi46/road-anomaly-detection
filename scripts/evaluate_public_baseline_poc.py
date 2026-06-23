"""Evaluate the 5-class public-data baseline on local PoC before/after images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RoadAnomalyDetection import config
from RoadAnomalyDetection.data_index import collect_labelled_images


CLASS_NAMES = {
    0: "pothole",
    1: "crack",
    2: "repair_negative",
    3: "neutral_negative",
    4: "deferred_damage",
}
ANOMALY_CLASSES = {0, 1}
REPAIR_CLASSES = {2}
NEUTRAL_CLASSES = {3}
DEFERRED_CLASSES = {4}


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def as_dict(self) -> dict:
        return {
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "confidence": round(self.confidence, 4),
            "xyxy": [round(v, 1) for v in self.xyxy],
            "area": round(self.area, 1),
        }


@dataclass
class EvalResult:
    image: Path
    truth: str
    verdict: str
    anomaly_score: float
    repair_score: float
    neutral_score: float
    deferred_score: float
    detections: list[Detection]


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not decode image: {path}")
    return img


def draw_result(img: np.ndarray, result: EvalResult) -> np.ndarray:
    colors = {
        "anomaly": (0, 0, 255),
        "repair": (0, 180, 0),
        "neutral": (180, 180, 180),
        "deferred": (0, 180, 220),
    }
    out = img.copy()
    for det in result.detections:
        if det.cls_id in ANOMALY_CLASSES:
            color = colors["anomaly"]
        elif det.cls_id in REPAIR_CLASSES:
            color = colors["repair"]
        elif det.cls_id in DEFERRED_CLASSES:
            color = colors["deferred"]
        else:
            color = colors["neutral"]
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    banner_color = (0, 0, 255) if result.verdict == config.LABEL_ABNORMAL else (0, 150, 0)
    banner = (
        f"truth={result.truth} pred={result.verdict} "
        f"a={result.anomaly_score:.2f} r={result.repair_score:.2f}"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), banner_color, -1)
    cv2.putText(out, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def detections_from_prediction(pred) -> list[Detection]:
    if pred.boxes is None or len(pred.boxes) == 0:
        return []
    cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
    confs = pred.boxes.conf.cpu().numpy().astype(float)
    boxes = pred.boxes.xyxy.cpu().numpy().astype(float)
    detections: list[Detection] = []
    for cls_id, conf, box in zip(cls_ids, confs, boxes):
        detections.append(
            Detection(
                cls_id=int(cls_id),
                cls_name=CLASS_NAMES.get(int(cls_id), str(int(cls_id))),
                confidence=float(conf),
                xyxy=tuple(float(v) for v in box),
            )
        )
    return detections


def score(
    detections: Iterable[Detection],
    min_anomaly_area: float,
    include_deferred_as_anomaly: bool,
) -> tuple[float, float, float, float]:
    anomaly_ids = set(ANOMALY_CLASSES)
    if include_deferred_as_anomaly:
        anomaly_ids |= DEFERRED_CLASSES

    anomaly_score = 0.0
    repair_score = 0.0
    neutral_score = 0.0
    deferred_score = 0.0
    for det in detections:
        if det.cls_id in anomaly_ids and det.area >= min_anomaly_area:
            anomaly_score += det.confidence
        elif det.cls_id in REPAIR_CLASSES:
            repair_score += det.confidence
        elif det.cls_id in NEUTRAL_CLASSES:
            neutral_score += det.confidence
        elif det.cls_id in DEFERRED_CLASSES:
            deferred_score += det.confidence
    return anomaly_score, repair_score, neutral_score, deferred_score


def decide(anomaly_score: float, repair_score: float, min_score: float, repair_margin: float) -> str:
    if anomaly_score >= min_score and anomaly_score >= repair_score - repair_margin:
        return config.LABEL_ABNORMAL
    return config.LABEL_NORMAL


def metrics(results: list[EvalResult]) -> dict:
    tp = sum(1 for r in results if r.truth == config.LABEL_ABNORMAL and r.verdict == config.LABEL_ABNORMAL)
    fn = sum(1 for r in results if r.truth == config.LABEL_ABNORMAL and r.verdict == config.LABEL_NORMAL)
    fp = sum(1 for r in results if r.truth == config.LABEL_NORMAL and r.verdict == config.LABEL_ABNORMAL)
    tn = sum(1 for r in results if r.truth == config.LABEL_NORMAL and r.verdict == config.LABEL_NORMAL)
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "truth",
                "verdict",
                "anomaly_score",
                "repair_score",
                "neutral_score",
                "deferred_score",
                "num_detections",
                "detections_json",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    str(result.image),
                    result.truth,
                    result.verdict,
                    f"{result.anomaly_score:.4f}",
                    f"{result.repair_score:.4f}",
                    f"{result.neutral_score:.4f}",
                    f"{result.deferred_score:.4f}",
                    len(result.detections),
                    json.dumps([d.as_dict() for d in result.detections], ensure_ascii=False),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("RoadDetection_v8/runs/local_public_fast_yolo26s_e30/weights/best.pt"),
    )
    parser.add_argument("--csv", type=Path, default=Path("RoadAnomalyDetection/outputs/public_baseline_poc_eval.csv"))
    parser.add_argument("--annotated-dir", type=Path, default=None)
    parser.add_argument("--save-all", action="store_true", help="Save annotations for all images, not only mistakes.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-anomaly-score", type=float, default=0.45)
    parser.add_argument("--min-anomaly-area", type=float, default=1500.0)
    parser.add_argument("--repair-margin", type=float, default=0.10)
    parser.add_argument("--deferred-as-anomaly", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    from ultralytics import YOLO

    pairs = collect_labelled_images()
    print(f"Found {len(pairs)} labelled PoC images")
    print(f"Using model: {args.model}")
    print(
        "Decision: "
        f"conf={args.conf}, min_anomaly_score={args.min_anomaly_score}, "
        f"min_anomaly_area={args.min_anomaly_area}, repair_margin={args.repair_margin}, "
        f"deferred_as_anomaly={args.deferred_as_anomaly}"
    )
    model = YOLO(str(args.model), task="detect")

    results: list[EvalResult] = []
    for index, (image_path, truth) in enumerate(pairs, 1):
        img = read_image(image_path)
        pred = model.predict(source=img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        dets = detections_from_prediction(pred)
        anomaly_score, repair_score, neutral_score, deferred_score = score(
            dets,
            min_anomaly_area=args.min_anomaly_area,
            include_deferred_as_anomaly=args.deferred_as_anomaly,
        )
        verdict = decide(
            anomaly_score=anomaly_score,
            repair_score=repair_score,
            min_score=args.min_anomaly_score,
            repair_margin=args.repair_margin,
        )
        result = EvalResult(
            image=image_path,
            truth=truth,
            verdict=verdict,
            anomaly_score=anomaly_score,
            repair_score=repair_score,
            neutral_score=neutral_score,
            deferred_score=deferred_score,
            detections=dets,
        )
        results.append(result)
        mark = "OK" if truth == verdict else "MISS"
        print(
            f"[{index:03d}/{len(pairs):03d}] {mark:<4} "
            f"truth={truth:<8} pred={verdict:<8} "
            f"a={anomaly_score:.2f} r={repair_score:.2f} n={neutral_score:.2f} d={deferred_score:.2f} "
            f"dets={len(dets):02d} {image_path.name}"
        )

        if args.annotated_dir and (args.save_all or truth != verdict):
            annotated = draw_result(img, result)
            rel_name = f"{index:03d}_{truth}_to_{verdict}_{image_path.stem}.jpg"
            out_path = args.annotated_dir / rel_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                buf.tofile(str(out_path))

    summary = metrics(results)
    print("\nConfusion matrix, positive=abnormal")
    print(f"  TP: {summary['tp']}  FP: {summary['fp']}")
    print(f"  FN: {summary['fn']}  TN: {summary['tn']}")
    print(f"  total:     {summary['total']}")
    print(f"  accuracy:  {summary['accuracy']:.4f}")
    print(f"  precision: {summary['precision']:.4f}")
    print(f"  recall:    {summary['recall']:.4f}")
    print(f"  f1:        {summary['f1']:.4f}")

    write_csv(args.csv, results)
    print(f"\nWrote CSV: {args.csv}")
    if args.annotated_dir:
        print(f"Wrote annotations: {args.annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
