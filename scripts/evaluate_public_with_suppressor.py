"""Evaluate YOLO detections with a crop-level second-stage suppressor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

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
    damage_prob: float | None = None
    suppress_prob: float | None = None
    suppressed: bool = False

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
            "damage_prob": None if self.damage_prob is None else round(self.damage_prob, 4),
            "suppress_prob": None if self.suppress_prob is None else round(self.suppress_prob, 4),
            "suppressed": self.suppressed,
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


def build_classifier() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def load_suppressor(path: Path, device: torch.device) -> tuple[nn.Module, dict, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    class_to_idx = checkpoint.get("class_to_idx", {"damage": 0, "suppress": 1})
    img_size = int(checkpoint.get("img_size", 224))
    return model, class_to_idx, img_size


def classifier_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def padded_bounds(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    px = bw * padding
    py = bh * padding
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(width, int(round(x2 + px))),
        min(height, int(round(y2 + py))),
    )


@torch.no_grad()
def classify_crop(
    model: nn.Module,
    tf,
    crop_bgr: np.ndarray,
    class_to_idx: dict,
    device: torch.device,
) -> tuple[float, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    tensor = tf(pil).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    damage_prob = float(probs[int(class_to_idx["damage"])])
    suppress_prob = float(probs[int(class_to_idx["suppress"])])
    return damage_prob, suppress_prob


def detections_from_prediction(pred) -> list[Detection]:
    if pred.boxes is None or len(pred.boxes) == 0:
        return []
    cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
    confs = pred.boxes.conf.cpu().numpy().astype(float)
    boxes = pred.boxes.xyxy.cpu().numpy().astype(float)
    return [
        Detection(
            cls_id=int(cls_id),
            cls_name=CLASS_NAMES.get(int(cls_id), str(int(cls_id))),
            confidence=float(conf),
            xyxy=tuple(float(v) for v in box),
        )
        for cls_id, conf, box in zip(cls_ids, confs, boxes)
    ]


def apply_suppressor(
    image: np.ndarray,
    detections: list[Detection],
    model: nn.Module,
    tf,
    class_to_idx: dict,
    device: torch.device,
    damage_threshold: float,
    crop_padding: float,
) -> None:
    height, width = image.shape[:2]
    for det in detections:
        if det.cls_id not in ANOMALY_CLASSES:
            continue
        x1, y1, x2, y2 = padded_bounds(det.xyxy, width, height, crop_padding)
        if x2 <= x1 or y2 <= y1:
            det.suppressed = True
            det.damage_prob = 0.0
            det.suppress_prob = 1.0
            continue
        crop = image[y1:y2, x1:x2]
        damage_prob, suppress_prob = classify_crop(model, tf, crop, class_to_idx, device)
        det.damage_prob = damage_prob
        det.suppress_prob = suppress_prob
        det.suppressed = damage_prob < damage_threshold


def score(
    detections: list[Detection],
    min_anomaly_area: float,
    include_deferred_as_anomaly: bool,
    score_mode: str,
) -> tuple[float, float, float, float]:
    anomaly_ids = set(ANOMALY_CLASSES)
    if include_deferred_as_anomaly:
        anomaly_ids |= DEFERRED_CLASSES

    anomaly_score = 0.0
    repair_score = 0.0
    neutral_score = 0.0
    deferred_score = 0.0
    for det in detections:
        if det.cls_id in anomaly_ids:
            if det.area < min_anomaly_area or det.suppressed:
                continue
            if score_mode == "conf_times_damage_prob" and det.damage_prob is not None:
                anomaly_score += det.confidence * det.damage_prob
            else:
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


def draw_result(image: np.ndarray, result: EvalResult) -> np.ndarray:
    out = image.copy()
    for det in result.detections:
        if det.cls_id in ANOMALY_CLASSES and det.suppressed:
            color = (180, 180, 180)
        elif det.cls_id in ANOMALY_CLASSES:
            color = (0, 0, 255)
        elif det.cls_id in REPAIR_CLASSES:
            color = (0, 180, 0)
        elif det.cls_id in DEFERRED_CLASSES:
            color = (0, 180, 220)
        else:
            color = (180, 180, 180)
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if det.damage_prob is None:
            label = f"{det.cls_name} {det.confidence:.2f}"
        else:
            label = f"{det.cls_name} {det.confidence:.2f} pd={det.damage_prob:.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    banner_color = (0, 0, 255) if result.verdict == config.LABEL_ABNORMAL else (0, 150, 0)
    banner = f"truth={result.truth} pred={result.verdict} a={result.anomaly_score:.2f} r={result.repair_score:.2f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), banner_color, -1)
    cv2.putText(out, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suppressor", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--annotated-dir", type=Path, default=None)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-anomaly-score", type=float, default=0.45)
    parser.add_argument("--min-anomaly-area", type=float, default=1500.0)
    parser.add_argument("--repair-margin", type=float, default=0.10)
    parser.add_argument("--damage-threshold", type=float, default=0.65)
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--score-mode", choices=["conf", "conf_times_damage_prob"], default="conf")
    parser.add_argument("--deferred-as-anomaly", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suppressor, class_to_idx, img_size = load_suppressor(args.suppressor, device)
    tf = classifier_transform(img_size)
    yolo = YOLO(str(args.model), task="detect")

    pairs = collect_labelled_images()
    print(f"Found {len(pairs)} labelled PoC images")
    print(f"Using YOLO: {args.model}")
    print(f"Using suppressor: {args.suppressor}")
    print(
        f"damage_threshold={args.damage_threshold} score_mode={args.score_mode} "
        f"min_score={args.min_anomaly_score} min_area={args.min_anomaly_area}"
    )

    results: list[EvalResult] = []
    for index, (image_path, truth) in enumerate(pairs, 1):
        image = read_image(image_path)
        pred = yolo.predict(source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        detections = detections_from_prediction(pred)
        apply_suppressor(
            image=image,
            detections=detections,
            model=suppressor,
            tf=tf,
            class_to_idx=class_to_idx,
            device=device,
            damage_threshold=args.damage_threshold,
            crop_padding=args.crop_padding,
        )
        anomaly_score, repair_score, neutral_score, deferred_score = score(
            detections,
            min_anomaly_area=args.min_anomaly_area,
            include_deferred_as_anomaly=args.deferred_as_anomaly,
            score_mode=args.score_mode,
        )
        verdict = decide(anomaly_score, repair_score, args.min_anomaly_score, args.repair_margin)
        result = EvalResult(
            image=image_path,
            truth=truth,
            verdict=verdict,
            anomaly_score=anomaly_score,
            repair_score=repair_score,
            neutral_score=neutral_score,
            deferred_score=deferred_score,
            detections=detections,
        )
        results.append(result)
        suppressed = sum(1 for det in detections if det.suppressed)
        mark = "OK" if truth == verdict else "MISS"
        print(
            f"[{index:03d}/{len(pairs):03d}] {mark:<4} truth={truth:<8} pred={verdict:<8} "
            f"a={anomaly_score:.2f} r={repair_score:.2f} dets={len(detections):02d} suppressed={suppressed:02d} "
            f"{image_path.name}"
        )

        if args.annotated_dir and (args.save_all or truth != verdict):
            annotated = draw_result(image, result)
            out_path = args.annotated_dir / f"{index:03d}_{truth}_to_{verdict}_{image_path.stem}.jpg"
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
