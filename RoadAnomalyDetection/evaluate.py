# coding: utf-8
"""Evaluate the baseline detector on the PoC ``before`` / ``after`` folders."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

from . import config
from .detector import AnomalyResult, RoadAnomalyDetector
from .data_index import collect_labelled_images


def _confusion(results: List[Tuple[str, AnomalyResult]]) -> Dict[Tuple[str, str], int]:
    cm: Dict[Tuple[str, str], int] = {}
    for truth, res in results:
        key = (truth, res.verdict)
        cm[key] = cm.get(key, 0) + 1
    return cm


def _metrics(cm: Dict[Tuple[str, str], int]) -> Dict[str, float]:
    pos = config.LABEL_ABNORMAL
    neg = config.LABEL_NORMAL
    tp = cm.get((pos, pos), 0)
    fn = cm.get((pos, neg), 0)
    fp = cm.get((neg, pos), 0)
    tn = cm.get((neg, neg), 0)
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def _print_report(cm: Dict[Tuple[str, str], int], metrics: Dict[str, float]) -> None:
    pos = config.LABEL_ABNORMAL
    neg = config.LABEL_NORMAL
    print()
    print("Confusion matrix (rows = ground truth, columns = prediction):")
    print(f"                pred={pos:<10} pred={neg:<10}")
    print(f"  truth={pos:<8} {cm.get((pos, pos), 0):<15} {cm.get((pos, neg), 0):<15}")
    print(f"  truth={neg:<8} {cm.get((neg, pos), 0):<15} {cm.get((neg, neg), 0):<15}")
    print()
    print(f"Total images : {metrics['total']}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}  (positive = {pos})")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"F1           : {metrics['f1']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the baseline road anomaly detector.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to dump per-image predictions as CSV.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only evaluate the first N images (debug).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Override path to YOLO weights (defaults to config.MODEL_PATH).")
    args = parser.parse_args()

    pairs = collect_labelled_images()
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("No labelled images found under the configured PoC roots.")
        return 1

    print(f"Found {len(pairs)} labelled images. Loading model...")
    detector = RoadAnomalyDetector(model_path=args.model)
    print(f"Using weights: {detector._model_path}")

    results: List[Tuple[str, AnomalyResult]] = []
    t0 = time.time()
    for i, (img_path, truth) in enumerate(pairs, 1):
        try:
            res = detector.analyze(img_path)
        except Exception as exc:  # pragma: no cover - robustness
            print(f"[{i}/{len(pairs)}] SKIP {img_path}: {exc}")
            continue
        results.append((truth, res))
        mark = "OK " if res.verdict == truth else "MISS"
        print(f"[{i}/{len(pairs)}] {mark} truth={truth:<8} pred={res.verdict:<8} "
              f"anom={res.anomaly_score:.2f} rep={res.repair_score:.2f}  {img_path.name}")
    elapsed = time.time() - t0

    cm = _confusion(results)
    metrics = _metrics(cm)
    _print_report(cm, metrics)
    print(f"\nElapsed: {elapsed:.1f}s  ({elapsed / max(len(results), 1):.2f}s/image)")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "truth", "verdict", "anomaly_score", "repair_score", "num_detections"])
            for truth, res in results:
                writer.writerow([str(res.image_path), truth, res.verdict,
                                 f"{res.anomaly_score:.4f}", f"{res.repair_score:.4f}",
                                 len(res.detections)])
        print(f"Wrote per-image CSV to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
