# coding: utf-8
"""Sweep confidence and decision thresholds on a labelled PoC set.

For each (conf, min_decisive) pair we report TP/FP/TN/FN, Precision,
Recall, F1. We also dump the best 0-FP operating point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from . import config
from .data_index import collect_labelled_images
from .detector import Detection, RoadAnomalyDetector


DEFAULT_WEIGHTS = config.MODEL_PATH

# Optional path substring used to identify a held-out scene.
HOLDOUT_FOLDER_HINT = "holdout"

CONF_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
MIN_GRID = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]


def _decide(dets: List[Detection], min_decisive: float) -> str:
    a = sum(d.confidence for d in dets if d.cls_id in config.ANOMALY_CLASSES)
    r = sum(d.confidence for d in dets if d.cls_id in config.REPAIR_CLASSES)
    if a >= min_decisive and a >= r - config.ANOMALY_MARGIN:
        return config.LABEL_ABNORMAL
    return config.LABEL_NORMAL


def _all_detections_per_conf(detector: RoadAnomalyDetector, pairs, conf: float):
    """Run YOLO once per (image, conf) and return raw detections list."""
    out = []
    for img_path, truth in pairs:
        img = detector._read_image(img_path)
        results = detector._model.predict(
            source=img, conf=conf, iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE, verbose=False,
        )
        dets: List[Detection] = []
        if results and results[0].boxes is not None and len(results[0].boxes):
            r = results[0].boxes
            cls_ids = r.cls.cpu().numpy().astype(int)
            confs = r.conf.cpu().numpy().astype(float)
            for cid, c in zip(cls_ids, confs):
                dets.append(Detection(
                    cls_id=int(cid),
                    cls_name=config.CLASS_NAMES.get(int(cid), str(cid)),
                    confidence=float(c), xyxy=(0, 0, 0, 0),
                ))
        out.append((truth, dets))
    return out


def _metrics(predictions):
    tp = fn = fp = tn = 0
    for truth, verdict in predictions:
        if truth == "abnormal" and verdict == "abnormal":
            tp += 1
        elif truth == "abnormal":
            fn += 1
        elif verdict == "abnormal":
            fp += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, tn, fn, p, r, f1


def _filter_subset(pairs, subset: str):
    """Slice the PoC pair list per the requested view.

    * ``full``    - all labelled pairs
    * ``holdout`` - only images whose path includes ``HOLDOUT_FOLDER_HINT``
    * ``leaked``  - all pairs NOT in holdout (the images v2 saw in training)
    """
    if subset == "full":
        return pairs
    in_hold = [pl for pl in pairs if HOLDOUT_FOLDER_HINT in str(pl[0])]
    if subset == "holdout":
        return in_hold
    if subset == "leaked":
        hold_set = {pl[0].resolve() for pl in in_hold}
        return [pl for pl in pairs if pl[0].resolve() not in hold_set]
    raise ValueError(f"unknown subset: {subset!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold sweep for road anomaly detection.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Path to the .pt weights to evaluate")
    parser.add_argument("--subset", choices=["full", "holdout", "leaked"],
                        default="full",
                        help="full = all labelled images; holdout = paths containing "
                             "the holdout folder hint; leaked = the rest")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Missing weights at {weights}")
        return 1
    pairs = collect_labelled_images()
    pairs = _filter_subset(pairs, args.subset)
    n_abn = sum(1 for _, t in pairs if t == config.LABEL_ABNORMAL)
    n_nrm = sum(1 for _, t in pairs if t == config.LABEL_NORMAL)
    print(f"Subset '{args.subset}': {len(pairs)} images "
          f"({n_abn} abnormal, {n_nrm} normal). Loading YOLO model from {weights} ...")
    detector = RoadAnomalyDetector(model_path=weights)

    # Pre-run YOLO at the lowest conf in the grid; higher confs are a
    # subset, so we can re-filter on the Python side without re-inferring.
    base_conf = min(CONF_GRID)
    print(f"Running inference once at conf={base_conf} (will filter higher confs in memory)...")
    raw = _all_detections_per_conf(detector, pairs, base_conf)

    print()
    print(f"{'conf':>6} {'min':>6} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 60)

    zero_fp_best = None
    for conf in CONF_GRID:
        for min_dec in MIN_GRID:
            preds = []
            for truth, dets in raw:
                filt = [d for d in dets if d.confidence >= conf]
                preds.append((truth, _decide(filt, min_dec)))
            tp, fp, tn, fn, p, r, f1 = _metrics(preds)
            print(f"{conf:>6.2f} {min_dec:>6.2f} {tp:>3} {fp:>3} {tn:>3} {fn:>3} "
                  f"{p:>6.3f} {r:>6.3f} {f1:>6.3f}")
            if fp == 0 and (zero_fp_best is None or r > zero_fp_best[-2]):
                zero_fp_best = (conf, min_dec, tp, fp, tn, fn, p, r, f1)

    print()
    if zero_fp_best:
        c, m, tp, fp, tn, fn, p, r, f1 = zero_fp_best
        print(f"Best 0-FP point: conf={c}, min={m} -> "
              f"TP={tp} FN={fn} TN={tn} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
    else:
        print("No 0-FP operating point found in the grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
