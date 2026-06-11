# coding: utf-8
"""Threshold sweep on the PoC set + dump false-positive images for inspection.

Runs the YOLO model once with a very permissive ``conf=0.05`` to capture
every weak detection, then evaluates the binary decision rule across a
grid of ``(CONF_THRESHOLD, MIN_DECISIVE_SCORE)`` values from cached
detections. Highlights the operating point that hits zero false
positives on the ``normal`` (repaired) subset.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config
from .data_index import collect_labelled_images


CAPTURE_CONF = 0.05  # very permissive YOLO predict threshold
CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
MIN_GRID = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50, 2.00]


def _read_image(image_path: Path) -> np.ndarray:
    import cv2
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not decode image: {image_path}")
    return img


def _capture_detections(pairs: List[Tuple[Path, str]]):
    """Returns list of (path, truth, [(cls_id, conf), ...])."""
    from ultralytics import YOLO
    model = YOLO(str(config.MODEL_PATH), task="detect")
    out = []
    for i, (path, truth) in enumerate(pairs, 1):
        img = _read_image(path)
        res = model.predict(source=img, conf=CAPTURE_CONF, iou=config.IOU_THRESHOLD,
                            imgsz=config.IMG_SIZE, verbose=False)
        dets: List[Tuple[int, float]] = []
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)
            confs = res[0].boxes.conf.cpu().numpy().astype(float)
            dets = [(int(c), float(p)) for c, p in zip(cls_ids, confs)]
        out.append((path, truth, dets))
        if i % 10 == 0 or i == len(pairs):
            print(f"  captured {i}/{len(pairs)}")
    return out


def _verdict(dets: List[Tuple[int, float]], conf_th: float, min_dec: float) -> Tuple[str, float, float]:
    anom = sum(p for c, p in dets if p >= conf_th and c in config.ANOMALY_CLASSES)
    rep = sum(p for c, p in dets if p >= conf_th and c in config.REPAIR_CLASSES)
    if anom >= min_dec and anom >= rep - config.ANOMALY_MARGIN:
        return config.LABEL_ABNORMAL, anom, rep
    return config.LABEL_NORMAL, anom, rep


def _eval(cache, conf_th: float, min_dec: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for _, truth, dets in cache:
        v, _, _ = _verdict(dets, conf_th, min_dec)
        if truth == config.LABEL_ABNORMAL and v == config.LABEL_ABNORMAL:
            tp += 1
        elif truth == config.LABEL_ABNORMAL and v == config.LABEL_NORMAL:
            fn += 1
        elif truth == config.LABEL_NORMAL and v == config.LABEL_ABNORMAL:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def _dump_fps(cache, conf_th: float, min_dec: float, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dumped: List[Path] = []
    for path, truth, dets in cache:
        if truth != config.LABEL_NORMAL:
            continue
        v, anom, rep = _verdict(dets, conf_th, min_dec)
        if v != config.LABEL_ABNORMAL:
            continue
        tag = f"FP_anom{anom:.2f}_rep{rep:.2f}__{path.stem}{path.suffix}"
        dst = out_dir / tag
        shutil.copyfile(path, dst)
        dumped.append(dst)
    return dumped


def main() -> int:
    pairs = collect_labelled_images()
    if not pairs:
        print("No images found.")
        return 1
    print(f"Loaded {len(pairs)} images. Running YOLO once at conf={CAPTURE_CONF}...")
    cache = _capture_detections(pairs)

    print("\nGrid: rows = MIN_DECISIVE_SCORE, cols = CONF_THRESHOLD")
    print("Cell = (Precision / Recall / FP-on-normal-count)\n")
    header = "MIN \\ CONF |" + "".join(f"  {c:>4.2f}        " for c in CONF_GRID)
    print(header)
    print("-" * len(header))
    for m in MIN_GRID:
        row = f"   {m:>4.2f}    |"
        for c in CONF_GRID:
            r = _eval(cache, c, m)
            row += f"  {r['precision']:.2f}/{r['recall']:.2f}/{r['fp']:>2d}  "
        print(row)

    print("\nDumping false-positive images at current default "
          f"(CONF_THRESHOLD={config.CONF_THRESHOLD}, MIN_DECISIVE_SCORE={config.MIN_DECISIVE_SCORE})...")
    fp_dir = config.OUTPUT_DIR / "false_positives"
    dumped = _dump_fps(cache, config.CONF_THRESHOLD, config.MIN_DECISIVE_SCORE, fp_dir)
    print(f"  Wrote {len(dumped)} FP images to {fp_dir}")
    for d in dumped:
        print(f"    {d.name}")

    print("\nBest operating points:")
    best_zero_fp = None
    for m in [v for v in MIN_GRID] + [0.80, 0.90, 1.10, 1.16, 1.20]:
        for c in CONF_GRID:
            r = _eval(cache, c, m)
            if r["fp"] == 0 and (best_zero_fp is None or r["recall"] > best_zero_fp[2]["recall"]):
                best_zero_fp = (c, m, r)
    if best_zero_fp:
        c, m, r = best_zero_fp
        print(f"  Highest recall at 0 FP: CONF={c:.2f} MIN={m:.2f}  "
              f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
