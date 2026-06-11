# coding: utf-8
"""Road anomaly detector built on top of an Ultralytics YOLO model.

Decision rule:

    strong(det) = (det.conf >= CONF_THRESHOLD) OR
                  (det.conf >= CONF_THRESHOLD_LOW
                   AND det.area >= MIN_BOX_AREA_LOW)

    anomaly_score = sum(conf for det in detections
                        if cls in ANOMALY_CLASSES and strong(det))
    repair_score  = sum(conf for det in detections
                        if cls in REPAIR_CLASSES and conf >= CONF_THRESHOLD)

    if anomaly_score >= MIN_DECISIVE_SCORE
       and anomaly_score >= repair_score - ANOMALY_MARGIN:
        verdict = "abnormal"
    else:
        verdict = "normal"   # default when nothing relevant is detected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    xyxy: tuple  # (x1, y1, x2, y2)


@dataclass
class AnomalyResult:
    image_path: Path
    verdict: str
    anomaly_score: float
    repair_score: float
    detections: List[Detection] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "image": str(self.image_path),
            "verdict": self.verdict,
            "anomaly_score": round(self.anomaly_score, 4),
            "repair_score": round(self.repair_score, 4),
            "num_detections": len(self.detections),
        }


class RoadAnomalyDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        from ultralytics import YOLO  # imported lazily so config-only use works

        self._model_path = Path(model_path) if model_path else config.MODEL_PATH
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self._model_path}")
        self._model = YOLO(str(self._model_path), task="detect")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _read_image(self, image_path: Path) -> np.ndarray:
        # cv2.imread chokes on non-ASCII paths on Windows; use imdecode.
        import cv2

        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not decode image: {image_path}")
        return img

    def _detect(self, image_path: Path) -> List[Detection]:
        img = self._read_image(image_path)
        # Query at the low threshold so decide() can apply the conf+area gate.
        results = self._model.predict(
            source=img,
            conf=config.CONF_THRESHOLD_LOW,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
        for cid, conf, box in zip(cls_ids, confs, xyxy):
            detections.append(
                Detection(
                    cls_id=int(cid),
                    cls_name=config.CLASS_NAMES.get(int(cid), str(int(cid))),
                    confidence=float(conf),
                    xyxy=tuple(box.tolist()),
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def _is_strong_anomaly(det: "Detection") -> bool:
        if det.confidence >= config.CONF_THRESHOLD:
            return True
        x1, y1, x2, y2 = det.xyxy
        area = (x2 - x1) * (y2 - y1)
        return (det.confidence >= config.CONF_THRESHOLD_LOW
                and area >= config.MIN_BOX_AREA_LOW)

    @staticmethod
    def decide(detections: List[Detection]) -> tuple:
        anomaly_score = sum(
            d.confidence for d in detections
            if d.cls_id in config.ANOMALY_CLASSES
            and RoadAnomalyDetector._is_strong_anomaly(d))
        repair_score = sum(
            d.confidence for d in detections
            if d.cls_id in config.REPAIR_CLASSES
            and d.confidence >= config.CONF_THRESHOLD)

        if (
            anomaly_score >= config.MIN_DECISIVE_SCORE
            and anomaly_score >= repair_score - config.ANOMALY_MARGIN
        ):
            verdict = config.LABEL_ABNORMAL
        else:
            verdict = config.LABEL_NORMAL
        return verdict, anomaly_score, repair_score

    def analyze(self, image_path: Path) -> AnomalyResult:
        image_path = Path(image_path)
        detections = self._detect(image_path)
        verdict, a_score, r_score = self.decide(detections)
        return AnomalyResult(
            image_path=image_path,
            verdict=verdict,
            anomaly_score=a_score,
            repair_score=r_score,
            detections=detections,
        )
