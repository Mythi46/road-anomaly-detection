# coding: utf-8
"""Configuration for the Road Anomaly Detection project.

The detector uses an 8-class road-surface schema. We bucket those
classes into three groups so detections can be turned into a binary
frame-level verdict:

    - ANOMALY_CLASSES  : the road is damaged       -> verdict "abnormal"
    - REPAIR_CLASSES   : the damage has been fixed -> verdict "normal"
    - NEUTRAL_CLASSES  : ignored by the decision rule
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

MODEL_PATH = REPO_ROOT / "models" / "best.pt"

# Any sub-tree containing a ``before`` and/or ``after`` directory will be
# picked up by the data indexer. Keep private image data outside git.
POC_ROOTS = [
    REPO_ROOT / "data" / "poc",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Manhole",
    1: "Patch-Net",
    2: "Patch-Crack",
    3: "Pothole",
    4: "Crack",
    5: "Patch-Pothole",
    6: "Net",
    7: "Other",
}

ANOMALY_CLASSES = {3, 4, 6}        # Pothole, Crack, Net
REPAIR_CLASSES = {1, 2, 5}         # Patch-Net, Patch-Crack, Patch-Pothole
NEUTRAL_CLASSES = {0, 7}           # Manhole, Other

# ---------------------------------------------------------------------------
# Detector & decision thresholds
# ---------------------------------------------------------------------------
# YOLO inference
CONF_THRESHOLD_LOW = 0.45      # low-confidence pass with area gate
CONF_THRESHOLD = 0.60          # high-confidence pass-through
MIN_BOX_AREA_LOW = 2000        # px^2 minimum at CONF_THRESHOLD_LOW
IOU_THRESHOLD = 0.7
IMG_SIZE = 640

# Decision rule
# ``MIN_DECISIVE_SCORE`` is the smallest bucket score (sum of confidences)
# required to flip the default verdict away from "normal".
MIN_DECISIVE_SCORE = 0.30
# ``ANOMALY_MARGIN`` makes the rule asymmetric: a tiny patch detection on
# an otherwise pothole-heavy image should not flip the verdict to normal.
ANOMALY_MARGIN = 0.10

# File extensions accepted by the data indexer
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Verdict labels used everywhere
LABEL_ABNORMAL = "abnormal"
LABEL_NORMAL = "normal"
