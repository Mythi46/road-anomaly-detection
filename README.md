# Road Anomaly Detection

Technical prototype for road-surface anomaly detection from inspection
vehicle imagery.

The current scope is frame-level detection of unrepaired road defects
such as potholes and cracks, with a conservative decision rule that
prioritizes avoiding false positives on repaired road surfaces.

## Repository Contents

```text
RoadAnomalyDetection/
  config.py              # Paths, class mapping, thresholds
  data_index.py          # Scans before/after labelled image folders
  detector.py            # YOLO wrapper + conservative decision rule
  predict.py             # Single-image / batch prediction CLI
  evaluate.py            # Precision / recall / FP evaluation
  sweep_threshold.py     # Threshold sweep for operating-point selection
  sweep_yolo26.py        # YOLO threshold sweep helper
  train_yolo26.py        # Training entry point
  requirements.txt

docs/
  feasibility_report.ja.md
  streaming_inference_considerations.ja.md
```

## Data and Weights

Private images, datasets, outputs, and model weights are intentionally
not tracked in git.

Expected local layout:

```text
data/
  poc/
    <scene>/
      before/   # abnormal images
      after/    # repaired / normal images
  data.yaml     # YOLO-format training YAML

models/
  best.pt       # inference weights
  base.pt       # optional training starting weights

runs/           # training outputs
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r RoadAnomalyDetection\requirements.txt
```

## Colab Pro+ Baseline Training

Open the training notebook directly in Colab:

[road_anomaly_colab_baseline.ipynb](https://colab.research.google.com/github/Mythi46/road-anomaly-detection/blob/main/notebooks/road_anomaly_colab_baseline.ipynb)

Recommended first run:

1. Runtime -> Change runtime type -> GPU.
2. Keep `RUN_MODE = "fast"` for the first baseline.
3. Run cells from top to bottom.
4. Training outputs and model weights are saved under
   `MyDrive/road-anomaly-detection/runs/`.

## Run Prediction

```powershell
python -m RoadAnomalyDetection.predict --image path\to\image.jpg
```

By default, the detector expects model weights at:

```text
models/best.pt
```

You can override this in code or by updating `RoadAnomalyDetection/config.py`.

## Evaluate a PoC Folder

The evaluator scans configured roots for this folder convention:

```text
before/ -> abnormal
after/  -> normal
```

Run:

```powershell
python -m RoadAnomalyDetection.evaluate --csv outputs\eval.csv
```

## Public Baseline Utilities

Build a combined public-data YAML:

```powershell
python scripts\write_combined_yolo_yaml.py `
  --converted-root data\public\converted `
  --output data\public\combined_fast.yaml
```

Build a hard-negative-biased manifest for a second training pass:

```powershell
python scripts\build_hard_negative_manifest.py `
  --input-yaml data\public\combined_fast.yaml `
  --output-dir data\public\hardneg_v2 `
  --repair-extra 8 `
  --neutral-extra 4 `
  --mixed-extra 1
```

Evaluate the 5-class public baseline on local PoC before/after folders:

```powershell
python scripts\evaluate_public_baseline_poc.py `
  --model RoadDetection_v8\runs\local_public_fast_yolo26s_e30\weights\best.pt `
  --csv RoadAnomalyDetection\outputs\public_baseline_poc_eval.csv `
  --annotated-dir RoadAnomalyDetection\outputs\public_baseline_poc_mistakes
```

Build and evaluate a second-stage crop suppressor:

```powershell
python scripts\build_suppressor_crops.py `
  --input-yaml data\public\combined_fast.yaml `
  --output-dir data\public\suppressor_crops_v1 `
  --clean

python scripts\train_suppressor_classifier.py `
  --data-dir data\public\suppressor_crops_v1 `
  --output-dir RoadAnomalyDetection\outputs\suppressor_v1

python scripts\evaluate_public_with_suppressor.py `
  --model RoadDetection_v8\runs\local_public_hardneg_v2_yolo26s_e20\weights\best.pt `
  --suppressor RoadAnomalyDetection\outputs\suppressor_v1\best.pt `
  --csv RoadAnomalyDetection\outputs\poc_with_suppressor.csv
```

Prepare a hard-negative review pack from false positives:

```powershell
python scripts\prepare_hard_negative_review_pack.py `
  --eval-csv RoadAnomalyDetection\outputs\poc_with_suppressor.csv `
  --output-dir annotation_workspace\hard_negative_review
```

## Technical Notes

The key design principle is precision-first operation:

- repaired road surfaces should not be flagged as abnormal;
- missing borderline defects is preferable to false positives on repairs;
- thresholds must remain externally configurable;
- additional hard-negative data is essential for improving reliability.

See `docs/` for the current feasibility report and streaming inference
performance notes.
