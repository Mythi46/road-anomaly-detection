# coding: utf-8
"""Fine-tune an Ultralytics YOLO detector for road anomaly detection.

Strategy:
    * Start from pretrained detector weights.
    * Train at 640x640 on a YOLO-format dataset YAML.
    * Save runs under ``runs/``.

Hardware assumption: GPU for training. For CPU-only environments, use a
small batch and expect longer training time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config


REPO_ROOT = config.REPO_ROOT
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "data.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "base.pt"
PROJECT_DIR = REPO_ROOT / "runs"
DEFAULT_RUN_NAME = "train_road_anomaly"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a road anomaly detector.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML),
                        help="Path to the YOLO-format data YAML")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Starting weights")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME,
                        help="Sub-folder name under runs/ for this training run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 lets Ultralytics auto-batch on the GPU")
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30,
                        help="EarlyStopping patience (epochs without val improvement)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--multi-scale", action="store_true",
                        help="vary imgsz by +/-50%% per batch so the model "
                             "learns scale robustness (helps small/distant "
                             "potholes at higher inference resolutions)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute() and not data_yaml.exists():
        data_yaml = REPO_ROOT / "data" / args.data
    weights = Path(args.weights)
    if not data_yaml.exists():
        print(f"Missing data YAML at {data_yaml}.")
        return 1
    if not weights.exists():
        print(f"Missing starting weights at {weights}.")
        return 1

    from ultralytics import YOLO

    print(f"Loading YOLO model from {weights} ...")
    model = YOLO(str(weights))

    print(f"Training on {data_yaml} for {args.epochs} epochs, imgsz={args.imgsz}, "
          f"batch={args.batch}, multi_scale={args.multi_scale}, "
          f"device={args.device}, run='{args.name}'")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        multi_scale=args.multi_scale,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"\nTraining done. Best weights: {results.save_dir}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
