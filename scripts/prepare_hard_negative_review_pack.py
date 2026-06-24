"""Prepare a review pack from PoC false-positive detections."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANOMALY_IDS = {0, 1}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not decode image: {path}")
    return img


def padded_bounds(xyxy: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
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


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise OSError(f"Could not encode {path}")
    buf.tofile(str(path))


def copy_annotated_if_available(annotated_dir: Path | None, output_dir: Path) -> int:
    if not annotated_dir or not annotated_dir.exists():
        return 0
    dst = output_dir / "annotated_false_positive_full"
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(annotated_dir.glob("*normal_to_abnormal*.jpg")):
        shutil.copy2(src, dst / src.name)
        count += 1
    return count


def make_contact_sheet(crop_paths: list[Path], output_path: Path) -> None:
    if not crop_paths:
        return
    thumb_w, thumb_h = 260, 200
    label_h = 44
    cols = 4 if len(crop_paths) >= 8 else 2
    rows = (len(crop_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, crop_path in enumerate(crop_paths):
        img = Image.open(crop_path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        sheet.paste(img, (x + (thumb_w - img.width) // 2, y))
        draw.text((x + 6, y + thumb_h + 4), crop_path.name[:42], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotated-dir", type=Path, default=None)
    parser.add_argument("--padding", type=float, default=0.25)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.eval_csv.open(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "false_positive_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    original_dir = args.output_dir / "original_false_positive_images"
    original_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    crop_paths: list[Path] = []
    copied_annotated = copy_annotated_if_available(args.annotated_dir, args.output_dir)

    for row in rows:
        if row["truth"] != "normal" or row["verdict"] != "abnormal":
            continue
        image_path = Path(row["image"])
        image = read_image(image_path)
        height, width = image.shape[:2]
        shutil.copy2(image_path, original_dir / image_path.name)
        detections = json.loads(row["detections_json"])
        for det_index, det in enumerate(detections):
            if int(det["cls_id"]) not in ANOMALY_IDS:
                continue
            x1, y1, x2, y2 = padded_bounds(det["xyxy"], width, height, args.padding)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            damage_prob = det.get("damage_prob")
            suppress_prob = det.get("suppress_prob")
            crop_name = (
                f"{image_path.stem}_det{det_index:02d}_"
                f"{det['cls_name']}_conf{float(det['confidence']):.2f}"
            )
            if damage_prob is not None:
                crop_name += f"_pd{float(damage_prob):.2f}"
            crop_name += ".jpg"
            crop_path = crop_dir / crop_name
            write_image(crop_path, crop)
            crop_paths.append(crop_path)
            manifest_rows.append(
                {
                    "source_image": str(image_path),
                    "crop": str(crop_path),
                    "cls_id": det["cls_id"],
                    "cls_name": det["cls_name"],
                    "confidence": det["confidence"],
                    "damage_prob": "" if damage_prob is None else damage_prob,
                    "suppress_prob": "" if suppress_prob is None else suppress_prob,
                    "xyxy": json.dumps(det["xyxy"]),
                    "area": det["area"],
                    "suggested_review_label": "repair_or_neutral_false_positive",
                }
            )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_image",
            "crop",
            "cls_id",
            "cls_name",
            "confidence",
            "damage_prob",
            "suppress_prob",
            "xyxy",
            "area",
            "suggested_review_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    contact_sheet_path = args.output_dir / "false_positive_crop_contact_sheet.jpg"
    make_contact_sheet(crop_paths, contact_sheet_path)

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Hard Negative Review Pack",
                "",
                "This pack contains PoC `after/normal` images that the current model still predicts as abnormal.",
                "",
                "- `original_false_positive_images/`: original normal images.",
                "- `annotated_false_positive_full/`: full-image annotations if provided.",
                "- `false_positive_crops/`: cropped anomaly detections to review as repair/neutral negatives.",
                "- `manifest.csv`: per-crop metadata.",
                "- `false_positive_crop_contact_sheet.jpg`: quick visual overview.",
                "",
                "Use this pack only for review or future hard-negative training. If it is used for training,",
                "keep a separate held-out repaired-road validation set.",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "false_positive_images": len({Path(r["source_image"]).name for r in manifest_rows}),
                "false_positive_crops": len(crop_paths),
                "copied_annotated_images": copied_annotated,
                "manifest": str(manifest_path),
                "contact_sheet": str(contact_sheet_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
