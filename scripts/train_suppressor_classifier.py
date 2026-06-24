"""Train a lightweight crop-level damage/suppress classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def make_loaders(data_dir: Path, img_size: int, batch_size: int, workers: int) -> tuple:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_ds, val_ds, train_loader, val_loader


def confusion_metrics(cm: torch.Tensor) -> dict:
    # Rows are true labels, columns are predictions.
    total = int(cm.sum().item())
    correct = int(cm.diag().sum().item())
    metrics = {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": cm.int().tolist(),
        "classes": {},
    }
    for cls_id, name in enumerate(["damage", "suppress"]):
        tp = int(cm[cls_id, cls_id].item())
        fn = int(cm[cls_id, :].sum().item() - tp)
        fp = int(cm[:, cls_id].sum().item() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["classes"][name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    cm = torch.zeros((2, 2), dtype=torch.long)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        preds = logits.argmax(dim=1)
        for truth, pred in zip(labels.cpu(), preds.cpu()):
            cm[int(truth), int(pred)] += 1
    metrics = confusion_metrics(cm)
    metrics["loss"] = total_loss / max(metrics["total"], 1)
    return metrics


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    started = time.time()

    print(f"device={device}")
    print(f"train={len(train_ds)} val={len(val_ds)} class_to_idx={train_ds.class_to_idx}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * labels.numel()
            seen += labels.numel()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device=device, amp=args.amp)
        suppress_recall = val_metrics["classes"]["suppress"]["recall"]
        damage_recall = val_metrics["classes"]["damage"]["recall"]
        score = 0.65 * suppress_recall + 0.35 * damage_recall
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "val": val_metrics,
            "score": score,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"damage_recall={damage_recall:.4f} "
            f"suppress_recall={suppress_recall:.4f} "
            f"score={score:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "img_size": args.img_size,
            "model": "mobilenet_v3_small",
            "epoch": epoch,
            "metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, best_path)

    summary = {
        "best_score": best_score,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "history": history,
        "class_to_idx": train_ds.class_to_idx,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    summary = train(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
