"""Huấn luyện CNN phân loại phương tiện.

Ví dụ:
    # CNN tự thiết kế, ảnh 128×128
    python -m ml.train_classifier --arch vehicle_cnn --data data/classification --epochs 40

    # Học chuyển giao ResNet-18: 5 epoch chỉ train head, sau đó fine-tune toàn bộ
    python -m ml.train_classifier --arch resnet18 --epochs 20 --warmup-epochs 5

Thành phần huấn luyện và cơ sở lý thuyết tương ứng trong [1]:
    * Hàm mất mát cross-entropy = cực đại hợp lý (MLE)     — §5.5 tr. 131, §6.2.1.1 tr. 178
    * Mini-batch SGD / Adam(W)                              — §8.1.3 tr. 277, §8.5.3 tr. 308
    * Weight decay (chính quy hoá L2)                       — §7.1.1 tr. 231
    * Dropout, BatchNorm, data augmentation                 — §7.12, §8.7.1, §7.4
    * Early stopping theo độ chính xác tập validation       — §7.8 tr. 246
    * Lan truyền ngược (back-propagation) do autograd thực hiện — §6.5 tr. 204
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import CHECKPOINT_DIR, IMG_SIZE_SCRATCH, IMG_SIZE_TRANSFER
from .data import build_dataloaders
from .models import build_model, count_parameters, unfreeze_backbone


def parse_args():
    p = argparse.ArgumentParser(description="Train vehicle classifier")
    p.add_argument("--arch", default="vehicle_cnn",
                   choices=["vehicle_cnn", "resnet18", "mobilenet_v2", "efficientnet_b0"])
    p.add_argument("--data", default="data/classification")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Số epoch chỉ huấn luyện head (transfer learning)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8, help="Early stopping")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Một epoch huấn luyện (optimizer != None) hoặc đánh giá."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)                      # lan truyền tiến
            loss = criterion(logits, y)            # J(θ) = −log p(y | x; θ)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()      # lan truyền ngược: ∇θ J
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()                   # θ ← θ − ε · ĝ
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = IMG_SIZE_SCRATCH if args.arch == "vehicle_cnn" else IMG_SIZE_TRANSFER

    train_dl, val_dl, classes = build_dataloaders(args.data, img_size, args.batch_size, args.workers)
    model = build_model(args.arch, num_classes=len(classes),
                        **({} if args.arch == "vehicle_cnn" else {"freeze_backbone": args.warmup_epochs > 0}))
    model.to(device)
    print(f"[{args.arch}] tham số: {count_parameters(model):,} | "
          f"huấn luyện được: {count_parameters(model, True):,} | thiết bị: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.arch}_best.pt"
    history, best_acc, bad_epochs = [], 0.0, 0

    for epoch in range(1, args.epochs + 1):
        if args.arch != "vehicle_cnn" and epoch == args.warmup_epochs + 1 and args.warmup_epochs > 0:
            # Giai đoạn 2: mở băng toàn bộ backbone, learning rate nhỏ hơn 10 lần
            unfreeze_backbone(model)
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
            print(">> Fine-tune toàn bộ mạng")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=optimizer.param_groups[0]["lr"]))
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | {time.time() - t0:.0f}s")

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save({"arch": args.arch, "classes": classes, "img_size": img_size,
                        "state_dict": model.state_dict(), "val_acc": va_acc}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:          # early stopping
                print(f"Dừng sớm tại epoch {epoch}")
                break

    (out_dir / f"{args.arch}_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val acc = {best_acc:.4f} → {ckpt_path}")


if __name__ == "__main__":
    main()
