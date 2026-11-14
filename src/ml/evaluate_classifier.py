"""Đánh giá mô hình phân loại trên tập test.

    python -m ml.evaluate_classifier --ckpt checkpoints/resnet18_best.pt --data data/classification

Báo cáo: Accuracy, Precision / Recall / F1 từng lớp, macro-F1, ma trận nhầm lẫn.
Việc chọn độ đo phù hợp với mục tiêu ứng dụng: [1, §11.1, tr. 422].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import CLASSES_VI
from .data import build_transforms
from .models import build_model


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    kw = {} if ckpt["arch"] == "vehicle_cnn" else {"pretrained": False, "freeze_backbone": False}
    model = build_model(ckpt["arch"], num_classes=len(ckpt["classes"]), **kw)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_all(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/classification")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    ds = datasets.ImageFolder(Path(args.data) / args.split, build_transforms(ckpt["img_size"], False))
    y_true, y_pred = predict_all(model, DataLoader(ds, batch_size=64), device)

    names = [CLASSES_VI.get(c, c) for c in ckpt["classes"]]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print("Accuracy:", float(np.mean(y_true == y_pred)))

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues", values_format=".2f")
    plt.title(f"Ma trận nhầm lẫn — {ckpt['arch']}")
    out = Path(args.ckpt).with_suffix(".confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
