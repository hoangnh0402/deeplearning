"""Grad-CAM: giải thích CNN "nhìn" vào đâu khi phân loại phương tiện.

    python -m ml.gradcam --ckpt checkpoints/vehicle_cnn_best.pt --image samples/bus.jpg

L^c = ReLU( Σ_k α_k^c · A^k ),  α_k^c = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}
(A^k: bản đồ đặc trưng thứ k của lớp tích chập cuối; y^c: logit lớp c).
Gradient được tính bằng lan truyền ngược ([1, §6.5]).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .evaluate_classifier import load_checkpoint


def last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


def grad_cam(model, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
    acts, grads = {}, {}
    layer = last_conv_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("g", go[0]))
    logits = model(x)
    target = int(logits.argmax(1)) if target is None else target
    model.zero_grad()
    logits[0, target].backward()
    h1.remove(); h2.remove()
    alpha = grads["g"].mean(dim=(2, 3), keepdim=True)            # trọng số kênh
    cam = F.relu((alpha * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    device = torch.device("cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    img = Image.open(args.image).convert("RGB")
    x = build_transforms(ckpt["img_size"], False)(img).unsqueeze(0)
    cam, cls = grad_cam(model, x)
    shown = np.asarray(img.resize((ckpt["img_size"], ckpt["img_size"])))
    plt.imshow(shown); plt.imshow(cam, cmap="jet", alpha=0.45); plt.axis("off")
    plt.title(f"Grad-CAM: {ckpt['classes'][cls]}")
    plt.savefig("gradcam.png", dpi=150, bbox_inches="tight")
    print("Đã lưu gradcam.png")


if __name__ == "__main__":
    main()
