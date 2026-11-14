"""Các kiến trúc CNN cho bài toán phân loại phương tiện.

1. VehicleCNN — CNN thiết kế và huấn luyện từ đầu (from scratch).
   Mỗi khối = [Conv 3×3 → BatchNorm → ReLU] × 2 → MaxPool 2×2,
   đúng cấu trúc "convolution stage → detector stage → pooling stage" của
   [1, §9.3, Hình 9.7, tr. 341].

   Bảng kích thước (đầu vào 3×128×128, K = 5 lớp):
   ┌─────────┬───────────────┬─────────────┬──────────┬────────────────┐
   │ Khối    │ Kênh          │ Đầu ra      │ Tham số  │ Receptive field│
   ├─────────┼───────────────┼─────────────┼──────────┼────────────────┤
   │ Block 1 │ 3 → 32 → 32   │ 32×64×64    │   10 208 │  6 px          │
   │ Block 2 │ 32 → 64 → 64  │ 64×32×32    │   55 552 │ 16 px          │
   │ Block 3 │ 64 → 128 →128 │ 128×16×16   │  221 696 │ 36 px          │
   │ Block 4 │ 128 → 256→256 │ 256×8×8     │  885 760 │ 76 px          │
   │ GAP+FC  │ 256 → 5       │ 5           │    1 285 │ toàn ảnh       │
   └─────────┴───────────────┴─────────────┴──────────┴────────────────┘
   Tổng: 1 174 501 tham số (≈ 1,17 M).
   So sánh: chỉ riêng 1 lớp fully-connected 49 152 → 512 đã cần 25,2 M tham số.

2. build_transfer_model — học chuyển giao (transfer learning, [1, §15.2, tr. 536])
   từ ResNet-18 / MobileNetV2 / EfficientNet-B0 huấn luyện trước trên ImageNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASSES


class ConvBNReLU(nn.Sequential):
    """Conv(k×k) → BatchNorm → ReLU.

    bias=False vì BatchNorm đã có tham số dịch β ([1, §8.7.1, tr. 317]).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class ConvBlock(nn.Sequential):
    """Hai lớp tích chập 3×3 xếp chồng + MaxPool 2×2.

    Hai lớp 3×3 có receptive field tương đương một lớp 5×5 nhưng ít tham số hơn
    (2·9·C² so với 25·C²) và thêm một phi tuyến.
    """

    def __init__(self, c_in: int, c_out: int, p_drop: float = 0.0):
        layers: list[nn.Module] = [
            ConvBNReLU(c_in, c_out),
            ConvBNReLU(c_out, c_out),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        super().__init__(*layers)


class VehicleCNN(nn.Module):
    """CNN 8 lớp tích chập cho phân loại 5 loại phương tiện."""

    def __init__(self, num_classes: int = len(CLASSES), widths=(32, 64, 128, 256),
                 p_drop: float = 0.5):
        super().__init__()
        blocks, c_in = [], 3
        for i, c_out in enumerate(widths):
            blocks.append(ConvBlock(c_in, c_out, p_drop=0.1 if i >= 2 else 0.0))
            c_in = c_out
        self.features = nn.Sequential(*blocks)          # bộ trích đặc trưng phân cấp
        self.pool = nn.AdaptiveAvgPool2d(1)              # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),                          # [1, §7.12, tr. 258]
            nn.Linear(c_in, num_classes),                # logits; softmax nằm trong loss
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Trả về bản đồ đặc trưng sau từng khối — dùng để trực quan hoá
        đặc trưng phân cấp (cạnh → kết cấu → bộ phận → đối tượng)."""
        maps = []
        for block in self.features:
            x = block(x)
            maps.append(x)
        return maps


def build_transfer_model(arch: str = "resnet18", num_classes: int = len(CLASSES),
                         pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Tạo mô hình học chuyển giao: giữ backbone ImageNet, thay lớp phân loại cuối.

    Giai đoạn 1: freeze_backbone=True  → chỉ huấn luyện head (hội tụ nhanh).
    Giai đoạn 2: unfreeze_backbone()   → fine-tune toàn bộ với learning rate nhỏ.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        head_name, in_f = "fc", model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        head_name, in_f = "classifier", model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        head_name, in_f = "classifier", model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {arch}")

    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head_name)
    model.head_name = head_name  # type: ignore[attr-defined]
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def build_model(arch: str, num_classes: int = len(CLASSES), **kw) -> nn.Module:
    if arch == "vehicle_cnn":
        return VehicleCNN(num_classes=num_classes)
    return build_transfer_model(arch, num_classes=num_classes, **kw)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    net = VehicleCNN()
    print(net)
    print("Số tham số VehicleCNN:", f"{count_parameters(net):,}")   # 1,174,501
    out = net(torch.randn(2, 3, 128, 128))
    print("Đầu ra:", out.shape)
