"""Nạp dữ liệu & tăng cường dữ liệu (data augmentation) cho bài toán phân loại.

Cấu trúc thư mục yêu cầu (định dạng ImageFolder):

    data/classification/
        train/{bicycle,bus,car,motorbike,truck}/*.jpg
        val/  {bicycle,bus,car,motorbike,truck}/*.jpg
        test/ {bicycle,bus,car,motorbike,truck}/*.jpg

Tăng cường dữ liệu là một dạng chính quy hoá: tạo thêm mẫu giả bằng các phép
biến đổi không làm đổi nhãn ([1, §7.4, tr. 240]). Lưu ý KHÔNG lật dọc
(xe không bao giờ "chổng ngược"), chỉ lật ngang.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),   # mô phỏng che khuất
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Lấy mẫu có trọng số nghịch đảo tần suất lớp — xử lý mất cân bằng dữ liệu
    (ví dụ xe máy chiếm đa số trong giao thông Việt Nam)."""
    counts = Counter(dataset.targets)
    weights = [1.0 / counts[t] for t in dataset.targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(data_dir: str | Path, img_size: int, batch_size: int = 32,
                      num_workers: int = 2, balance: bool = True):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", build_transforms(img_size, True))
    val_ds = datasets.ImageFolder(data_dir / "val", build_transforms(img_size, False))
    sampler = make_balanced_sampler(train_ds) if balance else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          shuffle=sampler is None, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_dl, val_dl, train_ds.classes
