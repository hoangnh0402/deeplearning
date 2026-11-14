"""Cấu hình dùng chung cho toàn bộ pipeline phân loại / phát hiện phương tiện.

Tham chiếu lý thuyết: Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]          # .../src
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Thứ tự theo bảng chữ cái — trùng với thứ tự torchvision.datasets.ImageFolder gán nhãn.
CLASSES = ["bicycle", "bus", "car", "motorbike", "truck"]
CLASSES_VI = {
    "bicycle": "Xe đạp",
    "bus": "Xe buýt",
    "car": "Ô tô con",
    "motorbike": "Xe máy",
    "truck": "Xe tải",
}

# Ánh xạ từ chỉ số lớp COCO (YOLO pretrained) sang lớp phương tiện của đề tài.
COCO_TO_VEHICLE = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# Tiền xử lý: chuẩn hoá theo thống kê ImageNet (xem [1, §12.2.1, tr. 453]).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE_SCRATCH = 128   # VehicleCNN huấn luyện từ đầu
IMG_SIZE_TRANSFER = 224  # ResNet / MobileNet pretrained
IMG_SIZE_DETECT = 640    # YOLOv8
