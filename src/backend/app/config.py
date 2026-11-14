"""Cấu hình backend, đọc từ biến môi trường (có giá trị mặc định để chạy demo ngay)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]   # .../src


@dataclass(frozen=True)
class Settings:
    # Trọng số YOLO: mặc định yolov8n.pt (COCO) — đã nhận biết sẵn car/bus/truck/motorcycle/bicycle.
    # Sau khi fine-tune, trỏ tới runs/detect/vehicles/weights/best.pt
    detector_weights: str = os.getenv("DETECTOR_WEIGHTS", "yolov8n.pt")
    # True nếu trọng số là mô hình COCO gốc (cần ánh xạ lớp COCO → lớp đề tài)
    detector_is_coco: bool = os.getenv("DETECTOR_IS_COCO", "1") == "1"
    classifier_ckpt: Path = Path(os.getenv("CLASSIFIER_CKPT", SRC_DIR / "checkpoints" / "resnet18_best.pt"))
    device: str = os.getenv("DEVICE", "cpu")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))


settings = Settings()
