"""Tầng dịch vụ suy luận: nạp mô hình một lần (lazy, singleton) và xử lý ảnh.

Luồng phát hiện:  ảnh → YOLOv8 (CNN backbone+neck+head) → lọc confidence → NMS
                  → (tuỳ chọn) cắt từng hộp và phân loại lại bằng CNN phân loại
                  → đếm số lượng theo loại xe.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import Counter

import numpy as np
from PIL import Image

from .config import SRC_DIR, settings

if str(SRC_DIR) not in sys.path:          # cho phép import gói ml/
    sys.path.insert(0, str(SRC_DIR))

from ml.config import CLASSES, CLASSES_VI, COCO_TO_VEHICLE  # noqa: E402


class VehicleDetector:
    _lock = threading.Lock()

    def __init__(self):
        self.model = None

    def load(self):
        with self._lock:
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO(settings.detector_weights)
        return self.model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, img: Image.Image, conf: float = 0.25, iou: float = 0.5) -> tuple[list[dict], float]:
        model = self.load()
        t0 = time.perf_counter()
        result = model.predict(np.asarray(img), conf=conf, iou=iou, device=settings.device,
                               verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        for xyxy, c, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy().astype(int)):
            if settings.detector_is_coco:
                label = COCO_TO_VEHICLE.get(int(cls))
                if label is None:           # bỏ người, đèn giao thông, ...
                    continue
            else:
                label = model.names[int(cls)]
            dets.append({"label": label, "confidence": float(c), "box": xyxy.tolist()})
        return dets, ms


class VehicleClassifier:
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.ckpt = None
        self.tf = None

    def load(self):
        with self._lock:
            if self.model is None:
                if not settings.classifier_ckpt.exists():
                    raise FileNotFoundError(
                        f"Chưa có checkpoint phân loại: {settings.classifier_ckpt}. "
                        "Hãy chạy `python -m ml.train_classifier` trước.")
                import torch
                from ml.data import build_transforms
                from ml.evaluate_classifier import load_checkpoint
                self.model, self.ckpt = load_checkpoint(settings.classifier_ckpt, torch.device(settings.device))
                self.tf = build_transforms(self.ckpt["img_size"], train=False)
        return self.model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def available(self) -> bool:
        return self.loaded or settings.classifier_ckpt.exists()

    def predict(self, img: Image.Image) -> tuple[list[tuple[str, float]], float]:
        import torch
        model = self.load()
        t0 = time.perf_counter()
        with torch.no_grad():
            x = self.tf(img).unsqueeze(0).to(settings.device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        ms = (time.perf_counter() - t0) * 1000
        classes = self.ckpt["classes"]
        ranked = sorted(zip(classes, probs.tolist()), key=lambda t: -t[1])
        return ranked, ms


detector = VehicleDetector()
classifier = VehicleClassifier()


def detect_and_count(img: Image.Image, conf: float, iou: float, refine: bool) -> dict:
    dets, ms = detector.predict(img, conf, iou)
    out = []
    for d in dets:
        label, score, refined = d["label"], d["confidence"], False
        if refine and classifier.available:
            x1, y1, x2, y2 = map(int, d["box"])
            if (x2 - x1) >= 24 and (y2 - y1) >= 24:          # bỏ hộp quá nhỏ
                ranked, cms = classifier.predict(img.crop((x1, y1, x2, y2)))
                ms += cms
                if ranked[0][1] > score:                    # tin bộ phân loại khi nó tự tin hơn
                    label, score, refined = ranked[0][0], ranked[0][1], True
        out.append({
            "label": label, "label_vi": CLASSES_VI.get(label, label), "confidence": score,
            "box": dict(zip(("x1", "y1", "x2", "y2"), d["box"])), "refined_by_classifier": refined,
        })
    counts = Counter(o["label"] for o in out)
    return {
        "width": img.width, "height": img.height, "detections": out,
        "counts": {c: counts.get(c, 0) for c in CLASSES}, "inference_ms": round(ms, 1),
        "model": str(settings.detector_weights),
    }
