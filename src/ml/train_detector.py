"""Huấn luyện / fine-tune bộ phát hiện phương tiện YOLOv8 (một giai đoạn, anchor-free).

    python -m ml.train_detector --data configs/vehicles.yaml --model yolov8n.pt --epochs 50
    python -m ml.train_detector --data configs/vehicles.yaml --model yolov8s.pt --imgsz 640 --batch 16

Kiến trúc YOLOv8 = CNN Backbone (CSPDarknet, trích đặc trưng phân cấp)
                  → Neck (FPN + PAN, trộn đặc trưng đa tỉ lệ P3/P4/P5)
                  → Head tách rời (decoupled): nhánh hồi quy hộp + nhánh phân loại.
Hàm mất mát:  L = λ_box · L_CIoU + λ_dfl · L_DFL + λ_cls · L_BCE
Hậu xử lý:     ngưỡng confidence → NMS theo lớp (xem ml/detection_utils.py).

Dữ liệu ở định dạng YOLO: mỗi ảnh có 1 file .txt, mỗi dòng
    <class_id> <x_center> <y_center> <width> <height>   (chuẩn hoá về [0, 1])
"""
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/vehicles.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt (pretrained COCO)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="0 | cpu")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="vehicles")
    return p.parse_args()


def main():
    from ultralytics import YOLO   # import muộn để các module khác không phụ thuộc

    args = parse_args()
    model = YOLO(args.model)        # khởi tạo từ trọng số COCO → học chuyển giao
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        cos_lr=True,
        patience=15,                # early stopping
        # Tăng cường dữ liệu cho cảnh giao thông
        mosaic=1.0, mixup=0.1, fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=0.0, scale=0.5,
    )
    metrics = model.val()           # mAP50, mAP50-95, precision, recall
    print("mAP@0.5     :", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)
    model.export(format="onnx")     # phục vụ triển khai nhẹ


if __name__ == "__main__":
    main()
