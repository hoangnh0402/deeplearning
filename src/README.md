# CNN trong Phát hiện và Phân loại Phương tiện — Mã nguồn

Tài liệu tham khảo chính: **[1] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016** (`knowledge_base/Deep Learning.pdf`), trọng tâm Chương 9 *Convolutional Networks* (tr. 330–372).

## Cấu trúc

```
src/
├── ml/                         # Mô hình & huấn luyện (PyTorch)
│   ├── config.py               # lớp, đường dẫn, hằng số tiền xử lý
│   ├── cnn_numpy.py            # conv2d / ReLU / max-pool cài đặt tay (Eq. 9.6, 9.7 trong [1])
│   ├── models.py               # VehicleCNN (từ đầu, 1,17M tham số) + transfer learning
│   ├── data.py                 # ImageFolder, augmentation, cân bằng lớp
│   ├── train_classifier.py     # huấn luyện phân loại (AdamW, cosine LR, early stopping)
│   ├── evaluate_classifier.py  # accuracy, P/R/F1, ma trận nhầm lẫn
│   ├── gradcam.py              # trực quan hoá vùng CNN chú ý
│   ├── detection_utils.py      # IoU, NMS, AP / mAP cài đặt tường minh
│   ├── prepare_coco_vehicles.py# trích tập con xe từ COCO → định dạng YOLO
│   └── train_detector.py       # fine-tune YOLOv8 cho 5 lớp phương tiện
├── configs/vehicles.yaml       # cấu hình dữ liệu YOLO
├── backend/                    # FastAPI
│   └── app/{main,services,schemas,config}.py
└── frontend/                   # ReactJS (Vite)
    └── src/{App.jsx, api.js, components/*}
```

5 lớp: `bicycle` (xe đạp), `bus` (xe buýt), `car` (ô tô con), `motorbike` (xe máy), `truck` (xe tải).

## 1. Cài đặt

```bash
cd src
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r backend/requirements.txt
```

## 2. Dữ liệu

* **Phân loại** — thư mục dạng ImageFolder: `data/classification/{train,val,test}/<lớp>/*.jpg`
  (có thể tự thu thập, hoặc cắt các hộp từ tập phát hiện).
* **Phát hiện** — định dạng YOLO: `data/detection/{images,labels}/{train,val}`.
  Có thể tạo nhanh từ COCO 2017: `python -m ml.prepare_coco_vehicles --coco <coco_dir> --max-images 5000`.

## 3. Huấn luyện

```bash
# CNN tự thiết kế
python -m ml.train_classifier --arch vehicle_cnn --epochs 40
# Học chuyển giao ResNet-18 (5 epoch train head, sau đó fine-tune)
python -m ml.train_classifier --arch resnet18 --epochs 20 --warmup-epochs 5 --lr 1e-3
# Đánh giá
python -m ml.evaluate_classifier --ckpt checkpoints/resnet18_best.pt
# Phát hiện: YOLOv8
python -m ml.train_detector --data configs/vehicles.yaml --model yolov8n.pt --epochs 50
```

## 4. Web demo

```bash
# Backend (mặc định dùng yolov8n.pt COCO, tự tải lần đầu — nhận diện được 5 loại xe ngay)
cd src/backend
uvicorn app.main:app --reload --port 8000        # Swagger: http://localhost:8000/docs

# Sau khi fine-tune, trỏ tới trọng số mới:
#   set DETECTOR_WEIGHTS=..\runs\detect\vehicles\weights\best.pt
#   set DETECTOR_IS_COCO=0

# Frontend
cd src/frontend
npm install
npm run dev                                       # http://localhost:5173
```

| Endpoint | Mô tả |
|---|---|
| `GET /api/health` | trạng thái mô hình |
| `GET /api/classes` | danh sách lớp |
| `POST /api/detect?conf=0.25&iou=0.5&refine=false` | ảnh → hộp, nhãn, số lượng từng loại |
| `POST /api/classify` | ảnh 1 phương tiện → xác suất softmax 5 lớp |

> Mã nguồn chưa được chạy thử (theo yêu cầu, để tránh nặng máy). Kiểm tra nhẹ trước khi huấn luyện:
> `python -m ml.cnn_numpy`, `python -m ml.detection_utils`, `python -m ml.models`.
