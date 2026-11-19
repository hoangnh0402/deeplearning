# Tài liệu trình bày phần mã nguồn (Slide 27 – 31)

> Đề tài: **Mạng nơ-ron tích chập (CNN) trong phát hiện và phân loại phương tiện giao thông**
> Phạm vi: Phần **04 – Triển khai hệ thống** của bộ slide `CNN_Phat_hien_Phan_loai_Phuong_tien.pptx`
> Tài liệu tham khảo chính: [1] Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press.

---

## Mục lục

0. [Tổng quan và phân bổ thời gian](#0-tổng-quan-và-phân-bổ-thời-gian)
1. [Slide 27 – Mở đầu phần Triển khai](#1-slide-27--mở-đầu-phần-triển-khai)
2. [Slide 28 – Cấu trúc mã nguồn](#2-slide-28--cấu-trúc-mã-nguồn)
3. [Slide 29 – Mã nguồn cốt lõi: VehicleCNN](#3-slide-29--mã-nguồn-cốt-lõi-vehiclecnn)
4. [Slide 30 – Kiến trúc web demo](#4-slide-30--kiến-trúc-web-demo-reactjs--fastapi--cnn)
5. [Slide 31 – Luồng xử lý một yêu cầu phát hiện](#5-slide-31--luồng-xử-lý-một-yêu-cầu-phát-hiện)
6. [Câu hỏi thường gặp khi bảo vệ](#6-câu-hỏi-thường-gặp-khi-bảo-vệ)
7. [Bảng thuật ngữ](#7-bảng-thuật-ngữ)
8. [Lưu ý trung thực khi trình bày](#8-lưu-ý-trung-thực-khi-trình-bày)

---

## 0. Tổng quan và phân bổ thời gian

| Slide | Nội dung | Thời gian gợi ý | File minh hoạ |
|---|---|---|---|
| 27 | Mở đầu phần 04 | ~15 giây | — |
| 28 | Cấu trúc mã nguồn, vai trò từng module | 2 – 3 phút | `src/ml/*` |
| 29 | Code VehicleCNN | 2 – 3 phút | `src/ml/models.py` |
| 30 | Kiến trúc 3 tầng React ↔ FastAPI ↔ PyTorch | 1,5 – 2 phút | `src/backend/app/main.py`, `src/frontend/src/App.jsx` |
| 31 | Luồng xử lý `/api/detect` | 1,5 – 2 phút | `src/backend/app/services.py`, `DetectionCanvas.jsx` |

**Thông điệp xuyên suốt cần truyền tải:** mọi khái niệm lý thuyết ở Phần 02 – 03 (tích chập, ReLU, pooling, softmax, cross-entropy, dropout, BatchNorm, IoU, NMS…) đều có **đoạn code tương ứng** trong dự án. Mã nguồn có hai lớp:

1. **Lớp minh hoạ** (viết tay bằng NumPy): `cnn_numpy.py`, `detection_utils.py` để chứng minh nhóm hiểu bản chất công thức.
2. **Lớp thực tế** (PyTorch, Ultralytics, FastAPI, React) để huấn luyện và triển khai được trên máy thật.

---

## 1. Slide 27 – Mở đầu phần Triển khai

**Lời dẫn gợi ý:**
> "Sau khi đã trình bày lý thuyết CNN và cách áp dụng cho bài toán, phần này nhóm trình bày cách hiện thực hoá thành một hệ thống chạy được, gồm ba thành phần: mã nguồn huấn luyện bằng PyTorch, API suy luận bằng FastAPI và giao diện web bằng ReactJS."

**Tại sao chọn bộ công nghệ này?**

| Công nghệ | Vai trò | Lý do chọn | Nhược điểm |
|---|---|---|---|
| **PyTorch** | Xây dựng và huấn luyện CNN | Phổ biến nhất trong nghiên cứu; *eager mode* dễ debug; autograd tự tính gradient (lan truyền ngược) | Triển khai production cần thêm bước (TorchScript/ONNX) |
| **torchvision** | Mô hình pretrained (ResNet, MobileNet…), augmentation | Có sẵn trọng số ImageNet, phục vụ học chuyển giao | — |
| **Ultralytics YOLOv8** | Phát hiện đối tượng | Huấn luyện, đánh giá, xuất ONNX chỉ vài dòng; tốc độ thời gian thực | Thư viện "hộp đen" hơn, giấy phép AGPL-3.0 khi thương mại hoá |
| **FastAPI** | REST API | Async, nhanh; tự sinh tài liệu Swagger tại `/docs`; kiểm tra dữ liệu bằng Pydantic | Chưa có sẵn hàng đợi tác vụ cho tải lớn |
| **ReactJS + Vite** | Giao diện web | Component tái sử dụng; Vite khởi động nhanh, có proxy dev | Cần Node.js, thêm một tiến trình chạy |

---

## 2. Slide 28 – Cấu trúc mã nguồn

```
src/
├─ ml/                      # Mô hình & huấn luyện (PyTorch)
│  ├─ config.py             # Hằng số dùng chung: lớp, đường dẫn, kích thước ảnh
│  ├─ cnn_numpy.py          # Conv / ReLU / MaxPool / softmax viết tay (NumPy)
│  ├─ models.py             # VehicleCNN + mô hình học chuyển giao
│  ├─ data.py               # Nạp dữ liệu, augmentation, cân bằng lớp
│  ├─ train_classifier.py   # Vòng lặp huấn luyện phân loại
│  ├─ evaluate_classifier.py# Accuracy, P/R/F1, ma trận nhầm lẫn
│  ├─ gradcam.py            # Giải thích CNN "nhìn" vào đâu
│  ├─ detection_utils.py    # IoU, NMS, AP/mAP viết tay
│  ├─ prepare_coco_vehicles.py # Trích lớp xe từ COCO → định dạng YOLO
│  └─ train_detector.py     # Fine-tune YOLOv8
├─ configs/vehicles.yaml    # Cấu hình dữ liệu YOLO
├─ backend/app/             # FastAPI
└─ frontend/src/            # ReactJS
```

**Lời dẫn gợi ý:**
> "Mã nguồn chia làm ba khối độc lập: `ml/` lo huấn luyện, `backend/` lo phục vụ mô hình, `frontend/` lo giao diện. Ba khối dùng chung một file cấu hình `ml/config.py` nên danh sách lớp và tiền xử lý luôn thống nhất."

Dưới đây là giải thích chi tiết từng module. Khi trình bày trên slide chỉ cần nói 1 – 2 câu mỗi module; phần chi tiết dùng để trả lời câu hỏi.

---

### 2.1. `config.py` – nguồn sự thật duy nhất (single source of truth)

```python
CLASSES = ["bicycle", "bus", "car", "motorbike", "truck"]          # thứ tự bảng chữ cái
COCO_TO_VEHICLE = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE_SCRATCH, IMG_SIZE_TRANSFER, IMG_SIZE_DETECT = 128, 224, 640
```

- **Tại sao các lớp xếp theo bảng chữ cái?** `torchvision.datasets.ImageFolder` gán nhãn 0..K-1 theo thứ tự bảng chữ cái của tên thư mục. Nếu danh sách trong code khác thứ tự này, nhãn dự đoán sẽ bị lệch (ví dụ dự đoán "car" nhưng hiển thị "bus").
- **`COCO_TO_VEHICLE`**: mô hình YOLOv8 gốc huấn luyện trên COCO (80 lớp). Ta chỉ giữ 5 chỉ số tương ứng với phương tiện, bỏ người, đèn giao thông…
- **Tại sao chuẩn hoá theo mean/std ImageNet?** Các mô hình pretrained (ResNet, MobileNet) được huấn luyện với phân phối đầu vào này; dùng đúng thống kê giúp đặc trưng học được vẫn có ý nghĩa ([1, §12.2.1]).
- **Tại sao 3 kích thước ảnh khác nhau?**
  - 128 px: VehicleCNN học từ đầu, mạng nhỏ, cần nhẹ để chạy CPU.
  - 224 px: kích thước chuẩn khi ResNet/MobileNet được huấn luyện trên ImageNet.
  - 640 px: kích thước chuẩn của YOLOv8; ảnh lớn giữ được các xe nhỏ ở xa.

---

### 2.2. `cnn_numpy.py` – các phép toán CNN viết tay

**Mục đích:** chứng minh rằng phép tích chập trong thư viện chỉ là các công thức trong sách. **Không dùng để huấn luyện** (chậm), chỉ để minh hoạ.

#### a) Kích thước đầu ra

```python
def output_size(w, k, p=0, s=1):
    return (w - k + 2 * p) // s + 1          # O = ⌊(W − K + 2P)/S⌋ + 1
```

- **W** kích thước vào, **K** kích thước kernel, **P** padding, **S** stride.
- Ví dụ: ảnh 128, kernel 3, padding 1, stride 1 → O = 128 (giữ nguyên kích thước, gọi là *same padding*).

#### b) Tích chập 2D (thực chất là cross-correlation, Eq. 9.6)

```python
def conv2d(image, kernel, stride=1, padding=0):
    ...
    for i in range(oh):
        for j in range(ow):
            patch = image[i*stride : i*stride+kh, j*stride : j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
```

- **Định nghĩa:** kernel (bộ lọc) trượt trên ảnh; tại mỗi vị trí, nhân từng phần tử của vùng ảnh với kernel rồi cộng lại → một giá trị của *feature map*.
- **Cross-correlation vs convolution:** tích chập toán học đúng nghĩa phải lật kernel 180°. Các thư viện DL không lật vì kernel được *học*, nên lật hay không cũng không ảnh hưởng kết quả ([1, §9.1, tr. 333]).
- **Thể hiện tương tác thưa (sparse interactions):** mỗi đầu ra chỉ phụ thuộc `kh × kw` điểm ảnh, không phải toàn bộ ảnh.

#### c) Tích chập đa kênh (Eq. 9.7)

```python
def conv2d_multi(x, weight, bias=None, ...):
    # x: (C_in, H, W) ; weight: (C_out, C_in, K, K)
    out[o, i, j] = np.sum(patch * weight[o])   # cộng qua mọi kênh vào
```

- Kernel là **tensor 4 chiều**. Mỗi kênh ra có một bộ lọc "dày" bằng số kênh vào.
- **Chia sẻ tham số (parameter sharing):** cùng `weight[o]` được dùng cho **mọi vị trí** (i, j). Số tham số = `C_out × C_in × K × K` (+ `C_out` bias), **không phụ thuộc kích thước ảnh**.

#### d) ReLU, Max pooling, Softmax

```python
def relu(x):      return np.maximum(0, x)
def max_pool2d(x, size=2, stride=2): ... .max(axis=(1, 2))
def softmax(z):   z = z - z.max(); e = np.exp(z); return e / e.sum()
```

| Thành phần | Định nghĩa | Tại sao dùng | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| **ReLU** `max(0, z)` | Hàm kích hoạt phi tuyến | Không có phi tuyến, nhiều lớp tuyến tính chồng lên vẫn chỉ là một phép tuyến tính | Tính nhanh; gradient = 1 khi z > 0 → giảm *vanishing gradient* | "ReLU chết": nơ-ron có z < 0 mãi thì gradient = 0, không học nữa |
| **Max pooling 2×2** | Lấy giá trị lớn nhất mỗi vùng 2×2 | Giảm kích thước 4 lần, bất biến với tịnh tiến nhỏ ([1, §9.3, tr. 342]) | Ít tính toán hơn; tăng trường tiếp nhận | Mất thông tin vị trí chính xác → bất lợi cho phát hiện |
| **Softmax** | Biến K logit thành phân phối xác suất | Đầu ra phân loại cần là xác suất có tổng = 1 | Kết hợp tự nhiên với cross-entropy | Dễ "quá tự tin" → dùng thêm label smoothing |

- **Mẹo `z - z.max()`:** tránh tràn số khi `exp` của số lớn (ổn định số học). Kết quả không đổi vì softmax bất biến khi cộng hằng số.
- **Demo:** `python -m ml.cnn_numpy` dùng kernel **Sobel-x** trên ảnh có một cạnh dọc → feature map có giá trị lớn đúng tại cạnh, minh hoạ trực quan slide 9.

---

### 2.3. `data.py` – dữ liệu và tăng cường dữ liệu

```python
transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),   # cắt + phóng ngẫu nhiên
transforms.RandomHorizontalFlip(p=0.5),                     # lật ngang
transforms.RandomRotation(10),                              # xoay nhẹ ±10°
transforms.ColorJitter(brightness=0.3, contrast=0.3, ...),  # mô phỏng ngày/đêm
transforms.RandomGrayscale(p=0.05),
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),       # mô phỏng che khuất
```

- **Data augmentation là gì?** Tạo thêm mẫu huấn luyện bằng các phép biến đổi **không làm đổi nhãn**. Đây là một dạng **chính quy hoá** chống quá khớp ([1, §7.4, tr. 240]).
- **Tại sao KHÔNG lật dọc?** Xe trong thực tế không bao giờ lộn ngược; lật dọc sinh dữ liệu phi thực tế, làm mô hình học sai.
- **Mỗi phép biến đổi ứng với một thách thức ở slide 6:** ColorJitter ↔ ánh sáng/thời tiết; RandomErasing ↔ che khuất; RandomResizedCrop ↔ đa tỉ lệ.
- **Tập val/test** chỉ Resize + CenterCrop + Normalize, **không augment**, để đánh giá khách quan.

**Cân bằng lớp:**

```python
def make_balanced_sampler(dataset):
    counts = Counter(dataset.targets)
    weights = [1.0 / counts[t] for t in dataset.targets]     # nghịch đảo tần suất
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

- **Vấn đề:** giao thông Việt Nam có rất nhiều xe máy, ít xe đạp/xe buýt. Nếu không xử lý, mô hình thiên về đoán "xe máy".
- **Giải pháp:** lớp hiếm được lấy mẫu với xác suất cao hơn → mỗi mini-batch xấp xỉ cân bằng.
- **Ưu:** đơn giản, không cần đổi hàm mất mát. **Nhược:** ảnh lớp hiếm bị lặp lại nhiều lần → có nguy cơ quá khớp lớp hiếm (augmentation giúp giảm bớt).

---

### 2.4. `train_classifier.py` – vòng lặp huấn luyện

**Một bước huấn luyện:**

```python
logits = model(x)                 # 1. Lan truyền tiến
loss = criterion(logits, y)       # 2. Tính J(θ) = −log p(y|x; θ)
optimizer.zero_grad()
loss.backward()                   # 3. Lan truyền ngược: tính ∇θ J
optimizer.step()                  # 4. Cập nhật: θ ← θ − ε·ĝ
```

**Các thành phần và lý do:**

| Thành phần | Code | Định nghĩa / Tại sao | Ưu | Nhược |
|---|---|---|---|---|
| **Cross-entropy + label smoothing** | `nn.CrossEntropyLoss(label_smoothing=0.1)` | Cực tiểu cross-entropy ⇔ cực đại hợp lý (MLE) ([1, §5.5]). Label smoothing thay nhãn one-hot [0,0,1,0,0] bằng [0.02,0.02,0.92,0.02,0.02] | Giảm quá tự tin, tổng quát hoá tốt hơn | Xác suất đầu ra thấp hơn thực tế một chút |
| **AdamW** | `AdamW(lr=1e-3, weight_decay=1e-4)` | Adam: tốc độ học thích nghi theo từng tham số ([1, §8.5.3]). "W": weight decay tách rời khỏi gradient | Hội tụ nhanh, ít phải tinh chỉnh lr | Đôi khi tổng quát hoá kém SGD+momentum một chút |
| **Weight decay (L2)** | `weight_decay=1e-4` | Phạt λ‖w‖², ép trọng số nhỏ ([1, §7.1.1]) | Chống quá khớp | Thêm một siêu tham số |
| **Cosine LR** | `CosineAnnealingLR` | Learning rate giảm dần theo đường cos | Đầu: bước lớn, khám phá nhanh; cuối: bước nhỏ, hội tụ tinh | — |
| **AMP** (mixed precision) | `torch.autocast`, `GradScaler` | Tính bằng float16 khi có GPU | Nhanh gần 2 lần, tiết kiệm bộ nhớ GPU | Chỉ bật khi có CUDA; cần `GradScaler` tránh gradient bị làm tròn về 0 |
| **Early stopping** | `patience=8` | Dừng khi val accuracy không cải thiện 8 epoch liên tiếp ([1, §7.8]) | Chống quá khớp, tiết kiệm thời gian | Có thể dừng sớm quá nếu val dao động |
| **Lưu checkpoint tốt nhất** | `torch.save({...}, f"{arch}_best.pt")` | Chỉ lưu khi val acc tăng | Luôn giữ mô hình tốt nhất | — |

**Định dạng checkpoint:** `{arch, classes, img_size, state_dict, val_acc}`. Lưu kèm `arch`, `classes`, `img_size` để lúc suy luận (backend, Grad-CAM, đánh giá) **tự dựng lại đúng mô hình và tiền xử lý** mà không cần nhớ tham số.

**Học chuyển giao hai giai đoạn (liên hệ slide 21):**

```python
if epoch == args.warmup_epochs + 1:
    unfreeze_backbone(model)                                   # mở băng toàn mạng
    optimizer = AdamW(model.parameters(), lr=args.lr * 0.1)    # lr nhỏ hơn 10 lần
```

- **Giai đoạn 1 (warm-up):** đóng băng backbone, chỉ huấn luyện lớp phân loại mới. Lý do: head mới khởi tạo ngẫu nhiên → gradient lớn và "hỗn loạn"; nếu cập nhật cả backbone ngay sẽ phá hỏng đặc trưng ImageNet tốt sẵn có.
- **Giai đoạn 2 (fine-tune):** mở băng, dùng lr nhỏ để tinh chỉnh nhẹ đặc trưng cho miền ảnh giao thông.

---

### 2.5. `evaluate_classifier.py` – đánh giá

- In **Precision, Recall, F1 từng lớp**, **Accuracy**, vẽ **ma trận nhầm lẫn** (chuẩn hoá theo hàng).
- **Tại sao không chỉ dùng Accuracy?** Dữ liệu lệch lớp: nếu 80% là xe máy, mô hình luôn đoán "xe máy" vẫn đạt 80% accuracy. **Macro-F1** tính trung bình đều giữa các lớp nên phản ánh đúng hơn.
- **Ma trận nhầm lẫn** chỉ ra cặp lớp hay nhầm, dự kiến là **xe tải ↔ xe buýt** (đều to, hình hộp).

---

### 2.6. `gradcam.py` – giải thích mô hình

**Công thức:**  L^c = ReLU( Σ_k α_k^c · A^k ),  với α_k^c = trung bình không gian của ∂y^c / ∂A^k

```python
h1 = layer.register_forward_hook(...)          # lấy feature map A của lớp conv cuối
h2 = layer.register_full_backward_hook(...)    # lấy gradient ∂y^c/∂A
logits[0, target].backward()
alpha = grads["g"].mean(dim=(2, 3), keepdim=True)   # trọng số từng kênh
cam = F.relu((alpha * acts["a"]).sum(1))            # tổng có trọng số
```

- **Ý tưởng:** kênh nào có gradient lớn với lớp dự đoán thì quan trọng; cộng các feature map theo trọng số đó → bản đồ nhiệt cho biết vùng ảnh nào ảnh hưởng đến quyết định.
- **Tại sao cần?** CNN thường bị coi là "hộp đen". Grad-CAM kiểm tra mô hình có nhìn vào **chiếc xe** hay chỉ nhìn vào **nền** (ví dụ đoán "xe buýt" vì thấy bến xe).
- **Ưu:** không cần sửa hay huấn luyện lại mô hình, dùng được cho mọi CNN.
- **Nhược:** độ phân giải thô (bằng feature map lớp cuối, 8×8 với VehicleCNN), chỉ là giải thích xấp xỉ.

---

### 2.7. `detection_utils.py` – IoU, NMS, mAP viết tay

**IoU (Intersection over Union):**

```python
lt = np.maximum(a[:, None, :2], b[None, :, :2])   # góc trên-trái phần giao
rb = np.minimum(a[:, None, 2:], b[None, :, 2:])   # góc dưới-phải phần giao
inter = clip(rb - lt).prod
iou = inter / (area_a + area_b - inter)
```

- Đo mức chồng lấp giữa 2 hộp, từ 0 (không giao) đến 1 (trùng khít). Code dùng **broadcasting** để tính ma trận IoU N×M một lần, không cần vòng lặp.

**NMS (Non-Maximum Suppression):**

```python
order = scores.argsort()[::-1]            # sắp xếp điểm giảm dần
while order.size > 0:
    i = order[0]; keep.append(i)          # giữ hộp điểm cao nhất
    ious = box_iou(boxes[i], boxes[order[1:]])[0]
    order = order[1:][ious <= iou_thr]    # loại các hộp trùng với nó
```

- **Tại sao cần NMS?** Bộ phát hiện sinh ra rất nhiều hộp (YOLOv8: 8 400 ứng viên) và nhiều hộp cùng bao một chiếc xe. NMS giữ lại hộp tốt nhất.
- **`batched_nms`** chạy NMS **riêng từng lớp**: một xe máy đứng sát ô tô không bị triệt tiêu chỉ vì hai hộp chồng lấp.
- **Nhược điểm của NMS tham lam:** ở cảnh rất đông (xe máy chen sát), hai xe thật chồng lấp nhiều có thể bị gộp làm một → đếm thiếu. Có thể khắc phục bằng Soft-NMS (hướng phát triển).

**AP / mAP:** AP = diện tích dưới đường Precision–Recall của một lớp; mAP = trung bình AP của 5 lớp. Hàm hỗ trợ cả mAP@0,5 (chuẩn VOC) và mAP@0,5:0,95 (chuẩn COCO).

**Kiểm tra nhanh:** `python -m ml.detection_utils` → IoU(0,1) ≈ 0,681, NMS giữ hộp [0, 2].

---

### 2.8. `prepare_coco_vehicles.py` và `train_detector.py` – phát hiện với YOLOv8

**Chuyển định dạng COCO → YOLO:**

```python
x, y, w, h = a["bbox"]                         # COCO: góc trên-trái + kích thước (pixel)
cx, cy = (x + w/2) / W, (y + h/2) / H          # YOLO: tâm + kích thước, chuẩn hoá [0,1]
```

Mỗi ảnh có một file `.txt`, mỗi dòng: `<class_id> <x_center> <y_center> <width> <height>`. Hộp nhỏ hơn 4 px bị bỏ (quá nhỏ, gần như nhiễu).

**Fine-tune YOLOv8:**

```python
model = YOLO("yolov8n.pt")                     # khởi tạo từ trọng số COCO → học chuyển giao
model.train(data="configs/vehicles.yaml", optimizer="AdamW", lr0=1e-3, cos_lr=True,
            patience=15, mosaic=1.0, mixup=0.1, fliplr=0.5, flipud=0.0, ...)
metrics = model.val()                          # mAP50, mAP50-95
model.export(format="onnx")                    # xuất ONNX để triển khai nhẹ
```

- **Mosaic:** ghép 4 ảnh thành 1, giúp mô hình thấy nhiều xe ở nhiều tỉ lệ và ngữ cảnh. **MixUp:** trộn 2 ảnh theo trọng số. **`flipud=0`:** không lật dọc, cùng lý do như phân loại.
- **Tại sao YOLOv8 (một giai đoạn) mà không phải Faster R-CNN (hai giai đoạn)?**

| | YOLOv8 (một giai đoạn) | Faster R-CNN (hai giai đoạn) |
|---|---|---|
| Tốc độ | Nhanh, thời gian thực (25–30 FPS dễ đạt) | Chậm hơn nhiều |
| Độ chính xác | Rất tốt, hơi kém với vật thể rất nhỏ/chồng lấp | Thường chính xác hơn với vật thể nhỏ |
| Dễ dùng | Huấn luyện/triển khai rất đơn giản | Cấu hình phức tạp hơn |
| Phù hợp đề tài | ✔ Camera giao thông cần thời gian thực | — |

- **Chọn bản `n` (nano):** nhỏ nhất (~3,2 M tham số), chạy được trên CPU cho demo. Bản `s`/`m` chính xác hơn nhưng chậm hơn (câu hỏi thực nghiệm 2 ở slide 26).
- **Lưu ý:** `configs/vehicles.yaml` dùng **đường dẫn tuyệt đối** vì Ultralytics hiểu đường dẫn tương đối theo thư mục riêng của nó.

---

## 3. Slide 29 – Mã nguồn cốt lõi: VehicleCNN

### 3.1. Code (`src/ml/models.py`)

```python
class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True))

class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, p_drop=0.0):
        layers = [ConvBNReLU(c_in, c_out), ConvBNReLU(c_out, c_out),
                  nn.MaxPool2d(kernel_size=2, stride=2)]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        super().__init__(*layers)

class VehicleCNN(nn.Module):
    def __init__(self, num_classes=5, widths=(32, 64, 128, 256), p_drop=0.5):
        super().__init__()
        blocks, c_in = [], 3
        for i, c_out in enumerate(widths):
            blocks.append(ConvBlock(c_in, c_out, p_drop=0.1 if i >= 2 else 0.0))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)                 # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
```

> Slide 29 hiển thị phiên bản rút gọn. Code thật có thêm `Dropout2d(0.1)` ở khối 3 và 4 (không làm thay đổi số tham số).

### 3.2. Giải thích theo luồng dữ liệu

| Khối | Kênh | Đầu ra | Tham số | Trường tiếp nhận |
|---|---|---|---|---|
| Đầu vào | 3 (RGB) | 3×128×128 | — | 1 px |
| Block 1 | 3 → 32 → 32 | 32×64×64 | 10 208 | 6 px |
| Block 2 | 32 → 64 → 64 | 64×32×32 | 55 552 | 16 px |
| Block 3 | 64 → 128 → 128 | 128×16×16 | 221 696 | 36 px |
| Block 4 | 128 → 256 → 256 | 256×8×8 | 885 760 | 76 px |
| GAP + FC | 256 → 5 | 5 logit | 1 285 | toàn ảnh |
| **Tổng** | | | **1 174 501 ≈ 1,17 M** | |

**Lời dẫn gợi ý:**
> "Mỗi khối đúng là cấu trúc ba tầng ở slide 11: convolution → detector (ReLU) → pooling. Sau mỗi khối, kích thước không gian giảm một nửa còn số kênh tăng gấp đôi: mạng đổi độ phân giải lấy độ giàu ngữ nghĩa. Lớp nông học cạnh, góc; lớp sâu học bánh xe, đèn pha, thùng xe."

### 3.3. Từng quyết định thiết kế: tại sao, ưu và nhược

**(1) Kernel 3×3, hai lớp xếp chồng thay vì một lớp 5×5**
- Hai lớp 3×3 có trường tiếp nhận bằng một lớp 5×5.
- Tham số: 2 × 9 × C² = 18C² so với 25C² → **ít hơn 28%**.
- Thêm một lớp ReLU → mô hình **phi tuyến hơn**, biểu diễn tốt hơn (ý tưởng của VGG).
- Nhược: sâu hơn, khó huấn luyện hơn một chút; đã có BatchNorm bù lại.

**(2) `padding = k // 2` (same padding)**
- Với k = 3 → padding = 1 → kích thước giữ nguyên qua lớp conv; chỉ MaxPool mới giảm kích thước. Giúp tính toán kích thước dễ dàng và giữ thông tin ở biên ảnh ([1, §9.5, tr. 349–351]).

**(3) BatchNorm và `bias=False`**
- **BatchNorm:** chuẩn hoá kích hoạt của mỗi mini-batch về trung bình 0, phương sai 1, rồi co giãn bằng tham số học được γ, β ([1, §8.7.1]).
- **Tại sao dùng:** cho phép learning rate lớn hơn, hội tụ nhanh hơn, giảm phụ thuộc vào khởi tạo, có tác dụng chính quy hoá nhẹ.
- **Tại sao `bias=False`:** BatchNorm trừ đi trung bình nên bias của Conv bị triệt tiêu; tham số β của BN đã làm nhiệm vụ dịch chuyển → bias là thừa.
- **Nhược:** hoạt động kém khi batch rất nhỏ (< 8); hành vi khác nhau giữa train và eval (phải gọi `model.eval()` khi suy luận).

**(4) Max pooling 2×2, stride 2**
- Giảm kích thước một nửa mỗi chiều, tăng nhanh trường tiếp nhận, bất biến với dịch chuyển nhỏ.
- Phù hợp phân loại (chỉ cần biết *có* xe gì, không cần biết *ở đâu*).

**(5) Global Average Pooling (GAP) thay vì Flatten**
- GAP lấy trung bình mỗi feature map 8×8 → 1 số; 256 kênh → vector 256 chiều.
- **So sánh:** nếu Flatten 256×8×8 = 16 384 chiều rồi nối vào FC 512 → 8,4 M tham số chỉ riêng một lớp, gấp 7 lần toàn bộ mạng. GAP + FC chỉ cần **1 285 tham số**.
- **Ưu:** ít tham số → ít quá khớp; chấp nhận ảnh đầu vào nhiều kích thước; tương thích Grad-CAM.
- **Nhược:** mất thông tin vị trí không gian (chấp nhận được với phân loại).

**(6) Dropout**
- `Dropout(0.5)` trước lớp FC: mỗi bước huấn luyện tắt ngẫu nhiên 50% nơ-ron → mạng không phụ thuộc vào một vài nơ-ron, xấp xỉ huấn luyện tổ hợp nhiều mạng con ([1, §7.12]).
- `Dropout2d(0.1)` ở khối 3, 4: tắt nguyên **cả kênh** (thích hợp cho feature map vì các điểm lân cận tương quan mạnh, tắt từng điểm lẻ ít tác dụng).
- **Nhược:** huấn luyện cần nhiều epoch hơn; khi suy luận phải tắt (`model.eval()`).

**(7) Không có Softmax trong `forward`**
- Mô hình trả **logit**; `nn.CrossEntropyLoss` đã gộp `log_softmax` + NLL bên trong, **ổn định số học hơn** so với tự tính softmax rồi lấy log. Khi suy luận, backend mới gọi `torch.softmax` để ra xác suất.

**(8) Khởi tạo Kaiming (He)**
- `kaiming_normal_(mode="fan_out", nonlinearity="relu")`: phương sai trọng số = 2 / fan, được thiết kế riêng cho ReLU để tín hiệu không bị tắt dần hay bùng nổ qua nhiều lớp.

### 3.4. Ưu và nhược tổng thể của VehicleCNN

| Ưu điểm | Nhược điểm |
|---|---|
| Nhẹ (1,17 M tham số), huấn luyện được trên CPU/GPU phổ thông | Cần nhiều dữ liệu hơn học chuyển giao vì học từ đầu |
| Minh hoạ trọn vẹn lý thuyết Chương 9 | Độ chính xác dự kiến thấp hơn ResNet-18 pretrained |
| Toàn quyền kiểm soát kiến trúc, dễ giải thích từng lớp | Chỉ phân loại ảnh đã cắt sẵn 1 xe, không định vị được |

### 3.5. So sánh với các mô hình học chuyển giao trong code

| Mô hình | Tham số (xấp xỉ) | Ưu | Nhược |
|---|---|---|---|
| **VehicleCNN** | 1,17 M | Nhẹ, tự thiết kế | Cần nhiều dữ liệu |
| **ResNet-18** | 11,2 M | Khối dư `y = F(x) + x` giúp huấn luyện mạng sâu; chính xác cao | Nặng hơn ~10 lần |
| **MobileNetV2** | 2,2 M | Tích chập tách biệt theo chiều sâu (depthwise separable), rất nhanh, hợp thiết bị biên | Chính xác hơi kém ResNet |
| **EfficientNet-B0** | 4,0 M | Cân bằng độ sâu/rộng/độ phân giải, hiệu quả tốt | Huấn luyện chậm hơn MobileNet |

Tất cả được dựng qua cùng một hàm `build_model(arch)` → đổi mô hình chỉ cần đổi tham số `--arch`.

---

## 4. Slide 30 – Kiến trúc web demo: ReactJS ↔ FastAPI ↔ CNN

### 4.1. Sơ đồ

```
┌───────────────────────┐  multipart/form-data   ┌───────────────────────┐        ┌────────────────────┐
│ Frontend · ReactJS    │ ─────────────────────▶ │ Backend · FastAPI     │ ─────▶ │ Mô hình · PyTorch  │
│ - Tải/kéo-thả ảnh     │                        │ - Kiểm tra định dạng  │        │ - YOLOv8 phát hiện │
│ - Chỉnh conf, IoU     │ ◀───────────────────── │ - Suy luận threadpool │ ◀───── │ - CNN phân loại    │
│ - Vẽ hộp lên canvas   │         JSON           │ - Schema Pydantic     │        │ - Nạp 1 lần        │
└───────────────────────┘                        └───────────────────────┘        └────────────────────┘
     localhost:5173                                   localhost:8000
```

**Lời dẫn gợi ý:**
> "Hệ thống chia ba tầng tách biệt. Giao diện chỉ lo hiển thị, không biết gì về mô hình. Backend nhận ảnh, kiểm tra, gọi mô hình và trả JSON. Nhờ tách biệt, ta có thể thay YOLOv8n bằng mô hình đã fine-tune mà không sửa một dòng giao diện nào."

### 4.2. Các endpoint

| Endpoint | Chức năng |
|---|---|
| `GET /api/health` | Trạng thái: mô hình đã nạp chưa, chạy trên CPU hay GPU |
| `GET /api/classes` | Danh sách 5 lớp kèm tên tiếng Việt |
| `POST /api/detect?conf=0.25&iou=0.5&refine=false` | Ảnh cảnh giao thông → hộp, nhãn, số lượng từng loại |
| `POST /api/classify` | Ảnh một phương tiện → xác suất softmax 5 lớp |

### 4.3. Backend: các kỹ thuật và lý do (`backend/app/main.py`, `services.py`)

**(1) Kiểm tra đầu vào**

```python
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
if file.content_type not in ALLOWED_TYPES: raise HTTPException(415, ...)
if len(data) > settings.max_upload_mb * 1024 * 1024: raise HTTPException(413, ...)
Image.open(io.BytesIO(data)).convert("RGB")       # ảnh hỏng → 400
```

- Tránh ảnh quá lớn làm treo server; mã lỗi HTTP chuẩn (415 sai định dạng, 413 quá lớn, 400 ảnh hỏng).
- `.convert("RGB")`: ảnh PNG có kênh alpha (4 kênh) hoặc ảnh xám (1 kênh) được đưa về 3 kênh như mô hình yêu cầu.

**(2) Schema Pydantic (`schemas.py`)**

```python
class Detection(BaseModel):
    label: str; label_vi: str
    confidence: float = Field(..., ge=0, le=1)
    box: BBox; refined_by_classifier: bool = False
```

- **Định nghĩa:** mô tả cấu trúc JSON trả về bằng class Python có kiểu dữ liệu.
- **Tại sao:** FastAPI tự kiểm tra dữ liệu ra, tự sinh tài liệu Swagger tại `/docs`; frontend biết chính xác định dạng nhận được.

**(3) Suy luận trong threadpool**

```python
return await run_in_threadpool(detect_and_count, img, conf, iou, refine)
```

- **Vấn đề:** FastAPI chạy trên *event loop* bất đồng bộ (một luồng). Suy luận CNN tốn hàng chục đến hàng trăm ms CPU; nếu chạy trực tiếp sẽ **chặn** event loop, mọi request khác (kể cả `/health`) phải chờ.
- **Giải pháp:** đẩy tác vụ nặng sang luồng phụ, event loop tiếp tục phục vụ request khác.
- **Nhược:** vẫn dùng chung CPU; tải lớn cần giải pháp mạnh hơn (hàng đợi Celery, Triton Inference Server, chạy nhiều worker).

**(4) Lazy singleton – nạp mô hình một lần**

```python
class VehicleDetector:
    _lock = threading.Lock()
    def load(self):
        with self._lock:
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO(settings.detector_weights)
        return self.model

detector = VehicleDetector()          # một thể hiện duy nhất cho cả ứng dụng
```

- **Singleton:** chỉ có một đối tượng mô hình trong bộ nhớ, dùng lại cho mọi request. Nạp trọng số mất vài giây, không thể nạp lại mỗi request.
- **Lazy (nạp muộn):** chỉ nạp khi có request đầu tiên → server khởi động nhanh; nếu chưa có checkpoint phân loại thì endpoint phát hiện vẫn hoạt động.
- **Lock:** hai request đầu tiên đến cùng lúc (chạy ở hai luồng khác nhau) không nạp mô hình hai lần.
- **Nhược:** request đầu tiên chậm hơn (phải chờ nạp mô hình).

**(5) Cấu hình bằng biến môi trường (`config.py`)**

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DETECTOR_WEIGHTS` | `yolov8n.pt` | Trọng số YOLO (tự tải lần đầu) |
| `DETECTOR_IS_COCO` | `1` | `1`: ánh xạ lớp COCO → 5 lớp xe; `0`: dùng tên lớp của mô hình đã fine-tune |
| `CLASSIFIER_CKPT` | `checkpoints/resnet18_best.pt` | Checkpoint phân loại |
| `DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `MAX_UPLOAD_MB` | `10` | Giới hạn dung lượng ảnh |

- **Tại sao:** đổi mô hình hoặc thiết bị **không cần sửa code**. Demo chạy ngay với YOLOv8n COCO; sau khi fine-tune chỉ cần đặt `DETECTOR_WEIGHTS=...best.pt` và `DETECTOR_IS_COCO=0`.

**(6) CORS và proxy Vite**
- Trình duyệt chặn trang ở cổng 5173 gọi API ở cổng 8000 (chính sách *same-origin*). Hai cách giải quyết, dự án dùng cả hai:
  - Backend bật `CORSMiddleware` cho phép `localhost:5173`.
  - Vite proxy `/api` → `localhost:8000` khi phát triển, nên frontend chỉ gọi đường dẫn tương đối `/api/...`.

### 4.4. Frontend (`frontend/src/`)

| File | Vai trò |
|---|---|
| `App.jsx` | Trạng thái chung (ảnh, chế độ phát hiện/phân loại, ngưỡng, kết quả), gọi API |
| `api.js` | Gửi ảnh bằng `FormData` (multipart), xử lý lỗi từ backend |
| `ImageUploader.jsx` | Chọn hoặc kéo-thả ảnh |
| `DetectionCanvas.jsx` | Vẽ ảnh gốc + hộp màu theo lớp lên `<canvas>` |
| `CountPanel.jsx` | Bảng đếm số lượng từng loại xe |
| `ProbabilityBars.jsx` | Thanh xác suất 5 lớp (chế độ phân loại) |

- **Tại sao gửi `multipart/form-data` mà không gửi base64 trong JSON?** Base64 làm dữ liệu to thêm ~33%, phải mã hoá/giải mã; multipart gửi nhị phân trực tiếp.
- **Tại sao vẽ trên `<canvas>`?** Toạ độ hộp từ backend tính theo pixel ảnh gốc; canvas đặt kích thước bằng ảnh gốc nên vẽ trực tiếp, không cần quy đổi tỉ lệ.

---

## 5. Slide 31 – Luồng xử lý một yêu cầu phát hiện

### 5.1. Sáu bước

| Bước | Diễn ra ở đâu | Chi tiết |
|---|---|---|
| 1 | React | Người dùng chọn ảnh, chỉnh thanh trượt conf (mặc định 0,25) và IoU (0,50) |
| 2 | `api.js` | `POST /api/detect?conf=..&iou=..&refine=..` kèm ảnh |
| 3 | `services.py` | YOLOv8 suy luận → lọc theo confidence → NMS → giữ 5 lớp xe |
| 4 | `services.py` | (Nếu `refine=true`) cắt từng hộp, CNN phân loại lại nhãn |
| 5 | FastAPI | Trả JSON: hộp, nhãn tiếng Việt, độ tin cậy, đếm số lượng, thời gian suy luận |
| 6 | `DetectionCanvas.jsx` | Vẽ hộp màu theo lớp + bảng thống kê |

### 5.2. Code bước 3: phát hiện và lọc lớp

```python
result = model.predict(np.asarray(img), conf=conf, iou=iou, device=settings.device)[0]
for xyxy, c, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
    if settings.detector_is_coco:
        label = COCO_TO_VEHICLE.get(int(cls))
        if label is None:            # bỏ người, đèn giao thông, ...
            continue
    else:
        label = model.names[int(cls)]
```

- **`conf`** (ngưỡng tin cậy): hộp có điểm thấp hơn bị bỏ. Tăng → ít hộp sai (precision tăng) nhưng dễ bỏ sót xe (recall giảm).
- **`iou`** (ngưỡng NMS): hai hộp cùng lớp chồng lấp hơn ngưỡng thì bỏ hộp điểm thấp. Giảm → loại trùng mạnh hơn nhưng dễ gộp nhầm hai xe đứng sát nhau.
- Ultralytics đã thực hiện NMS bên trong `predict`; `detection_utils.nms` là bản viết tay cùng thuật toán để giải thích.
- **Ánh xạ COCO:** mô hình COCO gốc đã biết sẵn car, bus, truck, motorcycle, bicycle → **demo chạy được ngay không cần huấn luyện**.

### 5.3. Code bước 4: tinh chỉnh nhãn bằng CNN phân loại

```python
if refine and classifier.available:
    x1, y1, x2, y2 = map(int, d["box"])
    if (x2 - x1) >= 24 and (y2 - y1) >= 24:             # bỏ hộp quá nhỏ
        ranked, cms = classifier.predict(img.crop((x1, y1, x2, y2)))
        if ranked[0][1] > score:                        # tin bộ phân loại khi nó tự tin hơn
            label, score, refined = ranked[0][0], ranked[0][1], True
```

- **Ý tưởng:** YOLO phải vừa định vị vừa phân loại trên cả bức ảnh; CNN phân loại chỉ tập trung vào một chiếc xe đã cắt, ở độ phân giải riêng → có thể phân biệt tốt hơn các cặp dễ nhầm (xe tải ↔ xe buýt). Đây là câu hỏi thực nghiệm 3 ở slide 26.
- **Tại sao bỏ hộp < 24 px?** Phóng hộp quá nhỏ lên 224 px chủ yếu là nhiễu, kết quả phân loại không đáng tin.
- **Quy tắc quyết định:** chỉ thay nhãn khi xác suất của bộ phân loại **cao hơn** confidence của YOLO. Đơn giản, dễ giải thích.
- **Ưu:** tận dụng được cả hai mô hình; bật/tắt bằng một checkbox.
- **Nhược:**
  - Chậm hơn: thêm một lần suy luận CNN **cho mỗi hộp** (ảnh 20 xe → thêm 20 lần).
  - Confidence của YOLO (sigmoid) và xác suất softmax không hoàn toàn cùng thang đo nên so sánh trực tiếp chỉ là heuristic.
  - Cần có checkpoint phân loại đã huấn luyện; nếu chưa có, bước này tự bỏ qua.

### 5.4. Code bước 5: đếm và trả kết quả

```python
counts = Counter(o["label"] for o in out)
return {"width": img.width, "height": img.height, "detections": out,
        "counts": {c: counts.get(c, 0) for c in CLASSES},      # đủ 5 lớp, kể cả lớp = 0
        "inference_ms": round(ms, 1), "model": str(settings.detector_weights)}
```

- Trả đủ 5 lớp (kể cả lớp có 0 xe) để giao diện luôn hiển thị bảng thống kê cố định.
- `inference_ms` đo thời gian suy luận thực tế, dùng để đánh giá khả năng chạy thời gian thực.

### 5.5. Bước 6: hiển thị

- Mỗi lớp một màu cố định (xe máy cam, ô tô xanh dương, xe buýt vàng…).
- Nhãn hiển thị: `Xe máy 91%`; nếu được CNN phân loại tinh chỉnh thì có thêm dấu `✓`.

> **Lưu ý:** hình giao diện trên slide 31 chỉ là **phác thảo bố cục, không phải kết quả thật**. Nên nói rõ điều này khi trình bày.

---

## 6. Câu hỏi thường gặp khi bảo vệ

**H1. Tại sao phải viết tay `cnn_numpy.py` khi PyTorch đã có sẵn?**
Để chứng minh hiểu bản chất: phép tích chập chỉ là vòng lặp nhân–cộng trên vùng lân cận theo Eq. 9.6/9.7. Code viết tay chạy rất chậm nên chỉ dùng minh hoạ; huấn luyện thật dùng PyTorch (tối ưu bằng cuDNN, GPU).

**H2. Tại sao không dùng mạng kết nối đầy đủ (MLP)?**
Ảnh 128×128×3 = 49 152 đầu vào; chỉ một lớp FC ra 512 nơ-ron đã cần 25,2 M tham số, gấp 21 lần toàn bộ VehicleCNN. MLP cũng không tận dụng được cấu trúc không gian (điểm ảnh lân cận liên quan nhau) và không đẳng biến tịnh tiến.

**H3. Phân loại và phát hiện khác nhau thế nào, sao cần cả hai?**
Phân loại: ảnh 1 xe → 1 nhãn. Phát hiện: ảnh nhiều xe → nhiều (hộp, nhãn, điểm). Nhánh phân loại minh hoạ lý thuyết CNN đầy đủ và so sánh học từ đầu với học chuyển giao; nhánh phát hiện giải quyết bài toán thực tế (đếm xe). Hai nhánh kết hợp ở tuỳ chọn `refine`.

**H4. Học chuyển giao là gì, tại sao hiệu quả?**
Dùng lại mạng đã huấn luyện trên tập lớn (ImageNet 1,2 triệu ảnh), chỉ thay lớp cuối. Các lớp đầu học đặc trưng chung (cạnh, kết cấu, hình dạng) dùng được cho nhiều bài toán, nên khi dữ liệu xe ít vẫn đạt kết quả tốt ([1, §15.2]).

**H5. Tại sao đóng băng backbone ở giai đoạn đầu?**
Head mới khởi tạo ngẫu nhiên sinh gradient lớn, nếu lan ngược vào backbone ngay sẽ phá hỏng đặc trưng ImageNet. Huấn luyện head trước cho ổn định, rồi mới fine-tune toàn mạng với learning rate nhỏ hơn 10 lần.

**H6. Làm sao xử lý dữ liệu mất cân bằng?**
`WeightedRandomSampler` lấy mẫu theo nghịch đảo tần suất lớp; đánh giá bằng macro-F1 và mAP thay vì chỉ accuracy.

**H7. Làm sao chống quá khớp?**
Data augmentation, Dropout, weight decay, BatchNorm, label smoothing, early stopping, GAP thay Flatten (ít tham số).

**H8. NMS hoạt động thế nào, có nhược điểm gì?**
Sắp xếp hộp theo điểm, giữ hộp cao nhất, loại các hộp cùng lớp có IoU > ngưỡng, lặp lại. Nhược điểm: ở cảnh xe máy chen sát, hai xe thật chồng lấp nhiều có thể bị gộp → đếm thiếu. Có thể dùng Soft-NMS hoặc tracking.

**H9. Tại sao chạy suy luận trong threadpool?**
FastAPI dùng event loop bất đồng bộ; tác vụ nặng CPU chạy trực tiếp sẽ chặn mọi request khác. Threadpool đẩy sang luồng phụ.

**H10. Mô hình đã được huấn luyện và có kết quả chưa?**
Trả lời trung thực: mã nguồn đã hoàn chỉnh; kết quả số sẽ cập nhật sau khi huấn luyện (slide 26). Demo hiện dùng YOLOv8n pretrained COCO, vốn đã nhận diện được 5 loại xe.

**H11. Grad-CAM dùng để làm gì?**
Tạo bản đồ nhiệt vùng ảnh ảnh hưởng nhiều nhất đến quyết định, để kiểm tra mô hình có nhìn vào chiếc xe thật hay chỉ học "đường tắt" từ nền ảnh.

**H12. Muốn chạy thời gian thực trên camera thì làm gì?**
Xuất ONNX/TensorRT (`train_detector.py` đã có `model.export(format="onnx")`), lượng tử hoá INT8, chạy trên Jetson; thêm DeepSORT/ByteTrack để theo vết và đếm xe qua vạch (slide 32).

---

## 7. Bảng thuật ngữ

| Thuật ngữ | Tiếng Việt | Giải thích ngắn |
|---|---|---|
| Convolution / Cross-correlation | Tích chập / Tương quan chéo | Kernel trượt trên ảnh, nhân–cộng vùng lân cận |
| Kernel / Filter | Bộ lọc | Ma trận trọng số nhỏ (3×3) được học |
| Feature map | Bản đồ đặc trưng | Đầu ra của một lớp tích chập |
| Stride | Bước trượt | Số pixel kernel dịch mỗi lần |
| Padding | Đệm biên | Thêm pixel 0 quanh ảnh để giữ kích thước |
| Receptive field | Trường tiếp nhận | Vùng ảnh gốc mà một nơ-ron "nhìn" thấy |
| Sparse interactions | Tương tác thưa | Mỗi đầu ra chỉ nối vài đầu vào lân cận |
| Parameter sharing | Chia sẻ tham số | Một kernel dùng cho mọi vị trí |
| Equivariance | Đẳng biến | Dịch đầu vào thì đầu ra dịch theo |
| Pooling | Gộp | Tóm tắt vùng lân cận (max/average) |
| GAP | Gộp trung bình toàn cục | Trung bình mỗi feature map thành 1 số |
| ReLU | Hàm chỉnh lưu tuyến tính | `max(0, z)` |
| Softmax | — | Biến logit thành phân phối xác suất |
| Logit | — | Đầu ra chưa chuẩn hoá của lớp cuối |
| Cross-entropy | Entropy chéo | Hàm mất mát cho phân loại |
| Label smoothing | Làm mềm nhãn | Thay nhãn one-hot bằng nhãn "mềm" |
| Back-propagation | Lan truyền ngược | Tính gradient bằng quy tắc dây chuyền |
| BatchNorm | Chuẩn hoá theo lô | Chuẩn hoá kích hoạt mỗi mini-batch |
| Dropout | — | Tắt ngẫu nhiên nơ-ron khi huấn luyện |
| Weight decay | Suy giảm trọng số | Chính quy hoá L2 |
| Early stopping | Dừng sớm | Dừng khi val không cải thiện |
| Data augmentation | Tăng cường dữ liệu | Biến đổi ảnh không đổi nhãn |
| Transfer learning | Học chuyển giao | Dùng lại mạng huấn luyện sẵn |
| Fine-tune | Tinh chỉnh | Huấn luyện tiếp với learning rate nhỏ |
| Backbone / Neck / Head | Xương sống / Cổ / Đầu | Trích đặc trưng / trộn đa tỉ lệ / dự đoán |
| Bounding box | Hộp bao | (x1, y1, x2, y2) |
| IoU | Tỉ lệ giao trên hợp | Mức chồng lấp hai hộp |
| NMS | Triệt tiêu phi cực đại | Loại hộp trùng lặp |
| AP / mAP | Độ chính xác trung bình | Diện tích dưới đường P–R / trung bình các lớp |
| Grad-CAM | — | Bản đồ nhiệt giải thích quyết định CNN |
| Checkpoint | Điểm lưu | File lưu trọng số mô hình |
| Inference | Suy luận | Dùng mô hình đã huấn luyện để dự đoán |
| Singleton | — | Chỉ có một thể hiện duy nhất của đối tượng |
| Threadpool | Nhóm luồng | Tập luồng phụ chạy tác vụ nặng |
| CORS | Chia sẻ tài nguyên khác nguồn | Cơ chế cho phép trang web gọi API khác cổng/miền |
| ONNX | — | Định dạng mô hình trung lập, chạy trên nhiều nền tảng |

---

## 8. Lưu ý trung thực khi trình bày

1. **Mã nguồn chưa được chạy huấn luyện** và **chưa có số liệu kết quả**. Không trình bày con số accuracy/mAP nào như là kết quả đã đạt.
2. **Hình giao diện slide 31 là phác thảo**, không phải ảnh chụp kết quả thật.
3. **Slide 29 là code rút gọn**: code thật có thêm `Dropout2d(0.1)` ở khối 3, 4 và khởi tạo Kaiming.
4. **NMS trong backend do Ultralytics thực hiện**; `detection_utils.py` là bản viết tay để giải thích (và có thể dùng tính mAP).
5. **Nếu muốn demo trực tiếp, nên chạy thử trước buổi trình bày.** Các lệnh nhẹ, không cần dữ liệu:
   ```bash
   cd src
   python -m ml.models            # in số tham số VehicleCNN: 1,174,501
   python -m ml.cnn_numpy         # demo tích chập Sobel viết tay
   python -m ml.detection_utils   # demo IoU, NMS
   ```
   Web demo (YOLOv8n tự tải lần đầu, cần mạng):
   ```bash
   cd src/backend && uvicorn app.main:app --port 8000     # Swagger: http://localhost:8000/docs
   cd src/frontend && npm install && npm run dev          # http://localhost:5173
   ```
   Chế độ **Phân loại** (`/api/classify`) sẽ báo lỗi 503 cho đến khi có checkpoint phân loại đã huấn luyện.
