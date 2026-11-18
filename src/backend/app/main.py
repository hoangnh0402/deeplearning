"""FastAPI — API phát hiện & phân loại phương tiện.

Chạy:  cd src/backend && uvicorn app.main:app --reload --port 8000
Docs:  http://localhost:8000/docs  (Swagger UI tự sinh)

Endpoint:
    GET  /api/health      trạng thái mô hình
    GET  /api/classes     danh sách lớp
    POST /api/detect      ảnh → danh sách hộp + nhãn + đếm số lượng
    POST /api/classify    ảnh (đã cắt sẵn 1 phương tiện) → xác suất từng lớp
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .config import settings
from .schemas import ClassifyResponse, DetectResponse, HealthResponse
from .services import CLASSES, CLASSES_VI, classifier, detect_and_count, detector

app = FastAPI(
    title="Vehicle Detection & Classification API",
    description="Demo đề tài: CNN trong phát hiện và phân loại phương tiện giao thông",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


async def read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Định dạng không hỗ trợ: {file.content_type}")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Ảnh vượt quá {settings.max_upload_mb} MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(400, "Không đọc được ảnh") from e


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", detector_loaded=detector.loaded,
                          classifier_loaded=classifier.loaded, device=settings.device)


@app.get("/api/classes")
def classes():
    return [{"label": c, "label_vi": CLASSES_VI[c]} for c in CLASSES]


@app.post("/api/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99, description="Ngưỡng confidence"),
    iou: float = Query(0.5, ge=0.1, le=0.95, description="Ngưỡng IoU cho NMS"),
    refine: bool = Query(False, description="Phân loại lại từng hộp bằng CNN phân loại"),
):
    img = await read_image(file)
    # Suy luận CNN là tác vụ nặng CPU/GPU → chạy trong threadpool để không chặn event loop
    return await run_in_threadpool(detect_and_count, img, conf, iou, refine)


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    img = await read_image(file)
    try:
        ranked, ms = await run_in_threadpool(classifier.predict, img)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    probs = [{"label": c, "label_vi": CLASSES_VI.get(c, c), "probability": p} for c, p in ranked]
    return {"top1": probs[0], "probabilities": probs, "inference_ms": round(ms, 1),
            "model": classifier.ckpt["arch"]}
