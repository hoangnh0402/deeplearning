# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University deep-learning project (Vietnamese): CNNs for detecting and classifying vehicles into 5 classes — `bicycle, bus, car, motorbike, truck`. Code comments, docstrings, API error messages and UI strings are in Vietnamese; keep that convention. Docstrings cite Goodfellow, Bengio & Courville, *Deep Learning* (2016) as `[1, §x.y, tr. N]` — the report (`docs/baocao/*.docx`) and slides (`docs/slide/*.pptx`) depend on those references and on numbers stated in code (e.g. VehicleCNN = 1,174,501 params), so keep them consistent when changing models.

All code lives in `src/`. Run Python commands from `src/` — the `ml` package is imported as `ml.*` and default paths (`data/`, `checkpoints/`, `runs/`, `configs/`) are relative to `src/`. These dirs and `*.pt` weights are gitignored.

The code has never actually been run (per the user's request, to avoid heavy compute). Don't launch training or download large datasets/weights without asking.

## Commands

```bash
# Setup (from src/)
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt          # includes ../requirements.txt (torch, ultralytics, ...)

# Lightweight smoke checks (no data needed) — the closest thing to tests; there is no test suite or linter config
python -m ml.cnn_numpy          # hand-written conv/ReLU/pool vs. formulas
python -m ml.detection_utils    # IoU / NMS / AP self-check
python -m ml.models             # builds VehicleCNN, prints param count, forward pass

# Training / evaluation
python -m ml.train_classifier --arch vehicle_cnn --epochs 40
python -m ml.train_classifier --arch resnet18 --epochs 20 --warmup-epochs 5 --lr 1e-3   # also mobilenet_v2, efficientnet_b0
python -m ml.evaluate_classifier --ckpt checkpoints/resnet18_best.pt
python -m ml.gradcam --ckpt <ckpt> --image <img>
python -m ml.prepare_coco_vehicles --coco <coco_dir> --max-images 5000
python -m ml.train_detector --data configs/vehicles.yaml --model yolov8n.pt --epochs 50

# Backend (from src/backend) — Swagger at http://localhost:8000/docs
uvicorn app.main:app --reload --port 8000

# Frontend (from src/frontend) — http://localhost:5173
npm install && npm run dev      # npm run build for production
```

## Architecture

Three layers, wired together through `ml/config.py`, which is the single source of truth for `CLASSES` (alphabetical — must match `ImageFolder` label order), Vietnamese labels `CLASSES_VI`, the `COCO_TO_VEHICLE` index mapping, ImageNet normalization stats and image sizes (128 scratch / 224 transfer / 640 YOLO).

**`ml/` (PyTorch)** — two independent pipelines:
- *Classification*: `models.build_model(arch)` returns either the custom `VehicleCNN` or a torchvision transfer model (ResNet-18 / MobileNetV2 / EfficientNet-B0) with a replaced head. Transfer training is two-phase: with `--warmup-epochs > 0` the backbone is frozen, then `unfreeze_backbone()` is called and the cosine LR scheduler is rebuilt. The best checkpoint is saved as `checkpoints/{arch}_best.pt`, a dict `{arch, classes, img_size, state_dict, val_acc}`; `evaluate_classifier.load_checkpoint` rebuilds the model from that dict (the backend reuses this function). Data is ImageFolder at `data/classification/{train,val,test}/<class>/`.
- *Detection*: Ultralytics YOLOv8 fine-tuned via `train_detector.py`, using YOLO-format data at `data/detection/{images,labels}/{train,val}`. `configs/vehicles.yaml` contains an **absolute** `path:` (Ultralytics resolves relative paths against its own datasets_dir) — update it per machine.
- `cnn_numpy.py` and `detection_utils.py` are educational from-scratch NumPy implementations (conv, pooling, IoU, NMS, mAP) used for the report; the production path uses PyTorch/Ultralytics equivalents.

**`backend/` (FastAPI)** — `services.py` inserts `src/` into `sys.path` to import `ml.*`. `VehicleDetector` and `VehicleClassifier` are module-level lazy singletons (loaded on first request, under a lock); inference runs via `run_in_threadpool`. By default the detector is plain COCO `yolov8n.pt` (auto-downloaded) and COCO class ids are mapped to project classes via `COCO_TO_VEHICLE`, dropping all other classes. After fine-tuning, set `DETECTOR_WEIGHTS=<path to best.pt>` and `DETECTOR_IS_COCO=0` so `model.names` are used directly. `/api/detect?refine=true` crops each box (≥24 px) and re-classifies it with the classifier, overriding the label only if the classifier is more confident. `/api/classify` returns 503 until a classifier checkpoint exists (`CLASSIFIER_CKPT`, default `checkpoints/resnet18_best.pt`). Other env settings in `backend/app/config.py`: `DEVICE` (default `cpu`), `MAX_UPLOAD_MB`, `CORS_ORIGINS`.

**`frontend/` (React 18 + Vite)** — `src/api.js` calls `/api/*`; in dev, Vite proxies `/api` to `localhost:8000` (override with `VITE_API_URL`). Response shapes are defined by `backend/app/schemas.py` — keep both sides in sync when changing the API.
