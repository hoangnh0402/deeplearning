"""Các thành phần lõi của bài toán PHÁT HIỆN đối tượng, cài đặt tường minh.

Hộp (bounding box) dùng định dạng (x1, y1, x2, y2) theo pixel.

* box_iou         — Intersection over Union: IoU(A, B) = |A ∩ B| / |A ∪ B|
* nms             — Non-Maximum Suppression: loại các hộp trùng lặp
* average_precision / mean_average_precision — độ đo mAP (chuẩn PASCAL VOC / COCO)

Các thư viện như Ultralytics đã có sẵn; module này giúp hiểu rõ cơ chế và
được dùng lại ở backend để hậu xử lý khi cần.
"""
from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa N hộp a và M hộp b → ma trận (N, M)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])        # góc trên-trái của phần giao
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])        # góc dưới-phải của phần giao
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Greedy NMS.

    1. Sắp xếp hộp theo điểm tin cậy giảm dần.
    2. Lấy hộp có điểm cao nhất, giữ lại.
    3. Loại mọi hộp còn lại có IoU với nó > iou_thr.
    4. Lặp lại tới khi hết hộp.
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """NMS độc lập theo từng lớp (xe máy không triệt tiêu ô tô)."""
    keep: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep.extend(idx[nms(boxes[idx], scores[idx], iou_thr)].tolist())
    return sorted(keep, key=lambda i: -scores[i])


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP = diện tích dưới đường Precision–Recall (nội suy all-point)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))   # bao lồi precision
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_per_class(preds: list[dict], gts: list[dict], cls: int, iou_thr: float = 0.5) -> float:
    """preds/gts: danh sách theo ảnh, mỗi phần tử {"boxes": (N,4), "labels": (N,), "scores": (N,)}."""
    records, n_gt = [], 0
    for p, g in zip(preds, gts):
        g_boxes = g["boxes"][g["labels"] == cls]
        n_gt += len(g_boxes)
        matched = np.zeros(len(g_boxes), dtype=bool)
        m = p["labels"] == cls
        p_boxes, p_scores = p["boxes"][m], p["scores"][m]
        for k in np.argsort(-p_scores):
            tp = 0
            if len(g_boxes):
                ious = box_iou(p_boxes[k], g_boxes)[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thr and not matched[j]:
                    matched[j], tp = True, 1
            records.append((p_scores[k], tp))
    if n_gt == 0:
        return float("nan")
    records.sort(key=lambda r: -r[0])
    tps = np.array([r[1] for r in records], dtype=float)
    tp_cum, fp_cum = np.cumsum(tps), np.cumsum(1 - tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return average_precision(recall, precision)


def mean_average_precision(preds, gts, num_classes: int, iou_thrs=(0.5,)) -> float:
    """mAP@0.5 (VOC) hoặc mAP@[.5:.95] (COCO) nếu iou_thrs = np.arange(0.5, 0.96, 0.05)."""
    aps = [ap_per_class(preds, gts, c, t) for t in iou_thrs for c in range(num_classes)]
    aps = [a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else 0.0


if __name__ == "__main__":
    b = np.array([[10, 10, 110, 110], [20, 20, 120, 120], [300, 300, 400, 380]], dtype=float)
    s = np.array([0.9, 0.75, 0.8])
    print("IoU(0,1) =", box_iou(b[0], b[1])[0, 0].round(3))   # ≈ 0.681
    print("NMS giữ:", nms(b, s, 0.5))                          # [0, 2]
