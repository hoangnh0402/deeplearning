"""Pydantic schema cho request / response của API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str = Field(..., examples=["car"])
    label_vi: str = Field(..., examples=["Ô tô con"])
    confidence: float = Field(..., ge=0, le=1)
    box: BBox
    refined_by_classifier: bool = False


class DetectResponse(BaseModel):
    width: int
    height: int
    detections: list[Detection]
    counts: dict[str, int]
    inference_ms: float
    model: str


class ClassProb(BaseModel):
    label: str
    label_vi: str
    probability: float


class ClassifyResponse(BaseModel):
    top1: ClassProb
    probabilities: list[ClassProb]
    inference_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    classifier_loaded: bool
    device: str
