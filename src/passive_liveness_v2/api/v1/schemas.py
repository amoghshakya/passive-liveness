"""Pydantic v2 request/response models for the v1 API."""

from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every endpoint."""

    error: dict


class HealthResponse(BaseModel):
    """Service liveness probe (no model check)."""

    status: Literal["ok"]
    version: str


class ModelInfo(BaseModel):
    """Metadata about the loaded model."""

    name: str
    backbone: str
    target_size: int
    threshold: float
    device: str
    checkpoint: str


class ReadyResponse(BaseModel):
    """Readiness probe: the model is loaded and ready to serve."""

    status: Literal["ok"]
    model: ModelInfo


class FaceInfo(BaseModel):
    """Detection details for the face the verdict is based on."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in the uploaded frame
    det_score: float = Field(ge=0.0, le=1.0)


class LivenessResponse(BaseModel):
    """Result of a single-frame liveness check."""

    request_id: str
    model: str
    label: Literal["live", "spoof"]
    score: float = Field(ge=0.0, le=1.0, description="Probability of live")
    threshold: float = Field(description="Operating point used for the label")
    face: FaceInfo | None = None
