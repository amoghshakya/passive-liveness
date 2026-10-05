"""Health and readiness routes."""

from fastapi import APIRouter, Depends

from passive_liveness_v2.api.deps import get_liveness_service
from passive_liveness_v2.api.v1.schemas import (
    HealthResponse,
    ModelInfo,
    ReadyResponse,
)
from passive_liveness_v2.core.config import Settings, get_settings
from passive_liveness_v2.inference.service import LivenessService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Cheap liveness probe; does not touch the model."""
    return HealthResponse(status="ok", version=settings.version)


@router.get("/ready", response_model=ReadyResponse, summary="Readiness and model metadata")
def ready(
    service: LivenessService = Depends(get_liveness_service),
    settings: Settings = Depends(get_settings),
) -> ReadyResponse:
    """Returns 503 until the model has finished loading."""
    return ReadyResponse(
        status="ok",
        model=ModelInfo(
            name=settings.model_name,
            backbone=settings.backbone_model_id,
            target_size=settings.target_size,
            threshold=settings.threshold,
            device=str(service.device),
            checkpoint=str(settings.checkpoint_path),
        ),
    )
