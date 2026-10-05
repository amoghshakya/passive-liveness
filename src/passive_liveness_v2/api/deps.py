"""FastAPI dependency providers."""

from fastapi import Request

from passive_liveness_v2.core.config import Settings, get_settings
from passive_liveness_v2.core.errors import APIError
from passive_liveness_v2.inference.service import LivenessService


def get_liveness_service(request: Request) -> LivenessService:
    """Provide the model service installed by the lifespan handler."""
    service: LivenessService | None = getattr(
        request.app.state, "liveness_service", None
    )
    if service is None or not service.is_loaded:
        raise APIError(
            "service_unavailable", "Model is not loaded", status_code=503
        )
    return service
