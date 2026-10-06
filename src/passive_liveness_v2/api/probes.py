"""Unversioned root probes for infrastructure health checks.

Container orchestrators, load balancers and monitors conventionally
probe ``/health`` and ``/ready`` at the root. These alias the versioned
``/v1`` routes by re-registering the same handlers, so probes don't
404; the ``/v1`` paths remain the canonical API.
"""

from fastapi import APIRouter

from passive_liveness_v2.api.v1.health import health, ready
from passive_liveness_v2.api.v1.schemas import HealthResponse, ReadyResponse

router = APIRouter()

router.add_api_route(
    "/health",
    health,
    methods=["GET"],
    response_model=HealthResponse,
    summary="Service liveness (alias of /v1/health)",
)
router.add_api_route(
    "/ready",
    ready,
    methods=["GET"],
    response_model=ReadyResponse,
    summary="Readiness and model metadata (alias of /v1/ready)",
)
