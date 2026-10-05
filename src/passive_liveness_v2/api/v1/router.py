"""Versioned API router: mounts all v1 sub-routers under /v1."""

from fastapi import APIRouter

from passive_liveness_v2.api.v1.health import router as health_router
from passive_liveness_v2.api.v1.liveness import router as liveness_router

api_router = APIRouter(prefix="/v1")
api_router.include_router(health_router, tags=["health"])
api_router.include_router(liveness_router, tags=["liveness"])
