"""FastAPI application factory and CLI entrypoint."""

import logging
from contextlib import asynccontextmanager

import anyio
import uvicorn
from fastapi import FastAPI

from passive_liveness_v2.api.v1.router import api_router
from passive_liveness_v2.core.config import get_settings
from passive_liveness_v2.core.errors import (
    RequestIdMiddleware,
    register_exception_handlers,
)
from passive_liveness_v2.inference.service import LivenessService

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release it at shutdown."""
    settings = get_settings()
    service = LivenessService(settings)
    await anyio.to_thread.run_sync(service.load)
    app.state.liveness_service = service
    yield
    app.state.liveness_service = None


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()


def run() -> None:
    """Serve the app with uvicorn (``passive-liveness-v2`` console script)."""
    settings = get_settings()
    uvicorn.run(
        "passive_liveness_v2.main:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    run()
