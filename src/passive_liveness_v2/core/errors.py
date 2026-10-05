"""Error model, exception handlers, and request-ID middleware.

Every API error is returned as a uniform envelope:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}
"""

import logging
from contextvars import ContextVar
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

# Holds the request ID for the duration of a request, so error handlers can
# include it in the response body without threading it through every call.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class APIError(Exception):
    """An error with a stable machine-readable code and HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def _envelope(
    code: str,
    message: str,
    request_id: str | None,
    details: dict | None = None,
) -> dict:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    if request_id:
        error["request_id"] = request_id
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that render every failure in the error envelope."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, request_id_ctx.get(), exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # exc.errors() carries a non-serializable "ctx" entry; strip it.
        errors = [{k: v for k, v in err.items() if k != "ctx"} for err in exc.errors()]
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error", "Request validation failed", request_id_ctx.get(),
                {"errors": errors},
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail), request_id_ctx.get()),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = request_id_ctx.get()
        log.exception(
            "Unhandled error on %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "Unexpected server error", request_id),
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign each request an ID (incoming X-Request-ID or a fresh UUID) and
    expose it as a response header and via ``request_id_ctx``."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response
