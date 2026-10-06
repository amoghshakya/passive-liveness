"""Liveness prediction routes."""

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from passive_liveness_v2.api.deps import get_liveness_service
from passive_liveness_v2.api.v1.schemas import ErrorResponse, LivenessResponse
from passive_liveness_v2.core.errors import APIError, request_id_ctx
from passive_liveness_v2.inference.preprocessing import decode_image
from passive_liveness_v2.inference.service import LivenessService

router = APIRouter()


@router.post(
    "/liveness",
    response_model=LivenessResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Classify a face frame as live or spoof",
)
async def predict_liveness(
    file: UploadFile = File(..., description="Single face frame (JPEG/PNG)"),
    skip_detection: bool = Form(
        False,
        description="Bypass detection, quality gate and alignment; "
        "classify a center crop directly (approximate scores)",
    ),
    request: Request = None,
    service: LivenessService = Depends(get_liveness_service),
) -> LivenessResponse:
    """Run the PAD model on one uploaded frame.

    The client captures a short burst and uploads the sharpest frame
    (see architecture.md section 3.3); the decision stays server-side.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise APIError(
            "unsupported_media_type",
            f"Expected an image, got {file.content_type!r}",
            status_code=415,
        )

    data = await file.read()
    if not data:
        raise APIError("empty_upload", "Uploaded file is empty")
    if (
        service.settings.max_upload_bytes is not None
        and len(data) > service.settings.max_upload_bytes
    ):
        raise APIError(
            "file_too_large",
            f"File exceeds the {service.settings.max_upload_bytes} byte limit",
            status_code=413,
        )

    # Decoding and inference are blocking; run them off the event loop.
    image = await anyio.to_thread.run_sync(decode_image, data)
    prediction = await anyio.to_thread.run_sync(
        service.predict, image, skip_detection
    )

    return LivenessResponse(
        request_id=request_id_ctx.get() or "",
        model=service.settings.model_name,
        label=prediction.label,
        score=prediction.score,
        threshold=service.settings.threshold,
        face=(
            None
            if prediction.face is None
            else {
                "bbox": prediction.face.bbox,
                "det_score": prediction.face.det_score,
            }
        ),
    )
