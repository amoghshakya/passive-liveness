"""Liveness prediction routes."""

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from passive_liveness_v2.api.deps import get_liveness_service
from passive_liveness_v2.api.v1.schemas import (
    BurstCounts,
    ErrorResponse,
    FrameResult,
    LivenessBurstResponse,
    LivenessResponse,
)
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
        skip_detection=skip_detection,
        face=(
            None
            if prediction.face is None
            else {
                "bbox": prediction.face.bbox,
                "det_score": prediction.face.det_score,
            }
        ),
    )


MAX_BURST_FRAMES = 8


@router.post(
    "/liveness/burst",
    response_model=LivenessBurstResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Classify a burst of frames by majority vote",
)
async def predict_liveness_burst(
    files: list[UploadFile] = File(
        ...,
        description="1-8 frames from the same burst (a ~2s window)",
    ),
    skip_detection: bool = Form(
        False,
        description="Bypass detection, quality gate and alignment "
        "for every frame",
    ),
    request: Request = None,
    service: LivenessService = Depends(get_liveness_service),
) -> LivenessBurstResponse:
    """Classify every frame and fuse the verdicts by majority vote.

    The fused score is the lower median of the frame scores,
    which for any burst size is exactly the strict majority
    vote over the per-frame threshold decisions; even-sized
    ties fail closed (spoof). Frames that fail the quality
    gate are reported per-frame and excluded from the vote;
    if every frame fails, the first failure is raised.
    """
    if not files or len(files) > MAX_BURST_FRAMES:
        raise APIError(
            "invalid_burst",
            f"Send between 1 and {MAX_BURST_FRAMES} frames",
            status_code=422,
        )

    results: list[FrameResult] = []
    first_error: APIError | None = None

    for index, file in enumerate(files):
        try:
            if file.content_type and not file.content_type.startswith(
                "image/"
            ):
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
                    f"File exceeds the "
                    f"{service.settings.max_upload_bytes} byte limit",
                    status_code=413,
                )
            image = await anyio.to_thread.run_sync(decode_image, data)
            prediction = await anyio.to_thread.run_sync(
                service.predict, image, skip_detection
            )
        except APIError as exc:
            if first_error is None:
                first_error = exc
            results.append(FrameResult(index=index, error=exc.code))
            continue

        results.append(
            FrameResult(
                index=index,
                label=prediction.label,
                score=prediction.score,
                face=(
                    None
                    if prediction.face is None
                    else {
                        "bbox": prediction.face.bbox,
                        "det_score": prediction.face.det_score,
                    }
                ),
            )
        )

    scored = [r.score for r in results if r.score is not None]
    if not scored:
        raise first_error or APIError(
            "invalid_burst",
            "No frames could be classified",
            status_code=422,
        )

    ordered = sorted(scored)
    fused = ordered[(len(ordered) - 1) // 2]
    label = "live" if fused >= service.settings.threshold else "spoof"

    return LivenessBurstResponse(
        request_id=request_id_ctx.get() or "",
        model=service.settings.model_name,
        label=label,
        score=fused,
        threshold=service.settings.threshold,
        skip_detection=skip_detection,
        counts=BurstCounts(
            live=sum(1 for r in results if r.label == "live"),
            spoof=sum(1 for r in results if r.label == "spoof"),
        ),
        frames=results,
    )
