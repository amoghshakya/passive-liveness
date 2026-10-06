"""Quality gate: decides whether a frame yields a usable face.

Ported from the quality cell in ``preprocess.py``. Failures here
are the system's non-response path (architecture.md: the
non-response rate must stay at or under 1%).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from passive_liveness_v2.inference.detection import Detection
from passive_liveness_v2.inference.geometry import compute_pose_asym

# reason -> stable API error code
REASON_TO_CODE = {
    "no_face_detected": "no_face_detected",
    "invalid_bbox": "invalid_bbox",
    "low_confidence": "low_confidence",
    "too_small": "face_too_small",
    "blurry": "low_quality",
    "extreme_pose": "extreme_pose",
}


@dataclass(frozen=True)
class QualityConfig:
    face_confidence_threshold: float = 0.5
    min_face_size: int = 48  # px, shorter bbox side in ORIGINAL frame
    min_face_ratio: float = 0.04  # bbox_height / frame_height
    blur_threshold: float = 50.0  # Laplacian variance on the face crop
    max_pose_asym: float | None = None  # None = pose filter disabled


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str | None
    det_prob: float
    bbox: tuple[int, int, int, int]
    face_w: int
    face_h: int
    blur: float | None
    pose_asym: float | None


def blur_score(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian on the face crop; low => blurry."""
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate(
    frame_bgr: np.ndarray, det: Detection | None, cfg: QualityConfig
) -> QualityResult:
    """Gate a single frame. ``det=None`` means no face was found."""
    frame_h, frame_w = frame_bgr.shape[:2]

    if det is None:
        return QualityResult(
            False, "no_face_detected", 0.0, (0, 0, 0, 0), 0, 0, None, None
        )

    x1, y1, x2, y2 = det.box
    x1c, y1c = max(0, round(x1)), max(0, round(y1))
    x2c, y2c = min(frame_w, round(x2)), min(frame_h, round(y2))

    if x2c <= x1c or y2c <= y1c:
        return QualityResult(
            False, "invalid_bbox", det.prob, (x1c, y1c, x2c, y2c), 0, 0, None, None
        )

    bbox = (x1c, y1c, x2c, y2c)
    face_w, face_h = x2c - x1c, y2c - y1c
    blur = blur_score(frame_bgr, bbox)
    pose_asym = compute_pose_asym(det.landmarks)

    if det.prob < cfg.face_confidence_threshold:
        return QualityResult(
            False, "low_confidence", det.prob, bbox, face_w, face_h, blur, pose_asym
        )

    if (
        min(face_w, face_h) < cfg.min_face_size
        or (face_h / frame_h) < cfg.min_face_ratio
    ):
        return QualityResult(
            False, "too_small", det.prob, bbox, face_w, face_h, blur, pose_asym
        )

    if blur < cfg.blur_threshold:
        return QualityResult(
            False, "blurry", det.prob, bbox, face_w, face_h, blur, pose_asym
        )

    if (
        cfg.max_pose_asym is not None
        and pose_asym is not None
        and pose_asym > cfg.max_pose_asym
    ):
        return QualityResult(
            False, "extreme_pose", det.prob, bbox, face_w, face_h, blur, pose_asym
        )

    return QualityResult(True, None, det.prob, bbox, face_w, face_h, blur, pose_asym)
