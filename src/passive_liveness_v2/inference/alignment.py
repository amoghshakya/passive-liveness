"""5-point similarity-transform face alignment.

Ported from the align cell in ``preprocess.py``. The reference
layout is the ArcFace 112x112 standard; the same transform the
training data was extracted with.
"""

import cv2
import numpy as np

# ArcFace 112x112 reference points:
# left_eye, right_eye, nose, mouth_left, mouth_right
REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    frame_bgr: np.ndarray,
    landmarks_5pt: np.ndarray,
    face_size: int,
    margin_scale: float = 1.0,
    fallback_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Align a face to the ArcFace reference layout.

    ``margin_scale`` < 1.0 shrinks the reference landmark layout
    toward its own centroid, zooming the face out within the same
    face_size canvas -- more surrounding context survives into the
    crop, at the cost of less resolution on the face itself.
    """
    ref = REF_112 * (face_size / 112.0)
    if margin_scale != 1.0:
        center = ref.mean(axis=0)
        ref = center + (ref - center) * margin_scale

    src = landmarks_5pt.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
    if M is not None:
        return cv2.warpAffine(
            frame_bgr,
            M,
            (face_size, face_size),
            borderMode=cv2.BORDER_REPLICATE,
        )

    # degenerate landmarks (rare): fall back to a square bbox crop+resize
    if fallback_bbox is None:
        raise ValueError("alignment failed and no fallback_bbox provided")
    x1, y1, x2, y2 = fallback_bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((face_size, face_size, 3), dtype=frame_bgr.dtype)
    return cv2.resize(crop, (face_size, face_size))


def adaptive_margin_scale(
    face_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    min_scale: float = 0.35,
    max_scale: float = 1.0,
) -> float:
    """Derive the crop margin from the face's share of the frame.

    A fixed margin is a bad fit across distances: a close-up face
    (large bbox) needs more zoom-out to keep context in the crop,
    while a face that is already small in frame is already zoomed
    out. Interpolates from face-area-to-frame-area ratio.
    """
    x1, y1, x2, y2 = face_bbox
    frame_h, frame_w = frame_shape[:2]
    face_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, frame_w * frame_h)
    return float(np.interp(face_area_ratio, [0.05, 0.3], [max_scale, min_scale]))
