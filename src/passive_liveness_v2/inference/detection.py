"""Face detection via InsightFace (RetinaFace, buffalo_l pack,
detection-only module). Ported from the face_detect cell in
``preprocess.py`` so extraction and serving stay numerically
consistent.

InsightFace downloads the buffalo_l detection weights (~16 MB) on
first ``prepare()``; they are cached locally afterwards.
"""

from dataclasses import dataclass

import numpy as np
import torch
from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class Detection:
    """A detected face in ORIGINAL frame coordinates."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    prob: float  # detection confidence
    landmarks: np.ndarray  # (5, 2): left_eye, right_eye, nose, mouth_left, mouth_right


@dataclass(frozen=True)
class FaceInfo:
    """Face details returned alongside a liveness verdict."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame coords
    det_score: float


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FaceDetector:
    """RetinaFace detection with the extraction pipeline's settings."""

    def __init__(
        self,
        device: str = "auto",
        detect_max_side: int = 640,
        det_thresh: float | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.detect_max_side = detect_max_side
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=providers,
        )
        prepare_kwargs = {"det_size": (detect_max_side, detect_max_side)}
        if det_thresh is not None:
            prepare_kwargs["det_thresh"] = det_thresh
        self._app.prepare(
            ctx_id=0 if self.device == "cuda" else -1, **prepare_kwargs
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        faces = self._app.get(frame_bgr)
        return [
            Detection(
                box=tuple(float(v) for v in f.bbox),
                prob=float(f.det_score),
                landmarks=np.asarray(f.kps, dtype=np.float32),
            )
            for f in faces
        ]

    def detect_primary(self, frame_bgr: np.ndarray) -> Detection | None:
        """The primary face: highest confidence x area (as in extraction)."""
        dets = self.detect(frame_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.prob * _area(d.box))
