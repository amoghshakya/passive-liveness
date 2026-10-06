"""Model loading and the full serving pipeline.

The service owns the model and detector for the lifetime of
the process: both are loaded once during app startup (FastAPI
lifespan), never per request.

Serving pipeline (mirrors the extraction pipeline in
``preprocess.py`` so train/serve stay numerically consistent):

    decode -> detect (RetinaFace) -> primary face -> quality
    gate -> 5-point alignment (adaptive margin) -> geometry
    ratios -> resize 224 -> normalize -> PAD head
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, Dinov2WithRegistersConfig

from passive_liveness_v2.core.config import Settings
from passive_liveness_v2.core.errors import APIError
from passive_liveness_v2.inference.alignment import (
    adaptive_margin_scale,
    align_face,
)
from passive_liveness_v2.inference.detection import (
    FaceDetector,
    FaceInfo,
)
from passive_liveness_v2.inference.geometry import (
    GEOMETRY_DIM,
    compute_geometry,
)
from passive_liveness_v2.inference.model import LivenessModel
from passive_liveness_v2.inference.preprocessing import build_transform
from passive_liveness_v2.inference.quality import (
    REASON_TO_CODE,
    QualityConfig,
    evaluate,
)

log = logging.getLogger(__name__)

# The checkpoint was saved with an older transformers version that used
# `attention.attention.{query,key,value}` and `attention.output.dense`
# naming; transformers >= 5.18 uses `attention.{q,k,v,o}_proj`. Remap
# the old names so the checkpoint loads on the installed version.
_ATTENTION_KEY_REMAP = (
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
)


def _remap_backbone_keys(state: dict) -> dict:
    """Rename legacy attention keys; no-op for newer checkpoints."""
    remapped = {}
    for key, value in state.items():
        for old, new in _ATTENTION_KEY_REMAP:
            if old in key:
                key = key.replace(old, new)
                break
        remapped[key] = value
    return remapped


# Fallback backbone layout, used only if the HF config cannot be
# fetched. Matches facebook/dinov2-with-registers-base, which
# this checkpoint was trained with.
_FALLBACK_CONFIG = dict(
    image_size=518,
    patch_size=14,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    num_register_tokens=4,
)


@dataclass(frozen=True)
class Prediction:
    """A single liveness decision."""

    score: float  # probability of live, in [0, 1]
    label: str  # "live" | "spoof"
    face: FaceInfo | None = None


class LivenessService:
    """Loads the PAD checkpoint once and serves predictions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: LivenessModel | None = None
        self._device: torch.device | None = None
        self._detector: FaceDetector | None = None
        self._quality_cfg = QualityConfig(
            face_confidence_threshold=settings.face_confidence_threshold,
            min_face_size=settings.min_face_size,
            min_face_ratio=settings.min_face_ratio,
            blur_threshold=settings.blur_threshold,
            max_pose_asym=settings.max_pose_asym,
        )
        self._transform = build_transform(settings.target_size)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._detector is not None

    @property
    def device(self) -> torch.device | None:
        return self._device

    def load(self) -> None:
        """Build the model and detector, load the checkpoint, warm up."""
        if self.is_loaded:
            return

        state = torch.load(
            self.settings.checkpoint_path, map_location="cpu", weights_only=True
        )
        state = _remap_backbone_keys(state)
        backbone = self._build_backbone()

        model = LivenessModel(backbone, hidden_dim=backbone.config.hidden_size)
        model.load_state_dict(state, strict=True)

        self._device = self.settings.resolve_device()
        model.to(self._device).eval()
        self._warmup(model)
        self._model = model

        # Prepares InsightFace; downloads the buffalo_l detection
        # weights (~16 MB) once, then caches them locally.
        self._detector = FaceDetector(
            device=self.settings.device,
            detect_max_side=self.settings.detect_max_side,
        )

        log.info(
            "model %s loaded on %s (threshold=%.4f)",
            self.settings.model_name,
            self._device,
            self.settings.threshold,
        )

    def _build_backbone(self) -> torch.nn.Module:
        """Build the DINOv2 backbone architecture (weights come from the checkpoint)."""
        try:
            config = AutoConfig.from_pretrained(self.settings.backbone_model_id)
        except Exception:
            log.warning(
                "could not fetch backbone config for %s; using built-in "
                "ViT-B/14 + registers defaults",
                self.settings.backbone_model_id,
            )
            config = Dinov2WithRegistersConfig(**_FALLBACK_CONFIG)

        return AutoModel.from_config(config)

    def _warmup(self, model: LivenessModel) -> None:
        """Run one dummy forward pass to initialize lazy CUDA buffers etc."""
        with torch.inference_mode():
            model(
                pixel_values=torch.zeros(
                    1, 3, self.settings.target_size, self.settings.target_size,
                    device=self._device,
                ),
                geometry=torch.zeros(
                    1, GEOMETRY_DIM, device=self._device
                ),
            )

    def predict(self, image: Image.Image, skip_detection: bool = False) -> Prediction:
        """Run the full pipeline on a single RGB frame.

        With ``skip_detection`` the detection, quality-gate and
        alignment stages are bypassed; see
        ``_predict_without_detection``.
        """
        if not self.is_loaded or self._device is None:
            raise RuntimeError("model is not loaded")

        if skip_detection:
            return self._predict_without_detection(image)

        # The pipeline operates on BGR frames, matching extraction.
        frame_bgr = np.array(image)[:, :, ::-1]

        det = self._detector.detect_primary(frame_bgr)
        qr = evaluate(frame_bgr, det, self._quality_cfg)
        if not qr.accepted or det is None:
            self._raise_gate_error(qr)

        margin_scale = (
            adaptive_margin_scale(qr.bbox, frame_bgr.shape)
            if self.settings.adaptive_crop_margin
            else self.settings.crop_margin_scale
        )
        aligned_bgr = align_face(
            frame_bgr,
            det.landmarks,
            self.settings.face_size,
            margin_scale=margin_scale,
            fallback_bbox=qr.bbox,
        )

        geometry = compute_geometry(det.landmarks, qr.bbox)
        if geometry is None:
            raise APIError(
                "geometry_extraction_failed",
                "Could not extract face geometry",
                status_code=422,
            )

        # Aligned BGR crop -> RGB PIL -> train/serve transform
        # (resize 224, ImageNet normalize), identical to training.
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        aligned_pil = Image.fromarray(aligned_rgb)
        tensor = self._transform(aligned_pil).unsqueeze(0).to(self._device)
        geometry_tensor = (
            torch.from_numpy(geometry).unsqueeze(0).to(self._device)
        )

        with torch.inference_mode():
            logit = self._model(
                pixel_values=tensor, geometry=geometry_tensor
            )

        score = float(torch.sigmoid(logit).item())
        label = "live" if score >= self.settings.threshold else "spoof"
        return Prediction(
            score=score,
            label=label,
            face=FaceInfo(bbox=qr.bbox, det_score=det.prob),
        )

    def _predict_without_detection(self, image: Image.Image) -> Prediction:
        """Classify a frame without face detection.

        Escape hatch for frames that visibly contain a face but defeat
        the detector: center-crop a square, resize it to the
        aligned-crop scale, and run the head with zeroed geometry
        (the model's own fallback input). The model was trained on
        aligned face crops, so scores on unaligned frames are
        approximate.
        """
        arr = np.array(image)  # RGB
        h, w = arr.shape[:2]
        side = min(h, w)
        if side == 0:
            raise APIError("invalid_image", "Image has zero dimensions")
        y, x = (h - side) // 2, (w - side) // 2
        square = cv2.resize(
            arr[y : y + side, x : x + side],
            (self.settings.face_size, self.settings.face_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = (
            self._transform(Image.fromarray(square))
            .unsqueeze(0)
            .to(self._device)
        )
        geometry_tensor = torch.zeros(1, GEOMETRY_DIM, device=self._device)

        with torch.inference_mode():
            logit = self._model(pixel_values=tensor, geometry=geometry_tensor)

        score = float(torch.sigmoid(logit).item())
        label = "live" if score >= self.settings.threshold else "spoof"
        return Prediction(score=score, label=label, face=None)

    def _raise_gate_error(self, qr) -> None:
        """Translate a quality-gate failure into an API error."""
        code = REASON_TO_CODE.get(qr.reason or "no_face_detected", "no_face_detected")
        messages = {
            "no_face_detected": "No face detected in the frame",
            "invalid_bbox": "Face detection returned an invalid bounding box",
            "low_confidence": "Detected face is below the confidence threshold",
            "face_too_small": "Face is too small in the frame",
            "low_quality": "Face crop is too blurry",
            "extreme_pose": "Face pose is too extreme",
        }
        details = {"reason": qr.reason}
        if qr.det_prob:
            details["det_score"] = round(qr.det_prob, 4)
        if qr.blur is not None:
            details["blur_score"] = round(qr.blur, 2)
        raise APIError(code, messages[code], status_code=422, details=details)
