"""Application configuration, driven by environment variables.

All settings use the ``LIVENESS_`` env prefix, e.g. ``LIVENESS_THRESHOLD=0.9``.
A ``.env`` file in the working directory is also picked up.
"""

from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the liveness API."""

    model_config = SettingsConfigDict(
        env_prefix="LIVENESS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- application ----
    app_name: str = "passive-liveness-v2"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- model ----
    model_name: str = "vitb_pad_v1"
    checkpoint_path: Path = Path("pl_checkpoints/vitb_pad_v1.pt")
    backbone_model_id: str = "facebook/dinov2-with-registers-base"
    target_size: int = 224
    # Operating point from eval_output/scores.txt (APCER 1.7% @ BPCER 10.2%).
    threshold: float = 0.93
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ---- request limits ----
    # None disables the upload size check entirely.
    max_upload_bytes: int | None = None

    # ---- preprocessing (mirrors QualityConfig in preprocess.py) ----
    face_confidence_threshold: float = 0.5
    min_face_size: int = 48  # px, shorter bbox side in original frame
    min_face_ratio: float = 0.04  # bbox_height / frame_height
    # Laplacian variance on the face crop. Extraction used 40.0 to filter
    # training data; the serving default is lower because webcam/phone frames
    # (smaller faces, JPEG compression) score systematically lower, and a
    # rejected frame is a non-response. Error details include the measured
    # blur_score so the operating point can be tuned without code changes.
    blur_threshold: float = 15.0
    max_pose_asym: float | None = None  # None = pose filter disabled
    face_size: int = 512  # aligned crop side length (not model input size)
    crop_margin_scale: float = 1.0  # <1.0 zooms out (more context)
    adaptive_crop_margin: bool = False  # derive margin from face-to-frame area
    detect_max_side: int = 640  # detector input side length

    def resolve_device(self) -> torch.device:
        """Resolve the ``device`` setting to a concrete torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance (env vars are read once per process)."""
    return Settings()
