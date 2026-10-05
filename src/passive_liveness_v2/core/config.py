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
    threshold: float = 0.9353
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ---- request limits ----
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB

    def resolve_device(self) -> torch.device:
        """Resolve the ``device`` setting to a concrete torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance (env vars are read once per process)."""
    return Settings()
