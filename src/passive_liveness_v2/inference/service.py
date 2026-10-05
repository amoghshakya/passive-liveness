"""Model loading and prediction service.

The service owns the model for the lifetime of the process: it is loaded
once during app startup (FastAPI lifespan), never per request.
"""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, Dinov2WithRegistersConfig

from passive_liveness_v2.core.config import Settings
from passive_liveness_v2.inference.model import LivenessModel
from passive_liveness_v2.inference.preprocessing import build_transform

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


class LivenessService:
    """Loads the PAD checkpoint once and serves predictions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: LivenessModel | None = None
        self._device: torch.device | None = None
        self._transform = build_transform(settings.target_size)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> torch.device | None:
        return self._device

    def load(self) -> None:
        """Build the model, load the checkpoint, and warm it up."""
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
                    1, model.geometry_dim, device=self._device
                ),
            )

    def predict(self, image: Image.Image) -> Prediction:
        """Classify a single RGB frame as live or spoof."""
        if not self.is_loaded or self._device is None:
            raise RuntimeError("model is not loaded")

        tensor = self._transform(image).unsqueeze(0).to(self._device)
        # TODO: replace with landmark-ratio extraction (see preprocess.py
        # pipeline); zeros until the extraction is ported.
        geometry = torch.zeros(1, self._model.geometry_dim, device=self._device)

        with torch.inference_mode():
            logit = self._model(pixel_values=tensor, geometry=geometry)

        score = float(torch.sigmoid(logit).item())
        label = "live" if score >= self.settings.threshold else "spoof"
        return Prediction(score=score, label=label)
