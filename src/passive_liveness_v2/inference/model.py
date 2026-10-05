"""PAD model: frozen DINOv2 (+registers) backbone with a classification head.

Inference port of the ``LivenessModel`` training cell in ``notebook.py``.

Head input layout (923 features):
    [CLS token (768) | geometry ratios (7) | freq bins (20) | depth features (128)]

Serve-time behaviour matches the notebook's own fallback paths:
- depth branch: the notebook only computes depth maps when a depth model is
  passed in, and otherwise falls back to zeros -- serving always uses the
  zeros fallback (no depth model wired).
- geometry: supplied by the caller; zeros until landmark-ratio extraction is
  ported from ``preprocess.py``. This is a known gap: scores are real but
  approximate until then.
"""

import torch
import torch.nn as nn


class LivenessModel(nn.Module):
    """DINOv2 backbone + frequency/geometry/depth-conditioned PAD head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 768,
        n_freq_bins: int = 20,
        use_geometry: bool = True,
        head_hidden: int = 128,
        geometry_dim: int = 7,
        use_depth_head: bool = True,
        depth_feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_geometry = use_geometry
        self.use_depth_head = use_depth_head
        self.geometry_dim = geometry_dim
        self.depth_feature_dim = depth_feature_dim

        cls_in = hidden_dim
        if use_geometry:
            cls_in += geometry_dim
        cls_in += n_freq_bins
        if use_depth_head:
            cls_in += depth_feature_dim

        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(head_hidden, 1),
        )
        # Auxiliary frequency head: supervised during training, and its output
        # is concatenated into the PAD head input (deployed-FAS trick).
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_freq_bins),
        )
        if use_depth_head:
            self.depth_head = nn.Sequential(
                nn.Linear(hidden_dim, depth_feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(depth_feature_dim, depth_feature_dim),
            )

        self.num_registers = int(getattr(backbone.config, "num_register_tokens", 0) or 0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the live/spoof logit (sigmoid -> probability of live)."""
        out = self.backbone(pixel_values=pixel_values)
        hidden = out.last_hidden_state
        cls_in = hidden[:, 0]
        # Patch tokens follow CLS and the register tokens; average them for
        # the frequency (and, in training, depth) auxiliary views.
        patches = hidden[:, 1 + self.num_registers :].mean(dim=1)

        if self.use_geometry:
            if geometry is None:
                raise ValueError("use_geometry=True but no geometry tensor")
            cls_in = torch.cat([cls_in, geometry], dim=-1)

        freq = self.freq_head(patches)
        cls_in = torch.cat([cls_in, freq], dim=-1)

        if self.use_depth_head:
            # Serve-time fallback (no depth model wired): zeros, as in notebook.py.
            depth_feat = torch.zeros(
                patches.shape[0],
                self.depth_feature_dim,
                device=patches.device,
                dtype=patches.dtype,
            )
            cls_in = torch.cat([cls_in, depth_feat], dim=-1)

        return self.cls_head(cls_in).squeeze(-1)
