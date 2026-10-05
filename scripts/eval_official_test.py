#!/usr/bin/env python3
"""Evaluate the ViT‑B model (pl_checkpoints/vitb_pad_v1.pt) on the official iBeta test set.

Outputs:
  - eval_output/official_test_results.csv  (scores from the model)
  - eval_output/vitb_pad_v1/false_positives/  (annotated images)
  - eval_output/vitb_pad_v1/false_negatives/  (annotated images)

Each annotated image has a bounding box on the face, the liveness score,
and the threshold used.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_MAPPING = {
    "vits": "facebook/dinov2-with-registers-small",
    "vitb": "facebook/dinov2-with-registers-base",
}
TARGET_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

GEOMETRY_COLS = [
    "interocular_dist_norm",
    "eye_to_nose_dist_norm",
    "nose_to_mouth_dist_norm",
    "eye_to_mouth_dist_norm",
    "mouth_width_norm",
    "face_aspect_ratio",
    "pose_asym",
]

# The checkpoint we want to evaluate
CHECKPOINT_PATH = "./pl_checkpoints/vitb_pad_v1.pt"
MODEL_NAME = "vitb_pad_v1"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LivenessModel(nn.Module):
    """Matches the definition used in the training notebook."""

    def __init__(
        self,
        backbone,
        hidden_dim: int,
        n_freq_bins: int = 20,
        use_freq_head: bool = True,
        use_geometry: bool = True,
        head_hidden: int = 128,
        geometry_dim: int = 7,
        use_depth_head: bool = True,
        depth_feature_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_freq_head = use_freq_head
        self.use_geometry = use_geometry
        self.use_depth_head = use_depth_head

        cls_in = hidden_dim
        if use_geometry:
            cls_in += geometry_dim
        if use_freq_head:
            cls_in += n_freq_bins
        if use_depth_head:
            cls_in += depth_feature_dim

        # Classification head – dropout 0.5 as in the notebook
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(head_hidden, 1),
        )
        # Depth head – dropout 0.3 as in the notebook
        if use_depth_head:
            self.depth_head = nn.Sequential(
                nn.Linear(hidden_dim, depth_feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(depth_feature_dim, depth_feature_dim),
            )
        # Frequency head – dropout 0.5 as in the notebook
        if use_freq_head:
            self.freq_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_freq_bins),
            )
        self.num_registers = getattr(backbone.config, "num_register_tokens", 0)

    def forward(
        self,
        pixel_values,
        geometry=None,
        depth_model=None,
        depth_transform=None,
        device=None,
    ):
        out = self.backbone(pixel_values=pixel_values)
        hidden = out.last_hidden_state
        cls_in = hidden[:, 0]
        patches = hidden[:, 1 + self.num_registers :].mean(dim=1)
        if self.use_geometry:
            if geometry is None:
                raise ValueError("use_geometry=True but no geometry tensor")
            cls_in = torch.cat([cls_in, geometry], dim=-1)

        if self.use_freq_head:
            freq = self.freq_head(patches)
            cls_in = torch.cat([cls_in, freq], dim=-1)

        depth_feat = None
        if self.use_depth_head:
            # During evaluation we do **not** compute depth maps (mirrors notebook behavior)
            # The notebook falls back to a zero tensor when depth_model/transform are None.
            depth_feat = torch.zeros(
                patches.shape[0],
                self.depth_head[-1].out_features
                if hasattr(self.depth_head[-1], "out_features")
                else depth_feature_dim,
                device=pixel_values.device,
            )
        cls_in = torch.cat([cls_in, depth_feat], dim=-1)

        logit = self.cls_head(cls_in).squeeze(-1)
        freq = self.freq_head(patches) if self.use_freq_head else None
        return logit, freq, depth_feat


def load_backbone(backbone_id):
    """Load a DinoV2 backbone and unfreeze the last 2 blocks (matches training)."""
    model = AutoModel.from_pretrained(backbone_id)
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze the last 2 blocks
    last = model.config.num_hidden_layers - 1
    for i in range(2):
        layer_idx = last - i
        for name, p in model.named_parameters():
            if f"encoder.layer.{layer_idx}." in name or name.startswith("layernorm"):
                p.requires_grad = True
    return model


def load_model(checkpoint_path: str, model_name: str) -> tuple[LivenessModel, float]:
    """Load checkpoint and instantiate the model with the correct heads."""
    blob = torch.load(checkpoint_path, map_location=DEVICE)
    # The notebook saves the raw state dict (no extra nesting)
    state = (
        blob
        if isinstance(blob, dict) and "model_state" not in blob
        else blob.get("model_state", blob)
    )
    threshold = blob.get("threshold", 0.8) if isinstance(blob, dict) else 0.8

    # Detect which heads are present in the checkpoint
    has_freq_head = any(key.startswith("freq_head.") for key in state.keys())
    has_depth_head = any(key.startswith("depth_head.") for key in state.keys())
    # Geometry is always present in our training config
    use_geometry = True

    backbone_id = BACKBONE_MAPPING["vitb"]  # we know we used vitb
    backbone = load_backbone(backbone_id)

    model = LivenessModel(
        backbone,
        backbone.config.hidden_size,
        use_freq_head=has_freq_head,
        use_geometry=use_geometry,
        head_hidden=128,
        use_depth_head=has_depth_head,
    ).to(DEVICE)
    
    # Convert state dict keys to match the current backbone naming
    converted_state = {}
    for key, value in state.items():
        new_key = key
        if key.startswith('backbone.encoder.layer.'):
            if '.attention.attention.' in key:
                # Handle query/key/value projections: attention.attention.{op} -> attention.{op}_proj
                new_key = key.replace('.attention.attention.', '.attention.')
                parts = new_key.split('.')
                if len(parts) >= 2:
                    op = parts[-2]
                    if op == 'query':
                        parts[-2] = 'q_proj'
                    elif op == 'key':
                        parts[-2] = 'k_proj'
                    elif op == 'value':
                        parts[-2] = 'v_proj'
                    new_key = '.'.join(parts)
            elif '.attention.output.dense' in key:
                # Handle output projection: attention.output.dense -> attention.o_proj
                # Be more explicit to avoid any substring issues
                new_key = key.replace('.attention.output.dense.weight', '.attention.o_proj.weight')
                new_key = new_key.replace('.attention.output.dense.bias', '.attention.o_proj.bias')
        converted_state[new_key] = value
    
    model.load_state_dict(converted_state)
    model.eval()
    return model, float(threshold)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OfficialTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_root: Path):
        self.df = df.reset_index(drop=True)
        self.root = dataset_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self.root / row["frame_path"]
        img = Image.open(img_path).convert("RGB")
        img = transforms.Resize((TARGET_SIZE, TARGET_SIZE))(img)
        tensor = transforms.ToTensor()(img)
        geometry = torch.tensor(
            [0.0 if pd.isna(row.get(c)) else float(row.get(c)) for c in GEOMETRY_COLS],
            dtype=torch.float32,
        )
        return (
            tensor,
            row["binary_label"],
            row["attack_type"],
            row["video_id"],
            geometry,
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(model, loader, use_geometry: bool = True) -> pd.DataFrame:
    """Run model over loader, return DataFrame with score + metadata."""
    model.eval()
    all_scores = []
    all_labels = []
    all_attack_types = []
    all_video_ids = []

    for batch in loader:
        imgs, labels, atk_types, vids, geometry = batch

        imgs = NORMALIZE(imgs.to(DEVICE))
        geometry = geometry.to(DEVICE) if use_geometry else None

        logits, _, _ = model(imgs, geometry, None, None, DEVICE)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_attack_types.extend(atk_types)
        all_video_ids.extend(vids)

    return pd.DataFrame(
        {
            "score": all_scores,
            "binary_label": all_labels,
            "attack_type": all_attack_types,
            "video_id": all_video_ids,
        }
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rate_metrics(scores, labels, threshold) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    live = labels == 1
    predicted_live = scores >= threshold
    apcer = float(predicted_live[~live].mean()) if (~live).any() else float("nan")
    bpcer = float((~predicted_live[live]).mean()) if live.any() else float("nan")
    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": 0.5 * (apcer + bpcer),
        "n_live": int(live.sum()),
        "n_spoof": int((~live).sum()),
    }


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def annotate_image(
    img_path: Path,
    output_path: Path,
    score: float,
    threshold: float,
    is_genuine: bool,
    attack_type: str,
):
    """Draw bbox + score + threshold on image, save to output_path."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Since images are pre‑cropped faces, draw a box around the whole image with a small margin
    w, h = img.size
    margin = 2
    draw.rectangle(
        [margin, margin, w - margin, h - margin],
        outline="red" if is_genuine else "lime",
        width=3,
    )

    # Text annotation
    pred = "live" if score >= threshold else "spoof"
    label = "GENUINE" if is_genuine else f"SPOOF ({attack_type})"
    text = f"score: {score:.4f} | thr: {threshold:.4f} | pred: {pred}"
    sub_text = f"true: {label}"

    # Draw text background
    text_y = 5
    draw.rectangle([2, text_y, w - 2, text_y + 32], fill="black")
    draw.text((5, text_y + 2), text, fill="white")
    draw.text((5, text_y + 16), sub_text, fill="yellow")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the ViT‑B model on the official iBeta test set"
    )
    parser.add_argument(
        "--dataset",
        default="processed_dataset_official_test",
        help="Path to extracted dataset directory",
    )
    parser.add_argument(
        "--zip",
        default="processed_dataset_official_test.zip",
        help="Path to dataset zip (used if --dataset doesn't exist)",
    )
    parser.add_argument(
        "--output", default="eval_output", help="Output directory for results"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom threshold to use for evaluation (overrides model's threshold)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        zip_path = Path(args.zip)
        if zip_path.exists():
            print(f"Extracting {zip_path} -> {dataset_root}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dataset_root)
        else:
            print(f"ERROR: neither {dataset_root} nor {zip_path} found")
            sys.exit(1)

    # Load metadata
    frames_csv = dataset_root / "metadata" / "frames.csv"
    if not frames_csv.exists():
        print(f"ERROR: {frames_csv} not found")
        sys.exit(1)

    df = pd.read_csv(frames_csv)
    df["binary_label"] = (df["label"] == "genuine").astype(int)
    print(
        f"Loaded {len(df)} frames ({df['binary_label'].sum()} genuine, {(1 - df['binary_label']).sum()} spoof)"
    )

    # Build loader
    dataset = OfficialTestDataset(df, dataset_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = df[
        ["frame_path", "video_id", "subject_id", "label", "attack_type", "binary_label"]
    ].copy()

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"{'=' * 60}")

    if not Path(CHECKPOINT_PATH).exists():
        print(f"  ERROR: checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    model, model_threshold = load_model(CHECKPOINT_PATH, MODEL_NAME)
    # Use custom threshold if provided, otherwise use model's threshold
    threshold = args.threshold if args.threshold is not None else model_threshold
    print(f"  Threshold: {threshold:.4f}" + (" (custom)" if args.threshold is not None else ""))

    scored = run_inference(model, loader, use_geometry=True)
    results_df[f"score_{MODEL_NAME}"] = scored["score"]
    results_df[f"pred_{MODEL_NAME}"] = (scored["score"] >= threshold).astype(int)

    # Metrics
    metrics = rate_metrics(
        scored["score"].values, scored["binary_label"].values, threshold
    )
    print(f"  APCER: {metrics['apcer']:.4f}")
    print(f"  BPCER: {metrics['bpcer']:.4f}")
    print(f"  ACER:  {metrics['acer']:.4f}")

    # FP/FN analysis
    scores = scored["score"].values
    labels = scored["binary_label"].values
    preds = (scores >= threshold).astype(int)

    fp_mask = (labels == 0) & (preds == 1)  # spoof predicted as live
    fn_mask = (labels == 1) & (preds == 0)  # genuine predicted as spoof

    print(f"  False Positives: {fp_mask.sum()}")
    print(f"  False Negatives: {fn_mask.sum()}")

    # Write annotated FP/FN images
    fp_dir = output_dir / MODEL_NAME / "false_positives"
    fn_dir = output_dir / MODEL_NAME / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    for idx in np.where(fp_mask)[0]:
        row = df.iloc[idx]
        img_path = dataset_root / row["frame_path"]
        out_path = fp_dir / f"{row['video_id']}_{row['frame_path'].split('/')[-1]}"
        annotate_image(
            img_path,
            out_path,
            scores[idx],
            threshold,
            is_genuine=False,
            attack_type=row["attack_type"],
        )

    for idx in np.where(fn_mask)[0]:
        row = df.iloc[idx]
        img_path = dataset_root / row["frame_path"]
        out_path = fn_dir / f"{row['video_id']}_{row['frame_path'].split('/')[-1]}"
        annotate_image(
            img_path,
            out_path,
            scores[idx],
            threshold,
            is_genuine=True,
            attack_type=row["attack_type"],
        )

    print(f"  FP images -> {fp_dir}")
    print(f"  FN images -> {fn_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write combined results CSV
    results_csv = output_dir / "official_test_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results CSV: {results_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
