# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "facenet-pytorch==2.5.3",
#     "huggingface-hub==1.32.0",
#     "matplotlib>=3.9.0",
#     "numpy==2.5.3",
#     "opencv-python==5.0.0.93",
#     "pillow==12.3.0",
#     "python-lsp-ruff==2.3.4",
#     "python-lsp-server==1.15.0",
#     "ruff==0.16.7",
#     "transformers==5.16.1",
#     "tqdm==4.70.1",
#     "websockets==17.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import json
    import os
    import random
    import time
    from dataclasses import asdict, dataclass, field, fields
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Sampler
    from torchvision import transforms
    from transformers import AutoModel
    from collections.abc import Iterator
    from tqdm.auto import tqdm


@app.cell
def _():
    @dataclass(frozen=False)
    class RunConfig:
        """Simplified training configuration with LR scheduling and class weighting"""

        name: str = "simple_training"

        # ---- model -------------------------------------------------------
        backbone: str = "vitb"  # "vits" | "vitb"
        target_size: int = 224
        unfreeze_last_n_blocks: int = (
            2  # Unfreeze last 2 blocks instead of just last block
        )
        use_freq_head: bool = True  # Step 2: auxiliary FFT+DCT band loss
        use_geometry: bool = True  # Step 3: 7 landmark ratios on the CLS token
        head_hidden: int = 128
        lambda_freq: float = 1.0  # Weight for freq branch loss

        # depth (monocular)
        use_depth_head: bool = True
        lambda_depth: float = 0.25  # Weight for depth branch loss

        # ---- optimisation -----------------------------------------------
        epochs: int = 50
        batch_size: int = 32
        lr: float = 5e-5
        weight_decay: float = 5e-4
        backbone_lr_mult: float = 0.1  # backbone learns slower than head
        patience: int = 15
        label_smoothing: float = 0.1
        seed: int = 37

        # ---- Learning Rate Scheduling ------------------------------------
        lr_scheduler_type: str = "cosine"  # Options: "step", "exponential", "cosine"
        lr_decay_factor: float = 0.5  # Factor to reduce LR by
        lr_decay_epochs: int = 10  # epochs between LR reductions for step scheduler

        # ---- Class Weighting for Imbalanced Data -------------------------
        # Calculated from data distribution: ~37% live, 63% spoof
        # Weights = 1 / frequency
        class_weight_live: float = 2.7
        class_weight_spoof: float = 1.6

        # ---- Focal Loss (optional) ------------------------------------
        use_focal_loss: bool = False  # Replace BCE with focal loss
        focal_gamma: float = 2.0  # Focusing parameter
        # Alpha weights derived from class frequencies (live ~37%, spoof ~63%)
        focal_alpha_live: float = 0.63  # Weight for live class
        focal_alpha_spoof: float = 0.37  # Weight for spoof class

        # ---- operating point --------------------------------------------
        max_bpcer: float = 0.14  # target.txt

        # ---- source sampling --------------------------------------------
        source_weights: dict = field(default_factory=lambda: SOURCE_WEIGHTS)
        default_source_weight: float = 0.10

        # ---- Early Stopping Metric --------------------------------------
        # Monitor BPCER instead of ACER to avoid over-prioritizing APCER=0
        monitor_metric_for_early_stop: str = "bpc"  # Options: "acer", "bpc", "loss"

    # ArcFace 5-point reference layout
    REF_112 = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # mouth left
            [70.7299, 92.2041],  # mouth right
        ],
        dtype=np.float32,
    )

    BACKBONE_IDS = {
        "vits": "facebook/dinov2-with-registers-small",
        "vitb": "facebook/dinov2-with-registers-base",
    }

    # Landmark ratios fed to the classifier (Step 3)
    GEOMETRY_COLS = [
        "interocular_dist_norm",
        "eye_to_nose_dist_norm",
        "nose_to_mouth_dist_norm",
        "eye_to_mouth_dist_norm",
        "mouth_width_norm",
        "face_aspect_ratio",
        "pose_asym",
    ]

    # Per-source sampling share within each label
    SOURCE_WEIGHTS = {
        1: {  # genuine
            "fas_ibeta_l1": 0.30,
            "lcc_fasd": 0.25,  # genuine-only pool
            "fas_ibeta_l1_printout": 0.20,
            "fas_ibeta_l1_a8": 0.10,
            "custom_pad_dataset": 0.10,
            "pad2d": 0.02,
            "fas_ibeta_l1_ytplayer": 0.03,
        },
        0: {  # spoof
            "fas_ibeta_l1": 0.45,
            "fas_ibeta_l1_printout": 0.18,
            "fas_ibeta_l1_a8": 0.10,
            "custom_pad_dataset": 0.10,
            "pad2d": 0.03,
        },
    }

    # A per-attack slice smaller than this is reported but flagged
    MIN_SLICE_N = 30

    # Directories
    PROCESSED_DIR = "processed_dataset"
    CHECKPOINT_DIR = "checkpoints"
    DEPTH_DIR = "depth_maps"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={DEVICE}  backbone-backends={list(BACKBONE_IDS)}")
    return (
        BACKBONE_IDS,
        CHECKPOINT_DIR,
        DEVICE,
        GEOMETRY_COLS,
        MIN_SLICE_N,
        PROCESSED_DIR,
        RunConfig,
    )


@app.function
def set_seed(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


@app.cell
def _():
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    mo.output.clear_console()
    return (fs,)


@app.cell
def _(PROCESSED_DIR, fs):
    import zipfile

    def fetch_bundle(filename: str, target_dir: str) -> None:
        """Download `filename` from Drive and extract into `target_dir`.

        Skips the download entirely when the directory already exists, so
        re-running this notebook after a training session costs nothing.
        """
        if os.path.isdir(target_dir):
            print(f"{target_dir} already present, skipping download")
            return
        matches = [
            p for p in fs.glob(f"*{filename}*") if p.rsplit("/", 1)[-1] == filename
        ]
        assert matches, f"{filename} not found on Drive"
        print(f"downloading {matches[0]} ...")
        with fs.open(matches[0], "rb") as rf, open(filename, "wb") as lf:
            lf.write(rf.read())
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(".")
        os.remove(filename)
        print(f"extracted -> {target_dir}")

    # Fetch the development bundle
    fetch_bundle("processed_dataset.zip", PROCESSED_DIR)
    return (fetch_bundle,)


@app.cell
def _(PROCESSED_DIR):
    def _read_frames(root: str) -> pd.DataFrame:
        """Load frames.csv for the DEVELOPMENT bundle, with per-video columns.

        `device` is a property of a video, not a frame, so it is merged in from
        videos.csv rather than duplicated across every frame row.

        Asserts no official-test rows are present.
        """
        meta = os.path.join(root, "metadata")
        videos = pd.read_csv(os.path.join(meta, "videos.csv"))
        frames = pd.read_csv(os.path.join(meta, "frames.csv"))
        if "split_role" in videos.columns:
            is_official = videos["split_role"] == "official_test"
            assert not is_official.any(), (
                "official-test rows leaked into the development bundle"
            )
        if "device" in frames.columns:
            frames = frames.drop(columns=["device"])
        frames = frames.merge(videos[["video_id", "device"]], on="video_id", how="left")
        frames["binary_label"] = (frames["label"] == "genuine").astype(int)
        frames["frame_path"] = frames[["frame_path"]].map(
            lambda p: os.path.join(root, p)
        )
        return frames.reset_index(drop=True)

    def load_fold_data(fold: int = 1):
        """Return (train_df, val_df) for one subject-disjoint fold.

        Val is the held-out fold: ~20% of subjects at 5 folds.
        """
        assert os.path.isdir(PROCESSED_DIR), (
            f"{PROCESSED_DIR} not found -- run preprocess.py first"
        )
        frames = _read_frames(PROCESSED_DIR)
        fold_dir = os.path.join(PROCESSED_DIR, "splits", f"fold_{fold}")
        ids = {
            s: set(pd.read_csv(os.path.join(fold_dir, f"{s}.csv"))["video_id"])
            for s in ("train", "val")
        }
        assert not (ids["train"] & ids["val"]), "train/val subject overlap"

        train = frames[frames["video_id"].isin(ids["train"])].reset_index(drop=True)
        val = frames[frames["video_id"].isin(ids["val"])].reset_index(drop=True)
        print(f"-- fold {fold}: VAL is development signal, not a test result --")
        for name, df in (("train", train), ("val", val)):
            live = int((df["binary_label"] == 1).sum())
            print(
                f"   {name:5s} {len(df):6d} frames | {df['video_id'].nunique():5d} videos"
                f" | {df['subject_id'].nunique():4d} subjects"
                f" | live {live:5d} | spoof {len(df) - live:5d}"
            )
        return train, val

    return (load_fold_data,)


@app.cell
def _():
    class LabelSourceSubjectBalancedSampler(Sampler):
        """Sample label -> source -> subject -> video -> frame, hierarchically.

        Each level is uniform given the one above, so a subject with 20 videos
        does not get 4x the training influence of one with 5.

        Source weights are *overrides*, not an allowlist: a source absent from
        the weights gets `default_source_weight` rather than zero. An earlier
        allowlist version silently gave later-added pools zero probability --
        they appeared in every test set having never been trained on, so their
        rows failed for free (e.g. webcam genuine BPCER 0.375).
        """

        def __init__(
            self,
            frames_df: pd.DataFrame,
            num_samples: int,
            source_weights: dict,
            default_source_weight: float,
            seed: int,
        ) -> None:
            self.df = frames_df.reset_index(drop=True)
            self.num_samples = num_samples
            self.seed = seed
            self._epoch = 0

            # label -> source -> subject -> video -> [row positions]
            self._tree: dict = {}
            for label, lg in self.df.groupby("binary_label"):
                by_source: dict = {}
                for source, sg in lg.groupby("dataset_source"):
                    by_subject: dict = {}
                    for subj, subg in sg.groupby("subject_id"):
                        by_subject[subj] = {
                            vid: vg.index.tolist()
                            for vid, vg in subg.groupby("video_id")
                        }
                    by_source[source] = by_subject
                self._tree[label] = by_source

            self._labels = sorted(self._tree)
            self._source_probs: dict = {}
            for label in self._labels:
                available = list(self._tree[label])
                w = source_weights.get(label, {})
                probs = np.array(
                    [w.get(s, default_source_weight) for s in available],
                    dtype=float,
                )
                if probs.sum() <= 0:  # every weight was 0 -> fall back to uniform
                    probs = np.full(len(available), 1.0 / len(available))
                probs = probs / probs.sum()
                self._source_probs[label] = dict(zip(available, probs))

        def set_epoch(self, epoch: int) -> None:
            self._epoch = epoch

        def __len__(self) -> int:
            return self.num_samples

        def __iter__(self) -> Iterator[int]:
            rng = np.random.default_rng(self.seed + self._epoch)
            for _ in range(self.num_samples):
                label = self._labels[rng.integers(len(self._labels))]
                sources = list(self._tree[label])
                probs = [self._source_probs[label][s] for s in sources]
                source = sources[rng.choice(len(sources), p=probs)]
                subjects = list(self._tree[label][source])
                subject = subjects[rng.integers(len(subjects))]
                videos = self._tree[label][source][subject]
                vids = list(videos)
                rows = videos[vids[rng.integers(len(vids))]]
                yield rows[rng.integers(len(rows))]

    def describe_sampler(sampler: LabelSourceSubjectBalancedSampler) -> None:
        """Print the realized sampling distribution.

        A source present in the data with ~0 probability would only ever be
        seen at test time, which is a bug rather than a weighting choice.
        """
        for label in sampler._labels:
            probs = {
                s: round(p, 3) for s, p in sorted(sampler._source_probs[label].items())
            }
            n_rows = sum(
                len(r)
                for subj in sampler._tree[label].values()
                for vids in subj.values()
                for r in vids.values()
            )
            print(f"  P(source | label={label})  n={n_rows}:  {probs}")
        draws = [
            (
                sampler.df.iloc[i]["binary_label"],
                sampler.df.iloc[i]["dataset_source"],
            )
            for i in list(sampler)[:500]
        ]
        print(
            f"  sampled genuine fraction = {np.mean([l for l, _ in draws]):.3f} (target ~0.5)"
        )
        for label in (0, 1):
            srcs = [s for l, s in draws if l == label]
            if srcs:
                dist = pd.Series(srcs).value_counts(normalize=True).round(3).to_dict()
                print(f"  label={label} realized: {dist}")

    return LabelSourceSubjectBalancedSampler, describe_sampler


@app.cell
def _():
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    TRAIN_AUGMENT = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ]
    )

    # Genuine-only augmentation
    GENUINE_PERSPECTIVE_AUGMENT = transforms.RandomPerspective(
        distortion_scale=0.5, p=0.4
    )

    # Spoof-only augmentation
    SPOOF_SATURATION_AUGMENT = transforms.RandomApply(
        [transforms.ColorJitter(saturation=(1.2, 1.7))], p=0.4
    )
    return (
        GENUINE_PERSPECTIVE_AUGMENT,
        NORMALIZE,
        SPOOF_SATURATION_AUGMENT,
        TRAIN_AUGMENT,
    )


@app.cell
def _(
    GENUINE_PERSPECTIVE_AUGMENT,
    GEOMETRY_COLS,
    SPOOF_SATURATION_AUGMENT,
    TRAIN_AUGMENT,
):
    class PADFrameDataset(Dataset):
        """One already-extracted, already-aligned face crop per row.

        Returns an *unnormalized* [0,1] tensor. Normalization happens in the
        eval path, not here, so the frequency target is computed on clean
        pixels and train/eval cannot drift apart on the normalize step.
        """

        def __init__(self, df: pd.DataFrame, augment: bool, target_size: int):
            self.df = df.reset_index(drop=True)
            self.augment = augment
            self.target_size = target_size

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, i: int):
            row = self.df.iloc[i]
            img = Image.open(row["frame_path"]).convert("RGB")
            img = transforms.Resize((self.target_size, self.target_size))(img)
            if self.augment:
                img = TRAIN_AUGMENT(img)
                img = (
                    GENUINE_PERSPECTIVE_AUGMENT(img)
                    if row["binary_label"] == 1
                    else SPOOF_SATURATION_AUGMENT(img)
                )
            tensor = transforms.ToTensor()(img)
            # NaN geometry (detector miss at extraction) becomes 0.0, a
            # plausible-looking landmark ratio. Flagged as a known gap rather
            # than dropped: dropping a frame loses the sample entirely.
            geometry = torch.tensor(
                [
                    0.0 if pd.isna(row.get(c)) else float(row.get(c))
                    for c in GEOMETRY_COLS
                ],
                dtype=torch.float32,
            )
            return (
                tensor,
                row["binary_label"],
                row["attack_type"],
                row["device"],
                geometry,
                row.get("pai_family") or "unknown",
            )

    return (PADFrameDataset,)


@app.cell
def _(
    LabelSourceSubjectBalancedSampler,
    PADFrameDataset,
    RunConfig,
    describe_sampler,
):
    def make_loaders(train: pd.DataFrame, val: pd.DataFrame, cfg: RunConfig):
        """Build the train (sampled) and val (sequential) loaders."""
        train_ds = PADFrameDataset(train, augment=True, target_size=cfg.target_size)
        val_ds = PADFrameDataset(val, augment=False, target_size=cfg.target_size)
        sampler = LabelSourceSubjectBalancedSampler(
            train,
            num_samples=len(train),
            source_weights=cfg.source_weights,
            default_source_weight=cfg.default_source_weight,
            seed=cfg.seed,
        )
        describe_sampler(sampler)
        return (
            DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
            ),
            DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            ),
            sampler,
        )

    return (make_loaders,)


@app.cell
def _(BACKBONE_IDS, DEVICE, RunConfig):
    def load_backbone(key: str, cfg: RunConfig):
        model = AutoModel.from_pretrained(BACKBONE_IDS[key]).to(DEVICE)
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze the last N blocks (default to 2 for improved capacity)
        n_blocks_to_unfreeze = getattr(cfg, "unfreeze_last_n_blocks", 2)

        if n_blocks_to_unfreeze > 0:
            last = model.config.num_hidden_layers - 1
            for i in range(n_blocks_to_unfreeze):
                layer_idx = last - i
                for name, p in model.named_parameters():
                    if f"encoder.layer.{layer_idx}." in name or name.startswith(
                        "layernorm"
                    ):
                        p.requires_grad = True

        print(f"Unfrozen last {n_blocks_to_unfreeze} backbone blocks for training")
        return model

    return (load_backbone,)


@app.function
def compute_freq_bands(imgs: torch.Tensor, n_fft_bins=12, n_dct_bins=8, fft_size=64):
    """Frequency target: ring-binned FFT energy + zigzag-grouped DCT energy.

    FFT rings capture global periodicities (moiré, halftone); DCT blocks
    capture localized compression and texture artefacts. Standardized
    per-image, so the target encodes *relative* band balance rather than
    absolute contrast.
    """
    B = imgs.shape[0]
    gray = torch.nn.functional.interpolate(
        imgs.mean(dim=1, keepdim=True),
        size=(fft_size,) * 2,
        mode="bilinear",
        align_corners=False,
    )
    fft = torch.fft.fftshift(torch.fft.fft2(gray.squeeze(1)))
    magnitude = torch.log1p(fft.abs())

    yy, xx = torch.meshgrid(
        torch.arange(fft_size, device=imgs.device),
        torch.arange(fft_size, device=imgs.device),
        indexing="ij",
    )
    radius = torch.sqrt((yy - fft_size / 2) ** 2 + (xx - fft_size / 2) ** 2)
    bin_idx = (radius / radius.max() * (n_fft_bins - 1)).long().clamp(0, n_fft_bins - 1)
    fft_bands = torch.stack(
        [
            magnitude[:, bin_idx == b].mean(dim=1)
            for b in range(n_fft_bins)
            if (bin_idx == b).sum() > 0
        ],
        dim=1,
    )

    # 8x8 DCT-II over non-overlapping blocks, grouped by zigzag order so
    # low/mid/high frequency energy is pooled the way JPEG scans it.
    blocks = gray.unfold(2, 8, 8).unfold(3, 8, 8).contiguous().view(B, -1, 8, 8)
    n = torch.arange(8, device=imgs.device, dtype=torch.float32)
    basis = torch.cos(torch.pi * n.unsqueeze(1) * (n.unsqueeze(0) + 0.5) / 8)
    dct = torch.einsum("bijk,kl->bijl", blocks, basis)
    dct = torch.einsum("bijk,kl->bijl", dct.transpose(-2, -1), basis).transpose(-2, -1)
    flat = torch.log1p(dct.abs()).reshape(B, -1)

    zigzag = torch.tensor(
        [
            0,
            1,
            8,
            16,
            9,
            2,
            3,
            10,
            17,
            24,
            32,
            25,
            18,
            11,
            4,
            5,
            12,
            19,
            26,
            33,
            40,
            41,
            34,
            27,
            20,
            13,
            6,
            7,
            14,
            21,
            28,
            35,
            42,
            49,
            56,
            57,
            50,
            43,
            36,
            29,
            22,
            15,
            23,
            30,
            37,
            44,
            51,
            58,
            59,
            52,
            45,
            38,
            31,
            39,
            46,
            53,
            60,
            61,
            54,
            47,
            55,
            62,
            63,
        ],
        device=imgs.device,
    )
    group = 64 // n_dct_bins
    dct_bands = torch.stack(
        [
            flat[:, zigzag[b * group : (b + 1) * group]].mean(dim=1)
            for b in range(n_dct_bins)
        ],
        dim=1,
    )

    bands = torch.cat([fft_bands, dct_bands], dim=1)
    return (bands - bands.mean(1, keepdim=True)) / (bands.std(1, keepdim=True) + 1e-6)


@app.cell
def _(DEVICE):
    def compute_depth_map(
        imgs: torch.Tensor,
        model=None,
        transform=None,
        device=DEVICE,
    ):
        """Compute depth map using a pre-trained depth estimation model.

        Args:
            imgs: Input images tensor of shape (B, 3, H, W)
            model: Pre-trained depth estimation model (DPT or similar)
            transform: Preprocessing transform for the depth model
            device: Device to run computation on

        Returns:
            Depth maps tensor of shape (B, 1, H, W) normalized to [0, 1]
        """
        if model is None or transform is None:
            return torch.zeros_like(imgs[:, :1, :, :])

        model.eval()
        with torch.no_grad():
            # Preprocess images for depth model
            # DepthAnything expects list of PIL images or numpy arrays
            # But our transform handles the conversion
            inputs = transform(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # Get depth prediction
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                outputs = model(pixel_values)
                # DepthAnything outputs predicted_depth directly
                depth = outputs.predicted_depth

            # Ensure depth is in [B, 1, H, W] format
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            elif depth.dim() == 4 and depth.shape[1] != 1:
                depth = depth[:, :1, :, :]

            # Resize to match input if needed
            if depth.shape[-2:] != imgs.shape[-2:]:
                depth = torch.nn.functional.interpolate(
                    depth,
                    size=imgs.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Normalize to [0, 1] per image (same as before)
            depth_min = depth.amin(dim=(1, 2, 3), keepdim=True)
            depth_max = depth.amax(dim=(1, 2, 3), keepdim=True)
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

            return depth

    return (compute_depth_map,)


@app.cell
def LivenessModel(DEVICE, compute_depth_map):
    class LivenessModel(nn.Module):
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

            self.cls_head = nn.Sequential(
                nn.Linear(cls_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(head_hidden, 1),
            )
            if use_depth_head:
                self.depth_head = nn.Sequential(
                    nn.Linear(hidden_dim, depth_feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(depth_feature_dim, depth_feature_dim),
                )
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
            device=DEVICE,
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
                # compute depth maps if model is provided
                if depth_model is not None and depth_transform is not None:
                    depth_maps = compute_depth_map(
                        pixel_values, depth_model, depth_transform, device
                    )
                    depth_feat = self.depth_head(patches)
                else:
                    # fallback: use zero tensor if depth models not available
                    depth_feat = torch.zeros(
                        patches.shape[0],
                        self.depth_head[-1].out_features
                        if hasattr(self.depth_head[-1], "out_features")
                        else 128,
                        device=patches.device,
                    )
                cls_in = torch.cat([cls_in, depth_feat], dim=-1)

            logit = self.cls_head(cls_in).squeeze(-1)
            freq = self.freq_head(patches) if self.use_freq_head else None
            return logit, freq, depth_feat

    return (LivenessModel,)


@app.function
def find_best_iberta_threshold(scores, labels, max_bpcer: float) -> tuple[float, str]:
    """Find threshold optimized for iBeta requirements:
    - Minimize APCER subject to BPCER ≤ max_bpcer
    - If impossible, find threshold with BPCER as close to max_bpcer as possible
    - Falls back to balancing APCER and BPCER if needed

    Returns (threshold, note).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    genuine, spoof = scores[labels == 1], scores[labels == 0]
    if len(genuine) == 0 or len(spoof) == 0:
        return 0.5, "empty class in val; threshold defaulted to 0.5"

    # Sort thresholds from low to high
    thresholds = np.unique(scores)
    thresholds = np.concatenate(
        ([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6])
    )

    best_apcer = float("inf")
    best_threshold = 0.5
    best_bpcer_at_threshold = 0.0

    # Track if we can achieve BPCER <= max_bpcer
    feasible_thresholds = []

    for t in thresholds:
        # Calculate metrics at this threshold
        bpcer = float(
            (genuine < t).mean()
        )  # proportion of genuine below threshold (false rejects)
        apcer = float(
            (spoof >= t).mean()
        )  # proportion of spoof above threshold (false accepts)

        # Check if this threshold satisfies BPCER constraint
        if bpcer <= max_bpcer:
            feasible_thresholds.append((t, apcer, bpcer))
            # Among feasible thresholds, pick the one with minimum APCER
            if apcer < best_apcer:
                best_apcer = apcer
                best_threshold = t
                best_bpcer_at_threshold = bpcer

    if feasible_thresholds:
        # We found thresholds that satisfy BPCER ≤ max_bpcer
        # Return the one with best (lowest) APCER
        note = (
            f"Optimized for BPCER ≤ {max_bpcer:.2f}: achieved APCER {best_apcer:.4f}"
            f" at BPCER {best_bpcer_at_threshold:.4f}"
        )
        return best_threshold, note
    else:
        # Cannot satisfy BPCER ≤ max_bpcer, find threshold with BPCER closest to max_bpcer
        # This minimizes the violation of the BPCER constraint
        best_bpcer_diff = float("inf")
        best_threshold = 0.5
        best_apcer_at_threshold = 0.0
        best_bpcer_at_threshold = 0.0

        for t in thresholds:
            bpcer = float((genuine < t).mean())
            apcer = float((spoof >= t).mean())
            bpcer_diff = abs(bpcer - max_bpcer)

            if bpcer_diff < best_bpcer_diff:
                best_bpcer_diff = bpcer_diff
                best_threshold = t
                best_apcer_at_threshold = apcer
                best_bpcer_at_threshold = bpcer

        note = (
            f"Cannot achieve BPCER ≤ {max_bpcer:.2f}; "
            f"best effort: APCER {best_apcer_at_threshold:.4f} at BPCER {best_bpcer_at_threshold:.4f}"
        )
        return best_threshold, note


@app.function
def find_best_acer_threshold(scores, labels, max_bpcer: float) -> tuple[float, str]:
    """Pick the operating threshold. Returns (threshold, note).

    Objective, in priority order, targeting 0% APCER:
      1. Among thresholds with zero val false-accepts, take the lowest BPCER.
      2. If none reaches zero APCER, fall back to min-ACER under the BPCER cap
         and say so loudly — that warning is the most decision-relevant signal
         available *before* touching the test set.
      3. If even the cap is unreachable, fall back to unconstrained min-ACER.

    Why not just min-ACER: ACER weights the two errors equally and will trade
    one point of APCER for one point of BPCER, which is exactly the trade a 0%
    APCER target cannot afford. Note this gives APCER unbounded priority, so
    BPCER can be arbitrarily bad — which is why the note is returned rather
    than swallowed.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    genuine, spoof = scores[labels == 1], scores[labels == 0]
    if len(genuine) == 0 or len(spoof) == 0:
        return 0.5, "empty class in val; threshold defaulted to 0.5"

    uniq = np.unique(scores)
    candidates = np.concatenate(
        ([uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2.0, [uniq[-1] + 1e-6])
    )

    zero_apcer = []
    best = (0.5, np.inf)
    feasible = (None, np.inf)
    for t in candidates:
        bpcer = float((genuine < t).mean())
        apcer = float((spoof >= t).mean())
        acer = 0.5 * (apcer + bpcer)
        if apcer == 0.0:
            zero_apcer.append((bpcer, float(t)))
        if acer < best[1]:
            best = (float(t), acer)
        if bpcer <= max_bpcer and acer < feasible[1]:
            feasible = (float(t), acer)

    if zero_apcer:
        zero_apcer.sort(key=lambda p: (p[0], -p[1]))  # ties -> wider margin
        bpcer, threshold = zero_apcer[0]
        note = (
            f"0% val APCER costs BPCER {bpcer:.4f} (cap {max_bpcer:.2f}), "
            f"n_spoof={len(spoof)}"
        )
        if bpcer > max_bpcer:
            note += " -- OUT OF SPEC on BPCER, kept because 0% APCER is primary"
        return threshold, note

    note = (
        f"0% val APCER UNREACHABLE over {len(spoof)} spoof val frames; "
        f"fell back to min-ACER under BPCER <= {max_bpcer:.2f}"
    )
    return (feasible[0] if feasible[0] is not None else best[0], note)


@app.cell
def _(MIN_SLICE_N):
    def rate_metrics(scores, labels, threshold) -> dict:
        """APCER / BPCER / ACER at one threshold. score >= threshold -> live."""
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

    def per_attack_apcer(
        scores, labels, attack_types, threshold, live_types, min_n
    ) -> pd.DataFrame:
        """APCER broken down by attack type, with honest sample sizes.

        Slices below `min_n` are kept but flagged: one sample moves such a
        slice by 1/n, so quoting it unqualified invites reading 1 sample as 1%.
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels)
        attack_types = np.asarray(attack_types)
        accepted = scores >= threshold
        rows = []
        for atk in sorted(set(attack_types)):
            if atk in live_types:
                continue
            mask = (attack_types == atk) & (labels == 0)
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "attack_type": atk,
                    "n": n,
                    "n_accepted": int(accepted[mask].sum()),
                    "apcer": float(accepted[mask].mean()),
                    "low_evidence": n < min_n,
                }
            )
        return pd.DataFrame(rows)

    def to_video_level(df: pd.DataFrame, scores) -> pd.DataFrame:
        """Average scores within a video.

        Frames inside one video are highly correlated, so a frame-level count
        of n=24 is really 1-2 videos. Video-level counting also matches how
        iBeta-style reports tally trials.
        """
        d = df.copy()
        d["score"] = np.asarray(scores)
        return (
            d.groupby("video_id", sort=False)
            .agg(
                binary_label=("binary_label", "first"),
                attack_type=("attack_type", "first"),
                score=("score", "mean"),
            )
            .reset_index()
        )

    def video_level_metrics(
        scores,
        labels,
        attack_types,
        video_ids,
        threshold,
        agg="median",
        min_n=MIN_SLICE_N,
    ):
        """Aggregate frame scores to video level and compute metrics.

        `agg` can be "mean", "median", or "fraction_live" (majority
        vote: fraction of frames in the video with score >= threshold).
        Returns (metrics_dict, per_attack_df).
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels)
        attack_types = np.asarray(attack_types)
        video_ids = np.asarray(video_ids)
        df = pd.DataFrame(
            {
                "video_id": video_ids,
                "score": scores,
                "binary_label": labels,
                "attack_type": attack_types,
            }
        )
        if agg == "fraction_live":
            grouped = df.groupby("video_id").agg(
                binary_label=("binary_label", "first"),
                attack_type=("attack_type", "first"),
                score=("score", lambda s: (np.asarray(s) >= threshold).mean()),
            )
        else:
            grouped = df.groupby("video_id").agg(
                binary_label=("binary_label", "first"),
                attack_type=("attack_type", "first"),
                score=("score", agg),
            )
        grouped = grouped.reset_index()
        live = grouped["binary_label"] == 1
        predicted_live = grouped["score"] >= threshold
        apcer = float(predicted_live[~live].mean()) if (~live).any() else float("nan")
        bpcer = float((~predicted_live[live]).mean()) if live.any() else float("nan")
        metrics = {
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": 0.5 * (apcer + bpcer),
            "n_videos_live": int(live.sum()),
            "n_videos_spoof": int((~live).sum()),
        }
        live_types = set(grouped.loc[live, "attack_type"])
        per_attack = per_attack_apcer(
            grouped["score"].values,
            grouped["binary_label"].values,
            grouped["attack_type"].values,
            threshold,
            live_types,
            min_n,
        )
        return metrics, per_attack

    return per_attack_apcer, rate_metrics, video_level_metrics


@app.cell
def _(DEVICE, NORMALIZE):
    @torch.no_grad()
    def score_loader(
        model,
        loader,
        use_geometry: bool,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run the model over a loader; return one row per item.

        Returns a DataFrame rather than a tuple of parallel lists: the lists
        had to be kept in lockstep across call sites, and a mismatch there
        would silently mislabel a whole metric.

        Pass `df` (the same frame the loader was built from, same order) to
        carry the identity + diagnostic columns through: `video_id`,
        `subject_id`, `dataset_source`, `frame_path`, and (when present)
        `detector_confidence`. Those make
        the cached CSV self-describing, and doing it here means the caller
        never merges a scored frame onto a metadata frame -- which is where a
        column-name collision (pai_family present in both) previously raised
        *after* a full training run.

        `detector_confidence` is carried deliberately: `bpcer_by_detector_tier`
        splits genuine rows around the 0.4 serve-time floor, and without it
        that breakdown silently returns nothing.
        """
        model.eval()
        rows = []
        for imgs, labels, atk, dev, geometry, fam in loader:
            imgs = NORMALIZE(imgs.to(DEVICE))
            geometry = geometry.to(DEVICE) if use_geometry else None
            logits, _, _depth_feat = model(imgs, geometry)
            scores = torch.sigmoid(logits).cpu().numpy()
            for s, l, a, d, f in zip(scores, labels.tolist(), atk, dev, fam):
                rows.append(
                    {
                        "score": float(s),
                        "binary_label": int(l),
                        "attack_type": a,
                        "device": d,
                        "pai_family": f,
                    }
                )
        out = pd.DataFrame(rows)
        if df is not None:
            if len(df) != len(out):
                raise ValueError(
                    f"score_loader: df has {len(df)} rows but the loader yielded "
                    f"{len(out)} -- the cached scores would be misaligned"
                )
            # frame_path is carried because the failure-inspection cards in
            # section 9 render `mo.image(row["frame_path"])` on these rows. It
            # was dropped here, so that cell raised KeyError as soon as the
            # official test produced a single false accept -- i.e. exactly
            # when it mattered most.
            carry = ["video_id", "subject_id", "dataset_source", "frame_path"]
            for _opt in ("detector_confidence",):
                if _opt in df.columns:
                    carry.append(_opt)
            out = pd.concat(
                [df[carry].reset_index(drop=True), out],
                axis=1,
            )
        return out

    @torch.no_grad()
    def score_loader_ensemble(
        models,
        loader,
        use_geometry: bool,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run an ensemble of models over a loader; average logits.

        Returns a DataFrame with the same schema as `score_loader`,
        but `score` is the sigmoid of the mean logit across models.
        """
        for m in models:
            m.eval()
        rows = []
        for imgs, labels, atk, dev, geometry, fam in loader:
            imgs = NORMALIZE(imgs.to(DEVICE))
            geometry = geometry.to(DEVICE) if use_geometry else None
            logits_list = []
            for model in models:
                logits, _, _ = model(imgs, geometry)
                logits_list.append(logits)
            mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            scores = torch.sigmoid(mean_logits).cpu().numpy()
            for s, l, a, d, f in zip(scores, labels.tolist(), atk, dev, fam):
                rows.append(
                    {
                        "score": float(s),
                        "binary_label": int(l),
                        "attack_type": a,
                        "device": d,
                        "pai_family": f,
                    }
                )
        out = pd.DataFrame(rows)
        if df is not None:
            if len(df) != len(out):
                raise ValueError(
                    f"score_loader_ensemble: df has {len(df)} rows but the loader yielded "
                    f"{len(out)} -- the cached scores would be misaligned"
                )
            carry = ["video_id", "subject_id", "dataset_source", "frame_path"]
            for _opt in ("detector_confidence",):
                if _opt in df.columns:
                    carry.append(_opt)
            out = pd.concat(
                [df[carry].reset_index(drop=True), out],
                axis=1,
            )
        return out

    return (score_loader, score_loader_ensemble)


@app.cell
def _(rate_metrics):
    def roc_table(scores, labels, thresholds=None) -> pd.DataFrame:
        """Full ROC / DET curve from cached scores.

        This is the point of persisting per-frame scores: the curve, the EER,
        and APCER at a BPCER cap are arithmetic on a file, not a GPU run. It is
        the measurement that says whether a recipe is capacity-bound or
        threshold-bound — which the single operating point cannot.
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels)
        genuine, spoof = scores[labels == 1], scores[labels == 0]
        if len(genuine) == 0 or len(spoof) == 0:
            return pd.DataFrame()
        if thresholds is None:
            thresholds = np.unique(scores)
            thresholds = np.concatenate(
                ([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6])
            )
        rows = []
        for t in thresholds:
            m = rate_metrics(scores, labels, float(t))
            rows.append({"threshold": float(t), **m})
        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    def summarize_roc(curve: pd.DataFrame, max_bpcer: float) -> dict:
        """Headline numbers off the ROC: EER, and APCER at a BPCER<=cap point.

        The APCER@cap row is the one that matters for target.txt: it is the
        best false-accept rate achievable while staying inside the BPCER
        budget, which is the trade the single chosen operating point hides.
        """
        if curve.empty:
            return {}
        feasible = curve[curve["bpcer"] <= max_bpcer]
        out = {
            "eer": float(curve["acer"].min()),
            "apcer_at_bpcer_cap": float(feasible["apcer"].min())
            if not feasible.empty
            else float("nan"),
            "bpcer_at_apcer_zero": float(curve[curve["apcer"] == 0]["bpcer"].min())
            if (curve["apcer"] == 0).any()
            else float("nan"),
        }
        return out

    return roc_table, summarize_roc


@app.cell
def _(DEVICE, NORMALIZE, rate_metrics):
    def focal_loss(logits, targets, labels, cfg):
        """Focal loss with per-class alpha weighting."""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        alpha = torch.where(
            labels > 0.5,
            torch.full_like(logits, cfg.focal_alpha_live),
            torch.full_like(logits, cfg.focal_alpha_spoof),
        )
        loss = alpha * (1 - pt) ** cfg.focal_gamma * bce_loss
        return loss.mean()

    def train_model(
        model,
        train_loader,
        val_loader,
        val_df,
        cfg,
        find_best_acer_threshold,
        score_loader,
        use_geometry,
        ckpt_path=None,
        depth_model=None,
        depth_transform=None,
    ) -> tuple:
        """Train one fold. Returns (model, history DataFrame).

        Selection metric is val ACER at the chosen operating threshold, not
        val BCE: BCE bottoms out within 2-3 epochs and then rises while the
        decision boundary can still improve, so selecting on BCE discarded
        every later checkpoint. The threshold is recomputed per epoch from the
        current model, so `val_acer` tracks the operating point rather than a
        fixed cut.
        """
        set_seed(cfg.seed)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.cls_head.parameters())
        if model.use_freq_head:
            head_params += list(model.freq_head.parameters())
        print(
            f"  training {sum(p.numel() for p in backbone_params + head_params):,} params"
            f"  ({sum(p.numel() for p in backbone_params):,} backbone @ lr={cfg.lr * cfg.backbone_lr_mult:.2e},"
            f" {sum(p.numel() for p in head_params):,} head @ lr={cfg.lr:.2e})"
        )
        # Create parameter groups with different learning rates
        param_groups = [
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
            {
                "params": head_params,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
        ]

        opt = torch.optim.Adam(param_groups)

        # Learning rate scheduler
        scheduler = None
        if cfg.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.lr_decay_epochs, gamma=cfg.lr_decay_factor
            )
        elif cfg.lr_scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=cfg.lr_decay_factor
            )
        elif cfg.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
            )
        bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
        loss_note = ""

        warmup_epochs = 5

        best = {"aper": np.inf, "state": None, "epoch": -1}
        history = []
        no_improve = 0

        # # Calculate class weights for balanced loss
        # # These weights help with class imbalance in the dataset
        # class_weights = torch.tensor(
        #     [
        #         cfg.class_weight_spoof,  # weight for class 0 (spoof)
        #         cfg.class_weight_live,  # weight for class 1 (live)
        #     ],
        #     device=DEVICE,
        # )

        for epoch in range(cfg.epochs):
            model.train()
            total = 0.0

            if epoch < warmup_epochs:
                warmup_lr = cfg.lr * (epoch + 1) / warmup_epochs
                for param_group in opt.param_groups:
                    base_lr = warmup_lr
                    if "backbone" in param_group.get("name", ""):
                        param_group["lr"] = base_lr * cfg.backbone_lr_mult
                    else:
                        param_group["lr"] = base_lr

            for imgs, labels, _atk, _dev, geometry, _fam in train_loader:
                raw = imgs.to(DEVICE)
                labels = labels.float().to(DEVICE)
                geometry = geometry.to(DEVICE) if use_geometry else None
                opt.zero_grad()
                logits, freq_pred, depth_feat = model(
                    NORMALIZE(raw),
                    geometry,
                    depth_model,
                    depth_transform,
                    DEVICE,
                )

                # Apply label smoothing and class weights
                # Label smoothing: target = label * (1 - smoothing) + 0.5 * smoothing
                target = labels * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

                # Classification loss: focal loss (or BCE fallback)
                if cfg.use_focal_loss:
                    loss = focal_loss(logits, target, labels, cfg)
                else:
                    loss = bce(logits, target)

                # Apply class weights manually since BCEWithLogitsLoss doesn't directly support per-sample weights
                # We'll weight the loss per sample based on class
                # if hasattr(cfg, "class_weight_live") and hasattr(
                #     cfg, "class_weight_spoof"
                # ):
                #     # Create weight tensor for each sample
                #     weights_per_sample = torch.where(
                #         labels == 1,
                #         torch.tensor(cfg.class_weight_live, device=DEVICE),
                #         torch.tensor(cfg.class_weight_spoof, device=DEVICE),
                #     )
                #     # Apply weights to loss
                #     loss = (loss * weights_per_sample).mean()

                if model.use_freq_head:
                    loss = loss + cfg.lambda_freq * mse(
                        freq_pred, compute_freq_bands(raw)
                    )

                depth_loss = torch.tensor(0.0, device=DEVICE)
                if model.use_depth_head and depth_model is not None:
                    # Encourage smooth depth maps for live faces, varied for spoof
                    depth_var = torch.var(depth_feat, dim=1)
                    depth_target = (
                        labels.float()
                    )  # Live=1 -> target=1, Spoof=0 -> target=0
                    depth_loss = mse(depth_var, depth_target) * cfg.lambda_depth

                loss = loss + depth_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item()

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            #     current_lr = opt.param_groups[0]["lr"]
            #     current_backbone_lr = opt.param_groups[0]["lr"]
            #     current_head_lr = (
            #         opt.param_groups[1]["lr"]
            #         if len(opt.param_groups) > 1
            #         else current_lr
            #     )
            # else:
            #     current_lr = opt.param_groups[0]["lr"]
            #     current_backbone_lr = opt.param_groups[0]["lr"]
            #     current_head_lr = (
            #         opt.param_groups[1]["lr"]
            #         if len(opt.param_groups) > 1
            #         else current_lr
            #     )

            val = score_loader(
                model,
                val_loader,
                use_geometry,
                df=val_df,
            )
            thr, note = find_best_iberta_threshold(
                val["score"], val["binary_label"], cfg.max_bpcer
            )
            m = rate_metrics(val["score"], val["binary_label"], thr)
            val_bce = float(
                nn.functional.binary_cross_entropy(
                    torch.tensor(
                        val["score"].clip(1e-7, 1 - 1e-7), dtype=torch.float32
                    ),
                    torch.tensor(val["binary_label"], dtype=torch.float32),
                )
            )

            # Determine metric to monitor for early stopping
            monitor_value = None
            if cfg.monitor_metric_for_early_stop == "acer":
                monitor_value = m["acer"]
            elif cfg.monitor_metric_for_early_stop == "bpc":
                monitor_value = m["bpcer"]
            elif cfg.monitor_metric_for_early_stop == "loss":
                monitor_value = val_bce
            else:
                # Default to ACER
                monitor_value = m["acer"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(train_loader),
                    "val_bce": val_bce,
                    "val_apcer": m["apcer"],
                    "val_bpcer": m["bpcer"],
                    "val_acer": m["acer"],
                    "threshold": thr,
                }
            )
            # One warning per run, not one per epoch.
            if not loss_note:
                loss_note = note

            # Check for improvement: lower APCER (since we're optimizing APCER subject to BPCER constraint)
            improved = (
                best["state"] is None or m["apcer"] < best["aper"] - 1e-6
            )  # Small epsilon for improvement

            if improved:
                best = {
                    "aper": m["apcer"],
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                }
                no_improve = 0
                if ckpt_path:
                    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                    torch.save(best["state"], ckpt_path)
            else:
                no_improve += 1
            h = history[-1]
            # Get current LR for logging
            current_lr = opt.param_groups[0]["lr"]
            current_backbone_lr = opt.param_groups[0]["lr"]
            current_head_lr = (
                opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else current_lr
            )
            print(
                f"  epoch {epoch:3d}  loss {h['train_loss']:.4f}  bce {h['val_bce']:.4f}"
                f"  acer {h['val_acer']:.4f}  bpc {h['val_bpcer']:.4f}  thr {h['threshold']:.3f}"
                f"  lr {current_lr:.2e}  bb_lr {current_backbone_lr:.2e}  hd_lr {current_head_lr:.2e}"
                f"  {'*' if improved else ' '} best {best['aper']:.4f}@{best['epoch']}"
                f"  patience {no_improve}/{cfg.patience}"
            )
            if no_improve >= cfg.patience:
                print(f"  early stop at epoch {epoch}")
                break

        model.load_state_dict(best["state"])
        return model, pd.DataFrame(history), loss_note

    return (train_model,)


@app.cell
def _(DEVICE):
    def load_depth():
        try:
            # For DepthAnything v2, we use the transformers library
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
            import torch

            # Using DepthAnything V2 Small (good balance of speed/accuracy)
            # Options: "depth-anything/Depth-Anything-V2-Small",
            #          "depth-anything/Depth-Anything-V2-Base",
            #          "depth-anything/Depth-Anything-V2-Large"
            depth_model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Base-hf"
            ).to(DEVICE)
            depth_model.eval()

            depth_transform = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Base-hf"
            )

            print("DepthAnything V2 loaded successfully.")
            return depth_model, depth_transform
        except Exception as e:
            print(f"Warning: Could not load DepthAnything: {e}")
            print("Falling back to MiDaS...")
            # Fallback to original MiDaS
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            depth_model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-hybrid-midas"
            ).to(DEVICE)
            depth_model.eval()
            depth_transform = DPTImageProcessor.from_pretrained(
                "Intel/dpt-hybrid-midas"
            )
            return depth_model, depth_transform

    return (load_depth,)


@app.cell
def _(
    CHECKPOINT_DIR,
    DEVICE,
    LivenessModel,
    MIN_SLICE_N,
    RunConfig,
    load_backbone,
    load_depth,
    load_fold_data,
    make_loaders,
    per_attack_apcer,
    rate_metrics,
    roc_table,
    score_loader,
    summarize_roc,
    train_model,
):
    def run_training(
        checkpoint_name: str = "vits_simple_training_fold1",
        seed: int | None = None,
        use_focal_loss: bool | None = None,
    ):
        """Main training function with tqdm progress bars.

        Pass `seed` and/or `use_focal_loss` to override the RunConfig
        defaults without editing the dataclass by hand -- useful for
        multi-seed ensemble runs.
        """

        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Configuration
        cfg = RunConfig()
        if seed is not None:
            cfg.seed = seed
        if use_focal_loss is not None:
            cfg.use_focal_loss = use_focal_loss
        fold = 1  # Using fold 1 for simplicity

        print(f"\n{'=' * 70}\n{cfg.name} · fold {fold}\n{'=' * 70}")

        # Load data
        train_df, val_df = load_fold_data(fold)
        live_types = set(train_df.loc[train_df["binary_label"] == 1, "attack_type"])

        # Create data loaders
        train_loader, val_loader, sampler = make_loaders(train_df, val_df, cfg)

        # depth
        depth_model = None
        depth_transform = None
        if cfg.use_depth_head:
            try:
                # try load a lightweight depth model
                depth_model, depth_transform = load_depth()
                print("Depth estimation model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load depth estimation model: {e}")
                print(
                    "Training will continue without depth information (depth features will be zero)"
                )

        # Load model
        backbone = load_backbone(cfg.backbone, cfg)
        set_seed(cfg.seed)
        model = LivenessModel(
            backbone,
            backbone.config.hidden_size,
            use_freq_head=cfg.use_freq_head,
            use_geometry=cfg.use_geometry,
            use_depth_head=cfg.use_depth_head,
            head_hidden=cfg.head_hidden,
        ).to(next(backbone.parameters()).device)

        if not checkpoint_name:
            # Setup checkpoint path
            ckpt = f"{CHECKPOINT_DIR}/{cfg.backbone}_{cfg.name}_fold{fold}.pt"
        else:
            ckpt = f"{CHECKPOINT_DIR}/{checkpoint_name}.pt"

        # Check if checkpoint exists
        if os.path.exists(ckpt):
            print(f"  Loading existing checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print("  Checkpoint loaded successfully!")
        else:
            print(f"  No existing checkpoint found. Starting fresh training...")
            # Train the model
            model, history, _ = train_model(
                model,
                train_loader,
                val_loader,
                val_df,
                cfg,
                find_best_iberta_threshold,
                score_loader=score_loader,
                use_geometry=cfg.use_geometry,
                depth_model=depth_model,
                depth_transform=depth_transform,
                ckpt_path=ckpt,
            )
            print(f"  Training completed. Model saved to: {ckpt}")

        # Evaluation
        print("\n" + "=" * 50)
        print("EVALUATION ON VALIDATION SET")
        print("=" * 50)

        val_scores = score_loader(model, val_loader, cfg.use_geometry, df=val_df)
        thr, note = find_best_iberta_threshold(
            val_scores["score"], val_scores["binary_label"], cfg.max_bpcer
        )
        print(f"  Threshold: {thr:.4f}  ({note})")

        frame_metrics = rate_metrics(
            val_scores["score"], val_scores["binary_label"], thr
        )
        print(f"  Frame-level:")
        print(f"    APCER: {frame_metrics['apcer']:.4f}")
        print(f"    BPCER: {frame_metrics['bpcer']:.4f}")
        print(f"    ACER:  {frame_metrics['acer']:.4f}")

        # Per-attack breakdown
        per_attack = per_attack_apcer(
            val_scores["score"],
            val_scores["binary_label"],
            val_scores["attack_type"],
            thr,
            live_types,
            MIN_SLICE_N,
        )
        if not per_attack.empty:
            print("\n  Per-attack APCER (validation):")
            for _, r in per_attack.iterrows():
                flag = ""
                if r["low_evidence"]:
                    flag += "  LOW EVIDENCE"
                if r["apcer"] > 0:
                    flag += "  <-- ABOVE 0% TARGET"
                print(
                    f"    {r['attack_type']:22s} {r['apcer']:.4f}  (n={r['n']}){flag}"
                )

        # ROC analysis
        curve = roc_table(val_scores["score"], val_scores["binary_label"])
        if not curve.empty:
            roc_summary = summarize_roc(curve, cfg.max_bpcer)
            print(f"\n  ROC Analysis:")
            print(f"    EER: {roc_summary['eer']:.4f}")
            print(
                f"    APCER@BPCER<={cfg.max_bpcer}: {roc_summary['apcer_at_bpcer_cap']:.4f}"
            )
            print(f"    BPCER@APCER=0: {roc_summary['bpcer_at_apcer_zero']:.4f}")

        print(f"\n  Model Info:")
        print(f"    Total params: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"    Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Show LR schedule info if used
        if hasattr(cfg, "lr_scheduler_type") and cfg.lr_scheduler_type != "none":
            print(f"\n  LR Schedule: {cfg.lr_scheduler_type}")
            if hasattr(cfg, "lr_decay_factor"):
                print(f"    Decay factor: {cfg.lr_decay_factor}")
                if cfg.lr_scheduler_type == "step" and hasattr(cfg, "lr_decay_epochs"):
                    print(f"    Decay every: {cfg.lr_decay_epochs} epochs")

        return model, val_scores, thr

    return (run_training,)


@app.cell
def _(fetch_bundle):
    def fetch_official_test_bundle():
        """Fetch the official test bundle for evaluation."""
        print("Fetching official test set...")
        fetch_bundle(
            "processed_dataset_official_test.zip",
            "processed_dataset_official_test",
        )
        return "processed_dataset_official_test"

    return (fetch_official_test_bundle,)


@app.cell
def _(
    CHECKPOINT_DIR,
    DEVICE,
    LivenessModel,
    MIN_SLICE_N,
    PADFrameDataset,
    RunConfig,
    fetch_official_test_bundle,
    load_backbone,
    load_fold_data,
    per_attack_apcer,
    rate_metrics,
    roc_table,
    score_loader,
    summarize_roc,
):
    def evaluate_test_set(checkpoint_path=None):
        """Evaluate the model on the official iBeta test set."""

        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = f"{CHECKPOINT_DIR}/vits_simple_training_fold1.pt"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please train the model first using the run_training() function."
            )

        print(f"Loading checkpoint from: {checkpoint_path}")

        # Fetch official test set if not already present
        test_dir = "processed_dataset_official_test"
        if not os.path.isdir(test_dir):
            test_dir = fetch_official_test_bundle()

        # Load test data
        def _read_frames_official(root: str) -> pd.DataFrame:
            """Load frames.csv for the OFFICIAL test bundle."""
            meta = os.path.join(root, "metadata")
            videos = pd.read_csv(os.path.join(meta, "videos.csv"))
            frames = pd.read_csv(os.path.join(meta, "frames.csv"))
            # Keep only official test rows
            is_official = videos["split_role"] == "official_test"
            videos = videos[is_official]
            frames = frames.merge(videos[["video_id"]], on="video_id", how="inner")
            frames["binary_label"] = (frames["label"] == "genuine").astype(int)
            frames["frame_path"] = frames[["frame_path"]].map(
                lambda p: os.path.join(root, p)
            )
            return frames.reset_index(drop=True)

        test_frames = _read_frames_official(test_dir)
        print(f"Test set loaded: {len(test_frames)} frames")

        # Create test dataset and loader
        test_dataset = PADFrameDataset(test_frames, augment=False, target_size=224)
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # load model
        cfg = RunConfig()
        backbone = load_backbone(cfg.backbone, cfg)
        model = LivenessModel(
            backbone,
            backbone.config.hidden_size,
            use_freq_head=cfg.use_freq_head,
            use_geometry=cfg.use_geometry,
            use_depth_head=cfg.use_depth_head,
            head_hidden=128,
        ).to(DEVICE)

        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully!")

        # Load validation data (fold 1)
        _, val_df = load_fold_data(fold=1)
        val_dataset = PADFrameDataset(val_df, augment=False, target_size=224)
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        val_scores = score_loader(
            model,
            val_loader,
            use_geometry=True,
            df=val_df,
        )

        # Compute threshold on validation set using iBeta-optimized objective
        thr, val_note = find_best_iberta_threshold(
            val_scores["score"], val_scores["binary_label"], cfg.max_bpcer
        )
        print(f"  Validation threshold: {thr:.4f}  ({val_note})")

        # -------- TEST SET EVALUATION (using fixed threshold) --------
        test_scores = score_loader(
            model,
            test_loader,
            use_geometry=True,
            df=test_frames,
        )
        test_metrics = rate_metrics(
            test_scores["score"], test_scores["binary_label"], thr
        )

        print(f"\n" + "=" * 50)
        print("OFFICIAL TEST SET EVALUATION (iBeta-optimized)")
        print("=" * 50)
        print(f"  Threshold: {thr:.4f}  (fixed from validation set)")
        print(f"  Test Results:")
        print(f"    APCER: {test_metrics['apcer']:.4f}")
        print(f"    BPCER: {test_metrics['bpcer']:.4f}")
        print(f"    ACER:  {test_metrics['acer']:.4f}")

        # Per-attack breakdown on test set
        live_types_test = set(
            test_frames.loc[test_frames["binary_label"] == 1, "attack_type"]
        )
        per_attack_test = per_attack_apcer(
            test_scores["score"],
            test_scores["binary_label"],
            test_frames["attack_type"],
            thr,
            live_types_test,
            MIN_SLICE_N,
        )
        if not per_attack_test.empty:
            print(f"\n  Per-attack APCER (test set):")
            for _, r in per_attack_test.iterrows():
                flag = ""
                if r["low_evidence"]:
                    flag += "  LOW EVIDENCE"
                if r["apcer"] > 0:
                    flag += "  <-- ABOVE 0% TARGET"
                print(
                    f"    {r['attack_type']:22s} {r['apcer']:.4f}  (n={r['n']}){flag}"
                )

        # ROC analysis on test set
        test_curve = roc_table(test_scores["score"], test_scores["binary_label"])
        if not test_curve.empty:
            test_roc_summary = summarize_roc(test_curve, 0.15)
            print(f"\n  Test ROC Analysis:")
            print(f"    EER: {test_roc_summary['eer']:.4f}")
            print(
                f"    APCER@BPCER<=0.15: {test_roc_summary['apcer_at_bpcer_cap']:.4f}"
            )
            print(f"    BPCER@APCER=0: {test_roc_summary['bpcer_at_apcer_zero']:.4f}")

        return test_scores, thr, test_metrics

    return (evaluate_test_set,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Simplified Passive Liveness Training & Evaluation

    This notebook provides a streamlined interface for:
    1. Training the model with Phase 1 improvements
    2. Evaluating on the official iBeta test set

    ## Phase 1 Improvements Implemented:
    - Increased model capacity: Unfreeze last 2 backbone blocks (instead of just last block)
    - Enhanced frequency branch: Increased lambda_freq from 0.3 to 0.6
    - Faster backbone adaptation: Increased backbone_lr_mult from 0.1 to 0.25
    - Added tqdm progress bars for training epochs

    ## Usage:
    1. Run `run_training()` to train the model (or load existing checkpoint)
    2. Run `evaluate_test_set()` to evaluate on the official test set
    """)
    return


@app.cell
def _(run_training):
    run_training()
    return


@app.cell
def _(evaluate_test_set):
    evaluate_test_set()
    return


@app.cell
def _(CHECKPOINT_DIR, fs):
    def save_checkpoint_to_drive(local_path: str, drive_dir: str = "pl_checkpoints"):
        """Upload a local checkpoint file to Google Drive under `drive_dir`."""
        assert os.path.exists(local_path), f"Local file not found: {local_path}"
        filename = os.path.basename(local_path)
        remote_path = f"{drive_dir}/{filename}"
        with open(local_path, "rb") as f:
            data = f.read()
        fs.pipe(remote_path, data)
        print(f"Uploaded {local_path} -> {remote_path} ({len(data) / 1e6:.1f} MB)")

    for ckpt_file in sorted(Path(CHECKPOINT_DIR).glob("*.pt")):
        save_checkpoint_to_drive(str(ckpt_file))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
