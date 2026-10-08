# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gdrive-fsspec>=2023.10.0",
#     "huggingface-hub>=0.23.0",
#     "fsspec>=2023.10.0",
#     "insightface>=0.7.3",
#     "marimo>=0.24.0",
#     "numpy",
#     "opencv-python-headless",
#     "pandas>=3.0.5",
#     "pillow>=12.0.0",
#     "python-lsp-ruff==2.3.4",
#     "python-lsp-server==1.15.0",
#     "ruff==0.16.7",
#     "scikit-learn>=1.9.0",
#     "torch>=2.14.0",
#     "tqdm>=4.70.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    # =====================================================================
    # Self-contained preprocessing pipeline.
    #
    # EVERYTHING below (pipeline code that used to live in
    # src/preprocess_data/*.py, plus the merge logic from
    # scripts/build_combined_pool.py) is inlined so this notebook runs on a
    # cloud GPU (molab) with no checkout of this repo. All Python deps come
    # from the script header above.
    #
    # Keep in sync: the serving path (src/passive_liveness/infer.py via
    # src/preprocess_data/{face_detect,align,geometry,quality,tracking}.py)
    # uses its own copy of the detection/alignment/quality code -- extraction
    # time and serve time must stay numerically consistent.
    # =====================================================================

    # --- stdlib ---
    import ast
    import csv
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import warnings
    import zipfile
    from collections import Counter
    from collections.abc import Callable, Iterator
    from dataclasses import asdict, dataclass, field
    from pathlib import Path

    # --- third-party ---
    import cv2
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from gdrive_fsspec import GoogleDriveFileSystem
    from insightface.app import FaceAnalysis
    from PIL import Image, ImageDraw, ImageFont
    from sklearn.model_selection import KFold, StratifiedKFold
    from tqdm import tqdm

    # =====================================================================
    # Canonical vocabularies. Both pools and the official test set resolve
    # through these, so a label is spelled once.
    #
    #   attack_type  the source of truth. Never inferred from folder names.
    #   pai_family   a pure rollup for coarse reporting, recomputable from
    #                attack_type, so it is never itself a source of truth.
    #   label        the binary split every pool's source CSV encodes
    #                differently (bona_fide / genuine / real / live). Resolved
    #                here once; the literal "bona_fide" used to be repeated in
    #                a second pool reader, so a rename in one place silently
    #                relabelled a whole pool as spoof.
    # =====================================================================

    # Source-CSV label spellings that mean a genuine capture.
    GENUINE_LABELS = frozenset({"bona_fide", "genuine", "real", "live", "bonafide"})

    def is_genuine_label(raw: str) -> bool:
        return str(raw).strip().lower() in GENUINE_LABELS

    PAI_FAMILY = {
        "live": "live",
        "print": "print",
        "print_3d": "print",
        "print_cutouts": "print",
        "print_eyeholes": "print",
        "cylinder": "print",
        "on_actor": "print",
        "replay": "replay",
        "spoof_untyped": "untyped",
    }

    def pai_family_of(attack_type: str) -> str:
        return PAI_FAMILY.get(attack_type, "untyped")

    # Shared slug regex -- used by source_metadata and direct-ingest cells.
    SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

    # =====================================================================
    # Per-pool extraction config -- thresholds calibrated per source:
    # All 5 datasets now in HF dataset, process all on GPU.
    # =====================================================================
    POOLS = [
        {
            "input_dir": "datasets/fas_ibeta_l1",
            "metadata_csv": "datasets/fas_ibeta_l1/metadata.csv",
            "output_dir": "processed_dataset",
            "dataset_source": "fas_ibeta_l1",
            "det_thresh": 0.3,
            "face_confidence_threshold": 0.55,
            "adaptive_crop_margin": True,
            "subpool_rules": [
                ("printout_id", "fas_ibeta_l1_printout"),
                ("a8_", "fas_ibeta_l1_a8"),
                ("kaggle_webcam_", "fas_ibeta_l1_webcam"),
                ("td_replay_", "fas_ibeta_l1_replay"),
                ("axon_print_", "fas_ibeta_l1_axonprint"),
            ],
        },
        {
            "input_dir": "datasets/lcc_fasd",
            "metadata_csv": "datasets/lcc_fasd/metadata.csv",
            "output_dir": "processed_dataset_lcc_fasd",
            "dataset_source": "lcc_fasd",
            "det_thresh": 0.05,
            "face_confidence_threshold": 0.1,
            "blur_threshold": 10,
        },
        {
            "input_dir": "datasets/trainingdatapro_2d_pad",
            "metadata_csv": "datasets/trainingdatapro_2d_pad/metadata.csv",
            "output_dir": "processed_dataset_pad2d",
            "dataset_source": "pad2d",
            "det_thresh": 0.05,
            "face_confidence_threshold": 0.1,
            "blur_threshold": 10,
        },
    ]

    # Pools that do NOT go through `extract` (no per-video config above): they
    # are either already-cropped images ingested verbatim, or the held-out test
    # set. `custom_pad_dataset` is metadata.csv-driven like the POOLS entries;
    # the other two are directory-convention scans. Kept here so the download
    # guard and the pre-flight verification below are both derived from one
    # list -- a pool that exists but is absent from the guard is exactly how a
    # stale copy survives a re-fetch.
    DIRECT_POOL_DIRS = [
        "datasets/frames_cleaned_cropped_v2",
        "datasets/custom_pad_dataset",
        "datasets/test_data_ibeta",
    ]

    # Which of those direct-ingest pools are metadata.csv-driven rather than
    # directory-scanned. Only custom_pad_dataset is; the other two have no CSV
    # and are enumerated by convention. Listed here so the pre-flight
    # verification derives its pool list instead of hardcoding one entry, which
    # is how a pool ends up checked in one place and not the other.
    CSV_DRIVEN_DIRECT_POOLS = [
        "datasets/custom_pad_dataset",
    ]

    HF_REPO_ID = "amoghshakya/passive-liveness"
    HF_TOKEN = os.environ.get("HF_TOKEN")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Preprocessing pipeline (cloud GPU)

    **Fully self-contained** -- all pipeline code (former
    `src/preprocess_data/*` + `scripts/build_combined_pool.py`) is inlined in
    the setup cell above, and all deps come from the script header. No repo
    checkout needed on the cloud machine.

    Standalone from the training notebook (`notebook.py`) on purpose -- run
    this only when raw source data actually changes (new subjects added, a
    detector threshold recalibrated, a new pool merged in), not every
    training session.

    Downloads raw datasets from Hugging Face (the `datasets/` tree, ~20 GB),
    processes on GPU, merges, and uploads `processed_dataset.zip` to Google Drive.

    Needs `HF_TOKEN` env var + Google Drive OAuth (first run).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1 · Fetch raw data

    Syncs the `datasets/` tree (~20 GB) from Hugging Face on first run; skipped
    once the pool directories exist.
    """)
    return


@app.cell
def _():
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    print(f"Connected to HF Hub: {HF_REPO_ID}")
    return (api,)


@app.cell
def _(api):
    # Derived from the canonical pool lists, not hand-maintained: a pool that
    # exists but is missing from this list lets the fetch be skipped while a
    # stale copy of that pool stays on disk.
    _required_sources = [p["input_dir"] for p in POOLS] + list(DIRECT_POOL_DIRS)
    if not all(os.path.isdir(path) for path in _required_sources):
        # Per-file sync, not one opaque blob. `datasets.tar.zst` cost the full
        # ~20 GB on the Hub for every change, no matter how small: tar+zstd of a
        # modified tree shares no bytes with the previous archive, so a 1.5 MB
        # metadata.csv edit produced a brand-new 20 GB LFS object. Two of those
        # were 40 GB of the free-tier quota. At file granularity the same edit
        # uploads 1.5 MB and the ~18k unchanged images dedup to nothing.
        # `snapshot_download` also resumes and re-checks per-file etags, so a
        # partial fetch is recoverable instead of all-or-nothing.
        print("Syncing datasets/ from Hugging Face (~20GB, ~18.5k files)...")
        api.snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=["*"],
            # Keeps the two hand-curated CSV backups next to metadata.csv out of
            # the Hub, so a future sync can't silently promote one over the other.
            ignore_patterns=["**/*.bak"],
            local_dir="./datasets",
            token=HF_TOKEN,
            max_workers=16,
        )
        print("Done.")
    else:
        print("raw datasets already present, skipping fetch")
    return


@app.cell
def _():
    # A directory existing proves PRESENCE but not FRESHNESS: the guard above
    # skips the fetch entirely when all pool directories are present, so a tree
    # left over from an earlier sync survives a newer Hub revision unnoticed,
    # and the only symptom is thousands of skipped rows much later. So verify
    # that every metadata row actually resolves. This is the same check that
    # diagnosed the custom_pad_dataset repo-root-relative `dst_path` bug,
    # promoted to a pre-flight gate.
    _CSV_POOLS = [
        (p["dataset_source"], Path(p["input_dir"]), Path(p["metadata_csv"]))
        for p in POOLS
    ] + [(d, Path(d), Path(d) / "metadata.csv") for d in CSV_DRIVEN_DIRECT_POOLS]

    def verify_raw_pools(tolerance: float = 0.01) -> None:
        """Fail loudly when a pool's metadata does not resolve on disk.

        `tolerance` is the fraction of rows allowed to be missing before the
        drift is treated as structural -- a wrong path convention, a half-renamed
        pool -- rather than single-file drift between the CSV and disk. fas_ibeta_l1
        currently sits at 1/695 (0.14%), so 1% leaves headroom without masking a
        systematic break.
        """
        problems: list[str] = []
        for rel in DIRECT_POOL_DIRS:
            if not Path(rel).is_dir():
                problems.append(f"{rel}: missing directory")
        for source, root, meta in _CSV_POOLS:
            if not root.is_dir():
                problems.append(f"{source}: missing directory {root}")
                continue
            if not meta.is_file():
                problems.append(f"{source}: missing {meta}")
                continue
            with meta.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                problems.append(f"{source}: {meta} has no data rows")
                continue
            missing = [
                r["dst_path"] for r in rows if not (root / r["dst_path"]).exists()
            ]
            if not missing:
                print(f"  ok: {source}: {len(rows)}/{len(rows)} rows resolve")
                continue
            frac = len(missing) / len(rows)
            detail = (
                f"{source}: {len(missing)}/{len(rows)} rows unresolvable "
                f"({frac:.2%}), first: {missing[0]!r}"
            )
            if frac > tolerance:
                problems.append(detail)
            else:
                print(f"  note: {detail} (within {tolerance:.2%} tolerance)")
        if problems:
            raise RuntimeError(
                "raw pool verification failed -- the Hugging Face archive is "
                "stale, or a metadata.csv uses the wrong path convention:\n  "
                + "\n  ".join(problems)
                + "\n  Re-sync the affected pool (the images did not change; "
                "only metadata.csv and directory names did) before running."
            )
        print("  all raw pools verified")

    verify_raw_pools()
    return


@app.cell
def _():
    fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    mo.output.clear_console()
    return (fs,)


@app.cell
def config():
    # Config dataclass + sampling/stage enum constants.
    SAMPLING_STRATEGIES = [
        "uniform",
        "random",
        "fibonacci",
        "quality_weighted",
        "center_biased",
    ]
    STAGES = [
        "extract",
        "metadata",
        "splits",
        "balance",
        "report",
        "contact_sheets",
    ]

    @dataclass
    class Config:
        # I/O
        input_dir: Path = Path("fas_ibeta_l1")
        metadata_csv: Path | None = None  # defaults to <input_dir>/metadata.csv
        output_dir: Path = Path("processed_dataset")
        dataset_source: str = (
            "fas_ibeta_l1"  # tag written on every row; one value per input dataset
        )
        subpool_rules: list[tuple[str, str]] = field(default_factory=list)
        # (subject_id prefix, dataset_source override) pairs, checked in order,
        # first match wins. Lets one raw input dataset split into finer
        # sub-pools (e.g. a batch that behaves like a distinct capture domain)
        # without a separate re-tagging pass after extraction.

        # frame extraction
        num_frames: int = 8
        sampling_strategy: str = "uniform"
        num_candidates: int = (
            48  # candidate frames sampled across whole video before detection
        )

        # face detection / tracking
        face_confidence_threshold: float = 0.5
        det_thresh: float | None = None
        iou_track_threshold: float = 0.3
        min_track_len: int = 3

        # quality filtering
        min_face_size: int = 48  # px, shorter bbox side in ORIGINAL frame
        min_face_ratio: float = 0.04  # bbox_height / frame_height
        blur_threshold: float = (
            40.0  # variance-of-Laplacian on the aligned crop; below => blurry
        )
        max_pose_asym: float | None = None  # None = pose filter disabled

        # face crop
        face_size: int = (
            512  # aligned crop side length; NOT the model's final input size
        )
        crop_margin_scale: float = (
            1.0  # <1.0 zooms out (align_face), revealing more context around the face
        )
        adaptive_crop_margin: bool = False  # if True, derive margin_scale per-face from face-to-frame area ratio instead of the fixed crop_margin_scale

        # splits
        num_folds: int = 5
        seed: int = 42

        # misc
        device: str = "auto"  # "auto" | "cpu" | "cuda"
        limit_videos: int | None = None
        skip_existing: bool = False
        stages: list[str] = field(default_factory=lambda: list(STAGES))
        contact_sheet_examples: int = 6

        def __post_init__(self) -> None:
            self.input_dir = Path(self.input_dir)
            self.output_dir = Path(self.output_dir)
            if self.metadata_csv is None:
                self.metadata_csv = self.input_dir / "metadata.csv"
            else:
                self.metadata_csv = Path(self.metadata_csv)
            if self.sampling_strategy not in SAMPLING_STRATEGIES:
                raise ValueError(
                    f"sampling_strategy must be one of {SAMPLING_STRATEGIES}"
                )
            for s in self.stages:
                if s not in STAGES:
                    raise ValueError(f"unknown stage {s!r}, must be one of {STAGES}")

        def to_json(self) -> str:
            d = asdict(self)
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
            return json.dumps(d, indent=2, sort_keys=True)

        def save(self, path: Path) -> None:
            path.write_text(self.to_json())

    return (Config,)


@app.cell
def video_io():
    # Video I/O: probing, candidate sampling, frame reading.
    @dataclass(frozen=True)
    class VideoProbe:
        fps: float
        frame_count: int
        duration_sec: float
        width: int
        height: int

    def probe(video_path: Path) -> VideoProbe:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"could not open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frame_count <= 0:
                frame_count = count_frames_by_reading(cap)
            duration_sec = (frame_count / fps) if fps > 0 else 0.0
            return VideoProbe(
                fps=fps,
                frame_count=frame_count,
                duration_sec=duration_sec,
                width=width,
                height=height,
            )
        finally:
            cap.release()

    def count_frames_by_reading(cap: cv2.VideoCapture) -> int:
        n = 0
        while cap.grab():
            n += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return n

    def candidate_indices(frame_count: int, num_candidates: int) -> list[int]:
        """Uniformly spaced candidate frame indices across the whole video."""
        if frame_count <= 0:
            return []
        n = min(num_candidates, frame_count)
        if n <= 1:
            return [0]
        return sorted({round(i * (frame_count - 1) / (n - 1)) for i in range(n)})

    @dataclass(frozen=True)
    class RawFrame:
        frame_index: int
        timestamp_sec: float
        image: np.ndarray  # BGR

    def read_frames_at(
        video_path: Path, indices: list[int], fallback_fps: float
    ) -> list[RawFrame]:
        if not indices:
            return []
        wanted = set(indices)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"could not open video: {video_path}")
        out: list[RawFrame] = []
        try:
            idx = 0
            last_wanted = max(wanted)
            while idx <= last_wanted:
                ok = cap.grab()
                if not ok:
                    break
                if idx in wanted:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        ts = (
                            ts_msec / 1000.0
                            if ts_msec and ts_msec > 0
                            else (
                                idx / fallback_fps if fallback_fps > 0 else float(idx)
                            )
                        )
                        out.append(
                            RawFrame(frame_index=idx, timestamp_sec=ts, image=frame)
                        )
                idx += 1
        finally:
            cap.release()
        return out

    return candidate_indices, probe, read_frames_at


@app.cell
def face_detect():
    # RetinaFace-based face detection (via insightface).
    @dataclass(frozen=True)
    class Detection:
        box: tuple[
            float, float, float, float
        ]  # x1, y1, x2, y2 in ORIGINAL frame coords
        prob: float
        landmarks: np.ndarray  # (5, 2) in ORIGINAL frame coords

    def resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    class FaceDetector:
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

        def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
            if not frames_bgr:
                return []
            results: list[list[Detection]] = []
            for frame in frames_bgr:
                faces = self._app.get(frame)
                dets = [
                    Detection(
                        box=tuple(float(v) for v in f.bbox),
                        prob=float(f.det_score),
                        landmarks=np.asarray(f.kps, dtype=np.float32),
                    )
                    for f in faces
                ]
                results.append(dets)
            return results

    return Detection, FaceDetector


@app.cell
def tracking(Detection):
    # Primary-face tracking across candidate frames.
    def area(box: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def box_diagonal(box: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = box
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    @dataclass(frozen=True)
    class TrackedFrame:
        frame_index: int
        detection: Detection | None
        track_status: str  # "ok" | "no_face" | "track_reacquired" | "primary_face_lost"

    def track_primary_face(
        frame_indices: list[int],
        detections_per_frame: list[list[Detection]],
        iou_track_threshold: float = 0.3,
        reacquire_center_factor: float = 1.5,
        reacquire_size_ratio_range: tuple[float, float] = (0.4, 2.5),
    ) -> list[TrackedFrame]:
        assert len(frame_indices) == len(detections_per_frame)
        out: list[TrackedFrame] = []
        last_box: tuple[float, float, float, float] | None = None

        for idx, dets in zip(frame_indices, detections_per_frame):
            if not dets:
                out.append(TrackedFrame(idx, None, "no_face"))
                continue

            if last_box is None:
                best = max(dets, key=lambda d: d.prob * area(d.box))
                out.append(TrackedFrame(idx, best, "ok"))
                last_box = best.box
                continue

            best_det, best_iou = max(
                ((d, iou(d.box, last_box)) for d in dets), key=lambda t: t[1]
            )
            if best_iou >= iou_track_threshold:
                out.append(TrackedFrame(idx, best_det, "ok"))
                last_box = best_det.box
                continue

            # try re-acquisition: same rough position/size, just below IoU threshold
            diag = box_diagonal(last_box) or 1.0
            lc = center(last_box)

            def dist(d: Detection, lc: tuple[float, float] = lc) -> float:
                c = center(d.box)
                return ((c[0] - lc[0]) ** 2 + (c[1] - lc[1]) ** 2) ** 0.5

            cand = min(dets, key=dist)
            size_ratio = area(cand.box) / area(last_box) if area(last_box) > 0 else 0.0
            lo, hi = reacquire_size_ratio_range
            if dist(cand) <= reacquire_center_factor * diag and lo <= size_ratio <= hi:
                out.append(TrackedFrame(idx, cand, "track_reacquired"))
                last_box = cand.box
            else:
                out.append(TrackedFrame(idx, None, "primary_face_lost"))
                # keep last_box as-is so a later frame can still re-acquire against it

        return out

    return TrackedFrame, track_primary_face


@app.cell
def quality(Detection, TrackedFrame):
    # Quality filtering for tracked face detections.
    REJECTION_REASONS = [
        "no_face",
        "primary_face_lost",
        "low_confidence",
        "too_small",
        "blurry",
        "extreme_pose",
        "invalid_bbox",
    ]

    @dataclass
    class QualityConfig:
        """Extraction-time accept/reject thresholds.

        Construct ONLY via `from_config`. This dataclass deliberately has no
        defaults: a previous version carried its own, including
        `face_confidence_threshold=0.90` -- a stale MTCNN-era value that would
        reject nearly everything under RetinaFace's 0.5-scale scores. Two
        copies of one threshold is one too many; every call site now derives
        from the single `Config` that POOLS sets per pool.
        """

        face_confidence_threshold: float
        min_face_size: int
        min_face_ratio: float
        blur_threshold: float
        max_pose_asym: float | None

        @classmethod
        def from_config(cls, cfg) -> "QualityConfig":
            return cls(
                face_confidence_threshold=cfg.face_confidence_threshold,
                min_face_size=cfg.min_face_size,
                min_face_ratio=cfg.min_face_ratio,
                blur_threshold=cfg.blur_threshold,
                max_pose_asym=cfg.max_pose_asym,
            )

    @dataclass(frozen=True)
    class QualityResult:
        accepted: bool
        reason: str | None
        detector_confidence: float | None
        face_bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2, clipped to frame
        face_width: int | None
        face_height: int | None
        blur_score: float | None
        pose_asym: float | None

    def blur_score(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_pose_asym(landmarks: np.ndarray) -> float | None:
        # insightface/RetinaFace 5-point order: left_eye, right_eye, nose, mouth_left, mouth_right
        if landmarks is None or landmarks.shape[0] < 3:
            return None
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        interocular = float(np.linalg.norm(left_eye - right_eye))
        if interocular <= 1e-6:
            return None
        d_left = abs(float(nose[0]) - float(left_eye[0]))
        d_right = abs(float(right_eye[0]) - float(nose[0]))
        return abs(d_left - d_right) / interocular

    def evaluate(
        frame_bgr: np.ndarray,
        tracked: TrackedFrame,
        cfg: QualityConfig,
    ) -> QualityResult:
        frame_h, frame_w = frame_bgr.shape[:2]

        if tracked.detection is None:
            return QualityResult(
                False, tracked.track_status, None, None, None, None, None, None
            )

        det: Detection = tracked.detection
        x1, y1, x2, y2 = det.box
        x1c, y1c = max(0, round(x1)), max(0, round(y1))
        x2c, y2c = min(frame_w, round(x2)), min(frame_h, round(y2))

        if x2c <= x1c or y2c <= y1c:
            return QualityResult(
                False, "invalid_bbox", det.prob, None, None, None, None, None
            )

        bbox = (x1c, y1c, x2c, y2c)
        face_w, face_h = x2c - x1c, y2c - y1c
        blur = blur_score(frame_bgr, bbox)
        pose_asym = compute_pose_asym(det.landmarks)

        if det.prob < cfg.face_confidence_threshold:
            return QualityResult(
                False,
                "low_confidence",
                det.prob,
                bbox,
                face_w,
                face_h,
                blur,
                pose_asym,
            )

        if (
            min(face_w, face_h) < cfg.min_face_size
            or (face_h / frame_h) < cfg.min_face_ratio
        ):
            return QualityResult(
                False,
                "too_small",
                det.prob,
                bbox,
                face_w,
                face_h,
                blur,
                pose_asym,
            )

        if blur < cfg.blur_threshold:
            return QualityResult(
                False,
                "blurry",
                det.prob,
                bbox,
                face_w,
                face_h,
                blur,
                pose_asym,
            )

        if (
            cfg.max_pose_asym is not None
            and pose_asym is not None
            and pose_asym > cfg.max_pose_asym
        ):
            return QualityResult(
                False,
                "extreme_pose",
                det.prob,
                bbox,
                face_w,
                face_h,
                blur,
                pose_asym,
            )

        return QualityResult(
            True, None, det.prob, bbox, face_w, face_h, blur, pose_asym
        )

    return QualityConfig, evaluate


@app.cell
def align():
    # 5-point similarity-transform face alignment.
    # ArcFace 112x112 reference points: left_eye, right_eye, nose, mouth_left, mouth_right
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
        """`margin_scale` < 1.0 shrinks the reference landmark layout toward
        its own centroid, which zooms the face out within the same face_size
        canvas -- more surrounding context (paper edges, occlusion
        boundaries, scene) survives into the crop, at the cost of less
        resolution on the face itself. 1.0 (default) is the original tight
        ArcFace-standard crop."""
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
        frame_shape: tuple[int, int],
        min_scale: float = 0.35,
        max_scale: float = 1.0,
    ) -> float:
        """A fixed margin_scale is a bad fit for a print/print_cut attack
        held at varying distance from the camera: a close-up print (large
        face_bbox relative to frame) needs more zoom-out to keep its cut edge
        / background boundary inside the crop, while a face that's already
        small in frame is already zoomed out and doesn't need it.
        Interpolates margin_scale from face-area-to-frame-area ratio: small
        ratio (already zoomed out) -> max_scale (no extra zoom-out), large
        ratio (zoomed in / close-up) -> min_scale (zoom out further)."""
        x1, y1, x2, y2 = face_bbox
        frame_h, frame_w = frame_shape[:2]
        face_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, frame_w * frame_h)
        return float(np.interp(face_area_ratio, [0.05, 0.3], [max_scale, min_scale]))

    return REF_112, adaptive_margin_scale, align_face


@app.cell
def geometry():
    # Per-face geometry ratios from 5-point landmarks.
    @dataclass(frozen=True)
    class GeometryFeatures:
        interocular_dist_norm: float
        eye_to_nose_dist_norm: float
        nose_to_mouth_dist_norm: float
        eye_to_mouth_dist_norm: float
        mouth_width_norm: float
        face_aspect_ratio: float  # bbox width / height

    def compute_geometry(
        landmarks_5pt: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> GeometryFeatures | None:
        if landmarks_5pt is None or landmarks_5pt.shape[0] < 5:
            return None
        x1, y1, x2, y2 = bbox
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        norm = (w * w + h * h) ** 0.5  # bbox diagonal, scale-normalizer

        left_eye, right_eye, nose, mouth_l, mouth_r = landmarks_5pt
        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (mouth_l + mouth_r) / 2.0

        return GeometryFeatures(
            interocular_dist_norm=float(np.linalg.norm(left_eye - right_eye) / norm),
            eye_to_nose_dist_norm=float(np.linalg.norm(eye_mid - nose) / norm),
            nose_to_mouth_dist_norm=float(np.linalg.norm(nose - mouth_mid) / norm),
            eye_to_mouth_dist_norm=float(np.linalg.norm(eye_mid - mouth_mid) / norm),
            mouth_width_norm=float(np.linalg.norm(mouth_l - mouth_r) / norm),
            face_aspect_ratio=float(w / h),
        )

    return (compute_geometry,)


@app.cell
def sampling():
    # Frame-selection strategies over accepted candidates.
    def clip_count(n_available: int, num_frames: int) -> int:
        return min(n_available, num_frames)

    def dedup_pad(idxs: list[int], n_available: int, num_frames: int) -> list[int]:
        """Round-off can collide two targets on the same index; pad with the
        nearest unused index so we return exactly min(n_available, num_frames)."""
        want = clip_count(n_available, num_frames)
        chosen = set(idxs)
        ordered = sorted(chosen)
        if len(ordered) >= want:
            return ordered[:want]
        remaining = [i for i in range(n_available) if i not in chosen]
        # fill by proximity to existing picks, cheap and good enough for small n
        remaining.sort(key=lambda r: min(abs(r - c) for c in ordered) if ordered else r)
        for r in remaining:
            ordered.append(r)
            if len(ordered) >= want:
                break
        return sorted(ordered)

    def uniform(
        n_available: int,
        num_frames: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> list[int]:
        want = clip_count(n_available, num_frames)
        if want <= 0:
            return []
        if want == 1:
            return [n_available // 2]
        positions = np.linspace(0, n_available - 1, want)
        idxs = [round(p) for p in positions]
        return dedup_pad(idxs, n_available, num_frames)

    def random_strategy(
        n_available: int,
        num_frames: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> list[int]:
        want = clip_count(n_available, num_frames)
        if want <= 0:
            return []
        idxs = rng.choice(n_available, size=want, replace=False).tolist()
        return sorted(idxs)

    def fibonacci(
        n_available: int,
        num_frames: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> list[int]:
        """Front-loaded sampling using Fibonacci-spaced cumulative fractions.
        Kept for later experiments per the architecture doc; NOT the default."""
        want = clip_count(n_available, num_frames)
        if want <= 0:
            return []
        if want == 1:
            return [n_available // 2]
        fibs = [1, 1]
        while len(fibs) < want:
            fibs.append(fibs[-1] + fibs[-2])
        fibs = fibs[:want]
        cum = np.cumsum(fibs, dtype=float)
        cum = cum / cum[-1]
        idxs = [round(f * (n_available - 1)) for f in cum]
        return dedup_pad(idxs, n_available, num_frames)

    def quality_weighted(
        n_available: int,
        num_frames: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> list[int]:
        want = clip_count(n_available, num_frames)
        if want <= 0:
            return []
        if weights is None or len(weights) != n_available or weights.sum() <= 0:
            probs = np.full(n_available, 1.0 / n_available)
        else:
            probs = weights / weights.sum()
        idxs = rng.choice(n_available, size=want, replace=False, p=probs).tolist()
        return sorted(idxs)

    def center_biased(
        n_available: int,
        num_frames: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> list[int]:
        want = clip_count(n_available, num_frames)
        if want <= 0:
            return []
        center = (n_available - 1) / 2.0
        sigma = max(1.0, n_available / 4.0)
        positions = np.arange(n_available)
        probs = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
        probs = probs / probs.sum()
        idxs = rng.choice(n_available, size=want, replace=False, p=probs).tolist()
        return sorted(idxs)

    STRATEGIES: dict[
        str,
        Callable[[int, int, np.random.Generator, np.ndarray | None], list[int]],
    ] = {
        "uniform": uniform,
        "random": random_strategy,
        "fibonacci": fibonacci,
        "quality_weighted": quality_weighted,
        "center_biased": center_biased,
    }
    return (STRATEGIES,)


@app.cell
def source_metadata():
    # VideoRecord + source-CSV loading. GENUINE_LABELS / is_genuine_label come
    # from the setup cell -- defined once, used by every pool reader.
    DATASET_SOURCE = "fas_ibeta_l1"

    @dataclass(frozen=True)
    class VideoRecord:
        subject_id: str
        video_id: str
        video_path: str  # relative to input_dir
        media_type: str  # "video" | "image"
        label: str  # "genuine" | "spoof"
        attack_type: str  # "live" for genuine, else the PAI class
        pai_family: str  # rollup of attack_type, for coarse reporting
        dataset_source: str
        device: str
        replay_device: str | None
        session: str | None
        environment: str | None
        active_zoom: str  # "unknown" unless a dataset explicitly records it
        raw_label: (
            str  # original label string from the source CSV, kept for traceability
        )
        split_role: str = "development"

    def slug(s: str) -> str:
        return SAFE.sub("_", s).strip("_")

    def normalize_device(device: str) -> str:
        """Casing/separator-only normalization (e.g. "Galaxy_A54" /
        "Galaxy_a54" -> "galaxy a54") so per-device breakdowns aren't
        fragmented by free-text device-name variance across ingestion
        batches. Does NOT attempt to unify genuinely different spellings of
        the same device (e.g. "Iphone11" vs "iPhone 11") -- that needs an
        explicit mapping, not a casing rule."""
        return re.sub(r"[\s_]+", " ", device).strip().lower()

    def load_video_records(
        metadata_csv: Path,
        input_dir: Path,
        dataset_source: str = DATASET_SOURCE,
        subpool_rules: list[tuple[str, str]] | None = None,
    ) -> list[VideoRecord]:
        if not metadata_csv.exists():
            raise FileNotFoundError(
                f"no source metadata mapping at {metadata_csv}. "
                "This pipeline requires an explicit label/attack-type mapping file "
                "rather than guessing labels from directory names."
            )
        records: list[VideoRecord] = []
        skipped = 0
        first_missing: str | None = None
        with metadata_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dst_path = row["dst_path"]
                abs_path = input_dir / dst_path
                if not abs_path.exists():
                    # Source CSV can drift from disk; skip rather than crash the
                    # whole run, but COUNT it. A bare `continue` here once hid a
                    # genuinely missing video from fas_ibeta_l1 entirely.
                    skipped += 1
                    if first_missing is None:
                        first_missing = dst_path
                    continue
                raw_label = row["label"]
                label = "genuine" if is_genuine_label(raw_label) else "spoof"
                # Three spellings of the same field exist across pools:
                #   attack_type  the newer dedicated column
                #   pai          lcc_fasd / trainingdatapro_2d_pad
                #   label        the original, which for those pools holds the
                #                PAI class directly ("on_actor", not "spoof")
                # Prefer the most specific available.
                explicit_attack_type = (row.get("attack_type") or "").strip() or (
                    row.get("pai") or ""
                ).strip()
                if label == "genuine":
                    attack_type = "live"
                elif explicit_attack_type:
                    attack_type = explicit_attack_type
                else:
                    attack_type = raw_label
                subject_id = row["user_id"] or "unknown_subject"
                file_id = row.get("file_id", "")
                stem = slug(Path(dst_path).stem)
                media_type = (
                    "image"
                    if Path(dst_path).suffix.lower()
                    in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    }
                    else "video"
                )
                video_id = f"v{file_id}_{slug(subject_id)}_{stem}"
                split_role = (row.get("split_role") or "development").strip()
                # capture_device carries the real value where both are
                # present; the newer "device" column is currently just a
                # placeholder ("UnknownDevice") on every row that has it.
                raw_device = (
                    (row.get("capture_device") or "").strip()
                    or (row.get("device") or "").strip()
                    or "unknown"
                )
                device = normalize_device(raw_device)
                replay_device = row.get("replay_device") or None
                row_dataset_source = dataset_source
                for prefix, override in subpool_rules or []:
                    if subject_id.startswith(prefix):
                        row_dataset_source = override
                        break
                records.append(
                    VideoRecord(
                        subject_id=subject_id,
                        video_id=video_id,
                        video_path=dst_path,
                        media_type=media_type,
                        label=label,
                        attack_type=attack_type,
                        pai_family=pai_family_of(attack_type),
                        dataset_source=row_dataset_source,
                        device=device,
                        replay_device=replay_device,
                        session=None,
                        environment=None,
                        active_zoom="unknown",
                        raw_label=raw_label,
                        split_role=split_role,
                    )
                )
        if skipped:
            print(
                f"  WARNING: {dataset_source}: skipped {skipped} metadata row(s) "
                f"pointing at missing files (first: {first_missing!r})"
            )
        return records

    return VideoRecord, load_video_records


@app.cell
def extract(
    Config,
    Detection,
    FaceDetector,
    QualityConfig,
    STRATEGIES: dict[str, Callable[[int, int, np.random.Generator, np.ndarray | None], list[int]]],
    VideoRecord,
    adaptive_margin_scale,
    align_face,
    candidate_indices,
    compute_geometry,
    evaluate,
    probe,
    read_frames_at,
    track_primary_face,
):
    # Per-video extraction pipeline.
    def seeded_rng(seed: int, video_id: str) -> np.random.Generator:
        h = hashlib.sha256(f"{seed}:{video_id}".encode()).digest()
        sub_seed = int.from_bytes(h[:8], "little")
        return np.random.default_rng(sub_seed)

    @dataclass
    class ExtractResult:
        video_row: dict
        frame_rows: list[dict]
        candidate_rows: list[dict]

    def frame_dir(output_dir: Path, subject_id: str, video_id: str) -> Path:
        return output_dir / "faces" / subject_id / video_id

    FINGERPRINT_FIELDS = [
        "num_frames",
        "sampling_strategy",
        "num_candidates",
        "face_confidence_threshold",
        "det_thresh",
        "iou_track_threshold",
        "min_track_len",
        "min_face_size",
        "min_face_ratio",
        "blur_threshold",
        "max_pose_asym",
        "face_size",
        "crop_margin_scale",
        "adaptive_crop_margin",
        "seed",
        "dataset_source",
        "subpool_rules",
    ]

    def compute_fingerprint(cfg: Config) -> dict:
        # json round-trips tuples as lists, so normalize here to keep the
        # in-memory (current run) and json.loads-restored (cached)
        # fingerprints comparable -- a bare tuple in subpool_rules would
        # otherwise never equal its own cached form and silently defeat
        # --skip-existing.
        return json.loads(json.dumps({k: getattr(cfg, k) for k in FINGERPRINT_FIELDS}))

    def manifest_path(out_dir: Path) -> Path:
        return out_dir / "manifest.json"

    def load_cached(out_dir: Path, fingerprint: dict) -> ExtractResult | None:
        mp = manifest_path(out_dir)
        if not mp.exists():
            return None
        try:
            data = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("fingerprint") != fingerprint:
            return None
        return ExtractResult(
            data["video_row"], data["frame_rows"], data["candidate_rows"]
        )

    def save_cache(out_dir: Path, result: ExtractResult, fingerprint: dict) -> None:
        mp = manifest_path(out_dir)
        mp.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "video_row": result.video_row,
                    "frame_rows": result.frame_rows,
                    "candidate_rows": result.candidate_rows,
                }
            )
        )

    def _base_media_row(video: VideoRecord) -> dict:
        return {
            "subject_id": video.subject_id,
            "video_id": video.video_id,
            "video_path": video.video_path,
            "media_type": video.media_type,
            "label": video.label,
            "attack_type": video.attack_type,
            "pai_family": video.pai_family,
            "dataset_source": video.dataset_source,
            "device": video.device,
            "replay_device": video.replay_device,
            "session": video.session,
            "environment": video.environment,
            "active_zoom": video.active_zoom,
            "raw_label": video.raw_label,
            "split_role": video.split_role,
        }

    # ---- shared row builders --------------------------------------------
    # The image and video paths used to spell out these dicts field by field.
    # Anything added to one was silently absent from the other, so both now go
    # through one builder each.

    def _candidate_row(media, frame_index, timestamp, tracked, qr, chosen, seq):
        return {
            "subject_id": media.subject_id,
            "video_id": media.video_id,
            "media_type": media.media_type,
            "split_role": media.split_role,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "quality_status": "accepted" if qr.accepted else "rejected",
            "rejection_reason": qr.reason,
            "track_status": tracked.track_status,
            "detector_confidence": qr.detector_confidence,
            "face_bbox": qr.face_bbox,
            "face_width": qr.face_width,
            "face_height": qr.face_height,
            "blur_score": qr.blur_score,
            "pose_asym": qr.pose_asym,
            "chosen": chosen,
            "sequence_position": seq,
            "label": media.label,
            "attack_type": media.attack_type,
            "pai_family": media.pai_family,
            "dataset_source": media.dataset_source,
        }

    def _frame_row(media, frame_index, timestamp, seq, qr, geo, out_dir, frame_name):
        rel_frame = Path("faces") / media.subject_id / media.video_id / frame_name
        rel_landmarks = (
            Path("faces")
            / media.subject_id
            / media.video_id
            / "landmarks"
            / frame_name.replace(".jpg", ".npy")
        )
        return {
            "subject_id": media.subject_id,
            "video_id": media.video_id,
            "media_type": media.media_type,
            "split_role": media.split_role,
            "frame_id": f"{media.video_id}_f{seq:03d}",
            "frame_index": frame_index,
            "timestamp": timestamp,
            "frame_path": str(rel_frame),
            "face_bbox": qr.face_bbox,
            "face_width": qr.face_width,
            "face_height": qr.face_height,
            "detector_confidence": qr.detector_confidence,
            "blur_score": qr.blur_score,
            "quality_status": "selected",
            "landmark_path": str(rel_landmarks),
            "sequence_position": seq,
            "label": media.label,
            "attack_type": media.attack_type,
            "pai_family": media.pai_family,
            "dataset_source": media.dataset_source,
            "interocular_dist_norm": geo.interocular_dist_norm if geo else None,
            "eye_to_nose_dist_norm": geo.eye_to_nose_dist_norm if geo else None,
            "nose_to_mouth_dist_norm": geo.nose_to_mouth_dist_norm if geo else None,
            "eye_to_mouth_dist_norm": geo.eye_to_mouth_dist_norm if geo else None,
            "mouth_width_norm": geo.mouth_width_norm if geo else None,
            "face_aspect_ratio": geo.face_aspect_ratio if geo else None,
            "pose_asym": qr.pose_asym,
        }

    def _write_frame(out_dir, aligned, landmarks, name="frame_{seq:03d}.jpg"):
        """Persist one aligned crop + its landmarks; return the frame filename."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "landmarks").mkdir(exist_ok=True)
        filename = name.format(seq=1)
        cv2.imwrite(str(out_dir / filename), aligned, [cv2.IMWRITE_JPEG_QUALITY, 97])
        np.save(out_dir / "landmarks" / filename.replace(".jpg", ".npy"), landmarks)
        return filename

    def _write_frame_seq(out_dir, aligned, landmarks, seq):
        """Persist one aligned crop + its landmarks with an explicit sequence
        number; return the frame filename."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "landmarks").mkdir(exist_ok=True)
        filename = f"frame_{seq:03d}.jpg"
        cv2.imwrite(str(out_dir / filename), aligned, [cv2.IMWRITE_JPEG_QUALITY, 97])
        np.save(out_dir / "landmarks" / filename.replace(".jpg", ".npy"), landmarks)
        return filename

    def process_image_impl(
        image: VideoRecord,
        cfg: Config,
        detector: FaceDetector,
        out_dir: Path,
    ) -> ExtractResult:
        """Process one still image as a one-frame media item.

        Images must not go through VideoCapture/probe. They still receive
        detection, quality filtering, alignment, landmarks, and geometry so
        they share the same training contract as extracted video frames.
        """
        image_path = cfg.input_dir / image.video_path
        base_row = _base_media_row(image)
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            base_row.update(
                {
                    "fps": None,
                    "frame_count": 1,
                    "duration": None,
                    "width": None,
                    "height": None,
                    "num_candidates_evaluated": 0,
                    "num_frames_accepted": 0,
                    "num_frames_selected": 0,
                    "usable_interval_start_idx": None,
                    "usable_interval_end_idx": None,
                    "status": "error",
                    "error_message": f"could not read image: {image_path}",
                }
            )
            return ExtractResult(base_row, [], [])

        height, width = frame_bgr.shape[:2]
        base_row.update(
            {
                "fps": None,
                "frame_count": 1,
                "duration": None,
                "width": width,
                "height": height,
            }
        )
        detections = detector.detect_batch([frame_bgr])
        tracked = track_primary_face([0], detections, cfg.iou_track_threshold)
        qcfg = QualityConfig.from_config(cfg)
        qr = evaluate(frame_bgr, tracked[0], qcfg)
        candidate = _candidate_row(
            image, 0, 0.0, tracked[0], qr, qr.accepted, 1 if qr.accepted else None
        )
        base_row.update(
            {
                "num_candidates_evaluated": 1,
                "num_frames_accepted": int(qr.accepted),
                "num_frames_selected": int(qr.accepted),
                "usable_interval_start_idx": 0 if qr.accepted else None,
                "usable_interval_end_idx": 0 if qr.accepted else None,
                "status": "ok" if qr.accepted else "no_usable_face",
                "error_message": None if qr.accepted else qr.reason,
            }
        )
        if not qr.accepted:
            return ExtractResult(base_row, [], [candidate])

        det: Detection = tracked[0].detection
        margin_scale = (
            adaptive_margin_scale(qr.face_bbox, frame_bgr.shape)
            if cfg.adaptive_crop_margin
            else cfg.crop_margin_scale
        )
        aligned = align_face(
            frame_bgr,
            det.landmarks,
            cfg.face_size,
            margin_scale=margin_scale,
            fallback_bbox=qr.face_bbox,
        )
        frame_filename = _write_frame(out_dir, aligned, det.landmarks, "frame_001.jpg")
        geo = compute_geometry(det.landmarks, qr.face_bbox)
        frame_row = _frame_row(image, 0, 0.0, 1, qr, geo, out_dir, frame_filename)
        return ExtractResult(base_row, [frame_row], [candidate])

    def process_video(
        video: VideoRecord,
        cfg: Config,
        detector: FaceDetector,
    ) -> ExtractResult:
        out_dir = frame_dir(cfg.output_dir, video.subject_id, video.video_id)
        fingerprint = compute_fingerprint(cfg)
        if cfg.skip_existing:
            cached = load_cached(out_dir, fingerprint)
            if cached is not None:
                return cached

        if video.media_type == "image":
            result = process_image_impl(video, cfg, detector, out_dir)
        else:
            result = process_video_impl(video, cfg, detector, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_cache(out_dir, result, fingerprint)
        return result

    def process_video_impl(
        video: VideoRecord,
        cfg: Config,
        detector: FaceDetector,
        out_dir: Path,
    ) -> ExtractResult:
        video_path = cfg.input_dir / video.video_path
        base_row = _base_media_row(video)

        try:
            vp = probe(video_path)
        except Exception as e:  # noqa: BLE001 - one bad video must not kill the run
            base_row.update(
                {
                    "fps": None,
                    "frame_count": None,
                    "duration": None,
                    "width": None,
                    "height": None,
                    "num_candidates_evaluated": 0,
                    "num_frames_accepted": 0,
                    "num_frames_selected": 0,
                    "usable_interval_start_idx": None,
                    "usable_interval_end_idx": None,
                    "status": "error",
                    "error_message": str(e),
                }
            )
            return ExtractResult(base_row, [], [])

        base_row.update(
            {
                "fps": vp.fps,
                "frame_count": vp.frame_count,
                "duration": vp.duration_sec,
                "width": vp.width,
                "height": vp.height,
            }
        )

        cand_idx = candidate_indices(vp.frame_count, cfg.num_candidates)
        raw_frames = read_frames_at(video_path, cand_idx, vp.fps)
        if not raw_frames:
            base_row.update(
                {
                    "num_candidates_evaluated": 0,
                    "num_frames_accepted": 0,
                    "num_frames_selected": 0,
                    "usable_interval_start_idx": None,
                    "usable_interval_end_idx": None,
                    "status": "no_usable_face",
                    "error_message": "no frames could be decoded",
                }
            )
            return ExtractResult(base_row, [], [])

        detections = detector.detect_batch([rf.image for rf in raw_frames])
        tracked = track_primary_face(
            [rf.frame_index for rf in raw_frames],
            detections,
            cfg.iou_track_threshold,
        )

        qcfg = QualityConfig.from_config(cfg)

        candidate_rows: list[dict] = []
        accepted: list[tuple] = []  # (raw_frame, tracked_frame, quality_result)
        for rf, tf in zip(raw_frames, tracked):
            qr = evaluate(rf.image, tf, qcfg)
            candidate_rows.append(
                _candidate_row(
                    video, rf.frame_index, rf.timestamp_sec, tf, qr, False, None
                )
            )
            if qr.accepted:
                accepted.append((rf, tf, qr))

        n_accepted = len(accepted)
        usable_start = accepted[0][0].frame_index if accepted else None
        usable_end = accepted[-1][0].frame_index if accepted else None

        frame_rows: list[dict] = []
        if n_accepted == 0:
            base_row.update(
                {
                    "num_candidates_evaluated": len(raw_frames),
                    "num_frames_accepted": 0,
                    "num_frames_selected": 0,
                    "usable_interval_start_idx": None,
                    "usable_interval_end_idx": None,
                    "status": "no_usable_face",
                    "error_message": None,
                }
            )
            return ExtractResult(base_row, [], candidate_rows)

        rng = seeded_rng(cfg.seed, video.video_id)
        weights = np.array(
            [a[2].detector_confidence or 0.0 for a in accepted], dtype=float
        )
        strategy_fn = STRATEGIES[cfg.sampling_strategy]
        picked_positions = strategy_fn(n_accepted, cfg.num_frames, rng, weights)

        chosen_frame_indices = {accepted[p][0].frame_index for p in picked_positions}
        # frame_index -> 1-based sequence position, so the candidate audit log
        # and the frame rows agree on ordering
        chosen_order = {
            accepted[p][0].frame_index: seq
            for seq, p in enumerate(picked_positions, start=1)
        }
        for row in candidate_rows:
            if row["frame_index"] in chosen_frame_indices:
                row["chosen"] = True
                row["sequence_position"] = chosen_order[row["frame_index"]]

        for seq, pos in enumerate(picked_positions, start=1):
            rf, tf, qr = accepted[pos]
            det: Detection = tf.detection
            margin_scale = (
                adaptive_margin_scale(qr.face_bbox, rf.image.shape)
                if cfg.adaptive_crop_margin
                else cfg.crop_margin_scale
            )
            aligned = align_face(
                rf.image,
                det.landmarks,
                cfg.face_size,
                margin_scale=margin_scale,
                fallback_bbox=qr.face_bbox,
            )

            frame_filename = _write_frame_seq(out_dir, aligned, det.landmarks, seq)
            geo = compute_geometry(det.landmarks, qr.face_bbox)
            frame_rows.append(
                _frame_row(
                    video,
                    rf.frame_index,
                    rf.timestamp_sec,
                    seq,
                    qr,
                    geo,
                    out_dir,
                    frame_filename,
                )
            )

        status = "ok"
        if n_accepted < cfg.min_track_len:
            status = "insufficient_frames"

        base_row.update(
            {
                "num_candidates_evaluated": len(raw_frames),
                "num_frames_accepted": n_accepted,
                "num_frames_selected": len(frame_rows),
                "usable_interval_start_idx": usable_start,
                "usable_interval_end_idx": usable_end,
                "status": status,
                "error_message": None,
            }
        )
        return ExtractResult(base_row, frame_rows, candidate_rows)

    return ExtractResult, process_video


@app.cell
def metadata_writer():
    # Assembles + writes metadata/{videos,frames,...}.csv.
    def write_metadata(
        output_dir: Path,
        video_rows: list[dict],
        frame_rows: list[dict],
        candidate_rows: list[dict],
    ) -> dict[str, pd.DataFrame]:
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        videos_df = pd.DataFrame(video_rows)
        frames_df = pd.DataFrame(frame_rows)
        candidates_df = pd.DataFrame(candidate_rows)
        subjects_df = build_subjects(videos_df, frames_df)

        videos_df.to_csv(meta_dir / "videos.csv", index=False)
        frames_df.to_csv(meta_dir / "frames.csv", index=False)
        candidates_df.to_csv(meta_dir / "candidates.csv", index=False)
        subjects_df.to_csv(meta_dir / "subjects.csv", index=False)

        return {
            "videos": videos_df,
            "frames": frames_df,
            "candidates": candidates_df,
            "subjects": subjects_df,
        }

    def build_subjects(
        videos_df: pd.DataFrame, frames_df: pd.DataFrame
    ) -> pd.DataFrame:
        if videos_df.empty:
            return pd.DataFrame(columns=["subject_id", "num_videos"])

        rows = []
        for subject_id, g in videos_df.groupby("subject_id"):
            frame_g = (
                frames_df[frames_df["subject_id"] == subject_id]
                if not frames_df.empty
                else frames_df
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "num_videos": len(g),
                    "num_genuine_videos": int((g["label"] == "genuine").sum()),
                    "num_spoof_videos": int((g["label"] == "spoof").sum()),
                    "attack_types_present": ",".join(sorted(g["attack_type"].unique())),
                    "devices_present": ",".join(
                        sorted(d for d in g["device"].unique() if d)
                    ),
                    "num_frames_selected_total": len(frame_g),
                }
            )
        return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)

    def validate_csv_schema(output_dir: Path) -> None:
        """Check that every written CSV has the columns downstream expects.

        A schema drift between preprocessing and training surfaces as a
        confusing KeyError deep in the notebook. Failing here names the
        missing column and the file, which is the whole difference.
        """
        required = {
            "videos.csv": [
                "subject_id",
                "video_id",
                "label",
                "attack_type",
                "dataset_source",
                "split_role",
            ],
            "frames.csv": [
                "subject_id",
                "video_id",
                "label",
                "attack_type",
                "dataset_source",
                "frame_path",
                "interocular_dist_norm",
                "eye_to_nose_dist_norm",
                "nose_to_mouth_dist_norm",
                "eye_to_mouth_dist_norm",
                "mouth_width_norm",
                "face_aspect_ratio",
                "pose_asym",
            ],
        }
        meta_dir = output_dir / "metadata"
        problems = []
        for fname, cols in required.items():
            path = meta_dir / fname
            if not path.is_file():
                problems.append(f"{path}: file missing")
                continue
            with path.open(newline="") as f:
                actual = set(csv.DictReader(f).fieldnames or [])
            missing = [c for c in cols if c not in actual]
            if missing:
                problems.append(f"{fname}: missing columns {missing}")
        if problems:
            raise RuntimeError(
                "CSV schema validation failed:\n  " + "\n  ".join(problems)
            )

    return validate_csv_schema, write_metadata


@app.cell
def balance():
    # Subject/label-balanced sampling and per-frame weights.
    try:
        from torch.utils.data import Sampler
    except ImportError:  # torch is a hard dep here, but keep this importable without it
        Sampler = object  # type: ignore[assignment,misc]

    class SubjectBalancedSampler(Sampler):
        """Yields row-positions into `frames_df` (0-indexed, matching a
        Dataset built directly from that dataframe) via subject -> video ->
        frame hierarchical sampling, for `num_samples` draws per epoch.
        """

        def __init__(
            self,
            frames_df: pd.DataFrame,
            num_samples: int | None = None,
            seed: int = 42,
        ) -> None:
            self.frames_df = frames_df.reset_index(drop=True)
            self.num_samples = (
                num_samples if num_samples is not None else len(self.frames_df)
            )
            self.seed = seed
            self._epoch = 0

            self._by_subject: dict[str, dict[str, list[int]]] = {}
            for subject_id, sub_g in self.frames_df.groupby("subject_id"):
                videos: dict[str, list[int]] = {}
                for video_id, vid_g in sub_g.groupby("video_id"):
                    videos[video_id] = vid_g.index.tolist()
                self._by_subject[subject_id] = videos
            self._subjects = sorted(self._by_subject)

        def set_epoch(self, epoch: int) -> None:
            self._epoch = epoch

        def __len__(self) -> int:
            return self.num_samples

        def __iter__(self) -> Iterator[int]:
            rng = np.random.default_rng(self.seed + self._epoch)
            subjects = self._subjects
            for _ in range(self.num_samples):
                subject_id = subjects[rng.integers(len(subjects))]
                videos = self._by_subject[subject_id]
                video_ids = list(videos)
                video_id = video_ids[rng.integers(len(video_ids))]
                frame_positions = videos[video_id]
                yield frame_positions[rng.integers(len(frame_positions))]

    class LabelSubjectBalancedSampler(Sampler):
        """Like `SubjectBalancedSampler`, with a label draw on top:

            sample label uniformly (genuine/spoof)
                -> sample subject uniformly (within that label)
                    -> sample video uniformly (within that subject)
                        -> sample frame uniformly (within that video)

        `SubjectBalancedSampler` alone only fixes videos-per-subject skew;
        it does nothing about label skew. On this dataset that matters a
        lot: only ~2-3 subjects are genuine at all, so a plain
        subject-uniform draw over e.g. 35 train subjects puts genuine frames
        in ~6% of draws, not 50%. With that few genuine frames, BCE has a
        near-zero-loss trivial solution ("always predict spoof") -- that's
        what a collapsed model (BPCER=1.0, train_loss near 0) usually means,
        not a code bug.
        """

        def __init__(
            self,
            frames_df: pd.DataFrame,
            label_col: str,
            num_samples: int | None = None,
            seed: int = 42,
        ) -> None:
            self.frames_df = frames_df.reset_index(drop=True)
            self.num_samples = (
                num_samples if num_samples is not None else len(self.frames_df)
            )
            self.seed = seed
            self._epoch = 0

            self._by_label: dict[object, dict[str, dict[str, list[int]]]] = {}
            for label, label_g in self.frames_df.groupby(label_col):
                by_subject: dict[str, dict[str, list[int]]] = {}
                for subject_id, sub_g in label_g.groupby("subject_id"):
                    videos: dict[str, list[int]] = {}
                    for video_id, vid_g in sub_g.groupby("video_id"):
                        videos[video_id] = vid_g.index.tolist()
                    by_subject[subject_id] = videos
                self._by_label[label] = by_subject
            self._labels = sorted(self._by_label, key=str)

        def set_epoch(self, epoch: int) -> None:
            self._epoch = epoch

        def __len__(self) -> int:
            return self.num_samples

        def __iter__(self) -> Iterator[int]:
            rng = np.random.default_rng(self.seed + self._epoch)
            labels = self._labels
            for _ in range(self.num_samples):
                by_subject = self._by_label[labels[rng.integers(len(labels))]]
                subjects = list(by_subject)
                videos = by_subject[subjects[rng.integers(len(subjects))]]
                video_ids = list(videos)
                frame_positions = videos[video_ids[rng.integers(len(video_ids))]]
                yield frame_positions[rng.integers(len(frame_positions))]

    def compute_frame_weights(frames_df: pd.DataFrame) -> pd.Series:
        """Per-frame weight s.t. expected sampling probability is uniform
        across subjects, then uniform across videos within a subject, then
        uniform across frames within a video. Usable with
        WeightedRandomSampler."""
        num_subjects = frames_df["subject_id"].nunique()
        videos_per_subject = frames_df.groupby("subject_id")["video_id"].transform(
            "nunique"
        )
        frames_per_video = frames_df.groupby("video_id")["frame_id"].transform("count")
        weight = 1.0 / (num_subjects * videos_per_subject * frames_per_video)
        return weight

    def videos_per_subject_stats(videos_df: pd.DataFrame) -> dict:
        counts = videos_df.groupby("subject_id").size()
        return {
            "num_subjects": int(counts.shape[0]),
            "min": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "std": float(counts.std(ddof=0)),
        }

    return compute_frame_weights, videos_per_subject_stats


@app.cell
def splits():
    # Subject-disjoint 5-fold split generator.
    def subject_table(videos_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for subject_id, g in videos_df.groupby("subject_id"):
            label_counts = Counter(g["label"])
            majority_label = label_counts.most_common(1)[0][0]
            attack_counts = Counter(g["attack_type"])
            majority_attack = attack_counts.most_common(1)[0][0]
            rows.append(
                {
                    "subject_id": subject_id,
                    "num_videos": len(g),
                    "majority_label": majority_label,
                    "majority_attack_type": majority_attack,
                }
            )
        return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)

    def stratified_or_plain_fold(
        subjects: pd.DataFrame, y_col: str, n_splits: int, seed: int
    ):
        """Returns a list of (train_idx, test_idx) over `subjects`,
        stratified if feasible, else a warning + plain shuffled KFold."""
        if len(subjects) < 2:
            all_idx = subjects.index.to_numpy()
            return [(all_idx, all_idx[:0])]
        if n_splits > len(subjects):
            warnings.warn(
                f"requested n_splits={n_splits} but only {len(subjects)} subject(s) available; "
                f"clamping to {len(subjects)}.",
                stacklevel=2,
            )
            n_splits = len(subjects)
        y = subjects[y_col]
        min_class_count = y.value_counts().min()
        if min_class_count >= n_splits:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(skf.split(subjects, y))
        warnings.warn(
            f"cannot stratify {n_splits}-fold by {y_col!r}: smallest class has only "
            f"{min_class_count} subject(s). Falling back to a plain (unstratified) "
            f"shuffled KFold over subjects -- subject-disjointness is preserved, "
            f"class balance across folds is not guaranteed. Check the split-report "
            f"for per-fold composition.",
            stacklevel=2,
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(subjects))

    def generate_folds(
        videos_df: pd.DataFrame,
        num_folds: int,
        seed: int,
    ) -> list[dict]:
        """Returns one dict per fold: {fold, train_subjects, val_subjects}.

        There is deliberately NO test split. `test_data_ibeta` is the single
        held-out test set and lives outside this pipeline entirely; emitting a
        per-fold test split here would recreate the two-test-set setup the
        supervisor ruled out. The held-out fold is therefore *validation* --
        it drives early stopping and threshold calibration -- not a second
        test set, and must never be reported as a result.

        Consequence worth knowing: val is now 1/num_folds of subjects (~20% at
        5 folds), up from ~13% under the old outer-test + inner-val scheme. A
        larger val set makes the calibrated threshold less noisy, which
        matters because fold-to-fold thresholds were previously observed
        swinging 0.366-0.691 on one recipe.
        """
        subjects = subject_table(videos_df)
        outer_splits = stratified_or_plain_fold(
            subjects, "majority_label", num_folds, seed
        )

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
            folds.append(
                {
                    "fold": fold_idx + 1,
                    "train_subjects": sorted(
                        subjects.iloc[train_idx]["subject_id"].tolist()
                    ),
                    "val_subjects": sorted(
                        subjects.iloc[val_idx]["subject_id"].tolist()
                    ),
                }
            )
        return folds

    def verify_no_leakage(fold: dict) -> list[str]:
        """Returns a list of human-readable leakage errors (empty if none).

        Only train/val are checked -- there is no test split to leak into.
        """
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        errors = []
        if train & val:
            errors.append(f"fold {fold['fold']}: train ∩ val = {sorted(train & val)}")
        return errors

    def write_splits(
        output_dir: Path, videos_df: pd.DataFrame, folds: list[dict]
    ) -> list[str]:
        """Writes splits/fold_k/{train,val}.csv (video-level rows, joinable to
        frames.csv on video_id). No test.csv is written -- see generate_folds.

        A stale test.csv from an earlier run is deleted if present, so a
        rebuilt dataset can never leave a reader (or a notebook) picking up a
        test split that no longer has a definition.
        """
        all_errors: list[str] = []
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        cols = [
            "subject_id",
            "video_id",
            "label",
            "attack_type",
            "pai_family",
            "dataset_source",
            "device",
        ]
        cols = [c for c in cols if c in videos_df.columns]

        for fold in folds:
            errors = verify_no_leakage(fold)
            all_errors.extend(errors)
            fold_dir = splits_dir / f"fold_{fold['fold']}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            stale_test = fold_dir / "test.csv"
            if stale_test.exists():
                stale_test.unlink()
            for split_name in ("train", "val"):
                subject_ids = set(fold[f"{split_name}_subjects"])
                sub_df = videos_df[videos_df["subject_id"].isin(subject_ids)][cols]
                sub_df.to_csv(fold_dir / f"{split_name}.csv", index=False)

        return all_errors

    return generate_folds, verify_no_leakage, write_splits


@app.cell
def report(verify_no_leakage, videos_per_subject_stats):
    # Dataset quality report.
    def build_report(
        videos_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        folds: list[dict],
    ) -> dict:
        totals = {
            "total_subjects": int(videos_df["subject_id"].nunique())
            if not videos_df.empty
            else 0,
            "total_videos": len(videos_df),
            "total_genuine_videos": int((videos_df["label"] == "genuine").sum())
            if not videos_df.empty
            else 0,
            "total_spoof_videos": int((videos_df["label"] == "spoof").sum())
            if not videos_df.empty
            else 0,
            "total_candidate_frames_evaluated": len(candidates_df),
            "total_frames_selected": len(frames_df),
            "note": (
                "total_frames_selected counts extracted face crops, NOT independent "
                "training samples: frames within a video (and videos within a subject) "
                "are highly correlated. Use subject-balanced sampling (see balance "
                "section) and subject-disjoint splits (see splits section) for "
                "training/evaluation."
            ),
        }

        subject_distribution = (
            videos_per_subject_stats(videos_df) if not videos_df.empty else {}
        )

        attack_distribution = []
        if not videos_df.empty:
            for attack_type, g in videos_df.groupby("attack_type"):
                attack_distribution.append(
                    {
                        "attack_type": attack_type,
                        "num_videos": len(g),
                        "num_subjects": int(g["subject_id"].nunique()),
                    }
                )
            attack_distribution.sort(key=lambda r: -r["num_videos"])

        video_quality = {
            "videos_with_no_usable_face": int(
                (videos_df["status"] == "no_usable_face").sum()
            )
            if not videos_df.empty
            else 0,
            "videos_with_insufficient_frames": int(
                (videos_df["status"] == "insufficient_frames").sum()
            )
            if not videos_df.empty
            else 0,
            "videos_with_error": int((videos_df["status"] == "error").sum())
            if not videos_df.empty
            else 0,
            "average_face_width_px": float(frames_df["face_width"].mean())
            if not frames_df.empty
            else None,
            "average_face_height_px": float(frames_df["face_height"].mean())
            if not frames_df.empty
            else None,
            "average_blur_score": float(frames_df["blur_score"].mean())
            if not frames_df.empty
            else None,
            "average_detection_confidence": float(
                frames_df["detector_confidence"].mean()
            )
            if not frames_df.empty
            else None,
        }

        device_distribution = []
        environment_distribution = []
        if not videos_df.empty:
            for device, g in videos_df.groupby("device"):
                device_distribution.append(
                    {"device": device or "unknown", "num_videos": len(g)}
                )
            if videos_df["environment"].notna().any():
                for env, g in videos_df.groupby("environment", dropna=False):
                    environment_distribution.append(
                        {
                            "environment": env if pd.notna(env) else "unknown",
                            "num_videos": len(g),
                        }
                    )
            else:
                environment_distribution = [
                    {"environment": "unknown", "num_videos": len(videos_df)}
                ]

        split_verification = []
        for fold in folds:
            errors = verify_no_leakage(fold)
            split_verification.append(
                {
                    "fold": fold["fold"],
                    "num_train_subjects": len(fold["train_subjects"]),
                    "num_val_subjects": len(fold["val_subjects"]),
                    "leakage_errors": errors,
                    "ok": len(errors) == 0,
                }
            )

        return {
            "totals": totals,
            "subject_distribution": subject_distribution,
            "attack_distribution": attack_distribution,
            "video_quality": video_quality,
            "device_distribution": device_distribution,
            "environment_distribution": environment_distribution,
            "split_verification": split_verification,
        }

    def write_report(output_dir: Path, report: dict) -> None:
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        (reports_dir / "dataset_summary.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

        flat_rows = []
        for section, content in report.items():
            if isinstance(content, dict):
                for k, v in content.items():
                    flat_rows.append({"section": section, "key": k, "value": v})
            elif isinstance(content, list):
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        for k, v in item.items():
                            flat_rows.append(
                                {
                                    "section": f"{section}[{i}]",
                                    "key": k,
                                    "value": v,
                                }
                            )
        pd.DataFrame(flat_rows).to_csv(reports_dir / "dataset_summary.csv", index=False)

        any_leakage = any(not s["ok"] for s in report["split_verification"])
        if any_leakage:
            raise RuntimeError(
                "SUBJECT LEAKAGE DETECTED across splits -- see reports/dataset_summary.json "
                "-> split_verification. Refusing to treat the split output as valid."
            )

    return build_report, write_report


@app.cell
def contact_sheets(read_frames_at):
    # Contact-sheet visualizations.
    FONT = ImageFont.load_default()
    CELL = 160
    PAD = 6
    CAPTION_H = 28

    def thumb(img: Image.Image, size: int = CELL) -> Image.Image:
        img = img.copy()
        img.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (32, 32, 32))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    def grid(
        cells: list[tuple[Image.Image, str]], cols: int, title: str
    ) -> Image.Image:
        rows = max(1, (len(cells) + cols - 1) // cols)
        cell_h = CELL + CAPTION_H
        title_h = 32
        W = cols * (CELL + PAD) + PAD
        H = rows * (cell_h + PAD) + PAD + title_h
        sheet = Image.new("RGB", (W, H), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        draw.text((PAD, 6), title, fill=(255, 255, 255), font=FONT)

        for i, (img, caption) in enumerate(cells):
            r, c = divmod(i, cols)
            x = PAD + c * (CELL + PAD)
            y = title_h + PAD + r * (cell_h + PAD)
            sheet.paste(thumb(img), (x, y))
            draw.text(
                (x, y + CELL + 2),
                caption[:26],
                fill=(230, 230, 230),
                font=FONT,
            )
        return sheet

    def sample_distinct(
        df: pd.DataFrame, n: int, distinct_cols: list[str]
    ) -> pd.DataFrame:
        if df.empty:
            return df
        deduped = df.drop_duplicates(subset=distinct_cols)
        return deduped.sample(n=min(n, len(deduped)), random_state=0)

    def genuine_and_spoof_sheets(
        frames_df: pd.DataFrame,
        out_dir: Path,
        input_output_dir: Path,
        examples_per_group: int,
    ) -> None:
        if frames_df.empty:
            return
        genuine = frames_df[frames_df["label"] == "genuine"]
        sample = sample_distinct(
            genuine, examples_per_group, ["subject_id", "video_id"]
        )
        cells = []
        for _, row in sample.iterrows():
            img_path = input_output_dir / row["frame_path"]
            if img_path.exists():
                cells.append(
                    (
                        Image.open(img_path),
                        f"{row['subject_id']}/{row['video_id']}",
                    )
                )
        if cells:
            grid(cells, cols=min(6, len(cells)), title="genuine examples").save(
                out_dir / "genuine_examples.jpg", quality=92
            )

        for attack_type, g in frames_df[frames_df["label"] == "spoof"].groupby(
            "attack_type"
        ):
            sample = sample_distinct(g, examples_per_group, ["subject_id", "video_id"])
            cells = []
            for _, row in sample.iterrows():
                img_path = input_output_dir / row["frame_path"]
                if img_path.exists():
                    cells.append(
                        (
                            Image.open(img_path),
                            f"{row['subject_id']}/{row['video_id']}",
                        )
                    )
            if cells:
                safe = str(attack_type).replace(" ", "_").replace("/", "_")
                grid(
                    cells,
                    cols=min(6, len(cells)),
                    title=f"spoof: {attack_type}",
                ).save(out_dir / f"spoof_{safe}.jpg", quality=92)

    def rejected_examples_sheet(
        candidates_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        out_dir: Path,
        input_dir: Path,
        examples_per_reason: int,
    ) -> None:
        if candidates_df.empty:
            return
        rejected = candidates_df[candidates_df["quality_status"] == "rejected"]
        if rejected.empty:
            return
        video_path_by_id = videos_df.set_index("video_id")["video_path"].to_dict()
        fps_by_id = videos_df.set_index("video_id")["fps"].to_dict()

        cells = []
        for reason, g in rejected.groupby("rejection_reason"):
            sample = sample_distinct(g, examples_per_reason, ["subject_id", "video_id"])
            for _, row in sample.iterrows():
                vid = row["video_id"]
                vpath = input_dir / video_path_by_id.get(vid, "")
                if not vpath.exists():
                    continue
                raw = read_frames_at(
                    vpath, [int(row["frame_index"])], fps_by_id.get(vid) or 0.0
                )
                if not raw:
                    continue
                frame_bgr = raw[0].image
                bbox = row.get("face_bbox")
                crop = frame_bgr
                if isinstance(bbox, str) and bbox not in ("", "nan"):
                    try:
                        x1, y1, x2, y2 = ast.literal_eval(bbox)
                        crop = frame_bgr[max(0, y1) : y2, max(0, x1) : x2]
                        if crop.size == 0:
                            crop = frame_bgr
                    except Exception:  # noqa: BLE001
                        crop = frame_bgr
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cells.append((Image.fromarray(rgb), f"{reason}"))

        if cells:
            grid(
                cells,
                cols=min(6, len(cells)),
                title="rejected examples (by reason)",
            ).save(out_dir / "rejected_examples.jpg", quality=92)

    def eight_frame_sequence_sheets(
        frames_df: pd.DataFrame,
        out_dir: Path,
        input_output_dir: Path,
        num_videos: int,
    ) -> None:
        if frames_df.empty:
            return
        video_ids = (
            frames_df["video_id"]
            .drop_duplicates()
            .sample(
                n=min(num_videos, frames_df["video_id"].nunique()),
                random_state=0,
            )
        )
        for vid in video_ids:
            g = frames_df[frames_df["video_id"] == vid].sort_values("sequence_position")
            cells = []
            for _, row in g.iterrows():
                img_path = input_output_dir / row["frame_path"]
                if img_path.exists():
                    cells.append((Image.open(img_path), f"t{row['sequence_position']}"))
            if cells:
                grid(cells, cols=len(cells), title=f"{vid} (chronological)").save(
                    out_dir / f"sequence_{vid}.jpg", quality=92
                )

    def build_contact_sheets(
        output_dir: Path,
        input_dir: Path,
        videos_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        examples_per_group: int = 6,
    ) -> None:
        out_dir = output_dir / "reports" / "contact_sheets"
        out_dir.mkdir(parents=True, exist_ok=True)
        genuine_and_spoof_sheets(frames_df, out_dir, output_dir, examples_per_group)
        rejected_examples_sheet(
            candidates_df, videos_df, out_dir, input_dir, examples_per_group
        )
        eight_frame_sequence_sheets(
            frames_df, out_dir, output_dir, num_videos=examples_per_group
        )

    return (build_contact_sheets,)


@app.cell
def pipeline(
    Config,
    ExtractResult,
    FaceDetector,
    build_contact_sheets,
    build_report,
    compute_frame_weights,
    generate_folds,
    load_video_records,
    process_video,
    validate_csv_schema,
    videos_per_subject_stats,
    write_metadata,
    write_report,
    write_splits,
):
    # Top-level orchestrator (run).
    def load_existing(output_dir: Path) -> dict[str, pd.DataFrame]:
        meta_dir = output_dir / "metadata"
        dfs = {}
        for name in ("videos", "frames", "candidates", "subjects"):
            path = meta_dir / f"{name}.csv"
            dfs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return dfs

    def run(cfg: Config) -> None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(cfg.output_dir / "config.json")

        dfs: dict[str, pd.DataFrame] = {}

        if "extract" in cfg.stages:
            records = load_video_records(
                cfg.metadata_csv,
                cfg.input_dir,
                cfg.dataset_source,
                cfg.subpool_rules,
            )
            records = sorted(records, key=lambda r: r.video_id)
            if cfg.limit_videos is not None:
                records = records[: cfg.limit_videos]

            detector = FaceDetector(device=cfg.device, det_thresh=cfg.det_thresh)
            video_rows, frame_rows, candidate_rows = [], [], []
            for video in tqdm(records, desc="extracting frames"):
                result: ExtractResult = process_video(video, cfg, detector)
                video_rows.append(result.video_row)
                frame_rows.extend(result.frame_rows)
                candidate_rows.extend(result.candidate_rows)

            dfs["videos"] = pd.DataFrame(video_rows)
            dfs["frames"] = pd.DataFrame(frame_rows)
            dfs["candidates"] = pd.DataFrame(candidate_rows)

        if "metadata" in cfg.stages:
            if "videos" not in dfs:
                existing = load_existing(cfg.output_dir)
                dfs.update(existing)
            written = write_metadata(
                cfg.output_dir,
                dfs["videos"].to_dict("records"),
                dfs["frames"].to_dict("records"),
                dfs["candidates"].to_dict("records"),
            )
            dfs.update(written)
            validate_csv_schema(cfg.output_dir)
        elif "videos" not in dfs:
            dfs.update(load_existing(cfg.output_dir))

        folds: list[dict] = []
        if "splits" in cfg.stages:
            folds = generate_folds(dfs["videos"], cfg.num_folds, cfg.seed)
            errors = write_splits(cfg.output_dir, dfs["videos"], folds)
            if errors:
                raise RuntimeError(
                    "subject leakage detected while writing splits:\n"
                    + "\n".join(errors)
                )
            print(
                f"wrote {len(folds)} subject-disjoint folds to {cfg.output_dir / 'splits'}"
            )

        if "balance" in cfg.stages:
            stats = videos_per_subject_stats(dfs["videos"])
            print("videos per subject:", stats)
            frames_with_w = dfs["frames"].copy()
            frames_with_w["sample_weight"] = compute_frame_weights(frames_with_w)
            out = cfg.output_dir / "metadata" / "frame_weights.csv"
            frames_with_w[
                ["frame_id", "subject_id", "video_id", "sample_weight"]
            ].to_csv(out, index=False)
            print(f"wrote {out}")

        if "report" in cfg.stages:
            rep = build_report(dfs["videos"], dfs["frames"], dfs["candidates"], folds)
            write_report(cfg.output_dir, rep)
            print(f"wrote report to {cfg.output_dir / 'reports'}")

        if "contact_sheets" in cfg.stages:
            build_contact_sheets(
                cfg.output_dir,
                cfg.input_dir,
                dfs["videos"],
                dfs["frames"],
                dfs["candidates"],
                examples_per_group=cfg.contact_sheet_examples,
            )
            print(
                f"wrote contact sheets to {cfg.output_dir / 'reports' / 'contact_sheets'}"
            )

    return (run,)


@app.cell
def direct_ingest_helpers(Detection, FaceDetector, REF_112, compute_geometry):
    # Direct-ingest helpers: naming, landmark recovery, the item record.
    # Every image extension the direct-ingest pools may use.
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def ingest_slug(value: str) -> str:
        return SAFE.sub("_", value).strip("_")

    def canonical_attack(value: str) -> str:
        """Map raw PAI-type tokens from filenames to the canonical attack_type
        vocabulary (shared with fas_ibeta_l1 and test_data_ibeta).

        Used only for `frames_cleaned_cropped_v2`, whose filenames encode the
        attack species. test_data_ibeta gets its species from the CATEGORY
        DIRECTORY instead, and custom_pad_dataset from its metadata.csv --
        neither needs (or should use) filename guessing.

        An unrecognized token is raised as a ValueError rather than silently
        mapped to "unknown": a silent fallback lets a typo in a filename
        propagate into training data as an attack_type the model has never
        seen, which is exactly how a pool ends up untestable.
        """
        value = (value or "unknown").lower()
        mapping = {
            "printed_photo": "print",
            "print_photo": "print",
            "printout": "print",
            "ipad_video": "replay",
            "tablet_video": "replay",
            "laptop_video": "replay",
            "iphone_video": "replay",
            "phone_video": "replay",
            "mobile_replay": "replay",
            "replay_video": "replay",
        }
        if value in mapping:
            return mapping[value]
        raise ValueError(
            f"canonical_attack: unrecognized attack token {value!r}. "
            f"Known tokens: {sorted(mapping)}"
        )

    @dataclass(frozen=True)
    class ExistingCrop:
        """Result of detecting landmarks on an already-cropped image.

        `landmarks` is ALWAYS a valid (5, 2) array -- real, or synthetic from
        the image dimensions on a detector miss -- so it can be written to
        `.npy` without breaking downstream visualization. `geometry` is None on
        a miss (the model already treats missing geometry as 0.0), and
        `detector_confidence` is None in the same case, which is how a missed
        detection stays auditable rather than being silently accepted.
        """

        detection: Detection | None
        bbox: tuple[int, int, int, int]
        landmarks: np.ndarray
        geometry: "GeometryFeatures | None"
        pose_asym: float | None

        @property
        def detected(self) -> bool:
            return self.detection is not None

    # Create a shared detector instance for landmark recovery on pre-cropped images.
    detector = FaceDetector(device="cuda", det_thresh=0.05)

    def detect_existing_crop(path: Path) -> "ExistingCrop":
        """Run RetinaFace on a pre-cropped image to get landmarks + geometry.

        A miss (the image is already a face crop by construction, so this
        usually means a detection-threshold issue rather than the absence of a
        face) degrades to the full-image bbox with synthetic landmarks and
        geometry=None rather than dropping the image.
        """
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read pre-cropped image: {path}")
        height, width = image.shape[:2]
        try:
            detections = detector.detect_batch([image])
        except Exception as e:
            print(
                f"  WARNING: RetinaFace raised {type(e).__name__} on "
                f"{path.name} (falling back to full-image bbox): {e}"
            )
            detections = []

        def _synthetic_landmarks(h: int, w: int) -> np.ndarray:
            scale = min(w, h) / 112.0
            ref = REF_112 * scale
            cx, cy = w / 2.0, h / 2.0
            return ref - ref.mean(axis=0) + np.array([cx, cy], dtype=np.float32)

        if not detections or not detections[0]:
            print(
                f"  WARNING: RetinaFace found no face in pre-cropped image "
                f"(falling back to full-image bbox): {path}"
            )
            return ExistingCrop(
                None,
                (0, 0, width, height),
                _synthetic_landmarks(height, width),
                None,
                None,
            )

        detection = max(
            detections[0],
            key=lambda item: (
                item.prob
                * max(
                    0.0,
                    (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]),
                )
            ),
        )
        x1, y1, x2, y2 = detection.box
        bbox = (
            max(0, min(width, round(x1))),
            max(0, min(height, round(y1))),
            max(0, min(width, round(x2))),
            max(0, min(height, round(y2))),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            print(
                f"  WARNING: invalid face bbox from RetinaFace "
                f"(falling back to full-image bbox): {path}"
            )
            return ExistingCrop(
                None,
                (0, 0, width, height),
                _synthetic_landmarks(height, width),
                None,
                None,
            )

        pose = None
        landmarks = detection.landmarks
        has_real_landmarks = landmarks is not None and len(landmarks) >= 5
        if not has_real_landmarks:
            landmarks = _synthetic_landmarks(height, width)
        if has_real_landmarks:
            left_eye, right_eye, nose = landmarks[:3]
            interocular = float(np.linalg.norm(left_eye - right_eye))
            if interocular > 1e-6:
                pose = (
                    abs(
                        abs(float(nose[0]) - float(left_eye[0]))
                        - abs(float(right_eye[0]) - float(nose[0]))
                    )
                    / interocular
                )
        geometry = compute_geometry(landmarks, bbox) if has_real_landmarks else None
        return ExistingCrop(detection, bbox, landmarks, geometry, pose)

    @dataclass(frozen=True)
    class DirectItem:
        path: Path
        subject_id: str
        video_id: str
        label: str  # "genuine" | "spoof"
        attack_type: str
        device: str
        frame_index: int

    return DirectItem, IMAGE_EXTS, canonical_attack, detect_existing_crop


@app.cell
def direct_ingest_readers(DirectItem, IMAGE_EXTS, canonical_attack):
    def iter_files(root: Path) -> list[Path]:
        return sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def iter_convention_items(
        input_dir: Path,
        dataset_source: str,
        split_role: str,
        nested_attack: bool,
    ) -> list[DirectItem]:
        """Directory-convention layouts.

        nested_attack=False -> <subject>/<label>/*.ext   (frames_cleaned_cropped_v2)
        nested_attack=True  -> test_data_ibeta layout:
            live:  <subject>/live/<clip>/<frame>     (4 levels)
            spoof: <subject>/spoof/<spoof_type>/<clip>/<frame>  (5 levels)
        """
        items: list[DirectItem] = []
        for path in iter_files(input_dir):
            parts = path.relative_to(input_dir).parts
            if len(parts) < 3:
                continue
            subject_dir, label_dir = parts[0], parts[1]
            if label_dir not in {"live", "spoof"}:
                continue
            label = "genuine" if label_dir == "live" else "spoof"

            if nested_attack:
                # Asymmetric depth: live has clip at parts[2], spoof at parts[3]
                if label == "genuine":
                    # parts = [subject, live, clip, frame...]
                    if len(parts) < 4:
                        continue
                    clip_name = parts[2]
                    attack = "live"
                else:
                    # parts = [subject, spoof, spoof_type, clip, frame...]
                    if len(parts) < 5:
                        continue
                    attack = parts[2]  # spoof_type (e.g., print, print_cutouts, replay)
                    clip_name = parts[3]

                # Extract frame_index from filename (e.g., frame_001.jpg)
                frame_match = re.search(r"_frame_(\d+)$", path.stem)
                frame_index = int(frame_match.group(1)) if frame_match else 0

                subject_id = f"official_test_{subject_dir}"
                # video_id = subject + clip, so all frames in same clip share video_id
                video_id = f"{subject_id}_{SAFE.sub('_', clip_name).strip('_')}"
                device = "unknown"
            else:
                # frames_cleaned_cropped_v2: flat <subject>/<label>/<frame>
                if label == "genuine":
                    attack = "live"
                else:
                    stem = re.sub(r"_frame_\d+$", "", path.stem)
                    token = stem.split("_SD_", 1)[-1].split("_scene", 1)[0]
                    attack = canonical_attack(token)

                frame_match = re.search(r"_frame_(\d+)$", path.stem)
                frame_index = int(frame_match.group(1)) if frame_match else 0

                subject_id = f"{dataset_source}_{subject_dir}"
                video_id = f"{dataset_source}_{subject_dir}_{SAFE.sub('_', re.sub(r'_frame_\d+$', '', path.stem)).strip('_')}"
                dev = re.search(r"_(android|laptop)_", path.stem)
                device = dev.group(1) if dev else "unknown"

            items.append(
                DirectItem(
                    path=path,
                    subject_id=subject_id,
                    video_id=video_id,
                    label=label,
                    attack_type=attack,
                    device=device,
                    frame_index=frame_index,
                )
            )
        return items

    def iter_metadata_items(
        input_dir: Path, dataset_source: str, split_role: str
    ) -> list[DirectItem]:
        """metadata.csv-driven layout (custom_pad_dataset).

        Labels are NOT inferred here -- the CSV is the sole source of truth,
        matching every other pool. Built by scripts/build_custom_pad_metadata.py.
        """
        meta_csv = input_dir / "metadata.csv"
        if not meta_csv.exists():
            raise FileNotFoundError(
                f"{meta_csv} not found -- build it first with "
                f"scripts/build_custom_pad_metadata.py"
            )
        items: list[DirectItem] = []
        skipped = 0
        first_missing: str | None = None
        total = 0
        with meta_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                total += 1
                path = input_dir / row["dst_path"]
                if not path.exists():
                    # Counted, not printed per row: this fired 5662 times before
                    # and scrolled the actual cause off the console. The
                    # pre-flight `verify_raw_pools` gate reports the same drift
                    # with a pool name and a fraction.
                    skipped += 1
                    if first_missing is None:
                        first_missing = row["dst_path"]
                    continue
                frame_match = re.search(r"_frame_(\d+)$", path.stem)
                items.append(
                    DirectItem(
                        path=path,
                        subject_id=row["user_id"],
                        video_id=row["video_id"],
                        label=(
                            "genuine" if is_genuine_label(row["label"]) else "spoof"
                        ),
                        attack_type=row["attack_type"],
                        device=row.get("device") or "unknown",
                        frame_index=(int(frame_match.group(1)) if frame_match else 0),
                    )
                )
        if skipped:
            print(
                f"  WARNING: {dataset_source}: skipped {skipped}/{total} metadata "
                f"row(s) pointing at missing files (first: {first_missing!r})"
            )
        return items

    return iter_convention_items, iter_metadata_items


@app.cell
def direct_ingest_writer(DirectItem, detect_existing_crop):
    def write_pool(
        items: list[DirectItem],
        output_dir: Path,
        dataset_source: str,
        split_role: str,
        input_dir: Path,
    ) -> None:
        if not items:
            raise RuntimeError(f"no images found for {dataset_source}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        (output_dir / "faces").mkdir(parents=True)

        media_rows: list[dict] = []
        frame_rows: list[dict] = []
        candidate_rows: list[dict] = []
        by_video: dict[str, list[DirectItem]] = {}
        for it in items:
            by_video.setdefault(it.video_id, []).append(it)

        for it in items:
            subject_dir = output_dir / "faces" / it.subject_id / it.video_id
            subject_dir.mkdir(parents=True, exist_ok=True)
            (subject_dir / "landmarks").mkdir(exist_ok=True)
            shutil.copy2(it.path, subject_dir / it.path.name)

            crop = detect_existing_crop(it.path)
            bbox, landmarks, geometry, pose = (
                crop.bbox,
                crop.landmarks,
                crop.geometry,
                crop.pose_asym,
            )
            landmark_rel = (
                Path("faces")
                / it.subject_id
                / it.video_id
                / "landmarks"
                / f"frame_{it.frame_index + 1:03d}.npy"
            )
            np.save(output_dir / landmark_rel, landmarks)

            g = geometry
            frame_rows.append(
                {
                    "subject_id": it.subject_id,
                    "video_id": it.video_id,
                    "media_type": "pre_cropped_image",
                    "frame_id": f"{it.video_id}_f{it.frame_index + 1:03d}",
                    "frame_index": it.frame_index,
                    "timestamp": None,
                    "frame_path": str(
                        Path("faces") / it.subject_id / it.video_id / it.path.name
                    ),
                    "landmark_path": str(landmark_rel),
                    "sequence_position": it.frame_index + 1,
                    "label": it.label,
                    "attack_type": it.attack_type,
                    "pai_family": pai_family_of(it.attack_type),
                    "dataset_source": dataset_source,
                    "split_role": split_role,
                    "quality_status": "selected",
                    "pre_cropped": True,
                    "face_bbox": bbox,
                    "face_width": bbox[2] - bbox[0],
                    "face_height": bbox[3] - bbox[1],
                    "detector_confidence": (
                        crop.detection.prob if crop.detected else None
                    ),
                    "blur_score": None,
                    "device": it.device,
                    "interocular_dist_norm": (g.interocular_dist_norm if g else None),
                    "eye_to_nose_dist_norm": (g.eye_to_nose_dist_norm if g else None),
                    "nose_to_mouth_dist_norm": (
                        g.nose_to_mouth_dist_norm if g else None
                    ),
                    "eye_to_mouth_dist_norm": (g.eye_to_mouth_dist_norm if g else None),
                    "mouth_width_norm": g.mouth_width_norm if g else None,
                    "face_aspect_ratio": g.face_aspect_ratio if g else None,
                    "pose_asym": pose,
                }
            )
            candidate_rows.append(
                {
                    "subject_id": it.subject_id,
                    "video_id": it.video_id,
                    "media_type": "pre_cropped_image",
                    "frame_index": it.frame_index,
                    "timestamp": None,
                    "quality_status": "pre_cropped",
                    "rejection_reason": None,
                    "track_status": "direct_ingest",
                    "detector_confidence": None,
                    "chosen": True,
                    "sequence_position": it.frame_index + 1,
                    "label": it.label,
                    "attack_type": it.attack_type,
                    "dataset_source": dataset_source,
                    "split_role": split_role,
                    "device": it.device,
                }
            )

        # one media row per source video (a "test item" for the official pool)
        for video_id, group in by_video.items():
            first = group[0]
            media_rows.append(
                {
                    "subject_id": first.subject_id,
                    "video_id": video_id,
                    "video_path": str(first.path),
                    "media_type": "pre_cropped_image",
                    "label": first.label,
                    "attack_type": first.attack_type,
                    "pai_family": pai_family_of(first.attack_type),
                    "dataset_source": dataset_source,
                    "device": first.device,
                    "replay_device": None,
                    "session": None,
                    "environment": "direct_ingest",
                    "active_zoom": "unknown",
                    "raw_label": first.label,
                    "split_role": split_role,
                    "frame_count": len(group),
                    "num_candidates_evaluated": len(group),
                    "num_frames_accepted": len(group),
                    "num_frames_selected": len(group),
                    "status": "ok",
                    "error_message": None,
                }
            )

        videos_df = pd.DataFrame(media_rows)
        frames_df = pd.DataFrame(frame_rows)
        candidates_df = pd.DataFrame(candidate_rows)
        if not videos_df.empty:
            videos_df = videos_df.drop_duplicates("video_id")
        meta = output_dir / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        videos_df.to_csv(meta / "videos.csv", index=False)
        frames_df.to_csv(meta / "frames.csv", index=False)
        candidates_df.to_csv(meta / "candidates.csv", index=False)

        subjects = []
        for subject_id, group in videos_df.groupby("subject_id"):
            subjects.append(
                {
                    "subject_id": subject_id,
                    "num_videos": len(group),
                    "num_genuine_videos": int((group["label"] == "genuine").sum()),
                    "num_spoof_videos": int((group["label"] == "spoof").sum()),
                    "attack_types_present": ",".join(
                        sorted(group["attack_type"].unique())
                    ),
                    "devices_present": ",".join(
                        sorted(str(d) for d in group["device"].dropna().unique())
                    ),
                    "num_frames_selected_total": int(
                        (frames_df["subject_id"] == subject_id).sum()
                    ),
                }
            )
        pd.DataFrame(subjects).to_csv(meta / "subjects.csv", index=False)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "dataset_source": dataset_source,
                    "split_role": split_role,
                    "media_type": "pre_cropped_image",
                    "ingest": "direct (verbatim copy, geometry only)",
                    "pai_family_map": PAI_FAMILY,
                },
                indent=2,
            )
        )
        print(
            f"direct ingest {dataset_source}: {len(videos_df)} media, "
            f"{len(frames_df)} frames, {len(subjects)} subjects -> {output_dir}"
        )
        print(f"  attack_type: {dict(Counter(frames_df['attack_type']))}")

    return (write_pool,)


@app.cell
def direct_ingest_pools(
    iter_convention_items,
    iter_metadata_items,
    write_pool,
):
    def build_pre_cropped_pool():
        write_pool(
            iter_convention_items(
                Path("datasets/frames_cleaned_cropped_v2"),
                "frames_cleaned_cropped_v2",
                "development",
                nested_attack=False,
            ),
            Path("processed_dataset_frames_cleaned"),
            "frames_cleaned_cropped_v2",
            "development",
            Path("datasets/frames_cleaned_cropped_v2"),
        )

    def build_custom_pad_pool():
        write_pool(
            iter_metadata_items(
                Path("datasets/custom_pad_dataset"),
                "custom_pad_dataset",
                "development",
            ),
            Path("processed_dataset_custom_pad"),
            "custom_pad_dataset",
            "development",
            Path("datasets/custom_pad_dataset"),
        )

    def build_official_test_pool():
        write_pool(
            iter_convention_items(
                Path("datasets/test_data_ibeta"),
                "ibeta_official_test",
                "official_test",
                nested_attack=True,
            ),
            Path("processed_dataset_official_test"),
            "ibeta_official_test",
            "official_test",
            Path("datasets/test_data_ibeta"),
        )

    build_pre_cropped_pool()
    build_custom_pad_pool()
    build_official_test_pool()
    return


@app.cell
def combined_pool(Config, run):
    # Merge per-dataset pools into processed_dataset_combined/ + zip.
    EVAL_PREFIX = "lcc_eval"
    TABLES = ("videos", "frames", "candidates")

    DEFAULT_POOLS = {
        "fas_ibeta_l1": "processed_dataset",
        "lcc_fasd": "processed_dataset_lcc_fasd",
        "pad2d": "processed_dataset_pad2d",
        "frames_cleaned_cropped_v2": "processed_dataset_frames_cleaned",
        "custom_pad_dataset": "processed_dataset_custom_pad",
    }

    zip_path = Path("processed_dataset.zip")
    official_test_zip_path = Path("processed_dataset_official_test.zip")

    def merge_faces(pools: dict[str, Path], faces_dir: Path) -> None:
        # Subject namespaces (axon_*/a8_*/unidata_* vs lcc_* vs pad2d_*) are
        # disjoint by construction; assert it BEFORE copying since cp -al
        # would silently merge same-named subject dirs.
        seen: set[str] = set()
        per_pool_names: dict[str, set[str]] = {}
        for name, src in pools.items():
            names = set(os.listdir(src / "faces"))
            clash = seen & names
            assert not clash, (
                f"subject dir collision across pools ({name}): {sorted(clash)}"
            )
            seen |= names
            per_pool_names[name] = names
        if faces_dir.exists():
            shutil.rmtree(faces_dir)
        faces_dir.mkdir(parents=True)
        n_copied, n_skipped_eval = 0, 0
        for name, src in pools.items():
            for subj in sorted(per_pool_names[name]):
                # lcc_eval_* subjects are the held-out cross-dataset probe --
                # excluded from every merged metadata table (see
                # merge_tables), so their face dirs must not be hardlinked
                # into the combined pool either, or the probe images would
                # sit inside processed_dataset_combined/faces unreferenced by
                # any row but reachable by anyone walking the tree.
                if subj.startswith(EVAL_PREFIX):
                    n_skipped_eval += 1
                    continue
                subprocess.run(
                    [
                        "cp",
                        "-al",
                        f"{src}/faces/{subj}",
                        f"{faces_dir}/{subj}",
                    ],
                    check=True,
                )
                n_copied += 1
        print(
            f"hardlinked {n_copied} subject dirs (skipped {n_skipped_eval} lcc_eval_* probe subjects)"
        )

    def merge_tables(pools: dict[str, Path], meta_dir: Path) -> pd.DataFrame:
        meta_dir.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            parts = []
            counts = []
            for name, src in pools.items():
                df = pd.read_csv(src / "metadata" / f"{table}.csv")
                n_eval = int(df["subject_id"].str.startswith(EVAL_PREFIX).sum())
                df = df[~df["subject_id"].str.startswith(EVAL_PREFIX)]
                df["pool"] = name
                parts.append(df)
                counts.append(
                    f"{len(df)} {name}"
                    + (f" (+{n_eval} eval excluded)" if n_eval else "")
                )
            merged = pd.concat(parts, ignore_index=True)
            merged.to_csv(meta_dir / f"{table}.csv", index=False)
            print(f"{table}: {' + '.join(counts)} = {len(merged)}")
        return pd.read_csv(meta_dir / "videos.csv")

    def run_late_stages(out_dir: Path, num_folds: int, seed: int) -> None:
        cfg = Config(
            input_dir=out_dir,
            output_dir=out_dir,
            dataset_source="combined",
            blur_threshold=20.0,  # cosmetic here (post-extract); per-video gates live in manifests
            num_folds=num_folds,
            seed=seed,
            stages=["metadata", "splits", "balance", "report"],
        )
        run(cfg)

    def pack_training_zip(
        out_dir: Path, zip_file: Path, archive_root: str = "processed_dataset"
    ) -> None:
        if zip_file.exists():
            zip_file.unlink()
        with zipfile.ZipFile(
            zip_file, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in sorted(files):
                    full = Path(root) / fn
                    arc = Path(archive_root) / full.relative_to(out_dir)
                    zf.write(full, arc.as_posix())
        print(f"wrote {zip_file} ({zip_file.stat().st_size / 1e6:.0f} MB)")

    return (
        DEFAULT_POOLS,
        merge_faces,
        merge_tables,
        official_test_zip_path,
        pack_training_zip,
        run_late_stages,
        zip_path,
    )


@app.cell
def extract_pools(Config, run):
    for _pool in POOLS:
        _cfg_kwargs = dict(
            _pool
        )  # copy: don't mutate POOLS (re-running this cell must be idempotent)
        _subpool_rules = _cfg_kwargs.pop("subpool_rules", None)
        print(
            f"\n========== {_cfg_kwargs['input_dir']} -> {_cfg_kwargs['output_dir']} =========="
        )
        run(Config(device="cuda", subpool_rules=_subpool_rules, **_cfg_kwargs))
    return


@app.cell
def merge_and_package(
    DEFAULT_POOLS,
    merge_faces,
    merge_tables,
    official_test_zip_path,
    pack_training_zip,
    run_late_stages,
    zip_path,
):
    pools = {
        name: Path(path) for name, path in DEFAULT_POOLS.items() if Path(path).is_dir()
    }
    missing = [name for name in DEFAULT_POOLS if name not in pools]
    if missing:
        raise SystemExit(
            f"expected pools missing from disk: {missing}. "
            f"Present: {sorted(pools)}. Re-run the direct-ingest and extraction "
            f"cells before merging."
        )
    if len(pools) < 2:
        raise SystemExit(f"fewer than 2 pools exist on disk ({list(pools)})")

    out_dir = Path("processed_dataset_combined")
    print(f"pools: {', '.join(f'{k} ({v})' for k, v in pools.items())}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    merge_faces(pools, out_dir / "faces")
    videos = merge_tables(pools, out_dir / "metadata")

    # Verify subject namespace disjointness across pools
    for name in pools:
        others = set(videos[videos["pool"] != name]["subject_id"])
        mine = set(videos[videos["pool"] == name]["subject_id"])
        overlap = others & mine
        assert not overlap, (
            f"subject namespace collision involving {name}: {sorted(overlap)}"
        )

    n_live = videos[videos["label"] == "genuine"]["subject_id"].nunique()
    n_spoof = videos[videos["label"] == "spoof"]["subject_id"].nunique()
    print(f"pool subjects: {n_live} genuine / {n_spoof} spoof")

    run_late_stages(out_dir, num_folds=5, seed=42)
    pack_training_zip(out_dir, zip_path)
    pack_training_zip(
        Path("processed_dataset_official_test"),
        official_test_zip_path,
        archive_root="processed_dataset_official_test",
    )
    return


@app.cell
def upload_to_drive(fs, official_test_zip_path, zip_path):
    for _zip in (zip_path, official_test_zip_path):
        if not _zip.exists():
            raise FileNotFoundError(f"expected processed artifact is missing: {_zip}")
        print(f"Uploading {_zip} to Google Drive...")
        with (
            open(_zip, "rb") as local_f,
            fs.open(_zip.name, "wb") as remote_f,
        ):
            remote_f.write(local_f.read())
        print(f"Uploaded {_zip}")
    return


@app.cell
def cleanup():
    """Remove all intermediate artifacts. Safe to run unconditionally --
    the next run re-downloads from Hugging Face and re-processes.
    """
    cleanup_targets = [
        # "datasets",
        "processed_dataset",
        "processed_dataset_lcc_fasd",
        "processed_dataset_pad2d",
        "processed_dataset_frames_cleaned",
        "processed_dataset_custom_pad",
        "processed_dataset_official_test",
        "processed_dataset_combined",
        "processed_dataset.zip",
        "processed_dataset_official_test.zip",
    ]

    removed = 0
    for target in cleanup_targets:
        path = Path(target)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1
    print(f"Removed {removed} path(s). Next run re-downloads and re-processes.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
