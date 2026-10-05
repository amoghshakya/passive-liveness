#!/usr/bin/env python3
"""Minimal preprocessing script for processing video folders.

This script extracts faces from videos and creates a processed dataset
compatible with the evaluation pipeline. It's a simplified version that
focuses on core functionality without the complexity of the full marimo notebook.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.model_selection import KFold
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json

# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    # I/O
    input_dir: Path = Path("input")
    metadata_csv: Optional[Path] = None  # defaults to <input_dir>/metadata.csv
    output_dir: Path = Path("processed_dataset")
    dataset_source: str = "custom"  # tag written on every row
    subpool_rules: List[Tuple[str, str]] = field(default_factory=list)

    # frame extraction
    num_frames: int = 8
    sampling_strategy: str = "uniform"
    num_candidates: int = 48  # candidate frames sampled across whole video before detection

    # face detection / tracking
    face_confidence_threshold: float = 0.3  # Lowered for better detection
    det_thresh: float | None = None
    iou_track_threshold: float = 0.3
    min_track_len: int = 3

    # quality filtering
    min_face_size: int = 20  # Reduced for better detection
    min_face_ratio: float = 0.01  # Reduced for better detection
    blur_threshold: float = (
        10.0  # Reduced to be more lenient (variance-of-Laplacian on the aligned crop)
    )
    max_pose_asym: float | None = None  # None = pose filter disabled

    # face crop
    face_size: int = (
        512  # aligned crop side length; NOT the model's final input size
    )
    crop_margin_scale: float = 1.0  # <1.0 zooms out (align_face), revealing more context around the face
    adaptive_crop_margin: bool = False  # if True, derive margin_scale per-face from face-to-frame area ratio instead of the fixed crop_margin_scale

    # splits
    num_folds: int = 5
    seed: int = 42

    # misc
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    limit_videos: int | None = None
    skip_existing: bool = False
    stages: List[str] = field(default_factory=lambda: ["extract", "metadata", "splits"])
    contact_sheet_examples: int = 6

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.metadata_csv is None:
            self.metadata_csv = self.input_dir / "metadata.csv"
        else:
            self.metadata_csv = Path(self.metadata_csv)
        if self.sampling_strategy not in ["uniform", "random", "fibonacci", "quality_weighted", "center_biased"]:
            raise ValueError(
                f"sampling_strategy must be one of ['uniform', 'random', 'fibonacci', 'quality_weighted', 'center_biased']"
            )
        for s in self.stages:
            if s not in ["extract", "metadata", "splits", "balance", "report", "contact_sheets"]:
                raise ValueError(
                    f"unknown stage {s!r}, must be one of ['extract', 'metadata', 'splits', 'balance', 'report', 'contact_sheets']"
                )


# =====================================================================
# Core Processing Components
# =====================================================================

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
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=providers,
        )
        prepare_kwargs = {"det_size": (detect_max_side, detect_max_side)}
        if det_thresh is not None:
            prepare_kwargs["det_thresh"] = det_thresh
        self.app.prepare(
            ctx_id=0 if self.device == "cuda" else -1, **prepare_kwargs
        )

    def detect_batch(self, frames_bgr: List[np.ndarray]) -> List[List[Dict]]:
        if not frames_bgr:
            return []
        results: List[List[Dict]] = []
        for frame in frames_bgr:
            faces = self.app.get(frame)
            dets = [
                {
                    "box": tuple(float(v) for v in f.bbox),
                    "prob": float(f.det_score),
                    "landmarks": np.asarray(f.kps, dtype=np.float32),
                }
                for f in faces
            ]
            results.append(dets)
        return results


def area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def track_primary_face(
    frame_indices: List[int],
    detections_per_frame: List[List[Dict]],
    iou_track_threshold: float = 0.3,
) -> List[Dict]:
    """Simple primary face tracking."""
    out: List[Dict] = []
    last_box: Optional[Tuple[float, float, float, float]] = None

    for idx, (frame_idx, dets) in enumerate(zip(frame_indices, detections_per_frame)):
        if not dets:
            out.append({"detection": None, "track_status": "no_face"})
            continue

        if last_box is None:
            best = max(dets, key=lambda d: d["prob"] * area(d["box"]))
            out.append({"detection": best, "track_status": "ok"})
            last_box = best["box"]
            continue

        best_det, best_iou = max(
            ((d, iou(d["box"], last_box)) for d in dets), key=lambda t: t[1]
        )
        if best_iou >= iou_track_threshold:
            out.append({"detection": best_det, "track_status": "ok"})
            last_box = best_det["box"]
            continue
        else:
            out.append({"detection": None, "track_status": "primary_face_lost"})
            # keep last_box as-is so a later frame can still re-acquire against it

    return out


def blur_score(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate(
    frame_bgr: np.ndarray,
    tracked: Dict,
    cfg: Config,
) -> Tuple[bool, Optional[str], float, Optional[Tuple[int, int, int, int]], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """Quality filtering for tracked face detections."""
    if tracked["detection"] is None:
        return False, tracked["track_status"], 0.0, None, None, None, None, None

    det: Dict = tracked["detection"]
    x1, y1, x2, y2 = det["box"]
    frame_h, frame_w = frame_bgr.shape[:2]
    x1c, y1c = max(0, round(x1)), max(0, round(y1))
    x2c, y2c = min(frame_w, round(x2)), min(frame_h, round(y2))

    if x2c <= x1c or y2c <= y1c:
        return False, "invalid_bbox", det["prob"], None, None, None, None, None

    bbox = (x1c, y1c, x2c, y2c)
    face_w, face_h = x2c - x1c, y2c - y1c
    blur = blur_score(frame_bgr, bbox)

    if det["prob"] < cfg.face_confidence_threshold:
        return False, "low_confidence", det["prob"], bbox, face_w, face_h, blur, None

    if (
        min(face_w, face_h) < cfg.min_face_size
        or (face_h / frame_h) < cfg.min_face_ratio
    ):
        return False, "too_small", det["prob"], bbox, face_w, face_h, blur, None

    if blur < cfg.blur_threshold:
        return False, "blurry", det["prob"], bbox, face_w, face_h, blur, None

    return True, None, det["prob"], bbox, face_w, face_h, blur, None


def align_face(
    frame_bgr: np.ndarray,
    landmarks_5pt: np.ndarray,
    face_size: int,
    margin_scale: float = 1.0,
    fallback_bbox: Tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Simple face alignment using similarity transform."""
    # ArcFace 112x112 reference points
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

    # degenerate landmarks: fall back to a square bbox crop+resize
    if fallback_bbox is None:
        raise ValueError("alignment failed and no fallback_bbox provided")
    x1, y1, x2, y2 = fallback_bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((face_size, face_size, 3), dtype=frame_bgr.dtype)
    return cv2.resize(crop, (face_size, face_size))


def uniform_sampling(n_available: int, num_frames: int, rng: np.random.Generator) -> List[int]:
    """Uniform sampling strategy."""
    want = min(n_available, num_frames)
    if want <= 0:
        return []
    if want == 1:
        return [n_available // 2]
    positions = np.linspace(0, n_available - 1, want)
    idxs = [round(p) for p in positions]
    # Simple dedup/pad
    chosen = set(idxs)
    if len(chosen) >= want:
        return sorted(list(chosen))[:want]
    remaining = [i for i in range(n_available) if i not in chosen]
    remaining.sort(key=lambda r: min(abs(r - c) for c in chosen) if chosen else r)
    for r in remaining:
        chosen.add(r)
        if len(chosen) >= want:
            break
    return sorted(list(chosen))[:want]


STRATEGIES = {
    "uniform": uniform_sampling,
    # For simplicity, we'll just use uniform for now
    # Other strategies could be added if needed
}


def compute_fingerprint(cfg: Config) -> Dict:
    """Compute config fingerprint for caching."""
    # Simple fingerprint based on key parameters
    return {
        "num_frames": cfg.num_frames,
        "sampling_strategy": cfg.sampling_strategy,
        "num_candidates": cfg.num_candidates,
        "face_confidence_threshold": cfg.face_confidence_threshold,
        "det_thresh": cfg.det_thresh,
        "iou_track_threshold": cfg.iou_track_threshold,
        "min_track_len": cfg.min_track_len,
        "min_face_size": cfg.min_face_size,
        "min_face_ratio": cfg.min_face_ratio,
        "blur_threshold": cfg.blur_threshold,
        "max_pose_asym": cfg.max_pose_asym,
        "face_size": cfg.face_size,
        "crop_margin_scale": cfg.crop_margin_scale,
        "adaptive_crop_margin": cfg.adaptive_crop_margin,
        "seed": cfg.seed,
        "dataset_source": cfg.dataset_source,
        "subpool_rules": cfg.subpool_rules,
    }


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def load_cached(out_dir: Path, fingerprint: Dict) -> Optional[Dict]:
    """Load cached extraction result."""
    mp = manifest_path(out_dir)
    if not mp.exists():
        return None
    try:
        data = json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    return {
        "video_row": data["video_row"],
        "frame_rows": data["frame_rows"],
        "candidate_rows": data["candidate_rows"],
    }


def save_cache(out_dir: Path, result: Dict, fingerprint: Dict) -> None:
    """Save extraction result to cache."""
    mp = manifest_path(out_dir)
    mp.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "video_row": result["video_row"],
                "frame_rows": result["frame_rows"],
                "candidate_rows": result["candidate_rows"],
            }
        )
    )


def process_video(
    video_path: Path,
    cfg: Config,
    detector: FaceDetector,
) -> Dict:
    """Process a single video and extract frames."""
    # Create output directory for this video
    # For simplicity, we'll use a hash of the video path as video_id
    video_id = hashlib.sha256(str(video_path).encode()).hexdigest()[:16]
    # Try to extract subject_id from path if possible, otherwise use default
    subject_id = "subject_01"  # Default - will be improved if metadata is available
    out_dir = cfg.output_dir / "faces" / subject_id / video_id
    
    fingerprint = compute_fingerprint(cfg)
    if cfg.skip_existing:
        cached = load_cached(out_dir, fingerprint)
        if cached is not None:
            return cached

    # Probe video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        # Count frames by reading
        n = 0
        while cap.grab():
            n += 1
        frame_count = n
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    
    cap.release()

    # Sample candidate frames
    candidate_indices_list = []
    if frame_count > 0:
        n_candidates = min(cfg.num_candidates, frame_count)
        if n_candidates <= 1:
            candidate_indices_list = [0] if frame_count > 0 else []
        else:
            candidate_indices_list = sorted(
                {round(i * (frame_count - 1) / (n_candidates - 1)) for i in range(n_candidates)}
            )
    
    # Read frames at candidate indices
    raw_frames = []
    if candidate_indices_list:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            wanted = set(candidate_indices_list)
            idx = 0
            last_wanted = max(wanted) if wanted else 0
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
                                idx / fps
                                if fps > 0
                                else float(idx)
                            )
                        )
                        raw_frames.append({
                            "frame_index": idx,
                            "timestamp": ts,
                            "image": frame
                        })
                idx += 1
        cap.release()
    
    if not raw_frames:
        # No frames could be decoded
        video_row = {
            "subject_id": subject_id,
            "video_id": video_id,
            "video_path": str(video_path.relative_to(cfg.input_dir) if cfg.input_dir in video_path.parents else video_path.name),
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
            "error_message": "could not read video",
            "dataset_source": cfg.dataset_source,
        }
        return {
            "video_row": video_row,
            "frame_rows": [],
            "candidate_rows": [],
        }

    # Detect faces in candidate frames
    frames_bgr = [rf["image"] for rf in raw_frames]
    detections = detector.detect_batch(frames_bgr)
    
    # Track primary face
    tracked = track_primary_face(
        [rf["frame_index"] for rf in raw_frames],
        detections,
        cfg.iou_track_threshold,
    )

    # Quality filtering
    qcfg = type('QualityConfig', (), {
        'face_confidence_threshold': cfg.face_confidence_threshold,
        'min_face_size': cfg.min_face_size,
        'min_face_ratio': cfg.min_face_ratio,
        'blur_threshold': cfg.blur_threshold,
        'max_pose_asym': cfg.max_pose_asym,
    })()
    
    candidate_rows: List[Dict] = []
    accepted: List[Tuple] = []  # (raw_frame, tracked_frame, quality_result)
    for rf, tf in zip(raw_frames, tracked):
        accepted_bool, rejection_reason, det_prob, bbox, face_w, face_h, blur, pose_asym = evaluate(
            rf["image"], tf, cfg
        )
        candidate_rows.append({
            "subject_id": subject_id,
            "video_id": video_id,
            "frame_index": rf["frame_index"],
            "timestamp": rf["timestamp"],
            "quality_status": "accepted" if accepted_bool else "rejected",
            "rejection_reason": rejection_reason,
            "track_status": tf["track_status"],
            "detector_confidence": tf["detection"]["prob"] if tf["detection"] else None,
            "face_bbox": bbox,
            "face_width": face_w,
            "face_height": face_h,
            "blur_score": blur,
            "pose_asym": pose_asym,
            "chosen": False,
            "sequence_position": None,
            "label": "genuine",  # All clips are bona fide/real
            "attack_type": "",
            "pai_family": "",
            "dataset_source": cfg.dataset_source,
        })
        if accepted_bool:
            accepted.append((rf, tf, (accepted_bool, rejection_reason, det_prob, bbox, face_w, face_h, blur, pose_asym)))

    n_accepted = len(accepted)
    usable_start = accepted[0][0]["frame_index"] if accepted else None
    usable_end = accepted[-1][0]["frame_index"] if accepted else None

    frame_rows: List[Dict] = []
    if n_accepted == 0:
        video_row = {
            "subject_id": subject_id,
            "video_id": video_id,
            "video_path": str(video_path.relative_to(cfg.input_dir) if cfg.input_dir in video_path.parents else video_path.name),
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration_sec,
            "width": width,
            "height": height,
            "num_candidates_evaluated": len(raw_frames),
            "num_frames_accepted": 0,
            "num_frames_selected": 0,
            "usable_interval_start_idx": None,
            "usable_interval_end_idx": None,
            "status": "no_usable_face",
            "error_message": None,
            "dataset_source": cfg.dataset_source,
        }
        return {
            "video_row": video_row,
            "frame_rows": [],
            "candidate_rows": candidate_rows,
        }

    # Frame selection
    rng = np.random.default_rng(cfg.seed + hash(video_id) % 2**32)
    weights = np.array(
        [a[2][2] or 0.0 for a in accepted], dtype=float  # detector_confidence
    )
    strategy_fn = STRATEGIES[cfg.sampling_strategy]
    strategy_fn = STRATEGIES[cfg.sampling_strategy]
    picked_positions = strategy_fn(
        n_accepted, cfg.num_frames, rng
    )

    chosen_frame_indices = {
        accepted[p][0]["frame_index"] for p in picked_positions
    }
    chosen_frame_indices = {
        accepted[p][0]["frame_index"] for p in picked_positions
    }
    chosen_order = {
        accepted[p][0]["frame_index"]: seq
        for seq, p in enumerate(picked_positions, start=1)
    }
    for row in candidate_rows:
        if row["frame_index"] in chosen_frame_indices:
            row["chosen"] = True
            row["sequence_position"] = chosen_order[row["frame_index"]]
    for row in candidate_rows:
        if row["frame_index"] in chosen_frame_indices:
            row["chosen"] = True
            row["sequence_position"] = chosen_order[row["frame_index"]]

    # Save selected frames
    for seq, pos in enumerate(picked_positions, start=1):
        rf, tf, qr_tuple = accepted[pos]
        det: Dict = tf["detection"]
        margin_scale = (
            1.0  # Simplified - not implementing adaptive margin for now
            if cfg.adaptive_crop_margin else cfg.crop_margin_scale
        )
        aligned = align_face(
            rf["image"],
            det["landmarks"],
            cfg.face_size,
            margin_scale=margin_scale,
            fallback_bbox=qr_tuple[3],  # bbox
        )
        
        # Create directories
        (out_dir / "landmarks").mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aligned frame
        frame_filename = f"frame_{seq:03d}.jpg"
        cv2.imwrite(str(out_dir / frame_filename), aligned, [cv2.IMWRITE_JPEG_QUALITY, 97])
        
        # Save landmarks
        landmark_file = frame_filename.replace(".jpg", ".npy")
        np.save(out_dir / "landmarks" / landmark_file, det["landmarks"])
        
        # Add to frame rows
        qr_bool, qr_reason, qr_det_prob, qr_bbox, qr_face_w, qr_face_h, qr_blur, qr_pose_asym = qr_tuple
        frame_rows.append({
            "subject_id": subject_id,
            "video_id": video_id,
            "frame_id": f"{video_id}_f{seq:03d}",
            "frame_index": rf["frame_index"],
            "timestamp": rf["timestamp"],
            "frame_path": str(Path("faces") / subject_id / video_id / frame_filename),
            "face_bbox": qr_bbox,
            "face_width": qr_face_w,
            "face_height": qr_face_h,
            "detector_confidence": qr_det_prob,
            "blur_score": qr_blur,
            "quality_status": "selected",
            "landmark_path": str(Path("faces") / subject_id / video_id / "landmarks" / landmark_file),
            "sequence_position": seq,
            "label": "genuine",
            "attack_type": "",
            "pai_family": "",
            "dataset_source": cfg.dataset_source,
            "interocular_dist_norm": None,  # Simplified - we could compute this but keeping it simple
            "eye_to_nose_dist_norm": None,
            "nose_to_mouth_dist_norm": None,
            "eye_to_mouth_dist_norm": None,
            "mouth_width_norm": None,
            "face_aspect_ratio": None,
            "pose_asym": qr_pose_asym,
        })

    status = "ok"
    if n_accepted < cfg.min_track_len:
        status = "insufficient_frames"

    video_row = {
        "subject_id": subject_id,
        "video_id": video_id,
        "video_path": str(video_path.relative_to(cfg.input_dir) if cfg.input_dir in video_path.parents else video_path.name),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration_sec,
        "width": width,
        "height": height,
        "num_candidates_evaluated": len(raw_frames),
        "num_frames_accepted": n_accepted,
        "num_frames_selected": len(frame_rows),
        "usable_interval_start_idx": usable_start,
        "usable_interval_end_idx": usable_end,
        "status": status,
        "error_message": None,
        "dataset_source": cfg.dataset_source,
    }
    
    return {
        "video_row": video_row,
        "frame_rows": frame_rows,
        "candidate_rows": candidate_rows,
    }


def write_metadata(
    output_dir: Path,
    video_rows: List[Dict],
    frame_rows: List[Dict],
    candidate_rows: List[Dict],
) -> Dict[str, pd.DataFrame]:
    """Write metadata CSV files."""
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    videos_df = pd.DataFrame(video_rows) if video_rows else pd.DataFrame()
    frames_df = pd.DataFrame(frame_rows) if frame_rows else pd.DataFrame()
    candidates_df = pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame()
    subjects_df = build_subjects(videos_df, frames_df) if not videos_df.empty else pd.DataFrame(columns=["subject_id", "num_videos"])

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
    videos_df: pd.DataFrame,
    frames_df: pd.DataFrame
) -> pd.DataFrame:
    if videos_df.empty:
        return pd.DataFrame(columns=["subject_id", "num_videos"])

    rows = []
    for subject_id, g in videos_df.groupby("subject_id"):
        # Handle empty frames_df case
        if frames_df.empty:
            frame_g = pd.DataFrame(columns=frames_df.columns)
        else:
            frame_g = frames_df[frames_df["subject_id"] == subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "num_videos": len(g),
                "num_genuine_videos": int((g["label"] == "genuine").sum()) if len(g) > 0 and "label" in g.columns else 0,
                "num_spoof_videos": int((g["label"] == "spoof").sum()) if len(g) > 0 and "label" in g.columns else 0,
                "attack_types_present": ",".join(sorted(g["attack_type"].unique())) if len(g) > 0 and "attack_type" in g.columns else "",
                "devices_present": ",".join(sorted(d for d in g["device"].unique() if d)) if len(g) > 0 and "device" in g.columns else "",
                "num_frames_selected_total": len(frame_g),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    )

    rows = []
    for subject_id, g in videos_df.groupby("subject_id"):
        # Handle empty frames_df case
        if frames_df.empty:
            frame_g = pd.DataFrame(columns=frames_df.columns)
        else:
            frame_g = frames_df[frames_df["subject_id"] == subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "num_videos": len(g),
                "num_genuine_videos": int((g["label"] == "genuine").sum()) if len(g) > 0 and "label" in g.columns else 0,
                "num_spoof_videos": int((g["label"] == "spoof").sum()) if len(g) > 0 and "label" in g.columns else 0,
                "attack_types_present": ",".join(sorted(g["attack_type"].unique())) if len(g) > 0 and "attack_type" in g.columns else "",
                "devices_present": ",".join(sorted(d for d in g["device"].unique() if d)) if len(g) > 0 and "device" in g.columns else "",
            }
        )
    return (
        pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    )

def generate_folds(
    videos_df: pd.DataFrame,
    num_folds: int,
    seed: int,
) -> List[Dict]:
    """Generate subject-disjoint folds."""
    if videos_df.empty or num_folds < 2:
        return []
    # Create subject table
    subject_rows = []
    for subject_id, g in videos_df.groupby("subject_id"):
        label_counts = g["label"].value_counts() if "label" in g.columns else pd.Series()
        majority_label = label_counts.index[0] if len(label_counts) > 0 else "unknown"

    subjects_df = pd.DataFrame(subject_rows)
    if subjects_df.empty:
        return []

    # Use stratified KFold if possible, otherwise plain KFold
    from sklearn.model_selection import StratifiedKFold, KFold

    try:
        y = subjects_df["majority_label"]
        min_class_count = y.value_counts().min()
        if min_class_count >= num_folds:
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            splits = list(skf.split(subjects_df, y))
        else:
            # Fall back to plain KFold
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(subjects_df))
    except Exception as e:
        # Fall back to plain KFold
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(subjects_df))

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        folds.append({
            "fold": fold_idx + 1,
            "train_subjects": sorted(subjects_df.iloc[train_idx]["subject_id"].tolist()),
            "val_subjects": sorted(subjects_df.iloc[val_idx]["subject_id"].tolist()),
        })
    return folds
def main():
    parser = argparse.ArgumentParser(
        description="Minimal preprocessing script for processing video folders"
    )
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        required=True,
        help="Input directory containing videos organized by subject"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("processed_dataset"),
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--metadata-csv", 
        type=Path, 
        default=None,
        help="Path to metadata CSV (defaults to <input-dir>/metadata.csv)"
    )
    parser.add_argument(
        "--dataset-source", 
        type=str, 
        default="arsenal_players",
        help="Dataset source tag"
    )
    parser.add_argument(
        "--num-frames", 
        type=int, 
        default=8,
        help="Number of frames to extract per video"
    )
    parser.add_argument(
        "--sampling-strategy", 
        type=str, 
        default="uniform",
        choices=["uniform", "random", "fibonacci", "quality_weighted", "center_biased"],
        help="Frame sampling strategy"
    )
    parser.add_argument(
        "--det-thresh", 
        type=float, 
        default=0.3,
        help="Face detection threshold"
    )
    parser.add_argument(
        "--face-confidence-threshold", 
        type=float, 
        default=0.3,
        help="Face confidence threshold for quality filtering"
    )
    parser.add_argument(
        "--blur-thresh", 
        type=float, 
        default=10.0,
        help="Blur threshold (variance of Laplacian)"
    )
    parser.add_argument(
        "--min-face-size", 
        type=int, 
        default=20,
        help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip existing extractions"
    )
    parser.add_argument(
        "--stages", 
        nargs="+", 
        default=["extract", "metadata", "splits"],
        choices=["extract", "metadata", "splits", "balance", "report", "contact_sheets"],
        help="Stages to run"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        dataset_source=args.dataset_source,
        num_frames=args.num_frames,
        sampling_strategy=args.sampling_strategy,
        det_thresh=args.det_thresh,
        face_confidence_threshold=args.face_confidence_threshold,
        blur_threshold=args.blur_thresh,
        min_face_size=args.min_face_size,
        skip_existing=args.skip_existing,
        stages=args.stages,
        seed=args.seed,
    )
    
    print(f"Processing videos from: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Dataset source: {config.dataset_source}")
    print(f"Stages: {config.stages}")
    
    # Run processing
    if "extract" in config.stages:
        print("\n=== Extracting frames ===")
        
        # Initialize face detector
        detector = FaceDetector(
            device=config.device,
            det_thresh=config.det_thresh
        )
        
        # Find all video files
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        video_files = []
        for ext in video_extensions:
            video_files.extend(config.input_dir.rglob(f"*{ext}"))
        
        if not video_files:
            print(f"No video files found in {config.input_dir}")
            return 1
            
        print(f"Found {len(video_files)} video files")
        
        # Process each video
        all_video_rows = []
        all_frame_rows = []
        all_candidate_rows = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                result = process_video(video_path, config, detector)
                all_video_rows.append(result["video_row"])
                all_frame_rows.extend(result["frame_rows"])
                all_candidate_rows.extend(result["candidate_rows"])
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                # Create error row
                video_row = {
                    "subject_id": "unknown",
                    "video_id": hashlib.sha256(str(video_path).encode()).hexdigest()[:16],
                    "video_path": str(video_path.relative_to(config.input_dir) if config.input_dir in video_path.parents else video_path.name),
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
                    "dataset_source": config.dataset_source,
                }
                all_video_rows.append(video_row)
        
        # Write metadata
        if "metadata" in config.stages:
            print("\n=== Writing metadata ===")
            metadata = write_metadata(
                config.output_dir,
                all_video_rows,
                all_frame_rows,
                all_candidate_rows,
            )
            print(f"Written metadata for {len(all_video_rows)} videos")
            print(f"Total frames extracted: {len(all_frame_rows)}")
        
        # Generate splits
        if "splits" in config.stages and all_video_rows:
            print("\n=== Generating splits ===")
            videos_df = metadata["videos"] if "metadata" in locals() else pd.DataFrame(all_video_rows)
            folds = generate_folds(videos_df, config.num_folds, config.seed)
            if folds:
                errors = write_splits(config.output_dir, videos_df, folds)
                if errors:
                    print("Warnings during split generation:")
                    for error in errors:
                        print(f"  {error}")
                else:
                    print(f"Generated {len(folds)} subject-disjoint folds")
            else:
                print("Could not generate splits (not enough subjects or videos)")
    
    print(f"\n✅ Processing complete! Output saved to: {config.output_dir}")
    print(f"Next steps:")
    print(f"  1. Check the output directory structure")
    print(f"  2. Run evaluation with: uv run python scripts/eval_official_test.py --dataset {config.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())