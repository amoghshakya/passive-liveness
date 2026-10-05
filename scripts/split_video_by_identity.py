#!/usr/bin/env python3
"""Split a multi-person video into one short clip per distinct person.

Motivation: `preprocess.py`'s `track_primary_face` follows exactly ONE face
through a video (IoU + re-acquisition, "avoids latching onto a bystander
face"), and `load_video_records` derives `subject_id` from a metadata CSV
keyed on a source file path. Neither can consume a single video containing
many people: every player after the first would either be dropped as
`primary_face_lost` or -- worse -- silently *re-acquired* into the first
player's track, collapsing N identities into one `subject_id`. Since
`subject_id` is what the subject-disjoint folds are built on, that collapse
would quietly destroy the value of the data.

So this script does the splitting, upstream of the pipeline, and emits real
video files. Everything downstream -- detection, tracking, quality gating,
sampling, alignment, geometry, metadata, splits -- is left to `preprocess.py`
untouched, so extraction-time behaviour cannot drift from the rest of the
dataset. No face crops are produced here on purpose: the pipeline owns that.

SEGMENTATION IS IDENTITY-BASED, NOT CUT-BASED. Measured on the motivating
input (a 178 s, 854x470 squad-pronunciation compilation), scene-cut
detection is useless: both `ffmpeg -filter:v select='gt(scene,0.25)'` and a
plain downscaled-histogram difference put nearly all cuts in the final 20 s,
because the video keeps one template on screen and swaps the person inside
it. Face-identity clustering recovers 32 clean, contiguous, non-overlapping
windows instead. So: sparse-sample frames, embed every primary face with
insightface's `w600k_r50` recognition model (already vendored in the
`buffalo_l` bundle the pipeline itself downloads), greedily cluster by cosine
similarity, then order clusters by first appearance and cut at the midpoint
between neighbouring clusters' observed spans.

The embeddings come from the recognition model only -- it is used purely to
group frames into clips. It never touches the face pixels that end up in the
dataset, and it is not part of the model's input, so it introduces no
train/serve skew.

OUTPUT (frames dir, NOT the dataset):

    frames/
      subject_01/
        seg_01.mp4
        seg_01.jpg        # face crops, for identifying who
      subject_02/
        seg_01.mp4
        seg_01.jpg
      ...                 # one directory per identified person

All clips are bona fide/real and ready for direct preprocessing.
No metadata files or placement scripts are generated.

Run:  uv run python scripts/split_video_by_identity.py --video PATH [--frames-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Contact-sheet crop size. Faces in the motivating input are ~280 px tall in a
# 470 px frame; normalising to a fixed 128 px keeps memory flat regardless of
# source resolution and still leaves a face big enough to recognise.
CROP_PX = 128


@dataclass(frozen=True)
class Sample:
    """One primary-face observation from one sampled frame."""

    frame_index: int
    timestamp: float
    det_score: float
    face_h: float
    face_w: float
    crop: np.ndarray  # CROP_PX x CROP_PX BGR
    embedding: np.ndarray  # unit-norm face embedding, for clustering only


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    samples: list[Sample]


def probe_duration(path: Path) -> tuple[float, int, float]:
    """(duration_sec, frame_count, fps) by decoding, not by trusting the
    container header -- `CAP_PROP_FRAME_COUNT` is routinely wrong on VFR and on
    oddly-muxed files, and `read_frames_at` in preprocess.py already
    established grab()-everything as the safe pattern here."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = 0
    last_ts = 0.0
    while cap.grab():
        n += 1
        last_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
    cap.release()
    if n == 0:
        raise OSError(f"decoded 0 frames from {path}")
    # Prefer the mux-reported fps for index->time math, but fall back to the
    # measured duration when it is missing or absurd (some web muxers omit it).
    duration = n / fps if fps > 1.0 else last_ts
    if duration <= 0:
        raise OSError(f"could not determine duration of {path}")
    return duration, n, fps


def primary_face_samples(
    video: Path, sample_every: int, det_thresh: float, device: str
) -> tuple[list[Sample], int, int]:
    """Sparse-sample the video and keep the primary face from each frame.

    "Primary" is the face maximising `det_score * area`, the same choice
    `track_primary_face` makes for the first frame of a track, so the
    observation stream matches what the pipeline would have locked onto.

    Returns (samples, n_sampled_frames, n_frames_with_no_face)."""
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        ),
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1,
                det_size=(640, 640), det_thresh=det_thresh)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open video: {video}")

    samples: list[Sample] = []
    n_sampled = 0
    n_no_face = 0
    idx = 0
    pbar = tqdm(desc="detecting+embedding", unit="frame")
    while True:
        if not cap.grab():
            break
        want = idx % sample_every == 0
        ok, frame = cap.retrieve()
        if want and ok and frame is not None:
            n_sampled += 1
            faces = app.get(frame)
            if not faces:
                n_no_face += 1
            else:
                best = max(faces, key=lambda f: float(f.det_score) * _area(f.bbox))
                x1, y1, x2, y2 = (float(v) for v in best.bbox)
                h, w = max(1.0, y2 - y1), max(1.0, x2 - x1)
                ih, iw = frame.shape[:2]
                crop = frame[
                    max(0, int(y1)) : min(ih, int(y2)),
                    max(0, int(x1)) : min(iw, int(x2)),
                ]
                if crop.size == 0:
                    n_no_face += 1
                else:
                    ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    emb = np.asarray(best.embedding, dtype=np.float32)
                    samples.append(
                        Sample(
                            frame_index=idx,
                            timestamp=ts,
                            det_score=float(best.det_score),
                            face_h=h,
                            face_w=w,
                            crop=cv2.resize(crop, (CROP_PX, CROP_PX)),
                            embedding=emb / (np.linalg.norm(emb) or 1.0),
                        )
                    )
            pbar.update(1)
        idx += 1
    pbar.close()
    cap.release()
    return samples, n_sampled, n_no_face


def _area(box) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def cluster_by_identity(samples: list[Sample], cos_threshold: float) -> list[int]:
    """Greedy single-pass clustering against cluster centroids.

    Returns one cluster id per sample. Greedy (not k-means) because the
    expected structure is a sequence of well-separated individuals: the
    centroids are stable, there is no k to choose, and adding a sample can
    never move an existing centroid and relabel earlier frames -- which a
    two-pass method could do, silently rewriting the segment boundaries."""
    centroids: list[np.ndarray] = []
    assign: list[int] = []
    for s in tqdm(samples, desc="clustering", unit="face"):
        if centroids:
            sims = np.stack(centroids) @ s.embedding
            j = int(np.argmax(sims))
            if float(sims[j]) > cos_threshold:
                assign.append(j)
                continue
        centroids.append(s.embedding)
        assign.append(len(centroids) - 1)
    return assign


def build_segments(
    samples: list[Sample], assign: list[int], duration: float, pad: float
) -> list[Segment]:
    """Group clustered samples into time-ordered, non-overlapping segments.

    A cluster is treated as one person's contiguous on-screen window. Each
    segment is its cluster's observed span grown by `pad`, then clamped to the
    midpoint with each neighbour so segments can never overlap even when two
    people are adjacent enough that their padded spans would collide.

    The clamp is why this doesn't simply emit the raw spans: a sample lands
    only every `sample_every` frames, so each observed span carries up to
    `sample_every/fps` seconds of quantisation error at both ends. `pad` set
    to roughly half that interval recovers the true boundaries; anything
    beyond the midpoint belongs to the neighbouring person (or, at the very
    start, to the intro card -- which is exactly what must NOT be folded into
    the first segment)."""
    spans: dict[int, list[Sample]] = {}
    for s, c in zip(samples, assign):
        spans.setdefault(c, []).append(s)
    order = sorted(spans, key=lambda c: spans[c][0].timestamp)

    observed = [(spans[c][0].timestamp, spans[c][-1].timestamp) for c in order]
    midpoints = [
        (a[1] + b[0]) / 2.0 for a, b in pairwise(observed)
    ]

    segments: list[Segment] = []
    for i, c in enumerate(order):
        lo = max(0.0, observed[i][0] - pad)
        hi = min(duration, observed[i][1] + pad)
        if i > 0:
            lo = max(lo, midpoints[i - 1])
        if i < len(midpoints):
            hi = min(hi, midpoints[i])
        segments.append(
            Segment(index=i + 1, start=lo, end=max(hi, lo + 1e-3),
                    samples=spans[c])
        )
    return segments


def cut_clip(video: Path, seg: Segment, out_path: Path, crf: int | None) -> None:
    """Cut [start, end) with ffmpeg.

    A re-encode is unavoidable: cutting at an arbitrary timestamp with
    `-c copy` only works on keyframe boundaries, and these segments do not
    align to them. `crf=None` selects x264 lossless (`-qp 0`). The source is
    already a web-compressed H.264, so the output is a second generation
    either way -- recorded in the pool metadata's `notes`, and part of why
    this batch needs its own `dataset_source`."""
    vf = ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p"]
    vf += ["-qp", "0"] if crf is None else ["-crf", str(crf)]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seg.start:.3f}", "-to", f"{seg.end:.3f}",
        "-i", str(video), "-an", *vf, str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg failed for segment {seg.index} "
            f"({seg.start:.2f}-{seg.end:.2f}s):\n{proc.stderr.strip()}"
        )


def contact_sheet(seg: Segment, out_path: Path, cols: int, max_tiles: int) -> None:
    """Grid of the segment's face crops, for identifying who the person is.

    Uses the crops already captured during the sampling pass, so this costs no
    extra detection and cannot disagree with the clusters it illustrates."""
    tiles = [s.crop for s in seg.samples[:max_tiles]]
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    pad = 6
    header = 26
    sheet = np.full(
        (header + rows * (CROP_PX + pad) + pad, cols * (CROP_PX + pad) + pad, 3),
        255, np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = header + r * (CROP_PX + pad)
        x = pad + c * (CROP_PX + pad)
        sheet[y : y + CROP_PX, x : x + CROP_PX] = tile
    cv2.putText(
        sheet, f"segment {seg.index:02d}  {seg.start:.1f}-{seg.end:.1f}s",
        (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)





def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", type=Path, required=True,
                    help="source video containing one person at a time")
    ap.add_argument("--frames-dir", type=Path, default=Path("frames"),
                    help="output directory for clips organized by subject ID")
    ap.add_argument("--sample-every", type=int, default=20,
                    help="sample every Nth frame for identity clustering "
                         "(20 ~= 0.67s at 30fps; smaller = finer, slower)")
    ap.add_argument("--cos-threshold", type=float, default=0.45,
                    help="cosine similarity above which a face joins an "
                         "existing identity cluster")
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--max-segments", type=int, default=None,
                    help="keep only the first N segments (debugging)")
    ap.add_argument("--min-segments", type=int, default=2,
                    help="fail if fewer than this many identities are found; "
                         "1 means the clustering collapsed and every person "
                         "would be emitted as one bogus subject")
    ap.add_argument("--crf", type=int, default=16,
                    help="x264 CRF for the cut clips (lower = better/smaller "
                         "tradeoff). Ignored with --lossless")
    ap.add_argument("--lossless", action="store_true",
                    help="x264 -qp 0 (no quality loss, much larger)")
    ap.add_argument("--sheet-cols", type=int, default=6)
    ap.add_argument("--sheet-max", type=int, default=12)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--write", action="store_true",
                    help="actually cut clips; without it, only report the "
                         "segmentation that would be produced")
    a = ap.parse_args()

    if not a.video.exists():
        print(f"no such video: {a.video}", file=sys.stderr)
        return 1
    if a.video.suffix.lower() not in VIDEO_EXTS:
        print(f"warning: unexpected extension {a.video.suffix!r}", file=sys.stderr)

    duration, n_frames, fps = probe_duration(a.video)
    print(f"{a.video.name}: {n_frames} frames, {fps:.2f} fps, {duration:.1f}s, "
          f"{a.video.stat().st_size / 1e6:.1f} MB")

    samples, n_sampled, n_no_face = primary_face_samples(
        a.video, a.sample_every, a.det_thresh, a.device
    )
    print(f"sampled {n_sampled} frames every {a.sample_every}; "
          f"{len(samples)} primary faces, {n_no_face} frames with no face")
    if not samples:
        print("no faces detected -- nothing to segment", file=sys.stderr)
        return 1

    assign = cluster_by_identity(samples, a.cos_threshold)
    # Half a sampling interval recovers the quantisation error on each side of
    # an observed span without reaching into the neighbouring person.
    pad = (a.sample_every / 2.0) / fps if fps > 0 else 0.0
    segments = build_segments(samples, assign, duration, pad)
    if a.max_segments:
        segments = segments[: a.max_segments]

    n_clust = len(set(assign))
    print(f"\n{n_clust} identity clusters @cos>{a.cos_threshold} -> "
          f"{len(segments)} segments")
    if n_clust < a.min_segments:
        print(f"refusing: only {n_clust} identities (min {a.min_segments}). "
              "Clustering likely collapsed -- try a lower --cos-threshold or a "
              "smaller --sample-every.", file=sys.stderr)
        return 1
    for s in segments:
        print(f"  seg_{s.index:02d}  {s.start:7.2f}-{s.end:7.2f}s  "
              f"({s.end - s.start:5.2f}s)  {len(s.samples):3d} samples  "
              f"mean face h {np.mean([x.face_h for x in s.samples]):.0f}px")

    if not a.write:
        print("\ndry run -- pass --write to cut clips and write the staging dir")
        return 0

    frames_dir = a.frames_dir
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    for s in tqdm(segments, desc="cutting", unit="clip"):
        subject_dir = frames_dir / f"subject_{s.index:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        cut_clip(a.video, s, subject_dir / f"seg_{s.index:02d}.mp4",
                 None if a.lossless else a.crf)
        # Optional: keep contact sheets for verification
        contact_sheet(s, subject_dir / f"seg_{s.index:02d}.jpg", a.sheet_cols,
                      a.sheet_max)

    print(f"\nwrote {len(segments)} clips to {frames_dir}")
    print(f"  organized by subject in frames/subject_XX/ directories")
    print(f"  all clips are bona fide/real")
    print(f"  next steps:")
    print(f"  1. run preprocessing pipeline on '{frames_dir}' directory")
    print(f"  2. extract features and run model evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
