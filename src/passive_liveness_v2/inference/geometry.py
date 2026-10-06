"""Per-face geometry ratios from 5-point landmarks (Step 3 branch).

Ported from the geometry cell in ``preprocess.py``. The seven
ratios, in the order the PAD head expects them (GEOMETRY_COLS in
notebook.py), normalized by the bbox diagonal:

    interocular_dist_norm, eye_to_nose_dist_norm,
    nose_to_mouth_dist_norm, eye_to_mouth_dist_norm,
    mouth_width_norm, face_aspect_ratio, pose_asym
"""

import numpy as np

GEOMETRY_DIM = 7


def compute_pose_asym(landmarks: np.ndarray) -> float | None:
    """Nose x-offset asymmetry between the eyes, normalized by the
    interocular distance. insightface/RetinaFace 5-point order:
    left_eye, right_eye, nose, mouth_left, mouth_right."""
    if landmarks is None or landmarks.shape[0] < 3:
        return None
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    interocular = float(np.linalg.norm(left_eye - right_eye))
    if interocular <= 1e-6:
        return None
    d_left = abs(float(nose[0]) - float(left_eye[0]))
    d_right = abs(float(right_eye[0]) - float(nose[0]))
    return abs(d_left - d_right) / interocular


def compute_geometry(
    landmarks_5pt: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray | None:
    """The 7 geometry ratios as a float32 vector, or None if the
    landmarks are unusable."""
    if landmarks_5pt is None or landmarks_5pt.shape[0] < 5:
        return None

    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    norm = (w * w + h * h) ** 0.5  # bbox diagonal, scale-normalizer

    left_eye, right_eye, nose, mouth_l, mouth_r = landmarks_5pt
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0

    pose_asym = compute_pose_asym(landmarks_5pt)
    if pose_asym is None:
        return None

    return np.array(
        [
            float(np.linalg.norm(left_eye - right_eye) / norm),
            float(np.linalg.norm(eye_mid - nose) / norm),
            float(np.linalg.norm(nose - mouth_mid) / norm),
            float(np.linalg.norm(eye_mid - mouth_mid) / norm),
            float(np.linalg.norm(mouth_l - mouth_r) / norm),
            float(w / h),
            float(pose_asym),
        ],
        dtype=np.float32,
    )
