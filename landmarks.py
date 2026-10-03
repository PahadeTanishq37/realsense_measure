"""
3D face landmark detection for landmark-guided registration.

This is the only file in the project that imports mediapipe, so it can be
swapped out later without touching anything else (same pattern as
camera/realsense_capture.py being the only file that imports pyrealsense2).

Landmarks are used to compute a RIGID TRANSFORM between frames (see
registration.py's rigid_transform_from_correspondences and
register_frame_pair_landmark_guided) — never to move or snap the actual
captured points. The raw geometry stays exactly as captured; only the
frame-to-frame alignment uses landmarks as reliable correspondence points.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import open3d as o3d

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# A small, fixed set of stable landmark indices, out of mediapipe's full 468,
# chosen because they sit on rigid bone/cartilage (not lips/cheeks, which
# deform with expression) and are reliably detected across head angles.
LANDMARK_INDICES: dict[str, int] = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "nose_tip": 1,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "chin": 152,
}

_landmarker = None  # lazy singleton -- model load is slow, do it once


def _ensure_model_downloaded() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not auto-download the face landmarker model ({e}). "
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {MODEL_PATH}"
        ) from e


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
        _ensure_model_downloaded()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_face_landmarks_3d(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic,
    depth_patch: int = 3,
) -> dict[str, np.ndarray] | None:
    """
    Detect facial landmarks and back-project them to 3D camera-space meters.

    Parameters
    ----------
    color_bgr:
        Color frame as captured elsewhere in this project (BGR, OpenCV
        convention).
    depth_m:
        Depth frame in meters, same resolution/alignment as color_bgr.
    intrinsics_o3d:
        Camera intrinsics, as already built elsewhere in this project
        (camera/realsense_capture.py).
    depth_patch:
        Side length of the square patch used for a median depth read at
        each landmark, instead of a single noisy pixel.

    Returns
    -------
    dict mapping each name in LANDMARK_INDICES to a 3D point [x, y, z] in
    camera-space meters, for every landmark that was both detected by
    mediapipe AND had valid (non-zero) depth nearby. Returns None if no
    face was detected in the frame at all.
    """
    import mediapipe as mp

    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]  # num_faces=1
    h, w = color_bgr.shape[:2]
    fx, fy = intrinsics_o3d.get_focal_length()
    cx, cy = intrinsics_o3d.get_principal_point()

    half = depth_patch // 2
    out: dict[str, np.ndarray] = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = face[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        y0, y1 = max(0, py - half), min(h, py + half + 1)
        x0, x1 = max(0, px - half), min(w, px + half + 1)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            continue
        z = float(np.median(valid))
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        out[name] = np.array([x, y, z])

    return out if out else None
