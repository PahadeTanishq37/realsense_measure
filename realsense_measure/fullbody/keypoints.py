"""
realsense_measure/fullbody/keypoints.py

Swappable 2D keypoint detection (KeypointDetector protocol),
MediaPipePose backend, and robust 3D keypoint lifting via local depth window.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import Keypoints3D

logger = logging.getLogger(__name__)

# Core anatomical keypoints used across alignment, segmentation, and anthropometry
CORE_LANDMARK_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


@runtime_checkable
class KeypointDetector(Protocol):
    """Protocol for pluggable 2D pose/keypoint detectors."""

    def detect(
        self, color_bgr: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
        """
        Detect 2D keypoints on a BGR image.

        Returns
        -------
        Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
            (pts_2d, confidence_dict, visibility_dict)
            where pts_2d maps joint_name -> (u_px, v_px).
        """
        ...


class MediaPipePoseDetector:
    """
    MediaPipe Pose backend for full-body 2D keypoint detection.
    """

    def __init__(self, static_image_mode: bool = True, model_complexity: int = 1):
        self._pose = None
        self._landmarker = None
        self._static_image_mode = static_image_mode
        self._model_complexity = model_complexity

    def _ensure_initialized(self):
        if self._pose is None and self._landmarker is None:
            try:
                import mediapipe as mp
                # Try solutions.pose first (legacy mediapipe < 0.10)
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                    self._pose = mp.solutions.pose.Pose(
                        static_image_mode=self._static_image_mode,
                        model_complexity=self._model_complexity,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                    )
                elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision
                    from pathlib import Path
                    candidates = [
                        Path(__file__).parent.parent / "models" / "pose_landmarker.task",
                        Path(__file__).resolve().parents[2] / "models" / "pose_landmarker.task",
                        Path("models/pose_landmarker.task"),
                    ]
                    model_path = next((p for p in candidates if p.exists()), None)
                    if model_path is not None:
                        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
                        options = vision.PoseLandmarkerOptions(
                            base_options=base_options,
                            output_segmentation_masks=False,
                            min_pose_detection_confidence=0.3,
                            min_tracking_confidence=0.3,
                        )
                        self._landmarker = vision.PoseLandmarker.create_from_options(options)
                    else:
                        logger.warning("  [keypoints] pose_landmarker.task not found for mediapipe tasks API.")
            except Exception as exc:
                logger.warning(f"  [keypoints] Could not initialize MediaPipe Pose: {exc}")

    def detect(
        self, color_bgr: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
        self._ensure_initialized()
        h, w = color_bgr.shape[:2]
        pts_2d: Dict[str, Tuple[float, float]] = {}
        conf_dict: Dict[str, float] = {}
        vis_dict: Dict[str, float] = {}

        mp_indices = {
            "nose": 0,
            "left_eye": 2, "right_eye": 5,
            "left_ear": 7, "right_ear": 8,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "left_heel": 29, "right_heel": 30,
            "left_foot_index": 31, "right_foot_index": 32,
        }

        if self._landmarker is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self._landmarker.detect(mp_image)
                if not results.pose_landmarks:
                    return pts_2d, conf_dict, vis_dict
                landmarks = results.pose_landmarks[0]
                for name, idx in mp_indices.items():
                    lm = landmarks[idx]
                    u = lm.x * (w - 1)
                    v = lm.y * (h - 1)
                    pts_2d[name] = (float(u), float(v))
                    conf_dict[name] = float(getattr(lm, "presence", 1.0))
                    vis_dict[name] = float(getattr(lm, "visibility", 1.0))
            except Exception as exc:
                logger.debug(f"  [keypoints] Tasks pose detection error: {exc}")
            return pts_2d, conf_dict, vis_dict

        if self._pose is not None:
            try:
                rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                results = self._pose.process(rgb)
                if not results.pose_landmarks:
                    return pts_2d, conf_dict, vis_dict

                for name, idx in mp_indices.items():
                    lm = results.pose_landmarks.landmark[idx]
                    u = lm.x * (w - 1)
                    v = lm.y * (h - 1)
                    pts_2d[name] = (float(u), float(v))
                    conf_dict[name] = float(getattr(lm, "presence", 1.0))
                    vis_dict[name] = float(getattr(lm, "visibility", 1.0))

            except Exception as exc:
                logger.debug(f"  [keypoints] Solutions pose detection error: {exc}")

        return pts_2d, conf_dict, vis_dict


def lift_keypoints_to_3d(
    pts_2d: Dict[str, Tuple[float, float]],
    conf_2d: Dict[str, float],
    vis_2d: Dict[str, float],
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cfg: FullBodyConfig,
) -> Keypoints3D:
    """
    Lift 2D pixel keypoints to 3D metric coordinates using local window depth.

    Rejects depth outliers via Median Absolute Deviation (MAD) in a window around
    the detected 2D position.
    """
    h, w = depth_m.shape[:2]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()
    win = cfg.keypoint_window_size_px

    names: List[str] = []
    points_3d: List[List[float]] = []
    conf_3d: List[float] = []
    vis_3d: List[float] = []

    for name, (u_f, v_f) in pts_2d.items():
        u_int = int(round(u_f))
        v_int = int(round(v_f))

        u_min = max(0, u_int - win)
        u_max = min(w, u_int + win + 1)
        v_min = max(0, v_int - win)
        v_max = min(h, v_int + win + 1)

        patch = depth_m[v_min:v_max, u_min:u_max]
        valid_depths = patch[(patch >= cfg.depth_min_m) & (patch <= cfg.depth_max_m)]

        c2d = conf_2d.get(name, 0.0)
        v2d = vis_2d.get(name, 0.0)

        if len(valid_depths) < 3 or c2d < cfg.keypoint_min_confidence or v2d < cfg.keypoint_min_visibility:
            names.append(name)
            points_3d.append([0.0, 0.0, 0.0])
            conf_3d.append(0.0)
            vis_3d.append(0.0)
            continue

        med_z = float(np.median(valid_depths))
        mad = float(np.median(np.abs(valid_depths - med_z)))
        mad_thresh = max(0.015, cfg.keypoint_depth_mad_scale * mad)
        inliers = valid_depths[np.abs(valid_depths - med_z) <= mad_thresh]

        if len(inliers) == 0:
            final_z = med_z
        else:
            final_z = float(np.median(inliers))

        # Backproject
        x = float((u_f - cx) * final_z / fx)
        y = float((v_f - cy) * final_z / fy)
        z = float(final_z)

        # Depth coverage factor
        frac = len(inliers) / float(patch.size)
        final_conf = float(c2d * min(1.0, frac * 1.5))

        names.append(name)
        points_3d.append([x, y, z])
        conf_3d.append(final_conf)
        vis_3d.append(v2d)

    # Compute derived anatomical midpoint landmarks
    keypoints = Keypoints3D(
        names=names,
        points=np.array(points_3d, dtype=np.float32),
        confidence=np.array(conf_3d, dtype=np.float32),
        visibility=np.array(vis_3d, dtype=np.float32),
        frame="camera",
    )
    _add_derived_midpoints(keypoints)
    return keypoints


def _add_derived_midpoints(kpts: Keypoints3D) -> None:
    """Add mid_hip, mid_shoulder, and neck_base to Keypoints3D in place."""
    pairs = [
        ("mid_hip", "left_hip", "right_hip"),
        ("mid_shoulder", "left_shoulder", "right_shoulder"),
    ]
    for mid_name, l_name, r_name in pairs:
        p_l, c_l, v_l = kpts.get(l_name)
        p_r, c_r, v_r = kpts.get(r_name)
        if p_l is not None and p_r is not None and c_l > 0.1 and c_r > 0.1:
            mid_pt = 0.5 * (p_l + p_r)
            mid_c = min(c_l, c_r)
            mid_v = min(v_l, v_r)
            kpts.names.append(mid_name)
            kpts.points = np.vstack([kpts.points, mid_pt[np.newaxis, :]])
            kpts.confidence = np.append(kpts.confidence, mid_c)
            kpts.visibility = np.append(kpts.visibility, mid_v)


def average_burst_keypoints_2d(
    burst_detections: List[Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
    """
    Robustly average 2D keypoints across a burst of frames before lifting to 3D.
    Takes median pixel coordinates across valid detections to reject transient tracking glitches.
    """
    if not burst_detections:
        return {}, {}, {}

    joint_pts: Dict[str, List[Tuple[float, float]]] = {}
    joint_confs: Dict[str, List[float]] = {}
    joint_vis: Dict[str, List[float]] = {}

    for pts_2d, confs, vis in burst_detections:
        for name, pt in pts_2d.items():
            c = confs.get(name, 0.0)
            v = vis.get(name, 0.0)
            if c > 0.1:
                joint_pts.setdefault(name, []).append(pt)
                joint_confs.setdefault(name, []).append(c)
                joint_vis.setdefault(name, []).append(v)

    avg_pts: Dict[str, Tuple[float, float]] = {}
    avg_confs: Dict[str, float] = {}
    avg_vis: Dict[str, float] = {}

    for name, pts in joint_pts.items():
        if not pts:
            continue
        pts_np = np.array(pts, dtype=np.float64)
        avg_pts[name] = (float(np.median(pts_np[:, 0])), float(np.median(pts_np[:, 1])))
        avg_confs[name] = float(np.mean(joint_confs[name]))
        avg_vis[name] = float(np.mean(joint_vis[name]))

    return avg_pts, avg_confs, avg_vis

