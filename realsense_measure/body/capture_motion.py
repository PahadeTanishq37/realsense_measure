"""
realsense_measure/body/capture_motion.py
========================================
Stillness verification and burst motion analysis for --target body stop-and-hold captures.

Monitors subject stillness during the multi-second burst capture:
- Silhouette centroid displacement (lateral and vertical motion in mm)
- 3D depth centroid displacement (depth/sway motion in mm)
- Flags any yaw stop where motion exceeds 10.0 mm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BurstMotionResult:
    """Diagnostic stillness results across a multi-frame burst."""
    frames_used: int
    real_elapsed_sec: float
    motion_mm: float
    silhouette_motion_mm: float
    depth_motion_mm: float
    is_flagged: bool
    threshold_mm: float = 10.0
    silhouette_centroids: List[Tuple[float, float, float]] = field(default_factory=list)
    depth_centroids: List[Tuple[float, float, float]] = field(default_factory=list)


def extract_subject_mask(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract subject binary mask for stillness tracking.
    Uses MediaPipe person mask if available, with depth-range connected-component fallback.
    """
    try:
        from tools.body_capture_report import extract_person_mask
        mask = extract_person_mask(color_bgr, depth_m, bg_depth_m)
        if mask is not None and np.sum(mask) >= 50:
            return mask
    except Exception:
        pass

    # Depth fallback: foreground subject in 0.40 - 2.80 m
    valid = (depth_m >= 0.40) & (depth_m <= 2.80)
    if np.any(valid):
        med_z = float(np.median(depth_m[valid]))
        slice_m = (depth_m >= max(0.40, med_z - 0.45)) & (depth_m <= med_z + 0.45)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(slice_m.astype(np.uint8), connectivity=8)
        if n_lbl > 1:
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[largest, cv2.CC_STAT_AREA] >= 100:
                return (lbls == largest)

    return (depth_m > 0.0)


def compute_frame_centroids(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (silhouette_centroid_3d, depth_centroid_3d) in metres for a single frame.

    Parameters
    ----------
    color_bgr : np.ndarray
        (H, W, 3) BGR image
    depth_m : np.ndarray
        (H, W) float32 depth map in metres
    bg_depth_m : Optional[np.ndarray]
        Optional background depth reference
    intrinsics : Optional[Tuple[float, float, float, float]]
        (fx, fy, cx, cy). If None, defaults to D455 portrait/landscape nominal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (silhouette_centroid_m, depth_centroid_m) as (3,) arrays [X, Y, Z]
    """
    h, w = depth_m.shape[:2]
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
    else:
        # Default nominal pinhole intrinsics
        fx, fy = float(w), float(w)
        cx, cy = float(w) / 2.0, float(h) / 2.0

    mask = extract_subject_mask(color_bgr, depth_m, bg_depth_m)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    # 1. Silhouette Centroid (2D image centroid projected to metric depth)
    u_sil = float(np.mean(xs))
    v_sil = float(np.mean(ys))

    valid_mask = mask & (depth_m > 0.0)
    if np.any(valid_mask):
        z_sil = float(np.median(depth_m[valid_mask]))
    else:
        z_sil = 1.5

    x_sil = ((u_sil - cx) / fx) * z_sil
    y_sil = ((v_sil - cy) / fy) * z_sil
    sil_centroid = np.array([x_sil, y_sil, z_sil], dtype=np.float64)

    # 2. 3D Depth Centroid (mean 3D point of valid body surface pixels)
    if np.any(valid_mask):
        val_ys, val_xs = np.where(valid_mask)
        val_z = depth_m[valid_mask].astype(np.float64)
        val_x = ((val_xs - cx) / fx) * val_z
        val_y = ((val_ys - cy) / fy) * val_z

        depth_centroid = np.array([
            float(np.mean(val_x)),
            float(np.mean(val_y)),
            float(np.mean(val_z)),
        ], dtype=np.float64)
    else:
        depth_centroid = sil_centroid.copy()

    return sil_centroid, depth_centroid


def evaluate_burst_stillness(
    burst_colors: List[np.ndarray],
    burst_depths: List[np.ndarray],
    burst_timestamps: Optional[List[float]] = None,
    bg_depth_m: Optional[np.ndarray] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    threshold_mm: float = 10.0,
) -> BurstMotionResult:
    """
    Evaluate subject stillness across a burst sequence.

    Calculates maximum pairwise motion (peak-to-peak) of both the silhouette
    centroid and the 3D depth centroid. If either exceeds threshold_mm (default 10 mm),
    is_flagged is set to True.
    """
    n_frames = len(burst_depths)
    if n_frames < 2:
        return BurstMotionResult(
            frames_used=n_frames,
            real_elapsed_sec=0.0,
            motion_mm=0.0,
            silhouette_motion_mm=0.0,
            depth_motion_mm=0.0,
            is_flagged=False,
            threshold_mm=threshold_mm,
        )

    if burst_timestamps is not None and len(burst_timestamps) >= 2:
        real_elapsed = float(burst_timestamps[-1] - burst_timestamps[0])
    else:
        real_elapsed = float(n_frames / 30.0)

    sil_centroids: List[np.ndarray] = []
    dep_centroids: List[np.ndarray] = []

    for c, d in zip(burst_colors, burst_depths):
        s_c, d_c = compute_frame_centroids(c, d, bg_depth_m=bg_depth_m, intrinsics=intrinsics)
        sil_centroids.append(s_c)
        dep_centroids.append(d_c)

    sil_stack = np.array(sil_centroids)  # (N, 3)
    dep_stack = np.array(dep_centroids)  # (N, 3)

    # Maximum pairwise displacement across all frames in burst (mm)
    # Range of centroids along each coordinate axis gives robust motion span
    sil_span_mm = float(np.linalg.norm(np.ptp(sil_stack, axis=0)) * 1000.0)
    dep_span_mm = float(np.linalg.norm(np.ptp(dep_stack, axis=0)) * 1000.0)

    motion_mm = max(sil_span_mm, dep_span_mm)
    is_flagged = bool(motion_mm > threshold_mm)

    return BurstMotionResult(
        frames_used=n_frames,
        real_elapsed_sec=real_elapsed,
        motion_mm=motion_mm,
        silhouette_motion_mm=sil_span_mm,
        depth_motion_mm=dep_span_mm,
        is_flagged=is_flagged,
        threshold_mm=threshold_mm,
        silhouette_centroids=[c.tolist() for c in sil_centroids],
        depth_centroids=[c.tolist() for c in dep_centroids],
    )

