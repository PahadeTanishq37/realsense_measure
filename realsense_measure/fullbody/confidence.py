"""
realsense_measure/fullbody/confidence.py

Per-pixel confidence calculation and depth filtering:
- Incidence angle falloff (cos(theta)^p) computed from locally smoothed surface normals.
- Distance-to-depth-edge weighting (silhouette edges against background, not internal dropouts).
- Sensor z^2 noise model: sigma(z) = sigma_0 * (z / z_0)^2.
- Temporal standard deviation from burst.
- Burst validity count ratio.
- Invalidation of low-confidence pixels.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig

logger = logging.getLogger(__name__)


def compute_depth_normals(
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
) -> np.ndarray:
    """
    Compute dense 3D surface normals from depth map.

    Applies a 3x3 median pre-filter on valid depth pixels to suppress
    high-frequency sensor noise from corrupting normal estimates while
    preserving physical anatomical contours.

    Returns
    -------
    np.ndarray:
        HxWx3 float32 array of unit normal vectors pointing toward the camera.
    """
    h, w = depth_m.shape[:2]
    valid = depth_m > 0

    # Median pre-filter suppresses single-pixel noise spikes in normal derivatives
    if np.sum(valid) > 50:
        depth_clean = cv2.medianBlur(depth_m.astype(np.float32), 3)
        depth_clean[~valid] = 0.0
    else:
        depth_clean = depth_m

    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Backproject to 3D point grid
    z = depth_clean.copy()
    valid_z = z > 0
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid_z] = (u_grid[valid_z] - cx) * z[valid_z] / fx
    y[valid_z] = (v_grid[valid_z] - cy) * z[valid_z] / fy

    # Central difference for tangent vectors
    dz_dx = np.zeros((h, w, 3), dtype=np.float32)
    dz_dy = np.zeros((h, w, 3), dtype=np.float32)

    # dx along width
    dz_dx[:, 1:-1, 0] = x[:, 2:] - x[:, :-2]
    dz_dx[:, 1:-1, 1] = y[:, 2:] - y[:, :-2]
    dz_dx[:, 1:-1, 2] = z[:, 2:] - z[:, :-2]

    # dy along height
    dz_dy[1:-1, :, 0] = x[2:, :] - x[:-2, :]
    dz_dy[1:-1, :, 1] = y[2:, :] - y[:-2, :]
    dz_dy[1:-1, :, 2] = z[2:, :] - z[:-2, :]

    # Cross product dx x dy gives surface normal
    normals = np.cross(dz_dx, dz_dy)
    norm_len = np.linalg.norm(normals, axis=2, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    # Orient normal toward the camera (Z component should be negative in camera frame)
    flip_mask = normals[:, :, 2] > 0
    normals[flip_mask] = -normals[flip_mask]

    normals[~valid] = 0.0
    return normals


def compute_confidence_map(
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cfg: FullBodyConfig,
    depth_std_m: Optional[np.ndarray] = None,
    valid_counts: Optional[np.ndarray] = None,
    burst_total: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel confidence map w in [0, 1] and filtered depth map.

    Parameters
    ----------
    depth_m:
        HxW metric depth.
    intrinsics:
        Camera intrinsics.
    cfg:
        FullBodyConfig instance.
    depth_std_m:
        Optional per-pixel standard deviation from burst.
    valid_counts:
        Optional per-pixel validity count from burst.
    burst_total:
        Total frames in burst.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        (confidence_map, filtered_depth_m)
    """
    h, w = depth_m.shape[:2]
    valid = (depth_m >= cfg.depth_min_m) & (depth_m <= cfg.depth_max_m) & np.isfinite(depth_m)
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    # 1. Incidence angle factor: cos(theta)^power
    normals = compute_depth_normals(depth_m, intrinsics)
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    ray_x = (u_grid - cx) / fx
    ray_y = (v_grid - cy) / fy
    ray_z = np.ones((h, w), dtype=np.float32)
    ray_len = np.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)
    view_rays = np.stack([ray_x / ray_len, ray_y / ray_len, ray_z / ray_len], axis=2)

    # Dot product of normal with view ray (ray points away from camera, normal towards)
    cos_theta = -np.sum(normals * view_rays, axis=2)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    w_angle = np.power(cos_theta, cfg.incidence_angle_power).astype(np.float32)

    # 2. Distance to depth edge factor:
    # Silhouette jump edges (> 5 cm step between adjacent valid pixels, or image borders).
    # Isolated zero pixels (sensor dropouts) are NOT treated as silhouette edges so that
    # the entire body interior is not artificially penalized.
    both_valid_x = valid[:, 1:] & valid[:, :-1]
    grad_x = np.zeros((h, w), dtype=np.float32)
    grad_x[:, 1:][both_valid_x] = np.abs(depth_m[:, 1:][both_valid_x] - depth_m[:, :-1][both_valid_x])

    both_valid_y = valid[1:, :] & valid[:-1, :]
    grad_y = np.zeros((h, w), dtype=np.float32)
    grad_y[1:, :][both_valid_y] = np.abs(depth_m[1:, :][both_valid_y] - depth_m[:-1, :][both_valid_y])

    edges = np.zeros((h, w), dtype=np.uint8)
    edges[:, 1:][grad_x[:, 1:] > 0.05] = 1
    edges[1:, :][grad_y[1:, :] > 0.05] = 1
    # Image frame borders
    edges[0, :] = 1
    edges[-1, :] = 1
    edges[:, 0] = 1
    edges[:, -1] = 1

    dist_to_edge = cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3)
    w_edge = np.clip(dist_to_edge / cfg.edge_distance_falloff_px, 0.0, 1.0).astype(np.float32)

    # 3. Quadratic distance noise model: sigma(z) = sigma_base * (z / z_ref)^2
    sigma_z = cfg.noise_sigma_base_m * np.square(np.maximum(0.1, depth_m) / cfg.noise_reference_distance_m)
    w_theory = (1.0 / (1.0 + np.square(sigma_z / 0.005))).astype(np.float32)

    # 4. Temporal standard deviation factor:
    # When burst statistics are available, take minimum with empirical temporal noise
    # (avoid squaring variance by multiplying two independent noise penalties)
    if depth_std_m is not None:
        w_temp = (1.0 / (1.0 + np.square(np.maximum(0.0, depth_std_m) / 0.005))).astype(np.float32)
        w_noise = np.minimum(w_theory, w_temp)
    else:
        w_noise = w_theory

    # 5. Validity count factor
    if valid_counts is not None and burst_total > 0:
        w_valid = np.clip(valid_counts / float(burst_total), 0.0, 1.0).astype(np.float32)
    else:
        w_valid = np.ones((h, w), dtype=np.float32)

    # Combined confidence
    confidence = w_angle * w_edge * w_noise * w_valid
    confidence[~valid] = 0.0

    # Filter out low-confidence pixels
    filtered_depth = depth_m.copy()
    filtered_depth[confidence < cfg.min_confidence_threshold] = 0.0

    return confidence, filtered_depth
