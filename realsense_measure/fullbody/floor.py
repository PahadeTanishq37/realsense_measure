"""
realsense_measure/fullbody/floor.py

Floor plane detection via RANSAC on the lower image region,
cross-stop floor consistency verification, and gravity-aligned
camera-to-world coordinate transform computation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane

logger = logging.getLogger(__name__)


def fit_floor_plane(
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cfg: FullBodyConfig,
    color_bgr: Optional[np.ndarray] = None,
    mask_exclude: Optional[np.ndarray] = None,
) -> Optional[FloorPlane]:
    """
    Fit a floor plane to the lower portion of a depth image.

    Parameters
    ----------
    depth_m:
        HxW metric depth map.
    intrinsics:
        Camera pinhole intrinsics.
    cfg:
        FullBodyConfig instance.
    color_bgr:
        Optional color image (for debug visualization).
    mask_exclude:
        Optional boolean mask of pixels to exclude (e.g. subject feet).

    Returns
    -------
    Optional[FloorPlane]:
        Fitted FloorPlane instance or None if fitting fails.
    """
    h, w = depth_m.shape[:2]
    row_start = int(h * (1.0 - cfg.floor_lower_image_fraction))

    # Mask lower region
    lower_mask = np.zeros((h, w), dtype=bool)
    lower_mask[row_start:, :] = True
    valid = (depth_m > 0.0) & (depth_m >= cfg.depth_min_m) & (depth_m <= cfg.depth_max_m)
    if mask_exclude is not None:
        valid &= (~mask_exclude)
    
    roi_mask = lower_mask & valid
    if np.sum(roi_mask) < 100:
        logger.warning("  [floor] Insufficient valid depth pixels in lower region.")
        return None

    # Backproject to 3D points
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_sel = u_grid[roi_mask].astype(np.float64)
    v_sel = v_grid[roi_mask].astype(np.float64)
    z_sel = depth_m[roi_mask].astype(np.float64)

    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    x_sel = (u_sel - cx) * z_sel / fx
    y_sel = (v_sel - cy) * z_sel / fy
    pts = np.stack([x_sel, y_sel, z_sel], axis=1)

    # RANSAC plane fitting using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=cfg.floor_distance_threshold_m,
            ransac_n=3,
            num_iterations=cfg.floor_ransac_iterations,
        )
    except Exception as exc:
        logger.warning(f"  [floor] Open3D plane segmentation failed: {exc}")
        return None

    if len(inliers) < 50:
        logger.warning("  [floor] RANSAC found too few plane inliers.")
        return None

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-8:
        return None
    normal /= norm_len
    d /= norm_len

    # Camera optical frame: +Y is downwards. Floor normal pointing UPWARDS
    # towards the ceiling must have negative Y component in camera frame.
    expected_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if np.dot(normal, expected_up) < 0:
        normal = -normal
        d = -d

    alignment = np.dot(normal, expected_up)
    if alignment < cfg.floor_normal_min_up_alignment:
        logger.warning(
            f"  [floor] Fitted plane normal [a={normal[0]:.2f}, b={normal[1]:.2f}, c={normal[2]:.2f}] "
            f"deviates from expected vertical (alignment={alignment:.2f} < {cfg.floor_normal_min_up_alignment:.2f})"
        )
        return None

    inlier_pts = pts[inliers]
    point_on_plane = np.mean(inlier_pts, axis=0)
    # Project exactly onto plane
    dist_to_plane = np.dot(point_on_plane, normal) + d
    point_on_plane -= dist_to_plane * normal

    rmse = float(np.sqrt(np.mean((inlier_pts @ normal + d) ** 2)))
    inlier_ratio = float(len(inliers) / len(pts))

    # Compute gravity alignment rotation R_grav_cam: normal -> [0, 1, 0]
    R_grav_cam = compute_rotation_aligning_vectors(normal, np.array([0.0, 1.0, 0.0]))

    return FloorPlane(
        coefficients=np.array([normal[0], normal[1], normal[2], d], dtype=np.float64),
        normal=normal,
        point_on_plane=point_on_plane,
        inlier_count=len(inliers),
        inlier_ratio=inlier_ratio,
        rmse_m=rmse,
        gravity_R_cam=R_grav_cam,
    )


def compute_rotation_aligning_vectors(
    src_vec: np.ndarray,
    dst_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute minimal 3x3 rotation matrix R such that R @ src_vec = dst_vec.
    """
    v1 = src_vec / np.linalg.norm(src_vec)
    v2 = dst_vec / np.linalg.norm(dst_vec)

    dot = np.dot(v1, v2)
    if dot > 0.9999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9999999:
        # 180 degree flip around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, axis)
        axis /= np.linalg.norm(axis)
        # Rodrigues formula for 180 degrees
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2.0 * (K @ K)

    cross = np.cross(v1, v2)
    s = np.linalg.norm(cross)
    kmat = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    R = np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - dot) / (s ** 2))
    return R


def verify_floor_consistency(
    planes: List[FloorPlane],
    cfg: FullBodyConfig,
) -> Tuple[bool, FloorPlane]:
    """
    Verify that floor planes detected across stops agree on orientation and height.

    Returns
    -------
    Tuple[bool, FloorPlane]:
        (is_consistent, consensus_plane)
    """
    if not planes:
        raise ValueError("Cannot verify consistency of empty floor planes list")

    normals = np.array([p.normal for p in planes])  # (S, 3)
    # Average normal
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)

    # Angles to mean normal
    dots = np.clip(normals @ mean_normal, -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(dots))
    max_angle_diff = float(np.max(angles_deg))

    is_consistent = max_angle_diff <= cfg.floor_max_angle_diff_deg
    if not is_consistent:
        logger.warning(
            f"  [floor] Floor normals across stops differ by up to {max_angle_diff:.2f} deg "
            f"(threshold: {cfg.floor_max_angle_diff_deg:.1f} deg)"
        )

    # Use median distance and mean normal for robust consensus plane
    mean_d = float(np.median([p.coefficients[3] for p in planes]))
    consensus_point = -mean_d * mean_normal
    R_grav_cam = compute_rotation_aligning_vectors(mean_normal, np.array([0.0, 1.0, 0.0]))

    consensus_plane = FloorPlane(
        coefficients=np.array([mean_normal[0], mean_normal[1], mean_normal[2], mean_d]),
        normal=mean_normal,
        point_on_plane=consensus_point,
        inlier_count=sum(p.inlier_count for p in planes),
        inlier_ratio=float(np.mean([p.inlier_ratio for p in planes])),
        rmse_m=float(np.mean([p.rmse_m for p in planes])),
        gravity_R_cam=R_grav_cam,
    )
    return is_consistent, consensus_plane

