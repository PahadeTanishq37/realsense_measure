"""
tests/fullbody/test_floor.py

Unit tests for floor plane detection and gravity frame transformation:
- RANSAC fitting on lower image region.
- Normal orientation check (points upward towards ceiling in camera frame).
- Gravity transform maps floor points to Y = 0.
- Cross-stop consistency check.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.floor import (
    compute_rotation_aligning_vectors,
    fit_floor_plane,
    verify_floor_consistency,
)
from realsense_measure.fullbody.types import FloorPlane


def test_floor_plane_ransac_and_gravity_alignment():
    """Verify floor RANSAC detects horizontal floor and T_grav_cam sets floor height to Y=0."""
    h, w = 120, 160
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, 100.0, 100.0, 80.0, 60.0)
    cfg = FullBodyConfig(floor_lower_image_fraction=0.40, depth_min_m=0.3, depth_max_m=3.0)

    # In camera coordinates, camera is at height 0.90 m above the floor.
    # Floor plane equation: Y_cam = 0.90 m (horizontal plane below camera).
    # Since y_cam = (v - cy) * z / fy, for a plane at Y_cam = 0.90:
    # z = 0.90 * fy / (v - cy).
    depth = np.zeros((h, w), dtype=np.float32)
    cy = 60.0
    fy = 100.0
    for v in range(int(h * 0.6), h):
        z_val = (0.90 * fy) / max(1.0, float(v - cy))
        if 0.4 <= z_val <= 2.5:
            depth[v, :] = z_val

    floor = fit_floor_plane(depth, intrinsics, cfg)
    assert floor is not None
    # In camera optical frame, floor normal pointing upward towards ceiling is [0, -1, 0]
    expected_normal = np.array([0.0, -1.0, 0.0])
    dot = np.dot(floor.normal, expected_normal)
    assert dot > 0.95

    # Compute gravity transform
    from realsense_measure.fullbody.alignment import compute_gravity_transform
    T_grav_cam = compute_gravity_transform(floor)

    # Any floor point transformed by T_grav_cam must have Y = 0
    p_floor_cam = floor.point_on_plane
    p_floor_grav = T_grav_cam[:3, :3] @ p_floor_cam + T_grav_cam[:3, 3]
    assert np.isclose(p_floor_grav[1], 0.0, atol=1e-4)


def test_cross_stop_floor_consistency():
    """Verify consistency checker passes when normals agree, and detects excessive tilt."""
    cfg = FullBodyConfig(floor_max_angle_diff_deg=4.0)

    n_up = np.array([0.0, -1.0, 0.0])
    p1 = FloorPlane(
        coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
        normal=n_up,
        point_on_plane=np.array([0.0, 0.90, 1.2]),
        inlier_count=500,
        inlier_ratio=0.8,
        rmse_m=0.005,
    )
    # Slight 1-degree tilt in stop 2
    tilt_rad = np.deg2rad(1.0)
    n2 = np.array([np.sin(tilt_rad), -np.cos(tilt_rad), 0.0])
    p2 = FloorPlane(
        coefficients=np.array([n2[0], n2[1], n2[2], 0.90]),
        normal=n2,
        point_on_plane=np.array([0.0, 0.90, 1.2]),
        inlier_count=500,
        inlier_ratio=0.8,
        rmse_m=0.005,
    )

    consistent, consensus = verify_floor_consistency([p1, p2], cfg)
    assert consistent is True
    assert consensus.normal[1] < -0.99

    # Excessive 10-degree tilt in stop 3
    large_tilt_rad = np.deg2rad(10.0)
    n3 = np.array([np.sin(large_tilt_rad), -np.cos(large_tilt_rad), 0.0])
    p3 = FloorPlane(
        coefficients=np.array([n3[0], n3[1], n3[2], 0.90]),
        normal=n3,
        point_on_plane=np.array([0.0, 0.90, 1.2]),
        inlier_count=500,
        inlier_ratio=0.8,
        rmse_m=0.005,
    )
    consistent_bad, _ = verify_floor_consistency([p1, p3], cfg)
    assert consistent_bad is False

