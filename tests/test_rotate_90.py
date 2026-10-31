"""
Unit tests for 90-degree CW and CCW camera rotation.

Verifies:
1. Swapping fx/fy and remapping cx/cy for CW and CCW rotations.
2. Back-projected 3D points before and after rotation are identical up to the rotation matrix.
3. Open3D PointCloud creation consistency.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pytest

from camera.frame import create_pcd_from_rgbd
from camera.rotation import rotate_intrinsics, rotate_rgbd_arrays


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_backprojected_3d_points_identical_up_to_rotation(mode: str) -> None:
    """
    Back-projected 3D points before and after rotation must be identical up to rotation.
    """
    W, H = 848, 480
    fx, fy = 615.0, 616.0
    cx, cy = 424.0, 240.0

    intr_orig = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Synthetic pattern with known features
    np.random.seed(123)
    color = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)

    # Create several non-trivial depth patches
    depth[80:360, 150:650] = 1.75
    depth[120:250, 220:450] = 1.20
    depth[300:400, 300:500] = 2.10

    # Rotate images and intrinsics
    color_rot, depth_rot = rotate_rgbd_arrays(color, depth, mode=mode)
    intr_rot = rotate_intrinsics(intr_orig, mode=mode)

    assert intr_rot.width == H
    assert intr_rot.height == W
    rot_fx, rot_fy = intr_rot.get_focal_length()
    rot_cx, rot_cy = intr_rot.get_principal_point()

    assert rot_fx == fy
    assert rot_fy == fx

    if mode == "cw":
        assert rot_cx == (H - 1) - cy
        assert rot_cy == cx
        # 3D rotation around +Z: (X, Y, Z) -> (-Y, X, Z)
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    else:  # ccw
        assert rot_cx == cy
        assert rot_cy == (W - 1) - cx
        # 3D rotation around +Z: (X, Y, Z) -> (Y, -X, Z)
        R = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    # 1. Direct back-projection verification
    ys_orig, xs_orig = np.where(depth > 0)
    zs_orig = depth[ys_orig, xs_orig]
    xs_3d_orig = (xs_orig - cx) * zs_orig / fx
    ys_3d_orig = (ys_orig - cy) * zs_orig / fy
    pts_orig = np.column_stack([xs_3d_orig, ys_3d_orig, zs_orig])

    ys_rot, xs_rot = np.where(depth_rot > 0)
    zs_rot = depth_rot[ys_rot, xs_rot]
    xs_3d_rot = (xs_rot - rot_cx) * zs_rot / rot_fx
    ys_3d_rot = (ys_rot - rot_cy) * zs_rot / rot_fy
    pts_rot = np.column_stack([xs_3d_rot, ys_3d_rot, zs_rot])

    # Transform original 3D points by R
    pts_orig_rotated = (R @ pts_orig.T).T

    # Both point sets should contain the exact same 3D coordinates
    order_orig = np.lexsort((pts_orig_rotated[:, 2], pts_orig_rotated[:, 1], pts_orig_rotated[:, 0]))
    order_rot = np.lexsort((pts_rot[:, 2], pts_rot[:, 1], pts_rot[:, 0]))

    sorted_orig_rot = pts_orig_rotated[order_orig]
    sorted_rot = pts_rot[order_rot]

    assert sorted_orig_rot.shape == sorted_rot.shape
    max_diff = np.max(np.abs(sorted_orig_rot - sorted_rot))
    assert max_diff < 1e-5, f"Max coordinate mismatch: {max_diff}"

    # 2. Open3D PointCloud creation verification
    pcd_orig = create_pcd_from_rgbd(color, depth, intr_orig, depth_min_m=0.4, depth_max_m=3.0)
    pcd_rot = create_pcd_from_rgbd(color_rot, depth_rot, intr_rot, depth_min_m=0.4, depth_max_m=3.0)

    pts_pcd_orig = np.asarray(pcd_orig.points)
    pts_pcd_rot = np.asarray(pcd_rot.points)

    assert len(pts_pcd_orig) == len(pts_pcd_rot)
    pts_pcd_orig_rotated = (R @ pts_pcd_orig.T).T

    order_pcd_orig = np.lexsort((pts_pcd_orig_rotated[:, 2], pts_pcd_orig_rotated[:, 1], pts_pcd_orig_rotated[:, 0]))
    order_pcd_rot = np.lexsort((pts_pcd_rot[:, 2], pts_pcd_rot[:, 1], pts_pcd_rot[:, 0]))

    diff_pcd = np.max(np.abs(pts_pcd_orig_rotated[order_pcd_orig] - pts_pcd_rot[order_pcd_rot]))
    assert diff_pcd < 1e-4, f"Open3D PCD coordinate mismatch: {diff_pcd}"
