"""
tests/camera/test_rotation_geometry.py

Unit test proving numerical correctness of rotate_intrinsics() and rotate_rgbd_arrays()
for both 90-degree clockwise (cw) and counter-clockwise (ccw) camera rotation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from realsense_measure.camera.rotation import rotate_intrinsics, rotate_rgbd_arrays


class MockPinholeCameraIntrinsic:
    """Mock for open3d.camera.PinholeCameraIntrinsic compatible with rotation.py."""

    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.intrinsic_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def get_focal_length(self) -> tuple[float, float]:
        return float(self.intrinsic_matrix[0, 0]), float(self.intrinsic_matrix[1, 1])

    def get_principal_point(self) -> tuple[float, float]:
        return float(self.intrinsic_matrix[0, 2]), float(self.intrinsic_matrix[1, 2])


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_rotation_geometry_numerical_exactness(mode: str) -> None:
    """
    Prove numerically for rotate_90 = 'cw' and 'ccw':
    1. Synthetic pinhole camera: 848x480, fx=fy=425, cx=424, cy=240.
    2. Known 3D camera-frame points (center, quadrants, near edges, near corners).
    3. Project to pixels with original intrinsics.
    4. Rotate pixel coordinates using exact mapping derived from rotate_rgbd_arrays.
    5. Back-project rotated pixels with intrinsics from rotate_intrinsics().
    6. Assert back-projected 3D point matches original 3D point rotated about optical axis
       by exactly +90 deg (cw) or -90 deg (ccw) within 1e-6 m.
    7. Assert width/height swapped, (fx, fy) and (cx, cy) exchanged/mirrored with pixel-center convention.
    """
    W, H = 848, 480
    fx, fy = 425.0, 425.0
    cx, cy = 424.0, 240.0

    intr_orig = MockPinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # 1. Known 3D camera-frame points
    points_3d = [
        (0.0, 0.0, 1.0),            # Optical center
        (0.2, 0.1, 1.5),            # Off-center quadrant 1 (+X, +Y)
        (-0.25, 0.15, 1.2),         # Off-center quadrant 2 (-X, +Y)
        (-0.2, -0.15, 1.8),         # Off-center quadrant 3 (-X, -Y)
        (0.18, -0.12, 1.4),         # Off-center quadrant 4 (+X, -Y)
        (0.0, -0.5, 1.0),           # Near top edge (v ≈ 27.5)
        (0.0, 0.5, 1.0),            # Near bottom edge (v ≈ 452.5)
        (-0.95, 0.0, 1.0),          # Near left edge (u ≈ 20.25)
        (0.95, 0.0, 1.0),           # Near right edge (u ≈ 827.75)
        (0.9, 0.5, 1.0),            # Near bottom-right corner
        (-0.9, -0.5, 1.0),          # Near top-left corner
    ]

    # Rotate intrinsics via module function
    intr_rot = rotate_intrinsics(intr_orig, mode=mode)

    # 5. Assert width/height are swapped, fx/fy exchanged, cx/cy remapped
    assert intr_rot.width == H
    assert intr_rot.height == W
    rot_fx, rot_fy = intr_rot.get_focal_length()
    rot_cx, rot_cy = intr_rot.get_principal_point()

    assert rot_fx == fy
    assert rot_fy == fx

    # Sign convention of rotation about camera optical axis (+Z into screen):
    # Camera coordinates: +X right, +Y down, +Z forward.
    # When camera is rotated CW by 90 deg:
    # A vector originally pointing along +X (right) now points along +Y (down): (1, 0, 0) -> (0, 1, 0).
    # A vector originally pointing along +Y (down) now points along -X (left): (0, 1, 0) -> (-1, 0, 0).
    # This corresponds to rotation about +Z by +90 deg:
    # R_cw = [[0, -1, 0], [1, 0, 0], [0, 0, 1]], (X, Y, Z) -> (-Y, X, Z).
    # When camera is rotated CCW by 90 deg:
    # R_ccw = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], (X, Y, Z) -> (Y, -X, Z).
    if mode == "cw":
        assert rot_cx == (H - 1) - cy
        assert rot_cy == cx
        R_expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    else:  # ccw
        assert rot_cx == cy
        assert rot_cy == (W - 1) - cx
        R_expected = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    # Derive exact pixel coordinate mapping from rotate_rgbd_arrays using unique index grid
    index_img = np.arange(H * W, dtype=np.int32).reshape((H, W))
    dummy_color = np.zeros((H, W, 3), dtype=np.uint8)
    _, rot_depth_indices = rotate_rgbd_arrays(dummy_color, index_img.astype(np.float32), mode=mode)
    rot_indices = np.round(rot_depth_indices).astype(np.int32)

    for X, Y, Z in points_3d:
        # 2. Project to pixel coordinates with original intrinsics
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        assert 0 <= u < W
        assert 0 <= v < H

        # Derive mapping for the integer pixel center from real rotate_rgbd_arrays
        u_int = int(round(u))
        v_int = int(round(v))
        pixel_val = v_int * W + u_int
        y_locs, x_locs = np.where(rot_indices == pixel_val)
        assert len(x_locs) == 1
        rot_u_int = int(x_locs[0])
        rot_v_int = int(y_locs[0])

        # Exact pixel mapping including subpixel offsets:
        # For CW: (u, v) -> ((H - 1) - v, u)
        # For CCW: (u, v) -> (v, (W - 1) - u)
        if mode == "cw":
            assert rot_u_int == (H - 1) - v_int
            assert rot_v_int == u_int
            rot_u = (H - 1.0) - v
            rot_v = u
        else:
            assert rot_u_int == v_int
            assert rot_v_int == (W - 1) - u_int
            rot_u = v
            rot_v = (W - 1.0) - u

        # 3. Back-project rotated pixel using rotated intrinsics and carried depth
        X_rot = (rot_u - rot_cx) * Z / rot_fx
        Y_rot = (rot_v - rot_cy) * Z / rot_fy
        Z_rot = Z

        pt_backprojected = np.array([X_rot, Y_rot, Z_rot], dtype=np.float64)
        pt_expected = R_expected @ np.array([X, Y, Z], dtype=np.float64)

        # 4. Assert result equals rotated point within 1e-6 m
        diff = np.linalg.norm(pt_backprojected - pt_expected)
        assert diff < 1e-6, f"Point {X, Y, Z} back-projection error {diff} >= 1e-6 m in mode {mode}"

