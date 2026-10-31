"""
Rotation utilities for 90-degree portrait/landscape camera orientation.

Provides coordinate transformations for color/depth image arrays and
pinhole camera intrinsics. Guarantees that back-projected 3D points
before and after rotation are identical up to the 3D rotation matrix.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d


def rotate_intrinsics(
    intr: o3d.camera.PinholeCameraIntrinsic,
    mode: str,
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Rotate pinhole camera intrinsics for 90-degree CW or CCW image rotation.

    Parameters
    ----------
    intr:
        Original Open3D PinholeCameraIntrinsic.
    mode:
        'cw' for 90 degrees clockwise, 'ccw' for 90 degrees counter-clockwise.

    Returns
    -------
    o3d.camera.PinholeCameraIntrinsic:
        Rotated intrinsics with swapped width/height, swapped fx/fy, and remapped cx/cy.
    """
    mode = mode.lower()
    w = intr.width
    h = intr.height
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    new_w = h
    new_h = w
    new_fx = fy
    new_fy = fx

    if mode == "cw":
        new_cx = float((h - 1) - cy)
        new_cy = float(cx)
    elif mode == "ccw":
        new_cx = float(cy)
        new_cy = float((w - 1) - cx)
    else:
        raise ValueError(f"Invalid rotation mode: '{mode}'. Expected 'cw' or 'ccw'.")

    return o3d.camera.PinholeCameraIntrinsic(
        new_w, new_h, new_fx, new_fy, new_cx, new_cy
    )


def rotate_rgbd_arrays(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate color and depth arrays by 90 degrees CW or CCW.

    Parameters
    ----------
    color_bgr:
        HxWx3 BGR uint8 image.
    depth_m:
        HxW float32 depth map in metres.
    mode:
        'cw' or 'ccw'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        (rotated_color_bgr, rotated_depth_m)
    """
    mode = mode.lower()
    if mode == "cw":
        rot_code = cv2.ROTATE_90_CLOCKWISE
    elif mode == "ccw":
        rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        raise ValueError(f"Invalid rotation mode: '{mode}'. Expected 'cw' or 'ccw'.")

    rot_color = cv2.rotate(color_bgr, rot_code)
    rot_depth = cv2.rotate(depth_m, rot_code)
    return rot_color, rot_depth
