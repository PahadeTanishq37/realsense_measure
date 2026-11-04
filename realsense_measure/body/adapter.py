"""
realsense_measure/body/adapter.py
=================================
RGB-D-to-Anny Stop Adapter.

Converts an incoming sensor frame (live RGBDFrame or replayed capture)
into the exact SimulatedStopData representation required by DifferentiableBodyFitter,
BodyUnposer, and PointToPlaneICPRefiner.

Preserves full sensor provenance via IndexedPointCloud while translating
3D point clouds into the Anny world / tripod coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame
from camera.indexed_pcd import IndexedPointCloud, create_indexed_pcd_from_rgbd
from .synth import SimulatedStopData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authoritative Coordinate Transformations
# ---------------------------------------------------------------------------
#
# RealSense Camera Optical Frame:
#   +X: Right
#   +Y: Down
#   +Z: Forward (optical depth into scene)
#
# Anny World / Tripod Frame:
#   +X: Anatomical Lateral (Subject's Left)
#   +Y: Forward (Anterior, pointing from camera towards subject)
#   +Z: Up (Superior, vertical stature axis)
#
# Fixed Tripod Convention:
#   Camera optical center is positioned at world coordinate (0.0, -distance_m, 0.0)
#   facing in the +Y direction towards the subject centered at (0.0, 0.0, 0.0).
#
# Direct Coordinate Relations:
#   X_w = X_c
#   Y_w = Z_c - distance_m
#   Z_w = -Y_c
#
# Inverse Relations:
#   X_c = X_w
#   Y_c = -Z_w
#   Z_c = Y_w + distance_m

R_CAM_TO_WORLD = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0, -1.0,  0.0],
], dtype=np.float32)

R_WORLD_TO_CAM = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float32)


def camera_to_world_points(
    points_cam: np.ndarray,
    distance_m: float,
    camera_height_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """
    Transform 3D point cloud from RealSense camera optical frame to Anny world frame.

    Parameters
    ----------
    points_cam : np.ndarray
        (N, 3) array of points in metres [X_c, Y_c, Z_c].
    distance_m : float
        Tripod distance along world Y in metres (> 0).
    camera_height_m : float, optional
        Camera optical center height along world Z in metres (default 0.0).
    pitch_deg : float, optional
        Downward camera pitch tilt in degrees (default 0.0).

    Returns
    -------
    np.ndarray
        (N, 3) array of points in world coordinates [X_w, Y_w, Z_w].
    """
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"Invalid distance_m: {distance_m}. Must be a finite positive number >= 0.10 m.")

    pts = np.asarray(points_cam, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_cam must have shape (N, 3), got {pts.shape}")

    if pitch_deg != 0.0 or camera_height_m != 0.0:
        T = camera_to_world_matrix(distance_m, camera_height_m, pitch_deg)
        R = T[:3, :3]
        t = T[:3, 3]
        return (pts @ R.T) + t

    cam_pos = np.array([0.0, -float(distance_m), 0.0], dtype=np.float32)
    return np.stack([
        pts[:, 0] + cam_pos[0],
        pts[:, 2] + cam_pos[1],
        -pts[:, 1] + cam_pos[2],
    ], axis=-1)


def world_to_camera_points(
    points_world: np.ndarray,
    distance_m: float,
    camera_height_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """
    Transform 3D point cloud from Anny world frame back to RealSense camera optical frame.

    Parameters
    ----------
    points_world : np.ndarray
        (N, 3) array of points in metres [X_w, Y_w, Z_w].
    distance_m : float
        Tripod distance along world Y in metres (> 0).
    camera_height_m : float, optional
        Camera optical center height along world Z in metres (default 0.0).
    pitch_deg : float, optional
        Downward camera pitch tilt in degrees (default 0.0).

    Returns
    -------
    np.ndarray
        (N, 3) array of points in camera coordinates [X_c, Y_c, Z_c].
    """
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"Invalid distance_m: {distance_m}. Must be a finite positive number >= 0.10 m.")

    pts = np.asarray(points_world, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3), got {pts.shape}")

    if pitch_deg != 0.0 or camera_height_m != 0.0:
        T_inv = world_to_camera_matrix(distance_m, camera_height_m, pitch_deg)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3]
        return (pts @ R_inv.T) + t_inv

    cam_pos = np.array([0.0, -float(distance_m), 0.0], dtype=np.float32)
    return np.stack([
        pts[:, 0] - cam_pos[0],
        -(pts[:, 2] - cam_pos[2]),
        pts[:, 1] - cam_pos[1],
    ], axis=-1)


def camera_to_world_matrix(
    distance_m: float,
    camera_height_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """
    Return 4x4 homogeneous transformation matrix from camera to world coordinates.
    """
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"Invalid distance_m: {distance_m}. Must be a finite positive number >= 0.10 m.")

    if pitch_deg != 0.0:
        from .hardware_smoke_test import CoordinateGeometryAuditor
        return CoordinateGeometryAuditor.camera_to_world_matrix(
            distance_m=distance_m,
            camera_height_m=camera_height_m,
            pitch_deg=pitch_deg,
        )

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_CAM_TO_WORLD
    T[:3, 3] = np.array([0.0, -float(distance_m), float(camera_height_m)], dtype=np.float32)
    return T


def world_to_camera_matrix(
    distance_m: float,
    camera_height_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """
    Return 4x4 homogeneous transformation matrix from world to camera coordinates.
    """
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"Invalid distance_m: {distance_m}. Must be a finite positive number >= 0.10 m.")

    if pitch_deg != 0.0 or camera_height_m != 0.0:
        T_c2w = camera_to_world_matrix(distance_m, camera_height_m, pitch_deg)
        return np.linalg.inv(T_c2w).astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_WORLD_TO_CAM
    T[:3, 3] = -R_WORLD_TO_CAM @ np.array([0.0, -float(distance_m), 0.0], dtype=np.float32)
    return T


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class AdaptedStopData(SimulatedStopData):
    """
    Subclasses SimulatedStopData to guarantee complete backward-compatibility with
    DifferentiableBodyFitter, while preserving sensor-level provenance and acquisition metadata.

    Attributes
    ----------
    indexed_pcd : Optional[IndexedPointCloud]
        Provenance-preserving point cloud containing original (u, v) pixel mappings,
        source frame ID, observation weights, and validity flags.
    frame_id : int
        Original capture frame sequence index.
    timestamp : float
        Hardware / system capture timestamp in seconds.
    color_bgr : Optional[np.ndarray]
        Original BGR color image if available.
    """
    indexed_pcd: Optional[IndexedPointCloud] = None
    frame_id: int = 0
    timestamp: float = 0.0
    color_bgr: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_intrinsics_dict(
    intrinsics: Any,
    distance_m: float,
    depth_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Extract and strictly validate pinhole camera intrinsics dictionary.

    Parameters
    ----------
    intrinsics : Any
        Open3D PinholeCameraIntrinsic, 3x3 array, or dictionary.
    distance_m : float
        Camera distance in metres (> 0.10).
    depth_shape : Optional[Tuple[int, int]]
        Optional (H, W) to cross-validate image dimensions.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: fx, fy, cx, cy, width, height, distance_m.
    """
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"distance_m must be a finite float >= 0.10 m, got {distance_m}")

    if intrinsics is None:
        raise ValueError("Camera intrinsics cannot be None.")

    if isinstance(intrinsics, o3d.camera.PinholeCameraIntrinsic):
        mat = np.array(intrinsics.intrinsic_matrix, dtype=np.float64)
        w = int(intrinsics.width)
        h = int(intrinsics.height)
    elif isinstance(intrinsics, np.ndarray):
        mat = np.asarray(intrinsics, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Intrinsics matrix must be 3x3, got {mat.shape}")
        if depth_shape is None:
            raise ValueError("depth_shape is required when intrinsics is a raw 3x3 matrix.")
        h, w = int(depth_shape[0]), int(depth_shape[1])
    elif isinstance(intrinsics, dict):
        req_keys = ["fx", "fy", "cx", "cy"]
        for k in req_keys:
            if k not in intrinsics:
                raise ValueError(f"Missing required intrinsic key '{k}' in dictionary.")
        fx_val = float(intrinsics["fx"])
        fy_val = float(intrinsics["fy"])
        cx_val = float(intrinsics["cx"])
        cy_val = float(intrinsics["cy"])
        w_val = int(intrinsics.get("width", depth_shape[1] if depth_shape else 0))
        h_val = int(intrinsics.get("height", depth_shape[0] if depth_shape else 0))
        mat = np.array([[fx_val, 0.0, cx_val], [0.0, fy_val, cy_val], [0.0, 0.0, 1.0]], dtype=np.float64)
        w, h = w_val, h_val
    else:
        raise TypeError(f"Unsupported intrinsics type: {type(intrinsics)}")

    fx = float(mat[0, 0])
    fy = float(mat[1, 1])
    cx = float(mat[0, 2])
    cy = float(mat[1, 2])

    if not (np.isfinite(fx) and fx > 0.0 and np.isfinite(fy) and fy > 0.0):
        raise ValueError(f"Invalid focal lengths: fx={fx}, fy={fy}. Must be positive finite numbers.")

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: width={w}, height={h}. Must be positive.")

    if not (0.0 <= cx <= float(w) and 0.0 <= cy <= float(h)):
        raise ValueError(f"Principal point ({cx}, {cy}) is outside image bounds [0, {w}]x[0, {h}].")

    if depth_shape is not None:
        dh, dw = depth_shape
        if (dh, dw) != (h, w):
            raise ValueError(f"Intrinsics dimensions ({w}x{h}) do not match depth image shape ({dw}x{dh}).")

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(w),
        "height": int(h),
        "distance_m": float(distance_m),
    }


def get_or_create_foreground_mask(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    foreground_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resolve or compute person foreground segmentation mask.

    Parameters
    ----------
    color_bgr : np.ndarray
        (H, W, 3) BGR image array.
    depth_m : np.ndarray
        (H, W) float32 depth map in metres.
    foreground_mask : Optional[np.ndarray]
        Explicit mask if already supplied.

    Returns
    -------
    np.ndarray
        (H, W) boolean foreground mask.
    """
    h, w = depth_m.shape[:2]

    if foreground_mask is not None:
        mask = np.asarray(foreground_mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"Supplied foreground_mask shape {mask.shape} does not match depth shape {(h, w)}.")
        return mask

    # Try MediaPipe person segmentation if available
    try:
        from segmentation import get_person_mask
        seg_mask = get_person_mask(color_bgr)
        if seg_mask is not None and np.sum(seg_mask) >= 50:
            valid_p = seg_mask & (depth_m > 0.10) & (depth_m <= 6.0) & np.isfinite(depth_m)
            if np.any(valid_p):
                d_med = float(np.median(depth_m[valid_p]))
                depth_gate = (depth_m >= max(0.20, d_med - 0.70)) & (depth_m <= min(6.0, d_med + 0.70))
                gated = seg_mask & depth_gate
                if np.sum(gated) >= 50:
                    return gated.astype(bool)
            return seg_mask.astype(bool)
    except Exception as exc:
        logger.debug(f"MediaPipe segmentation skipped or failed: {exc}")

    # Fallback: depth range connected component in typical subject capture zone [0.40, 3.20] m
    valid_d = (depth_m >= 0.40) & (depth_m <= 3.20) & np.isfinite(depth_m)
    if np.any(valid_d):
        d_med = float(np.median(depth_m[valid_d]))
        slab = (depth_m >= max(0.40, d_med - 0.50)) & (depth_m <= d_med + 0.50) & np.isfinite(depth_m)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(slab.astype(np.uint8), connectivity=8)
        if n_lbl > 1:
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[largest, cv2.CC_STAT_AREA] >= 100:
                return (lbls == largest)

    # Return valid depth pixels as final fallback
    return (depth_m > 0.10) & np.isfinite(depth_m)


# ---------------------------------------------------------------------------
# Primary Adapter
# ---------------------------------------------------------------------------

def frame_to_body_stop(
    frame: RGBDFrame,
    nominal_yaw_deg: float,
    distance_m: float,
    foreground_mask: Optional[np.ndarray] = None,
    stop_idx: int = 0,
    depth_min_m: float = 0.10,
    depth_max_m: float = 3.50,
    raise_on_empty: bool = True,
    true_yaw_deg: Optional[float] = None,
    true_translation_m: Optional[np.ndarray] = None,
    true_arm_abduction_deg: Optional[Tuple[float, float]] = None,
    camera_height_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> AdaptedStopData:
    """
    Convert an RGBDFrame into the SimulatedStopData structure required by DifferentiableBodyFitter.

    Parameters
    ----------
    frame : RGBDFrame
        Captured or replayed sensor frame with synchronized color, depth, and intrinsics.
    nominal_yaw_deg : float
        Nominal turntable / subject yaw angle in degrees (e.g. 0.0, 90.0, 180.0, 270.0).
    distance_m : float
        Calibrated tripod distance from subject rotation center in metres.
    foreground_mask : Optional[np.ndarray]
        Explicit (H, W) boolean foreground mask. If None, uses frame.foreground_mask or segmentation.
    stop_idx : int
        Sequence index for this stop (default 0).
    depth_min_m : float
        Minimum valid depth clipping distance in metres (default 0.10 m).
    depth_max_m : float
        Maximum valid depth clipping distance in metres (default 3.50 m).
    raise_on_empty : bool
        If True (default), raises ValueError if the mask or point cloud is empty.
        If False, returns a valid AdaptedStopData with empty point clouds.
    true_yaw_deg : Optional[float]
        Ground-truth yaw in degrees if known (synthetic benchmarks).
        Defaults to NaN for real/replayed captures to avoid contaminating evaluation.
    true_translation_m : Optional[np.ndarray]
        Ground-truth translation in metres if known (synthetic benchmarks).
        Defaults to NaN for real/replayed captures.
    true_arm_abduction_deg : Optional[Tuple[float, float]]
        Ground-truth arm abduction angles if known (synthetic benchmarks).
        Defaults to (NaN, NaN) for real/replayed captures.

    Returns
    -------
    AdaptedStopData
        Subclass of SimulatedStopData ready for body fitting, pose refinement, and unposing.
    """
    # 1. Validate Input Frame
    if frame is None:
        raise ValueError("frame cannot be None.")

    color_bgr = getattr(frame, "color_bgr", None)
    depth_m = getattr(frame, "depth_m", None)
    intrinsics = getattr(frame, "intrinsics", None)

    if color_bgr is None or not isinstance(color_bgr, np.ndarray):
        raise ValueError("frame.color_bgr must be a valid numpy array.")
    if depth_m is None or not isinstance(depth_m, np.ndarray):
        raise ValueError("frame.depth_m must be a valid numpy array.")

    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError(f"frame.color_bgr must have shape (H, W, 3), got {color_bgr.shape}")
    if depth_m.ndim != 2:
        raise ValueError(f"frame.depth_m must have shape (H, W), got {depth_m.shape}")

    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError(
            f"Color shape {color_bgr.shape[:2]} does not match depth shape {depth_m.shape}."
        )

    # 2. Validate Nominal Yaw and Distance Parameters
    if not np.isfinite(nominal_yaw_deg):
        raise ValueError(f"nominal_yaw_deg must be a finite float, got {nominal_yaw_deg}")
    if not np.isfinite(distance_m) or distance_m <= 0.10:
        raise ValueError(f"distance_m must be a positive finite float >= 0.10 m, got {distance_m}")

    # 3. Extract and Validate Intrinsics
    intr_dict = extract_intrinsics_dict(intrinsics, distance_m=distance_m, depth_shape=depth_m.shape)
    intr_dict["camera_height_m"] = float(camera_height_m)
    intr_dict["pitch_deg"] = float(pitch_deg)

    # 4. Resolve Foreground Mask
    candidate_mask = foreground_mask if foreground_mask is not None else getattr(frame, "foreground_mask", None)
    effective_mask = get_or_create_foreground_mask(color_bgr, depth_m, candidate_mask)

    if np.sum(effective_mask) == 0:
        if raise_on_empty:
            raise ValueError("Foreground mask is empty: no foreground subject pixels detected.")
        logger.warning("Foreground mask is empty for frame %s.", getattr(frame, "frame_id", stop_idx))

    # 5. Back-project to IndexedPointCloud (Preserving Sensor Provenance)
    frame_id = int(getattr(frame, "frame_id", stop_idx))
    timestamp = float(getattr(frame, "timestamp", 0.0))

    indexed_pcd = create_indexed_pcd_from_rgbd(
        color=color_bgr,
        depth_m=depth_m,
        intrinsics=intrinsics,
        depth_scale=1.0,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        mask=effective_mask,
        frame_id=frame_id,
        timestamp=timestamp,
        is_bgr=True,
    )

    pcd_cam = np.asarray(indexed_pcd.points, dtype=np.float32)

    # Ensure no NaN or Inf in coordinates
    if len(pcd_cam) > 0 and not np.all(np.isfinite(pcd_cam)):
        raise ValueError("Back-projected point cloud contains non-finite coordinates.")

    if len(pcd_cam) == 0:
        if raise_on_empty:
            raise ValueError(
                "No valid 3D points found within depth bounds and foreground mask."
            )
        pcd_world = np.zeros((0, 3), dtype=np.float32)
    else:
        # 6. Apply Authoritative Camera-to-World Coordinate Conversion
        pcd_world = camera_to_world_points(
            pcd_cam,
            distance_m=distance_m,
            camera_height_m=camera_height_m,
            pitch_deg=pitch_deg,
        )

    # 7. Construct and Return AdaptedStopData
    # Mask depth image with effective foreground mask to match SimulatedStopData contract (background = 0.0)
    depth_image_f32 = np.where(effective_mask, np.asarray(depth_m, dtype=np.float32), 0.0).astype(np.float32)

    # Ground-truth semantics: real captures have unknown pose
    t_yaw = float(true_yaw_deg) if true_yaw_deg is not None else float("nan")
    t_trans = (
        np.asarray(true_translation_m, dtype=np.float32)
        if true_translation_m is not None
        else np.full(3, np.nan, dtype=np.float32)
    )
    t_abd = (
        (float(true_arm_abduction_deg[0]), float(true_arm_abduction_deg[1]))
        if true_arm_abduction_deg is not None
        else (float("nan"), float("nan"))
    )

    return AdaptedStopData(
        stop_idx=int(stop_idx),
        nominal_yaw_deg=float(nominal_yaw_deg),
        true_yaw_deg=t_yaw,
        true_translation_m=t_trans,
        true_arm_abduction_deg=t_abd,
        depth_image=depth_image_f32,
        foreground_mask=effective_mask,
        point_cloud=pcd_cam,
        point_cloud_world=pcd_world,
        intrinsics=intr_dict,
        indexed_pcd=indexed_pcd,
        frame_id=frame_id,
        timestamp=timestamp,
        color_bgr=color_bgr,
    )
