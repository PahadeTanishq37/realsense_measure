"""
Rich multi-modal representation of a captured RGB-D sensor frame.

Preserves synchronized color, depth, calibrated camera intrinsics,
colored 3D point cloud, and hardware/system timestamp.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import open3d as o3d


@dataclass
class RGBDFrame:
    """
    Synchronized RGB-D frame containing color, depth, point cloud, and intrinsics.

    Attributes
    ----------
    frame_id:
        Zero-based index of this frame in capture sequence.
    color_bgr:
        HxWx3 uint8 numpy array in OpenCV BGR format.
    depth_m:
        HxW float32 numpy array of depth values in metres (0.0 = invalid/truncated).
    pcd:
        Open3D PointCloud created from this frame, retaining RGB colors and normals.
    timestamp:
        Capture timestamp in seconds.
    intrinsics:
        Open3D PinholeCameraIntrinsic calibration parameters.
    quality_metrics:
        Optional dictionary of live quality diagnostics (ROI distance, valid fraction, face detected).
    """

    frame_id: int
    color_bgr: np.ndarray
    depth_m: np.ndarray
    pcd: o3d.geometry.PointCloud
    timestamp: float
    intrinsics: o3d.camera.PinholeCameraIntrinsic
    quality_metrics: dict[str, Any] | None = None
    foreground_mask: np.ndarray | None = None

    def to_rgbd_image(
        self,
        depth_scale: float = 1.0,
        depth_trunc: float = 1.5,
        convert_rgb_to_intensity: bool = False,
        depth_override: np.ndarray | None = None,
        color_override: np.ndarray | None = None,
    ) -> o3d.geometry.RGBDImage:
        """
        Convert to an Open3D RGBDImage for RGB-D odometry and volumetric integration.

        Parameters
        ----------
        depth_scale:
            Multiplier to convert depth array to metres (1.0 since depth_m is already in metres).
        depth_trunc:
            Truncation distance in metres beyond which depth values are clipped to zero.
        convert_rgb_to_intensity:
            Whether to convert RGB to single-channel luminance intensity.
            Typically True for RGBDOdometry, False for colored TSDF integration.
        depth_override:
            Optional pre-smoothed or masked depth array to use instead of self.depth_m.
        color_override:
            Optional color array to use instead of self.color_bgr (e.g. background set to constant).

        Returns
        -------
        o3d.geometry.RGBDImage
        """
        active_color = self.color_bgr if color_override is None else color_override
        color_rgb = np.ascontiguousarray(active_color[:, :, ::-1])
        depth_base = self.depth_m if depth_override is None else depth_override
        depth_clipped = depth_base.copy()
        depth_clipped[depth_clipped < 0.10] = 0.0
        depth_clipped[depth_clipped > depth_trunc] = 0.0

        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_clipped)

        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity,
        )

    def clone(self) -> "RGBDFrame":
        """Return a deep copy of this frame."""
        return RGBDFrame(
            frame_id=self.frame_id,
            color_bgr=self.color_bgr.copy(),
            depth_m=self.depth_m.copy(),
            pcd=copy.deepcopy(self.pcd),
            timestamp=self.timestamp,
            intrinsics=copy.deepcopy(self.intrinsics),
            quality_metrics=copy.deepcopy(self.quality_metrics) if self.quality_metrics is not None else None,
            foreground_mask=self.foreground_mask.copy() if self.foreground_mask is not None else None,
        )


    def to_point_cloud(
        self,
        depth_min_m: float = 0.40,
        depth_max_m: float = 1.20,
        roi: tuple[int, int, int, int] | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Recompute Open3D PointCloud from color and depth without requiring pyrealsense2.
        """
        return create_pcd_from_rgbd(
            color_bgr=self.color_bgr,
            depth_m=self.depth_m,
            intrinsics=self.intrinsics,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            roi=roi,
        )


def compute_burst_median_frame(
    burst_colors: list[np.ndarray],
    burst_depths: list[np.ndarray],
    min_valid_frac: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute burst-averaged RGB-D frame from N observations.

    Uses the middle colour frame and computes the per-pixel median of valid depth
    (ignoring zeros). Requires at least ceil(N * min_valid_frac) valid frames for a pixel
    to have valid depth in the output; otherwise 0.0. Outliers are rejected.
    Does not invent depth where the burst has none.

    Parameters
    ----------
    burst_colors:
        List of HxWx3 uint8 BGR images.
    burst_depths:
        List of HxW float32 metric depth maps.
    min_valid_frac:
        Minimum ratio of burst frames that must contain non-zero depth for that pixel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        (middle_color_bgr, median_depth_m)
    """
    if not burst_depths:
        raise ValueError("burst_depths list cannot be empty")
    n_frames = len(burst_depths)
    mid_idx = n_frames // 2
    middle_color = burst_colors[mid_idx].copy()

    depth_stack = np.stack(burst_depths, axis=0)  # (N, H, W)
    valid_mask = depth_stack > 0.0
    valid_counts = np.sum(valid_mask, axis=0)

    min_required = int(np.ceil(n_frames * min_valid_frac))

    nan_stack = np.where(valid_mask, depth_stack, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med_depth = np.nanmedian(nan_stack, axis=0)

    med_depth = np.nan_to_num(med_depth, nan=0.0)
    med_depth[valid_counts < min_required] = 0.0

    return middle_color, med_depth.astype(np.float32)


def create_pcd_from_rgbd(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_min_m: float = 0.40,
    depth_max_m: float = 1.20,
    roi: tuple[int, int, int, int] | None = None,
) -> o3d.geometry.PointCloud:
    """
    Create an Open3D PointCloud from color and depth arrays with camera intrinsics.

    Matches RealSenseCapture.to_point_cloud() exactly without requiring pyrealsense2.

    Parameters
    ----------
    color_bgr:
        HxWx3 uint8 BGR image.
    depth_m:
        HxW float32 depth map in metres.
    intrinsics:
        Open3D PinholeCameraIntrinsic.
    depth_min_m:
        Minimum depth distance to retain (metres).
    depth_max_m:
        Maximum depth distance to retain (metres).
    roi:
        Optional (x, y, w, h) pixel bounding box crop.

    Returns
    -------
    o3d.geometry.PointCloud
    """
    color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    depth_clipped = depth_m.copy()
    if roi is not None:
        rx, ry, rw, rh = roi
        mask = np.zeros_like(depth_clipped, dtype=bool)
        mask[max(0, ry) : max(0, ry + rh), max(0, rx) : max(0, rx + rw)] = True
        depth_clipped[~mask] = 0.0

    depth_clipped[depth_clipped < depth_min_m] = 0.0
    depth_clipped[depth_clipped > depth_max_m] = 0.0

    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(depth_clipped)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=1.0,
        depth_trunc=depth_max_m,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsics,
    )
    return pcd

