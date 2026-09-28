"""
Rich multi-modal representation of a captured RGB-D sensor frame.

Preserves synchronized color, depth, calibrated camera intrinsics,
colored 3D point cloud, and hardware/system timestamp.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

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
    """

    frame_id: int
    color_bgr: np.ndarray
    depth_m: np.ndarray
    pcd: o3d.geometry.PointCloud
    timestamp: float
    intrinsics: o3d.camera.PinholeCameraIntrinsic

    def to_rgbd_image(
        self,
        depth_scale: float = 1.0,
        depth_trunc: float = 1.5,
        convert_rgb_to_intensity: bool = False,
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

        Returns
        -------
        o3d.geometry.RGBDImage
        """
        color_rgb = np.ascontiguousarray(self.color_bgr[:, :, ::-1])
        depth_clipped = self.depth_m.copy()
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
        )
