"""
BodyTarget: measurement target for human body scans.

Registered under the name "body". See targets/base.py for the extension contract.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from realsense_measure.targets.base import Target, register_target


@register_target
class BodyTarget(Target):
    """
    Measurement target for human body scans.
    """

    name: str = "body"

    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Segment or refine the body point cloud.
        """
        return pcd

    def measure(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        Compute measurements and bounding extents for body scan.
        """
        if len(pcd.points) == 0:
            return {"target": "body", "status": "empty"}

        pts = np.asarray(pcd.points)
        min_b = pts.min(axis=0)
        max_b = pts.max(axis=0)
        extent = max_b - min_b

        return {
            "target": "body",
            "point_count": len(pts),
            "bounding_box_extent_m": {
                "width_m": float(extent[0]),
                "height_m": float(extent[1]),
                "depth_m": float(extent[2]),
            },
        }
