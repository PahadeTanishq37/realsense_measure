"""
HeadTarget: measure human head / cranial dimensions from a fused reconstruction.

Registered under the name "head".
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from config import PreprocessConfig
from preprocessing import extract_largest_cluster
from targets.base import Target, register_target


@register_target
class HeadTarget(Target):
    """
    Segment and measure human head / face geometry.

    Measures cranial dimensions:
      - Cranial breadth (width, ear-to-ear / biparietal)
      - Head length (depth, glabella to back of head)
      - Facial / cranial height (chin / vertex)
      - Minimum bounding box and volume
    """

    name: str = "head"

    def __init__(self, preprocess_cfg: PreprocessConfig | None = None) -> None:
        self.cfg: PreprocessConfig = preprocess_cfg or PreprocessConfig()

    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Isolate head and facial geometry, removing disconnected neck/torso outliers.
        """
        if len(pcd.points) < 50:
            return pcd

        # Retain the largest connected head component
        cleaned = extract_largest_cluster(pcd, self.cfg)
        return cleaned

    def measure(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        Fit bounding boxes and calculate head anatomical dimensions.
        """
        if len(pcd.points) < 50:
            return {
                "target": self.name,
                "error": "Insufficient points in segmented head cloud",
                "point_count": len(pcd.points),
            }

        # Oriented bounding box
        obb = pcd.get_minimal_oriented_bounding_box() if hasattr(pcd, "get_minimal_oriented_bounding_box") else pcd.get_oriented_bounding_box()
        extents = sorted(list(obb.extent))  # sorted: [height, depth, width] or similar

        # Axis-aligned bounding box
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [float(v) for v in aabb.get_extent()]

        # Typical human head: ~15-20 cm width, ~18-22 cm length, ~20-25 cm height
        dim_x, dim_y, dim_z = aabb_extent

        result = {
            "target": self.name,
            "point_count": len(pcd.points),
            "head_dimensions_mm": {
                "cranial_width_mm": float(dim_x * 1000.0),
                "cranial_depth_mm": float(dim_z * 1000.0),
                "head_height_mm": float(dim_y * 1000.0),
            },
            "oriented_bbox": {
                "length_m": float(extents[2]),
                "width_m": float(extents[1]),
                "height_m": float(extents[0]),
                "volume_m3": float(obb.volume()),
            },
            "axis_aligned_bbox_extent_m": aabb_extent,
            "geometry_for_viz": obb,
        }
        return result
