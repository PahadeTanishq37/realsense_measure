"""
BoxTarget: measure a rigid, roughly box-shaped object.

Registered under the name "box".  See targets/base.py for the extension
contract — adding a new target type (head, body) follows the same pattern.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from config import PreprocessConfig
from preprocessing import extract_largest_cluster, remove_dominant_plane
from targets.base import Target, register_target


@register_target
class BoxTarget(Target):
    """
    Segment and measure a rigid box-shaped object from a fused point cloud.

    The measurement pipeline is:
      1. :meth:`segment` — strip any residual planar background, keep the
         single largest connected component.
      2. :meth:`measure` — fit a minimum-volume oriented bounding box and
         return trimmed (bias-corrected) axis lengths, volume, and OBB
         geometry for the visualizer.
    """

    name: str = "box"

    def __init__(self, preprocess_cfg: PreprocessConfig | None = None) -> None:
        """
        Parameters
        ----------
        preprocess_cfg:
            Preprocessing settings shared with the rest of the pipeline.
            If ``None``, sensible defaults are used (``PreprocessConfig()``).
        """
        self.cfg: PreprocessConfig = preprocess_cfg or PreprocessConfig()

    # ------------------------------------------------------------------
    # Target interface
    # ------------------------------------------------------------------

    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove residual planar background and return the dominant cluster.

        Even though the fused cloud has already been processed per-frame,
        small amounts of table/wall geometry can be dragged back in during
        registration (especially near the edges of the reference cloud).
        This extra plane-removal pass mops those up before measurement.

        Parameters
        ----------
        pcd:
            Fused, mostly-clean object cloud.

        Returns
        -------
        o3d.geometry.PointCloud
            Box points only.  Falls back to the input cloud if plane removal
            produces an empty result.
        """
        no_plane, _ = remove_dominant_plane(pcd, self.cfg)

        if not no_plane.has_points():
            # Plane removal consumed everything — return the original cloud
            # rather than propagating an empty cloud to measure().
            return pcd

        return extract_largest_cluster(no_plane, self.cfg)

    def measure(
        self,
        pcd: o3d.geometry.PointCloud,
        trim_percentile: float = 0.5,
    ) -> dict:
        """
        Fit an oriented bounding box and return trimmed axis measurements.

        Parameters
        ----------
        pcd:
            Segmented box cloud, as returned by :meth:`segment`.
        trim_percentile:
            Percentile (0–50) used for outlier-trimming when computing
            dimensions.  Default 0.5 trims the outermost 0.5 % of points
            on each side of each local axis.

        Returns
        -------
        dict
            JSON-serialisable measurement dict with one extra key
            ``"geometry_for_viz"`` holding the fitted OBB for the visualizer.

        Raises
        ------
        ValueError
            If the cloud has fewer than 10 points after segmentation.
        """
        if len(pcd.points) < 10:
            raise ValueError(
                "Not enough points survived segmentation to measure the box "
                f"(got {len(pcd.points)}, need at least 10)."
            )

        # ------------------------------------------------------------------
        # Fit a minimum-volume oriented bounding box.
        # get_minimal_oriented_bounding_box was added in Open3D ≥ 0.18;
        # fall back to get_oriented_bounding_box for older installs.
        # ------------------------------------------------------------------
        try:
            obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        except AttributeError:
            obb = pcd.get_oriented_bounding_box(robust=True)

        # ------------------------------------------------------------------
        # CRITICAL — bias-corrected dimension estimation.
        #
        # Using obb.extent directly gives the min/max span of the cloud in
        # each local axis, which is measurably biased HIGH on real sensor
        # data.  Even a handful of noisy points sitting at the cloud's
        # extremities can inflate every axis by several millimetres, and the
        # bias grows with point density.  This is a systematic over-estimate,
        # not random noise.
        #
        # Fix: project all points into the OBB's local frame, then use
        # trimmed percentiles (default: 0.5 % each tail) instead of the
        # absolute min/max.  This robustly clips outlier points that push
        # the boundary outward without pulling in the bulk of real surface
        # points — giving unbiased, repeatable dimension estimates.
        # ------------------------------------------------------------------
        pts = np.asarray(pcd.points)
        R = np.asarray(obb.R)           # 3×3 rotation matrix (OBB axes as columns)
        center = np.asarray(obb.center)

        # Translate to OBB centre, then rotate into the OBB's local frame.
        local_pts = (pts - center) @ R

        lo = np.percentile(local_pts, trim_percentile, axis=0)
        hi = np.percentile(local_pts, 100.0 - trim_percentile, axis=0)
        trimmed_extent = hi - lo        # trimmed span along each local axis

        # Sort longest → shortest so index 0=length, 1=width, 2=height
        # regardless of which physical axis the OBB happened to assign each.
        dims = np.sort(trimmed_extent)[::-1]
        length, width, height = float(dims[0]), float(dims[1]), float(dims[2])

        # Axis-aligned bounding box for reference / sanity checking.
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [
            round(float(v), 4) for v in np.asarray(aabb.get_extent())
        ]

        return {
            "target": "box",
            "num_points": len(pcd.points),
            "oriented_bbox": {
                "length_m": round(length, 4),
                "width_m": round(width, 4),
                "height_m": round(height, 4),
                "volume_m3": round(length * width * height, 6),
                "center": center.tolist(),
                "rotation": R.tolist(),
            },
            "axis_aligned_bbox_extent_m": aabb_extent,
            # Original (untrimmed) OBB geometry — used by the visualizer to
            # draw the fitted box on top of the point cloud.
            "geometry_for_viz": obb,
        }
