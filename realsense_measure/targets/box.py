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


def _fit_plane_rms_mm(pts: np.ndarray) -> float:
    """
    Fit a plane to 3D points using SVD and compute the RMS perpendicular distance in mm.
    """
    if len(pts) < 3:
        return 0.0
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # normal corresponds to smallest singular value
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-12:
            normal = normal / norm_len
        dists = np.abs(centered @ normal)
        rms_m = float(np.sqrt(np.mean(dists**2)))
        return rms_m * 1000.0
    except Exception:
        return 0.0


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

        # ------------------------------------------------------------------
        # Planarity self-check and dimension estimation:
        # For each of the 3 local axes:
        #   1. Extract the outer 10% face points on near and far ends.
        #   2. Compute SVD plane-fit RMS perpendicular distances (planarity check).
        #   3. Compute dimension as distance between near-face and far-face mean
        #      positions along that axis (plane-fit dimension).
        #   4. Fallback: if either face has < 30 points, use trimmed percentiles.
        # ------------------------------------------------------------------
        rms_values: list[float] = []
        per_axis_dims: list[float] = []
        dimension_methods: list[str] = []

        for axis in range(3):
            vals = local_pts[:, axis]
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            span = max_v - min_v

            near_mask = vals <= (min_v + 0.10 * span)
            far_mask = vals >= (max_v - 0.10 * span)

            near_pts = local_pts[near_mask]
            far_pts = local_pts[far_mask]

            near_rms = _fit_plane_rms_mm(near_pts)
            far_rms = _fit_plane_rms_mm(far_pts)

            rms_values.append(round(near_rms, 2))
            rms_values.append(round(far_rms, 2))

            if len(near_pts) >= 30 and len(far_pts) >= 30:
                near_mean = float(np.mean(near_pts[:, axis]))
                far_mean = float(np.mean(far_pts[:, axis]))
                axis_dim = abs(far_mean - near_mean)
                dimension_methods.append("plane_fit")
            else:
                lo_k = float(np.percentile(vals, trim_percentile))
                hi_k = float(np.percentile(vals, 100.0 - trim_percentile))
                axis_dim = float(hi_k - lo_k)
                dimension_methods.append("percentile_fallback")

            per_axis_dims.append(axis_dim)

        max_rms = float(max(rms_values)) if rms_values else 0.0
        is_valid_box = bool(max_rms <= 4.0)
        warning_msg = (
            f"This does not look like a rigid box (face RMS deviation {max_rms:.1f} mm > 4 mm threshold) "
            "— measurements may be meaningless."
            if not is_valid_box else None
        )

        planarity_check = {
            "rms_deviation_mm": rms_values,
            "max_rms_mm": round(max_rms, 2),
            "is_valid_box": is_valid_box,
            "warning": warning_msg,
        }

        # Sort longest → shortest so index 0=length, 1=width, 2=height
        # regardless of which physical axis the OBB happened to assign each.
        dims = np.sort(per_axis_dims)[::-1]
        length, width, height = float(dims[0]), float(dims[1]), float(dims[2])

        # Axis-aligned bounding box for reference / sanity checking.
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [
            round(float(v), 4) for v in np.asarray(aabb.get_extent())
        ]

        return {
            "target": "box",
            "num_points": len(pcd.points),
            "dimension_method": dimension_methods,
            "planarity_check": planarity_check,
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
