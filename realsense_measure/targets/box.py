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


def _check_face_support(
    face_pts: np.ndarray,
    axis: int,
    other_axes: list[int],
    nominal_extents: list[float],
) -> tuple[bool, str, float]:
    """
    Evaluate physical evidence and 2D spatial support for a candidate box face.

    Parameters
    ----------
    face_pts:
        Points in canonical local coordinate frame belonging to this boundary slab.
    axis:
        The principal axis index (0, 1, or 2) normal to this face.
    other_axes:
        The two orthogonal axis indices lying in the plane of this face.
    nominal_extents:
        Estimated full extent of the object along each principal axis.

    Returns
    -------
    tuple[bool, str, float]
        (is_supported, diagnostic_reason, rms_mm)
    """
    n_pts = len(face_pts)
    if n_pts < 25:
        return False, f"too few points ({n_pts} < 25 threshold)", 0.0

    rms = float(np.sqrt(np.mean((face_pts[:, axis] - np.mean(face_pts[:, axis])) ** 2))) * 1000.0

    u = face_pts[:, other_axes[0]]
    v = face_pts[:, other_axes[1]]
    L_u = nominal_extents[other_axes[0]]
    L_v = nominal_extents[other_axes[1]]

    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))
    span_u = u_max - u_min
    span_v = v_max - v_min

    u_span_rel = span_u / max(L_u, 1e-4)
    v_span_rel = span_v / max(L_v, 1e-4)

    # 1. Span coverage: face points must span a plausible fraction of the object's width/length
    if u_span_rel < 0.40 or v_span_rel < 0.40:
        return False, f"narrow orthogonal coverage (u_span={u_span_rel:.2f}, v_span={v_span_rel:.2f} < 0.40)", rms

    # 2. Interior presence: distinguishes a solid planar face from a hollow cut rim
    # On a hollow rim (missing face), points exist only on the thin perimeter from adjacent faces.
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0
    inner_mask = (np.abs(u - u_mid) <= 0.25 * span_u) & (np.abs(v - v_mid) <= 0.25 * span_v)
    inner_pts = int(np.sum(inner_mask))
    inner_ratio = inner_pts / float(n_pts)

    if inner_pts < 8 or inner_ratio < 0.08:
        return False, f"hollow/sparse face interior (inner_pts={inner_pts}, ratio={inner_ratio:.2f})", rms

    # 3. 2D grid dispersion: points must be spread across multiple cells, not bunched in one corner
    u_bins = np.digitize(u, np.linspace(u_min, u_max, 4)) - 1
    v_bins = np.digitize(v, np.linspace(v_min, v_max, 4)) - 1
    valid_bins = (u_bins >= 0) & (u_bins < 3) & (v_bins >= 0) & (v_bins < 3)
    grid = np.zeros((3, 3), dtype=int)
    for ub, vb in zip(u_bins[valid_bins], v_bins[valid_bins]):
        grid[ub, vb] += 1
    occupied_cells = int(np.sum(grid > 0))

    if occupied_cells < 5:
        return False, f"clustered coverage ({occupied_cells}/9 cells occupied)", rms

    # 4. Planarity RMS limit
    if rms > 4.0:
        return False, f"RMS deviation {rms:.2f} mm exceeds 4.0 mm threshold", rms

    return True, "face supported", rms


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
        This extra plane-removal pass mops those up before measurement while
        protecting valid box faces.

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
        from config import TargetConfig
        target_cfg = TargetConfig()
        no_plane, _ = remove_dominant_plane(pcd, self.cfg, target_cfg=target_cfg)

        if not no_plane.has_points():
            # Plane removal consumed everything — return the original cloud
            # rather than propagating an empty cloud to measure().
            return pcd

        return extract_largest_cluster(no_plane, self.cfg)

    def is_plausible_box(self, pcd: o3d.geometry.PointCloud) -> tuple[bool, str]:
        """
        Quick pre-check before OBB fitting to verify whether the cloud is plausibly box-shaped.

        Checks:
          1. len(pcd.points) >= 200 (minimum points required to judge 3D box shape).
          2. Aspect ratio sanity: AABB extents sorted descending [a, b, c].
             If a / max(c, 1e-6) > 20 -> implausible aspect ratio (sliver or plane fragment).
          3. Point density sanity: volume = a * b * c.
             If volume > 1e-6 and points/volume < 500 (pts/m^3) -> too sparse.

        Parameters
        ----------
        pcd:
            Segmented candidate box cloud.

        Returns
        -------
        tuple[bool, str]
            (is_plausible, reason_if_false)
        """
        n_pts = len(pcd.points)
        if n_pts < 200:
            return False, f"too few points ({n_pts}) to determine box shape (minimum 200)"

        pts = np.asarray(pcd.points)
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        extent = maxs - mins
        extents_sorted = sorted(extent, reverse=True)
        a, b, c = float(extents_sorted[0]), float(extents_sorted[1]), float(extents_sorted[2])

        c_safe = max(c, 1e-6)
        aspect_ratio = a / c_safe
        if aspect_ratio > 20.0:
            return False, f"aspect ratio {aspect_ratio:.1f}:1 is implausible for a rigid box"

        volume = a * b * c
        if volume > 1e-6:
            density = n_pts / volume
            if density < 500.0:
                return False, "too sparse for a solid object; likely a small fragment stretched over a large empty region"

        return True, ""

    def measure(
        self,
        pcd: o3d.geometry.PointCloud,
        trim_percentile: float = 0.5,
    ) -> dict:
        """
        Estimate box dimensions using 3D point-to-point Euclidean distances between
        identified physical boundary endpoints.

        Parameters
        ----------
        pcd:
            Segmented box cloud, as returned by :meth:`segment`.
        trim_percentile:
            Percentile (0–50) used for outlier-trimming when computing
            boundary endpoints. Default 0.5 trims the outermost 0.5 % of points
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
        # Identify principal box axes via Oriented Bounding Box
        # ------------------------------------------------------------------
        try:
            obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        except AttributeError:
            obb = pcd.get_oriented_bounding_box(robust=True)

        pts = np.asarray(pcd.points)
        R = np.asarray(obb.R)           # 3×3 rotation matrix (principal axes as columns)
        center = np.asarray(obb.center)

        # Project all points into the canonical local coordinate frame
        local_pts = (pts - center) @ R
        nominal_extents = [float(np.max(local_pts[:, i]) - np.min(local_pts[:, i])) for i in range(3)]

        # ------------------------------------------------------------------
        # Approach 1: 3D Point-to-Point Euclidean Distance between Endpoints
        # For each axis k:
        #   P1 = center + lo_val * R[:, k]
        #   P2 = center + hi_val * R[:, k]
        #   d = sqrt((X2-X1)^2 + (Y2-Y1)^2 + (Z2-Z1)^2)
        # ------------------------------------------------------------------
        endpoints_3d = []
        per_axis_dims: list[float] = []
        rms_values: list[float] = []
        is_axis_supported: list[bool] = []
        face_diagnostics: list[str] = []

        for axis in range(3):
            other_axes = [i for i in range(3) if i != axis]
            vals = local_pts[:, axis]
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            span = max_v - min_v

            # Identify robust physical endpoints along this principal axis
            lo_val = float(np.percentile(vals, trim_percentile))
            hi_val = float(np.percentile(vals, 100.0 - trim_percentile))

            # 3D endpoints expressed in the shared world/camera coordinate frame
            P1 = center + lo_val * R[:, axis]
            P2 = center + hi_val * R[:, axis]

            # Point-to-point Euclidean distance in 3D
            axis_dim = float(np.sqrt(np.sum((P2 - P1) ** 2)))
            per_axis_dims.append(axis_dim)
            endpoints_3d.append((P1.tolist(), P2.tolist()))

            # Quality validation: boundary point support, spatial coverage & planarity RMS
            face_tolerance = max(0.003, 0.02 * span)
            near_pts = local_pts[np.abs(vals - lo_val) <= face_tolerance]
            far_pts = local_pts[np.abs(vals - hi_val) <= face_tolerance]

            near_ok, near_msg, near_rms = _check_face_support(near_pts, axis, other_axes, nominal_extents)
            far_ok, far_msg, far_rms = _check_face_support(far_pts, axis, other_axes, nominal_extents)

            rms_values.append(round(near_rms, 2))
            rms_values.append(round(far_rms, 2))
            face_diagnostics.append(f"Axis {axis} near: {near_msg}")
            face_diagnostics.append(f"Axis {axis} far: {far_msg}")

            is_supported = bool(near_ok and far_ok)
            is_axis_supported.append(is_supported)

        max_rms = float(max(rms_values)) if rms_values else 0.0
        all_supported = all(is_axis_supported)
        min_dim_valid = bool(min(per_axis_dims) >= 0.02)
        is_valid_box = bool(max_rms <= 4.0 and all_supported and min_dim_valid)

        if not min_dim_valid:
            warning_msg = f"Degenerate dimension detected (smallest span {min(per_axis_dims) * 1000.0:.1f} mm < 20 mm minimum) — likely a 2D plane fragment, measurements are unverified."
        elif not all_supported:
            failed_msgs = [msg for msg in face_diagnostics if "face supported" not in msg]
            warning_msg = f"Incomplete boundary coverage ({'; '.join(failed_msgs)}) — measurements are unverified."
        elif not (max_rms <= 4.0):
            warning_msg = f"This does not look like a rigid box (face RMS deviation {max_rms:.1f} mm > 4 mm threshold) — measurements are unverified."
        else:
            warning_msg = None

        planarity_check = {
            "rms_deviation_mm": rms_values,
            "max_rms_mm": round(max_rms, 2),
            "is_valid_box": is_valid_box,
            "all_faces_supported": all_supported,
            "face_diagnostics": face_diagnostics,
            "warning": warning_msg,
        }

        # Sort longest → shortest so index 0=length, 1=width, 2=height
        order = np.argsort(per_axis_dims)[::-1]
        length = float(per_axis_dims[order[0]])
        width = float(per_axis_dims[order[1]])
        height = float(per_axis_dims[order[2]])

        axis_names = ["length", "width", "height"]
        sorted_endpoints = {
            axis_names[i]: {
                "P1": endpoints_3d[order[i]][0],
                "P2": endpoints_3d[order[i]][1],
                "distance_m": round(float(per_axis_dims[order[i]]), 4),
                "distance_mm": round(float(per_axis_dims[order[i]]) * 1000.0, 1),
            }
            for i in range(3)
        }

        # Axis-aligned bounding box for reference / sanity checking.
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [
            round(float(v), 4) for v in np.asarray(aabb.get_extent())
        ]

        return {
            "target": "box",
            "status": "validated" if is_valid_box else "unverified",
            "measurement_method": "3d_point_to_point_distance",
            "num_points": len(pcd.points),
            "endpoints_3d": sorted_endpoints,
            "planarity_check": planarity_check,
            "oriented_bbox": {
                "length_m": round(length, 4),
                "width_m": round(width, 4),
                "height_m": round(height, 4),
                "volume_m3": round(length * width * height, 6) if is_valid_box else None,
                "estimated_volume_m3": round(length * width * height, 6),
                "center": center.tolist(),
                "rotation": R.tolist(),
            },
            "axis_aligned_bbox_extent_m": aabb_extent,
            "geometry_for_viz": obb,
        }
