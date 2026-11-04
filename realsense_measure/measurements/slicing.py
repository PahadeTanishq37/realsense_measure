"""
realsense_measure/measurements/slicing.py
=========================================
Stage 17: Robust Cross-Sectional Anatomical Slicing & Ring Completeness Auditor.

Extracts cross-sectional point slabs along explicit anatomical planes and audits
angular surface coverage, gap size, and closed-ring defensibility.

CRITICAL RESEARCH SAFEGUARD:
Single-view RGB-D captures only view the front of the human subject (typically
~120-150 degrees of angular coverage). The posterior/back is completely unobserved.
Under NO circumstances does this module connect incomplete arcs across empty back space
to claim a full body circumference. When coverage < 75% or max gap > 60 degrees,
closed circumference extraction is explicitly marked as INCOMPLETE_PERIMETER and blocked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SliceRingAnalysis:
    """Detailed geometric analysis of an anatomical plane cross-section."""
    plane_name: str
    plane_point: List[float]       # [X, Y, Z] anchor point in metres
    plane_normal: List[float]      # [Nx, Ny, Nz] unit normal vector
    slab_thickness_mm: float
    num_points_in_slab: int
    transverse_breadth_mm: float   # Max lateral dimension across slice
    anteroposterior_depth_mm: float # Observed anterior-posterior depth
    observed_bins_count: int
    total_angular_bins: int        # Standard: 72 bins (5 deg each)
    observed_coverage_fraction: float  # [0.0, 1.0]
    missing_surface_fraction: float    # 1.0 - coverage_fraction
    max_angular_gap_deg: float         # Largest unobserved angular sector
    is_closed_ring: bool               # True only if coverage >= 0.75 and max_gap <= 60 deg
    observed_arc_length_mm: float      # Metric arc length of observed physical points
    closed_circumference_mm: Optional[float]  # None if is_closed_ring is False
    ring_status: str                   # "VALID_CLOSED_RING", "INCOMPLETE_PERIMETER_ANTERIOR_ONLY", "INSUFFICIENT_POINTS"
    blocking_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnatomicalPlaneSlicer:
    """
    Extracts and evaluates planar cross-sections from 3D point clouds.
    """

    DEFAULT_TOTAL_BINS = 72       # 5.0 degree bins
    MIN_CLOSED_COVERAGE = 0.75    # 75% surface coverage required for full circumference
    MAX_CLOSED_GAP_DEG = 60.0     # Maximum gap allowed for closed-loop interpolation
    MIN_SLAB_POINTS = 30          # Minimum point count required for slice analysis

    @classmethod
    def slice_point_cloud(
        cls,
        points_3d: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        plane_name: str = "anatomical_plane",
        slab_thickness_m: float = 0.020,  # 20 mm (+-10 mm)
        ref_lateral: Optional[np.ndarray] = None,
        ref_anterior: Optional[np.ndarray] = None,
        legacy_swap: bool = False,
    ) -> Tuple[np.ndarray, SliceRingAnalysis]:
        """
        Slices 3D point cloud along plane and audits coverage.

        Parameters
        ----------
        points_3d : np.ndarray
            (N, 3) metric 3D point array (in metres).
        plane_point : np.ndarray
            (3,) anchor point on plane.
        plane_normal : np.ndarray
            (3,) normal vector to the slicing plane.
        plane_name : str
            Descriptive anatomical plane name.
        slab_thickness_m : float
            Total thickness of cutting slab in metres.
        ref_lateral : Optional[np.ndarray]
            Reference unit vector for lateral axis (defaults to [1, 0, 0]).
        ref_anterior : Optional[np.ndarray]
            Reference unit vector for anterior axis (defaults to [0, 0, 1]).
        legacy_swap : bool
            If True, uses legacy uncorrected cross-product basis construction (defect reproduction).

        Returns
        -------
        Tuple[np.ndarray, SliceRingAnalysis]
            (M, 3) 3D points inside slab, and comprehensive analysis dataclass.
        """
        pts = np.asarray(points_3d, dtype=np.float32)
        p0 = np.asarray(plane_point, dtype=np.float32).reshape(3)
        n = np.asarray(plane_normal, dtype=np.float32).reshape(3)
        norm_len = np.linalg.norm(n)
        if norm_len < 1e-6:
            raise ValueError("Plane normal vector cannot be zero length.")
        n = n / norm_len

        half_slab = slab_thickness_m / 2.0
        dist_to_plane = np.dot(pts - p0, n)
        in_slab_mask = np.abs(dist_to_plane) <= half_slab
        slab_pts_3d = pts[in_slab_mask]

        if len(slab_pts_3d) < cls.MIN_SLAB_POINTS:
            analysis = SliceRingAnalysis(
                plane_name=plane_name,
                plane_point=p0.tolist(),
                plane_normal=n.tolist(),
                slab_thickness_mm=slab_thickness_m * 1000.0,
                num_points_in_slab=len(slab_pts_3d),
                transverse_breadth_mm=0.0,
                anteroposterior_depth_mm=0.0,
                observed_bins_count=0,
                total_angular_bins=cls.DEFAULT_TOTAL_BINS,
                observed_coverage_fraction=0.0,
                missing_surface_fraction=1.0,
                max_angular_gap_deg=360.0,
                is_closed_ring=False,
                observed_arc_length_mm=0.0,
                closed_circumference_mm=None,
                ring_status="INSUFFICIENT_POINTS",
                blocking_reason=f"Only {len(slab_pts_3d)} points in slice slab (minimum {cls.MIN_SLAB_POINTS} required).",
            )
            return slab_pts_3d, analysis

        # Construct 2D orthonormal coordinate system on plane (u=lateral, v=sagittal)
        u_axis, v_axis = cls._construct_plane_basis(
            n, ref_lateral=ref_lateral, ref_anterior=ref_anterior, legacy_swap=legacy_swap
        )

        # Project 3D points to 2D plane coordinates
        rel_pts = slab_pts_3d - p0
        u = np.dot(rel_pts, u_axis)  # Lateral / transverse axis
        v = np.dot(rel_pts, v_axis)  # Sagittal / anterior-posterior axis

        # Basic dimensions
        transverse_breadth_mm = float((np.max(u) - np.min(u)) * 1000.0)
        anteroposterior_depth_mm = float((np.max(v) - np.min(v)) * 1000.0)

        # Centroid
        u_c = float(np.median(u))
        v_c = float(np.median(v))

        # Polar coordinates relative to centroid
        du = u - u_c
        dv = v - v_c
        angles_rad = np.arctan2(dv, du)  # [-pi, pi]
        radii_m = np.hypot(du, dv)

        # Angular binning
        total_bins = cls.DEFAULT_TOTAL_BINS
        bin_width_rad = 2.0 * np.pi / total_bins
        # Shift [-pi, pi] to [0, 2pi)
        angles_pos = np.mod(angles_rad, 2.0 * np.pi)
        bin_indices = np.floor(angles_pos / bin_width_rad).astype(int)
        bin_indices = np.clip(bin_indices, 0, total_bins - 1)

        unique_bins = np.unique(bin_indices)
        observed_bins_count = len(unique_bins)
        coverage_fraction = float(observed_bins_count / total_bins)
        missing_surface_fraction = float(1.0 - coverage_fraction)

        # Compute maximum angular gap
        max_gap_deg = cls._compute_max_angular_gap(angles_pos)

        # Determine ring closure
        is_closed_ring = (coverage_fraction >= cls.MIN_CLOSED_COVERAGE) and (max_gap_deg <= cls.MAX_CLOSED_GAP_DEG)

        # Compute observed physical arc length (along observed points)
        observed_arc_length_mm = cls._compute_observed_arc_length(u, v, angles_pos)

        if is_closed_ring:
            # Closed circumference valid
            closed_circ_mm = float(cls._compute_closed_perimeter(u, v))
            status = "VALID_CLOSED_RING"
            blocking_reason = None
        else:
            closed_circ_mm = None
            status = "INCOMPLETE_PERIMETER_ANTERIOR_ONLY"
            blocking_reason = (
                f"Incomplete cross-sectional surface coverage ({coverage_fraction*100.0:.1f}% observed < "
                f"{cls.MIN_CLOSED_COVERAGE*100.0:.0f}% required; maximum angular gap {max_gap_deg:.1f} deg > "
                f"{cls.MAX_CLOSED_GAP_DEG:.0f} deg). Single-view physical capture lacks posterior surface. "
                f"Closed circumference extraction blocked to prevent perimeter under-estimation."
            )

        analysis = SliceRingAnalysis(
            plane_name=plane_name,
            plane_point=p0.tolist(),
            plane_normal=n.tolist(),
            slab_thickness_mm=slab_thickness_m * 1000.0,
            num_points_in_slab=len(slab_pts_3d),
            transverse_breadth_mm=transverse_breadth_mm,
            anteroposterior_depth_mm=anteroposterior_depth_mm,
            observed_bins_count=observed_bins_count,
            total_angular_bins=total_bins,
            observed_coverage_fraction=coverage_fraction,
            missing_surface_fraction=missing_surface_fraction,
            max_angular_gap_deg=max_gap_deg,
            is_closed_ring=is_closed_ring,
            observed_arc_length_mm=observed_arc_length_mm,
            closed_circumference_mm=closed_circ_mm,
            ring_status=status,
            blocking_reason=blocking_reason,
        )

        return slab_pts_3d, analysis

    @classmethod
    def _construct_plane_basis(
        cls,
        n: np.ndarray,
        ref_lateral: Optional[np.ndarray] = None,
        ref_anterior: Optional[np.ndarray] = None,
        legacy_swap: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs two orthogonal vectors lying in the plane normal to n.

        u_axis: Explicit lateral (transverse) axis lying in the plane.
        v_axis: Explicit sagittal (anteroposterior) axis lying in the plane.

        Ensures:
        1. u_axis is orthogonal to n: dot(u, n) == 0.
        2. v_axis is orthogonal to n: dot(v, n) == 0.
        3. u_axis and v_axis are mutually orthogonal: dot(u, v) == 0.
        4. u_axis measures lateral breadth (along subject/camera X axis).
        5. v_axis measures sagittal depth (along subject/camera Z axis).
        """
        n = np.asarray(n, dtype=np.float32).reshape(3)
        norm_n = float(np.linalg.norm(n))
        if norm_n < 1e-6:
            raise ValueError("Normal vector cannot be zero length.")
        n = n / norm_n

        if legacy_swap:
            # Reproduces the Stage 20 legacy defect for diagnostic auditing
            if abs(n[0]) < 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            u = np.cross(n, ref)
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            v = v / np.linalg.norm(v)
            return u, v

        ref_lat = np.array([1.0, 0.0, 0.0], dtype=np.float32) if ref_lateral is None else np.asarray(ref_lateral, dtype=np.float32).reshape(3)
        ref_ant = np.array([0.0, 0.0, 1.0], dtype=np.float32) if ref_anterior is None else np.asarray(ref_anterior, dtype=np.float32).reshape(3)

        # 1. Project ref_lateral onto plane to get in-plane lateral vector u
        u = ref_lat - float(np.dot(ref_lat, n)) * n
        norm_u = float(np.linalg.norm(u))
        if norm_u < 1e-4:
            # Degenerate case: plane normal is parallel to X (sagittal plane); fallback to anterior ref
            u = ref_ant - float(np.dot(ref_ant, n)) * n
            norm_u = float(np.linalg.norm(u))
        u = u / norm_u

        # 2. Project ref_anterior onto plane and orthogonalize against u to get in-plane sagittal vector v
        v = ref_ant - float(np.dot(ref_ant, n)) * n - float(np.dot(ref_ant, u)) * u
        norm_v = float(np.linalg.norm(v))
        if norm_v < 1e-4:
            v = np.cross(n, u)
            norm_v = float(np.linalg.norm(v))
        v = v / norm_v

        return u, v

    @staticmethod
    def _compute_max_angular_gap(angles_pos: np.ndarray) -> float:
        """Finds largest angular gap in degrees across circle [0, 2*pi)."""
        if len(angles_pos) < 2:
            return 360.0
        sorted_angles = np.sort(angles_pos)
        # Differences between consecutive angles
        gaps = np.diff(sorted_angles)
        # Wrap-around gap from last point back to first point
        wrap_gap = (2.0 * np.pi + sorted_angles[0]) - sorted_angles[-1]
        all_gaps = np.append(gaps, wrap_gap)
        return float(np.max(all_gaps) * 180.0 / np.pi)

    @staticmethod
    def _compute_observed_arc_length(u: np.ndarray, v: np.ndarray, angles_pos: np.ndarray) -> float:
        """Computes length of observed contiguous surface arc in mm."""
        order = np.argsort(angles_pos)
        u_sorted = u[order]
        v_sorted = v[order]
        ang_sorted = angles_pos[order]

        # Calculate distances only between adjacent points where angular step is small (< 15 degrees)
        max_step_rad = 15.0 * np.pi / 180.0
        total_len_m = 0.0
        for i in range(len(u_sorted) - 1):
            if (ang_sorted[i+1] - ang_sorted[i]) <= max_step_rad:
                step = math.hypot(u_sorted[i+1] - u_sorted[i], v_sorted[i+1] - v_sorted[i])
                total_len_m += step
        return float(total_len_m * 1000.0)

    @staticmethod
    def _compute_closed_perimeter(u: np.ndarray, v: np.ndarray) -> float:
        """Computes convex hull perimeter of a closed cross-section in mm."""
        from scipy.spatial import ConvexHull
        pts_2d = np.stack([u, v], axis=-1)
        hull = ConvexHull(pts_2d)
        # ConvexHull perimeter is available via area/volume or summing vertices
        hull_pts = pts_2d[hull.vertices]
        diffs = np.diff(np.vstack([hull_pts, hull_pts[0:1]]), axis=0)
        perim_m = np.sum(np.linalg.norm(diffs, axis=1))
        return float(perim_m * 1000.0)

