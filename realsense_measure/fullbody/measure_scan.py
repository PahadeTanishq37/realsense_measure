"""
realsense_measure/fullbody/measure_scan.py

Model-free anthropometric measurements derived directly from the fused scan:
- Stature (floor to crown with FOV clipping guard).
- Shoulder breadth and linear skeletal segment lengths.
- 2D cross-sectional slicing perpendicular to torso/limb long axes.
- Arm masking for trunk rings.
- Dual-method circumference extraction:
    (a) Polar radius profile with periodic interpolation & noise bias correction.
    (b) Convex hull perimeter (tape-line proxy).
- Angular coverage, interpolated fraction, bootstrap uncertainty, and quality flags.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.landmarks import SlicingPlane, build_slicing_planes, extract_anatomical_levels
from realsense_measure.fullbody.parts import BodyPart, is_arm_point, label_points_by_body_part
from realsense_measure.fullbody.types import Keypoints3D, MeasurementResult, QualityFlag

logger = logging.getLogger(__name__)


def estimate_crown_by_head_fit(
    pts,
    cfg,
):
    """
    Fit a sphere to the head-region point cloud and return (apex_y_m, uncertainty_mm).

    The apex is the top of the fitted sphere: apex_y = cy + r.
    Uncertainty is 2x the RMS fit residual. Returns (nan, nan) when < 20 head pts.
    """
    if len(pts) < 20:
        return float("nan"), float("nan")

    y_max = float(np.percentile(pts[:, 1], 99.9))
    y_range = y_max - float(np.min(pts[:, 1]))
    y_min_head = y_max - 0.25 * y_range
    head_mask = (
        (pts[:, 1] >= y_min_head)
        & (np.abs(pts[:, 0]) < 0.20)
        & (np.abs(pts[:, 2]) < 0.20)
    )
    head_pts = pts[head_mask]
    if len(head_pts) < 20:
        return float("nan"), float("nan")

    # Linearised least-squares sphere fit
    # 2*cx*x + 2*cy*y + 2*cz*z + (r^2 - cx^2 - cy^2 - cz^2) = x^2 + y^2 + z^2
    A = np.column_stack([
        2.0 * head_pts[:, 0],
        2.0 * head_pts[:, 1],
        2.0 * head_pts[:, 2],
        np.ones(len(head_pts)),
    ])
    b_vec = np.sum(head_pts ** 2, axis=1)
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx_fit, cy_fit, cz_fit = result[0], result[1], result[2]
    r_sq = result[3] + cx_fit ** 2 + cy_fit ** 2 + cz_fit ** 2
    if r_sq <= 0.0:
        return float("nan"), float("nan")
    r_fit = float(np.sqrt(r_sq))
    apex_y = float(cy_fit) + r_fit

    centre = np.array([cx_fit, cy_fit, cz_fit])
    residuals = np.abs(np.sqrt(np.sum((head_pts - centre) ** 2, axis=1)) - r_fit)
    uncertainty_mm = float(2.0 * np.sqrt(np.mean(residuals ** 2)) * 1000.0)
    return apex_y, uncertainty_mm


def measure_fused_scan(
    pcd: o3d.geometry.PointCloud,
    keypoints: Optional[Keypoints3D],
    cfg: FullBodyConfig,
    crown_clipped_in_camera: bool = False,
    hole_map: Optional[Dict[str, float]] = None,
) -> Dict[str, MeasurementResult]:
    """
    Extract all anthropometric measurements from the canonical fused point cloud.
    """
    results: Dict[str, MeasurementResult] = {}
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return results

    # 1. Anatomical levels and slicing planes
    levels = extract_anatomical_levels(pcd, keypoints, cfg)
    planes = build_slicing_planes(levels, keypoints, cfg)

    # 2. Body part labels for masking arms from torso rings
    labels = None
    if keypoints is not None:
        labels = label_points_by_body_part(pts, keypoints, cfg)

    # 3. Stature
    y_floor = levels.get("floor", 0.0)
    y_crown = levels.get("crown", float(np.percentile(pts[:, 1], 99.95)))
    stature_mm = (y_crown - y_floor) * 1000.0

    crown_hole = hole_map.get("crown", 0.0) if hole_map is not None else 0.0
    if crown_clipped_in_camera:
        stature_flag = QualityFlag.PARTIAL.value
        stature_notes = "crown_clipped (top of head touched camera FOV boundary)"
        stature_cov = 0.85
    elif crown_hole > cfg.crown_hole_threshold:
        stature_flag = QualityFlag.PARTIAL.value
        stature_notes = (
            f"crown_unobserved (missing fraction {crown_hole:.2f} > {cfg.crown_hole_threshold:.2f} "
            f"due to camera viewing horizon)"
        )
        stature_cov = float(max(0.0, 1.0 - crown_hole))
    else:
        stature_flag = QualityFlag.OK.value
        stature_notes = ""
        stature_cov = 1.0

    results["stature"] = MeasurementResult(
        name="stature",
        value_mm=stature_mm,
        method="vertical_height",
        coverage=stature_cov,
        interpolated_fraction=0.0,
        quality_flag=stature_flag,
        uncertainty_mm=3.0,
        notes=stature_notes,
    )

    # 3b. Head-fit crown estimation (secondary stature estimate via sphere fit)
    apex_hf_y, apex_hf_unc_mm = estimate_crown_by_head_fit(pts, cfg)
    if not math.isnan(apex_hf_y):
        stature_hf_mm = (apex_hf_y - y_floor) * 1000.0
        results["stature_head_fit"] = MeasurementResult(
            name="stature_head_fit",
            value_mm=stature_hf_mm,
            method="head_sphere_fit",
            coverage=stature_cov,
            interpolated_fraction=0.0,
            quality_flag=stature_flag,
            uncertainty_mm=max(apex_hf_unc_mm, 3.0),
            notes=(
                f"stature from sphere fit to head region; "
                f"sphere-fit uncertainty={apex_hf_unc_mm:.1f}mm"
            ),
        )

    # 4. Shoulder width and skeletal segment lengths from keypoints
    if keypoints is not None:
        _measure_skeletal_segments(keypoints, results)

    # 5. Circumference rings
    for plane in planes:
        res_polar, res_hull = extract_ring_circumference(pts, plane, labels, cfg)
        # Store polar as primary and hull as tape-proxy
        results[f"{plane.name}_circumference"] = res_polar
        results[f"{plane.name}_circumference_convexhull"] = res_hull

    return results


def _measure_skeletal_segments(
    keypoints: Keypoints3D,
    results: Dict[str, MeasurementResult],
) -> None:
    """Measure shoulder breadth and limb segment lengths from 3D keypoints."""
    l_sh, c_ls, _ = keypoints.get("left_shoulder")
    r_sh, c_rs, _ = keypoints.get("right_shoulder")
    if l_sh is not None and r_sh is not None and c_ls >= 0.2 and c_rs >= 0.2:
        sh_width_mm = float(np.linalg.norm(l_sh - r_sh) * 1000.0)
        results["shoulder_breadth"] = MeasurementResult(
            name="shoulder_breadth",
            value_mm=sh_width_mm,
            method="skeletal_keypoints",
            coverage=float(min(c_ls, c_rs)),
            interpolated_fraction=0.0,
            quality_flag=QualityFlag.OK.value if min(c_ls, c_rs) > 0.5 else QualityFlag.PARTIAL.value,
            uncertainty_mm=8.0,
            notes="Euclidean distance between left and right acromion keypoints",
        )

    # Segment lengths
    segments = [
        ("shoulder_elbow_length_left", "left_shoulder", "left_elbow"),
        ("shoulder_elbow_length_right", "right_shoulder", "right_elbow"),
        ("shoulder_wrist_length_left", "left_shoulder", "left_wrist"),
        ("shoulder_wrist_length_right", "right_shoulder", "right_wrist"),
        ("inseam_left", "left_hip", "left_ankle"),
        ("inseam_right", "right_hip", "right_ankle"),
    ]
    for seg_name, j_a, j_b in segments:
        pt_a, c_a, _ = keypoints.get(j_a)
        pt_b, c_b, _ = keypoints.get(j_b)
        if pt_a is not None and pt_b is not None and c_a >= 0.2 and c_b >= 0.2:
            seg_len_mm = float(np.linalg.norm(pt_a - pt_b) * 1000.0)
            results[seg_name] = MeasurementResult(
                name=seg_name,
                value_mm=seg_len_mm,
                method="skeletal_keypoints",
                coverage=float(min(c_a, c_b)),
                interpolated_fraction=0.0,
                quality_flag=QualityFlag.OK.value if min(c_a, c_b) > 0.5 else QualityFlag.PARTIAL.value,
                uncertainty_mm=12.0,
                notes="skeletal segment distance (lower accuracy than tape)",
            )


def extract_ring_circumference(
    points: np.ndarray,
    plane: SlicingPlane,
    labels: Optional[np.ndarray],
    cfg: FullBodyConfig,
) -> Tuple[MeasurementResult, MeasurementResult]:
    """
    Extract 2D cross section within slab and compute polar and convex-hull circumferences.
    """
    # 1. Project points onto plane
    diff = points - plane.origin
    dist_to_plane = np.dot(diff, plane.normal)
    thickness = cfg.slice_thickness_m if not plane.is_limb else max(0.025, cfg.slice_thickness_m * 1.5)
    slab_mask = np.abs(dist_to_plane) <= (thickness * 0.5)

    # 2. Filter points belonging to other body parts (mask arms from torso rings)
    if not plane.is_limb:
        if labels is not None:
            # Mask out arm points so arms never bridge into the chest/waist ring.
            # Points within trunk boundary (|X| <= 0.20m) are protected from arm labeling.
            is_arm = is_arm_point(labels) & (np.abs(points[:, 0]) > 0.20)
            slab_mask &= (~is_arm)
        # Exclude extreme lateral points
        slab_mask &= (np.abs(points[:, 0]) < 0.28)
    else:
        # For limb rings, isolate limb from adjacent anatomy using limb-appropriate radial bounds
        limb_rad_max = 0.09 if "thigh" in plane.name else 0.07
        dist_rad = np.linalg.norm(diff - dist_to_plane[:, np.newaxis] * plane.normal, axis=1)
        slab_mask &= (dist_rad < limb_rad_max)

    pts_slab = points[slab_mask]
    min_pts_required = 15 if not plane.is_limb else 8
    if len(pts_slab) < min_pts_required:
        res_empty = MeasurementResult(
            name=plane.name,
            value_mm=float("nan"),
            method="polar_contour",
            coverage=0.0,
            interpolated_fraction=1.0,
            quality_flag=QualityFlag.NOT_MEASURABLE.value,
            notes="Insufficient points in slicing slab",
        )
        return res_empty, res_empty

    # Project to 2D coordinates on plane
    diff_slab = pts_slab - plane.origin
    u_2d = np.dot(diff_slab, plane.u_axis)
    v_2d = np.dot(diff_slab, plane.v_axis)
    pts_2d = np.stack([u_2d, v_2d], axis=1)

    # Centroid
    c_u, c_v = float(np.mean(u_2d)), float(np.mean(v_2d))
    du = u_2d - c_u
    dv = v_2d - c_v
    angles = np.arctan2(dv, du)  # [-pi, pi]
    radii = np.sqrt(du ** 2 + dv ** 2)

    # 3. Bin into M angular sectors (torso uses dense bins; limbs use 36 bins matched to limb point spacing)
    M = cfg.num_angular_bins if not plane.is_limb else 36
    bin_edges = np.linspace(-np.pi, np.pi, M + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(angles, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, M - 1)

    bin_radii = np.full(M, np.nan, dtype=np.float64)
    for b in range(M):
        in_b = radii[bin_indices == b]
        if len(in_b) > 0:
            bin_radii[b] = float(np.median(in_b))

    valid_bins = ~np.isnan(bin_radii)
    observed_count = int(np.sum(valid_bins))
    coverage = float(observed_count / M)
    interpolated_frac = float(1.0 - coverage)

    # Classification
    if coverage >= cfg.min_angular_coverage_for_ok:
        q_flag = QualityFlag.OK.value
    elif coverage >= cfg.min_angular_coverage_for_partial:
        q_flag = QualityFlag.PARTIAL.value
    else:
        q_flag = QualityFlag.NOT_MEASURABLE.value

    # If unmeasurable, return NaN
    if q_flag == QualityFlag.NOT_MEASURABLE.value:
        res_unmeasurable = MeasurementResult(
            name=plane.name,
            value_mm=float("nan"),
            method="polar_contour",
            coverage=coverage,
            interpolated_fraction=interpolated_frac,
            quality_flag=q_flag,
            notes=f"Coverage {coverage*100:.1f}% below minimum {cfg.min_angular_coverage_for_partial*100:.0f}%",
        )
        return res_unmeasurable, res_unmeasurable

    # Periodic linear interpolation across missing bins
    full_radii = _interpolate_periodic_profile(bin_centers, bin_radii, valid_bins)

    # Noise bias correction: r_corr = sqrt(max(0, r^2 - sigma^2))
    sigma_noise_m = 0.002
    full_radii = np.sqrt(np.maximum(0.001, full_radii ** 2 - sigma_noise_m ** 2))

    # Polar perimeter
    x_poly = full_radii * np.cos(bin_centers)
    y_poly = full_radii * np.sin(bin_centers)
    dx = np.diff(np.append(x_poly, x_poly[0]))
    dy = np.diff(np.append(y_poly, y_poly[0]))
    polar_perimeter_mm = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)) * 1000.0)

    # Convex hull perimeter
    hull = ConvexHull(pts_2d)
    hull_perimeter_mm = float(hull.area * 1000.0)

    # Bootstrap uncertainty over points
    uncertainty_mm = _bootstrap_polar_uncertainty(u_2d, v_2d, cfg)

    res_polar = MeasurementResult(
        name=plane.name,
        value_mm=polar_perimeter_mm,
        method="polar_contour",
        coverage=coverage,
        interpolated_fraction=interpolated_frac,
        quality_flag=q_flag,
        uncertainty_mm=uncertainty_mm,
        notes="periodic-interpolated polar radius with bias correction",
    )

    res_hull = MeasurementResult(
        name=f"{plane.name}_convexhull",
        value_mm=hull_perimeter_mm,
        method="convex_hull",
        coverage=coverage,
        interpolated_fraction=interpolated_frac,
        quality_flag=q_flag,
        uncertainty_mm=uncertainty_mm,
        notes="2D convex hull tape proxy",
    )

    return res_polar, res_hull


def _interpolate_periodic_profile(
    angles: np.ndarray,
    radii: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Interpolate missing angular bins periodically on circle."""
    valid_idx = np.where(valid_mask)[0]
    out = radii.copy()
    if len(valid_idx) == 0:
        return np.zeros_like(radii)

    # Replicate for periodic boundary conditions
    m = len(angles)
    ext_angles = np.concatenate([angles[valid_idx] - 2 * np.pi, angles[valid_idx], angles[valid_idx] + 2 * np.pi])
    ext_radii = np.concatenate([radii[valid_idx], radii[valid_idx], radii[valid_idx]])

    out = np.interp(angles, ext_angles, ext_radii)
    return out


def _bootstrap_polar_uncertainty(
    u: np.ndarray,
    v: np.ndarray,
    cfg: FullBodyConfig,
) -> float:
    """Compute standard deviation of perimeter via bootstrapping over points."""
    n_pts = len(u)
    if n_pts < 20:
        return 4.0

    iters = min(30, cfg.bootstrap_iterations)
    perimeters = []
    c_u, c_v = float(np.mean(u)), float(np.mean(v))

    for _ in range(iters):
        idx = np.random.choice(n_pts, size=n_pts, replace=True)
        u_b = u[idx] - c_u
        v_b = v[idx] - c_v
        r_b = np.sqrt(u_b ** 2 + v_b ** 2)
        # Average radius
        r_mean = float(np.mean(r_b))
        perimeters.append(2.0 * math.pi * r_mean * 1000.0)

    return float(np.std(perimeters))

