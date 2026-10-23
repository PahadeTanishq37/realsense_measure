"""
test_head_validation_false_positives_reproduction.py - Problem 12B Reproduction Suite.

Evaluates whether HeadTarget.segment() / HeadTarget.measure() falsely validates
incomplete, cropped, occluded, or non-head ellipsoidal point clouds.

Scenarios audited:
  1. Missing chin: Symmetric head with lower chin region removed by planar crop.
  2. Missing crown: Symmetric head with top cranial region removed by planar crop.
  3. Posterior truncation: Head with controlled amount of posterior surface removed.
  4. Headrest-like occlusion: Head missing lower posterior surface (nape/lower occiput).
  5. Non-head oval: Smooth, plausible-sized ellipsoid with controlled asymmetric protrusion.

Note: Expected behavior of an anthropometric cranial validator is to withhold validation
(status='unverified', volume_m3=None) when cranial anatomy is incomplete or non-human.
Tests where the current production code exhibits false-positive validation are marked with
@pytest.mark.xfail(strict=True) documenting the reproduced vulnerability pending fix.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import open3d as o3d
import pytest

from targets.head import HeadTarget
from test_head_synthetic import create_synthetic_head


def _record_scenario_metrics(name: str, pcd_in: o3d.geometry.PointCloud, target: HeadTarget) -> dict[str, Any]:
    """
    Run production segment() -> measure() pipeline and extract structured audit metrics.
    """
    n_in = len(pcd_in.points)
    seg_pcd = target.segment(pcd_in)
    n_out = len(seg_pcd.points)
    seg_meta = dict(target.last_segmentation_result)

    meas = target.measure(seg_pcd)

    metrics = {
        "scenario": name,
        "input_point_count": n_in,
        "output_point_count": n_out,
        "segmentation_outcome": seg_meta.get("outcome"),
        "cut_applied": seg_meta.get("cut_applied"),
        "longitudinal_span_m": seg_meta.get("longitudinal_span_m"),
        "status": meas.get("status"),
        "volume_m3": meas.get("volume_m3"),
        "obb_volume_m3": meas.get("oriented_bbox", {}).get("volume_m3"),
        "cranial_envelope_3d": meas.get("cranial_envelope_3d"),
        "symmetry_residual_mm": meas.get("symmetry_residual_mm"),
        "boundary_checks": meas.get("boundary_checks"),
        "diagnostics": meas.get("diagnostics", []),
    }
    return metrics


# ===========================================================================
# 1. Missing Chin
# ===========================================================================

def create_chin_cropped_head(crop_height_m: float = 0.035, seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Symmetric head with lower chin region removed by planar horizontal crop.
    Base head: W=160, D=200, H=240 mm. y spans [-0.12, +0.12].
    Cropping y < -0.12 + crop_height_m removes the chin.
    """
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)
    pts = np.asarray(pcd.points)
    # y is longitudinal height; chin is at y = -0.12
    cut_plane_y = -0.12 + crop_height_m  # -0.085 m
    mask = pts[:, 1] >= cut_plane_y
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    return cropped


@pytest.mark.xfail(reason="Reproduced false positive: chin-cropped head passes validation and receives volume", strict=True)
def test_scenario_1_missing_chin():
    """
    Scenario 1: Missing chin.
    Cranial height is truncated by 35 mm (measured ~205 mm, within [160, 350] mm range).
    Expected: unverified, volume None.
    Current Production: falsely validated with non-null volume.
    """
    target = HeadTarget()
    pcd = create_chin_cropped_head(crop_height_m=0.035)
    metrics = _record_scenario_metrics("1. Missing Chin", pcd, target)

    # Intentionally failing assertion documenting the defect:
    assert metrics["status"] == "unverified", f"Falsely validated: {metrics}"
    assert metrics["volume_m3"] is None
    assert metrics["obb_volume_m3"] is None


# ===========================================================================
# 2. Missing Crown
# ===========================================================================

def create_crown_cropped_head(crop_height_m: float = 0.035, seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Symmetric head with top cranial region removed by planar horizontal crop.
    Base head: W=160, D=200, H=240 mm. y spans [-0.12, +0.12].
    Cropping y > +0.12 - crop_height_m removes the vertex/calvarium.
    """
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)
    pts = np.asarray(pcd.points)
    cut_plane_y = 0.12 - crop_height_m  # +0.085 m
    mask = pts[:, 1] <= cut_plane_y
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    return cropped


@pytest.mark.xfail(reason="Reproduced false positive: crown-cropped head passes validation and receives volume", strict=True)
def test_scenario_2_missing_crown():
    """
    Scenario 2: Missing crown.
    Cranial vertex is truncated by 35 mm (measured ~205 mm, within [160, 350] mm range).
    Expected: unverified, volume None.
    Current Production: falsely validated with non-null volume.
    """
    target = HeadTarget()
    pcd = create_crown_cropped_head(crop_height_m=0.035)
    metrics = _record_scenario_metrics("2. Missing Crown", pcd, target)

    assert metrics["status"] == "unverified", f"Falsely validated: {metrics}"
    assert metrics["volume_m3"] is None
    assert metrics["obb_volume_m3"] is None


# ===========================================================================
# 3. Posterior Truncation
# ===========================================================================

def create_posterior_truncated_head(crop_depth_m: float = 0.020, seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Head with controlled amount of posterior surface removed.
    Base head: W=160, D=200, H=240 mm. z spans [-0.10, +0.10 + 0.035].
    Cropping z < -0.10 + crop_depth_m removes the back cap.
    """
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)
    pts = np.asarray(pcd.points)
    cut_plane_z = -0.10 + crop_depth_m  # -0.080 m for 20 mm crop
    mask = pts[:, 2] >= cut_plane_z
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    return cropped


@pytest.mark.xfail(reason="Reproduced false positive: controlled posterior-truncated head passes validation", strict=True)
def test_scenario_3_posterior_truncation_controlled():
    """
    Scenario 3: Controlled posterior truncation (20 mm removed from back of head).
    Remaining depth is ~180 mm without nose / ~215 mm with nose (in [130, 280] mm range).
    Expected: unverified, volume None.
    Current Production: falsely validated with non-null volume.
    """
    target = HeadTarget()
    pcd = create_posterior_truncated_head(crop_depth_m=0.020)
    metrics = _record_scenario_metrics("3. Controlled Posterior Truncation (20mm)", pcd, target)

    assert metrics["status"] == "unverified", f"Falsely validated: {metrics}"
    assert metrics["volume_m3"] is None
    assert metrics["obb_volume_m3"] is None


def test_scenario_3_posterior_truncation_large_rejected():
    """
    Scenario 3 variation: Large posterior truncation (40 mm removed from back of head).
    Removes points z < -0.06 m.
    Result: Current production correctly rejects because PCA tilt increases symmetry residual > 12.0 mm.
    """
    target = HeadTarget()
    pcd = create_posterior_truncated_head(crop_depth_m=0.040)
    metrics = _record_scenario_metrics("3. Large Posterior Truncation (40mm)", pcd, target)

    # Correctly fails closed in production:
    assert metrics["status"] == "unverified"
    assert metrics["volume_m3"] is None
    assert metrics["symmetry_residual_mm"] > 12.0


# ===========================================================================
# 4. Headrest-Like Occlusion
# ===========================================================================

def create_headrest_occluded_head(seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Head missing lower posterior surface region (nape/lower occiput occluded by chair/headrest).
    Upper occiput (z ~ -0.10, y > 0.0), crown (y ~ +0.12), chin (y ~ -0.12),
    face/nose (z > 0.10), and lateral temples (x ~ +/-0.08) are all present.
    Extreme caliper bounding points exist on all 6 faces.
    """
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=6000)
    pts = np.asarray(pcd.points)
    # Occlude lower posterior quadrant: z < -0.03 (posterior) and y < 0.02 (inferior)
    mask_occluded = (pts[:, 2] < -0.03) & (pts[:, 1] < 0.02)
    pts_visible = pts[~mask_occluded]
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pts_visible)
    return pcd_vis


@pytest.mark.xfail(reason="Reproduced false positive: headrest-occluded head passes validation and receives volume", strict=True)
def test_scenario_4_headrest_occlusion():
    """
    Scenario 4: Headrest-like occlusion.
    Lower posterior quadrant is missing, but all 6 bounding-box extremes are intact.
    Expected: unverified, volume None.
    Current Production: falsely validated with non-null volume.
    """
    target = HeadTarget()
    pcd = create_headrest_occluded_head()
    metrics = _record_scenario_metrics("4. Headrest Occlusion", pcd, target)

    assert metrics["status"] == "unverified", f"Falsely validated: {metrics}"
    assert metrics["volume_m3"] is None
    assert metrics["obb_volume_m3"] is None


# ===========================================================================
# 5. Non-Head Oval (Ellipsoid with Asymmetric Bump)
# ===========================================================================

def create_non_head_oval(seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Smooth, plausible-sized ellipsoid (W=150, D=190, H=220 mm) with a controlled
    asymmetric protrusion on the anterior surface (+z) to pass the A-P relief test.
    Non-cranial object (e.g. melon, watermelon, dummy).
    """
    np.random.seed(seed)
    n = 4000
    u = np.random.uniform(0, 2 * np.pi, n)
    v = np.random.uniform(0, np.pi, n)
    x = 0.075 * np.sin(v) * np.cos(u)   # breadth 150 mm
    y = 0.110 * np.cos(v)               # height 220 mm
    z = 0.095 * np.sin(v) * np.sin(u)   # depth 190 mm

    # Controlled localized protrusion on anterior pole (+z) to pass A-P relief asymmetry
    mask_bump = (z > 0.08) & (np.abs(x) < 0.03) & (np.abs(y) < 0.03)
    z[mask_bump] += 0.015

    pts = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


@pytest.mark.xfail(reason="Reproduced false positive: non-cranial ellipsoid passes validation and receives volume", strict=True)
def test_scenario_5_non_head_oval():
    """
    Scenario 5: Non-head oval.
    A smooth non-cranial ellipsoid with human dimensions.
    Expected: unverified, volume None.
    Current Production: falsely validated with non-null volume.
    """
    target = HeadTarget()
    pcd = create_non_head_oval()
    metrics = _record_scenario_metrics("5. Non-Head Oval", pcd, target)

    assert metrics["status"] == "unverified", f"Falsely validated: {metrics}"
    assert metrics["volume_m3"] is None
    assert metrics["obb_volume_m3"] is None
