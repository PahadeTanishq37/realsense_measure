"""
tests/fullbody/test_measure_scan.py

Unit tests for model-free anthropometric measurement extraction:
- Circumference on analytic circular and elliptical cross sections.
- Arm masking for trunk slices.
- Perpendicular slicing for limbs.
- Quality flag triggering (OK, PARTIAL, NOT_MEASURABLE).
- Stature calculation and crown_clipped guard.
"""

from __future__ import annotations

import math
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.landmarks import SlicingPlane
from realsense_measure.fullbody.measure_scan import (
    extract_ring_circumference,
    measure_fused_scan,
)
from realsense_measure.fullbody.types import Keypoints3D, QualityFlag
from tests.fullbody.conftest import create_ground_truth_keypoints, ramanujan_ellipse_perimeter


def test_analytic_circle_circumference_accuracy():
    """Verify extracted circumference on an analytic cylinder matches 2*pi*r within 1%."""
    cfg = FullBodyConfig(slice_thickness_m=0.02, num_angular_bins=180)
    radius_m = 0.15  # 150 mm radius -> perimeter ~ 942.5 mm
    expected_mm = 2.0 * math.pi * radius_m * 1000.0

    # Generate 3D point ring on cylinder at Y = 1.0 m
    angles = np.linspace(0, 2 * math.pi, 500, endpoint=False)
    y_vals = np.random.uniform(0.99, 1.01, size=len(angles))
    pts = np.stack([radius_m * np.cos(angles), y_vals, radius_m * np.sin(angles)], axis=1)

    plane = SlicingPlane(
        name="test_circle",
        origin=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        is_limb=False,
    )

    res_polar, res_hull = extract_ring_circumference(pts, plane, None, cfg)

    assert res_polar.quality_flag == QualityFlag.OK.value
    assert res_polar.coverage > 0.95
    # Value within 1% of analytic
    err_pct = abs(res_polar.value_mm - expected_mm) / expected_mm * 100.0
    assert err_pct < 1.0


def test_analytic_ellipse_circumference_accuracy():
    """Verify extracted circumference on an ellipse matches Ramanujan's formula within 1.5%."""
    cfg = FullBodyConfig(slice_thickness_m=0.02, num_angular_bins=180)
    a, b = 0.18, 0.12  # Semi-major and semi-minor axes in metres
    expected_mm = ramanujan_ellipse_perimeter(a, b) * 1000.0

    angles = np.linspace(0, 2 * math.pi, 600, endpoint=False)
    y_vals = np.random.uniform(0.99, 1.01, size=len(angles))
    pts = np.stack([a * np.cos(angles), y_vals, b * np.sin(angles)], axis=1)

    plane = SlicingPlane(
        name="test_ellipse",
        origin=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        is_limb=False,
    )

    res_polar, res_hull = extract_ring_circumference(pts, plane, None, cfg)

    assert res_polar.quality_flag == QualityFlag.OK.value
    err_pct = abs(res_polar.value_mm - expected_mm) / expected_mm * 100.0
    assert err_pct < 1.5


def test_arm_masking_from_torso_ring():
    """Verify arm points labeled by parts.py are excluded from torso slicing slabs."""
    cfg = FullBodyConfig(slice_thickness_m=0.03)

    # Torso ring points (chest circle, radius 0.15m)
    angles = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    torso_pts = np.stack([0.15 * np.cos(angles), np.full(200, 1.25), 0.15 * np.sin(angles)], axis=1)

    # Extraneous arm points located at X = -0.30 m
    arm_pts = np.array([
        [-0.30, 1.25, 0.0],
        [-0.30, 1.25, 0.05],
        [-0.30, 1.25, -0.05],
    ])
    all_pts = np.vstack([torso_pts, arm_pts])

    # Body part labels: 2 = TORSO, 4 = UPPER_ARM_LEFT
    labels = np.array([2] * len(torso_pts) + [4] * len(arm_pts), dtype=np.int32)

    plane = SlicingPlane(
        name="chest",
        origin=np.array([0.0, 1.25, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        is_limb=False,
    )

    res_polar, res_hull = extract_ring_circumference(all_pts, plane, labels, cfg)

    # If arm points were NOT masked, convex hull perimeter would be significantly larger (> 1100 mm)
    expected_circle_mm = 2 * math.pi * 0.15 * 1000.0
    assert abs(res_hull.value_mm - expected_circle_mm) < 40.0


def test_quality_flag_triggers_correctly():
    """Verify OK, PARTIAL, and NOT_MEASURABLE flags trigger according to angular coverage."""
    cfg = FullBodyConfig(
        slice_thickness_m=0.03,
        num_angular_bins=100,
        min_angular_coverage_for_ok=0.70,
        min_angular_coverage_for_partial=0.35,
    )
    plane = SlicingPlane(
        name="test_qual",
        origin=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
    )

    # 1. 85% observed -> OK
    ang_85 = np.linspace(0, 0.85 * 2 * np.pi, 200)
    pts_85 = np.stack([0.15 * np.cos(ang_85), np.full(200, 1.0), 0.15 * np.sin(ang_85)], axis=1)
    res_85, _ = extract_ring_circumference(pts_85, plane, None, cfg)
    assert res_85.quality_flag == QualityFlag.OK.value

    # 2. 50% observed -> PARTIAL
    ang_50 = np.linspace(0, 0.50 * 2 * np.pi, 100)
    pts_50 = np.stack([0.15 * np.cos(ang_50), np.full(100, 1.0), 0.15 * np.sin(ang_50)], axis=1)
    res_50, _ = extract_ring_circumference(pts_50, plane, None, cfg)
    assert res_50.quality_flag == QualityFlag.PARTIAL.value

    # 3. 20% observed -> NOT_MEASURABLE
    ang_20 = np.linspace(0, 0.20 * 2 * np.pi, 40)
    pts_20 = np.stack([0.15 * np.cos(ang_20), np.full(40, 1.0), 0.15 * np.sin(ang_20)], axis=1)
    res_20, _ = extract_ring_circumference(pts_20, plane, None, cfg)
    assert res_20.quality_flag == QualityFlag.NOT_MEASURABLE.value
    assert math.isnan(res_20.value_mm)


def test_stature_and_crown_clipped_flag():
    """Verify stature calculation and crown_clipped guard."""
    cfg = FullBodyConfig()
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.9, 0.0],
        [0.0, 1.80, 0.0],  # crown at 1.80 m
    ])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # Without crown clipping
    res_normal = measure_fused_scan(pcd, None, cfg, crown_clipped_in_camera=False)
    assert np.isclose(res_normal["stature"].value_mm, 1800.0, atol=20.0)
    assert res_normal["stature"].quality_flag == QualityFlag.OK.value

    # With crown clipping
    res_clipped = measure_fused_scan(pcd, None, cfg, crown_clipped_in_camera=True)
    assert res_clipped["stature"].quality_flag == QualityFlag.PARTIAL.value
    assert "crown_clipped" in res_clipped["stature"].notes

