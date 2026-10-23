"""
test_head_segmentation_failure_guard.py - Regression test suite for Problem 11B.

Covers the 7 required test cases for segmentation failure propagation and fail-safe guards:
  1. Problem 11A head-plus-neck case (~290 mm longitudinal span): must NOT be validated when neck is attached.
  2. No constriction candidate found: must produce outcome 'no_constriction_found' and status 'unverified'.
  3. Candidate cut leaving fewer than 150 retained points: must produce 'insufficient_retained_points' and status 'unverified'.
  4. Valid constriction cut: must produce 'constriction_cut', cut_applied=True, status 'validated', volume_m3 not None.
  5. Clearly isolated head bypassing neck search: must produce 'bypassed_short_span', status 'validated', volume_m3 not None.
  6. Repeated calls on the same HeadTarget instance: must reset metadata and not reuse stale cut metadata.
  7. Empty, sparse, or invalid input: must not produce false success metadata.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from targets.head import HeadTarget
from test_head_synthetic import create_head_with_neck_shoulders, create_synthetic_head


def create_head_plus_neck_short_span(seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Construct deterministic synthetic head + neck geometry with longitudinal span ~290 mm (< 300 mm).
    Head: 150 x 190 x 200 mm ellipsoid.
    Neck: radius 50 mm cylinder extending 90 mm inferiorly.
    Total height: ~290 mm.
    """
    np.random.seed(seed)
    n_head = 4000
    u = np.random.uniform(0, 2 * np.pi, n_head)
    v = np.random.uniform(0, np.pi, n_head)
    hx = 0.075 * np.sin(v) * np.cos(u)
    hy = 0.095 * np.sin(v) * np.sin(u)
    hz = 0.100 * np.cos(v) + 0.04  # z in [-0.06, 0.14]

    # Facial relief (nose protrusion)
    mask_nose = (hy > 0.08) & (np.abs(hx) < 0.02) & (hz > 0.0) & (hz < 0.05)
    hy[mask_nose] += 0.02

    # Neck cylinder from z = -0.06 down to -0.15 (90 mm neck length, total z-span ~0.29 m = 290 mm)
    n_neck = 1500
    neck_z = np.random.uniform(-0.15, -0.06, n_neck)
    neck_theta = np.random.uniform(0, 2 * np.pi, n_neck)
    neck_r = np.random.uniform(0.045, 0.055, n_neck)
    neck_x = neck_r * np.cos(neck_theta)
    neck_y = neck_r * np.sin(neck_theta)

    all_x = np.concatenate([hx, neck_x])
    all_y = np.concatenate([hy, neck_y])
    all_z = np.concatenate([hz, neck_z])
    pts = np.column_stack([all_x, all_y, all_z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_case_1_head_plus_neck_short_span_rejected():
    """
    Case 1: Problem 11A reproduction.
    Head-plus-neck with ~290 mm span (< 300 mm) must NOT be marked validated with a non-null volume.
    """
    pcd = create_head_plus_neck_short_span(seed=42)
    target = HeadTarget()

    seg_pcd = target.segment(pcd)
    meta = target.last_segmentation_result

    # Longitudinal span is < 300 mm, so constriction search was bypassed
    assert meta["outcome"] == "bypassed_short_span"
    assert meta["cut_applied"] is False
    assert 0.280 <= meta["longitudinal_span_m"] <= 0.300

    meas = target.measure(seg_pcd)

    # Fail-safe guard must reject attached neck: status unverified, null validated volume
    assert meas["status"] == "unverified", f"Expected unverified, got: {meas['status']}"
    assert meas["oriented_bbox"]["volume_m3"] is None, "Validated volume must be None when neck is attached"
    assert meas["volume_m3"] is None, "Validated volume_m3 must be None"
    assert len(meas["diagnostics"]) > 0, "Diagnostics must report why validation was withheld"
    assert any("unconfirmed" in d.lower() or "attached neck" in d.lower() for d in meas["diagnostics"])


def test_case_2_no_constriction_candidate_found_rejected():
    """
    Case 2: No constriction candidate found.
    Uniform cylinder (height 400 mm > 300 mm gate) has no constriction; search runs and finds no cut.
    Must produce outcome 'no_constriction_found' and status 'unverified'.
    """
    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.08, height=0.40)
    pcd_cyl = mesh_cyl.sample_points_poisson_disk(number_of_points=3000)

    target = HeadTarget()
    seg = target.segment(pcd_cyl)
    meta = target.last_segmentation_result

    assert meta["outcome"] == "no_constriction_found"
    assert meta["cut_applied"] is False
    assert meta["retained_point_count"] == len(pcd_cyl.points)

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert any("no_constriction_found" in d for d in meas["diagnostics"])


def test_case_3_insufficient_retained_points_rejected(monkeypatch):
    """
    Case 3: Candidate cut leaves fewer than 150 retained points.
    Must record outcome 'insufficient_retained_points', cut_applied=False, and status 'unverified'.
    """
    pcd = create_head_with_neck_shoulders(n_pts=8000)
    target = HeadTarget()

    # Intercept np.sum to simulate a candidate cut finding only 120 points (< 150)
    orig_sum = np.sum

    def mock_sum(a, *args, **kwargs):
        res = orig_sum(a, *args, **kwargs)
        if isinstance(a, np.ndarray) and a.dtype == bool and res > 1000:
            return 120
        return res

    monkeypatch.setattr(np, "sum", mock_sum)

    seg = target.segment(pcd)
    meta = target.last_segmentation_result

    assert meta["outcome"] == "insufficient_retained_points"
    assert meta["cut_applied"] is False
    assert "retains too few points" in meta["reason"]

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert any("insufficient_retained_points" in d for d in meas["diagnostics"])


def test_case_4_valid_constriction_cut_validated():
    """
    Case 4: Valid constriction cut still follows existing measurement-validation logic.
    Head + neck + broad shoulders must produce outcome 'constriction_cut', cut_applied=True,
    status 'validated', and a valid non-null volume_m3.
    """
    pcd = create_head_with_neck_shoulders(n_pts=8000)
    target = HeadTarget()

    seg = target.segment(pcd)
    meta = target.last_segmentation_result

    assert meta["outcome"] == "constriction_cut"
    assert meta["cut_applied"] is True
    assert meta["retained_point_count"] < len(pcd.points)
    assert meta["retained_point_count"] >= 150
    assert 0.10 <= meta["retention_ratio"] <= 0.60

    meas = target.measure(seg)
    assert meas["status"] == "validated"
    assert meas["oriented_bbox"]["volume_m3"] is not None
    assert meas["oriented_bbox"]["volume_m3"] > 0.0
    assert meas["diagnostics"] == []

    dims = meas["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0
    assert 190.0 <= dims["length_mm"] <= 240.0
    assert 220.0 <= dims["height_mm"] <= 260.0


def test_case_5_clearly_isolated_head_short_span_validated():
    """
    Case 5: Clearly isolated head that bypasses the neck search is validated according to policy.
    Clean synthetic head (span ~240 mm <= 300 mm) has no neck; must produce 'bypassed_short_span',
    status 'validated', and volume_m3 not None.
    """
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=4000)
    target = HeadTarget()

    seg = target.segment(pcd)
    meta = target.last_segmentation_result

    assert meta["outcome"] == "bypassed_short_span"
    assert meta["cut_applied"] is False
    assert meta["longitudinal_span_m"] <= 0.300

    meas = target.measure(seg)
    assert meas["status"] == "validated"
    assert meas["oriented_bbox"]["volume_m3"] is not None
    assert meas["oriented_bbox"]["volume_m3"] > 0.0
    assert meas["diagnostics"] == []

    dims = meas["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0
    assert 190.0 <= dims["length_mm"] <= 240.0
    assert 225.0 <= dims["height_mm"] <= 255.0


def test_case_6_repeated_calls_do_not_reuse_stale_metadata():
    """
    Case 6: Repeated calls on the same HeadTarget instance do not reuse stale segmentation metadata.
    """
    target = HeadTarget()

    # Step 1: Run on head + shoulders (produces constriction_cut)
    pcd_cut = create_head_with_neck_shoulders(n_pts=8000)
    seg_1 = target.segment(pcd_cut)
    assert target.last_segmentation_result["outcome"] == "constriction_cut"
    meas_1 = target.measure(seg_1)
    assert meas_1["status"] == "validated"

    # Step 2: Next, call segment() on Problem 11A head+neck (should overwrite metadata)
    pcd_neck = create_head_plus_neck_short_span(seed=42)
    seg_2 = target.segment(pcd_neck)
    assert target.last_segmentation_result["outcome"] == "bypassed_short_span"
    assert target.last_segmentation_result["cut_applied"] is False

    meas_2 = target.measure(seg_2)
    assert meas_2["status"] == "unverified"
    assert meas_2["oriented_bbox"]["volume_m3"] is None

    # Step 3: Call measure() directly on an untracked unsegmented cloud
    # Must NOT reuse metadata from Step 1 or Step 2
    meas_untracked = target.measure(pcd_neck)
    assert meas_untracked["status"] == "unverified"
    assert meas_untracked["oriented_bbox"]["volume_m3"] is None


def test_case_7_empty_sparse_invalid_input_metadata():
    """
    Case 7: Empty, sparse, or invalid input does not produce false success metadata.
    """
    target = HeadTarget()

    # Empty cloud
    pcd_empty = o3d.geometry.PointCloud()
    seg_empty = target.segment(pcd_empty)
    meta_empty = target.last_segmentation_result
    assert meta_empty["outcome"] == "insufficient_input"
    assert meta_empty["cut_applied"] is False
    assert meta_empty["retained_point_count"] == 0
    meas_empty = target.measure(seg_empty)
    assert meas_empty["status"] == "unverified"
    assert meas_empty["volume_m3"] is None

    # Sparse cloud (< 50 points)
    np.random.seed(42)
    pts_sparse = np.random.randn(30, 3) * 0.1
    pcd_sparse = o3d.geometry.PointCloud()
    pcd_sparse.points = o3d.utility.Vector3dVector(pts_sparse)
    seg_sparse = target.segment(pcd_sparse)
    meta_sparse = target.last_segmentation_result
    assert meta_sparse["outcome"] == "insufficient_input"
    assert meta_sparse["cut_applied"] is False
    meas_sparse = target.measure(seg_sparse)
    assert meas_sparse["status"] == "unverified"
    assert meas_sparse["volume_m3"] is None

    # Non-finite coordinates (NaN/Inf)
    pts_nan = np.array([[np.nan, np.nan, np.nan], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pcd_nan = o3d.geometry.PointCloud()
    pcd_nan.points = o3d.utility.Vector3dVector(pts_nan)
    seg_nan = target.segment(pcd_nan)
    meta_nan = target.last_segmentation_result
    assert meta_nan["outcome"] == "insufficient_input"
    assert meta_nan["cut_applied"] is False
    meas_nan = target.measure(seg_nan)
    assert meas_nan["status"] == "unverified"
    assert meas_nan["volume_m3"] is None
