"""
test_head_segmentation_mutation_guard.py - Regression test suite for Problem 11E.

Validates that HeadTarget.measure() detects point-cloud mutation after segment(),
invalidates prior segmentation metadata, fails closed with status 'unverified',
sets both volume_m3 and oriented_bbox.volume_m3 to None, and provides clear diagnostics.

Covers:
  1. An unchanged segmented cloud retains existing behavior (validated, non-null volume).
  2. Replacing seg.points with a different cloud of the same point count invalidates metadata.
  3. Appending points invalidates metadata.
  4. Removing points invalidates metadata.
  5. Applying an in-place rigid transformation (translation or rotation) invalidates metadata.
  6. Restoring a previously attached neck cannot use stale constriction_cut metadata to validate the cloud or return non-null volume.
  7. Reordering points invalidates metadata.
  8. Injecting invalid or non-finite coordinates (NaN/Inf) safely invalidates metadata without error.
  9. Failed or insufficient segmentation remains fail-closed.
  10. Repeated measurement without mutation remains safe and deterministic.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from targets.head import HeadTarget, _compute_point_cloud_fingerprint
from test_head_synthetic import create_head_with_neck_shoulders, create_synthetic_head


def test_fingerprint_helper_properties():
    """Unit test for fingerprint helper: deterministic, sensitive to order, coordinates, shape."""
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=200)
    fp1 = _compute_point_cloud_fingerprint(pcd)
    fp2 = _compute_point_cloud_fingerprint(pcd)
    assert fp1 is not None and len(fp1) == 64
    assert fp1 == fp2

    # In-place translation
    pcd.translate(np.array([0.001, 0.0, 0.0]))
    fp3 = _compute_point_cloud_fingerprint(pcd)
    assert fp3 != fp1

    # Empty cloud
    assert _compute_point_cloud_fingerprint(o3d.geometry.PointCloud()) is None
    assert _compute_point_cloud_fingerprint(None) is None

    # Non-finite coordinates
    nan_pcd = o3d.geometry.PointCloud()
    nan_pcd.points = o3d.utility.Vector3dVector([[np.nan, 0.0, 0.0]])
    assert _compute_point_cloud_fingerprint(nan_pcd) is None


def test_unchanged_segmented_cloud_retains_existing_behavior():
    """Requirement: An unchanged segmented cloud retains existing behavior."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    meta = target.last_segmentation_result
    assert meta["outcome"] == "constriction_cut"
    assert meta["cut_applied"] is True
    assert meta["fingerprint"] is not None

    meas = target.measure(seg)
    assert meas["status"] == "validated"
    assert meas["volume_m3"] is not None and meas["volume_m3"] > 0.0
    assert meas["oriented_bbox"]["volume_m3"] is not None and meas["oriented_bbox"]["volume_m3"] > 0.0
    assert meas["segmentation"]["outcome"] == "constriction_cut"
    assert meas["segmentation"]["cut_applied"] is True
    assert meas["diagnostics"] == []


def test_replacing_points_same_count_invalidates_metadata():
    """Requirement: Replacing seg.points with a different cloud of the same point count invalidates metadata."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    n_pts = len(seg.points)
    assert n_pts >= 150

    # Replace with same number of points but different coordinates
    rng = np.random.RandomState(42)
    alt_pts = rng.normal(loc=0.0, scale=0.1, size=(n_pts, 3))
    seg.points = o3d.utility.Vector3dVector(alt_pts)

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["cut_applied"] is False
    assert meas["segmentation"]["fingerprint_mismatch"] is True
    assert any("mutated_after_segmentation" in d for d in meas.get("diagnostics", []))


def test_appending_points_invalidates_metadata():
    """Requirement: Appending points invalidates metadata."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    pts = np.asarray(seg.points)
    extra_pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    seg.points = o3d.utility.Vector3dVector(np.vstack([pts, extra_pts]))

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["fingerprint_mismatch"] is True
    assert any("mutated_after_segmentation" in d for d in meas.get("diagnostics", []))


def test_removing_points_invalidates_metadata():
    """Requirement: Removing points invalidates metadata."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    pts = np.asarray(seg.points)
    seg.points = o3d.utility.Vector3dVector(pts[:-5])

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["fingerprint_mismatch"] is True
    assert any("mutated_after_segmentation" in d for d in meas.get("diagnostics", []))


def test_in_place_rigid_transformation_invalidates_metadata():
    """Requirement: Applying an in-place rigid transformation invalidates metadata."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)

    # In-place translation
    T = np.eye(4)
    T[0, 3] = 0.05  # 50 mm shift
    seg.transform(T)

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["fingerprint_mismatch"] is True
    assert any("mutated_after_segmentation" in d for d in meas.get("diagnostics", []))


def test_restoring_attached_neck_fails_closed():
    """
    Requirement: A mutation that restores a previously attached neck cannot use
    stale constriction_cut metadata to validate the cloud or return a non-null volume.
    """
    target = HeadTarget()
    pcd_neck = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd_neck)
    assert target.last_segmentation_result["outcome"] == "constriction_cut"

    # Restore the uncut unsegmented points (with neck/shoulders) into seg
    seg.points = o3d.utility.Vector3dVector(np.asarray(pcd_neck.points))

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["fingerprint_mismatch"] is True
    assert any("mutated_after_segmentation" in d for d in meas.get("diagnostics", []))


def test_reordering_points_invalidates_metadata():
    """Requirement: Reordering points is treated conservatively as invalidation."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    pts = np.asarray(seg.points)
    # Reverse order of points
    seg.points = o3d.utility.Vector3dVector(pts[::-1])

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"
    assert meas["segmentation"]["fingerprint_mismatch"] is True


def test_injecting_non_finite_coordinates_fails_safely():
    """Requirement: Handle invalid or non-finite point arrays safely."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)
    pts = np.asarray(seg.points).copy()
    pts[0] = [np.nan, np.inf, 0.0]
    seg.points = o3d.utility.Vector3dVector(pts)

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "mutated_after_segmentation"


def test_failed_or_insufficient_segmentation_remains_fail_closed():
    """Requirement: Failed or insufficient segmentation remains fail-closed."""
    target = HeadTarget()
    empty_pcd = o3d.geometry.PointCloud()

    seg = target.segment(empty_pcd)
    assert target.last_segmentation_result["outcome"] == "insufficient_input"
    assert target.last_segmentation_result["fingerprint"] is None

    meas = target.measure(seg)
    assert meas["status"] == "unverified"
    assert meas["volume_m3"] is None
    assert meas["oriented_bbox"]["volume_m3"] is None
    assert meas["segmentation"]["outcome"] == "insufficient_input"


def test_repeated_measurement_without_mutation_safe():
    """Requirement: Repeated measurement without mutation remains safe and deterministic."""
    target = HeadTarget()
    pcd = create_head_with_neck_shoulders(n_pts=8000)

    seg = target.segment(pcd)

    meas1 = target.measure(seg)
    meas2 = target.measure(seg)

    assert meas1["status"] == "validated"
    assert meas2["status"] == "validated"
    assert meas1["volume_m3"] is not None and meas1["volume_m3"] > 0.0
    assert meas1["volume_m3"] == meas2["volume_m3"]
    assert meas1["oriented_bbox"]["volume_m3"] == meas2["oriented_bbox"]["volume_m3"]
    assert meas1["segmentation"]["outcome"] == "constriction_cut"
    assert meas2["segmentation"]["outcome"] == "constriction_cut"
