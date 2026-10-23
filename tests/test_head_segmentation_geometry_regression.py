"""
Regression test suite for Head Cross-Section Geometry and Orientation Handling.

Covers Problem 9B requirements:
  1. Exactly upright subject: transverse areas must not collapse to zero when v_long is aligned with Y.
  2. Tilted subject: transverse area calculation remains valid under non-zero pitch and roll.
  3. Compound rotation: transverse spans are based on the orthonormal plane, not fixed camera columns.
  4. Orientation inversion: reversing the candidate axis must not cause the algorithm to retain torso and discard cranium.
  5. No valid constriction: method must not make an arbitrary cut; failure falls back safely.
  6. Partial head scan: measurement remains unverified when boundary support evidence is insufficient.
  7. Orthonormal basis mathematical properties: unit length, mutual orthogonality, and perpendicularity to u.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from targets.head import (
    HeadTarget,
    _compute_slice_bounding_areas,
    _orthonormal_transverse_basis,
)
from test_head_synthetic import create_head_with_neck_shoulders, create_synthetic_head


def test_orthonormal_transverse_basis_properties():
    """Verify that _orthonormal_transverse_basis produces unit, mutually orthogonal vectors perpendicular to u."""
    test_vectors = [
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.57735, 0.57735, 0.57735]),
        np.array([0.1, -0.98, 0.17]),
        np.array([0.95, 0.05, 0.31]),
    ]

    for v in test_vectors:
        u = v / np.linalg.norm(v)
        e1, e2 = _orthonormal_transverse_basis(u)

        # Unit length
        assert np.isclose(np.linalg.norm(e1), 1.0, atol=1e-7), f"e1 not unit length for u={u}"
        assert np.isclose(np.linalg.norm(e2), 1.0, atol=1e-7), f"e2 not unit length for u={u}"

        # Perpendicular to u
        assert np.isclose(float(np.dot(e1, u)), 0.0, atol=1e-7), f"e1 not orthogonal to u={u}"
        assert np.isclose(float(np.dot(e2, u)), 0.0, atol=1e-7), f"e2 not orthogonal to u={u}"

        # Mutually orthogonal
        assert np.isclose(float(np.dot(e1, e2)), 0.0, atol=1e-7), f"e1 and e2 not mutually orthogonal for u={u}"


def test_exact_upright_subject_non_collapsing_areas():
    """
    Requirement 1: Exactly upright subject.
    Transverse areas along the slices must not collapse to zero when v_long aligns with camera Y.
    """
    np.random.seed(42)
    pcd = create_head_with_neck_shoulders(n_pts=8000)
    pts = np.asarray(pcd.points)
    C = np.mean(pts, axis=0)
    Pc = pts - C

    # Force exact upright longitudinal axis
    u_exact = np.array([0.0, 1.0, 0.0])
    e1, e2 = _orthonormal_transverse_basis(u_exact)

    # For u=[0, 1, 0], basis must span X and Z
    assert np.isclose(abs(float(e1[1])), 0.0, atol=1e-7), "e1 has non-zero Y component on upright axis"
    assert np.isclose(abs(float(e2[1])), 0.0, atol=1e-7), "e2 has non-zero Y component on upright axis"

    proj = Pc @ u_exact
    bins = np.linspace(np.min(proj), np.max(proj), 25)
    areas, centers = _compute_slice_bounding_areas(Pc, proj, bins, e1, e2)

    non_zero_areas = [a for a in areas if a > 0.0]
    assert len(non_zero_areas) >= 15, f"Too few non-zero area bins ({len(non_zero_areas)} < 15)"
    assert max(areas) > 0.05, f"Maximum area ({max(areas):.4f}) is unexpectedly small; expected > 0.05 m^2 for shoulders"

    # End-to-end segmentation and measurement
    target = HeadTarget()
    seg = target.segment(pcd)
    assert len(seg.points) < len(pcd.points), "Segmented cloud should have excluded neck and shoulders"

    res = target.measure(seg)
    assert res["status"] == "validated", f"Measurement failed on upright head: {res.get('diagnostics')}"
    dims = res["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0, f"Breadth out of bounds: {dims['breadth_mm']}"
    assert 190.0 <= dims["length_mm"] <= 240.0, f"Length out of bounds: {dims['length_mm']}"
    assert 220.0 <= dims["height_mm"] <= 260.0, f"Height out of bounds: {dims['height_mm']}"


def test_tilted_subject_cross_sectional_validity():
    """
    Requirement 2: Tilted subject.
    Area calculation remains valid when the longitudinal axis is tilted by 15 deg pitch and 10 deg roll.
    """
    np.random.seed(42)
    pcd = create_head_with_neck_shoulders(n_pts=8000)
    R_tilt = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(15.0), 0.0, np.radians(10.0)))
    pcd.rotate(R_tilt, center=(0.0, 0.0, 0.0))

    target = HeadTarget()
    seg = target.segment(pcd)

    # Verify head isolation
    seg_pts = np.asarray(seg.points)
    assert 1500 <= len(seg_pts) <= 3500, f"Unexpected segmented point count: {len(seg_pts)}"

    res = target.measure(seg)
    assert res["status"] == "validated", f"Measurement failed on tilted head: {res.get('diagnostics')}"
    dims = res["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0
    assert 190.0 <= dims["length_mm"] <= 240.0
    assert 220.0 <= dims["height_mm"] <= 260.0


def test_compound_rotation_transverse_plane():
    """
    Requirement 3: Compound rotation.
    Transverse spans are based on the orthonormal plane, not fixed camera Cartesian columns.
    """
    np.random.seed(42)
    pcd = create_head_with_neck_shoulders(n_pts=8000)
    R_compound = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(15.0), np.radians(45.0), 0.0))
    pcd.rotate(R_compound, center=(0.0, 0.0, 0.0))

    target = HeadTarget()
    seg = target.segment(pcd)
    res = target.measure(seg)

    assert res["status"] == "validated", f"Measurement failed on compound rotation: {res.get('diagnostics')}"
    dims = res["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0
    assert 190.0 <= dims["length_mm"] <= 240.0
    assert 220.0 <= dims["height_mm"] <= 260.0


def test_orientation_inversion_preserves_cranium():
    """
    Requirement 4: Orientation inversion.
    Reversing the axis must not cause the algorithm to retain the torso and discard the cranium.
    """
    np.random.seed(42)
    target = HeadTarget()

    # Original orientation (head at +Y, shoulders at -Y)
    pcd_normal = create_head_with_neck_shoulders(n_pts=8000)
    seg_normal = target.segment(pcd_normal)
    pts_normal = np.asarray(seg_normal.points)

    # Inverted orientation (rotated 180 deg around X: head at -Y, shoulders at +Y)
    R_flip = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0.0, 0.0))
    pcd_flipped = create_head_with_neck_shoulders(n_pts=8000)
    pcd_flipped.rotate(R_flip, center=(0.0, 0.0, 0.0))
    seg_flipped = target.segment(pcd_flipped)
    pts_flipped = np.asarray(seg_flipped.points)

    # In both cases, the segmented cloud must isolate the cranium (roughly ~2400 points)
    assert abs(len(pts_normal) - len(pts_flipped)) < 100, (
        f"Point count mismatch between normal ({len(pts_normal)}) and flipped ({len(pts_flipped)})"
    )

    # Check that flipped segmentation did not keep the torso (which would have > 4000 points)
    assert len(pts_flipped) < 3500, f"Flipped segmentation retained torso ({len(pts_flipped)} points)"

    # Verify that flipped measurement succeeds
    res_flipped = target.measure(seg_flipped)
    assert res_flipped["status"] == "validated", f"Flipped measurement failed: {res_flipped.get('diagnostics')}"
    dims = res_flipped["cranial_envelope_3d"]
    assert 150.0 <= dims["breadth_mm"] <= 170.0
    assert 190.0 <= dims["length_mm"] <= 240.0
    assert 220.0 <= dims["height_mm"] <= 260.0


def test_no_valid_constriction_safe_fallback():
    """
    Requirement 5: No valid constriction.
    The method must not make an arbitrary anatomical cut; failure falls back safely.
    """
    np.random.seed(42)
    # Create a uniform cylinder of radius 8 cm, height 40 cm (aspect ratio 2.5:1, length 40 cm > 30 cm)
    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.08, height=0.40)
    pcd_cyl = mesh_cyl.sample_points_poisson_disk(number_of_points=3000)

    target = HeadTarget()
    seg = target.segment(pcd_cyl)

    # Because a uniform cylinder has no neck constriction, segmentation must return unsegmented cloud
    assert len(seg.points) == len(pcd_cyl.points), (
        f"Segmented cylinder unexpectedly cut points ({len(seg.points)} vs {len(pcd_cyl.points)})"
    )

    # Downstream measurement must mark cylinder as unverified
    res = target.measure(seg)
    assert res["status"] == "unverified", f"Cylinder should be unverified, got {res['status']}"


def test_partial_head_scan_remains_unverified():
    """
    Requirement 6: Partial head scan.
    Measurement must remain unverified when boundary support evidence is insufficient.
    """
    np.random.seed(42)
    pcd_full = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=4000)
    pts = np.asarray(pcd_full.points)

    # Truncate posterior skull (Z < 0)
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(pts[pts[:, 2] > 0.0])

    target = HeadTarget()
    seg = target.segment(pcd_partial)
    res = target.measure(seg)

    assert res["status"] == "unverified", f"Partial scan should be unverified, got {res['status']}"
    assert res["oriented_bbox"]["volume_m3"] is None, "Partial scan volume must be None"
    diags = res.get("diagnostics", [])
    assert len(diags) > 0, "Expected failure diagnostics for partial head scan"
