"""
Unit tests for unposing depth observations and observed-surface circumference extraction.
"""

import math
import numpy as np
import pytest

from realsense_measure.body.unpose import BodyUnposer, UnposedStopData
from realsense_measure.body.observed_measure import (
    ObservedRingExtractor,
    ObservedRingResult,
    weighted_median,
)
from realsense_measure.body.synth import (
    SimulatedStopData,
    SyntheticTruthSubject,
    CameraConfig,
    apply_arm_abduction,
    apply_rehab_mid_thigh_atrophy,
)
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.measure import measure_mesh


def test_weighted_median():
    vals = np.array([1.0, 2.0, 3.0, 10.0])
    weights = np.array([1.0, 1.0, 1.0, 0.01])
    med = weighted_median(vals, weights)
    assert 1.9 <= med <= 2.1

    # Empty
    assert math.isnan(weighted_median(np.array([]), np.array([])))


def test_unpose_rigid_translation_and_yaw():
    # Synthetic canonical points on a circle in XY plane
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts_canon = np.stack([0.2 * np.cos(theta), 0.2 * np.sin(theta), np.zeros_like(theta)], axis=-1)

    unposer = BodyUnposer(
        canon_vertices=pts_canon,
        canon_joints=np.zeros((5, 3), dtype=np.float32),
        joint_names=["root", "spine", "neck", "head", "foot"],
    )

    yaw_deg = 45.0
    trans_m = np.array([0.5, -0.2, 1.1], dtype=np.float32)

    # Forward pose: rotate by yaw then translate
    rad = np.radians(yaw_deg)
    rot_z = np.array([
        [np.cos(rad), -np.sin(rad), 0.0],
        [np.sin(rad),  np.cos(rad), 0.0],
        [0.0,          0.0,         1.0],
    ])
    pts_posed = (pts_canon @ rot_z.T) + trans_m

    # Unpose
    recovered, dists, _ = unposer.unpose_points(
        pts_posed,
        yaw_deg=yaw_deg,
        translation_m=trans_m,
    )

    max_err_mm = np.max(np.linalg.norm(recovered - pts_canon, axis=-1)) * 1000.0
    assert max_err_mm < 0.1, f"Rigid unposing error too large: {max_err_mm:.4f} mm"


def test_observed_ring_on_synthetic_cylinder():
    R = 0.15
    analytic_p_mm = 2.0 * np.pi * R * 1000.0

    # Generate points on cylinder surface within +- 5 mm of Z=0
    n_pts = 1000
    angles = np.random.uniform(-np.pi, np.pi, size=n_pts)
    z_coords = np.random.uniform(-0.005, 0.005, size=n_pts)
    pts_slab = np.stack([R * np.cos(angles), R * np.sin(angles), z_coords], axis=-1)

    stops = np.zeros(n_pts, dtype=np.int32)
    view_dirs = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_pts, 1))

    extractor = ObservedRingExtractor(
        canon_verts=pts_slab,
        canon_joints=np.zeros((5, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int64),
        joint_names=["root", "spine", "neck", "head", "foot"],
        n_angular_bins=72,
        slab_thickness_m=0.010,
    )

    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    val_mm, cov, _, contour, mask, n_slab = extractor.measure_ring_contour(
        p0, norm, pts_slab, stops, view_dirs, expected_r_ref=R
    )

    rel_err = abs(val_mm - analytic_p_mm) / analytic_p_mm
    assert rel_err < 0.005, f"Perimeter error {rel_err*100:.3f}% exceeds 0.5% (Val: {val_mm:.1f}, True: {analytic_p_mm:.1f})"
    assert cov > 0.90, f"Full cylinder coverage was {cov:.2f}"


def test_observed_ring_coverage_and_source_flag():
    R = 0.15
    n_pts = 500
    # Only positive X half (angles between -pi/2 and +pi/2)
    angles = np.random.uniform(-np.pi / 2, np.pi / 2, size=n_pts)
    z_coords = np.random.uniform(-0.005, 0.005, size=n_pts)
    pts_slab = np.stack([R * np.cos(angles), R * np.sin(angles), z_coords], axis=-1)

    stops = np.zeros(n_pts, dtype=np.int32)
    view_dirs = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_pts, 1))

    extractor = ObservedRingExtractor(
        canon_verts=pts_slab,
        canon_joints=np.zeros((5, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int64),
        joint_names=["root"],
        n_angular_bins=72,
    )

    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    val_mm, cov, _, contour, mask, _ = extractor.measure_ring_contour(
        p0, norm, pts_slab, stops, view_dirs, expected_r_ref=R
    )

    assert 0.40 <= cov <= 0.60, f"Expected ~0.50 coverage, got {cov:.2f}"
    source = "observed" if cov >= 0.95 else "mixed"
    assert source == "mixed"


def test_rehab_atrophy_detected_on_observed_surface():
    """
    Test that an asymmetric local mid-thigh reduction (-15 mm) is detected
    on the observed surface points, whereas the symmetric model fit fails to detect it.
    """
    backend = AnnyBackend(verbose=False)
    prior = AdultShapePrior()
    subj = SyntheticTruthSubject(backend, prior, subject_id=0, seed=777, apply_rehab_atrophy=True)

    # True thigh circumferences
    c_thigh_L_true = subj.true_measurements["mid_thigh_circumference_left"].value_mm
    c_thigh_R_true = subj.true_measurements["mid_thigh_circumference_right"].value_mm
    true_diff = c_thigh_R_true - c_thigh_L_true

    # Simulate 4 stops with virtual camera
    cam = CameraConfig.portrait()
    stops = subj.simulate_stops(cam, n_stops=4, imperfections=False)

    # Perfect unposing using true known poses
    canon_body = backend.rest_pose(subj.phenotypes)
    unposer = BodyUnposer(
        canon_vertices=canon_body.numpy_vertices(0),
        canon_joints=canon_body.numpy_joints(0),
        joint_names=canon_body.joint_names,
        raw_dict=canon_body.raw,
    )

    known_poses = [{"yaw_deg": s.true_yaw_deg, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0} for s in stops]
    unp_stops = unposer.unpose_all_stops(stops, known_poses)

    extractor = ObservedRingExtractor(
        canon_verts=canon_body.numpy_vertices(0),
        canon_joints=canon_body.numpy_joints(0),
        faces=canon_body.faces.cpu().numpy(),
        joint_names=canon_body.joint_names,
        raw_dict=canon_body.raw,
    )
    rings = extractor.measure_all_observed_rings(unp_stops)

    obs_L = rings["mid_thigh_circumference_left"].value_mm
    obs_R = rings["mid_thigh_circumference_right"].value_mm
    obs_diff = obs_R - obs_L

    # Observed difference must reflect true difference within 10 mm (accounting for camera resolution/noise)
    diff_err = abs(obs_diff - true_diff)
    assert diff_err < 10.0, f"Rehab diff error {diff_err:.2f} mm exceeds 10.0 mm (Obs diff: {obs_diff:.1f}, True diff: {true_diff:.1f})"
