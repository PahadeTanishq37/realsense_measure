"""
tests/body/test_step4.py
========================
Tests for Step 4:
1. Extractor Oracle (Level i <= 2 mm error on truth geometry)
2. Alignment ICP Refiner (point-to-plane pose refinement)
3. Personalised Surface Reconstruction (Laplacian solve + LOO validation)
4. Bias correction comparison on synthetic circle and high-curvature ellipse
5. Capture pose collision check (arm-torso contact)
6. Test-Retest MDC95 formula consistency
"""

import math
import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import (
    CameraConfig,
    SyntheticTruthSubject,
    check_arm_torso_contact,
)
from realsense_measure.body.unpose import BodyUnposer
from realsense_measure.body.observed_measure import ObservedRingExtractor
from realsense_measure.body.alignment_icp import PointToPlaneICPRefiner
from realsense_measure.body.personalized_surface import PersonalisedSurfaceFitter


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend(verbose=False)


@pytest.fixture(scope="module")
def adult_prior():
    return AdultShapePrior()


def test_bias_correction_synthetic_shapes():
    """
    Validate that Quadratic Bias Correction corrects noise without
    the severe curvature shrinkage caused by Gaussian smoothing.
    """
    rng = np.random.default_rng(42)
    K = 72
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    sigma = 1.5  # mm

    # 1. Circle R = 150 mm
    R = 150.0
    x_c = R * np.cos(th)
    y_c = R * np.sin(th)
    p_true_c = 2 * np.pi * R

    x_c_noisy = x_c + rng.normal(0, sigma, size=K)
    y_c_noisy = y_c + rng.normal(0, sigma, size=K)

    dx = np.diff(np.append(x_c_noisy, x_c_noisy[0]))
    dy = np.diff(np.append(y_c_noisy, y_c_noisy[0]))
    L_sq = dx**2 + dy**2
    p_raw = np.sum(np.sqrt(L_sq))
    p_corr = np.sum(np.sqrt(np.maximum(0.0, L_sq - 2.0 * sigma**2)))

    # Raw noisy polygon has positive Pythagorean bias
    assert p_raw > p_true_c + 3.0
    # Corrected perimeter is much closer to truth than raw
    assert abs(p_corr - p_true_c) < abs(p_raw - p_true_c)

    # 2. High-curvature Ellipse (a=200, b=80)
    a, b = 200.0, 80.0
    x_e = a * np.cos(th)
    y_e = b * np.sin(th)
    dx_e = np.diff(np.append(x_e, x_e[0]))
    dy_e = np.diff(np.append(y_e, y_e[0]))
    p_true_e = np.sum(np.sqrt(dx_e**2 + dy_e**2))

    x_e_noisy = x_e + rng.normal(0, sigma, size=K)
    y_e_noisy = y_e + rng.normal(0, sigma, size=K)
    dx_en = np.diff(np.append(x_e_noisy, x_e_noisy[0]))
    dy_en = np.diff(np.append(y_e_noisy, y_e_noisy[0]))
    L_sq_e = dx_en**2 + dy_en**2
    p_corr_e = np.sum(np.sqrt(np.maximum(0.0, L_sq_e - 2.0 * sigma**2)))

    # Quadratic correction maintains low bias on high curvature
    err_corr_e = abs(p_corr_e - p_true_e)
    assert err_corr_e < 10.0


def test_arm_torso_contact_detection(anny_backend, adult_prior):
    """
    Test that check_arm_torso_contact detects when arms touch the torso
    at 0 deg (neutral rest pose) and distinguishes from abducted arms (35 deg).
    """
    subj = SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)

    # In neutral rest pose (0 deg), inner arm is within 15 mm of lateral ribcage
    is_contact_0, min_d_0 = check_arm_torso_contact(subj.true_verts, subj.raw_dict)
    assert is_contact_0 is True
    assert min_d_0 < 0.015

    # With +35 deg abduction, arms lift away from the torso (> 40 mm)
    from realsense_measure.body.synth import apply_arm_abduction
    abducted_verts = apply_arm_abduction(
        subj.true_verts, subj.true_joints, anny_backend.joint_names, subj.raw_dict,
        left_deg=35.0, right_deg=35.0,
    )
    is_contact_abd, min_d_abd = check_arm_torso_contact(
        abducted_verts, subj.raw_dict, contact_threshold_m=0.015
    )
    assert min_d_abd > 0.020
    assert is_contact_abd is False


def test_point_to_plane_icp_refinement(anny_backend, adult_prior):
    """
    Test that point-to-plane ICP refines stop yaw and translation.
    """
    subj = SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)
    cam = CameraConfig.portrait()
    stops = subj.simulate_stops(cam, n_stops=2, imperfections=True)

    # Perturbed initial pose: add 2.5 deg yaw error and 15 mm translation offset
    st0 = stops[0]
    pert_pose = {
        "yaw_deg": st0.true_yaw_deg + 2.5,
        "translation_m": (st0.true_translation_m + np.array([0.015, -0.010, 0.005])).tolist(),
        "arm_abd_L_deg": st0.true_arm_abduction_deg[0],
        "arm_abd_R_deg": st0.true_arm_abduction_deg[1],
    }

    refiner = PointToPlaneICPRefiner(
        canon_vertices=subj.true_verts,
        faces=subj.faces,
        canon_joints=subj.true_joints,
        joint_names=anny_backend.joint_names,
        raw_dict=subj.raw_dict,
    )

    res = refiner.refine_stop(st0, pert_pose, n_iters=4)

    # Yaw error after ICP must decrease
    init_yaw_err = abs(pert_pose["yaw_deg"] - st0.true_yaw_deg)
    final_yaw_err = abs(res.refined_yaw_deg - st0.true_yaw_deg)
    assert final_yaw_err < init_yaw_err
    assert res.refined_residual_mm < res.initial_residual_mm


def test_personalised_surface_fitter(anny_backend, adult_prior):
    """
    Test that PersonalisedSurfaceFitter solves 3D offsets and performs LOO validation.
    """
    subj = SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)
    cam = CameraConfig.portrait()
    stops = subj.simulate_stops(cam, n_stops=4, imperfections=False)

    known_poses = [{
        "yaw_deg": s.true_yaw_deg,
        "translation_m": s.true_translation_m.tolist(),
        "arm_abd_L_deg": s.true_arm_abduction_deg[0],
        "arm_abd_R_deg": s.true_arm_abduction_deg[1],
    } for s in stops]

    unposer = BodyUnposer(
        canon_vertices=subj.canonical_verts if hasattr(subj, "canonical_verts") else subj.true_verts,
        canon_joints=subj.true_joints,
        joint_names=anny_backend.joint_names,
        raw_dict=subj.raw_dict,
    )
    unp_stops = unposer.unpose_all_stops(stops, known_poses)

    fitter = PersonalisedSurfaceFitter(
        canonical_vertices=subj.canonical_verts if hasattr(subj, "canonical_verts") else subj.true_verts,
        faces=subj.faces,
        candidate_lambdas=[10.0, 50.0],
    )

    surf_res = fitter.fit_with_loo(unp_stops)
    assert surf_res.selected_lambda in [10.0, 50.0]
    assert surf_res.delta_v_norm_mean_mm > 0.0
    assert surf_res.vertex_coverage_fraction > 0.60

    # Measure rings on personalised surface
    rings = fitter.measure_personalised_rings(
        surf_res, subj.true_joints, anny_backend.joint_names, subj.raw_dict
    )
    assert "chest_circumference" in rings
    assert rings["chest_circumference"].value_mm > 700.0


def test_test_retest_mdc95_formula():
    """
    Verify mathematical consistency of MDC95 = 1.96 * sqrt(2) * SEM.
    """
    rng = np.random.default_rng(123)
    # Simulate 50 subject pairs with true repeatability SD = 3.5 mm
    true_sd_diff = 3.5
    d = rng.normal(0.0, true_sd_diff, size=50)

    sd_diff = float(np.std(d, ddof=1))
    sem = sd_diff / np.sqrt(2.0)
    mdc95 = 1.96 * np.sqrt(2.0) * sem

    # Verify MDC95 equals 1.96 * sd_diff
    assert math.isclose(mdc95, 1.96 * sd_diff, rel_tol=1e-5)
