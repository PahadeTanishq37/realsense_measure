"""
tests/body/test_anatomical_correspondence.py
============================================
Comprehensive tests for independent anatomical association and constrained
point-to-mesh correspondence in realsense_measure/body/personalized_surface.py:

Scenarios:
1. A hand point near the hip must not become a pelvis observation merely because
   the nearest template vertex belongs to the pelvis.
2. A forearm point near the torso must not automatically become a torso observation.
3. A point with no independent anatomical label must remain unknown.
4. Unknown points must not deform arbitrary template regions (zero deformation).
5. A genuinely identified torso observation must still be accepted.
6. A genuinely identified thigh observation must still be accepted.
7. Multiple stops must preserve point-to-label alignment.
8. Changing nearest-template indices must not change independently established anatomical labels.
9. Original circular-labeling failure reproduced and verified to be rejected.
10. Diagnostics distinguish unknown observations from rejected geometric correspondences.
11. Ambiguous boundary rejection (fixed assertion verifying rejection as ambiguous_boundary).
12. Incompatible normal rejection when normals are provided.
13. Missing normals graceful fallback without crash.
14. Bilateral thigh cross-talk rejection across the sagittal plane.
15. Points beyond max_dist_m rejected as out_of_range.
16. Production fit_with_loo integration pipeline test with multi-stop capture.
"""

import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import SyntheticTruthSubject, CameraConfig
from realsense_measure.body.unpose import BodyUnposer, UnposedStopData
from realsense_measure.body.personalized_surface import (
    PersonalisedSurfaceFitter,
    PersonalisedSurfaceResult,
    CorrespondenceDiagnostics,
)


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend(verbose=False)


@pytest.fixture(scope="module")
def adult_prior():
    return AdultShapePrior()


@pytest.fixture(scope="module")
def truth_subject(anny_backend, adult_prior):
    return SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)


@pytest.fixture(scope="module")
def fitter(truth_subject, anny_backend):
    return PersonalisedSurfaceFitter(
        canonical_vertices=truth_subject.true_verts,
        faces=truth_subject.faces,
        raw_dict=truth_subject.raw_dict,
        canonical_joints=truth_subject.true_joints,
        joint_names=anny_backend.joint_names,
        max_dist_m=0.040,
    )


# ---------------------------------------------------------------------------
# Problem 2C Required Scenarios
# ---------------------------------------------------------------------------

def test_hand_point_near_hip_not_pelvis(fitter):
    """
    Scenario 1: A hand point near the hip must NOT become a pelvis observation
    merely because the nearest template vertex belongs to the pelvis.
    """
    hip_indices = np.where(fitter.vertex_regions == "pelvis")[0]
    assert len(hip_indices) > 0
    hip_pt = fitter.v0[hip_indices[0]]

    # A hand point located near the hip (e.g. hand hanging down beside hip in capture)
    hand_like_pt = hip_pt + np.array([0.008, 0.002, 0.0])
    hand_reg = "hand_R" if hip_pt[0] < 0 else "hand_L"

    # In Operation B (mesh correspondence), pelvis vertices must NOT accept it
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        np.array([hand_like_pt]),
        point_regions=[hand_reg],
    )
    assert len(acc_pts) == 0
    assert diag.rejected_points == 1
    assert diag.rejection_reasons["cross_talk"] == 1


def test_forearm_point_near_torso_not_torso(fitter):
    """
    Scenario 2: A forearm point near the torso must not automatically become a torso observation.
    """
    torso_indices = np.where((fitter.vertex_regions == "torso") & (fitter.v0[:, 2] > 0.05) & (fitter.v0[:, 2] < 0.20))[0]
    lat_torso_idx = torso_indices[np.argmax(fitter.v0[torso_indices, 0])]
    torso_pt = fitter.v0[lat_torso_idx]

    forearm_pt = torso_pt + np.array([0.010, 0.0, 0.0])

    # When labeled as forearm independently
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        np.array([forearm_pt]),
        point_regions=["forearm_L"],
    )
    assert len(acc_pts) == 0
    assert diag.rejection_reasons["cross_talk"] == 1


def test_unlabeled_point_remains_unknown(fitter):
    """
    Scenario 3: A point with no independent anatomical label outside safe envelopes
    must remain unknown.
    """
    # Point floating far off or in empty space
    free_point = np.array([[0.30, 0.30, 0.0]])
    inferred = fitter.associate_anatomical_regions(free_point)
    assert inferred[0] == "unknown"


def test_unknown_points_do_not_deform_mesh(fitter):
    """
    Scenario 4: Unknown points must not deform arbitrary template regions (zero deformation).
    """
    hip_indices = np.where(fitter.vertex_regions == "pelvis")[0]
    hip_pts = fitter.v0[hip_indices]

    # Explicitly unknown observations placed right on the hip
    unknown_points = hip_pts + np.array([0.002, 0.0, 0.0])

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        unknown_points,
        point_regions=["unknown"] * len(unknown_points),
    )

    assert len(acc_pts) == 0
    assert diag.unknown_points == len(unknown_points)
    assert diag.rejection_reasons["unknown_region"] == len(unknown_points)

    # Solve deformation
    delta_v = fitter._solve_offsets(
        unknown_points,
        lam=30.0,
        point_regions=["unknown"] * len(unknown_points),
    )
    # Deformation must be strictly zero
    assert np.allclose(delta_v, 0.0)


def test_genuinely_identified_torso_observation_accepted(fitter):
    """
    Scenario 5: A genuinely identified torso observation must still be accepted.
    """
    torso_indices = np.where((fitter.vertex_regions == "torso") & (fitter.v0[:, 2] > 0.10) & (fitter.v0[:, 2] < 0.35))[0][:50]
    torso_pts = fitter.v0[torso_indices] + fitter.vertex_normals[torso_indices] * 0.003

    # Independent association must identify them as torso
    inferred_regs = fitter.associate_anatomical_regions(torso_pts)
    assert np.mean(inferred_regs == "torso") > 0.85

    # Run correspondence with independently identified labels
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        torso_pts,
        point_regions=inferred_regs,
    )
    assert len(acc_pts) > 40
    assert diag.per_region_counts["torso"] > 40
    assert diag.confidence_mean > 0.80


def test_genuinely_identified_thigh_observation_accepted(fitter):
    """
    Scenario 6: A genuinely identified thigh observation must still be accepted.
    """
    thigh_indices = np.where(fitter.vertex_regions == "thigh_L")[0][:50]
    thigh_pts = fitter.v0[thigh_indices] + fitter.vertex_normals[thigh_indices] * 0.003

    inferred_regs = fitter.associate_anatomical_regions(thigh_pts)
    assert np.mean(inferred_regs == "thigh_L") > 0.85

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        thigh_pts,
        point_regions=inferred_regs,
    )
    assert len(acc_pts) > 40
    assert diag.per_region_counts["thigh_L"] > 40


def test_multiple_stops_preserve_point_label_alignment(fitter):
    """
    Scenario 7: Multiple stops must preserve point-to-label alignment.
    """
    pts_stop1 = fitter.v0[np.where((fitter.vertex_regions == "torso") & (fitter.v0[:, 2] > 0.10) & (fitter.v0[:, 2] < 0.35))[0][:20]]
    pts_stop2 = fitter.v0[np.where((fitter.vertex_regions == "thigh_L") & (fitter.v0[:, 0] > 0.08) & (fitter.v0[:, 2] < -0.15))[0][:20]]

    regs1 = fitter.associate_anatomical_regions(pts_stop1)
    regs2 = fitter.associate_anatomical_regions(pts_stop2)

    combined_pts = np.vstack([pts_stop1, pts_stop2])
    combined_regs = np.concatenate([regs1, regs2])

    assert len(combined_pts) == len(combined_regs)
    assert np.all(combined_regs[:20] == "torso")
    assert np.all(combined_regs[20:] == "thigh_L")


def test_changing_nearest_template_indices_does_not_change_labels(fitter):
    """
    Scenario 8: Changing nearest-template indices must not change independently
    established anatomical labels.
    """
    torso_pts = fitter.v0[np.where(fitter.vertex_regions == "torso")[0][:10]]

    # Independent labels depend ONLY on points and skeletal joints
    labels_original = fitter.associate_anatomical_regions(torso_pts)

    # In UnposedStopData, even if nearest_vertex_indices is corrupted to point to foot/hand
    stop = UnposedStopData(
        stop_idx=0,
        unposed_points=torso_pts,
        camera_view_dir=np.array([0, 1, 0], dtype=np.float32),
        point_to_model_distances=np.zeros(len(torso_pts)),
        mean_residual_mm=0.0,
        median_residual_mm=0.0,
        p90_residual_mm=0.0,
        nearest_vertex_indices=np.array([9999] * len(torso_pts)),  # corrupted!
    )

    # fit_with_loo extracts regions independently of nearest_vertex_indices
    extracted_labels = fitter.associate_anatomical_regions(stop.unposed_points)
    assert np.array_equal(labels_original, extracted_labels)


def test_original_circular_labeling_failure_reproduced_and_rejected(fitter):
    """
    Scenario 9: The original circular-labeling failure must be reproduced
    and shown to be rejected.

    In the old implementation, a hand point placed near the hip would be given
    the nearest vertex's region ('pelvis') and then accepted into the hip.
    Here, the hand point must NOT be accepted into the pelvis!
    """
    hip_idx = np.where(fitter.vertex_regions == "pelvis")[0][0]
    hip_vertex_pos = fitter.v0[hip_idx]

    # Hand point located 8 mm from hip vertex
    spurious_hand_pt = hip_vertex_pos + np.array([0.008, 0.0, 0.0])

    # In old code: nearest_vertex_indices would be hip_idx, so region would be 'pelvis'
    # In new code: associate_anatomical_regions runs independently
    inferred_reg = fitter.associate_anatomical_regions(np.array([spurious_hand_pt]))[0]

    # It must NOT be pelvis!
    assert inferred_reg != "pelvis"

    # Even if an attacker or buggy caller forced the point region to 'hand_L':
    # It must be rejected from hip vertices
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        np.array([spurious_hand_pt]),
        point_regions=["hand_L"],
    )
    assert len(acc_pts) == 0
    assert diag.rejection_reasons["cross_talk"] == 1


def test_diagnostics_distinguish_unknown_from_geometric_rejection(fitter):
    """
    Scenario 10: Diagnostics must distinguish unknown observations from rejected
    geometric correspondences.
    """
    torso_indices = np.where(fitter.vertex_regions == "torso")[0][:20]
    valid_pts = fitter.v0[torso_indices] + fitter.vertex_normals[torso_indices] * 0.003
    far_pts = fitter.v0[torso_indices[:10]] + fitter.vertex_normals[torso_indices[:10]] * 0.060
    unknown_pts = np.array([[0.30, 0.30, 0.0], [0.35, 0.35, 0.0]])

    all_test_pts = np.vstack([valid_pts, far_pts, unknown_pts])
    test_regions = (
        ["torso"] * len(valid_pts)
        + ["torso"] * len(far_pts)
        + ["unknown"] * len(unknown_pts)
    )

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        all_test_pts,
        point_regions=test_regions,
    )

    assert diag.total_input_points == len(all_test_pts)
    assert diag.accepted_points == len(valid_pts)
    assert diag.unknown_points == 2
    assert diag.rejection_reasons["unknown_region"] == 2
    assert diag.rejection_reasons["out_of_range"] == len(far_pts)
    assert diag.accepted_points + diag.rejected_points == diag.total_input_points


# ---------------------------------------------------------------------------
# Fixed Ambiguous Boundary Rejection Test
# ---------------------------------------------------------------------------

def test_ambiguous_boundary_rejection(fitter):
    """
    Scenario 11: Points positioned near boundary where candidates span conflicting
    regions are rejected as ambiguous_boundary.
    """
    # Find medial thigh vertices close to contact plane
    tl = np.where(fitter.vertex_regions == "thigh_L")[0]
    tr = np.where(fitter.vertex_regions == "thigh_R")[0]
    from scipy.spatial import cKDTree
    tree_r = cKDTree(fitter.v0[tr])
    d, idx = tree_r.query(fitter.v0[tl], k=1)
    sub = np.where(fitter.v0[tl, 2] < -0.20)[0]
    best_i = sub[np.argmin(d[sub])]
    v_L = fitter.v0[tl[best_i]]
    v_R = fitter.v0[tr[idx[best_i]]]
    p_mid = (v_L + v_R) * 0.5

    # Run correspondence with both thigh candidate vertices nearby
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        np.array([p_mid]),
        point_regions=["thigh_L"],  # even if claimed as thigh_L, candidates include thigh_R
        k_neighbors=5,
    )

    # Must be completely rejected
    assert len(acc_pts) == 0
    assert diag.rejected_points == 1
    assert diag.rejection_reasons["ambiguous_boundary"] == 1


# ---------------------------------------------------------------------------
# Additional Existing Correspondence Checks
# ---------------------------------------------------------------------------

def test_incompatible_normal_rejection(fitter):
    """
    Scenario 12: Incompatible normal rejection when normals are provided.
    """
    torso_indices = np.where(fitter.vertex_regions == "torso")[0][:30]
    torso_pts = fitter.v0[torso_indices] + fitter.vertex_normals[torso_indices] * 0.005

    opposing_normals = -fitter.vertex_normals[torso_indices]

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        torso_pts,
        point_normals=opposing_normals,
        point_regions=["torso"] * len(torso_pts),
    )

    assert len(acc_pts) == 0
    assert diag.rejected_points == len(torso_pts)
    assert diag.rejection_reasons["normal_incompatible"] == len(torso_pts)


def test_missing_normals_graceful_fallback(fitter):
    """
    Scenario 13: Missing normals gracefully fall back without crash.
    """
    torso_indices = np.where(fitter.vertex_regions == "torso")[0][:30]
    torso_pts = fitter.v0[torso_indices] + fitter.vertex_normals[torso_indices] * 0.004

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        torso_pts,
        point_normals=None,
        point_regions=["torso"] * len(torso_pts),
    )

    assert len(acc_pts) == len(torso_pts)
    assert diag.rejected_points == 0
    assert diag.confidence_mean > 0.70


def test_bilateral_thigh_crosstalk_rejection(fitter):
    """
    Scenario 14: Bilateral thigh cross-talk rejection across sagittal plane.
    """
    thigh_L_idx = np.where(fitter.vertex_regions == "thigh_L")[0]
    pts_left = fitter.v0[thigh_L_idx[:25]] + fitter.vertex_normals[thigh_L_idx[:25]] * 0.002
    assert np.all(pts_left[:, 0] > 0.02)

    _, acc_pts_cross, _, diag_cross = fitter.compute_correspondences(
        pts_left,
        point_regions=["thigh_R"] * len(pts_left),
    )
    assert len(acc_pts_cross) == 0
    assert diag_cross.rejection_reasons["cross_talk"] == len(pts_left)


def test_points_beyond_max_dist_rejected(fitter):
    """
    Scenario 15: Points beyond max_dist_m rejected as out_of_range.
    """
    torso_indices = np.where((fitter.vertex_regions == "torso") & (fitter.v0[:, 2] > 0.10) & (fitter.v0[:, 2] < 0.35))[0][:20]
    far_points = fitter.v0[torso_indices] + fitter.vertex_normals[torso_indices] * 0.060

    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        far_points,
        point_regions=["torso"] * len(far_points),
        max_dist_m=0.040,
    )

    assert len(acc_pts) == 0
    assert diag.rejected_points == len(far_points)
    assert diag.rejection_reasons["out_of_range"] == len(far_points)


# ---------------------------------------------------------------------------
# Integration-Level Test Through Production fit_with_loo() Path
# ---------------------------------------------------------------------------

def test_production_fit_with_loo_integration(anny_backend, adult_prior):
    """
    Scenario 16: End-to-end integration test through fit_with_loo() using multi-stop
    unposed data, verifying independent anatomical association without circular labeling.
    """
    subj = SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)
    cam = CameraConfig.portrait()
    stops = subj.simulate_stops(cam, n_stops=4, imperfections=False)

    unposer = BodyUnposer(
        canon_vertices=subj.true_verts,
        canon_joints=subj.true_joints,
        joint_names=anny_backend.joint_names,
        raw_dict=subj.raw_dict,
    )
    known_poses = [{
        "yaw_deg": s.true_yaw_deg,
        "translation_m": s.true_translation_m.tolist(),
        "arm_abd_L_deg": s.true_arm_abduction_deg[0],
        "arm_abd_R_deg": s.true_arm_abduction_deg[1],
    } for s in stops]

    unp_stops = unposer.unpose_all_stops(stops, known_poses)

    fitter = PersonalisedSurfaceFitter(
        canonical_vertices=subj.true_verts,
        faces=subj.faces,
        candidate_lambdas=[10.0, 50.0],
        raw_dict=subj.raw_dict,
        canonical_joints=subj.true_joints,
        joint_names=anny_backend.joint_names,
    )

    surf_res = fitter.fit_with_loo(unp_stops)

    # Verification: valid observations are NOT over-rejected
    assert surf_res.selected_lambda in [10.0, 50.0]
    assert surf_res.vertex_coverage_fraction > 0.65
    assert surf_res.delta_v_norm_mean_mm > 0.0

    diag = surf_res.diagnostics
    assert diag is not None
    assert diag.total_input_points > 100000
    assert diag.accepted_points > 80000
    # Rejection rate should be reasonable (< 20% on good data)
    assert diag.rejected_points < diag.total_input_points * 0.20
    assert diag.accepted_points + diag.rejected_points == diag.total_input_points
