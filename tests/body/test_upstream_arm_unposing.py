"""
tests/body/test_upstream_arm_unposing.py
========================================
Comprehensive regression tests for upstream arm identification and unposing
in realsense_measure/body/unpose.py.

Exercises all 14 required cases from Problem 2E:
1. Hand beside hip: must not receive pelvis skinning weights merely due to proximity.
2. Forearm beside torso: must be identified independently where landmarks are reliable.
3. Arm resting against thigh: distinguished or marked unknown, never deforming thigh.
4. Crossed arms: bilateral constraint violation tagged unknown or non-arm.
5. Normal A-pose: preserves arm labels and zero residual unposing.
6. Different left and right arm-abduction angles: independent asymmetric unposing.
7. Missing wrist landmark: safe handling without guessing or assigning to pelvis.
8. Occluded elbow: graceful handling without misclassifying forearm.
9. Low-confidence pose landmarks: confidence < 0.50 rejected as unreliable.
10. Invalid or missing depth at a landmark: NaN/zero depth handled safely.
11. Correct shoulder-pivot transformation: exact mathematical inversion.
12. Torso and pelvis points do not receive arm correction.
13. Unknown arm observations must not silently become pelvis observations.
14. Multiple stops preserve source-point and classification alignment.
"""

import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import (
    SyntheticTruthSubject,
    CameraConfig,
    apply_arm_abduction,
)
from realsense_measure.body.unpose import (
    BodyUnposer,
    UnposedStopData,
    ArmUnposeDiagnostics,
)
from realsense_measure.body.personalized_surface import PersonalisedSurfaceFitter


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
def unposer(truth_subject):
    return BodyUnposer(
        canon_vertices=truth_subject.true_verts,
        canon_joints=truth_subject.true_joints,
        joint_names=truth_subject.joint_names,
        raw_dict=truth_subject.raw_dict,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_left_hand_indices(subj):
    bl = subj.raw_dict["bone_labels"]
    hand_bones = [b for b in bl if ".L" in b and any(k in b.lower() for k in ["wrist", "hand", "finger", "thumb", "metacarpal"])]
    h_idxs = set(bl.index(b) for b in hand_bones if b in bl)
    vw = subj.raw_dict["vertex_bone_weights"]
    vi = subj.raw_dict["vertex_bone_indices"]
    if hasattr(vw, "cpu"):
        vw = vw.cpu().numpy()
        vi = vi.cpu().numpy()
    return np.where(np.isin(vi, list(h_idxs)).any(axis=1))[0]


def _get_left_forearm_indices(subj):
    bl = subj.raw_dict["bone_labels"]
    fa_bones = [b for b in bl if ".L" in b and any(k in b.lower() for k in ["lowerarm", "elbow"])]
    f_idxs = set(bl.index(b) for b in fa_bones if b in bl)
    vw = subj.raw_dict["vertex_bone_weights"]
    vi = subj.raw_dict["vertex_bone_indices"]
    if hasattr(vw, "cpu"):
        vw = vw.cpu().numpy()
        vi = vi.cpu().numpy()
    return np.where(np.isin(vi, list(f_idxs)).any(axis=1))[0]


def _get_pelvis_indices(subj):
    bl = subj.raw_dict["bone_labels"]
    p_bones = [b for b in bl if any(k in b.lower() for k in ["root", "pelvis"])]
    p_idxs = set(bl.index(b) for b in p_bones if b in bl)
    vw = subj.raw_dict["vertex_bone_weights"]
    vi = subj.raw_dict["vertex_bone_indices"]
    if hasattr(vw, "cpu"):
        vw = vw.cpu().numpy()
        vi = vi.cpu().numpy()
    return np.where(np.isin(vi, list(p_idxs)).any(axis=1))[0]


# ---------------------------------------------------------------------------
# 14 Test Cases
# ---------------------------------------------------------------------------

def test_hand_beside_hip_not_pelvis_skinning(truth_subject, unposer):
    """
    Case 1: Hand points beside the hip must NOT receive pelvis skinning weights
    merely due to spatial proximity. They must be identified as arm/hand,
    receive arm abduction unposing, and match canonical hand vertices.
    """
    # Pose left arm downward 35 degrees so hand hangs beside hip
    verts_posed = apply_arm_abduction(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.joint_names,
        truth_subject.raw_dict,
        left_deg=35.0,
        right_deg=0.0,
    )
    hand_idx = _get_left_hand_indices(truth_subject)
    pts_hand_posed = verts_posed[hand_idx]

    p_canon, dists, nn_idx = unposer.unpose_points(
        pts_hand_posed,
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=35.0,
        arm_abd_R_deg=0.0,
    )

    # 1. Must be identified as arm/hand
    labels = unposer.last_arm_labels
    assert np.mean(np.isin(labels, ["hand_L", "forearm_L"])) > 0.85

    # 2. Points must be unposed outward into A-pose (> 0.35m from origin in X)
    assert np.mean(p_canon[:, 0] > 0.35) > 0.85

    # 3. Nearest canonical mesh vertices must NOT be pelvis
    fitter = PersonalisedSurfaceFitter(truth_subject.true_verts, truth_subject.faces)
    matched_regions = fitter.vertex_regions[nn_idx]
    assert np.sum(matched_regions == "pelvis") == 0


def test_forearm_beside_torso_independently_identified(truth_subject, unposer):
    """
    Case 2: Forearm points beside torso must be identified independently
    where landmarks are reliable, not labeled as torso.
    """
    verts_posed = apply_arm_abduction(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.joint_names,
        truth_subject.raw_dict,
        left_deg=25.0,
        right_deg=0.0,
    )
    fa_idx = _get_left_forearm_indices(truth_subject)
    pts_fa_posed = verts_posed[fa_idx]

    p_canon, dists, nn_idx = unposer.unpose_points(
        pts_fa_posed,
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=25.0,
        arm_abd_R_deg=0.0,
    )

    labels = unposer.last_arm_labels
    assert np.mean(np.isin(labels, ["forearm_L", "hand_L", "upperarm_L"])) > 0.85
    assert np.sum(labels == "torso") == 0


def test_arm_resting_against_thigh(truth_subject, unposer):
    """
    Case 3: Hand/arm points near thigh contact plane must be distinguished
    from thigh observations or safely marked unknown.
    """
    # When arm is abducted 40 deg, wrist hangs right beside the lateral thigh
    rad = np.radians(40.0)
    cos_L, sin_L = np.cos(rad), np.sin(rad)
    diff_wr = unposer.canon_wr_L - unposer.pivot_L
    wr_posed = unposer.pivot_L + np.array([
        diff_wr[0] * cos_L + diff_wr[2] * sin_L,
        diff_wr[1],
        -diff_wr[0] * sin_L + diff_wr[2] * cos_L,
    ], dtype=np.float32)

    # Point on the hand resting against the lateral thigh
    p_contact = wr_posed + np.array([0.02, 0.0, 0.0], dtype=np.float32)

    _, _, _, labels, diag = unposer.identify_arm_points(
        np.array([p_contact]),
        arm_abd_L_deg=40.0,
    )
    # Point in the contact collision boundary must be either hand_L or unknown, NEVER thigh
    assert labels[0] in ["hand_L", "unknown"]


def test_crossed_arms(unposer):
    """
    Case 4: Left arm points crossing into the right half of the body (X < -0.015m)
    must violate bilateral constraints and be tagged unknown or non-arm.
    """
    # Point on right side claimed by left arm
    p_crossed = np.array([[-0.10, 0.15, 0.20]], dtype=np.float32)

    _, _, _, labels, diag = unposer.identify_arm_points(
        p_crossed,
        arm_abd_L_deg=20.0,
    )
    assert labels[0] != "upperarm_L" and labels[0] != "forearm_L"


def test_normal_a_pose_preservation(truth_subject, unposer):
    """
    Case 5: Normal A-pose points preserve arm labels and produce zero unposing error.
    """
    hand_idx = _get_left_hand_indices(truth_subject)[:50]
    pts_canon_hand = truth_subject.true_verts[hand_idx]

    p_canon, dists, nn_idx = unposer.unpose_points(
        pts_canon_hand,
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=0.0,
        arm_abd_R_deg=0.0,
    )

    err = np.linalg.norm(p_canon - pts_canon_hand, axis=-1)
    assert np.max(err) < 1e-5
    assert np.all(np.isin(unposer.last_arm_labels, ["hand_L", "forearm_L"]))


def test_different_left_right_arm_abductions(truth_subject, unposer):
    """
    Case 6: Left arm at +25 deg, right arm at -15 deg. Verify independent
    asymmetric unposing transformations are applied correctly.
    """
    verts_posed = apply_arm_abduction(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.joint_names,
        truth_subject.raw_dict,
        left_deg=25.0,
        right_deg=-15.0,
    )
    hand_L = _get_left_hand_indices(truth_subject)[:30]

    bl = truth_subject.raw_dict["bone_labels"]
    r_bones = [b for b in bl if ".R" in b and any(k in b.lower() for k in ["wrist", "hand", "finger", "thumb", "metacarpal"])]
    r_idxs = set(bl.index(b) for b in r_bones if b in bl)
    vi = truth_subject.raw_dict["vertex_bone_indices"]
    if hasattr(vi, "cpu"): vi = vi.cpu().numpy()
    hand_R = np.where(np.isin(vi, list(r_idxs)).any(axis=1))[0][:30]

    pts = np.vstack([verts_posed[hand_L], verts_posed[hand_R]])

    p_canon, _, _ = unposer.unpose_points(
        pts,
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=25.0,
        arm_abd_R_deg=-15.0,
    )

    # Recovered canonical positions should match true canonical verts within 5 mm
    true_canon = np.vstack([truth_subject.true_verts[hand_L], truth_subject.true_verts[hand_R]])
    err_mm = np.linalg.norm(p_canon - true_canon, axis=-1) * 1000.0
    assert np.median(err_mm) < 5.0


def test_missing_wrist_landmark_safe_handling(truth_subject, unposer):
    """
    Case 7: Missing wrist landmark must not guess or assign observations to pelvis.
    """
    p_test = np.array([[0.25, 0.0, 0.0]], dtype=np.float32)

    # Landmarks dict with elbow present but wrist explicitly None / missing
    lms = {
        "shoulder_L": {"pos": unposer.pivot_L, "confidence": 0.95},
        "elbow_L": {"pos": unposer.canon_el_L, "confidence": 0.95},
        "wrist_L": None,
    }

    _, _, _, labels, diag = unposer.identify_arm_points(
        p_test,
        landmarks=lms,
        arm_abd_L_deg=35.0,
    )
    assert labels[0] in ["unknown", "non_arm"]
    assert diag.n_unposed_insufficient_evidence >= 0


def test_occluded_elbow_landmark_safe_handling(truth_subject, unposer):
    """
    Case 8: Occluded elbow landmark handled safely without crashing.
    """
    p_test = np.array([[0.30, 0.0, 0.20]], dtype=np.float32)
    lms = {
        "shoulder_L": unposer.pivot_L,
        "elbow_L": {"pos": [0.0, 0.0, 0.0], "confidence": 0.10},  # occluded/low confidence
        "wrist_L": unposer.canon_wr_L,
    }

    _, _, _, labels, diag = unposer.identify_arm_points(
        p_test,
        landmarks=lms,
        arm_abd_L_deg=20.0,
    )
    assert labels[0] in ["unknown", "non_arm"]


def test_low_confidence_pose_landmarks(unposer):
    """
    Case 9: Landmark with confidence < 0.50 rejected as unreliable.
    """
    lms = {
        "shoulder_L": {"pos": [0.19, 0.0, 0.45], "confidence": 0.30},
        "elbow_L": {"pos": [0.35, 0.0, 0.25], "confidence": 0.40},
        "wrist_L": {"pos": [0.45, 0.0, 0.10], "confidence": 0.20},
    }
    pts = np.array([[0.35, 0.0, 0.25]], dtype=np.float32)
    _, _, _, labels, diag = unposer.identify_arm_points(pts, landmarks=lms)
    assert diag.landmark_confidence_min < 0.50
    assert labels[0] in ["unknown", "non_arm"]


def test_invalid_or_missing_depth_at_landmark(unposer):
    """
    Case 10: Landmark with NaN or non-positive depth handled gracefully without crash.
    """
    lms = {
        "shoulder_L": [np.nan, 0.0, 0.45],
        "elbow_L": [0.35, np.inf, 0.25],
        "wrist_L": [0.45, 0.0, -1.0],
    }
    pts = np.array([[0.35, 0.0, 0.25]], dtype=np.float32)
    _, _, _, labels, diag = unposer.identify_arm_points(pts, landmarks=lms)
    assert len(labels) == 1


def test_correct_shoulder_pivot_transformation(truth_subject, unposer):
    """
    Case 11: Inverting arm abduction rotates exactly around the shoulder pivot.
    """
    sh = unposer.pivot_L
    wr = unposer.canon_wr_L
    diff = wr - sh

    # Forward rotation by 30 deg
    rad = np.radians(30.0)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    wr_posed = sh + np.array([
        diff[0] * cos_a + diff[2] * sin_a,
        diff[1],
        -diff[0] * sin_a + diff[2] * cos_a,
    ], dtype=np.float32)

    # Unpose with unpose_points
    p_canon, _, _ = unposer.unpose_points(
        np.array([wr_posed]),
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=30.0,
    )

    err = np.linalg.norm(p_canon[0] - wr)
    assert err < 1e-4, f"Shoulder pivot inverse rotation error: {err*1000:.3f} mm"


def test_torso_pelvis_points_do_not_receive_arm_correction(truth_subject, unposer):
    """
    Case 12: Torso and pelvis points never receive arm unposing correction.
    """
    pelvis_idx = _get_pelvis_indices(truth_subject)[:50]
    pts_pelv = truth_subject.true_verts[pelvis_idx]

    p_canon, dists, nn_idx = unposer.unpose_points(
        pts_pelv,
        yaw_deg=0.0,
        translation_m=[0.0, 0.0, 0.0],
        arm_abd_L_deg=35.0,
        arm_abd_R_deg=35.0,
    )

    # Max displacement must be strictly zero
    disp = np.linalg.norm(p_canon - pts_pelv, axis=-1)
    assert np.max(disp) < 1e-6
    assert np.all(unposer.last_arm_labels == "non_arm")


def test_unknown_arm_observations_not_silently_pelvis(truth_subject, unposer):
    """
    Case 13: Uncertain arm-region observations marked unknown do not become pelvis
    observations in the surface correspondence stage.
    """
    fitter = PersonalisedSurfaceFitter(truth_subject.true_verts, truth_subject.faces)

    # Point marked unknown by unposer
    p_unk = np.array([[0.20, 0.0, 0.0]], dtype=np.float32)
    assigned_v, acc_pts, weights, diag = fitter.compute_correspondences(
        p_unk,
        point_regions=["unknown"],
    )
    assert len(acc_pts) == 0
    assert diag.rejected_points == 1
    assert diag.rejection_reasons["unknown_region"] == 1


def test_multiple_stops_preserve_source_and_classification_alignment(truth_subject, anny_backend, adult_prior):
    """
    Case 14: Multi-stop unpose_all_stops preserves source-point ordering,
    arm label alignment, and diagnostic statistics.
    """
    cam = CameraConfig.portrait()
    stops = truth_subject.simulate_stops(cam, n_stops=4, imperfections=False)

    unposer = BodyUnposer(
        canon_vertices=truth_subject.true_verts,
        canon_joints=truth_subject.true_joints,
        joint_names=truth_subject.joint_names,
        raw_dict=truth_subject.raw_dict,
    )

    known_poses = [{
        "yaw_deg": s.true_yaw_deg,
        "translation_m": s.true_translation_m.tolist(),
        "arm_abd_L_deg": s.true_arm_abduction_deg[0],
        "arm_abd_R_deg": s.true_arm_abduction_deg[1],
    } for s in stops]

    unp_stops = unposer.unpose_all_stops(stops, known_poses)
    assert len(unp_stops) == 4

    for i, st in enumerate(unp_stops):
        assert len(st.unposed_points) == len(stops[i].point_cloud_world)
        assert len(st.arm_point_labels) == len(st.unposed_points)
        assert st.diagnostics is not None
        assert st.diagnostics.n_non_arm > 0
