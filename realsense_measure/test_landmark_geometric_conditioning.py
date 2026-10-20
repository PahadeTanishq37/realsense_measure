"""
Deterministic unit and regression tests for landmark geometric conditioning checks (Problem 4).

Validates the geometric conditioning gate before landmark-guided Kabsch registration:
1. Well-distributed, non-collinear facial landmarks: accepted.
2. Collinear midline landmarks: rejected for landmark-guided registration.
3. Nearly collinear correspondences: behavior consistent with documented thresholds.
4. Compact but valid non-collinear configuration: sufficiently constrained, accepted.
5. Planar but non-collinear points: not rejected solely for planarity (s3 == 0 is valid).
6. Repeated or nearly coincident points: insufficient unique geometry is rejected.
7. Missing landmarks that leave insufficient spread: safe FPFH fallback.
8. Unreliable landmark set with deceptively low residual: rejected before Kabsch.
9. Source and destination geometry disagreeing in conditioning: rejected safely.
10. Explicit config toggle (enable_landmark_conditioning_check) behavior.

Units:
- Point coordinates and thresholds are in METRES (m) unless explicitly named _mm or _deg.
- Residuals and distances are reported in MILLIMETRES (mm) where scaled by 1000.
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest

from config import RegistrationConfig
from registration import (
    check_landmark_geometric_conditioning,
    compute_rotation_deg,
    evaluate_point_set_geometry,
    register_frame_pair_landmark_guided,
    rigid_transform_from_correspondences,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Helper Data
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg() -> RegistrationConfig:
    cfg = RegistrationConfig()
    cfg.use_landmarks = True
    cfg.enable_landmark_conditioning_check = True
    cfg.landmark_min_spread_m = 0.025            # 25 mm
    cfg.landmark_min_secondary_spread_m = 0.005   # 5 mm
    cfg.landmark_max_collinear_ratio = 10.0
    cfg.landmark_min_unique_dist_m = 0.005        # 5 mm
    return cfg


@pytest.fixture
def well_distributed_landmarks() -> dict[str, dict]:
    """Realistic facial landmarks well-distributed across eyes, nose, mouth, and chin."""
    return {
        "left_eye_outer": {"position": np.array([-0.045, 0.035, 0.65]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "right_eye_outer": {"position": np.array([0.045, 0.035, 0.65]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "nose_tip": {"position": np.array([0.0, 0.0, 0.61]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "chin": {"position": np.array([0.0, -0.070, 0.64]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "left_mouth_corner": {"position": np.array([-0.025, -0.035, 0.63]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "right_mouth_corner": {"position": np.array([0.025, -0.035, 0.63]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
    }


def _make_dummy_frame_data():
    """Create lightweight dummy point clouds, depth, color, and camera intrinsics."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((20, 3)))
    color = np.zeros((20, 20, 3), dtype=np.uint8)
    depth = np.ones((20, 20), dtype=np.float32) * 0.6
    intr = o3d.camera.PinholeCameraIntrinsic(20, 20, 20.0, 20.0, 10.0, 10.0)
    return pcd, color, depth, intr


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_1_well_distributed_landmarks_accepted(default_cfg, well_distributed_landmarks):
    """
    Case 1: Well-distributed, non-collinear facial landmarks must be accepted by
    both the geometric helper and the production registration entry point.
    """
    pts = np.array([item["position"] for item in well_distributed_landmarks.values()])
    ok, reason = check_landmark_geometric_conditioning(pts, pts, cfg=default_cfg)
    assert ok, f"Expected well-distributed landmarks to pass, but failed: {reason}"

    # Verify production entry point invokes Kabsch
    pcd, color, depth, intr = _make_dummy_frame_data()
    mock_rigid = MagicMock(return_value=(np.eye(4), 1.0))

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[well_distributed_landmarks, well_distributed_landmarks]), \
         patch("registration.rigid_transform_from_correspondences", mock_rigid), \
         patch("registration.register_frame_pair", return_value=(np.eye(4), 0.5, 0.001)):

        register_frame_pair_landmark_guided(pcd, color, depth, pcd, color, depth, intr, default_cfg)

    assert mock_rigid.call_count == 1, "Kabsch rigid transform must be called for well-distributed landmarks"


def test_2_collinear_midline_landmarks_rejected(default_cfg):
    """
    Case 2: Collinear midline landmarks (e.g. nose bridge, nose tip, upper lip, chin)
    all lying on a 1D line must be rejected, and Kabsch must be skipped.
    """
    # 4 strictly collinear points along Y
    collinear_pts = np.array([
        [0.0, 0.06, 0.60],
        [0.0, 0.02, 0.60],
        [0.0, -0.02, 0.60],
        [0.0, -0.06, 0.60],
    ])

    ok, reason = check_landmark_geometric_conditioning(collinear_pts, collinear_pts, cfg=default_cfg)
    assert not ok, "Collinear points must be rejected"
    assert "nearly collinear" in reason or "secondary spread too small" in reason or "collinearity ratio" in reason

    # Verify production entry point skips Kabsch and routes to FPFH fallback
    pcd, color, depth, intr = _make_dummy_frame_data()
    collinear_lm = {
        f"pt_{i}": {"position": collinear_pts[i], "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False}
        for i in range(4)
    }

    mock_rigid = MagicMock(return_value=(np.eye(4), 0.5))
    mock_fallback = MagicMock(return_value=(np.eye(4), 0.7, 0.002))

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[collinear_lm, collinear_lm]), \
         patch("registration.rigid_transform_from_correspondences", mock_rigid), \
         patch("registration.register_frame_pair", mock_fallback):

        T, fit, rmse, method, res_mm = register_frame_pair_landmark_guided(
            pcd, color, depth, pcd, color, depth, intr, default_cfg
        )

    assert mock_rigid.call_count == 0, "Kabsch MUST NOT be called when landmarks are collinear!"
    assert mock_fallback.call_count == 1, "Must route to FPFH fallback"
    assert method == "fpfh_fallback"
    assert np.isnan(res_mm)


def test_3_nearly_collinear_threshold_consistency(default_cfg):
    """
    Case 3: Nearly collinear correspondences: behavior is consistent with documented thresholds.
    When lateral deviation produces ratio > max_collinear_ratio, it is rejected.
    When lateral deviation is widened so ratio <= max_collinear_ratio and s2 >= min_secondary_spread,
    it is accepted.
    """
    # Baseline line along Y: span 120 mm (s1 ~ 43 mm)
    # With tiny 0.4 mm lateral spread: s2 ~ 0.3 mm, ratio ~ 140 >> 10.0 -> REJECTED
    nearly_collinear = np.array([
        [0.0004, 0.06, 0.60],
        [-0.0004, 0.02, 0.60],
        [0.0003, -0.02, 0.60],
        [-0.0003, -0.06, 0.60],
    ])
    ok_reject, reason_reject, m_reject = evaluate_point_set_geometry(
        nearly_collinear,
        min_secondary_spread_m=default_cfg.landmark_min_secondary_spread_m,
        max_collinear_ratio=default_cfg.landmark_max_collinear_ratio,
    )
    assert not ok_reject
    assert m_reject["collinear_ratio"] > 10.0 or m_reject["s2_m"] < 0.005

    # Widened configuration with 16 mm lateral spread: s2 ~ 11 mm >= 5 mm, ratio ~ 3.8 <= 10.0 -> ACCEPTED
    widened = np.array([
        [0.015, 0.06, 0.60],
        [-0.015, 0.02, 0.60],
        [0.015, -0.02, 0.60],
        [-0.015, -0.06, 0.60],
    ])
    ok_accept, reason_accept, m_accept = evaluate_point_set_geometry(
        widened,
        min_secondary_spread_m=default_cfg.landmark_min_secondary_spread_m,
        max_collinear_ratio=default_cfg.landmark_max_collinear_ratio,
    )
    assert ok_accept, f"Expected widened configuration to pass: {reason_accept}"
    assert m_accept["s2_m"] >= default_cfg.landmark_min_secondary_spread_m
    assert m_accept["collinear_ratio"] <= default_cfg.landmark_max_collinear_ratio


def test_4_compact_valid_non_collinear_configuration(default_cfg):
    """
    Case 4: A compact but valid non-collinear configuration (e.g. nostrils, nose tip, upper lip)
    spanning ~30 mm in X and ~25 mm in Y is assessed. It is sufficiently constrained and accepted.
    """
    compact_valid_pts = np.array([
        [0.0, 0.0, 0.61],          # nose_tip
        [-0.015, -0.008, 0.62],    # left_nostril (15 mm left)
        [0.015, -0.008, 0.62],     # right_nostril (15 mm right)
        [0.0, -0.025, 0.625],      # upper_lip_center (25 mm down)
    ])

    ok, reason, metrics = evaluate_point_set_geometry(
        compact_valid_pts,
        min_spread_m=default_cfg.landmark_min_spread_m,
        min_secondary_spread_m=default_cfg.landmark_min_secondary_spread_m,
        max_collinear_ratio=default_cfg.landmark_max_collinear_ratio,
    )

    assert ok, f"Compact valid non-collinear geometry should be accepted: {reason}"
    assert metrics["max_spread_m"] >= 0.025, "Span should meet 25 mm threshold"
    assert metrics["s2_m"] >= 0.005, "Secondary spread should meet 5 mm threshold"
    assert metrics["collinear_ratio"] <= 3.0, "Collinearity ratio should be near 1.0"


def test_5_planar_non_collinear_points_not_rejected_for_planarity(default_cfg):
    """
    Case 5: Planar but non-collinear points (s3 == 0) must NOT be rejected merely
    for planarity. Non-collinear planar points uniquely constrain a 3D rigid transform.
    """
    # 4 coplanar points on Z = 0.65 forming an 80 mm x 70 mm rectangle
    coplanar_pts = np.array([
        [-0.04, 0.035, 0.65],
        [0.04, 0.035, 0.65],
        [-0.04, -0.035, 0.65],
        [0.04, -0.035, 0.65],
    ])

    ok, reason, metrics = evaluate_point_set_geometry(coplanar_pts)
    assert ok, f"Planar non-collinear points must be accepted: {reason}"
    assert np.isclose(metrics["s3_m"], 0.0, atol=1e-12), "Must be strictly coplanar (s3 == 0)"
    assert metrics["s2_m"] > 0.025, "s2 must have substantial spread"
    assert metrics["collinear_ratio"] < 2.0, "Aspect ratio must be close to 1"

    # Verify Kabsch succeeds and accurately recovers 3D rotation from planar points
    theta = np.radians(12.0)
    R_gt = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    t_gt = np.array([0.010, -0.005, 0.015])
    dst_pts = (R_gt @ coplanar_pts.T).T + t_gt

    T_est, res_mm = rigid_transform_from_correspondences(coplanar_pts, dst_pts)
    rot_err = compute_rotation_deg(T_est[:3, :3] @ R_gt.T)
    assert rot_err < 0.01, f"Planar points must accurately constrain 3D rotation, err={rot_err:.4f} deg"
    assert res_mm < 0.01


def test_6_repeated_or_nearly_coincident_points_rejected(default_cfg):
    """
    Case 6: Points containing repeated or nearly coincident correspondences that make
    the nominal count misleading (e.g. only 2 unique spatial locations) are rejected.
    """
    # 4 nominal points, but only 2 unique locations (2 duplicates at A, 2 duplicates at B)
    repeated_pairs = np.array([
        [0.0, 0.0, 0.60],
        [0.0, 0.0, 0.60],
        [0.05, 0.0, 0.60],
        [0.05, 0.0, 0.60],
    ])
    ok_pairs, reason_pairs, m_pairs = evaluate_point_set_geometry(repeated_pairs)
    assert not ok_pairs
    assert "insufficient unique landmarks" in reason_pairs
    assert m_pairs["n_unique"] == 2.0

    # 4 nominal points, but 3 clustered within 0.5 mm noise
    nearly_coincident = np.array([
        [0.0, 0.0, 0.60],
        [0.0002, 0.0001, 0.60],
        [-0.0001, 0.0002, 0.60],
        [0.05, 0.0, 0.60],
    ])
    ok_coin, reason_coin, m_coin = evaluate_point_set_geometry(nearly_coincident, min_unique_dist_m=0.005)
    assert not ok_coin
    assert "insufficient unique landmarks" in reason_coin
    assert m_coin["n_unique"] == 2.0


def test_7_missing_landmarks_insufficient_spread_fallback(default_cfg):
    """
    Case 7: Missing landmarks that leave a small cluster (diameter < 25 mm) are rejected,
    routing safely to FPFH fallback without running Kabsch.
    """
    # 4 points all in a tiny 6 mm patch
    tiny_cluster = np.array([
        [0.0, 0.0, 0.60],
        [0.004, 0.002, 0.60],
        [-0.002, 0.003, 0.60],
        [0.001, -0.003, 0.60],
    ])
    ok, reason, metrics = evaluate_point_set_geometry(tiny_cluster, min_spread_m=0.025)
    assert not ok
    assert "spread too small" in reason or "insufficient unique" in reason

    pcd, color, depth, intr = _make_dummy_frame_data()
    tiny_lm = {
        f"pt_{i}": {"position": tiny_cluster[i], "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False}
        for i in range(4)
    }

    mock_rigid = MagicMock()
    mock_fallback = MagicMock(return_value=(np.eye(4), 0.6, 0.002))

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[tiny_lm, tiny_lm]), \
         patch("registration.rigid_transform_from_correspondences", mock_rigid), \
         patch("registration.register_frame_pair", mock_fallback):

        T, fit, rmse, method, res_mm = register_frame_pair_landmark_guided(
            pcd, color, depth, pcd, color, depth, intr, default_cfg
        )

    assert mock_rigid.call_count == 0, "Kabsch must not be called when spread is insufficient"
    assert method == "fpfh_fallback"


def test_8_deceptively_low_residual_rejected_before_kabsch(default_cfg):
    """
    Case 8: An unreliable (collinear) landmark set with a deceptively low Kabsch residual.
    Demonstrates that Kabsch would report a tiny residual (< 0.5 mm) while having a large
    rotation error (> 50 deg). The geometric conditioning check rejects it BEFORE Kabsch.
    """
    collinear_src = np.array([
        [0.0, 0.06, 0.60],
        [0.0, 0.02, 0.60],
        [0.0, -0.02, 0.60],
        [0.0, -0.06, 0.60],
    ])

    # Ground truth rotation: 10 deg around Y, 5 deg around X
    th_y = np.radians(10.0)
    Ry = np.array([[np.cos(th_y), 0, np.sin(th_y)], [0, 1, 0], [-np.sin(th_y), 0, np.cos(th_y)]])
    th_x = np.radians(5.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(th_x), -np.sin(th_x)], [0, np.sin(th_x), np.cos(th_x)]])
    R_gt = Ry @ Rx

    # Target points with 0.5 mm noise
    collinear_dst = (R_gt @ collinear_src.T).T + np.array([
        [0.0005, 0, 0],
        [-0.0005, 0, 0],
        [0.0003, 0, 0],
        [-0.0003, 0, 0],
    ])

    # Prove that raw Kabsch would produce a deceptively low residual (< 0.5 mm) with huge rotation error
    T_raw, raw_res_mm = rigid_transform_from_correspondences(collinear_src, collinear_dst)
    raw_rot_err = compute_rotation_deg(T_raw[:3, :3] @ R_gt.T)
    assert raw_res_mm < 1.0, f"Demonstration premise: residual must be deceptively low, got {raw_res_mm} mm"
    assert raw_rot_err > 40.0, f"Demonstration premise: rotation error must be severe, got {raw_rot_err} deg"

    # Now verify that the production entry point catches this and skips Kabsch
    pcd, color, depth, intr = _make_dummy_frame_data()
    src_lm = {f"k_{i}": {"position": collinear_src[i], "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False} for i in range(4)}
    dst_lm = {f"k_{i}": {"position": collinear_dst[i], "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False} for i in range(4)}

    mock_rigid = MagicMock()
    mock_fallback = MagicMock(return_value=(np.eye(4), 0.75, 0.002))

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[src_lm, dst_lm]), \
         patch("registration.rigid_transform_from_correspondences", mock_rigid), \
         patch("registration.register_frame_pair", mock_fallback):

        T, fit, rmse, method, res_mm = register_frame_pair_landmark_guided(
            pcd, color, depth, pcd, color, depth, intr, default_cfg
        )

    assert mock_rigid.call_count == 0, "Kabsch MUST be skipped despite deceptively low potential residual"
    assert method == "fpfh_fallback"


def test_9_source_and_destination_geometry_disagreeing_rejected(default_cfg, well_distributed_landmarks):
    """
    Case 9: Source and destination geometry disagreeing in conditioning:
    - If source is valid but destination is collinear: rejected safely.
    - If destination is valid but source is collinear: rejected safely.
    """
    well_pts = np.array([item["position"] for item in well_distributed_landmarks.values()][:4])
    collinear_pts = np.array([
        [0.0, 0.06, 0.60],
        [0.0, 0.02, 0.60],
        [0.0, -0.02, 0.60],
        [0.0, -0.06, 0.60],
    ])

    # Source ok, Target collinear
    ok1, reason1 = check_landmark_geometric_conditioning(well_pts, collinear_pts, cfg=default_cfg)
    assert not ok1
    assert "Destination geometry rejected" in reason1

    # Source collinear, Target ok
    ok2, reason2 = check_landmark_geometric_conditioning(collinear_pts, well_pts, cfg=default_cfg)
    assert not ok2
    assert "Source geometry rejected" in reason2


def test_10_disable_conditioning_gate_via_config(default_cfg, well_distributed_landmarks):
    """
    Case 10: Disabling the gate via config (enable_landmark_conditioning_check=False)
    allows bypassing the check when explicitly requested by user configuration.
    """
    default_cfg.enable_landmark_conditioning_check = False
    collinear_pts = np.array([
        [0.0, 0.06, 0.60],
        [0.0, 0.02, 0.60],
        [0.0, -0.02, 0.60],
        [0.0, -0.06, 0.60],
    ])
    collinear_lm = {
        f"pt_{i}": {"position": collinear_pts[i], "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False}
        for i in range(4)
    }

    pcd, color, depth, intr = _make_dummy_frame_data()
    mock_rigid = MagicMock(return_value=(np.eye(4), 0.5))

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[collinear_lm, collinear_lm]), \
         patch("registration.rigid_transform_from_correspondences", mock_rigid), \
         patch("registration.register_frame_pair", return_value=(np.eye(4), 0.5, 0.001)):

        register_frame_pair_landmark_guided(pcd, color, depth, pcd, color, depth, intr, default_cfg)

    assert mock_rigid.call_count == 1, "Kabsch should be called when check is explicitly disabled"


def test_11_vectorized_duplicate_detection_known_pair():
    """
    Case 11: Vectorized duplicate detection correctly identifies known duplicate pairs.
    Tests known duplicate points closer than min_unique_dist_m to confirm exact semantics.
    """
    # 5 points with points 0 and 1 being known duplicates (< 0.5 mm apart)
    pts_with_duplicate = np.array([
        [0.000, 0.000, 0.60],
        [0.0002, 0.0001, 0.60],  # duplicate of point 0 (< 5 mm)
        [0.050, 0.000, 0.60],
        [0.000, 0.050, 0.60],
        [-0.050, 0.000, 0.60],
    ])
    is_valid, reason, metrics = evaluate_point_set_geometry(pts_with_duplicate, min_unique_dist_m=0.005)
    assert is_valid, f"Expected 4 distinct landmarks to be valid, got: {reason}"
    assert metrics["n_unique"] == 4.0

    # 5 points with 3 duplicates of point 0 (only 3 unique points total, below min_unique_count=4)
    pts_insufficient_unique = np.array([
        [0.000, 0.000, 0.60],
        [0.0001, 0.0000, 0.60],  # dup of 0
        [0.0000, 0.0002, 0.60],  # dup of 0
        [0.050, 0.000, 0.60],    # unique pt 2
        [0.000, 0.050, 0.60],    # unique pt 3
    ])
    is_valid_bad, reason_bad, metrics_bad = evaluate_point_set_geometry(pts_insufficient_unique, min_unique_dist_m=0.005)
    assert not is_valid_bad
    assert "insufficient unique landmarks" in reason_bad
    assert metrics_bad["n_unique"] == 3.0

