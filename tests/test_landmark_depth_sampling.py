"""
Regression tests for facial landmark depth sampling, background rejection,
and production registration variance weighting.

Tests:
1. Non-finite (NaN, Inf), zero, and negative depth values are properly excluded.
2. Step edge:
   - Valid center at foreground: accepted with low_confidence=True.
   - Valid center on farther background: rejected as ambiguous/contaminated.
3. Invalid center:
   - Invalid center + flat background-only patch: rejected (no 3D point fabricated).
   - Invalid center + mixed foreground/background: selects defensible foreground surface, sets low_confidence=True.
   - Invalid center + isolated single-pixel noise: rejected.
4. No usable depth: returns None (no fabricated 3D points).
5. Production registration variance weighting (exercising registration.py directly):
   - Monotonic decrease across threshold boundaries (var 0.02 vs 1e-4).
   - Non-finite variance handling.
   - Low confidence down-weighting policy.
   - Safe fallthrough to FPFH when fewer than 4 common landmarks exist.
6. Output compatibility for all 13 landmarks contract.
"""
import copy
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest

from config import LandmarkConfig, RegistrationConfig
from landmarks import LANDMARK_INDICES, detect_face_landmarks_3d
from registration import register_frame_pair_landmark_guided


class MockNormalizedLandmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _build_mock_face(w: int, h: int, landmark_px_coords: dict[str, tuple[int, int]] | None = None):
    """Create a mock MediaPipe face landmark list with 500 entries."""
    face = [MockNormalizedLandmark(0.5, 0.5) for _ in range(500)]
    if landmark_px_coords:
        for name, (px, py) in landmark_px_coords.items():
            if name in LANDMARK_INDICES:
                idx = LANDMARK_INDICES[name]
                face[idx] = MockNormalizedLandmark(px / w, py / h)
    return face


def _create_mock_landmarker(mock_face):
    mock_landmarker = MagicMock()
    mock_result = MagicMock()
    mock_result.face_landmarks = [mock_face]
    mock_landmarker.detect.return_value = mock_result
    return mock_landmarker


@pytest.fixture
def pinhole_intrinsics():
    return o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)


# ============================================================================
# Test 1: Non-finite, zero, and negative depth exclusion
# ============================================================================
def test_non_finite_and_invalid_depth_exclusion(pinhole_intrinsics):
    w, h = 640, 480
    depth_m = np.zeros((h, w), dtype=np.float32)
    py, px = 240, 320
    # Center is valid 0.50m; patch has NaN, Inf, 0, negative, and 0.49/0.51m
    depth_m[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [np.nan, np.inf, 0.0],
        [-1.0, 0.50, 0.51],
        [0.0, 0.49, -np.inf],
    ], dtype=np.float32)

    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    mock_face = _build_mock_face(w, h, {"nose_tip": (px, py)})

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        result = detect_face_landmarks_3d(color_bgr, depth_m, pinhole_intrinsics, quiet=True)

    assert result is not None
    assert "nose_tip" in result
    lm = result["nose_tip"]
    assert np.isclose(lm["position"][2], 0.50, atol=0.01)
    assert np.isfinite(lm["position"]).all()
    assert np.isclose(lm["valid_ratio"], 3.0 / 9.0)
    assert lm["low_confidence"] is True


# ============================================================================
# Test 2: Background rejection & Step edge handling
# ============================================================================
def test_step_edge_foreground_vs_background_center(pinhole_intrinsics):
    w, h = 640, 480
    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Case A: Center pixel is valid foreground (0.50m), background (1.80m) in patch
    depth_a = np.zeros((h, w), dtype=np.float32)
    py, px = 240, 320
    depth_a[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [0.50, 0.50, 1.80],
        [0.50, 0.50, 1.80],
        [1.80, 1.80, 1.80],
    ], dtype=np.float32)

    mock_face = _build_mock_face(w, h, {"nose_tip": (px, py)})
    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        result_a = detect_face_landmarks_3d(color_bgr, depth_a, pinhole_intrinsics, quiet=True)

    assert result_a is not None and "nose_tip" in result_a
    lm_a = result_a["nose_tip"]
    assert np.isclose(lm_a["position"][2], 0.50, atol=0.005)
    # Because there was a depth discontinuity in the patch, low_confidence must be True
    assert lm_a["low_confidence"] is True

    # Case B: Center pixel landed on background (1.80m) while foreground face is in patch (0.50m)
    depth_b = np.zeros((h, w), dtype=np.float32)
    depth_b[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [0.50, 0.50, 1.80],
        [0.50, 1.80, 1.80],  # Center (py, px) is 1.80m
        [1.80, 1.80, 1.80],
    ], dtype=np.float32)

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        result_b = detect_face_landmarks_3d(color_bgr, depth_b, pinhole_intrinsics, quiet=True)

    # Must REJECT the landmark because center belongs to the farther cluster at a discontinuity
    assert result_b is None or "nose_tip" not in result_b


# ============================================================================
# Test 3: Invalid center depth scenarios
# ============================================================================
def test_invalid_center_depth_scenarios(pinhole_intrinsics):
    w, h = 640, 480
    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    py, px = 240, 320
    mock_face = _build_mock_face(w, h, {"chin": (px, py)})

    # Scenario 3A: Center invalid + flat background only (e.g. wall at 1.80m)
    depth_wall = np.zeros((h, w), dtype=np.float32)
    depth_wall[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [1.80, 1.80, 1.80],
        [1.80, 0.00, 1.80],  # Center is 0.0
        [1.80, 1.80, 1.80],
    ], dtype=np.float32)

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        res_wall = detect_face_landmarks_3d(color_bgr, depth_wall, pinhole_intrinsics, quiet=True)

    # Must NOT fabricate a 3D landmark on the wall when center is invalid with no face cluster!
    assert res_wall is None or "chin" not in res_wall

    # Scenario 3B: Center invalid + mixed foreground (0.52m, 3 px) and background (1.75m, 5 px)
    depth_mixed = np.zeros((h, w), dtype=np.float32)
    depth_mixed[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [0.52, 0.52, 1.75],
        [0.52, 0.00, 1.75],
        [1.75, 1.75, 1.75],
    ], dtype=np.float32)

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        res_mixed = detect_face_landmarks_3d(color_bgr, depth_mixed, pinhole_intrinsics, quiet=True)

    assert res_mixed is not None and "chin" in res_mixed
    lm_mixed = res_mixed["chin"]
    assert np.isclose(lm_mixed["position"][2], 0.52, atol=0.005)
    assert lm_mixed["low_confidence"] is True

    # Scenario 3C: Center invalid + isolated single-pixel noise (0.30m, 1 px) and background (1.80m, 7 px)
    depth_spike = np.zeros((h, w), dtype=np.float32)
    depth_spike[py - 1 : py + 2, px - 1 : px + 2] = np.array([
        [0.30, 1.80, 1.80],
        [1.80, 0.00, 1.80],
        [1.80, 1.80, 1.80],
    ], dtype=np.float32)

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        res_spike = detect_face_landmarks_3d(color_bgr, depth_spike, pinhole_intrinsics, quiet=True)

    # Single pixel is unreliable; must be rejected
    assert res_spike is None or "chin" not in res_spike


# ============================================================================
# Test 4: Patch with no usable depth
# ============================================================================
def test_patch_with_no_usable_depth(pinhole_intrinsics):
    w, h = 640, 480
    depth_m = np.zeros((h, w), dtype=np.float32)
    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    mock_face = _build_mock_face(w, h, {"nose_tip": (320, 240)})

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        result = detect_face_landmarks_3d(color_bgr, depth_m, pinhole_intrinsics, quiet=True)

    assert result is None


# ============================================================================
# Test 5: Production registration variance weighting & fallback safety
# ============================================================================
def test_production_registration_variance_weighting():
    """
    Directly exercises registration.py:register_frame_pair_landmark_guided().
    Intercepts the weights passed to rigid_transform_from_correspondences()
    to verify actual production code behavior.
    """
    import registration

    captured_weights = {}

    orig_rigid = registration.rigid_transform_from_correspondences

    def spy_rigid(src_pts, dst_pts, weights=None):
        captured_weights["weights"] = copy.deepcopy(weights)
        return np.eye(4), 1.0  # low residual to avoid early fail

    cfg = RegistrationConfig()
    cfg.landmark_cfg = LandmarkConfig()

    # Create 4 common landmarks to meet the len(common) >= 4 threshold:
    # 0: base clean landmark (var = 1e-6)
    # 1: moderate variance (var = 1e-4)
    # 2: high variance crossing old 0.01 threshold (var = 0.02)
    # 3: low_confidence landmark (var = 1e-6, low_confidence = True)
    src_landmarks = {
        "nose_tip": {"position": np.array([0.0, 0.0, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "chin": {"position": np.array([0.0, -0.08, 0.5]), "valid_ratio": 1.0, "variance": 1e-4, "low_confidence": False},
        "left_eye_inner": {"position": np.array([-0.02, 0.03, 0.5]), "valid_ratio": 1.0, "variance": 0.02, "low_confidence": False},
        "left_eye_outer": {"position": np.array([-0.04, 0.03, 0.5]), "valid_ratio": 1.0, "variance": np.nan, "low_confidence": False},
        "right_eye_inner": {"position": np.array([0.02, 0.03, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": True},
    }

    dst_landmarks = {
        "nose_tip": {"position": np.array([0.0, 0.0, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "chin": {"position": np.array([0.0, -0.08, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "left_eye_inner": {"position": np.array([-0.02, 0.03, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "left_eye_outer": {"position": np.array([-0.04, 0.03, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "right_eye_inner": {"position": np.array([0.02, 0.03, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
    }

    dummy_pcd = o3d.geometry.PointCloud()
    dummy_pcd.points = o3d.utility.Vector3dVector(np.zeros((10, 3)))
    dummy_color = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_depth = np.ones((10, 10), dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(10, 10, 10.0, 10.0, 5.0, 5.0)

    with patch("registration.rigid_transform_from_correspondences", side_effect=spy_rigid), \
         patch("landmarks.detect_face_landmarks_3d", side_effect=[src_landmarks, dst_landmarks]), \
         patch("registration.register_frame_pair", return_value=(np.eye(4), 0.5, 0.001)):

        register_frame_pair_landmark_guided(
            source_pcd=dummy_pcd,
            source_color=dummy_color,
            source_depth=dummy_depth,
            target_pcd=dummy_pcd,
            target_color=dummy_color,
            target_depth=dummy_depth,
            intrinsics_o3d=intr,
            cfg=cfg,
        )

    w = captured_weights.get("weights")
    assert w is not None and len(w) == 5

    # Order of common keys: ['chin', 'left_eye_inner', 'left_eye_outer', 'nose_tip', 'right_eye_inner']
    w_chin = w[0]         # var 1e-4
    w_eye_in = w[1]       # var 0.02
    w_eye_nan = w[2]      # var NaN
    w_nose = w[3]         # var 1e-6
    w_eye_low = w[4]      # var 1e-6, low_confidence = True

    print(f"\n  Production weights: nose={w_nose:.4f}, chin={w_chin:.4f}, high_var={w_eye_in:.4f}, nan_var={w_eye_nan:.4f}, low_conf={w_eye_low:.4f}")

    # 1. Monotonicity: Higher variance MUST have strictly lower weight
    assert w_eye_in < w_chin, f"Monotonicity failed: w(var=0.02)={w_eye_in} should be < w(var=1e-4)={w_chin}"
    assert w_eye_in < 0.001, f"High variance (0.02 m^2) weight must be near zero, got {w_eye_in}"

    # 2. Non-finite variance handling: NaN variance must not receive high weight
    assert w_eye_nan < 0.001, f"Non-finite variance weight must be near zero, got {w_eye_nan}"

    # 3. Low confidence policy: low_confidence must be down-weighted relative to identical confident landmark
    assert w_eye_low < w_nose, f"Low-confidence landmark ({w_eye_low}) must have lower weight than confident landmark ({w_nose})"
    assert np.isclose(w_eye_low, w_nose * 0.5, rtol=1e-3)


def test_production_registration_reduced_correspondences_safety():
    """Verify that fewer than 4 common landmarks falls back to FPFH without crashing."""
    cfg = RegistrationConfig()
    dummy_pcd = o3d.geometry.PointCloud()
    dummy_pcd.points = o3d.utility.Vector3dVector(np.zeros((10, 3)))
    dummy_color = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_depth = np.ones((10, 10), dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(10, 10, 10.0, 10.0, 5.0, 5.0)

    # Only 2 landmarks detected (others rejected as background)
    sparse_lm = {
        "nose_tip": {"position": np.array([0.0, 0.0, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
        "chin": {"position": np.array([0.0, -0.08, 0.5]), "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False},
    }

    with patch("landmarks.detect_face_landmarks_3d", side_effect=[sparse_lm, sparse_lm]), \
         patch("registration.register_frame_pair", return_value=(np.eye(4), 0.5, 0.001)):

        T, fit, rmse, method, res_mm = register_frame_pair_landmark_guided(
            source_pcd=dummy_pcd,
            source_color=dummy_color,
            source_depth=dummy_depth,
            target_pcd=dummy_pcd,
            target_color=dummy_color,
            target_depth=dummy_depth,
            intrinsics_o3d=intr,
            cfg=cfg,
        )

    assert method == "fpfh_fallback"
    assert np.isnan(res_mm)


# ============================================================================
# Test 6: 13-landmark output contract
# ============================================================================
def test_output_compatibility_all_13_landmarks(pinhole_intrinsics):
    w, h = 640, 480
    depth_m = np.full((h, w), 0.55, dtype=np.float32)
    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    coords = {
        name: (320 + i * 5, 240 + i * 5)
        for i, name in enumerate(LANDMARK_INDICES.keys())
    }
    mock_face = _build_mock_face(w, h, coords)

    with patch("landmarks._get_landmarker", return_value=_create_mock_landmarker(mock_face)):
        result = detect_face_landmarks_3d(color_bgr, depth_m, pinhole_intrinsics, quiet=True)

    assert result is not None
    assert len(result) == 13
    assert set(result.keys()) == set(LANDMARK_INDICES.keys())

    for name, lm in result.items():
        assert "position" in lm
        assert "valid_ratio" in lm
        assert "variance" in lm
        assert "low_confidence" in lm
        assert isinstance(lm["position"], np.ndarray)
        assert lm["position"].shape == (3,)
        assert isinstance(lm["valid_ratio"], float)
        assert isinstance(lm["variance"], float)
        assert isinstance(lm["low_confidence"], bool)
        assert 0.0 <= lm["valid_ratio"] <= 1.0
        assert lm["variance"] >= 0.0
        assert np.isclose(lm["position"][2], 0.55)
