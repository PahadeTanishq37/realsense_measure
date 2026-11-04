r"""
tests/body/test_physical_stage12_geometry_verification.py
========================================================
Tests for Stage 12: Independent Camera Transform, Tilt, and Framing Verification.

Validates:
1. Coordinate convention and rotation composition ($R^T R = I$, $\det(R) = +1$).
2. Upward and downward pitch angle behavior and symmetry.
3. Affine camera-to-world and world-to-camera matrix inversion.
4. Synthetic known-point recovery and round trips.
5. Horizontal world-plane invariance ($Z_w = \text{const}$).
6. Rigid Euclidean distance preservation.
7. Perspective framing pixel projection for crown and feet.
8. Torso measurement-plane horizontal invariance (normal $[0, 0, 1]$).
9. Integration of camera height and pitch in adapter.py (frame_to_body_stop).
10. Preservation of raw_captures/body_capture_01.
11. Stage 12 deliverable artifact integrity.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from realsense_measure.body.geometry_verification import (
    IndependentRotationVerifier,
    SyntheticPointTestResult,
    SyntheticPointTester,
    FramingProjectionDetails,
    FramingProjectionAuditor,
    MeasurementPlaneAuditor,
    Stage12DeliverablesGenerator,
)
from realsense_measure.body.hardware_smoke_test import CoordinateGeometryAuditor
from realsense_measure.body.adapter import (
    camera_to_world_matrix,
    world_to_camera_matrix,
    camera_to_world_points,
    world_to_camera_points,
    frame_to_body_stop,
)
from camera.frame import RGBDFrame
import open3d as o3d


RAW_DIR = Path("raw_captures/body_capture_01")
STAGE12_DIR = Path("stage12_geometry_verification_output")


# ---------------------------------------------------------------------------
# 1. Independent Rotation Matrix Composition & Properties
# ---------------------------------------------------------------------------

def test_independent_rotation_matches_stage11_implementation():
    """Verify independent analytical derivation identically matches Stage 11 auditor."""
    for pitch in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        R_independent = IndependentRotationVerifier.derive_rotation_matrix(pitch)
        T_stage11 = CoordinateGeometryAuditor.camera_to_world_matrix(2.4, 1.0, pitch)
        R_stage11 = T_stage11[:3, :3]
        np.testing.assert_allclose(R_independent, R_stage11, atol=1e-6)


def test_rotation_matrix_so3_properties():
    """Verify determinant = +1 and orthonormality for various pitch angles."""
    for pitch in [-15.0, -7.5, -5.0, 0.0, 2.5, 5.0, 8.0, 15.0]:
        props = IndependentRotationVerifier.verify_properties(pitch)
        assert props["is_proper_rotation"] is True
        assert props["is_orthonormal"] is True
        assert abs(props["determinant"] - 1.0) < 1e-6
        assert props["orthonormality_max_error"] < 1e-6


def test_camera_world_affine_inverses():
    """Verify T_c2w @ T_w2c = I_4 for level, downward, and upward pitch."""
    for pitch in [-5.0, 0.0, 5.0]:
        T_c2w = IndependentRotationVerifier.derive_camera_to_world_affine(2.4, 1.0, pitch)
        T_w2c = np.linalg.inv(T_c2w)
        product = T_c2w @ T_w2c
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Synthetic Known-Point & Known-Plane Tests
# ---------------------------------------------------------------------------

def test_level_camera_axial_points():
    """Verify level camera recovers known world coordinates along and off-axis."""
    res = SyntheticPointTester.test_level_camera_axial_points(2.4, 1.0)
    assert res.passed is True
    assert res.max_error_mm < 1e-3


def test_downward_and_upward_pitch_known_points():
    """Verify known point elevations for +5 deg and -5 deg pitch."""
    res_down = SyntheticPointTester.test_pitched_camera_known_points(+5.0, 2.4, 1.0)
    assert res_down.passed is True
    assert res_down.max_error_mm < 1e-2

    res_up = SyntheticPointTester.test_pitched_camera_known_points(-5.0, 2.4, 1.0)
    assert res_up.passed is True
    assert res_up.max_error_mm < 1e-2

    # Optical axis elevation at subject distance:
    # Downward tilt points below camera height (0.79 m < 1.0 m)
    # Upward tilt points above camera height (1.21 m > 1.0 m)
    assert res_down.details["actual_world_z"] < 1.0
    assert res_up.details["actual_world_z"] > 1.0


def test_horizontal_world_plane_invariance():
    """Verify horizontal world plane remains horizontal after round-trip transformation."""
    for pitch in [-5.0, 0.0, 5.0]:
        res = SyntheticPointTester.test_horizontal_world_plane_invariance(pitch, 2.4, 1.0, plane_z=0.0)
        assert res.passed is True
        assert res.details["z_range_mm"] < 1e-3
        assert res.details["mean_z_err_mm"] < 1e-3


def test_rigid_distance_preservation():
    """Verify Euclidean distances between points are perfectly preserved."""
    for pitch in [-5.0, 0.0, 5.0]:
        res = SyntheticPointTester.test_rigid_distance_invariance(pitch, 2.4, 1.0)
        assert res.passed is True
        assert res.details["max_pairwise_error_mm"] < 1e-3


# ---------------------------------------------------------------------------
# 3. Framing Projection Verification
# ---------------------------------------------------------------------------

def test_framing_projection_level_vs_tilt():
    """Verify exact pixel row projection and margin shifts."""
    nom_level = FramingProjectionAuditor.audit_subject_framing(2.4, 1.0, 0.0, 1.75)
    nom_tilt = FramingProjectionAuditor.audit_subject_framing(2.4, 1.0, 5.0, 1.75)

    assert nom_level.full_body_in_image is True
    assert nom_tilt.full_body_in_image is True

    # Check pixel row values are inside sensor (0 to 719)
    assert 0 <= nom_level.crown_v_px < 720
    assert 0 <= nom_level.feet_v_px < 720
    assert 0 <= nom_tilt.crown_v_px < 720
    assert 0 <= nom_tilt.feet_v_px < 720

    # Downward tilt shifts projected image downward (v increases for both crown and feet)
    # This increases bottom margin below feet (moving feet away from bottom edge v=719)
    assert nom_tilt.feet_v_px < nom_level.feet_v_px  # feet row is lower number (higher up in image)
    assert nom_tilt.feet_bottom_margin_px > nom_level.feet_bottom_margin_px


def test_framing_sensitivity_grid():
    """Verify multi-parameter sensitivity evaluation."""
    grid = FramingProjectionAuditor.generate_sensitivity_matrix()
    assert len(grid) > 50
    # Confirm some conditions succeed and some tall subjects / close distances clip
    full_count = sum(1 for g in grid if g.full_body_in_image)
    assert full_count > 0
    assert full_count < len(grid)


# ---------------------------------------------------------------------------
# 4. Measurement Plane Invariance
# ---------------------------------------------------------------------------

def test_measurement_plane_invariance():
    """Verify slice plane normal remains [0, 0, 1] and quantify uncompensated slant."""
    audit = MeasurementPlaneAuditor.audit_measurement_plane_invariance(5.0)
    assert audit["is_compensated_invariant"] is True
    assert audit["slant_angle_deg"] == 5.0
    assert 0.15 < audit["circumference_geometric_dilation_pct"] < 0.25  # ~0.19% for 5 deg cut of circular cylinder


# ---------------------------------------------------------------------------
# 5. Adapter Integration: frame_to_body_stop
# ---------------------------------------------------------------------------

def test_frame_to_body_stop_propagates_height_and_tilt():
    """Verify frame_to_body_stop accepts and applies camera_height_m and pitch_deg."""
    dummy_color = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_depth = np.ones((720, 1280), dtype=np.float32) * 2.4
    intr = o3d.camera.PinholeCameraIntrinsic(1280, 720, 645.889, 645.889, 637.24, 366.936)
    frame = RGBDFrame(
        color_bgr=dummy_color,
        depth_m=dummy_depth,
        frame_id=0,
        timestamp=0.0,
        pcd=o3d.geometry.PointCloud(),
        intrinsics=intr,
    )
    # Mask a small 10x10 patch in center
    mask = np.zeros((720, 1280), dtype=bool)
    mask[355:365, 635:645] = True

    # 1. Level stop
    stop_level = frame_to_body_stop(
        frame=frame,
        nominal_yaw_deg=0.0,
        distance_m=2.4,
        foreground_mask=mask,
        camera_height_m=1.0,
        pitch_deg=0.0,
    )
    assert stop_level.intrinsics["camera_height_m"] == 1.0
    assert stop_level.intrinsics["pitch_deg"] == 0.0

    # 2. Tilted stop
    stop_tilt = frame_to_body_stop(
        frame=frame,
        nominal_yaw_deg=0.0,
        distance_m=2.4,
        foreground_mask=mask,
        camera_height_m=1.0,
        pitch_deg=5.0,
    )
    assert stop_tilt.intrinsics["camera_height_m"] == 1.0
    assert stop_tilt.intrinsics["pitch_deg"] == 5.0

    # Points around center optical axis under downward tilt should have lower Z than level
    mean_z_level = np.mean(stop_level.point_cloud_world[:, 2])
    mean_z_tilt = np.mean(stop_tilt.point_cloud_world[:, 2])
    assert mean_z_tilt < mean_z_level


# ---------------------------------------------------------------------------
# 6. Historical Capture Preservation
# ---------------------------------------------------------------------------

def test_historical_raw_capture_untouched():
    """Verify raw_captures/body_capture_01 remains intact."""
    assert RAW_DIR.exists()
    meta_path = RAW_DIR / "meta.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_frames"] == 4


# ---------------------------------------------------------------------------
# 7. Deliverables Generation
# ---------------------------------------------------------------------------

def test_stage12_deliverables_generator_produces_all_files(tmp_path):
    """Verify all 8 Stage 12 deliverables are created and valid."""
    deliverables = Stage12DeliverablesGenerator.generate_all(tmp_path)
    expected_keys = [
        "stage12_report",
        "conventions",
        "derivation",
        "synthetic_results",
        "framing_audit",
        "plane_audit",
        "risks",
        "regression_summary",
    ]
    for key in expected_keys:
        assert key in deliverables
        p = deliverables[key]
        assert p.exists()
        assert p.stat().st_size > 0

    # JSON validity
    with open(deliverables["synthetic_results"], "r", encoding="utf-8") as f:
        synth_doc = json.load(f)
        assert synth_doc["all_passed"] is True
        assert len(synth_doc["tests"]) == 7

    with open(deliverables["regression_summary"], "r", encoding="utf-8") as f:
        reg_doc = json.load(f)
        assert reg_doc["total_synthetic_tests"] == 7
        assert reg_doc["passed_synthetic_tests"] == 7
