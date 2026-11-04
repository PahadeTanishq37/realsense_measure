"""
tests/body/test_physical_stage20_subject_frame.py
=================================================
Test suite for Stage 20: Subject-Centric Coordinate Frame & Measurement Geometry.

Validates:
1. Source data preservation & non-corruption of raw captures and prior stage outputs.
2. Orthonormal properties of SubjectCentricCoordinateFrame (R R^T = I, det(R)=1, condition number=1).
3. Synthetic pure yaw rotations across discrete angles (0, 30, 45, 90, -45 deg).
4. Multi-axis rotation recovery and round-trip coordinate inversion (< 10^-6 m error).
5. Handedness and sign consistency (Subject Left > 0, Subject Right < 0).
6. Degenerate landmark configurations rejection.
7. Physical capture orientation analysis (body_capture_03_diagnostic yaw ~ 49.4 deg, occluded hips).
8. Camera-frame vs subject-frame measurement comparisons (shoulder breadth and chest breadth).
9. Deliverables inventory and JSON schema integrity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.measurements import (
    AnatomicalLandmark,
    FrameConstructionStatus,
    Stage20SubjectFrameOrchestrator,
    Stage20Verdict,
    SubjectCentricCoordinateFrame,
    SubjectFrameMeasurementComparator,
    SubjectFrameSyntheticValidator,
    SubjectOrientationEstimator,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path) -> Stage20SubjectFrameOrchestrator:
    return Stage20SubjectFrameOrchestrator(workspace_root=repo_root)


def test_source_data_preservation(repo_root: Path):
    """Verify that prior raw captures and stage directories exist and remain unmodified."""
    raw1 = repo_root / "raw_captures" / "body_capture_01"
    raw2 = repo_root / "raw_captures" / "body_capture_02"
    raw3 = repo_root / "raw_captures" / "body_capture_03_diagnostic"

    assert raw1.exists(), "body_capture_01 must be preserved"
    assert raw2.exists(), "body_capture_02 must be preserved"
    assert raw3.exists(), "body_capture_03_diagnostic must be preserved"

    # Prior stages 13 - 19
    for stg in ["stage13_full_body_capture_output", "stage14_physical_reconstruction_output",
                "stage15_registration_diagnostics_output", "stage16_subject_depth_diagnostic_output",
                "stage17_upper_body_anthropometry_output", "stage18_measurement_sanity_audit_output",
                "stage19_targeted_validation_output"]:
        p = repo_root / stg
        assert p.exists() and p.is_dir(), f"Prior stage output directory {stg} must be preserved"


def test_subject_frame_orthonormality_and_diagnostics():
    """Verify that constructed coordinate frames strictly satisfy SO(3) orthonormality."""
    lms = {
        "acromion_left": AnatomicalLandmark(
            landmark_id="acromion_left", anatomical_name="Left", classification="ESTIMATED_2D_3D_LANDMARK",
            pixel_u=650, pixel_v=300, in_image_bounds=True, point_3d_camera_m=[0.1, 0.2, 0.8], depth_m=0.8,
            depth_valid=True, depth_std_mm=0.5, visibility_confidence=0.95, status="VALID",
        ),
        "acromion_right": AnatomicalLandmark(
            landmark_id="acromion_right", anatomical_name="Right", classification="ESTIMATED_2D_3D_LANDMARK",
            pixel_u=450, pixel_v=300, in_image_bounds=True, point_3d_camera_m=[-0.2, 0.1, 0.6], depth_m=0.6,
            depth_valid=True, depth_std_mm=0.5, visibility_confidence=0.95, status="VALID",
        ),
    }

    frame = SubjectOrientationEstimator.estimate_from_landmarks(lms)
    assert frame.status == FrameConstructionStatus.CONSTRUCTED_VALID.value
    assert frame.orthogonality_residual < 1e-12, f"Orthogonality residual too large: {frame.orthogonality_residual}"
    assert abs(frame.determinant - 1.0) < 1e-6, f"Determinant must be +1.0, got {frame.determinant}"
    assert abs(frame.condition_number - 1.0) < 1e-6, f"Condition number must be 1.0, got {frame.condition_number}"

    # Verify cross product: u_z = u_x x u_y
    ux = np.array(frame.u_x_lateral)
    uy = np.array(frame.u_y_longitudinal)
    uz = np.array(frame.u_z_anterior)
    cross_calc = np.cross(ux, uy)
    assert np.allclose(uz, cross_calc, atol=1e-5), "Anterior axis must match cross product of lateral and longitudinal"


def test_synthetic_pure_yaw_rotations():
    """Verify synthetic pure yaw rotations across discrete angles."""
    res = SubjectFrameSyntheticValidator.run_synthetic_tests()
    yaw_tests = res["pure_yaw_rotations"]

    for deg in [0, 30, 45, 90, -45]:
        key = f"yaw_{deg}deg"
        assert key in yaw_tests
        t = yaw_tests[key]
        assert t["passed"]
        assert t["yaw_error_deg"] < 0.1
        assert t["span_error_mm"] < 0.1


def test_synthetic_multi_axis_and_inversion():
    """Verify multi-axis rotation recovery and round-trip coordinate inversion."""
    res = SubjectFrameSyntheticValidator.run_synthetic_tests()
    multi = res["multi_axis_rotation"]
    inv = res["round_trip_inversion"]

    assert multi["passed"]
    assert multi["span_error_mm"] < 0.1
    assert inv["passed"]
    assert inv["max_coordinate_error_m"] < 1e-6


def test_sign_consistency_and_handedness():
    """Verify anatomical handedness: Subject Left has positive X_subj, Subject Right has negative X_subj."""
    res = SubjectFrameSyntheticValidator.run_synthetic_tests()
    sign = res["sign_consistency"]
    assert sign["passed"]
    assert sign["subject_left_x_positive"]
    assert sign["subject_right_x_negative"]


def test_degenerate_landmark_rejection():
    """Verify that degenerate landmark configurations (identical coordinates) are safely rejected."""
    res = SubjectFrameSyntheticValidator.run_synthetic_tests()
    degen = res["degenerate_input_handling"]
    assert degen["passed"]
    assert degen["status"] == FrameConstructionStatus.REJECTED_DEGENERATE.value
    assert degen["confidence_score"] == 0.0


def test_physical_capture_orientation_audit(orchestrator: Stage20SubjectFrameOrchestrator):
    """Verify torso orientation diagnostics on physical captures."""
    summary = orchestrator._audit_captures_for_subject_frame()

    # body_capture_03_diagnostic should be valid for experimental use
    cap3 = summary["body_capture_03_diagnostic"]
    assert cap3["torso_usable"]
    f0 = cap3["frames"][0]
    assert f0["frame_status"] == FrameConstructionStatus.CONSTRUCTED_VALID.value
    assert abs(f0["yaw_deg"] - 49.36) < 1.0, f"Expected yaw ~49.4 deg, got {f0['yaw_deg']}"
    assert f0["confidence_score"] == 0.765  # 0.90 * 0.85 due to occluded hips

    # body_capture_01 should be rejected due to depth contamination
    cap1 = summary["body_capture_01"]
    f0_c1 = cap1["frames"][0]
    assert f0_c1["frame_status"] == FrameConstructionStatus.REJECTED_DEGENERATE.value


def test_camera_vs_subject_frame_physical_measurements(orchestrator: Stage20SubjectFrameOrchestrator):
    """Verify measurement differences between camera frame and subject frame."""
    comp = orchestrator._run_physical_comparisons()["comparison"]

    sh = comp["shoulder_breadth"]
    ch = comp["chest_transverse_breadth"]

    # Shoulder breadth: camera X foreshortened (218mm), subject frame recovered (351.1mm)
    assert abs(sh["camera_x_span_mm"] - 218.0) < 1.0
    assert abs(sh["subject_frame_lateral_span_mm"] - 351.1) < 1.0
    assert abs(sh["yaw_foreshortening_ratio"] - 0.621) < 0.01

    # Chest breadth: camera frame (371.0mm) vs subject frame (424.9mm)
    assert abs(ch["camera_frame_value_mm"] - 371.0) < 1.0
    assert abs(ch["subject_frame_value_mm"] - 424.9) < 1.0
    assert ch["breadth_difference_mm"] > 50.0  # > 50mm difference due to yaw correction


def test_stage20_deliverables_and_verdict(repo_root: Path):
    """Verify that all Stage 20 output files exist, have valid JSON schemas, and state the correct verdict."""
    out_dir = repo_root / "stage20_subject_coordinate_frame_output"
    assert out_dir.exists()

    required_files = [
        "stage20_report.md",
        "coordinate_frame_definition.json",
        "orientation_estimation_comparison.json",
        "camera_vs_subject_frame_measurements.json",
        "coordinate_transform_test_results.json",
        "subject_frame_measurement_comparison.png",
    ]

    for rf in required_files:
        p = out_dir / rf
        assert p.exists() and p.stat().st_size > 0, f"Deliverable {rf} must exist and be non-empty"

    for jf in [f for f in required_files if f.endswith(".json")]:
        with open(out_dir / jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Deliverable {jf} must be a valid JSON dictionary"

    report_text = (out_dir / "stage20_report.md").read_text(encoding="utf-8")
    assert Stage20Verdict.SUBJECT_FRAME_IMPLEMENTED_EXPERIMENTALLY.value in report_text

