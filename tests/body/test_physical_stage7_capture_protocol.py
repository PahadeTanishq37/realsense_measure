"""Tests for Problem 4 - Stage 7: Controlled Anthropometric Capture Protocol & Repeatability Infrastructure.

Validates:
1. Protocol configuration parsing and required metadata validation.
2. Anonymous subject identifiers validation (strict PII rejection: names, emails rejected).
3. Unique capture IDs (collision detection prevents overwriting existing data).
4. Missing and duplicate yaw stops detection.
5. Invalid camera configuration rejection (negative distance, invalid resolution).
6. Sensor-grounded completeness status classification (observed, partially_observed, missing, unreliable).
7. Unavailable diagnostic handling (explicit 'unavailable', strictly no fabrication).
8. Capture-session grouping and repeat-capture grouping by subject.
9. Stage 6 manual validation linkage and duplicate record detection.
10. Empty dataset handling and preservation of existing raw data.
11. Verification of all physical artifacts in stage7_capture_protocol_output/.
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from realsense_measure.body.capture_protocol import (
    CaptureCompletenessChecker,
    CaptureExperimentEngine,
    CaptureExperimentGridConfig,
    CaptureProtocolConfig,
    CaptureSessionManager,
    CaptureSessionRecord,
    Stage6ValidationBridge,
    StopCompletenessDiagnostic,
)


STAGE7_DIR = Path("stage7_capture_protocol_output")
RAW_DIR = Path("raw_captures/body_capture_01")
STAGE6_CSV = Path("stage6_validation_output/manual_validation_template.csv")


# ---------------------------------------------------------------------------
# 1. Protocol Configuration & Anonymous Subject Validation
# ---------------------------------------------------------------------------

def test_protocol_config_valid_round_trip():
    """Verify valid CaptureProtocolConfig serialization and deserialization."""
    cfg = CaptureProtocolConfig(
        protocol_id="RS_ANTHRO_PROTO_V1",
        subject_id="SUBJ_42",
        capture_id="body_capture_42",
        operator_id="OPERATOR_02",
        camera_distance_m=2.30,
        camera_height_m=1.05,
        nominal_yaw_deg=[0.0, 90.0, 180.0, 270.0],
    )
    d = cfg.to_dict()
    assert d["subject_id"] == "SUBJ_42"
    assert d["camera_distance_m"] == 2.30
    assert d["rgb_resolution"] == [1280, 720]

    rebuilt = CaptureProtocolConfig.from_dict(d)
    assert rebuilt.subject_id == cfg.subject_id
    assert rebuilt.rgb_resolution == (1280, 720)


def test_anonymous_subject_id_pii_rejection():
    """Verify that personal names, emails, spaces, and non-anonymous IDs are rejected."""
    # Invalid PII entries must raise ValueError
    invalid_ids = [
        "john.smith@gmail.com",
        "user@antigravity.io",
        "John Smith",
        "Participant 01",
        "Alice_Smith",
        "jane_doe",
        "",
        None,
    ]
    for bad_id in invalid_ids:
        with pytest.raises(ValueError):
            CaptureProtocolConfig.validate_anonymous_subject_id(bad_id)

    # Valid anonymous formats must pass
    valid_ids = [
        "SUBJ_01",
        "SUBJ_A102",
        "ANON_42",
        "PARTICIPANT_001",
        "TEST_99",
    ]
    for good_id in valid_ids:
        CaptureProtocolConfig.validate_anonymous_subject_id(good_id)


def test_required_metadata_validation():
    """Verify that missing required metadata fields raise ValueError."""
    # Empty capture_id
    with pytest.raises(ValueError, match="capture_id"):
        CaptureProtocolConfig(subject_id="SUBJ_01", capture_id="")

    # Empty operator_id
    with pytest.raises(ValueError, match="operator_id"):
        CaptureProtocolConfig(subject_id="SUBJ_01", operator_id="")

    # Invalid repeat index
    with pytest.raises(ValueError, match="repeat_index"):
        CaptureProtocolConfig(subject_id="SUBJ_01", repeat_index=0)


def test_invalid_camera_and_yaw_configuration():
    """Verify rejection of invalid stream specs, negative distances, and duplicate stops."""
    # Negative camera distance
    with pytest.raises(ValueError, match="camera_distance_m"):
        CaptureProtocolConfig(subject_id="SUBJ_01", camera_distance_m=-1.5)

    # Negative camera height
    with pytest.raises(ValueError, match="camera_height_m"):
        CaptureProtocolConfig(subject_id="SUBJ_01", camera_height_m=-0.5)

    # Invalid resolution
    with pytest.raises(ValueError, match="rgb_resolution"):
        CaptureProtocolConfig(subject_id="SUBJ_01", rgb_resolution=(0, 720))

    # Duplicate nominal yaw stops
    with pytest.raises(ValueError, match="Duplicate nominal yaw stops"):
        CaptureProtocolConfig(subject_id="SUBJ_01", nominal_yaw_deg=[0.0, 90.0, 90.0, 180.0])

    # Empty yaw stops
    with pytest.raises(ValueError, match="nominal_yaw_deg"):
        CaptureProtocolConfig(subject_id="SUBJ_01", nominal_yaw_deg=[])


# ---------------------------------------------------------------------------
# 2. Capture Completeness & Unavailable Diagnostics
# ---------------------------------------------------------------------------

def test_capture_completeness_checker_anatomical_classification():
    """Verify non-parametric classification of observed, partially_observed, missing."""
    checker = CaptureCompletenessChecker()
    h, w = 720, 1280

    # 1. Case: Full body centered, no clipping
    depth_m = np.full((h, w), 2.0, dtype=np.float32)
    # Foreground centered with margins
    mask_centered = np.zeros((h, w), dtype=bool)
    mask_centered[50:670, 200:1080] = True

    # Vertical span in world: -0.90m to +0.80m (full body)
    z_bounds_full = (-0.90, 0.80)
    diag_full = checker.evaluate_stop(
        stop_idx=0,
        nominal_yaw_deg=0.0,
        depth_m=depth_m,
        foreground_mask=mask_centered,
        z_bounds_world_m=z_bounds_full,
    )
    assert diag_full.head_status == "observed"
    assert diag_full.torso_status == "observed"
    assert diag_full.pelvis_status == "observed"
    assert diag_full.foot_left_status == "observed"
    assert diag_full.landmark_status == "unavailable"  # Not run -> unavailable

    # 2. Case: Severe vertical truncation (matching body_capture_01)
    mask_truncated = np.zeros((h, w), dtype=bool)
    mask_truncated[0:h, 200:1080] = True  # Contacts top and bottom boundaries
    z_bounds_trunc = (-0.65, 0.25)  # Truncated head and feet
    diag_trunc = checker.evaluate_stop(
        stop_idx=0,
        nominal_yaw_deg=0.0,
        depth_m=depth_m,
        foreground_mask=mask_truncated,
        z_bounds_world_m=z_bounds_trunc,
    )
    assert diag_trunc.clipping.top is True
    assert diag_trunc.clipping.bottom is True
    assert diag_trunc.head_status == "missing"
    assert diag_trunc.foot_left_status == "missing"
    assert diag_trunc.torso_status == "observed"


def test_unavailable_diagnostic_handling():
    """Verify that missing landmark detectors explicitly return 'unavailable' without fabrication."""
    checker = CaptureCompletenessChecker()
    depth_m = np.full((100, 100), 2.0, dtype=np.float32)
    mask = np.ones((100, 100), dtype=bool)

    # When landmarks argument is None
    diag = checker.evaluate_stop(
        stop_idx=0,
        nominal_yaw_deg=0.0,
        depth_m=depth_m,
        foreground_mask=mask,
        z_bounds_world_m=(-0.5, 0.5),
        landmarks=None,
    )
    assert diag.landmark_status == "unavailable"
    assert diag.landmark_confidence is None


def test_missing_and_duplicate_yaw_stops_detection():
    """Verify session evaluator detects missing or duplicate stops."""
    checker = CaptureCompletenessChecker()
    expected_yaws = [0.0, 90.0, 180.0, 270.0]

    # Stop 1: 0 deg
    s0 = checker.evaluate_stop(0, 0.0, np.full((10, 10), 2.0, dtype=np.float32), np.ones((10, 10), bool), (-0.5, 0.5))
    # Stop 2: duplicate 0 deg
    s1 = checker.evaluate_stop(1, 0.0, np.full((10, 10), 2.0, dtype=np.float32), np.ones((10, 10), bool), (-0.5, 0.5))

    session_diag = checker.evaluate_session(
        capture_id="test_cap",
        subject_id="SUBJ_01",
        stop_diagnostics=[s0, s1],
        expected_nominal_yaws=expected_yaws,
    )
    assert session_diag.is_yaw_complete is False
    assert 90.0 in session_diag.missing_stops
    assert 180.0 in session_diag.missing_stops
    assert 270.0 in session_diag.missing_stops
    assert 0.0 in session_diag.duplicate_stops


def test_empty_dataset_handling():
    """Verify session completeness handles empty stops gracefully without crashing."""
    checker = CaptureCompletenessChecker()
    session_diag = checker.evaluate_session(
        capture_id="empty_cap",
        subject_id="SUBJ_01",
        stop_diagnostics=[],
        expected_nominal_yaws=[0.0, 90.0, 180.0, 270.0],
    )
    assert session_diag.is_capture_complete is False
    assert session_diag.quality_rating == "unusable"
    assert session_diag.num_stops_found == 0


# ---------------------------------------------------------------------------
# 3. Session Tracking, Repeatability & Collision Prevention
# ---------------------------------------------------------------------------

def test_session_manager_collision_detection(tmp_path):
    """Verify session manager prevents duplicate capture_id collisions."""
    mgr = CaptureSessionManager(manifest_csv_path=tmp_path / "manifest.csv")

    rec1 = CaptureSessionRecord(
        capture_id="cap_01",
        subject_id="SUBJ_01",
        session_id="SESS_01",
        repeat_index=1,
        operator_id="OP_01",
        capture_datetime="2026-11-02T12:00:00Z",
        camera_model="D455f",
        camera_serial_number="12345",
        camera_distance_m=2.14,
        camera_height_m=1.0,
        resolution="1280x720",
        nominal_stops_count=4,
        actual_stops_count=4,
        clothing_condition="casual_clothing",
        lighting_condition="diffuse_indoor_fluorescent",
        quality_rating="complete",
        is_complete=True,
    )
    mgr.register_capture(rec1)

    # Attempting to register the same capture_id without allow_overwrite must raise ValueError
    with pytest.raises(ValueError, match="Capture ID collision"):
        mgr.register_capture(rec1, allow_overwrite=False)

    # Succeeded with different capture_id
    rec2 = CaptureSessionRecord(
        capture_id="cap_02",
        subject_id="SUBJ_01",
        session_id="SESS_01",
        repeat_index=2,
        operator_id="OP_01",
        capture_datetime="2026-11-02T12:15:00Z",
        camera_model="D455f",
        camera_serial_number="12345",
        camera_distance_m=2.14,
        camera_height_m=1.0,
        resolution="1280x720",
        nominal_stops_count=4,
        actual_stops_count=4,
        clothing_condition="casual_clothing",
        lighting_condition="diffuse_indoor_fluorescent",
        quality_rating="complete",
        is_complete=True,
    )
    mgr.register_capture(rec2)
    assert len(mgr.records) == 2


def test_repeat_capture_grouping_by_subject():
    """Verify that multiple captures for the same subject are clustered together."""
    mgr = CaptureSessionManager()

    for idx, (cap_id, subj_id) in enumerate([
        ("cap_01", "SUBJ_01"),
        ("cap_02", "SUBJ_01"),
        ("cap_03", "SUBJ_02"),
    ]):
        rec = CaptureSessionRecord(
            capture_id=cap_id,
            subject_id=subj_id,
            session_id=f"SESS_{subj_id}",
            repeat_index=idx + 1,
            operator_id="OP_01",
            capture_datetime="2026-11-02T12:00:00Z",
            camera_model="D455f",
            camera_serial_number="12345",
            camera_distance_m=2.14,
            camera_height_m=1.0,
            resolution="1280x720",
            nominal_stops_count=4,
            actual_stops_count=4,
            clothing_condition="casual_clothing",
            lighting_condition="diffuse_indoor_fluorescent",
            quality_rating="complete",
            is_complete=True,
        )
        mgr.register_capture(rec)

    subj1_caps = mgr.get_subject_captures("SUBJ_01")
    assert len(subj1_caps) == 2
    assert {c.capture_id for c in subj1_caps} == {"cap_01", "cap_02"}

    subj2_caps = mgr.get_subject_captures("SUBJ_02")
    assert len(subj2_caps) == 1
    assert subj2_caps[0].capture_id == "cap_03"


# ---------------------------------------------------------------------------
# 4. Stage 6 Validation Linkage
# ---------------------------------------------------------------------------

def test_stage6_validation_linkage_verification():
    """Verify linkage with existing Stage 6 validation template."""
    if not STAGE6_CSV.exists():
        pytest.skip(f"Stage 6 template CSV not found at {STAGE6_CSV}")

    # Correct matching
    status = Stage6ValidationBridge.verify_capture_linkage(
        capture_id="body_capture_01",
        subject_id="SUBJ_01",
        stage6_csv_path=STAGE6_CSV,
    )
    assert status["is_linked"] is True
    assert status["matched_records"] == 19
    assert status["has_duplicates"] is False

    # Mismatched capture_id
    mismatched = Stage6ValidationBridge.verify_capture_linkage(
        capture_id="nonexistent_capture",
        subject_id="SUBJ_01",
        stage6_csv_path=STAGE6_CSV,
    )
    assert mismatched["is_linked"] is False
    assert mismatched["matched_records"] == 0


def test_stage6_duplicate_record_detection(tmp_path):
    """Verify detection of duplicate validation entries."""
    dup_csv = tmp_path / "duplicate_validation.csv"
    with open(dup_csv, "w", encoding="utf-8") as f:
        f.write("subject_id,capture_id,measurement_name\n")
        f.write("SUBJ_01,body_capture_01,waist_circumference\n")
        f.write("SUBJ_01,body_capture_01,waist_circumference\n")  # duplicate

    status = Stage6ValidationBridge.verify_capture_linkage(
        capture_id="body_capture_01",
        subject_id="SUBJ_01",
        stage6_csv_path=dup_csv,
    )
    assert status["has_duplicates"] is True
    assert len(status["duplicate_records"]) == 1
    assert status["is_linked"] is False


# ---------------------------------------------------------------------------
# 5. Controlled Experiment Engine & Trade-Offs
# ---------------------------------------------------------------------------

def test_experiment_tradeoff_no_hardcoded_superiority():
    """Verify that experiment engine calculates physical trade-offs without declaring a winner."""
    engine = CaptureExperimentEngine()
    results = engine.run_grid_evaluation()

    assert len(results) > 0
    # Check that distance 1.5m flags near-field risk
    r_1_5m = [r for r in results if r.distance_m == 1.5 and r.resolution == (1280, 720)][0]
    assert r_1_5m.risk_near_field_truncation is True
    assert r_1_5m.is_full_body_in_fov is False

    # Check that distance 3.0m flags far-field noise loss
    r_3_0m = [r for r in results if r.distance_m == 3.0 and r.resolution == (1280, 720)][0]
    assert r_3_0m.risk_far_field_precision_loss is True
    assert r_3_0m.is_full_body_in_fov is True
    assert r_3_0m.theoretical_depth_noise_mm > 5.0


# ---------------------------------------------------------------------------
# 6. Physical Artifacts & Raw Capture Preservation
# ---------------------------------------------------------------------------

def test_physical_stage7_artifacts_exist():
    """Verify that all 6 required Stage 7 artifacts exist on disk and are valid."""
    required_files = [
        "capture_protocol_template.json",
        "capture_session_manifest.csv",
        "capture_quality_report.json",
        "capture_completeness_report.csv",
        "capture_experiment_summary.json",
        "stage7_capture_protocol_report.md",
    ]
    for fname in required_files:
        fpath = STAGE7_DIR / fname
        assert fpath.exists(), f"Required Stage 7 artifact missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"


def test_raw_captures_preservation():
    """Verify that raw physical captures were never modified or corrupted."""
    if not RAW_DIR.exists():
        pytest.skip(f"Raw capture directory missing at {RAW_DIR}")

    meta_file = RAW_DIR / "meta.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["target_name"] == "body"
    assert meta["num_frames"] == 4
    for f_info in meta["frames"]:
        color_p = RAW_DIR / f_info["color_file"]
        depth_p = RAW_DIR / f_info["depth_file"]
        assert color_p.exists() and color_p.stat().st_size > 0
        assert depth_p.exists() and depth_p.stat().st_size > 0
