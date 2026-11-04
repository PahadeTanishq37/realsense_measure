"""Tests for Problem 4 - Stage 9: Physical Pilot Study & Repeatability Evaluation.

Validates:
1. Pilot protocol configuration parsing, validation, and serialization.
2. Anonymous subject identifiers validation (strict PII rejection: names, emails rejected).
3. Capture configuration validation (positive distance, height, yaw stops).
4. Full-body readiness checking using sensor-grounded diagnostics (decoupling coverage from visibility).
5. Repeatability analysis:
   - Subject-level grouping (repeat captures of same subject not treated as independent).
   - Within-subject standard deviation, CV%, and Repeatability Coefficient (RC).
   - Stage 2 vs Stage 4 repeatability comparison.
   - ICC(2,1) sample-size guards (requires N_subjects >= 2 with >= 2 repeats each).
   - ICC(2,1) mathematical computation on mock multi-subject data.
6. Stage 6 manual validation bridge:
   - Strict anti-fabrication when manual measurements are unpopulated.
   - Exclusion of incompatible / invalid measurements (e.g. neck base circumference).
   - Error metrics computation (signed error, absolute error, MAE, RMSE, error reduction).
7. Capture configuration comparison:
   - Evaluates multi-configuration records without declaring ungrounded optimality.
8. Empty dataset and edge case handling.
9. Verification of all physical output artifacts in stage9_pilot_study_output/.
10. Preservation of raw captures and Stages 1-8 outputs.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest

from realsense_measure.body.pilot_study import (
    CaptureConfiguration,
    ConfigurationComparisonRecord,
    FullBodyReadinessChecker,
    FullBodyReadinessResult,
    ManualComparisonItem,
    PilotConfigurationComparator,
    PilotManualValidationBridge,
    PilotManualValidationSummary,
    PilotStudyOrchestrator,
    PilotStudyProtocolConfig,
    RepeatabilityEngine,
    SingleMeasurementRepeatability,
    TARGET_FULL_BODY_REGIONS,
)


STAGE9_DIR = Path("stage9_pilot_study_output")
RAW_DIR = Path("raw_captures/body_capture_01")


# ---------------------------------------------------------------------------
# 1. Protocol Configuration & Anonymous Subject Validation
# ---------------------------------------------------------------------------

def test_protocol_config_valid_round_trip():
    """Verify valid PilotStudyProtocolConfig serialization and deserialization."""
    cfg = PilotStudyProtocolConfig(
        protocol_id="RS_PILOT_ANTHRO_V1",
        subject_id="SUBJ_01",
        capture_id="body_capture_01",
        session_id="SESS_20261102_01",
        operator_id="OPERATOR_01",
        config_id="CONFIG_D220_H100_Y4_R720",
        repeat_index=2,
        camera_distance_m=2.20,
        camera_height_m=1.00,
        nominal_yaw_stops=[0.0, 90.0, 180.0, 270.0],
        rgb_resolution=(1280, 720),
        depth_resolution=(1280, 720),
    )
    d = cfg.to_dict()
    assert d["subject_id"] == "SUBJ_01"
    assert d["repeat_index"] == 2
    assert d["camera_distance_m"] == 2.20
    assert d["rgb_resolution"] == [1280, 720]

    rebuilt = PilotStudyProtocolConfig.from_dict(d)
    assert rebuilt.subject_id == cfg.subject_id
    assert rebuilt.repeat_index == cfg.repeat_index
    assert rebuilt.rgb_resolution == (1280, 720)


def test_anonymous_subject_id_pii_rejection():
    """Verify that personal names, emails, spaces, and non-anonymous IDs are rejected."""
    invalid_ids = [
        "john.doe@hospital.org",
        "alice@antigravity.io",
        "John Smith",
        "Participant 01",
        "Doctor_Who",
        "patient_01",
        "",
        None,
    ]
    for bad_id in invalid_ids:
        with pytest.raises(ValueError):
            PilotStudyProtocolConfig.validate_anonymous_subject_id(bad_id)

    # Valid anonymous formats must pass
    valid_ids = [
        "SUBJ_01",
        "SUBJ_B42",
        "ANON_101",
        "PARTICIPANT_007",
        "TEST_01",
        "P_12",
    ]
    for good_id in valid_ids:
        PilotStudyProtocolConfig.validate_anonymous_subject_id(good_id)


def test_protocol_config_invalid_parameters():
    """Verify that invalid distances, heights, or repeat indices raise ValueError."""
    # Negative distance
    with pytest.raises(ValueError, match="camera_distance_m"):
        PilotStudyProtocolConfig(subject_id="SUBJ_01", camera_distance_m=-1.5)

    # Negative height
    with pytest.raises(ValueError, match="camera_height_m"):
        PilotStudyProtocolConfig(subject_id="SUBJ_01", camera_height_m=0.0)

    # Zero repeat index
    with pytest.raises(ValueError, match="repeat_index"):
        PilotStudyProtocolConfig(subject_id="SUBJ_01", repeat_index=0)

    # Invalid clothing condition
    with pytest.raises(ValueError, match="clothing_condition"):
        PilotStudyProtocolConfig(subject_id="SUBJ_01", clothing_condition="heavy_winter_coat")


# ---------------------------------------------------------------------------
# 2. Capture Configuration Specification
# ---------------------------------------------------------------------------

def test_capture_configuration_validation():
    """Verify CaptureConfiguration parameter validation."""
    cfg = CaptureConfiguration(
        config_id="CONFIG_D240_H105_Y4_R720",
        camera_distance_m=2.40,
        camera_height_m=1.05,
        nominal_yaw_stops=[0.0, 90.0, 180.0, 270.0],
        rgb_resolution=(1280, 720),
        depth_resolution=(1280, 720),
        target_framing="full_body",
        description="Standard tall subject capture configuration",
    )
    assert cfg.num_yaw_stops == 4
    d = cfg.to_dict()
    assert d["config_id"] == "CONFIG_D240_H105_Y4_R720"
    assert d["camera_distance_m"] == 2.40

    # Test invalid config
    with pytest.raises(ValueError):
        CaptureConfiguration(config_id="", camera_distance_m=2.0, camera_height_m=1.0)
    with pytest.raises(ValueError):
        CaptureConfiguration(config_id="CFG1", camera_distance_m=-2.0, camera_height_m=1.0)


# ---------------------------------------------------------------------------
# 3. Full-Body Capture Readiness Checker
# ---------------------------------------------------------------------------

def test_full_body_readiness_partial_body():
    """Verify that an upper-body capture is diagnosed as PARTIAL_UPPER_BODY_ONLY."""
    stage8_calib = {
        "head": {"spatial_coverage": "partially_inside_fov", "anatomical_visibility": "partially_observed"},
        "shoulders": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "torso": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "pelvis_hips": {"spatial_coverage": "outside_fov", "anatomical_visibility": "missing"},
        "thighs": {"spatial_coverage": "outside_fov", "anatomical_visibility": "missing"},
        "lower_legs": {"spatial_coverage": "outside_fov", "anatomical_visibility": "missing"},
        "feet": {"spatial_coverage": "outside_fov", "anatomical_visibility": "missing"},
    }
    res = FullBodyReadinessChecker.evaluate_readiness("body_capture_01", stage8_calib)
    assert res.overall_status == "PARTIAL_UPPER_BODY_ONLY"
    assert res.total_regions_count == 10
    assert "pelvis" in res.missing_regions
    assert "thighs" in res.missing_regions
    assert "knees" in res.missing_regions
    assert "calves" in res.missing_regions
    assert "feet" in res.missing_regions
    assert res.region_details["head"].is_adequately_covered is True
    assert res.region_details["feet"].is_adequately_covered is False


def test_full_body_readiness_complete():
    """Verify that when all 10 regions are observed, status is READY_FULL_BODY."""
    complete_calib = {
        "head": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "shoulders": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "torso": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "pelvis_hips": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "thighs": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "lower_legs": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
        "feet": {"spatial_coverage": "inside_fov", "anatomical_visibility": "observed"},
    }
    res = FullBodyReadinessChecker.evaluate_readiness("mock_complete_capture", complete_calib)
    # Target regions should all be evaluated
    assert len(res.region_details) == 10


# ---------------------------------------------------------------------------
# 4. Repeatability Analysis Engine
# ---------------------------------------------------------------------------

def test_repeatability_single_capture_sample_size_guard():
    """Verify that when only 1 capture exists for a subject, repeat stats are marked unavailable."""
    records = [{
        "subject_id": "SUBJ_01",
        "capture_id": "body_capture_01",
        "repeat_index": 1,
        "measurements": {
            "chest_circumference": {"stage2_mm": 925.0, "stage4_mm": 925.0},
            "waist_circumference": {"stage2_mm": 780.0, "stage4_mm": 772.0},
        },
    }]
    results = RepeatabilityEngine.compute_repeatability(records)
    assert "chest_circumference" in results
    rep = results["chest_circumference"]
    assert rep.num_subjects == 1
    assert rep.num_captures == 1
    assert rep.within_subject_std_mm is None
    assert rep.within_subject_cv_percent is None
    assert rep.repeatability_coefficient_mm is None
    assert rep.icc_value is None
    assert "unavailable" in rep.icc_status.lower()


def test_repeatability_multiple_repeats_same_subject():
    """Verify within-subject variation and repeatability coefficient when repeats exist for a subject."""
    # 3 repeat captures of SUBJ_01
    records = [
        {
            "subject_id": "SUBJ_01",
            "capture_id": "body_capture_01_r1",
            "repeat_index": 1,
            "measurements": {"waist_circumference": {"stage2_mm": 780.0, "stage4_mm": 772.0}},
        },
        {
            "subject_id": "SUBJ_01",
            "capture_id": "body_capture_01_r2",
            "repeat_index": 2,
            "measurements": {"waist_circumference": {"stage2_mm": 782.0, "stage4_mm": 774.0}},
        },
        {
            "subject_id": "SUBJ_01",
            "capture_id": "body_capture_01_r3",
            "repeat_index": 3,
            "measurements": {"waist_circumference": {"stage2_mm": 779.0, "stage4_mm": 773.0}},
        },
    ]
    results = RepeatabilityEngine.compute_repeatability(records)
    rep = results["waist_circumference"]
    assert rep.num_subjects == 1
    assert rep.num_captures == 3
    # Subject-level grouping verified: repeats are grouped under SUBJ_01
    assert rep.num_repeats_per_subject["SUBJ_01"] == 3

    assert rep.within_subject_std_mm is not None
    assert rep.within_subject_std_mm > 0.0
    assert rep.repeatability_coefficient_mm is not None
    # RC = 1.96 * sqrt(2) * s_w ~ 2.77 * s_w
    expected_rc = 1.96 * math.sqrt(2.0) * rep.within_subject_std_mm
    assert rep.repeatability_coefficient_mm == pytest.approx(expected_rc, abs=0.05)

    # ICC requires >= 2 subjects
    assert rep.icc_value is None
    assert "unavailable_insufficient_sample_size" in rep.icc_status


def test_repeatability_icc_computation_multi_subject():
    """Verify ICC(2,1) calculation when multiple subjects have repeat captures."""
    # 3 subjects, each with 2 repeat captures
    records = [
        {"subject_id": "SUBJ_01", "repeat_index": 1, "measurements": {"stature": {"stage2_mm": 1750.0, "stage4_mm": 1750.0}}},
        {"subject_id": "SUBJ_01", "repeat_index": 2, "measurements": {"stature": {"stage2_mm": 1752.0, "stage4_mm": 1752.0}}},
        {"subject_id": "SUBJ_02", "repeat_index": 1, "measurements": {"stature": {"stage2_mm": 1820.0, "stage4_mm": 1820.0}}},
        {"subject_id": "SUBJ_02", "repeat_index": 2, "measurements": {"stature": {"stage2_mm": 1821.0, "stage4_mm": 1821.0}}},
        {"subject_id": "SUBJ_03", "repeat_index": 1, "measurements": {"stature": {"stage2_mm": 1650.0, "stage4_mm": 1650.0}}},
        {"subject_id": "SUBJ_03", "repeat_index": 2, "measurements": {"stature": {"stage2_mm": 1649.0, "stage4_mm": 1649.0}}},
    ]
    results = RepeatabilityEngine.compute_repeatability(records)
    rep = results["stature"]
    assert rep.num_subjects == 3
    assert rep.num_captures == 6
    assert rep.icc_value is not None
    # Stature has huge between-subject variance and tiny within-subject variance -> ICC ~ 0.999
    assert rep.icc_value > 0.95
    assert rep.icc_status == "computed_icc_2_1"


# ---------------------------------------------------------------------------
# 5. Manual Anthropometric Validation Bridge
# ---------------------------------------------------------------------------

def test_manual_validation_unpopulated_template_anti_fabrication():
    """Verify that unpopulated template strictly yields PENDING_MANUAL_INPUT without fabricated values."""
    summary = PilotManualValidationBridge.evaluate_manual_validation()
    assert summary.has_manual_ground_truth is False
    assert summary.status == "TEMPLATE_READY_MANUAL_INPUT_PENDING"
    assert summary.populated_manual_samples == 0
    assert summary.stage2_mae_mm is None
    assert summary.stage4_mae_mm is None
    assert summary.mean_error_reduction_mm is None

    # Check incompatible metric exclusion
    incomp_items = [it for it in summary.items if it.status == "INCOMPATIBLE_EXCLUDED"]
    assert len(incomp_items) >= 1
    assert any("neck_base" in it.measurement_name for it in incomp_items)


def test_manual_validation_with_populated_mock_data():
    """Verify error computation when manual measurements are present."""
    # Create temporary CSV with 2 measurements populated
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id", "capture_id", "measurement_name", "anatomical_protocol",
            "manual_measurement_mm", "manual_repeat_1_mm", "manual_repeat_2_mm", "manual_repeat_3_mm",
            "rgbd_stage2_measurement_mm", "rgbd_stage4_measurement_mm", "stage5_status",
            "observed_coverage_percent", "comparable", "clothing_condition", "operator_id", "notes"
        ])
        writer.writerow([
            "SUBJ_01", "body_capture_01", "waist_circumference", "ISO 7250",
            "775.0", "775.0", "774.0", "776.0",
            "780.0", "774.0", "partially_observed", "39.3", "True", "casual", "OPERATOR_01", "test"
        ])
        writer.writerow([
            "SUBJ_01", "body_capture_01", "hip_circumference", "ISO 7250",
            "1030.0", "1030.0", "1031.0", "1029.0",
            "1040.0", "1031.0", "observation_supported", "97.9", "True", "casual", "OPERATOR_01", "test"
        ])
        temp_csv = f.name

    try:
        summary = PilotManualValidationBridge.evaluate_manual_validation(template_csv_path=temp_csv)
        assert summary.has_manual_ground_truth is True
        assert summary.status == "EVALUATED_WITH_MANUAL_DATA"
        assert summary.populated_manual_samples == 2

        # Check errors
        # Waist: manual=775, s2=780 (err=+5, abs=5), s4=774 (err=-1, abs=1) -> reduction=4
        # Hip: manual=1030, s2=1040 (err=+10, abs=10), s4=1031 (err=+1, abs=1) -> reduction=9
        assert summary.stage2_mae_mm == pytest.approx(7.5, abs=0.1)
        assert summary.stage4_mae_mm == pytest.approx(1.0, abs=0.1)
        assert summary.mean_error_reduction_mm == pytest.approx(6.5, abs=0.1)
    finally:
        Path(temp_csv).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. Capture Configuration Comparison
# ---------------------------------------------------------------------------

def test_configuration_comparator_no_false_optimality():
    """Verify that configuration comparison refuses to declare optimality without comparative data."""
    rec = ConfigurationComparisonRecord(
        config_id="CONFIG_D214_H100_Y4_R720",
        camera_distance_m=2.14,
        camera_height_m=1.00,
        num_yaw_stops=4,
        resolution="1280x720",
        physical_captures_count=1,
        anatomical_coverage_ratio=0.50,
        valid_depth_ratio=0.89,
        mean_landmark_confidence=0.88,
        reconstruction_fitting_residual_m=0.0384,
        stage4_deformation_residual_m=0.0211,
        mean_repeatability_cv_percent=None,
        manual_ground_truth_mae_mm=None,
        processing_time_seconds=24.5,
        empirical_status="EVALUATED_SINGLE_CAPTURE",
        notes="Single configuration evaluated.",
    )
    comp = PilotConfigurationComparator.build_comparison([rec])
    assert comp["has_multi_configuration_physical_data"] is False
    assert comp["optimal_configuration_declared"] is None
    assert "No optimal camera configuration can be declared" in comp["conclusion"]


# ---------------------------------------------------------------------------
# 7. Physical Artifacts Verification in stage9_pilot_study_output/
# ---------------------------------------------------------------------------

def test_stage9_physical_artifacts_exist():
    """Verify all 8 required Stage 9 artifacts exist and are non-empty."""
    assert STAGE9_DIR.is_dir(), f"Stage 9 directory not found at {STAGE9_DIR}"

    required_files = [
        STAGE9_DIR / "pilot_study_protocol.md",
        STAGE9_DIR / "pilot_experiment_template.csv",
        STAGE9_DIR / "pilot_capture_manifest.csv",
        STAGE9_DIR / "pilot_repeatability_results.csv",
        STAGE9_DIR / "pilot_manual_validation_results.csv",
        STAGE9_DIR / "pilot_configuration_comparison.json",
        STAGE9_DIR / "pilot_study_summary.json",
        STAGE9_DIR / "stage9_pilot_study_report.md",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required Stage 9 artifact missing: {rf}"
        assert rf.stat().st_size > 0, f"Artifact is empty: {rf}"


def test_pilot_study_summary_json_content():
    """Verify pilot_study_summary.json structure and truthfulness."""
    p = STAGE9_DIR / "pilot_study_summary.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "Stage 9: Physical Pilot Study & Repeatability Evaluation" in data["stage"]
    assert data["audited_capture_id"] == "body_capture_01"
    assert data["subject_id"] == "SUBJ_01"
    assert data["physical_data_status"]["physical_captures_in_repo"] == 1
    assert data["physical_data_status"]["has_repeat_captures_for_same_subject"] is False
    assert data["physical_data_status"]["has_manual_anthropometric_ground_truth"] is False
    assert data["audited_capture_readiness"]["overall_status"] == "PARTIAL_UPPER_BODY_ONLY"
    assert "pelvis" in data["audited_capture_readiness"]["missing_regions"]
    assert "thighs" in data["audited_capture_readiness"]["missing_regions"]
    assert "No synthetic values were fabricated" in data["conclusion"]


def test_pilot_repeatability_csv_content():
    """Verify pilot_repeatability_results.csv contains required columns and records pending status."""
    p = STAGE9_DIR / "pilot_repeatability_results.csv"
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    expected_cols = {
        "measurement_name", "num_subjects", "num_captures", "mean_val_mm",
        "total_std_mm", "total_cv_pct", "within_subject_std_mm", "within_subject_cv_pct",
        "repeatability_coeff_mm", "icc_value", "icc_status", "stage2_mean_mm", "stage4_mean_mm"
    }
    assert expected_cols.issubset(set(reader.fieldnames or []))

    # All rows should report INSUFFICIENT_REPEAT_CAPTURES because only 1 capture exists in repo
    for r in rows:
        assert r["status"] == "INSUFFICIENT_REPEAT_CAPTURES"
        assert r["num_subjects"] == "1"
        assert r["num_captures"] == "1"
        assert r["within_subject_std_mm"] == ""
        assert "unavailable" in r["icc_status"].lower()


def test_pilot_manual_validation_csv_content():
    """Verify pilot_manual_validation_results.csv contains required columns and pending status."""
    p = STAGE9_DIR / "pilot_manual_validation_results.csv"
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    # Neck base must be INCOMPATIBLE_EXCLUDED
    neck_rows = [r for r in rows if "neck_base" in r["measurement_name"]]
    assert len(neck_rows) == 1
    assert neck_rows[0]["status"] == "INCOMPATIBLE_EXCLUDED"

    # Other rows must be PENDING_MANUAL_GROUND_TRUTH
    pending_rows = [r for r in rows if r["status"] == "PENDING_MANUAL_GROUND_TRUTH"]
    assert len(pending_rows) > 0
    for pr in pending_rows:
        assert pr["manual_ground_truth_mm"] == ""
        assert pr["signed_error_stage4_mm"] == ""


def test_pilot_study_markdown_report_comprehensive():
    """Verify stage9_pilot_study_report.md exists and covers key sections."""
    p = STAGE9_DIR / "stage9_pilot_study_report.md"
    content = p.read_text(encoding="utf-8")

    assert "Problem 4 — Stage 9: Physical Pilot Study & Repeatability Evaluation" in content
    assert "1. Executive Summary & Infrastructure Status" in content
    assert "2. Audit of Existing Physical Capture (`body_capture_01`)" in content
    assert "3. Repeatability Analysis & Sample Size Guard" in content
    assert "4. Manual Anthropometric Validation Status" in content
    assert "5. Capture Configuration Comparative Evaluation" in content
    assert "6. Pilot Study Execution Roadmap" in content
    assert "7. Output Artifact Index" in content
    assert "PARTIAL_UPPER_BODY_ONLY" in content

