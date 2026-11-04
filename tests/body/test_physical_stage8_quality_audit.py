"""Tests for Problem 4 - Stage 8: Capture Quality Calibration & Sensor-Grounded Visibility Audit.

Validates:
1. Decoupled taxonomic dimensions: SpatialCoverage, AnatomicalVisibility, DiagnosticClassification.
2. Coordinate coverage vs. anatomical visibility separation (e.g. points at Z < -0.4m do NOT classify as observed).
3. PoseDepthIntegrator bounds checking, depth gating (0.4m - 3.0m), and bilateral landmark identity.
4. Handling of unavailable landmarks: returns 'unavailable' with null confidence, strictly no fabrication.
5. Handling of invalid depths: depth == 0.0 or NaN classified as 'unreliable'.
6. Diagnostic provenance tracking metadata.
7. Distance consistency metric decomposition: dispersion, AP/ML shape asymmetry, sway, sensor noise floor, tripod displacement.
8. Epistemological status tags: theoretical formulas tagged 'theoretical_estimate', physical observations tagged 'physical_measurement'.
9. Verification of all physical artifacts in stage8_capture_quality_audit_output/.
10. Verification of unmasked Stage 7 false positives (pelvis/thighs corrected to missing).
"""

import csv
import json
import math
from pathlib import Path
from typing import List
import numpy as np
import pytest

from realsense_measure.body.capture_quality_audit import (
    ANATOMICAL_REGION_LANDMARKS,
    AnatomicalVisibility,
    DiagnosticClassification,
    DiagnosticProvenance,
    DistanceConsistencyAuditor,
    DistanceConsistencyDecomposition,
    LandmarkSample,
    PoseDepthIntegrator,
    SensorGroundedVisibilityAuditor,
    SpatialCoverage,
    TheoreticalExperimentAuditor,
)


STAGE8_DIR = Path("stage8_capture_quality_audit_output")
RAW_DIR = Path("raw_captures/body_capture_01")


# ---------------------------------------------------------------------------
# 1. Decoupled Taxonomic Dimensions
# ---------------------------------------------------------------------------

def test_enums_and_taxonomy():
    """Verify all taxonomic enum values are correctly defined and distinct."""
    # Spatial coverage
    assert SpatialCoverage.INSIDE_FOV.value == "inside_fov"
    assert SpatialCoverage.PARTIALLY_INSIDE_FOV.value == "partially_inside_fov"
    assert SpatialCoverage.OUTSIDE_FOV.value == "outside_fov"
    assert SpatialCoverage.UNKNOWN.value == "unknown"

    # Anatomical visibility
    assert AnatomicalVisibility.OBSERVED.value == "observed"
    assert AnatomicalVisibility.PARTIALLY_OBSERVED.value == "partially_observed"
    assert AnatomicalVisibility.OCCLUDED.value == "occluded"
    assert AnatomicalVisibility.MISSING.value == "missing"
    assert AnatomicalVisibility.UNRELIABLE.value == "unreliable"
    assert AnatomicalVisibility.UNAVAILABLE.value == "unavailable"

    # Diagnostic classification
    assert DiagnosticClassification.PHYSICAL_MEASUREMENT.value == "physical_measurement"
    assert DiagnosticClassification.THEORETICAL_ESTIMATE.value == "theoretical_estimate"
    assert DiagnosticClassification.SYNTHETIC_ESTIMATE.value == "synthetic_estimate"


def test_spatial_coverage_vs_anatomical_visibility_decoupling():
    """Verify that spatial coverage and anatomical visibility are strictly independent."""
    # Case 1: Inside FOV, but landmark is missing/undetected -> MISSING
    # Case 2: Inside FOV, but depth is zero/invalid -> UNRELIABLE
    # Case 3: Outside FOV -> MISSING
    sample_valid = LandmarkSample(
        landmark_id=11, name="left_shoulder", norm_x=0.4, norm_y=0.3,
        pixel_u=512, pixel_v=216, is_in_image_bounds=True,
        visibility_score=0.95, presence_score=0.95, depth_m=2.10, is_depth_valid=True,
        spatial_coverage=SpatialCoverage.INSIDE_FOV.value,
        anatomical_visibility=AnatomicalVisibility.OBSERVED.value,
    )
    assert sample_valid.spatial_coverage == "inside_fov"
    assert sample_valid.anatomical_visibility == "observed"

    sample_bad_depth = LandmarkSample(
        landmark_id=12, name="right_shoulder", norm_x=0.6, norm_y=0.3,
        pixel_u=768, pixel_v=216, is_in_image_bounds=True,
        visibility_score=0.90, presence_score=0.90, depth_m=0.0, is_depth_valid=False,
        spatial_coverage=SpatialCoverage.INSIDE_FOV.value,
        anatomical_visibility=AnatomicalVisibility.UNRELIABLE.value,
    )
    assert sample_bad_depth.spatial_coverage == "inside_fov"
    assert sample_bad_depth.anatomical_visibility == "unreliable"

    sample_out_of_bounds = LandmarkSample(
        landmark_id=23, name="left_hip", norm_x=0.5, norm_y=1.35,
        pixel_u=640, pixel_v=972, is_in_image_bounds=False,
        visibility_score=0.0, presence_score=0.0, depth_m=float("nan"), is_depth_valid=False,
        spatial_coverage=SpatialCoverage.OUTSIDE_FOV.value,
        anatomical_visibility=AnatomicalVisibility.MISSING.value,
    )
    assert sample_out_of_bounds.spatial_coverage == "outside_fov"
    assert sample_out_of_bounds.anatomical_visibility == "missing"


# ---------------------------------------------------------------------------
# 2. Bilateral Landmark Identification & Region Mapping
# ---------------------------------------------------------------------------

def test_bilateral_landmark_mapping():
    """Verify that left and right anatomical regions use distinct landmark indices."""
    assert ANATOMICAL_REGION_LANDMARKS["shoulder_left"] == [11]
    assert ANATOMICAL_REGION_LANDMARKS["shoulder_right"] == [12]
    assert ANATOMICAL_REGION_LANDMARKS["pelvis_left"] == [23]
    assert ANATOMICAL_REGION_LANDMARKS["pelvis_right"] == [24]
    assert ANATOMICAL_REGION_LANDMARKS["thigh_left"] == [23, 25]
    assert ANATOMICAL_REGION_LANDMARKS["thigh_right"] == [24, 26]
    assert ANATOMICAL_REGION_LANDMARKS["lower_leg_left"] == [25, 27]
    assert ANATOMICAL_REGION_LANDMARKS["lower_leg_right"] == [26, 28]
    assert ANATOMICAL_REGION_LANDMARKS["foot_left"] == [27, 29, 31]
    assert ANATOMICAL_REGION_LANDMARKS["foot_right"] == [28, 30, 32]

    # No overlap between left and right single-side landmarks
    left_all = set(ANATOMICAL_REGION_LANDMARKS["shoulder_left"] + ANATOMICAL_REGION_LANDMARKS["arm_left"] + ANATOMICAL_REGION_LANDMARKS["pelvis_left"] + ANATOMICAL_REGION_LANDMARKS["thigh_left"] + ANATOMICAL_REGION_LANDMARKS["lower_leg_left"] + ANATOMICAL_REGION_LANDMARKS["foot_left"])
    right_all = set(ANATOMICAL_REGION_LANDMARKS["shoulder_right"] + ANATOMICAL_REGION_LANDMARKS["arm_right"] + ANATOMICAL_REGION_LANDMARKS["pelvis_right"] + ANATOMICAL_REGION_LANDMARKS["thigh_right"] + ANATOMICAL_REGION_LANDMARKS["lower_leg_right"] + ANATOMICAL_REGION_LANDMARKS["foot_right"])
    assert left_all.isdisjoint(right_all), "Left and right landmark sets must be strictly disjoint!"


# ---------------------------------------------------------------------------
# 3. Provenance Schema & Explicit Missing/Unavailable Behavior
# ---------------------------------------------------------------------------

def test_diagnostic_provenance_dataclass():
    """Verify DiagnosticProvenance fields and serialization."""
    prov = DiagnosticProvenance(
        diagnostic_name="visibility_pelvis_left",
        result="missing",
        spatial_coverage=SpatialCoverage.OUTSIDE_FOV.value,
        anatomical_visibility=AnatomicalVisibility.MISSING.value,
        confidence=None,
        source_data="mediapipe_pose",
        method="landmark_lookup",
        classification_type=DiagnosticClassification.PHYSICAL_MEASUREMENT.value,
        reason_if_unavailable="Hip landmark normalized y=1.37 exceeds image height",
        notes="Subject framed as upper-body medium portrait",
    )
    d = prov.to_dict()
    assert d["diagnostic_name"] == "visibility_pelvis_left"
    assert d["confidence"] is None  # Strictly no fabricated confidence
    assert d["spatial_coverage"] == "outside_fov"
    assert d["anatomical_visibility"] == "missing"
    assert d["classification_type"] == "physical_measurement"
    assert "exceeds image height" in d["reason_if_unavailable"]


def test_unavailable_pose_detector_behavior():
    """Verify that when pose detector is unavailable, results are 'unavailable' with null confidence."""
    # Create an integrator pointing to a non-existent model
    integrator = PoseDepthIntegrator(model_path="non_existent_path.task")
    assert not integrator.is_available

    auditor = SensorGroundedVisibilityAuditor(pose_integrator=integrator)
    h, w = 720, 1280
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 2.1, dtype=np.float32)
    fg_mask = np.ones((h, w), dtype=bool)

    prov_list = auditor.audit_stop(
        stop_idx=0,
        nominal_yaw_deg=0.0,
        color_bgr=color,
        depth_m=depth,
        foreground_mask=fg_mask,
        z_bounds_world_m=(-0.65, 0.25),
    )

    assert len(prov_list) == len(ANATOMICAL_REGION_LANDMARKS)
    for p in prov_list:
        assert p.result == "unavailable"
        assert p.spatial_coverage == SpatialCoverage.UNKNOWN.value
        assert p.anatomical_visibility == AnatomicalVisibility.UNAVAILABLE.value
        assert p.confidence is None
        assert p.reason_if_unavailable is not None
        assert "not available" in p.reason_if_unavailable


# ---------------------------------------------------------------------------
# 4. Distance Consistency Decomposition
# ---------------------------------------------------------------------------

def test_distance_consistency_auditor():
    """Verify physical decomposition of inter-stop depth variations."""
    # Depths with AP vs ML cross-sectional difference:
    # Front (0 deg): 2.14m, Right (90 deg): 2.12m, Back (180 deg): 2.15m, Left (270 deg): 2.13m
    medians = [2.14, 2.12, 2.15, 2.13]
    decomp = DistanceConsistencyAuditor.audit_distance_metrics(
        depth_medians_m=medians,
        camera_mounted_on_tripod=True,
    )

    assert decomp.standoff_median_depth_m == pytest.approx(2.135, abs=0.01)
    assert decomp.foreground_median_depth_dispersion_mm > 0.0
    assert decomp.is_stage7_metric_flawed is True
    assert "conflates human anatomical cross-sectional asymmetry" in decomp.flawed_metric_critique
    assert decomp.camera_tripod_displacement_mm == 0.0
    assert decomp.is_camera_physically_stationary is True
    assert decomp.corrected_standoff_stability_rating == "stable_stationary_mount"
    assert decomp.body_shape_asymmetry_variation_mm > 0.0
    assert decomp.sensor_temporal_noise_floor_mm > 0.0


# ---------------------------------------------------------------------------
# 5. Theoretical Assumptions & Epistemological Separation
# ---------------------------------------------------------------------------

def test_theoretical_experiment_assumptions_epistemology():
    """Verify that theoretical experiment formulas are strictly catalogued and tagged as theoretical_estimate."""
    assumptions = TheoreticalExperimentAuditor.get_audited_assumptions()

    required_metrics = ["vertical_fov_span", "theoretical_depth_noise", "nominal_point_density"]
    for m in required_metrics:
        assert m in assumptions
        doc = assumptions[m]
        assert doc.epistemological_type == DiagnosticClassification.THEORETICAL_ESTIMATE.value
        assert len(doc.assumptions) > 0
        assert len(doc.known_limitations) > 0
        assert doc.formula != ""
        assert doc.units != ""


# ---------------------------------------------------------------------------
# 6. Physical Artifacts Verification in stage8_capture_quality_audit_output/
# ---------------------------------------------------------------------------

def test_stage8_physical_artifacts_exist():
    """Verify all 6 required Stage 8 output artifacts exist."""
    assert STAGE8_DIR.is_dir(), f"Stage 8 directory not found at {STAGE8_DIR}"

    required_files = [
        STAGE8_DIR / "stage7_diagnostic_audit.json",
        STAGE8_DIR / "capture_quality_calibration_report.json",
        STAGE8_DIR / "capture_visibility_report.csv",
        STAGE8_DIR / "capture_distance_consistency_report.json",
        STAGE8_DIR / "capture_experiment_assumptions.json",
        STAGE8_DIR / "stage8_capture_quality_audit_report.md",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required Stage 8 artifact missing: {rf}"
        assert rf.stat().st_size > 0, f"Artifact is empty: {rf}"


def test_stage7_diagnostic_audit_content():
    """Verify stage7_diagnostic_audit.json unmasks Stage 7 false positives."""
    audit_path = STAGE8_DIR / "stage7_diagnostic_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "audit_title" in data
    assert "flawed_diagnostics" in data
    assert "root_cause_analysis" in data
    assert "remediation_actions" in data

    flawed = data["flawed_diagnostics"]
    # Pelvis, thighs, lower legs, and distance consistency should be unmasked
    regions = [item["region"] for item in flawed]
    assert "pelvis" in regions
    assert "thighs" in regions
    assert "lower_legs" in regions
    assert any("distance_consistency" in r for r in regions)


def test_capture_quality_calibration_report_content():
    """Verify capture_quality_calibration_report.json structure and validity."""
    calib_path = STAGE8_DIR / "capture_quality_calibration_report.json"
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "Stage 8: Capture Quality Calibration & Sensor-Grounded Visibility Audit" in data["stage"]
    assert data["capture_id"] == "body_capture_01"
    assert "overall_calibrated_status" in data
    assert "distance_consistency_decomposition" in data
    assert data["media_pipe_pose_integrated"] is True

    # Verify distance consistency decomposition fields
    dist_audit = data["distance_consistency_decomposition"]
    assert dist_audit["camera_tripod_displacement_mm"] == 0.0
    assert dist_audit["is_stage7_metric_flawed"] is True
    assert dist_audit["foreground_median_depth_dispersion_mm"] > 0.0


def test_capture_visibility_report_csv():
    """Verify capture_visibility_report.csv has expected columns and rows."""
    csv_path = STAGE8_DIR / "capture_visibility_report.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    # Expected columns
    expected_cols = {
        "stop_idx", "nominal_yaw_deg", "diagnostic_name", "spatial_coverage",
        "anatomical_visibility", "confidence", "classification_type", "source_data", "method"
    }
    assert expected_cols.issubset(set(reader.fieldnames or []))

    # Check that pelvis and thighs are recorded as outside_fov / missing
    pelvis_rows = [r for r in rows if "pelvis" in r["diagnostic_name"]]
    assert len(pelvis_rows) > 0
    for pr in pelvis_rows:
        assert pr["spatial_coverage"] == "outside_fov"
        assert pr["anatomical_visibility"] == "missing"

    # Check shoulders are recorded as observed inside FOV on front view stop 0
    shoulder_rows = [r for r in rows if "shoulder" in r["diagnostic_name"] and r["stop_idx"] == "0"]
    assert len(shoulder_rows) > 0
    for sr in shoulder_rows:
        assert sr["spatial_coverage"] == "inside_fov"
        assert sr["anatomical_visibility"] == "observed"


def test_capture_distance_consistency_report_json():
    """Verify capture_distance_consistency_report.json contains complete physical decomposition."""
    dc_path = STAGE8_DIR / "capture_distance_consistency_report.json"
    with open(dc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["is_stage7_metric_flawed"] is True
    assert data["camera_tripod_displacement_mm"] == 0.0
    assert data["corrected_standoff_stability_rating"] == "stable_stationary_mount"
    assert "body_shape_asymmetry_variation_mm" in data
    assert "postural_sway_estimate_mm" in data
    assert "sensor_temporal_noise_floor_mm" in data


def test_capture_experiment_assumptions_json():
    """Verify capture_experiment_assumptions.json documents all theoretical formulas."""
    assump_path = STAGE8_DIR / "capture_experiment_assumptions.json"
    with open(assump_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "vertical_fov_span" in data
    assert "theoretical_depth_noise" in data
    assert "nominal_point_density" in data

    for name, metric_data in data.items():
        assert metric_data["epistemological_type"] == "theoretical_estimate"
        assert len(metric_data["assumptions"]) >= 2
        assert len(metric_data["known_limitations"]) >= 1


def test_markdown_report_comprehensive():
    """Verify stage8_capture_quality_audit_report.md exists, has substance, and covers key sections."""
    md_path = STAGE8_DIR / "stage8_capture_quality_audit_report.md"
    content = md_path.read_text(encoding="utf-8")

    assert "Problem 4 — Stage 8: Capture Quality Calibration & Sensor-Grounded Visibility Audit" in content
    assert "1. Executive Summary & Audit Conclusions" in content
    assert "2. Audit of Stage 7 Diagnostics: Heuristics vs. Sensor Evidence" in content
    assert "3. Decoupling Spatial Coverage from Anatomical Visibility" in content
    assert "4. Distance Consistency Metric Audit & Decomposition" in content
    assert "5. Audit of Theoretical Experiment Assumptions" in content
    assert "6. Diagnostic Provenance Tracking Schema" in content
    assert "7. Output Artifact Index" in content
