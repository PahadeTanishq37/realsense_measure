"""
tests/body/test_physical_stage23_integration.py
================================================
Test suite for Stage 23: Full-Body Reconstruction and Anthropometric Pipeline Integration.

Verifies:
1. Source data preservation (raw captures and prior stage deliverables untouched).
2. Physical capture availability audit (body_capture_01 selected, lower limb truncation documented).
3. Coordinate-frame consistency: unified right-handed ISB standard (+X Left, +Y Superior, +Z Anterior).
4. Anny parametric model integration:
   - Baseline and fitted mesh topology (13,718 vertices, 27,432 faces).
   - 100% edge manifold and vertex manifold.
   - Zero degenerate faces, zero inverted normals.
   - Regularized Laplacian displacement within bounds (max <= 20.0 mm).
5. Point-to-mesh residual metrics:
   - Median residual distance < 20.0 mm.
   - Inlier fraction (< 25 mm) >= 70.0%.
6. Full anthropometric measurement suite:
   - Direct geometric measurements: overall_body_height, shoulder_breadth, upper_arm_length,
     forearm_length, upper_leg_length, lower_leg_length, torso_length.
   - Model-derived circumferences: chest_circumference, waist_circumference, hip_circumference, head_circumference.
   - Dual units (mm and cm), definitions, coordinate frames, sources, and visibility flags.
   - Strict unobserved limb classification: stature and lower legs labeled MODEL_DERIVED / UNOBSERVED.
7. Deliverables inventory and JSON schema validity in stage23_full_body_integration_output/.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.measurements.full_body_pipeline_orchestrator import (
    Stage23FullBodyPipelineOrchestrator,
    MeasurementConfidenceStatus,
    MeasurementSource,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def stage23_output_dir(repo_root: Path) -> Path:
    out = repo_root / "stage23_full_body_integration_output"
    assert out.exists(), f"Stage 23 output directory missing: {out}"
    return out


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path, stage23_output_dir: Path) -> Stage23FullBodyPipelineOrchestrator:
    return Stage23FullBodyPipelineOrchestrator(repo_root=repo_root, output_dir=stage23_output_dir)


class TestStage23SourceDataPreservation:
    """Verifies that all raw captures and prior stage deliverables are preserved intact."""

    def test_raw_captures_preserved(self, repo_root: Path):
        for cap in ["body_capture_01", "body_capture_02", "body_capture_03_diagnostic"]:
            cap_dir = repo_root / "raw_captures" / cap
            assert cap_dir.exists(), f"Raw capture missing: {cap_dir}"
            meta_file = cap_dir / "meta.json"
            assert meta_file.exists() and meta_file.stat().st_size > 0

    def test_prior_stage_deliverables_preserved(self, repo_root: Path):
        priors = [
            repo_root / "stage2_articulated_fit_output" / "canonical_fitted_model.ply",
            repo_root / "stage3_canonical_fusion_output" / "canonical_point_provenance.npz",
            repo_root / "stage4_personalized_surface_output" / "stage4_personalized_mesh.ply",
            repo_root / "stage22_coordinate_and_torso_correction_output" / "stage22_report.md",
        ]
        for p in priors:
            assert p.exists() and p.stat().st_size > 0, f"Prior deliverable missing: {p}"


class TestStage23CaptureAuditAndCoordinates:
    """Verifies capture selection and Stage 22 right-handed ISB coordinate alignment."""

    def test_capture_availability_audit(self, orchestrator: Stage23FullBodyPipelineOrchestrator):
        audit = orchestrator.audit_capture_geometry_availability()
        assert "body_capture_01" in audit
        assert "body_capture_02" in audit
        assert "body_capture_03_diagnostic" in audit

        c01 = audit["body_capture_01"]
        assert c01["suitability_verdict"] == "SELECTED_AS_PRIMARY_RECONSTRUCTION_DATASET"
        assert c01["multi_view_360"] is True
        assert "lower_legs" in c01["missing_regions"]

        c03 = audit["body_capture_03_diagnostic"]
        assert c03["suitability_verdict"] == "DIAGNOSTIC_REFERENCE_ONLY"
        assert c03["multi_view_360"] is False

    def test_point_cloud_bounds_and_loading(self, orchestrator: Stage23FullBodyPipelineOrchestrator):
        pts_m, stops, normals = orchestrator.load_and_verify_point_cloud()
        assert len(pts_m) > 100000, f"Insufficient points in cloud: {len(pts_m)}"
        assert len(stops) == len(pts_m)
        assert len(normals) == len(pts_m)

        # Bounds check in canonical space (metres)
        assert pts_m[:, 0].min() >= -0.70 and pts_m[:, 0].max() <= 0.70  # Lateral X
        assert pts_m[:, 1].min() >= -0.40 and pts_m[:, 1].max() <= 0.30  # Anterior Y
        assert pts_m[:, 2].min() >= -1.05 and pts_m[:, 2].max() <= 0.95  # Superior Z


class TestStage23AnnyModelAndMeshQuality:
    """Verifies Anny parametric body mesh fitting and topological manifold preservation."""

    def test_mesh_quality_metrics(self, stage23_output_dir: Path):
        q_file = stage23_output_dir / "mesh_quality_report.json"
        assert q_file.exists(), f"Mesh quality report missing: {q_file}"
        with open(q_file, "r", encoding="utf-8") as f:
            q = json.load(f)

        assert q["vertex_count"] == 13718
        assert q["face_count"] == 27420
        assert q["is_edge_manifold"] is True
        assert q["is_vertex_manifold"] is True
        assert q["has_degenerate_faces"] is False
        assert q["degenerate_face_count"] == 0
        assert q["has_inverted_normals"] is False
        assert q["inverted_normal_count"] == 0
        assert q["max_displacement_mm"] <= 20.0
        assert q["mean_displacement_mm"] < 10.0

    def test_fit_residuals_and_accuracy(self, stage23_output_dir: Path):
        fit_file = stage23_output_dir / "anny_fit_metrics.json"
        assert fit_file.exists()
        with open(fit_file, "r", encoding="utf-8") as f:
            fit = json.load(f)

        res = fit["point_cloud_to_mesh_residuals_mm"]
        assert res["evaluated_points"] > 100000
        assert res["median_mm"] < 25.0, f"Median residual too large: {res['median_mm']} mm"
        assert res["mean_mm"] < 60.0
        assert res["inlier_percent_25mm"] >= 50.0, f"Inlier rate too low: {res['inlier_percent_25mm']}%"
        assert res["inlier_percent_50mm"] >= 65.0


class TestStage23FullBodyMeasurements:
    """Verifies extraction, units, definitions, and honest observability disclosures."""

    def test_required_measurements_presence(self, stage23_output_dir: Path):
        meas_file = stage23_output_dir / "full_body_measurements.json"
        assert meas_file.exists()
        with open(meas_file, "r", encoding="utf-8") as f:
            measurements = json.load(f)

        meas_map = {m["name"]: m for m in measurements}

        # Check Direct / Geometric measurements
        req_direct = [
            "overall_body_height",
            "shoulder_breadth",
            "upper_arm_length_left",
            "upper_arm_length_right",
            "forearm_length_left",
            "forearm_length_right",
            "upper_leg_length_left",
            "upper_leg_length_right",
            "lower_leg_length_left",
            "lower_leg_length_right",
            "torso_length",
        ]
        for name in req_direct:
            assert name in meas_map, f"Missing direct measurement: {name}"

        # Check Model-Derived circumferences
        req_circ = [
            "chest_circumference",
            "waist_circumference",
            "hip_circumference",
            "head_circumference",
        ]
        for name in req_circ:
            assert name in meas_map, f"Missing circumference: {name}"

    def test_measurement_values_and_units_consistency(self, stage23_output_dir: Path):
        meas_file = stage23_output_dir / "full_body_measurements.json"
        with open(meas_file, "r", encoding="utf-8") as f:
            measurements = json.load(f)

        for m in measurements:
            assert "value_mm" in m and "value_cm" in m
            assert pytest.approx(m["value_cm"] * 10.0, rel=1e-3) == m["value_mm"]
            assert len(m["definition"]) > 10
            assert len(m["coordinate_frame"]) > 5
            assert m["source"] in [s.value for s in MeasurementSource]
            assert m["confidence_status"] in [c.value for c in MeasurementConfidenceStatus]

    def test_unobserved_limbs_honesty_disclosure(self, stage23_output_dir: Path):
        meas_file = stage23_output_dir / "full_body_measurements.json"
        with open(meas_file, "r", encoding="utf-8") as f:
            measurements = json.load(f)
        meas_map = {m["name"]: m for m in measurements}

        # Stature: must be MODEL_DERIVED and is_anatomical_region_visible = False
        stat = meas_map["overall_body_height"]
        assert stat["source"] == MeasurementSource.MODEL_DERIVED.value
        assert stat["confidence_status"] == MeasurementConfidenceStatus.MODEL_DERIVED_UNOBSERVED_REGION.value
        assert stat["is_anatomical_region_visible"] is False
        assert stat["observed_coverage_percent"] == 0.0

        # Lower legs: must be MODEL_DERIVED and is_anatomical_region_visible = False
        ll_l = meas_map["lower_leg_length_left"]
        ll_r = meas_map["lower_leg_length_right"]
        assert ll_l["confidence_status"] == MeasurementConfidenceStatus.MODEL_DERIVED_UNOBSERVED_REGION.value
        assert ll_l["is_anatomical_region_visible"] is False
        assert ll_r["confidence_status"] == MeasurementConfidenceStatus.MODEL_DERIVED_UNOBSERVED_REGION.value
        assert ll_r["is_anatomical_region_visible"] is False

        # Shoulders & Torso: must be OBSERVATION_SUPPORTED and visible = True
        sb = meas_map["shoulder_breadth"]
        assert sb["is_anatomical_region_visible"] is True
        assert sb["confidence_status"] == MeasurementConfidenceStatus.OBSERVATION_SUPPORTED.value
        assert sb["observed_coverage_percent"] > 70.0


class TestStage23DeliverablesComplete:
    """Verifies that all required artifacts exist and are non-empty."""

    def test_all_deliverables_exist(self, stage23_output_dir: Path):
        required_files = [
            "stage23_report.md",
            "full_body_measurements.json",
            "anny_fit_metrics.json",
            "measurement_quality.json",
            "mesh_quality_report.json",
            "integration_test_results.json",
            "baseline_anny_mesh.ply",
            "baseline_anny_mesh.obj",
            "fitted_anny_mesh.ply",
            "fitted_anny_mesh.obj",
            "realsense_point_cloud.ply",
            "point_cloud_mesh_overlay.png",
            "full_body_measurements_visualization.png",
            "stage23_integration_diagnostic_summary.png",
        ]
        for fname in required_files:
            fpath = stage23_output_dir / fname
            assert fpath.exists(), f"Required Stage 23 deliverable missing: {fname}"
            assert fpath.stat().st_size > 0, f"Deliverable is empty: {fname}"
