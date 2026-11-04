"""
tests/body/test_physical_stage24_smpl.py
=========================================
Unit and integration test suite for Stage 24:
SMPL Integration and RealSense-Based Human Mesh Reconstruction.

Verifies:
1. SMPL setup audit & licensing transparency (CC BY-NC 4.0 license, installation instructions).
2. Canonical SMPL mesh topology validation (6,890 vertices, 13,776 faces).
3. Mesh quality & topological manifoldness (is_edge_manifold, is_vertex_manifold, 0 degenerate faces).
4. Stage 23 RealSense point cloud connection & fitting convergence.
5. Point-to-SMPL residual distance metrics (median residual < 30.0 mm).
6. SMPL anthropometric measurement extraction:
   - Stature, shoulder breadth, upper arm lengths, forearm lengths, torso length,
     upper leg lengths, lower leg lengths, chest, waist, hip, and head circumferences.
   - Dual units (mm and cm), definitions, coordinate frames, sources, and visibility flags.
   - Truncated lower legs & stature labeled MODEL_DERIVED / UNOBSERVED.
7. Anny vs SMPL comparative visualization and deliverables inventory in stage24_smpl_integration_output/.
8. Final verdict validity conforming to Section 10 allowed verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.body.smpl_backend import (
    SMPLModelBackend,
    Stage24Verdict,
)
from realsense_measure.measurements.smpl_orchestrator import (
    Stage24SMPLOrchestrator,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def stage24_output_dir(repo_root: Path) -> Path:
    out = repo_root / "stage24_smpl_integration_output"
    assert out.exists(), f"Stage 24 output directory missing: {out}"
    return out


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path, stage24_output_dir: Path) -> Stage24SMPLOrchestrator:
    return Stage24SMPLOrchestrator(repo_root=repo_root, output_dir=stage24_output_dir)


class TestStage24SMPLSetupAndLicensing:
    """Verifies SMPL model setup, licensing, and asset availability disclosures."""

    def test_smpl_setup_status_schema(self, stage24_output_dir: Path):
        setup_file = stage24_output_dir / "smpl_setup_status.json"
        assert setup_file.exists()
        with open(setup_file, "r", encoding="utf-8") as f:
            status = json.load(f)

        assert status["model_family"] == "SMPL (Skinned Multi-Person Linear Model)"
        assert "MPI-IS" in status["official_authors"] or "Max Planck" in status["official_authors"]
        assert "CC BY-NC" in status["licensing_terms"]
        assert "https://smpl.is.tue.mpg.de/" in status["official_website"]
        assert status["canonical_vertex_count"] == 6890
        assert status["canonical_face_count"] == 13776
        assert len(status["installation_instructions_if_missing"]) >= 4

    def test_canonical_smpl_topology_integrity(self, orchestrator: Stage24SMPLOrchestrator):
        m_base = orchestrator.backend.get_baseline_mesh()
        assert len(m_base.vertices) == 6890, f"Expected 6,890 SMPL vertices, got {len(m_base.vertices)}"
        assert len(m_base.faces) == 13776, f"Expected 13,776 SMPL faces, got {len(m_base.faces)}"

        # Check coordinate bounds in metres
        v = m_base.vertices
        assert v[:, 0].min() < -0.40 and v[:, 0].max() > 0.40  # Lateral span
        assert v[:, 2].min() < -0.70 and v[:, 2].max() > 0.70  # Longitudinal stature


class TestStage24SMPLFittingAndResiduals:
    """Verifies SMPL fitting against Stage 23 RealSense physical point cloud."""

    def test_smpl_fit_parameters_schema(self, stage24_output_dir: Path):
        params_file = stage24_output_dir / "smpl_fit_parameters.json"
        assert params_file.exists()
        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        assert "translation_m" in params and len(params["translation_m"]) == 3
        assert "global_orientation_rad" in params and len(params["global_orientation_rad"]) == 3
        assert "scale_factor" in params and params["scale_factor"] > 0.5
        assert "shape_betas" in params and len(params["shape_betas"]) >= 10
        assert params["convergence_status"] == "CONVERGED"

    def test_smpl_fit_metrics(self, stage24_output_dir: Path):
        metrics_file = stage24_output_dir / "smpl_fit_metrics.json"
        assert metrics_file.exists()
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["evaluated_point_count"] > 100000
        assert metrics["median_residual_mm"] < 30.0, f"Median residual too large: {metrics['median_residual_mm']} mm"
        assert metrics["mean_residual_mm"] < 60.0
        assert metrics["inlier_percent_25mm"] >= 45.0
        assert metrics["inlier_percent_50mm"] >= 65.0


class TestStage24SMPLAnthropometricMeasurements:
    """Verifies anthropometric measurement extraction on the fitted SMPL mesh."""

    def test_measurements_presence_and_definitions(self, stage24_output_dir: Path):
        meas_file = stage24_output_dir / "smpl_measurements.json"
        assert meas_file.exists()
        with open(meas_file, "r", encoding="utf-8") as f:
            measurements = json.load(f)

        meas_map = {m["name"]: m for m in measurements}

        # Check required direct measurements
        required_direct = [
            "overall_body_height",
            "shoulder_breadth",
            "upper_arm_length_left",
            "upper_arm_length_right",
            "forearm_length_left",
            "forearm_length_right",
            "torso_length",
            "upper_leg_length_left",
            "upper_leg_length_right",
            "lower_leg_length_left",
            "lower_leg_length_right",
        ]
        for name in required_direct:
            assert name in meas_map, f"Missing direct measurement: {name}"

        # Check required circumferences
        required_circ = [
            "chest_circumference",
            "waist_circumference",
            "hip_circumference",
            "head_circumference",
        ]
        for name in required_circ:
            assert name in meas_map, f"Missing circumference: {name}"

    def test_measurement_units_and_honesty_disclosures(self, stage24_output_dir: Path):
        meas_file = stage24_output_dir / "smpl_measurements.json"
        with open(meas_file, "r", encoding="utf-8") as f:
            measurements = json.load(f)

        for m in measurements:
            assert pytest.approx(m["value_cm"] * 10.0, rel=1e-3) == m["value_mm"]
            assert len(m["definition"]) > 10
            assert "source" in m and "confidence_status" in m

        meas_map = {m["name"]: m for m in measurements}

        # Stature & lower limbs must be marked MODEL_DERIVED / UNOBSERVED
        stat = meas_map["overall_body_height"]
        assert stat["confidence_status"] == "MODEL_DERIVED_UNOBSERVED_REGION"
        assert stat["is_anatomical_region_visible"] is False

        ll_l = meas_map["lower_leg_length_left"]
        assert ll_l["confidence_status"] == "MODEL_DERIVED_UNOBSERVED_REGION"
        assert ll_l["is_anatomical_region_visible"] is False

        # Shoulders & Torso must be observed
        sb = meas_map["shoulder_breadth"]
        assert sb["is_anatomical_region_visible"] is True
        assert sb["confidence_status"] == "OBSERVATION_SUPPORTED"


class TestStage24SMPLMeshQualityAndDeliverables:
    """Verifies mesh topology, manifold preservation, and deliverable files."""

    def test_mesh_quality(self, stage24_output_dir: Path):
        q_file = stage24_output_dir / "mesh_quality_report.json"
        assert q_file.exists()
        with open(q_file, "r", encoding="utf-8") as f:
            q = json.load(f)

        assert q["vertex_count"] == 6890
        assert q["face_count"] == 13776
        assert q["is_edge_manifold"] is True
        assert q["is_vertex_manifold"] is True
        assert q["has_degenerate_faces"] is False

    def test_deliverables_inventory(self, stage24_output_dir: Path):
        required_files = [
            "stage24_report.md",
            "smpl_setup_status.json",
            "smpl_fit_parameters.json",
            "smpl_fit_metrics.json",
            "smpl_measurements.json",
            "measurement_quality.json",
            "mesh_quality_report.json",
            "integration_test_results.json",
            "baseline_smpl_mesh.ply",
            "baseline_smpl_mesh.obj",
            "fitted_smpl_mesh.ply",
            "fitted_smpl_mesh.obj",
            "point_cloud_smpl_overlay.png",
            "anny_vs_smpl_comparison.png",
            "stage24_smpl_summary.png",
        ]
        for fname in required_files:
            fpath = stage24_output_dir / fname
            assert fpath.exists(), f"Required Stage 24 deliverable missing: {fname}"
            assert fpath.stat().st_size > 0, f"Deliverable is empty: {fname}"

    def test_verdict_validity(self, stage24_output_dir: Path):
        test_file = stage24_output_dir / "integration_test_results.json"
        assert test_file.exists()
        with open(test_file, "r", encoding="utf-8") as f:
            res = json.load(f)

        allowed_verdicts = [v.value for v in Stage24Verdict]
        assert res["verdict"] in allowed_verdicts, f"Invalid verdict: {res['verdict']}"
        assert res["fitting_verdict"] == Stage24Verdict.SMPL_FITTING_CONVERGED.value

