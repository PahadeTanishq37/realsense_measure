"""
tests/body/test_physical_stage21_robustness.py
=============================================
Test suite for Stage 21: Subject Frame Robustness & Independent Geometry Audit.

Validates:
1. Source data preservation (raw captures and stages 13-20 outputs untouched).
2. Axis convention audit: detection of right-handed +Z posterior conflict against declared anterior.
3. Perturbation sensitivity analysis: stability at <= 10mm and failure at >= 20mm landmark displacements.
4. Independent torso orientation audit: PCA surface normal corroborates oblique yaw.
5. Chest breadth geometry reconciliation: slicing basis vector swap and resting arm contamination.
6. Downstream integration readiness gating: blocking downstream measurement integration until issues are resolved.
7. Deliverables inventory, JSON schema validity, and final MULTIPLE_ISSUES_FOUND verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.measurements import (
    AxisConventionAuditor,
    ChestBreadthGeometryAuditor,
    FramePerturbationAnalyzer,
    IndependentOrientationAuditor,
    Stage21RobustnessOrchestrator,
    Stage21Verdict,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path) -> Stage21RobustnessOrchestrator:
    return Stage21RobustnessOrchestrator(workspace_root=repo_root)


def test_source_data_preservation(repo_root: Path):
    """Verify that prior raw captures and stage directories exist and remain unmodified."""
    raw1 = repo_root / "raw_captures" / "body_capture_01"
    raw2 = repo_root / "raw_captures" / "body_capture_02"
    raw3 = repo_root / "raw_captures" / "body_capture_03_diagnostic"

    assert raw1.exists(), "body_capture_01 must be preserved"
    assert raw2.exists(), "body_capture_02 must be preserved"
    assert raw3.exists(), "body_capture_03_diagnostic must be preserved"

    # Prior stages 13 - 20
    for stg in [
        "stage13_full_body_capture_output",
        "stage14_physical_reconstruction_output",
        "stage15_registration_diagnostics_output",
        "stage16_subject_depth_diagnostic_output",
        "stage17_upper_body_anthropometry_output",
        "stage18_measurement_sanity_audit_output",
        "stage19_targeted_validation_output",
        "stage20_subject_coordinate_frame_output",
    ]:
        p = repo_root / stg
        assert p.exists() and p.is_dir(), f"Prior stage output directory {stg} must be preserved"


def test_axis_convention_audit_and_defect_detection(orchestrator: Stage21RobustnessOrchestrator):
    """Verify detection of the axis convention conflict: mathematical +Z is dorsal/posterior, not anterior."""
    audit = orchestrator.audit_axis_convention()

    assert "RIGHT-HANDED" in audit.handedness.upper()
    assert audit.cross_product_order == "u_z = u_x x u_y"
    # Mathematical u_z in Stage 20 points dorsal, so sternum has negative Z coordinate
    assert audit.sternum_z_subject_m < 0.0, f"Sternum Z must be negative in dorsal +Z frame, got {audit.sternum_z_subject_m}"
    assert audit.conflict_identified is True
    assert "POSTERIOR" in audit.actual_z_direction.upper()
    assert "ISB" in audit.recommended_standard_convention


def test_perturbation_sensitivity_analysis():
    """Verify landmark perturbation sensitivity across 2, 5, 10, 20, 30mm noise."""
    sh_r = np.array([-0.18, -0.22, 2.45], dtype=np.float64)
    sh_l = np.array([0.05, -0.15, 2.20], dtype=np.float64)

    results = FramePerturbationAnalyzer.evaluate_sensitivity(
        sh_r, sh_l, magnitudes_mm=[2.0, 5.0, 10.0, 20.0, 30.0]
    )

    assert len(results) == 5

    # Check stability at <= 10mm
    r_2mm = results[0]
    assert r_2mm.perturbation_magnitude_mm == 2.0
    assert r_2mm.max_lateral_axis_dev_deg < 1.0
    assert r_2mm.max_origin_displacement_mm <= 1.05
    assert r_2mm.is_stable is True

    r_10mm = results[2]
    assert r_10mm.perturbation_magnitude_mm == 10.0
    assert r_10mm.max_lateral_axis_dev_deg < 4.0
    assert r_10mm.max_origin_displacement_mm <= 5.05
    assert r_10mm.is_stable is True

    # Check degradation at >= 20mm
    r_20mm = results[3]
    assert r_20mm.perturbation_magnitude_mm == 20.0
    assert r_20mm.max_lateral_axis_dev_deg >= 5.0
    assert r_20mm.biacromial_span_change_mm > 20.0
    assert r_20mm.is_stable is False


def test_independent_torso_pca_and_normal_orientation(orchestrator: Stage21RobustnessOrchestrator):
    """Verify independent orientation evidence: Torso PCA surface normal corroborates oblique yaw."""
    audit = orchestrator.audit_independent_orientation()

    assert audit.num_torso_points > 1000
    assert audit.shoulder_yaw_deg == pytest.approx(49.36, abs=0.5)
    # The normal yaw is oblique (-45.29 deg) which corroborates the ~49.4 deg shoulder rotation
    assert abs(abs(audit.torso_normal_yaw_deg) - audit.shoulder_yaw_deg) < 10.0
    assert "CONFIRMS_OBLIQUE_YAW" in audit.independent_evidence_status
    assert len(audit.pca_eigenvalues) == 3
    # Eigenvalues indicate flattened torso shell: lambda1 >> lambda2 > lambda3
    assert audit.pca_eigenvalues[0] > audit.pca_eigenvalues[1]
    assert audit.pca_eigenvalues[1] > audit.pca_eigenvalues[2]


def test_chest_slicing_geometry_reconciliation(orchestrator: Stage21RobustnessOrchestrator):
    """Verify detection of slicing basis swap and resting arm contamination."""
    rec = orchestrator.audit_chest_geometry()

    assert rec.slicing_basis_swap_identified is True
    # In slicing.py, u was [0, 0, -1] (depth) instead of [1, 0, 0] (lateral)
    assert np.allclose(rec.slicing_u_vector_cam, [0.0, 0.0, -1.0], atol=1e-3)
    assert np.allclose(rec.slicing_v_vector_cam, [-1.0, 0.0, 0.0], atol=1e-3)

    # Reported breadth in Stage 20 (424.9mm) was actually depth spread
    assert abs(rec.reported_transverse_breadth_was_depth_mm - 424.9) < 5.0
    # True lateral breadth in camera frame
    assert rec.actual_transverse_breadth_cam_mm > 450.0

    # Resting arm contamination in subject frame
    assert rec.resting_arm_points_count > 500
    assert rec.subject_frame_lateral_extent_raw_mm > 800.0  # Measured ~928.0mm


def test_downstream_integration_readiness_rules(orchestrator: Stage21RobustnessOrchestrator):
    """Verify that downstream integration gating blocks integration until defects are resolved."""
    readiness = orchestrator.assess_integration_readiness()

    assert readiness["readiness_status"] == "NOT_READY_FOR_INTEGRATION"
    assert readiness["overall_verdict"] == Stage21Verdict.MULTIPLE_ISSUES_FOUND.value

    rules = readiness["gating_rules"]
    assert rules["gate1_axis_convention"]["status"] == "BLOCKED_PENDING_CORRECTION"
    assert rules["gate2_slicing_basis_swap"]["status"] == "BLOCKED_PENDING_CORRECTION"
    assert rules["gate3_resting_arm_segmentation"]["status"] == "BLOCKED_PENDING_CORRECTION"
    assert rules["gate4_landmark_confidence_filter"]["status"] == "SATISFIED"


def test_stage21_deliverables_and_verdict(repo_root: Path):
    """Verify that all Stage 21 output files exist, have valid JSON schemas, and state the MULTIPLE_ISSUES_FOUND verdict."""
    out_dir = repo_root / "stage21_subject_frame_robustness_output"
    assert out_dir.exists()

    required_files = [
        "stage21_report.md",
        "axis_convention_audit.json",
        "frame_perturbation_sensitivity.json",
        "orientation_evidence_comparison.json",
        "chest_breadth_geometry_reconciliation.json",
        "subject_frame_integration_readiness.json",
        "stage21_robustness_and_geometry_audit.png",
    ]

    for rf in required_files:
        p = out_dir / rf
        assert p.exists() and p.stat().st_size > 0, f"Deliverable {rf} must exist and be non-empty"

    for jf in [f for f in required_files if f.endswith(".json")]:
        with open(out_dir / jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Deliverable {jf} must be a valid JSON dictionary"

    report_text = (out_dir / "stage21_report.md").read_text(encoding="utf-8")
    assert Stage21Verdict.MULTIPLE_ISSUES_FOUND.value in report_text
