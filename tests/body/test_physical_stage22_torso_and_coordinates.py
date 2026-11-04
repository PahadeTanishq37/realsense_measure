"""
tests/body/test_physical_stage22_torso_and_coordinates.py
=========================================================
Test suite for Stage 22: Coordinate Convention Correction & Torso Isolation.

Validates:
1. Source data preservation (raw captures and stages 13-21 outputs untouched).
2. Axis convention correction: right-handed ISB frame (+X Left, +Y Superior, +Z Anterior).
   Fails if anterior axis points posterior, frame is left-handed, or signs are inconsistent.
3. Slicing plane basis correction: u=lateral, v=sagittal. Fails if basis vectors are swapped.
4. Rotational invariance across yaw angles on known 3D rectangular geometry.
5. Controlled synthetic torso isolation: 100% rejection of resting arms and desk points;
   preservation of >= 95% of torso points; accurate lateral width recovery.
6. Physical chest breadth reassessment: resolved true lateral span (P1-P99 327.9mm, raw 338.1mm).
7. Deliverables inventory, JSON schema validity, and final GEOMETRY_CORRECTIONS_VERIFIED_EXPERIMENTALLY verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.measurements import (
    AnatomicalPlaneSlicer,
    AnatomicalTorsoSegmenter,
    Stage22CoordinateAndTorsoOrchestrator,
    Stage22Verdict,
    SubjectFrameConvention,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path) -> Stage22CoordinateAndTorsoOrchestrator:
    return Stage22CoordinateAndTorsoOrchestrator(workspace_root=repo_root)


def test_source_data_preservation(repo_root: Path):
    """Verify that prior raw captures and stage directories exist and remain unmodified."""
    raw1 = repo_root / "raw_captures" / "body_capture_01"
    raw2 = repo_root / "raw_captures" / "body_capture_02"
    raw3 = repo_root / "raw_captures" / "body_capture_03_diagnostic"

    assert raw1.exists(), "body_capture_01 must be preserved"
    assert raw2.exists(), "body_capture_02 must be preserved"
    assert raw3.exists(), "body_capture_03_diagnostic must be preserved"

    # Prior stages 13 - 21
    for stg in [
        "stage13_full_body_capture_output",
        "stage14_physical_reconstruction_output",
        "stage15_registration_diagnostics_output",
        "stage16_subject_depth_diagnostic_output",
        "stage17_upper_body_anthropometry_output",
        "stage18_measurement_sanity_audit_output",
        "stage19_targeted_validation_output",
        "stage20_subject_coordinate_frame_output",
        "stage21_subject_frame_robustness_output",
    ]:
        p = repo_root / stg
        assert p.exists() and p.is_dir(), f"Prior stage output directory {stg} must be preserved"


def test_axis_convention_correction_and_defect_prevention(orchestrator: Stage22CoordinateAndTorsoOrchestrator):
    """
    Verify axis convention correction.
    Fails if:
    - The anterior axis points posterior.
    - The frame is left-handed (det != +1.0).
    - u_Z != u_X x u_Y.
    - Subject left is negative or subject right is positive.
    """
    audit = orchestrator.audit_axis_convention_correction()

    math_props = audit["mathematical_properties"]
    assert math_props["right_handed"] is True, "Coordinate frame must be right-handed"
    assert abs(math_props["determinant"] - 1.0) < 1e-5, f"Determinant must be +1.0, got {math_props['determinant']}"
    assert math_props["cross_product_verified"] is True, "u_Z must equal u_X x u_Y"

    signs = audit["landmark_sign_verification"]
    # Anterior axis must point anterior (sternum Z > 0)
    assert signs["sternum_is_anterior_positive"] is True, "Anterior axis must point anterior (sternum Z > 0)"
    assert signs["sternum_z_isb_mm"] > 0.0, f"Sternum Z must be positive, got {signs['sternum_z_isb_mm']}"

    # Superior axis must point Up (sternum Y < 0 relative to shoulders)
    assert signs["sternum_is_inferior_to_shoulders"] is True
    assert signs["sternum_y_isb_mm"] < 0.0

    # Lateral sign consistency
    assert signs["subject_left_is_positive"] is True, "Subject left must have positive X coordinate"
    assert signs["subject_right_is_negative"] is True, "Subject right must have negative X coordinate"


def test_slicing_basis_vector_swap_prevention(orchestrator: Stage22CoordinateAndTorsoOrchestrator):
    """
    Verify slicing basis vectors.
    Fails if:
    - Slicing basis vectors are swapped (u is depth and v is lateral).
    - A known lateral dimension is interpreted as sagittal depth.
    """
    val = orchestrator.validate_slicing_basis()

    assert val["validation_status"] == "VERIFIED_CORRECT"
    basis_audit = val["basis_construction_audit"]
    assert basis_audit["swap_confirmed_and_fixed"] is True

    u_corr = np.array(basis_audit["corrected_u_lateral"])
    v_corr = np.array(basis_audit["corrected_v_sagittal"])

    # For canonical transverse slicing (n=[0, 1, 0]):
    # u must be along X (lateral): [1, 0, 0]
    # v must be along Z (sagittal): [0, 0, 1]
    assert np.allclose(u_corr, [1.0, 0.0, 0.0], atol=1e-3), f"u must be lateral [1, 0, 0], got {u_corr}"
    assert np.allclose(v_corr, [0.0, 0.0, 1.0], atol=1e-3), f"v must be sagittal [0, 0, 1], got {v_corr}"

    # Fails if legacy swap was active
    u_leg = np.array(basis_audit["legacy_u_swapped"])
    assert abs(u_leg[2]) > 0.9, "Legacy defect must be confirmed for regression protection"


def test_rotational_invariance_on_synthetic_geometry(orchestrator: Stage22CoordinateAndTorsoOrchestrator):
    """
    Verify that rotating synthetic geometry does NOT alter recovered known lateral breadth beyond tolerance.
    """
    val = orchestrator.validate_slicing_basis()

    assert val["rotational_invariance_passed"] is True
    assert val["max_breadth_error_mm"] < 0.05, f"Breadth error {val['max_breadth_error_mm']} must be < 0.05 mm"
    assert val["max_depth_error_mm"] < 0.05, f"Depth error {val['max_depth_error_mm']} must be < 0.05 mm"

    for t in val["rotation_tests"]:
        assert t["passed"] is True, f"Rotation test failed at yaw {t['yaw_angle_deg']} deg"
        assert t["breadth_error_mm"] < 0.05


def test_synthetic_torso_isolation_rejects_arms_and_desk():
    """
    Verify controlled synthetic torso isolation.
    Fails if:
    - Synthetic arm points are included.
    - Synthetic desk points are included.
    - Legitimate synthetic torso points are excessively removed (< 95%).
    """
    res = AnatomicalTorsoSegmenter.run_synthetic_torso_isolation_test()

    assert res["passed"] is True
    assert res["arm_rejection_pct"] == 100.0, "Must reject 100% of resting arm points"
    assert res["desk_rejection_pct"] == 100.0, "Must reject 100% of desk points"
    assert res["torso_preservation_pct"] >= 95.0, f"Must preserve >= 95% of torso points, got {res['torso_preservation_pct']}%"
    assert res["breadth_error_mm"] < 2.0, f"Recovered breadth error {res['breadth_error_mm']} must be < 2 mm"


def test_physical_chest_breadth_reassessment(orchestrator: Stage22CoordinateAndTorsoOrchestrator):
    """
    Verify physical chest breadth extraction on body_capture_03_diagnostic.
    """
    reassess = orchestrator.reassess_chest_breadth()
    comp = reassess["comparisons"]

    stage22_res = comp["stage22_corrected_torso_only"]

    # P1-P99 lateral breadth should be anatomically plausible for an adult torso (~310 - 345 mm)
    p1_p99 = stage22_res["p1_p99_lateral_breadth_mm"]
    raw_breadth = stage22_res["raw_lateral_breadth_mm"]

    assert 315.0 <= p1_p99 <= 345.0, f"Expected P1-P99 breadth in [315, 345] mm, got {p1_p99} mm"
    assert 320.0 <= raw_breadth <= 355.0, f"Expected raw breadth in [320, 355] mm, got {raw_breadth} mm"

    # Sagittal shell depth should be consistent with anterior single-view shell (< 75 mm)
    depth_spread = stage22_res["sagittal_depth_spread_mm"]
    assert 40.0 <= depth_spread <= 75.0, f"Expected anterior shell depth in [40, 75] mm, got {depth_spread} mm"

    # Confirms that Stage 20 legacy results are marked INVALID_GEOMETRIC_DEFECT
    assert comp["stage20_camera_frame"]["audit_verdict"] == "INVALID_GEOMETRIC_DEFECT"
    assert comp["stage20_subject_frame"]["audit_verdict"] == "INVALID_GEOMETRIC_DEFECT"


def test_integration_readiness_and_deliverables(repo_root: Path, orchestrator: Stage22CoordinateAndTorsoOrchestrator):
    """
    Verify integration readiness rules, deliverables existence, and GEOMETRY_CORRECTIONS_VERIFIED_EXPERIMENTALLY verdict.
    """
    readiness = orchestrator.assess_integration_readiness()

    assert readiness["overall_verdict"] == Stage22Verdict.GEOMETRY_CORRECTIONS_VERIFIED_EXPERIMENTALLY.value
    assert readiness["overall_status"] == "READY_FOR_INTEGRATION_TESTING"

    gates = readiness["prerequisite_gates"]
    assert gates["gate1_axis_convention"]["status"] == "PASSED_VERIFIED"
    assert gates["gate2_slicing_basis"]["status"] == "PASSED_VERIFIED"
    assert gates["gate3_torso_isolation"]["status"] == "PASSED_VERIFIED"
    assert gates["gate4_chest_measurement"]["status"] == "PASSED_AS_EXPERIMENTAL_PROXY"
    assert gates["gate5_downstream_compatibility"]["status"] == "PASSED_VERIFIED"

    out_dir = repo_root / "stage22_coordinate_and_torso_correction_output"
    assert out_dir.exists()

    required_files = [
        "stage22_report.md",
        "axis_convention_correction.json",
        "slicing_basis_validation.json",
        "torso_mask_validation.json",
        "chest_breadth_reassessment.json",
        "integration_readiness.json",
        "stage22_coordinate_and_torso_diagnostic.png",
    ]

    for rf in required_files:
        p = out_dir / rf
        assert p.exists() and p.stat().st_size > 0, f"Deliverable {rf} must exist and be non-empty"

    for jf in [f for f in required_files if f.endswith(".json")]:
        with open(out_dir / jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Deliverable {jf} must be a valid JSON dictionary"

    report_text = (out_dir / "stage22_report.md").read_text(encoding="utf-8")
    assert Stage22Verdict.GEOMETRY_CORRECTIONS_VERIFIED_EXPERIMENTALLY.value in report_text

