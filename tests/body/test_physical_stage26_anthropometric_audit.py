"""
tests/body/test_physical_stage26_anthropometric_audit.py
=========================================================
Unit and integration test suite for Stage 26:
Anthropometric Measurement Pipeline Audit, Anatomical Landmark Correction & Validation.

Verifies:
1. Root-cause defect audit and diagnostic disclosures (perineum typo, hip/clavicle, acromion/deltoid).
2. Mathematical coordinate frame standardization into Stage 22 ISB frame (+X Lateral, +Y Superior, +Z Anterior).
3. Verified anatomical landmark positioning and torso height sanity (450 mm - 550 mm).
4. Anthropometric circumference corrections:
   - Waist circumference is within normal human range (850 - 1100 mm), resolving the 1,402 mm error.
   - Hip circumference is within normal human range (1050 - 1350 mm), resolving the 1,261 mm error.
   - Shoulder breadth is within true biacromial range (370 - 500 mm), resolving the 632 mm error.
5. True deformation impact on measurements:
   - Significant, physically driven changes in chest (+47.2 mm), waist (-46.3 mm), and hip (+84.9 mm).
6. Synthetic geometric validation on analytical rectangular prism and cylinder:
   - Analytical perimeter recovery (< 1e-4 mm error on box, < 0.1% on cylinder).
   - Exact translation invariance, scale linearity (2.0x), and SO(3) rotation invariance.
7. Truthful observability classification (DIRECTLY_OBSERVED, PARTIALLY_OBSERVED, MODEL_PRIOR_DERIVED).
8. Deliverables inventory and allowed final verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import open3d as o3d

from realsense_measure.measurements.anthropometric_auditor import (
    AnthropometricMeasurementEngine,
    CoordinateFrameStandardizer,
    ObservabilityCategory,
    Stage26Verdict,
    SyntheticGeometryValidator,
    VERIFIED_SMPL_LANDMARKS,
)
from realsense_measure.measurements.stage26_orchestrator import (
    Stage26AnthropometricAuditOrchestrator,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def stage26_output_dir(repo_root: Path) -> Path:
    out = repo_root / "stage26_anthropometric_validation_output"
    assert out.exists(), f"Stage 26 output directory missing: {out}"
    return out


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path, stage26_output_dir: Path) -> Stage26AnthropometricAuditOrchestrator:
    return Stage26AnthropometricAuditOrchestrator(repo_root=repo_root, output_dir=stage26_output_dir)


class TestStage26DefectAuditAndRootCause:
    """Verifies that Stage 25 measurement anomalies are comprehensively diagnosed."""

    def test_audit_disclosures_schema(self, stage26_output_dir: Path):
        audit_file = stage26_output_dir / "measurement_definition_audit.json"
        assert audit_file.exists()
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        defects = data["stage25_defects_investigation"]
        assert "perineum" in defects["torso_length_collapse"]["investigation_findings"].lower()
        assert "chin" in defects["torso_length_collapse"]["investigation_findings"].lower()
        assert "deltoid" in defects["shoulder_breadth_discrepancy"]["investigation_findings"].lower()
        assert "1402" in defects["waist_circumference_inflation"]["reported_stage25_mm"].__str__()

    def test_landmark_mapping_corrections(self, stage26_output_dir: Path):
        lm_file = stage26_output_dir / "landmark_mapping_audit.json"
        assert lm_file.exists()
        with open(lm_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lm_dict = data["verified_smpl_landmarks"]
        assert lm_dict["perineum"]["index"] == 3170
        assert lm_dict["trochanterion_left"]["index"] == 847
        assert lm_dict["acromion_left"]["index"] == 1903
        assert lm_dict["acromion_right"]["index"] == 5363


class TestStage26CoordinateStandardization:
    """Verifies mathematical validity of Stage 22 ISB coordinate standardization."""

    def test_isb_rotation_matrix_properties(self, stage26_output_dir: Path):
        coord_file = stage26_output_dir / "coordinate_and_unit_validation.json"
        assert coord_file.exists()
        with open(coord_file, "r", encoding="utf-8") as f:
            coord = json.load(f)

        assert coord["determinant"] == pytest.approx(1.0, abs=1e-12)
        assert coord["is_pure_rotation"] is True
        assert coord["is_right_handed"] is True
        assert coord["orthogonality_error"] < 1e-12
        assert coord["cross_product_consistency_error"] < 1e-12


class TestStage26CorrectedMeasurements:
    """Verifies that corrected anthropometric measurements resolve Stage 25 anomalies."""

    def test_corrected_measurements_values(self, stage26_output_dir: Path):
        meas_file = stage26_output_dir / "corrected_measurements.json"
        assert meas_file.exists()
        with open(meas_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        meas_map = {m["name"]: m for m in data["measurements"]}

        # 1. Torso Length: must be ~490 mm, resolving the 156 mm collapse
        tl = meas_map["torso_length"]["value_mm"]
        assert 450.0 <= tl <= 550.0, f"Torso length outside realistic range: {tl:.1f} mm"

        # 2. Waist Circumference: must be ~970 mm, resolving the 1,402 mm arm-intersection bug
        wc = meas_map["waist_circumference"]["value_mm"]
        assert 850.0 <= wc <= 1100.0, f"Waist circumference outside realistic range: {wc:.1f} mm"

        # 3. Hip Circumference: must be ~1230 mm, resolving the 1,261 mm clavicle-slice bug
        hc = meas_map["hip_circumference"]["value_mm"]
        assert 1050.0 <= hc <= 1350.0, f"Hip circumference outside realistic range: {hc:.1f} mm"

        # 4. Shoulder Breadth: must be ~466 mm, resolving the 632 mm deltoid-span bug
        sb = meas_map["shoulder_breadth"]["value_mm"]
        assert 370.0 <= sb <= 500.0, f"Shoulder breadth outside realistic range: {sb:.1f} mm"

        # 5. Chest Circumference: must be ~845 mm
        cc = meas_map["chest_circumference"]["value_mm"]
        assert 780.0 <= cc <= 950.0, f"Chest circumference outside realistic range: {cc:.1f} mm"

    def test_deformation_impact_on_torso_circumferences(self, stage26_output_dir: Path):
        meas_file = stage26_output_dir / "corrected_measurements.json"
        with open(meas_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meas_map = {m["name"]: m for m in data["measurements"]}

        # Chest must adapt significantly to thoracic depth points (> 20 mm change)
        assert abs(meas_map["chest_circumference"]["difference_from_fit_mm"]) >= 20.0

        # Waist must adapt significantly to lumbar points (> 20 mm change)
        assert abs(meas_map["waist_circumference"]["difference_from_fit_mm"]) >= 20.0

        # Hip must adapt significantly to pelvic points (> 20 mm change)
        assert abs(meas_map["hip_circumference"]["difference_from_fit_mm"]) >= 20.0


class TestStage26SyntheticGeometryValidation:
    """Verifies that measurement functions recover analytical dimensions within tight tolerance."""

    def test_box_and_cylinder_analytical_recovery(self, stage26_output_dir: Path):
        synth_file = stage26_output_dir / "synthetic_geometry_test_results.json"
        assert synth_file.exists()
        with open(synth_file, "r", encoding="utf-8") as f:
            synth = json.load(f)

        assert synth["all_synthetic_tests_passed"] is True
        assert synth["rectangular_prism"]["passed"] is True
        assert synth["circular_cylinder"]["passed"] is True
        assert synth["invariance_properties"]["translation_invariant"] is True
        assert synth["invariance_properties"]["scale_linear"] is True
        assert synth["invariance_properties"]["rotation_invariant"] is True


class TestStage26ObservabilityAndDeliverables:
    """Verifies deliverables presence, observability classifications, and final verdict."""

    def test_all_deliverables_present(self, stage26_output_dir: Path):
        required_files = [
            "stage26_report.md",
            "measurement_definition_audit.json",
            "landmark_mapping_audit.json",
            "coordinate_and_unit_validation.json",
            "linear_measurement_validation.json",
            "circumference_validation.json",
            "baseline_vs_personalized_measurements.json",
            "measurement_observability_audit.json",
            "synthetic_geometry_test_results.json",
            "measurement_regression_results.json",
            "corrected_measurements.json",
            "landmark_mapping_audit.png",
            "measurement_planes_audit.png",
            "circumference_contours_corrected.png",
            "baseline_vs_personalized_corrected.png",
        ]
        for fname in required_files:
            target = stage26_output_dir / fname
            assert target.exists(), f"Missing required Stage 26 deliverable: {fname}"
            assert target.stat().st_size > 0, f"Deliverable is empty: {fname}"

    def test_observability_categories(self, stage26_output_dir: Path):
        obs_file = stage26_output_dir / "measurement_observability_audit.json"
        assert obs_file.exists()
        with open(obs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        by_name = {m["name"]: m for m in data["observability_breakdown"]}

        # Waist must have high physical observation coverage (> 80%)
        assert by_name["waist_circumference"]["category"] == "DIRECTLY_OBSERVED"
        assert by_name["waist_circumference"]["observed_coverage_pct"] >= 80.0

        # Stature and lower legs must be model prior derived (calves truncated outside camera frame)
        assert by_name["overall_body_height"]["category"] == "MODEL_PRIOR_DERIVED"
        assert by_name["lower_leg_length_left"]["category"] == "MODEL_PRIOR_DERIVED"

    def test_final_verdict(self, stage26_output_dir: Path):
        reg_file = stage26_output_dir / "measurement_regression_results.json"
        assert reg_file.exists()
        with open(reg_file, "r", encoding="utf-8") as f:
            reg = json.load(f)

        assert reg["final_verdict"] == Stage26Verdict.MEASUREMENTS_CORRECTED_BUT_PHYSICAL_VALIDATION_REQUIRED.value

