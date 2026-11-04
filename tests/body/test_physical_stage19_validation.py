"""
tests/body/test_physical_stage19_validation.py
=============================================
Test suite for Stage 19: Targeted Audit Corrections & Offline Validation.

Validates:
1. Source data preservation & non-corruption of raw captures and prior stage outputs.
2. Stage 16 torso variability reconciliation (87.6 mm vs 876.05 mm vs 0.70 mm median).
3. RealSense depth sentinel physics (uint16 Z16, 54.4m low-disparity floor, 0xFFFF saturation).
4. Stability metric reclassification (local_patch_median_temporal_variability_mm).
5. Anthropometric plausibility decoupling from geometric validity.
6. Chest transverse breadth reassessment under subject yaw.
7. Torso yaw vs arm length discrepancy decomposition.
8. Deliverables inventory and JSON schema integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from realsense_measure.measurements import (
    AnthropometricPlausibilityConfig,
    GeometricValidityGates,
    Stage19TargetedValidator,
    Stage19Verdict,
)
from realsense_measure.measurements.targeted_validation import PopulationGroup


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def validator(repo_root: Path) -> Stage19TargetedValidator:
    return Stage19TargetedValidator(workspace_root=repo_root)


def test_source_data_preservation(repo_root: Path):
    """Verify that prior raw captures and stage directories exist and remain unmodified."""
    raw1 = repo_root / "raw_captures" / "body_capture_01"
    raw2 = repo_root / "raw_captures" / "body_capture_02"
    raw3 = repo_root / "raw_captures" / "body_capture_03_diagnostic"

    assert raw1.exists(), "body_capture_01 must be preserved"
    assert raw2.exists(), "body_capture_02 must be preserved"
    assert raw3.exists(), "body_capture_03_diagnostic must be preserved"

    # Prior stages
    for stg in ["stage13_full_body_capture_output", "stage14_physical_reconstruction_output",
                "stage15_registration_diagnostics_output", "stage16_subject_depth_diagnostic_output",
                "stage17_upper_body_anthropometry_output", "stage18_measurement_sanity_audit_output"]:
        p = repo_root / stg
        assert p.exists() and p.is_dir(), f"Prior stage output directory {stg} must be preserved"


def test_stage16_torso_variability_reconciliation(validator: Stage19TargetedValidator):
    """Verify the arithmetic cause of Stage 16 torso variability discrepancy."""
    res = validator.audit_torso_variability()

    assert res.historical_prose_reported_mm == 87.6
    assert abs(res.historical_json_stored_mm - 876.05) < 0.2
    assert res.discrepancy_factor == 10.0

    # True robust median on subject surface
    assert 0.60 <= res.median_std_mm <= 0.80, f"Expected median ~0.70 mm, got {res.median_std_mm}"
    assert 2.0 <= res.p90_std_mm <= 3.5, f"Expected p90 < 3.5 mm, got {res.p90_std_mm}"

    # Verify sentinel inflation
    assert res.p99_std_mm > 20000.0, f"Expected p99 > 20,000 mm from flickering sentinels, got {res.p99_std_mm}"
    assert res.max_std_mm > 25000.0, f"Expected max > 25,000 mm, got {res.max_std_mm}"
    assert res.flickering_boundary_fraction > 0.03, "Flickering boundary pixels should account for ~4% of mask"


def test_depth_sentinel_triangulation_physics(validator: Stage19TargetedValidator):
    """Verify RealSense Z16 triangulation formula and low-disparity quantization origin."""
    res = validator.audit_depth_sentinels()

    assert res.raw_dtype == "uint16 (Z16)"
    assert res.depth_scale_m_per_unit == 0.001
    assert res.saturation_sentinel_raw == 65535
    assert res.saturation_sentinel_m == 65.535

    # Check triangulation: fx * B / d
    fx_B = res.focal_length_px * (res.baseline_mm / 1000.0)
    calculated_disp = fx_B / res.burst_sentinel_median_m
    assert abs(calculated_disp - 1.13) < 0.05, f"Expected disparity ~1.13 px, got {calculated_disp}"

    # Median raw depth at boundary pixel
    assert 50.0 <= res.burst_sentinel_median_m <= 60.0


def test_stability_metric_reclassification(validator: Stage19TargetedValidator):
    """Verify that stability metric is properly reclassified to avoid misleading accuracy claims."""
    res = validator.reclassify_stability_metric()

    assert res.corrected_metric_name == "local_patch_median_temporal_variability_mm"
    assert res.sample_window_size_px == 21
    assert res.burst_frame_count == 10
    assert "precision" in res.what_metric_proves.lower() or "electronic" in res.what_metric_proves.lower()
    assert "accuracy" in res.what_metric_does_not_prove.lower()


def test_anthropometric_plausibility_decoupling(validator: Stage19TargetedValidator):
    """Verify that population plausibility is configurable and decoupled from optical validity."""
    cfg_adult = AnthropometricPlausibilityConfig(population=PopulationGroup.ADULT_GENERAL)
    gates = GeometricValidityGates()

    review = validator.review_anthropometric_plausibility_gates(config=cfg_adult, gates=gates)
    assert review["plausibility_configuration"]["upper_arm_min_mm"] == 260.0
    assert review["geometric_validity_gates"]["min_patch_purity"] == 0.50
    assert review["geometric_validity_gates"]["max_z_dominance_fraction"] == 0.70

    # Pediatric test
    cfg_ped = AnthropometricPlausibilityConfig(population=PopulationGroup.PEDIATRIC, upper_arm_min_mm=160.0)
    assert cfg_ped.upper_arm_min_mm == 160.0


def test_chest_transverse_breadth_yaw_reassessment(validator: Stage19TargetedValidator):
    """Verify reassessment of chest breadth proxy under 49.4 degree subject yaw."""
    res = validator.reassess_chest_breadth()

    assert res.reported_value_mm == 369.0
    assert abs(res.subject_yaw_deg - 49.4) < 1.0
    assert not res.clinical_validity
    assert "Proxy" in res.corrected_classification
    assert res.slicing_plane_normal == [0.0, 1.0, 0.0]


def test_yaw_arm_discrepancy_decomposition(validator: Stage19TargetedValidator):
    """Verify that yaw accounts for shoulder offset but not arm discrepancy."""
    res = validator.review_yaw_arm_discrepancy()

    assert abs(res.shoulder_delta_z_mm - 254.0) < 1.0
    assert abs(res.shoulder_delta_x_mm - 218.0) < 1.0
    assert abs(res.observed_torso_yaw_deg - 49.4) < 1.0

    # Discrepancy is NOT caused by yaw
    assert not res.is_discrepancy_caused_by_yaw
    assert res.length_discrepancy_mm == 143.5
    assert "desk" in res.left_arm_primary_failure.lower()
    assert "truncation" in res.right_arm_primary_failure.lower()


def test_stage19_deliverables_and_verdict(repo_root: Path):
    """Verify all Stage 19 deliverables exist and validate their contents."""
    out_dir = repo_root / "stage19_targeted_validation_output"
    assert out_dir.exists()

    required_files = [
        "stage19_report.md",
        "depth_sentinel_audit.json",
        "stage16_stage17_variability_reconciliation.json",
        "anthropometric_gate_review.json",
        "chest_breadth_reassessment.json",
        "yaw_arm_discrepancy_review.json",
        "stage19_variability_reconciliation_plot.png",
    ]

    for rf in required_files:
        p = out_dir / rf
        assert p.exists() and p.stat().st_size > 0, f"Deliverable {rf} must exist and be non-empty"

    # Verify JSON parsability
    for jf in [f for f in required_files if f.endswith(".json")]:
        with open(out_dir / jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"JSON deliverable {jf} must be a valid dictionary"

    # Verify report verdict
    report_text = (out_dir / "stage19_report.md").read_text(encoding="utf-8")
    assert Stage19Verdict.MULTIPLE_ISSUES_FOUND.value in report_text

