"""Tests for Problem 4 - Stage 6: Manual Anthropometric Ground-Truth Validation.

Validates:
1. ISO 7250 / ISAK anatomical protocol mapping and compatibility checks.
2. Anti-fabrication integrity (manual fields empty by default).
3. Synthetic ground-truth validation (error metrics, multi-repeat averaging, Bland-Altman LoA).
4. Quality tier stratification and Stage 2 vs Stage 4 error comparison.
5. Physical artifacts in stage6_validation_output/.
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from tools.run_physical_stage6_validation import (
    ANTHROPOMETRIC_PROTOCOLS,
    PhysicalStage6Validator,
    ValidationRow,
    TierValidationStats,
)


STAGE5_CSV = Path("stage5_anthropometric_output/dual_mesh_measurement_comparison.csv")
STAGE6_DIR = Path("stage6_validation_output")


def test_anatomical_protocols_definitions():
    """Verify ISO 7250 / ISAK definitions and neck_base_circumference incompatibility exclusion."""
    assert len(ANTHROPOMETRIC_PROTOCOLS) == 19
    # neck_base_circumference must be explicitly excluded as non-comparable
    assert ANTHROPOMETRIC_PROTOCOLS["neck_base_circumference"]["comparable"] is False
    assert "INCOMPATIBLE" in ANTHROPOMETRIC_PROTOCOLS["neck_base_circumference"]["protocol"]

    # Key ISO/ISAK circumferences must be marked comparable
    for metric in [
        "hip_circumference",
        "waist_circumference",
        "chest_circumference",
        "mid_thigh_circumference_left",
        "mid_thigh_circumference_right",
        "stature",
    ]:
        assert ANTHROPOMETRIC_PROTOCOLS[metric]["comparable"] is True
        assert len(ANTHROPOMETRIC_PROTOCOLS[metric]["protocol"]) > 20


def test_validation_template_anti_fabrication(tmp_path):
    """Verify that generated template never fabricates manual measurements."""
    if not STAGE5_CSV.exists():
        pytest.skip(f"Stage 5 CSV not found at {STAGE5_CSV}")

    validator = PhysicalStage6Validator(stage5_csv_path=STAGE5_CSV, output_dir=tmp_path)
    stage5_records = validator.load_stage5_measurements()
    template_rows = validator.build_validation_template(stage5_records)

    assert len(template_rows) == len(stage5_records)
    assert len(template_rows) == 19

    # Crucial check: All manual fields must be empty / NaN
    for row in template_rows:
        assert math.isnan(row.manual_measurement_mm)
        assert math.isnan(row.manual_repeat_1_mm)
        assert math.isnan(row.manual_repeat_2_mm)
        assert math.isnan(row.manual_repeat_3_mm)
        assert row.has_manual_data is False

    # Check comparable flags
    neck_row = [r for r in template_rows if r.measurement_name == "neck_base_circumference"][0]
    assert neck_row.comparable is False

    hip_row = [r for r in template_rows if r.measurement_name == "hip_circumference"][0]
    assert hip_row.comparable is True
    assert hip_row.stage5_status == "observation_supported"


def test_synthetic_ground_truth_error_computation(tmp_path):
    """Verify mathematical accuracy of error metrics and multi-repeat averaging using synthetic ground truth."""
    validator = PhysicalStage6Validator()

    # Create mock template rows
    mock_rows = [
        ValidationRow(
            subject_id="SUBJ_TEST",
            capture_id="CAP_TEST",
            measurement_name="hip_circumference",
            anatomical_protocol="ISO 7250 / ISAK standard",
            manual_measurement_mm=float("nan"),
            manual_repeat_1_mm=float("nan"),
            manual_repeat_2_mm=float("nan"),
            manual_repeat_3_mm=float("nan"),
            rgbd_stage2_measurement_mm=1020.0,
            rgbd_stage4_measurement_mm=1010.0,
            stage5_status="observation_supported",
            observed_coverage_percent=95.0,
            comparable=True,
            clothing_condition="casual_clothing",
            operator_id="OP_01",
        ),
        ValidationRow(
            subject_id="SUBJ_TEST",
            capture_id="CAP_TEST",
            measurement_name="mid_thigh_circumference_left",
            anatomical_protocol="ISAK standard",
            manual_measurement_mm=float("nan"),
            manual_repeat_1_mm=float("nan"),
            manual_repeat_2_mm=float("nan"),
            manual_repeat_3_mm=float("nan"),
            rgbd_stage2_measurement_mm=520.0,
            rgbd_stage4_measurement_mm=510.0,
            stage5_status="observation_supported",
            observed_coverage_percent=85.0,
            comparable=True,
            clothing_condition="casual_clothing",
            operator_id="OP_01",
        ),
        ValidationRow(
            subject_id="SUBJ_TEST",
            capture_id="CAP_TEST",
            measurement_name="mid_thigh_circumference_right",
            anatomical_protocol="ISAK standard",
            manual_measurement_mm=float("nan"),
            manual_repeat_1_mm=float("nan"),
            manual_repeat_2_mm=float("nan"),
            manual_repeat_3_mm=float("nan"),
            rgbd_stage2_measurement_mm=515.0,
            rgbd_stage4_measurement_mm=505.0,
            stage5_status="observation_supported",
            observed_coverage_percent=100.0,
            comparable=True,
            clothing_condition="casual_clothing",
            operator_id="OP_01",
        ),
        ValidationRow(
            subject_id="SUBJ_TEST",
            capture_id="CAP_TEST",
            measurement_name="neck_base_circumference",
            anatomical_protocol="INCOMPATIBLE",
            manual_measurement_mm=float("nan"),
            manual_repeat_1_mm=float("nan"),
            manual_repeat_2_mm=float("nan"),
            manual_repeat_3_mm=float("nan"),
            rgbd_stage2_measurement_mm=880.0,
            rgbd_stage4_measurement_mm=880.0,
            stage5_status="invalid",
            observed_coverage_percent=0.0,
            comparable=False,  # Excluded
            clothing_condition="casual_clothing",
            operator_id="OP_01",
        ),
    ]

    # Write mock manual ground truth CSV
    mock_csv_path = tmp_path / "mock_manual.csv"
    with open(mock_csv_path, "w", encoding="utf-8") as f:
        f.write("measurement_name,manual_measurement_mm,manual_repeat_1_mm,manual_repeat_2_mm,manual_repeat_3_mm\n")
        f.write("hip_circumference,1000.0,,,\n")
        f.write("mid_thigh_circumference_left,500.0,,,\n")
        # Test 3 repeats averaging: (500 + 502 + 498) / 3 = 500.0
        f.write("mid_thigh_circumference_right,,500.0,502.0,498.0\n")
        f.write("neck_base_circumference,400.0,,,\n")

    updated_rows = validator.ingest_manual_measurements(mock_csv_path, mock_rows)
    evaluated_rows, tier_stats, summary = validator.compute_validation_statistics(updated_rows)

    # Check right thigh multi-repeat
    right_thigh = [r for r in evaluated_rows if r.measurement_name == "mid_thigh_circumference_right"][0]
    assert np.isclose(right_thigh.manual_measurement_mm, 500.0)
    assert np.isclose(right_thigh.stage4_signed_error_mm, 5.0)  # 505 - 500 = +5
    assert np.isclose(right_thigh.stage4_abs_error_mm, 5.0)
    assert np.isclose(right_thigh.stage4_pct_error, 1.0)
    assert np.isclose(right_thigh.personalization_error_reduction_mm, 10.0)  # |15| - |5| = 10

    # Check observation_supported tier aggregates:
    # manual: [1000, 500, 500]
    # s2: [1020, 520, 515] -> errs: [20, 20, 15] -> MAE = 18.333, bias = +18.333
    # s4: [1010, 510, 505] -> errs: [10, 10, 5] -> MAE = 8.333, bias = +8.333
    obs_stats = tier_stats["observation_supported"]
    assert obs_stats.sample_count == 3
    assert np.isclose(obs_stats.stage2_mae_mm, 18.333, atol=1e-2)
    assert np.isclose(obs_stats.stage4_mae_mm, 8.333, atol=1e-2)
    assert np.isclose(obs_stats.mean_error_reduction_mm, 10.0, atol=1e-2)

    # Incompatible neck_base must be excluded from all_comparable
    all_stats = tier_stats["all_comparable"]
    assert all_stats.sample_count == 3


def test_bland_altman_limits_of_agreement():
    """Verify that Bland-Altman limits are exactly bias +/- 1.96 * SD."""
    validator = PhysicalStage6Validator()

    # Setup 5 measurements with errors: [2, 4, 6, 8, 10]
    # manual: 100, s4: [102, 104, 106, 108, 110]
    rows = []
    for i, err in enumerate([2.0, 4.0, 6.0, 8.0, 10.0]):
        rows.append(
            ValidationRow(
                subject_id="TEST",
                capture_id="TEST",
                measurement_name=f"metric_{i}",
                anatomical_protocol="ISO",
                manual_measurement_mm=100.0,
                manual_repeat_1_mm=float("nan"),
                manual_repeat_2_mm=float("nan"),
                manual_repeat_3_mm=float("nan"),
                rgbd_stage2_measurement_mm=100.0 + err + 5.0,
                rgbd_stage4_measurement_mm=100.0 + err,
                stage5_status="observation_supported",
                observed_coverage_percent=100.0,
                comparable=True,
                clothing_condition="casual_clothing",
                operator_id="OP_01",
                has_manual_data=True,
            )
        )

    _, tier_stats, _ = validator.compute_validation_statistics(rows)
    stats = tier_stats["observation_supported"]

    mean_bias = 6.0
    errs = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    sd = float(np.std(errs, ddof=1))
    expected_lower = mean_bias - 1.96 * sd
    expected_upper = mean_bias + 1.96 * sd

    assert np.isclose(stats.stage4_bias_mm, mean_bias)
    assert np.isclose(stats.stage4_sd_mm, sd)
    assert np.isclose(stats.bland_altman_loa_stage4[0], expected_lower)
    assert np.isclose(stats.bland_altman_loa_stage4[1], expected_upper)


def test_physical_stage6_artifacts_exist():
    """Verify that all required Stage 6 artifacts exist on disk and are valid."""
    required_files = [
        "manual_validation_template.csv",
        "validation_results.csv",
        "validation_summary.json",
        "measurement_error_comparison.png",
        "stage2_vs_stage4_error.png",
        "bland_altman_plots.png",
        "stage6_validation_report.md",
    ]
    for fname in required_files:
        fpath = STAGE6_DIR / fname
        assert fpath.exists(), f"Required Stage 6 artifact missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"


def test_physical_stage6_summary_integrity():
    """Verify summary JSON schema, metadata, and anti-fabrication status."""
    summary_path = STAGE6_DIR / "validation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "TEMPLATE_READY_MANUAL_INPUT_PENDING"
    assert summary["has_empirical_ground_truth"] is False
    assert summary["meta"]["total_metrics_evaluated"] == 19
    assert summary["meta"]["comparable_metrics"] == 18
    assert summary["meta"]["incompatible_metrics"] == 1
    assert summary["meta"]["populated_manual_samples"] == 0

    # Ensure tier sample counts are 0 since no manual data was fabricated
    assert summary["tier_statistics"]["observation_supported"]["sample_count"] == 0
    assert summary["tier_statistics"]["partially_observed"]["sample_count"] == 0
    assert summary["tier_statistics"]["model_inferred"]["sample_count"] == 0
