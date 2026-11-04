"""
tests/fullbody/test_validate_and_sweet_spot.py

Unit tests for tape measurement validation arithmetic and sweet-spot configuration scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.sweet_spot import evaluate_capture_configuration
from realsense_measure.fullbody.types import MeasurementResult, QualityFlag
from realsense_measure.fullbody.validate import (
    compare_with_tape,
    format_validation_table,
    load_tape_measurements,
)


def test_validation_arithmetic_and_repeatability_floor(tmp_path: Path):
    """Verify tape comparison computes error, bias, MAE, and repeatability floor."""
    tape_json = tmp_path / "tape_test.json"
    data = {
        "subject_id": "test_01",
        "measurements_mm": {
            "stature": [1750, 1752, 1748],  # Repeated: mean 1750, std 1.63
            "chest_circumference": 900.0,
            "waist_circumference": 800.0,
            "hip_circumference": None,      # Missing in tape
        }
    }
    with open(tape_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    tape_vals, rep_floor, subj_id = load_tape_measurements(tape_json)
    assert subj_id == "test_01"
    assert tape_vals["stature"] == 1750.0
    assert np.isclose(rep_floor["stature"], 1.63, atol=0.05)

    scan_measurements = {
        "stature": MeasurementResult("stature", 1760.0, "vertical_height", 1.0, 0.0, QualityFlag.OK.value),
        "chest_circumference": MeasurementResult("chest_circumference", 910.0, "polar_contour", 0.9, 0.1, QualityFlag.OK.value),
        "waist_circumference": MeasurementResult("waist_circumference", 790.0, "polar_contour", 0.9, 0.1, QualityFlag.OK.value),
    }

    report = compare_with_tape(scan_measurements, tape_vals, rep_floor)
    assert report["num_compared"] == 3
    # Errors: stature +10, chest +10, waist -10 -> mean bias = (+10 + 10 - 10) / 3 = +3.33 mm
    assert np.isclose(report["mean_error_bias_mm"], 3.33, atol=0.05)
    # MAE = (10 + 10 + 10) / 3 = 10.0 mm
    assert np.isclose(report["mean_absolute_error_mm"], 10.0, atol=0.05)
    assert report["max_error_mm"] == 10.0

    table_text = format_validation_table(report)
    assert "chest_circumference" in table_text
    assert "Bias = +3.3 mm" in table_text


def test_sweet_spot_configuration_scoring():
    """Verify sweet spot evaluates quadratic noise score and identifies clipped regions."""
    cfg = FullBodyConfig()

    # Generate synthetic point cloud with high feet (Y >= 0.15 m -> feet clipped)
    np.random.seed(42)
    pts = np.random.uniform([-0.2, 0.15, 1.2], [0.2, 1.3, 1.4], (3000, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    sw_rep = evaluate_capture_configuration(pcd, None, cfg)

    assert sw_rep.total_score > 0.0
    assert "FEET_CLIPPED" in sw_rep.under_covered_parts
    assert any("Feet appear clipped" in r for r in sw_rep.recommendations)

