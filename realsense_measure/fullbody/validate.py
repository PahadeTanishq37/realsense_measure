"""
realsense_measure/fullbody/validate.py

Comparison of model-free scan measurements against ground-truth physical tape measurements.
Computes per-dimension error, system bias, absolute residuals, and tape repeatability floor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from realsense_measure.fullbody.types import MeasurementResult

logger = logging.getLogger(__name__)


def load_tape_measurements(
    tape_path: Union[str, Path]
) -> Tuple[Dict[str, float], Dict[str, float], Optional[str]]:
    """
    Load tape measurements JSON file.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Optional[str]]:
        (tape_values_mm, repeatability_floor_mm, subject_id)
    """
    p = Path(tape_path)
    if not p.exists():
        raise FileNotFoundError(f"Tape measurement file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    subject_id = data.get("subject_id")
    raw_measurements = data.get("measurements_mm", {})

    tape_values: Dict[str, float] = {}
    repeatability: Dict[str, float] = {}

    for k, v in raw_measurements.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            tape_values[k] = float(v)
            repeatability[k] = 0.0
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            valid_nums = [float(x) for x in v if x is not None and np.isfinite(x)]
            if valid_nums:
                tape_values[k] = float(np.mean(valid_nums))
                repeatability[k] = float(np.std(valid_nums)) if len(valid_nums) > 1 else 0.0

    return tape_values, repeatability, subject_id


def compare_with_tape(
    scan_measurements: Dict[str, MeasurementResult],
    tape_values: Dict[str, float],
    repeatability_floor: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare scan-derived measurements with physical tape values.

    Returns
    -------
    Dict[str, Any]:
        Validation summary containing per-dimension comparison records and aggregate metrics.
    """
    records = []
    signed_errors = []
    abs_errors = []

    # Measurement name aliases mapping scan keys to tape keys
    aliases = {
        "stature": "stature",
        "neck_base_circumference": "neck_base_circumference",
        "chest_circumference": "chest_circumference",
        "waist_circumference": "waist_circumference",
        "hip_circumference": "hip_circumference",
        "mid_thigh_circumference_left": "mid_thigh_circumference_left",
        "mid_thigh_circumference_right": "mid_thigh_circumference_right",
        "max_calf_circumference_left": "max_calf_circumference_left",
        "max_calf_circumference_right": "max_calf_circumference_right",
        "upper_arm_circumference_left": "upper_arm_circumference_left",
        "upper_arm_circumference_right": "upper_arm_circumference_right",
        "forearm_circumference_left": "forearm_circumference_left",
        "forearm_circumference_right": "forearm_circumference_right",
    }

    for scan_name, scan_m in scan_measurements.items():
        tape_name = aliases.get(scan_name, scan_name)
        if tape_name in tape_values:
            tape_val = tape_values[tape_name]
            scan_val = scan_m.value_mm

            if np.isfinite(scan_val) and np.isfinite(tape_val) and tape_val > 0:
                err = scan_val - tape_val
                abs_err = abs(err)
                rel_err_pct = (abs_err / tape_val) * 100.0

                rep = repeatability_floor.get(tape_name, 0.0) if repeatability_floor else 0.0

                signed_errors.append(err)
                abs_errors.append(abs_err)

                records.append({
                    "dimension": tape_name,
                    "scan_mm": round(scan_val, 1),
                    "tape_mm": round(tape_val, 1),
                    "error_mm": round(err, 1),
                    "abs_error_mm": round(abs_err, 1),
                    "rel_error_pct": round(rel_err_pct, 2),
                    "tape_std_mm": round(rep, 1),
                    "coverage": round(scan_m.coverage, 2),
                    "quality_flag": scan_m.quality_flag,
                })

    bias = float(np.mean(signed_errors)) if signed_errors else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    max_err = float(np.max(abs_errors)) if abs_errors else 0.0

    return {
        "num_compared": len(records),
        "mean_error_bias_mm": round(bias, 2),
        "mean_absolute_error_mm": round(mae, 2),
        "max_error_mm": round(max_err, 2),
        "comparisons": records,
    }


def format_validation_table(report: Dict[str, Any]) -> str:
    """Format validation comparison as a terminal table."""
    lines = [
        f"{'Dimension':<30} | {'Scan (mm)':>9} | {'Tape (mm)':>9} | {'Diff (mm)':>9} | {'Rel %':>6} | {'Qual':<7}",
        "-" * 80,
    ]
    for c in report.get("comparisons", []):
        lines.append(
            f"{c['dimension']:<30} | {c['scan_mm']:9.1f} | {c['tape_mm']:9.1f} | {c['error_mm']:+9.1f} | {c['rel_error_pct']:5.1f}% | {c['quality_flag']:<7}"
        )
    lines.append("-" * 80)
    lines.append(
        f"Summary: Bias = {report.get('mean_error_bias_mm', 0.0):+.1f} mm | MAE = {report.get('mean_absolute_error_mm', 0.0):.1f} mm | Max = {report.get('max_error_mm', 0.0):.1f} mm"
    )
    return "\n".join(lines)

