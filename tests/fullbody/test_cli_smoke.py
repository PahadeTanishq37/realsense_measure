"""
tests/fullbody/test_cli_smoke.py

Smoke tests for fullbody CLI argument parsing and execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.cli import build_parser, main
from realsense_measure.fullbody.io import save_scan_result
from realsense_measure.fullbody.types import MeasurementResult, QualityFlag, ScanResult


def test_cli_parser_build():
    """Verify CLI parser builds with all subcommands."""
    parser = build_parser()
    subcommands = {
        "scan": ["scan", "--capture", "foo"],
        "measure": ["measure", "--scan", "foo"],
        "validate": ["validate", "--scan", "foo"],
        "sweet-spot": ["sweet-spot", "--scan", "foo"],
        "review": ["review", "--scan", "foo"],
        "keypoints-check": ["keypoints-check", "dummy.png"],
    }
    for sub, argv in subcommands.items():
        args = parser.parse_args(argv)
        assert args.subcommand == sub


def test_cli_measure_subcommand_smoke(tmp_path: Path, capsys):
    """Verify 'measure' subcommand loads scan and prints summary table."""
    scan_dir = tmp_path / "test_scan"
    res = ScanResult(
        subject_id="sub_test",
        measurements={
            "stature": MeasurementResult("stature", 1750.0, "vertical_height", 1.0, 0.0, QualityFlag.OK.value),
            "chest_circumference": MeasurementResult("chest_circumference", 920.0, "polar_contour", 0.9, 0.1, QualityFlag.OK.value),
        }
    )
    save_scan_result(res, scan_dir)

    exit_code = main(["measure", "--scan", str(scan_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "chest_circumference" in captured.out
    assert "1750.0" in captured.out


def test_cli_validate_subcommand_smoke(tmp_path: Path, capsys):
    """Verify 'validate' subcommand compares scan with tape data."""
    scan_dir = tmp_path / "test_scan"
    res = ScanResult(
        measurements={
            "stature": MeasurementResult("stature", 1755.0, "vertical_height", 1.0, 0.0, QualityFlag.OK.value),
        }
    )
    save_scan_result(res, scan_dir)

    tape_file = tmp_path / "tape.json"
    with open(tape_file, "w") as f:
        json.dump({"measurements_mm": {"stature": 1750.0}}, f)

    exit_code = main(["validate", "--scan", str(scan_dir), "--tape", str(tape_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "stature" in captured.out
    assert "Diff (mm)" in captured.out


def test_cli_module_execution():
    """Verify 'python -m realsense_measure.fullbody --help' executes cleanly."""
    res = subprocess.run(
        [sys.executable, "-m", "realsense_measure.fullbody", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Model-free metric full-body scanning" in res.stdout


@pytest.mark.skipif("FULLBODY_SAMPLE_IMAGE" not in os.environ, reason="FULLBODY_SAMPLE_IMAGE not set")
def test_cli_keypoints_check_with_sample_image(tmp_path: Path):
    """Verify 'keypoints-check' executes on real or provided sample image."""
    sample_img = os.environ["FULLBODY_SAMPLE_IMAGE"]
    out_overlay = tmp_path / "overlay.png"
    exit_code = main(["keypoints-check", sample_img, "--out", str(out_overlay)])
    assert exit_code == 0
    assert out_overlay.exists()
