"""
Tests for evaluate_head_scan_offline.py.

Verifies:
  1. Single-file evaluation on valid synthetic head point cloud.
  2. Directory and recursive file discovery.
  3. Handling of missing or non-existent PLY file.
  4. Handling of empty or non-finite point cloud.
  5. Per-file exception resilience (one corrupt/missing file does not abort batch).
  6. Output generation (JSON and CSV structure, unit labels in mm/m3).
  7. Explicit 'not_exposed' labeling of internal segmentation diagnostics.
  8. Handling of unverified measurement and null volume without crashing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from evaluate_head_scan_offline import (
    discover_ply_files,
    evaluate_single_ply,
    run_evaluation,
)
from test_head_synthetic import create_synthetic_head


@pytest.fixture
def temp_eval_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory with synthetic test PLY files."""
    test_dir = tmp_path / "offline_test_assets"
    test_dir.mkdir()

    # 1. Valid complete synthetic head
    pcd_head = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=3000)
    valid_path = test_dir / "valid_head.ply"
    o3d.io.write_point_cloud(str(valid_path), pcd_head)

    # 2. Subdirectory with recursive PLY
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    pcd_sub = create_synthetic_head(w=0.15, d=0.18, h=0.22, n_pts=1500)
    sub_path = sub_dir / "sub_head.ply"
    o3d.io.write_point_cloud(str(sub_path), pcd_sub)

    # 3. Empty point cloud PLY (valid PLY header with 0 vertices)
    empty_path = test_dir / "empty.ply"
    empty_content = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 0\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    empty_path.write_text(empty_content, encoding="utf-8")

    # 4. Corrupt non-PLY file with .ply extension
    corrupt_path = test_dir / "corrupt.ply"
    corrupt_path.write_text("This is not a valid ply header or data", encoding="utf-8")

    return test_dir


def test_single_file_evaluation(temp_eval_dir: Path):
    """Test evaluating a single valid synthetic head PLY file."""
    ply_path = temp_eval_dir / "valid_head.ply"
    rec = evaluate_single_ply(ply_path)

    assert rec["execution_error"] is None
    assert rec["input_path"] == str(ply_path)
    assert rec["input_filename"] == "valid_head.ply"

    inp = rec["input_and_preprocessing"]
    assert inp["raw_point_count"] == 3000
    assert inp["finite_point_count"] == 3000
    assert inp["dominant_cluster_point_count"] == "not_exposed"
    assert inp["max_span_mm"] > 150.0

    seg = rec["segmentation_diagnostics"]
    assert seg["output_point_count"] > 0
    assert 0.0 < seg["retention_ratio_raw"] <= 1.0
    assert seg["longitudinal_span_gate_passed"] == "not_exposed"
    assert seg["longitudinal_axis_u"] == "not_exposed"
    assert seg["cross_sectional_areas_evaluated"] == "not_exposed"

    down = rec["downstream_validation"]
    assert down["is_plausible"] is True
    assert down["measurement_status"] in ("validated", "unverified")
    dims = down["cranial_envelope_mm"]
    assert dims["breadth_mm"] > 0.0
    assert dims["length_mm"] > 0.0
    assert dims["height_mm"] > 0.0


def test_directory_and_recursive_discovery(temp_eval_dir: Path):
    """Test shallow and recursive PLY file discovery."""
    # Non-recursive should find files only in root of temp_eval_dir
    shallow = discover_ply_files(temp_eval_dir, recursive=False)
    shallow_names = [p.name for p in shallow]
    assert "valid_head.ply" in shallow_names
    assert "empty.ply" in shallow_names
    assert "sub_head.ply" not in shallow_names

    # Recursive should find files in subdirectories too
    recursive = discover_ply_files(temp_eval_dir, recursive=True)
    rec_names = [p.name for p in recursive]
    assert "valid_head.ply" in rec_names
    assert "sub_head.ply" in rec_names
    assert len(recursive) >= 4


def test_missing_or_unreadable_ply(tmp_path: Path):
    """Test handling of a non-existent PLY path."""
    missing_path = tmp_path / "does_not_exist.ply"
    rec = evaluate_single_ply(missing_path)

    assert rec["execution_error"] is not None
    assert "File not found" in rec["execution_error"]
    assert rec["input_and_preprocessing"] == {}


def test_corrupt_ply_handling(temp_eval_dir: Path):
    """Test handling of an unreadable/corrupt PLY file."""
    corrupt_path = temp_eval_dir / "corrupt.ply"
    rec = evaluate_single_ply(corrupt_path)

    # Must record error cleanly without raising exception
    assert rec["execution_error"] is not None
    assert "Corrupt or invalid PLY header" in rec["execution_error"]


def test_empty_point_cloud(temp_eval_dir: Path):
    """Test handling of an empty point cloud."""
    empty_path = temp_eval_dir / "empty.ply"
    rec = evaluate_single_ply(empty_path)

    assert rec["execution_error"] == "Point cloud contains zero finite coordinates"
    inp = rec["input_and_preprocessing"]
    assert inp["raw_point_count"] == 0
    assert inp["finite_point_count"] == 0
    seg = rec["segmentation_diagnostics"]
    assert seg["output_point_count"] == 0
    down = rec["downstream_validation"]
    assert down["is_plausible"] is False
    assert down["measurement_status"] == "unverified"
    assert down["validated_volume_m3"] is None


def test_batch_resilience_and_output_generation(temp_eval_dir: Path, tmp_path: Path):
    """
    Test that run_evaluation handles mixed valid, empty, corrupt, and missing files,
    does not abort, and properly generates both JSON and CSV files.
    """
    missing_file = temp_eval_dir / "missing.ply"
    test_inputs = [
        temp_eval_dir / "valid_head.ply",
        temp_eval_dir / "empty.ply",
        temp_eval_dir / "corrupt.ply",
        missing_file,
    ]

    out_dir = tmp_path / "eval_reports"
    summary = run_evaluation(
        inputs=test_inputs,
        output_dir=out_dir,
        output_name="test_summary",
        quiet=True,
    )

    # Check overall summary statistics
    assert summary["total_files_evaluated"] == 4
    assert summary["successful_evaluations"] == 1
    assert summary["failed_evaluations"] == 3
    assert len(summary["records"]) == 4

    # Verify JSON file written and valid
    json_path = out_dir / "test_summary.json"
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["evaluator_name"] == "evaluate_head_scan_offline"
    assert len(loaded_json["records"]) == 4

    # Verify CSV file written and valid
    csv_path = out_dir / "test_summary.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    assert len(rows) == 4

    # Check units and column presence
    row_valid = rows[0]
    assert row_valid["input_filename"] == "valid_head.ply"
    assert float(row_valid["raw_point_count"]) == 3000
    assert float(row_valid["breadth_mm"]) > 0.0
    assert float(row_valid["length_mm"]) > 0.0
    assert float(row_valid["height_mm"]) > 0.0
    assert row_valid["segmentation_diagnostics"] == "not_exposed"


def test_unverified_measurement_and_null_volume(tmp_path: Path):
    """
    Test that unverified measurements correctly record volume as None / empty
    and maintain diagnostics without raising an error.
    """
    # Create a partial head (truncated anterior-posterior) that fails face support
    pcd_partial = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=3000)
    pts = np.asarray(pcd_partial.points)
    # Strip posterior cranium (Z < 0)
    pcd_partial.points = o3d.utility.Vector3dVector(pts[pts[:, 2] > 0.05])
    partial_path = tmp_path / "partial_head.ply"
    o3d.io.write_point_cloud(str(partial_path), pcd_partial)

    rec = evaluate_single_ply(partial_path)
    assert rec["execution_error"] is None

    down = rec["downstream_validation"]
    assert down["measurement_status"] == "unverified"
    assert down["validated_volume_m3"] is None
    assert len(down["validation_diagnostics"]) > 0
