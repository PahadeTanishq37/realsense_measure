"""
Pipeline-level Integration Tests for HeadTarget in realsense_measure.

Tests:
  1. Pipeline-level measurement on a complete synthetic head.
  2. Pipeline-level measurement on an incomplete / ambiguous synthetic head.
  3. Pipeline-level measurement on a synthetic box to confirm backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import open3d as o3d

from config import PipelineConfig, TargetConfig, CameraConfig
from pipeline import ScanPipeline
from test_head_synthetic import create_synthetic_head


def test_pipeline_complete_head() -> None:
    print("\n" + "=" * 80)
    print("  [TEST 1] Pipeline Head Measurement -- Complete Synthetic Head")
    print("=" * 80)

    cfg = PipelineConfig(
        target=TargetConfig(name="head"),
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=Path("test_output_head_complete"),
    )
    pipeline = ScanPipeline(cfg)

    # Generate synthetic head point cloud (160 x 200 x 240 mm)
    pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)

    result = pipeline.measure(pcd)

    assert result["target"] == "head", f"Expected target 'head', got {result['target']}"
    assert result["measurement_type"] == "3d_geometric_cranial_envelope"
    assert result["anatomical_landmarks"] is None
    assert result["status"] == "validated"
    obb = result["oriented_bbox"]
    assert obb["volume_m3"] is not None

    cranial = result["cranial_envelope_3d"]
    b_mm, l_mm, h_mm = cranial["breadth_mm"], cranial["length_mm"], cranial["height_mm"]
    print(f"  Result: validated, Breadth={b_mm:.1f} mm, Length={l_mm:.1f} mm, Height={h_mm:.1f} mm, Volume={obb['volume_m3']*1e6:.1f} cm^3")

    # Verify JSON serializability
    json_str = json.dumps(result, indent=2)
    assert len(json_str) > 0

    print("  -> TEST 1 PASSED [OK]")


def test_pipeline_ambiguous_head() -> None:
    print("\n" + "=" * 80)
    print("  [TEST 2] Pipeline Head Measurement -- Ambiguous / Incomplete Head")
    print("=" * 80)

    cfg = PipelineConfig(
        target=TargetConfig(name="head"),
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=Path("test_output_head_ambiguous"),
    )
    pipeline = ScanPipeline(cfg)

    # 1. Ambiguous sphere
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    sphere_pcd = sphere_mesh.sample_points_poisson_disk(number_of_points=4000)

    result_sphere = pipeline.measure(sphere_pcd)
    assert result_sphere["target"] == "head"
    assert result_sphere["status"] == "unverified"
    assert result_sphere["oriented_bbox"]["volume_m3"] is None
    print(f"  Ambiguous Sphere Result: status={result_sphere['status']}, volume_m3={result_sphere['oriented_bbox']['volume_m3']}")

    # 2. Incomplete head (missing posterior)
    head_pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)
    pts = np.asarray(head_pcd.points)
    pts_incomplete = pts[pts[:, 1] > -0.02]  # remove back of head
    inc_pcd = o3d.geometry.PointCloud()
    inc_pcd.points = o3d.utility.Vector3dVector(pts_incomplete)

    result_inc = pipeline.measure(inc_pcd)
    assert result_inc["target"] == "head"
    assert result_inc["status"] == "unverified"
    assert result_inc["oriented_bbox"]["volume_m3"] is None
    print(f"  Incomplete Head Result: status={result_inc['status']}, volume_m3={result_inc['oriented_bbox']['volume_m3']}")

    print("  -> TEST 2 PASSED [OK]")


def test_pipeline_box_compatibility() -> None:
    print("\n" + "=" * 80)
    print("  [TEST 3] Pipeline Box Measurement -- Backward Compatibility")
    print("=" * 80)

    cfg = PipelineConfig(
        target=TargetConfig(name="box"),
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=Path("test_output_box_compat"),
    )
    pipeline = ScanPipeline(cfg)

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.30, height=0.20, depth=0.12)
    mesh.translate((-0.15, -0.10, -0.06))
    pcd_box = mesh.sample_points_poisson_disk(number_of_points=6000)

    result = pipeline.measure(pcd_box)

    assert result["target"] == "box", f"Expected target 'box', got {result['target']}"
    assert result["measurement_method"] == "3d_point_to_point_distance"
    assert result["status"] == "validated"
    obb = result["oriented_bbox"]
    assert obb["volume_m3"] is not None
    assert "planarity_check" in result

    l_mm, w_mm, h_mm = obb["length_m"] * 1000.0, obb["width_m"] * 1000.0, obb["height_m"] * 1000.0
    print(f"  Result: validated, Length={l_mm:.1f} mm, Width={w_mm:.1f} mm, Height={h_mm:.1f} mm, Volume={obb['volume_m3']*1e6:.1f} cm^3")

    print("  -> TEST 3 PASSED [OK]")


if __name__ == "__main__":
    test_pipeline_complete_head()
    test_pipeline_ambiguous_head()
    test_pipeline_box_compatibility()
    print("\n" + "=" * 80)
    print("  ALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)
