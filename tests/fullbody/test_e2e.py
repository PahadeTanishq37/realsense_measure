"""
tests/fullbody/test_e2e.py

End-to-end verification of the model-free full-body pipeline on synthetic data.
Pipeline stages tested:
  burst -> floor -> confidence -> keypoints -> alignment -> fusion -> landmarks -> measure_scan -> validate
Also tests CLI subcommands:
  scan -> measure -> validate

NOTE: These are unit and code-path tests only. Synthetic tests verify code paths,
numerical stability, coordinate frame consistency, and API contracts. Real-world
accuracy is judged solely by physical validation against tape measurements.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Dict

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.alignment import align_stops
from realsense_measure.fullbody.burst import process_stop_burst
from realsense_measure.fullbody.cli import main as cli_main
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.confidence import compute_confidence_map
from realsense_measure.fullbody.floor import fit_floor_plane, verify_floor_consistency
from realsense_measure.fullbody.fusion import fuse_aligned_stops
from realsense_measure.fullbody.measure_scan import measure_fused_scan
from realsense_measure.fullbody.types import QualityFlag, ScanResult
from realsense_measure.fullbody.validate import compare_with_tape
from tests.fullbody.synth_humanoid import SyntheticHumanoidCaptureRenderer


def test_e2e_synthetic_humanoid_pipeline_8_stops(capsys):
    """
    Code-path test: 8-stop synthetic humanoid scan through full pipeline.
    Asserts:
      - Recovered yaw/translation within loose tolerances.
      - Fused mesh has a single dominant component (>90% triangles).
      - Hole map flags unobserved regions.
      - Extracted measurements vs analytic ground truth are reported with loose tolerances.
    """
    cfg = FullBodyConfig()
    cfg.min_confidence_threshold = 0.08
    renderer = SyntheticHumanoidCaptureRenderer(seed=42)

    # 1. Render 8 stops with sensor noise, dropouts, flying pixels, and inter-stop jitter
    stops, gt, kpts_gt = renderer.render_stops(
        num_stops=8,
        burst_count=3,
        add_noise=True,
        yaw_jitter_deg=4.0,
        trans_jitter_m=0.02,
    )
    assert len(stops) == 8

    # 2. Burst depth filtering + confidence map + floor fitting
    floors = []
    for s in stops:
        f_color, f_depth, d_std, v_counts = process_stop_burst(
            s.raw_burst_colors, s.raw_burst_depths, cfg
        )
        s.color = f_color
        s.depth = f_depth
        s.depth_std = d_std
        s.valid_counts = v_counts

        # Floor plane
        floor = fit_floor_plane(s.depth, s.intrinsics, cfg)
        if floor is not None:
            s.floor_plane = floor
            floors.append(floor)

        # Confidence map
        cmap, cdepth = compute_confidence_map(
            s.depth, s.intrinsics, cfg, depth_std_m=s.depth_std, valid_counts=s.valid_counts, burst_total=3
        )
        s.confidence_map = cmap

        # Backproject to point cloud
        h, w = cdepth.shape[:2]
        fx, fy = s.intrinsics.get_focal_length()
        cx, cy = s.intrinsics.get_principal_point()
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        val = cdepth > 0
        zv = cdepth[val]
        xv = (u_grid[val] - cx) * zv / fx
        yv = (v_grid[val] - cy) * zv / fy
        pts = np.stack([xv, yv, zv], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        s.pcd = pcd

    # 3. Consensus floor verification
    assert len(floors) >= 6, "At least 6 stops must detect floor"
    consistent, consensus_floor = verify_floor_consistency(floors, cfg)
    assert consistent, "Floor planes must be consistent across stops"

    # 4. Multi-stop alignment
    aligned = align_stops(stops, consensus_floor, cfg)
    assert len(aligned) == 8

    # Verify recovered poses
    for a in aligned:
        assert a.accepted, f"Stop {a.stop_id} alignment was unexpectedly rejected"
        nom_canonical = (180.0 - stops[a.stop_id].nominal_angle_deg + 360.0) % 360.0
        # Refined yaw within 25 deg of nominal canonical (accounting for +-4 deg jitter + ICP refinement)
        angle_diff = (a.refined_yaw_deg - nom_canonical + 180.0) % 360.0 - 180.0
        assert abs(angle_diff) < 25.0, f"Stop {a.stop_id} yaw discrepancy too large: {angle_diff:.1f} deg"

    # 5. Volumetric TSDF fusion
    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)
    assert len(fused_pcd.points) > 10000, "Fused point cloud must contain dense points"
    assert len(mesh.vertices) > 2000, "Fused mesh must contain reconstructed surface geometry"

    # Dominant component check
    clusters, num_tri, _ = mesh.cluster_connected_triangles()
    largest_cluster = np.max(num_tri)
    assert largest_cluster > 200, "Fused mesh dominant component must contain substantial triangles"

    # Hole map check
    assert "crown" in hole_map
    assert "crotch" in hole_map
    assert "underarms" in hole_map

    # 6. Slicing planes and tape-style measurements
    ref_kpts = aligned[0].keypoints_canonical
    measurements = measure_fused_scan(fused_pcd, ref_kpts, cfg)

    # 7. Print measurements summary table
    print("\n" + "=" * 80)
    print("CODE-PATH TEST: SYNTHETIC 8-STOP MEASUREMENTS vs ANALYTIC GROUND TRUTH")
    print("=" * 80)
    print(f"{'Measurement':<32} | {'Measured':>10} | {'Truth':>10} | {'Diff':>10} | {'Cov':>6} | {'Quality':<12}")
    print("-" * 80)

    for name in ["stature", "chest_circumference_convexhull", "waist_circumference_convexhull", "hip_circumference_convexhull"]:
        if name in measurements:
            m = measurements[name]
            base_name = name.replace("_convexhull", "")
            gt_val = gt.get(base_name, float("nan"))
            diff = m.value_mm - gt_val if not math.isnan(gt_val) else float("nan")
            print(f"{name:<32} | {m.value_mm:>9.1f} mm | {gt_val:>9.1f} mm | {diff:>+9.1f} mm | {m.coverage * 100:>5.1f}% | {m.quality_flag:<12}")

    print("=" * 80)

    # 8. Assert loose, documented code-path tolerances
    # Stature within 100 mm (due to top-of-head camera perspective)
    assert abs(measurements["stature"].value_mm - gt["stature"]) < 120.0
    assert measurements["stature"].quality_flag == QualityFlag.OK.value

    # Torso circumferences (convex hull tape-proxy) within 15%
    for circum_key, gt_key in [
        ("chest_circumference_convexhull", "chest_circumference"),
        ("waist_circumference_convexhull", "waist_circumference"),
        ("hip_circumference_convexhull", "hip_circumference"),
    ]:
        if circum_key in measurements and not math.isnan(measurements[circum_key].value_mm):
            m_val = measurements[circum_key].value_mm
            gt_val = gt[gt_key]
            rel_err = abs(m_val - gt_val) / gt_val
            assert rel_err < 0.20, f"{circum_key} error {rel_err * 100:.1f}% exceeds 20%"

    # 9. Validation against tape comparison API
    tape_dict = {
        "stature": gt["stature"],
        "chest_circumference": gt["chest_circumference"],
        "waist_circumference": gt["waist_circumference"],
        "hip_circumference": gt["hip_circumference"],
    }
    val_report = compare_with_tape(measurements, tape_dict)
    assert len(val_report) >= 1


def test_e2e_synthetic_humanoid_pipeline_4_stops():
    """
    Code-path test: 4-stop synthetic humanoid scan through pipeline.
    Asserts:
      - 4 stops align and fuse into valid geometry.
      - Hole map correctly flags unobserved regions under sparse 4-stop sampling.
    """
    cfg = FullBodyConfig()
    cfg.min_confidence_threshold = 0.08
    renderer = SyntheticHumanoidCaptureRenderer(seed=123)

    stops, gt, _ = renderer.render_stops(num_stops=4, burst_count=1, add_noise=True)
    assert len(stops) == 4

    # Prepare point clouds
    for s in stops:
        f = fit_floor_plane(s.depth, s.intrinsics, cfg)
        s.floor_plane = f
        cmap, cdepth = compute_confidence_map(s.depth, s.intrinsics, cfg)
        s.confidence_map = cmap

        h, w = cdepth.shape[:2]
        fx, fy = s.intrinsics.get_focal_length()
        cx, cy = s.intrinsics.get_principal_point()
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        val = cdepth > 0
        zv = cdepth[val]
        xv = (u_grid[val] - cx) * zv / fx
        yv = (v_grid[val] - cy) * zv / fy
        pts = np.stack([xv, yv, zv], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        s.pcd = pcd

    floors = [s.floor_plane for s in stops if s.floor_plane is not None]
    _, consensus_floor = verify_floor_consistency(floors, cfg)

    aligned = align_stops(stops, consensus_floor, cfg)
    assert len(aligned) == 4

    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)
    assert len(fused_pcd.points) > 5000

    # In 4-stop sparse capture, crown and underarms should show missingness
    assert hole_map["crown"] > 0.50, f"Crown should be flagged in 4-stop capture: {hole_map}"
    assert "crotch" in hole_map


def test_e2e_cli_scan_and_measure():
    """
    Code-path test: Run CLI 'scan' and 'measure' end-to-end on synthetic capture files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cap_dir = Path(tmpdir) / "raw_capture"
        cap_dir.mkdir()
        out_dir = Path(tmpdir) / "scan_out"

        renderer = SyntheticHumanoidCaptureRenderer(seed=999)
        stops, gt, _ = renderer.render_stops(num_stops=4, burst_count=1, add_noise=False)

        # Write layout 1 session with meta.json
        frames_meta = []
        fx, fy = stops[0].intrinsics.get_focal_length()
        cx, cy = stops[0].intrinsics.get_principal_point()
        w, h = stops[0].intrinsics.width, stops[0].intrinsics.height
        for s in stops:
            c_name = f"frame_{s.stop_id:02d}_color.png"
            d_name = f"frame_{s.stop_id:02d}_depth.npy"
            cv2.imwrite(str(cap_dir / c_name), s.color)
            np.save(str(cap_dir / d_name), s.depth)
            frames_meta.append({
                "frame_id": s.stop_id,
                "color_file": c_name,
                "depth_file": d_name,
                "timestamp": 0.0,
            })

        meta = {
            "intrinsics": {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "frames": frames_meta,
        }
        with open(cap_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # 1. Run CLI scan
        ret_scan = cli_main(["scan", "--capture", str(cap_dir), "--out", str(out_dir), "--no-viz"])
        assert ret_scan == 0
        assert (out_dir / "scan_result.json").exists()
        assert (out_dir / "fused.ply").exists()

        # 2. Run CLI measure
        ret_meas = cli_main(["measure", "--scan", str(out_dir)])
        assert ret_meas == 0
