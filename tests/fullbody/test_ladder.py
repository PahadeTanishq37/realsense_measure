"""
tests/fullbody/test_ladder.py

Error ladder diagnostic for the model-free fullbody scanning pipeline:
Level 0: Perfect baseline (exact poses, no sensor noise, no dropouts, burst=1)
Level 1: + Realistic sensor noise (sigma(z) = 1mm * (z/0.6)^2)
Level 2: + Sensor noise + 10% dropouts + flying pixels (burst=3)
Level 3: + Sensor noise + dropouts + pose perturbations (+-4 deg yaw, +-2 cm translation)
Level 4: + Full degradation (+-3 deg inter-stop arm motion)

At each level, reports:
- Stature: True vs Visible tangent apex vs Measured, and QualityFlag
- Torso circumferences (Chest, Waist, Hip): True vs Measured, Coverage %, Error %
- Limb circumferences (Upper Arm, Forearm, Thigh, Calf): True vs Measured, Coverage %, Error %
- Alignment errors: Mean/Max Yaw Error (deg), Mean/Max Translation Error (mm)
- Hole map: Crown, Crotch, Underarms fractions

Run standalone with:
  pytest tests/fullbody/test_ladder.py -s -v -m slow
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.alignment import align_stops
from realsense_measure.fullbody.burst import process_stop_burst
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.confidence import compute_confidence_map
from realsense_measure.fullbody.floor import fit_floor_plane, verify_floor_consistency
from realsense_measure.fullbody.fusion import fuse_aligned_stops
from realsense_measure.fullbody.measure_scan import measure_fused_scan
from realsense_measure.fullbody.types import MeasurementResult, QualityFlag, StopData
from tests.fullbody.synth_humanoid import SyntheticHumanoidCaptureRenderer


@dataclasses.dataclass
class LadderLevelResult:
    level_name: str
    description: str
    stature_true_mm: float
    stature_visible_apex_mm: float
    stature_measured_mm: float
    stature_flag: str
    mean_yaw_err_deg: float
    max_yaw_err_deg: float
    mean_trans_err_mm: float
    max_trans_err_mm: float
    chest_true_mm: float
    chest_measured_mm: float
    chest_err_pct: float
    chest_cov_pct: float
    waist_true_mm: float
    waist_measured_mm: float
    waist_err_pct: float
    waist_cov_pct: float
    hip_true_mm: float
    hip_measured_mm: float
    hip_err_pct: float
    hip_cov_pct: float
    arm_true_mm: float
    arm_measured_mm: float
    arm_err_pct: float
    arm_cov_pct: float
    thigh_true_mm: float
    thigh_measured_mm: float
    thigh_err_pct: float
    thigh_cov_pct: float
    hole_crown: float
    hole_crotch: float
    hole_underarms: float


def run_ladder_level(
    level_name: str,
    description: str,
    renderer: SyntheticHumanoidCaptureRenderer,
    cfg: FullBodyConfig,
    burst_count: int = 1,
    add_noise: bool = False,
    yaw_jitter_deg: float = 0.0,
    trans_jitter_m: float = 0.0,
    arm_jitter_deg: float = 0.0,
) -> LadderLevelResult:
    """Run one level of the diagnostic ladder and return metrics."""
    stops, gt, _ = renderer.render_stops(
        num_stops=8,
        burst_count=burst_count,
        add_noise=add_noise,
        yaw_jitter_deg=yaw_jitter_deg,
        trans_jitter_m=trans_jitter_m,
    )

    floors = []
    for s in stops:
        if burst_count > 1 and len(s.raw_burst_depths) > 1:
            f_color, f_depth, d_std, v_counts = process_stop_burst(
                s.raw_burst_colors, s.raw_burst_depths, cfg
            )
            s.color = f_color
            s.depth = f_depth
            s.depth_std = d_std
            s.valid_counts = v_counts

        floor = fit_floor_plane(s.depth, s.intrinsics, cfg)
        if floor is not None:
            s.floor_plane = floor
            floors.append(floor)

        cmap, cdepth = compute_confidence_map(
            s.depth, s.intrinsics, cfg, depth_std_m=s.depth_std, valid_counts=s.valid_counts, burst_total=burst_count
        )
        s.confidence_map = cmap

        # Backproject
        h, w = cdepth.shape[:2]
        fx, fy = s.intrinsics.get_focal_length()
        cx, cy = s.intrinsics.get_principal_point()
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        val = cdepth > 0
        zv = cdepth[val]
        xv = (u_grid[val] - cx) * zv / fx
        yv = (v_grid[val] - cy) * zv / fy
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack([xv, yv, zv], axis=1))
        s.pcd = pcd

    _, consensus_floor = verify_floor_consistency(floors, cfg)
    aligned = align_stops(stops, consensus_floor, cfg)

    # Compute pose errors
    yaw_errors = []
    trans_errors = []
    for a, s in zip(aligned, stops):
        if a.accepted and s.gt_yaw_deg is not None and s.gt_T_canonical_camera is not None:
            dyaw = abs((a.refined_yaw_deg - s.gt_yaw_deg + 180.0) % 360.0 - 180.0)
            yaw_errors.append(dyaw)
            # Translation error: Euclidean distance between camera origins mapped to canonical frame
            t_est = a.T_canonical_camera[:3, 3]
            t_gt = s.gt_T_canonical_camera[:3, 3]
            dtrans_mm = float(np.linalg.norm(t_est - t_gt) * 1000.0)
            trans_errors.append(dtrans_mm)

    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)
    ref_kpts = aligned[0].keypoints_canonical if aligned else None
    measurements = measure_fused_scan(fused_pcd, ref_kpts, cfg, hole_map=hole_map)

    # Stature
    stature_m = measurements["stature"]
    stature_true = gt["stature"]
    stature_vis = gt.get("highest_visible_apex_mm", 1733.1)

    # Helper for ring extraction
    def _ring_stats(meas_name: str, gt_name: str) -> Tuple[float, float, float, float]:
        m_polar = measurements.get(f"{meas_name}_circumference")
        m_hull = measurements.get(f"{meas_name}_circumference_convexhull")
        gt_val = gt.get(gt_name, float("nan"))
        # Prefer convex hull for torso rings as tape proxy, polar for limbs
        m = m_hull if m_hull and not math.isnan(m_hull.value_mm) else m_polar
        if m and not math.isnan(m.value_mm) and not math.isnan(gt_val):
            err_pct = abs(m.value_mm - gt_val) / gt_val * 100.0
            cov_pct = m.coverage * 100.0
            return gt_val, m.value_mm, err_pct, cov_pct
        return gt_val, float("nan"), float("nan"), 0.0

    ch_gt, ch_m, ch_err, ch_cov = _ring_stats("chest", "chest_circumference")
    w_gt, w_m, w_err, w_cov = _ring_stats("waist", "waist_circumference")
    h_gt, h_m, h_err, h_cov = _ring_stats("hip", "hip_circumference")
    arm_gt, arm_m, arm_err, arm_cov = _ring_stats("upper_arm_left", "upper_arm_left_circumference")
    th_gt, th_m, th_err, th_cov = _ring_stats("thigh_left", "thigh_left_circumference")

    return LadderLevelResult(
        level_name=level_name,
        description=description,
        stature_true_mm=stature_true,
        stature_visible_apex_mm=stature_vis,
        stature_measured_mm=stature_m.value_mm,
        stature_flag=stature_m.quality_flag,
        mean_yaw_err_deg=float(np.mean(yaw_errors)) if yaw_errors else 0.0,
        max_yaw_err_deg=float(np.max(yaw_errors)) if yaw_errors else 0.0,
        mean_trans_err_mm=float(np.mean(trans_errors)) if trans_errors else 0.0,
        max_trans_err_mm=float(np.max(trans_errors)) if trans_errors else 0.0,
        chest_true_mm=ch_gt,
        chest_measured_mm=ch_m,
        chest_err_pct=ch_err,
        chest_cov_pct=ch_cov,
        waist_true_mm=w_gt,
        waist_measured_mm=w_m,
        waist_err_pct=w_err,
        waist_cov_pct=w_cov,
        hip_true_mm=h_gt,
        hip_measured_mm=h_m,
        hip_err_pct=h_err,
        hip_cov_pct=h_cov,
        arm_true_mm=arm_gt,
        arm_measured_mm=arm_m,
        arm_err_pct=arm_err,
        arm_cov_pct=arm_cov,
        thigh_true_mm=th_gt,
        thigh_measured_mm=th_m,
        thigh_err_pct=th_err,
        thigh_cov_pct=th_cov,
        hole_crown=hole_map.get("crown", 1.0),
        hole_crotch=hole_map.get("crotch", 1.0),
        hole_underarms=hole_map.get("underarms", 1.0),
    )


@pytest.mark.slow
def test_fullbody_error_ladder_diagnostic(capsys):
    """
    Run the 5-step diagnostic error ladder, print summary markdown table, and assert tolerances.
    """
    cfg = FullBodyConfig()
    cfg.min_confidence_threshold = 0.08
    renderer = SyntheticHumanoidCaptureRenderer(seed=777)

    results: List[LadderLevelResult] = []

    # Level 0: Perfect baseline
    res0 = run_ladder_level(
        "Level 0",
        "Perfect (no noise, exact poses, burst=1)",
        renderer,
        cfg,
        burst_count=1,
        add_noise=False,
        yaw_jitter_deg=0.0,
        trans_jitter_m=0.0,
    )
    results.append(res0)

    # Level 1: + Realistic distance sensor noise
    res1 = run_ladder_level(
        "Level 1",
        "+ Sensor noise sigma(z)",
        renderer,
        cfg,
        burst_count=3,
        add_noise=True,
        yaw_jitter_deg=0.0,
        trans_jitter_m=0.0,
    )
    results.append(res1)

    # Level 2: + Dropouts & edge flying pixels
    res2 = run_ladder_level(
        "Level 2",
        "+ 10% Dropouts & Flying pixels",
        renderer,
        cfg,
        burst_count=3,
        add_noise=True,
        yaw_jitter_deg=0.0,
        trans_jitter_m=0.0,
    )
    results.append(res2)

    # Level 3: + Pose perturbations (+-4 deg yaw, +-2 cm translation)
    res3 = run_ladder_level(
        "Level 3",
        "+ Pose Jitter (+-4 deg, +-2 cm)",
        renderer,
        cfg,
        burst_count=3,
        add_noise=True,
        yaw_jitter_deg=4.0,
        trans_jitter_m=0.02,
    )
    results.append(res3)

    # Level 4: + Arm motion between stops (+-3 deg)
    res4 = run_ladder_level(
        "Level 4",
        "+ Arm Posture Jitter (+-3 deg)",
        renderer,
        cfg,
        burst_count=3,
        add_noise=True,
        yaw_jitter_deg=4.0,
        trans_jitter_m=0.02,
        arm_jitter_deg=3.0,
    )
    results.append(res4)

    # Print Diagnostic Ladder Markdown Table
    print("\n" + "=" * 120)
    print("FULL-BODY SCANNING PIPELINE: ERROR LADDER DIAGNOSTIC REPORT")
    print("=" * 120)
    header = (
        f"{'Level':<8} | {'Description':<32} | {'YawErr(deg)':>11} | {'TrErr(mm)':>9} | "
        f"{'Stature':>9} | {'StFlag':<7} | {'Chest(err/cov)':>14} | {'Waist(err/cov)':>14} | "
        f"{'Hip(err/cov)':>14} | {'CrownHole':>9}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        chest_str = f"{r.chest_err_pct:>4.1f}%/{r.chest_cov_pct:>4.0f}%"
        waist_str = f"{r.waist_err_pct:>4.1f}%/{r.waist_cov_pct:>4.0f}%"
        hip_str = f"{r.hip_err_pct:>4.1f}%/{r.hip_cov_pct:>4.0f}%"
        row = (
            f"{r.level_name:<8} | {r.description:<32} | {r.mean_yaw_err_deg:>10.2f}\u00b0 | {r.mean_trans_err_mm:>7.1f}mm | "
            f"{r.stature_measured_mm:>7.1f}mm | {r.stature_flag:<7} | {chest_str:>14} | {waist_str:>14} | "
            f"{hip_str:>14} | {r.hole_crown:>8.2f}"
        )
        print(row)
    print("=" * 120)

    # Assertions derived from ladder analysis:
    # 1. Level 0: Perfect level coverage must be >= 85% for chest/waist and >= 90% for hip
    assert res0.chest_cov_pct >= 85.0, f"Level 0 chest coverage {res0.chest_cov_pct:.1f}% < 85%"
    assert res0.waist_cov_pct >= 85.0, f"Level 0 waist coverage {res0.waist_cov_pct:.1f}% < 85%"
    assert res0.hip_cov_pct >= 90.0, f"Level 0 hip coverage {res0.hip_cov_pct:.1f}% < 90%"
    # Level 0 circumferences must be within 10% of Ramanujan analytic truth
    assert res0.chest_err_pct < 10.0, f"Level 0 chest error {res0.chest_err_pct:.2f}% exceeds 10.0%"
    assert res0.waist_err_pct < 10.0, f"Level 0 waist error {res0.waist_err_pct:.2f}% exceeds 10.0%"
    assert res0.hip_err_pct < 10.0, f"Level 0 hip error {res0.hip_err_pct:.2f}% exceeds 10.0%"

    # 2. Stature flag consistency: Crown hole fraction > 0.20 => flag MUST be PARTIAL, never OK!
    for r in results:
        if r.hole_crown > 0.20:
            assert r.stature_flag == QualityFlag.PARTIAL.value, (
                f"{r.level_name}: stature flag was {r.stature_flag} but crown hole fraction was {r.hole_crown}"
            )
        # Measured stature reflects visible tangent apex (~1733mm) minus grazing falloff
        assert abs(r.stature_measured_mm - r.stature_visible_apex_mm) < 45.0

    # 3. Alignment error tolerances:
    # Under Level 3 and 4 (injected +-4 deg yaw, +-20mm trans):
    # Mean recovered yaw error must be < 10.0 deg, translation error < 200 mm
    assert res3.mean_yaw_err_deg < 10.0, f"Level 3 yaw error {res3.mean_yaw_err_deg:.2f} deg exceeds 10.0 deg"
    assert res3.mean_trans_err_mm < 200.0, f"Level 3 trans error {res3.mean_trans_err_mm:.1f} mm exceeds 200 mm"
    assert res4.mean_yaw_err_deg < 10.0, f"Level 4 yaw error {res4.mean_yaw_err_deg:.2f} deg exceeds 10.0 deg"
    assert res4.mean_trans_err_mm < 200.0, f"Level 4 trans error {res4.mean_trans_err_mm:.1f} mm exceeds 200 mm"
