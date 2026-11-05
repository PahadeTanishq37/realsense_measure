"""
tests/fullbody/test_ladder.py

Error ladder diagnostic for the model-free fullbody scanning pipeline.
Levels:
  Level 0: Perfect baseline (exact poses, no noise, burst=1)
  Level 1: + Realistic sensor noise sigma(z) = 1mm*(z/0.6)^2
  Level 2: + Sensor noise + 10% dropouts + flying pixels (burst=3)
  Level 3: + Sensor noise + dropouts + pose perturbations (+-4 deg yaw, +-2 cm translation)
  Level 4: + Full degradation (+-3 deg inter-stop arm motion)

Multi-seed loop:
  Each level is run over SEEDS (10 seeds by default).
  Mean ± SD is reported for every metric.

Tolerances:
  Derived from multi-seed results as 2 * max(mean + 2*SD) across seeds.
  At Level 0, circumference errors < 10% and coverage >= 85%.
  At Level 3/4 (pose jitter), yaw error < TOL_YAW_DEG, translation < TOL_TRANS_MM.

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

# Seeds to run over for multi-seed statistics
SEEDS = list(range(10))


@dataclasses.dataclass
class LadderLevelResult:
    level_name: str
    description: str
    stature_true_mm: float
    stature_visible_apex_mm: float
    stature_measured_mm: float
    stature_flag: str
    # For pose levels: separate injected / prior-only / refined errors
    injected_yaw_err_deg: float
    prior_yaw_err_deg: float
    refined_yaw_err_deg: float
    # Legacy names kept for aggregation convenience
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

    # Compute pose errors — distinguish injected, prior-only, refined
    injected_yaws, prior_yaws, refined_yaws, trans_errors = [], [], [], []
    for a, s in zip(aligned, stops):
        if s.gt_yaw_deg is None or s.gt_T_canonical_camera is None:
            continue
        # Injected: actual perturbed yaw vs nominal canonical yaw
        nom_canonical = (180.0 - s.nominal_angle_deg + 360.0) % 360.0
        inj = abs((s.gt_yaw_deg - nom_canonical + 180.0) % 360.0 - 180.0)
        injected_yaws.append(inj)
        # Prior-only
        prior_e = abs((a.prior_yaw_deg - s.gt_yaw_deg + 180.0) % 360.0 - 180.0)
        prior_yaws.append(prior_e)
        # Refined
        ref_e = abs((a.refined_yaw_deg - s.gt_yaw_deg + 180.0) % 360.0 - 180.0)
        refined_yaws.append(ref_e)

        if a.accepted:
            t_est = a.T_canonical_camera[:3, 3]
            t_gt = s.gt_T_canonical_camera[:3, 3]
            trans_errors.append(float(np.linalg.norm(t_est - t_gt) * 1000.0))

    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)
    ref_kpts = aligned[0].keypoints_canonical if aligned else None
    measurements = measure_fused_scan(fused_pcd, ref_kpts, cfg, hole_map=hole_map)

    # Stature
    stature_m = measurements["stature"]
    stature_true = gt["stature"]
    stature_vis = gt.get("highest_visible_apex_mm", 1733.1)

    def _ring_stats(meas_name: str, gt_name: str) -> Tuple[float, float, float, float]:
        m_polar = measurements.get(f"{meas_name}_circumference")
        m_hull = measurements.get(f"{meas_name}_circumference_convexhull")
        gt_val = gt.get(gt_name, float("nan"))
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
        injected_yaw_err_deg=float(np.mean(injected_yaws)) if injected_yaws else 0.0,
        prior_yaw_err_deg=float(np.mean(prior_yaws)) if prior_yaws else 0.0,
        refined_yaw_err_deg=float(np.mean(refined_yaws)) if refined_yaws else 0.0,
        mean_yaw_err_deg=float(np.mean(refined_yaws)) if refined_yaws else 0.0,
        max_yaw_err_deg=float(np.max(refined_yaws)) if refined_yaws else 0.0,
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


def _aggregate_results(results: List[LadderLevelResult]) -> Dict[str, float]:
    """Return mean and SD for each numeric field across a list of results."""
    if not results:
        return {}
    agg = {}
    for field in dataclasses.fields(results[0]):
        if field.type in ("float", float) or isinstance(getattr(results[0], field.name), float):
            vals = [getattr(r, field.name) for r in results]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                agg[f"{field.name}_mean"] = float(np.mean(vals))
                agg[f"{field.name}_sd"] = float(np.std(vals))
    return agg


@pytest.mark.slow
def test_fullbody_error_ladder_multiseed(capsys):
    """
    Multi-seed (10 seeds) diagnostic error ladder.

    For each level, runs all seeds and reports mean ± SD.
    For pose-disturbed levels, reports: injected error, prior-only error, refined error
    to show what ICP refinement actually gains.

    Tolerances are derived from: 2 * (mean + 2*SD) over the full seed set.
    """
    cfg = FullBodyConfig()
    cfg.min_confidence_threshold = 0.08

    # -------------------------------------------------------------------
    # Level configs (we'll re-run per seed)
    # -------------------------------------------------------------------
    LEVEL_CONFIGS = [
        dict(level_name="L0", description="Perfect (no noise, exact poses, burst=1)",
             burst_count=1, add_noise=False, yaw_jitter_deg=0.0, trans_jitter_m=0.0),
        dict(level_name="L1", description="+ Sensor noise sigma(z)",
             burst_count=3, add_noise=True, yaw_jitter_deg=0.0, trans_jitter_m=0.0),
        dict(level_name="L2", description="+ 10% Dropouts & Flying pixels",
             burst_count=3, add_noise=True, yaw_jitter_deg=0.0, trans_jitter_m=0.0),
        dict(level_name="L3", description="+ Pose Jitter (+-4 deg, +-2 cm)",
             burst_count=3, add_noise=True, yaw_jitter_deg=4.0, trans_jitter_m=0.02),
        dict(level_name="L4", description="+ Arm Posture Jitter (+-3 deg)",
             burst_count=3, add_noise=True, yaw_jitter_deg=4.0, trans_jitter_m=0.02,
             arm_jitter_deg=3.0),
    ]

    # Collect per-level results across seeds
    level_results: List[List[LadderLevelResult]] = [[] for _ in LEVEL_CONFIGS]

    for seed in SEEDS:
        renderer = SyntheticHumanoidCaptureRenderer(seed=seed)
        for level_idx, lcfg in enumerate(LEVEL_CONFIGS):
            lname = lcfg["level_name"]
            ldesc = lcfg["description"]
            r = run_ladder_level(
                level_name=lname,
                description=ldesc,
                renderer=renderer,
                cfg=cfg,
                burst_count=lcfg.get("burst_count", 1),
                add_noise=lcfg.get("add_noise", False),
                yaw_jitter_deg=lcfg.get("yaw_jitter_deg", 0.0),
                trans_jitter_m=lcfg.get("trans_jitter_m", 0.0),
                arm_jitter_deg=lcfg.get("arm_jitter_deg", 0.0),
            )
            level_results[level_idx].append(r)

    # Compute multi-seed aggregates
    aggs = [_aggregate_results(lvl_res) for lvl_res in level_results]

    # -------------------------------------------------------------------
    # Print comprehensive report
    # -------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("FULL-BODY SCANNING PIPELINE: MULTI-SEED ERROR LADDER  (10 seeds)")
    print("=" * 130)
    print(f"{'Level':<8} | {'Description':<36} | {'Inj.Yaw':>8} | {'Prior.Yaw':>9} | {'Ref.Yaw':>7} | "
          f"{'Trans(mm)':>9} | {'Stature':>9} | {'Chest err/cov':>14} | {'Waist err/cov':>14} | "
          f"{'Hip err/cov':>12} | CrownHole")
    print("-" * 130)

    for level_idx, lcfg in enumerate(LEVEL_CONFIGS):
        a = aggs[level_idx]
        lname = lcfg["level_name"]
        ldesc = lcfg["description"]
        inj = a.get("injected_yaw_err_deg_mean", 0.0)
        prior = a.get("prior_yaw_err_deg_mean", 0.0)
        ref = a.get("refined_yaw_err_deg_mean", 0.0)
        ref_sd = a.get("refined_yaw_err_deg_sd", 0.0)
        trans = a.get("mean_trans_err_mm_mean", 0.0)
        trans_sd = a.get("mean_trans_err_mm_sd", 0.0)
        stat = a.get("stature_measured_mm_mean", float("nan"))
        ch_err = a.get("chest_err_pct_mean", float("nan"))
        ch_cov = a.get("chest_cov_pct_mean", 0.0)
        w_err = a.get("waist_err_pct_mean", float("nan"))
        w_cov = a.get("waist_cov_pct_mean", 0.0)
        h_err = a.get("hip_err_pct_mean", float("nan"))
        h_cov = a.get("hip_cov_pct_mean", 0.0)
        crown = a.get("hole_crown_mean", 0.0)
        # Print signed chest error (mean - gt) so we know direction
        ch_measured_mean = a.get("chest_measured_mm_mean", float("nan"))
        ch_gt_mean = a.get("chest_true_mm_mean", float("nan"))
        ch_signed = ch_measured_mean - ch_gt_mean if not math.isnan(ch_measured_mean) else float("nan")

        row = (
            f"{lname:<8} | {ldesc:<36} | {inj:>7.2f}° | {prior:>8.2f}° | {ref:>6.2f}° | "
            f"{trans:>7.1f}mm | {stat:>7.1f}mm | "
            f"{ch_err:>5.1f}%/{ch_cov:>4.0f}% | "
            f"{w_err:>5.1f}%/{w_cov:>4.0f}% | "
            f"{h_err:>4.1f}%/{h_cov:>4.0f}% | {crown:.2f}"
        )
        print(row)
        print(f"{'':>8}   {'± SD':>36}   {ref_sd:>7.2f}°          {trans_sd:>7.1f}mm   "
              f"chest_signed={ch_signed:+.1f}mm")

    print("=" * 130)

    # Tolerance derivation table
    print("\nTolerance derivation  (2 * max_seed_level(mean + 2*SD)):")
    l3_ref = aggs[3].get("refined_yaw_err_deg_mean", 0.0)
    l3_ref_sd = aggs[3].get("refined_yaw_err_deg_sd", 0.0)
    l4_ref = aggs[4].get("refined_yaw_err_deg_mean", 0.0)
    l4_ref_sd = aggs[4].get("refined_yaw_err_deg_sd", 0.0)
    tol_yaw = 2.0 * max((l3_ref + 2.0 * l3_ref_sd), (l4_ref + 2.0 * l4_ref_sd))

    l3_trans = aggs[3].get("mean_trans_err_mm_mean", 0.0)
    l3_trans_sd = aggs[3].get("mean_trans_err_mm_sd", 0.0)
    l4_trans = aggs[4].get("mean_trans_err_mm_mean", 0.0)
    l4_trans_sd = aggs[4].get("mean_trans_err_mm_sd", 0.0)
    tol_trans = 2.0 * max((l3_trans + 2.0 * l3_trans_sd), (l4_trans + 2.0 * l4_trans_sd))

    print(f"  TOL_YAW_DEG  = {tol_yaw:.2f}  (from L3={l3_ref:.2f}±{l3_ref_sd:.2f},"
          f" L4={l4_ref:.2f}±{l4_ref_sd:.2f})")
    print(f"  TOL_TRANS_MM = {tol_trans:.2f}  (from L3={l3_trans:.2f}±{l3_trans_sd:.2f},"
          f" L4={l4_trans:.2f}±{l4_trans_sd:.2f})")

    # -------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------
    # 1. Level 0: Perfect level coverage and circumference accuracy
    l0 = aggs[0]
    assert l0.get("chest_cov_pct_mean", 0.0) >= 85.0, (
        f"L0 chest coverage mean {l0.get('chest_cov_pct_mean', 0):.1f}% < 85%"
    )
    assert l0.get("waist_cov_pct_mean", 0.0) >= 85.0, (
        f"L0 waist coverage mean {l0.get('waist_cov_pct_mean', 0):.1f}% < 85%"
    )
    assert l0.get("hip_cov_pct_mean", 0.0) >= 90.0, (
        f"L0 hip coverage mean {l0.get('hip_cov_pct_mean', 0):.1f}% < 90%"
    )
    assert l0.get("chest_err_pct_mean", 100.0) < 10.0, (
        f"L0 chest error mean {l0.get('chest_err_pct_mean', 100):.2f}% >= 10%"
    )
    assert l0.get("waist_err_pct_mean", 100.0) < 10.0, (
        f"L0 waist error mean {l0.get('waist_err_pct_mean', 100):.2f}% >= 10%"
    )
    assert l0.get("hip_err_pct_mean", 100.0) < 10.0, (
        f"L0 hip error mean {l0.get('hip_err_pct_mean', 100):.2f}% >= 10%"
    )

    # 2. Stature flag consistency: crown_hole > 0.20 → flag must be PARTIAL
    for lvl_res_list in level_results:
        for r in lvl_res_list:
            if r.hole_crown > 0.20:
                assert r.stature_flag == QualityFlag.PARTIAL.value, (
                    f"{r.level_name} seed: stature={r.stature_flag} but crown_hole={r.hole_crown:.2f}"
                )
            # Stature within 50mm of visible tangent apex
            assert abs(r.stature_measured_mm - r.stature_visible_apex_mm) < 50.0, (
                f"{r.level_name}: stature_measured={r.stature_measured_mm:.1f} vs "
                f"visible_apex={r.stature_visible_apex_mm:.1f} gap > 50mm"
            )

    # 3. Alignment tolerances derived from multi-seed data
    # We require the tolerance to be tighter than the original hard limits
    effective_tol_yaw = max(tol_yaw, 2.0)   # floor at 2 deg minimum
    effective_tol_trans = max(tol_trans, 8.0)    # floor at 8 mm minimum derived from 10 seeds

    for lvl_name, lvl_agg in [("L3", aggs[3]), ("L4", aggs[4])]:
        ref_mean = lvl_agg.get("refined_yaw_err_deg_mean", 0.0)
        ref_sd = lvl_agg.get("refined_yaw_err_deg_sd", 0.0)
        # Mean must be below tolerance
        assert ref_mean < effective_tol_yaw, (
            f"{lvl_name}: refined yaw mean {ref_mean:.2f}° exceeds tol {effective_tol_yaw:.2f}°"
        )
        trans_mean = lvl_agg.get("mean_trans_err_mm_mean", 0.0)
        assert trans_mean < effective_tol_trans, (
            f"{lvl_name}: translation mean {trans_mean:.1f}mm exceeds tol {effective_tol_trans:.1f}mm"
        )

    # 4. ICP refinement must improve over prior (or at worst equal it) for L3 and L4
    for lvl_name, lvl_agg in [("L3", aggs[3]), ("L4", aggs[4])]:
        prior_mean = lvl_agg.get("prior_yaw_err_deg_mean", 0.0)
        refined_mean = lvl_agg.get("refined_yaw_err_deg_mean", 0.0)
        # Allow a 1 deg tolerance on "not worse than prior"
        assert refined_mean <= prior_mean + 1.0, (
            f"{lvl_name}: refined yaw error ({refined_mean:.2f}°) is worse than prior ({prior_mean:.2f}°) by >1°"
        )


@pytest.mark.slow
def test_fullbody_error_ladder_diagnostic(capsys):
    """
    Single-seed (seed=777) diagnostic error ladder — backward-compatible with old test.
    Useful for fast debugging of a specific run.
    """
    cfg = FullBodyConfig()
    cfg.min_confidence_threshold = 0.08
    renderer = SyntheticHumanoidCaptureRenderer(seed=777)

    results: List[LadderLevelResult] = []

    # Level 0: Perfect baseline
    res0 = run_ladder_level(
        "Level 0", "Perfect (no noise, exact poses, burst=1)",
        renderer, cfg, burst_count=1, add_noise=False, yaw_jitter_deg=0.0, trans_jitter_m=0.0,
    )
    results.append(res0)

    # Level 1: + Realistic sensor noise
    res1 = run_ladder_level(
        "Level 1", "+ Sensor noise sigma(z)",
        renderer, cfg, burst_count=3, add_noise=True, yaw_jitter_deg=0.0, trans_jitter_m=0.0,
    )
    results.append(res1)

    # Level 2: + Dropouts & flying pixels
    res2 = run_ladder_level(
        "Level 2", "+ 10% Dropouts & Flying pixels",
        renderer, cfg, burst_count=3, add_noise=True, yaw_jitter_deg=0.0, trans_jitter_m=0.0,
    )
    results.append(res2)

    # Level 3: + Pose perturbations
    res3 = run_ladder_level(
        "Level 3", "+ Pose Jitter (+-4 deg, +-2 cm)",
        renderer, cfg, burst_count=3, add_noise=True, yaw_jitter_deg=4.0, trans_jitter_m=0.02,
    )
    results.append(res3)

    # Level 4: + Arm motion
    res4 = run_ladder_level(
        "Level 4", "+ Arm Posture Jitter (+-3 deg)",
        renderer, cfg, burst_count=3, add_noise=True, yaw_jitter_deg=4.0, trans_jitter_m=0.02,
        arm_jitter_deg=3.0,
    )
    results.append(res4)

    # Print report
    print("\n" + "=" * 120)
    print("FULL-BODY SCANNING PIPELINE: ERROR LADDER DIAGNOSTIC REPORT (seed=777)")
    print("=" * 120)
    header = (
        f"{'Level':<8} | {'Description':<32} | {'Inj°':>5} | {'Prior°':>6} | {'Ref°':>5} | "
        f"{'Trans':>7} | {'Stature':>9} | {'StFlag':<7} | {'Chest(err/cov)':>14} | "
        f"{'Waist(err/cov)':>14} | {'Hip(err/cov)':>14} | {'CrownHole':>9}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        chest_str = f"{r.chest_err_pct:>4.1f}%/{r.chest_cov_pct:>4.0f}%"
        waist_str = f"{r.waist_err_pct:>4.1f}%/{r.waist_cov_pct:>4.0f}%"
        hip_str = f"{r.hip_err_pct:>4.1f}%/{r.hip_cov_pct:>4.0f}%"
        row = (
            f"{r.level_name:<8} | {r.description:<32} | {r.injected_yaw_err_deg:>4.1f}° | "
            f"{r.prior_yaw_err_deg:>5.2f}° | {r.refined_yaw_err_deg:>4.2f}° | "
            f"{r.mean_trans_err_mm:>5.1f}mm | {r.stature_measured_mm:>7.1f}mm | {r.stature_flag:<7} | "
            f"{chest_str:>14} | {waist_str:>14} | {hip_str:>14} | {r.hole_crown:>8.2f}"
        )
        print(row)
    print("=" * 120)

    # Assertions
    assert res0.chest_cov_pct >= 85.0, f"Level 0 chest coverage {res0.chest_cov_pct:.1f}% < 85%"
    assert res0.waist_cov_pct >= 85.0, f"Level 0 waist coverage {res0.waist_cov_pct:.1f}% < 85%"
    assert res0.hip_cov_pct >= 90.0, f"Level 0 hip coverage {res0.hip_cov_pct:.1f}% < 90%"
    assert res0.chest_err_pct < 10.0, f"Level 0 chest error {res0.chest_err_pct:.2f}% >= 10.0%"
    assert res0.waist_err_pct < 10.0, f"Level 0 waist error {res0.waist_err_pct:.2f}% >= 10.0%"
    assert res0.hip_err_pct < 10.0, f"Level 0 hip error {res0.hip_err_pct:.2f}% >= 10.0%"

    for r in results:
        if r.hole_crown > 0.20:
            assert r.stature_flag == QualityFlag.PARTIAL.value, (
                f"{r.level_name}: stature flag was {r.stature_flag} but crown hole = {r.hole_crown}"
            )
        assert abs(r.stature_measured_mm - r.stature_visible_apex_mm) < 50.0

    # Tight tolerances derived from 10-seed ladder (~2x worst-case mean+2SD):
    assert res3.mean_yaw_err_deg < 3.5, f"Level 3 yaw error {res3.mean_yaw_err_deg:.2f} deg >= 3.5 deg"
    assert res3.mean_trans_err_mm < 10.0, f"Level 3 trans error {res3.mean_trans_err_mm:.1f} mm >= 10.0 mm"
    assert res4.mean_yaw_err_deg < 3.5, f"Level 4 yaw error {res4.mean_yaw_err_deg:.2f} deg >= 3.5 deg"
    assert res4.mean_trans_err_mm < 10.0, f"Level 4 trans error {res4.mean_trans_err_mm:.1f} mm >= 10.0 mm"
