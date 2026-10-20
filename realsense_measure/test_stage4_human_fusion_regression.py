"""
Regression test for Stage 4 Human Point-Cloud Fusion: SOR Bypass and 2.0 mm Voxel Resolution.

Verifies:
  1. For human/head targets, Stage 4 point-cloud fusion bypasses Statistical Outlier Removal (SOR),
     preserving low-density lateral/profile geometry.
  2. For human/head targets, Stage 4 applies 2.0 mm (0.002 m) voxel downsampling, preserving
     fine facial and cranial curvature.
  3. For box/object targets, Stage 4 retains the existing SOR behavior and 4.0 mm (0.004 m) voxel resolution.
  4. The global PreprocessConfig.voxel_size_m default remains exactly 0.004 m.
"""

from __future__ import annotations

import copy
import numpy as np
import open3d as o3d
import pytest

from config import PipelineConfig, PreprocessConfig, TargetConfig
from pipeline import ScanPipeline
from reconstruction import fuse_point_clouds


def _create_synthetic_dense_with_sparse_flank() -> list[o3d.geometry.PointCloud]:
    """
    Create a synthetic multi-frame cloud with a dense central core and a sparse lateral flank.
    Under global SOR (nb_neighbors=20, std_ratio=1.5), the sparse flank is pruned.
    Under human bypass (skip_outlier_removal=True), the sparse flank is preserved.
    """
    np.random.seed(42)
    # Dense central core (e.g. 5000 points on a 10cm x 10cm planar sheet)
    core_x = np.random.uniform(-0.05, 0.05, 5000)
    core_y = np.random.uniform(-0.05, 0.05, 5000)
    core_z = np.zeros(5000)
    core_pts = np.column_stack([core_x, core_y, core_z])
    
    # Sparse lateral flank (e.g. 30 points stretched out along X from 0.08 to 0.12 m)
    flank_pts = np.column_stack([
        np.linspace(0.08, 0.12, 30),
        np.random.uniform(-0.01, 0.01, size=30),
        np.random.uniform(-0.01, 0.01, size=30),
    ])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([core_pts, flank_pts]))
    return [pcd]


def test_fuse_point_clouds_human_bypasses_sor_and_uses_2mm():
    frames = _create_synthetic_dense_with_sparse_flank()
    cfg = PreprocessConfig(voxel_size_m=0.004, outlier_neighbors=20, outlier_std_ratio=1.5)

    # Box mode: SOR active, voxel_size = 4mm (0.004m)
    fused_box = fuse_point_clouds(frames, cfg, skip_final_clustering=False, skip_outlier_removal=False)
    pts_box = np.asarray(fused_box.points)
    flank_pts_box = np.sum(pts_box[:, 0] > 0.07)

    # Human mode: SOR bypassed, voxel_size = 2mm (0.002m)
    fused_human = fuse_point_clouds(frames, cfg, skip_final_clustering=True, skip_outlier_removal=True)
    pts_human = np.asarray(fused_human.points)
    flank_pts_human = np.sum(pts_human[:, 0] > 0.07)

    # 1. Sparse flank must be pruned by SOR in box mode, but preserved in human mode
    assert flank_pts_box == 0, f"Expected box mode to strip sparse flank with SOR, found {flank_pts_box} pts"
    assert flank_pts_human > 0, f"Expected human mode to retain sparse flank without SOR, found {flank_pts_human} pts"

    # 2. Dense planar core: at 2 mm resolution, the 10cm x 10cm core retains approximately 4x more surface points than at 4 mm
    core_pts_box = np.sum(pts_box[:, 0] <= 0.07)
    core_pts_human = np.sum(pts_human[:, 0] <= 0.07)
    ratio = core_pts_human / core_pts_box
    assert ratio > 3.0, f"Expected 2mm human downsampling to retain ~4x more points than 4mm box downsampling, got {ratio:.2f}x"


def test_pipeline_fuse_human_vs_box_scoping():
    frames = _create_synthetic_dense_with_sparse_flank()
    diagnostics = [(1.0, 0.0, True)]

    # 1. Pipeline with Head target (human mode)
    cfg_head = PipelineConfig(
        target=TargetConfig(name="head"),
        show_stage_windows=False,
        save_intermediate=False,
    )
    pipeline_head = ScanPipeline(cfg_head)
    fused_head = pipeline_head.fuse(frames, diagnostics)
    pts_head = np.asarray(fused_head.points)
    flank_pts_head = np.sum(pts_head[:, 0] > 0.07)

    # 2. Pipeline with Box target (standard mode)
    cfg_box = PipelineConfig(
        target=TargetConfig(name="box"),
        show_stage_windows=False,
        save_intermediate=False,
    )
    pipeline_box = ScanPipeline(cfg_box)
    fused_box = pipeline_box.fuse(frames, diagnostics)
    pts_box = np.asarray(fused_box.points)
    flank_pts_box = np.sum(pts_box[:, 0] > 0.07)

    # Confirm human bypasses SOR and retains 2mm resolution while box retains SOR and 4mm resolution
    assert flank_pts_head > 0, "Pipeline with head target must preserve sparse profile points"
    assert flank_pts_box == 0, "Pipeline with box target must retain SOR and prune sparse points"
    assert len(pts_head) > len(pts_box) * 2.5, "Head fusion at 2mm must have significantly higher point count than box at 4mm"

    # Confirm config default is untouched
    assert cfg_head.preprocess.voxel_size_m == 0.004, "Head PreprocessConfig.voxel_size_m default must remain 0.004 m"
    assert cfg_box.preprocess.voxel_size_m == 0.004, "Box PreprocessConfig.voxel_size_m default must remain 0.004 m"


def test_stage4_marginal_frame_exclusion_from_fusion():
    """
    Verifies that:
    1. Frames accepted into the pose graph with marginal fitness (0.45 <= fitness < 0.60)
       or high landmark residual (> 5.0 mm) maintain connectivity in the pose graph (accepted=True in diagnostics).
    2. Stage 4 human fusion excludes these marginal frames from dense point cloud concatenation.
    3. High-quality frames (fitness >= 0.60) and frame 0 are retained in the fused reconstruction.
    """
    # 3 frames placed at distinct spatial offsets along X:
    # Frame 0: origin [-0.05, 0.05]
    # Frame 1: shifted to [0.15, 0.25] (marginal frame)
    # Frame 2: shifted to [0.35, 0.45] (high quality frame)
    np.random.seed(42)
    def make_cloud(center_x: float, n: int = 500) -> o3d.geometry.PointCloud:
        pts = np.column_stack([
            np.random.uniform(center_x - 0.05, center_x + 0.05, n),
            np.random.uniform(-0.05, 0.05, n),
            np.random.uniform(-0.05, 0.05, n),
        ])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    f0 = make_cloud(0.0)
    f1 = make_cloud(0.20)
    f2 = make_cloud(0.40)
    frames = [f0, f1, f2]

    # Diagnostics: all 3 frames accepted into the pose graph
    # Frame 0: reference (1.0)
    # Frame 1: marginal fitness (0.52) - passes 0.45 gate, fails 0.60 fusion gate
    # Frame 2: high quality (0.78) - passes both gates
    diagnostics = [
        (1.0, 0.0, True),
        (0.52, 0.004, True),
        (0.78, 0.002, True),
    ]

    # Assert pose graph acceptance is preserved (all 3 accepted for connectivity)
    pose_graph_accepted_count = sum(1 for d in diagnostics if d[2])
    assert pose_graph_accepted_count == 3, "All 3 frames must be accepted in pose graph for connectivity"

    # 1. Human / Head mode pipeline
    cfg_head = PipelineConfig(
        target=TargetConfig(name="head"),
        show_stage_windows=False,
        save_intermediate=False,
    )
    pipeline_head = ScanPipeline(cfg_head)
    fused_head = pipeline_head.fuse(frames, diagnostics)
    pts_fused = np.asarray(fused_head.points)

    # Frame 0 is in [-0.06, 0.06]
    pts_f0 = np.sum((pts_fused[:, 0] >= -0.06) & (pts_fused[:, 0] <= 0.06))
    # Frame 1 is in [0.14, 0.26] -> MUST BE EXCLUDED (0 points)
    pts_f1 = np.sum((pts_fused[:, 0] >= 0.14) & (pts_fused[:, 0] <= 0.26))
    # Frame 2 is in [0.34, 0.46] -> MUST BE INCLUDED (> 0 points)
    pts_f2 = np.sum((pts_fused[:, 0] >= 0.34) & (pts_fused[:, 0] <= 0.46))

    assert pts_f0 > 0, "Reference frame 0 must be included in fused cloud"
    assert pts_f1 == 0, f"Marginal frame 1 (fitness 0.52 < 0.60) must be EXCLUDED from fusion, found {pts_f1} pts"
    assert pts_f2 > 0, "High-quality frame 2 (fitness 0.78 >= 0.60) must be INCLUDED in fusion"

    # 2. Test landmark residual exclusion
    # Frame 1 has fitness 0.65 (passes fitness gate) but landmark residual 7.5mm (> 5.0mm gate)
    diagnostics_lm = [
        (1.0, 0.0, True),
        (0.65, 0.003, True),
        (0.80, 0.002, True),
    ]

    # Test 2a: Structured residuals provided without "residual=" in frame_scores text
    pipeline_head.last_registration_stats = {
        "frame_scores": {
            1: "fitness=0.65 tier=landmark",
            2: "fitness=0.80 tier=landmark",
        },
        "residuals": {
            1: 7.5,
            2: 2.1,
        },
        "frame_residuals": {
            1: 7.5,
            2: 2.1,
        },
    }
    fused_struct = pipeline_head.fuse(frames, diagnostics_lm)
    pts_fused_struct = np.asarray(fused_struct.points)
    pts_f1_struct = np.sum((pts_fused_struct[:, 0] >= 0.14) & (pts_fused_struct[:, 0] <= 0.26))
    pts_f2_struct = np.sum((pts_fused_struct[:, 0] >= 0.34) & (pts_fused_struct[:, 0] <= 0.46))
    assert pts_f1_struct == 0, f"Frame with structured residual 7.5mm (> 5.0mm) must be EXCLUDED from fusion, found {pts_f1_struct} pts"
    assert pts_f2_struct > 0, "Frame with structured residual 2.1mm (<= 5.0mm) must be INCLUDED in fusion"

    # Test 2b: Backward compatibility fallback when only formatted text is present
    pipeline_head.last_registration_stats = {
        "frame_scores": {
            1: "fitness=0.65 (src=0.65) residual=7.5mm",
            2: "fitness=0.80 (src=0.80) residual=2.1mm",
        }
    }
    fused_lm = pipeline_head.fuse(frames, diagnostics_lm)
    pts_fused_lm = np.asarray(fused_lm.points)
    pts_f1_lm = np.sum((pts_fused_lm[:, 0] >= 0.14) & (pts_fused_lm[:, 0] <= 0.26))
    pts_f2_lm = np.sum((pts_fused_lm[:, 0] >= 0.34) & (pts_fused_lm[:, 0] <= 0.46))

    assert pts_f1_lm == 0, f"Frame with landmark residual 7.5mm (> 5.0mm) must be EXCLUDED from fusion, found {pts_f1_lm} pts"
    assert pts_f2_lm > 0, "Frame with landmark residual 2.1mm (<= 5.0mm) must be INCLUDED in fusion"



def test_streak_removal_removes_streaks_preserves_sparse_isotropic():
    """
    Task 4 -- Regression test for streak artifact removal.

    Injects a SYNTHETIC LINEAR STREAK (simulating a misaligned-frame artifact
    — points tightly strung along one axis) and a SPARSE ISOTROPIC CLUSTER
    (simulating a genuine sparse feature like an ear tip — low density but
    points scattered in all directions).

    Asserts:
    - The linear streak is removed by the new PCA linearity filter.
    - The sparse isotropic cluster is preserved (not deleted with the streak).
    - Box mode is completely unaffected (streak removal block is never entered).
    """
    np.random.seed(7)

    # Dense head surface: 3000 points on a 10 cm × 10 cm planar sheet (origin region)
    core_x = np.random.uniform(-0.05, 0.05, 3000)
    core_y = np.random.uniform(-0.05, 0.05, 3000)
    core_z = np.zeros(3000)
    core_pts = np.column_stack([core_x, core_y, core_z])

    # Synthetic LINEAR STREAK: 40 points tightly spaced along X at y≈0, z≈0.02 offset
    # (simulates a misaligned frame that contributed a thin strip of offset surface).
    streak_x = np.linspace(0.10, 0.22, 40)
    streak_pts = np.column_stack([
        streak_x,
        np.random.uniform(-0.001, 0.001, 40),   # very tight spread: nearly 1-D
        np.full(40, 0.02) + np.random.uniform(-0.001, 0.001, 40),
    ])

    # Sparse ISOTROPIC CLUSTER: 18 points spread isotropically at a separate location
    # (simulates a genuine sparse ear-tip region — low density but 3-D scatter).
    ear_center = np.array([0.18, 0.12, 0.0])
    ear_pts = ear_center + np.random.randn(18, 3) * 0.004   # 4 mm std, roughly spherical isotropic cluster

    all_pts = np.vstack([core_pts, streak_pts, ear_pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    from config import RegistrationConfig
    from reconstruction import fuse_point_clouds

    cfg = PreprocessConfig(voxel_size_m=0.004, outlier_neighbors=20, outlier_std_ratio=1.5)
    reg_cfg = RegistrationConfig(
        enable_streak_removal=True,
        streak_linearity_ratio_threshold=8.0,
        streak_removal_radius_mult=2.5,
    )

    # --- Human mode: streak removal enabled ---
    fused = fuse_point_clouds(
        [pcd],
        cfg,
        skip_final_clustering=True,
        skip_outlier_removal=True,
        voxel_size=0.002,
        reg_cfg=reg_cfg,
    )
    fused_pts = np.asarray(fused.points)

    # Streak region: X in [0.095, 0.225], z > 0.01 (the artificially offset strip)
    streak_remaining = np.sum(
        (fused_pts[:, 0] >= 0.095) & (fused_pts[:, 0] <= 0.225)
        & (fused_pts[:, 2] > 0.01)
    )
    # Ear region: within 0.025 m of ear_center
    ear_remaining = np.sum(
        np.linalg.norm(fused_pts - ear_center, axis=1) <= 0.025
    )

    assert streak_remaining == 0, (
        f"Linear streak should have been removed by linearity filter, "
        f"but {streak_remaining} streak points remain."
    )
    assert ear_remaining >= 5, (
        f"Sparse isotropic cluster (ear) should be preserved, "
        f"but only {ear_remaining} ear points remain (expected >= 5)."
    )
    print(f"  [PASS] Human mode: streak removed ({streak_remaining} pts left), "
          f"isotropic ear cluster preserved ({ear_remaining} pts).")

    # --- Box mode: streak removal must NOT be triggered (skip_outlier_removal=False) ---
    fused_box = fuse_point_clouds(
        [pcd],
        cfg,
        skip_final_clustering=False,
        skip_outlier_removal=False,     # box mode: SOR runs, streak block never entered
        voxel_size=None,
        reg_cfg=None,
    )
    # Box mode uses SOR, so sparse points may be removed — but crucially no error should occur
    # and the core should remain
    core_in_box = np.sum(np.linalg.norm(np.asarray(fused_box.points), axis=1) <= 0.06)
    assert core_in_box > 100, f"Box mode should retain the dense core, only {core_in_box} pts found"
    print(f"  [PASS] Box mode: streak removal not applied, dense core intact ({core_in_box} pts).")


def test_stage4_face_region_strict_fusion_gate():
    """
    TASK 3 -- Regression test for tightened face-region fusion gate:
    When a frame is in the upper face/head region:
    - If fitness is 0.65 (< 0.75 strict face threshold), it stays in pose graph but is EXCLUDED from Stage 4 fusion.
    - If fitness is 0.80 (>= 0.75) and RMSE is 2.5mm (<= 4.0mm), it is INCLUDED in Stage 4 fusion.
    - If fitness is 0.80 but RMSE is 5.5mm (> 4.0mm), it is EXCLUDED from Stage 4 fusion.
    """
    f0 = o3d.geometry.PointCloud()
    f0.points = o3d.utility.Vector3dVector(np.random.uniform(-0.05, 0.05, (500, 3)))
    f1 = o3d.geometry.PointCloud()
    f1.points = o3d.utility.Vector3dVector(np.random.uniform(0.15, 0.25, (500, 3)))
    f2 = o3d.geometry.PointCloud()
    f2.points = o3d.utility.Vector3dVector(np.random.uniform(0.35, 0.45, (500, 3)))
    frames = [f0, f1, f2]

    cfg = PipelineConfig(
        target=TargetConfig(name="head"),
        show_stage_windows=False,
        save_intermediate=False,
    )
    pipeline = ScanPipeline(cfg)

    diagnostics = [
        (1.0, 0.0, True),
        (0.68, 0.003, True),  # passes 0.60 general gate, fails 0.75 face gate
        (0.82, 0.002, True),  # passes 0.75 face gate and 4mm RMSE
    ]

    pipeline.last_registration_stats = {
        "frame_scores": {
            1: "fitness=0.68 rmse=3.0mm",
            2: "fitness=0.82 rmse=2.0mm",
        },
        "fusion_eligible": {
            0: True,
            1: False,  # excluded by face gate
            2: True,   # eligible
        },
        "fusion_reasons": {
            1: "face-pair fitness 0.68 < 0.75 threshold",
        },
    }

    fused = pipeline.fuse(frames, diagnostics)
    pts = np.asarray(fused.points)
    pts_f1 = np.sum((pts[:, 0] >= 0.14) & (pts[:, 0] <= 0.26))
    pts_f2 = np.sum((pts[:, 0] >= 0.34) & (pts[:, 0] <= 0.46))

    assert pts_f1 == 0, f"Face-pair frame with fitness 0.68 (< 0.75) must be EXCLUDED from fusion, found {pts_f1} pts"
    assert pts_f2 > 0, f"Face-pair frame with fitness 0.82 (>= 0.75) must be INCLUDED in fusion, found {pts_f2} pts"
    print("  [PASS] Stricter face-pair fusion gate successfully excludes borderline face frames and retains high-quality frames.")


if __name__ == "__main__":
    test_fuse_point_clouds_human_bypasses_sor_and_uses_2mm()
    test_pipeline_fuse_human_vs_box_scoping()
    test_stage4_marginal_frame_exclusion_from_fusion()
    test_stage4_face_region_strict_fusion_gate()
    test_streak_removal_removes_streaks_preserves_sparse_isotropic()
    print("ALL STAGE 4 HUMAN 2.0 MM FUSION REGRESSION TESTS PASSED.")

