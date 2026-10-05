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


if __name__ == "__main__":
    test_fuse_point_clouds_human_bypasses_sor_and_uses_2mm()
    test_pipeline_fuse_human_vs_box_scoping()
    print("ALL STAGE 4 HUMAN 2.0 MM FUSION REGRESSION TESTS PASSED.")
