"""
Focused regression test for human head mutual-support registration fitness.

Validates the fix for the asymmetric point-count disparity between head-only frames
(e.g., Frame 26, ~3,750 points) and head+torso connected frames (e.g., Frame 27, ~11,500 points).
Under Open3D's source-normalized fitness (inliers / N_source), the maximum possible fitness is
capped around 3,750 / 11,500 ≈ 0.326, which is below both the Colored ICP threshold (0.45)
and FPFH threshold (0.35).

The mutual fitness metric:
    mutual_fitness = inlier_count / min(N_source, N_target)
allows genuine head-to-head overlap to be correctly recognized while preserving all
geometric safety gates (RMSE, rotation, translation, and pose jump).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from config import RegistrationConfig
from registration import compute_mutual_fitness, register_rgbd_pair
from test_head_synthetic import create_synthetic_head


def _make_intrinsics() -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(848, 480, 428.46, 427.90, 422.17, 244.60)


def test_compute_mutual_fitness_formula():
    """Unit test for the mutual fitness mathematical formulation."""
    # S has 11500 pts, T has 3750 pts. All 3750 head pts overlap perfectly.
    # Open3D source fitness is 3750 / 11500 = 0.326087
    src_fit = 3750.0 / 11500.0
    mut_fit = compute_mutual_fitness(src_fit, n_source=11500, n_target=3750)
    assert mut_fit == pytest.approx(1.0, abs=1e-4)

    # Symmetric case: S has 3750 pts, T has 11500 pts.
    src_fit_sym = 1.0  # 3750 / 3750
    mut_fit_sym = compute_mutual_fitness(src_fit_sym, n_source=3750, n_target=11500)
    assert mut_fit_sym == pytest.approx(1.0, abs=1e-4)

    # Equal size clouds: mutual fitness equals source fitness
    mut_equal = compute_mutual_fitness(0.80, n_source=4000, n_target=4000)
    assert mut_equal == pytest.approx(0.80, abs=1e-4)

    # Partial overlap: only 500 of 3750 points overlap
    src_fit_part = 500.0 / 11500.0
    mut_fit_part = compute_mutual_fitness(src_fit_part, n_source=11500, n_target=3750)
    assert mut_fit_part == pytest.approx(500.0 / 3750.0, abs=1e-4)


def test_asymmetric_head_torso_registration_accepted():
    """
    Reproduces the exact Frame 26 vs Frame 27 failure:
    - Target: head only (~3,750 points)
    - Source: head (~3,750 points) + torso (~7,750 points) -> 11,500 points total
    - Known rigid transform: rotation ~2.8 deg, translation ~17 mm

    Verifies:
      A. Open3D source-normalized fitness is capped around 3750/11500 ≈ 0.326 (< 0.35).
      B. Mutual fitness correctly reflects the high-quality head overlap (> 0.90 >= 0.45).
      C. RMSE, rotation, and translation satisfy all safety gates.
      D. The registration is cleanly ACCEPTED.
    """
    cfg = RegistrationConfig()
    intr = _make_intrinsics()

    # Target: head only (canonical synthetic head with nose)
    tgt_head = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=3750, with_nose=True)
    tgt_head.paint_uniform_color([0.8, 0.6, 0.5])

    # Source: head + additional torso
    torso_mesh = o3d.geometry.TriangleMesh.create_box(width=0.40, height=0.20, depth=0.18)
    torso_mesh.translate((-0.20, 0.12, -0.09))
    torso_pcd = torso_mesh.sample_points_poisson_disk(number_of_points=7750)
    torso_pcd.paint_uniform_color([0.2, 0.4, 0.8])

    # Apply known realistic rigid transform to source: R(2.0, 2.0, 0.0) deg, t=(10, 5, 10) mm
    R_known = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(2.0), np.radians(2.0), 0.0))
    t_known = np.array([0.010, 0.005, 0.010])
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_known
    T_gt[:3, 3] = t_known

    T_inv = np.linalg.inv(T_gt)
    src_head = o3d.geometry.PointCloud(tgt_head)
    src_head.transform(T_inv)
    src_torso = o3d.geometry.PointCloud(torso_pcd)
    src_torso.transform(T_inv)

    src_pcd = src_head + src_torso

    assert len(tgt_head.points) == 3750
    assert len(src_pcd.points) == 11500

    f_src = RGBDFrame(frame_id=1, color_bgr=None, depth_m=None, pcd=src_pcd, timestamp=1.0, intrinsics=intr)
    f_tgt = RGBDFrame(frame_id=0, color_bgr=None, depth_m=None, pcd=tgt_head, timestamp=0.0, intrinsics=intr)

    T_est, fit, rmse, acc, method, rot, trans, reason = register_rgbd_pair(f_src, f_tgt, cfg)

    # 1. Registration must be ACCEPTED
    assert acc is True, f"Registration failed unexpectedly: {reason}"
    assert method in ("colored_icp", "fpfh_fallback")

    # 2. Mutual fitness must be high (>= 0.45)
    assert fit >= cfg.min_colored_icp_fitness
    assert fit > 0.90, f"Expected mutual fitness > 0.90, got {fit:.3f}"

    # 3. Reason string must audit both mutual and source-normalized fitness
    assert "src=" in reason, f"Reason string should report source-normalized fitness: {reason}"

    # 4. Recovered transform must closely match ground truth
    assert rot < 5.0, f"Rotation error too large: {rot:.2f} deg"
    assert abs(trans - 15.0) < 5.0, f"Translation error too large: {trans:.2f} mm"
    assert rmse < cfg.max_colored_icp_rmse_m


def test_small_unrelated_patch_overlap_rejected():
    """
    Negative safety requirement:
    Verify that a candidate is NOT accepted merely because a small cloud
    happens to fit a small portion of a much larger unrelated cloud.
    """
    cfg = RegistrationConfig()
    intr = _make_intrinsics()

    tgt_head = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=3750, with_nose=True)
    tgt_head.paint_uniform_color([0.8, 0.6, 0.5])

    # Source has 11,500 points, but only a small patch (~500 pts = 13% of target) overlaps
    patch_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    patch_mesh.translate((0.0, 0.0, 0.10))
    patch_pcd = patch_mesh.sample_points_poisson_disk(number_of_points=500)

    unrelated_mesh = o3d.geometry.TriangleMesh.create_box(width=0.40, height=0.40, depth=0.40)
    unrelated_mesh.translate((0.30, 0.30, 0.30))
    unrelated_pcd = unrelated_mesh.sample_points_poisson_disk(number_of_points=11000)

    src_pcd = patch_pcd + unrelated_pcd
    src_pcd.paint_uniform_color([0.2, 0.4, 0.8])

    f_src = RGBDFrame(frame_id=1, color_bgr=None, depth_m=None, pcd=src_pcd, timestamp=1.0, intrinsics=intr)
    f_tgt = RGBDFrame(frame_id=0, color_bgr=None, depth_m=None, pcd=tgt_head, timestamp=0.0, intrinsics=intr)

    T_est, fit, rmse, acc, method, rot, trans, reason = register_rgbd_pair(f_src, f_tgt, cfg)

    # Must be REJECTED because mutual fitness (~0.13) < min_accept_fitness (0.35)
    assert acc is False, "Small partial overlap must not be accepted"
    assert fit < cfg.min_accept_fitness


def test_excessive_motion_jump_rejected_despite_overlap():
    """
    Negative safety requirement:
    Verify that even if mutual fitness is 1.0, rotation jump > 25 deg is strictly REJECTED.
    """
    cfg = RegistrationConfig()
    intr = _make_intrinsics()

    tgt_head = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=3750, with_nose=True)
    tgt_head.paint_uniform_color([0.8, 0.6, 0.5])

    # Rotate source by 35 degrees (exceeds max_rotation_deg_per_frame = 25.0 deg)
    R_35 = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.radians(35.0), 0.0))
    src_rot = o3d.geometry.PointCloud(tgt_head)
    src_rot.rotate(R_35, center=(0, 0, 0))

    f_src = RGBDFrame(frame_id=1, color_bgr=None, depth_m=None, pcd=src_rot, timestamp=1.0, intrinsics=intr)
    f_tgt = RGBDFrame(frame_id=0, color_bgr=None, depth_m=None, pcd=tgt_head, timestamp=0.0, intrinsics=intr)

    T_est, fit, rmse, acc, method, rot, trans, reason = register_rgbd_pair(f_src, f_tgt, cfg)

    assert acc is False, "Excessive rotation jump (> 25 deg) must be rejected"
    assert "rotation" in reason or "Fallback also failed" in reason
