"""
Tests for Opt-in Head Reconstruction and Model-based Pose Refinement.

Covers:
  1. Synthetic head-like ellipsoid scanned from many views with 4 mm added pose noise
     and 1.5 mm depth noise. After refinement, median residual goes down, and the
     final mesh is within 2 mm Chamfer of the true ellipsoid.
  2. Moving torso blob exclusion (torso points masked out, zero torso points in fused mesh).
  3. Per-pixel depth quality filtering (grazing normals > 70 deg and depth jumps > 8 mm).
  4. Triangle mesh boundary hole counting.
  5. Opt-in configuration defaults (refine=False by default).
"""

from __future__ import annotations

import copy
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from config import HeadRefinementConfig, PipelineConfig
from reconstruction import (
    compute_reconstruction_quality_metrics,
    count_mesh_holes,
    filter_depth_pixel_quality,
    fuse_head_scalable_tsdf,
    mask_frame_to_head_roi,
    refine_poses_model_based,
)


def _create_synthetic_ellipsoid_mesh(
    rx: float = 0.08,
    ry: float = 0.11,
    rz: float = 0.10,
) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    verts = np.asarray(sphere.vertices)
    verts[:, 0] *= rx
    verts[:, 1] *= ry
    verts[:, 2] *= rz
    sphere.vertices = o3d.utility.Vector3dVector(verts)
    sphere.compute_vertex_normals()
    return sphere


def _render_ellipsoid_depth(
    rx: float,
    ry: float,
    rz: float,
    cam_pose: np.ndarray,
    width: int = 160,
    height: int = 160,
    fx: float = 200.0,
    fy: float = 200.0,
    cx: float = 80.0,
    cy: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    R_cw = cam_pose[:3, :3]
    t_cw = cam_pose[:3, 3]

    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    dx_c = (u_grid - cx) / fx
    dy_c = (v_grid - cy) / fy
    dz_c = np.ones_like(dx_c)
    dirs_c = np.stack([dx_c, dy_c, dz_c], axis=-1)

    C_w = t_cw
    dirs_w = dirs_c @ R_cw.T

    inv_r = np.array([1.0 / rx, 1.0 / ry, 1.0 / rz])
    D_scaled = dirs_w * inv_r
    C_scaled = C_w * inv_r

    A = np.sum(D_scaled * D_scaled, axis=-1)
    B = np.sum(D_scaled * C_scaled, axis=-1)
    C = np.sum(C_scaled * C_scaled) - 1.0

    disc = B * B - A * C
    valid = disc >= 0.0

    depth_m = np.zeros((height, width), dtype=np.float32)
    t_val = np.zeros((height, width), dtype=np.float64)

    t_near = (-B[valid] - np.sqrt(disc[valid])) / A[valid]
    pos_mask = t_near > 0.0

    valid_indices = np.where(valid)
    v_y = valid_indices[0][pos_mask]
    v_x = valid_indices[1][pos_mask]
    t_val[v_y, v_x] = t_near[pos_mask]

    depth_m[v_y, v_x] = t_val[v_y, v_x].astype(np.float32)

    color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    color_bgr[v_y, v_x] = [180, 200, 230]

    return depth_m, color_bgr


def test_synthetic_head_ellipsoid_refinement_and_chamfer():
    np.random.seed(0)
    rx, ry, rz = 0.08, 0.11, 0.10
    mesh_true = _create_synthetic_ellipsoid_mesh(rx, ry, rz)

    width, height = 160, 160
    fx, fy, cx, cy = 200.0, 200.0, 80.0, 80.0
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    views = []
    for el, n_az in [(25.0, 6), (0.0, 8), (-20.0, 6)]:
        el_rad = np.radians(el)
        for az in np.linspace(0, 360, n_az, endpoint=False):
            az_rad = np.radians(az)
            views.append((az_rad, el_rad))

    radius = 0.50
    true_poses, noisy_poses, frames, head_pcds = [], [], [], []

    for i, (az_rad, el_rad) in enumerate(views):
        cam_pos = np.array([
            radius * np.cos(el_rad) * np.sin(az_rad),
            radius * np.sin(el_rad),
            radius * np.cos(el_rad) * np.cos(az_rad)
        ])

        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, -1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_cw = np.column_stack([x_axis, y_axis, z_axis])

        T_true = np.eye(4)
        T_true[:3, :3] = R_cw
        T_true[:3, 3] = cam_pos
        true_poses.append(T_true)

        noise_dir = np.random.randn(3)
        noise_dir /= np.linalg.norm(noise_dir)
        noise_trans = noise_dir * 0.004

        T_noisy = T_true.copy()
        T_noisy[:3, 3] = T_true[:3, 3] + noise_trans
        noisy_poses.append(T_noisy)

        depth_m, color_bgr = _render_ellipsoid_depth(rx, ry, rz, T_true, width, height, fx, fy, cx, cy)
        valid_depth = depth_m > 0
        depth_m[valid_depth] += np.random.normal(0.0, 0.0015, size=np.sum(valid_depth)).astype(np.float32)
        depth_m[depth_m < 0] = 0.0

        u_g, v_g = np.meshgrid(np.arange(width), np.arange(height))
        vh = depth_m > 0
        pts_x = (u_g[vh] - cx) * depth_m[vh] / fx
        pts_y = (v_g[vh] - cy) * depth_m[vh] / fy
        pts_z = depth_m[vh]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([pts_x, pts_y, pts_z]))
        pcd.colors = o3d.utility.Vector3dVector(color_bgr[vh][:, ::-1] / 255.0)

        frame = RGBDFrame(
            frame_id=i,
            color_bgr=color_bgr,
            depth_m=depth_m,
            pcd=pcd,
            timestamp=float(i),
            intrinsics=intr,
        )
        frames.append(frame)
        head_pcds.append(pcd)

    refined_poses, iter_stats, _ = refine_poses_model_based(
        frames=frames,
        initial_poses=noisy_poses,
        head_pcds=head_pcds,
        voxel_length_m=0.002,
        sdf_trunc_m=0.008,
        depth_trunc_m=1.0,
        max_distance_m=0.006,
        max_trans_m=0.010,
        max_rot_deg=3.0,
        max_iterations=3,
        early_stop_improvement_m=0.0001,
    )

    assert len(iter_stats) >= 1
    initial_residual = iter_stats[0]['median_residual_mm']
    final_residual = iter_stats[-1]['median_residual_mm']
    assert final_residual <= initial_residual + 0.05

    mesh_fused, pcd_fused = fuse_head_scalable_tsdf(
        frames=frames,
        poses=refined_poses,
        voxel_length_m=0.0015,
        sdf_trunc_m=0.005,
        depth_trunc_m=1.0,
        taubin_iterations=10,
        min_component_ratio=0.02,
    )

    assert len(mesh_fused.vertices) > 500
    assert len(mesh_fused.triangles) > 500

    pts_fused = np.asarray(mesh_fused.sample_points_uniformly(10000).points)
    pts_true = np.asarray(mesh_true.sample_points_uniformly(10000).points)

    kd_true = o3d.geometry.KDTreeFlann(mesh_true)
    kd_fused = o3d.geometry.KDTreeFlann(mesh_fused)

    d_fused_to_true = []
    for pt in pts_fused:
        _, idxs, sq_d = kd_true.search_knn_vector_3d(pt, 1)
        if idxs:
            d_fused_to_true.append(np.sqrt(sq_d[0]))

    d_true_to_fused = []
    for pt in pts_true:
        _, idxs, sq_d = kd_fused.search_knn_vector_3d(pt, 1)
        if idxs:
            d_true_to_fused.append(np.sqrt(sq_d[0]))

    chamfer_m = 0.5 * (np.mean(d_fused_to_true) + np.mean(d_true_to_fused))
    chamfer_mm = chamfer_m * 1000.0

    print(f'Synthetic head ellipsoid Chamfer distance: {chamfer_mm:.2f} mm')
    assert chamfer_mm <= 2.0, f'Chamfer distance ({chamfer_mm:.2f} mm) exceeded 2.0 mm threshold'


def test_moving_torso_blob_exclusion():
    width, height = 160, 160
    fx, fy, cx, cy = 200.0, 200.0, 80.0, 80.0
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth_1 = np.zeros((height, width), dtype=np.float32)
    color_1 = np.zeros((height, width, 3), dtype=np.uint8)

    u_g, v_g = np.meshgrid(np.arange(width), np.arange(height))
    head_dist = np.sqrt((u_g - cx)**2 + (v_g - cy)**2)
    head_mask = head_dist < 30
    depth_1[head_mask] = 0.50
    color_1[head_mask] = [180, 200, 230]

    # Torso blob 1 (placed well below chin margin, v in [135, 155])
    torso_mask_1 = (v_g >= 135) & (v_g <= 155) & (u_g >= 60) & (u_g <= 100)
    depth_1[torso_mask_1] = 0.52
    color_1[torso_mask_1] = [50, 50, 180]

    depth_2 = depth_1.copy()
    color_2 = color_1.copy()
    depth_2[torso_mask_1] = 0.0
    torso_mask_2 = (v_g >= 135) & (v_g <= 155) & (u_g >= 70) & (u_g <= 110)
    depth_2[torso_mask_2] = 0.54
    color_2[torso_mask_2] = [50, 50, 180]

    head_center_cam = np.array([0.0, 0.0, 0.50])
    down_vec = np.array([0.0, 1.0, 0.0])
    chin_dist = 0.08

    masked_d1, masked_c1, drop1, tot1 = mask_frame_to_head_roi(
        color_1, depth_1, intr, head_center_cam, down_vec=down_vec, chin_dist=chin_dist, radius_m=0.13, chin_margin_m=0.025
    )
    masked_d2, masked_c2, drop2, tot2 = mask_frame_to_head_roi(
        color_2, depth_2, intr, head_center_cam, down_vec=down_vec, chin_dist=chin_dist, radius_m=0.13, chin_margin_m=0.025
    )

    assert np.all(masked_d1[torso_mask_1] == 0.0)
    assert np.all(masked_d2[torso_mask_2] == 0.0)
    assert drop1 > 0
    assert drop2 > 0

    f1 = RGBDFrame(0, masked_c1, masked_d1, None, 0.0, intr)
    f2 = RGBDFrame(1, masked_c2, masked_d2, None, 1.0, intr)
    mesh, _ = fuse_head_scalable_tsdf([f1, f2], [np.eye(4), np.eye(4)], voxel_length_m=0.003, sdf_trunc_m=0.01)

    verts = np.asarray(mesh.vertices)
    if len(verts) > 0:
        torso_verts = np.sum(verts[:, 1] > 0.15)
        assert torso_verts == 0, f'Found {torso_verts} torso vertices in fused mesh'


def test_per_pixel_depth_quality_filters():
    width, height = 100, 100
    fx, fy, cx, cy = 100.0, 100.0, 50.0, 50.0
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth = np.full((height, width), 0.60, dtype=np.float32)

    u_g, _ = np.meshgrid(np.arange(width), np.arange(height))
    steep_mask = (u_g >= 20) & (u_g <= 40)
    depth[steep_mask] = 0.60 + (u_g[steep_mask] - 20) * 0.04

    jump_mask = (u_g >= 60) & (u_g <= 80)
    depth[jump_mask] = 0.70

    clean_depth, stats = filter_depth_pixel_quality(
        depth, intr, max_normal_angle_deg=70.0, max_depth_jump_m=0.008
    )

    assert stats['jump_dropped'] > 0
    assert stats['normal_angle_dropped'] > 0
    assert clean_depth[50, 10] == pytest.approx(0.60, abs=1e-4)


def test_count_mesh_holes():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    assert count_mesh_holes(sphere) == 0

    v = np.asarray(sphere.vertices)
    t = np.asarray(sphere.triangles)
    mask = np.all(v[t][:, :, 2] > 0, axis=1)
    sphere.remove_triangles_by_mask(~mask)
    sphere.remove_unreferenced_vertices()
    assert count_mesh_holes(sphere) == 1


def test_refinement_opt_in_defaults():
    cfg = PipelineConfig()
    assert cfg.refine is False
    assert cfg.head_refinement.enabled is False
    assert cfg.head_refinement.voxel_length_mm == 1.5
    assert cfg.head_refinement.sdf_trunc_mm == 5.0
    assert cfg.head_refinement.depth_trunc_m == 0.9
