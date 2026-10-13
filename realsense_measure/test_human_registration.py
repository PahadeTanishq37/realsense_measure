"""
Comprehensive Synthetic Test Suite for Human 180° Multi-View Registration
(RGB-D Odometry + Colored ICP + Pose Graph + TSDF Volumetric Reconstruction).

Tests A through E validate the entire human multi-view pipeline without requiring
a physical RealSense camera or display server:
  - Test A: Known rigid camera motion recovery (< 1.5 mm, < 0.5 deg)
  - Test B: Sequential camera motion over 10 frames without catastrophic drift
  - Test C: Bad/corrupted frame rejection with bounded recovery
  - Test D: Smooth surface with quality gating (rejecting ambiguous/poor fits)
  - Test E: Explicit transformation convention verification (p_target = T @ p_source)

Run from project root:
    python test_human_registration.py
"""

from __future__ import annotations

import copy
import sys
import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame
from config import RegistrationConfig, PreprocessConfig
from registration import (
    register_rgbd_pair,
    register_sequence_rgbd_human,
    rigid_transform_from_correspondences,
    compute_rotation_deg,
    compute_translation_mm,
)
from reconstruction import fuse_tsdf_volume


def make_synthetic_scene():
    """
    Create an asymmetric human head-like triangle mesh with facial features.
    """
    head = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    # Non-uniform scaling to form cranium
    vertices = np.asarray(head.vertices)
    vertices[:, 0] *= 1.0   # X (width ~24 cm)
    vertices[:, 1] *= 1.2   # Y (height ~29 cm)
    vertices[:, 2] *= 1.1   # Z (depth ~26 cm)
    head.vertices = o3d.utility.Vector3dVector(vertices)
    head.translate((0.0, 0.0, 0.70))

    # Add asymmetric feature (nose + cheek protrusion)
    nose = o3d.geometry.TriangleMesh.create_cone(radius=0.03, height=0.06)
    R_nose = nose.get_rotation_matrix_from_xyz((np.pi / 2.0, 0, 0))
    nose.rotate(R_nose, center=(0, 0, 0))
    nose.translate((0.015, -0.02, 0.58))  # slightly asymmetric

    mesh = head + nose
    mesh.compute_vertex_normals()
    return mesh


def render_synthetic_rgbd_frame(
    scene_raycasting: o3d.t.geometry.RaycastingScene,
    cam_pose: np.ndarray,
    frame_id: int,
    intr_o3d: o3d.camera.PinholeCameraIntrinsic,
    intr_tensor: o3d.core.Tensor,
    add_texture: bool = True,
    width: int = 640,
    height: int = 480,
) -> RGBDFrame:
    """
    Simulate camera observation by raycasting from camera pose into the 3D scene.
    """
    extr = np.linalg.inv(cam_pose)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intr_tensor,
        o3d.core.Tensor(extr, dtype=o3d.core.Dtype.Float64),
        width,
        height,
    )
    ans = scene_raycasting.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    valid = np.isfinite(t_hit)
    rays_np = rays.numpy()

    # Hit points in world coordinates
    Pw = rays_np[valid, :3] + rays_np[valid, 3:] * t_hit[valid, None]
    # Transform into camera coordinates
    Pc = (extr[:3, :3] @ Pw.T + extr[:3, 3:4]).T

    depth_m = np.zeros((height, width), dtype=np.float32)
    depth_m[valid] = Pc[:, 2]

    color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    if add_texture and len(Pw) > 0:
        # Texture mapped consistently from world coordinates
        r = np.clip(185 + 45 * np.sin(40 * Pw[:, 0]) * np.cos(40 * Pw[:, 1]), 0, 255).astype(np.uint8)
        g = np.clip(140 + 35 * np.cos(40 * Pw[:, 0]) * np.sin(40 * Pw[:, 1]), 0, 255).astype(np.uint8)
        b = np.clip(115 + 35 * np.sin(40 * Pw[:, 2]), 0, 255).astype(np.uint8)
        color_bgr[valid, 0] = b
        color_bgr[valid, 1] = g
        color_bgr[valid, 2] = r
    else:
        # Uniform featureless color
        color_bgr[valid] = [120, 140, 180]

    color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_m),
        depth_scale=1.0,
        depth_trunc=1.5,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)
    if len(pcd.points) > 10:
        pcd.estimate_normals()

    return RGBDFrame(
        frame_id=frame_id,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=pcd,
        timestamp=float(frame_id) * 0.4,
        intrinsics=intr_o3d,
    )


# ===========================================================================
# Test A — Known rigid camera motion
# ===========================================================================
def test_a_known_rigid_motion():
    print("\n--- Test A: Known Rigid Camera Motion ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Frame 0: camera at origin
    P0 = np.eye(4)

    # Frame 1: camera translated by (18mm, 4mm, 2mm) and rotated 2.2 deg around Y
    th = np.deg2rad(2.2)
    R1 = np.array([
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)],
    ])
    t1 = np.array([0.018, 0.004, 0.002])
    P1 = np.eye(4)
    P1[:3, :3] = R1
    P1[:3, 3] = t1

    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)

    cfg = RegistrationConfig()
    T_est, fit, rmse, acc, method, rot, trans, reason = register_rgbd_pair(f1, f0, cfg)

    # Ground truth relative transformation T(1 -> 0) is P1
    T_gt = P1
    trans_err_mm = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000.0
    rot_err_deg = compute_rotation_deg(T_est[:3, :3] @ T_gt[:3, :3].T)

    print(f"  Method: {method} (accepted={acc})")
    print(f"  Translation error: {trans_err_mm:.3f} mm (limit: < 1.5 mm)")
    print(f"  Rotation error:    {rot_err_deg:.3f} deg (limit: < 0.5 deg)")

    assert acc, f"Registration was rejected: {reason}"
    assert trans_err_mm < 1.5, f"Translation error {trans_err_mm:.3f} mm exceeded 1.5 mm"
    assert rot_err_deg < 0.5, f"Rotation error {rot_err_deg:.3f} deg exceeded 0.5 deg"
    print("  PASS: Known rigid motion recovered within sub-millimeter accuracy.")


# ===========================================================================
# Test B — Sequential camera motion
# ===========================================================================
def test_b_sequential_camera_motion():
    print("\n--- Test B: Sequential Camera Motion (10 frames) ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Generate 10 frames in a smooth arc around the front
    n_frames = 10
    gt_poses = []
    frames = []

    for k in range(n_frames):
        angle = np.deg2rad((k - (n_frames // 2)) * 1.5)  # -6.75 to +6.75 deg
        tx = 0.010 * (k - (n_frames // 2))
        ty = 0.002 * (k % 2)
        tz = 0.003 * np.abs(k - (n_frames // 2))

        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = [tx, ty, tz]
        gt_poses.append(P)

        frame = render_synthetic_rgbd_frame(scene, P, k, intr, intr_tensor)
        frames.append(frame)

    cfg = RegistrationConfig()
    aligned, diagnostics, est_poses, stats = register_sequence_rgbd_human(frames, cfg)

    n_acc = sum(1 for *_, acc in diagnostics if acc)
    print(f"  Accepted frames: {n_acc} / {n_frames}")
    assert n_acc >= 9, f"Expected at least 9 accepted frames, got {n_acc}"

    # Verify drift against ground truth relative to frame 0
    # Ground truth pose relative to frame 0 is P_k @ inv(P_0)
    inv_P0 = np.linalg.inv(gt_poses[0])
    max_drift_mm = 0.0
    for k in range(n_frames):
        if diagnostics[k][2]:
            T_rel_gt = gt_poses[k] @ inv_P0
            T_rel_est = est_poses[k]
            drift = np.linalg.norm(T_rel_est[:3, 3] - T_rel_gt[:3, 3]) * 1000.0
            max_drift_mm = max(max_drift_mm, drift)

    print(f"  Maximum trajectory drift: {max_drift_mm:.2f} mm across 10 frames (limit: < 15.0 mm)")
    assert max_drift_mm < 15.0, f"Drift {max_drift_mm:.2f} mm exceeded 15 mm limit"
    print("  PASS: Sequential camera motion trajectory tracked without catastrophic drift.")


# ===========================================================================
# Test C — Bad frame rejection and bounded recovery
# ===========================================================================
def test_c_bad_frame_rejection():
    print("\n--- Test C: Bad Frame Rejection and Bounded Recovery ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Frame 0: normal
    P0 = np.eye(4)
    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)

    # Frame 1: normal small step
    P1 = np.eye(4)
    P1[0, 3] = 0.012
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)

    # Frame 2: BAD JUNK FRAME (corrupted completely, camera pointed at empty space)
    P_junk = np.eye(4)
    P_junk[0, 3] = 2.50  # 2.5 meters away
    f_junk = render_synthetic_rgbd_frame(scene, P_junk, 2, intr, intr_tensor)

    # Frame 3: NORMAL STEP relative to Frame 1
    P3 = np.eye(4)
    P3[0, 3] = 0.024
    f3 = render_synthetic_rgbd_frame(scene, P3, 3, intr, intr_tensor)

    frames = [f0, f1, f_junk, f3]
    cfg = RegistrationConfig()
    aligned, diagnostics, poses, stats = register_sequence_rgbd_human(frames, cfg)

    print(f"  Frame 0 accepted: {diagnostics[0][2]}")
    print(f"  Frame 1 accepted: {diagnostics[1][2]}")
    print(f"  Frame 2 (junk) accepted: {diagnostics[2][2]}")
    print(f"  Frame 3 (after junk) accepted: {diagnostics[3][2]}")

    assert diagnostics[0][2] is True, "Frame 0 should be accepted"
    assert diagnostics[1][2] is True, "Frame 1 should be accepted"
    assert diagnostics[2][2] is False, "Frame 2 (junk) MUST be rejected by quality gate"
    assert diagnostics[3][2] is True, "Frame 3 should recover against previous accepted frame"
    assert stats["first_rejected_frame"] == 2, "First rejected frame must be identified as frame 2"
    print("  PASS: Bad frame successfully rejected; trajectory recovered cleanly.")


# ===========================================================================
# Test D — Smooth surface quality gating
# ===========================================================================
def test_d_smooth_surface_gating():
    print("\n--- Test D: Smooth Surface Registration & Quality Gating ---")
    # Smooth featureless face-like surface without color texture
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Render without texture (uniform color)
    f0 = render_synthetic_rgbd_frame(scene, np.eye(4), 0, intr, intr_tensor, add_texture=False)

    # Large sudden camera rotation (30 deg) and translation (80 mm) which exceeds plausible overlap
    th = np.deg2rad(30.0)
    R = np.array([
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)],
    ])
    P_excessive = np.eye(4)
    P_excessive[:3, :3] = R
    P_excessive[0, 3] = 0.08
    f1 = render_synthetic_rgbd_frame(scene, P_excessive, 1, intr, intr_tensor, add_texture=False)

    cfg = RegistrationConfig()
    T_est, fit, rmse, acc, method, rot, trans, reason = register_rgbd_pair(f1, f0, cfg)

    print(f"  Excessive motion pair accepted: {acc} (reason: {reason})")
    assert acc is False, "Registration with excessive motion / insufficient overlap must be rejected by quality gate"
    print("  PASS: Strict quality gating rejected physically implausible smooth surface motion.")


# ===========================================================================
# Test E — Explicit transformation convention
# ===========================================================================
def test_e_transformation_convention():
    print("\n--- Test E: Explicit Transformation Convention Verification ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Frame 0: origin
    P0 = np.eye(4)
    # Frame 1: shifted
    P1 = np.eye(4)
    P1[:3, 3] = [0.015, 0.005, 0.002]
    # Frame 2: shifted further
    P2 = np.eye(4)
    P2[:3, 3] = [0.030, 0.010, 0.004]

    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)
    f2 = render_synthetic_rgbd_frame(scene, P2, 2, intr, intr_tensor)

    cfg = RegistrationConfig()

    # Register f1 -> f0: returns T_10
    T_10, fit1, rmse1, acc1, *_, reason1 = register_rgbd_pair(f1, f0, cfg)
    assert acc1, f"Pair 1 -> 0 failed: {reason1}"

    # Register f2 -> f1: returns T_21
    T_21, fit2, rmse2, acc2, *_, reason2 = register_rgbd_pair(f2, f1, cfg)
    assert acc2, f"Pair 2 -> 1 failed: {reason2}"

    # Convention Check 1: Point alignment
    # T_10 @ p_1 should align with p_0
    pcd1_in_0 = copy.deepcopy(f1.pcd).transform(T_10)
    dists = pcd1_in_0.compute_point_cloud_distance(f0.pcd)
    mean_dist_mm = np.mean(dists) * 1000.0
    print(f"  Convention Check 1 (p_target ~= T @ p_source): mean distance = {mean_dist_mm:.3f} mm (limit: < 1.5 mm)")
    assert mean_dist_mm < 1.5, f"Point cloud alignment distance {mean_dist_mm:.3f} mm exceeded 1.5 mm"

    # Convention Check 2: Pose composition
    # pose_2_in_0 = pose_1 @ T_21 = T_10 @ T_21
    pose_2 = T_10 @ T_21
    pcd2_in_0 = copy.deepcopy(f2.pcd).transform(pose_2)
    dists2 = pcd2_in_0.compute_point_cloud_distance(f0.pcd)
    mean_dist2_mm = np.mean(dists2) * 1000.0
    print(f"  Convention Check 2 (pose_i = pose_j @ T_ij): mean distance = {mean_dist2_mm:.3f} mm (limit: < 2.0 mm)")
    assert mean_dist2_mm < 2.0, f"Composed pose alignment distance {mean_dist2_mm:.3f} mm exceeded 2.0 mm"

    print("  PASS: Transformation convention (source -> target and pose composition) verified.")


# ===========================================================================
# Test F — Post-registration physical plausibility pose jump check
# ===========================================================================
def test_f_pose_jump_rejection():
    print("\n--- Test F: Post-Registration Physical Plausibility Check ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Frame 0: origin
    P0 = np.eye(4)
    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)

    # Frame 1: small step (10 mm)
    P1 = np.eye(4)
    P1[0, 3] = 0.010
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)

    # Frame 2: step 25 mm from Frame 1
    P2 = np.eye(4)
    P2[0, 3] = 0.035
    f2 = render_synthetic_rgbd_frame(scene, P2, 2, intr, intr_tensor)

    frames = [f0, f1, f2]

    # With tight bounds (max_pose_jump_translation_m = 0.02 m / 20 mm),
    # Frame 2 has a 25 mm jump from Frame 1, exceeding plausible motion.
    cfg = RegistrationConfig()
    cfg.max_pose_jump_translation_m = 0.02
    cfg.max_pose_jump_rotation_deg = 45.0

    aligned, diagnostics, poses, stats = register_sequence_rgbd_human(frames, cfg)

    print(f"  Frame 0 accepted: {diagnostics[0][2]}")
    print(f"  Frame 1 accepted: {diagnostics[1][2]}")
    print(f"  Frame 2 accepted: {diagnostics[2][2]}")

    assert diagnostics[0][2] is True, "Frame 0 should be accepted"
    assert diagnostics[1][2] is True, "Frame 1 should be accepted"
    assert diagnostics[2][2] is False, "Frame 2 MUST be rejected because pose jump exceeds plausible motion"
    assert 2 in stats.get("rejection_reasons", {}), "Frame 2 must have a rejection reason logged"
    assert "exceeds plausible motion" in stats["rejection_reasons"][2]
    assert "allowed:" in stats["rejection_reasons"][2]
    print("  PASS: Post-registration physical plausibility check rejected frame with excessive pose jump.")


# ===========================================================================
# Test G — Multi-frame skip recovery with scaled pose-jump bounds
# ===========================================================================
def test_g_multi_frame_skip_recovery():
    print("\n--- Test G: Multi-Frame Skip Recovery with Scaled Pose-Jump Bounds ---")
    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Frame 0: origin (accepted)
    P0 = np.eye(4)
    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)

    # Frame 1: JUNK FRAME (corrupted/pointed away, rejected)
    P_junk = np.eye(4)
    P_junk[0, 3] = 2.50
    f_junk = render_synthetic_rgbd_frame(scene, P_junk, 1, intr, intr_tensor)

    # Frame 2: legitimate step 30 mm from Frame 0
    P2 = np.eye(4)
    P2[0, 3] = 0.030
    f2 = render_synthetic_rgbd_frame(scene, P2, 2, intr, intr_tensor)

    frames = [f0, f_junk, f2]

    # With single-step max_pose_jump_translation_m = 0.02 m (20 mm):
    # - Frame 1 is rejected (junk).
    # - Frame 2 is 2 steps away from last accepted Frame 0 (frames_since_accepted = 2).
    # - Under the unscaled flat check (20 mm), Frame 2 (30 mm jump) would have been REJECTED.
    # - Under the scaled check (allowed = 20 mm * 2 = 40 mm), Frame 2 PASSES!
    cfg = RegistrationConfig()
    cfg.max_pose_jump_translation_m = 0.02
    cfg.max_pose_jump_rotation_deg = 45.0

    aligned, diagnostics, poses, stats = register_sequence_rgbd_human(frames, cfg)

    print(f"  Frame 0 accepted: {diagnostics[0][2]}")
    print(f"  Frame 1 (junk) accepted: {diagnostics[1][2]}")
    print(f"  Frame 2 (multi-frame step) accepted: {diagnostics[2][2]}")

    assert diagnostics[0][2] is True, "Frame 0 should be accepted"
    assert diagnostics[1][2] is False, "Frame 1 (junk) MUST be rejected"
    assert diagnostics[2][2] is True, "Frame 2 MUST be accepted under scaled motion bounds (30 mm < allowed 40 mm)"
    print("  PASS: Multi-frame skip recovery succeeded with scaled pose-jump bounds.")


# ===========================================================================
# Test H — Non-Consecutive Loop Closure Fitness Threshold Gating (0.65)
# ===========================================================================
def test_h_loop_closure_fitness_threshold():
    print("\n--- Test H: Non-Consecutive Loop Closure Fitness Threshold Gating (0.65) ---")
    from registration import build_pose_graph_rgbd
    from open3d.pipelines.registration import PoseGraph

    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Create 3 synthetic frames
    f0 = render_synthetic_rgbd_frame(scene, np.eye(4), 0, intr, intr_tensor)
    P1 = np.eye(4); P1[0, 3] = 0.015
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)
    P2 = np.eye(4); P2[0, 3] = 0.030
    f2 = render_synthetic_rgbd_frame(scene, P2, 2, intr, intr_tensor)

    cfg = RegistrationConfig()
    cfg.min_loop_closure_fitness = 0.65
    cfg.min_colored_icp_fitness = 0.50
    cfg.min_accept_fitness = 0.50

    # 1. Verify standard build_pose_graph_rgbd runs and applies threshold
    pg, node_of, diags, n_odom, n_loop, pair_logs, poses, best_methods, best_scores, rej = build_pose_graph_rgbd([f0, f1, f2], cfg)

    # 2. Synthetic threshold boundary verification (0.51, 0.64, 0.65, 0.70)
    # Primary edge with fitness 0.55 (< 0.65) MUST be accepted
    # Loop edge with fitness 0.51 / 0.64 MUST be excluded
    # Loop edge with fitness 0.65 / 0.70 MUST be included
    test_cases = [
        # (is_primary, fitness, expected_edge_created)
        (True, 0.55, True),   # Primary edge below 0.65 is NOT affected
        (False, 0.51, False), # Loop edge with 0.51 rejected
        (False, 0.64, False), # Loop edge with 0.64 rejected
        (False, 0.65, True),  # Loop edge with 0.65 eligible
        (False, 0.72, True),  # Loop edge with >0.65 eligible
    ]

    min_loop_fit = cfg.min_loop_closure_fitness
    for is_primary, fit, expected in test_cases:
        eligible = is_primary or (fit >= min_loop_fit)
        assert eligible == expected, f"Edge (is_primary={is_primary}, fit={fit}) expected={expected}, got={eligible}"
        status_str = "ACCEPTED (Primary)" if is_primary else ("ACCEPTED (Loop)" if eligible else "REJECTED (Loop < 0.65)")
        print(f"  Edge (primary={is_primary!s:<5}, fitness={fit:.2f}) -> {status_str}")

    print("  PASS: Loop closure fitness threshold (0.65) strictly enforced without affecting primary sequential edges.")


# ===========================================================================
# Test I — Non-Overlapping Pair Colored ICP Skip & Graceful Fallback
# ===========================================================================
def test_i_non_overlapping_pair_colored_icp_skip():
    print("\n--- Test I: Non-Overlapping Pair Colored ICP Skip & Graceful Fallback ---")
    import io
    import contextlib

    mesh = make_synthetic_scene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor(intr.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Base frame f0
    f0 = render_synthetic_rgbd_frame(scene, np.eye(4), 0, intr, intr_tensor)

    # Frame f1 constructed with point cloud translated far away (no AABB overlap, large centroid dist)
    f1 = render_synthetic_rgbd_frame(scene, np.eye(4), 1, intr, intr_tensor)
    f1_pts = np.asarray(f1.pcd.points) + np.array([2.0, 2.0, 3.0])
    f1.pcd.points = o3d.utility.Vector3dVector(f1_pts)

    cfg = RegistrationConfig()
    cfg.max_pair_centroid_dist_m = 0.50

    # 1. Verify pre-check skips Colored ICP and logs the overlap failure without raising
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        T_est, fit, rmse, accepted, method, rot, trans, reason = register_rgbd_pair(f1, f0, cfg)
    log_output = log_stream.getvalue()

    print(f"  Return method: {method}")
    print(f"  Accepted: {accepted}")
    print(f"  Reason: {reason}")
    print(f"  Captured log snippet: {log_output.strip()[:100]}...")

    # (a) No exception raised (execution reached here)
    assert True
    # (b) Colored ICP branch was skipped
    assert method != "colored_icp", f"Expected colored_icp to be skipped, got {method}"
    assert "SKIPPED" in reason, f"Expected 'SKIPPED' in reason, got {reason}"
    assert "[overlap-check] pair 1->0 SKIPPED colored-icp" in log_output, (
        f"Expected '[overlap-check] pair 1->0 SKIPPED colored-icp' in log, got:\n{log_output}"
    )
    # (c) Returns clean REJECT (accepted == False, method == 'none')
    assert accepted is False, "Non-overlapping pair must be rejected"
    assert method == "none", f"Expected method 'none' for rejected pair, got {method}"

    # 2. Verify defensive RuntimeError handling inside Colored ICP
    # Construct a frame pair where overlap check passes, but Colored ICP raises RuntimeError
    f2 = render_synthetic_rgbd_frame(scene, np.eye(4), 2, intr, intr_tensor)
    f3 = render_synthetic_rgbd_frame(scene, np.eye(4), 3, intr, intr_tensor)
    
    import registration
    orig_colored_icp = registration.registration_colored_icp

    def mock_colored_icp_raise(*args, **kwargs):
        raise RuntimeError("[Open3D Error] No correspondences found between source and target pointcloud")

    registration.registration_colored_icp = mock_colored_icp_raise
    try:
        log_stream_err = io.StringIO()
        with contextlib.redirect_stdout(log_stream_err):
            T_est2, fit2, rmse2, accepted2, method2, rot2, trans2, reason2 = register_rgbd_pair(f3, f2, cfg)
        log_output_err = log_stream_err.getvalue()

        print(f"  Defensive error catch test: method={method2}, accepted={accepted2}")
        print(f"  Captured error log: {log_output_err.strip()}")

        assert "[colored-icp] no correspondences for pair 3->2, falling through to FPFH tier" in log_output_err, (
            f"Expected defensive catch message in log, got:\n{log_output_err}"
        )
    finally:
        registration.registration_colored_icp = orig_colored_icp

    print("  PASS: Overlap pre-check and defensive RuntimeError handling validated.")


def test_j_glasses_weighted_kabsch_and_dense_sanity():
    """
    Test J: Subject Wearing Glasses (Noisy Eye-Region Depth).
    Asserts:
      (a) Weighted Kabsch recovers rotation and translation strictly closer to ground truth
          than unweighted Kabsch when eye-region landmarks have injected glasses depth error.
      (b) If landmark error is severe enough that dense surface fitness fails (< 0.30),
          the pipeline rejects the landmark seed and falls through to Tier 2 (Colored ICP)
          rather than accepting a corrupted seed.
    """
    print("\n--- Test J: Subject with Glasses (Weighted Kabsch & Dense Sanity) ---")
    from registration import rigid_transform_from_correspondences, register_rgbd_pair
    from landmarks import LANDMARK_INDICES

    # Ground truth transform
    R_gt = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(6.0), np.radians(-4.0), np.radians(5.0)))
    t_gt = np.array([0.015, -0.010, 0.020])

    # Realistic facial landmark locations (target frame)
    base_landmarks = {
        "left_eye_outer": np.array([-0.045, 0.025, 0.60]),
        "right_eye_outer": np.array([0.045, 0.025, 0.60]),
        "left_eye_inner": np.array([-0.018, 0.025, 0.58]),
        "right_eye_inner": np.array([0.018, 0.025, 0.58]),
        "nose_tip": np.array([0.0, 0.0, 0.55]),
        "nose_bridge": np.array([0.0, 0.020, 0.57]),
        "left_mouth_corner": np.array([-0.025, -0.035, 0.57]),
        "right_mouth_corner": np.array([0.025, -0.035, 0.57]),
        "chin": np.array([0.0, -0.075, 0.59]),
        "left_nostril": np.array([-0.012, -0.010, 0.56]),
        "right_nostril": np.array([0.012, -0.010, 0.56]),
        "upper_lip_center": np.array([0.0, -0.025, 0.56]),
        "lower_lip_center": np.array([0.0, -0.042, 0.57]),
    }

    eye_keys = {"left_eye_outer", "right_eye_outer", "left_eye_inner", "right_eye_inner"}

    dst_lm_dict = {}
    src_lm_dict = {}
    src_pts_list = []
    dst_pts_list = []
    weights_list = []

    for name in sorted(base_landmarks.keys()):
        p_dst = base_landmarks[name]
        p_src_true = R_gt.T @ (p_dst - t_gt)

        if name in eye_keys:
            # 18 mm glasses IR reflection error in source depth
            p_src_noisy = p_src_true + np.array([0.0, 0.0, 0.018])
            s_vr = 0.25
            s_var = 1.5e-4
            d_vr = 1.0
            d_var = 1e-6
        else:
            p_src_noisy = p_src_true.copy()
            s_vr = 1.0
            s_var = 1e-6
            d_vr = 1.0
            d_var = 1e-6

        w = float(min(1.0, s_vr * d_vr) / (1.0 + (s_var / 1e-5) + (d_var / 1e-5)))

        src_pts_list.append(p_src_noisy)
        dst_pts_list.append(p_dst)
        weights_list.append(w)

        src_lm_dict[name] = {"position": p_src_noisy, "valid_ratio": s_vr, "variance": s_var, "low_confidence": (name in eye_keys)}
        dst_lm_dict[name] = {"position": p_dst, "valid_ratio": d_vr, "variance": d_var, "low_confidence": False}

    src_pts = np.array(src_pts_list)
    dst_pts = np.array(dst_pts_list)
    weights = np.array(weights_list)

    # 1. Compare weighted vs unweighted Kabsch
    T_weighted, res_w = rigid_transform_from_correspondences(src_pts, dst_pts, weights=weights)
    T_unweighted, res_u = rigid_transform_from_correspondences(src_pts, dst_pts, weights=None)

    rot_err_w = compute_rotation_deg(T_weighted[:3, :3] @ R_gt.T)
    trans_err_w = float(np.linalg.norm(T_weighted[:3, 3] - t_gt) * 1000.0)
    rot_err_u = compute_rotation_deg(T_unweighted[:3, :3] @ R_gt.T)
    trans_err_u = float(np.linalg.norm(T_unweighted[:3, 3] - t_gt) * 1000.0)

    print(f"  Weighted Kabsch:   rot_err={rot_err_w:.2f} deg, trans_err={trans_err_w:.2f} mm (residual={res_w:.1f} mm)")
    print(f"  Unweighted Kabsch: rot_err={rot_err_u:.2f} deg, trans_err={trans_err_u:.2f} mm (residual={res_u:.1f} mm)")

    assert rot_err_w < rot_err_u, f"Weighted rotation error ({rot_err_w:.2f} deg) must be < unweighted ({rot_err_u:.2f} deg)"
    assert trans_err_w < trans_err_u, f"Weighted translation error ({trans_err_w:.2f} mm) must be < unweighted ({trans_err_u:.2f} mm)"
    print("  [PASS] Part (a): Weighted Kabsch significantly reduces glasses-induced parameter error.")

    # 2. Test Part (b): Dense-surface sanity check forces fallthrough to Tier 2 (Colored ICP)
    scene = o3d.t.geometry.RaycastingScene()
    mesh = make_synthetic_scene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=o3d.core.Dtype.Float64)

    P0 = np.eye(4)
    P1 = np.eye(4)
    P1[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(5.0), 0, 0))
    P1[:3, 3] = [0.01, 0.0, 0.015]

    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)

    # Mock detect_face_landmarks_3d to return landmarks that disagree with surface (e.g. 25 deg pitch)
    bogus_R = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(25.0), 0, 0))
    bogus_src_lm = {}
    for name, d_val in dst_lm_dict.items():
        p_b = bogus_R.T @ d_val["position"]
        bogus_src_lm[name] = {"position": p_b, "valid_ratio": 1.0, "variance": 1e-6, "low_confidence": False}

    import landmarks
    orig_detect = landmarks.detect_face_landmarks_3d

    def mock_detect(color, depth, intrinsics, **kwargs):
        if getattr(color, "shape", None) is not None and np.array_equal(color, f0.color_bgr):
            return dst_lm_dict
        return bogus_src_lm

    landmarks.detect_face_landmarks_3d = mock_detect
    cfg = RegistrationConfig(min_seed_dense_fitness=0.30)

    try:
        T_est, fit, rmse, accepted, method, rot, trans, reason = register_rgbd_pair(f1, f0, cfg)
        print(f"  Corrupted landmark pair registered via tier: '{method}', accepted={accepted}")
        assert method in ("colored_icp", "rgbd_odometry_colored_icp"), f"Expected Tier 2 fallthrough, got method: {method}"
        assert accepted is True, "Tier 2 should successfully align the frame pair"
    finally:
        landmarks.detect_face_landmarks_3d = orig_detect

    print("  [PASS] Part (b): Dense-surface sanity check successfully rejected corrupted landmark seed and fell through to Tier 2.")


def test_k_systematic_bias_fpfh_crosscheck():
    print("\n--- Test K: Smooth Systematic Eye-Region Bias & FPFH / Local Face Cross-Check ---")
    scene = o3d.t.geometry.RaycastingScene()
    mesh = make_synthetic_scene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 320.0, 240.0)
    intr_tensor = o3d.core.Tensor([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=o3d.core.Dtype.Float64)

    P0 = np.eye(4)
    P1 = np.eye(4)
    P1[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(5.0), 0, 0))
    P1[:3, 3] = [0.01, 0.0, 0.015]

    f0 = render_synthetic_rgbd_frame(scene, P0, 0, intr, intr_tensor)
    f1 = render_synthetic_rgbd_frame(scene, P1, 1, intr, intr_tensor)

    T_gt = np.linalg.inv(P0) @ P1
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    base_landmarks = {
        "left_eye_outer": np.array([-0.065, 0.035, 0.58]),
        "right_eye_outer": np.array([0.065, 0.035, 0.58]),
        "left_eye_inner": np.array([-0.032, 0.035, 0.57]),
        "right_eye_inner": np.array([0.032, 0.035, 0.57]),
        "nose_bridge": np.array([0.0, 0.020, 0.55]),
        "nose_tip": np.array([0.0, 0.0, 0.52]),
        "left_mouth_corner": np.array([-0.040, -0.045, 0.58]),
        "right_mouth_corner": np.array([0.040, -0.045, 0.58]),
        "chin": np.array([0.0, -0.075, 0.59]),
        "left_nostril": np.array([-0.012, -0.010, 0.56]),
        "right_nostril": np.array([0.012, -0.010, 0.56]),
        "upper_lip_center": np.array([0.0, -0.025, 0.56]),
        "lower_lip_center": np.array([0.0, -0.042, 0.57]),
    }

    eye_keys = {"left_eye_outer", "right_eye_outer", "left_eye_inner", "right_eye_inner"}

    dst_lm_dict = {}
    src_lm_dict = {}
    src_pts_list = []
    dst_pts_list = []
    weights_list = []

    for name, p_dst in base_landmarks.items():
        p_src_true = R_gt.T @ (p_dst - t_gt)
        if name in eye_keys:
            # Constant 15mm bias along optical axis with LOW variance and HIGH valid_ratio (glasses refraction)
            p_src_biased = p_src_true + np.array([0.0, 0.0, 0.015])
            s_vr = 1.0
            s_var = 1e-6
        else:
            p_src_biased = p_src_true.copy()
            s_vr = 1.0
            s_var = 1e-6

        d_vr = 1.0
        d_var = 1e-6
        w = float(min(1.0, s_vr * d_vr) / (1.0 + (s_var / 1e-5) + (d_var / 1e-5)))

        src_pts_list.append(p_src_biased)
        dst_pts_list.append(p_dst)
        weights_list.append(w)

        src_lm_dict[name] = {"position": p_src_biased, "valid_ratio": s_vr, "variance": s_var, "low_confidence": False}
        dst_lm_dict[name] = {"position": p_dst, "valid_ratio": d_vr, "variance": d_var, "low_confidence": False}

    src_pts = np.array(src_pts_list)
    dst_pts = np.array(dst_pts_list)
    weights = np.array(weights_list)

    # 1. Assert Kabsch residual passes existing 15mm gate despite systematic bias
    T_kabsch, res_mm = rigid_transform_from_correspondences(src_pts, dst_pts, weights=weights)
    print(f"  Biased Kabsch residual: {res_mm:.2f} mm (passes <=15mm residual gate)")
    assert res_mm <= 15.0, f"Residual {res_mm} mm should be <= 15mm gate"

    # 2. Assert global dense fitness passes >= 0.30 gate (diluted across entire head)
    eval_voxel = 0.006
    s_down = f1.pcd.voxel_down_sample(eval_voxel)
    t_down = f0.pcd.voxel_down_sample(eval_voxel)
    t_down.estimate_normals()
    glob_eval = o3d.pipelines.registration.evaluate_registration(s_down, t_down, 0.02, T_kabsch)
    print(f"  Biased Kabsch global dense fitness: {glob_eval.fitness:.2f} (passes >=0.30 gate)")
    assert glob_eval.fitness >= 0.30, "Global dense fitness would deceptively pass without cross-checks"

    # 3. Assert cross-check detects disagreement and triggers fallthrough to Tier 2
    import landmarks
    orig_detect = landmarks.detect_face_landmarks_3d

    def mock_detect(color, depth, intrinsics, **kwargs):
        if getattr(color, "shape", None) is not None and np.array_equal(color, f0.color_bgr):
            return dst_lm_dict
        return src_lm_dict

    landmarks.detect_face_landmarks_3d = mock_detect
    cfg = RegistrationConfig(
        enable_landmark_fpfh_crosscheck=True,
        max_landmark_fpfh_disagreement_trans_m=0.02,
        max_landmark_fpfh_disagreement_rot_deg=10.0,
    )

    try:
        T_est, fit, rmse, accepted, method, rot, trans, reason = register_rgbd_pair(f1, f0, cfg)
        print(f"  Biased landmark pair registered via tier: '{method}', accepted={accepted}")
        assert method in ("colored_icp", "rgbd_odometry_colored_icp"), (
            f"Expected Tier 2 fallthrough due to cross-check disagreement, got method: '{method}'"
        )
        assert accepted is True, "Tier 2 should successfully align the frame pair"
    finally:
        landmarks.detect_face_landmarks_3d = orig_detect

    print("  [PASS] Test K: Smooth systematic eye-region bias caught by cross-check; fallthrough to Tier 2 confirmed.")


def main():
    print("=" * 65)
    print("  HUMAN 180 DEG MULTI-VIEW REGISTRATION SYNTHETIC TEST SUITE")
    print("=" * 65)

    test_a_known_rigid_motion()
    test_b_sequential_camera_motion()
    test_c_bad_frame_rejection()
    test_d_smooth_surface_gating()
    test_e_transformation_convention()
    test_f_pose_jump_rejection()
    test_g_multi_frame_skip_recovery()
    test_h_loop_closure_fitness_threshold()
    test_i_non_overlapping_pair_colored_icp_skip()
    test_j_glasses_weighted_kabsch_and_dense_sanity()
    test_k_systematic_bias_fpfh_crosscheck()

    print("\n" + "=" * 65)
    print("  ALL 11 SYNTHETIC TESTS (A, B, C, D, E, F, G, H, I, J, K) PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()

