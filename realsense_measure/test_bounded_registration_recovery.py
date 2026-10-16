"""
Deterministic Unit and Regression Tests for Problem 7A Bounded Registration Recovery.

Validates all 12 required scenarios from Section 6:
1. Valid independent bridge cross-check.
2. Translation disagreement beyond 20 mm.
3. Rotation disagreement beyond 10 degrees.
4. Unanchored chain remains fusion-ineligible.
5. Valid loop closure promotes an unanchored chain only after all promotion conditions pass.
6. Loop closure fails or is pruned, so fusion quarantine remains.
7. Corrupted transform within the kinematic envelope but inconsistent with the independent transform.
8. Two-frame recovery horizon and third-failure handling.
9. Broken local link prevents reuse of stale transforms.
10. Buffer clearing after successful direct registration.
11. The six-frame profile-transition scenario, with exactly two fusion-eligible frames when no closure exists.
12. Bridge-edge uncertainty and hop-based information weighting.
"""

from __future__ import annotations

import copy
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from config import RegistrationConfig
import registration
from registration import (
    build_pose_graph_rgbd,
    register_sequence_rgbd_human,
    optimize_pose_graph,
    compute_rotation_deg,
    compute_translation_mm,
    get_information_matrix,
)


def make_dummy_frame(frame_id: int, center_y: float = 0.1, n_pts: int = 1500) -> RGBDFrame:
    """Create a lightweight synthetic RGBDFrame with valid point cloud and normals."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(n_pts)
    pts = np.asarray(pcd.points)
    pts[:, 1] += center_y
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    color_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    depth_m = np.ones((48, 64), dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(64, 48, 50.0, 50.0, 32.0, 24.0)

    return RGBDFrame(
        frame_id=frame_id,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=pcd,
        timestamp=float(frame_id) * 0.033,
        intrinsics=intr,
    )


# ---------------------------------------------------------------------------
# Test 1: Valid independent bridge cross-check (<= 20 mm, <= 10 deg)
# ---------------------------------------------------------------------------
def test_1_valid_independent_bridge_crosscheck():
    frames = [make_dummy_frame(i) for i in range(3)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    # Frame 0 is anchor.
    # Frame 1: fails global against 0, succeeds local to 0 -> PROVISIONAL_BRIDGE.
    # T(1 -> 0) = translation [0.010, 0, 0]
    T_1_to_0 = np.eye(4)
    T_1_to_0[0, 3] = 0.010

    # Frame 2:
    # Direct to frame 0: T(2 -> 0) = translation [0.022, 0, 0]
    T_2_to_0_dir = np.eye(4)
    T_2_to_0_dir[0, 3] = 0.022

    # Local to frame 1: T(2 -> 1) = translation [0.011, 0, 0]
    # Composed: T(1->0) @ T(2->1) = [0.021, 0, 0]
    # Disagreement with direct: |0.022 - 0.021| = 1 mm <= 20 mm.
    T_2_to_1_loc = np.eye(4)
    T_2_to_1_loc[0, 3] = 0.011

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            if not is_bridge:
                return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "global fail"
            return T_1_to_0, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "bridge ok"
        if s == 2 and d == 0:
            return T_2_to_0_dir, 0.85, 0.002, True, "colored_icp", 0.0, 22.0, "direct ok"
        if s == 2 and d == 1:
            return T_2_to_1_loc, 0.82, 0.002, True, "colored_icp", 0.0, 11.0, "crosscheck ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    assert stats["frame_states"][1] == "PROMOTED_BRIDGE"
    assert stats["frame_states"][2] == "TERMINAL_RECOVERED"
    assert diags[1][2] is False  # primary acceptance remains False for bridge
    assert diags[2][2] is True   # terminal recovered is accepted
    assert stats["n_nodes"] == 3


# ---------------------------------------------------------------------------
# Test 2: Translation disagreement beyond 20 mm
# ---------------------------------------------------------------------------
def test_2_translation_disagreement_beyond_20mm():
    frames = [make_dummy_frame(i) for i in range(3)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    T_1_to_0 = np.eye(4)
    T_1_to_0[0, 3] = 0.010

    # Direct: [0.015, 0, 0]
    T_2_to_0_dir = np.eye(4)
    T_2_to_0_dir[0, 3] = 0.015

    # Local: [0.035, 0, 0] -> Composed: [0.045, 0, 0]
    # Disagreement: |0.045 - 0.015| = 30 mm > 20 mm
    T_2_to_1_loc = np.eye(4)
    T_2_to_1_loc[0, 3] = 0.035

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            if not is_bridge:
                return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "global fail"
            return T_1_to_0, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "bridge ok"
        if s == 2 and d == 0:
            return T_2_to_0_dir, 0.85, 0.002, True, "colored_icp", 0.0, 15.0, "direct ok"
        if s == 2 and d == 1:
            return T_2_to_1_loc, 0.82, 0.002, True, "colored_icp", 0.0, 35.0, "crosscheck"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Disagreement > 20 mm rejects recovery and frame 2
    assert stats["frame_states"][2] == "REJECTED"
    assert diags[2][2] is False
    assert "cross-check failed" in stats["rejection_reasons"][2]


# ---------------------------------------------------------------------------
# Test 3: Rotation disagreement beyond 10 degrees
# ---------------------------------------------------------------------------
def test_3_rotation_disagreement_beyond_10deg():
    frames = [make_dummy_frame(i) for i in range(3)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    T_1_to_0 = np.eye(4)

    T_2_to_0_dir = np.eye(4)

    # 15 degrees rotation around Y axis
    th = np.deg2rad(15.0)
    R = np.array([
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)],
    ])
    T_2_to_1_loc = np.eye(4)
    T_2_to_1_loc[:3, :3] = R

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            if not is_bridge:
                return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "global fail"
            return T_1_to_0, 0.80, 0.002, True, "colored_icp", 0.0, 0.0, "bridge ok"
        if s == 2 and d == 0:
            return T_2_to_0_dir, 0.85, 0.002, True, "colored_icp", 0.0, 0.0, "direct ok"
        if s == 2 and d == 1:
            return T_2_to_1_loc, 0.82, 0.002, True, "colored_icp", 15.0, 0.0, "crosscheck"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    assert stats["frame_states"][2] == "REJECTED"
    assert diags[2][2] is False
    assert "rot 15.0deg > 10.0deg" in stats["rejection_reasons"][2]


# ---------------------------------------------------------------------------
# Test 4: Unanchored chain remains fusion-ineligible without loop closure
# ---------------------------------------------------------------------------
def test_4_unanchored_chain_remains_fusion_ineligible():
    frames = [make_dummy_frame(i) for i in range(5)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    # 0 accepted
    # 1, 2 fail global, succeed local -> PROVISIONAL then PROMOTED_BRIDGE when buffer exhausted
    # 3, 4 unanchored tracking odometry
    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 4 and d == 3:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    assert stats["frame_states"][3] == "UNANCHORED_ODOMETRY"
    assert stats["frame_states"][4] == "UNANCHORED_ODOMETRY"
    # Both unanchored frames must remain quarantined from fusion
    assert stats["fusion_eligible"][3] is False
    assert stats["fusion_eligible"][4] is False
    assert stats["n_fusion_eligible"] == 1  # only frame 0 is fusion eligible


# ---------------------------------------------------------------------------
# Test 5: Valid loop closure promotes unanchored chain after all checks pass
# ---------------------------------------------------------------------------
def test_5_valid_loop_closure_promotes_unanchored_chain():
    frames = [make_dummy_frame(i) for i in range(4)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    step = np.eye(4)
    step[0, 3] = 0.010

    # Frames 1, 2 form bridge
    # Frame 3 is unanchored odometry that ALSO closes loop to frame 0 with consistent geometry
    # Poses:
    # Frame 0: origin
    # Frame 1: x = 0.010
    # Frame 2: x = 0.020
    # Frame 3: x = 0.030 (relative to 2: step)
    # Loop from 3 to 0: T(3->0) has x = 0.030. Consistent with odometry!
    T_3_to_0 = np.eye(4)
    T_3_to_0[0, 3] = 0.030

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 0:
            # Loop closure edge to anchor 0
            return T_3_to_0, 0.75, 0.002, True, "colored_icp", 0.0, 30.0, "loop closure"
        if s == 3 and d == 2:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Unanchored chain was connected via frame 3's survived loop closure
    # Both frame 0 and frame 3 should be fusion eligible
    assert stats["fusion_eligible"][0] is True
    assert stats["fusion_eligible"][3] is True


# ---------------------------------------------------------------------------
# Test 6: Loop closure fails or is pruned, so fusion quarantine remains
# ---------------------------------------------------------------------------
def test_6_pruned_loop_closure_preserves_fusion_quarantine():
    frames = [make_dummy_frame(i) for i in range(4)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    step = np.eye(4)
    step[0, 3] = 0.010

    # Frame 3 has an inconsistent / corrupted loop closure to frame 0
    # True pose is x = 0.030, but loop claims x = 0.250 (25 cm off!).
    # In optimize_pose_graph with prune threshold 0.25, this edge will be pruned!
    T_3_to_0_bad = np.eye(4)
    T_3_to_0_bad[0, 3] = 0.250

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 0:
            return T_3_to_0_bad, 0.70, 0.003, True, "colored_icp", 0.0, 250.0, "bad loop"
        if s == 3 and d == 2:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Edge was pruned in optimizer; unanchored chain remains quarantined
    assert stats["fusion_eligible"][3] is False


# ---------------------------------------------------------------------------
# Test 7: Corrupted transform within kinematic envelope but inconsistent
# ---------------------------------------------------------------------------
def test_7_corrupted_transform_within_kinematic_envelope():
    frames = [make_dummy_frame(i) for i in range(3)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    T_1_to_0 = np.eye(4)
    T_1_to_0[0, 3] = 0.010

    # Direct transform to anchor: 15 mm motion (well within 450 mm kinematic jump)
    T_2_to_0_dir = np.eye(4)
    T_2_to_0_dir[0, 3] = 0.015

    # Local transform gives 45 mm motion (also well within 450 mm kinematic jump!)
    # But independent crosscheck disagreement is |45 - 15| = 30 mm > 20 mm!
    T_2_to_1_loc = np.eye(4)
    T_2_to_1_loc[0, 3] = 0.035  # Composed = 0.010 + 0.035 = 0.045 m

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            if not is_bridge:
                return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "global fail"
            return T_1_to_0, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "bridge ok"
        if s == 2 and d == 0:
            return T_2_to_0_dir, 0.85, 0.002, True, "colored_icp", 0.0, 15.0, "direct ok"
        if s == 2 and d == 1:
            return T_2_to_1_loc, 0.82, 0.002, True, "colored_icp", 0.0, 35.0, "local"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Rejected by independent crosscheck despite passing kinematic jump
    assert stats["frame_states"][2] == "REJECTED"
    assert diags[2][2] is False


# ---------------------------------------------------------------------------
# Test 8: Two-frame recovery horizon and third-failure handling
# ---------------------------------------------------------------------------
def test_8_two_frame_recovery_horizon():
    # 5 frames: 0 accepted, 1 and 2 fill buffer (size 2), 3 exhausts horizon without overlap
    frames = [make_dummy_frame(i) for i in range(4)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True
    cfg.max_bridge_buffer_size = 2

    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Frames 1 and 2 were promoted as bridge nodes when buffer reached max capacity 2
    assert stats["frame_states"][1] == "PROMOTED_BRIDGE"
    assert stats["frame_states"][2] == "PROMOTED_BRIDGE"
    # Frame 3 became unanchored odometry
    assert stats["frame_states"][3] == "UNANCHORED_ODOMETRY"


# ---------------------------------------------------------------------------
# Test 9: Broken local link prevents reuse of stale transforms
# ---------------------------------------------------------------------------
def test_9_broken_local_link_prevents_stale_reuse():
    # Frame 0 accepted.
    # Frame 1 fails global, succeeds local -> buffer has [1].
    # Frame 2 fails global AND fails local link to 1!
    # Buffer must be cleared immediately. Frame 3 must not link to frame 1!
    frames = [make_dummy_frame(i) for i in range(4)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    attempted_targets = {i: [] for i in range(4)}
    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        attempted_targets[s].append(d)
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            # Broken local link!
            return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "broken link"
        if s == 3 and d == 0:
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Frame 2 local link broke, so frame 2 is REJECTED
    assert stats["frame_states"][2] == "REJECTED"
    # Frame 3 must only attempt accepted frame 0, never stale frame 1
    assert 1 not in attempted_targets[3]


# ---------------------------------------------------------------------------
# Test 10: Buffer clearing after successful direct registration
# ---------------------------------------------------------------------------
def test_10_buffer_clearing_after_direct_registration():
    frames = [make_dummy_frame(i) for i in range(3)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 0:
            # Direct registration succeeds!
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 2 and d == 1:
            # Local crosscheck fails (no local overlap)
            return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "no crosscheck"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Direct registration succeeded, buffer cleared
    assert stats["frame_states"][2] == "DIRECT_ACCEPTED"
    assert diags[2][2] is True


# ---------------------------------------------------------------------------
# Test 11: The six-frame profile-transition scenario
# ---------------------------------------------------------------------------
def test_11_six_frame_profile_transition_scenario():
    frames = [make_dummy_frame(i) for i in range(6)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        # Frames 0 and 1 accepted
        if s == 1 and d == 0:
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "direct ok"
        # Frames 2 and 3 fail global, succeed local
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        # Frames 4 and 5 succeed local odometry
        if s == 4 and d == 3:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 5 and d == 4:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Exactly 2 fusion-eligible frames (0 and 1)
    assert stats["n_fusion_eligible"] == 2
    assert stats["fusion_eligible"][0] is True
    assert stats["fusion_eligible"][1] is True
    assert stats["fusion_eligible"][2] is False
    assert stats["fusion_eligible"][3] is False
    assert stats["fusion_eligible"][4] is False
    assert stats["fusion_eligible"][5] is False

    # Acceptance semantics
    assert stats["accepted"] == 4  # 0, 1, 4, 5
    assert stats["rejected"] == 2  # 2, 3
    assert stats["n_nodes"] == 6
    assert len(stats["node_of"]) == 6


# ---------------------------------------------------------------------------
# Test 12: Bridge-edge uncertainty and hop-based information weighting
# ---------------------------------------------------------------------------
def test_12_bridge_edge_uncertainty_and_information_weighting():
    frames = [make_dummy_frame(i) for i in range(4)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True

    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        pg_res = build_pose_graph_rgbd(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    pose_graph = pg_res[0]
    bridge_edges = pg_res.bridge_edges

    # We expect bridge edges for (1 -> 0, hop=1), (2 -> 1, hop=2), (3 -> 2, hop=3)
    assert len(bridge_edges) >= 2
    for src, tgt, hop in bridge_edges:
        # Find the edge in pose_graph
        matching_edges = [
            e for e in pose_graph.edges
            if (e.source_node_id == pg_res[1][src] and e.target_node_id == pg_res[1][tgt])
        ]
        assert len(matching_edges) == 1
        edge = matching_edges[0]
        # Must be marked uncertain
        assert edge.uncertain is True

        # Base info matrix without scaling
        base_info = get_information_matrix(frames[src].pcd, frames[tgt].pcd, edge.transformation, cfg)
        expected_scaled_info = base_info * (1.0 / (1.0 + hop))
        np.testing.assert_allclose(edge.information, expected_scaled_info, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 13: Broken unanchored chain prevents stale skip and starts new chain
# ---------------------------------------------------------------------------
def test_13_broken_unanchored_chain_prevents_stale_skip_and_starts_new_chain():
    frames = [make_dummy_frame(i) for i in range(7)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True
    cfg.max_bridge_buffer_size = 2

    step = np.eye(4)
    step[0, 3] = 0.010

    attempted_targets = {i: [] for i in range(7)}

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        attempted_targets[s].append(d)
        if s == 1 and d == 0:
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 4 and d == 3:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 5 and d == 4:
            # Broken local link
            return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "broken local link"
        if s == 6 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Frame 6 must not attempt broken unanchored frame 4 or failed frame 5
    assert 4 not in attempted_targets[6], f"Frame 6 must not attempt broken unanchored frame 4, attempted: {attempted_targets[6]}"
    assert 5 not in attempted_targets[6], f"Frame 6 must not attempt broken frame 5, attempted: {attempted_targets[6]}"
    assert attempted_targets[6] == [1, 0, 1], f"Frame 6 should attempt accepted frames [1, 0] then fallback anchor 1, got {attempted_targets[6]}"

    # State transitions
    assert stats["frame_states"][4] == "UNANCHORED_ODOMETRY"
    assert stats["frame_states"][5] == "REJECTED"
    assert stats["frame_states"][6] == "PROVISIONAL_BRIDGE"

    # Fusion quarantine
    assert stats["fusion_eligible"][4] is False
    assert stats["fusion_eligible"][5] is False
    assert stats["fusion_eligible"][6] is False


# ---------------------------------------------------------------------------
# Test 14: Broken unanchored chain cannot be promoted by later loop closure
# ---------------------------------------------------------------------------
def test_14_broken_unanchored_chain_cannot_be_promoted_by_later_loop_closure():
    frames = [make_dummy_frame(i) for i in range(7)]
    cfg = RegistrationConfig()
    cfg.enable_bounded_recovery = True
    cfg.max_bridge_buffer_size = 2

    step = np.eye(4)
    step[0, 3] = 0.010

    def mock_register(src, dst, cfg_arg, initial_guess=None, is_bridge=False):
        s, d = src.frame_id, dst.frame_id
        if s == 1 and d == 0:
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 2 and d == 1:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 3 and d == 2:
            return (step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok") if is_bridge else (np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail")
        if s == 4 and d == 3:
            return step, 0.80, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        if s == 5 and d == 4:
            # Broken local link
            return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "broken"
        if s == 6 and d == 0:
            # Frame 6 direct registers to anchor 0
            return step, 0.85, 0.002, True, "colored_icp", 0.0, 10.0, "ok"
        return np.eye(4), 0.0, 999.0, False, "none", 0.0, 0.0, "fail"

    orig = registration.register_rgbd_pair
    registration.register_rgbd_pair = mock_register
    try:
        aligned, diags, poses, stats = register_sequence_rgbd_human(frames, cfg)
    finally:
        registration.register_rgbd_pair = orig

    # Frame 4 was in unanchored chain that broke at frame 5
    # Even though frame 6 succeeded direct to frame 0, frame 4 must remain quarantined from fusion
    assert stats["fusion_eligible"][4] is False
    assert stats["fusion_eligible"][0] is True
    assert stats["fusion_eligible"][1] is True
    assert stats["fusion_eligible"][6] is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
