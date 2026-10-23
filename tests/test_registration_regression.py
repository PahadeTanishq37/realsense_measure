"""
Regression test for the registration stage (no camera needed).

Builds one asymmetric cloud, makes frames from it with small KNOWN motions,
and checks that every registration path recovers them.  Also checks that a
junk frame (unrelated flat plane) is REJECTED by the gating instead of
corrupting the result.

Run from the project folder:   python test_registration_regression.py
"""
import contextlib
import copy
import io
import logging
import sys
from unittest.mock import patch

import numpy as np
import open3d as o3d

from config import RegistrationConfig
import registration as reg

TOL_MM = 3.0


def make_asymmetric_cloud() -> o3d.geometry.PointCloud:
    a = o3d.geometry.TriangleMesh.create_box(0.30, 0.20, 0.12)
    b = o3d.geometry.TriangleMesh.create_box(0.10, 0.08, 0.10)
    b.translate([0.30, 0.0, 0.0])
    c = o3d.geometry.TriangleMesh.create_box(0.06, 0.06, 0.15)
    c.translate([0.05, 0.20, 0.0])
    mesh = a + b + c
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(8000).voxel_down_sample(0.004)


def make_junk_plane() -> o3d.geometry.PointCloud:
    pts = np.random.uniform(-0.3, 0.3, size=(3000, 3))
    pts[:, 2] = 0.0
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    return p


def make_frames(base, n=6, with_junk_at=None):
    """frame_k = base moved by the inverse of a known cumulative pose G_k."""
    frames, Gk = [], np.eye(4)
    axis = np.array([0.3, 1.0, 0.2]); axis /= np.linalg.norm(axis)
    for k in range(n):
        if k > 0:
            step = np.eye(4)
            step[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(5.0))
            step[:3, 3] = [0.005, -0.003, 0.002]
            Gk = Gk @ step
        frames.append(copy.deepcopy(base).transform(np.linalg.inv(Gk)))
    if with_junk_at is not None:
        frames.insert(with_junk_at, make_junk_plane())
    return frames


def mean_nn_mm(a, ref):
    return float(np.mean(np.asarray(a.compute_point_cloud_distance(ref))) * 1000)


def run(fn, frames, cfg):
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(frames, cfg)


def main() -> int:
    np.random.seed(1)
    base = make_asymmetric_cloud()
    cfg = RegistrationConfig(ransac_max_iter=100_000)
    failures = []

    paths = [("sequential", reg.register_sequence),
             ("multiway", reg.register_sequence_multiway)]

    # 1) small known motions must be recovered by every path
    frames = make_frames(base)
    for name, fn in paths:
        aligned, diag = run(fn, frames, cfg)
        errs = [mean_nn_mm(a, aligned[0]) for a in aligned[1:]]
        ok = max(errs) < TOL_MM and all(d[2] for d in diag)
        print(f"[known motion] {name:10s} errors(mm)={np.round(errs, 1)}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"known motion / {name}")

    # 2) a junk frame must be rejected and must not damage the others
    junk_at = 3
    frames = make_frames(base, with_junk_at=junk_at)
    for name, fn in paths:
        aligned, diag = run(fn, frames, cfg)
        rejected = not diag[junk_at][2]
        good = [i for i in range(1, len(frames)) if i != junk_at and diag[i][2]]
        errs = [mean_nn_mm(aligned[i], aligned[0]) for i in good]
        ok = rejected and len(good) == len(frames) - 2 and max(errs) < TOL_MM
        print(f"[junk frame  ] {name:10s} junk rejected={rejected}  others ok={max(errs) < TOL_MM if errs else False}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"junk frame / {name}")

    # 3) pose-jump bounds scaling for human pose-graph (TASK 3)
    from camera.frame import RGBDFrame

    cfg_jump = RegistrationConfig()
    cfg_jump.max_pose_jump_translation_m = 0.15
    cfg_jump.max_pose_jump_rotation_deg = 45.0

    flat_bound_t = cfg_jump.max_pose_jump_translation_m
    flat_bound_r = cfg_jump.max_pose_jump_rotation_deg

    # Frame N occurs 5 frames after the last accepted frame (frame 0), i.e. 4 intermediate rejected frames
    frames_since_accepted_step = 4
    eff_bound_t = min(
        cfg_jump.max_pose_jump_translation_m * (1 + frames_since_accepted_step),
        cfg_jump.max_pose_jump_translation_m * 6.0
    )
    eff_bound_r = min(
        cfg_jump.max_pose_jump_rotation_deg * (1 + frames_since_accepted_step),
        cfg_jump.max_pose_jump_rotation_deg * 6.0
    )

    jump_t = 0.40  # 0.4m
    jump_r = 50.0  # 50 deg

    # Assert scaled_bound > flat_bound
    assert eff_bound_t > flat_bound_t, f"eff_bound_t ({eff_bound_t}) must be > flat_bound_t ({flat_bound_t})"
    assert eff_bound_r > flat_bound_r, f"eff_bound_r ({eff_bound_r}) must be > flat_bound_r ({flat_bound_r})"

    # Assert it would have been REJECTED under flat bounds
    flat_passed = (jump_t <= flat_bound_t) and (jump_r <= flat_bound_r)
    assert not flat_passed, "0.4m / 50deg jump must fail flat bounds"

    # Assert it is ACCEPTED under scaled bounds
    scaled_passed = (jump_t <= eff_bound_t) and (jump_r <= eff_bound_r)
    assert scaled_passed, "0.4m / 50deg jump must pass scaled bounds"

    # Assert truly implausible jump (2m) is still REJECTED even with maximum scaling (6x cap)
    eff_bound_t_max = min(
        cfg_jump.max_pose_jump_translation_m * (1 + 10),
        cfg_jump.max_pose_jump_translation_m * 6.0
    )
    implausible_jump_t = 2.0
    cap_passed = (implausible_jump_t <= eff_bound_t_max)
    assert not cap_passed, "2.0m translation must be rejected under 6x max cap"

    # End-to-end simulation through build_pose_graph_rgbd
    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 600, 600, 320, 240)
    dummy_pcd = o3d.geometry.PointCloud()
    dummy_pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-0.1, 0.1, size=(50, 3)))
    sim_frames = [
        RGBDFrame(i, np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=np.float32), dummy_pcd, float(i), intr)
        for i in range(7)
    ]

    R_50 = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * np.deg2rad(50.0))
    T_large = np.eye(4)
    T_large[:3, :3] = R_50
    T_large[:3, 3] = [0.40, 0.0, 0.0]

    T_implausible = np.eye(4)
    T_implausible[0, 3] = 2.0

    def mock_register_pair(src, dst, c, initial_guess=None, **kwargs):
        if src.frame_id in (1, 2, 3, 4):
            return np.eye(4), 0.1, 0.1, False, "none", 0.0, 0.0, "low fitness"
        elif src.frame_id == 5:
            return T_large, 0.95, 0.001, True, "colored_icp", 50.0, 400.0, "fitness=0.95"
        elif src.frame_id == 6:
            return T_implausible, 0.95, 0.001, True, "colored_icp", 0.0, 2000.0, "fitness=0.95"
        return np.eye(4), 0.95, 0.001, True, "colored_icp", 0.0, 0.0, "fitness=0.95"

    orig_pair = reg.register_rgbd_pair
    reg.register_rgbd_pair = mock_register_pair
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pg, node_of, diag, n_odom, n_loop, pair_logs, poses, best_methods, best_scores, rej = reg.build_pose_graph_rgbd(sim_frames, cfg_jump)
    finally:
        reg.register_rgbd_pair = orig_pair

    f5_ok = diag[5] is not None and diag[5][2] is True
    f6_rejected = diag[6] is None or diag[6][2] is False

    ok_jump = (eff_bound_t > flat_bound_t) and not flat_passed and scaled_passed and not cap_passed and f5_ok and f6_rejected
    print(f"[pose jump   ] scaled bound {eff_bound_t:.2f}m > flat {flat_bound_t:.2f}m  frame5 accepted={f5_ok} (flat would reject={not flat_passed})  frame6(2m) rejected={f6_rejected}  {'PASS' if ok_jump else 'FAIL'}")
    if not ok_jump:
        failures.append("pose jump scaling")

    print("\nRESULT:", "PASS" if not failures else f"FAIL -> {failures}")
    return 0 if not failures else 1


# ===========================================================================
# Pytest Unit Tests for register_against_candidates (Fix #3)
# ===========================================================================

def test_register_against_candidates_fallback_on_exception(caplog):
    """
    Test register_against_candidates when candidate registration raises an exception (Fix #3).

    Asserts:
      (a) The exception is logged/visible as a warning per fix #3 ([fusion-fallback]).
      (b) The pipeline correctly falls through to the next candidate rather than
          silently returning a wrong or empty (None) result.
    """
    cfg = RegistrationConfig()

    # Create dummy point clouds with >= 10 points
    new_frame = o3d.geometry.PointCloud()
    new_frame.points = o3d.utility.Vector3dVector(np.ones((20, 3)))

    cand0 = o3d.geometry.PointCloud()
    cand0.points = o3d.utility.Vector3dVector(np.ones((20, 3)) * 0.1)

    cand1 = o3d.geometry.PointCloud()
    cand1.points = o3d.utility.Vector3dVector(np.ones((20, 3)) * 0.2)

    expected_transform = np.eye(4)
    expected_transform[0, 3] = 0.05
    expected_fitness = 0.88
    expected_rmse = 0.0015

    def mock_candidate_fn(src, dst, c):
        if dst is cand0:
            raise RuntimeError("Deliberate registration failure on candidate 0")
        elif dst is cand1:
            return expected_transform, expected_fitness, expected_rmse
        raise ValueError("Unexpected candidate")

    with patch.object(reg, "register_frame_pair", side_effect=mock_candidate_fn):
        with caplog.at_level(logging.WARNING):
            result = reg.register_against_candidates(new_frame, [cand0, cand1], cfg)

    # (a) Assert exception is logged/visible per fix #3
    warning_records = [
        r for r in caplog.records
        if "[fusion-fallback]" in r.message and "Candidate 0" in r.message
    ]
    assert len(warning_records) >= 1, (
        f"Expected '[fusion-fallback] Candidate 0' warning in logs, found:\n{caplog.text}"
    )
    assert "RuntimeError" in warning_records[0].message
    assert "Deliberate registration failure on candidate 0" in warning_records[0].message

    # (b) Assert pipeline correctly falls through to candidate 1
    assert result is not None, (
        "register_against_candidates silently returned None instead of falling through to candidate 1"
    )
    transform, fitness, rmse = result
    np.testing.assert_array_equal(transform, expected_transform)
    assert fitness == expected_fitness
    assert rmse == expected_rmse


def test_register_against_candidates_all_candidates_raise_returns_none(caplog):
    """
    When all candidates raise exceptions, all exceptions are logged and
    the function safely returns None.
    """
    cfg = RegistrationConfig()
    new_frame = o3d.geometry.PointCloud()
    new_frame.points = o3d.utility.Vector3dVector(np.ones((20, 3)))

    cand0 = o3d.geometry.PointCloud()
    cand0.points = o3d.utility.Vector3dVector(np.ones((20, 3)) * 0.1)
    cand1 = o3d.geometry.PointCloud()
    cand1.points = o3d.utility.Vector3dVector(np.ones((20, 3)) * 0.2)

    def failing_candidate_fn(src, dst, c):
        raise ValueError(f"Failing registration for candidate {dst}")

    with patch.object(reg, "register_frame_pair", side_effect=failing_candidate_fn):
        with caplog.at_level(logging.WARNING):
            result = reg.register_against_candidates(new_frame, [cand0, cand1], cfg)

    assert result is None
    logged_warnings = [
        r.message for r in caplog.records if "[fusion-fallback]" in r.message
    ]
    assert len(logged_warnings) == 2
    assert "Candidate 0 registration failed with ValueError" in logged_warnings[0]
    assert "Candidate 1 registration failed with ValueError" in logged_warnings[1]


def test_registration_regression_known_motions_and_junk_frame():
    """Verify known motion recovery and junk frame rejection across pipeline paths."""
    res = main()
    assert res == 0


if __name__ == "__main__":
    sys.exit(main())


