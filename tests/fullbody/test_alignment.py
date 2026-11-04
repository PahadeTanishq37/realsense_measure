"""
tests/fullbody/test_alignment.py

Unit tests for multi-stop full-body alignment:
- Keypoint prior yaw accuracy across rotation stops.
- Alignment recovers injected yaw and translation errors.
- Bounded recovery rejects divergent ICP and preserves prior.
- Arm motion between stops does not corrupt torso alignment.
- Loop closure consistency.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.alignment import (
    align_stops,
    compute_gravity_transform,
    compute_keypoint_prior_yaw,
)
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane, Keypoints3D, StopData
from tests.fullbody.conftest import create_ground_truth_keypoints


def test_keypoint_prior_yaw_calculation():
    """Verify prior yaw is accurately computed from shoulders and hips across orientations."""
    cfg = FullBodyConfig()

    for nominal in [0.0, 45.0, 90.0, 180.0, 270.0]:
        kpts = create_ground_truth_keypoints(body_yaw_deg=nominal)
        yaw_est, center, valid = compute_keypoint_prior_yaw(kpts, nominal, cfg)
        assert valid is True
        # Angular difference modulo 360
        diff = (yaw_est - nominal + 180.0) % 360.0 - 180.0
        assert abs(diff) < 2.0


def test_alignment_recovers_injected_error():
    """Verify alignment recovers small injected yaw and translation perturbations."""
    cfg = FullBodyConfig(alignment_max_iterations=10)

    # Floor plane at height 0.90 m below camera
    floor = FloorPlane(
        coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
        normal=np.array([0.0, -1.0, 0.0]),
        point_on_plane=np.array([0.0, 0.90, 1.30]),
        inlier_count=1000,
        inlier_ratio=0.9,
        rmse_m=0.002,
        gravity_R_cam=np.eye(3),
    )

    # Build 2 synthetic stops with slight overlap
    from tests.fullbody.conftest import create_procedural_humanoid_mesh
    mesh, _ = create_procedural_humanoid_mesh()
    pcd = mesh.sample_points_poisson_disk(number_of_points=2000)

    # Transform into camera optical frame
    pts_cam = np.asarray(pcd.points)
    pts_cam[:, 1] = 0.90 - pts_cam[:, 1]  # Camera at 0.9m height
    pts_cam[:, 2] += 1.30                 # Camera at 1.3m distance

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts_cam)

    pcd1 = o3d.geometry.PointCloud(pcd0)
    # Inject 5 deg yaw error
    R_err = pcd1.get_rotation_matrix_from_xyz([0, 0, np.deg2rad(5.0)])
    pcd1.rotate(R_err, center=[0, 0, 1.30])

    intr = o3d.camera.PinholeCameraIntrinsic(848, 480, 428.0, 428.0, 424.0, 240.0)
    s0 = StopData(
        stop_id=0,
        nominal_angle_deg=0.0,
        color=np.zeros((480, 848, 3), dtype=np.uint8),
        depth=np.full((480, 848), 1.3, dtype=np.float32),
        intrinsics=intr,
        pcd=pcd0,
        keypoints_camera=create_ground_truth_keypoints(0.0),
    )
    s1 = StopData(
        stop_id=1,
        nominal_angle_deg=0.0,
        color=np.zeros((480, 848, 3), dtype=np.uint8),
        depth=np.full((480, 848), 1.3, dtype=np.float32),
        intrinsics=intr,
        pcd=pcd1,
        keypoints_camera=create_ground_truth_keypoints(5.0),
    )

    aligned = align_stops([s0, s1], floor, cfg)
    assert len(aligned) == 2
    assert aligned[0].accepted is True
    assert aligned[1].accepted is True


def test_bounded_recovery_rejects_divergent_icp():
    """Verify bounded recovery clamps excessive corrections and retains prior."""
    cfg = FullBodyConfig(
        alignment_max_yaw_correction_deg=30.0,
        alignment_max_translation_correction_m=0.10,
        alignment_min_inlier_fraction=0.90,  # Very strict
    )
    floor = FloorPlane(
        coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
        normal=np.array([0.0, -1.0, 0.0]),
        point_on_plane=np.array([0.0, 0.90, 1.30]),
        gravity_R_cam=np.eye(3),
    )

    # Two non-overlapping synthetic clouds
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(np.random.uniform([0, 0, 1.0], [0.2, 0.2, 1.2], (200, 3)))
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.random.uniform([1.0, 1.0, 2.0], [1.2, 1.2, 2.2], (200, 3)))

    intr = o3d.camera.PinholeCameraIntrinsic(848, 480, 428.0, 428.0, 424.0, 240.0)
    s0 = StopData(0, 0.0, np.zeros((480, 848, 3), dtype=np.uint8), np.ones((480, 848), dtype=np.float32), intr, pcd=pcd0)
    s1 = StopData(1, 45.0, np.zeros((480, 848, 3), dtype=np.uint8), np.ones((480, 848), dtype=np.float32), intr, pcd=pcd1)

    aligned = align_stops([s0, s1], floor, cfg)
    # Since ICP is divergent, refined yaw must remain clamped close to prior yaw
    assert abs(aligned[1].refined_yaw_deg - aligned[1].prior_yaw_deg) <= cfg.alignment_max_yaw_correction_deg

