"""
tests/fullbody/test_keypoints_and_parts.py

Unit tests for 3D keypoint lifting and body part capsule segmentation:
- 3D lifting with window-based MAD outlier rejection.
- Skeletal capsule distance labeling.
- Strict separation of rigid torso/legs from non-rigid arms.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.keypoints import lift_keypoints_to_3d
from realsense_measure.fullbody.parts import (
    ARM_PARTS,
    RIGID_PARTS,
    BodyPart,
    is_arm_point,
    is_rigid_point,
    label_points_by_body_part,
)
from realsense_measure.fullbody.types import Keypoints3D


def test_keypoint_lifting_mad_outlier_rejection():
    """Verify local window lifting rejects background depth spike via MAD."""
    h, w = 40, 40
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 100.0, 100.0, 20.0, 20.0)
    cfg = FullBodyConfig(keypoint_window_size_px=3, keypoint_min_confidence=0.3)

    # Depth map is foreground at 1.20 m, with 2 background spike pixels at 2.40 m
    depth = np.full((h, w), 1.20, dtype=np.float32)
    depth[19:21, 19] = 2.40  # outlier spike near nose at (20, 20)

    pts_2d = {"nose": (20.0, 20.0)}
    conf_2d = {"nose": 0.90}
    vis_2d = {"nose": 0.90}

    kpts_3d = lift_keypoints_to_3d(pts_2d, conf_2d, vis_2d, depth, intr, cfg)

    pt, conf, vis = kpts_3d.get("nose")
    assert pt is not None
    assert np.isclose(pt[2], 1.20, atol=0.01)  # Rejected the 2.40m spike!
    assert conf > 0.5


def test_body_part_capsule_labeling():
    """Verify points within capsules are labeled as expected body parts."""
    cfg = FullBodyConfig()

    # Generate synthetic keypoints
    from tests.fullbody.conftest import create_ground_truth_keypoints
    kpts = create_ground_truth_keypoints()

    # Test points at known locations:
    # 1. Torso center point: (0.0, 1.20, 0.0)
    pt_torso = np.array([[0.0, 1.20, 0.0]], dtype=np.float32)
    # 2. Left arm point: (-0.32, 1.15, 0.0)
    pt_arm = np.array([[-0.32, 1.15, 0.0]], dtype=np.float32)
    # 3. Leg point: (-0.09, 0.30, 0.0)
    pt_leg = np.array([[-0.09, 0.30, 0.0]], dtype=np.float32)

    test_pts = np.vstack([pt_torso, pt_arm, pt_leg])
    labels = label_points_by_body_part(test_pts, kpts, cfg)

    assert labels[0] == int(BodyPart.TORSO)
    assert labels[1] in [int(p) for p in ARM_PARTS]
    assert labels[2] in [int(BodyPart.THIGH_LEFT), int(BodyPart.SHANK_LEFT)]

    assert is_rigid_point(labels)[0] == True
    assert is_arm_point(labels)[0] == False

    assert is_rigid_point(labels)[1] == False
    assert is_arm_point(labels)[1] == True

    assert is_rigid_point(labels)[2] == True
    assert is_arm_point(labels)[2] == False

