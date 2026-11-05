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



def test_handedness_real_photo():
    """
    Verify handedness convention on a real photo (skipped if absent).
    Subject facing the camera and raising the right hand.
    Asserts right_wrist.x < left_wrist.x in pixel space (subject's right is on image left).
    """
    import os
    from pathlib import Path

    img_path = os.environ.get("FULLBODY_HANDEDNESS_IMAGE") or os.environ.get("FULLBODY_SAMPLE_IMAGE")
    if not img_path or not Path(img_path).is_file():
        pytest.skip("No user-supplied photo provided (set FULLBODY_HANDEDNESS_IMAGE to run)")

    import cv2
    from realsense_measure.fullbody.keypoints import MediaPipePoseDetector

    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Failed to read image at {img_path}"

    detector = MediaPipePoseDetector()
    pts_2d, confs, _ = detector.detect(img_bgr)
    assert "right_wrist" in pts_2d and "left_wrist" in pts_2d, "Wrists not detected in sample image"
    assert confs.get("right_wrist", 0) > 0.3 and confs.get("left_wrist", 0) > 0.3, "Low wrist confidence"

    rw_x = pts_2d["right_wrist"][0]
    lw_x = pts_2d["left_wrist"][0]
    assert rw_x < lw_x, (
        f"Handedness convention check failed: right_wrist x ({rw_x:.1f}) >= left_wrist x ({lw_x:.1f}). "
        f"When facing camera, right hand must be on camera-left (smaller pixel X)."
    )


def test_yaw_prior_and_crossproduct_sign_convention():
    """
    Re-check yaw prior and cross-product sign against standard right-handed optical convention.
    Ensures anterior vector and canonical yaw sign conventions are strictly consistent.
    """
    from realsense_measure.fullbody.alignment import compute_keypoint_prior_yaw

    cfg = FullBodyConfig()

    # Subject facing camera in gravity frame:
    # Camera looks along +Z, +Y is up.
    # Subject's right shoulder is on camera-left (X = -0.18 m).
    # Subject's left shoulder is on camera-right (X = +0.18 m).
    # Subject's nose is anterior (Z = 1.20 m, closer than shoulders at Z = 1.30 m).
    kpts_facing_cam = Keypoints3D(
        names=["left_shoulder", "right_shoulder", "nose"],
        points=np.array([
            [ 0.18, 1.40, 1.30],   # left_shoulder (camera-right)
            [-0.18, 1.40, 1.30],   # right_shoulder (camera-left)
            [ 0.00, 1.65, 1.20],   # nose (closer to camera)
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        visibility=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        frame="gravity",
    )

    yaw_deg, center, valid = compute_keypoint_prior_yaw(kpts_facing_cam, nominal_deg=180.0, cfg=cfg)
    assert valid is True
    # Stop 0 facing camera has canonical yaw ~ 180°
    assert abs(abs(yaw_deg) - 180.0) < 1.0, f"Expected canonical yaw ~ 180°, got {yaw_deg:.2f}°"

    # Subject turned 90° clockwise (facing subject's right):
    # Chest (+Z) points to camera-left (X = -Z_cam).
    # Left shoulder is anterior to right shoulder.
    kpts_turned_90 = Keypoints3D(
        names=["left_shoulder", "right_shoulder", "nose"],
        points=np.array([
            [ 0.00, 1.40, 1.48],   # left_shoulder (deeper)
            [ 0.00, 1.40, 1.12],   # right_shoulder (closer)
            [-0.10, 1.65, 1.30],   # nose (pointing left)
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        visibility=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        frame="gravity",
    )

    yaw_90, _, valid_90 = compute_keypoint_prior_yaw(kpts_turned_90, nominal_deg=90.0, cfg=cfg)
    assert valid_90 is True
    # In canonical frame (+Y up, +X left, +Z anterior):
    # Nose pointing to camera-left means yaw is ~ 90° (or -90° depending on convention)
    assert abs(abs(yaw_90) - 90.0) < 2.0, f"Expected canonical yaw ~ 90°, got {yaw_90:.2f}°"
