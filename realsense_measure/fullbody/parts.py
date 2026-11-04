"""
realsense_measure/fullbody/parts.py

Body-part segmentation and point labeling via skeletal capsule distances.
Labels points as head, torso, pelvis, upper arms, forearms, thighs, or shanks.
Used to isolate trunk cross-sections, mask arms from circumferences, and
exclude non-rigid limb segments during global multi-stop alignment.
"""

from __future__ import annotations

from enum import IntEnum
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import Keypoints3D

logger = logging.getLogger(__name__)


class BodyPart(IntEnum):
    UNASSIGNED = 0
    HEAD = 1
    TORSO = 2
    PELVIS = 3
    UPPER_ARM_LEFT = 4
    UPPER_ARM_RIGHT = 5
    FOREARM_LEFT = 6
    FOREARM_RIGHT = 7
    THIGH_LEFT = 8
    THIGH_RIGHT = 9
    SHANK_LEFT = 10
    SHANK_RIGHT = 11


RIGID_PARTS = {
    BodyPart.HEAD,
    BodyPart.TORSO,
    BodyPart.PELVIS,
    BodyPart.THIGH_LEFT,
    BodyPart.THIGH_RIGHT,
    BodyPart.SHANK_LEFT,
    BodyPart.SHANK_RIGHT,
}

ARM_PARTS = {
    BodyPart.UPPER_ARM_LEFT,
    BodyPart.UPPER_ARM_RIGHT,
    BodyPart.FOREARM_LEFT,
    BodyPart.FOREARM_RIGHT,
}


def point_to_segment_distance(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance from each point in points (N, 3) to segment [seg_a, seg_b].

    Returns
    -------
    np.ndarray: shape (N,) distances in metres.
    """
    v = seg_b - seg_a
    v_len_sq = float(np.sum(v ** 2))
    if v_len_sq < 1e-10:
        return np.linalg.norm(points - seg_a, axis=1)

    # Project points onto line: t = (P - A) . v / ||v||^2
    p_minus_a = points - seg_a
    t = np.sum(p_minus_a * v, axis=1) / v_len_sq
    t = np.clip(t, 0.0, 1.0)[:, np.newaxis]

    closest = seg_a + t * v
    return np.linalg.norm(points - closest, axis=1)


def label_points_by_body_part(
    points: np.ndarray,
    keypoints: Keypoints3D,
    cfg: FullBodyConfig,
) -> np.ndarray:
    """
    Assign each 3D point (N, 3) an integer BodyPart label based on nearest capsule.

    Points further than 1.5x capsule radius from all segments remain UNASSIGNED (0).

    Returns
    -------
    np.ndarray: shape (N,) integer labels corresponding to BodyPart enum.
    """
    n_pts = len(points)
    if n_pts == 0:
        return np.zeros(0, dtype=np.int32)

    labels = np.zeros(n_pts, dtype=np.int32)

    # Define segment capsules: (BodyPart, joint_a, joint_b, radius_m)
    segments = [
        (BodyPart.HEAD, "nose", "mid_shoulder", cfg.capsule_radius_head_m),
        (BodyPart.TORSO, "mid_shoulder", "mid_hip", cfg.capsule_radius_torso_m),
        (BodyPart.PELVIS, "left_hip", "right_hip", cfg.capsule_radius_pelvis_m),
        (BodyPart.UPPER_ARM_LEFT, "left_shoulder", "left_elbow", cfg.capsule_radius_upper_arm_m),
        (BodyPart.UPPER_ARM_RIGHT, "right_shoulder", "right_elbow", cfg.capsule_radius_upper_arm_m),
        (BodyPart.FOREARM_LEFT, "left_elbow", "left_wrist", cfg.capsule_radius_forearm_m),
        (BodyPart.FOREARM_RIGHT, "right_elbow", "right_wrist", cfg.capsule_radius_forearm_m),
        (BodyPart.THIGH_LEFT, "left_hip", "left_knee", cfg.capsule_radius_thigh_m),
        (BodyPart.THIGH_RIGHT, "right_hip", "right_knee", cfg.capsule_radius_thigh_m),
        (BodyPart.SHANK_LEFT, "left_knee", "left_ankle", cfg.capsule_radius_shank_m),
        (BodyPart.SHANK_RIGHT, "right_knee", "right_ankle", cfg.capsule_radius_shank_m),
    ]

    min_norm_dist = np.full(n_pts, 1e9, dtype=np.float32)

    for part, j_a, j_b, radius in segments:
        pt_a, c_a, _ = keypoints.get(j_a)
        pt_b, c_b, _ = keypoints.get(j_b)

        if pt_a is None or pt_b is None or c_a < 0.1 or c_b < 0.1:
            continue

        dist = point_to_segment_distance(points, pt_a, pt_b)
        norm_dist = dist / max(0.01, radius)

        # Update closest capsule if within 1.5x capsule radius
        is_closer = (norm_dist < min_norm_dist) & (norm_dist <= 1.5)
        labels[is_closer] = int(part)
        min_norm_dist[is_closer] = norm_dist[is_closer]

    return labels


def is_rigid_point(labels: np.ndarray) -> np.ndarray:
    """Return boolean mask indicating points that belong to rigid torso/pelvis/leg parts."""
    rigid_int_values = {int(p) for p in RIGID_PARTS}
    return np.isin(labels, list(rigid_int_values))


def is_arm_point(labels: np.ndarray) -> np.ndarray:
    """Return boolean mask indicating points that belong to arms."""
    arm_int_values = {int(p) for p in ARM_PARTS}
    return np.isin(labels, list(arm_int_values))

