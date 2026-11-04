"""
realsense_measure/fullbody/landmarks.py

Derive anatomical measurement levels, planes, and long axes from
the canonical fused scan and transformed 3D keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import Keypoints3D

logger = logging.getLogger(__name__)


@dataclass
class SlicingPlane:
    """
    Parametric plane for extracting 2D cross-sectional rings.
    Defined by point on plane and unit normal vector.
    """
    name: str
    origin: np.ndarray      # shape (3,), float64
    normal: np.ndarray      # shape (3,), float64 unit normal
    u_axis: np.ndarray      # shape (3,), in-plane local coordinate axis
    v_axis: np.ndarray      # shape (3,), in-plane local coordinate axis
    is_limb: bool = False
    limb_part: Optional[str] = None


def extract_anatomical_levels(
    pcd: o3d.geometry.PointCloud,
    keypoints: Optional[Keypoints3D],
    cfg: FullBodyConfig,
) -> Dict[str, float]:
    """
    Derive vertical Y levels (in metres above floor) for standard anatomical planes.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return {}

    y_floor = 0.0
    # Robust crown: 99.8th percentile of Y
    y_crown = float(np.percentile(pts[:, 1], 99.8))
    stature = max(0.5, y_crown - y_floor)

    levels: Dict[str, float] = {
        "floor": y_floor,
        "crown": y_crown,
        "stature": stature,
    }

    # Extract levels from keypoints if available
    mid_sh, _, _ = keypoints.get("mid_shoulder") if keypoints else (None, 0, 0)
    mid_hp, _, _ = keypoints.get("mid_hip") if keypoints else (None, 0, 0)
    l_hp, _, _ = keypoints.get("left_hip") if keypoints else (None, 0, 0)
    r_hp, _, _ = keypoints.get("right_hip") if keypoints else (None, 0, 0)
    l_kn, _, _ = keypoints.get("left_knee") if keypoints else (None, 0, 0)
    r_kn, _, _ = keypoints.get("right_knee") if keypoints else (None, 0, 0)

    # 1. Neck base: between mid-shoulder and chin/nose
    if mid_sh is not None:
        levels["neck_base"] = float(mid_sh[1] + 0.04)
        levels["chest"] = float(mid_sh[1] - 0.12)
    else:
        levels["neck_base"] = float(y_floor + 0.85 * stature)
        levels["chest"] = float(y_floor + 0.74 * stature)

    # 2. Waist: search for minimum trunk cross-sectional width between 56% and 66% stature
    waist_y_min = y_floor + 0.56 * stature
    waist_y_max = y_floor + 0.67 * stature
    levels["waist"] = _find_narrowest_trunk_level(pts, waist_y_min, waist_y_max)

    # 3. Hip: search for maximum trunk cross-sectional width between 48% and 56% stature
    hip_y_min = y_floor + 0.48 * stature
    hip_y_max = y_floor + 0.57 * stature
    levels["hip"] = _find_widest_trunk_level(pts, hip_y_min, hip_y_max)

    # 4. Crotch level: immediately below mid-hip or ~48% stature
    if mid_hp is not None:
        levels["crotch"] = float(mid_hp[1] - 0.08)
    else:
        levels["crotch"] = float(y_floor + 0.47 * stature)

    return levels


def _find_narrowest_trunk_level(pts: np.ndarray, y_min: float, y_max: float) -> float:
    """Find vertical Y level with minimum torso lateral bounding extent."""
    y_steps = np.linspace(y_min, y_max, 15)
    best_y = 0.5 * (y_min + y_max)
    min_width = 1e9

    for y in y_steps:
        slab = pts[(pts[:, 1] >= y - 0.01) & (pts[:, 1] <= y + 0.01) & (np.abs(pts[:, 0]) < 0.25)]
        if len(slab) >= 20:
            width = float(np.percentile(slab[:, 0], 95) - np.percentile(slab[:, 0], 5))
            if width < min_width:
                min_width = width
                best_y = float(y)
    return best_y


def _find_widest_trunk_level(pts: np.ndarray, y_min: float, y_max: float) -> float:
    """Find vertical Y level with maximum hip/buttock lateral bounding extent."""
    y_steps = np.linspace(y_min, y_max, 15)
    best_y = 0.5 * (y_min + y_max)
    max_width = 0.0

    for y in y_steps:
        slab = pts[(pts[:, 1] >= y - 0.01) & (pts[:, 1] <= y + 0.01) & (np.abs(pts[:, 0]) < 0.28)]
        if len(slab) >= 20:
            width = float(np.percentile(slab[:, 0], 95) - np.percentile(slab[:, 0], 5))
            if width > max_width:
                max_width = width
                best_y = float(y)
    return best_y


def build_slicing_planes(
    levels: Dict[str, float],
    keypoints: Optional[Keypoints3D],
    cfg: FullBodyConfig,
) -> List[SlicingPlane]:
    """
    Build parametric slicing planes for all torso rings and perpendicular limb rings.
    """
    planes: List[SlicingPlane] = []

    # Torso horizontal planes (normal is +Y = [0, 1, 0])
    torso_rings = ["neck_base", "chest", "waist", "hip"]
    for ring_name in torso_rings:
        if ring_name in levels:
            y = levels[ring_name]
            planes.append(
                SlicingPlane(
                    name=ring_name,
                    origin=np.array([0.0, y, 0.0]),
                    normal=np.array([0.0, 1.0, 0.0]),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 0.0, 1.0]),
                    is_limb=False,
                )
            )

    if keypoints is None:
        return planes

    # Limb rings (perpendicular to limb long axis!)
    limb_segments = [
        ("upper_arm_left", "left_shoulder", "left_elbow", "mid"),
        ("upper_arm_right", "right_shoulder", "right_elbow", "mid"),
        ("forearm_left", "left_elbow", "left_wrist", "mid"),
        ("forearm_right", "right_elbow", "right_wrist", "mid"),
        ("thigh_left", "left_hip", "left_knee", "mid"),
        ("thigh_right", "right_hip", "right_knee", "mid"),
        ("calf_left", "left_knee", "left_ankle", "max_gastrocnemius"),
        ("calf_right", "right_knee", "right_ankle", "max_gastrocnemius"),
    ]

    for name, j_a, j_b, loc in limb_segments:
        pt_a, c_a, _ = keypoints.get(j_a)
        pt_b, c_b, _ = keypoints.get(j_b)

        if pt_a is None or pt_b is None or c_a < 0.1 or c_b < 0.1:
            continue

        axis = pt_b - pt_a
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-4:
            continue
        normal = axis / axis_len

        # Location along segment
        t = 0.5 if loc == "mid" else 0.35  # max calf belly is ~35% down from knee
        origin = pt_a + t * axis

        # Orthogonal coordinate frame on plane
        ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.8 else np.array([1.0, 0.0, 0.0])
        u_axis = np.cross(normal, ref)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)

        planes.append(
            SlicingPlane(
                name=name,
                origin=origin,
                normal=normal,
                u_axis=u_axis,
                v_axis=v_axis,
                is_limb=True,
                limb_part=name,
            )
        )

    return planes

