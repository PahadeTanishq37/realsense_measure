"""
tests/fullbody/conftest.py

Procedural humanoid generator, synthetic raycasting renderer,
sensor noise injector, and ground-truth keypoint provider for fullbody unit tests.
All synthetic tests are strictly code-path and geometry verification tests.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane, Keypoints3D, StopData


def ramanujan_ellipse_perimeter(a: float, b: float) -> float:
    """Compute analytic perimeter of ellipse with semi-axes a and b using Ramanujan's formula."""
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def create_procedural_humanoid_mesh() -> Tuple[o3d.geometry.TriangleMesh, Dict[str, float]]:
    """
    Generate clean, manifold humanoid triangle mesh with known analytic dimensions:
    - Floor plane at Y = 0.
    - Stature = 1.75 m.
    - Torso: elliptical cylinder (a = 0.16 m, b = 0.12 m -> chest circ approx 885 mm).
    - Waist: elliptical cylinder (a = 0.14 m, b = 0.10 m -> waist circ approx 760 mm).
    - Head: ellipsoid (radius = 0.10 m, crown at Y = 1.75 m).
    - Limbs in canonical A-pose.
    """
    meshes = []
    ground_truth: Dict[str, float] = {}

    # 1. Torso: lofted elliptical cylinders from Y = 0.85 to 1.45 m
    a_chest, b_chest = 0.16, 0.12
    ground_truth["chest_circumference"] = ramanujan_ellipse_perimeter(a_chest, b_chest) * 1000.0

    a_waist, b_waist = 0.14, 0.10
    ground_truth["waist_circumference"] = ramanujan_ellipse_perimeter(a_waist, b_waist) * 1000.0

    a_hip, b_hip = 0.17, 0.13
    ground_truth["hip_circumference"] = ramanujan_ellipse_perimeter(a_hip, b_hip) * 1000.0

    ground_truth["stature"] = 1750.0  # mm

    def _make_elliptical_tube(a_bot, b_bot, a_top, b_top, y_bot, y_top, n=48):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        bot_pts = np.stack([a_bot * np.cos(angles), np.full(n, y_bot), b_bot * np.sin(angles)], axis=1)
        top_pts = np.stack([a_top * np.cos(angles), np.full(n, y_top), b_top * np.sin(angles)], axis=1)
        v = np.vstack([bot_pts, top_pts])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([i, j, n + i])
            f.append([j, n + j, n + i])
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(np.array(f, dtype=np.int32))
        return m

    # Pelvis to waist (0.80 - 1.05m)
    m_pelvis = _make_elliptical_tube(a_hip, b_hip, a_waist, b_waist, 0.80, 1.05)
    # Waist to chest (1.05 - 1.42m)
    m_torso = _make_elliptical_tube(a_waist, b_waist, a_chest, b_chest, 1.05, 1.42)
    meshes.extend([m_pelvis, m_torso])

    # Head: sphere at Y = 1.65m, radius = 0.10m -> crown at Y = 1.75m
    m_head = o3d.geometry.TriangleMesh.create_sphere(radius=0.10, resolution=20)
    m_head.translate([0.0, 1.65, 0.0])
    meshes.append(m_head)

    # Left and right legs (cylinders down to floor Y = 0)
    r_leg = 0.06
    ground_truth["thigh_circumference"] = 2 * math.pi * r_leg * 1000.0
    m_leg_l = o3d.geometry.TriangleMesh.create_cylinder(radius=r_leg, height=0.80, resolution=20)
    m_leg_l.translate([-0.09, 0.40, 0.0])
    m_leg_r = o3d.geometry.TriangleMesh.create_cylinder(radius=r_leg, height=0.80, resolution=20)
    m_leg_r.translate([0.09, 0.40, 0.0])
    meshes.extend([m_leg_l, m_leg_r])

    # Left and right arms in A-pose (angled 30 deg away from body)
    r_arm = 0.04
    ground_truth["arm_circumference"] = 2 * math.pi * r_arm * 1000.0
    m_arm_l = o3d.geometry.TriangleMesh.create_cylinder(radius=r_arm, height=0.55, resolution=20)
    R_l = m_arm_l.get_rotation_matrix_from_xyz([0.0, 0.0, np.deg2rad(30.0)])
    m_arm_l.rotate(R_l, center=[0, 0, 0])
    m_arm_l.translate([-0.28, 1.15, 0.0])

    m_arm_r = o3d.geometry.TriangleMesh.create_cylinder(radius=r_arm, height=0.55, resolution=20)
    R_r = m_arm_r.get_rotation_matrix_from_xyz([0.0, 0.0, np.deg2rad(-30.0)])
    m_arm_r.rotate(R_r, center=[0, 0, 0])
    m_arm_r.translate([0.28, 1.15, 0.0])
    meshes.extend([m_arm_l, m_arm_r])

    # Floor plane: 2x2 m square quad at Y = 0.0
    floor_verts = np.array([
        [-1.0, 0.0, -1.0],
        [ 1.0, 0.0, -1.0],
        [ 1.0, 0.0,  1.0],
        [-1.0, 0.0,  1.0],
    ])
    floor_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    m_floor = o3d.geometry.TriangleMesh()
    m_floor.vertices = o3d.utility.Vector3dVector(floor_verts)
    m_floor.triangles = o3d.utility.Vector3iVector(floor_tris)
    meshes.append(m_floor)

    combined = meshes[0]
    for m in meshes[1:]:
        combined += m

    combined.compute_vertex_normals()
    return combined, ground_truth


def create_ground_truth_keypoints(body_yaw_deg: float = 0.0) -> Keypoints3D:
    """Generate ground-truth 3D keypoints for the procedural humanoid in canonical frame."""
    base_kpts = {
        "nose": np.array([0.0, 1.65, 0.10]),
        "left_eye": np.array([-0.03, 1.68, 0.09]),
        "right_eye": np.array([0.03, 1.68, 0.09]),
        "left_ear": np.array([-0.09, 1.66, 0.0]),
        "right_ear": np.array([0.09, 1.66, 0.0]),
        "left_shoulder": np.array([-0.18, 1.40, 0.0]),
        "right_shoulder": np.array([0.18, 1.40, 0.0]),
        "left_elbow": np.array([-0.32, 1.15, 0.0]),
        "right_elbow": np.array([0.32, 1.15, 0.0]),
        "left_wrist": np.array([-0.42, 0.90, 0.0]),
        "right_wrist": np.array([0.42, 0.90, 0.0]),
        "left_hip": np.array([-0.10, 0.85, 0.0]),
        "right_hip": np.array([0.10, 0.85, 0.0]),
        "left_knee": np.array([-0.09, 0.45, 0.0]),
        "right_knee": np.array([0.09, 0.45, 0.0]),
        "left_ankle": np.array([-0.09, 0.08, 0.0]),
        "right_ankle": np.array([0.09, 0.08, 0.0]),
        "mid_shoulder": np.array([0.0, 1.40, 0.0]),
        "mid_hip": np.array([0.0, 0.85, 0.0]),
    }

    yaw_rad = np.deg2rad(body_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([
        [c, 0,  s],
        [0, 1,  0],
        [-s, 0, c],
    ])

    names = list(base_kpts.keys())
    pts = np.array([R @ base_kpts[k] for k in names], dtype=np.float32)
    conf = np.ones(len(names), dtype=np.float32)
    vis = np.ones(len(names), dtype=np.float32)

    return Keypoints3D(names=names, points=pts, confidence=conf, visibility=vis, frame="canonical")


@pytest.fixture
def synthetic_humanoid_mesh():
    """Pytest fixture providing procedural humanoid mesh and ground-truth dimensions."""
    return create_procedural_humanoid_mesh()
