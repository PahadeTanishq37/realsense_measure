"""
tests/fullbody/synth_humanoid.py

Synthetic humanoid generator, raycasting renderer, and ground-truth keypoint provider.
- Lofted elliptical cross-sections for torso/pelvis (known analytic Ramanujan circumferences).
- Ellipsoid head and articulated limbs (A-pose with adjustable abduction).
- Open3D RaycastingScene multi-stop rendering with noise model sigma(z)=1mm*(z/0.6)^2,
  10% dropouts, edge flying pixels, per-stop yaw perturbation (+-5 deg),
  translation perturbation (+-3 cm), and small inter-stop arm motion.
- Ground-truth keypoint detector replacing MediaPipe in deterministic tests.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane, Keypoints3D, StopData


def ramanujan_perimeter(a: float, b: float) -> float:
    """Analytic perimeter of an ellipse with semi-axes a and b using Ramanujan's formula."""
    return float(math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def build_articulated_humanoid_mesh(
    arm_abduction_deg: float = 35.0,
    arm_jitter_deg: float = 0.0,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Construct a clean manifold humanoid mesh in canonical ISB frame:
      +X: Subject's left
      +Y: Upwards (superior)
      +Z: Anterior (chest)
      Origin (0, 0, 0) on floor plane beneath vertical axis.
    """
    meshes: List[o3d.geometry.TriangleMesh] = []
    ground_truth: Dict[str, float] = {}
    skeleton: Dict[str, np.ndarray] = {}

    # Analytic torso dimensions (metres)
    a_chest, b_chest = 0.16, 0.12   # Chest semi-axes -> ~885 mm
    a_waist, b_waist = 0.14, 0.10   # Waist semi-axes -> ~760 mm
    a_hip, b_hip = 0.17, 0.13       # Hip semi-axes   -> ~948 mm

    ground_truth["chest_circumference"] = ramanujan_perimeter(a_chest, b_chest) * 1000.0
    ground_truth["waist_circumference"] = ramanujan_perimeter(a_waist, b_waist) * 1000.0
    ground_truth["hip_circumference"] = ramanujan_perimeter(a_hip, b_hip) * 1000.0
    ground_truth["stature"] = 1750.0  # mm
    ground_truth["shoulder_breadth"] = 360.0  # mm

    def _tube(a0, b0, a1, b1, y0, y1, n=48):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        bot = np.stack([a0 * np.cos(angles), np.full(n, y0), b0 * np.sin(angles)], axis=1)
        top = np.stack([a1 * np.cos(angles), np.full(n, y1), b1 * np.sin(angles)], axis=1)
        v = np.vstack([bot, top])
        fs = []
        for i in range(n):
            j = (i + 1) % n
            fs.append([i, j, n + i])
            fs.append([j, n + j, n + i])
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(np.array(fs, dtype=np.int32))
        return m

    # Pelvis (Y: 0.80 - 1.02 m), Lower Torso (Y: 1.02 - 1.24 m), Chest (Y: 1.24 - 1.42 m)
    # The chest region maintains (a_chest, b_chest) across the slicing plane (Y ~ 1.28 m)
    # so that the analytic ground truth matches the geometry at the measurement plane.
    m_pelvis = _tube(a_hip, b_hip, a_waist, b_waist, 0.80, 1.02)
    m_waist_to_chest = _tube(a_waist, b_waist, a_chest, b_chest, 1.02, 1.24)
    m_chest = _tube(a_chest, b_chest, a_chest, b_chest, 1.24, 1.42)
    meshes.extend([m_pelvis, m_waist_to_chest, m_chest])

    # Head: sphere centered at Y = 1.65 m, radius 0.10 m -> crown exactly at Y = 1.75 m
    m_head = o3d.geometry.TriangleMesh.create_sphere(radius=0.10, resolution=20)
    m_head.translate([0.0, 1.65, 0.0])
    meshes.append(m_head)

    # Highest point visible from camera (geometric horizon)
    # Camera at (0, 0.90, -1.30), sphere center (0, 1.65, 0), radius 0.10
    dy_cam = 0.90 - 1.65
    dz_cam = -1.30
    d_cam = math.sqrt(dy_cam**2 + dz_cam**2)
    theta_cam = math.atan2(abs(dy_cam), abs(dz_cam))
    alpha_cam = math.acos(0.10 / d_cam)
    phi_apex = math.pi / 2.0 - (alpha_cam - theta_cam)
    y_vis_max = 1.65 + 0.10 * math.cos(phi_apex)
    ground_truth["highest_visible_apex_mm"] = y_vis_max * 1000.0  # ~1733 mm

    # Legs: cylinders radius 0.06 m down to Y = 0.04 m (feet on floor)
    r_leg = 0.06
    ground_truth["thigh_circumference"] = 2.0 * math.pi * r_leg * 1000.0
    ground_truth["calf_circumference"] = 2.0 * math.pi * r_leg * 1000.0
    ground_truth["thigh_left_circumference"] = 2.0 * math.pi * r_leg * 1000.0
    ground_truth["thigh_right_circumference"] = 2.0 * math.pi * r_leg * 1000.0
    ground_truth["calf_left_circumference"] = 2.0 * math.pi * r_leg * 1000.0
    ground_truth["calf_right_circumference"] = 2.0 * math.pi * r_leg * 1000.0

    R_z_to_y = o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0])

    m_leg_l = o3d.geometry.TriangleMesh.create_cylinder(radius=r_leg, height=0.76, resolution=20)
    m_leg_l.rotate(R_z_to_y, center=[0, 0, 0])
    m_leg_l.translate([0.09, 0.42, 0.0])
    m_leg_r = o3d.geometry.TriangleMesh.create_cylinder(radius=r_leg, height=0.76, resolution=20)
    m_leg_r.rotate(R_z_to_y, center=[0, 0, 0])
    m_leg_r.translate([-0.09, 0.42, 0.0])
    meshes.extend([m_leg_l, m_leg_r])

    # Arms in A-pose with adjustable abduction and per-stop jitter
    r_arm = 0.04
    ground_truth["arm_circumference"] = 2.0 * math.pi * r_arm * 1000.0
    ground_truth["upper_arm_left_circumference"] = 2.0 * math.pi * r_arm * 1000.0
    ground_truth["upper_arm_right_circumference"] = 2.0 * math.pi * r_arm * 1000.0
    ground_truth["forearm_left_circumference"] = 2.0 * math.pi * r_arm * 1000.0
    ground_truth["forearm_right_circumference"] = 2.0 * math.pi * r_arm * 1000.0
    abd_l = np.deg2rad(arm_abduction_deg + arm_jitter_deg)
    abd_r = np.deg2rad(arm_abduction_deg - arm_jitter_deg)

    # Left arm: rotate to Y then abduct in XY (outward toward positive X)
    m_arm_l = o3d.geometry.TriangleMesh.create_cylinder(radius=r_arm, height=0.55, resolution=20)
    m_arm_l.rotate(R_z_to_y, center=[0, 0, 0])
    R_l = m_arm_l.get_rotation_matrix_from_xyz([0.0, 0.0, abd_l])
    m_arm_l.rotate(R_l, center=[0, 0, 0])
    m_arm_l.translate([0.18 + 0.275 * np.sin(abd_l), 1.40 - 0.275 * np.cos(abd_l), 0.0])

    # Right arm: rotate to Y then abduct in XY (outward toward negative X)
    m_arm_r = o3d.geometry.TriangleMesh.create_cylinder(radius=r_arm, height=0.55, resolution=20)
    m_arm_r.rotate(R_z_to_y, center=[0, 0, 0])
    R_r = m_arm_r.get_rotation_matrix_from_xyz([0.0, 0.0, -abd_r])
    m_arm_r.rotate(R_r, center=[0, 0, 0])
    m_arm_r.translate([-0.18 - 0.275 * np.sin(abd_r), 1.40 - 0.275 * np.cos(abd_r), 0.0])
    meshes.extend([m_arm_l, m_arm_r])

    # Floor plane: quad at Y = 0.0 with upward normal (+Y)
    f_verts = np.array([
        [-2.0, 0.0, -2.0],
        [-2.0, 0.0,  4.0],
        [ 2.0, 0.0,  4.0],
        [ 2.0, 0.0, -2.0],
    ], dtype=np.float64)
    f_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    m_floor = o3d.geometry.TriangleMesh()
    m_floor.vertices = o3d.utility.Vector3dVector(f_verts)
    m_floor.triangles = o3d.utility.Vector3iVector(f_tris)
    meshes.append(m_floor)

    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    combined.compute_vertex_normals()

    # Skeleton landmarks in canonical ISB frame (+X left, +Y up, +Z chest)
    skeleton["nose"] = np.array([0.0, 1.65, 0.10], dtype=np.float64)
    skeleton["left_shoulder"] = np.array([0.18, 1.40, 0.0], dtype=np.float64)
    skeleton["right_shoulder"] = np.array([-0.18, 1.40, 0.0], dtype=np.float64)
    skeleton["mid_shoulder"] = np.array([0.0, 1.40, 0.0], dtype=np.float64)
    skeleton["left_elbow"] = np.array([0.18 + 0.28 * np.sin(abd_l), 1.40 - 0.28 * np.cos(abd_l), 0.0], dtype=np.float64)
    skeleton["right_elbow"] = np.array([-0.18 - 0.28 * np.sin(abd_r), 1.40 - 0.28 * np.cos(abd_r), 0.0], dtype=np.float64)
    skeleton["left_wrist"] = np.array([0.18 + 0.55 * np.sin(abd_l), 1.40 - 0.55 * np.cos(abd_l), 0.0], dtype=np.float64)
    skeleton["right_wrist"] = np.array([-0.18 - 0.55 * np.sin(abd_r), 1.40 - 0.55 * np.cos(abd_r), 0.0], dtype=np.float64)
    skeleton["left_hip"] = np.array([0.10, 0.82, 0.0], dtype=np.float64)
    skeleton["right_hip"] = np.array([-0.10, 0.82, 0.0], dtype=np.float64)
    skeleton["mid_hip"] = np.array([0.0, 0.82, 0.0], dtype=np.float64)
    skeleton["left_knee"] = np.array([0.09, 0.44, 0.0], dtype=np.float64)
    skeleton["right_knee"] = np.array([-0.09, 0.44, 0.0], dtype=np.float64)
    skeleton["left_ankle"] = np.array([0.09, 0.06, 0.0], dtype=np.float64)
    skeleton["right_ankle"] = np.array([-0.09, 0.06, 0.0], dtype=np.float64)

    return combined, ground_truth, skeleton


class SyntheticHumanoidCaptureRenderer:
    """
    Renders realistic synthetic multi-stop capture sessions from fixed portrait camera:
    - Fixed portrait camera at height 0.90m, distance 1.30m, level pitch.
    - Rotates body through N stops.
    - Injects noise: sigma(z) = 0.001 * (z/0.6)^2.
    - Injects dropouts (10%) and flying pixels.
    - Injects per-stop yaw perturbation (+-5 deg) and translation (+-3 cm).
    - Injects small arm posture jitter between stops (+-3 deg).
    """

    def __init__(
        self,
        width: int = 480,
        height: int = 848,
        fx: float = 428.0,
        fy: float = 428.0,
        cam_height_m: float = 0.90,
        cam_dist_m: float = 1.30,
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.cam_height = cam_height_m
        self.cam_dist = cam_dist_m
        self.rng = np.random.RandomState(seed)

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, self.cx, self.cy
        )

    def render_stops(
        self,
        num_stops: int = 8,
        burst_count: int = 1,
        add_noise: bool = True,
        yaw_jitter_deg: float = 5.0,
        trans_jitter_m: float = 0.03,
    ) -> Tuple[List[StopData], Dict[str, float], List[Keypoints3D]]:
        """
        Render N stops and return (stops, ground_truth_measurements, ground_truth_keypoints).
        """
        stops: List[StopData] = []
        gt_keypoints_list: List[Keypoints3D] = []
        step_deg = 360.0 / num_stops

        # Base humanoid mesh
        base_mesh, gt_measurements, base_skeleton = build_articulated_humanoid_mesh()

        for stop_idx in range(num_stops):
            nominal_yaw = stop_idx * step_deg
            # Perturbations
            dyaw = self.rng.uniform(-yaw_jitter_deg, yaw_jitter_deg) if add_noise else 0.0
            actual_yaw = nominal_yaw + dyaw

            dx = self.rng.uniform(-trans_jitter_m, trans_jitter_m) if add_noise else 0.0
            dz = self.rng.uniform(-trans_jitter_m, trans_jitter_m) if add_noise else 0.0
            arm_jit = self.rng.uniform(-3.0, 3.0) if add_noise else 0.0

            # 1. Build mesh for this stop with slight arm jitter
            mesh_stop, _, skel_stop = build_articulated_humanoid_mesh(arm_jitter_deg=arm_jit)

            # Rotate body around Y (clockwise looking from above: +Z rotates toward -X)
            yaw_rad = np.deg2rad(actual_yaw)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            R_body = np.array([
                [ c, 0, -s],
                [ 0, 1,  0],
                [ s, 0,  c],
            ], dtype=np.float64)

            # Apply body pose to mesh
            mesh_copy = o3d.geometry.TriangleMesh(mesh_stop)
            mesh_copy.rotate(R_body, center=[0, 0, 0])
            mesh_copy.translate([dx, 0.0, dz])

            # Transform mesh into camera optical frame:
            # World: +Y up, origin on floor at (0, 0, 0).
            # Camera: position at (0, cam_height, -cam_dist), looking down +Z.
            # In camera frame: X_cam = -X_world (right-handed optical frame), Y_cam = cam_height - Y_world, Z_cam = Z_world + cam_dist.
            T_cam_world = np.array([
                [-1.0,  0.0, 0.0,  0.0],
                [ 0.0, -1.0, 0.0,  self.cam_height],
                [ 0.0,  0.0, 1.0,  self.cam_dist],
                [ 0.0,  0.0, 0.0,  1.0],
            ], dtype=np.float64)

            mesh_cam = o3d.geometry.TriangleMesh(mesh_copy)
            mesh_cam.transform(T_cam_world)

            # 2. Render depth via Open3D RaycastingScene
            scene = o3d.t.geometry.RaycastingScene()
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_cam)
            scene.add_triangles(tmesh)

            # Create camera rays
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(self.intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float64),
                o3d.core.Tensor(np.eye(4, dtype=np.float64)),
                self.width,
                self.height,
            )
            cast_res = scene.cast_rays(rays)
            t_hit = cast_res["t_hit"].numpy()  # (H, W)

            # In Open3D create_rays_pinhole, ray direction is (dx, dy, 1.0) so t_hit is directly planar depth Z_cam
            clean_depth = np.zeros((self.height, self.width), dtype=np.float32)
            hit_mask = np.isfinite(t_hit) & (t_hit < 4.0) & (t_hit > 0.3)
            clean_depth[hit_mask] = t_hit[hit_mask].astype(np.float32)

            # Synthetic color frame: shade by depth and normal
            color_bgr = np.full((self.height, self.width, 3), 160, dtype=np.uint8)
            color_bgr[clean_depth > 0] = 200

            # 3. Add realistic sensor noise, dropouts, flying pixels
            # 3. Burst frames with realistic sensor noise, dropouts, flying pixels
            def _apply_noise(base_depth: np.ndarray) -> np.ndarray:
                d = base_depth.copy()
                if not add_noise:
                    return d
                valid_mask = d > 0
                z_vals = d[valid_mask]
                sigma_z = 0.001 * np.square(z_vals / 0.6)
                noise = self.rng.normal(0.0, sigma_z).astype(np.float32)
                d[valid_mask] = np.maximum(0.2, z_vals + noise)

                # 10% random dropouts
                drop_mask = (self.rng.uniform(0.0, 1.0, (self.height, self.width)) < 0.10) & valid_mask
                d[drop_mask] = 0.0

                # Edge flying pixels
                grad_x = np.abs(d[:, 1:] - d[:, :-1])
                jump = np.zeros_like(valid_mask)
                jump[:, 1:] = grad_x > 0.08
                fly_candidates = jump & (self.rng.uniform(0.0, 1.0, (self.height, self.width)) < 0.20)
                d[fly_candidates] += self.rng.uniform(0.02, 0.06, size=int(np.sum(fly_candidates)))
                return d

            burst_colors = [color_bgr.copy() for _ in range(burst_count)]
            burst_depths = [_apply_noise(clean_depth) for _ in range(burst_count)]
            depth_out = burst_depths[0]

            # 4. Compute ground-truth 3D keypoints in camera frame
            kpt_names = list(skel_stop.keys())
            kpt_pts_cam = []
            for name in kpt_names:
                pt_w = R_body @ skel_stop[name] + np.array([dx, 0.0, dz])
                pt_c = T_cam_world[:3, :3] @ pt_w + T_cam_world[:3, 3]
                kpt_pts_cam.append(pt_c)

            kpt_pts_cam_np = np.array(kpt_pts_cam, dtype=np.float32)
            kpts_cam = Keypoints3D(
                names=kpt_names,
                points=kpt_pts_cam_np,
                confidence=np.ones(len(kpt_names), dtype=np.float32),
                visibility=np.ones(len(kpt_names), dtype=np.float32),
                frame="camera",
            )
            gt_keypoints_list.append(kpts_cam)

            # Compute ground truth transform camera -> canonical
            T_canonical_world = np.eye(4, dtype=np.float64)
            T_canonical_world[:3, :3] = R_body.T
            T_canonical_world[:3, 3] = -R_body.T @ np.array([dx, 0.0, dz])
            T_canonical_cam_gt = T_canonical_world @ np.linalg.inv(T_cam_world)
            gt_canonical_yaw = (180.0 - actual_yaw + 360.0) % 360.0

            stop_data = StopData(
                stop_id=stop_idx,
                nominal_angle_deg=nominal_yaw,
                color=color_bgr,
                depth=depth_out,
                intrinsics=self.intrinsics,
                keypoints_camera=kpts_cam,
                raw_burst_colors=burst_colors,
                raw_burst_depths=burst_depths,
                gt_yaw_deg=gt_canonical_yaw,
                gt_translation_m=np.array([dx, 0.0, dz]),
                gt_T_canonical_camera=T_canonical_cam_gt,
            )
            stops.append(stop_data)

        return stops, gt_measurements, gt_keypoints_list


class GroundTruthKeypointDetector:
    """
    Mock 2D keypoint detector using ground-truth skeleton projections for fast, deterministic testing.
    """

    def __init__(self, keypoints_camera: Keypoints3D, intrinsics: o3d.camera.PinholeCameraIntrinsic):
        self.kpts = keypoints_camera
        self.intrinsics = intrinsics

    def detect(
        self, color_bgr: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
        fx, fy = self.intrinsics.get_focal_length()
        cx, cy = self.intrinsics.get_principal_point()

        pts_2d: Dict[str, Tuple[float, float]] = {}
        conf: Dict[str, float] = {}
        vis: Dict[str, float] = {}

        for name, pt, c, v in zip(self.kpts.names, self.kpts.points, self.kpts.confidence, self.kpts.visibility):
            if pt[2] > 0.1:
                u = float(fx * (pt[0] / pt[2]) + cx)
                v_px = float(fy * (pt[1] / pt[2]) + cy)
                pts_2d[name] = (u, v_px)
                conf[name] = float(c)
                vis[name] = float(v)

        return pts_2d, conf, vis
