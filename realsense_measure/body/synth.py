"""
realsense_measure/body/synth.py
================================
Synthetic ground truth and virtual depth camera simulation for Step 2.

Features:
  1. Truth Body Generation:
     - Adult phenotype sampled from population prior.
     - Smooth 3D Gaussian Random Field (GRF) displacement field:
       correlation length ~5 cm, std 5–8 mm.
     - Local 15 mm reduction on left mid-thigh for rehab test.
     - True measurements evaluated on the displaced mesh.
  2. 4-Stop Virtual Acquisition (0°, 90°, 180°, 270°):
     - Imperfections: yaw error ±5°, translation ±3 cm, arm abduction ±10°.
  3. Virtual Depth Camera (Open3D RaycastingScene):
     - Intel RealSense D455 depth FOV: 87° × 58°.
     - Landscape at 1.6 m, Portrait at 1.1 m.
     - Quadratic depth noise: sigma(z) = sigma_ref * (z / 0.6)^2
       with typical (1.0 mm) and pessimistic (2.0 mm).
     - 10% random dropouts.
     - Edge flying pixels along depth boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import scipy.ndimage
import torch

from .measure import Measurement, measure_mesh
from .model import BodyOutput

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    mode: str                    # 'landscape' or 'portrait'
    distance_m: float            # 1.6 for landscape, 1.1 for portrait
    hfov_deg: float              # 87 for landscape, 58 for portrait
    vfov_deg: float              # 58 for landscape, 87 for portrait
    width: int                   # 640 or 480
    height: int                  # 480 or 640
    sigma_ref_mm: float          # 1.0 (typical) or 2.0 (pessimistic)
    dropout_rate: float = 0.10   # 10% random dropouts
    edge_flying_pixels: bool = True

    @classmethod
    def landscape(cls, pessimistic: bool = False) -> CameraConfig:
        return cls(
            mode="landscape",
            distance_m=1.6,
            hfov_deg=87.0,
            vfov_deg=58.0,
            width=640,
            height=480,
            sigma_ref_mm=2.0 if pessimistic else 1.0,
        )

    @classmethod
    def portrait(cls, pessimistic: bool = False) -> CameraConfig:
        return cls(
            mode="portrait",
            distance_m=1.1,
            hfov_deg=58.0,
            vfov_deg=87.0,
            width=480,
            height=640,
            sigma_ref_mm=2.0 if pessimistic else 1.0,
        )


@dataclass
class SimulatedStopData:
    stop_idx: int
    nominal_yaw_deg: float
    true_yaw_deg: float
    true_translation_m: np.ndarray        # (3,)
    true_arm_abduction_deg: Tuple[float, float]  # (left, right)
    depth_image: np.ndarray               # (H, W) float32 in metres
    foreground_mask: np.ndarray           # (H, W) bool
    point_cloud: np.ndarray               # (N, 3) metres in camera frame
    point_cloud_world: np.ndarray         # (N, 3) metres in world frame
    intrinsics: Dict[str, float]


def generate_smooth_displacement_field(
    vertices: np.ndarray,
    correlation_length_m: float = 0.05,
    std_mm: float = 6.5,
    seed: Optional[int] = None,
    n_modes: int = 64,
) -> np.ndarray:
    """
    Generate a 3D smooth Gaussian Random Field displacement field over mesh vertices
    using Random Fourier Features.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices (V, 3) in metres.
    correlation_length_m : float
        Spatial correlation length in metres (~5 cm).
    std_mm : float
        Target standard deviation in millimetres (5–8 mm).
    seed : Optional[int]
        Random seed.
    n_modes : int
        Number of Fourier wavevectors.

    Returns
    -------
    disp : np.ndarray
        Displacement field (V, 3) in metres.
    """
    rng = np.random.default_rng(seed)
    V = len(vertices)

    # Wavevectors sampled from isotropic Gaussian
    # Power spectrum corresponding to Gaussian covariance
    sigma_k = 1.0 / correlation_length_m
    K = rng.normal(0.0, sigma_k, size=(n_modes, 3))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_modes, 3))
    amplitudes = rng.normal(0.0, 1.0, size=(n_modes, 3))

    # Evaluate sum_m A_m * cos(K_m . x + phi_m)
    # Projection: (V, 3) @ (3, n_modes) -> (V, n_modes)
    proj = vertices @ K.T
    disp = np.zeros((V, 3), dtype=np.float64)
    for c in range(3):
        # (V, n_modes)
        cos_terms = np.cos(proj + phases[:, c])
        disp[:, c] = cos_terms @ amplitudes[:, c]

    # Normalize standard deviation across vertices
    current_std = np.std(disp)
    if current_std > 1e-8:
        target_std_m = std_mm * 1e-3
        disp = disp * (target_std_m / current_std)

    return disp


def apply_rehab_mid_thigh_atrophy(
    vertices: np.ndarray,
    joints: np.ndarray,
    joint_names: List[str],
    circumference_reduction_mm: float = 15.0,
    axial_sigma_m: float = 0.04,
) -> np.ndarray:
    """
    Rehab test: Apply a local 15 mm circumference reduction to the true left mid-thigh.
    Displaces vertices inward toward the femur axis.
    """
    jmap = {name: i for i, name in enumerate(joint_names)}
    hip_idx = jmap.get("upperleg01.L")
    knee_idx = jmap.get("lowerleg01.L")
    if hip_idx is None or knee_idx is None:
        return vertices

    hip = joints[hip_idx]
    knee = joints[knee_idx]
    mid_thigh = 0.5 * (hip + knee)
    axis = knee - hip
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-4:
        return vertices
    axis_unit = axis / axis_len

    # Radial reduction delta_r = delta_C / (2 * pi)
    delta_r_m = (circumference_reduction_mm * 1e-3) / (2.0 * np.pi)

    # For each vertex, compute projection onto femur axis
    diff = vertices - mid_thigh
    axial_dist = np.dot(diff, axis_unit)
    rad_vec = diff - np.outer(axial_dist, axis_unit)
    rad_dist = np.linalg.norm(rad_vec, axis=1)

    # Select left thigh vertices (axial within ±10 cm, radial < 15 cm, X > 0)
    thigh_mask = (np.abs(axial_dist) < 0.12) & (rad_dist < 0.16) & (vertices[:, 0] > 0.0)

    displaced_verts = vertices.copy()
    if np.any(thigh_mask):
        weight = np.exp(-0.5 * (axial_dist[thigh_mask] / axial_sigma_m) ** 2)
        unit_inward = -rad_vec[thigh_mask] / (rad_dist[thigh_mask, None] + 1e-6)
        displaced_verts[thigh_mask] += unit_inward * (delta_r_m * weight[:, None])

    return displaced_verts


def apply_arm_abduction(
    vertices: np.ndarray,
    joints: np.ndarray,
    joint_names: List[str],
    raw_dict: dict,
    left_deg: float,
    right_deg: float,
) -> np.ndarray:
    """
    Apply coronal arm abduction rotations to left and right arms around shoulder joints.
    """
    if abs(left_deg) < 1e-3 and abs(right_deg) < 1e-3:
        return vertices

    v_weights = raw_dict.get("vertex_bone_weights")
    v_indices = raw_dict.get("vertex_bone_indices")
    bl = raw_dict.get("bone_labels")
    if v_weights is None or v_indices is None or bl is None:
        return vertices

    if hasattr(v_weights, "cpu"):
        v_weights = v_weights.cpu().numpy()
    if hasattr(v_indices, "cpu"):
        v_indices = v_indices.cpu().numpy()

    jmap = {name: i for i, name in enumerate(joint_names)}
    out_verts = vertices.copy()

    for side, deg in [("L", left_deg), ("R", right_deg)]:
        if abs(deg) < 1e-3:
            continue
        sh_idx = jmap.get(f"upperarm01.{side}")
        if sh_idx is None:
            continue
        pivot = joints[sh_idx]

        # Arm bones for this side
        arm_bones = [b for b in bl if f".{side}" in b and any(k in b.lower() for k in [
            "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
        ])]
        b_indices = set(bl.index(b) for b in arm_bones if b in bl)
        if not b_indices:
            continue

        weight = np.zeros(len(vertices), dtype=np.float32)
        for k in range(v_indices.shape[1]):
            in_b = np.isin(v_indices[:, k], list(b_indices))
            weight += np.where(in_b, v_weights[:, k], 0.0)

        # Coronal rotation around Y axis:
        # For left arm (+X): positive abduction tilts arm upward (+Z, -X)
        rad = np.radians(deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        # R_y rotation in X-Z:
        # x' = x*cos + z*sin, z' = -x*sin + z*cos
        diff = vertices - pivot
        x_rot = diff[:, 0] * cos_a + diff[:, 2] * sin_a
        z_rot = -diff[:, 0] * sin_a + diff[:, 2] * cos_a
        rot_diff = np.stack([x_rot, diff[:, 1], z_rot], axis=-1)

        # Blend by skinning weight
        w_expanded = np.clip(weight, 0.0, 1.0)[:, None]
        out_verts = out_verts + (rot_diff - diff) * w_expanded

    return out_verts


class VirtualDepthSensor:
    """
    Simulates an Intel RealSense D455 depth camera using Open3D RaycastingScene.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.fx = (config.width / 2.0) / np.tan(np.radians(config.hfov_deg / 2.0))
        self.fy = (config.height / 2.0) / np.tan(np.radians(config.vfov_deg / 2.0))
        self.cx = config.width / 2.0
        self.cy = config.height / 2.0

        # Precompute camera ray directions
        u, v = np.meshgrid(np.arange(config.width), np.arange(config.height))
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        # Camera convention: X right, Y down, Z forward
        # In our world convention: X lateral, Y forward, Z up
        # When camera is placed at (0, -distance, z_cam) looking at +Y:
        # Camera X -> World +X
        # Camera Y -> World -Z
        # Camera Z -> World +Y
        self.ray_dirs_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
        self.ray_dirs_cam /= np.linalg.norm(self.ray_dirs_cam, axis=-1, keepdims=True)

    def render_depth(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raycast mesh and add realistic depth sensor noise.

        Returns
        -------
        depth_m : np.ndarray (H, W) float32
        mask : np.ndarray (H, W) bool
        pcd_cam : np.ndarray (N, 3) float32
        """
        rng = np.random.default_rng(seed)

        # Fixed calibrated tripod position in world coordinates (looking at +Y)
        cam_pos_world = np.array([0.0, -self.config.distance_m, 0.0], dtype=np.float32)

        # World ray directions
        # Camera X -> World X, Camera Y -> World -Z, Camera Z -> World Y
        dirs_world = np.stack([
            self.ray_dirs_cam[:, :, 0],
            self.ray_dirs_cam[:, :, 2],
            -self.ray_dirs_cam[:, :, 1],
        ], axis=-1).astype(np.float32)

        origins_world = np.broadcast_to(
            cam_pos_world, dirs_world.shape
        ).copy().astype(np.float32)

        # Build Open3D RaycastingScene
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        rays = np.concatenate([origins_world, dirs_world], axis=-1)
        rays_tensor = o3d.core.Tensor(rays)
        ans = scene.cast_rays(rays_tensor)
        t_hit = ans["t_hit"].numpy()

        valid = np.isfinite(t_hit) & (t_hit < 5.0) & (t_hit > 0.1)

        # True depth Z in camera frame = t_hit * cos_angle = t_hit * ray_dirs_cam[:,:,2]
        true_depth = np.where(valid, t_hit * self.ray_dirs_cam[:, :, 2], 0.0).astype(np.float32)

        # -------------------------------------------------------------
        # Realistic Quadratic Sensor Noise: sigma(z) = sigma_ref * (z / 0.6)^2
        # -------------------------------------------------------------
        sigma_ref_m = self.config.sigma_ref_mm * 1e-3
        sigma_z = sigma_ref_m * ((np.maximum(true_depth, 0.3) / 0.6) ** 2)
        noise = rng.normal(0.0, sigma_z)
        noisy_depth = np.where(valid, true_depth + noise, 0.0).astype(np.float32)

        # -------------------------------------------------------------
        # 10% Random Dropouts
        # -------------------------------------------------------------
        drop_mask = rng.uniform(0.0, 1.0, size=noisy_depth.shape) < self.config.dropout_rate
        noisy_depth[drop_mask & valid] = 0.0
        final_valid = valid & (~drop_mask) & (noisy_depth > 0.1)

        # -------------------------------------------------------------
        # Edge Flying Pixels
        # -------------------------------------------------------------
        if self.config.edge_flying_pixels and np.any(final_valid):
            # Compute depth gradient
            grad_y, grad_x = np.gradient(noisy_depth)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_pixels = (grad_mag > 0.08) & final_valid
            # Corrupt 30% of edge pixels with intermediate sightline depth
            corrupt = edge_pixels & (rng.uniform(0.0, 1.0, size=edge_pixels.shape) < 0.3)
            noisy_depth[corrupt] += rng.uniform(0.02, 0.08, size=np.sum(corrupt)).astype(np.float32)

        # Back-project to 3D camera point cloud
        v_idx, u_idx = np.where(final_valid)
        z_pts = noisy_depth[v_idx, u_idx]
        x_pts = (u_idx - self.cx) * z_pts / self.fx
        y_pts = (v_idx - self.cy) * z_pts / self.fy
        pcd_cam = np.stack([x_pts, y_pts, z_pts], axis=-1)

        return noisy_depth, final_valid, pcd_cam


def check_arm_torso_contact(
    vertices: np.ndarray,
    raw_dict: dict,
    contact_threshold_m: float = 0.015,
) -> Tuple[bool, float]:
    """
    Check if arms touch or graze the lateral torso.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
    raw_dict : dict
    contact_threshold_m : float
        Proximity threshold in metres (default 15 mm).

    Returns
    -------
    is_contact : bool
    min_dist_m : float
    """
    v_weights = raw_dict.get("vertex_bone_weights")
    v_indices = raw_dict.get("vertex_bone_indices")
    bl = raw_dict.get("bone_labels")
    if v_weights is None or v_indices is None or bl is None:
        return False, 999.0

    if hasattr(v_weights, "cpu"):
        v_weights = v_weights.cpu().numpy()
    if hasattr(v_indices, "cpu"):
        v_indices = v_indices.cpu().numpy()

    arm_bones = [b for b in bl if any(k in b.lower() for k in [
        "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
    ])]
    b_indices = set(bl.index(b) for b in arm_bones if b in bl)
    if not b_indices:
        return False, 999.0

    w_arm = np.zeros(len(vertices), dtype=np.float32)
    for k in range(v_indices.shape[1]):
        in_b = np.isin(v_indices[:, k], list(b_indices))
        w_arm += np.where(in_b, v_weights[:, k], 0.0)

    # Distal arm (lowerarm, wrist, hand, fingers) and mid-lower torso (Z between -0.25 and 0.25)
    # where hanging arms collide with hips and waist
    distal_bones = [b for b in bl if any(k in b.lower() for k in [
        "lowerarm", "wrist", "hand", "finger", "thumb"
    ])]
    distal_indices = set(bl.index(b) for b in distal_bones if b in bl)
    w_distal = np.zeros(len(vertices), dtype=np.float32)
    for k in range(v_indices.shape[1]):
        in_d = np.isin(v_indices[:, k], list(distal_indices))
        w_distal += np.where(in_d, v_weights[:, k], 0.0)

    arm_mask = (w_distal > 0.4) & (vertices[:, 2] < 0.25)
    torso_mask = (w_arm < 0.05) & (vertices[:, 2] > -0.25) & (vertices[:, 2] < 0.25)

    if not np.any(arm_mask) or not np.any(torso_mask):
        return False, 999.0

    from scipy.spatial import cKDTree
    torso_tree = cKDTree(vertices[torso_mask])
    dists, _ = torso_tree.query(vertices[arm_mask])
    min_d = float(np.min(dists)) if len(dists) > 0 else 999.0
    return bool(min_d < contact_threshold_m), min_d


class SyntheticTruthSubject:
    """
    Generates a synthetic ground-truth subject with non-ideal displacement field,
    true anthropometric measurements, and multi-stop depth simulations.
    """

    def __init__(
        self,
        backend,
        prior,
        subject_id: int = 0,
        seed: int = 42,
        apply_rehab_atrophy: bool = False,
        displacement_std_mm: Optional[float] = None,
        phenotypes: Optional[Dict[str, float]] = None,
    ):
        self.subject_id = subject_id
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Sample adult phenotype from prior or use provided
        if phenotypes is not None:
            self.phenotypes = {k: float(v) for k, v in phenotypes.items()}
        else:
            pheno_dict = prior.sample_adult(n=1, seed=seed)
            self.phenotypes = {k: float(v[0]) for k, v in pheno_dict.items()}

        # Generate canonical rest-pose body from backend
        out = backend.rest_pose(phenotypes=self.phenotypes)
        self.base_verts = out.numpy_vertices(0)
        self.base_joints = out.numpy_joints(0)
        self.faces = out.faces.cpu().numpy()
        self.joint_names = out.joint_names
        self.raw_dict = out.raw if hasattr(out, "raw") else {}

        # Apply smooth random displacement field (std 5–8 mm, correlation ~5 cm)
        if displacement_std_mm is not None and displacement_std_mm == 0.0:
            self.disp_field = np.zeros_like(self.base_verts)
        else:
            disp_std_mm = displacement_std_mm if displacement_std_mm is not None else self.rng.uniform(5.5, 7.5)
            self.disp_field = generate_smooth_displacement_field(
                self.base_verts,
                correlation_length_m=0.05,
                std_mm=disp_std_mm,
                seed=seed + 100,
            )
        self.true_verts = self.base_verts + self.disp_field

        # Optional rehab atrophy
        if apply_rehab_atrophy:
            self.true_verts = apply_rehab_mid_thigh_atrophy(
                self.true_verts,
                self.base_joints,
                self.joint_names,
                circumference_reduction_mm=15.0,
            )

        # True joints track local vertices
        self.true_joints = self.base_joints.copy()
        for j in range(len(self.true_joints)):
            dists = np.linalg.norm(self.base_verts - self.base_joints[j], axis=1)
            near_idx = np.where(dists < 0.05)[0]
            if len(near_idx) > 0:
                self.true_joints[j] += np.mean(self.disp_field[near_idx], axis=0)

        # True measurements computed on the displaced mesh
        self.true_measurements = measure_mesh(
            self.true_verts,
            self.true_joints,
            self.faces,
            self.joint_names,
            self.raw_dict,
        )

    def simulate_stops(
        self,
        camera_config: CameraConfig,
        n_stops: int = 4,
        imperfections: bool = True,
        nominal_arm_abduction_deg: Optional[float] = None,
    ) -> List[SimulatedStopData]:
        """
        Simulate depth camera acquisitions across 1, 2, or 4 yaw stops.

        Stops:
          - n_stops=1: [0°]
          - n_stops=2: [0°, 180°]
          - n_stops=4: [0°, 90°, 180°, 270°]
        """
        nominal_yaws = [0.0, 90.0, 180.0, 270.0] if n_stops == 4 else (
            [0.0, 180.0] if n_stops == 2 else [0.0]
        )

        sensor = VirtualDepthSensor(camera_config)
        stops_data: List[SimulatedStopData] = []

        for i, nom_yaw in enumerate(nominal_yaws):
            stop_seed = self.seed + 1000 + i * 37
            stop_rng = np.random.default_rng(stop_seed)

            # Per-stop imperfections:
            if imperfections:
                yaw_err = float(stop_rng.uniform(-5.0, 5.0))
                true_yaw = nom_yaw + yaw_err
                true_trans = stop_rng.uniform(-0.03, 0.03, size=3)
                if nominal_arm_abduction_deg is not None:
                    arm_abd_L = float(nominal_arm_abduction_deg + stop_rng.uniform(-3.0, 3.0))
                    arm_abd_R = float(nominal_arm_abduction_deg + stop_rng.uniform(-3.0, 3.0))
                else:
                    arm_abd_L = float(stop_rng.uniform(-10.0, 10.0))
                    arm_abd_R = float(stop_rng.uniform(-10.0, 10.0))
            else:
                yaw_err = 0.0
                true_yaw = nom_yaw
                true_trans = np.zeros(3, dtype=np.float32)
                arm_abd_L = float(nominal_arm_abduction_deg) if nominal_arm_abduction_deg is not None else 0.0
                arm_abd_R = float(nominal_arm_abduction_deg) if nominal_arm_abduction_deg is not None else 0.0

            # 1. Apply arm abduction to true mesh
            stop_verts = apply_arm_abduction(
                self.true_verts,
                self.true_joints,
                self.joint_names,
                self.raw_dict,
                left_deg=arm_abd_L,
                right_deg=arm_abd_R,
            )

            # 2. Rotate around world Z by true_yaw (metres, centered at root)
            rad = np.radians(true_yaw)
            cos_y = np.cos(rad)
            sin_y = np.sin(rad)
            rot_z = np.array([
                [cos_y, -sin_y, 0.0],
                [sin_y,  cos_y, 0.0],
                [0.0,    0.0,   1.0],
            ])
            stop_verts = stop_verts @ rot_z.T

            # 3. Apply translation
            stop_verts = stop_verts + true_trans

            # 4. Raycast with virtual depth camera
            depth_img, mask, pcd_cam = sensor.render_depth(
                stop_verts, self.faces, seed=stop_seed
            )

            # 5. Transform pcd_cam back to world coordinates
            # Camera frame -> World frame:
            # cam X -> world X, cam Y -> world -Z, cam Z -> world Y
            # with calibrated camera at (0, -dist, 0.0)
            cam_world_pos = np.array([0.0, -camera_config.distance_m, 0.0])
            pcd_world = np.stack([
                pcd_cam[:, 0] + cam_world_pos[0],
                pcd_cam[:, 2] + cam_world_pos[1],
                -pcd_cam[:, 1] + cam_world_pos[2],
            ], axis=-1)

            floor_z = float(self.true_verts[:, 2].min())
            stops_data.append(SimulatedStopData(
                stop_idx=i,
                nominal_yaw_deg=nom_yaw,
                true_yaw_deg=true_yaw,
                true_translation_m=true_trans,
                true_arm_abduction_deg=(arm_abd_L, arm_abd_R),
                depth_image=depth_img,
                foreground_mask=mask,
                point_cloud=pcd_cam,
                point_cloud_world=pcd_world,
                intrinsics={
                    "fx": sensor.fx,
                    "fy": sensor.fy,
                    "cx": sensor.cx,
                    "cy": sensor.cy,
                    "width": camera_config.width,
                    "height": camera_config.height,
                    "distance_m": camera_config.distance_m,
                    "floor_z": floor_z,
                },
            ))

        return stops_data

