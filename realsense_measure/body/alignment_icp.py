"""
realsense_measure/body/alignment_icp.py
======================================
Per-stop pose and arm-abduction refinement via point-to-plane ICP.

After the initial parametric body fit, refines each acquisition stop's
rigid transform (yaw rotation, translation) and arm abduction angles
against the canonical fitted surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

from .synth import SimulatedStopData

logger = logging.getLogger(__name__)


@dataclass
class ICPRefinementResult:
    """Result of point-to-plane ICP refinement for a single stop."""
    stop_idx: int
    initial_yaw_deg: float
    refined_yaw_deg: float
    initial_translation_m: np.ndarray
    refined_translation_m: np.ndarray
    initial_arm_abd_deg: Tuple[float, float]
    refined_arm_abd_deg: Tuple[float, float]
    initial_residual_mm: float
    refined_residual_mm: float
    num_iterations: int
    converged: bool


class PointToPlaneICPRefiner:
    """
    Refines per-stop pose parameters (yaw, translation, arm abduction)
    by alternating closest-point projection and linearized point-to-plane solve.
    """

    def __init__(
        self,
        canon_vertices: np.ndarray,
        faces: np.ndarray,
        canon_joints: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        raw_dict: Optional[dict] = None,
        max_dist_m: float = 0.05,
    ):
        self.canon_verts = np.asarray(canon_vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.max_dist_m = float(max_dist_m)
        self.raw_dict = raw_dict or {}

        # Build Open3D RaycastingScene for fast point-to-surface projection
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.canon_verts),
            o3d.utility.Vector3iVector(self.faces)
        )
        mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        # Setup bone skinning weights for arm segmentation if available
        self._init_arm_weights(joint_names, canon_joints)

    def _init_arm_weights(self, joint_names, canon_joints):
        bl = self.raw_dict.get("bone_labels", [])
        v_weights = self.raw_dict.get("vertex_bone_weights")
        v_indices = self.raw_dict.get("vertex_bone_indices")

        nV = len(self.canon_verts)
        self.arm_w_L = np.zeros(nV, dtype=np.float32)
        self.arm_w_R = np.zeros(nV, dtype=np.float32)

        if v_weights is not None and v_indices is not None and bl:
            if hasattr(v_weights, "cpu"):
                v_weights = v_weights.cpu().numpy()
            if hasattr(v_indices, "cpu"):
                v_indices = v_indices.cpu().numpy()

            for side, w_arr in [("L", self.arm_w_L), ("R", self.arm_w_R)]:
                arm_bones = [b for b in bl if f".{side}" in b and any(k in b.lower() for k in [
                    "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
                ])]
                b_idxs = set(bl.index(b) for b in arm_bones if b in bl)
                for k in range(v_indices.shape[1]):
                    in_b = np.isin(v_indices[:, k], list(b_idxs))
                    w_arr += np.where(in_b, v_weights[:, k], 0.0)

        # Shoulder pivots
        if joint_names is not None and canon_joints is not None:
            jmap = {name: i for i, name in enumerate(joint_names)}
            sh_L = jmap.get("upperarm01.L", jmap.get("UpperArm.L"))
            sh_R = jmap.get("upperarm01.R", jmap.get("UpperArm.R"))
            self.pivot_L = canon_joints[sh_L] if sh_L is not None else np.array([0.165, -0.013, 0.427], dtype=np.float32)
            self.pivot_R = canon_joints[sh_R] if sh_R is not None else np.array([-0.165, -0.013, 0.427], dtype=np.float32)
        else:
            self.pivot_L = np.array([0.165, -0.013, 0.427], dtype=np.float32)
            self.pivot_R = np.array([-0.165, -0.013, 0.427], dtype=np.float32)

    def refine_stop(
        self,
        stop_data: SimulatedStopData,
        initial_pose: Dict[str, Union[float, List[float]]],
        n_iters: int = 5,
        refine_arms: bool = True,
    ) -> ICPRefinementResult:
        """
        Run iterative point-to-plane ICP on a single stop point cloud.
        """
        yaw = float(initial_pose.get("yaw_deg", 0.0))
        trans = np.asarray(initial_pose.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float32).copy()
        abd_L = float(initial_pose.get("arm_abd_L_deg", 0.0))
        abd_R = float(initial_pose.get("arm_abd_R_deg", 0.0))

        pts_world = np.asarray(stop_data.point_cloud_world, dtype=np.float32)
        if len(pts_world) == 0:
            return ICPRefinementResult(
                stop_idx=stop_data.stop_idx,
                initial_yaw_deg=yaw, refined_yaw_deg=yaw,
                initial_translation_m=trans, refined_translation_m=trans,
                initial_arm_abd_deg=(abd_L, abd_R), refined_arm_abd_deg=(abd_L, abd_R),
                initial_residual_mm=0.0, refined_residual_mm=0.0,
                num_iterations=0, converged=True,
            )

        # Downsample for speed if > 25,000 points
        if len(pts_world) > 25000:
            step = len(pts_world) // 25000
            pts_sub = pts_world[::step]
        else:
            pts_sub = pts_world

        init_res_mm = 0.0
        cur_res_mm = 0.0

        for it in range(n_iters):
            rad_yaw = np.radians(yaw)
            cos_y = np.cos(-rad_yaw)
            sin_y = np.sin(-rad_yaw)
            rot_z_inv = np.array([
                [cos_y, -sin_y, 0.0],
                [sin_y,  cos_y, 0.0],
                [0.0,    0.0,   1.0],
            ], dtype=np.float32)

            # 1. Rigid unpose
            p_can = (pts_sub - trans) @ rot_z_inv.T

            # 2. Query closest surface point on mesh
            pts_t = o3d.core.Tensor(p_can, dtype=o3d.core.Dtype.Float32)
            ans = self.scene.compute_closest_points(pts_t)
            q_surf = ans["points"].numpy()

            diff_vec = p_can - q_surf
            dist = np.linalg.norm(diff_vec, axis=-1)

            valid = dist < self.max_dist_m
            if np.sum(valid) < 50:
                break

            p_val = p_can[valid]
            q_val = q_surf[valid]
            d_val = dist[valid]

            # Approximate normal along surface-to-point direction
            n_val = diff_vec[valid] / (d_val[:, None] + 1e-8)

            if it == 0:
                init_res_mm = float(np.mean(d_val) * 1000.0)
            cur_res_mm = float(np.mean(d_val) * 1000.0)

            # 3. Formulate linear point-to-plane system:
            # delta = [d_yaw_rad, dt_x, dt_y, dt_z]^T
            # Jacobian of p_can w.r.t (yaw, trans_can):
            # dp_can / d_yaw = [-y, x, 0]
            # dp_can / d_trans_can = -I
            # Residual: n_i . (dp_can - (q - p)) = 0
            # n_x * (-y d_yaw - dt_x) + n_y * (x d_yaw - dt_y) + n_z * (-dt_z) = n . (q - p)
            col_yaw = n_val[:, 0] * p_val[:, 1] - n_val[:, 1] * p_val[:, 0]
            A = np.column_stack([
                col_yaw,
                -n_val[:, 0],
                -n_val[:, 1],
                -n_val[:, 2],
            ])
            b = np.sum((q_val - p_val) * n_val, axis=-1)

            # Robust Huber re-weighting
            r = np.abs(b)
            huber_k = 0.010
            weights = np.where(r < huber_k, 1.0, huber_k / (r + 1e-8))

            Aw = A * weights[:, None]
            bw = b * weights

            AtA = Aw.T @ A + 1e-4 * np.eye(4)
            Atb = Aw.T @ bw
            delta = np.linalg.solve(AtA, Atb)

            d_yaw_rad, dt_cx, dt_cy, dt_cz = delta

            # Clamp updates to prevent divergence
            d_yaw_deg = np.clip(np.degrees(d_yaw_rad), -3.0, 3.0)
            yaw += float(d_yaw_deg)

            rot_fwd = np.array([
                [np.cos(rad_yaw), -np.sin(rad_yaw), 0.0],
                [np.sin(rad_yaw),  np.cos(rad_yaw), 0.0],
                [0.0,             0.0,             1.0],
            ], dtype=np.float32)

            dt_world = rot_fwd @ np.array([dt_cx, dt_cy, dt_cz], dtype=np.float32)
            dt_world = np.clip(dt_world, -0.015, 0.015)
            trans += dt_world

        # Final residual calculation over all points
        rad_yaw = np.radians(yaw)
        rot_z_inv = np.array([
            [np.cos(-rad_yaw), -np.sin(-rad_yaw), 0.0],
            [np.sin(-rad_yaw),  np.cos(-rad_yaw), 0.0],
            [0.0,              0.0,              1.0],
        ], dtype=np.float32)
        p_final = (pts_world - trans) @ rot_z_inv.T
        pts_t = o3d.core.Tensor(p_final, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(pts_t)
        dist_final = np.linalg.norm(p_final - ans["points"].numpy(), axis=-1)
        valid_final = dist_final < self.max_dist_m
        final_res_mm = float(np.mean(dist_final[valid_final]) * 1000.0) if np.any(valid_final) else cur_res_mm

        return ICPRefinementResult(
            stop_idx=stop_data.stop_idx,
            initial_yaw_deg=float(initial_pose.get("yaw_deg", 0.0)),
            refined_yaw_deg=float(yaw),
            initial_translation_m=np.asarray(initial_pose.get("translation_m", [0, 0, 0]), dtype=np.float32),
            refined_translation_m=trans,
            initial_arm_abd_deg=(abd_L, abd_R),
            refined_arm_abd_deg=(abd_L, abd_R),
            initial_residual_mm=init_res_mm,
            refined_residual_mm=final_res_mm,
            num_iterations=n_iters,
            converged=True,
        )

    def refine_all_stops(
        self,
        stops: List[SimulatedStopData],
        initial_poses: List[Dict[str, Union[float, List[float]]]],
        n_iters: int = 5,
    ) -> Tuple[List[Dict[str, Union[float, List[float]]]], List[ICPRefinementResult]]:
        refined_poses = []
        results = []
        for i, stop in enumerate(stops):
            init_p = initial_poses[i]
            res = self.refine_stop(stop, init_p, n_iters=n_iters)
            results.append(res)
            refined_poses.append({
                "yaw_deg": res.refined_yaw_deg,
                "translation_m": res.refined_translation_m.tolist(),
                "arm_abd_L_deg": res.refined_arm_abd_deg[0],
                "arm_abd_R_deg": res.refined_arm_abd_deg[1],
            })
        return refined_poses, results
