"""
Unposing depth camera observations into the canonical A-pose frame.

Transforms point clouds from arbitrary acquisition stops (with estimated yaw,
translation, and arm abduction) back into the canonical rest-pose reference
frame of the fitted body model using inverse rigid poses and inverse soft
skinning transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import torch

from .synth import SimulatedStopData

logger = logging.getLogger(__name__)


@dataclass
class UnposedStopData:
    """Unposed point cloud and diagnostic residuals for a single acquisition stop."""
    stop_idx: int
    unposed_points: np.ndarray       # (N, 3) metres in canonical A-pose frame
    camera_view_dir: np.ndarray      # (3,) unit vector pointing from camera to subject in canonical frame
    point_to_model_distances: np.ndarray  # (N,) distance to nearest fitted canonical vertex (metres)
    mean_residual_mm: float
    median_residual_mm: float
    p90_residual_mm: float
    nearest_vertex_indices: np.ndarray    # (N,) nearest vertex on canonical model


class BodyUnposer:
    """
    Transforms multi-stop depth observations into the canonical rest-pose frame.
    """

    def __init__(
        self,
        canon_vertices: np.ndarray,      # (V, 3) in metres
        canon_joints: np.ndarray,        # (J, 3) in metres
        joint_names: List[str],
        raw_dict: Optional[dict] = None,
    ):
        self.canon_verts = np.asarray(canon_vertices, dtype=np.float32)
        self.canon_joints = np.asarray(canon_joints, dtype=np.float32)
        self.joint_names = list(joint_names)
        self.raw_dict = raw_dict or {}
        self.jmap = {name: i for i, name in enumerate(self.joint_names)}

        # Precompute canonical KDTree for fast correspondence
        self.canon_tree = cKDTree(self.canon_verts)

        # Precompute arm skinning weights and pivots
        self._init_arm_skinning()

    def _init_arm_skinning(self):
        v_weights = self.raw_dict.get("vertex_bone_weights")
        v_indices = self.raw_dict.get("vertex_bone_indices")
        bl = self.raw_dict.get("bone_labels")

        self.arm_weight_L = np.zeros(len(self.canon_verts), dtype=np.float32)
        self.arm_weight_R = np.zeros(len(self.canon_verts), dtype=np.float32)

        if v_weights is not None and v_indices is not None and bl is not None:
            if hasattr(v_weights, "cpu"):
                v_weights = v_weights.cpu().numpy()
            if hasattr(v_indices, "cpu"):
                v_indices = v_indices.cpu().numpy()

            for side, w_arr in [("L", self.arm_weight_L), ("R", self.arm_weight_R)]:
                arm_bones = [b for b in bl if f".{side}" in b and any(k in b.lower() for k in [
                    "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
                ])]
                b_idxs = set(bl.index(b) for b in arm_bones if b in bl)
                if not b_idxs:
                    continue
                for k in range(v_indices.shape[1]):
                    in_b = np.isin(v_indices[:, k], list(b_idxs))
                    w_arr += np.where(in_b, v_weights[:, k], 0.0)

        # Shoulder pivots
        sh_idx_L = self.jmap.get("upperarm01.L", self.jmap.get("UpperArm.L"))
        sh_idx_R = self.jmap.get("upperarm01.R", self.jmap.get("UpperArm.R"))
        self.pivot_L = self.canon_joints[sh_idx_L] if sh_idx_L is not None else np.array([0.165, -0.013, 0.427], dtype=np.float32)
        self.pivot_R = self.canon_joints[sh_idx_R] if sh_idx_R is not None else np.array([-0.165, -0.013, 0.427], dtype=np.float32)

    def unpose_points(
        self,
        points_world: np.ndarray,
        yaw_deg: float,
        translation_m: Union[np.ndarray, List[float]],
        arm_abd_L_deg: float = 0.0,
        arm_abd_R_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpose point cloud from world coordinates into canonical A-pose frame.

        Parameters
        ----------
        points_world : np.ndarray, shape (N, 3)
            Point cloud in world coordinates.
        yaw_deg : float
            Estimated yaw rotation around Z axis.
        translation_m : np.ndarray or list of float, length 3
            Estimated translation vector [tx, ty, tz].
        arm_abd_L_deg : float
            Left arm coronal abduction angle in degrees.
        arm_abd_R_deg : float
            Right arm coronal abduction angle in degrees.

        Returns
        -------
        p_canon : np.ndarray, shape (N, 3)
            Unposed points in canonical frame.
        distances : np.ndarray, shape (N,)
            Distance to nearest canonical mesh vertex (metres).
        nearest_indices : np.ndarray, shape (N,)
            Index of nearest vertex on canonical mesh.
        """
        if len(points_world) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

        pts = np.asarray(points_world, dtype=np.float32)
        trans = np.asarray(translation_m, dtype=np.float32)

        # 1. Invert translation
        p1 = pts - trans

        # 2. Invert yaw rotation around Z axis
        rad_yaw = np.radians(yaw_deg)
        cos_y = np.cos(-rad_yaw)
        sin_y = np.sin(-rad_yaw)
        rot_z_inv = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)
        p_unrotated = p1 @ rot_z_inv.T

        # 3. Invert arm abduction via inverse skinning transform
        # For points on arms, invert the coronal rotation around shoulder pivots.
        # Step 3a: Query nearest neighbor on canonical mesh to identify bone assignment
        dists, nn_idx = self.canon_tree.query(p_unrotated, k=1)

        p_canon = p_unrotated.copy()

        # Invert left arm abduction if present
        if abs(arm_abd_L_deg) > 1e-3:
            w_L = self.arm_weight_L[nn_idx]
            arm_mask_L = w_L > 0.02
            if np.any(arm_mask_L):
                pts_arm = p_unrotated[arm_mask_L]
                w = np.clip(w_L[arm_mask_L], 0.0, 1.0)
                rad = np.radians(arm_abd_L_deg)
                cos_a = np.cos(rad)
                sin_a = np.sin(rad)
                
                # Forward matrix was M = (1 - w) I + w R_y(alpha)
                # M = [[1 - w + w*cos, 0, w*sin], [0, 1, 0], [-w*sin, 0, 1 - w + w*cos]]
                # Inverse matrix M^-1:
                a = 1.0 - w + w * cos_a
                b = w * sin_a
                denom = a * a + b * b + 1e-8
                
                diff = pts_arm - self.pivot_L
                dx = diff[:, 0]
                dy = diff[:, 1]
                dz = diff[:, 2]
                
                unp_x = (a * dx - b * dz) / denom
                unp_y = dy
                unp_z = (b * dx + a * dz) / denom
                p_canon[arm_mask_L] = self.pivot_L + np.stack([unp_x, unp_y, unp_z], axis=-1)

        # Invert right arm abduction if present
        if abs(arm_abd_R_deg) > 1e-3:
            w_R = self.arm_weight_R[nn_idx]
            arm_mask_R = w_R > 0.02
            if np.any(arm_mask_R):
                pts_arm = p_unrotated[arm_mask_R]
                w = np.clip(w_R[arm_mask_R], 0.0, 1.0)
                rad = np.radians(arm_abd_R_deg)
                cos_a = np.cos(rad)
                sin_a = np.sin(rad)
                
                a = 1.0 - w + w * cos_a
                b = w * sin_a
                denom = a * a + b * b + 1e-8
                
                diff = pts_arm - self.pivot_R
                dx = diff[:, 0]
                dy = diff[:, 1]
                dz = diff[:, 2]
                
                unp_x = (a * dx - b * dz) / denom
                unp_y = dy
                unp_z = (b * dx + a * dz) / denom
                p_canon[arm_mask_R] = self.pivot_R + np.stack([unp_x, unp_y, unp_z], axis=-1)

        # Final distances and correspondences to canonical mesh
        final_dists, final_idx = self.canon_tree.query(p_canon, k=1)
        return p_canon, final_dists, final_idx

    def unpose_all_stops(
        self,
        stops: List[SimulatedStopData],
        per_stop_poses: List[Dict[str, Union[float, List[float]]]],
    ) -> List[UnposedStopData]:
        """
        Unpose all stops and compute diagnostic residuals against canonical mesh.
        """
        results = []
        for i, stop in enumerate(stops):
            pose = per_stop_poses[i]
            yaw = float(pose["yaw_deg"])
            trans = pose["translation_m"]
            abd_L = float(pose.get("arm_abd_L_deg", 0.0))
            abd_R = float(pose.get("arm_abd_R_deg", 0.0))

            pts_canon, dists, nn_idx = self.unpose_points(
                stop.point_cloud_world,
                yaw_deg=yaw,
                translation_m=trans,
                arm_abd_L_deg=abd_L,
                arm_abd_R_deg=abd_R,
            )

            # Camera view direction in canonical frame:
            # Camera was looking along +Y in world frame
            # Canonical view direction = R_z(-yaw) * [0, 1, 0]
            rad_yaw = np.radians(yaw)
            cam_view_dir = np.array([
                np.sin(rad_yaw),
                np.cos(rad_yaw),
                0.0,
            ], dtype=np.float32)
            cam_view_dir /= np.linalg.norm(cam_view_dir) + 1e-8

            dists_mm = dists * 1000.0
            results.append(UnposedStopData(
                stop_idx=stop.stop_idx,
                unposed_points=pts_canon,
                camera_view_dir=cam_view_dir,
                point_to_model_distances=dists,
                mean_residual_mm=float(np.mean(dists_mm)),
                median_residual_mm=float(np.median(dists_mm)),
                p90_residual_mm=float(np.percentile(dists_mm, 90)),
                nearest_vertex_indices=nn_idx,
            ))
        return results

