"""
Unposing depth camera observations into the canonical A-pose frame.

Transforms point clouds from arbitrary acquisition stops (with estimated yaw,
translation, and arm abduction) back into the canonical rest-pose reference
frame of the fitted body model using inverse rigid poses, independent arm
identification, and inverse soft skinning transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import torch

from .synth import SimulatedStopData

logger = logging.getLogger(__name__)


@dataclass
class ArmUnposeDiagnostics:
    """Audit diagnostics for independent arm identification and unposing."""
    n_upperarm: int = 0
    n_forearm: int = 0
    n_hand: int = 0
    n_non_arm: int = 0
    n_unknown: int = 0
    n_rejected: int = 0
    n_arm_corrected: int = 0
    n_unposed_insufficient_evidence: int = 0
    landmark_confidence_mean: float = 1.0
    landmark_confidence_min: float = 1.0
    landmark_confidence_stats: Dict[str, float] = field(default_factory=dict)


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
    # Backward-compatible additions:
    arm_point_labels: Optional[np.ndarray] = None
    point_arm_weights: Optional[np.ndarray] = None
    diagnostics: Optional[ArmUnposeDiagnostics] = None
    n_upperarm: int = 0
    n_forearm: int = 0
    n_hand: int = 0
    n_non_arm: int = 0
    n_unknown: int = 0
    n_rejected: int = 0
    n_arm_corrected: int = 0
    n_unposed_insufficient_evidence: int = 0


def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized distance and projection parameter from points to 3D line segment [a, b]."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-8:
        d = np.linalg.norm(pts - a, axis=-1).astype(np.float32)
        return d, np.zeros(len(pts), dtype=np.float32)
    t = np.clip(np.sum((pts - a) * ab, axis=-1) / ab_len_sq, 0.0, 1.0).astype(np.float32)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(pts - proj, axis=-1).astype(np.float32)
    return d, t


def _parse_single_landmark(
    lm_dict: Optional[Dict[str, Any]],
    keys: List[str],
    depth_image: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[Dict[str, float]] = None,
    cam_world_pos: Optional[np.ndarray] = None,
    rot_z_inv: Optional[np.ndarray] = None,
    trans_m: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Extracts a 3D landmark in unrotated subject frame coordinates.
    Returns (pos_unrot, confidence, is_valid).
    """
    if not lm_dict:
        return None, 1.0, False

    val = None
    for k in keys:
        if k in lm_dict:
            val = lm_dict[k]
            break

    if val is None:
        return None, 1.0, False

    pos_raw = None
    conf = 1.0
    frame = "unrotated"

    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=np.float32)
        if len(arr) == 3:
            pos_raw = arr
    elif isinstance(val, dict):
        conf = float(val.get("confidence", val.get("visibility", val.get("score", 1.0))))
        frame = str(val.get("frame", "unrotated")).lower()
        if "xyz" in val or "pos" in val:
            pos_raw = np.asarray(val.get("xyz", val.get("pos")), dtype=np.float32)
        elif "x" in val and "y" in val and "z" in val:
            pos_raw = np.array([val["x"], val["y"], val["z"]], dtype=np.float32)
        elif ("pixel" in val or ("u" in val and "v" in val)) and depth_image is not None and camera_intrinsics is not None:
            pix = val.get("pixel", [val.get("u"), val.get("v")])
            u, v = float(pix[0]), float(pix[1])
            H, W = depth_image.shape[:2]
            c = int(np.clip(round(u), 0, W - 1))
            r = int(np.clip(round(v), 0, H - 1))
            patch = depth_image[max(0, r - 1):min(H, r + 2), max(0, c - 1):min(W, c + 2)]
            valid_z = patch[(patch > 0.1) & (patch < 5.0) & np.isfinite(patch)]
            if len(valid_z) > 0:
                z_m = float(np.median(valid_z))
                fx = camera_intrinsics["fx"]
                fy = camera_intrinsics["fy"]
                cx = camera_intrinsics["cx"]
                cy = camera_intrinsics["cy"]
                x_m = (u - cx) * z_m / fx
                y_m = (v - cy) * z_m / fy
                pos_raw = np.array([x_m, y_m, z_m], dtype=np.float32)
                frame = "camera"
            else:
                return None, conf, False
    elif hasattr(val, "x") and hasattr(val, "y") and hasattr(val, "z"):
        conf = float(getattr(val, "visibility", getattr(val, "confidence", 1.0)))
        pos_raw = np.array([val.x, val.y, val.z], dtype=np.float32)
        frame = "camera"

    if pos_raw is None or not np.all(np.isfinite(pos_raw)):
        return None, conf, False

    if conf < 0.50:
        return pos_raw, conf, False

    # Coordinate frame transformation to unrotated subject frame
    if frame == "camera":
        dist_cam = float(camera_intrinsics.get("distance_m", 1.1)) if camera_intrinsics else 1.1
        c_pos = cam_world_pos if cam_world_pos is not None else np.array([0.0, -dist_cam, 0.0], dtype=np.float32)
        pos_world = np.array([
            pos_raw[0] + c_pos[0],
            pos_raw[2] + c_pos[1],
            -pos_raw[1] + c_pos[2],
        ], dtype=np.float32)
        t = trans_m if trans_m is not None else np.zeros(3, dtype=np.float32)
        R_inv = rot_z_inv if rot_z_inv is not None else np.eye(3, dtype=np.float32)
        pos_unrot = (pos_world - t) @ R_inv.T
        return pos_unrot, conf, True
    elif frame == "world":
        t = trans_m if trans_m is not None else np.zeros(3, dtype=np.float32)
        R_inv = rot_z_inv if rot_z_inv is not None else np.eye(3, dtype=np.float32)
        pos_unrot = (pos_raw - t) @ R_inv.T
        return pos_unrot, conf, True
    else:
        return pos_raw, conf, True


class BodyUnposer:
    """
    Transforms multi-stop depth observations into the canonical rest-pose frame
    using independent arm identification before unposing, decoupling arm detection
    from nearest-pelvis/torso mesh proximity.
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
                    "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb", "metacarpal"
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

        # Canonical arm joints
        el_idx_L = self.jmap.get("lowerarm01.L", self.jmap.get("LowerArm.L"))
        wr_idx_L = self.jmap.get("wrist.L", self.jmap.get("Wrist.L"))
        self.canon_el_L = self.canon_joints[el_idx_L] if el_idx_L is not None else np.array([0.334, -0.019, 0.239], dtype=np.float32)
        self.canon_wr_L = self.canon_joints[wr_idx_L] if wr_idx_L is not None else np.array([0.450, -0.174, 0.121], dtype=np.float32)

        el_idx_R = self.jmap.get("lowerarm01.R", self.jmap.get("LowerArm.R"))
        wr_idx_R = self.jmap.get("wrist.R", self.jmap.get("Wrist.R"))
        self.canon_el_R = self.canon_joints[el_idx_R] if el_idx_R is not None else np.array([-0.334, -0.019, 0.239], dtype=np.float32)
        self.canon_wr_R = self.canon_joints[wr_idx_R] if wr_idx_R is not None else np.array([-0.450, -0.174, 0.121], dtype=np.float32)

        # Canonical trunk reference landmarks
        sp04_idx = self.jmap.get("spine04")
        sp01_idx = self.jmap.get("spine01")
        root_idx = self.jmap.get("root")
        hip_idx_L = self.jmap.get("upperleg01.L")
        hip_idx_R = self.jmap.get("upperleg01.R")
        kn_idx_L = self.jmap.get("lowerleg01.L")
        kn_idx_R = self.jmap.get("lowerleg01.R")

        self.canon_spine04 = self.canon_joints[sp04_idx] if sp04_idx is not None else np.array([0.0, 0.032, 0.068], dtype=np.float32)
        self.canon_spine01 = self.canon_joints[sp01_idx] if sp01_idx is not None else np.array([0.0, 0.035, 0.345], dtype=np.float32)
        self.canon_root = self.canon_joints[root_idx] if root_idx is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.canon_hip_L = self.canon_joints[hip_idx_L] if hip_idx_L is not None else np.array([0.102, 0.004, -0.007], dtype=np.float32)
        self.canon_hip_R = self.canon_joints[hip_idx_R] if hip_idx_R is not None else np.array([-0.102, 0.004, -0.007], dtype=np.float32)
        self.canon_kn_L = self.canon_joints[kn_idx_L] if kn_idx_L is not None else np.array([0.133, -0.028, -0.396], dtype=np.float32)
        self.canon_kn_R = self.canon_joints[kn_idx_R] if kn_idx_R is not None else np.array([-0.133, -0.028, -0.396], dtype=np.float32)

    def identify_arm_points(
        self,
        p_unrotated: np.ndarray,
        arm_abd_L_deg: float = 0.0,
        arm_abd_R_deg: float = 0.0,
        landmarks: Optional[Dict[str, Any]] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        points_cam: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        yaw_deg: float = 0.0,
        translation_m: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ArmUnposeDiagnostics]:
        """
        Independently classify points into arm regions (upperarm, forearm, hand) or non-arm/unknown
        BEFORE canonical unposing, using pose landmarks or estimated abduction segments.

        Returns
        -------
        arm_mask_L : np.ndarray (N,) bool
        arm_mask_R : np.ndarray (N,) bool
        weights_arm : np.ndarray (N,) float32
        point_labels : np.ndarray (N,) object string
        diagnostics : ArmUnposeDiagnostics
        """
        N = len(p_unrotated)
        if N == 0:
            return (
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=object),
                ArmUnposeDiagnostics(),
            )

        pts = np.asarray(p_unrotated, dtype=np.float32)
        trans = np.asarray(translation_m, dtype=np.float32) if translation_m is not None else np.zeros(3, dtype=np.float32)
        rad_y = np.radians(yaw_deg)
        rot_z_inv = np.array([
            [np.cos(-rad_y), -np.sin(-rad_y), 0.0],
            [np.sin(-rad_y),  np.cos(-rad_y), 0.0],
            [0.0,            0.0,           1.0],
        ], dtype=np.float32)

        dist_cam = float(camera_intrinsics.get("distance_m", 1.1)) if camera_intrinsics else 1.1
        cam_world_pos = np.array([0.0, -dist_cam, 0.0], dtype=np.float32)

        # 1. Parse independent landmarks
        conf_stats: Dict[str, float] = {}

        sh_L_pos, conf_sh_L, v_sh_L = _parse_single_landmark(
            landmarks, ["shoulder_L", "left_shoulder", "upperarm01.L", "Shoulder.L"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )
        el_L_pos, conf_el_L, v_el_L = _parse_single_landmark(
            landmarks, ["elbow_L", "left_elbow", "lowerarm01.L", "Elbow.L"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )
        wr_L_pos, conf_wr_L, v_wr_L = _parse_single_landmark(
            landmarks, ["wrist_L", "left_wrist", "wrist.L", "Wrist.L"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )

        sh_R_pos, conf_sh_R, v_sh_R = _parse_single_landmark(
            landmarks, ["shoulder_R", "right_shoulder", "upperarm01.R", "Shoulder.R"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )
        el_R_pos, conf_el_R, v_el_R = _parse_single_landmark(
            landmarks, ["elbow_R", "right_elbow", "lowerarm01.R", "Elbow.R"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )
        wr_R_pos, conf_wr_R, v_wr_R = _parse_single_landmark(
            landmarks, ["wrist_R", "right_wrist", "wrist.R", "Wrist.R"],
            depth_image, camera_intrinsics, cam_world_pos, rot_z_inv, trans,
        )

        for k, c in [
            ("shoulder_L", conf_sh_L), ("elbow_L", conf_el_L), ("wrist_L", conf_wr_L),
            ("shoulder_R", conf_sh_R), ("elbow_R", conf_el_R), ("wrist_R", conf_wr_R),
        ]:
            conf_stats[k] = c

        all_confs = list(conf_stats.values())
        mean_conf = float(np.mean(all_confs)) if all_confs else 1.0
        min_conf = float(np.min(all_confs)) if all_confs else 1.0

        # Construct posed landmarks (falling back to abduction angles if landmarks absent)
        # Left side
        has_lm_L = (landmarks is not None) and any(
            k in landmarks for k in ["shoulder_L", "left_shoulder", "elbow_L", "left_elbow", "wrist_L", "left_wrist"]
        )
        unreliable_L = has_lm_L and (not v_el_L or not v_wr_L)

        sh_L = sh_L_pos if (v_sh_L and sh_L_pos is not None) else self.pivot_L
        if v_el_L and el_L_pos is not None:
            el_L = el_L_pos
        else:
            rad_L = np.radians(arm_abd_L_deg)
            cos_L, sin_L = np.cos(rad_L), np.sin(rad_L)
            d_el = self.canon_el_L - self.pivot_L
            el_L = self.pivot_L + np.array([
                d_el[0] * cos_L + d_el[2] * sin_L,
                d_el[1],
                -d_el[0] * sin_L + d_el[2] * cos_L,
            ], dtype=np.float32)

        if v_wr_L and wr_L_pos is not None:
            wr_L = wr_L_pos
        else:
            rad_L = np.radians(arm_abd_L_deg)
            cos_L, sin_L = np.cos(rad_L), np.sin(rad_L)
            d_wr = self.canon_wr_L - self.pivot_L
            wr_L = self.pivot_L + np.array([
                d_wr[0] * cos_L + d_wr[2] * sin_L,
                d_wr[1],
                -d_wr[0] * sin_L + d_wr[2] * cos_L,
            ], dtype=np.float32)

        hand_axis_L = wr_L + 0.85 * (wr_L - el_L)

        # Right side
        has_lm_R = (landmarks is not None) and any(
            k in landmarks for k in ["shoulder_R", "right_shoulder", "elbow_R", "right_elbow", "wrist_R", "right_wrist"]
        )
        unreliable_R = has_lm_R and (not v_el_R or not v_wr_R)

        sh_R = sh_R_pos if (v_sh_R and sh_R_pos is not None) else self.pivot_R
        if v_el_R and el_R_pos is not None:
            el_R = el_R_pos
        else:
            rad_R = np.radians(arm_abd_R_deg)
            cos_R, sin_R = np.cos(rad_R), np.sin(rad_R)
            d_el = self.canon_el_R - self.pivot_R
            el_R = self.pivot_R + np.array([
                d_el[0] * cos_R + d_el[2] * sin_R,
                d_el[1],
                -d_el[0] * sin_R + d_el[2] * cos_R,
            ], dtype=np.float32)

        if v_wr_R and wr_R_pos is not None:
            wr_R = wr_R_pos
        else:
            rad_R = np.radians(arm_abd_R_deg)
            cos_R, sin_R = np.cos(rad_R), np.sin(rad_R)
            d_wr = self.canon_wr_R - self.pivot_R
            wr_R = self.pivot_R + np.array([
                d_wr[0] * cos_R + d_wr[2] * sin_R,
                d_wr[1],
                -d_wr[0] * sin_R + d_wr[2] * cos_R,
            ], dtype=np.float32)

        hand_axis_R = wr_R + 0.85 * (wr_R - el_R)

        # 2. Vectorized segment distances
        # Left arm
        d_ua_L, t_ua_L = _dist_to_segment(pts, sh_L, el_L)
        d_fa_L, t_fa_L = _dist_to_segment(pts, el_L, wr_L)
        d_hd_L, t_hd_L = _dist_to_segment(pts, wr_L, hand_axis_L)
        d_wr_L = np.linalg.norm(pts - wr_L, axis=-1)

        # Right arm
        d_ua_R, t_ua_R = _dist_to_segment(pts, sh_R, el_R)
        d_fa_R, t_fa_R = _dist_to_segment(pts, el_R, wr_R)
        d_hd_R, t_hd_R = _dist_to_segment(pts, wr_R, hand_axis_R)
        d_wr_R = np.linalg.norm(pts - wr_R, axis=-1)

        # Trunk reference
        d_spine, _ = _dist_to_segment(pts, self.canon_spine04, self.canon_spine01)
        d_pelv, _ = _dist_to_segment(pts, self.canon_root + np.array([0, 0, -0.09], dtype=np.float32), self.canon_root + np.array([0, 0, 0.05], dtype=np.float32))
        d_trunk = np.minimum(d_spine, d_pelv)

        d_thigh_L, _ = _dist_to_segment(pts, self.canon_hip_L, self.canon_kn_L)
        d_thigh_R, _ = _dist_to_segment(pts, self.canon_hip_R, self.canon_kn_R)

        # 3. Classify points
        point_labels = np.array(["non_arm"] * N, dtype=object)
        arm_weights = np.zeros(N, dtype=np.float32)
        n_unposed_insufficient = 0

        px = pts[:, 0]
        pz = pts[:, 2]

        for i in range(N):
            p = pts[i]
            # Left arm evaluation
            if px[i] > -0.015:
                # If left arm landmarks were unreliable/missing when required
                if unreliable_L:
                    # Check if point falls within left arm volume
                    if (d_hd_L[i] <= 0.12 or d_wr_L[i] <= 0.18 or d_fa_L[i] <= 0.10 or d_ua_L[i] <= 0.12) and px[i] > 0.03:
                        point_labels[i] = "unknown"
                        n_unposed_insufficient += 1
                        continue

                # Hand Left
                is_hand_L = (d_hd_L[i] <= 0.12 or d_wr_L[i] <= 0.18) and d_hd_L[i] < d_trunk[i] and d_hd_L[i] < d_thigh_L[i] and px[i] > 0.03
                if is_hand_L:
                    # Contact ambiguity check: if also close to pelvis/torso
                    if d_trunk[i] < 0.10 or d_thigh_L[i] < 0.08:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "hand_L"
                        arm_weights[i] = 1.0
                    continue

                # Forearm Left
                is_fa_L = d_fa_L[i] <= 0.10 and d_fa_L[i] < d_trunk[i] and d_fa_L[i] < d_thigh_L[i] and px[i] > 0.03
                if is_fa_L:
                    if d_trunk[i] < 0.10:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "forearm_L"
                        arm_weights[i] = 1.0
                    continue

                # Upperarm Left
                is_ua_L = d_ua_L[i] <= 0.12 and (d_ua_L[i] < d_trunk[i] or pz[i] > 0.32) and px[i] > 0.03
                if is_ua_L:
                    if d_trunk[i] < 0.09 and pz[i] < 0.32:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "upperarm_L"
                        arm_weights[i] = float(np.clip(0.2 + 0.8 * t_ua_L[i], 0.0, 1.0))
                    continue

            # Right arm evaluation
            if px[i] < 0.015:
                if unreliable_R:
                    if (d_hd_R[i] <= 0.12 or d_wr_R[i] <= 0.18 or d_fa_R[i] <= 0.10 or d_ua_R[i] <= 0.12) and px[i] < -0.03:
                        point_labels[i] = "unknown"
                        n_unposed_insufficient += 1
                        continue

                # Hand Right
                is_hand_R = (d_hd_R[i] <= 0.12 or d_wr_R[i] <= 0.18) and d_hd_R[i] < d_trunk[i] and d_hd_R[i] < d_thigh_R[i] and px[i] < -0.03
                if is_hand_R:
                    if d_trunk[i] < 0.10 or d_thigh_R[i] < 0.08:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "hand_R"
                        arm_weights[i] = 1.0
                    continue

                # Forearm Right
                is_fa_R = d_fa_R[i] <= 0.10 and d_fa_R[i] < d_trunk[i] and d_fa_R[i] < d_thigh_R[i] and px[i] < -0.03
                if is_fa_R:
                    if d_trunk[i] < 0.10:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "forearm_R"
                        arm_weights[i] = 1.0
                    continue

                # Upperarm Right
                is_ua_R = d_ua_R[i] <= 0.12 and (d_ua_R[i] < d_trunk[i] or pz[i] > 0.32) and px[i] < -0.03
                if is_ua_R:
                    if d_trunk[i] < 0.09 and pz[i] < 0.32:
                        point_labels[i] = "unknown"
                    else:
                        point_labels[i] = "upperarm_R"
                        arm_weights[i] = float(np.clip(0.2 + 0.8 * t_ua_R[i], 0.0, 1.0))
                    continue

        arm_mask_L = np.isin(point_labels, ["upperarm_L", "forearm_L", "hand_L"])
        arm_mask_R = np.isin(point_labels, ["upperarm_R", "forearm_R", "hand_R"])

        # Diagnostics counts
        n_ua = int(np.sum(np.isin(point_labels, ["upperarm_L", "upperarm_R"])))
        n_fa = int(np.sum(np.isin(point_labels, ["forearm_L", "forearm_R"])))
        n_hd = int(np.sum(np.isin(point_labels, ["hand_L", "hand_R"])))
        n_non = int(np.sum(point_labels == "non_arm"))
        n_unk = int(np.sum(point_labels == "unknown"))
        n_corr = int(np.sum(arm_mask_L | arm_mask_R))

        diag = ArmUnposeDiagnostics(
            n_upperarm=n_ua,
            n_forearm=n_fa,
            n_hand=n_hd,
            n_non_arm=n_non,
            n_unknown=n_unk,
            n_rejected=n_unk,
            n_arm_corrected=n_corr,
            n_unposed_insufficient_evidence=n_unposed_insufficient,
            landmark_confidence_mean=mean_conf,
            landmark_confidence_min=min_conf,
            landmark_confidence_stats=conf_stats,
        )

        return arm_mask_L, arm_mask_R, arm_weights, point_labels, diag

    def unpose_points(
        self,
        points_world: np.ndarray,
        yaw_deg: float,
        translation_m: Union[np.ndarray, List[float]],
        arm_abd_L_deg: float = 0.0,
        arm_abd_R_deg: float = 0.0,
        landmarks: Optional[Dict[str, Any]] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        points_cam: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
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
        landmarks : Optional dict
            Independently estimated pose landmarks.
        camera_intrinsics : Optional dict
            Camera parameters (fx, fy, cx, cy, distance_m).
        points_cam : Optional np.ndarray
            Points in camera frame.
        depth_image : Optional np.ndarray
            Depth image for landmark de-projection.

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
            self.last_arm_labels = np.zeros(0, dtype=object)
            self.last_arm_weights = np.zeros(0, dtype=np.float32)
            self.last_arm_diagnostics = ArmUnposeDiagnostics()
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

        # 3. Independent Arm Identification (BEFORE canonical unposing)
        arm_mask_L, arm_mask_R, weights_arm, point_labels, diag = self.identify_arm_points(
            p_unrotated,
            arm_abd_L_deg=arm_abd_L_deg,
            arm_abd_R_deg=arm_abd_R_deg,
            landmarks=landmarks,
            camera_intrinsics=camera_intrinsics,
            points_cam=points_cam,
            depth_image=depth_image,
            yaw_deg=yaw_deg,
            translation_m=trans,
        )

        self.last_arm_labels = point_labels
        self.last_arm_weights = weights_arm
        self.last_arm_diagnostics = diag

        p_canon = p_unrotated.copy()

        # Invert left arm abduction for points independently identified as left arm
        if abs(arm_abd_L_deg) > 1e-3 and np.any(arm_mask_L):
            pts_arm = p_unrotated[arm_mask_L]
            w = weights_arm[arm_mask_L]
            rad = np.radians(arm_abd_L_deg)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            # Inverse linear blend skinning around left shoulder pivot:
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

        # Invert right arm abduction for points independently identified as right arm
        if abs(arm_abd_R_deg) > 1e-3 and np.any(arm_mask_R):
            pts_arm = p_unrotated[arm_mask_R]
            w = weights_arm[arm_mask_R]
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
            landmarks = pose.get("landmarks", getattr(stop, "pose_landmarks", None))

            pts_canon, dists, nn_idx = self.unpose_points(
                stop.point_cloud_world,
                yaw_deg=yaw,
                translation_m=trans,
                arm_abd_L_deg=abd_L,
                arm_abd_R_deg=abd_R,
                landmarks=landmarks,
                camera_intrinsics=getattr(stop, "intrinsics", None),
                points_cam=getattr(stop, "point_cloud", None),
                depth_image=getattr(stop, "depth_image", None),
            )

            # Camera view direction in canonical frame:
            rad_yaw = np.radians(yaw)
            cam_view_dir = np.array([
                np.sin(rad_yaw),
                np.cos(rad_yaw),
                0.0,
            ], dtype=np.float32)
            cam_view_dir /= np.linalg.norm(cam_view_dir) + 1e-8

            dists_mm = dists * 1000.0
            diag = getattr(self, "last_arm_diagnostics", ArmUnposeDiagnostics())
            arm_labels = getattr(self, "last_arm_labels", None)
            arm_weights = getattr(self, "last_arm_weights", None)

            results.append(UnposedStopData(
                stop_idx=stop.stop_idx,
                unposed_points=pts_canon,
                camera_view_dir=cam_view_dir,
                point_to_model_distances=dists,
                mean_residual_mm=float(np.mean(dists_mm)),
                median_residual_mm=float(np.median(dists_mm)),
                p90_residual_mm=float(np.percentile(dists_mm, 90)),
                nearest_vertex_indices=nn_idx,
                arm_point_labels=arm_labels,
                point_arm_weights=arm_weights,
                diagnostics=diag,
                n_upperarm=diag.n_upperarm,
                n_forearm=diag.n_forearm,
                n_hand=diag.n_hand,
                n_non_arm=diag.n_non_arm,
                n_unknown=diag.n_unknown,
                n_rejected=diag.n_rejected,
                n_arm_corrected=diag.n_arm_corrected,
                n_unposed_insufficient_evidence=diag.n_unposed_insufficient_evidence,
            ))
        return results
