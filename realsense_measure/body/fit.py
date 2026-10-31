"""
realsense_measure/body/fit.py
==============================
Optimization engine to fit Anny body model to simulated partial depth data.

Parameters:
  - Shared across stops (6 phenotypes):
    [gender, age, muscle, weight, height, proportions] in [0, 1]
  - Per stop s:
    yaw (1), translation (3), arm abduction (2: left, right independently)

Loss formulation (all terms documented with explicit weights):
  (a) Data-to-surface distance with KDTree correspondences on detached model samples,
      robust Geman-McClure loss: rho(r) = r^2 / (r^2 + c^2). (w_data = 2.0)
  (b) Visible model-to-depth term: backface-culled, z-buffer occlusion tested,
      compared against measured depth at valid pixels with robust residual. (w_vis = 2.0)
  (c) Free-space term: penalises model surface in front of measured depth by more
      than a margin (15 mm). (w_free = 10.0)
  (d) Silhouette distance transform penalty for points projecting outside. (w_sil = 50.0)
  (e) Shape prior from AdultShapePrior (negative log-likelihood). (w_prior = 0.02)
  (f) Optional floor-plane constraint: model's lowest vertices on the floor plane. (w_floor = 100.0)
  (g) Optional known-height constraint. (w_height = 1.0)

Ablation support: any loss term can be selectively removed via `ablation_remove_term`.
Strictly NO symmetry terms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
import scipy.spatial
import torch
import torch.nn as nn
import torch.optim as optim

from .measure import Measurement, measure_mesh
from .prior import ADULT_AGE_NEUTRAL, AdultShapePrior
from .synth import CameraConfig, SimulatedStopData

logger = logging.getLogger(__name__)

PHENOTYPE_NAMES = ["gender", "age", "muscle", "weight", "height", "proportions"]


@dataclass
class LossBreakdown:
    loss_data: float
    loss_vis: float
    loss_free: float
    loss_sil: float
    loss_prior: float
    loss_floor: float
    loss_height: float
    total_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss_data": self.loss_data,
            "loss_vis": self.loss_vis,
            "loss_free": self.loss_free,
            "loss_sil": self.loss_sil,
            "loss_prior": self.loss_prior,
            "loss_floor": self.loss_floor,
            "loss_height": self.loss_height,
            "total_loss": self.total_loss,
        }


@dataclass
class FitResult:
    fitted_phenotypes: Dict[str, float]
    measurements: Dict[str, Measurement]
    per_stop_poses: List[Dict[str, float]]
    final_loss: float
    loss_breakdown: LossBreakdown
    converged: bool
    iterations: int
    elapsed_sec: float


def compute_vertex_normals(verts: torch.Tensor, faces_t: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex surface normals from mesh faces."""
    v0 = verts[faces_t[:, 0]]
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / (torch.norm(fn, dim=-1, keepdim=True) + 1e-8)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces_t[:, 0], fn)
    vn.index_add_(0, faces_t[:, 1], fn)
    vn.index_add_(0, faces_t[:, 2], fn)
    return vn / (torch.norm(vn, dim=-1, keepdim=True) + 1e-8)


class DifferentiableBodyFitter:
    """
    Fits Anny body model to simulated depth camera stops.
    """

    def __init__(
        self,
        backend,
        prior: AdultShapePrior,
        device: str = "cpu",
    ):
        self.backend = backend
        self.model = backend._model
        self.prior = prior
        self.device = torch.device(device)
        self.faces = backend._faces.cpu().numpy()
        self.faces_t = torch.tensor(self.faces, dtype=torch.long, device=self.device)
        self.joint_names = list(self.model.bone_labels)

        self.raw_dict = {
            "vertex_bone_weights": getattr(self.model, "vertex_bone_weights", None),
            "vertex_bone_indices": getattr(self.model, "vertex_bone_indices", None),
            "bone_labels": self.joint_names,
        }
        self._init_arm_masks()

    def _init_arm_masks(self):
        """Precompute soft skinning weights for Left and Right arm chains."""
        v_weights = self.raw_dict["vertex_bone_weights"]
        v_indices = self.raw_dict["vertex_bone_indices"]
        bl = self.raw_dict["bone_labels"]

        if v_weights is None or v_indices is None:
            self.arm_weight_L = torch.zeros(13718, dtype=torch.float32, device=self.device)
            self.arm_weight_R = torch.zeros(13718, dtype=torch.float32, device=self.device)
            return

        vw = v_weights.to(self.device) if hasattr(v_weights, "to") else torch.tensor(v_weights, device=self.device)
        vi = v_indices.to(self.device) if hasattr(v_indices, "to") else torch.tensor(v_indices, device=self.device)

        def _get_weight(side):
            arm_bones = [b for b in bl if f".{side}" in b and any(k in b.lower() for k in [
                "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
            ])]
            b_idxs = [bl.index(b) for b in arm_bones if b in bl]
            w = torch.zeros(vw.shape[0], dtype=torch.float32, device=self.device)
            for k in range(vi.shape[1]):
                col_idx = vi[:, k]
                col_w = vw[:, k]
                for b_i in b_idxs:
                    w += torch.where(col_idx == b_i, col_w, 0.0)
            return torch.clamp(w, 0.0, 1.0)

        self.arm_weight_L = _get_weight("L")
        self.arm_weight_R = _get_weight("R")

    def _forward_model(self, pheno_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiably compute rest vertices (V, 3) and joints (J, 3) in canonical A-pose
        from phenotypes (6,).
        """
        p = pheno_tensor.unsqueeze(0)
        p_dict = {k: p[:, i] for i, k in enumerate(PHENOTYPE_NAMES)}
        pose_params = self.backend._identity_pose.expand(1, self.backend._bone_count, 4, 4)
        raw = self.model.forward(
            pose_parameters=pose_params,
            phenotype_kwargs=p_dict,
        )
        verts = raw["vertices"][0]
        joints = raw["rest_bone_heads"][0]
        return verts, joints

    def _transform_stop(
        self,
        verts: torch.Tensor,
        joints: torch.Tensor,
        yaw_rad: torch.Tensor,
        trans: torch.Tensor,
        abd_L_rad: torch.Tensor,
        abd_R_rad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply arm abduction, yaw rotation around Z, and translation.
        Returns posed vertices and posed vertex normals.
        """
        jmap = {name: i for i, name in enumerate(self.joint_names)}
        idx_L = jmap.get("upperarm01.L", 0)
        idx_R = jmap.get("upperarm01.R", 0)
        pivot_L = joints[idx_L]
        pivot_R = joints[idx_R]

        out = verts.clone()

        # Left arm abduction in coronal plane
        diff_L = verts - pivot_L
        cos_L = torch.cos(abd_L_rad)
        sin_L = torch.sin(abd_L_rad)
        rot_x_L = diff_L[:, 0] * cos_L + diff_L[:, 2] * sin_L
        rot_z_L = -diff_L[:, 0] * sin_L + diff_L[:, 2] * cos_L
        rot_L = torch.stack([rot_x_L, diff_L[:, 1], rot_z_L], dim=-1)
        w_L = self.arm_weight_L.unsqueeze(-1)
        out = out + (rot_L - diff_L) * w_L

        # Right arm abduction in coronal plane
        diff_R = verts - pivot_R
        cos_R = torch.cos(abd_R_rad)
        sin_R = torch.sin(abd_R_rad)
        rot_x_R = diff_R[:, 0] * cos_R + diff_R[:, 2] * sin_R
        rot_z_R = -diff_R[:, 0] * sin_R + diff_R[:, 2] * cos_R
        rot_R = torch.stack([rot_x_R, diff_R[:, 1], rot_z_R], dim=-1)
        w_R = self.arm_weight_R.unsqueeze(-1)
        out = out + (rot_R - diff_R) * w_R

        # Yaw rotation around Z axis
        cos_y = torch.cos(yaw_rad)
        sin_y = torch.sin(yaw_rad)
        rx = out[:, 0] * cos_y - out[:, 1] * sin_y
        ry = out[:, 0] * sin_y + out[:, 1] * cos_y
        rz = out[:, 2]
        posed = torch.stack([rx, ry, rz], dim=-1) + trans.unsqueeze(0)

        # Normals
        vn = compute_vertex_normals(out, self.faces_t)
        nx = vn[:, 0] * cos_y - vn[:, 1] * sin_y
        ny = vn[:, 0] * sin_y + vn[:, 1] * cos_y
        nz = vn[:, 2]
        posed_normals = torch.stack([nx, ny, nz], dim=-1)

        return posed, posed_normals

    def fit(
        self,
        stops: List[SimulatedStopData],
        known_height_mm: Optional[float] = None,
        use_floor_constraint: bool = False,
        floor_z: Optional[float] = None,
        n_coarse_iter: int = 40,
        n_fine_iter: int = 30,
        multi_starts: int = 3,
        seed: int = 42,
        ablation_remove_term: Optional[str] = None,
    ) -> FitResult:
        """
        Execute multi-start coarse-to-fine optimization with complete loss formulation.
        """
        start_time = time.time()
        S = len(stops)

        # Precompute EDT distance transforms for each stop mask
        dist_transforms = []
        for s in stops:
            edt = scipy.ndimage.distance_transform_edt(~s.foreground_mask)
            pixel_size_m = s.intrinsics["distance_m"] / s.intrinsics["fx"]
            edt_m = (edt * pixel_size_m).astype(np.float32)
            dist_transforms.append(torch.tensor(edt_m, device=self.device))

        # Subsample observed point clouds for KDTree loss
        obs_sub_pts = []
        for s in stops:
            pts = s.point_cloud_world
            if len(pts) > 2000:
                rng = np.random.default_rng(seed)
                sub_idx = rng.choice(len(pts), size=2000, replace=False)
                sub_pts = pts[sub_idx]
            else:
                sub_pts = pts
            obs_sub_pts.append(torch.tensor(sub_pts, dtype=torch.float32, device=self.device))

        # Camera positions in world (fixed tripod at (0, -dist, 0.0))
        cam_world_pos = torch.tensor([0.0, -stops[0].intrinsics["distance_m"], 0.0], device=self.device)

        # Reference floor Z
        ref_floor_z = floor_z if floor_z is not None else stops[0].intrinsics.get("floor_z")

        # Estimate stature from depth bounding box to initialize height phenotype
        all_obs_z = torch.cat([pts[:, 2] for pts in obs_sub_pts])
        top_obs_z = float(torch.max(all_obs_z).item())
        floor_val = float(ref_floor_z) if ref_floor_z is not None else float(torch.min(all_obs_z).item())
        stature_obs_mm = (top_obs_z - floor_val) * 1000.0
        h_from_depth = float(np.clip((stature_obs_mm - 1360.0) / 960.0, 0.05, 0.95))

        # Initial candidates for multi-start
        init_phenos = [
            np.array([0.5, ADULT_AGE_NEUTRAL, 0.45, 0.45, h_from_depth, 0.50]),
            np.array([0.5, ADULT_AGE_NEUTRAL, 0.55, 0.70, h_from_depth, 0.50]),
            np.array([0.5, ADULT_AGE_NEUTRAL, 0.35, 0.25, h_from_depth, 0.50]),
        ][:multi_starts]

        if known_height_mm is not None:
            init_h = float(np.clip((known_height_mm - 1360.0) / 960.0, 0.05, 0.95))
            for cand in init_phenos:
                cand[4] = init_h

        best_loss = float("inf")
        best_state = None
        best_breakdown = None

        for start_idx, pheno_init in enumerate(init_phenos):
            p_logit = torch.tensor(
                scipy.special.logit(np.clip(pheno_init, 0.05, 0.95)),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )

            yaws = []
            translations = []
            abds_L = []
            abds_R = []

            for s_idx, s in enumerate(stops):
                nom_rad = np.radians(s.nominal_yaw_deg)
                yaws.append(torch.tensor(nom_rad, dtype=torch.float32, device=self.device, requires_grad=True))

                pts = s.point_cloud_world
                # Model center is near (0, 0, 0); translation initialized to 0
                translations.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device, requires_grad=True))

                abds_L.append(torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True))
                abds_R.append(torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True))

            param_groups = [
                {"params": [p_logit], "lr": 0.03},
                {"params": yaws, "lr": 0.02},
                {"params": translations, "lr": 0.01},
                {"params": abds_L + abds_R, "lr": 0.015},
            ]
            optimizer = optim.Adam(param_groups)

            # Optimization stages: coarse -> fine
            for stage, n_iter, c_geman in [("coarse", n_coarse_iter, 0.035), ("fine", n_fine_iter, 0.018)]:
                if stage == "fine":
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.4

                for step in range(n_iter):
                    optimizer.zero_grad()
                    pheno_curr = torch.sigmoid(p_logit)
                    verts_rest, joints_rest = self._forward_model(pheno_curr)

                    term_data = torch.tensor(0.0, device=self.device)
                    term_vis = torch.tensor(0.0, device=self.device)
                    term_free = torch.tensor(0.0, device=self.device)
                    term_sil = torch.tensor(0.0, device=self.device)

                    for s_i, s in enumerate(stops):
                        posed_verts, posed_normals = self._transform_stop(
                            verts_rest, joints_rest,
                            yaws[s_i], translations[s_i],
                            abds_L[s_i], abds_R[s_i],
                        )

                        # (a) Data-to-surface with KDTree
                        with torch.no_grad():
                            detached_verts = posed_verts.cpu().numpy()
                            kdtree = scipy.spatial.cKDTree(detached_verts)
                            obs_np = obs_sub_pts[s_i].cpu().numpy()
                            dists, nn_idx = kdtree.query(obs_np)
                            nn_idx_t = torch.tensor(nn_idx, device=self.device)

                        matched_model = posed_verts[nn_idx_t]
                        matched_normals = posed_normals[nn_idx_t].detach()
                        diff = obs_sub_pts[s_i] - matched_model
                        r_plane = torch.sum(diff * matched_normals, dim=-1)
                        residuals = 0.75 * torch.abs(r_plane) + 0.25 * torch.norm(diff, dim=-1)
                        term_data = term_data + torch.mean(residuals ** 2 / (residuals ** 2 + c_geman ** 2))

                        # Camera projection
                        intr = s.intrinsics
                        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
                        w, h = intr["width"], intr["height"]

                        # Camera frame coords: X_c = X_w, Y_c = -Z_w, Z_c = Y_w + dist
                        x_c = posed_verts[:, 0] - cam_world_pos[0]
                        y_c = -(posed_verts[:, 2] - cam_world_pos[2])
                        z_c = posed_verts[:, 1] - cam_world_pos[1]

                        u_proj = fx * (x_c / torch.clamp(z_c, min=0.3)) + cx
                        v_proj = fy * (y_c / torch.clamp(z_c, min=0.3)) + cy

                        u_idx = torch.clamp(u_proj.long(), 0, w - 1)
                        v_idx = torch.clamp(v_proj.long(), 0, h - 1)

                        # (d) Silhouette penalty
                        sil_dists = dist_transforms[s_i][v_idx, u_idx]
                        term_sil = term_sil + torch.mean(sil_dists ** 2)

                        # (b) Visible model-to-depth & (c) Free-space term
                        # Front-facing: normal Y component < -0.15 (pointing toward camera at -Y)
                        front_mask = posed_normals[:, 1] < -0.15
                        if torch.any(front_mask):
                            v_idx_f = v_idx[front_mask]
                            u_idx_f = u_idx[front_mask]
                            z_c_f = z_c[front_mask]

                            depth_map_t = torch.tensor(s.depth_image, device=self.device)
                            meas_d = depth_map_t[v_idx_f, u_idx_f]
                            valid_meas = (meas_d > 0.1)

                            if torch.any(valid_meas):
                                meas_d_v = meas_d[valid_meas]
                                z_c_v = z_c_f[valid_meas]
                                # (b) Visible model-to-depth: only unoccluded (within 25 mm of depth)
                                unoccluded = (z_c_v <= meas_d_v + 0.025)
                                if torch.any(unoccluded):
                                    r_depth = z_c_v[unoccluded] - meas_d_v[unoccluded]
                                    term_vis = term_vis + torch.mean(r_depth ** 2 / (r_depth ** 2 + c_geman ** 2))
                                # (c) Free-space penetration (model in front of depth by > 15 mm)
                                penetration = torch.relu(meas_d_v - z_c_v - 0.015)
                                term_free = term_free + torch.mean(penetration ** 2)

                    # (e) Shape prior
                    term_prior = -self.prior.log_prob(pheno_curr)

                    # (f) Floor constraint (optional)
                    term_floor = torch.tensor(0.0, device=self.device)
                    if use_floor_constraint and ref_floor_z is not None:
                        min_v_z = torch.min(verts_rest[:, 2]) + translations[0][2]
                        term_floor = (min_v_z - ref_floor_z) ** 2

                    # (g) Height constraint (optional)
                    term_height = torch.tensor(0.0, device=self.device)
                    if known_height_mm is not None:
                        stature_est_mm = (verts_rest[:, 2].max() - verts_rest[:, 2].min()) * 1000.0
                        term_height = ((stature_est_mm - known_height_mm) / 10.0) ** 2

                    # Weights:
                    w_dict = {
                        "data": 2.0,
                        "vis": 2.0,
                        "free": 10.0,
                        "sil": 50.0,
                        "prior": 0.02,
                        "floor": 100.0,
                        "height": 1.0,
                    }

                    # Ablation support
                    if ablation_remove_term and ablation_remove_term in w_dict:
                        w_dict[ablation_remove_term] = 0.0

                    total_loss = (
                        w_dict["data"] * term_data +
                        w_dict["vis"] * term_vis +
                        w_dict["free"] * term_free +
                        w_dict["sil"] * term_sil +
                        w_dict["prior"] * term_prior +
                        w_dict["floor"] * term_floor +
                        w_dict["height"] * term_height
                    )

                    total_loss.backward()
                    optimizer.step()

            # L-BFGS refinement on phenotypes
            lbfgs = optim.LBFGS([p_logit], lr=0.15, max_iter=6, line_search_fn="strong_wolfe")
            c_geman = 0.015

            def closure():
                lbfgs.zero_grad()
                pheno_curr = torch.sigmoid(p_logit)
                verts_rest, joints_rest = self._forward_model(pheno_curr)

                t_data = torch.tensor(0.0, device=self.device)
                t_vis = torch.tensor(0.0, device=self.device)
                t_free = torch.tensor(0.0, device=self.device)
                t_sil = torch.tensor(0.0, device=self.device)

                for s_i, s in enumerate(stops):
                    posed_verts, posed_normals = self._transform_stop(
                        verts_rest, joints_rest,
                        yaws[s_i].detach(), translations[s_i].detach(),
                        abds_L[s_i].detach(), abds_R[s_i].detach(),
                    )
                    with torch.no_grad():
                        detached_verts = posed_verts.cpu().numpy()
                        kdtree = scipy.spatial.cKDTree(detached_verts)
                        obs_np = obs_sub_pts[s_i].cpu().numpy()
                        dists, nn_idx = kdtree.query(obs_np)
                        nn_idx_t = torch.tensor(nn_idx, device=self.device)

                    matched_model = posed_verts[nn_idx_t]
                    matched_normals = posed_normals[nn_idx_t].detach()
                    diff = obs_sub_pts[s_i] - matched_model
                    r_plane = torch.sum(diff * matched_normals, dim=-1)
                    residuals = 0.75 * torch.abs(r_plane) + 0.25 * torch.norm(diff, dim=-1)
                    t_data = t_data + torch.mean(residuals ** 2 / (residuals ** 2 + c_geman ** 2))

                    intr = s.intrinsics
                    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
                    w, h = intr["width"], intr["height"]
                    x_c = posed_verts[:, 0] - cam_world_pos[0]
                    y_c = -(posed_verts[:, 2] - cam_world_pos[2])
                    z_c = posed_verts[:, 1] - cam_world_pos[1]
                    u_proj = fx * (x_c / torch.clamp(z_c, min=0.3)) + cx
                    v_proj = fy * (y_c / torch.clamp(z_c, min=0.3)) + cy
                    u_idx = torch.clamp(u_proj.long(), 0, w - 1)
                    v_idx = torch.clamp(v_proj.long(), 0, h - 1)

                    sil_dists = dist_transforms[s_i][v_idx, u_idx]
                    t_sil = t_sil + torch.mean(sil_dists ** 2)

                    front_mask = posed_normals[:, 1] < -0.15
                    if torch.any(front_mask):
                        v_idx_f = v_idx[front_mask]
                        u_idx_f = u_idx[front_mask]
                        z_c_f = z_c[front_mask]
                        depth_map_t = torch.tensor(s.depth_image, device=self.device)
                        meas_d = depth_map_t[v_idx_f, u_idx_f]
                        valid_meas = (meas_d > 0.1)
                        if torch.any(valid_meas):
                            meas_d_v = meas_d[valid_meas]
                            z_c_v = z_c_f[valid_meas]
                            unoccluded = (z_c_v <= meas_d_v + 0.025)
                            if torch.any(unoccluded):
                                r_depth = z_c_v[unoccluded] - meas_d_v[unoccluded]
                                t_vis = t_vis + torch.mean(r_depth ** 2 / (r_depth ** 2 + c_geman ** 2))
                            penetration = torch.relu(meas_d_v - z_c_v - 0.015)
                            t_free = t_free + torch.mean(penetration ** 2)

                t_prior = -self.prior.log_prob(pheno_curr)
                t_floor = torch.tensor(0.0, device=self.device)
                if use_floor_constraint and ref_floor_z is not None:
                    min_v_z = torch.min(verts_rest[:, 2]) + translations[0][2]
                    t_floor = (min_v_z - ref_floor_z) ** 2
                t_height = torch.tensor(0.0, device=self.device)
                if known_height_mm is not None:
                    stature_est_mm = (verts_rest[:, 2].max() - verts_rest[:, 2].min()) * 1000.0
                    t_height = ((stature_est_mm - known_height_mm) / 10.0) ** 2

                total_l = (
                    w_dict["data"] * t_data +
                    w_dict["vis"] * t_vis +
                    w_dict["free"] * t_free +
                    w_dict["sil"] * t_sil +
                    w_dict["prior"] * t_prior +
                    w_dict["floor"] * t_floor +
                    w_dict["height"] * t_height
                )
                total_l.backward()
                return total_l

            try:
                lbfgs.step(closure)
            except Exception:
                pass

            final_val = float(total_loss.item())
            if final_val < best_loss:
                best_loss = final_val
                fitted_pheno = {k: float(torch.sigmoid(p_logit)[i].item()) for i, k in enumerate(PHENOTYPE_NAMES)}
                best_state = (fitted_pheno, yaws, translations, abds_L, abds_R)
                best_breakdown = LossBreakdown(
                    loss_data=float(term_data.item()) * w_dict["data"],
                    loss_vis=float(term_vis.item()) * w_dict["vis"],
                    loss_free=float(term_free.item()) * w_dict["free"],
                    loss_sil=float(term_sil.item()) * w_dict["sil"],
                    loss_prior=float(term_prior.item()) * w_dict["prior"],
                    loss_floor=float(term_floor.item()) * w_dict["floor"],
                    loss_height=float(term_height.item()) * w_dict["height"],
                    total_loss=final_val,
                )

        fitted_phenos, yaws, translations, abds_L, abds_R = best_state
        fitted_output = self.backend.rest_pose(phenotypes=fitted_phenos)
        measurements = measure_mesh(
            fitted_output.numpy_vertices(0),
            fitted_output.numpy_joints(0),
            fitted_output.faces.cpu().numpy(),
            fitted_output.joint_names,
            fitted_output.raw if hasattr(fitted_output, "raw") else {},
        )

        per_stop_poses = []
        for s_i in range(S):
            per_stop_poses.append({
                "yaw_deg": float(np.degrees(yaws[s_i].item())),
                "translation_m": [float(x) for x in translations[s_i].tolist()],
                "arm_abd_L_deg": float(np.degrees(abds_L[s_i].item())),
                "arm_abd_R_deg": float(np.degrees(abds_R[s_i].item())),
            })

        elapsed = time.time() - start_time
        return FitResult(
            fitted_phenotypes=fitted_phenos,
            measurements=measurements,
            per_stop_poses=per_stop_poses,
            final_loss=best_loss,
            loss_breakdown=best_breakdown,
            converged=best_loss < 8.0,
            iterations=n_coarse_iter + n_fine_iter,
            elapsed_sec=elapsed,
        )
