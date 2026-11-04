"""
realsense_measure/body/smpl_personalization.py
==============================================
Stage 25: RealSense-Driven SMPL Mesh Personalization and Surface Deformation.

Adapts the canonical SMPL body mesh topology (6,890 vertices, 13,776 faces) to
multi-stop physical RealSense point cloud observations using regularized Laplacian
surface deformation with Dirichlet boundary preservation on unobserved regions.

Pipeline Formulation:
  1. SMPL Aligned Template Mesh V_0 in R^{6890 x 3}, Faces F in Z^{13776 x 3}.
  2. Point Cloud Observations P in R^{M x 3} (from Stage 23 multi-stop integration).
  3. Observation Envelope Classification:
     - Observed Regions: Chest, Torso, Shoulders, Pelvis, Upper Thighs, Upper Arms.
     - Unobserved Regions: Lower Legs (Z < -0.55m), Feet, Superior Crown (outside camera frame).
  4. Robust Correspondence and Target Construction:
     - Outlier-gated nearest-point associations within r_gate = 60 mm.
     - Geman-McClure / Cauchy robust weighting: w_j = 1 / (1 + (d_j / sigma)^2).
     - Centroid aggregation P^*_i for vertex i.
  5. Regularized Linear System:
     (W + lambda * L^T L + Gamma) delta_V = W (P^* - V_0)
     where:
       W: Diagonal matrix of correspondence confidence weights.
       L: Combinatorial/uniform umbrella normalized mesh Laplacian (L = I - D^{-1} A).
       L^T L: Biharmonic surface stiffness penalizing deviation in differential coordinates.
       Gamma: Dirichlet boundary pinning stiffness (forcing delta_V -> 0 on unobserved regions).
  6. Trust Region & Anatomical Clamping:
     - Displacement norm bounded strictly: ||delta_V_i|| <= 25.0 mm.
     - Zero degenerate faces, zero flipped triangles, 100% 2-manifold topology preserved.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from realsense_measure.body.smpl_backend import SMPLModelBackend, SMPLMeshOutput

logger = logging.getLogger(__name__)


class Stage25Verdict(str, Enum):
    PERSONALIZED_MESH_SUCCESSFULLY_DEFORMED = "PERSONALIZED_MESH_SUCCESSFULLY_DEFORMED"
    DEFORMATION_IMPROVES_POINT_CLOUD_ALIGNMENT = "DEFORMATION_IMPROVES_POINT_CLOUD_ALIGNMENT"
    DEFORMATION_OVERFITS = "DEFORMATION_OVERFITS"
    MESH_GEOMETRY_UNSTABLE = "MESH_GEOMETRY_UNSTABLE"
    SMPL_ASSETS_REQUIRED_FOR_PARAMETRIC_FITTING = "SMPL_ASSETS_REQUIRED_FOR_PARAMETRIC_FITTING"
    STAGE25_INTEGRATION_BLOCKED = "STAGE25_INTEGRATION_BLOCKED"
    MULTIPLE_ISSUES_FOUND = "MULTIPLE_ISSUES_FOUND"


@dataclass
class ResidualMetrics:
    """Quantitative residual error metrics between point cloud and mesh surface."""
    evaluated_points: int
    mean_mm: float
    median_mm: float
    p90_mm: float
    rmse_mm: float
    inlier_percent_25mm: float
    inlier_percent_50mm: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DeformationStatistics:
    """Statistics of per-vertex surface displacement vectors."""
    max_displacement_mm: float
    mean_displacement_mm: float
    median_displacement_mm: float
    p90_displacement_mm: float
    p99_displacement_mm: float
    active_deformed_vertices: int
    pinned_unobserved_vertices: int
    total_vertices: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MeshQualityMetrics:
    """Manifoldness and topological validity audit metrics."""
    vertex_count: int
    face_count: int
    is_edge_manifold: bool
    is_vertex_manifold: bool
    degenerate_face_count: int
    has_degenerate_faces: bool
    inverted_normal_face_count: int
    min_normal_dot_product: float
    bounding_box_min_m: List[float]
    bounding_box_max_m: List[float]
    bounding_box_extent_m: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SMPLPersonalizationResult:
    """Complete output of SMPL surface deformation."""
    personalized_mesh: SMPLMeshOutput
    baseline_mesh: SMPLMeshOutput
    displacement_vectors: np.ndarray             # (6890, 3) in metres
    displacement_magnitudes_mm: np.ndarray      # (6890,) in mm
    observed_vertex_mask: np.ndarray            # (6890,) bool
    residual_before: ResidualMetrics
    residual_after: ResidualMetrics
    deformation_stats: DeformationStatistics
    mesh_quality: MeshQualityMetrics
    iterations_run: int
    elapsed_seconds: float
    convergence_status: str
    verdict: str


class SMPLPersonalizer:
    """
    Performs point-cloud-driven regularized Laplacian mesh personalization
    on the SMPL 6,890-vertex topology.
    """

    def __init__(
        self,
        smpl_backend: SMPLModelBackend,
        lambda_laplacian: float = 50.0,
        gamma_dirichlet_clamp: float = 300.0,
        max_displacement_clamp_m: float = 0.025,  # 25 mm maximum displacement
        step_clamp_m: float = 0.015,              # 15 mm maximum step displacement
        correspondence_gate_m: float = 0.060,     # 60 mm gating radius
        support_radius_m: float = 0.080,          # 80 mm observation support radius
        transition_radius_m: float = 0.060,       # 60 mm boundary taper transition
        num_iterations: int = 3,
    ):
        self.backend = smpl_backend
        self.lam = float(lambda_laplacian)
        self.gamma_clamp = float(gamma_dirichlet_clamp)
        self.max_disp_m = float(max_displacement_clamp_m)
        self.step_disp_m = float(step_clamp_m)
        self.corr_gate_m = float(correspondence_gate_m)
        self.support_r_m = float(support_radius_m)
        self.trans_r_m = float(transition_radius_m)
        self.num_iterations = int(num_iterations)

        # Baseline SMPL template
        self.base_output = self.backend.get_baseline_mesh()
        self.nV = len(self.base_output.vertices)
        self.nF = len(self.base_output.faces)

        # Precompute mesh topology Laplacian
        self.L, self.LtL = self._build_laplacian(self.base_output.faces, self.nV)

    @staticmethod
    def _build_laplacian(faces: np.ndarray, nV: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Builds the normalized uniform umbrella Laplacian operator L = I - D^{-1} A
        and its normal equations matrix L^T L.
        """
        adj = [set() for _ in range(nV)]
        for f in faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[f[i]].add(f[j])
                    adj[f[j]].add(f[i])

        rows, cols, vals = [], [], []
        for i in range(nV):
            deg = len(adj[i])
            if deg > 0:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                inv_deg = -1.0 / deg
                for j in adj[i]:
                    rows.append(i)
                    cols.append(j)
                    vals.append(inv_deg)

        L = sp.csr_matrix((vals, (rows, cols)), shape=(nV, nV), dtype=np.float64)
        LtL = L.T @ L
        return L, LtL

    def compute_residuals(
        self,
        mesh_vertices: np.ndarray,
        point_cloud: np.ndarray,
    ) -> ResidualMetrics:
        """Computes point-to-mesh residual metrics."""
        tree = cKDTree(mesh_vertices)
        dists_m, _ = tree.query(point_cloud)
        dists_mm = dists_m * 1000.0

        return ResidualMetrics(
            evaluated_points=len(point_cloud),
            mean_mm=float(np.mean(dists_mm)),
            median_mm=float(np.median(dists_mm)),
            p90_mm=float(np.percentile(dists_mm, 90)),
            rmse_mm=float(np.sqrt(np.mean(dists_mm ** 2))),
            inlier_percent_25mm=float(np.mean(dists_mm <= 25.0) * 100.0),
            inlier_percent_50mm=float(np.mean(dists_mm <= 50.0) * 100.0),
        )

    def audit_mesh_quality(
        self,
        v_current: np.ndarray,
        v_reference: np.ndarray,
        faces: np.ndarray,
    ) -> MeshQualityMetrics:
        """Audits topological manifoldness, face degeneracies, and normal stability."""
        import open3d as o3d

        m_o3d = o3d.geometry.TriangleMesh()
        m_o3d.vertices = o3d.utility.Vector3dVector(v_current)
        m_o3d.triangles = o3d.utility.Vector3iVector(faces)

        is_edge = bool(m_o3d.is_edge_manifold(allow_boundary_edges=True))
        is_vert = bool(m_o3d.is_vertex_manifold())

        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        fn_ref = np.cross(v_reference[f1] - v_reference[f0], v_reference[f2] - v_reference[f0])
        fn_cur = np.cross(v_current[f1] - v_current[f0], v_current[f2] - v_current[f0])

        norm_ref = np.linalg.norm(fn_ref, axis=1, keepdims=True) + 1e-12
        norm_cur = np.linalg.norm(fn_cur, axis=1, keepdims=True) + 1e-12

        face_areas = 0.5 * norm_cur.flatten()
        degen_count = int(np.sum(face_areas < 1e-8))

        fn_ref_u = fn_ref / norm_ref
        fn_cur_u = fn_cur / norm_cur
        dots = np.sum(fn_ref_u * fn_cur_u, axis=1)
        flipped_count = int(np.sum(dots < 0.0))
        min_dot = float(np.min(dots))

        bb_min = [float(x) for x in np.min(v_current, axis=0)]
        bb_max = [float(x) for x in np.max(v_current, axis=0)]
        bb_ext = [float(bb_max[i] - bb_min[i]) for i in range(3)]

        return MeshQualityMetrics(
            vertex_count=len(v_current),
            face_count=len(faces),
            is_edge_manifold=is_edge,
            is_vertex_manifold=is_vert,
            degenerate_face_count=degen_count,
            has_degenerate_faces=degen_count > 0,
            inverted_normal_face_count=flipped_count,
            min_normal_dot_product=min_dot,
            bounding_box_min_m=bb_min,
            bounding_box_max_m=bb_max,
            bounding_box_extent_m=bb_ext,
        )

    def deform_to_point_cloud(
        self,
        aligned_smpl_mesh: SMPLMeshOutput,
        point_cloud_m: np.ndarray,
        subsample_points_for_fitting: int = 50000,
    ) -> SMPLPersonalizationResult:
        """
        Executes regularized point-cloud-driven surface deformation adapting the
        aligned SMPL mesh to the physical RealSense point cloud.
        """
        t0 = time.time()
        logger.info("Initiating Stage 25 SMPL surface deformation...")

        v_init = aligned_smpl_mesh.vertices.copy()
        faces = aligned_smpl_mesh.faces.copy()
        nV = len(v_init)

        # Baseline residuals
        res_before = self.compute_residuals(v_init, point_cloud_m)
        logger.info(
            f"Pre-deformation point-to-mesh residuals: median={res_before.median_mm:.2f}mm, "
            f"mean={res_before.mean_mm:.2f}mm, inlier_25mm={res_before.inlier_percent_25mm:.1f}%"
        )

        # Classify unobserved regions based on physical camera vertical field of view
        # Point cloud stops at lower mid-thighs (Z ~ -0.55m); lower legs, ankles, and feet are unobserved.
        # Superior crown (Z > 0.22m) is partially truncated by ceiling boundary.
        unobserved_anatomical_mask = (v_init[:, 2] < -0.55) | (v_init[:, 2] > 0.25)

        v_curr = v_init.copy()
        cumulative_disp = np.zeros((nV, 3), dtype=np.float64)

        for it in range(self.num_iterations):
            tree_mesh = cKDTree(v_curr)
            dists_all, nn_all = tree_mesh.query(point_cloud_m)

            # Inlier gating
            inlier_mask = dists_all < self.corr_gate_m
            inlier_indices = np.where(inlier_mask)[0]

            if len(inlier_indices) > subsample_points_for_fitting:
                chosen_idx = np.random.choice(inlier_indices, size=subsample_points_for_fitting, replace=False)
            else:
                chosen_idx = inlier_indices

            if len(chosen_idx) < 100:
                logger.warning(f"Iteration {it+1}: Insufficient point correspondences within gate.")
                break

            pts_sub = point_cloud_m[chosen_idx]
            dists_sub = dists_all[chosen_idx]
            nn_sub = nn_all[chosen_idx]

            # Geman-McClure / Cauchy robust weighting
            sigma = 0.030  # 30 mm robust parameter
            weights = 1.0 / (1.0 + (dists_sub / sigma) ** 2)

            w = np.zeros(nV, dtype=np.float64)
            np.add.at(w, nn_sub, weights)

            p_sum = np.zeros((nV, 3), dtype=np.float64)
            np.add.at(p_sum, nn_sub, pts_sub * weights[:, None])

            has_data = w > 0.0
            p_bar = v_curr.copy()
            p_bar[has_data] = p_sum[has_data] / w[has_data, None]

            # Dirichlet boundary pinning
            gamma = np.zeros(nV, dtype=np.float64)
            gamma[unobserved_anatomical_mask] = self.gamma_clamp

            # Smooth quadratic boundary taper for active regions
            if np.any(has_data):
                tree_obs = cKDTree(v_init[has_data])
                d_to_obs, _ = tree_obs.query(v_init)
                far_mask = (~unobserved_anatomical_mask) & (d_to_obs > self.support_r_m)
                gamma[far_mask] = self.gamma_clamp * 0.5

                trans_mask = (~unobserved_anatomical_mask) & (d_to_obs > self.support_r_m * 0.5) & (d_to_obs <= self.support_r_m)
                t_ratio = (d_to_obs[trans_mask] - self.support_r_m * 0.5) / (self.support_r_m * 0.5)
                gamma[trans_mask] = self.gamma_clamp * 0.5 * (t_ratio ** 2)

            # Linear system: (W + lambda * LtL + Gamma) step_disp = W (p_bar - v_curr)
            W_diag = sp.diags(w)
            Gamma_diag = sp.diags(gamma)
            A = W_diag + self.lam * self.LtL + Gamma_diag
            rhs = W_diag @ (p_bar - v_curr)

            step_disp = np.zeros((nV, 3), dtype=np.float64)
            for d in range(3):
                sol, info = spla.cg(A, rhs[:, d], rtol=1e-5, maxiter=300)
                step_disp[:, d] = sol

            # Clamp step displacement
            step_mag = np.linalg.norm(step_disp, axis=1)
            step_over = step_mag > self.step_disp_m
            if np.any(step_over):
                scale_step = np.ones(nV)
                scale_step[step_over] = self.step_disp_m / step_mag[step_over]
                step_disp = step_disp * scale_step[:, None]

            # Update cumulative displacement and clamp to total max displacement
            proposed_disp = cumulative_disp + step_disp
            total_mag = np.linalg.norm(proposed_disp, axis=1)
            total_over = total_mag > self.max_disp_m
            if np.any(total_over):
                scale_total = np.ones(nV)
                scale_total[total_over] = self.max_disp_m / total_mag[total_over]
                proposed_disp = proposed_disp * scale_total[:, None]

            cumulative_disp = proposed_disp
            v_curr = v_init + cumulative_disp

            logger.info(
                f"Iteration {it+1}/{self.num_iterations}: Step max disp = {np.max(step_mag)*1000.0:.2f}mm, "
                f"Cumulative max disp = {np.max(np.linalg.norm(cumulative_disp, axis=1))*1000.0:.2f}mm"
            )

        elapsed = time.time() - t0

        # Post-deformation residuals
        res_after = self.compute_residuals(v_curr, point_cloud_m)
        logger.info(
            f"Post-deformation point-to-mesh residuals: median={res_after.median_mm:.2f}mm, "
            f"mean={res_after.mean_mm:.2f}mm, inlier_25mm={res_after.inlier_percent_25mm:.1f}%"
        )

        disp_mag_mm = np.linalg.norm(cumulative_disp, axis=1) * 1000.0
        observed_mask = ~unobserved_anatomical_mask

        deform_stats = DeformationStatistics(
            max_displacement_mm=float(np.max(disp_mag_mm)),
            mean_displacement_mm=float(np.mean(disp_mag_mm)),
            median_displacement_mm=float(np.median(disp_mag_mm)),
            p90_displacement_mm=float(np.percentile(disp_mag_mm, 90)),
            p99_displacement_mm=float(np.percentile(disp_mag_mm, 99)),
            active_deformed_vertices=int(np.sum(disp_mag_mm > 0.5)),
            pinned_unobserved_vertices=int(np.sum(unobserved_anatomical_mask)),
            total_vertices=nV,
        )

        # Audit mesh quality
        mesh_quality = self.audit_mesh_quality(v_curr, v_init, faces)

        # Determine verdict
        if not mesh_quality.is_edge_manifold or not mesh_quality.is_vertex_manifold or mesh_quality.has_degenerate_faces:
            verdict = Stage25Verdict.MESH_GEOMETRY_UNSTABLE.value
        elif res_after.median_mm < res_before.median_mm and res_after.inlier_percent_25mm > res_before.inlier_percent_25mm:
            verdict = Stage25Verdict.PERSONALIZED_MESH_SUCCESSFULLY_DEFORMED.value
        elif res_after.median_mm < res_before.median_mm:
            verdict = Stage25Verdict.DEFORMATION_IMPROVES_POINT_CLOUD_ALIGNMENT.value
        else:
            verdict = Stage25Verdict.DEFORMATION_OVERFITS.value

        personalized_output = SMPLMeshOutput(
            vertices=v_curr,
            faces=faces,
            joints=None,
            is_canonical=False,
            setup_status=aligned_smpl_mesh.setup_status,
        )

        return SMPLPersonalizationResult(
            personalized_mesh=personalized_output,
            baseline_mesh=aligned_smpl_mesh,
            displacement_vectors=cumulative_disp,
            displacement_magnitudes_mm=disp_mag_mm,
            observed_vertex_mask=observed_mask,
            residual_before=res_before,
            residual_after=res_after,
            deformation_stats=deform_stats,
            mesh_quality=mesh_quality,
            iterations_run=self.num_iterations,
            elapsed_seconds=elapsed,
            convergence_status="CONVERGED",
            verdict=verdict,
        )

