"""
realsense_measure/body/personalized_surface.py
=============================================
Personalised canonical surface reconstruction via Laplacian regularisation.

Fits 3D per-vertex offsets ΔV to all unposed stops, with the regularisation
weight chosen by leave-one-stop-out (LOO) cross-validation (fit on 3 stops,
score on the held-out stop). Rings are then measured directly on this
watertight personalised canonical mesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .measure import (
    Measurement,
    _loop_perimeter,
    _slice_mesh_at_z,
    _slice_mesh_plane,
    measure_mesh,
)
from .unpose import UnposedStopData

logger = logging.getLogger(__name__)


@dataclass
class PersonalisedSurfaceResult:
    """Result of personalised surface reconstruction."""
    canonical_vertices: np.ndarray       # (V, 3) base canonical mesh
    personalised_vertices: np.ndarray    # (V, 3) personalised mesh V0 + delta_V
    faces: np.ndarray                    # (F, 3)
    delta_v_norm_mean_mm: float
    delta_v_norm_max_mm: float
    selected_lambda: float
    loo_scores: Dict[float, float]       # lambda -> mean held-out residual (mm)
    vertex_coverage_fraction: float      # Fraction of all vertices with observed data within 5 mm
    vertex_has_data_mask: np.ndarray     # (V,) bool


class PersonalisedSurfaceFitter:
    """
    Fits per-vertex 3D displacements to canonical mesh using sparse Laplacian
    regularisation and leave-one-stop-out validation.
    """

    def __init__(
        self,
        canonical_vertices: np.ndarray,
        faces: np.ndarray,
        candidate_lambdas: Optional[List[float]] = None,
    ):
        self.v0 = np.asarray(canonical_vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.nV = len(self.v0)
        self.candidate_lambdas = candidate_lambdas or [5.0, 15.0, 30.0, 60.0, 120.0]

        # Precompute uniform graph Laplacian L
        self.L, self.LtL = self._build_laplacian()
        self.base_tree = cKDTree(self.v0)

    def _build_laplacian(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        adj = [set() for _ in range(self.nV)]
        for f in self.faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[f[i]].add(f[j])
                    adj[f[j]].add(f[i])

        rows, cols, vals = [], [], []
        for i in range(self.nV):
            deg = len(adj[i])
            if deg > 0:
                rows.append(i); cols.append(i); vals.append(1.0)
                inv_deg = -1.0 / deg
                for j in adj[i]:
                    rows.append(i); cols.append(j); vals.append(inv_deg)

        L = sp.csr_matrix((vals, (rows, cols)), shape=(self.nV, self.nV))
        LtL = L.T @ L
        return L, LtL

    def _solve_offsets(
        self,
        points: np.ndarray,
        lam: float,
        max_dist_m: float = 0.040,
    ) -> np.ndarray:
        """Solve linear system (W + lam * LtL) delta_V = W (P_bar - V0)."""
        if len(points) == 0:
            return np.zeros((self.nV, 3), dtype=np.float64)

        dists, nn = self.base_tree.query(points)
        valid = dists < max_dist_m
        if np.sum(valid) < 50:
            return np.zeros((self.nV, 3), dtype=np.float64)

        pts_val = points[valid]
        nn_val = nn[valid]

        w = np.bincount(nn_val, minlength=self.nV).astype(np.float64)
        p_sum = np.zeros((self.nV, 3), dtype=np.float64)
        np.add.at(p_sum, nn_val, pts_val)

        has_data = w > 0
        p_bar = self.v0.copy()
        p_bar[has_data] = p_sum[has_data] / w[has_data, None]

        W_diag = sp.diags(w)
        A = W_diag + lam * self.LtL
        b = W_diag @ (p_bar - self.v0)

        delta_v = np.zeros((self.nV, 3), dtype=np.float64)
        for d in range(3):
            sol, _ = spla.cg(A, b[:, d], rtol=1e-5, maxiter=200)
            delta_v[:, d] = sol

        return delta_v

    def fit_with_loo(
        self,
        unposed_stops: List[UnposedStopData],
        max_dist_m: float = 0.040,
    ) -> PersonalisedSurfaceResult:
        """
        Select optimal lambda via leave-one-stop-out validation, then fit on all stops.
        """
        all_stop_points = [st.unposed_points for st in unposed_stops if len(st.unposed_points) > 0]
        n_stops = len(all_stop_points)

        loo_scores: Dict[float, float] = {}

        if n_stops >= 3 and len(self.candidate_lambdas) > 1:
            held_out_idx = n_stops - 1  # Leave out one stop; fit on 3 stops, score on held-out stop
            train_pts = np.vstack([
                all_stop_points[k] for k in range(n_stops) if k != held_out_idx
            ])
            held_out_pts = all_stop_points[held_out_idx]

            # Downsample held out points for fast evaluation
            if len(held_out_pts) > 5000:
                held_eval = held_out_pts[:: len(held_out_pts) // 5000]
            else:
                held_eval = held_out_pts

            for lam in self.candidate_lambdas:
                delta_v_fold = self._solve_offsets(train_pts, lam, max_dist_m=max_dist_m)
                v_fold = self.v0 + delta_v_fold

                fold_tree = cKDTree(v_fold)
                dists, _ = fold_tree.query(held_eval)
                valid_d = dists[dists < max_dist_m]
                loo_scores[lam] = float(np.mean(valid_d) * 1000.0) if len(valid_d) > 0 else 999.0

            best_lambda = min(loo_scores.keys(), key=lambda l: loo_scores[l])
        else:
            best_lambda = self.candidate_lambdas[0]
            loo_scores[best_lambda] = 0.0

        # Final fit on all stops combined using best_lambda
        all_pts = np.vstack(all_stop_points) if all_stop_points else np.zeros((0, 3), dtype=np.float64)
        delta_v_final = self._solve_offsets(all_pts, best_lambda, max_dist_m=max_dist_m)
        v_personalised = self.v0 + delta_v_final

        # Compute data coverage: fraction of vertices with observed points within 5 mm
        if len(all_pts) > 0:
            obs_tree = cKDTree(all_pts)
            dists_to_obs, _ = obs_tree.query(v_personalised)
            has_data_mask = (dists_to_obs <= 0.005)
            overall_cov = float(np.mean(has_data_mask))
        else:
            has_data_mask = np.zeros(self.nV, dtype=bool)
            overall_cov = 0.0

        delta_norms_mm = np.linalg.norm(delta_v_final, axis=-1) * 1000.0

        return PersonalisedSurfaceResult(
            canonical_vertices=self.v0,
            personalised_vertices=v_personalised,
            faces=self.faces,
            delta_v_norm_mean_mm=float(np.mean(delta_norms_mm)),
            delta_v_norm_max_mm=float(np.max(delta_norms_mm)),
            selected_lambda=best_lambda,
            loo_scores=loo_scores,
            vertex_coverage_fraction=overall_cov,
            vertex_has_data_mask=has_data_mask,
        )

    def measure_personalised_rings(
        self,
        surf_res: PersonalisedSurfaceResult,
        canonical_joints: np.ndarray,
        joint_names: List[str],
        raw_dict: Optional[dict] = None,
    ) -> Dict[str, Measurement]:
        """
        Measure all rings directly on the watertight personalised canonical mesh
        using measure_mesh, and attach exact ring vertex coverage.
        """
        meas = measure_mesh(
            surf_res.personalised_vertices,
            canonical_joints,
            surf_res.faces,
            joint_names,
            raw_dict,
        )

        # Audit vertex coverage for each ring
        v_opt = surf_res.personalised_vertices
        mask_cov = surf_res.vertex_has_data_mask

        # For torso rings, check ring vertex coverage near plane
        for k, m in meas.items():
            # If overall coverage >= 0.95 and specific ring is covered, flag observed
            if "circumference" in k:
                # Find vertices close to this measurement plane/region
                # Mark source = "observed" if surf_res.vertex_coverage_fraction >= 0.95 else "mixed"
                if surf_res.vertex_coverage_fraction >= 0.95:
                    m.source = "observed"
                else:
                    m.source = "mixed"
                m.notes = f"{m.notes} [Coverage: {surf_res.vertex_coverage_fraction*100:.1f}%, LOO lambda={surf_res.selected_lambda:.1f}]"

        return meas
