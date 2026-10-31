"""
Observed-surface circumference measurement on unposed depth point clouds.

Extracts closed cross-sectional contours directly from unposed depth observations,
computes surface-path perimeters, tracks angular coverage fractions, and flags
occlusion gaps without silent model filling.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .measure import (
    LandmarkHeights,
    _loop_perimeter,
    _orthonormal_basis,
    _slice_mesh_at_z,
    _slice_mesh_plane,
    _get_torso_mesh,
    _select_torso_loop,
)
from .unpose import UnposedStopData

logger = logging.getLogger(__name__)


@dataclass
class ObservedRingResult:
    """Measurement result computed directly on observed depth points."""
    name: str
    value_mm: float
    coverage_fraction: float
    source: str                     # "observed" if coverage >= 0.95 else "mixed"
    method: str
    unposed_residual_mm: float      # Mean distance of in-slab observed points to canonical fitted model
    num_points_in_slab: int
    num_covered_bins: int
    total_bins: int
    contour_pts: np.ndarray = field(repr=False)       # (K, 3) closed 3D polygon
    covered_bin_mask: np.ndarray = field(repr=False)  # (K,) bool

    def is_valid(self) -> bool:
        return not math.isnan(self.value_mm) and self.value_mm > 0


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of 1D array."""
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])

    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]

    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    if total_w <= 1e-8:
        return float(np.median(values))

    med_idx = np.searchsorted(cum_w, 0.5 * total_w)
    med_idx = min(med_idx, len(values) - 1)
    return float(v_sorted[med_idx])


class ObservedRingExtractor:
    """
    Extracts observed-surface circumferences from unposed multi-stop point clouds.
    """

    def __init__(
        self,
        canon_verts: np.ndarray,
        canon_joints: np.ndarray,
        faces: np.ndarray,
        joint_names: List[str],
        raw_dict: Optional[dict] = None,
        n_angular_bins: int = 72,        # 5-degree bins around ring axis
        slab_thickness_m: float = 0.010,  # +-5 mm slab around cutting plane
        use_gaussian_smoothing: bool = True,
    ):
        self.verts = np.asarray(canon_verts, dtype=np.float32)
        self.joints = np.asarray(canon_joints, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int64) if len(faces) > 0 else np.zeros((0, 3), dtype=np.int64)
        self.joint_names = list(joint_names)
        self.raw_dict = raw_dict or {}
        self.n_bins = n_angular_bins
        self.half_slab = slab_thickness_m / 2.0  # 0.005 m = 5 mm
        self.use_gaussian_smoothing = use_gaussian_smoothing

        self.lm = LandmarkHeights(self.joints, self.joint_names, self.verts)
        self.jmap = {name: i for i, name in enumerate(self.joint_names)}

        # Precompute bone weights for segment classification
        self._init_segment_masks()

    def _init_segment_masks(self):
        v_weights = self.raw_dict.get("vertex_bone_weights")
        v_indices = self.raw_dict.get("vertex_bone_indices")
        bl = self.raw_dict.get("bone_labels")

        n_v = len(self.verts)
        self.arm_weight_total = np.zeros(n_v, dtype=np.float32)
        self.upperarm_w_L = np.zeros(n_v, dtype=np.float32)
        self.upperarm_w_R = np.zeros(n_v, dtype=np.float32)
        self.forearm_w_L = np.zeros(n_v, dtype=np.float32)
        self.forearm_w_R = np.zeros(n_v, dtype=np.float32)
        self.thigh_w_L = np.zeros(n_v, dtype=np.float32)
        self.thigh_w_R = np.zeros(n_v, dtype=np.float32)
        self.calf_w_L = np.zeros(n_v, dtype=np.float32)
        self.calf_w_R = np.zeros(n_v, dtype=np.float32)

        if v_weights is not None and v_indices is not None and bl is not None:
            if hasattr(v_weights, "cpu"):
                v_weights = v_weights.cpu().numpy()
            if hasattr(v_indices, "cpu"):
                v_indices = v_indices.cpu().numpy()

            for k in range(v_indices.shape[1]):
                col_idx = v_indices[:, k]
                col_w = v_weights[:, k]
                for b_i, b_name in enumerate(bl):
                    b_lower = b_name.lower()
                    mask = (col_idx == b_i)
                    if not np.any(mask):
                        continue

                    if any(x in b_lower for x in ["upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"]):
                        self.arm_weight_total[mask] += col_w[mask]

                    if "upperarm" in b_lower:
                        if ".l" in b_lower:
                            self.upperarm_w_L[mask] += col_w[mask]
                        elif ".r" in b_lower:
                            self.upperarm_w_R[mask] += col_w[mask]
                    elif any(x in b_lower for x in ["lowerarm", "wrist"]):
                        if ".l" in b_lower:
                            self.forearm_w_L[mask] += col_w[mask]
                        elif ".r" in b_lower:
                            self.forearm_w_R[mask] += col_w[mask]
                    elif "upperleg" in b_lower:
                        if ".l" in b_lower:
                            self.thigh_w_L[mask] += col_w[mask]
                        elif ".r" in b_lower:
                            self.thigh_w_R[mask] += col_w[mask]
                    elif any(x in b_lower for x in ["lowerleg", "foot", "ankle"]):
                        if ".l" in b_lower:
                            self.calf_w_L[mask] += col_w[mask]
                        elif ".r" in b_lower:
                            self.calf_w_R[mask] += col_w[mask]

    def _get_slice_centroid_and_ref(
        self,
        p_init: np.ndarray,
        axis: np.ndarray,
        sign_x: Optional[float] = None,
        is_torso: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Slice fitted canonical mesh to find exact cross-sectional centroid and radius."""
        if len(self.faces) == 0:
            return p_init, 0.10

        if is_torso:
            t_verts, t_faces = _get_torso_mesh(self.verts, self.faces, self.raw_dict)
            loops = _slice_mesh_at_z(t_verts, t_faces, float(p_init[2]))
        else:
            loops = _slice_mesh_plane(self.verts, self.faces, p_init, axis)

        if sign_x is not None and loops:
            side_loops = [lp for lp in loops if np.mean(lp[:, 0]) * sign_x > 0]
            if side_loops:
                loops = side_loops

        if not loops:
            return p_init, 0.10

        if is_torso:
            best_lp = max(loops, key=_loop_perimeter)
        else:
            best_lp = min(loops, key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_init))

        centroid = np.mean(best_lp, axis=0)
        n_axis = axis / (np.linalg.norm(axis) + 1e-8)
        u, v = _orthonormal_basis(n_axis)
        diff = best_lp - centroid
        r_ref = float(np.mean(np.sqrt((diff @ u)**2 + (diff @ v)**2)))
        return centroid, r_ref

    def _aggregate_unposed_points(
        self,
        unposed_stops: List[UnposedStopData],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_pts, all_stops, all_view_dirs, all_nn_idx = [], [], [], []
        for s_idx, stop_data in enumerate(unposed_stops):
            pts = stop_data.unposed_points
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_stops.append(np.full(len(pts), s_idx, dtype=np.int32))
            all_view_dirs.append(np.tile(stop_data.camera_view_dir, (len(pts), 1)))
            all_nn_idx.append(stop_data.nearest_vertex_indices)

        if not all_pts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )

        return (
            np.vstack(all_pts),
            np.concatenate(all_stops),
            np.vstack(all_view_dirs),
            np.concatenate(all_nn_idx),
        )

    def measure_ring_contour(
        self,
        p0: np.ndarray,             # (3,) centroid on cutting plane
        normal: np.ndarray,         # (3,) plane normal vector (segment axis)
        pts_pool: np.ndarray,       # (N, 3) candidate points in canonical frame
        pts_stops: np.ndarray,      # (N,) stop indices
        pts_view_dirs: np.ndarray,  # (N, 3) camera viewing vectors
        expected_r_ref: Optional[float] = None,  # Reference radius from model
        min_points_per_bin: int = 2,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, int]:
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-8 or len(pts_pool) == 0:
            return float("nan"), 0.0, float("nan"), np.zeros((0, 3)), np.zeros(0, dtype=bool), 0

        axis = normal / n_norm
        u, v = _orthonormal_basis(axis)

        # 1. Slab filtering: |(p - p0) . n| <= 5 mm
        diff = pts_pool - p0
        proj_axial = diff @ axis
        slab_mask = np.abs(proj_axial) <= self.half_slab
        if np.sum(slab_mask) < 4:
            return float("nan"), 0.0, float("nan"), np.zeros((0, 3)), np.zeros(0, dtype=bool), int(np.sum(slab_mask))

        p_slab = pts_pool[slab_mask]
        s_slab = pts_stops[slab_mask]
        vd_slab = pts_view_dirs[slab_mask]
        d_axial = proj_axial[slab_mask]

        # 2. In-plane polar projection
        p_in_plane = p_slab - np.outer(d_axial, axis)
        rel_2d_u = (p_in_plane - p0) @ u
        rel_2d_v = (p_in_plane - p0) @ v
        rad_dist = np.sqrt(rel_2d_u ** 2 + rel_2d_v ** 2)

        # Plausible radial bounds
        if expected_r_ref is not None and expected_r_ref > 0.02:
            r_min = 0.50 * expected_r_ref
            r_max = 1.50 * expected_r_ref
        else:
            r_min = 0.025
            r_max = 0.35

        rad_mask = (rad_dist >= r_min) & (rad_dist <= r_max)
        if np.sum(rad_mask) < 4:
            return float("nan"), 0.0, float("nan"), np.zeros((0, 3)), np.zeros(0, dtype=bool), int(np.sum(rad_mask))

        u_coords = rel_2d_u[rad_mask]
        v_coords = rel_2d_v[rad_mask]
        r_vals = rad_dist[rad_mask]
        view_dirs = vd_slab[rad_mask]

        angles = np.arctan2(v_coords, u_coords)  # [-pi, pi)

        # 3. Angular binning into K bins
        bin_width = 2.0 * np.pi / self.n_bins
        bin_indices = np.floor((angles + np.pi) / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        bin_radii = np.full(self.n_bins, np.nan, dtype=np.float32)
        covered_mask = np.zeros(self.n_bins, dtype=bool)

        for b_idx in range(self.n_bins):
            in_b = (bin_indices == b_idx)
            n_pts_b = np.sum(in_b)
            if n_pts_b >= min_points_per_bin:
                r_b = r_vals[in_b]
                v_b = view_dirs[in_b]

                theta_b = -np.pi + (b_idx + 0.5) * bin_width
                radial_vec_3d = np.cos(theta_b) * u + np.sin(theta_b) * v

                cos_view = -np.sum(v_b * radial_vec_3d, axis=-1)
                weights = np.maximum(0.15, cos_view)

                med_r = weighted_median(r_b, weights)
                if not math.isnan(med_r):
                    bin_radii[b_idx] = med_r
                    covered_mask[b_idx] = True

        covered_count = int(np.sum(covered_mask))
        coverage_frac = covered_count / float(self.n_bins)

        if covered_count < 4:
            return float("nan"), coverage_frac, float("nan"), np.zeros((0, 3)), covered_mask, len(r_vals)

        # 4. Circular periodic interpolation for missing bins
        interp_radii = bin_radii.copy()
        if covered_count < self.n_bins:
            covered_indices = np.where(covered_mask)[0]
            ext_indices = np.concatenate([covered_indices - self.n_bins, covered_indices, covered_indices + self.n_bins])
            ext_radii = np.concatenate([bin_radii[covered_indices], bin_radii[covered_indices], bin_radii[covered_indices]])

            for b_idx in range(self.n_bins):
                if not covered_mask[b_idx]:
                    interp_radii[b_idx] = float(np.interp(b_idx, ext_indices, ext_radii))

        # 5. Build 3D closed polygon contour
        if self.use_gaussian_smoothing:
            from scipy.ndimage import gaussian_filter1d
            radii_for_poly = gaussian_filter1d(interp_radii, sigma=1.2, mode="wrap")
        else:
            radii_for_poly = interp_radii

        bin_angles = -np.pi + (np.arange(self.n_bins) + 0.5) * bin_width
        contour_2d_u = radii_for_poly * np.cos(bin_angles)
        contour_2d_v = radii_for_poly * np.sin(bin_angles)
        contour_3d = p0 + np.outer(contour_2d_u, u) + np.outer(contour_2d_v, v)

        # Quadratic Pythagorean noise bias correction on edge lengths
        dx = np.diff(np.append(contour_3d[:, 0], contour_3d[0, 0]))
        dy = np.diff(np.append(contour_3d[:, 1], contour_3d[0, 1]))
        dz = np.diff(np.append(contour_3d[:, 2], contour_3d[0, 2]))
        edge_sq = dx**2 + dy**2 + dz**2

        noise_sigma_m = getattr(self, "noise_sigma_m", 0.0)
        if noise_sigma_m > 0 and not self.use_gaussian_smoothing:
            edge_corr = np.sqrt(np.maximum(1e-10, edge_sq - 2.0 * (noise_sigma_m ** 2)))
            perimeter_mm = float(np.sum(edge_corr) * 1000.0)
        else:
            perimeter_mm = float(np.sum(np.sqrt(edge_sq)) * 1000.0)

        return perimeter_mm, coverage_frac, 0.0, contour_3d, covered_mask, len(r_vals)

    def measure_all_observed_rings(
        self,
        unposed_stops: List[UnposedStopData],
    ) -> Dict[str, ObservedRingResult]:
        all_pts, all_stops, all_view_dirs, all_nn_idx = self._aggregate_unposed_points(unposed_stops)
        results: Dict[str, ObservedRingResult] = {}

        if len(all_pts) == 0:
            return results

        # Torso candidate points mask: exclude arms (arm_weight_total < 0.05)
        arm_w_pts = self.arm_weight_total[all_nn_idx]
        torso_mask = (arm_w_pts < 0.05)
        pts_torso = all_pts[torso_mask]
        stops_torso = all_stops[torso_mask]
        vd_torso = all_view_dirs[torso_mask]

        norm_torso = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # ------------------------------------------------------------------
        # 1. Torso Rings: Chest, Waist, Hip, Neck Base
        # ------------------------------------------------------------------

        # A. Chest Circumference
        sp01 = self.lm._j("spine01")
        sp02 = self.lm._j("spine02")
        if sp01 is not None and sp02 is not None:
            chest_z = float(sp02[2] + 0.85 * (sp01[2] - sp02[2]))
        elif sp01 is not None:
            chest_z = float(sp01[2] - 0.02)
        else:
            chest_z = self.lm.floor_z() + (self.lm.head_top_z() - self.lm.floor_z()) * 0.55

        p0_chest, r_ref_chest = self._get_slice_centroid_and_ref(
            np.array([0.0, float(sp01[1]) if sp01 is not None else 0.0, chest_z], dtype=np.float32),
            norm_torso,
            is_torso=True,
        )

        c_val, c_cov, _, c_cont, c_mask, c_n = self.measure_ring_contour(
            p0_chest, norm_torso, pts_torso, stops_torso, vd_torso, expected_r_ref=r_ref_chest
        )
        results["chest_circumference"] = ObservedRingResult(
            name="chest_circumference",
            value_mm=c_val,
            coverage_fraction=c_cov,
            source="observed" if c_cov >= 0.95 else "mixed",
            method="observed_plane_slice_weighted_median",
            unposed_residual_mm=0.0,
            num_points_in_slab=c_n,
            num_covered_bins=int(np.sum(c_mask)),
            total_bins=self.n_bins,
            contour_pts=c_cont,
            covered_bin_mask=c_mask,
        )

        # B. Waist Circumference (minimum trunk circumference across lumbar region)
        sp04 = self.lm._j("spine04")
        if sp04 is not None and sp02 is not None:
            waist_zs = np.linspace(float(sp04[2]), float(sp02[2]), 9)
        else:
            stature_m = self.lm.head_top_z() - self.lm.floor_z()
            waist_zs = np.linspace(self.lm.floor_z() + stature_m * 0.40, self.lm.floor_z() + stature_m * 0.48, 9)

        w_candidates = []
        for z_w in waist_zs:
            p0_w, r_ref_w = self._get_slice_centroid_and_ref(
                np.array([0.0, 0.0, z_w], dtype=np.float32),
                norm_torso,
                is_torso=True,
            )
            val, cov, _, cont, mask, n_pts = self.measure_ring_contour(
                p0_w, norm_torso, pts_torso, stops_torso, vd_torso, expected_r_ref=r_ref_w
            )
            if not math.isnan(val) and val > 0:
                w_candidates.append((val, cov, cont, mask, n_pts))

        if w_candidates:
            w_val, w_cov, w_cont, w_mask, w_n = min(w_candidates, key=lambda x: x[0])
        else:
            w_val, w_cov, w_cont, w_mask, w_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, dtype=bool), 0

        results["waist_circumference"] = ObservedRingResult(
            name="waist_circumference",
            value_mm=w_val,
            coverage_fraction=w_cov,
            source="observed" if w_cov >= 0.95 else "mixed",
            method="observed_plane_slice_weighted_median_min",
            unposed_residual_mm=0.0,
            num_points_in_slab=w_n,
            num_covered_bins=int(np.sum(w_mask)),
            total_bins=self.n_bins,
            contour_pts=w_cont,
            covered_bin_mask=w_mask,
        )

        # C. Hip Circumference (maximum circumference around pelvis/buttocks)
        root_j = self.lm._j("root", "pelvis")
        root_z = float(root_j[2]) if root_j is not None else 0.0
        hip_zs = np.linspace(root_z - 0.04, root_z + 0.06, 7)

        h_candidates = []
        for z_h in hip_zs:
            p0_h, r_ref_h = self._get_slice_centroid_and_ref(
                np.array([0.0, 0.0, z_h], dtype=np.float32),
                norm_torso,
                is_torso=True,
            )
            # Filter arm leakage points at hip height (distance to centerline > max torso bound)
            torso_bound = max(0.26, 1.25 * r_ref_h)
            h_mask_pts = (np.abs(pts_torso[:, 0] - p0_h[0]) < torso_bound)
            val, cov, _, cont, mask, n_pts = self.measure_ring_contour(
                p0_h, norm_torso, pts_torso[h_mask_pts], stops_torso[h_mask_pts], vd_torso[h_mask_pts], expected_r_ref=r_ref_h
            )
            if not math.isnan(val) and val > 0:
                h_candidates.append((val, cov, cont, mask, n_pts))

        if h_candidates:
            h_val, h_cov, h_cont, h_mask, h_n = max(h_candidates, key=lambda x: x[0])
        else:
            h_val, h_cov, h_cont, h_mask, h_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, dtype=bool), 0

        results["hip_circumference"] = ObservedRingResult(
            name="hip_circumference",
            value_mm=h_val,
            coverage_fraction=h_cov,
            source="observed" if h_cov >= 0.95 else "mixed",
            method="observed_plane_slice_weighted_median_max",
            unposed_residual_mm=0.0,
            num_points_in_slab=h_n,
            num_covered_bins=int(np.sum(h_mask)),
            total_bins=self.n_bins,
            contour_pts=h_cont,
            covered_bin_mask=h_mask,
        )

        # D. Neck Base Circumference
        nb_z = self.lm.neck_base_z()
        neck_pt = self.lm._j("neck01", "neck_01", "Neck")
        origin_neck = np.array([0.0, float(neck_pt[1]) if neck_pt is not None else 0.0, nb_z], dtype=np.float32)

        p0_neck, r_ref_neck = self._get_slice_centroid_and_ref(origin_neck, norm_torso, is_torso=True)
        dist_to_neck = np.linalg.norm(pts_torso[:, :2] - p0_neck[:2], axis=1)
        nb_mask = (dist_to_neck < 0.14)

        n_val, n_cov, _, n_cont, n_mask, n_n = self.measure_ring_contour(
            p0_neck, norm_torso, pts_torso[nb_mask], stops_torso[nb_mask], vd_torso[nb_mask], expected_r_ref=r_ref_neck
        )
        results["neck_base_circumference"] = ObservedRingResult(
            name="neck_base_circumference",
            value_mm=n_val,
            coverage_fraction=n_cov,
            source="observed" if n_cov >= 0.95 else "mixed",
            method="observed_plane_slice_weighted_median",
            unposed_residual_mm=0.0,
            num_points_in_slab=n_n,
            num_covered_bins=int(np.sum(n_mask)),
            total_bins=self.n_bins,
            contour_pts=n_cont,
            covered_bin_mask=n_mask,
        )

        # ------------------------------------------------------------------
        # 2. Set B: Limb Circumferences (Left & Right separate)
        # Sliced perpendicular to bone axis with canonical centroid origin
        # ------------------------------------------------------------------
        for side_code, side_label in [("L", "left"), ("R", "right")]:
            sign_x = 1.0 if side_code == "L" else -1.0

            sh = self.lm._j(f"upperarm01.{side_code}", f"shoulder.{side_code}", f"UpperArm.{side_code}")
            el = self.lm._j(f"lowerarm01.{side_code}", f"forearm.{side_code}", f"LowerArm.{side_code}")
            wr = self.lm._j(f"wrist.{side_code}", f"hand.{side_code}", f"Wrist.{side_code}", f"Hand.{side_code}")

            hip_j = self.lm._j(f"upperleg01.{side_code}", f"pelvis.{side_code}", f"UpperLeg.{side_code}")
            kn_j  = self.lm._j(f"lowerleg01.{side_code}", f"shin.{side_code}", f"LowerLeg.{side_code}")
            ft_j  = self.lm._j(f"foot.{side_code}", f"ankle.{side_code}", f"Foot.{side_code}")

            ua_w = self.upperarm_w_L if side_code == "L" else self.upperarm_w_R
            fa_w = self.forearm_w_L if side_code == "L" else self.forearm_w_R
            th_w = self.thigh_w_L if side_code == "L" else self.thigh_w_R
            cf_w = self.calf_w_L if side_code == "L" else self.calf_w_R

            ua_pts_mask = (ua_w[all_nn_idx] > 0.15) & (all_pts[:, 0] * sign_x > 0)
            fa_pts_mask = (fa_w[all_nn_idx] > 0.15) & (all_pts[:, 0] * sign_x > 0)
            th_pts_mask = (th_w[all_nn_idx] > 0.15) & (all_pts[:, 0] * sign_x > 0)
            cf_pts_mask = (cf_w[all_nn_idx] > 0.15) & (all_pts[:, 0] * sign_x > 0)

            # B1. Upper Arm Circumference (Midpoint of shoulder-elbow)
            if sh is not None and el is not None:
                p_mid_ua = 0.5 * (sh + el)
                ua_axis = el - sh
                p0_ua, r_ref_ua = self._get_slice_centroid_and_ref(p_mid_ua, ua_axis, sign_x=sign_x)
                ua_val, ua_cov, _, ua_cont, ua_mask, ua_n = self.measure_ring_contour(
                    p0_ua, ua_axis, all_pts[ua_pts_mask], all_stops[ua_pts_mask], all_view_dirs[ua_pts_mask],
                    expected_r_ref=r_ref_ua
                )
            else:
                ua_val, ua_cov, ua_cont, ua_mask, ua_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, bool), 0

            results[f"upper_arm_circumference_{side_label}"] = ObservedRingResult(
                name=f"upper_arm_circumference_{side_label}",
                value_mm=ua_val,
                coverage_fraction=ua_cov,
                source="observed" if ua_cov >= 0.95 else "mixed",
                method="observed_perpendicular_limb_slice",
                unposed_residual_mm=0.0,
                num_points_in_slab=ua_n,
                num_covered_bins=int(np.sum(ua_mask)),
                total_bins=self.n_bins,
                contour_pts=ua_cont,
                covered_bin_mask=ua_mask,
            )

            # B2. Forearm Circumference (Proximal 25% down elbow-wrist)
            if el is not None and wr is not None:
                fa_axis = wr - el
                p_fa = el + 0.25 * fa_axis
                p0_fa, r_ref_fa = self._get_slice_centroid_and_ref(p_fa, fa_axis, sign_x=sign_x)
                fa_val, fa_cov, _, fa_cont, fa_mask, fa_n = self.measure_ring_contour(
                    p0_fa, fa_axis, all_pts[fa_pts_mask], all_stops[fa_pts_mask], all_view_dirs[fa_pts_mask],
                    expected_r_ref=r_ref_fa
                )
            else:
                fa_val, fa_cov, fa_cont, fa_mask, fa_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, bool), 0

            results[f"forearm_circumference_{side_label}"] = ObservedRingResult(
                name=f"forearm_circumference_{side_label}",
                value_mm=fa_val,
                coverage_fraction=fa_cov,
                source="observed" if fa_cov >= 0.95 else "mixed",
                method="observed_perpendicular_limb_slice",
                unposed_residual_mm=0.0,
                num_points_in_slab=fa_n,
                num_covered_bins=int(np.sum(fa_mask)),
                total_bins=self.n_bins,
                contour_pts=fa_cont,
                covered_bin_mask=fa_mask,
            )

            # B3. Mid-Thigh Circumference (Midpoint of hip-knee)
            if hip_j is not None and kn_j is not None:
                p_mid_th = 0.5 * (hip_j + kn_j)
                th_axis = kn_j - hip_j
                p0_th, r_ref_th = self._get_slice_centroid_and_ref(p_mid_th, th_axis, sign_x=sign_x)
                th_val, th_cov, _, th_cont, th_mask, th_n = self.measure_ring_contour(
                    p0_th, th_axis, all_pts[th_pts_mask], all_stops[th_pts_mask], all_view_dirs[th_pts_mask],
                    expected_r_ref=r_ref_th
                )
            else:
                th_val, th_cov, th_cont, th_mask, th_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, bool), 0

            results[f"mid_thigh_circumference_{side_label}"] = ObservedRingResult(
                name=f"mid_thigh_circumference_{side_label}",
                value_mm=th_val,
                coverage_fraction=th_cov,
                source="observed" if th_cov >= 0.95 else "mixed",
                method="observed_perpendicular_limb_slice",
                unposed_residual_mm=0.0,
                num_points_in_slab=th_n,
                num_covered_bins=int(np.sum(th_mask)),
                total_bins=self.n_bins,
                contour_pts=th_cont,
                covered_bin_mask=th_mask,
            )

            # B4. Calf Maximum Circumference (Proximal 30% down lower leg)
            if kn_j is not None and ft_j is not None:
                calf_axis = ft_j - kn_j
                p_calf = kn_j + 0.30 * calf_axis
                p0_cf, r_ref_cf = self._get_slice_centroid_and_ref(p_calf, calf_axis, sign_x=sign_x)
                calf_val, calf_cov, _, calf_cont, calf_mask, calf_n = self.measure_ring_contour(
                    p0_cf, calf_axis, all_pts[cf_pts_mask], all_stops[cf_pts_mask], all_view_dirs[cf_pts_mask],
                    expected_r_ref=r_ref_cf
                )
            else:
                calf_val, calf_cov, calf_cont, calf_mask, calf_n = float("nan"), 0.0, np.zeros((0, 3)), np.zeros(0, bool), 0

            results[f"max_calf_circumference_{side_label}"] = ObservedRingResult(
                name=f"max_calf_circumference_{side_label}",
                value_mm=calf_val,
                coverage_fraction=calf_cov,
                source="observed" if calf_cov >= 0.95 else "mixed",
                method="observed_perpendicular_limb_slice",
                unposed_residual_mm=0.0,
                num_points_in_slab=calf_n,
                num_covered_bins=int(np.sum(calf_mask)),
                total_bins=self.n_bins,
                contour_pts=calf_cont,
                covered_bin_mask=calf_mask,
            )

        return results
