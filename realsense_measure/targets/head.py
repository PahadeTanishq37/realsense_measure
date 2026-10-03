"""
HeadTarget: geometric 3D cranial envelope estimation for human head scans.

Registered under the name "head". See targets/base.py for the extension contract.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from config import PreprocessConfig
from preprocessing import extract_largest_cluster
from targets.base import Target, register_target


def _sanitize_points(pts: np.ndarray) -> np.ndarray:
    """Filter out NaN, Inf, and non-finite coordinate rows."""
    if len(pts) == 0:
        return pts
    finite_mask = np.all(np.isfinite(pts), axis=1)
    return pts[finite_mask]


def _check_face_support_2d(
    face_pts: np.ndarray,
    axis: int,
    other_axes: list[int],
    nominal_extents: list[float],
) -> tuple[bool, str, int]:
    """
    Check point support and 2D spatial dispersion on a candidate boundary slab.

    Returns
    -------
    tuple[bool, str, int]
        (is_supported, diagnostic_reason, point_count)
    """
    n_pts = len(face_pts)
    if n_pts < 15:
        return False, f"too few points ({n_pts} < 15 threshold)", n_pts

    u = face_pts[:, other_axes[0]]
    v = face_pts[:, other_axes[1]]
    L_u = nominal_extents[other_axes[0]]
    L_v = nominal_extents[other_axes[1]]

    u_span = float(np.max(u) - np.min(u))
    v_span = float(np.max(v) - np.min(v))

    u_rel = u_span / max(L_u, 1e-4)
    v_rel = v_span / max(L_v, 1e-4)

    # For anatomical facial protrusions (e.g. nose tip) and rounded poles, span is >= 5%
    if u_rel < 0.05 or v_rel < 0.05:
        return False, f"narrow orthogonal coverage (u_span={u_rel:.2f}, v_span={v_rel:.2f} < 0.05)", n_pts

    # 3x3 grid occupancy
    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))
    u_bins = np.digitize(u, np.linspace(u_min, u_max, 4)) - 1
    v_bins = np.digitize(v, np.linspace(v_min, v_max, 4)) - 1
    valid_bins = (u_bins >= 0) & (u_bins < 3) & (v_bins >= 0) & (v_bins < 3)
    grid = np.zeros((3, 3), dtype=int)
    for ub, vb in zip(u_bins[valid_bins], v_bins[valid_bins]):
        grid[ub, vb] += 1
    occupied_cells = int(np.sum(grid > 0))

    if occupied_cells < 3:
        return False, f"clustered boundary points ({occupied_cells}/9 cells occupied)", n_pts

    return True, "face supported", n_pts


@register_target
class HeadTarget(Target):
    """
    Segment and estimate 3D geometric surface-envelope dimensions of a human head.

    Calculates:
      - Cranial envelope breadth (Left-Right lateral width)
      - Cranial envelope length (Anterior-Posterior depth)
      - Cranial envelope height (Superior-Inferior height)
      - Reconstructed 3D boundary endpoints and volume
    """

    name: str = "head"

    def __init__(self, preprocess_cfg: PreprocessConfig | None = None) -> None:
        self.cfg: PreprocessConfig = preprocess_cfg or PreprocessConfig()

    def is_plausible_head(self, pcd: o3d.geometry.PointCloud) -> tuple[bool, str]:
        """
        Fast geometric sanity check before cranial envelope fitting.

        Checks:
          1. len(pcd.points) >= 200.
          2. Finite coordinates check.
          3. Bounding box extents: aspect ratio <= 3.5.
          4. Dimension bounds: smallest span >= 0.06 m, largest span <= 0.60 m.
        """
        raw_pts = np.asarray(pcd.points)
        pts = _sanitize_points(raw_pts)
        n_pts = len(pts)
        if n_pts < 200:
            return False, f"too few points ({n_pts}) to evaluate head geometry (need >= 200)"

        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        extents = maxs - mins
        sorted_extents = sorted(extents, reverse=True)
        max_ext, min_ext = float(sorted_extents[0]), float(sorted_extents[2])

        if min_ext < 0.06:
            return False, f"degenerate 2D or sliver geometry (minimum span {min_ext * 1000.0:.1f} mm < 60 mm)"
        if max_ext > 0.60:
            return False, f"excessive spatial extent ({max_ext * 1000.0:.1f} mm > 600 mm)"

        aspect_ratio = max_ext / max(min_ext, 1e-4)
        if aspect_ratio > 3.5:
            return False, f"implausible aspect ratio ({aspect_ratio:.1f}:1 > 3.5:1 for human head)"

        return True, ""

    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Isolate cranial geometry, separating disconnected outliers and torso attachments.
        """
        raw_pts = np.asarray(pcd.points)
        pts = _sanitize_points(raw_pts)
        if len(pts) < 50:
            return pcd

        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(pts)

        labels = np.array(
            cleaned_pcd.cluster_dbscan(
                eps=max(self.cfg.cluster_eps_m, 0.025),
                min_points=min(self.cfg.cluster_min_points, 20),
                print_progress=False,
            )
        )
        valid_mask = labels >= 0
        if valid_mask.any():
            valid_labels = labels[valid_mask]
            largest_label = int(np.bincount(valid_labels).argmax())
            indices = np.where(labels == largest_label)[0].tolist()
            dominant = cleaned_pcd.select_by_index(indices)
        else:
            dominant = cleaned_pcd

        pts_dom = np.asarray(dominant.points)
        if len(pts_dom) < 200:
            return dominant

        # Check for neck constriction along vertical long axis
        C = np.mean(pts_dom, axis=0)
        Pc = pts_dom - C
        cov = Pc.T @ Pc / len(Pc)
        evals, evecs = np.linalg.eigh(cov)
        v_long = evecs[:, np.argmax(evals)]  # principal long axis

        proj = Pc @ v_long
        p_min, p_max = float(np.min(proj)), float(np.max(proj))
        span_long = p_max - p_min

        # If long span > 30 cm, check if neck/shoulders are attached
        if span_long > 0.30:
            n_bins = 25
            bins = np.linspace(p_min, p_max, n_bins)
            areas: list[float] = []
            bin_centers: list[float] = []

            for i in range(len(bins) - 1):
                mask = (proj >= bins[i]) & (proj < bins[i + 1])
                s_pts = Pc[mask]
                if len(s_pts) < 10:
                    areas.append(0.0)
                else:
                    orth_proj = s_pts - (s_pts @ v_long[:, None]) * v_long
                    w_span = np.ptp(orth_proj[:, 0]) if len(orth_proj) else 0.0
                    d_span = np.ptp(orth_proj[:, 1]) if len(orth_proj) else 0.0
                    areas.append(float(w_span * d_span))
                bin_centers.append(float((bins[i] + bins[i + 1]) / 2.0))

            # Orient v_long so head is always on the positive side (areas[0] shoulders > areas[-1] head)
            if areas and areas[-1] > areas[0]:
                v_long = -v_long
                proj = -proj
                p_min, p_max = float(np.min(proj)), float(np.max(proj))
                bins = np.linspace(p_min, p_max, n_bins)
                areas = []
                bin_centers = []
                for i in range(len(bins) - 1):
                    mask = (proj >= bins[i]) & (proj < bins[i + 1])
                    s_pts = Pc[mask]
                    if len(s_pts) < 10:
                        areas.append(0.0)
                    else:
                        orth_proj = s_pts - (s_pts @ v_long[:, None]) * v_long
                        w_span = np.ptp(orth_proj[:, 0]) if len(orth_proj) else 0.0
                        d_span = np.ptp(orth_proj[:, 1]) if len(orth_proj) else 0.0
                        areas.append(float(w_span * d_span))
                    bin_centers.append(float((bins[i] + bins[i + 1]) / 2.0))

            mid_idx = len(areas) // 2
            candidate_min_idx = -1
            min_area_val = float("inf")
            for idx in range(3, mid_idx + 4):
                if 0 < idx < len(areas) - 1:
                    if areas[idx] < areas[idx - 1] and areas[idx] <= areas[idx + 1]:
                        if areas[idx] < min_area_val:
                            min_area_val = areas[idx]
                            candidate_min_idx = idx

            if candidate_min_idx != -1 and min_area_val > 0:
                head_mass_above = max(areas[candidate_min_idx:])
                shoulder_mass_below = max(areas[:candidate_min_idx]) if candidate_min_idx > 0 else 0.0

                if min_area_val <= 0.65 * head_mass_above and shoulder_mass_below >= 1.40 * min_area_val:
                    cut_val = bin_centers[candidate_min_idx]
                    head_mask = proj >= cut_val
                    if np.sum(head_mask) >= 150:
                        seg_pcd = o3d.geometry.PointCloud()
                        seg_pcd.points = o3d.utility.Vector3dVector(pts_dom[head_mask])
                        return seg_pcd

        return dominant

    def _estimate_anatomical_frame(
        self, pts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, list[str]]:
        """
        Estimate canonical Left-Right (u_LR), Superior-Inferior (u_SI), and
        Anterior-Posterior (u_AP) unit vectors.

        Returns
        -------
        tuple[u_LR, u_SI, u_AP, symmetry_err_mm, is_confident, diagnostics]
        """
        diag: list[str] = []
        C = np.mean(pts, axis=0)
        Pc = pts - C

        # PCA decomposition
        cov = Pc.T @ Pc / len(Pc)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        v_long = evecs[:, 0]   # axis with largest variance (height candidate)
        v_mid = evecs[:, 1]    # axis with second largest variance (depth candidate)
        v_short = evecs[:, 2]  # axis with smallest variance (lateral breadth candidate)

        # 1. Verify bilateral symmetry on lateral candidate (v_short)
        tree = cKDTree(Pc)
        P_ref = Pc - 2.0 * (Pc @ v_short[:, None]) * v_short
        dists, _ = tree.query(P_ref, k=1)
        sym_err_mm = float(np.mean(dists) * 1000.0)

        u_LR = v_short
        u_SI = v_long
        u_AP = np.cross(u_SI, u_LR)
        norm_ap = np.linalg.norm(u_AP)
        if norm_ap > 1e-6:
            u_AP = u_AP / norm_ap
        else:
            u_AP = v_mid

        is_confident = True
        if sym_err_mm > 12.0:
            diag.append(f"High bilateral symmetry residual ({sym_err_mm:.1f} mm > 12.0 mm)")
            is_confident = False

        # 2. Check Anterior-Posterior facial relief asymmetry (nose/brow protrusion)
        ap_proj = Pc @ u_AP
        skew_ap = float(np.mean(ap_proj**3) / max((np.mean(ap_proj**2)) ** 1.5, 1e-8))

        pos_extent = float(np.percentile(ap_proj, 99.5))
        neg_extent = float(np.abs(np.percentile(ap_proj, 0.5)))
        asym_ratio = abs(pos_extent - neg_extent) / max(pos_extent + neg_extent, 1e-4)

        if asym_ratio >= 0.015 or abs(skew_ap) >= 0.015:
            if pos_extent < neg_extent:
                u_AP = -u_AP
        else:
            # Ambiguous orientation (e.g. pure sphere or symmetric oval)
            diag.append("Ambiguous Anterior-Posterior orientation (no detectable facial relief or skewness)")
            is_confident = False

        # 3. Check Superior-Inferior crown vs chin taper
        si_proj = Pc @ u_SI
        top_mask = si_proj > 0.33 * np.max(si_proj)
        bot_mask = si_proj < 0.33 * np.min(si_proj)

        if np.sum(top_mask) > 10 and np.sum(bot_mask) > 10:
            top_var = np.var(Pc[top_mask] @ u_LR)
            bot_var = np.var(Pc[bot_mask] @ u_LR)
            if top_var < 0.70 * bot_var:
                u_SI = -u_SI

        return u_LR, u_SI, u_AP, sym_err_mm, is_confident, diag

    def measure(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        Estimate 3D geometric cranial surface-envelope dimensions using 3D point-to-point
        Euclidean distances between identified physical boundary endpoints.
        """
        raw_pts = np.asarray(pcd.points)
        pts = _sanitize_points(raw_pts)
        n_pts = len(pts)

        fallback_envelope = {
            "breadth_m": 0.0,
            "length_m": 0.0,
            "height_m": 0.0,
            "breadth_mm": 0.0,
            "length_mm": 0.0,
            "height_mm": 0.0,
        }
        fallback_bbox = {
            "length_m": 0.0,
            "width_m": 0.0,
            "height_m": 0.0,
            "volume_m3": None,
            "estimated_volume_m3": 0.0,
        }

        if n_pts < 50:
            return {
                "target": self.name,
                "status": "unverified",
                "reason": f"Insufficient points in segmented head cloud ({n_pts} < 50)",
                "num_points": n_pts,
                "cranial_envelope_3d": fallback_envelope,
                "endpoints_3d": {},
                "anatomical_landmarks": None,
                "oriented_bbox": fallback_bbox,
                "head_dimensions_mm": {
                    "cranial_width_mm": 0.0,
                    "cranial_depth_mm": 0.0,
                    "head_height_mm": 0.0,
                },
                "volume_m3": None,
                "estimated_volume_m3": 0.0,
            }

        is_plaus, plaus_reason = self.is_plausible_head(pcd)
        if not is_plaus:
            return {
                "target": self.name,
                "status": "unverified",
                "reason": plaus_reason,
                "num_points": n_pts,
                "cranial_envelope_3d": fallback_envelope,
                "endpoints_3d": {},
                "anatomical_landmarks": None,
                "oriented_bbox": fallback_bbox,
                "head_dimensions_mm": {
                    "cranial_width_mm": 0.0,
                    "cranial_depth_mm": 0.0,
                    "head_height_mm": 0.0,
                },
                "volume_m3": None,
                "estimated_volume_m3": 0.0,
            }

        # ------------------------------------------------------------------
        # Anatomical alignment & coordinate frame
        # ------------------------------------------------------------------
        C = np.mean(pts, axis=0)
        u_LR, u_SI, u_AP, sym_err_mm, is_confident, frame_diags = self._estimate_anatomical_frame(pts)

        axes_dict = {
            "breadth": u_LR,
            "length": u_AP,
            "height": u_SI,
        }
        axis_names = ["breadth", "length", "height"]
        axes_matrix = np.column_stack([u_LR, u_AP, u_SI])  # 3x3

        # Project points into canonical anatomical frame
        local_pts = (pts - C) @ axes_matrix
        nominal_extents = [
            float(np.max(local_pts[:, i]) - np.min(local_pts[:, i])) for i in range(3)
        ]

        # ------------------------------------------------------------------
        # Approach 1: 3D Point-to-Point Euclidean Distance Formulation
        # ------------------------------------------------------------------
        endpoints_3d = {}
        cranial_dims: dict[str, float] = {}
        boundary_checks: dict[str, dict] = {}
        all_faces_supported = True
        diagnostics: list[str] = list(frame_diags)

        trim_percentile = 0.5
        for axis_idx, name in enumerate(axis_names):
            other_axes = [i for i in range(3) if i != axis_idx]
            vals = local_pts[:, axis_idx]
            u_vec = axes_matrix[:, axis_idx]
            span = nominal_extents[axis_idx]

            lo_val = float(np.percentile(vals, trim_percentile))
            hi_val = float(np.percentile(vals, 100.0 - trim_percentile))

            # Reconstructed 3D endpoints in shared coordinate frame
            P1 = C + lo_val * u_vec
            P2 = C + hi_val * u_vec

            # Direct 3D Euclidean distance (no unexplained factor)
            distance_m = float(np.sqrt(np.sum((P2 - P1) ** 2)))
            distance_mm = distance_m * 1000.0

            cranial_dims[f"{name}_m"] = round(distance_m, 4)
            cranial_dims[f"{name}_mm"] = round(distance_mm, 1)

            endpoints_3d[name] = {
                "P1": [round(float(v), 5) for v in P1],
                "P2": [round(float(v), 5) for v in P2],
                "distance_m": round(distance_m, 4),
                "distance_mm": round(distance_mm, 1),
                "axis_vector": [round(float(v), 4) for v in u_vec],
            }

            # Check boundary support on both sides
            tol = max(0.005, 0.03 * span)
            near_pts = local_pts[np.abs(vals - lo_val) <= tol]
            far_pts = local_pts[np.abs(vals - hi_val) <= tol]

            near_ok, near_msg, n_near = _check_face_support_2d(near_pts, axis_idx, other_axes, nominal_extents)
            far_ok, far_msg, n_far = _check_face_support_2d(far_pts, axis_idx, other_axes, nominal_extents)

            boundary_checks[name] = {
                "near_supported": near_ok,
                "near_reason": near_msg,
                "near_points": n_near,
                "far_supported": far_ok,
                "far_reason": far_msg,
                "far_points": n_far,
            }

            if not near_ok or not far_ok:
                all_faces_supported = False
                if not near_ok:
                    diagnostics.append(f"{name} lower boundary: {near_msg}")
                if not far_ok:
                    diagnostics.append(f"{name} upper boundary: {far_msg}")

        # Plausibility range checks
        breadth_m = cranial_dims["breadth_m"]
        length_m = cranial_dims["length_m"]
        height_m = cranial_dims["height_m"]

        dim_in_range = bool(
            0.10 <= breadth_m <= 0.24
            and 0.13 <= length_m <= 0.28
            and 0.16 <= height_m <= 0.35
        )
        if not dim_in_range:
            diagnostics.append(
                f"Dimensions ({breadth_m*1000:.0f}x{length_m*1000:.0f}x{height_m*1000:.0f} mm) fall outside typical anthropometric range"
            )

        is_validated = bool(is_confident and all_faces_supported and dim_in_range)

        # Volume: Ellipsoid / OBB volume
        # V = 4/3 * pi * (breadth/2) * (length/2) * (height/2)
        ellipsoid_vol_m3 = round(float((4.0 / 3.0) * np.pi * (breadth_m / 2.0) * (length_m / 2.0) * (height_m / 2.0)), 6)

        try:
            obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        except AttributeError:
            obb = pcd.get_oriented_bounding_box(robust=True)

        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [round(float(v), 4) for v in np.asarray(aabb.get_extent())]

        return {
            "target": self.name,
            "status": "validated" if is_validated else "unverified",
            "measurement_type": "3d_geometric_cranial_envelope",
            "num_points": n_pts,
            "cranial_envelope_3d": cranial_dims,
            "endpoints_3d": endpoints_3d,
            "anatomical_landmarks": None,  # null: purely geometric outer envelope
            "oriented_bbox": {
                "length_m": round(length_m, 4),
                "width_m": round(breadth_m, 4),
                "height_m": round(height_m, 4),
                "volume_m3": ellipsoid_vol_m3 if is_validated else None,
                "estimated_volume_m3": ellipsoid_vol_m3,
                "center": [round(float(v), 5) for v in C],
                "rotation": [round(float(v), 5) for v in axes_matrix.flatten()],
            },
            # Preserved for backward compatibility
            "head_dimensions_mm": {
                "cranial_width_mm": cranial_dims["breadth_mm"],
                "cranial_depth_mm": cranial_dims["length_mm"],
                "head_height_mm": cranial_dims["height_mm"],
            },
            "boundary_checks": boundary_checks,
            "symmetry_residual_mm": round(sym_err_mm, 2),
            "diagnostics": diagnostics if not is_validated else [],
            "axis_aligned_bbox_extent_m": aabb_extent,
            "geometry_for_viz": obb,
        }

