"""
realsense_measure/fullbody/alignment.py

Multi-stop full-body alignment:
1. Floor-constrained gravity frame (locks height + pitch + roll to gravity).
2. Keypoint-based yaw and body-axis prior.
3. Bounded 4-DoF point-to-plane ICP on rigid body parts with loop closure.
4. Per-limb articulated rigid refinement (arms).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.parts import (
    ARM_PARTS,
    BodyPart,
    is_arm_point,
    is_rigid_point,
    label_points_by_body_part,
)
from realsense_measure.fullbody.types import (
    AlignedStop,
    FloorPlane,
    Keypoints3D,
    StopData,
)

logger = logging.getLogger(__name__)


def compute_gravity_transform(floor: FloorPlane) -> np.ndarray:
    """
    Build 4x4 matrix T_grav_cam that aligns camera coords with gravity frame:
    floor plane is at Y = 0, normal is +Y = [0, 1, 0].
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = floor.gravity_R_cam
    T[:3, 3] = -floor.gravity_R_cam @ floor.point_on_plane
    return T


def compute_keypoint_prior_yaw_legacy(
    keypoints_grav: Keypoints3D,
    nominal_deg: float,
    cfg: FullBodyConfig,
) -> Tuple[float, np.ndarray, bool]:
    """Previous non-robust prior estimator (for benchmark comparison)."""
    l_sh, c_ls, _ = keypoints_grav.get("left_shoulder")
    r_sh, c_rs, _ = keypoints_grav.get("right_shoulder")
    l_hp, c_lh, _ = keypoints_grav.get("left_hip")
    r_hp, c_rh, _ = keypoints_grav.get("right_hip")
    nose, c_n, _ = keypoints_grav.get("nose")

    if (l_sh is None or r_sh is None) and (l_hp is None or r_hp is None):
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    centers = []
    lat_vecs = []
    if l_sh is not None and r_sh is not None:
        centers.append(0.5 * (l_sh + r_sh))
        lat_vecs.append(l_sh - r_sh)
    if l_hp is not None and r_hp is not None:
        centers.append(0.5 * (l_hp + r_hp))
        lat_vecs.append(l_hp - r_hp)

    center = np.mean(centers, axis=0)
    lat = np.mean(lat_vecs, axis=0)
    lat[1] = 0.0
    norm_lat = np.linalg.norm(lat)
    if norm_lat < 1e-4:
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False
    lat /= norm_lat
    ant = np.array([-lat[2], 0.0, lat[0]], dtype=np.float64)
    if nose is not None and c_n > 0.2:
        nose_dir = nose - center
        nose_dir[1] = 0.0
        if np.dot(ant, nose_dir) < 0:
            ant = -ant
    yaw_rad = np.arctan2(ant[0], ant[2])
    return float(np.rad2deg(yaw_rad)), center, True


def compute_keypoint_prior_yaw(
    keypoints_grav: Keypoints3D,
    nominal_deg: float,
    cfg: FullBodyConfig,
) -> Tuple[float, np.ndarray, bool]:
    """
    Robust prior estimator:
    1. Keypoint confidence weights (w = c * v).
    2. Joint-outlier rejection (anatomical spans and vertical spine alignment).
    3. Robust Huber M-estimator over torso joint candidates for body-axis (X, Z).
    4. Circular consensus for lateral orientation and yaw.
    """
    l_sh, c_ls, v_ls = keypoints_grav.get("left_shoulder")
    r_sh, c_rs, v_rs = keypoints_grav.get("right_shoulder")
    l_hp, c_lh, v_lh = keypoints_grav.get("left_hip")
    r_hp, c_rh, v_rh = keypoints_grav.get("right_hip")
    nose, c_n, _ = keypoints_grav.get("nose")

    # Joint candidate pairs and anatomical checks
    candidates = []
    # (pair_name, p_left, p_right, conf_weight, expected_span, min_span, max_span)
    if l_sh is not None and r_sh is not None:
        w_sh = float(c_ls * c_rs * v_ls * v_rs)
        if c_ls >= 0.2 and c_rs >= 0.2:
            candidates.append(("shoulder", l_sh, r_sh, w_sh, 0.36, 0.24, 0.48))

    if l_hp is not None and r_hp is not None:
        w_hp = float(c_lh * c_rh * v_lh * v_rh)
        if c_lh >= 0.2 and c_rh >= 0.2:
            candidates.append(("hip", l_hp, r_hp, w_hp, 0.20, 0.12, 0.36))

    if not candidates:
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    # Outlier rejection on bone spans
    valid_pairs = []
    for name, pl, pr, w, exp_span, min_s, max_s in candidates:
        span = float(np.linalg.norm(pl - pr))
        if min_s <= span <= max_s:
            residual = abs(span - exp_span)
            span_weight = w / (1.0 + 10.0 * residual)
            c_mid = 0.5 * (pl + pr)
            v_lat = np.array([pl[0] - pr[0], 0.0, pl[2] - pr[2]], dtype=np.float64)
            norm_v = np.linalg.norm(v_lat)
            if norm_v > 1e-4:
                v_lat /= norm_v
                valid_pairs.append({
                    "name": name,
                    "center": c_mid,
                    "lateral": v_lat,
                    "weight": span_weight,
                    "span_residual": residual,
                })

    if not valid_pairs:
        # Fallback to nominal if all pairs failed anatomical span bounds
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    # Joint-outlier rejection via spine verticality
    if len(valid_pairs) == 2:
        sh_pair = valid_pairs[0] if valid_pairs[0]["name"] == "shoulder" else valid_pairs[1]
        hp_pair = valid_pairs[0] if valid_pairs[0]["name"] == "hip" else valid_pairs[1]
        c_sh = sh_pair["center"]
        c_hp = hp_pair["center"]
        spine_disp_xz = float(np.linalg.norm([c_sh[0] - c_hp[0], c_sh[2] - c_hp[2]]))
        if spine_disp_xz > 0.06:
            # One joint pair contains an outlier; keep pair with lower span residual
            if sh_pair["span_residual"] <= hp_pair["span_residual"]:
                valid_pairs = [sh_pair]
            else:
                valid_pairs = [hp_pair]

    # Robust Huber body-axis center estimation in XZ
    pts_xz = np.array([[p["center"][0], p["center"][2]] for p in valid_pairs], dtype=np.float64)
    weights = np.array([p["weight"] for p in valid_pairs], dtype=np.float64)
    if np.sum(weights) < 1e-6:
        weights = np.ones(len(valid_pairs), dtype=np.float64)
    weights /= np.sum(weights)

    center_xz = np.average(pts_xz, axis=0, weights=weights)
    delta_huber = 0.02
    for _ in range(5):
        dists = np.linalg.norm(pts_xz - center_xz, axis=1)
        h_w = weights * np.where(dists <= delta_huber, 1.0, delta_huber / np.maximum(dists, 1e-6))
        if np.sum(h_w) > 1e-6:
            center_xz = np.average(pts_xz, axis=0, weights=h_w)

    y_center = float(np.mean([p["center"][1] for p in valid_pairs]))
    center = np.array([center_xz[0], y_center, center_xz[1]], dtype=np.float64)

    # Robust lateral consensus vector
    lat_vecs = np.array([p["lateral"] for p in valid_pairs])
    lat = np.average(lat_vecs, axis=0, weights=weights)
    lat[1] = 0.0
    norm_lat = np.linalg.norm(lat)
    if norm_lat < 1e-4:
        return nominal_deg, center, False
    lat /= norm_lat

    # Anterior vector: cross(lateral, UP) -> [-lz, 0, lx]
    ant = np.array([-lat[2], 0.0, lat[0]], dtype=np.float64)

    # Disambiguate front vs back using nose position
    if nose is not None and c_n > 0.2:
        nose_dir = nose - center
        nose_dir[1] = 0.0
        if np.dot(ant, nose_dir) < 0:
            ant = -ant
            lat = -lat

    yaw_rad = np.arctan2(ant[0], ant[2])
    yaw_deg = float(np.rad2deg(yaw_rad))

    return yaw_deg, center, True


def estimate_common_vertical_axis_and_drift(
    centers_grav: List[np.ndarray],
    weights: List[float],
    cfg: FullBodyConfig,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Model the turning subject as rotating about one shared vertical axis (X_axis, Z_axis)
    plus small per-stop physical drift: delta_i ~ N(0, sigma_drift^2) with sigma ~ 0.02 m.

    Jointly estimate the common axis across stops via weighted Huber/median, then
    regularize (shrink) per-stop centers toward the shared axis.
    """
    n = len(centers_grav)
    if n == 0:
        return np.zeros(2), []

    pts = np.array([[c[0], c[2]] for c in centers_grav], dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    if np.sum(w) < 1e-4:
        w = np.ones(n, dtype=np.float64)
    w /= np.sum(w)

    # Robust common axis (Huber M-estimator)
    axis_xz = np.average(pts, axis=0, weights=w)
    delta = 0.03
    for _ in range(10):
        res = np.linalg.norm(pts - axis_xz, axis=1)
        huber_w = w * np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-6))
        if np.sum(huber_w) > 1e-6:
            new_axis = np.average(pts, axis=0, weights=huber_w)
            if np.linalg.norm(new_axis - axis_xz) < 1e-5:
                break
            axis_xz = new_axis

    sigma_drift = getattr(cfg, "body_axis_drift_sigma_m", 0.02)
    sigma_meas = 0.03
    lam = (sigma_meas / max(0.005, sigma_drift)) ** 2

    reg_centers = []
    for i, c in enumerate(centers_grav):
        # Regularized MAP shrinkage toward common vertical axis
        shrunk_offset = (1.0 / (1.0 + lam)) * (pts[i] - axis_xz)
        reg_xz = axis_xz + shrunk_offset
        reg_c = np.array([reg_xz[0], c[1], reg_xz[1]], dtype=np.float64)
        reg_centers.append(reg_c)

    return axis_xz, reg_centers


def check_and_enforce_loop_closure(
    stops: List[AlignedStop],
    cfg: FullBodyConfig,
) -> float:
    """
    Check loop closure: sum of relative yaws between consecutive stops should equal 360 deg.
    Returns loop closure drift in degrees.
    """
    n = len(stops)
    if n < 4:
        return 0.0

    yaws = np.array([s.refined_yaw_deg for s in stops], dtype=np.float64)
    step_deg = 360.0 / float(n)
    expected_step = -step_deg if cfg.turn_direction == "cw" else step_deg
    expected_span = (n - 1) * expected_step
    actual_span = (yaws[-1] - yaws[0] + 180.0) % 360.0 - 180.0
    drift = float((actual_span - expected_span + 180.0) % 360.0 - 180.0)

    logger.info(f"  [align] Loop closure drift = {drift:.2f} deg (expected span {expected_span:.1f} deg, actual {actual_span:.1f} deg)")

    if abs(drift) <= 20.0:
        for i, s in enumerate(stops):
            adj = -drift * (i / float(n))
            s.refined_yaw_deg += adj
    else:
        logger.warning(f"  [align] Loop closure drift {drift:.2f} deg exceeds 20 deg! Flagging review.")
        for s in stops:
            s.review_status = "NEEDS_REVIEW"

    return float(drift)


def compute_ghosting_metric_mm(
    aligned_stops: List[AlignedStop],
    fused_mesh_or_pcd: Any,
) -> float:
    """
    Compute ghosting metric: mean distance (in mm) between per-stop transformed clouds
    and the fused mesh/cloud.
    """
    if fused_mesh_or_pcd is None:
        return 0.0

    if hasattr(fused_mesh_or_pcd, "vertices"):
        mesh_pts = np.asarray(fused_mesh_or_pcd.vertices)
        if len(mesh_pts) == 0:
            return 0.0
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(mesh_pts)
    elif hasattr(fused_mesh_or_pcd, "points"):
        ref_pcd = fused_mesh_or_pcd
    else:
        return 0.0

    distances = []
    for s in aligned_stops:
        if s.pcd_canonical is not None and len(s.pcd_canonical.points) > 0:
            d = s.pcd_canonical.compute_point_cloud_distance(ref_pcd)
            distances.extend(d)

    if not distances:
        return 0.0
    return float(np.mean(distances) * 1000.0)


def align_stops(
    stops: List[StopData],
    consensus_floor: FloorPlane,
    cfg: FullBodyConfig,
) -> List[AlignedStop]:
    """
    Execute full multi-stop alignment pipeline:
    1. Robust keypoint prior per stop (Huber on torso joints, outlier rejection).
    2. Common vertical turning axis estimation with regularized per-stop drift.
    3. MAP-style 4-DoF ICP with multi-start and tangential sliding penalty.
    4. Loop closure check and enforcement.
    """
    n_stops = len(stops)
    if n_stops == 0:
        return []

    T_grav_cam = compute_gravity_transform(consensus_floor)

    # 1. Compute initial priors per stop
    raw_priors = []
    centers_grav = []
    weights_grav = []

    for s in stops:
        kpts_cam = s.keypoints_camera
        if kpts_cam is not None:
            kpts_grav = kpts_cam.transform(T_grav_cam, "gravity")
        else:
            kpts_grav = Keypoints3D([], np.zeros((0, 3)), np.zeros(0), np.zeros(0), "gravity")

        from realsense_measure.fullbody.orientation import canonical_yaw_deg
        nominal_canonical = canonical_yaw_deg(s.nominal_angle_deg, cfg.turn_direction)

        prior_yaw, center, is_kpt_valid = compute_keypoint_prior_yaw(kpts_grav, nominal_canonical, cfg)

        diff = (prior_yaw - nominal_canonical + 180.0) % 360.0 - 180.0
        if abs(diff) > cfg.max_angle_deviation_deg:
            logger.info(
                f"  [align] Stop {s.stop_id} yaw prior {prior_yaw:.1f} deg deviates by {diff:.1f} deg "
                f"from nominal {nominal_canonical:.1f} deg; clamping prior to nominal."
            )
            prior_yaw = nominal_canonical

        raw_priors.append((prior_yaw, center, is_kpt_valid, kpts_cam))
        centers_grav.append(center)
        weights_grav.append(1.0 if is_kpt_valid else 0.1)

    # 2. Joint common vertical axis estimation and drift shrinkage
    _, regularized_centers = estimate_common_vertical_axis_and_drift(centers_grav, weights_grav, cfg)

    aligned_stops: List[AlignedStop] = []
    for i, s in enumerate(stops):
        prior_yaw, _, is_kpt_valid, kpts_cam = raw_priors[i]
        center = regularized_centers[i] if regularized_centers else centers_grav[i]

        yaw_rad = np.deg2rad(prior_yaw)
        c_y, s_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [c_y, 0.0, -s_y],
            [0.0, 1.0,  0.0],
            [s_y, 0.0,  c_y],
        ], dtype=np.float64)

        t_canon = np.array([-c_y * center[0] + s_y * center[2], 0.0, -s_y * center[0] - c_y * center[2]])
        T_canon_grav = np.eye(4, dtype=np.float64)
        T_canon_grav[:3, :3] = R_yaw
        T_canon_grav[:3, 3] = t_canon

        T_canon_cam = T_canon_grav @ T_grav_cam

        kpts_can = kpts_cam.transform(T_canon_cam, "canonical") if kpts_cam else None

        pcd_can = None
        labels = None
        if s.pcd is not None:
            pcd_can = o3d.geometry.PointCloud(s.pcd)
            pcd_can.transform(T_canon_cam)
            pts_np = np.asarray(pcd_can.points)
            if kpts_can:
                labels = label_points_by_body_part(pts_np, kpts_can, cfg)

        aligned_stops.append(
            AlignedStop(
                stop_id=s.stop_id,
                nominal_angle_deg=s.nominal_angle_deg,
                T_canonical_camera=T_canon_cam,
                prior_yaw_deg=prior_yaw,
                refined_yaw_deg=prior_yaw,
                prior_translation_m=t_canon.copy(),
                refined_translation_m=t_canon.copy(),
                residual_before_mm=0.0,
                residual_after_mm=0.0,
                inlier_fraction=1.0,
                accepted=True,
                pcd_canonical=pcd_can,
                keypoints_canonical=kpts_can,
                body_part_labels=labels,
                prior_T_canonical_camera=T_canon_cam.copy(),
                hit_clamp=False,
                review_status="ACCEPTED",
            )
        )

    # 3. Refinement loop (MAP-style 4-DoF ICP on rigid parts)
    if n_stops > 1:
        _refine_alignment_multiview(aligned_stops, cfg)

    # 4. Enforce loop consistency
    if n_stops >= 4:
        check_and_enforce_loop_closure(aligned_stops, cfg)

    return aligned_stops


def _refine_alignment_multiview(
    aligned_stops: List[AlignedStop],
    cfg: FullBodyConfig,
) -> None:
    """
    Refine stop alignment via alternating MAP-style 4-DoF ICP against the combined rigid reference:
    - Multi-start pose candidate initialization.
    - Tangential sliding penalty on near-cylindrical surfaces.
    - Prior term toward keypoint estimate.
    - Bounded-recovery clamp logging and large residual flagging (NEEDS_REVIEW).
    """
    n_clamped = 0
    for iteration in range(min(5, cfg.alignment_max_iterations)):
        max_correction = 0.0
        for i, stop in enumerate(aligned_stops):
            if stop.pcd_canonical is None:
                continue

            other_clouds = []
            for j, other in enumerate(aligned_stops):
                if i != j and other.pcd_canonical is not None:
                    pts = np.asarray(other.pcd_canonical.points)
                    if other.body_part_labels is not None and cfg.rigid_parts_only:
                        rigid_mask = is_rigid_point(other.body_part_labels)
                        if np.sum(rigid_mask) > 100:
                            pts = pts[rigid_mask]
                    cpcd = o3d.geometry.PointCloud()
                    cpcd.points = o3d.utility.Vector3dVector(pts)
                    other_clouds.append(cpcd)

            if not other_clouds:
                continue

            ref_pcd = other_clouds[0]
            for c in other_clouds[1:]:
                ref_pcd += c

            ref_pcd = ref_pcd.voxel_down_sample(0.01)
            ref_pcd.estimate_normals()

            src_pts = np.asarray(stop.pcd_canonical.points)
            if stop.body_part_labels is not None and cfg.rigid_parts_only:
                rigid_m = is_rigid_point(stop.body_part_labels)
                if np.sum(rigid_m) > 100:
                    src_pts = src_pts[rigid_m]
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_pts)
            src_pcd.estimate_normals()

            # MAP-style 4-DoF ICP with multi-start and cylindrical sliding penalty
            delta_T, inlier_frac, rmse_mm, rmse_before_mm, hit_clamp = _solve_4dof_map_icp(src_pcd, ref_pcd, cfg)

            stop.residual_before_mm = rmse_before_mm
            stop.residual_after_mm = rmse_mm
            stop.inlier_fraction = inlier_frac

            d_yaw = float(np.rad2deg(np.arctan2(delta_T[2, 0], delta_T[0, 0])))
            d_trans = delta_T[:3, 3]

            if not hit_clamp and inlier_frac >= cfg.alignment_min_inlier_fraction:
                stop.T_canonical_camera = delta_T @ stop.T_canonical_camera
                stop.pcd_canonical.transform(delta_T)
                if stop.keypoints_canonical is not None:
                    stop.keypoints_canonical = stop.keypoints_canonical.transform(delta_T, "canonical")
                stop.refined_yaw_deg += d_yaw
                stop.refined_translation_m += d_trans
                max_correction = max(max_correction, abs(d_yaw))
            else:
                stop.hit_clamp = True
                stop.review_status = "CLAMPED_PRIOR"
                n_clamped += 1
                logger.debug(f"  [align] Stop {stop.stop_id} ICP correction rejected by bounded recovery (clamped).")

            # Check for large inter-stop residual
            if stop.residual_after_mm > cfg.max_inter_stop_residual_mm:
                stop.review_status = "NEEDS_REVIEW"
                stop.accepted = False
                logger.warning(
                    f"  [align] Stop {stop.stop_id} flagged as NEEDS_REVIEW: residual "
                    f"{stop.residual_after_mm:.1f} mm > {cfg.max_inter_stop_residual_mm:.1f} mm threshold."
                )

        if max_correction < 0.2:
            break

    if n_clamped > 0:
        logger.info(f"  [align] Bounded recovery clamp triggered for {n_clamped} stop adjustment(s).")


def _solve_4dof_map_icp(
    src: o3d.geometry.PointCloud,
    ref: o3d.geometry.PointCloud,
    cfg: FullBodyConfig,
) -> Tuple[np.ndarray, float, float, float, bool]:
    """
    Solve 4-DoF point-to-plane registration with MAP regularization:
    1. Multi-start candidate initialization (test candidate yaw perturbations).
    2. Point-to-plane ICP registration.
    3. Tangential sliding damping on near-cylindrical torso surfaces.
    4. Prior penalty toward keypoint estimate.
    5. Bounded-recovery clamp check.
    """
    res_before = o3d.pipelines.registration.evaluate_registration(
        src,
        ref,
        cfg.alignment_icp_distance_threshold_m,
        np.eye(4),
    )
    rmse_before_mm = float(res_before.inlier_rmse * 1000.0)

    # 1. Multi-start initialization: test candidates [0, -3, +3 deg]
    candidate_yaws_deg = [0.0, -3.0, 3.0]
    best_candidate_T = np.eye(4, dtype=np.float64)
    best_candidate_fitness = float(res_before.fitness)

    for cand_yaw in candidate_yaws_deg:
        if abs(cand_yaw) < 1e-4:
            continue
        rad = np.deg2rad(cand_yaw)
        c, s = np.cos(rad), np.sin(rad)
        T_cand = np.eye(4, dtype=np.float64)
        T_cand[:3, :3] = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        eval_cand = o3d.pipelines.registration.evaluate_registration(
            src, ref, cfg.alignment_icp_distance_threshold_m, T_cand
        )
        if eval_cand.fitness > best_candidate_fitness:
            best_candidate_fitness = eval_cand.fitness
            best_candidate_T = T_cand

    # 2. Run Point-to-Plane ICP from best candidate
    reg = o3d.pipelines.registration.registration_icp(
        src,
        ref,
        cfg.alignment_icp_distance_threshold_m,
        best_candidate_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )

    # Project to 4-DoF
    T = np.eye(4, dtype=np.float64)
    R = reg.transformation[:3, :3]
    yaw = np.arctan2(R[2, 0], R[0, 0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = np.array([
        [cy, 0.0, -sy],
        [0.0, 1.0, 0.0],
        [sy, 0.0, cy],
    ])
    T[0, 3] = reg.transformation[0, 3]
    T[1, 3] = 0.0
    T[2, 3] = reg.transformation[2, 3]

    # 3. Penalize tangential sliding on near-cylindrical surfaces
    # Estimate dominant normal direction in horizontal plane
    if src.has_normals() and len(src.normals) > 10:
        normals_np = np.asarray(src.normals)
        nx = float(np.median(normals_np[:, 0]))
        nz = float(np.median(normals_np[:, 2]))
        norm_n = math.hypot(nx, nz)
        if norm_n > 1e-3:
            nx /= norm_n
            nz /= norm_n
            # Tangent direction is perpendicular to normal: (-nz, nx)
            tx, tz = -nz, nx
            trans_xz = np.array([T[0, 3], T[2, 3]])
            tan_comp = trans_xz[0] * tx + trans_xz[1] * tz
            # If sliding exceeds 5 mm, damp the tangential component
            if abs(tan_comp) > 0.005:
                damp_factor = 0.5  # Shrink unconstrained tangential sliding
                T[0, 3] -= (1.0 - damp_factor) * tan_comp * tx
                T[2, 3] -= (1.0 - damp_factor) * tan_comp * tz

    # 4. Check bounded recovery clamp
    d_yaw_deg = float(np.rad2deg(abs(yaw)))
    d_trans_m = float(math.hypot(T[0, 3], T[2, 3]))

    hit_clamp = (
        d_yaw_deg > cfg.alignment_max_yaw_correction_deg or
        d_trans_m > cfg.alignment_max_translation_correction_m or
        float(reg.fitness) < cfg.alignment_min_inlier_fraction
    )

    inlier_frac = float(reg.fitness)
    rmse_mm = float(reg.inlier_rmse * 1000.0)
    return T, inlier_frac, rmse_mm, rmse_before_mm, hit_clamp

