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


def compute_keypoint_prior_yaw(
    keypoints_grav: Keypoints3D,
    nominal_deg: float,
    cfg: FullBodyConfig,
) -> Tuple[float, np.ndarray, bool]:
    """
    Compute initial yaw and body axis center from shoulders and hips in gravity frame.

    Returns
    -------
    Tuple[float, np.ndarray, bool]:
        (prior_yaw_deg, body_axis_center_3d, is_valid)
    """
    l_sh, c_ls, _ = keypoints_grav.get("left_shoulder")
    r_sh, c_rs, _ = keypoints_grav.get("right_shoulder")
    l_hp, c_lh, _ = keypoints_grav.get("left_hip")
    r_hp, c_rh, _ = keypoints_grav.get("right_hip")
    nose, c_n, _ = keypoints_grav.get("nose")

    # Fallback to nominal if keypoints absent
    if (l_sh is None or r_sh is None or c_ls < 0.2 or c_rs < 0.2) and \
       (l_hp is None or r_hp is None or c_lh < 0.2 or c_rh < 0.2):
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    # Robust anatomical extraction of lateral orientation and body axis center
    c_sh_cand, v_sh_cand = None, None
    c_hp_cand, v_hp_cand = None, None

    if l_sh is not None and r_sh is not None and c_ls >= 0.2 and c_rs >= 0.2:
        sh_span = float(np.linalg.norm(l_sh - r_sh))
        if 0.22 <= sh_span <= 0.50:
            v_sh_cand = np.array([l_sh[0] - r_sh[0], 0.0, l_sh[2] - r_sh[2]])
            c_sh_cand = 0.5 * (l_sh + r_sh)

    if l_hp is not None and r_hp is not None and c_lh >= 0.2 and c_rh >= 0.2:
        hp_span = float(np.linalg.norm(l_hp - r_hp))
        if 0.12 <= hp_span <= 0.38:
            v_hp_cand = np.array([l_hp[0] - r_hp[0], 0.0, l_hp[2] - r_hp[2]])
            c_hp_cand = 0.5 * (l_hp + r_hp)

    lat_vecs = []
    centers = []

    if c_sh_cand is not None and c_hp_cand is not None:
        # Cross-check horizontal spine alignment (shoulders and hips should align vertically)
        spine_disp_xz = float(np.linalg.norm([c_sh_cand[0] - c_hp_cand[0], c_sh_cand[2] - c_hp_cand[2]]))
        if spine_disp_xz > 0.05:
            # One pair has an outlier: compare spans against typical proportions (sh~0.36m, hp~0.20m)
            sh_residual = abs(float(np.linalg.norm(l_sh - r_sh)) - 0.36)
            hp_residual = abs(float(np.linalg.norm(l_hp - r_hp)) - 0.20)
            if sh_residual < hp_residual:
                lat_vecs.append(v_sh_cand)
                centers.append(c_sh_cand)
            else:
                lat_vecs.append(v_hp_cand)
                centers.append(c_hp_cand)
        else:
            lat_vecs.extend([v_sh_cand, v_hp_cand])
            centers.extend([c_sh_cand, c_hp_cand])
    elif c_sh_cand is not None:
        lat_vecs.append(v_sh_cand)
        centers.append(c_sh_cand)
    elif c_hp_cand is not None:
        lat_vecs.append(v_hp_cand)
        centers.append(c_hp_cand)
    else:
        # Ultimate fallback
        if l_sh is not None and r_sh is not None:
            lat_vecs.append(np.array([l_sh[0] - r_sh[0], 0.0, l_sh[2] - r_sh[2]]))
            centers.append(0.5 * (l_sh + r_sh))
        elif l_hp is not None and r_hp is not None:
            lat_vecs.append(np.array([l_hp[0] - r_hp[0], 0.0, l_hp[2] - r_hp[2]]))
            centers.append(0.5 * (l_hp + r_hp))
        else:
            return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    lat = np.mean(lat_vecs, axis=0)
    lat[1] = 0.0
    norm_lat = np.linalg.norm(lat)
    if norm_lat < 1e-4:
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False
    lat /= norm_lat

    # Anterior vector: cross(lateral, UP) where UP is [0, 1, 0]
    # cross([lx, 0, lz], [0, 1, 0]) = [-lz, 0, lx]
    ant = np.array([-lat[2], 0.0, lat[0]], dtype=np.float64)

    # Disambiguate front vs back using nose position relative to shoulders
    center = np.mean(centers, axis=0)
    if nose is not None and c_n > 0.2:
        nose_dir = nose - center
        nose_dir[1] = 0.0
        if np.dot(ant, nose_dir) < 0:
            ant = -ant
            lat = -lat

    # Compute rotation around +Y that aligns (lat -> [1, 0, 0], ant -> [0, 0, 1])
    # Angle in radians
    yaw_rad = np.arctan2(ant[0], ant[2])
    yaw_deg = float(np.rad2deg(yaw_rad))

    return yaw_deg, center, True


def align_stops(
    stops: List[StopData],
    consensus_floor: FloorPlane,
    cfg: FullBodyConfig,
) -> List[AlignedStop]:
    """
    Execute full multi-stop alignment pipeline.
    """
    n_stops = len(stops)
    if n_stops == 0:
        return []

    T_grav_cam = compute_gravity_transform(consensus_floor)

    aligned_stops: List[AlignedStop] = []

    # 1. Compute initial priors per stop
    for s in stops:
        kpts_cam = s.keypoints_camera
        if kpts_cam is not None:
            kpts_grav = kpts_cam.transform(T_grav_cam, "gravity")
        else:
            kpts_grav = Keypoints3D([], np.zeros((0, 3)), np.zeros(0), np.zeros(0), "gravity")

        if cfg.turn_direction == "cw":
            nominal_canonical = (180.0 - s.nominal_angle_deg + 360.0) % 360.0
        else:
            nominal_canonical = (180.0 + s.nominal_angle_deg + 360.0) % 360.0
        prior_yaw, center, is_kpt_valid = compute_keypoint_prior_yaw(kpts_grav, nominal_canonical, cfg)

        # Check deviation against nominal canonical yaw
        diff = (prior_yaw - nominal_canonical + 180.0) % 360.0 - 180.0
        if abs(diff) > cfg.max_angle_deviation_deg:
            logger.info(
                f"  [align] Stop {s.stop_id} yaw prior {prior_yaw:.1f} deg deviates by {diff:.1f} deg "
                f"from nominal {nominal_canonical:.1f} deg; clamping prior to nominal."
            )
            prior_yaw = nominal_canonical


        # Canonical transform: rotates yaw, centers X-Z body axis, sets floor at Y=0
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

        # Transform point cloud to canonical frame
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
            )
        )

    # 2. Refinement loop (4-DoF point-to-plane ICP on rigid parts)
    if n_stops > 1:
        _refine_alignment_multiview(aligned_stops, cfg)

    # 3. Enforce loop consistency
    if n_stops >= 4:
        _enforce_loop_consistency(aligned_stops)

    return aligned_stops


def _refine_alignment_multiview(
    aligned_stops: List[AlignedStop],
    cfg: FullBodyConfig,
) -> None:
    """
    Refine stop alignment via alternating 4-DoF ICP against the combined rigid reference.
    """
    for iteration in range(min(5, cfg.alignment_max_iterations)):
        max_correction = 0.0
        for i, stop in enumerate(aligned_stops):
            if stop.pcd_canonical is None:
                continue

            # Build reference point cloud from all OTHER stops
            other_clouds = []
            for j, other in enumerate(aligned_stops):
                if i != j and other.pcd_canonical is not None:
                    # Filter only rigid parts
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

            # Source rigid points
            src_pts = np.asarray(stop.pcd_canonical.points)
            if stop.body_part_labels is not None and cfg.rigid_parts_only:
                rigid_m = is_rigid_point(stop.body_part_labels)
                if np.sum(rigid_m) > 100:
                    src_pts = src_pts[rigid_m]
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_pts)
            src_pcd.estimate_normals()

            # Run 4-DoF planar ICP
            delta_T, inlier_frac, rmse_mm, rmse_before_mm = _solve_4dof_icp(src_pcd, ref_pcd, cfg)

            stop.residual_before_mm = rmse_before_mm
            stop.residual_after_mm = rmse_mm
            stop.inlier_fraction = inlier_frac

            # Extract yaw change and planar translation
            d_yaw = float(np.rad2deg(np.arctan2(delta_T[2, 0], delta_T[0, 0])))
            d_trans = delta_T[:3, 3]

            # Bounded recovery check
            if abs(d_yaw) <= cfg.alignment_max_yaw_correction_deg and \
               np.linalg.norm(d_trans) <= cfg.alignment_max_translation_correction_m and \
               inlier_frac >= cfg.alignment_min_inlier_fraction:
                stop.T_canonical_camera = delta_T @ stop.T_canonical_camera
                stop.pcd_canonical.transform(delta_T)
                if stop.keypoints_canonical is not None:
                    stop.keypoints_canonical = stop.keypoints_canonical.transform(delta_T, "canonical")
                stop.refined_yaw_deg += d_yaw
                stop.refined_translation_m += d_trans
                max_correction = max(max_correction, abs(d_yaw))
            else:
                logger.debug(f"  [align] Stop {stop.stop_id} ICP correction rejected by bounded recovery.")

        if max_correction < 0.2:
            break


def _solve_4dof_icp(
    src: o3d.geometry.PointCloud,
    ref: o3d.geometry.PointCloud,
    cfg: FullBodyConfig,
) -> Tuple[np.ndarray, float, float, float]:
    """Solve 4-DoF point-to-plane registration restricted to yaw + XZ translation."""
    res_before = o3d.pipelines.registration.evaluate_registration(
        src,
        ref,
        cfg.alignment_icp_distance_threshold_m,
        np.eye(4),
    )
    rmse_before_mm = float(res_before.inlier_rmse * 1000.0)

    reg = o3d.pipelines.registration.registration_icp(
        src,
        ref,
        cfg.alignment_icp_distance_threshold_m,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )
    # Project 6-DoF to 4-DoF (constrain rotation to Y-axis, translation to XZ plane)
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
    T[1, 3] = 0.0  # Height offset is zero (locked to floor)
    T[2, 3] = reg.transformation[2, 3]

    inlier_frac = float(reg.fitness)
    rmse_mm = float(reg.inlier_rmse * 1000.0)
    return T, inlier_frac, rmse_mm, rmse_before_mm


def _enforce_loop_consistency(stops: List[AlignedStop]) -> None:
    """Enforce that sum of relative yaws equals nominal 360 deg."""
    n = len(stops)
    if n < 4:
        return
    yaws = np.array([s.refined_yaw_deg for s in stops])
    total_span = (yaws[-1] - yaws[0] + 360.0) % 360.0
    expected_span = 360.0 - (360.0 / n)
    drift = total_span - expected_span
    if abs(drift) < 20.0:
        # Distribute drift linearly across stops
        for i, s in enumerate(stops):
            adj = -drift * (i / float(n))
            s.refined_yaw_deg += adj

