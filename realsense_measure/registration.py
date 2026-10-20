"""
Multi-view point-cloud registration for hand-repositioned object captures.

Context
-------
Frames are captured from a single FIXED camera while the operator rotates or
repositions the object by hand between shots.  There is no known relative
transform between frames, and inter-frame rotations can be large and
arbitrary -- so plain ICP with an identity initial guess will fail.

Two-stage registration is used for every frame pair:

  1. **Global registration** (FPFH + RANSAC) -- produces a coarse but
     rotation-agnostic initial alignment with no initial guess required.
  2. **Point-to-plane ICP** -- refines the coarse alignment to sub-millimetre
     accuracy using the RANSAC result as its starting pose.

Growing-reference design
------------------------
The *most important* design choice in this file is that frames are NOT all
registered independently back to frame 0.  Instead they are accumulated onto
a **growing reference cloud**:

  reference <- frame[0]
  for i in 1 ... N:
      transform frame[i] onto reference
      append transformed frame[i] to reference
      voxel-downsample reference  (keep it manageable)

Why this matters: frame 0 is a single partial view that covers at most ~180 deg
of the object surface.  Registering every subsequent frame against that one
partial view means later frames have very little overlap to match against,
especially if they show a previously-unseen side.  By growing the reference
cloud, each new frame is aligned against all surface area accumulated so far,
giving RANSAC and ICP far more inlier correspondences and dramatically
improving robustness for objects where each capture adds new geometry.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from open3d.pipelines.odometry import (
    OdometryOption,
    RGBDOdometryJacobianFromColorTerm,
    RGBDOdometryJacobianFromHybridTerm,
    compute_rgbd_odometry,
)
from open3d.pipelines.registration import (
    CorrespondenceCheckerBasedOnDistance,
    CorrespondenceCheckerBasedOnEdgeLength,
    GlobalOptimizationConvergenceCriteria,
    GlobalOptimizationLevenbergMarquardt,
    GlobalOptimizationOption,
    ICPConvergenceCriteria,
    PoseGraph,
    PoseGraphEdge,
    PoseGraphNode,
    RANSACConvergenceCriteria,
    TransformationEstimationForColoredICP,
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
    get_information_matrix_from_point_clouds,
    global_optimization,
    registration_colored_icp,
    registration_icp,
    registration_ransac_based_on_feature_matching,
)
from camera.frame import RGBDFrame
from config import RegistrationConfig


def rigid_transform_from_correspondences(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Weighted Kabsch algorithm: finds the rigid transform (R, t) that best maps
    src_pts onto dst_pts in a weighted least-squares sense.

    Needs at least 3 non-collinear corresponding points to be well-defined;
    works best with 4 or more (see LANDMARK_INDICES in landmarks.py, which
    provides up to 8/13).

    Parameters
    ----------
    src_pts:
        (N, 3) array of source points.
    dst_pts:
        (N, 3) array of destination points.
    weights:
        Optional (N,) array of non-negative quality weights per correspondence.

    Returns
    -------
    T:
        4x4 homogeneous transform mapping src_pts onto dst_pts.
    residual_mm:
        Weighted mean per-point distance (in mm) between dst_pts and src_pts after
        applying T. Use this as a quality gate — a high residual means the
        landmark correspondences themselves were unreliable.
    """
    n_pts = len(src_pts)
    if weights is None:
        w = np.ones(n_pts, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).flatten()
        if w.size != n_pts or float(np.sum(w)) <= 1e-9:
            w = np.ones(n_pts, dtype=np.float64)

    W = float(np.sum(w))
    src_c = np.sum(src_pts * w[:, None], axis=0) / W
    dst_c = np.sum(dst_pts * w[:, None], axis=0) / W

    src_centered = src_pts - src_c
    dst_centered = dst_pts - dst_c

    H = (src_centered * w[:, None]).T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    recovered = (R @ src_pts.T).T + t
    dists = np.linalg.norm(recovered - dst_pts, axis=1)
    residual_mm = float(np.sum(dists * w) / W * 1000.0)
    return T, residual_mm


def evaluate_point_set_geometry(
    pts: np.ndarray,
    weights: np.ndarray | None = None,
    label: str = "points",
    min_spread_m: float = 0.025,
    min_secondary_spread_m: float = 0.005,
    max_collinear_ratio: float = 10.0,
    min_unique_dist_m: float = 0.005,
    min_unique_count: int = 4,
) -> tuple[bool, str, dict[str, float]]:
    """
    Evaluate the geometric conditioning of a single 3D landmark point set.

    Assesses:
    1. Count and coordinate finiteness.
    2. Number of spatially distinct landmarks (clustering closer than min_unique_dist_m).
    3. Maximum geometric spread (diameter / max pairwise distance).
    4. Centered SVD singular values for 2D spread and collinearity.
       - Planar configurations (s3 ~ 0, s2 >= min_secondary_spread_m) are ACCEPTED.
       - Collinear configurations (s2 < min_secondary_spread_m or s1/s2 > max_collinear_ratio)
         are REJECTED.

    All spatial dimensions and thresholds are in METRES (m).

    Parameters
    ----------
    pts:
        (N, 3) float array of 3D coordinates in camera frame (metres).
    weights:
        Optional (N,) non-negative quality weights. If provided, SVD uses weighted
        centered coordinates to assess the effective geometry seen by weighted Kabsch.
    label:
        Descriptive string for diagnostic messages (e.g. "source" or "destination").
    min_spread_m:
        Minimum allowable maximum pairwise distance across all points (metres). Default 25 mm.
    min_secondary_spread_m:
        Minimum allowable RMS spread along the 2nd principal axis (metres). Default 5 mm.
    max_collinear_ratio:
        Maximum allowable ratio of 1st to 2nd singular value (sigma_1 / sigma_2). Default 10.0.
    min_unique_dist_m:
        Minimum separation distance to consider two points distinct (metres). Default 5 mm.
    min_unique_count:
        Minimum number of distinct spatial clusters required. Default 4.

    Returns
    -------
    (is_valid, reason, metrics)
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
        return False, f"{label} shape invalid: {pts_arr.shape}", {}

    n_pts = pts_arr.shape[0]
    if n_pts < min_unique_count:
        return False, f"{label} point count {n_pts} < {min_unique_count}", {}

    if not np.all(np.isfinite(pts_arr)):
        return False, f"{label} contains non-finite coordinates", {}

    # Check for repeated / nearly coincident points using vectorized pairwise distances
    dists = cdist(pts_arr, pts_arr)
    kept = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        if not np.any(kept & (dists[i] < min_unique_dist_m)):
            kept[i] = True

    n_unique = int(np.count_nonzero(kept))
    if n_unique < min_unique_count:
        return (
            False,
            f"{label} insufficient unique landmarks: {n_unique} < {min_unique_count} distinct clusters",
            {"n_unique": float(n_unique)},
        )

    # Maximum pairwise distance (spread / diameter)
    max_dist = float(np.max(dists))
    if max_dist < min_spread_m:
        return (
            False,
            f"{label} spread too small: {max_dist * 1000.0:.1f}mm < {min_spread_m * 1000.0:.1f}mm",
            {"max_spread_m": max_dist, "n_unique": float(n_unique)},
        )

    # Centered SVD analysis (weighted if weights provided)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).flatten()
        if w.size != n_pts or float(np.sum(w)) <= 1e-9:
            w = np.ones(n_pts, dtype=np.float64)
    else:
        w = np.ones(n_pts, dtype=np.float64)

    W = float(np.sum(w))
    w_norm = w / W
    centroid = np.sum(pts_arr * w_norm[:, None], axis=0)
    centered = pts_arr - centroid
    X_w = np.sqrt(w_norm[:, None]) * centered
    _, S, _ = np.linalg.svd(X_w)

    s1 = float(S[0])
    s2 = float(S[1])
    s3 = float(S[2])
    collinear_ratio = (s1 / s2) if s2 > 1e-9 else float("inf")

    metrics = {
        "max_spread_m": max_dist,
        "n_unique": float(n_unique),
        "s1_m": s1,
        "s2_m": s2,
        "s3_m": s3,
        "collinear_ratio": collinear_ratio,
    }

    # Secondary spread gate (sufficient 2D spread to constrain rotation)
    if s2 < min_secondary_spread_m:
        return (
            False,
            f"{label} secondary spread too small: {s2 * 1000.0:.1f}mm < {min_secondary_spread_m * 1000.0:.1f}mm (nearly collinear)",
            metrics,
        )

    # Collinearity ratio gate (elongation along 1D line)
    if collinear_ratio > max_collinear_ratio:
        return (
            False,
            f"{label} collinearity ratio too high: {collinear_ratio:.1f} > {max_collinear_ratio:.1f}",
            metrics,
        )

    return True, f"{label} conditioning passed", metrics


def check_landmark_geometric_conditioning(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    weights: np.ndarray | None = None,
    cfg: RegistrationConfig | None = None,
    *,
    min_spread_m: float | None = None,
    min_secondary_spread_m: float | None = None,
    max_collinear_ratio: float | None = None,
    min_unique_dist_m: float | None = None,
    min_unique_count: int = 4,
) -> tuple[bool, str]:
    """
    Validate the geometric conditioning of matched 3D landmark correspondences
    before calling landmark-guided Kabsch registration.

    Assesses both source and destination correspondences for:
    1. Sufficient geometric spread (diameter).
    2. Collinear or nearly collinear configurations.
    3. Repeated or coincident points that make the nominal count misleading.
    4. Sufficient 2D constraint for stable 3D rigid transform estimation.

    Planar non-collinear configurations are accepted (rotation about plane normal
    and in-plane axes are well-constrained). Collinear configurations are rejected.

    Parameters
    ----------
    src_pts:
        (N, 3) array of source landmark 3D positions (metres).
    dst_pts:
        (N, 3) array of destination landmark 3D positions (metres).
    weights:
        Optional (N,) array of correspondence weights.
    cfg:
        Optional RegistrationConfig providing threshold defaults.

    Returns
    -------
    (is_valid, reason):
        True if correspondence geometry is well-conditioned for Kabsch.
        False with a human-readable reason if Kabsch should be bypassed.
    """
    spread_thresh = min_spread_m if min_spread_m is not None else getattr(cfg, "landmark_min_spread_m", 0.025)
    s2_thresh = min_secondary_spread_m if min_secondary_spread_m is not None else getattr(cfg, "landmark_min_secondary_spread_m", 0.005)
    ratio_thresh = max_collinear_ratio if max_collinear_ratio is not None else getattr(cfg, "landmark_max_collinear_ratio", 10.0)
    uniq_dist = min_unique_dist_m if min_unique_dist_m is not None else getattr(cfg, "landmark_min_unique_dist_m", 0.005)

    # 1. Validate source point set
    src_ok, src_reason, _ = evaluate_point_set_geometry(
        src_pts,
        weights=weights,
        label="source",
        min_spread_m=spread_thresh,
        min_secondary_spread_m=s2_thresh,
        max_collinear_ratio=ratio_thresh,
        min_unique_dist_m=uniq_dist,
        min_unique_count=min_unique_count,
    )
    if not src_ok:
        return False, f"Source geometry rejected: {src_reason}"

    # 2. Validate destination point set
    dst_ok, dst_reason, _ = evaluate_point_set_geometry(
        dst_pts,
        weights=weights,
        label="destination",
        min_spread_m=spread_thresh,
        min_secondary_spread_m=s2_thresh,
        max_collinear_ratio=ratio_thresh,
        min_unique_dist_m=uniq_dist,
        min_unique_count=min_unique_count,
    )
    if not dst_ok:
        return False, f"Destination geometry rejected: {dst_reason}"

    return True, "Landmark correspondence geometry is well-conditioned"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    cfg: RegistrationConfig,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Downsample, estimate normals and compute FPFH features.

    Parameters
    ----------
    pcd:
        Input point cloud (any density).
    voxel_size:
        Voxel size in metres used for downsampling and as the scale
        reference for normal/feature search radii.
    cfg:
        Registration configuration with normal and feature radius multipliers.

    Returns
    -------
    down:
        Voxel-downsampled copy of ``pcd``.
    fpfh:
        FPFH feature descriptor for each point in ``down``.
    """
    down = pcd.voxel_down_sample(voxel_size)

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * cfg.fpfh_radius_normal_mult,
            max_nn=30,
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * cfg.fpfh_radius_feature_mult,
            max_nn=100,
        ),
    )

    return down, fpfh


def _global_register(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    cfg: RegistrationConfig,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    FPFH-based RANSAC global registration.

    No initial pose guess is required; RANSAC searches over the full SO(3)
    rotation space via feature correspondences.

    Parameters
    ----------
    source_down, target_down:
        Downsampled source and target clouds with normals.
    source_fpfh, target_fpfh:
        Corresponding FPFH feature descriptors.
    cfg:
        Registration configuration.
    voxel_size:
        Voxel size used during ``_prepare``, used to set the correspondence
        distance threshold.

    Returns
    -------
    open3d RegistrationResult with the coarse transformation.
    """
    distance_threshold = voxel_size * cfg.ransac_distance_mult

    # mutual_filter=False: Open3D silently falls back to non-mutual correspondences
    # when the mutual-filtered set is too small, which produces a console warning.
    # Setting mutual_filter to False directly avoids that redundant first attempt,
    # eliminates the warning, and is faster with equivalent alignment quality.
    return registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=RANSACConvergenceCriteria(
            cfg.ransac_max_iter,
            cfg.ransac_confidence,
        ),
    )


def _icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    cfg: RegistrationConfig,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Point-to-plane ICP refinement starting from ``init_transform``.

    Deep-copies both clouds so the caller's data is never mutated.
    Normals are estimated on each cloud if not already present.

    Parameters
    ----------
    source:
        Source (moving) point cloud.
    target:
        Target (fixed) point cloud.
    init_transform:
        4×4 numpy array — the coarse initial pose from global registration.
    cfg:
        Registration configuration.

    Returns
    -------
    open3d RegistrationResult with the refined transformation.
    """
    if not source.has_normals():
        src = o3d.geometry.PointCloud(source)
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_max_dist_m * 2, max_nn=30)
        )
    else:
        src = source

    if not target.has_normals():
        tgt = o3d.geometry.PointCloud(target)
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_max_dist_m * 2, max_nn=30)
        )
    else:
        tgt = target

    return registration_icp(
        src,
        tgt,
        cfg.icp_max_dist_m,
        init_transform,
        TransformationEstimationPointToPlane(),
        ICPConvergenceCriteria(max_iteration=cfg.icp_max_iterations),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_frame_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    cfg: RegistrationConfig,
) -> tuple[np.ndarray, float, float]:
    """
    Align ``source`` onto ``target`` with adaptive fast-path ICP.

    Speed vs Global Search Trade-off:
    ---------------------------------
    1. First attempts a cheap Point-to-Plane ICP starting from identity (np.eye(4)).
       In continuous / auto-capture workflows with small inter-frame camera motion,
       this converges almost instantaneously (milliseconds vs seconds).
    2. If ICP fitness >= cfg.icp_only_fitness_threshold, it is accepted immediately.
    3. If ICP fitness is insufficient, it falls back to full FPFH feature
       extraction and RANSAC global registration followed by ICP refinement.

    Parameters
    ----------
    source:
        The frame to be moved (query frame).
    target:
        The fixed reference frame.
    cfg:
        Registration configuration.

    Returns
    -------
    transform:
        4x4 numpy float64 array that maps ``source`` into ``target`` space.
    fitness:
        ICP fitness score (fraction of inlier correspondences).
    rmse:
        ICP inlier RMSE in metres.
    """
    # 1. Fast path: try ICP directly from identity
    cfg.resolve_defaults()
    fast_icp = _icp_refine(source, target, np.eye(4), cfg)
    if float(fast_icp.fitness) >= cfg.icp_only_fitness_threshold:
        return (
            fast_icp.transformation,
            float(fast_icp.fitness),
            float(fast_icp.inlier_rmse),
        )

    # 2. Slow / thorough path: full FPFH + RANSAC global registration -> ICP
    voxel_size = cfg.voxel_size_m

    source_down, source_fpfh = _prepare(source, voxel_size, cfg)
    target_down, target_fpfh = _prepare(target, voxel_size, cfg)

    global_result = _global_register(
        source_down, target_down,
        source_fpfh, target_fpfh,
        cfg, voxel_size,
    )

    icp_result = _icp_refine(source, target, global_result.transformation, cfg)

    # If fast ICP had a higher fitness than the global result, prefer it
    if float(fast_icp.fitness) > float(icp_result.fitness):
        return (
            fast_icp.transformation,
            float(fast_icp.fitness),
            float(fast_icp.inlier_rmse),
        )

    return (
        icp_result.transformation,
        float(icp_result.fitness),
        float(icp_result.inlier_rmse),
    )


def register_frame_pair_landmark_guided(
    source_pcd: o3d.geometry.PointCloud,
    source_color: np.ndarray,
    source_depth: np.ndarray,
    target_pcd: o3d.geometry.PointCloud,
    target_color: np.ndarray,
    target_depth: np.ndarray,
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic,
    cfg: RegistrationConfig,
    residual_accept_mm: float = 15.0,
) -> tuple[np.ndarray, float, float, str, float]:
    """
    Register one frame pair using facial landmarks as the initial guess for
    ICP, instead of FPFH+RANSAC global registration. Falls back to the
    existing register_frame_pair() (unchanged FPFH+RANSAC+ICP) whenever
    landmarks aren't usable for this particular pair — e.g. the back of the
    head or an extreme profile won't have a detectable face, and that's
    expected, not a bug.

    Returns
    -------
    (transform, fitness, rmse, method, residual_mm) where method is "landmark" or
    "fpfh_fallback", for logging/diagnostics.
    """
    from landmarks import detect_face_landmarks_3d

    lm_cfg = getattr(cfg, "landmark_cfg", None)
    src_lm = detect_face_landmarks_3d(source_color, source_depth, intrinsics_o3d, cfg=lm_cfg)
    dst_lm = detect_face_landmarks_3d(target_color, target_depth, intrinsics_o3d, cfg=lm_cfg)

    if src_lm is not None and dst_lm is not None:
        common = sorted(set(src_lm) & set(dst_lm))
        if len(common) >= 4:
            src_pts_list = []
            dst_pts_list = []
            weights_list = []
            max_var_thresh = getattr(cfg, "max_depth_variance_m2", 1e-5)

            for k in common:
                s_item = src_lm[k]
                d_item = dst_lm[k]
                if isinstance(s_item, dict):
                    s_pos = s_item["position"]
                    s_vr = s_item.get("valid_ratio", 1.0)
                    s_var = s_item.get("variance", 0.0)
                    s_low = s_item.get("low_confidence", False)
                else:
                    s_pos = s_item
                    s_vr = 1.0
                    s_var = 0.0
                    s_low = False

                if isinstance(d_item, dict):
                    d_pos = d_item["position"]
                    d_vr = d_item.get("valid_ratio", 1.0)
                    d_var = d_item.get("variance", 0.0)
                    d_low = d_item.get("low_confidence", False)
                else:
                    d_pos = d_item
                    d_vr = 1.0
                    d_var = 0.0
                    d_low = False

                src_pts_list.append(s_pos)
                dst_pts_list.append(d_pos)

                # Weight: min(1.0, valid_ratio_src * valid_ratio_dst) / (1.0 + variance_src + variance_dst)
                max_var_safe = max_var_thresh if (max_var_thresh is not None and np.isfinite(max_var_thresh) and max_var_thresh > 0) else 1e-5
                v_src_norm = (max(0.0, s_var) / max_var_safe) if (np.isfinite(s_var) and s_var > 0) else (1e4 if not np.isfinite(s_var) else 0.0)
                v_dst_norm = (max(0.0, d_var) / max_var_safe) if (np.isfinite(d_var) and d_var > 0) else (1e4 if not np.isfinite(d_var) else 0.0)
                base_w = float(min(1.0, max(0.0, s_vr) * max(0.0, d_vr)) / (1.0 + v_src_norm + v_dst_norm))
                w = base_w * (0.5 if (s_low or d_low) else 1.0)
                weights_list.append(w)

            src_pts = np.array(src_pts_list)
            dst_pts = np.array(dst_pts_list)
            weights = np.array(weights_list)

            # Geometric conditioning check (Problem 4)
            if getattr(cfg, "enable_landmark_conditioning_check", True):
                is_conditioned, cond_reason = check_landmark_geometric_conditioning(
                    src_pts, dst_pts, weights=weights, cfg=cfg
                )
                if not is_conditioned:
                    logger.warning(f"      [landmarks] Kabsch skipped: {cond_reason} -- falling back to FPFH")
                    transform, fitness, rmse = register_frame_pair(source_pcd, target_pcd, cfg)
                    return transform, fitness, rmse, "fpfh_fallback", float("nan")

            init_T, residual_mm = rigid_transform_from_correspondences(src_pts, dst_pts, weights=weights)
            if residual_mm <= residual_accept_mm:
                # Post-seed dense-surface sanity check (single pass evaluation)
                min_dense_fit = getattr(cfg, "min_seed_dense_fitness", 0.30)
                eval_voxel = max(getattr(cfg, "voxel_size_m", 0.003) * 2.0, 0.006)
                src_down = source_pcd.voxel_down_sample(eval_voxel) if len(source_pcd.points) > 50 else source_pcd
                tgt_down = target_pcd.voxel_down_sample(eval_voxel) if len(target_pcd.points) > 50 else target_pcd

                if not tgt_down.has_normals():
                    if tgt_down is target_pcd:
                        tgt_down = o3d.geometry.PointCloud(target_pcd)
                    tgt_down.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
                    )

                eval_res = o3d.pipelines.registration.evaluate_registration(
                    src_down,
                    tgt_down,
                    cfg.colored_icp_max_dist_m,
                    init_T,
                )
                seed_dense_fitness = float(eval_res.fitness)

                if seed_dense_fitness < min_dense_fit:
                    logger.warning(
                        f"      [landmarks] Kabsch seed rejected: dense-surface fitness "
                        f"{seed_dense_fitness:.2f} < {min_dense_fit:.2f} threshold (residual was {residual_mm:.1f}mm) "
                        f"-- falling through to Tier 2"
                    )
                    return np.eye(4), 0.0, float("inf"), "dense_sanity_failed", float(residual_mm)

                # Localized (face-region) fitness check (Task 2)
                if getattr(cfg, "enable_landmark_face_fitness_check", True):
                    face_crop_r = getattr(cfg, "face_crop_radius_m", 0.08)
                    min_face_fit = getattr(cfg, "min_landmark_face_fitness", 0.25)
                    face_corr_dist = getattr(cfg, "face_fitness_dist_m", 0.005)

                    if "nose_tip" in dst_lm:
                        d_nose = dst_lm["nose_tip"]
                        face_anchor = d_nose["position"] if isinstance(d_nose, dict) else d_nose
                    else:
                        face_anchor = np.mean(dst_pts, axis=0)

                    src_trans = o3d.geometry.PointCloud(src_down).transform(init_T)
                    src_trans_pts = np.asarray(src_trans.points)
                    tgt_down_pts = np.asarray(tgt_down.points)

                    src_face_mask = np.linalg.norm(src_trans_pts - face_anchor, axis=1) <= face_crop_r
                    tgt_face_mask = np.linalg.norm(tgt_down_pts - face_anchor, axis=1) <= face_crop_r

                    if np.count_nonzero(src_face_mask) >= 10 and np.count_nonzero(tgt_face_mask) >= 10:
                        src_face_crop = src_trans.select_by_index(np.where(src_face_mask)[0])
                        tgt_face_crop = tgt_down.select_by_index(np.where(tgt_face_mask)[0])

                        eval_face = o3d.pipelines.registration.evaluate_registration(
                            src_face_crop,
                            tgt_face_crop,
                            face_corr_dist,
                            np.eye(4),
                        )
                        face_fitness = float(eval_face.fitness)

                        if face_fitness < min_face_fit:
                            logger.warning(
                                f"      [landmarks] Kabsch seed rejected: local face-region fitness "
                                f"{face_fitness:.2f} < {min_face_fit:.2f} threshold (residual was {residual_mm:.1f}mm) "
                                f"-- falling through to Tier 2"
                            )
                            return np.eye(4), 0.0, float("inf"), "face_sanity_failed", float(residual_mm)

                # Independent geometric FPFH+RANSAC cross-check (Task 1)
                if getattr(cfg, "enable_landmark_fpfh_crosscheck", True):
                    max_trans_disagree = getattr(cfg, "max_landmark_fpfh_disagreement_trans_m", 0.02)
                    max_rot_disagree = getattr(cfg, "max_landmark_fpfh_disagreement_rot_deg", 10.0)

                    try:
                        voxel_fpfh = getattr(cfg, "voxel_size_m", 0.003)
                        s_down_fpfh, s_fpfh = _prepare(source_pcd, voxel_fpfh, cfg)
                        t_down_fpfh, t_fpfh = _prepare(target_pcd, voxel_fpfh, cfg)
                        fpfh_res = _global_register(s_down_fpfh, t_down_fpfh, s_fpfh, t_fpfh, cfg, voxel_fpfh)

                        fpfh_refined = _icp_refine(s_down_fpfh, t_down_fpfh, fpfh_res.transformation, cfg)
                        T_fpfh = fpfh_refined.transformation if fpfh_refined.fitness >= 0.15 else fpfh_res.transformation
                        fpfh_quality = max(float(fpfh_res.fitness), float(fpfh_refined.fitness))

                        if fpfh_quality >= 0.15:
                            trans_disagree = float(np.linalg.norm(init_T[:3, 3] - T_fpfh[:3, 3]))
                            rot_disagree = float(compute_rotation_deg(init_T[:3, :3] @ T_fpfh[:3, :3].T))

                            if trans_disagree > max_trans_disagree or rot_disagree > max_rot_disagree:
                                logger.warning(
                                    f"      [landmarks] Kabsch disagrees with independent FPFH check "
                                    f"(trans {trans_disagree:.3f}m, rot {rot_disagree:.1f}deg) "
                                    f"-- falling through to Tier 2"
                                )
                                return np.eye(4), 0.0, float("inf"), "fpfh_crosscheck_failed", float(residual_mm)
                    except Exception as e:
                        logger.warning(
                            f"      [fusion-fallback] FPFH cross-check failed with "
                            f"{type(e).__name__}: {e}"
                        )

                # Tightened Colored ICP refinement specifically for landmark seed
                if not source_pcd.has_normals():
                    src_temp = o3d.geometry.PointCloud(source_pcd)
                    src_temp.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
                    )
                else:
                    src_temp = source_pcd

                if not target_pcd.has_normals():
                    tgt_temp = o3d.geometry.PointCloud(target_pcd)
                    tgt_temp.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
                    )
                else:
                    tgt_temp = target_pcd
                try:
                    colored_criteria = ICPConvergenceCriteria(
                        max_iteration=50,
                        relative_fitness=1e-7,
                        relative_rmse=1e-7,
                    )
                    icp_result = registration_colored_icp(
                        src_temp,
                        tgt_temp,
                        cfg.colored_icp_max_dist_m,
                        init_T,
                        TransformationEstimationForColoredICP(lambda_geometric=cfg.colored_icp_lambda_geom),
                        colored_criteria,
                    )
                except Exception as e:
                    logger.warning(
                        f"      [fusion-fallback] Colored ICP refinement for landmark seed failed with "
                        f"{type(e).__name__}: {e} -- falling back to point-to-plane ICP"
                    )
                    icp_result = _icp_refine(source_pcd, target_pcd, init_T, cfg)
                return (
                    icp_result.transformation,
                    float(icp_result.fitness),
                    float(icp_result.inlier_rmse),
                    "landmark",
                    float(residual_mm),
                )
            else:
                logger.debug(f"      [landmarks] Kabsch residual too high: {residual_mm:.1f}mm > {residual_accept_mm:.1f}mm")
        else:
            logger.debug(f"      [landmarks] Insufficient common landmarks: {len(common)} < 4 (src={len(src_lm)}, dst={len(dst_lm)})")
    else:
        src_n = len(src_lm) if src_lm is not None else 0
        dst_n = len(dst_lm) if dst_lm is not None else 0
        logger.debug(f"      [landmarks] Face not detected on both frames (src={src_n}, dst={dst_n})")

    transform, fitness, rmse = register_frame_pair(source_pcd, target_pcd, cfg)
    return transform, fitness, rmse, "fpfh_fallback", float("nan")


def register_against_candidates(
    new_frame: o3d.geometry.PointCloud,
    candidates: list[o3d.geometry.PointCloud | None],
    cfg: RegistrationConfig,
) -> tuple[np.ndarray, float, float] | None:
    """
    Tries registering `new_frame` against each candidate reference cloud
    in `candidates` (in order), returning the (transform, fitness, rmse)
    from whichever candidate produced the BEST fitness. This lets a frame
    succeed by matching either the immediately preceding frame (best when
    consecutive frames have small motion between them, e.g. from
    auto-capture) OR the full accumulated reference (best when a frame
    doesn't overlap much with just the last frame but does overlap with
    earlier accumulated geometry) -- whichever actually works for that
    particular frame.
    """
    if new_frame is None or len(new_frame.points) < 10:
        return None

    best: tuple[np.ndarray, float, float] | None = None
    for idx, candidate in enumerate(candidates):
        if candidate is None or len(candidate.points) < 10:
            continue
        try:
            transform, fitness, rmse = register_frame_pair(new_frame, candidate, cfg)
        except Exception as e:
            logger.warning(
                f"  [fusion-fallback] Candidate {idx} registration failed with "
                f"{type(e).__name__}: {e}"
            )
            continue
        if best is None or fitness > best[1]:
            best = (transform, fitness, rmse)
    return best


def register_sequence(
    frames: list[o3d.geometry.PointCloud],
    cfg: RegistrationConfig,
) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]]]:
    """
    Register a sequence of per-frame object clouds into one coordinate system.

    Uses a multi-candidate growing reference strategy with acceptance gating:
    - Frame 0 becomes the initial reference and last aligned frame.
    - Each subsequent frame is matched against [last_aligned_frame, reference]
    - and picks the candidate alignment with the highest fitness.
    - If fitness >= cfg.min_accept_fitness, the frame is marked as accepted,
      updates last_aligned_frame, and is merged into reference.
    - If fitness < cfg.min_accept_fitness, the frame is rejected from the
      reference to prevent poisoning subsequent alignments.

    Parameters
    ----------
    frames:
        Ordered list of preprocessed, per-frame object clouds.
    cfg:
        Registration configuration.

    Returns
    -------
    aligned:
        List of point clouds, all expressed in frame-0 coordinates.
    diagnostics:
        List of ``(fitness, rmse, accepted)`` tuples, one per frame.
        Frame 0 always gets ``(1.0, 0.0, True)``.
    """
    cfg.resolve_defaults()
    if not frames:
        return [], []

    aligned: list[o3d.geometry.PointCloud] = [copy.deepcopy(frames[0])]
    reference = copy.deepcopy(frames[0])
    last_aligned_frame = copy.deepcopy(frames[0])
    diagnostics: list[tuple[float, float, bool]] = [(1.0, 0.0, True)]

    for i in range(1, len(frames)):
        candidates = [last_aligned_frame, reference]
        result = register_against_candidates(frames[i], candidates, cfg)
        if result is None:
            aligned.append(copy.deepcopy(frames[i]))
            diagnostics.append((0.0, float("inf"), False))
            continue

        transform, fitness, rmse = result
        moved = copy.deepcopy(frames[i]).transform(transform)
        aligned.append(moved)

        accepted = bool(fitness >= cfg.min_accept_fitness)
        diagnostics.append((fitness, rmse, accepted))

        if accepted:
            last_aligned_frame = moved
            reference = reference + moved
            reference = reference.voxel_down_sample(cfg.voxel_size_m)

    return aligned, diagnostics


def get_information_matrix(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    cfg: RegistrationConfig,
) -> np.ndarray:
    """
    Compute 6x6 information matrix for a registered frame pair.

    Produces the uncertainty/confidence weighting each edge needs for
    pose graph optimization.
    """
    return get_information_matrix_from_point_clouds(
        source,
        target,
        cfg.icp_max_dist_m,
        transform,
    )


def build_pose_graph(
    frames: list[o3d.geometry.PointCloud],
    cfg: RegistrationConfig,
):
    """
    Build a gated pose graph.

    CONVENTION:
    ``register_frame_pair(frames[i], frames[j])`` returns T that maps frame i
    INTO frame j's coordinates. Open3D's ``PoseGraphEdge(source, target, T)``
    expects exactly that, so the edge is declared (i -> j), and node poses
    (frame -> frame-0 coordinates) compose as ``pose_i = pose_j @ T``.

    GATING: a new frame is registered against the last few *accepted* frames.
    If even the best of those registrations has fitness below
    ``cfg.min_accept_fitness`` the frame is REJECTED: it gets no node and no
    edges, so it cannot corrupt the graph.
    """
    cfg.resolve_defaults()
    n = len(frames)
    pose_graph = PoseGraph()
    pose_graph.nodes.append(PoseGraphNode(np.eye(4)))
    node_of: dict[int, int] = {0: 0}
    poses: dict[int, np.ndarray] = {0: np.eye(4)}
    accepted_idx: list[int] = [0]
    diagnostics: list = [(1.0, 0.0, True)] + [None] * (n - 1)
    n_odom = n_loop = 0

    for i in range(1, n):
        window = accepted_idx[-(cfg.loop_closure_search_window + 1):]
        results = []
        for j in window:
            try:
                T, fit, rmse = register_frame_pair(frames[i], frames[j], cfg)
            except Exception as e:
                logger.warning(
                    f"  [fusion-fallback] Pose-graph pair registration ({i} -> {j}) failed with "
                    f"{type(e).__name__}: {e}"
                )
                continue
            results.append((j, T, fit, rmse))

        if not results:
            diagnostics[i] = (0.0, float("inf"), False)
            continue

        j_best, T_best, fit_best, rmse_best = max(results, key=lambda r: r[2])
        if fit_best < cfg.min_accept_fitness:
            diagnostics[i] = (fit_best, rmse_best, False)
            continue

        pose_i = poses[j_best] @ T_best
        node_id = len(pose_graph.nodes)
        pose_graph.nodes.append(PoseGraphNode(pose_i))
        node_of[i] = node_id
        poses[i] = pose_i

        for j, T, fit, rmse in results:
            if fit < cfg.min_accept_fitness:
                continue
            info = get_information_matrix(frames[i], frames[j], T, cfg)
            certain = (j == j_best)
            pose_graph.edges.append(
                PoseGraphEdge(node_id, node_of[j], T, info, uncertain=not certain)
            )
            if certain:
                n_odom += 1
            else:
                n_loop += 1

        accepted_idx.append(i)
        diagnostics[i] = (fit_best, rmse_best, True)

    return pose_graph, node_of, diagnostics, n_odom, n_loop


def optimize_pose_graph(
    pose_graph: PoseGraph,
    cfg: RegistrationConfig,
) -> PoseGraph:
    """Global Levenberg-Marquardt optimisation with edge pruning."""
    option = GlobalOptimizationOption(
        max_correspondence_distance=cfg.icp_max_dist_m,
        edge_prune_threshold=cfg.pose_graph_edge_prune_threshold,
        reference_node=0,
    )
    global_optimization(
        pose_graph,
        GlobalOptimizationLevenbergMarquardt(),
        GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return pose_graph


def register_sequence_multiway(
    frames: list[o3d.geometry.PointCloud],
    cfg: RegistrationConfig,
) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]]]:
    """
    Gated pose-graph registration (see :func:`build_pose_graph`).
    """
    cfg.resolve_defaults()
    if not frames:
        return [], []
    if len(frames) == 1:
        return [copy.deepcopy(frames[0])], [(1.0, 0.0, True)]

    pose_graph, node_of, diagnostics, n_odom, n_loop = build_pose_graph(frames, cfg)
    n_before = len(pose_graph.edges)
    if len(pose_graph.nodes) > 1:
        optimize_pose_graph(pose_graph, cfg)
    logger.info(
        f"  Pose graph: {len(pose_graph.nodes)} of {len(frames)} frames in graph, "
        f"{n_before} edges ({n_odom} primary + {n_loop} loop-closure), "
        f"{len(pose_graph.edges)} survived pruning."
    )

    aligned: list[o3d.geometry.PointCloud] = []
    for i, f in enumerate(frames):
        if i in node_of:
            aligned.append(copy.deepcopy(f).transform(pose_graph.nodes[node_of[i]].pose))
        else:
            aligned.append(copy.deepcopy(f))
    return aligned, diagnostics


# ---------------------------------------------------------------------------
# Human 180° Multi-View Registration (RGB-D Odometry + Colored ICP)
# ---------------------------------------------------------------------------

def compute_rotation_deg(R: np.ndarray) -> float:
    """Compute the geodesic rotation angle in degrees from a 3x3 rotation matrix."""
    trace = float(np.trace(R[:3, :3]))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


def compute_translation_mm(T: np.ndarray) -> float:
    """Compute the translation norm in millimetres from a 4x4 matrix."""
    return float(np.linalg.norm(T[:3, 3]) * 1000.0)


def compute_mutual_fitness(source_fitness: float, n_source: int, n_target: int) -> float:
    """
    Compute mutual / minimum-cardinality inlier fitness for asymmetric point clouds.

    In Open3D ICP, fitness = inlier_count / n_source. When registering a larger cloud
    (e.g., connected head + torso, ~11.5k points) onto a smaller cloud (e.g.,
    isolated head, ~3.7k points), source-normalized fitness is mathematically
    bounded by n_target / n_source (~0.32), falsely triggering rejection even when
    the head surface matches with high physical accuracy.

    Mutual fitness normalizes the inlier count by min(n_source, n_target):
        mutual_fitness = inlier_count / min(n_source, n_target)
                       = (source_fitness * n_source) / min(n_source, n_target)
    """
    min_n = min(n_source, n_target)
    if min_n <= 0 or source_fitness <= 0.0:
        return 0.0
    inliers = float(source_fitness) * float(n_source)
    return min(1.0, float(inliers / float(min_n)))


class PairRegistrationResult(tuple):
    """Result tuple from register_rgbd_pair, supporting backward-compatible 8-element unpacking:
    (T, fitness, rmse, accepted, method_name, rot_deg, trans_mm, reason)
    while exposing structured attributes including residual_mm."""

    def __new__(
        cls,
        transformation,
        fitness=None,
        rmse=None,
        accepted=None,
        method=None,
        rotation_deg=None,
        translation_mm=None,
        reason=None,
        residual_mm=None,
    ):
        if isinstance(transformation, (tuple, list)) and fitness is None:
            items = list(transformation)
            if len(items) >= 9:
                residual_mm = items[8]
                items = items[:8]
            t, f, rm, acc, meth, rot, tr, rea = items
        else:
            t = transformation
            f = fitness
            rm = rmse
            acc = accepted
            meth = method
            rot = rotation_deg
            tr = translation_mm
            rea = reason

        inst = super().__new__(cls, (t, f, rm, acc, meth, rot, tr, rea))
        inst.transformation = t
        inst.fitness = f
        inst.rmse = rm
        inst.accepted = acc
        inst.method = meth
        inst.rotation_deg = rot
        inst.translation_mm = tr
        inst.reason = rea
        inst.residual_mm = residual_mm
        return inst


def register_rgbd_pair(
    source_frame: RGBDFrame,
    target_frame: RGBDFrame,
    cfg: RegistrationConfig,
    initial_guess: np.ndarray | None = None,
    is_bridge: bool = False,
) -> tuple[np.ndarray, float, float, bool, str, float, float, str]:
    """
    Register source_frame onto target_frame with Landmark-Guided ICP -> RGB-D Odometry -> Colored ICP -> Quality Gate -> FPFH/RANSAC Fallback.

    CONVENTION
    ----------
    Returns T that maps source_frame INTO target_frame coordinate system:
        p_target = T @ p_source

    Returns
    -------
    (T, fitness, rmse, accepted, method_name, rot_deg, trans_mm, reason)
    """
    cfg.resolve_defaults(target_name="head" if cfg.registration_mode == "rgbd_human" else "box")
    # Guard against empty / degenerate point clouds
    if (
        not hasattr(source_frame, "pcd") or len(source_frame.pcd.points) < 20
        or not hasattr(target_frame, "pcd") or len(target_frame.pcd.points) < 20
    ):
        return PairRegistrationResult(
            np.eye(4),
            0.0,
            float("inf"),
            False,
            "none",
            0.0,
            0.0,
            "Insufficient points in source or target frame (< 20 points)",
            residual_mm=None,
        )

    src_pcd = source_frame.pcd
    tgt_pcd = target_frame.pcd
    n_src = len(src_pcd.points)
    n_tgt = len(tgt_pcd.points)

    src_id = getattr(source_frame, "frame_id", "src")
    tgt_id = getattr(target_frame, "frame_id", "tgt")
    pair_str = f"{src_id}->{tgt_id}"

    # Overlap pre-check: AABB intersection and Centroid-to-Centroid distance
    src_min = np.asarray(src_pcd.get_min_bound())
    src_max = np.asarray(src_pcd.get_max_bound())
    tgt_min = np.asarray(tgt_pcd.get_min_bound())
    tgt_max = np.asarray(tgt_pcd.get_max_bound())
    aabb_intersects = bool(np.all(src_min <= tgt_max) and np.all(tgt_min <= src_max))

    src_centroid = np.asarray(src_pcd.get_center())
    tgt_centroid = np.asarray(tgt_pcd.get_center())
    centroid_dist = float(np.linalg.norm(src_centroid - tgt_centroid))
    max_centroid_dist = float(getattr(cfg, "max_pair_centroid_dist_m", 0.50))
    centroid_ok = centroid_dist <= max_centroid_dist
    overlap_ok = aabb_intersects and centroid_ok

    if not aabb_intersects and not centroid_ok:
        overlap_skip_reason = f"AABB non-overlap, centroid dist {centroid_dist:.2f}m > {max_centroid_dist:.2f}m"
    elif not aabb_intersects:
        overlap_skip_reason = "AABB non-overlap"
    else:
        overlap_skip_reason = f"centroid dist {centroid_dist:.2f}m > {max_centroid_dist:.2f}m"

    init_trans = np.eye(4) if initial_guess is None else initial_guess.copy()

    # 1. Landmark-guided registration
    is_landmark_seeded = False
    if (
        getattr(cfg, "use_landmarks", True)
        and hasattr(source_frame, "color_bgr") and source_frame.color_bgr is not None
        and hasattr(source_frame, "depth_m") and source_frame.depth_m is not None
        and hasattr(target_frame, "color_bgr") and target_frame.color_bgr is not None
        and hasattr(target_frame, "depth_m") and target_frame.depth_m is not None
        and hasattr(source_frame, "intrinsics") and source_frame.intrinsics is not None
    ):
        try:
            lm_T, lm_fit_src, lm_rmse, lm_method, lm_res_mm = register_frame_pair_landmark_guided(
                source_pcd=src_pcd,
                source_color=source_frame.color_bgr,
                source_depth=source_frame.depth_m,
                target_pcd=tgt_pcd,
                target_color=target_frame.color_bgr,
                target_depth=target_frame.depth_m,
                intrinsics_o3d=source_frame.intrinsics,
                cfg=cfg,
            )
            if lm_method == "landmark":
                is_landmark_seeded = True
                lm_rot = compute_rotation_deg(lm_T)
                lm_trans = compute_translation_mm(lm_T)
                lm_fit = compute_mutual_fitness(lm_fit_src, n_src, n_tgt)
                if (
                    lm_fit >= cfg.min_accept_fitness
                    and lm_rmse <= cfg.max_colored_icp_rmse_m * 1.5
                    and lm_rot <= cfg.max_rotation_deg_per_frame
                    and lm_trans <= (cfg.max_translation_m_per_frame * 1000.0)
                ):
                    return PairRegistrationResult(
                        lm_T,
                        lm_fit,
                        lm_rmse,
                        True,
                        "landmark",
                        lm_rot,
                        lm_trans,
                        f"fitness={lm_fit:.2f} (src={lm_fit_src:.2f}) residual={lm_res_mm:.1f}mm",
                        residual_mm=float(lm_res_mm),
                    )
                # Seed downstream refinement with coarse landmark transform if not immediately accepted
                if lm_rot <= cfg.max_rotation_deg_per_frame and lm_trans <= (cfg.max_translation_m_per_frame * 1000.0):
                    init_trans = lm_T
        except Exception as e:
            logger.warning(
                f"  [fusion-fallback] Landmark-guided stage for pair {pair_str} failed with "
                f"{type(e).__name__}: {e}"
            )

    # Check overlap before invoking RGB-D odometry or Colored ICP
    if not overlap_ok:
        logger.warning(f"[overlap-check] pair {pair_str} SKIPPED colored-icp ({overlap_skip_reason}) -- trying FPFH fallback")
        colored_status = f"SKIPPED ({overlap_skip_reason})"
        T_colored = init_trans
        fit_colored = 0.0
        rmse_colored = float("inf")
        rot_colored = 0.0
        trans_colored = 0.0
    else:
        # 2. RGB-D Odometry initial pose estimation
        odo_status = "NOT RUN"
        try:
            rgbd_source = source_frame.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True)
            rgbd_target = target_frame.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True)

            if cfg.rgbd_odometry_method == "hybrid":
                jacobian = RGBDOdometryJacobianFromHybridTerm()
            else:
                jacobian = RGBDOdometryJacobianFromColorTerm()

            option = OdometryOption()
            option.depth_diff_max = cfg.colored_icp_max_dist_m * 1.5
            option.depth_min = 0.20
            option.depth_max = 1.50

            odo_success, odo_trans, odo_info = compute_rgbd_odometry(
                rgbd_source,
                rgbd_target,
                source_frame.intrinsics,
                init_trans,
                jacobian,
                option,
            )

            if odo_success:
                odo_rot = compute_rotation_deg(odo_trans)
                odo_trans_mm = compute_translation_mm(odo_trans)
                if odo_rot <= cfg.max_rotation_deg_per_frame and odo_trans_mm <= (cfg.max_translation_m_per_frame * 1000.0):
                    init_trans = odo_trans
                    odo_status = "PASS"
                else:
                    odo_status = f"MOTION_EXCESSIVE (rot={odo_rot:.1f}deg, trans={odo_trans_mm:.0f}mm)"
            else:
                odo_status = "FAIL"
        except Exception as exc:
            odo_status = f"ERROR ({exc})"

        # 3. Colored ICP Refinement
        if not src_pcd.has_normals():
            src_pcd = o3d.geometry.PointCloud(src_pcd)
            src_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
            )
        if not tgt_pcd.has_normals():
            tgt_pcd = o3d.geometry.PointCloud(tgt_pcd)
            tgt_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
            )

        colored_status = "NOT RUN"
        try:
            # Tightened convergence criteria specifically for landmark-seeded refinement
            colored_criteria = (
                ICPConvergenceCriteria(max_iteration=50, relative_fitness=1e-7, relative_rmse=1e-7)
                if is_landmark_seeded
                else ICPConvergenceCriteria(max_iteration=cfg.colored_icp_max_iterations)
            )
            colored_res = registration_colored_icp(
                src_pcd,
                tgt_pcd,
                cfg.colored_icp_max_dist_m,
                init_trans,
                TransformationEstimationForColoredICP(lambda_geometric=cfg.colored_icp_lambda_geom),
                colored_criteria,
            )
            T_colored = colored_res.transformation
            fit_colored_src = float(colored_res.fitness)
            fit_colored = compute_mutual_fitness(fit_colored_src, n_src, n_tgt)
            rmse_colored = float(colored_res.inlier_rmse)
            rot_colored = compute_rotation_deg(T_colored)
            trans_colored = compute_translation_mm(T_colored)

            # Quality gating
            issues = []
            if fit_colored < cfg.min_colored_icp_fitness:
                issues.append(f"fitness {fit_colored:.3f} (src {fit_colored_src:.3f}) < {cfg.min_colored_icp_fitness:.2f}")
            if rmse_colored > cfg.max_colored_icp_rmse_m:
                issues.append(f"rmse {rmse_colored * 1000.0:.2f}mm > {cfg.max_colored_icp_rmse_m * 1000.0:.1f}mm")
            if rot_colored > cfg.max_rotation_deg_per_frame:
                issues.append(f"rotation {rot_colored:.1f}deg > {cfg.max_rotation_deg_per_frame:.0f}deg")
            if trans_colored > (cfg.max_translation_m_per_frame * 1000.0):
                issues.append(f"translation {trans_colored:.1f}mm > {cfg.max_translation_m_per_frame * 1000.0:.0f}mm")

            if not issues:
                return PairRegistrationResult(
                    T_colored,
                    fit_colored,
                    rmse_colored,
                    True,
                    "colored_icp",
                    rot_colored,
                    trans_colored,
                    f"fitness={fit_colored:.2f} (src={fit_colored_src:.2f}) rmse={rmse_colored * 1000.0:.1f}mm",
                    residual_mm=None,
                )
            colored_status = "FAIL: " + ", ".join(issues)
        except RuntimeError as exc:
            logger.warning(f"[colored-icp] no correspondences for pair {pair_str}, falling through to FPFH tier")
            colored_status = f"ERROR ({exc})"
            T_colored = init_trans
            fit_colored = 0.0
            rmse_colored = float("inf")
            rot_colored = 0.0
            trans_colored = 0.0
        except Exception as exc:
            logger.warning(
                f"  [fusion-fallback] Colored ICP stage for pair {pair_str} failed with "
                f"{type(exc).__name__}: {exc}"
            )
            colored_status = f"ERROR ({exc})"
            T_colored = init_trans
            fit_colored = 0.0
            rmse_colored = float("inf")
            rot_colored = 0.0
            trans_colored = 0.0

    # 4. Geometric Fallback (FPFH + RANSAC -> Point-to-Plane ICP)
    try:
        fb_T, fb_fit_src, fb_rmse = register_frame_pair(src_pcd, tgt_pcd, cfg)
        fb_fit = compute_mutual_fitness(fb_fit_src, n_src, n_tgt)
        fb_rot = compute_rotation_deg(fb_T)
        fb_trans = compute_translation_mm(fb_T)

        if (
            fb_fit >= cfg.min_accept_fitness
            and fb_rmse <= cfg.max_colored_icp_rmse_m * 1.5
            and fb_rot <= cfg.max_rotation_deg_per_frame
            and fb_trans <= (cfg.max_translation_m_per_frame * 1000.0)
        ):
            src_center_y = float(src_pcd.get_axis_aligned_bounding_box().get_center()[1])
            region_str = "lower (neck/shoulder)" if src_center_y > 0.0 else "upper (face/head)"
            logger.debug(f"[registration-diagnostic] pair {pair_str} FPFH fallback SUCCEEDED (region: {region_str}, bbox_center_y={src_center_y:+.3f}m)")
            return PairRegistrationResult(
                fb_T,
                fb_fit,
                fb_rmse,
                True,
                "fpfh_fallback",
                fb_rot,
                fb_trans,
                f"fitness={fb_fit:.2f} (src={fb_fit_src:.2f}) rmse={fb_rmse * 1000.0:.1f}mm",
                residual_mm=None,
            )
    except Exception as exc:
        logger.warning(
            f"  [fusion-fallback] Geometric FPFH fallback for pair {pair_str} failed with "
            f"{type(exc).__name__}: {exc}"
        )

    src_center_y = float(src_pcd.get_axis_aligned_bounding_box().get_center()[1])
    region_str = "lower (neck/shoulder)" if src_center_y > 0.0 else "upper (face/head)"
    fail_reason = f"Colored ICP failed ({colored_status}); Fallback also failed"
    logger.debug(
        f"[registration-diagnostic] pair {pair_str} REJECTED / LOW-FITNESS "
        f"(region: {region_str}, bbox_center_y={src_center_y:+.3f}m) -- {fail_reason}"
    )
    return PairRegistrationResult(
        T_colored,
        fit_colored,
        rmse_colored,
        False,
        "none",
        rot_colored,
        trans_colored,
        fail_reason,
        residual_mm=None,
    )


def is_upper_face_frame(frame: RGBDFrame | o3d.geometry.PointCloud, threshold_y: float = 0.0) -> bool:
    """Classify whether a frame's point cloud is in the upper (face/head) region (center_y <= threshold_y)."""
    pcd = frame.pcd if hasattr(frame, "pcd") else frame
    if pcd is None or not hasattr(pcd, "has_points") or not pcd.has_points():
        return False
    cy = float(pcd.get_axis_aligned_bounding_box().get_center()[1])
    return cy <= threshold_y


class BuildPoseGraphResult(tuple):
    """Result tuple from build_pose_graph_rgbd, supporting backward-compatible 10-element unpacking."""
    def __new__(
        cls,
        items,
        frame_states: dict[int, str] | None = None,
        unanchored_chain: list[int] | None = None,
        bridge_edges: list[tuple[int, int, int]] | None = None,
        potential_loop_closures: list[dict] | None = None,
        residuals: dict[int, float | None] | None = None,
    ):
        inst = super().__new__(cls, items)
        inst.frame_states = frame_states or {}
        inst.unanchored_chain = unanchored_chain or []
        inst.bridge_edges = bridge_edges or []
        inst.potential_loop_closures = potential_loop_closures or []
        inst.residuals = residuals or {}
        return inst


def _call_register_pair(src, dst, cfg, initial_guess=None, is_bridge=False):
    return register_rgbd_pair(src, dst, cfg, initial_guess=initial_guess, is_bridge=is_bridge)


def build_pose_graph_rgbd(
    frames: list[RGBDFrame],
    cfg: RegistrationConfig,
):
    """
    Build a multiway pose graph for human RGB-D frames with bounded search and quality gating.

    Returns
    -------
    (pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses, best_methods, best_scores, rejection_reasons)
    """
    cfg.resolve_defaults(target_name="head")
    n = len(frames)
    pose_graph = PoseGraph()
    pose_graph.nodes.append(PoseGraphNode(np.eye(4)))
    node_of: dict[int, int] = {0: 0}
    poses: dict[int, np.ndarray] = {0: np.eye(4)}
    accepted_idx: list[int] = [0]
    diagnostics: list[tuple[float, float, bool] | None] = [(1.0, 0.0, True)] + [None] * (n - 1)
    pair_logs: list[dict] = []
    best_methods: dict[int, str] = {0: "reference"}
    best_scores: dict[int, str] = {0: "(origin)"}
    best_residuals: dict[int, float | None] = {0: None}
    rejection_reasons: dict[int, str] = {}
    n_odom = 0
    n_loop = 0

    last_rel_trans: np.ndarray | None = None
    frames_since_accepted: int = 0

    # Bounded recovery tracking structures
    bridge_buffer: list[dict] = []
    active_unanchored_chain: list[int] = []
    promotable_unanchored_chain: list[int] = []
    all_unanchored_nodes: list[int] = []
    frame_states: dict[int, str] = {0: "DIRECT_ACCEPTED"}
    bridge_edges_info: list[tuple[int, int, int]] = []
    potential_loop_closures: list[dict] = []
    frame_hops: dict[int, int] = {0: 0}
    composed_to_anchor: dict[int, np.ndarray] = {0: np.eye(4)}

    for i in range(1, n):
        # Region classification for source frame i
        face_y_thresh = getattr(cfg, "face_region_center_y_threshold_m", 0.0)
        src_is_upper = is_upper_face_frame(frames[i], face_y_thresh)
        src_cy = (
            float(frames[i].pcd.get_axis_aligned_bounding_box().get_center()[1])
            if (hasattr(frames[i], "pcd") and frames[i].pcd is not None and frames[i].pcd.has_points())
            else 0.0
        )

        # Multi-candidate strategy: check preceding accepted frames in the search window
        # AND frame 0 (the anchor reference) to pick whichever gives higher inlier fitness.
        window = accepted_idx[-(cfg.search_accepted_window + 1):]
        target_indices = []
        for idx in reversed(window):
            if idx not in target_indices:
                target_indices.append(idx)
        if 0 not in target_indices:
            target_indices.append(0)

        candidates = []
        misleading_candidates = []

        prev_accepted_idx = accepted_idx[-1]
        prev_accepted_pose = poses[prev_accepted_idx]

        effective_translation_bound = min(
            cfg.max_pose_jump_translation_m * (1 + frames_since_accepted),
            cfg.max_pose_jump_translation_m * 6.0
        )
        effective_rotation_bound = min(
            cfg.max_pose_jump_rotation_deg * (1 + frames_since_accepted),
            cfg.max_pose_jump_rotation_deg * 6.0
        )

        for j in target_indices:
            tgt_is_upper = is_upper_face_frame(frames[j], face_y_thresh)
            tgt_cy = (
                float(frames[j].pcd.get_axis_aligned_bounding_box().get_center()[1])
                if (hasattr(frames[j], "pcd") and frames[j].pcd is not None and frames[j].pcd.has_points())
                else 0.0
            )
            is_face_pair = (src_is_upper and tgt_is_upper)

            init_guess = np.eye(4)
            if j == prev_accepted_idx and frames_since_accepted == 0 and last_rel_trans is not None:
                rot_p = compute_rotation_deg(last_rel_trans)
                trans_p_mm = compute_translation_mm(last_rel_trans)
                if rot_p <= cfg.max_rotation_deg_per_frame and trans_p_mm <= (cfg.max_translation_m_per_frame * 1000.0):
                    init_guess = last_rel_trans.copy()

            res = _call_register_pair(frames[i], frames[j], cfg, initial_guess=init_guess, is_bridge=False)
            T, fit, rmse, accepted, method, rot, trans, reason = res
            res_residual = getattr(res, "residual_mm", None)
            if res_residual is None and "residual=" in reason:
                try:
                    res_residual = float(reason.split("residual=")[1].split("mm")[0])
                except Exception:
                    pass

            # Post-registration physical-plausibility check (scaled by frames elapsed)
            exceeds_jump = False
            jump_trans_m = 0.0
            jump_rot_deg = 0.0

            if accepted:
                cand_pose = poses[j] @ T
                delta_pose = np.linalg.inv(prev_accepted_pose) @ cand_pose
                jump_trans_m = float(np.linalg.norm(delta_pose[:3, 3]))
                jump_rot_deg = float(compute_rotation_deg(delta_pose[:3, :3]))

                if jump_trans_m > effective_translation_bound or jump_rot_deg > effective_rotation_bound:
                    exceeds_jump = True
                    misleading_candidates.append({
                        "target": j,
                        "T": T,
                        "fitness": fit,
                        "rmse": rmse,
                        "rot": rot,
                        "trans": trans,
                        "method": method,
                        "residual_mm": res_residual,
                        "score_str": reason,
                        "jump_trans_m": jump_trans_m,
                        "jump_rot_deg": jump_rot_deg,
                        "allowed_trans_m": effective_translation_bound,
                        "allowed_rot_deg": effective_rotation_bound,
                        "frames_since_accepted": frames_since_accepted,
                    })

            pair_logs.append({
                "source": i,
                "target": j,
                "success": accepted and not exceeds_jump,
                "method": method,
                "fitness": fit,
                "rmse_mm": rmse * 1000.0 if np.isfinite(rmse) else 999.0,
                "rotation_deg": rot,
                "translation_mm": trans,
                "residual_mm": res_residual,
                "reason": (
                    f"pose jump ({jump_trans_m:.2f}m/{jump_rot_deg:.1f}deg) > allowed ({effective_translation_bound:.2f}m/{effective_rotation_bound:.1f}deg)"
                    if exceeds_jump else reason
                ),
                "jump_trans_m": jump_trans_m,
                "jump_rot_deg": jump_rot_deg,
                "is_face_pair": is_face_pair,
                "src_center_y": src_cy,
                "tgt_center_y": tgt_cy,
            })

            # Print pair log with exact tier and score
            if accepted and not exceeds_jump:
                logger.debug(f"    pair {i:02d} -> {j:02d}: tier={method} {reason} ACCEPTED")
                candidates.append((j, T, fit, rmse, rot, trans, method, reason, res_residual))
            elif accepted and exceeds_jump:
                logger.debug(
                    f"    pair {i:02d} -> {j:02d}: tier={method} {reason} "
                    f"REJECTED -- pose jump ({jump_trans_m:.2f}m/{jump_rot_deg:.1f}deg) exceeds plausible motion "
                    f"(bound {effective_translation_bound:.2f}m/{effective_rotation_bound:.1f}deg, scaled for {frames_since_accepted} "
                    f"frames since last accept)"
                )
            else:
                logger.debug(f"    pair {i:02d} -> {j:02d}: tier=none REJECTED ({reason})")

        if not candidates:
            # If candidates were rejected due to excessive pose jump, this is a physical motion violation,
            # not a tracking loss gap. Reject and do not attempt bridging.
            if misleading_candidates:
                frames_since_accepted += 1
                last_rel_trans = None
                mis = max(misleading_candidates, key=lambda m: m["fitness"])
                diagnostics[i] = (mis["fitness"], mis["rmse"], False)
                best_methods[i] = mis["method"]
                score_str = mis["score_str"]
                best_scores[i] = score_str
                best_residuals[i] = mis.get("residual_mm")
                n_skip = mis["frames_since_accepted"]
                rejection_msg = (
                    f"REJECTED -- pose jump ({mis['jump_trans_m']:.2f}m/{mis['jump_rot_deg']:.1f}deg) exceeds plausible motion "
                    f"(bound/allowed: {mis['allowed_trans_m']:.2f}m/{mis['allowed_rot_deg']:.1f}deg, scaled for {n_skip} "
                    f"frames since last accept)"
                )
                rejection_reasons[i] = rejection_msg
                frame_states[i] = "REJECTED"
                logger.debug(
                    f"    frame {i:02d}: tier={mis['method']} {score_str}\n"
                    f"              {rejection_msg}"
                )
                continue

            # All candidates in the window failed: check if bounded recovery is enabled
            enable_recovery = getattr(cfg, "enable_bounded_recovery", True)
            if not enable_recovery:
                frames_since_accepted += 1
                last_rel_trans = None
                best_prev = pair_logs[-1] if pair_logs else {}
                diagnostics[i] = (best_prev.get("fitness", 0.0), best_prev.get("rmse_mm", float("inf")) / 1000.0, False)
                best_methods[i] = "none"
                best_scores[i] = ""
                best_residuals[i] = None
                logger.debug(f"    frame {i:02d}: tier=none REJECTED")
                frame_states[i] = "REJECTED"
                continue

            # Bounded recovery path
            # Determine predecessor to link against
            if bridge_buffer:
                pred_idx = bridge_buffer[-1]["frame_idx"]
                anchor_idx = bridge_buffer[-1]["anchor_idx"]
                hop = bridge_buffer[-1]["hop"] + 1
                prev_T_to_anchor = bridge_buffer[-1]["T_to_anchor"]
            elif active_unanchored_chain:
                pred_idx = active_unanchored_chain[-1]
                anchor_idx = accepted_idx[-1]
                hop = frame_hops.get(pred_idx, len(active_unanchored_chain)) + 1
                prev_T_to_anchor = composed_to_anchor.get(pred_idx, np.eye(4))
            else:
                pred_idx = accepted_idx[-1]
                anchor_idx = pred_idx
                hop = 1
                prev_T_to_anchor = np.eye(4)

            # Attempt local registration against predecessor
            res_local = _call_register_pair(frames[i], frames[pred_idx], cfg, is_bridge=True)
            T_loc, fit_loc, rmse_loc, acc_loc, meth_loc, rot_loc, trans_loc, reason_loc = res_local

            if not acc_loc:
                # Local link broken: clear buffer and terminate active unanchored chain to prevent reusing stale transforms
                bridge_buffer.clear()
                active_unanchored_chain.clear()
                promotable_unanchored_chain.clear()
                frames_since_accepted += 1
                last_rel_trans = None
                diagnostics[i] = (fit_loc, rmse_loc, False)
                best_methods[i] = "none"
                best_scores[i] = ""
                best_residuals[i] = None
                rejection_reasons[i] = f"local bridge link {i}->{pred_idx} failed ({reason_loc})"
                frame_states[i] = "REJECTED"
                logger.debug(f"    frame {i:02d}: local bridge link {i}->{pred_idx} REJECTED ({reason_loc})")
                continue

            # Local link succeeded: check kinematic jump of composed transform from anchor
            T_to_anchor_cand = prev_T_to_anchor @ T_loc
            cand_pose = poses[anchor_idx] @ T_to_anchor_cand
            delta_pose = np.linalg.inv(poses[anchor_idx]) @ cand_pose
            jump_trans_m = float(np.linalg.norm(delta_pose[:3, 3]))
            jump_rot_deg = float(compute_rotation_deg(delta_pose[:3, :3]))

            effective_trans_bound = min(
                cfg.max_pose_jump_translation_m * (1 + hop),
                cfg.max_pose_jump_translation_m * 6.0
            )
            effective_rot_bound = min(
                cfg.max_pose_jump_rotation_deg * (1 + hop),
                cfg.max_pose_jump_rotation_deg * 6.0
            )

            if jump_trans_m > effective_trans_bound or jump_rot_deg > effective_rot_bound:
                # Pose jump exceeds motion bounds: reject, clear buffer, and terminate unanchored chain
                bridge_buffer.clear()
                active_unanchored_chain.clear()
                promotable_unanchored_chain.clear()
                frames_since_accepted += 1
                last_rel_trans = None
                diagnostics[i] = (fit_loc, rmse_loc, False)
                best_methods[i] = meth_loc
                best_scores[i] = reason_loc
                best_residuals[i] = getattr(res_local, "residual_mm", None)
                rejection_msg = (
                    f"REJECTED -- pose jump ({jump_trans_m:.2f}m/{jump_rot_deg:.1f}deg) exceeds plausible motion "
                    f"(bound {effective_trans_bound:.2f}m/{effective_rot_bound:.1f}deg, scaled for {hop} hops)"
                )
                rejection_reasons[i] = rejection_msg
                frame_states[i] = "REJECTED"
                logger.debug(f"    frame {i:02d}: {rejection_msg}")
                continue

            # Kinematic check passed!
            max_buf_size = getattr(cfg, "max_bridge_buffer_size", 2)
            if len(bridge_buffer) < max_buf_size and not active_unanchored_chain:
                # Store in bridge buffer as provisional bridge frame
                bridge_buffer.append({
                    "frame_idx": i,
                    "target_idx": pred_idx,
                    "T_local": T_loc,
                    "T_to_anchor": T_to_anchor_cand,
                    "anchor_idx": anchor_idx,
                    "fitness": fit_loc,
                    "rmse": rmse_loc,
                    "method": meth_loc,
                    "reason": reason_loc,
                    "residual_mm": getattr(res_local, "residual_mm", None),
                    "hop": hop,
                })
                diagnostics[i] = (fit_loc, rmse_loc, False)
                frame_states[i] = "PROVISIONAL_BRIDGE"
                best_methods[i] = meth_loc
                best_scores[i] = reason_loc
                best_residuals[i] = getattr(res_local, "residual_mm", None)
                frames_since_accepted += 1
                last_rel_trans = None
                logger.debug(f"    frame {i:02d}: tier={meth_loc} PROVISIONAL_BRIDGE (hop={hop}, buffered={len(bridge_buffer)}/{max_buf_size})")
                continue
            else:
                # Buffer capacity reached (or continuing unanchored chain).
                # No independent reference overlap is possible: mark UNANCHORED_ODOMETRY
                # 1. Promote provisional bridge buffer entries to pose graph as uncertain bridge nodes
                for b in bridge_buffer:
                    b_idx = b["frame_idx"]
                    b_pose = poses[b["anchor_idx"]] @ b["T_to_anchor"]
                    b_node_id = len(pose_graph.nodes)
                    pose_graph.nodes.append(PoseGraphNode(b_pose))
                    node_of[b_idx] = b_node_id
                    poses[b_idx] = b_pose
                    diagnostics[b_idx] = (b["fitness"], b["rmse"], False)
                    frame_states[b_idx] = "PROMOTED_BRIDGE"
                    best_methods[b_idx] = b["method"]
                    best_scores[b_idx] = b["reason"]
                    best_residuals[b_idx] = b.get("residual_mm")
                    # Add bridge edge with uncertain=True and info scaled by 1/(1+h)
                    # Heuristic scaling: 1 / (1 + h), where h is topological hop distance
                    b_info = get_information_matrix(frames[b_idx].pcd, frames[b["target_idx"]].pcd, b["T_local"], cfg)
                    b_info_scaled = b_info * (1.0 / (1.0 + b["hop"]))
                    pose_graph.edges.append(
                        PoseGraphEdge(node_of[b_idx], node_of[b["target_idx"]], b["T_local"], b_info_scaled, uncertain=True)
                    )
                    bridge_edges_info.append((b_idx, b["target_idx"], b["hop"]))
                bridge_buffer.clear()

                # 2. Add frame i to pose graph as UNANCHORED_ODOMETRY
                pose_i = poses[pred_idx] @ T_loc
                node_id = len(pose_graph.nodes)
                pose_graph.nodes.append(PoseGraphNode(pose_i))
                node_of[i] = node_id
                poses[i] = pose_i
                diagnostics[i] = (fit_loc, rmse_loc, True)
                frame_states[i] = "UNANCHORED_ODOMETRY"
                best_methods[i] = meth_loc
                best_scores[i] = reason_loc
                best_residuals[i] = getattr(res_local, "residual_mm", None)
                info = get_information_matrix(frames[i].pcd, frames[pred_idx].pcd, T_loc, cfg)
                info_scaled = info * (1.0 / (1.0 + hop))
                pose_graph.edges.append(
                    PoseGraphEdge(node_id, node_of[pred_idx], T_loc, info_scaled, uncertain=True)
                )
                bridge_edges_info.append((i, pred_idx, hop))
                active_unanchored_chain.append(i)
                promotable_unanchored_chain.append(i)
                all_unanchored_nodes.append(i)
                frame_hops[i] = hop
                composed_to_anchor[i] = T_to_anchor_cand
                frames_since_accepted += 1
                last_rel_trans = None
                logger.debug(f"    frame {i:02d}: tier={meth_loc} UNANCHORED_ODOMETRY (hop={hop}, node={node_id})")
                continue

        # Prioritize sequential candidate (prev_accepted_idx) if it succeeded and meets quality criteria
        seq_candidates = [c for c in candidates if c[0] == prev_accepted_idx]
        if seq_candidates:
            j_best, T_best, fit_best, rmse_best, rot_best, trans_best, method_best, reason_best, res_best = seq_candidates[0]
            last_rel_trans = T_best.copy() if frames_since_accepted == 0 else None
        else:
            j_best, T_best, fit_best, rmse_best, rot_best, trans_best, method_best, reason_best, res_best = max(
                candidates, key=lambda c: c[2]
            )
            last_rel_trans = None

        frame_state_i = "DIRECT_ACCEPTED"
        res_crosscheck = None
        if bridge_buffer:
            # Independent geometric validation against bridge buffer
            b_head = bridge_buffer[-1]
            res_crosscheck = _call_register_pair(frames[i], frames[b_head["frame_idx"]], cfg, is_bridge=True)
            if res_crosscheck[3]:
                T_loc_cross = res_crosscheck[0]
                T_comp_to_anchor = b_head["T_to_anchor"] @ T_loc_cross
                anchor_idx = b_head["anchor_idx"]
                if j_best == anchor_idx:
                    T_dir_to_anchor = T_best
                else:
                    T_dir_to_anchor = np.linalg.inv(poses[anchor_idx]) @ poses[j_best] @ T_best

                delta_T = np.linalg.inv(T_dir_to_anchor) @ T_comp_to_anchor
                trans_disagree = float(np.linalg.norm(delta_T[:3, 3]))
                rot_disagree = float(compute_rotation_deg(delta_T[:3, :3]))
                max_trans_cross = getattr(cfg, "max_bridge_crosscheck_trans_m", 0.020)
                max_rot_cross = getattr(cfg, "max_bridge_crosscheck_rot_deg", 10.0)

                if trans_disagree <= max_trans_cross and rot_disagree <= max_rot_cross:
                    logger.debug(f"    frame {i:02d}: independent bridge cross-check PASSED (trans disagree {trans_disagree*1000.0:.1f}mm <= {max_trans_cross*1000.0:.1f}mm, rot {rot_disagree:.1f}deg <= {max_rot_cross:.1f}deg)")
                    for b in bridge_buffer:
                        b_idx = b["frame_idx"]
                        b_pose = poses[b["anchor_idx"]] @ b["T_to_anchor"]
                        b_node_id = len(pose_graph.nodes)
                        pose_graph.nodes.append(PoseGraphNode(b_pose))
                        node_of[b_idx] = b_node_id
                        poses[b_idx] = b_pose
                        diagnostics[b_idx] = (b["fitness"], b["rmse"], False)
                        frame_states[b_idx] = "PROMOTED_BRIDGE"
                        best_methods[b_idx] = b["method"]
                        best_scores[b_idx] = b["reason"]
                        best_residuals[b_idx] = b.get("residual_mm")
                        b_info = get_information_matrix(frames[b_idx].pcd, frames[b["target_idx"]].pcd, b["T_local"], cfg)
                        b_info_scaled = b_info * (1.0 / (1.0 + b["hop"]))
                        pose_graph.edges.append(
                            PoseGraphEdge(node_of[b_idx], node_of[b["target_idx"]], b["T_local"], b_info_scaled, uncertain=True)
                        )
                        bridge_edges_info.append((b_idx, b["target_idx"], b["hop"]))
                    bridge_buffer.clear()
                    frame_state_i = "TERMINAL_RECOVERED"
                else:
                    logger.debug(f"    frame {i:02d}: independent bridge cross-check FAILED (trans disagree {trans_disagree*1000.0:.1f}mm > {max_trans_cross*1000.0:.1f}mm or rot {rot_disagree:.1f}deg > {max_rot_cross:.1f}deg)")
                    bridge_buffer.clear()
                    diagnostics[i] = (fit_best, rmse_best, False)
                    best_methods[i] = method_best
                    best_scores[i] = reason_best
                    best_residuals[i] = res_best
                    rejection_reasons[i] = f"REJECTED -- independent bridge cross-check failed: trans disagree {trans_disagree*1000.0:.1f}mm > {max_trans_cross*1000.0:.1f}mm or rot {rot_disagree:.1f}deg > {max_rot_cross:.1f}deg"
                    frame_states[i] = "REJECTED"
                    frames_since_accepted += 1
                    continue
            else:
                bridge_buffer.clear()
                frame_state_i = "DIRECT_ACCEPTED"
        else:
            frame_state_i = "DIRECT_ACCEPTED"

        frames_since_accepted = 0
        best_methods[i] = method_best
        best_scores[i] = reason_best
        best_residuals[i] = res_best
        frame_states[i] = frame_state_i

        # Mark primary candidate in pair_logs for frame i
        for pl in pair_logs:
            if pl["source"] == i:
                pl["is_primary"] = (pl["target"] == j_best)

        # Fusion eligibility check (secondary stricter gate for Stage 4 fusion)
        min_fit_fusion = getattr(cfg, "min_fitness_for_fusion", 0.60)
        max_res_fusion = getattr(cfg, "max_landmark_residual_for_fusion_mm", 5.0)
        min_face_fit = getattr(cfg, "min_face_pair_fitness_for_fusion", 0.75)
        max_face_rmse = getattr(cfg, "max_face_pair_rmse_for_fusion_mm", 4.0)

        best_tgt_is_upper = is_upper_face_frame(frames[j_best], face_y_thresh)
        best_is_face_pair = (src_is_upper and best_tgt_is_upper)

        lm_res_val = res_best
        if lm_res_val is None and "residual=" in reason_best:
            try:
                lm_res_val = float(reason_best.split("residual=")[1].split("mm")[0])
            except Exception as e:
                logger.warning(
                    f"    [fusion-fallback] Parsing landmark residual for frame {i} failed with "
                    f"{type(e).__name__}: {e}"
                )

        rmse_best_mm = rmse_best * 1000.0 if np.isfinite(rmse_best) else 999.0

        if best_is_face_pair:
            if fit_best < min_face_fit:
                logger.debug(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (face-pair fitness {fit_best:.2f} < {min_face_fit:.2f} threshold)")
            elif rmse_best_mm > max_face_rmse:
                logger.debug(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (face-pair rmse {rmse_best_mm:.1f}mm > {max_face_rmse:.1f}mm threshold)")
            else:
                logger.debug(f"    frame {i:02d}: tier={method_best} {reason_best} ACCEPTED")
        else:
            if fit_best < min_fit_fusion:
                logger.debug(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (fitness {fit_best:.2f} < {min_fit_fusion:.2f} fusion threshold)")
            elif lm_res_val is not None and lm_res_val > max_res_fusion:
                logger.debug(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (landmark residual {lm_res_val:.1f}mm > {max_res_fusion:.1f}mm fusion threshold)")
            else:
                logger.debug(f"    frame {i:02d}: tier={method_best} {reason_best} ACCEPTED")

        pose_i = poses[j_best] @ T_best
        node_id = len(pose_graph.nodes)
        pose_graph.nodes.append(PoseGraphNode(pose_i))
        node_of[i] = node_id
        poses[i] = pose_i

        # Add edges: primary sequential candidate is certain (odometry);
        # other non-consecutive candidates are uncertain (loop closures) and must satisfy min_loop_closure_fitness.
        min_loop_fit = getattr(cfg, "min_loop_closure_fitness", 0.65)
        for c in candidates:
            j, T, fit, rmse, rot, trans, method, reason = c[:8]
            certain = (j == j_best)
            if not certain and fit < min_loop_fit:
                continue
            info = get_information_matrix(frames[i].pcd, frames[j].pcd, T, cfg)
            pose_graph.edges.append(
                PoseGraphEdge(node_id, node_of[j], T, info, uncertain=not certain)
            )
            if certain:
                n_odom += 1
            else:
                n_loop += 1
                if promotable_unanchored_chain and (i in promotable_unanchored_chain or j in promotable_unanchored_chain):
                    potential_loop_closures.append({"source": i, "target": j, "fitness": fit, "rmse": rmse})

        # If terminal recovered, also add bridge edge to buffer head with uncertain=True and scaled info
        if frame_state_i == "TERMINAL_RECOVERED" and res_crosscheck is not None and res_crosscheck[3]:
            head_idx = b_head["frame_idx"]
            b_hop = b_head["hop"] + 1
            br_info = get_information_matrix(frames[i].pcd, frames[head_idx].pcd, res_crosscheck[0], cfg)
            br_info_scaled = br_info * (1.0 / (1.0 + b_hop))
            pose_graph.edges.append(
                PoseGraphEdge(node_id, node_of[head_idx], res_crosscheck[0], br_info_scaled, uncertain=True)
            )
            bridge_edges_info.append((i, head_idx, b_hop))

        # If active unanchored chain exists, check if frame i connects to the unanchored chain head
        if active_unanchored_chain and active_unanchored_chain[-1] in node_of and i not in active_unanchored_chain:
            last_u = active_unanchored_chain[-1]
            res_u = _call_register_pair(frames[i], frames[last_u], cfg, is_bridge=True)
            if res_u[3] and res_u[1] >= min_loop_fit:
                u_hop = frame_hops.get(last_u, 1) + 1
                u_info = get_information_matrix(frames[i].pcd, frames[last_u].pcd, res_u[0], cfg)
                u_info_scaled = u_info * (1.0 / (1.0 + u_hop))
                pose_graph.edges.append(
                    PoseGraphEdge(node_id, node_of[last_u], res_u[0], u_info_scaled, uncertain=True)
                )
                n_loop += 1
                bridge_edges_info.append((i, last_u, u_hop))
                potential_loop_closures.append({"source": i, "target": j_best, "fitness": fit_best, "rmse": rmse_best})
            else:
                promotable_unanchored_chain.clear()
            active_unanchored_chain.clear()

        accepted_idx.append(i)
        diagnostics[i] = (fit_best, rmse_best, True)

    # TASK 2: Diagnostic summary of loop-closure density and accepted-pair fitness/RMSE distribution
    accepted_pairs = [p for p in pair_logs if p.get("success", False)]
    total_accepted_frames = len(accepted_idx)
    avg_loop_per_frame = (n_loop / max(1, total_accepted_frames))

    logger.info("\n" + "=" * 65)
    logger.info("RGB-D POSE GRAPH & LOOP CLOSURE DIAGNOSTICS (Face & Cloud Metrics)")
    logger.info("=" * 65)
    logger.info(f"  Total accepted frames: {total_accepted_frames} of {n}")
    logger.info(f"  Total loop-closure edges formed: {n_loop}")
    logger.info(f"  Average loop-closure edges per frame: {avg_loop_per_frame:.2f}")

    if accepted_pairs:
        min_face_fit = getattr(cfg, "min_face_pair_fitness_for_fusion", 0.75)
        max_face_rmse = getattr(cfg, "max_face_pair_rmse_for_fusion_mm", 4.0)
        min_fit_fusion = getattr(cfg, "min_fitness_for_fusion", 0.60)
        max_colored_rmse_mm = getattr(cfg, "max_colored_icp_rmse_m", 0.006) * 1000.0
        min_accept_fit = getattr(cfg, "min_accept_fitness", 0.50) if getattr(cfg, "min_accept_fitness", None) is not None else 0.45

        fits = [p["fitness"] for p in accepted_pairs]
        rmses = [p["rmse_mm"] for p in accepted_pairs if p["rmse_mm"] < 990.0]
        n_pairs = len(accepted_pairs)

        fit_high = sum(1 for f in fits if f >= min_face_fit)
        fit_mod = sum(1 for f in fits if min_fit_fusion <= f < min_face_fit)
        fit_border = sum(1 for f in fits if f < min_fit_fusion)

        rmse_tight = sum(1 for r in rmses if r <= max_face_rmse)
        rmse_mod = sum(1 for r in rmses if max_face_rmse < r <= max_colored_rmse_mm)
        rmse_border = sum(1 for r in rmses if r > max_colored_rmse_mm)

        logger.info(f"\n  Accepted Pair Fitness Distribution (total accepted pairs: {n_pairs}):")
        logger.info(f"    High (>= {min_face_fit:.2f}):        {fit_high:3d} ({fit_high/n_pairs*100.0:5.1f}%) [comfortable alignment]")
        logger.info(f"    Moderate [{min_fit_fusion:.2f}, {min_face_fit:.2f}): {fit_mod:3d} ({fit_mod/n_pairs*100.0:5.1f}%) [moderate alignment]")
        logger.info(f"    Borderline (< {min_fit_fusion:.2f}):   {fit_border:3d} ({fit_border/n_pairs*100.0:5.1f}%) [near acceptance threshold]")
        logger.info(f"    Fitness: mean={np.mean(fits):.3f}, median={np.median(fits):.3f}, min={np.min(fits):.3f}, max={np.max(fits):.3f}")

        logger.info(f"\n  Accepted Pair RMSE Distribution:")
        logger.info(f"    Tight (<= {max_face_rmse:.1f} mm):     {rmse_tight:3d} ({rmse_tight/n_pairs*100.0:5.1f}%) [clears strict face threshold]")
        logger.info(f"    Moderate ({max_face_rmse:.1f}-{max_colored_rmse_mm:.1f} mm): {rmse_mod:3d} ({rmse_mod/n_pairs*100.0:5.1f}%)")
        logger.info(f"    Borderline (> {max_colored_rmse_mm:.1f} mm): {rmse_border:3d} ({rmse_border/n_pairs*100.0:5.1f}%) [near max RMSE limit]")
        if rmses:
            logger.info(f"    RMSE: mean={np.mean(rmses):.2f}mm, median={np.median(rmses):.2f}mm, min={np.min(rmses):.2f}mm, max={np.max(rmses):.2f}mm")

        face_pairs = [p for p in accepted_pairs if p.get("is_face_pair", False)]
        if face_pairs:
            n_fp = len(face_pairs)
            fp_fits = [p["fitness"] for p in face_pairs]
            fp_rmses = [p["rmse_mm"] for p in face_pairs if p["rmse_mm"] < 990.0]
            fp_clear = sum(1 for f in fp_fits if f >= min_face_fit)
            fp_border = sum(1 for f in fp_fits if min_accept_fit <= f < min_face_fit)
            logger.info(f"\n  Face-Region Pairs specifically (both frames upper face/head, n={n_fp}):")
            logger.info(f"    Clears strict threshold (>= {min_face_fit:.2f}): {fp_clear:3d} ({fp_clear/n_fp*100.0:5.1f}%)")
            logger.info(f"    Borderline [{min_accept_fit:.2f}, {min_face_fit:.2f}):           {fp_border:3d} ({fp_border/n_fp*100.0:5.1f}%) [excluded from face fusion]")
            if fp_rmses:
                fp_rmse_le = sum(1 for r in fp_rmses if r <= max_face_rmse)
                logger.info(f"    RMSE <= {max_face_rmse:.1f} mm:                    {fp_rmse_le:3d} ({fp_rmse_le/n_fp*100.0:5.1f}%)")
        else:
            logger.info("\n  Face-Region Pairs specifically: 0 accepted upper-to-upper pairs detected (frames in lower/neutral region)")
    logger.info("=" * 65 + "\n")

    return BuildPoseGraphResult(
        (pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses, best_methods, best_scores, rejection_reasons),
        frame_states=frame_states,
        unanchored_chain=promotable_unanchored_chain,
        bridge_edges=bridge_edges_info,
        potential_loop_closures=potential_loop_closures,
        residuals=best_residuals,
    )


def register_sequence_rgbd_human(
    frames: list[RGBDFrame],
    cfg: RegistrationConfig,
) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]], list[np.ndarray], dict]:
    """
    Multi-view human head/body registration using RGB-D odometry, colored ICP, and global pose graph.

    Returns
    -------
    aligned:
        Point clouds transformed into the common world coordinate system.
    diagnostics:
        Per-frame (fitness, rmse, accepted) tuples.
    poses:
        List of 4x4 camera-to-world poses for all frames.
    summary_stats:
        Dictionary of human registration metrics for the required report.
    """
    cfg.resolve_defaults(target_name="head")
    if not frames:
        return [], [], [], {}

    if len(frames) == 1:
        return [copy.deepcopy(frames[0].pcd)], [(1.0, 0.0, True)], [np.eye(4)], {
            "captured": 1,
            "accepted": 1,
            "rejected": 0,
            "avg_rmse_mm": 0.0,
            "median_rmse_mm": 0.0,
            "max_trans_jump_mm": 0.0,
            "max_rot_jump_deg": 0.0,
            "first_rejected_frame": None,
            "frame_methods": {0: "reference"},
            "frame_scores": {0: "(origin)"},
            "residuals": {0: None},
            "frame_residuals": {0: None},
            "rejection_reasons": {},
            "tier_counts": {"landmark": 0, "colored_icp": 0, "fpfh_fallback": 0, "rejected": 0},
            "tier_summary_str": "0 landmark, 0 colored_icp, 0 fpfh_fallback, 0 rejected",
        }

    logger.info("\n  Running Human RGB-D Pairwise Registration with Landmark Guidance, Colored ICP & Bounded Search...")
    pg_res = build_pose_graph_rgbd(frames, cfg)
    pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, unopt_poses, best_methods, best_scores, rejection_reasons = pg_res

    frame_states = getattr(pg_res, "frame_states", {})
    unanchored_chain = getattr(pg_res, "unanchored_chain", [])
    potential_loop_closures = getattr(pg_res, "potential_loop_closures", [])
    bridge_edges = getattr(pg_res, "bridge_edges", [])
    residuals = getattr(pg_res, "residuals", {})

    n_before = len(pose_graph.edges)
    if len(pose_graph.nodes) > 1:
        optimize_pose_graph(pose_graph, cfg)

    # Check whether a loop closure edge connecting the unanchored chain survived pruning in optimize_pose_graph
    unanchored_promoted = False
    if unanchored_chain and potential_loop_closures:
        surviving_edges = {(e.source_node_id, e.target_node_id) for e in pose_graph.edges}
        surviving_edges.update({(e.target_node_id, e.source_node_id) for e in pose_graph.edges})
        for lc in potential_loop_closures:
            src_node = node_of.get(lc["source"])
            tgt_node = node_of.get(lc["target"])
            if (src_node, tgt_node) in surviving_edges:
                unanchored_promoted = True
                logger.info(f"  Unanchored chain {unanchored_chain} PROMOTED to fusion eligibility via survived loop closure {lc['source']}->{lc['target']}")
                break

    logger.info(
        f"  Pose graph: {len(pose_graph.nodes)} of {len(frames)} frames in graph, "
        f"{n_before} edges ({n_odom} primary + {n_loop} loop-closure), "
        f"{len(pose_graph.edges)} survived pruning."
    )

    # Extract final camera poses
    poses: list[np.ndarray] = []
    aligned: list[o3d.geometry.PointCloud] = []

    for i in range(len(frames)):
        if i in node_of:
            p = pose_graph.nodes[node_of[i]].pose
            poses.append(p)
            aligned.append(copy.deepcopy(frames[i].pcd).transform(p))
        else:
            p = unopt_poses.get(i, np.eye(4))
            poses.append(p)
            aligned.append(copy.deepcopy(frames[i].pcd))

    # Calculate metrics
    accepted_pairs = [log for log in pair_logs if log["success"]]
    rmse_values = [log["rmse_mm"] for log in accepted_pairs if log["rmse_mm"] < 990.0]
    avg_rmse = float(np.mean(rmse_values)) if rmse_values else 0.0
    median_rmse = float(np.median(rmse_values)) if rmse_values else 0.0

    rot_jumps = [log["rotation_deg"] for log in accepted_pairs]
    trans_jumps = [log["translation_mm"] for log in accepted_pairs]
    max_rot_jump = float(np.max(rot_jumps)) if rot_jumps else 0.0
    max_trans_jump = float(np.max(trans_jumps)) if trans_jumps else 0.0

    first_rejected = None
    for i, diag in enumerate(diagnostics):
        if diag is not None and not diag[2]:
            first_rejected = i
            break

    n_accepted = sum(1 for d in diagnostics if d is not None and d[2])
    n_rejected = len(frames) - n_accepted

    tier_counts = {
        "landmark": sum(1 for i, m in best_methods.items() if m == "landmark" and diagnostics[i] is not None and diagnostics[i][2]),
        "colored_icp": sum(1 for i, m in best_methods.items() if m in ("colored_icp", "rgbd_odometry_colored_icp") and diagnostics[i] is not None and diagnostics[i][2]),
        "fpfh_fallback": sum(1 for i, m in best_methods.items() if m == "fpfh_fallback" and diagnostics[i] is not None and diagnostics[i][2]),
        "rejected": n_rejected,
    }
    tier_summary_str = f"{tier_counts['landmark']} landmark, {tier_counts['colored_icp']} colored_icp, {tier_counts['fpfh_fallback']} fpfh_fallback, {tier_counts['rejected']} rejected"

    # Compute fusion eligibility for accepted frames
    fusion_eligible_map: dict[int, bool] = {}
    fusion_reasons_map: dict[int, str] = {}
    min_fit_fusion = getattr(cfg, "min_fitness_for_fusion", 0.60)
    max_res_fusion = getattr(cfg, "max_landmark_residual_for_fusion_mm", 5.0)
    min_face_fit = getattr(cfg, "min_face_pair_fitness_for_fusion", 0.75)
    max_face_rmse = getattr(cfg, "max_face_pair_rmse_for_fusion_mm", 4.0)
    face_y_thresh = getattr(cfg, "face_region_center_y_threshold_m", 0.0)

    for i, d in enumerate(diagnostics):
        if i == 0:
            fusion_eligible_map[0] = True
            continue
        if d is not None and d[2]:
            st = frame_states.get(i, "DIRECT_ACCEPTED")
            if st == "PROMOTED_BRIDGE":
                fusion_eligible_map[i] = False
                fusion_reasons_map[i] = "bridge node excluded from TSDF fusion"
                continue
            if st == "UNANCHORED_ODOMETRY":
                if not unanchored_promoted:
                    fusion_eligible_map[i] = False
                    fusion_reasons_map[i] = "unanchored odometry chain in fusion quarantine (no loop closure)"
                    continue
                # Unanchored chain promoted via loop closure: check local reconstruction quality gate
                fit_val = d[0]
                rmse_val_mm = d[1] * 1000.0 if np.isfinite(d[1]) else 999.0
                src_is_upper = is_upper_face_frame(frames[i], face_y_thresh)
                if src_is_upper:
                    if fit_val < min_face_fit:
                        fusion_eligible_map[i] = False
                        fusion_reasons_map[i] = f"face-pair fitness {fit_val:.2f} < {min_face_fit:.2f} threshold"
                    elif rmse_val_mm > max_face_rmse:
                        fusion_eligible_map[i] = False
                        fusion_reasons_map[i] = f"face-pair rmse {rmse_val_mm:.1f}mm > {max_face_rmse:.1f}mm threshold"
                    else:
                        fusion_eligible_map[i] = True
                else:
                    if fit_val < min_fit_fusion:
                        fusion_eligible_map[i] = False
                        fusion_reasons_map[i] = f"fitness {fit_val:.2f} < {min_fit_fusion:.2f} fusion threshold"
                    else:
                        fusion_eligible_map[i] = True
                continue

            # Standard direct / terminal recovered frame
            fit_val = d[0]
            rmse_val_mm = d[1] * 1000.0 if np.isfinite(d[1]) else 999.0
            score_str = best_scores.get(i, "")
            res_val = residuals.get(i)
            if res_val is None and "residual=" in score_str:
                try:
                    res_val = float(score_str.split("residual=")[1].split("mm")[0])
                except Exception as e:
                    logger.warning(
                        f"    [fusion-fallback] Parsing landmark residual for frame {i} summary failed with "
                        f"{type(e).__name__}: {e}"
                    )

            src_is_upper = is_upper_face_frame(frames[i], face_y_thresh)
            # Find primary candidate target for frame i
            target_j = None
            for pl in pair_logs:
                if pl["source"] == i and pl["success"] and pl.get("is_primary", False):
                    target_j = pl["target"]
                    break
            if target_j is None:
                for pl in pair_logs:
                    if pl["source"] == i and pl["success"] and abs(pl["fitness"] - fit_val) < 1e-4:
                        target_j = pl["target"]
                        break

            tgt_is_upper = False
            if target_j is not None and target_j < len(frames):
                tgt_is_upper = is_upper_face_frame(frames[target_j], face_y_thresh)

            is_face_pair = (src_is_upper and tgt_is_upper)

            if is_face_pair:
                if fit_val < min_face_fit:
                    fusion_eligible_map[i] = False
                    fusion_reasons_map[i] = f"face-pair fitness {fit_val:.2f} < {min_face_fit:.2f} threshold"
                elif rmse_val_mm > max_face_rmse:
                    fusion_eligible_map[i] = False
                    fusion_reasons_map[i] = f"face-pair rmse {rmse_val_mm:.1f}mm > {max_face_rmse:.1f}mm threshold"
                else:
                    fusion_eligible_map[i] = True
            else:
                if fit_val < min_fit_fusion:
                    fusion_eligible_map[i] = False
                    fusion_reasons_map[i] = f"fitness {fit_val:.2f} < {min_fit_fusion:.2f} fusion threshold"
                elif res_val is not None and res_val > max_res_fusion:
                    fusion_eligible_map[i] = False
                    fusion_reasons_map[i] = f"landmark residual {res_val:.1f}mm > {max_res_fusion:.1f}mm fusion threshold"
                else:
                    fusion_eligible_map[i] = True
        else:
            fusion_eligible_map[i] = False

    fusion_eligible_indices = [i for i, ok in fusion_eligible_map.items() if ok]

    summary_stats = {
        "captured": len(frames),
        "accepted": n_accepted,
        "rejected": n_rejected,
        "avg_rmse_mm": avg_rmse,
        "median_rmse_mm": median_rmse,
        "max_trans_jump_mm": max_trans_jump,
        "max_rot_jump_deg": max_rot_jump,
        "first_rejected_frame": first_rejected,
        "n_nodes": len(pose_graph.nodes),
        "n_edges": len(pose_graph.edges),
        "n_loop": n_loop,
        "frame_methods": best_methods,
        "frame_scores": best_scores,
        "residuals": residuals,
        "frame_residuals": residuals,
        "rejection_reasons": rejection_reasons,
        "tier_counts": tier_counts,
        "tier_summary_str": tier_summary_str,
        "fusion_eligible": fusion_eligible_map,
        "fusion_reasons": fusion_reasons_map,
        "fusion_eligible_indices": fusion_eligible_indices,
        "n_fusion_eligible": len(fusion_eligible_indices),
        "frame_states": frame_states,
        "unanchored_chain": unanchored_chain,
        "bridge_edges": bridge_edges,
        "node_of": node_of,
    }

    return aligned, diagnostics, poses, summary_stats





