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

import numpy as np
import open3d as o3d
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
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)

    if not src.has_normals():
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_max_dist_m * 2, max_nn=30)
        )
    if not tgt.has_normals():
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_max_dist_m * 2, max_nn=30)
        )

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
                else:
                    s_pos = s_item
                    s_vr = 1.0
                    s_var = 0.0

                if isinstance(d_item, dict):
                    d_pos = d_item["position"]
                    d_vr = d_item.get("valid_ratio", 1.0)
                    d_var = d_item.get("variance", 0.0)
                else:
                    d_pos = d_item
                    d_vr = 1.0
                    d_var = 0.0

                src_pts_list.append(s_pos)
                dst_pts_list.append(d_pos)

                # Weight: min(1.0, valid_ratio_src * valid_ratio_dst) / (1.0 + variance_src + variance_dst)
                v_src_norm = s_var / max_var_thresh if (0.0 < s_var < 0.01 and max_var_thresh > 0) else s_var
                v_dst_norm = d_var / max_var_thresh if (0.0 < d_var < 0.01 and max_var_thresh > 0) else d_var
                w = float(min(1.0, s_vr * d_vr) / (1.0 + v_src_norm + v_dst_norm))
                weights_list.append(w)

            src_pts = np.array(src_pts_list)
            dst_pts = np.array(dst_pts_list)
            weights = np.array(weights_list)

            init_T, residual_mm = rigid_transform_from_correspondences(src_pts, dst_pts, weights=weights)
            if residual_mm <= residual_accept_mm:
                # Post-seed dense-surface sanity check (single pass evaluation)
                min_dense_fit = getattr(cfg, "min_seed_dense_fitness", 0.30)
                eval_voxel = max(getattr(cfg, "voxel_size_m", 0.003) * 2.0, 0.006)
                src_down = source_pcd.voxel_down_sample(eval_voxel) if len(source_pcd.points) > 50 else source_pcd
                tgt_down = target_pcd.voxel_down_sample(eval_voxel) if len(target_pcd.points) > 50 else target_pcd

                if not tgt_down.has_normals():
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
                    print(
                        f"      [landmarks] Kabsch seed rejected: dense-surface fitness "
                        f"{seed_dense_fitness:.2f} < {min_dense_fit:.2f} threshold (residual was {residual_mm:.1f}mm) "
                        f"-- falling through to Tier 2"
                    )
                    return np.eye(4), 0.0, float("inf"), "dense_sanity_failed", float(residual_mm)

                # Tightened Colored ICP refinement specifically for landmark seed
                src_temp = copy.deepcopy(source_pcd)
                tgt_temp = copy.deepcopy(target_pcd)
                if not src_temp.has_normals():
                    src_temp.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
                    )
                if not tgt_temp.has_normals():
                    tgt_temp.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
                    )
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
                except Exception:
                    icp_result = _icp_refine(source_pcd, target_pcd, init_T, cfg)
                return (
                    icp_result.transformation,
                    float(icp_result.fitness),
                    float(icp_result.inlier_rmse),
                    "landmark",
                    float(residual_mm),
                )
            else:
                print(f"      [landmarks] Kabsch residual too high: {residual_mm:.1f}mm > {residual_accept_mm:.1f}mm")
        else:
            print(f"      [landmarks] Insufficient common landmarks: {len(common)} < 4 (src={len(src_lm)}, dst={len(dst_lm)})")
    else:
        src_n = len(src_lm) if src_lm is not None else 0
        dst_n = len(dst_lm) if dst_lm is not None else 0
        print(f"      [landmarks] Face not detected on both frames (src={src_n}, dst={dst_n})")

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
    for candidate in candidates:
        if candidate is None or len(candidate.points) < 10:
            continue
        try:
            transform, fitness, rmse = register_frame_pair(new_frame, candidate, cfg)
        except Exception:
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
            except Exception:
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
    if not frames:
        return [], []
    if len(frames) == 1:
        return [copy.deepcopy(frames[0])], [(1.0, 0.0, True)]

    pose_graph, node_of, diagnostics, n_odom, n_loop = build_pose_graph(frames, cfg)
    n_before = len(pose_graph.edges)
    if len(pose_graph.nodes) > 1:
        optimize_pose_graph(pose_graph, cfg)
    print(
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


def register_rgbd_pair(
    source_frame: RGBDFrame,
    target_frame: RGBDFrame,
    cfg: RegistrationConfig,
    initial_guess: np.ndarray | None = None,
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
    # Guard against empty / degenerate point clouds
    if (
        not hasattr(source_frame, "pcd") or len(source_frame.pcd.points) < 20
        or not hasattr(target_frame, "pcd") or len(target_frame.pcd.points) < 20
    ):
        return (
            np.eye(4),
            0.0,
            float("inf"),
            False,
            "none",
            0.0,
            0.0,
            "Insufficient points in source or target frame (< 20 points)",
        )

    src_pcd = copy.deepcopy(source_frame.pcd)
    tgt_pcd = copy.deepcopy(target_frame.pcd)
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
                    return (
                        lm_T,
                        lm_fit,
                        lm_rmse,
                        True,
                        "landmark",
                        lm_rot,
                        lm_trans,
                        f"fitness={lm_fit:.2f} (src={lm_fit_src:.2f}) residual={lm_res_mm:.1f}mm",
                    )
                # Seed downstream refinement with coarse landmark transform if not immediately accepted
                if lm_rot <= cfg.max_rotation_deg_per_frame and lm_trans <= (cfg.max_translation_m_per_frame * 1000.0):
                    init_trans = lm_T
        except Exception:
            pass

    # Check overlap before invoking RGB-D odometry or Colored ICP
    if not overlap_ok:
        print(f"[overlap-check] pair {pair_str} SKIPPED colored-icp ({overlap_skip_reason}) -- trying FPFH fallback")
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
            src_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.colored_icp_max_dist_m * 2.0, max_nn=30)
            )
        if not tgt_pcd.has_normals():
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
                return (
                    T_colored,
                    fit_colored,
                    rmse_colored,
                    True,
                    "colored_icp",
                    rot_colored,
                    trans_colored,
                    f"fitness={fit_colored:.2f} (src={fit_colored_src:.2f}) rmse={rmse_colored * 1000.0:.1f}mm",
                )
            colored_status = "FAIL: " + ", ".join(issues)
        except RuntimeError as exc:
            print(f"[colored-icp] no correspondences for pair {pair_str}, falling through to FPFH tier")
            colored_status = f"ERROR ({exc})"
            T_colored = init_trans
            fit_colored = 0.0
            rmse_colored = float("inf")
            rot_colored = 0.0
            trans_colored = 0.0
        except Exception as exc:
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
            print(f"[registration-diagnostic] pair {pair_str} FPFH fallback SUCCEEDED (region: {region_str}, bbox_center_y={src_center_y:+.3f}m)")
            return (
                fb_T,
                fb_fit,
                fb_rmse,
                True,
                "fpfh_fallback",
                fb_rot,
                fb_trans,
                f"fitness={fb_fit:.2f} (src={fb_fit_src:.2f}) rmse={fb_rmse * 1000.0:.1f}mm",
            )
    except Exception as exc:
        pass

    src_center_y = float(src_pcd.get_axis_aligned_bounding_box().get_center()[1])
    region_str = "lower (neck/shoulder)" if src_center_y > 0.0 else "upper (face/head)"
    fail_reason = f"Colored ICP failed ({colored_status}); Fallback also failed"
    print(
        f"[registration-diagnostic] pair {pair_str} REJECTED / LOW-FITNESS "
        f"(region: {region_str}, bbox_center_y={src_center_y:+.3f}m) -- {fail_reason}"
    )
    return (
        T_colored,
        fit_colored,
        rmse_colored,
        False,
        "none",
        rot_colored,
        trans_colored,
        fail_reason,
    )


def build_pose_graph_rgbd(
    frames: list[RGBDFrame],
    cfg: RegistrationConfig,
):
    """
    Build a multiway pose graph for human RGB-D frames with bounded search and quality gating.

    Returns
    -------
    (pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses, best_methods, best_scores)
    """
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
    rejection_reasons: dict[int, str] = {}
    n_odom = 0
    n_loop = 0

    last_rel_trans: np.ndarray | None = None
    frames_since_accepted: int = 0

    for i in range(1, n):
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
            init_guess = np.eye(4)
            if j == prev_accepted_idx and frames_since_accepted == 0 and last_rel_trans is not None:
                rot_p = compute_rotation_deg(last_rel_trans)
                trans_p_mm = compute_translation_mm(last_rel_trans)
                if rot_p <= cfg.max_rotation_deg_per_frame and trans_p_mm <= (cfg.max_translation_m_per_frame * 1000.0):
                    init_guess = last_rel_trans.copy()

            res = register_rgbd_pair(frames[i], frames[j], cfg, initial_guess=init_guess)
            T, fit, rmse, accepted, method, rot, trans, reason = res

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
                "reason": (
                    f"pose jump ({jump_trans_m:.2f}m/{jump_rot_deg:.1f}deg) > allowed ({effective_translation_bound:.2f}m/{effective_rotation_bound:.1f}deg)"
                    if exceeds_jump else reason
                ),
                "jump_trans_m": jump_trans_m,
                "jump_rot_deg": jump_rot_deg,
            })

            # Print pair log with exact tier and score
            if accepted and not exceeds_jump:
                print(f"    pair {i:02d} -> {j:02d}: tier={method} {reason} ACCEPTED")
                candidates.append((j, T, fit, rmse, rot, trans, method, reason))
            elif accepted and exceeds_jump:
                print(
                    f"    pair {i:02d} -> {j:02d}: tier={method} {reason} "
                    f"REJECTED -- pose jump ({jump_trans_m:.2f}m/{jump_rot_deg:.1f}deg) exceeds plausible motion "
                    f"(bound {effective_translation_bound:.2f}m/{effective_rotation_bound:.1f}deg, scaled for {frames_since_accepted} "
                    f"frames since last accept)"
                )
            else:
                print(f"    pair {i:02d} -> {j:02d}: tier=none REJECTED ({reason})")

        if not candidates:
            # All candidates in the window failed: increment counter of unaccepted frames
            frames_since_accepted += 1
            last_rel_trans = None
            if misleading_candidates:
                mis = max(misleading_candidates, key=lambda m: m["fitness"])
                diagnostics[i] = (mis["fitness"], mis["rmse"], False)
                best_methods[i] = mis["method"]
                score_str = mis["score_str"]
                best_scores[i] = score_str
                n_skip = mis["frames_since_accepted"]
                rejection_msg = (
                    f"REJECTED -- pose jump ({mis['jump_trans_m']:.2f}m/{mis['jump_rot_deg']:.1f}deg) exceeds plausible motion "
                    f"(bound/allowed: {mis['allowed_trans_m']:.2f}m/{mis['allowed_rot_deg']:.1f}deg, scaled for {n_skip} "
                    f"frames since last accept)"
                )
                rejection_reasons[i] = rejection_msg
                print(
                    f"    frame {i:02d}: tier={mis['method']} {score_str}\n"
                    f"              {rejection_msg}"
                )
            else:
                best_prev = pair_logs[-1] if pair_logs else {}
                diagnostics[i] = (best_prev.get("fitness", 0.0), best_prev.get("rmse_mm", float("inf")) / 1000.0, False)
                best_methods[i] = "none"
                best_scores[i] = ""
                print(f"    frame {i:02d}: tier=none REJECTED")
            continue

        # Prioritize sequential candidate (prev_accepted_idx) if it succeeded and meets quality criteria
        seq_candidates = [c for c in candidates if c[0] == prev_accepted_idx]
        if seq_candidates:
            j_best, T_best, fit_best, rmse_best, rot_best, trans_best, method_best, reason_best = seq_candidates[0]
            last_rel_trans = T_best.copy() if frames_since_accepted == 0 else None
        else:
            j_best, T_best, fit_best, rmse_best, rot_best, trans_best, method_best, reason_best = max(
                candidates, key=lambda c: c[2]
            )
            last_rel_trans = None

        frames_since_accepted = 0
        best_methods[i] = method_best
        best_scores[i] = reason_best

        # Fusion eligibility check (secondary stricter gate for Stage 4 fusion)
        min_fit_fusion = getattr(cfg, "min_fitness_for_fusion", 0.60)
        max_res_fusion = getattr(cfg, "max_landmark_residual_for_fusion_mm", 5.0)
        lm_res_val = None
        if "residual=" in reason_best:
            try:
                lm_res_val = float(reason_best.split("residual=")[1].split("mm")[0])
            except Exception:
                pass

        if fit_best < min_fit_fusion:
            print(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (fitness {fit_best:.2f} < {min_fit_fusion:.2f} fusion threshold)")
        elif lm_res_val is not None and lm_res_val > max_res_fusion:
            print(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion (landmark residual {lm_res_val:.1f}mm > {max_res_fusion:.1f}mm fusion threshold)")
        else:
            print(f"    frame {i:02d}: tier={method_best} {reason_best} ACCEPTED")

        pose_i = poses[j_best] @ T_best
        node_id = len(pose_graph.nodes)
        pose_graph.nodes.append(PoseGraphNode(pose_i))
        node_of[i] = node_id
        poses[i] = pose_i

        # Add edges: primary sequential candidate is certain (odometry);
        # other non-consecutive candidates are uncertain (loop closures) and must satisfy min_loop_closure_fitness.
        min_loop_fit = getattr(cfg, "min_loop_closure_fitness", 0.65)
        for j, T, fit, rmse, rot, trans, method, reason in candidates:
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

        accepted_idx.append(i)
        diagnostics[i] = (fit_best, rmse_best, True)

    return pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses, best_methods, best_scores, rejection_reasons


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
            "rejection_reasons": {},
            "tier_counts": {"landmark": 0, "colored_icp": 0, "fpfh_fallback": 0, "rejected": 0},
            "tier_summary_str": "0 landmark, 0 colored_icp, 0 fpfh_fallback, 0 rejected",
        }

    print("\n  Running Human RGB-D Pairwise Registration with Landmark Guidance, Colored ICP & Bounded Search...")
    pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, unopt_poses, best_methods, best_scores, rejection_reasons = build_pose_graph_rgbd(frames, cfg)

    n_before = len(pose_graph.edges)
    if len(pose_graph.nodes) > 1:
        optimize_pose_graph(pose_graph, cfg)

    print(
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

    for i, d in enumerate(diagnostics):
        if i == 0:
            fusion_eligible_map[0] = True
            continue
        if d is not None and d[2]:
            fit_val = d[0]
            score_str = best_scores.get(i, "")
            res_val = None
            if "residual=" in score_str:
                try:
                    res_val = float(score_str.split("residual=")[1].split("mm")[0])
                except Exception:
                    pass
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
        "rejection_reasons": rejection_reasons,
        "tier_counts": tier_counts,
        "tier_summary_str": tier_summary_str,
        "fusion_eligible": fusion_eligible_map,
        "fusion_reasons": fusion_reasons_map,
        "fusion_eligible_indices": fusion_eligible_indices,
        "n_fusion_eligible": len(fusion_eligible_indices),
    }

    return aligned, diagnostics, poses, summary_stats





