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
) -> tuple[np.ndarray, float]:
    """
    Kabsch algorithm: finds the rigid transform (R, t) that best maps
    src_pts onto dst_pts in a least-squares sense.

    Needs at least 3 non-collinear corresponding points to be well-defined;
    works best with 4 or more (see LANDMARK_INDICES in landmarks.py, which
    provides up to 8).

    Returns
    -------
    T:
        4x4 homogeneous transform mapping src_pts onto dst_pts.
    residual_mm:
        Mean per-point distance (in mm) between dst_pts and src_pts after
        applying T. Use this as a quality gate — a high residual means the
        landmark correspondences themselves were unreliable (e.g. a bad
        detection on one of the two frames), and the caller should fall
        back to a different registration method rather than trust this
        transform.
    """
    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)
    H = (src_pts - src_c).T @ (dst_pts - dst_c)
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
    residual_mm = float(np.linalg.norm(recovered - dst_pts, axis=1).mean() * 1000)
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
) -> tuple[np.ndarray, float, float, str]:
    """
    Register one frame pair using facial landmarks as the initial guess for
    ICP, instead of FPFH+RANSAC global registration. Falls back to the
    existing register_frame_pair() (unchanged FPFH+RANSAC+ICP) whenever
    landmarks aren't usable for this particular pair — e.g. the back of the
    head or an extreme profile won't have a detectable face, and that's
    expected, not a bug.

    Returns
    -------
    (transform, fitness, rmse, method) where method is "landmark" or
    "fpfh_fallback", for logging/diagnostics.
    """
    from landmarks import detect_face_landmarks_3d

    src_lm = detect_face_landmarks_3d(source_color, source_depth, intrinsics_o3d)
    dst_lm = detect_face_landmarks_3d(target_color, target_depth, intrinsics_o3d)

    if src_lm is not None and dst_lm is not None:
        common = sorted(set(src_lm) & set(dst_lm))
        if len(common) >= 4:
            src_pts = np.array([src_lm[k] for k in common])
            dst_pts = np.array([dst_lm[k] for k in common])
            init_T, residual_mm = rigid_transform_from_correspondences(src_pts, dst_pts)
            if residual_mm <= residual_accept_mm:
                icp_result = _icp_refine(source_pcd, target_pcd, init_T, cfg)
                return (
                    icp_result.transformation,
                    float(icp_result.fitness),
                    float(icp_result.inlier_rmse),
                    "landmark",
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
    return transform, fitness, rmse, "fpfh_fallback"


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
      and picks the candidate alignment with the highest fitness.
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

    CONVENTION (this is what the previous version got wrong):
    ``register_frame_pair(frames[i], frames[j])`` returns T that maps frame i
    INTO frame j's coordinates.  Open3D's ``PoseGraphEdge(source, target, T)``
    expects exactly that, so the edge is declared (i -> j), and node poses
    (frame -> frame-0 coordinates) compose as ``pose_i = pose_j @ T``.

    GATING: a new frame is registered against the last few *accepted* frames.
    If even the best of those registrations has fitness below
    ``cfg.min_accept_fitness`` the frame is REJECTED: it gets no node and no
    edges, so it cannot corrupt the graph.  The best candidate becomes a
    certain edge; other good candidates become "uncertain" (loop-closure)
    edges that the optimiser may down-weight.

    Returns
    -------
    (pose_graph, node_of, diagnostics, n_odometry_edges, n_loop_edges)
      node_of: dict frame_index -> pose-graph node id (accepted frames only)
      diagnostics: list of (fitness, rmse, accepted) per frame
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

    Diagnostics carry the REAL best-candidate fitness/rmse per frame and
    ``accepted=False`` for frames that could not be registered reliably, so
    the pipeline's REJECTED/exclude-from-fusion logic actually works.
    Rejected frames are returned un-transformed (for inspection only).
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


def register_rgbd_pair(
    source_frame: RGBDFrame,
    target_frame: RGBDFrame,
    cfg: RegistrationConfig,
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
            "FAILED",
            0.0,
            0.0,
            "Insufficient points in source or target frame (< 20 points)",
        )

    src_pcd = copy.deepcopy(source_frame.pcd)
    tgt_pcd = copy.deepcopy(target_frame.pcd)

    # 1. Landmark-guided registration
    if (
        hasattr(source_frame, "color_bgr") and source_frame.color_bgr is not None
        and hasattr(source_frame, "depth_m") and source_frame.depth_m is not None
        and hasattr(target_frame, "color_bgr") and target_frame.color_bgr is not None
        and hasattr(target_frame, "depth_m") and target_frame.depth_m is not None
        and hasattr(source_frame, "intrinsics") and source_frame.intrinsics is not None
    ):
        try:
            lm_T, lm_fit, lm_rmse, lm_method = register_frame_pair_landmark_guided(
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
                lm_rot = compute_rotation_deg(lm_T)
                lm_trans = compute_translation_mm(lm_T)
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
                        "Landmark-guided registration PASS",
                    )
        except Exception:
            pass

    # 2. RGB-D Odometry initial pose estimation
    init_trans = np.eye(4)
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
            np.eye(4),
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
        colored_res = registration_colored_icp(
            src_pcd,
            tgt_pcd,
            cfg.colored_icp_max_dist_m,
            init_trans,
            TransformationEstimationForColoredICP(lambda_geometric=cfg.colored_icp_lambda_geom),
            ICPConvergenceCriteria(max_iteration=cfg.colored_icp_max_iterations),
        )
        T_colored = colored_res.transformation
        fit_colored = float(colored_res.fitness)
        rmse_colored = float(colored_res.inlier_rmse)
        rot_colored = compute_rotation_deg(T_colored)
        trans_colored = compute_translation_mm(T_colored)

        # Quality gating
        issues = []
        if fit_colored < cfg.min_colored_icp_fitness:
            issues.append(f"fitness {fit_colored:.3f} < {cfg.min_colored_icp_fitness:.2f}")
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
                "rgbd_odometry_colored_icp",
                rot_colored,
                trans_colored,
                f"Odometry: {odo_status} -> Colored ICP PASS",
            )
        colored_status = "FAIL: " + ", ".join(issues)
    except Exception as exc:
        colored_status = f"ERROR ({exc})"
        T_colored = init_trans
        fit_colored = 0.0
        rmse_colored = float("inf")
        rot_colored = 0.0
        trans_colored = 0.0

    # 4. Geometric Fallback (FPFH + RANSAC -> Point-to-Plane ICP)
    try:
        fb_T, fb_fit, fb_rmse = register_frame_pair(src_pcd, tgt_pcd, cfg)
        fb_rot = compute_rotation_deg(fb_T)
        fb_trans = compute_translation_mm(fb_T)

        if (
            fb_fit >= cfg.min_accept_fitness
            and fb_rmse <= cfg.max_colored_icp_rmse_m * 1.5
            and fb_rot <= cfg.max_rotation_deg_per_frame
            and fb_trans <= (cfg.max_translation_m_per_frame * 1000.0)
        ):
            return (
                fb_T,
                fb_fit,
                fb_rmse,
                True,
                "fpfh_fallback",
                fb_rot,
                fb_trans,
                f"Colored ICP failed ({colored_status}); recovered via FPFH+RANSAC",
            )
    except Exception as exc:
        pass

    fail_reason = f"Colored ICP failed ({colored_status}); Fallback also failed"
    return (
        T_colored,
        fit_colored,
        rmse_colored,
        False,
        "FAILED",
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
    (pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses)
    """
    n = len(frames)
    pose_graph = PoseGraph()
    pose_graph.nodes.append(PoseGraphNode(np.eye(4)))
    node_of: dict[int, int] = {0: 0}
    poses: dict[int, np.ndarray] = {0: np.eye(4)}
    accepted_idx: list[int] = [0]
    diagnostics: list[tuple[float, float, bool] | None] = [(1.0, 0.0, True)] + [None] * (n - 1)
    pair_logs: list[dict] = []
    best_methods: dict[int, str] = {0: "origin_reference"}
    n_odom = 0
    n_loop = 0

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

        for j in target_indices:
            res = register_rgbd_pair(frames[i], frames[j], cfg)
            T, fit, rmse, accepted, method, rot, trans, reason = res

            pair_logs.append({
                "source": i,
                "target": j,
                "success": accepted,
                "method": method,
                "fitness": fit,
                "rmse_mm": rmse * 1000.0 if np.isfinite(rmse) else 999.0,
                "rotation_deg": rot,
                "translation_mm": trans,
                "reason": reason,
            })

            # Print pair log immediately
            status_str = "ACCEPTED" if accepted else f"REJECTED ({reason})"
            print(f"    pair {i:02d} -> {j:02d}: fitness={fit:.2f} rmse={rmse * 1000.0:.1f}mm method={method} [{status_str}]")

            if accepted:
                candidates.append((j, T, fit, rmse, rot, trans, method))

        if not candidates:
            # All candidates in the window failed
            best_prev = pair_logs[-1] if pair_logs else {}
            diagnostics[i] = (best_prev.get("fitness", 0.0), best_prev.get("rmse_mm", float("inf")) / 1000.0, False)
            best_methods[i] = "FAILED"
            print(f"  [!] Frame {i:02d} could not be registered to any recent accepted frame -- REJECTED.")
            continue

        # Choose best candidate (highest inlier fitness)
        j_best, T_best, fit_best, rmse_best, rot_best, trans_best, method_best = max(
            candidates, key=lambda c: c[2]
        )
        best_methods[i] = method_best

        pose_i = poses[j_best] @ T_best
        node_id = len(pose_graph.nodes)
        pose_graph.nodes.append(PoseGraphNode(pose_i))
        node_of[i] = node_id
        poses[i] = pose_i

        # Add edges: best candidate is certain (odometry); other candidates in window are uncertain (loop closures)
        for j, T, fit, rmse, rot, trans, method in candidates:
            info = get_information_matrix(frames[i].pcd, frames[j].pcd, T, cfg)
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

    return pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, poses, best_methods


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
        }

    print("\n  Running Human RGB-D Pairwise Registration with Landmark Guidance, Colored ICP & Bounded Search...")
    pose_graph, node_of, diagnostics, n_odom, n_loop, pair_logs, unopt_poses, best_methods = build_pose_graph_rgbd(frames, cfg)

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
    }

    return aligned, diagnostics, poses, summary_stats


