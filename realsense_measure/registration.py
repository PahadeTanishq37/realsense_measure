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
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
    get_information_matrix_from_point_clouds,
    global_optimization,
    registration_icp,
    registration_ransac_based_on_feature_matching,
)

from config import RegistrationConfig


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

