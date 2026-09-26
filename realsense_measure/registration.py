"""
Multi-view point-cloud registration for hand-repositioned object captures.

Context
-------
Frames are captured from a single FIXED camera while the operator rotates or
repositions the object by hand between shots.  There is no known relative
transform between frames, and inter-frame rotations can be large and
arbitrary — so plain ICP with an identity initial guess will fail.

Two-stage registration is used for every frame pair:

  1. **Global registration** (FPFH + RANSAC) — produces a coarse but
     rotation-agnostic initial alignment with no initial guess required.
  2. **Point-to-plane ICP** — refines the coarse alignment to sub-millimetre
     accuracy using the RANSAC result as its starting pose.

Growing-reference design
------------------------
The *most important* design choice in this file is that frames are NOT all
registered independently back to frame 0.  Instead they are accumulated onto
a **growing reference cloud**:

  reference ← frame[0]
  for i in 1 … N:
      transform frame[i] onto reference
      append transformed frame[i] to reference
      voxel-downsample reference  (keep it manageable)

Why this matters: frame 0 is a single partial view that covers at most ~180°
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
    ICPConvergenceCriteria,
    RANSACConvergenceCriteria,
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
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
            radius=voxel_size * 2,
            max_nn=30,
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
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

    return registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
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
    Align ``source`` onto ``target`` using global registration then ICP.

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
        4×4 numpy float64 array that maps ``source`` into ``target`` space.
    fitness:
        ICP fitness score (fraction of inlier correspondences).
    rmse:
        ICP inlier RMSE in metres.
    """
    voxel_size = cfg.voxel_size_m

    source_down, source_fpfh = _prepare(source, voxel_size)
    target_down, target_fpfh = _prepare(target, voxel_size)

    global_result = _global_register(
        source_down, target_down,
        source_fpfh, target_fpfh,
        cfg, voxel_size,
    )

    icp_result = _icp_refine(source, target, global_result.transformation, cfg)

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
    earlier accumulated geometry) — whichever actually works for that
    particular frame.
    """
    best: tuple[np.ndarray, float, float] | None = None
    for candidate in candidates:
        if candidate is None or len(candidate.points) == 0:
            continue
        transform, fitness, rmse = register_frame_pair(new_frame, candidate, cfg)
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
