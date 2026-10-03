"""
Point-cloud preprocessing: raw sensor cloud → clean, isolated object cloud.

The ORDER of operations matters and must not be changed without careful thought:

  1. downsample_and_denoise  – reduce point count and kill sensor noise *first*,
                               so that the plane-fitting step works on clean data.
  2. remove_dominant_plane   – strip the single largest planar surface (table,
                               wall, floor) before clustering so it doesn't
                               swamp the cluster detector.
  3. extract_largest_cluster – isolate the one connected blob we care about.

Each function is stateless and operates on a single Open3D PointCloud.
Run the full pipeline via isolate_object(), which is the only function
pipeline.py needs to call — once per captured frame, before any multi-view
registration happens.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from config import PreprocessConfig, TargetConfig


def downsample_and_denoise(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
) -> o3d.geometry.PointCloud:
    """
    Voxel-downsample then remove statistical outliers.

    Parameters
    ----------
    pcd:
        Raw input point cloud (may be dense / noisy).
    cfg:
        Preprocessing configuration.

    Returns
    -------
    o3d.geometry.PointCloud
        Cleaned, downsampled point cloud.  If the cloud is empty after
        downsampling, it is returned as-is (no crash).
    """
    pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size_m)

    if not pcd.has_points():
        return pcd

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.outlier_neighbors,
        std_ratio=cfg.outlier_std_ratio,
    )
    return pcd


def remove_dominant_plane(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
    target_cfg: TargetConfig | None = None,
) -> tuple[o3d.geometry.PointCloud, list[float]]:
    """
    Segment and remove the single largest planar surface from the cloud.

    Uses RANSAC plane fitting (Open3D ``segment_plane``).  The inliers
    (table / wall / floor) are discarded; the remainder is returned.

    If target_cfg is provided and the detected plane's spatial extent is
    within the plausible target object dimensions, the plane is preserved
    as an object face rather than deleted.

    Parameters
    ----------
    pcd:
        Input point cloud, already downsampled and denoised.
    cfg:
        Preprocessing configuration.
    target_cfg:
        Optional target configuration specifying expected object bounds.

    Returns
    -------
    object_pcd:
        Cloud with the dominant plane removed (or preserved if part of the object).
    plane_model:
        [a, b, c, d] coefficients of the fitted plane (ax+by+cz+d=0).
        Returns [0, 0, 0, 0] when the plane fit was skipped.
    """
    # segment_plane requires at least plane_ransac_n points.
    if len(pcd.points) < cfg.plane_ransac_n:
        return pcd, [0.0, 0.0, 0.0, 0.0]

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=cfg.plane_dist_threshold_m,
        ransac_n=cfg.plane_ransac_n,
        num_iterations=cfg.plane_num_iterations,
    )

    if not inliers:
        return pcd, list(plane_model)

    if target_cfg is not None and target_cfg.expected_max_size_m > 0:
        inlier_pcd = pcd.select_by_index(inliers)
        extent = inlier_pcd.get_axis_aligned_bounding_box().get_extent()
        max_dim = float(np.max(extent))
        # If the detected plane fits inside the expected target bounds, it is
        # a face of the target object — preserve it to prevent corrupting object geometry.
        if max_dim <= target_cfg.expected_max_size_m:
            return pcd, [0.0, 0.0, 0.0, 0.0]

    object_pcd = pcd.select_by_index(inliers, invert=True)
    return object_pcd, list(plane_model)


def extract_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
) -> o3d.geometry.PointCloud:
    """
    Run DBSCAN and return only the largest cluster.

    If DBSCAN finds nothing (all labels are -1, or the cloud is empty),
    the input cloud is returned unchanged so downstream stages still have
    something to work with.

    Parameters
    ----------
    pcd:
        Plane-removed point cloud.
    cfg:
        Preprocessing configuration.

    Returns
    -------
    o3d.geometry.PointCloud
        The single largest connected cluster, or the original cloud if
        no clusters were found.
    """
    if not pcd.has_points():
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(
            eps=cfg.cluster_eps_m,
            min_points=cfg.cluster_min_points,
            print_progress=False,
        )
    )

    # Labels == -1 are noise; ignore them when looking for the largest cluster.
    valid_mask = labels >= 0
    if not valid_mask.any():
        # No cluster found — return the cloud as-is rather than an empty cloud.
        return pcd

    # np.bincount only accepts non-negative integers; shift to valid labels only.
    valid_labels = labels[valid_mask]
    largest_label = int(np.bincount(valid_labels).argmax())

    indices = np.where(labels == largest_label)[0].tolist()
    return pcd.select_by_index(indices)


def extract_plausible_cluster(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
    target_cfg: TargetConfig | None = None,
    last_centroid: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """
    Among DBSCAN clusters, picks the one that's both reasonably large
    AND a plausible physical size for the target object — not just
    whichever cluster has the most points. This prevents large
    background/clutter blobs (which often have MORE points than the
    actual object in a cluttered room) from being mistaken for it.

    If last_centroid is provided, prefers the size-plausible cluster whose
    centroid is closest to last_centroid.
    """
    if not pcd.has_points():
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(
            eps=cfg.cluster_eps_m,
            min_points=cfg.cluster_min_points,
            print_progress=False,
        )
    )
    if labels.size == 0 or labels.max() < 0:
        return pcd

    min_size = target_cfg.expected_min_size_m if target_cfg else 0.0
    max_size = target_cfg.expected_max_size_m if target_cfg else float("inf")

    candidates: list[tuple[np.ndarray, o3d.geometry.PointCloud]] = []
    for label in np.unique(labels[labels >= 0]):
        idx = np.where(labels == label)[0]
        cluster = pcd.select_by_index(idx)
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        largest_dim = float(np.max(extent))
        # Implausible size -> heavily penalize regardless of point count,
        # instead of ever letting it win purely on point count
        if largest_dim < min_size or largest_dim > max_size:
            continue
        candidates.append((idx, cluster))

    if not candidates:
        # Nothing plausible found — return empty rather than silently
        # picking an implausible cluster; the caller will flag this
        return o3d.geometry.PointCloud()

    if last_centroid is not None:
        # Prefer the plausible cluster closest to last_centroid
        best_idx, _ = min(
            candidates,
            key=lambda c: (
                np.linalg.norm(np.asarray(c[1].points).mean(axis=0) - last_centroid),
                -len(c[0]),
            ),
        )
        return pcd.select_by_index(best_idx)

    # If last_centroid is None, prefer the largest plausible cluster
    best_idx, _ = max(candidates, key=lambda c: len(c[0]))
    return pcd.select_by_index(best_idx)


def isolate_object(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
    target_cfg: TargetConfig | None = None,
    last_centroid: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """
    Full per-frame preprocessing pipeline.

    Runs, in order:
      1. :func:`downsample_and_denoise`
      2. :func:`remove_dominant_plane`
      3. :func:`extract_plausible_cluster` (if target_cfg provided) or :func:`extract_largest_cluster`

    Parameters
    ----------
    pcd:
        Raw point cloud from the depth sensor (one frame).
    cfg:
        Preprocessing configuration.
    target_cfg:
        Optional Target configuration specifying plausible size & point limits.
    last_centroid:
        Optional (3,) centroid from previous frame(s) to track the object position.

    Returns
    -------
    o3d.geometry.PointCloud
        Clean, isolated object cloud ready for registration.
    """
    pcd = downsample_and_denoise(pcd, cfg)
    if target_cfg is not None:
        # First attempt: Try direct plausible clustering to preserve all object faces intact
        obj = extract_plausible_cluster(pcd, cfg, target_cfg, last_centroid=last_centroid)
        if obj.has_points():
            return obj
        # Fallback if connected to a large supporting surface: strip dominant plane and retry
        no_plane, _ = remove_dominant_plane(pcd, cfg, target_cfg=target_cfg)
        obj = extract_plausible_cluster(no_plane, cfg, target_cfg, last_centroid=last_centroid)
        return obj
    else:
        no_plane, _ = remove_dominant_plane(pcd, cfg)
        obj = extract_largest_cluster(no_plane, cfg)
        return obj


def reject_positional_outliers(
    frames: list[o3d.geometry.PointCloud],
    max_dev_m: float,
) -> tuple[list[int], list[tuple[int, float]]]:
    """
    Drop frames whose object centroid is far from the median centroid.

    A size-plausibility check cannot tell a background blob from the object
    when both are object-sized.  But if the camera is roughly fixed on the
    object, the object's centroid should stay put from frame to frame while a
    wrongly-picked background cluster jumps to a very different place.

    Returns
    -------
    keep:      indices of frames to keep
    rejected:  list of (index, distance_from_median_m) for dropped frames
    """
    if len(frames) < 3:
        return list(range(len(frames))), []
    cents = np.array([np.asarray(f.points).mean(axis=0) for f in frames])
    median = np.median(cents, axis=0)
    dists = np.linalg.norm(cents - median, axis=1)
    keep = [i for i, d in enumerate(dists) if d <= max_dev_m]
    rejected = [(i, float(dists[i])) for i in range(len(frames)) if dists[i] > max_dev_m]
    return keep, rejected