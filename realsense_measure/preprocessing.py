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

from config import PreprocessConfig


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
) -> tuple[o3d.geometry.PointCloud, list[float]]:
    """
    Segment and remove the single largest planar surface from the cloud.

    Uses RANSAC plane fitting (Open3D ``segment_plane``).  The inliers
    (table / wall / floor) are discarded; the remainder is returned.

    Parameters
    ----------
    pcd:
        Input point cloud, already downsampled and denoised.
    cfg:
        Preprocessing configuration.

    Returns
    -------
    object_pcd:
        Cloud with the dominant plane removed.
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


def isolate_object(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig,
) -> o3d.geometry.PointCloud:
    """
    Full per-frame preprocessing pipeline.

    Runs, in order:
      1. :func:`downsample_and_denoise`
      2. :func:`remove_dominant_plane`
      3. :func:`extract_largest_cluster`

    This is the only function ``pipeline.py`` needs to call.  It is designed
    to operate on a single frame independently, before any multi-view
    registration takes place.

    Parameters
    ----------
    pcd:
        Raw point cloud from the depth sensor (one frame).
    cfg:
        Preprocessing configuration.

    Returns
    -------
    o3d.geometry.PointCloud
        Clean, isolated object cloud ready for registration.
    """
    pcd = downsample_and_denoise(pcd, cfg)
    pcd, _ = remove_dominant_plane(pcd, cfg)
    pcd = extract_largest_cluster(pcd, cfg)
    return pcd
