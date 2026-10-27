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

import logging

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import open3d as o3d

from camera.frame import create_pcd_from_rgbd
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

    is_human = target_cfg is not None and getattr(target_cfg, "name", None) in ("head", "body")
    dbscan_eps = max(cfg.cluster_eps_m, 0.035) if is_human else cfg.cluster_eps_m
    dbscan_min_points = min(cfg.cluster_min_points, 30) if is_human else cfg.cluster_min_points

    labels = np.array(
        pcd.cluster_dbscan(
            eps=dbscan_eps,
            min_points=dbscan_min_points,
            print_progress=False,
        )
    )
    if labels.size == 0 or labels.max() < 0:
        return pcd

    unique_labels = np.unique(labels[labels >= 0])
    cluster_sizes = [int(np.sum(labels == lbl)) for lbl in unique_labels]
    print(f"[preprocessing] found {len(unique_labels)} clusters (sizes: {cluster_sizes})")

    min_size = target_cfg.expected_min_size_m if target_cfg else 0.0
    max_size = target_cfg.expected_max_size_m if target_cfg else float("inf")

    candidates: list[tuple[np.ndarray, o3d.geometry.PointCloud]] = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cluster = pcd.select_by_index(idx)
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        largest_dim = float(np.max(extent))
        # Implausible size -> heavily penalize regardless of point count,
        # instead of ever letting it win purely on point count
        if largest_dim < min_size or largest_dim > max_size:
            continue
        candidates.append((idx, cluster))

    if not candidates and not is_human:
        # Nothing plausible found for box/general target
        return o3d.geometry.PointCloud()

    # Special handling for "head" and "body" targets:
    # When jaw/chin shadows split the subject into separate head and torso/neck clusters,
    # the head is anatomically the superior/upper structure (smaller Y in camera frame:
    # +X right, +Y down, +Z forward). We identify the primary cranial/head cluster, and then
    # merge adjacent clusters (neck, chin, shoulders) within anatomical proximity to prevent gaps.
    if is_human:
        if candidates:
            if getattr(target_cfg, "name", None) == "head":
                if len(candidates) == 1:
                    best_idx = candidates[0][0]
                else:
                    max_pts = max(len(c[0]) for c in candidates)
                    substantive_thresh = max(int(0.20 * max_pts), getattr(target_cfg, "min_object_points", 150))
                    substantive = [c for c in candidates if len(c[0]) >= substantive_thresh]
                    if not substantive:
                        substantive = candidates

                    cluster_y = [float(np.asarray(c[1].points).mean(axis=0)[1]) for c in substantive]
                    y_top = min(cluster_y)

                    vertical_band_m = 0.12
                    upper_candidates = [
                        candidates[i] for i in range(len(candidates))
                        if float(np.asarray(candidates[i][1].points).mean(axis=0)[1]) <= y_top + vertical_band_m
                    ]
                    if not upper_candidates:
                        upper_candidates = candidates

                    if last_centroid is not None:
                        def _head_score(cand):
                            idx, cluster = cand
                            pts = np.asarray(cluster.points)
                            c = pts.mean(axis=0)
                            dist_xz = float(np.linalg.norm(c[[0, 2]] - last_centroid[[0, 2]]))
                            return (len(idx), -dist_xz)

                        best_idx, _ = max(upper_candidates, key=_head_score)
                    else:
                        best_idx, _ = max(upper_candidates, key=lambda c: len(c[0]))
            else:
                # Body target
                if last_centroid is not None:
                    best_idx, _ = min(
                        candidates,
                        key=lambda c: (
                            np.linalg.norm(np.asarray(c[1].points).mean(axis=0) - last_centroid),
                            -len(c[0]),
                        ),
                    )
                else:
                    best_idx, _ = max(candidates, key=lambda c: len(c[0]))
        else:
            largest_label = int(np.bincount(labels[labels >= 0]).argmax())
            best_idx = np.where(labels == largest_label)[0]

        # For body target: merge adjacent / nearby clusters within anatomical distance of main centroid
        if getattr(target_cfg, "name", None) == "body":
            main_cluster = pcd.select_by_index(best_idx)
            main_centroid = np.asarray(main_cluster.points).mean(axis=0)
            merged_indices = [best_idx]
            proximity_limit_m = 0.35
            for label in unique_labels:
                other_idx = np.where(labels == label)[0]
                if np.array_equal(other_idx, best_idx):
                    continue
                other_cluster = pcd.select_by_index(other_idx)
                other_centroid = np.asarray(other_cluster.points).mean(axis=0)
                dist_to_main = float(np.linalg.norm(other_centroid - main_centroid))

                if dist_to_main <= proximity_limit_m:
                    merged_indices.append(other_idx)
                    print(
                        f"[preprocessing] merged adjacent body cluster "
                        f"(size: {len(other_idx)}, dist: {dist_to_main:.2f}m) into main body cluster"
                    )

            final_indices = np.concatenate(merged_indices)
            return pcd.select_by_index(final_indices)

        return pcd.select_by_index(best_idx)

    # General / box target handling (unchanged):
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



def isolate_with_segmentation(
    frame: "RGBDFrame",
    cfg: PreprocessConfig,
    target_cfg: "TargetConfig | None" = None,
    last_centroid: "np.ndarray | None" = None,
) -> "tuple[o3d.geometry.PointCloud, str]":
    """
    Segmentation-guided Stage 2 isolation for human frames.

    Algorithm
    ---------
    1. Run MediaPipe selfie segmenter on color_bgr to obtain a binary person mask.
    2. Zero out depth_m wherever the mask is False (never invent depth data).
    3. Rebuild the point cloud from the masked depth via create_pcd_from_rgbd.
    4. Apply the full geometric pipeline (downsample+denoise, plane removal,
       plausible-cluster selection) on the masked cloud.
    5. Fragment merging: keep the cluster whose centroid is nearest to
       last_centroid (or the largest cluster when last_centroid is None)
       AND union any other clusters whose centroid is within
       cfg.seg_merge_radius_m of the primary cluster centroid.
       Never silently fall back to "largest only".
    6. If the mask is None or the result has too few points, fall back to
       the standard geometric isolate_object() for this frame and log the
       reason.

    Parameters
    ----------
    frame : RGBDFrame
        Raw captured frame (color_bgr + depth_m + intrinsics must be set).
    cfg : PreprocessConfig
        Preprocessing configuration.
    target_cfg : TargetConfig, optional
        Target configuration for plausibility checks.
    last_centroid : np.ndarray (3,), optional
        Centroid of the last accepted frame in world-camera coordinates,
        used to identify the primary cluster across fragmented views.

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        Isolated subject point cloud.
    method : str
        "segmentation" if the mask path succeeded, "geometric_fallback"
        if we fell back.
    """
    from camera.frame import RGBDFrame as _RGBDFrame
    from segmentation import get_person_mask

    min_pts = (target_cfg.min_object_points if target_cfg is not None else 200)
    cam_cfg_min = 0.40
    cam_cfg_max = 1.20

    # --- attempt segmentation path ---
    mask = get_person_mask(frame.color_bgr)

    if mask is not None:
        masked_depth = frame.depth_m.copy()
        masked_depth[~mask] = 0.0

        # Rebuild point cloud from masked depth (reuses existing helper)
        masked_pcd = create_pcd_from_rgbd(
            color_bgr=frame.color_bgr,
            depth_m=masked_depth,
            intrinsics=frame.intrinsics,
            depth_min_m=cam_cfg_min,
            depth_max_m=cam_cfg_max,
        )

        if len(masked_pcd.points) >= min_pts:
            # Apply geometric pipeline on already-masked cloud
            cleaned = downsample_and_denoise(masked_pcd, cfg)
            if not cleaned.has_points():
                pass  # fall through to geometric fallback below
            else:
                # Plane removal only if the target is not a body/head that
                # fills the full depth range (preserve subject geometry)
                if target_cfg is not None and target_cfg.expected_max_size_m is not None:
                    no_plane, _ = remove_dominant_plane(cleaned, cfg, target_cfg=target_cfg)
                else:
                    no_plane, _ = remove_dominant_plane(cleaned, cfg)

                if not no_plane.has_points():
                    no_plane = cleaned  # keep cleaned if plane removal ate everything

                # DBSCAN clustering to get candidate fragments
                labels = np.array(no_plane.cluster_dbscan(
                    eps=cfg.cluster_eps_m,
                    min_points=cfg.cluster_min_points,
                    print_progress=False,
                ))
                unique_labels = [l for l in np.unique(labels) if l >= 0]

                if unique_labels:
                    # Build list of (indices, cloud, centroid) per cluster
                    clusters = []
                    for lbl in unique_labels:
                        idx = np.where(labels == lbl)[0]
                        c   = no_plane.select_by_index(idx)
                        cen = np.asarray(c.points).mean(axis=0)
                        clusters.append((idx, c, cen))

                    if last_centroid is not None:
                        # Primary: cluster nearest to last known position
                        primary = min(
                            clusters,
                            key=lambda t: np.linalg.norm(t[2] - last_centroid),
                        )
                    else:
                        # Primary: largest cluster
                        primary = max(clusters, key=lambda t: len(t[0]))

                    primary_idx, _, primary_cen = primary

                    # Merge any satellite clusters within seg_merge_radius_m
                    merge_radius = cfg.seg_merge_radius_m
                    merged_indices = [primary_idx]
                    for idx, c, cen in clusters:
                        if np.array_equal(idx, primary_idx):
                            continue
                        dist = float(np.linalg.norm(cen - primary_cen))
                        if dist <= merge_radius:
                            merged_indices.append(idx)
                            n_pts = len(idx)
                            print(
                                f"  [seg-isolation] merging satellite cluster "
                                f"(size: {n_pts}, dist: {dist:.3f} m)"
                            )

                    final_indices = np.concatenate(merged_indices)
                    result_pcd = no_plane.select_by_index(final_indices)
                else:
                    # No DBSCAN labels -> use the whole cleaned+masked cloud
                    result_pcd = no_plane

                n_result = len(result_pcd.points)
                if n_result >= min_pts:
                    cen_str = (
                        f"{np.asarray(result_pcd.points).mean(axis=0).round(3)}"
                        if result_pcd.has_points() else "N/A"
                    )
                    print(
                        f"  [seg-isolation] frame: method=segmentation  "
                        f"pts_before={len(masked_pcd.points)}  "
                        f"pts_after={n_result}  centroid={cen_str}"
                    )
                    return result_pcd, "segmentation"

                # Too few points after segmentation path
                reason = f"only {n_result} pts after segmentation (need >= {min_pts})"
                print(
                    f"  [seg-isolation] segmentation path gave {reason}; "
                    f"falling back to geometric isolation."
                )
        else:
            reason = f"masked cloud has only {len(masked_pcd.points)} pts (need >= {min_pts})"
            print(
                f"  [seg-isolation] segmentation path gave {reason}; "
                f"falling back to geometric isolation."
            )
    else:
        reason = "person mask is None (segmentation unavailable or failed)"
        print(
            f"  [seg-isolation] {reason}; "
            f"falling back to geometric isolation."
        )

    # --- geometric fallback ---
    geo_pcd = isolate_object(frame.pcd, cfg, target_cfg, last_centroid=last_centroid)
    n_geo = len(geo_pcd.points)
    cen_str = (
        f"{np.asarray(geo_pcd.points).mean(axis=0).round(3)}"
        if geo_pcd.has_points() else "N/A"
    )
    print(
        f"  [seg-isolation] frame: method=geometric_fallback  "
        f"pts_after={n_geo}  centroid={cen_str}"
    )
    return geo_pcd, "geometric_fallback"


def isolate_subject_by_background_subtraction(
    frame: "RGBDFrame",
    background_depth: np.ndarray,
    diff_threshold_m: float = 0.05,
    depth_min_m: float = 0.35,
    depth_max_m: float = 1.20,
    morph_kernel_size: int = 7,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Isolate foreground subject against a pre-captured static background reference.

    Foreground mask:
      - Valid depth in current frame (depth_min_m <= depth <= depth_max_m)
      - AND at least diff_threshold_m (default 5 cm) closer than background_depth,
        OR valid where background reference had none.
      - Cleaned with 2D morphological close.
      - Keeps largest 2D connected component.
      - Builds point cloud from masked depth only (NO 3D DBSCAN).

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        Isolated subject point cloud.
    fg_mask : np.ndarray
        2D boolean foreground mask (H x W).
    """
    from camera.frame import create_pcd_from_rgbd

    d = frame.depth_m
    fg_candidate = (d >= depth_min_m) & (d <= depth_max_m)
    is_nearer = (background_depth <= 0.0) | np.isnan(background_depth) | (d <= (background_depth - diff_threshold_m))
    raw_fg = fg_candidate & is_nearer

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(raw_fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels > 1:
        largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        final_fg = (labels == largest_label)
    else:
        final_fg = np.zeros_like(raw_fg, dtype=bool)

    masked_depth = np.where(final_fg, d, 0.0).astype(np.float32)
    pcd = create_pcd_from_rgbd(
        color_bgr=frame.color_bgr,
        depth_m=masked_depth,
        intrinsics=frame.intrinsics,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    n_pts = len(pcd.points)
    fid = getattr(frame, "frame_id", 0)
    logger.info(f"  [subject-turns isolation] Frame {fid:02d}: foreground mask -> {n_pts} points (no 3D DBSCAN)")
    return pcd, final_fg


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
    target_name: str | None = None,
) -> tuple[list[int], list[tuple[int, float]]]:
    """
    Drop frames whose object centroid jumps away from the expected object trajectory.

    For human head/body mode:
      The subject remains static while the camera sweeps in an arc. The camera-relative
      object centroid shifts legitimately across the orbit. An EMA-smoothed running centroid
      tolerates gradual arc motion and only rejects sudden jumps (> 1.5x step and > max_dev_m).

    For box/object mode:
      Compares each frame's centroid against the global median position (unchanged).

    Returns
    -------
    keep:      indices of frames to keep
    rejected:  list of (index, distance_from_reference_m) for dropped frames
    """
    if len(frames) < 3:
        return list(range(len(frames))), []

    cents = np.array([np.asarray(f.points).mean(axis=0) for f in frames])

    # Human head/body orbit tracking
    if target_name in ("head", "body"):
        keep: list[int] = [0]
        rejected: list[tuple[int, float]] = []

        running_centroid = cents[0].copy()
        alpha = 0.35          # EMA smoothing factor for orbital arc
        typical_step = 0.08   # baseline nominal step delta (80 mm)

        print(f"  [positional-filter] frame #00: reference origin centroid -- KEPT")
        for i in range(1, len(frames)):
            c_i = cents[i]
            dev = float(np.linalg.norm(c_i - running_centroid))
            last_kept_c = cents[keep[-1]]
            step_dist = float(np.linalg.norm(c_i - last_kept_c))

            # Dynamic threshold tolerating orbital sweep:
            # Tolerates gradual arc sweep, only flags sudden jumps exceeding 1.5x typical step / threshold
            allowed_threshold = max(max_dev_m * 1.5, 1.5 * typical_step, 0.25)

            if dev <= allowed_threshold:
                keep.append(i)
                typical_step = 0.7 * typical_step + 0.3 * step_dist
                running_centroid = alpha * c_i + (1.0 - alpha) * running_centroid
                print(f"  [positional-filter] frame #{i:02d}: centroid deviation {dev:.2f}m <= {allowed_threshold:.2f}m (gradual arc) -- KEPT")
            else:
                rejected.append((i, dev))
                print(f"  [positional-filter] frame #{i:02d}: sudden centroid jump {dev:.2f}m > {allowed_threshold:.2f}m -- DISCARDED")

        return keep, rejected

    # Box / standard mode (exact original global median comparison):
    median = np.median(cents, axis=0)
    dists = np.linalg.norm(cents - median, axis=1)
    keep = [i for i, d in enumerate(dists) if d <= max_dev_m]
    rejected = [(i, float(dists[i])) for i in range(len(frames)) if dists[i] > max_dev_m]
    return keep, rejected