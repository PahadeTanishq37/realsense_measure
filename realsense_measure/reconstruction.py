"""
Point-cloud fusion: merge all registered per-frame clouds into one clean mesh.

After registration every frame lives in the same coordinate system, so fusion
is simply concatenation followed by a final cleaning pass.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import open3d as o3d

from config import PreprocessConfig
from preprocessing import downsample_and_denoise, extract_largest_cluster



def _remove_radius_stragglers(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    nb_points: int = 2,
) -> o3d.geometry.PointCloud:
    """
    Gentle radius-based outlier removal (Task 2).

    Removes isolated single-frame straggler points that have fewer than
    ``nb_points`` neighbours within ``radius`` metres.  This is far looser
    than SOR: real sparse features (ear tips, nose bridge wisps) still survive
    because they accumulate a handful of consistent points across multiple
    accepted frames.  Only truly isolated specks — often the product of a
    single borderline-accepted frame — are dropped.

    Parameters
    ----------
    pcd:
        Input point cloud (head/body mode only; box mode must never call this).
    radius:
        Neighbourhood search radius in metres.  Recommended ≈ 2–3× the fusion
        voxel size (2 mm), so ~4–6 mm.
    nb_points:
        Minimum neighbours required to keep a point.  2 is appropriate (removes
        truly isolated single-point specks that have 0 or 1 nearby neighbour);
        lower values are more permissive, higher values remove more aggressively.

    Returns
    -------
    Cleaned point cloud with straggler specks removed.
    """
    if not pcd.has_points() or len(pcd.points) < nb_points + 1:
        return pcd
    cleaned, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cleaned


def _remove_streak_artifacts(
    pcd: o3d.geometry.PointCloud,
    linearity_ratio: float = 8.0,
    k_neighbors: int = 20,
    voxel_size: float = 0.002,
) -> o3d.geometry.PointCloud:
    """
    Linearity-aware streak artifact removal (Task 1).

    A true surface patch has a locally 2-D (planar / curved) neighbourhood.
    A streak artifact from slight frame misalignment has a locally 1-D
    (linear) neighbourhood — points strung out along the direction of the
    registration error.

    For each point, PCA eigenvalues λ₁ ≥ λ₂ ≥ λ₃ of its k-nearest
    neighbours are computed.  A neighbourhood is labelled *linear* when
    λ₁ / max(λ₂, ε) > ``linearity_ratio``.  Linear neighbourhoods that also
    sit in below-median local-density regions are flagged as streaks and
    removed.  Genuinely sparse-but-isotropic regions (ear tip, nose bridge)
    have small but roughly equal eigenvalues, so they survive.

    Parameters
    ----------
    pcd:
        Input point cloud (head/body mode only; box mode must never call this).
    linearity_ratio:
        λ₁/λ₂ threshold above which a neighbourhood is deemed linear.
        Default 8.0 means λ₁ must be 8× larger than λ₂.
    k_neighbors:
        Number of neighbours used for PCA.  15–20 is a good balance between
        stability and locality.
    voxel_size:
        Fusion voxel size in metres (used to set the KDTree search radius for
        local-density estimation).

    Returns
    -------
    Cleaned point cloud with linear streak artifacts removed.
    """
    if not pcd.has_points() or len(pcd.points) < k_neighbors + 1:
        return pcd

    pts = np.asarray(pcd.points)
    n = len(pts)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    linearity_flags = np.zeros(n, dtype=bool)
    local_counts = np.zeros(n, dtype=int)

    # PCA window radius: large enough to capture linear streaks and local surface patches
    # without reaching across to distant disconnected structures.
    # 10× voxel_size = 20 mm for 2 mm voxel.
    pca_radius = voxel_size * 10.0
    density_radius = voxel_size * 3.0
    min_pca_pts = 4  # Need at least 4 points in the neighborhood for stable 3D PCA

    for i in range(n):
        k_r_pca, idx_r_pca, _ = kdtree.search_radius_vector_3d(pts[i], pca_radius)
        k_r_den, _, _ = kdtree.search_radius_vector_3d(pts[i], density_radius)
        local_counts[i] = k_r_den - 1

        # Genuinely sparse patch: not enough points to build a stable PCA -> keep unconditionally
        if k_r_pca < min_pca_pts:
            continue

        # Use up to k_neighbors from the radius window for PCA
        neighbour_pts = pts[np.asarray(idx_r_pca[:min(k_neighbors, k_r_pca)])]
        cov = np.cov(neighbour_pts.T)
        eigvals = np.linalg.eigvalsh(cov)   # ascending: λ₃ ≤ λ₂ ≤ λ₁
        lam1 = float(eigvals[2])
        lam2 = float(eigvals[1])
        eps = 1e-12
        if lam1 / max(lam2, eps) > linearity_ratio:
            linearity_flags[i] = True

    median_density = float(np.median(local_counts))
    # Remove points that are BOTH locally linear AND below-median density
    streak_mask = linearity_flags & (local_counts < median_density)

    keep_idx = np.where(~streak_mask)[0]
    n_removed = int(np.sum(streak_mask))
    if n_removed > 0:
        print(f"  [streak-removal] Removed {n_removed} streak/artifact points "
              f"({n_removed / n * 100:.1f}% of {n} total).")
    return pcd.select_by_index(keep_idx.tolist())


def fuse_point_clouds(
    aligned_frames: list[o3d.geometry.PointCloud],
    cfg: PreprocessConfig,
    skip_final_clustering: bool = False,
    skip_outlier_removal: bool = False,
    voxel_size: float | None = None,
    reg_cfg=None,
) -> o3d.geometry.PointCloud:
    """
    Merge a list of registered, per-frame object clouds into one clean cloud.

    Steps
    -----
    1. Concatenate all frames into a single cloud.
    2. Statistical-outlier-removal via :func:`~open3d.geometry.PointCloud.remove_statistical_outlier`.
       This step is SKIPPED for human reconstructions (``skip_outlier_removal=True``)
       because global SOR penalizes single-view profile sweeps, stripping sparse
       lateral profile and ear geometry.
    3. Light voxel downsample to merge near-duplicate points from overlapping angles.
    4. (BOX only) Retain only the largest DBSCAN cluster via
       :func:`~preprocessing.extract_largest_cluster`.

       This step is SKIPPED for human reconstructions (``skip_final_clustering=True``)
       because a human body consists of multiple connected regions (face, ears,
       shoulders) that would be wrongly discarded as separate small blobs.
    5. (HUMAN/HEAD only) Gentle radius-based straggler removal then PCA-based
       linearity/streak removal — catches misalignment artifacts without removing
       sparse-but-real features like ear tips or nose bridge.

    Parameters
    ----------
    aligned_frames:
        Ordered list of point clouds, all expressed in the shared reference
        coordinate system produced by :func:`~registration.register_sequence`.
    cfg:
        Preprocessing configuration (voxel size, outlier thresholds, DBSCAN
        parameters) — the same config used during per-frame preprocessing.
    skip_final_clustering:
        When True, skip the DBSCAN largest-cluster pass.  Use for human scans
        where the body has complex topology.
    skip_outlier_removal:
        When True, skip statistical outlier removal. Use for human scans
        to avoid stripping sparse lateral/profile facial features.
    reg_cfg:
        Optional RegistrationConfig instance.  When provided, controls whether
        and how streak removal is applied (``enable_streak_removal``,
        ``streak_linearity_ratio_threshold``, ``streak_removal_radius_mult``).

    Returns
    -------
    o3d.geometry.PointCloud
        Fused, cleaned, single-component object cloud.
    """
    if not aligned_frames:
        return o3d.geometry.PointCloud()

    # Concatenate: start from a deep copy of the first frame so the caller's
    # data is never mutated, then accumulate the rest.
    fused = copy.deepcopy(aligned_frames[0])
    for frame in aligned_frames[1:]:
        fused += frame

    # 1. Statistical outlier removal to kill genuine noise/speckle from across frames
    # Bypassed for human scans to preserve fine lateral/profile geometry
    if not skip_outlier_removal and fused.has_points():
        fused, _ = fused.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )

    # 2. Light voxel downsample to merge near-duplicate points from overlapping angles
    # Human scans use 2.0 mm (0.002 m) resolution; box/general scans preserve cfg.voxel_size_m (4.0 mm).
    if voxel_size is not None:
        eff_voxel_size = voxel_size
    elif skip_outlier_removal:
        eff_voxel_size = 0.002
    else:
        eff_voxel_size = cfg.voxel_size_m

    if eff_voxel_size > 0 and fused.has_points():
        fused = fused.voxel_down_sample(voxel_size=eff_voxel_size)

    # 3. Final cluster pass: drop any small stray blobs from imperfect per-frame
    # isolation that only became visible once all views were merged together.
    # Skip this for human scans — ears, eyebrows, etc. would be discarded.
    if not skip_final_clustering and fused.has_points():
        fused = extract_largest_cluster(fused, cfg)

    # 4. Head/body-only post-fusion artifact cleanup (Task 1 & 2):
    #    Streak artifacts and single-frame ghost points that pass through when SOR is
    #    bypassed.  Box mode must never reach this block.
    if skip_outlier_removal and fused.has_points():
        enable = False
        linearity_ratio = 8.0
        radius_mult = 2.5
        if reg_cfg is not None:
            enable = getattr(reg_cfg, "enable_streak_removal", True)
            linearity_ratio = getattr(reg_cfg, "streak_linearity_ratio_threshold", 8.0)
            radius_mult = getattr(reg_cfg, "streak_removal_radius_mult", 2.5)

        if enable:
            # Task 2: Gentle radius-based straggler pass first (cheap O(n log n))
            straggler_radius = radius_mult * eff_voxel_size
            fused = _remove_radius_stragglers(fused, radius=straggler_radius, nb_points=2)

            # Task 1: PCA linearity-based streak removal (more expensive, but post-downsample)
            if fused.has_points():
                fused = _remove_streak_artifacts(
                    fused,
                    linearity_ratio=linearity_ratio,
                    k_neighbors=20,
                    voxel_size=eff_voxel_size,
                )

    return fused



def fuse_tsdf_volume(
    frames: list,
    poses: list[np.ndarray],
    cfg: PreprocessConfig,
    voxel_length: float = 0.004,
    sdf_trunc: float = 0.02,
    skip_final_clustering: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Volumetric integration of accepted RGB-D frames using Open3D TSDFVolume.

    Provides a clean, watertight surface reconstruction for human scans by
    integrating depth observations along optical rays into a voxel grid.

    Parameters
    ----------
    frames:
        List of accepted RGBDFrame objects.
    poses:
        List of 4x4 camera-to-world (frame-to-frame0) transformations.
    cfg:
        Preprocessing configuration for post-cleaning.
    voxel_length:
        Voxel resolution in metres (e.g. 0.004 = 4 mm).
    sdf_trunc:
        Truncation distance for signed distance function in metres.
    skip_final_clustering:
        When True, skip DBSCAN largest-cluster stripping after extraction.
        Use for human scans where body topology is complex.

    Returns
    -------
    o3d.geometry.PointCloud
        Integrated, cleaned point cloud with RGB color.
    """
    if not frames:
        return o3d.geometry.PointCloud()

    # Determine volume origin and extents in WORLD coordinates across all accepted frames
    all_pts = []
    for f, pose in zip(frames, poses):
        if hasattr(f, "pcd") and len(f.pcd.points) > 0:
            pts_cam = np.asarray(f.pcd.points)
            # Transform local camera points into world coordinates: p_world = pose @ p_camera
            R = pose[:3, :3]
            t = pose[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            all_pts.append(pts_world)

    if all_pts:
        pts_concat = np.vstack(all_pts)
        min_b = np.min(pts_concat, axis=0) - 0.20
        max_b = np.max(pts_concat, axis=0) + 0.20
        length = float(max(np.max(max_b - min_b), 1.2))
        origin = min_b
    else:
        origin = np.array([-0.6, -0.6, 0.2])
        length = 1.6

    # For head/body target, ensure fine 1.5 - 2.0 mm voxel resolution and tight truncation margin (roughly 4x voxel_length)
    if skip_final_clustering:
        if voxel_length >= 0.003:
            voxel_length = 0.002
        if sdf_trunc > 5.0 * voxel_length:
            sdf_trunc = 4.0 * voxel_length

    resolution = int(np.clip(length / voxel_length, 128, 1024))

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=resolution,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=origin,
    )

    integrated_count = 0
    for frame, pose in zip(frames, poses):
        if not hasattr(frame, "to_rgbd_image"):
            continue
        try:
            # Extrinsic transforms world to camera: inv(pose)
            extrinsic = np.linalg.inv(pose)

            # TASK 4: Depth pre-smoothing & isolated geometry masking for head/body mode
            depth_override = None
            if hasattr(frame, "depth_m") and frame.depth_m is not None:
                depth_clean = frame.depth_m.copy()
                if skip_final_clustering:  # Head/body mode
                    # 1. Depth pre-smoothing via morphological closing (cv2.morphologyEx)
                    # Closes sensor dropout holes and smooths noise on facial contours
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    depth_clean = cv2.morphologyEx(depth_clean, cv2.MORPH_CLOSE, kernel)

                    # 2. Fix TSDF masking: retain only isolated geometry from Stage 2 if present
                    if hasattr(frame, "pcd") and len(frame.pcd.points) > 0 and hasattr(frame, "intrinsics") and frame.intrinsics is not None:
                        h, w = depth_clean.shape[:2]
                        fx, fy = frame.intrinsics.get_focal_length()
                        cx, cy = frame.intrinsics.get_principal_point()
                        pts_cam = np.asarray(frame.pcd.points)
                        z = pts_cam[:, 2]
                        valid = z > 0.05
                        pts_valid = pts_cam[valid]
                        z_valid = z[valid]
                        u = np.round((pts_valid[:, 0] * fx / z_valid) + cx).astype(int)
                        v = np.round((pts_valid[:, 1] * fy / z_valid) + cy).astype(int)
                        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                        if np.sum(in_img) > 50:
                            obj_mask = np.zeros((h, w), dtype=np.uint8)
                            obj_mask[v[in_img], u[in_img]] = 255
                            obj_mask = cv2.dilate(obj_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
                            depth_clean[obj_mask == 0] = 0.0

                    depth_override = depth_clean

            try:
                rgbd = frame.to_rgbd_image(
                    depth_trunc=1.5,
                    convert_rgb_to_intensity=False,
                    depth_override=depth_override,
                )
            except TypeError:
                rgbd = frame.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=False)

            volume.integrate(rgbd, frame.intrinsics, extrinsic)
            integrated_count += 1
        except Exception as exc:
            print(f"  [TSDF integration warning] frame skipped: {exc}")

    if integrated_count > 0:
        fused = volume.extract_point_cloud()
        if len(fused.points) > 100:
            fused = downsample_and_denoise(fused, cfg)
            # Skip cluster stripping for human scans — would discard ears, nose etc.
            if not skip_final_clustering:
                fused = extract_largest_cluster(fused, cfg)
            return fused

    # Graceful fallback: point-cloud concatenation
    print("  [TSDF fallback] Volumetric extraction yielded sparse points; using point-cloud fusion.")
    pcds = [copy.deepcopy(f.pcd).transform(p) for f, p in zip(frames, poses) if hasattr(f, "pcd")]
    return fuse_point_clouds(
        pcds,
        cfg,
        skip_final_clustering=skip_final_clustering,
        skip_outlier_removal=skip_final_clustering,
    )

