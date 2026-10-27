"""
Point-cloud fusion: merge all registered per-frame clouds into one clean mesh.

After registration every frame lives in the same coordinate system, so fusion
is simply concatenation followed by a final cleaning pass.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d

from config import HeadRefinementConfig, PreprocessConfig
from preprocessing import downsample_and_denoise, extract_largest_cluster

logger = logging.getLogger(__name__)



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


# ---------------------------------------------------------------------------
# Opt-in Head Reconstruction & Model Refinement (--refine)
# ---------------------------------------------------------------------------

def filter_depth_pixel_quality(
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    max_normal_angle_deg: float = 70.0,
    max_depth_jump_m: float = 0.008,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Apply per-pixel depth quality filtering before TSDF integration:
    - Drop pixels whose surface normal makes an angle > 70 deg with the view ray.
    - Drop flying pixels at depth discontinuities (depth jump > 8 mm between 4-neighbors).

    Parameters
    ----------
    depth_m:
        Input depth map in metres.
    intrinsics:
        Camera pinhole intrinsics.
    max_normal_angle_deg:
        Maximum allowed angle with optical view ray (default 70 deg).
    max_depth_jump_m:
        Maximum allowed depth difference to neighbor (default 0.008 m = 8 mm).

    Returns
    -------
    (filtered_depth_m, stats_dict)
    """
    h, w = depth_m.shape[:2]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    clean_depth = depth_m.copy()
    valid = (clean_depth > 0.05) & np.isfinite(clean_depth)

    # 1. Flying pixel filter (jump > 8 mm between 4-neighbors)
    diff_up = np.zeros_like(clean_depth)
    diff_down = np.zeros_like(clean_depth)
    diff_left = np.zeros_like(clean_depth)
    diff_right = np.zeros_like(clean_depth)

    diff_up[1:, :] = np.abs(clean_depth[1:, :] - clean_depth[:-1, :]) * (valid[1:, :] & valid[:-1, :])
    diff_down[:-1, :] = np.abs(clean_depth[:-1, :] - clean_depth[1:, :]) * (valid[:-1, :] & valid[1:, :])
    diff_left[:, 1:] = np.abs(clean_depth[:, 1:] - clean_depth[:, :-1]) * (valid[:, 1:] & valid[:, :-1])
    diff_right[:, :-1] = np.abs(clean_depth[:, :-1] - clean_depth[:, 1:]) * (valid[:, :-1] & valid[:, 1:])

    max_jump = np.maximum.reduce([diff_up, diff_down, diff_left, diff_right])
    jump_mask = valid & (max_jump > max_depth_jump_m)

    # 2. Surface normal angle filter (> 70 deg to view ray)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * clean_depth / fx
    Y = (v - cy) * clean_depth / fy
    Z = clean_depth

    # Central differences dP/du
    dx_du = np.zeros_like(clean_depth)
    dy_du = np.zeros_like(clean_depth)
    dz_du = np.zeros_like(clean_depth)
    dx_du[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2.0
    dy_du[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / 2.0
    dz_du[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / 2.0

    # Central differences dP/dv
    dx_dv = np.zeros_like(clean_depth)
    dy_dv = np.zeros_like(clean_depth)
    dz_dv = np.zeros_like(clean_depth)
    dx_dv[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2.0
    dy_dv[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / 2.0
    dz_dv[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / 2.0

    # Cross product N = dP/du x dP/dv
    nx = dy_du * dz_dv - dz_du * dy_dv
    ny = dz_du * dx_dv - dx_du * dz_dv
    nz = dx_du * dy_dv - dy_du * dx_dv

    norm_n = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm_n[norm_n == 0] = 1.0
    nx /= norm_n
    ny /= norm_n
    nz /= norm_n

    # View ray V = P / ||P|| pointing away from camera; view vector to camera is -V
    norm_p = np.sqrt(X * X + Y * Y + Z * Z)
    norm_p[norm_p == 0] = 1.0
    vx = X / norm_p
    vy = Y / norm_p
    vz = Z / norm_p

    # Orient normal towards camera
    dot_v = -(nx * vx + ny * vy + nz * vz)
    flip = dot_v < 0
    nx[flip] = -nx[flip]
    ny[flip] = -ny[flip]
    nz[flip] = -nz[flip]
    dot_v = -(nx * vx + ny * vy + nz * vz)

    cos_thresh = np.cos(np.radians(max_normal_angle_deg))
    normal_mask = valid & (dot_v < cos_thresh)

    drop_mask = jump_mask | normal_mask
    clean_depth[drop_mask] = 0.0

    stats = {
        "jump_dropped": int(np.count_nonzero(jump_mask)),
        "normal_angle_dropped": int(np.count_nonzero(normal_mask)),
        "total_dropped": int(np.count_nonzero(drop_mask)),
    }
    return clean_depth, stats


def mask_frame_to_head_roi(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    head_center_cam: np.ndarray,
    down_vec: np.ndarray | None = None,
    chin_dist: float | None = None,
    radius_m: float = 0.13,
    chin_margin_m: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Mask frame's depth + colour to a head region: a sphere around the head
    centre (~0.13 m radius), extended below the chin by at most 25 mm.
    Everything outside is dropped from fusion so torso/arms never enter TSDF.

    Returns
    -------
    (masked_depth_m, masked_color_bgr, dropped_count, total_valid_count)
    """
    h, w = depth_m.shape[:2]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth_m > 0.05) & np.isfinite(depth_m)
    total_valid = int(np.count_nonzero(valid))

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    pts = np.stack([X, Y, Z], axis=-1)

    dist = np.linalg.norm(pts - head_center_cam, axis=-1)
    head_mask = valid & (dist <= radius_m)

    if down_vec is not None and chin_dist is not None:
        norm_down = np.linalg.norm(down_vec)
        if norm_down > 1e-4:
            d_norm = down_vec / norm_down
            proj_down = np.sum((pts - head_center_cam) * d_norm, axis=-1)
            max_down = chin_dist + chin_margin_m
            head_mask = head_mask & (proj_down <= max_down)

    masked_depth = np.where(head_mask, depth_m, 0.0).astype(np.float32)
    masked_color = color_bgr.copy()
    masked_color[~head_mask] = 0

    dropped_count = total_valid - int(np.count_nonzero(head_mask))
    return masked_depth, masked_color, dropped_count, total_valid


def count_mesh_holes(mesh: o3d.geometry.TriangleMesh) -> int:
    """
    Count boundary loops (holes) in a triangle mesh.
    """
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return 0

    edge_counts = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_counts[e] += 1

    boundary_edges = [e for e, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return 0

    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    num_loops = 0
    for node in adj:
        if node not in visited:
            num_loops += 1
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    return num_loops


def fuse_head_scalable_tsdf(
    frames: list,
    poses: list[np.ndarray],
    voxel_length_m: float = 0.0015,
    sdf_trunc_m: float = 0.005,
    depth_trunc_m: float = 0.9,
    taubin_iterations: int = 10,
    min_component_ratio: float = 0.02,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Volumetric integration of head-only frames using ScalableTSDFVolume with colour,
    extracts triangle mesh with marching cubes, removes connected components smaller
    than 2% of the largest, and applies Taubin smoothing.

    Returns
    -------
    (smoothed_mesh, point_cloud)
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame, pose in zip(frames, poses):
        try:
            extrinsic = np.linalg.inv(pose)
            rgbd = frame.to_rgbd_image(
                depth_trunc=depth_trunc_m,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, frame.intrinsics, extrinsic)
        except Exception as exc:
            logger.debug(f"  [ScalableTSDF warning] frame skipped: {exc}")

    mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()

    if len(mesh.vertices) == 0:
        # Fall back to bounded UniformTSDFVolume (handles fine sub-centimeter voxel sizes reliably across platforms)
        pts_list = []
        for f, p in zip(frames, poses):
            if hasattr(f, "pcd") and hasattr(f.pcd, "points") and len(f.pcd.points) > 0:
                cam_pts = np.asarray(f.pcd.points)
                w_pts = (p[:3, :3] @ cam_pts.T).T + p[:3, 3]
                pts_list.append(w_pts)
        if pts_list:
            all_pts = np.vstack(pts_list)
            min_b = np.min(all_pts, axis=0) - 0.05
            max_b = np.max(all_pts, axis=0) + 0.05
            length = float(np.max(max_b - min_b))
            origin = min_b
        else:
            origin = np.array([-0.3, -0.3, 0.3])
            length = 0.6
        res = int(np.clip(round(length / voxel_length_m), 128, 512))
        vol_u = o3d.pipelines.integration.UniformTSDFVolume(
            length=length,
            resolution=res,
            sdf_trunc=sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=origin,
        )
        for frame, pose in zip(frames, poses):
            try:
                rgbd = frame.to_rgbd_image(depth_trunc=depth_trunc_m, convert_rgb_to_intensity=False)
                vol_u.integrate(rgbd, frame.intrinsics, np.linalg.inv(pose))
            except Exception:
                pass
        mesh = vol_u.extract_triangle_mesh()
        pcd = vol_u.extract_point_cloud()

    if len(mesh.triangles) > 0:
        # Remove connected components smaller than 2% of largest
        tri_clusters, num_triangles, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        num_triangles = np.asarray(num_triangles)
        if len(num_triangles) > 0:
            max_tri = np.max(num_triangles)
            thresh = min_component_ratio * max_tri
            keep_clusters = np.where(num_triangles >= thresh)[0]
            remove_mask = ~np.isin(tri_clusters, keep_clusters)
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()

        # Taubin smoothing (10 iterations; preserves volume, unlike Laplacian)
        if taubin_iterations > 0 and len(mesh.vertices) > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iterations)
            mesh.compute_vertex_normals()

    return mesh, pcd


def refine_poses_model_based(
    frames: list,
    initial_poses: list[np.ndarray],
    head_pcds: list[o3d.geometry.PointCloud],
    voxel_length_m: float = 0.002,
    sdf_trunc_m: float = 0.008,
    depth_trunc_m: float = 0.9,
    max_distance_m: float = 0.004,
    max_trans_m: float = 0.010,
    max_rot_deg: float = 3.0,
    max_iterations: int = 3,
    early_stop_improvement_m: float = 0.0001,
) -> tuple[list[np.ndarray], list[dict], o3d.geometry.PointCloud]:
    """
    Model-based refinement:
    a. Fuse head-only frames into a TSDF (voxel 2 mm) and extract a point cloud with normals.
    b. For each frame, run point-to-plane ICP of that frame (head ROI) against the model
       (distance threshold 4 mm). Accept updated pose only if RMSE improves and pose change
       is < 10 mm and < 3 deg.
    c. Repeat 2-3 iterations, re-fusing each time. Stop early if median RMSE improves < 0.1 mm.

    Returns
    -------
    (refined_poses, iteration_stats, final_model_pcd)
    """
    current_poses = [p.copy() for p in initial_poses]
    iteration_stats = []
    final_model = o3d.geometry.PointCloud()
    prev_median_rmse = float("inf")

    for iter_idx in range(max_iterations):
        # a. Fuse into TSDF (voxel 2 mm) and extract point cloud with normals
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length_m,
            sdf_trunc=sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for frame, pose in zip(frames, current_poses):
            try:
                rgbd = frame.to_rgbd_image(depth_trunc=depth_trunc_m, convert_rgb_to_intensity=False)
                vol.integrate(rgbd, frame.intrinsics, np.linalg.inv(pose))
            except Exception:
                pass

        model_pcd = vol.extract_point_cloud()
        if len(model_pcd.points) < 50:
            # Fall back to bounded UniformTSDFVolume for fine voxel resolution
            pts_list = []
            for f, p in zip(frames, current_poses):
                if hasattr(f, "pcd") and hasattr(f.pcd, "points") and len(f.pcd.points) > 0:
                    cam_pts = np.asarray(f.pcd.points)
                    w_pts = (p[:3, :3] @ cam_pts.T).T + p[:3, 3]
                    pts_list.append(w_pts)
            if pts_list:
                all_pts = np.vstack(pts_list)
                min_b = np.min(all_pts, axis=0) - 0.05
                max_b = np.max(all_pts, axis=0) + 0.05
                length = float(np.max(max_b - min_b))
                origin = min_b
            else:
                origin = np.array([-0.3, -0.3, 0.3])
                length = 0.6
            res = int(np.clip(round(length / voxel_length_m), 128, 512))
            vol_u = o3d.pipelines.integration.UniformTSDFVolume(
                length=length,
                resolution=res,
                sdf_trunc=sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=origin,
            )
            for frame, pose in zip(frames, current_poses):
                try:
                    rgbd = frame.to_rgbd_image(depth_trunc=depth_trunc_m, convert_rgb_to_intensity=False)
                    vol_u.integrate(rgbd, frame.intrinsics, np.linalg.inv(pose))
                except Exception:
                    pass
            model_pcd = vol_u.extract_point_cloud()

        if len(model_pcd.points) < 50:
            logger.warning("  [model-refine] TSDF extraction produced sparse cloud; stopping refinement.")
            break

        model_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
        final_model = model_pcd
        model_kdtree = o3d.geometry.KDTreeFlann(model_pcd)

        # b. For each frame, run point-to-plane ICP against the model
        accepted_count = 0
        frame_residuals = []

        for i, (f, pcd_local) in enumerate(zip(frames, head_pcds)):
            if len(pcd_local.points) < 20:
                continue

            T_old = current_poses[i]
            pts_world = (T_old[:3, :3] @ np.asarray(pcd_local.points).T).T + T_old[:3, 3]

            # Measure baseline residual before ICP
            dists = []
            for pt in pts_world:
                _, idxs, sq_dists = model_kdtree.search_knn_vector_3d(pt, 1)
                if idxs:
                    d = np.sqrt(sq_dists[0])
                    if d <= max_distance_m:
                        dists.append(d)

            rmse_old = np.sqrt(np.mean(np.square(dists))) if len(dists) >= 10 else float("inf")

            # Point-to-plane ICP of head ROI against model
            source = copy.deepcopy(pcd_local)
            reg_res = o3d.pipelines.registration.registration_icp(
                source,
                model_pcd,
                max_correspondence_distance=max_distance_m,
                init=T_old,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )

            T_new = reg_res.transformation
            delta_T = T_new @ np.linalg.inv(T_old)
            trans_change = float(np.linalg.norm(delta_T[:3, 3]))
            tr = float(np.trace(delta_T[:3, :3]))
            rot_deg = float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

            rmse_new = float(reg_res.inlier_rmse)
            accepted = (
                reg_res.fitness > 0.25
                and rmse_new < rmse_old
                and trans_change < max_trans_m
                and rot_deg < max_rot_deg
            )

            if accepted:
                current_poses[i] = T_new
                accepted_count += 1
                frame_residuals.append(rmse_new)
            else:
                frame_residuals.append(rmse_old if np.isfinite(rmse_old) else max_distance_m)

        med_residual_m = float(np.median(frame_residuals)) if frame_residuals else 0.0
        p90_residual_m = float(np.percentile(frame_residuals, 90.0)) if frame_residuals else 0.0

        stat = {
            "iteration": iter_idx + 1,
            "median_residual_mm": med_residual_m * 1000.0,
            "p90_residual_mm": p90_residual_m * 1000.0,
            "accepted_pose_updates": accepted_count,
            "total_frames": len(frames),
        }
        iteration_stats.append(stat)

        logger.info(
            f"  [model-refine] Iteration {iter_idx + 1}/{max_iterations}: "
            f"median residual = {stat['median_residual_mm']:.2f} mm, "
            f"p90 = {stat['p90_residual_mm']:.2f} mm "
            f"(accepted updates: {accepted_count}/{len(frames)})"
        )

        improvement_m = prev_median_rmse - med_residual_m
        prev_median_rmse = med_residual_m

        if iter_idx > 0 and improvement_m < early_stop_improvement_m:
            logger.info(
                f"  [model-refine] Stopping early at iteration {iter_idx + 1}: "
                f"median residual improvement ({improvement_m * 1000.0:.2f} mm) < {early_stop_improvement_m * 1000.0:.1f} mm"
            )
            break

    return current_poses, iteration_stats, final_model


def compute_reconstruction_quality_metrics(
    mesh: o3d.geometry.TriangleMesh,
    frames: list,
    poses_before: list[np.ndarray],
    poses_after: list[np.ndarray],
    head_pcds: list[o3d.geometry.PointCloud],
    cfg: HeadRefinementConfig,
) -> dict:
    """
    Compute quality metrics for head reconstruction:
    - Median and p90 distance of each frame's head points to the final mesh (mm), before and after refinement.
    - REPEATABILITY: fuse odd and even frames separately, report symmetric Chamfer distance and p95 distance (mm).
    - Mesh vertex count, triangle count, and hole count.

    Returns
    -------
    dict
    """
    if len(mesh.vertices) == 0:
        return {
            "mesh_vertex_count": 0,
            "mesh_triangle_count": 0,
            "mesh_hole_count": 0,
            "residual_before_refinement_mm": {"median": None, "p90": None},
            "residual_after_refinement_mm": {"median": None, "p90": None},
            "repeatability_odd_even_mm": {"chamfer_distance_mm": None, "p95_distance_mm": None},
        }

    # Sample points on the final mesh for distance queries
    n_sample = max(min(len(mesh.vertices) * 2, 50000), 10000)
    mesh_sampled = mesh.sample_points_uniformly(number_of_points=n_sample)
    mesh_kdtree = o3d.geometry.KDTreeFlann(mesh_sampled)

    def _eval_distances(poses: list[np.ndarray]) -> list[float]:
        all_dists = []
        for pose, pcd in zip(poses, head_pcds):
            if len(pcd.points) == 0:
                continue
            pts = np.asarray(pcd.points)
            pts_world = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
            for pt in pts_world:
                _, idxs, sq_d = mesh_kdtree.search_knn_vector_3d(pt, 1)
                if idxs:
                    all_dists.append(np.sqrt(sq_d[0]) * 1000.0)  # mm
        return all_dists

    dists_before = _eval_distances(poses_before)
    dists_after = _eval_distances(poses_after)

    res_before = {
        "median": round(float(np.median(dists_before)), 2) if dists_before else None,
        "p90": round(float(np.percentile(dists_before, 90.0)), 2) if dists_before else None,
    }
    res_after = {
        "median": round(float(np.median(dists_after)), 2) if dists_after else None,
        "p90": round(float(np.percentile(dists_after, 90.0)), 2) if dists_after else None,
    }

    # Repeatability: fuse odd and even frames separately
    repeatability = {"chamfer_distance_mm": None, "p95_distance_mm": None}
    odd_frames = frames[1::2]
    odd_poses = poses_after[1::2]
    even_frames = frames[0::2]
    even_poses = poses_after[0::2]

    if len(odd_frames) >= 2 and len(even_frames) >= 2:
        try:
            mesh_odd, _ = fuse_head_scalable_tsdf(
                odd_frames,
                odd_poses,
                voxel_length_m=cfg.voxel_length_mm / 1000.0,
                sdf_trunc_m=cfg.sdf_trunc_mm / 1000.0,
                depth_trunc_m=cfg.depth_trunc_m,
                taubin_iterations=cfg.taubin_iterations,
                min_component_ratio=cfg.min_component_ratio,
            )
            mesh_even, _ = fuse_head_scalable_tsdf(
                even_frames,
                even_poses,
                voxel_length_m=cfg.voxel_length_mm / 1000.0,
                sdf_trunc_m=cfg.sdf_trunc_mm / 1000.0,
                depth_trunc_m=cfg.depth_trunc_m,
                taubin_iterations=cfg.taubin_iterations,
                min_component_ratio=cfg.min_component_ratio,
            )

            if len(mesh_odd.vertices) > 20 and len(mesh_even.vertices) > 20:
                p_odd = np.asarray(mesh_odd.sample_points_uniformly(10000).points)
                p_even = np.asarray(mesh_even.sample_points_uniformly(10000).points)

                kd_even = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_even)))
                kd_odd = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_odd)))

                d_odd_to_even = []
                for pt in p_odd:
                    _, idxs, sq_d = kd_even.search_knn_vector_3d(pt, 1)
                    if idxs:
                        d_odd_to_even.append(np.sqrt(sq_d[0]) * 1000.0)

                d_even_to_odd = []
                for pt in p_even:
                    _, idxs, sq_d = kd_odd.search_knn_vector_3d(pt, 1)
                    if idxs:
                        d_even_to_odd.append(np.sqrt(sq_d[0]) * 1000.0)

                if d_odd_to_even and d_even_to_odd:
                    chamfer = 0.5 * (np.mean(d_odd_to_even) + np.mean(d_even_to_odd))
                    all_pair_d = np.concatenate([d_odd_to_even, d_even_to_odd])
                    p95 = np.percentile(all_pair_d, 95.0)

                    repeatability = {
                        "chamfer_distance_mm": round(float(chamfer), 2),
                        "p95_distance_mm": round(float(p95), 2),
                    }
        except Exception as exc:
            logger.warning(f"  [repeatability evaluation failed]: {exc}")

    hole_count = count_mesh_holes(mesh)

    metrics = {
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_triangle_count": int(len(mesh.triangles)),
        "mesh_hole_count": hole_count,
        "residual_before_refinement_mm": res_before,
        "residual_after_refinement_mm": res_after,
        "repeatability_odd_even_mm": repeatability,
    }
    return metrics

