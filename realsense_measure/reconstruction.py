"""
Point-cloud fusion: merge all registered per-frame clouds into one clean mesh.

After registration every frame lives in the same coordinate system, so fusion
is simply concatenation followed by a final cleaning pass.
"""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

from config import PreprocessConfig
from preprocessing import downsample_and_denoise, extract_largest_cluster


def fuse_point_clouds(
    aligned_frames: list[o3d.geometry.PointCloud],
    cfg: PreprocessConfig,
    skip_final_clustering: bool = False,
    skip_outlier_removal: bool = False,
    voxel_size: float | None = None,
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

    resolution = int(np.clip(length / voxel_length, 128, 512))

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

