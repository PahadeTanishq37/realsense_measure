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
) -> o3d.geometry.PointCloud:
    """
    Merge a list of registered, per-frame object clouds into one clean cloud.

    Steps
    -----
    1. Concatenate all frames into a single cloud.
    2. Voxel-downsample + statistical-outlier-removal via
       :func:`~preprocessing.downsample_and_denoise`.
    3. (BOX only) Retain only the largest DBSCAN cluster via
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
    if fused.has_points():
        fused, _ = fused.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )

    # 2. Light voxel downsample to merge near-duplicate points from overlapping angles
    if cfg.voxel_size_m > 0 and fused.has_points():
        fused = fused.voxel_down_sample(voxel_size=cfg.voxel_size_m)

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

    # Determine volume origin and extents from initial frames
    all_pts = []
    for f in frames[:min(5, len(frames))]:
        if hasattr(f, "pcd") and len(f.pcd.points) > 0:
            all_pts.append(np.asarray(f.pcd.points))

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
    return fuse_point_clouds(pcds, cfg, skip_final_clustering=skip_final_clustering)

