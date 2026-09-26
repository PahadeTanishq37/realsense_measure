"""
Point-cloud fusion: merge all registered per-frame clouds into one clean mesh.

After registration every frame lives in the same coordinate system, so fusion
is simply concatenation followed by a final cleaning pass.
"""

from __future__ import annotations

import copy

import open3d as o3d

from config import PreprocessConfig
from preprocessing import downsample_and_denoise, extract_largest_cluster


def fuse_point_clouds(
    aligned_frames: list[o3d.geometry.PointCloud],
    cfg: PreprocessConfig,
) -> o3d.geometry.PointCloud:
    """
    Merge a list of registered, per-frame object clouds into one clean cloud.

    Steps
    -----
    1. Concatenate all frames into a single cloud.
    2. Voxel-downsample + statistical-outlier-removal via
       :func:`~preprocessing.downsample_and_denoise`.
    3. Retain only the largest DBSCAN cluster via
       :func:`~preprocessing.extract_largest_cluster`.

       Even with thorough per-frame isolation, a small misregistered chunk of
       leftover background (e.g. a sliver of table edge that survived plane
       removal in one frame) can end up as a separate disconnected blob in the
       fused cloud.  This final clustering pass identifies and drops it,
       ensuring only the dominant — and correct — object geometry is returned.

    Parameters
    ----------
    aligned_frames:
        Ordered list of point clouds, all expressed in the shared reference
        coordinate system produced by :func:`~registration.register_sequence`.
    cfg:
        Preprocessing configuration (voxel size, outlier thresholds, DBSCAN
        parameters) — the same config used during per-frame preprocessing.

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

    # Downsample the dense fused cloud and kill any sensor noise that
    # survived because it happened to appear in multiple frames.
    fused = downsample_and_denoise(fused, cfg)

    # Final cluster pass: drop any small stray blobs from imperfect per-frame
    # isolation that only became visible once all views were merged together.
    fused = extract_largest_cluster(fused, cfg)

    return fused
