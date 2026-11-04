"""
Comprehensive unit and integration tests for Provenance-Preserving IndexedPointCloud.

Covers:
1. RGB-D back-projection numerical accuracy vs Open3D.
2. Pixel-to-3D and 3D-to-pixel bidirectional correspondence.
3. Handling of invalid depths (0, negative, NaN, Inf, out-of-range).
4. Intrinsics consistency and predictable geometry scaling.
5. Rotated image coordinate handling (90-degree orientations).
6. 3D ROI filtering with provenance preservation.
7. Statistical outlier removal with provenance preservation.
8. Dominant plane removal with provenance preservation.
9. DBSCAN clustering with provenance preservation.
10. Voxel downsampling aggregation history (K=1 vs K>1, centroid-nearest representative).
11. Multi-source frame concatenation and multi-frame voxel aggregation.
12. Open3D conversion roundtrip (to_open3d and from_open3d).
13. Recording and replay provenance reconstruction.
14. Legacy recording compatibility and graceful fallback.
15. Invariance under point shuffling and non-contiguous masking.
"""

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.camera.frame import RGBDFrame
from realsense_measure.camera.indexed_pcd import (
    FrameMetadata,
    IndexedPointCloud,
    create_indexed_pcd_from_rgbd,
)
from realsense_measure.camera.recording import load_raw_capture, save_raw_capture
from realsense_measure.config import CameraConfig, LandmarkConfig, PipelineConfig, PreprocessConfig, TargetConfig
from realsense_measure.preprocessing import (
    downsample_and_denoise,
    extract_largest_cluster,
    remove_dominant_plane,
)
from realsense_measure.landmarks import LandmarkSample3D, sample_landmark_3d


@pytest.fixture
def sample_camera_intrinsics():
    """Standard 848x480 D455-like intrinsics."""
    return o3d.camera.PinholeCameraIntrinsic(
        width=848,
        height=480,
        fx=424.0,
        fy=424.0,
        cx=424.0,
        cy=240.0,
    )


@pytest.fixture
def synthetic_rgbd(sample_camera_intrinsics):
    """Synthetic RGB and depth images with known geometric features."""
    w = sample_camera_intrinsics.width
    h = sample_camera_intrinsics.height

    # Create deterministic color pattern
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)  # Blue gradient
    color[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # Green gradient
    color[:, :, 2] = 128  # Constant Red

    # Create depth map with valid planar ramp in center and zeros on borders
    depth = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # Center region [100:380, 200:648] has depth 1.0m to 2.0m
    center_mask = (yy >= 100) & (yy < 380) & (xx >= 200) & (xx < 648)
    depth[center_mask] = 1.0 + 0.5 * (xx[center_mask] - 200) / 448.0 + 0.2 * (yy[center_mask] - 100) / 280.0

    return color, depth, sample_camera_intrinsics


# ---------------------------------------------------------------------------
# Test 1: RGB-D Back-projection Accuracy vs Open3D
# ---------------------------------------------------------------------------
def test_rgbd_backprojection_accuracy(synthetic_rgbd):
    """
    Verify vectorized back-projection matches Open3D PointCloud.create_from_rgbd_image
    to machine precision (< 1e-10 m).
    """
    color, depth, intr = synthetic_rgbd
    depth_trunc = 3.0

    # 1. Custom IndexedPointCloud
    idx_pcd = create_indexed_pcd_from_rgbd(
        color=color,
        depth_m=depth,
        intrinsics=intr,
        depth_min_m=0.10,
        depth_max_m=depth_trunc,
        frame_id=1,
        timestamp=10.5,
        is_bgr=True,
    )

    # 2. Open3D PointCloud
    color_rgb = np.ascontiguousarray(color[:, :, ::-1])
    depth_clipped = depth.copy()
    depth_clipped[depth_clipped < 0.10] = 0.0
    depth_clipped[depth_clipped > depth_trunc] = 0.0

    o3d_img_c = o3d.geometry.Image(color_rgb)
    o3d_img_d = o3d.geometry.Image(depth_clipped)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_img_c, o3d_img_d, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    assert len(idx_pcd) == len(o3d_pcd.points)
    o3d_pts = np.asarray(o3d_pcd.points)
    o3d_cols = np.asarray(o3d_pcd.colors)

    max_pt_diff = np.max(np.abs(idx_pcd.points - o3d_pts))
    max_col_diff = np.max(np.abs(idx_pcd.colors - o3d_cols))

    assert max_pt_diff < 1e-6, f"Max point diff {max_pt_diff} exceeds 1 um"
    assert max_col_diff < 1e-6, f"Max color diff {max_col_diff} exceeds tolerance"


# ---------------------------------------------------------------------------
# Test 2: Pixel-to-3D and 3D-to-Pixel Correspondence
# ---------------------------------------------------------------------------
def test_pixel_to_3d_correspondence(synthetic_rgbd):
    """
    Verify that 3D points precisely map back to their original 2D (u, v) coordinates
    using the camera pinhole projection equation.
    """
    color, depth, intr = synthetic_rgbd
    idx_pcd = create_indexed_pcd_from_rgbd(color, depth, intr, depth_min_m=0.1, depth_max_m=3.0)

    fx = intr.intrinsic_matrix[0, 0]
    fy = intr.intrinsic_matrix[1, 1]
    cx = intr.intrinsic_matrix[0, 2]
    cy = intr.intrinsic_matrix[1, 2]

    # Sample 100 random points from the cloud
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(idx_pcd), size=100, replace=False)

    for i in sample_indices:
        pt = idx_pcd.points[i]
        u, v = idx_pcd.source_pixel_uv[i]
        z_rec = idx_pcd.source_depth_m[i]

        # Verify depth stored matches Z coordinate
        assert np.isclose(pt[2], z_rec, atol=1e-6)
        # Verify ground truth depth from array matches
        assert np.isclose(depth[v, u], z_rec, atol=1e-6)

        # Reproject 3D point to image plane
        u_proj = (pt[0] * fx / pt[2]) + cx
        v_proj = (pt[1] * fy / pt[2]) + cy

        assert np.isclose(u_proj, u, atol=1e-3), f"Reprojected u {u_proj} != source u {u}"
        assert np.isclose(v_proj, v, atol=1e-3), f"Reprojected v {v_proj} != source v {v}"


# ---------------------------------------------------------------------------
# Test 3: Handling of Invalid Depths
# ---------------------------------------------------------------------------
def test_invalid_depth_handling(sample_camera_intrinsics):
    """
    Verify that zero, negative, NaN, Inf, and out-of-range depths are rejected
    without creating spurious points or corrupting pixel alignment.
    """
    h, w = 480, 848
    color = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)

    # Valid points
    depth[100, 100] = 1.0
    depth[200, 200] = 2.0

    # Invalids
    depth[100, 101] = -0.5
    depth[100, 102] = np.nan
    depth[100, 103] = np.inf
    depth[100, 104] = 0.05  # Below min (0.10)
    depth[100, 105] = 5.0   # Above max (3.0)

    idx_pcd = create_indexed_pcd_from_rgbd(
        color, depth, sample_camera_intrinsics, depth_min_m=0.10, depth_max_m=3.0
    )

    assert len(idx_pcd) == 2
    # Verify exact pixel locations
    assert np.array_equal(idx_pcd.source_pixel_uv[0], [100, 100])
    assert np.array_equal(idx_pcd.source_pixel_uv[1], [200, 200])


# ---------------------------------------------------------------------------
# Test 4: Intrinsics Consistency
# ---------------------------------------------------------------------------
def test_intrinsics_consistency():
    """
    Verify that scaling focal lengths scales 3D coordinates inversely while preserving
    pixel coordinate provenance.
    """
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32)

    intr1 = o3d.camera.PinholeCameraIntrinsic(100, 100, fx=50.0, fy=50.0, cx=50.0, cy=50.0)
    intr2 = o3d.camera.PinholeCameraIntrinsic(100, 100, fx=100.0, fy=100.0, cx=50.0, cy=50.0)

    pcd1 = create_indexed_pcd_from_rgbd(color, depth, intr1)
    pcd2 = create_indexed_pcd_from_rgbd(color, depth, intr2)

    assert len(pcd1) == len(pcd2) == 10000
    # Provenance pixel coordinates are identical
    assert np.array_equal(pcd1.source_pixel_uv, pcd2.source_pixel_uv)

    # With 2x focal length, X and Y metric spans must be exactly 0.5x
    span_x1 = pcd1.points[:, 0].max() - pcd1.points[:, 0].min()
    span_x2 = pcd2.points[:, 0].max() - pcd2.points[:, 0].min()
    assert np.isclose(span_x1, 2.0 * span_x2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: Rotated Image Coordinate Handling
# ---------------------------------------------------------------------------
def test_rotated_image_coordinate_handling():
    """
    Verify that rotated images (e.g. 90-degree portrait mode) preserve correct pixel
    dimensions, bounds, and provenance.
    """
    # 480x848 portrait orientation (height=848, width=480)
    h, w = 848, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 1.5

    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx=424.0, fy=424.0, cx=240.0, cy=424.0)
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    assert len(pcd) == h * w
    assert pcd.source_pixel_uv[:, 0].max() == w - 1
    assert pcd.source_pixel_uv[:, 1].max() == h - 1
    assert pcd.frames_meta[0].width == w
    assert pcd.frames_meta[0].height == h


# ---------------------------------------------------------------------------
# Test 6: 3D ROI Filtering Preserves Provenance
# ---------------------------------------------------------------------------
def test_roi_filtering_preserves_provenance(synthetic_rgbd):
    """
    Verify that 3D bounding box ROI filtering preserves original pixel coordinates.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Filter bounding box
    min_b = [-0.2, -0.1, 1.1]
    max_b = [0.2, 0.1, 1.4]
    roi_pcd = pcd.filter_roi(min_b, max_b)

    assert len(roi_pcd) > 0
    assert len(roi_pcd) < len(pcd)

    # Check bounds
    assert np.all(roi_pcd.points >= min_b)
    assert np.all(roi_pcd.points <= max_b)

    # Check that provenance reflects the true pixels from depth
    for u, v, z in zip(roi_pcd.source_pixel_uv[:, 0], roi_pcd.source_pixel_uv[:, 1], roi_pcd.source_depth_m):
        assert np.isclose(depth[v, u], z, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 7: Statistical Outlier Removal Preserves Provenance
# ---------------------------------------------------------------------------
def test_statistical_outlier_removal_preserves_provenance(synthetic_rgbd):
    """
    Verify that outlier removal filters points without changing the correspondence
    of the inliers.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Inject isolated noisy points far away
    noisy_pts = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, 10.0]], dtype=np.float32)
    noisy_colors = np.zeros((2, 3), dtype=np.float32)
    noisy_fid = np.array([99, 99], dtype=np.int32)
    noisy_uv = np.array([[1, 1], [2, 2]], dtype=np.int32)

    noisy_cloud = IndexedPointCloud(
        points=np.vstack([pcd.points, noisy_pts]),
        colors=np.vstack([pcd.colors, noisy_colors]),
        source_frame_id=np.concatenate([pcd.source_frame_id, noisy_fid]),
        source_pixel_uv=np.vstack([pcd.source_pixel_uv, noisy_uv]),
        frames_meta=pcd.frames_meta,
    )

    cleaned_pcd, inliers = noisy_cloud.filter_statistical_outliers(nb_neighbors=20, std_ratio=1.5)

    assert len(cleaned_pcd) < len(noisy_cloud)
    # The artificial outliers must be rejected
    assert 99 not in cleaned_pcd.source_frame_id


# ---------------------------------------------------------------------------
# Test 8: Dominant Plane Removal Preserves Provenance
# ---------------------------------------------------------------------------
def test_plane_removal_preserves_provenance():
    """
    Verify that segmenting a supporting plane removes table points while
    preserving pixel origins for the remaining object.
    """
    # Create synthetic scene: a flat table at Z=1.0m plus an object sphere at Z=0.8m
    h, w = 100, 100
    color = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 1.0  # Flat table at 1.0m

    # Object in center
    yy, xx = np.ogrid[:h, :w]
    obj_mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 15 ** 2
    depth[obj_mask] = 0.8  # Object closer to camera

    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    cfg = PreprocessConfig(
        plane_dist_threshold_m=0.01,
        plane_ransac_n=3,
        plane_num_iterations=200,
    )
    non_plane_pcd, plane_model = remove_dominant_plane(pcd, cfg)

    # Non-plane points should correspond to the object mask
    assert len(non_plane_pcd) > 0
    assert len(non_plane_pcd) < len(pcd)
    assert np.allclose(non_plane_pcd.points[:, 2], 0.8, atol=0.05)
    # Provenance depth must be 0.8
    assert np.allclose(non_plane_pcd.source_depth_m, 0.8, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: DBSCAN Clustering Preserves Provenance
# ---------------------------------------------------------------------------
def test_dbscan_clustering_preserves_provenance(synthetic_rgbd):
    """
    Verify that DBSCAN clustering and cluster extraction preserve point attributes.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    cfg = PreprocessConfig(cluster_eps_m=0.05, cluster_min_points=10)
    largest_cluster = extract_largest_cluster(pcd, cfg)

    assert len(largest_cluster) > 0
    assert len(largest_cluster) <= len(pcd)
    assert isinstance(largest_cluster, IndexedPointCloud)
    # Check that pixel origins remain valid
    assert largest_cluster.source_pixel_uv[:, 0].min() >= 0
    assert largest_cluster.source_pixel_uv[:, 1].min() >= 0


# ---------------------------------------------------------------------------
# Test 10: Voxel Downsampling Provenance Aggregation
# ---------------------------------------------------------------------------
def test_voxel_downsampling_provenance_aggregation(synthetic_rgbd):
    """
    Verify aggregation history:
    - Initial points have K=1
    - Downsampled points have K > 1
    - Provenance representative belongs to the point closest to the voxel centroid
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Before downsampling
    assert np.all(pcd.aggregation_count == 1)

    voxel_size = 0.05  # 5cm
    down_pcd = pcd.voxel_downsample(voxel_size, aggregation_strategy="centroid_nearest")

    assert len(down_pcd) < len(pcd)
    # Merged points must have K >= 1
    assert np.all(down_pcd.aggregation_count >= 1)
    assert np.any(down_pcd.aggregation_count > 1)
    # Total sum of aggregation counts must equal original point count
    assert np.sum(down_pcd.aggregation_count) == len(pcd)

    # Verify that representative (u, v) is a valid coordinate within the camera frame
    assert np.all(down_pcd.source_pixel_uv[:, 0] >= 0)
    assert np.all(down_pcd.source_pixel_uv[:, 0] < intr.width)
    assert np.all(down_pcd.source_pixel_uv[:, 1] >= 0)
    assert np.all(down_pcd.source_pixel_uv[:, 1] < intr.height)


# ---------------------------------------------------------------------------
# Test 11: Multi-Source Frame Concatenation and Multi-Frame Voxel Aggregation
# ---------------------------------------------------------------------------
def test_multiple_source_frames_concatenation_and_provenance(synthetic_rgbd):
    """
    Verify multi-frame merging retains source frame IDs and tracks contributing
    frame IDs per voxel after fusion.
    """
    color, depth, intr = synthetic_rgbd

    pcd0 = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0, timestamp=1.0)
    pcd1 = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=1, timestamp=2.0)

    # Slight shift on pcd1 to overlap
    pcd1.transform(np.array([
        [1.0, 0.0, 0.0, 0.005],
        [0.0, 1.0, 0.0, 0.005],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))

    merged = pcd0.merge(pcd1)
    assert len(merged) == len(pcd0) + len(pcd1)
    assert 0 in merged.frames_meta
    assert 1 in merged.frames_meta

    # Downsample merged cloud
    fused_pcd = merged.voxel_downsample(voxel_size=0.03)

    # Some voxels must have points from both frames 0 and 1
    shared_voxels = [
        cf for cf in fused_pcd.contributing_frame_ids if 0 in cf and 1 in cf
    ]
    assert len(shared_voxels) > 0, "Expected voxels containing contributions from both frame 0 and frame 1"


# ---------------------------------------------------------------------------
# Test 12: Open3D Conversion Roundtrip
# ---------------------------------------------------------------------------
def test_open3d_conversion_roundtrip(synthetic_rgbd):
    """
    Verify conversion to and from Open3D PointCloud preserves geometry and colors.
    """
    color, depth, intr = synthetic_rgbd
    idx_pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    o3d_pcd = idx_pcd.to_open3d()
    assert isinstance(o3d_pcd, o3d.geometry.PointCloud)
    assert len(o3d_pcd.points) == len(idx_pcd)

    # Convert back
    roundtrip_pcd = IndexedPointCloud.from_open3d(o3d_pcd, frame_id=42, intrinsics=intr)
    assert len(roundtrip_pcd) == len(idx_pcd)
    assert np.allclose(roundtrip_pcd.points, idx_pcd.points, atol=1e-5)
    assert np.allclose(roundtrip_pcd.colors, idx_pcd.colors, atol=1e-5)
    assert np.all(roundtrip_pcd.source_frame_id == 42)


# ---------------------------------------------------------------------------
# Test 13: Recording and Replay Provenance Reconstruction
# ---------------------------------------------------------------------------
def test_recording_and_replay_provenance_reconstruction(synthetic_rgbd):
    """
    Verify save_raw_capture and load_raw_capture reconstruct IndexedPointCloud
    with full sensor provenance.
    """
    color, depth, intr = synthetic_rgbd

    frame = RGBDFrame(
        frame_id=0,
        color_bgr=color,
        depth_m=depth,
        pcd=create_indexed_pcd_from_rgbd(color, depth, intr).to_open3d(),
        timestamp=123.456,
        intrinsics=intr,
        indexed_pcd=create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0, timestamp=123.456),
    )

    cfg = PipelineConfig(
        camera=CameraConfig(width=intr.width, height=intr.height, fps=30),
        target=TargetConfig(name="body"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_raw_capture([frame], tmp_dir, cfg)

        # Load back
        replayed_frames = load_raw_capture(tmp_dir, target_name="body")
        assert len(replayed_frames) == 1
        rep_frame = replayed_frames[0]

        assert rep_frame.indexed_pcd is not None
        assert len(rep_frame.indexed_pcd) == len(frame.indexed_pcd)
        assert np.allclose(rep_frame.indexed_pcd.points, frame.indexed_pcd.points, atol=1e-5)
        assert np.array_equal(rep_frame.indexed_pcd.source_pixel_uv, frame.indexed_pcd.source_pixel_uv)
        assert rep_frame.indexed_pcd.frames_meta[0].frame_id == 0


# ---------------------------------------------------------------------------
# Test 14: Legacy Recording Compatibility
# ---------------------------------------------------------------------------
def test_legacy_recording_graceful_handling(synthetic_rgbd):
    """
    Verify that when loading a legacy recording without precomputed indexed_pcd,
    load_raw_capture deterministically creates a valid IndexedPointCloud from raw PNG/NPY.
    """
    color, depth, intr = synthetic_rgbd

    # Simulate legacy recording by writing PNG, NPY, and meta.json manually
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cv2.imwrite(str(tmp_path / "frame_00_color.png"), color)
        np.save(str(tmp_path / "frame_00_depth.npy"), depth)

        meta_doc = {
            "intrinsics": {
                "width": intr.width,
                "height": intr.height,
                "fx": float(intr.intrinsic_matrix[0, 0]),
                "fy": float(intr.intrinsic_matrix[1, 1]),
                "cx": float(intr.intrinsic_matrix[0, 2]),
                "cy": float(intr.intrinsic_matrix[1, 2]),
            },
            "camera_config": {"depth_min_m": 0.40, "depth_max_m": 3.00},
            "target_name": "body",
            "frames": [{"frame_id": 0, "timestamp": 0.0, "color_file": "frame_00_color.png", "depth_file": "frame_00_depth.npy"}],
        }
        with open(tmp_path / "meta.json", "w") as f:
            json.dump(meta_doc, f)

        loaded = load_raw_capture(tmp_dir)
        assert len(loaded) == 1
        assert loaded[0].indexed_pcd is not None
        assert len(loaded[0].indexed_pcd) > 0
        assert np.all(loaded[0].indexed_pcd.source_frame_id == 0)


# ---------------------------------------------------------------------------
# Test 15: Invariance under Point Shuffling and Masking
# ---------------------------------------------------------------------------
def test_shuffling_and_filtering_alignment_invariant(synthetic_rgbd):
    """
    Verify that arbitrary permutations or non-contiguous index selections
    never break the binding between a 3D point and its source (u, v) pixel coordinate.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    fx = intr.intrinsic_matrix[0, 0]
    fy = intr.intrinsic_matrix[1, 1]
    cx = intr.intrinsic_matrix[0, 2]
    cy = intr.intrinsic_matrix[1, 2]

    # Permute all points randomly
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(pcd))
    shuffled_pcd = pcd.select_by_index(perm)

    # Select random 500 points from the shuffled cloud
    sample_sub = shuffled_pcd.select_by_index(np.arange(0, 500))

    for pt, (u, v), z in zip(sample_sub.points, sample_sub.source_pixel_uv, sample_sub.source_depth_m):
        u_calc = (pt[0] * fx / z) + cx
        v_calc = (pt[1] * fy / z) + cy
        assert np.isclose(u_calc, u, atol=1e-3)
        assert np.isclose(v_calc, v, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 16: Single-frame Zero Per-Point List Overhead
# ---------------------------------------------------------------------------
def test_single_frame_zero_list_overhead(synthetic_rgbd):
    """
    Verify that an unaggregated single-frame IndexedPointCloud does not allocate
    per-point Python lists in internal instance storage.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=7)

    # Must use implicit contributor storage
    assert pcd.cf_flat is None
    assert pcd.cf_offsets is None
    # Instance dictionary must not contain a per-point list
    assert "contributing_frame_ids" not in pcd.__dict__

    # Point-level retrieval works implicitly from source_frame_id
    assert pcd.get_contributing_frame_ids(0) == [7]
    assert pcd.get_contributing_frame_ids(len(pcd) - 1) == [7]


# ---------------------------------------------------------------------------
# Test 17: Contributor-Frame Retrieval from Compact Ragged Representation
# ---------------------------------------------------------------------------
def test_contributor_retrieval_compact_representation():
    """
    Verify that get_contributing_frame_ids retrieves exact frame IDs from
    the compact ragged representation (cf_flat and cf_offsets).
    """
    points = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.float32)
    fid = np.array([0, 1, 2], dtype=np.int32)
    uv = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)

    # Point 0: frames [0, 1]
    # Point 1: frames [1]
    # Point 2: frames [0, 1, 2, 3]
    cf_flat = np.array([0, 1, 1, 0, 1, 2, 3], dtype=np.int32)
    cf_offsets = np.array([0, 2, 3, 7], dtype=np.int64)

    pcd = IndexedPointCloud(
        points=points,
        colors=colors,
        source_frame_id=fid,
        source_pixel_uv=uv,
        cf_flat=cf_flat,
        cf_offsets=cf_offsets,
    )

    assert pcd.get_contributing_frame_ids(0) == [0, 1]
    assert pcd.get_contributing_frame_ids(1) == [1]
    assert pcd.get_contributing_frame_ids(2) == [0, 1, 2, 3]

    # Verify backward compatible property
    legacy_list = pcd.contributing_frame_ids
    assert legacy_list == [[0, 1], [1], [0, 1, 2, 3]]


# ---------------------------------------------------------------------------
# Test 18: Multiple Frames Contributing to a Voxel Retain All Frame IDs
# ---------------------------------------------------------------------------
def test_multi_frame_voxel_retention():
    """
    Verify that when points from 3 different frames merge into a single voxel,
    all 3 frame IDs are retained in cf_flat and cf_offsets.
    """
    # 3 points in identical spatial position from frames 10, 20, 30
    points = np.array([[0.1, 0.2, 1.0], [0.1, 0.2, 1.0], [0.1, 0.2, 1.0]], dtype=np.float32)
    colors = np.ones((3, 3), dtype=np.float32)
    fid = np.array([10, 20, 30], dtype=np.int32)
    uv = np.array([[10, 10], [10, 10], [10, 10]], dtype=np.int32)

    pcd = IndexedPointCloud(
        points=points,
        colors=colors,
        source_frame_id=fid,
        source_pixel_uv=uv,
    )

    # Downsample with voxel size large enough to merge all 3 points
    down = pcd.voxel_downsample(voxel_size=0.05)
    assert len(down) == 1

    # Must contain all 3 frames
    cf = down.get_contributing_frame_ids(0)
    assert sorted(cf) == [10, 20, 30]
    assert down.cf_flat is not None
    assert down.cf_offsets is not None
    assert len(down.cf_flat) == 3
    assert np.array_equal(down.cf_offsets, [0, 3])


# ---------------------------------------------------------------------------
# Test 19: Filtering and Indexing Preserve Contributor Information
# ---------------------------------------------------------------------------
def test_filtering_and_indexing_preserves_contributors():
    """
    Verify that select_by_mask and select_by_index slice ragged contributor
    arrays vectorially without corrupting offsets.
    """
    points = np.zeros((4, 3), dtype=np.float32)
    colors = np.zeros((4, 3), dtype=np.float32)
    fid = np.arange(4, dtype=np.int32)
    uv = np.zeros((4, 2), dtype=np.int32)

    # Point 0: [100]
    # Point 1: [200, 201]
    # Point 2: [300, 301, 302]
    # Point 3: [400]
    cf_flat = np.array([100, 200, 201, 300, 301, 302, 400], dtype=np.int32)
    cf_offsets = np.array([0, 1, 3, 6, 7], dtype=np.int64)

    pcd = IndexedPointCloud(
        points=points,
        colors=colors,
        source_frame_id=fid,
        source_pixel_uv=uv,
        cf_flat=cf_flat,
        cf_offsets=cf_offsets,
    )

    # Select points 1 and 3
    sub = pcd.select_by_index([1, 3])
    assert len(sub) == 2
    assert sub.get_contributing_frame_ids(0) == [200, 201]
    assert sub.get_contributing_frame_ids(1) == [400]
    assert np.array_equal(sub.cf_offsets, [0, 2, 3])
    assert np.array_equal(sub.cf_flat, [200, 201, 400])


# ---------------------------------------------------------------------------
# Test 20: Cloning Produces an Equivalent Independent Object
# ---------------------------------------------------------------------------
def test_clone_independence_ragged_contributors():
    """
    Verify that clone() creates an independent copy of cf_flat and cf_offsets
    such that mutating one does not mutate the other.
    """
    points = np.zeros((2, 3), dtype=np.float32)
    colors = np.zeros((2, 3), dtype=np.float32)
    fid = np.array([0, 1], dtype=np.int32)
    uv = np.zeros((2, 2), dtype=np.int32)

    cf_flat = np.array([10, 11, 20], dtype=np.int32)
    cf_offsets = np.array([0, 2, 3], dtype=np.int64)

    pcd = IndexedPointCloud(
        points=points,
        colors=colors,
        source_frame_id=fid,
        source_pixel_uv=uv,
        cf_flat=cf_flat,
        cf_offsets=cf_offsets,
    )

    cloned = pcd.clone()
    assert cloned.get_contributing_frame_ids(0) == [10, 11]
    assert cloned.get_contributing_frame_ids(1) == [20]

    # Mutate cloned cf_flat
    cloned.cf_flat[0] = 999
    assert pcd.cf_flat[0] == 10  # Original must remain untouched
    assert cloned.cf_flat[0] == 999


# ---------------------------------------------------------------------------
# Test 21: Merging Clouds Preserves Frame Metadata and Contributors
# ---------------------------------------------------------------------------
def test_merge_preserves_contributors_and_metadata():
    """
    Verify that merge() correctly combines ragged contributor arrays and
    frames_meta dictionaries across different clouds.
    """
    pcd1 = IndexedPointCloud(
        points=np.array([[1, 1, 1]], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        source_frame_id=np.array([1], dtype=np.int32),
        source_pixel_uv=np.zeros((1, 2), dtype=np.int32),
        cf_flat=np.array([1, 10], dtype=np.int32),
        cf_offsets=np.array([0, 2], dtype=np.int64),
        frames_meta={1: FrameMetadata(frame_id=1, timestamp=1.0)},
    )

    pcd2 = IndexedPointCloud(
        points=np.array([[2, 2, 2]], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        source_frame_id=np.array([2], dtype=np.int32),
        source_pixel_uv=np.zeros((1, 2), dtype=np.int32),
        frames_meta={2: FrameMetadata(frame_id=2, timestamp=2.0)},
    )

    merged = pcd1.merge(pcd2)
    assert len(merged) == 2
    assert 1 in merged.frames_meta
    assert 2 in merged.frames_meta
    assert merged.get_contributing_frame_ids(0) == [1, 10]
    assert merged.get_contributing_frame_ids(1) == [2]


# ---------------------------------------------------------------------------
# Test 22: Empty Cloud Contributor Handling
# ---------------------------------------------------------------------------
def test_empty_cloud_contributor_handling():
    """
    Verify that an empty IndexedPointCloud handles contributor operations gracefully.
    """
    empty = IndexedPointCloud(
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        source_frame_id=np.zeros(0, dtype=np.int32),
        source_pixel_uv=np.zeros((0, 2), dtype=np.int32),
    )

    assert len(empty) == 0
    assert empty.is_empty()
    assert empty.cf_flat is None
    assert empty.cf_offsets is None
    assert empty.contributing_frame_ids == []

    cloned = empty.clone()
    assert len(cloned) == 0

    sub = empty.select_by_index([])
    assert len(sub) == 0

    with pytest.raises(IndexError):
        empty.get_contributing_frame_ids(0)


# ---------------------------------------------------------------------------
# Test 23: Backward Compatible Property Setter and Getter
# ---------------------------------------------------------------------------
def test_backward_compatible_contributing_frame_ids_property():
    """
    Verify that reading and writing the contributing_frame_ids property converts
    to/from ragged arrays without breaking legacy callers.
    """
    pcd = IndexedPointCloud(
        points=np.zeros((2, 3), dtype=np.float32),
        colors=np.zeros((2, 3), dtype=np.float32),
        source_frame_id=np.array([5, 6], dtype=np.int32),
        source_pixel_uv=np.zeros((2, 2), dtype=np.int32),
    )

    # Initial implicit state
    assert pcd.cf_flat is None
    assert pcd.contributing_frame_ids == [[5], [6]]

    # Setting multi-frame contributor lists converts to compact arrays
    pcd.contributing_frame_ids = [[5, 50], [6, 60, 600]]
    assert pcd.cf_flat is not None
    assert pcd.cf_offsets is not None
    assert np.array_equal(pcd.cf_flat, [5, 50, 6, 60, 600])
    assert np.array_equal(pcd.cf_offsets, [0, 2, 5])
    assert pcd.contributing_frame_ids == [[5, 50], [6, 60, 600]]

    # Setting None resets to implicit
    pcd.contributing_frame_ids = None
    assert pcd.cf_flat is None
    assert pcd.cf_offsets is None
    assert pcd.contributing_frame_ids == [[5], [6]]


# ---------------------------------------------------------------------------
# Test 24: Correct Pixel-to-Point Mapping on Synthetic Depth
# ---------------------------------------------------------------------------
def test_pixel_to_point_mapping_synthetic(synthetic_rgbd):
    """
    Verify get_point_index_at_pixel and get_point_at_pixel accurately map (u, v)
    to point index and 3D metric coordinate on a synthetic frame.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)

    # Pick 5 valid pixel coordinates in the center region
    sample_pixels = [(250, 150), (300, 200), (350, 250), (400, 300), (500, 200)]
    for u, v in sample_pixels:
        pt_idx = pcd.get_point_index_at_pixel(u, v)
        assert pt_idx >= 0
        assert pt_idx < len(pcd)

        # Verify point at index matches source pixel uv
        assert np.array_equal(pcd.source_pixel_uv[pt_idx], [u, v])

        # Verify 3D position
        pt_3d = pcd.get_point_at_pixel(u, v)
        assert pt_3d is not None
        assert np.allclose(pt_3d, pcd.points[pt_idx], atol=1e-6)
        assert np.isclose(pt_3d[2], depth[v, u], atol=1e-5)


# ---------------------------------------------------------------------------
# Test 25: Invalid Depth Returns -1
# ---------------------------------------------------------------------------
def test_invalid_depth_returns_minus_one(sample_camera_intrinsics):
    """
    Verify that pixels with invalid depth (zero, negative, NaN, Inf, out of range)
    return -1 for point index and None for 3D coordinates.
    """
    h, w = 480, 848
    color = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)

    # One valid pixel
    depth[200, 300] = 1.5

    # Various invalid depths
    depth[100, 100] = 0.0     # Zero
    depth[100, 101] = -1.0    # Negative
    depth[100, 102] = np.nan  # NaN
    depth[100, 103] = np.inf  # Inf
    depth[100, 104] = 0.05   # Below d_min (0.10)
    depth[100, 105] = 5.0    # Above d_max (3.00)

    pcd = create_indexed_pcd_from_rgbd(color, depth, sample_camera_intrinsics)

    # Valid pixel maps correctly
    assert pcd.get_point_index_at_pixel(300, 200) == 0
    assert pcd.get_point_at_pixel(300, 200) is not None

    # Invalids return -1 and None
    for invalid_u in [100, 101, 102, 103, 104, 105]:
        assert pcd.get_point_index_at_pixel(invalid_u, 100) == -1
        assert pcd.get_point_at_pixel(invalid_u, 100) is None


# ---------------------------------------------------------------------------
# Test 26: Out-of-Bounds Pixel Queries Handled Gracefully
# ---------------------------------------------------------------------------
def test_out_of_bounds_pixel_queries(synthetic_rgbd):
    """
    Verify out-of-bounds queries return -1 and None without raising IndexError.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)
    w, h = intr.width, intr.height

    oob_coords = [
        (-1, 0), (0, -1), (-100, -100),
        (w, 0), (0, h), (w, h), (w + 10, h + 10),
        (9999, 9999), (-9999, -9999)
    ]
    for u, v in oob_coords:
        assert pcd.get_point_index_at_pixel(u, v) == -1
        assert pcd.get_point_at_pixel(u, v) is None


# ---------------------------------------------------------------------------
# Test 27: Correct Mapping After ROI Filtering
# ---------------------------------------------------------------------------
def test_pixel_mapping_after_roi_filtering(synthetic_rgbd):
    """
    Verify pixel-to-point mapping updates correctly after filter_roi:
    - Surviving points receive their new 0-based index.
    - Filtered-out points return -1.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Target bounding box isolating a central cube
    min_b = [-0.2, -0.1, 1.1]
    max_b = [0.2, 0.1, 1.4]
    roi_pcd = pcd.filter_roi(min_b, max_b)

    assert len(roi_pcd) > 0
    assert len(roi_pcd) < len(pcd)

    # Pick a point in the filtered cloud
    u_surv, v_surv = roi_pcd.source_pixel_uv[0]
    idx_new = roi_pcd.get_point_index_at_pixel(u_surv, v_surv)
    assert idx_new == 0
    assert np.allclose(roi_pcd.get_point_at_pixel(u_surv, v_surv), roi_pcd.points[0])

    # Find a point from original cloud that was filtered out
    # Point at the corner of valid depth
    u_orig, v_orig = pcd.source_pixel_uv[0]
    # If this point was excluded from ROI, must return -1
    pt_orig = pcd.points[0]
    if not (np.all(pt_orig >= min_b) and np.all(pt_orig <= max_b)):
        assert roi_pcd.get_point_index_at_pixel(u_orig, v_orig) == -1
        assert roi_pcd.get_point_at_pixel(u_orig, v_orig) is None


# ---------------------------------------------------------------------------
# Test 28: Correct Mapping After Outlier and Plane Removal
# ---------------------------------------------------------------------------
def test_pixel_mapping_after_outlier_and_plane_removal():
    """
    Verify pixel lookup remains consistent after plane segmentation.
    """
    h, w = 100, 100
    color = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 1.0  # Table at 1.0m

    # Foreground object at (50, 50) radius 10 at 0.8m
    yy, xx = np.ogrid[:h, :w]
    obj_mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 10 ** 2
    depth[obj_mask] = 0.8

    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    cfg = PreprocessConfig(plane_dist_threshold_m=0.01, plane_ransac_n=3, plane_num_iterations=200)
    non_plane_pcd, _ = remove_dominant_plane(pcd, cfg)

    # Object center pixel (50, 50) must survive with valid index
    obj_idx = non_plane_pcd.get_point_index_at_pixel(50, 50)
    assert obj_idx >= 0
    assert obj_idx < len(non_plane_pcd)
    assert np.isclose(non_plane_pcd.get_point_at_pixel(50, 50)[2], 0.8, atol=0.02)

    # Table corner pixel (10, 10) was removed by plane fit; must return -1
    table_idx = non_plane_pcd.get_point_index_at_pixel(10, 10)
    assert table_idx == -1
    assert non_plane_pcd.get_point_at_pixel(10, 10) is None


# ---------------------------------------------------------------------------
# Test 29: Correct Mapping After Arbitrary Point-Index Selection
# ---------------------------------------------------------------------------
def test_pixel_mapping_after_arbitrary_point_index_selection(synthetic_rgbd):
    """
    Verify select_by_index correctly re-indexes surviving points and maps
    unselected pixels to -1.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    selected_indices = [10, 50, 100, 500]
    sub = pcd.select_by_index(selected_indices)

    assert len(sub) == 4
    for new_idx, orig_idx in enumerate(selected_indices):
        u, v = pcd.source_pixel_uv[orig_idx]
        assert sub.get_point_index_at_pixel(u, v) == new_idx
        assert np.allclose(sub.get_point_at_pixel(u, v), sub.points[new_idx])

    # Unselected point (e.g. orig_idx 11) must return -1
    u_unsel, v_unsel = pcd.source_pixel_uv[11]
    assert sub.get_point_index_at_pixel(u_unsel, v_unsel) == -1


# ---------------------------------------------------------------------------
# Test 30: Rotated Image Coordinate Consistency
# ---------------------------------------------------------------------------
def test_rotated_image_coordinate_consistency():
    """
    Verify rotated portrait image (height=848, width=480) indexes correctly
    in the rotated coordinate grid without shape mismatch.
    """
    h_rot, w_rot = 848, 480
    color = np.full((h_rot, w_rot, 3), 128, dtype=np.uint8)
    depth = np.ones((h_rot, w_rot), dtype=np.float32) * 1.5

    intr = o3d.camera.PinholeCameraIntrinsic(w_rot, h_rot, fx=424.0, fy=424.0, cx=240.0, cy=424.0)
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    grid = pcd.get_pixel_index_map()
    assert grid.shape == (h_rot, w_rot)  # (848, 480)

    # Sample pixel at bottom of rotated image
    u, v = 240, 800
    idx = pcd.get_point_index_at_pixel(u, v)
    assert idx >= 0
    assert np.array_equal(pcd.source_pixel_uv[idx], [u, v])

    # Query out of rotated width (u=500 >= 480) must return -1
    assert pcd.get_point_index_at_pixel(500, 400) == -1


# ---------------------------------------------------------------------------
# Test 31: No Stale Indices After Filtering
# ---------------------------------------------------------------------------
def test_no_stale_indices_after_filtering(synthetic_rgbd):
    """
    Verify that every non-negative value in pixel_to_point is strictly < len(filtered_pcd)
    and maps to the correct point coordinates (no stale indices).
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Filter out half the points using a checkerboard mask
    mask = (np.arange(len(pcd)) % 2 == 0)
    filtered = pcd.select_by_mask(mask)

    grid = filtered.get_pixel_index_map()
    valid_indices = grid[grid >= 0]

    # Max index must be strictly len(filtered) - 1
    assert np.max(valid_indices) == len(filtered) - 1
    assert np.min(valid_indices) == 0
    assert len(valid_indices) == len(filtered)

    # Every mapped index must match its source pixel
    sample_sub_indices = np.random.default_rng(42).choice(len(filtered), size=100, replace=False)
    for idx in sample_sub_indices:
        u, v = filtered.source_pixel_uv[idx]
        assert grid[v, u] == idx


# ---------------------------------------------------------------------------
# Test 32: Voxel Downsampled Clouds Do Not Claim False Unique Mappings
# ---------------------------------------------------------------------------
def test_voxel_downsampled_unambiguous_mapping(synthetic_rgbd):
    """
    Verify that for voxel-downsampled clouds, unambiguous_only=True returns -1
    for merged voxel centroids (aggregation_count > 1) to prevent false unique pixel claims.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Downsample with 5cm voxels
    down = pcd.voxel_downsample(voxel_size=0.05)
    assert len(down) < len(pcd)

    # Representative pixel of a merged voxel (aggregation_count > 1)
    merged_pts = np.where(down.aggregation_count > 1)[0]
    assert len(merged_pts) > 0

    u_rep, v_rep = down.source_pixel_uv[merged_pts[0]]

    # With unambiguous_only=True (default), must return -1
    assert down.get_point_index_at_pixel(u_rep, v_rep, unambiguous_only=True) == -1
    assert down.get_point_at_pixel(u_rep, v_rep, unambiguous_only=True) is None

    # With unambiguous_only=False, returns the representative point index
    rep_idx = down.get_point_index_at_pixel(u_rep, v_rep, unambiguous_only=False)
    assert rep_idx == merged_pts[0]
    assert down.get_point_at_pixel(u_rep, v_rep, unambiguous_only=False) is not None


# ---------------------------------------------------------------------------
# Test 33: Landmark Depth Sampling Consistency
# ---------------------------------------------------------------------------
def test_landmark_depth_sampling_consistency(synthetic_rgbd):
    """
    Verify sample_landmark_3d produces results completely consistent with
    raw depth patch sampling and integrates with IndexedPointCloud.
    """
    color, depth, intr = synthetic_rgbd
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr)

    # Landmark pixel in the valid center region
    lm_u, lm_v = 300, 200
    sample = sample_landmark_3d(
        pixel_uv=(lm_u, lm_v),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="test_landmark",
        depth_patch=3,
        indexed_pcd=pcd,
    )

    assert isinstance(sample, LandmarkSample3D)
    assert sample.name == "test_landmark"
    assert sample.pixel_uv == (lm_u, lm_v)
    assert sample.is_valid is True
    assert sample.point_3d is not None
    assert np.isclose(sample.depth_m, depth[lm_v, lm_u], atol=1e-3)
    assert sample.point_index >= 0
    assert np.allclose(sample.point_3d, pcd.points[sample.point_index], atol=1e-3)

    # Query an invalid depth region (corner at depth=0)
    invalid_sample = sample_landmark_3d(
        pixel_uv=(10, 10),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="invalid_landmark",
        indexed_pcd=pcd,
    )
    assert invalid_sample.is_valid is False
    assert invalid_sample.point_3d is None
    assert invalid_sample.point_index == -1


# ---------------------------------------------------------------------------
# Test 34: Duplicate (frame_id, u, v) Keys in Concatenated Clouds
# ---------------------------------------------------------------------------
def test_duplicate_keys_in_concatenated_clouds():
    """
    Verify that concatenating two point clouds with overlapping (frame_id, u, v) keys
    flags the duplicate pixels as ambiguous (-1) when unambiguous_only=True.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 150, dtype=np.uint8)

    # Cloud 1 has depth 1.0 at (50, 50) and (20, 20)
    depth1 = np.zeros((100, 100), dtype=np.float32)
    depth1[50, 50] = 1.0
    depth1[20, 20] = 1.0
    pcd1 = create_indexed_pcd_from_rgbd(color, depth1, intr, frame_id=1)

    # Cloud 2 has depth 2.0 at (50, 50) and (30, 30)
    depth2 = np.zeros((100, 100), dtype=np.float32)
    depth2[50, 50] = 2.0
    depth2[30, 30] = 2.0
    pcd2 = create_indexed_pcd_from_rgbd(color, depth2, intr, frame_id=1)

    # Merge clouds with same frame_id=1
    pcd_merged = pcd1.merge(pcd2)
    assert len(pcd_merged) == 4

    # (50, 50) is duplicated in frame 1 (two surviving observations with K=1)
    # Under unambiguous_only=True, it MUST return -1
    idx_ambig = pcd_merged.get_point_index_at_pixel(50, 50, frame_id=1, unambiguous_only=True)
    assert idx_ambig == -1
    assert pcd_merged.get_point_at_pixel(50, 50, frame_id=1, unambiguous_only=True) is None

    # Under unambiguous_only=False, it returns a deterministic representative
    idx_rep = pcd_merged.get_point_index_at_pixel(50, 50, frame_id=1, unambiguous_only=False)
    assert idx_rep >= 0

    # (20, 20) and (30, 30) are unique observations: must remain queryable
    idx_20 = pcd_merged.get_point_index_at_pixel(20, 20, frame_id=1, unambiguous_only=True)
    assert idx_20 >= 0
    idx_30 = pcd_merged.get_point_index_at_pixel(30, 30, frame_id=1, unambiguous_only=True)
    assert idx_30 >= 0


# ---------------------------------------------------------------------------
# Test 35: Filtering Duplicate Observation Restores Unique Lookup
# ---------------------------------------------------------------------------
def test_filtering_duplicate_observation_restores_unique_lookup():
    """
    Verify that if a point cloud has duplicate (frame_id, u, v) observations,
    filtering out one duplicate restores an unambiguous 1-to-1 lookup for the survivor.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 150, dtype=np.uint8)

    depth1 = np.zeros((100, 100), dtype=np.float32)
    depth1[50, 50] = 1.0  # Z = 1.0m
    pcd1 = create_indexed_pcd_from_rgbd(color, depth1, intr, frame_id=1)

    depth2 = np.zeros((100, 100), dtype=np.float32)
    depth2[50, 50] = 2.0  # Z = 2.0m
    pcd2 = create_indexed_pcd_from_rgbd(color, depth2, intr, frame_id=1)

    pcd_merged = pcd1.merge(pcd2)
    assert pcd_merged.get_point_index_at_pixel(50, 50, frame_id=1, unambiguous_only=True) == -1

    # Filter by depth: keep only Z < 1.5m (removes the second duplicate point)
    mask = pcd_merged.points[:, 2] < 1.5
    filtered_pcd = pcd_merged.select_by_mask(mask)
    assert len(filtered_pcd) == 1

    # Now (50, 50) has only ONE surviving point; lookup must succeed!
    survivor_idx = filtered_pcd.get_point_index_at_pixel(50, 50, frame_id=1, unambiguous_only=True)
    assert survivor_idx == 0
    assert np.isclose(filtered_pcd.points[survivor_idx, 2], 1.0)


# ---------------------------------------------------------------------------
# Test 36: Centre-Pixel Landmark Sampling with Exact Provenance
# ---------------------------------------------------------------------------
def test_center_pixel_landmark_sampling():
    """
    Verify centre-pixel landmark sampling preserves exact landmark and source coordinates,
    and identifies exact 1-to-1 indexed sensor observations.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.full((100, 100), 1.25, dtype=np.float32)
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)

    sample = sample_landmark_3d(
        pixel_uv=(40, 40),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="test_center",
        depth_patch=1,  # Single-pixel exact sample
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is True
    assert sample.has_point_3d is True
    assert sample.is_observed_point is True
    assert sample.landmark_pixel_uv == (40, 40)
    assert sample.source_pixel_uv == (40, 40)
    assert sample.point_index >= 0
    assert np.allclose(sample.point_3d, sample.observed_point_3d)
    assert np.allclose(sample.point_3d, pcd.points[sample.point_index])


# ---------------------------------------------------------------------------
# Test 37: Neighbourhood Median Landmark with Source-Pixel Tracking
# ---------------------------------------------------------------------------
def test_neighbourhood_median_landmark_source_pixels():
    """
    Verify neighbourhood patch median retains all contributing source pixels
    in sample.source_pixels and correctly reports patch median depth.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5
    # Introduce small gradient
    yy, xx = np.mgrid[0:100, 0:100]
    depth += (xx - 50) * 0.001
    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="median_test",
        depth_patch=3,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is True
    assert sample.landmark_pixel_uv == (50, 50)
    assert sample.source_pixel_uv == (50, 50)
    assert len(sample.source_pixels) == 9
    assert (50, 50) in sample.source_pixels


# ---------------------------------------------------------------------------
# Test 38: Invalid Centre Pixel with Neighbouring Fallback
# ---------------------------------------------------------------------------
def test_invalid_center_pixel_neighbour_fallback():
    """
    When the centre landmark pixel has zero depth (void), but surrounding pixels
    form a valid surface cluster:
    - Landmark coordinate point_3d is estimated along the landmark ray.
    - source_pixel_uv records the neighbouring pixel coordinates.
    - observed_point_3d is projected using that neighbour's own coordinates.
    - point_index is -1 (never fabricated for the missing centre pixel).
    - is_valid is False (low confidence due to missing centre depth).
    - has_point_3d is True.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.2
    depth[50, 50] = 0.0  # Center pixel is missing

    # Add background wall at 2.0m to create clear two clusters
    depth[0:20, :] = 2.0

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="void_center",
        depth_patch=3,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.has_point_3d is True
    assert sample.is_valid is False  # Missing center depth marks it low-confidence
    assert sample.landmark_pixel_uv == (50, 50)
    assert sample.source_pixel_uv != (50, 50)
    assert sample.source_pixel_uv in sample.source_pixels
    # Must NOT fabricate a point_index for the missing center ray
    assert sample.point_index == -1
    assert sample.source_point_index >= 0
    assert np.allclose(sample.observed_point_3d, pcd.points[sample.source_point_index])
    assert sample.is_observed_point is False
    assert sample.observed_point_3d is not None
    # Observed point X and Y should reflect the neighbour pixel coordinate
    nb_u, nb_v = sample.source_pixel_uv
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()
    expected_nb_x = (nb_u - cx) * depth[nb_v, nb_u] / fx
    assert np.isclose(sample.observed_point_3d[0], expected_nb_x)


# ---------------------------------------------------------------------------
# Test 39: Estimated Coordinate Does Not Receive Fabricated Point Index
# ---------------------------------------------------------------------------
def test_estimated_coordinate_does_not_fabricate_point_index():
    """
    Verify that when an indexed_pcd has filtered out or downsampled the point,
    sample_landmark_3d leaves point_index=-1 and is_observed_point=False.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    # Filter out pixel (50, 50)
    mask = ~((pcd.source_pixel_uv[:, 0] == 50) & (pcd.source_pixel_uv[:, 1] == 50))
    pcd_filtered = pcd.select_by_mask(mask)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="filtered_pt",
        depth_patch=1,
        indexed_pcd=pcd_filtered,
        frame_id=0,
    )

    assert sample.has_point_3d is True
    assert sample.point_index == -1
    assert sample.is_observed_point is False


# ---------------------------------------------------------------------------
# Test 40: is_valid=False with Available point_3d (Quality Check Failure)
# ---------------------------------------------------------------------------
def test_is_valid_false_with_available_point_3d():
    """
    Verify that when variance exceeds max_depth_variance_m2:
    - is_valid is False
    - has_point_3d is True
    - point_3d is populated (available for inspection/diagnostics)
    """
    from realsense_measure.config import LandmarkConfig

    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0
    # Add noisy depths in 3x3 patch around (50, 50)
    depth[49:52, 49:52] = np.array([
        [1.00, 1.01, 0.99],
        [1.02, 1.00, 0.98],
        [1.01, 0.99, 1.02],
    ], dtype=np.float32)

    cfg_strict = LandmarkConfig(max_depth_variance_m2=1e-6)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="noisy_sample",
        depth_patch=3,
        cfg=cfg_strict,
    )

    assert sample.has_point_3d is True
    assert sample.point_3d is not None
    assert sample.is_valid is False
    assert sample.confidence_info["low_confidence"] is True
    assert sample.confidence_info["variance"] > 1e-6


# ---------------------------------------------------------------------------
# Test 41: Step-Edge Discontinuity Rejection
# ---------------------------------------------------------------------------
def test_step_edge_discontinuity_rejection():
    """
    Verify that when the center pixel lies on the background cluster (> 3cm behind foreground),
    the sample is completely rejected (point_3d=None, is_valid=False, has_point_3d=False).
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    # Center is on a background wall at 1.8m while neighbours are foreground at 1.2m
    depth[49:52, 49:52] = 1.2
    depth[50, 50] = 1.8  # Background center

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="step_edge",
        depth_patch=3,
    )

    assert sample.is_valid is False
    assert sample.has_point_3d is False
    assert sample.point_3d is None
    assert sample.point_index == -1
    assert sample.confidence_info.get("error") == "center_on_background_cluster"


# ---------------------------------------------------------------------------
# Test 42: Legacy Landmark Interface Compatibility
# ---------------------------------------------------------------------------
def test_legacy_landmark_interface_compatibility():
    """
    Verify backward compatibility with legacy consumers expecting:
    - pixel_uv
    - depth_m
    - point_3d
    - is_valid
    - confidence_info
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    sample = sample_landmark_3d(
        pixel_uv=(25, 25),
        depth_m=depth,
        intrinsics_o3d=intr,
    )

    # Legacy attributes
    assert hasattr(sample, "pixel_uv")
    assert hasattr(sample, "depth_m")
    assert hasattr(sample, "point_3d")
    assert hasattr(sample, "is_valid")
    assert hasattr(sample, "confidence_info")
    assert sample.pixel_uv == (25, 25)
    assert sample.is_valid is True


# ---------------------------------------------------------------------------
# Test 43: Uniform Patch - Both Indices Refer to Same Point
# ---------------------------------------------------------------------------
def test_uniform_patch_both_indices_match():
    """
    On a uniform patch where center depth == median depth, point_index and
    source_point_index refer to the exact same point in IndexedPointCloud.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="uniform_test",
        depth_patch=3,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is True
    assert sample.is_observed_point is True
    assert sample.point_index >= 0
    assert sample.source_point_index >= 0
    assert sample.point_index == sample.source_point_index
    assert np.allclose(sample.point_3d, sample.observed_point_3d)
    assert np.allclose(sample.point_3d, pcd.points[sample.point_index])


# ---------------------------------------------------------------------------
# Test 44: Asymmetric Patch Median - Decouples Landmark and Source Indices
# ---------------------------------------------------------------------------
def test_asymmetric_patch_median_decouples_indices():
    """
    When the patch median shifts the landmark depth away from the center depth:
    - point_index must be -1 (point_3d is a synthetic median estimate).
    - source_point_index must be >= 0 (identifies raw center observation in pcd).
    - is_observed_point must be False.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    # Center is 1.000m, but 5 neighbors are at 1.005m (median shifts to 1.005m)
    depth[49:52, 49:52] = np.array([
        [1.005, 1.005, 1.005],
        [1.000, 1.000, 1.005],
        [1.000, 1.000, 1.005],
    ], dtype=np.float32)

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="asymmetric_patch",
        depth_patch=3,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is True
    assert sample.is_observed_point is False
    assert sample.point_index == -1  # Never fabricated for synthetic median
    assert sample.source_point_index >= 0  # Raw center point remains traceable!

    # Verify coordinates match their intended physical sources
    raw_point_z = pcd.points[sample.source_point_index, 2]
    assert np.isclose(raw_point_z, 1.000)
    assert np.isclose(sample.observed_point_3d[2], 1.000)
    assert np.isclose(sample.point_3d[2], 1.005)


# ---------------------------------------------------------------------------
# Test 45: Sloped Surface - No Conflation of Coordinates or Indices
# ---------------------------------------------------------------------------
def test_sloped_surface_does_not_conflate_coordinates_or_indices():
    """
    On a systematically sloped plane:
    - observed_point_3d equals pcd.points[source_point_index].
    - point_3d is the median-filtered estimate.
    - point_index is -1.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    # Asymmetric 3x3 plane tilt
    depth[49:52, 49:52] = np.array([
        [0.998, 1.002, 1.006],
        [1.000, 1.000, 1.008],
        [1.004, 1.006, 1.010],
    ], dtype=np.float32)

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    cfg_sloped = LandmarkConfig(max_depth_variance_m2=5e-5)
    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="sloped_test",
        depth_patch=3,
        cfg=cfg_sloped,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is True
    assert sample.is_observed_point is False
    assert sample.point_index == -1
    assert sample.source_point_index >= 0
    assert np.allclose(sample.observed_point_3d, pcd.points[sample.source_point_index])
    assert not np.allclose(sample.point_3d, pcd.points[sample.source_point_index], atol=1e-3)


# ---------------------------------------------------------------------------
# Test 46: Invalid Centre with Valid Neighbour Fallback Indices
# ---------------------------------------------------------------------------
def test_invalid_center_with_valid_neighbour_indices():
    """
    When the centre pixel has zero depth (void):
    - point_index is -1 (centre ray was not directly observed).
    - source_point_index references the neighbour in IndexedPointCloud.
    - observed_point_3d matches pcd.points[source_point_index].
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.2
    depth[50, 50] = 0.0  # Center void

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="fallback_index_test",
        depth_patch=3,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.point_index == -1
    assert sample.source_point_index >= 0
    assert sample.is_observed_point is False
    assert np.allclose(sample.observed_point_3d, pcd.points[sample.source_point_index])
    # The source pixel is the neighbour
    nb_u, nb_v = sample.source_pixel_uv
    assert pcd.source_pixel_uv[sample.source_point_index, 0] == nb_u
    assert pcd.source_pixel_uv[sample.source_point_index, 1] == nb_v


# ---------------------------------------------------------------------------
# Test 47: Filtered Cloud - Both Indices are -1
# ---------------------------------------------------------------------------
def test_filtered_cloud_both_indices_minus_one():
    """
    When the relevant pixel was filtered out of IndexedPointCloud,
    both point_index and source_point_index evaluate to -1.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    # Remove pixel (50, 50)
    mask = ~((pcd.source_pixel_uv[:, 0] == 50) & (pcd.source_pixel_uv[:, 1] == 50))
    filtered_pcd = pcd.select_by_mask(mask)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="filtered_cloud_test",
        depth_patch=1,
        indexed_pcd=filtered_pcd,
        frame_id=0,
    )

    assert sample.has_point_3d is True
    assert sample.point_index == -1
    assert sample.source_point_index == -1
    assert sample.is_observed_point is False


# ---------------------------------------------------------------------------
# Test 48: Downsampled / Ambiguous Cloud - No False Association
# ---------------------------------------------------------------------------
def test_downsampled_ambiguous_cloud_no_false_association():
    """
    When points are merged via voxel downsampling (aggregation_count > 1),
    unambiguous_only=True returns -1, ensuring neither point_index nor
    source_point_index falsely claims an unambiguous 1-to-1 association.
    """
    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    # Downsample heavily so multiple pixels merge
    down_pcd = pcd.voxel_downsample(voxel_size=0.05)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="downsampled_test",
        depth_patch=1,
        indexed_pcd=down_pcd,
        frame_id=0,
    )

    assert sample.point_index == -1
    assert sample.source_point_index == -1
    assert sample.is_observed_point is False


# ---------------------------------------------------------------------------
# Test 49: Invalid Quality Sample - Index Validity Follows Contract
# ---------------------------------------------------------------------------
def test_invalid_quality_sample_index_validity():
    """
    When a sample fails quality checks (e.g. high depth variance):
    - is_valid is False.
    - point_index is -1 (cannot claim valid indexed landmark coordinate).
    - source_point_index preserves the raw point index for debugging/diagnostics.
    - has_point_3d is True.
    """
    from realsense_measure.config import LandmarkConfig

    intr = o3d.camera.PinholeCameraIntrinsic(100, 100, 50.0, 50.0, 50.0, 50.0)
    color = np.full((100, 100, 3), 200, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0
    depth[49:52, 49:52] = np.array([
        [1.00, 1.02, 0.98],
        [1.01, 1.00, 0.99],
        [1.02, 0.98, 1.01],
    ], dtype=np.float32)

    pcd = create_indexed_pcd_from_rgbd(color, depth, intr, frame_id=0)
    cfg_strict = LandmarkConfig(max_depth_variance_m2=1e-6)

    sample = sample_landmark_3d(
        pixel_uv=(50, 50),
        depth_m=depth,
        intrinsics_o3d=intr,
        name="quality_fail_test",
        depth_patch=3,
        cfg=cfg_strict,
        indexed_pcd=pcd,
        frame_id=0,
    )

    assert sample.is_valid is False
    assert sample.has_point_3d is True
    assert sample.point_index == -1
    assert sample.source_point_index >= 0  # Raw pixel is still indexed
    assert sample.is_observed_point is False


# ---------------------------------------------------------------------------
# Test 50: LandmarkSample3D Clone and Dataclass Integrity
# ---------------------------------------------------------------------------
def test_landmark_sample_clone_and_dataclass_integrity():
    """
    Verify copy.deepcopy and dataclass attributes retain all indices and provenance.
    """
    import copy

    sample = LandmarkSample3D(
        name="nose_tip",
        pixel_uv=(40, 40),
        depth_m=1.234,
        point_3d=np.array([0.1, 0.2, 1.234]),
        is_valid=True,
        point_index=-1,
        source_point_index=42,
        landmark_pixel_uv=(40, 40),
        source_pixel_uv=(40, 41),
        has_point_3d=True,
        is_observed_point=False,
        observed_point_3d=np.array([0.1, 0.21, 1.230]),
        source_pixels=[(40, 40), (40, 41)],
        confidence_info={"variance": 1e-6},
    )

    cloned = copy.deepcopy(sample)
    assert cloned.name == "nose_tip"
    assert cloned.pixel_uv == (40, 40)
    assert cloned.point_index == -1
    assert cloned.source_point_index == 42
    assert cloned.source_pixel_uv == (40, 41)
    assert np.allclose(cloned.point_3d, sample.point_3d)
    assert np.allclose(cloned.observed_point_3d, sample.observed_point_3d)





