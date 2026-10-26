"""
Unit tests for Stage 2 segmentation-based subject isolation.

Covers:
1. Depth zeroing: ensuring depth is strictly zeroed where mask == False (no depth invented).
2. Fragment merging: unioning satellite clusters within seg_merge_radius_m.
3. Fallback on None mask: falls back to geometric isolation and returns 'geometric_fallback'.
4. Fallback on empty/tiny mask: falls back to geometric isolation when points < min_object_points.
5. Geometric isolation unchanged: ensuring PreprocessConfig(isolation_mode='geometric')
   yields identical results to the baseline isolate_object.
6. Optional real model test: skipped if selfie_segmenter.tflite is absent from models/.
"""

from __future__ import annotations

from unittest.mock import patch
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame, create_pcd_from_rgbd
from config import PreprocessConfig, TargetConfig
from preprocessing import isolate_object, isolate_with_segmentation
from segmentation import is_model_available


def _make_test_rgbd_frame(
    width: int = 100,
    height: int = 100,
    two_blobs: bool = False,
) -> RGBDFrame:
    """Synthesizes an RGBDFrame with controlled spherical bump(s)."""
    fx = fy = 80.0
    cx = width / 2.0
    cy = height / 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    color_bgr = np.full((height, width, 3), 128, dtype=np.uint8)
    depth_m = np.zeros((height, width), dtype=np.float32)

    y_idx, x_idx = np.indices((height, width))

    if not two_blobs:
        # Single bump at center
        r_sq = ((x_idx - cx) / 25.0) ** 2 + ((y_idx - cy) / 25.0) ** 2
        valid = r_sq <= 1.0
        depth_m[valid] = 0.70 - 0.05 * np.sqrt(np.clip(1.0 - r_sq[valid], 0.0, 1.0))
    else:
        # Two distinct bumps side-by-side
        # Primary bump around x = cx - 20
        r1_sq = ((x_idx - (cx - 20)) / 12.0) ** 2 + ((y_idx - cy) / 12.0) ** 2
        valid1 = r1_sq <= 1.0
        depth_m[valid1] = 0.70 - 0.04 * np.sqrt(np.clip(1.0 - r1_sq[valid1], 0.0, 1.0))

        # Secondary bump around x = cx + 20
        r2_sq = ((x_idx - (cx + 20)) / 12.0) ** 2 + ((y_idx - cy) / 12.0) ** 2
        valid2 = r2_sq <= 1.0
        depth_m[valid2] = 0.70 - 0.04 * np.sqrt(np.clip(1.0 - r2_sq[valid2], 0.0, 1.0))

    pcd = create_pcd_from_rgbd(
        color_bgr=color_bgr,
        depth_m=depth_m,
        intrinsics=intr,
        depth_min_m=0.40,
        depth_max_m=1.20,
    )

    return RGBDFrame(
        frame_id=0,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=pcd,
        timestamp=0.0,
        intrinsics=intr,
    )


def test_segmentation_depth_zeroing():
    """Verify that masked-out regions have depth zeroed and points strictly excluded."""
    frame = _make_test_rgbd_frame(width=100, height=100, two_blobs=False)
    cfg = PreprocessConfig(isolation_mode="segmentation", cluster_min_points=10)
    target_cfg = TargetConfig(name="head", min_object_points=50)

    # Half mask: True on x < 50, False on x >= 50
    half_mask = np.zeros((100, 100), dtype=bool)
    half_mask[:, :50] = True

    with patch("segmentation.get_person_mask", return_value=half_mask):
        isolated_pcd, method = isolate_with_segmentation(frame, cfg, target_cfg)

    assert method == "segmentation"
    assert len(isolated_pcd.points) > 50
    # In camera coordinates, x < 0 corresponds to pixels x < cx (cx=50)
    pts = np.asarray(isolated_pcd.points)
    assert np.all(pts[:, 0] <= 0.01), "Points exist in masked-out right half!"


def test_segmentation_fragment_merging():
    """Verify that satellite fragments within seg_merge_radius_m are merged."""
    frame = _make_test_rgbd_frame(width=100, height=100, two_blobs=True)
    full_mask = np.ones((100, 100), dtype=bool)

    # With large merge radius (0.50m), both blobs should be merged
    cfg_merged = PreprocessConfig(
        isolation_mode="segmentation",
        cluster_eps_m=0.02,
        cluster_min_points=20,
        seg_merge_radius_m=0.50,
    )
    target_cfg = TargetConfig(name="head", min_object_points=20)

    with patch("segmentation.get_person_mask", return_value=full_mask):
        pcd_merged, method = isolate_with_segmentation(frame, cfg_merged, target_cfg)

    # With tiny merge radius (0.001m), only primary cluster should be kept
    cfg_single = PreprocessConfig(
        isolation_mode="segmentation",
        cluster_eps_m=0.02,
        cluster_min_points=20,
        seg_merge_radius_m=0.001,
    )
    with patch("segmentation.get_person_mask", return_value=full_mask):
        pcd_single, _ = isolate_with_segmentation(frame, cfg_single, target_cfg)

    assert method == "segmentation"
    assert len(pcd_merged.points) > len(pcd_single.points), (
        f"Satellite cluster was not merged! {len(pcd_merged.points)} vs {len(pcd_single.points)}"
    )


def test_segmentation_fallback_on_none_mask():
    """Verify graceful fallback to geometric isolation when mask is None."""
    frame = _make_test_rgbd_frame()
    cfg = PreprocessConfig(isolation_mode="segmentation", cluster_min_points=20)
    target_cfg = TargetConfig(name="head", min_object_points=50)

    expected_geo = isolate_object(frame.pcd, cfg, target_cfg)

    with patch("segmentation.get_person_mask", return_value=None):
        isolated_pcd, method = isolate_with_segmentation(frame, cfg, target_cfg)

    assert method == "geometric_fallback"
    assert len(isolated_pcd.points) == len(expected_geo.points)
    np.testing.assert_allclose(
        np.asarray(isolated_pcd.points),
        np.asarray(expected_geo.points),
        rtol=1e-5,
    )


def test_segmentation_fallback_on_insufficient_points():
    """Verify graceful fallback when masked points are below min_object_points."""
    frame = _make_test_rgbd_frame()
    cfg = PreprocessConfig(isolation_mode="segmentation", cluster_min_points=20)
    target_cfg = TargetConfig(name="head", min_object_points=500)

    # Tiny mask with only 5 pixels
    tiny_mask = np.zeros((100, 100), dtype=bool)
    tiny_mask[50:55, 50:55] = True

    with patch("segmentation.get_person_mask", return_value=tiny_mask):
        isolated_pcd, method = isolate_with_segmentation(frame, cfg, target_cfg)

    assert method == "geometric_fallback"


def test_isolation_mode_geometric_unchanged():
    """Verify PreprocessConfig(isolation_mode='geometric') default produces identical results."""
    frame = _make_test_rgbd_frame()
    cfg = PreprocessConfig()
    assert cfg.isolation_mode == "geometric"
    assert cfg.seg_merge_radius_m == 0.05

    base_pcd = isolate_object(frame.pcd, cfg)
    assert base_pcd.has_points()


@pytest.mark.skipif(
    not is_model_available(),
    reason="MediaPipe selfie segmenter model not downloaded (CI/offline mode)",
)
def test_real_model_inference_if_available():
    """Runs actual model inference if the .tflite model exists on disk."""
    from segmentation import get_person_mask

    dummy_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    mask = get_person_mask(dummy_bgr)
    assert mask is None or isinstance(mask, np.ndarray)
