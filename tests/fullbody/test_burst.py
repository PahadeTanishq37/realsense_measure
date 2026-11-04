"""
tests/fullbody/test_burst.py

Unit tests for per-stop burst processing:
- Temporal median and sample standard deviation.
- Validity threshold (< 50% valid frames zeroed out).
- Flying pixel filtering on steep edge gradients.
- Conservative small hole filling.
"""

from __future__ import annotations

import numpy as np
import pytest

from realsense_measure.fullbody.burst import (
    fill_small_holes,
    filter_flying_pixels,
    process_stop_burst,
)
from realsense_measure.fullbody.config import FullBodyConfig


def test_burst_temporal_median_and_std():
    """Verify burst processing computes median and sample standard deviation accurately."""
    cfg = FullBodyConfig(depth_min_m=0.4, depth_max_m=2.5, burst_min_valid_frac=0.5)
    h, w = 40, 40

    # 10 frames with ground truth depth 1.20 m + small Gaussian noise
    np.random.seed(42)
    burst_depths = [
        np.full((h, w), 1.20, dtype=np.float32) + np.random.normal(0, 0.005, (h, w)).astype(np.float32)
        for _ in range(10)
    ]
    burst_colors = [np.full((h, w, 3), 128, dtype=np.uint8) for _ in range(10)]

    rep_color, fused_depth, depth_std, valid_counts = process_stop_burst(burst_colors, burst_depths, cfg)

    assert fused_depth.shape == (h, w)
    assert np.all(valid_counts == 10)
    # Median should be very close to 1.20 m
    assert np.isclose(np.mean(fused_depth), 1.20, atol=0.005)
    # Std should be approximately 0.005 m
    assert np.isclose(np.mean(depth_std), 0.005, atol=0.003)


def test_burst_rejects_pixels_below_min_valid_fraction():
    """Pixels with valid depth in less than 50% of frames are rejected (set to 0.0)."""
    cfg = FullBodyConfig(depth_min_m=0.4, depth_max_m=2.5, burst_min_valid_frac=0.5)
    h, w = 20, 20

    # 10 frames: pixel (10, 10) only has valid depth in 3 frames (< 50%)
    burst_depths = []
    for i in range(10):
        d = np.full((h, w), 1.30, dtype=np.float32)
        if i >= 3:
            d[10, 10] = 0.0  # Invalid in 7 frames
        burst_depths.append(d)
    burst_colors = [np.full((h, w, 3), 100, dtype=np.uint8) for _ in range(10)]

    _, fused_depth, _, valid_counts = process_stop_burst(burst_colors, burst_depths, cfg)

    assert valid_counts[10, 10] == 3
    assert fused_depth[10, 10] == 0.0
    # Other pixels should be valid
    assert fused_depth[5, 5] > 1.25


def test_flying_pixel_filtering():
    """Verify sharp depth edge jumps are stripped as flying pixel artifacts."""
    h, w = 30, 30
    depth = np.full((h, w), 1.00, dtype=np.float32)
    # Right half is far background at 2.20 m (step of 1.2 m > 0.04 m threshold)
    depth[:, 15:] = 2.20

    filtered = filter_flying_pixels(depth, gradient_threshold_m=0.04)

    # Boundary pixels along column 14 and 15 should be zeroed out
    assert np.all(filtered[:, 14] == 0.0)
    assert np.all(filtered[:, 15] == 0.0)
    # Flat interior regions should remain untouched
    assert np.all(filtered[:, 5] == 1.00)
    assert np.all(filtered[:, 25] == 2.20)


def test_small_hole_filling():
    """Verify isolated single-pixel dropouts are filled without expanding outer boundary."""
    h, w = 30, 30
    depth = np.full((h, w), 1.25, dtype=np.float32)
    # Create isolated 1-pixel hole in interior
    depth[15, 15] = 0.0
    # Outer margin is completely missing (unobserved outer region)
    depth[:5, :] = 0.0

    filled = fill_small_holes(depth, max_radius_px=2)

    # Interior small hole should be filled with neighbor depth 1.25 m
    assert np.isclose(filled[15, 15], 1.25, atol=0.01)
    # Outer unobserved region must NOT be filled
    assert np.all(filled[:4, :] == 0.0)

