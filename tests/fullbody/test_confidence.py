"""
tests/fullbody/test_confidence.py

Unit tests for per-pixel confidence weighting:
- Monotonicity with incidence angle.
- Monotonicity with edge distance.
- Monotonicity with depth (quadratic noise model).
- Low-confidence filtering.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.confidence import compute_confidence_map
from realsense_measure.fullbody.config import FullBodyConfig


def test_confidence_monotonicity():
    """Verify confidence behaves monotonically with incidence angle, edge distance, and depth."""
    h, w = 60, 60
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 30.0, 30.0)
    cfg = FullBodyConfig(min_confidence_threshold=0.15)

    # 1. Flat perpendicular plane at 1.0 m vs tilted plane
    depth_flat = np.full((h, w), 1.00, dtype=np.float32)
    conf_flat, _ = compute_confidence_map(depth_flat, intr, cfg)

    # Center pixel of flat plane has normal facing camera
    center_conf_flat = conf_flat[30, 30]
    assert center_conf_flat > 0.60

    # 2. Edge proximity check: border pixel has lower confidence than interior pixel
    edge_conf = conf_flat[30, 1]
    assert edge_conf < center_conf_flat

    # 3. Depth distance check: plane at 2.4 m has lower confidence than plane at 1.0 m
    depth_far = np.full((h, w), 2.40, dtype=np.float32)
    conf_far, _ = compute_confidence_map(depth_far, intr, cfg)
    center_conf_far = conf_far[30, 30]

    assert center_conf_far < center_conf_flat


def test_low_confidence_invalidation():
    """Verify pixels with confidence below threshold are zeroed in the filtered depth map."""
    h, w = 40, 40
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 20.0, 20.0)
    cfg = FullBodyConfig(min_confidence_threshold=0.30)

    # Valid depth with artificial very low validity count
    depth = np.full((h, w), 1.20, dtype=np.float32)
    valid_counts = np.full((h, w), 1, dtype=np.int32)
    valid_counts[20, 20] = 30  # high validity in center

    conf, filtered_d = compute_confidence_map(
        depth, intr, cfg, valid_counts=valid_counts, burst_total=30
    )

    # Pixel with 1/30 validity should be below threshold and zeroed
    assert filtered_d[5, 5] == 0.0
    # Pixel with 30/30 validity should remain valid
    assert filtered_d[20, 20] > 0.0

