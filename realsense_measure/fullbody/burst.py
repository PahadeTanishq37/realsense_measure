"""
realsense_measure/fullbody/burst.py

Per-stop burst processing:
- Temporal median and sample standard deviation of depth.
- Validity count per pixel.
- Depth-edge and flying-pixel removal.
- Conservative small-hole filling.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from realsense_measure.fullbody.config import FullBodyConfig

logger = logging.getLogger(__name__)


def process_stop_burst(
    burst_colors: List[np.ndarray],
    burst_depths: List[np.ndarray],
    cfg: FullBodyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process burst frames for a single stop.

    Parameters
    ----------
    burst_colors:
        List of N color frames (HxWx3 uint8).
    burst_depths:
        List of N depth frames (HxW float32 metres).
    cfg:
        FullBodyConfig instance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (representative_color, filtered_depth_m, depth_std_m, valid_counts)
    """
    if not burst_depths:
        raise ValueError("Cannot process empty burst")

    n_frames = len(burst_depths)
    h, w = burst_depths[0].shape[:2]

    # Select representative color (middle frame)
    mid_idx = n_frames // 2
    rep_color = burst_colors[mid_idx].copy() if burst_colors else np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Temporal statistics
    depth_stack = np.stack(burst_depths, axis=0)  # (N, H, W)
    valid_mask = (
        (depth_stack >= cfg.depth_min_m) &
        (depth_stack <= cfg.depth_max_m) &
        np.isfinite(depth_stack)
    )
    valid_counts = np.sum(valid_mask, axis=0).astype(np.int32)
    min_required = int(np.ceil(n_frames * cfg.burst_min_valid_frac))

    # Compute masked median and std
    fused_depth = np.zeros((h, w), dtype=np.float32)
    depth_std = np.zeros((h, w), dtype=np.float32)

    qualifying = valid_counts >= min_required
    if np.any(qualifying):
        # Sort depth values along time axis for qualifying pixels
        # Fill non-valid with NaN for robust calculation
        clean_stack = depth_stack.copy()
        clean_stack[~valid_mask] = np.nan

        # Fast nanmedian and nanstd
        fused_depth[qualifying] = np.nanmedian(clean_stack[:, qualifying], axis=0)
        # Standard deviation (0 if only 1 valid observation)
        with np.errstate(divide="ignore", invalid="ignore"):
            std_vals = np.nanstd(clean_stack[:, qualifying], axis=0)
        std_vals[np.isnan(std_vals)] = 0.0
        depth_std[qualifying] = std_vals

    # 2. Flying pixel / depth-edge removal
    fused_depth = filter_flying_pixels(
        fused_depth,
        gradient_threshold_m=cfg.flying_pixel_depth_gradient_threshold_m,
    )

    # 3. Conservative small-hole filling (only for pixels not rejected by temporal validity)
    if cfg.hole_fill_max_radius_px > 0:
        fused_depth = fill_small_holes(
            fused_depth,
            max_radius_px=cfg.hole_fill_max_radius_px,
        )
        fused_depth[~qualifying] = 0.0

    return rep_color, fused_depth, depth_std, valid_counts


def filter_flying_pixels(
    depth_m: np.ndarray,
    gradient_threshold_m: float = 0.04,
) -> np.ndarray:
    """
    Zero out flying pixels near occluding boundaries.

    A pixel whose depth differs drastically from neighbors or lies along
    steep depth gradients is rejected as an optical interpolation artifact.
    """
    valid = depth_m > 0.0
    if not np.any(valid):
        return depth_m.copy()

    # Compute morphological gradient / neighbor differences
    out = depth_m.copy()
    
    # 4-connectivity neighbor differences
    up_diff = np.abs(depth_m[1:, :] - depth_m[:-1, :])
    left_diff = np.abs(depth_m[:, 1:] - depth_m[:, :-1])

    # Mark edges where adjacent pixels have steep depth step
    edge_up = (up_diff > gradient_threshold_m) & (depth_m[1:, :] > 0) & (depth_m[:-1, :] > 0)
    edge_left = (left_diff > gradient_threshold_m) & (depth_m[:, 1:] > 0) & (depth_m[:, :-1] > 0)

    # Pixels on either side of sharp depth jump get flagged
    flagged = np.zeros_like(valid, dtype=bool)
    flagged[1:, :] |= edge_up
    flagged[:-1, :] |= edge_up
    flagged[:, 1:] |= edge_left
    flagged[:, :-1] |= edge_left

    # Additional check: isolated pixels with fewer than 3 valid 8-neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    valid_u8 = valid.astype(np.uint8)
    neighbor_count = cv2.filter2D(valid_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    isolated = valid & (neighbor_count < 3)

    out[flagged | isolated] = 0.0
    return out


def fill_small_holes(
    depth_m: np.ndarray,
    max_radius_px: int = 3,
) -> np.ndarray:
    """
    Fill small isolated depth dropouts using a local median filter.
    Only fills pixels where at least 65% of the local kernel has valid depth.
    Does not expand borders or fill large missing body sections.
    """
    ksize = 2 * max_radius_px + 1
    valid = depth_m > 0.0
    missing = ~valid

    if not np.any(valid) or not np.any(missing):
        return depth_m.copy()

    kernel = np.ones((ksize, ksize), dtype=np.float32)
    valid_float = valid.astype(np.float32)
    neighbor_valid_ratio = cv2.filter2D(
        valid_float, -1, kernel / (ksize * ksize), borderType=cv2.BORDER_CONSTANT
    )

    # Candidate pixels for hole filling
    fill_candidates = missing & (neighbor_valid_ratio >= 0.65)
    if not np.any(fill_candidates):
        return depth_m.copy()

    # Median filter on valid depth using scipy.ndimage.median_filter
    from scipy.ndimage import median_filter
    depth_blurred = median_filter(depth_m, size=ksize, mode="reflect")
    
    out = depth_m.copy()
    out[fill_candidates] = depth_blurred[fill_candidates]
    return out
