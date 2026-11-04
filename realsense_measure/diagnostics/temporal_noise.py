"""
realsense_measure/diagnostics/temporal_noise.py
================================================
Temporal depth noise evaluation and fair raw vs filtered jitter comparison.

Calculates pixel-wise and regional temporal depth statistics across burst captures:
- Per-pixel sample standard deviation (N - 1 degrees of freedom)
- Per-pixel linearly detrended residual standard deviation (N - 2 degrees of freedom)
- Stable-pixel masking (pixels valid in >= 90% of frames within [depth_min_m, depth_max_m])
- Mean, median, and 95th-percentile regional noise metrics across stable pixels
- Three distinct evaluation paths:
    1. Raw individual metrics (using raw stream's own stable-pixel mask)
    2. Filtered individual metrics (using filtered stream's own stable-pixel mask)
    3. Matched-population comparison (evaluated strictly on the intersection of raw and filtered stable masks)

Scientific Methodology:
-----------------------
1. Invalid observations (depth <= 0, depth < depth_min_m, depth > depth_max_m, NaN, Inf)
   are strictly treated as missing data (NaN) and excluded from variance sums. Zero depth
   is NEVER imputed as 0.0m.
2. Stable pixels are identified using a temporal intersection rule: at least 90% of burst
   frames must contain valid depth observations, and at least 3 valid observations are
   required to permit 2 degrees of freedom (slope and intercept) during linear detrending.
3. Standard temporal standard deviation uses (N - 1) degrees of freedom:
       s = sqrt( (1 / (N - 1)) * sum_i (z_i - z_bar)^2 )
4. Linearly detrended residual standard deviation fits z(t) = a * t + b via ordinary least
   squares and evaluates residuals with (N - 2) degrees of freedom:
       s_detrended = sqrt( (1 / (N - 2)) * sum_i (z_i - (a * t_i + b))^2 )
   Timestamps are normalized by subtracting t[0] upfront to guarantee numerical stability
   and avoid floating-point cancellation on absolute epoch timestamps.
5. Linear detrending is NOT a complete removal of human motion. Non-rigid deformations,
   lateral limb translation, and oscillatory sway remain present in living subjects.
   Static planar target trials provide the true ground-truth sensor noise floor.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_float(val: Any, decimals: int = 3) -> Optional[float]:
    """Ensure a float value is finite and rounded, returning None if NaN or Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if not np.isfinite(f) or math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


@dataclass
class SingleStreamNoiseMetrics:
    """Quantitative temporal noise metrics for a single depth stream (raw or filtered)."""
    stream_name: str
    num_frames: int
    num_total_pixels: int
    num_stable_pixels: int
    stable_pixel_ratio_pct: float
    mean_temporal_validity_pct: float

    # Degrees of freedom conventions
    dof_convention_standard: str = "N - 1"
    dof_convention_detrended: str = "N - 2"

    # Standard temporal noise (mm)
    mean_std_mm: Optional[float] = None
    median_std_mm: Optional[float] = None
    p95_std_mm: Optional[float] = None
    min_std_mm: Optional[float] = None
    max_std_mm: Optional[float] = None

    # Linearly detrended residual noise (mm)
    mean_detrended_std_mm: Optional[float] = None
    median_detrended_std_mm: Optional[float] = None
    p95_detrended_std_mm: Optional[float] = None
    min_detrended_std_mm: Optional[float] = None
    max_detrended_std_mm: Optional[float] = None

    # Binary stable mask (H, W) and 2D spatial maps (optional, omitted from default JSON serialization)
    stable_mask: Optional[np.ndarray] = field(default=None, repr=False)
    std_map_mm: Optional[np.ndarray] = field(default=None, repr=False)
    detrended_std_map_mm: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, include_maps: bool = False) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary with strict non-finite sanitization."""
        d = {
            "stream_name": self.stream_name,
            "num_frames": int(self.num_frames),
            "num_total_pixels": int(self.num_total_pixels),
            "num_stable_pixels": int(self.num_stable_pixels),
            "stable_pixel_ratio_pct": _sanitize_float(self.stable_pixel_ratio_pct, 2),
            "mean_temporal_validity_pct": _sanitize_float(self.mean_temporal_validity_pct, 2),
            "dof_convention_standard": self.dof_convention_standard,
            "dof_convention_detrended": self.dof_convention_detrended,
            "mean_std_mm": _sanitize_float(self.mean_std_mm, 3),
            "median_std_mm": _sanitize_float(self.median_std_mm, 3),
            "p95_std_mm": _sanitize_float(self.p95_std_mm, 3),
            "min_std_mm": _sanitize_float(self.min_std_mm, 3),
            "max_std_mm": _sanitize_float(self.max_std_mm, 3),
            "mean_detrended_std_mm": _sanitize_float(self.mean_detrended_std_mm, 3),
            "median_detrended_std_mm": _sanitize_float(self.median_detrended_std_mm, 3),
            "p95_detrended_std_mm": _sanitize_float(self.p95_detrended_std_mm, 3),
            "min_detrended_std_mm": _sanitize_float(self.min_detrended_std_mm, 3),
            "max_detrended_std_mm": _sanitize_float(self.max_detrended_std_mm, 3),
        }
        if include_maps:
            if self.std_map_mm is not None:
                d["std_map_mm"] = np.where(np.isfinite(self.std_map_mm), self.std_map_mm, None).tolist()
            if self.detrended_std_map_mm is not None:
                d["detrended_std_map_mm"] = np.where(
                    np.isfinite(self.detrended_std_map_mm), self.detrended_std_map_mm, None
                ).tolist()
        return d


@dataclass
class MatchedPopulationComparison:
    """Fair matched-pixel comparison evaluated exclusively over the intersection of raw and filtered stable masks."""
    matched_pixel_count: int
    raw_stable_retention_pct: float
    filtered_stable_retention_pct: float

    # Raw metrics on matched population
    raw_mean_std_mm: Optional[float] = None
    raw_median_std_mm: Optional[float] = None
    raw_mean_detrended_std_mm: Optional[float] = None
    raw_median_detrended_std_mm: Optional[float] = None

    # Filtered metrics on matched population
    filtered_mean_std_mm: Optional[float] = None
    filtered_median_std_mm: Optional[float] = None
    filtered_mean_detrended_std_mm: Optional[float] = None
    filtered_median_detrended_std_mm: Optional[float] = None

    # Denoising metrics on matched population
    ratio_filtered_to_raw_mean_jitter: Optional[float] = None
    denoising_reduction_pct: Optional[float] = None
    status: str = "valid"  # "valid", "empty_intersection", "zero_raw_jitter"

    def to_dict(self) -> Dict[str, Any]:
        """Convert matched comparison to a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "matched_pixel_count": int(self.matched_pixel_count),
            "raw_stable_retention_pct": _sanitize_float(self.raw_stable_retention_pct, 2),
            "filtered_stable_retention_pct": _sanitize_float(self.filtered_stable_retention_pct, 2),
            "raw_mean_std_mm": _sanitize_float(self.raw_mean_std_mm, 3),
            "raw_median_std_mm": _sanitize_float(self.raw_median_std_mm, 3),
            "raw_mean_detrended_std_mm": _sanitize_float(self.raw_mean_detrended_std_mm, 3),
            "raw_median_detrended_std_mm": _sanitize_float(self.raw_median_detrended_std_mm, 3),
            "filtered_mean_std_mm": _sanitize_float(self.filtered_mean_std_mm, 3),
            "filtered_median_std_mm": _sanitize_float(self.filtered_median_std_mm, 3),
            "filtered_mean_detrended_std_mm": _sanitize_float(self.filtered_mean_detrended_std_mm, 3),
            "filtered_median_detrended_std_mm": _sanitize_float(self.filtered_median_detrended_std_mm, 3),
            "ratio_filtered_to_raw_mean_jitter": _sanitize_float(self.ratio_filtered_to_raw_mean_jitter, 4),
            "denoising_reduction_pct": _sanitize_float(self.denoising_reduction_pct, 2),
        }


@dataclass
class TemporalNoiseComparison:
    """Comprehensive comparison report incorporating individual and matched-population metrics."""
    raw: SingleStreamNoiseMetrics
    filtered: SingleStreamNoiseMetrics
    matched: MatchedPopulationComparison

    # Backward compatibility fields (populated from matched population)
    ratio_filtered_to_raw_mean_jitter: Optional[float] = None
    jitter_reduction_pct: Optional[float] = None
    motion_centroid_displacement_mm: Optional[float] = None
    motion_notice: str = (
        "Linear detrending isolates monotonic axial drift from high-frequency stereo disparity jitter. "
        "It does not represent complete removal of non-rigid human breathing or swaying. "
        "Static planar targets provide the unconfounded sensor noise baseline."
    )

    def to_dict(self, include_maps: bool = False) -> Dict[str, Any]:
        """Convert comparison to a JSON-serializable dictionary."""
        return {
            "raw": self.raw.to_dict(include_maps=include_maps),
            "filtered": self.filtered.to_dict(include_maps=include_maps),
            "matched": self.matched.to_dict(),
            "ratio_filtered_to_raw_mean_jitter": _sanitize_float(self.ratio_filtered_to_raw_mean_jitter, 4),
            "jitter_reduction_pct": _sanitize_float(self.jitter_reduction_pct, 2),
            "motion_centroid_displacement_mm": _sanitize_float(self.motion_centroid_displacement_mm, 2),
            "motion_notice": self.motion_notice,
        }


def compute_temporal_depth_noise(
    depth_frames: Union[np.ndarray, List[np.ndarray]],
    timestamps: Optional[Union[np.ndarray, List[float]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    depth_min_m: float = 0.40,
    depth_max_m: float = 3.00,
    stable_threshold: float = 0.90,
    min_valid_samples: int = 3,
    stream_name: str = "depth",
    return_maps: bool = False,
) -> SingleStreamNoiseMetrics:
    """
    Compute pixel-level temporal standard deviation and linearly detrended residual noise.

    Parameters
    ----------
    depth_frames : Union[np.ndarray, List[np.ndarray]]
        Sequence of depth maps of shape (K, H, W) in metres, or list of K (H, W) arrays.
    timestamps : Optional[Union[np.ndarray, List[float]]]
        Sequence of K timestamps in seconds. If None, normalized frame indices [0, 1, ..., K-1]
        are used.
    roi_mask : Optional[np.ndarray]
        Binary mask of shape (H, W) indicating the spatial region of interest (e.g. subject
        silhouette or planar target). If None, all pixels are considered.
    depth_min_m : float
        Minimum valid depth threshold in metres (inclusive).
    depth_max_m : float
        Maximum valid depth threshold in metres (inclusive).
    stable_threshold : float
        Minimum fraction of frames in which a pixel must be valid to be classified as stable
        (default 0.90 for 90%).
    min_valid_samples : int
        Minimum number of valid temporal observations required for linear detrending (default 3,
        yielding N - 2 >= 1 degree of freedom).
    stream_name : str
        Name of depth stream ("raw", "filtered", etc.).
    return_maps : bool
        If True, stores the full (H, W) spatial noise maps in the returned dataclass.

    Returns
    -------
    SingleStreamNoiseMetrics
        Calculated temporal noise metrics with attached binary stable_mask.
    """
    if isinstance(depth_frames, list):
        if len(depth_frames) == 0:
            raise ValueError("depth_frames list cannot be empty.")
        depth_arr = np.stack(depth_frames, axis=0).astype(np.float32)
    else:
        depth_arr = np.asarray(depth_frames, dtype=np.float32)

    if depth_arr.ndim != 3:
        raise ValueError(f"depth_frames must have shape (K, H, W), got {depth_arr.shape}")

    K, H, W = depth_arr.shape
    total_pixels = H * W

    # Prepare timestamps with t0-normalization for strict numerical stability
    if timestamps is not None:
        ts_arr = np.asarray(timestamps, dtype=np.float64).flatten()
        if len(ts_arr) != K:
            raise ValueError(f"Length of timestamps ({len(ts_arr)}) must match frame count ({K}).")
        # Subtract t0 to center timestamps near zero, preventing float precision loss
        ts_arr = ts_arr - ts_arr[0]
    else:
        ts_arr = np.arange(K, dtype=np.float64)

    # 1. Identify valid observations
    valid_mask = (
        (depth_arr >= depth_min_m)
        & (depth_arr <= depth_max_m)
        & np.isfinite(depth_arr)
        & (depth_arr > 0.0)
    )

    # Convert invalid observations to NaN for clean separation
    depth_clean = np.where(valid_mask, depth_arr, np.nan)

    # Valid observation count per pixel across the K frames
    valid_counts = np.sum(valid_mask, axis=0)  # shape (H, W)
    valid_ratios = valid_counts.astype(np.float32) / float(K)

    # Stable pixel criterion
    stable_mask = (valid_ratios >= stable_threshold) & (valid_counts >= min_valid_samples)
    if roi_mask is not None:
        roi_bool = roi_mask.astype(bool)
        if roi_bool.shape != (H, W):
            raise ValueError(f"roi_mask shape {roi_bool.shape} does not match frame shape {(H, W)}")
        stable_mask = stable_mask & roi_bool
        active_pixel_count = int(np.sum(roi_bool))
        mean_validity_pct = float(np.mean(valid_ratios[roi_bool]) * 100.0) if active_pixel_count > 0 else 0.0
    else:
        active_pixel_count = total_pixels
        mean_validity_pct = float(np.mean(valid_ratios) * 100.0)

    num_stable = int(np.sum(stable_mask))
    stable_ratio_pct = (num_stable / float(active_pixel_count) * 100.0) if active_pixel_count > 0 else 0.0

    if num_stable == 0:
        logger.warning(f"  [temporal-noise] No stable pixels found for stream '{stream_name}' (threshold={stable_threshold}).")
        return SingleStreamNoiseMetrics(
            stream_name=stream_name,
            num_frames=K,
            num_total_pixels=active_pixel_count,
            num_stable_pixels=0,
            stable_pixel_ratio_pct=0.0,
            mean_temporal_validity_pct=mean_validity_pct,
            stable_mask=stable_mask,
            std_map_mm=np.full((H, W), np.nan, dtype=np.float32) if return_maps else None,
            detrended_std_map_mm=np.full((H, W), np.nan, dtype=np.float32) if return_maps else None,
        )

    # 2. Extract stable pixel sequences: shape (K, P)
    depth_p = depth_clean[:, stable_mask]      # (K, P)
    valid_p = valid_mask[:, stable_mask]       # (K, P)
    N_p = valid_counts[stable_mask].astype(np.float64)  # (P,)

    # 3. Standard temporal sample standard deviation (N - 1 DOF)
    mean_p = np.nansum(depth_p, axis=0) / N_p  # (P,)
    diff_p = np.where(valid_p, depth_p - mean_p[None, :], 0.0)
    sum_sq_p = np.sum(diff_p ** 2, axis=0)
    dof_std_p = np.maximum(N_p - 1.0, 1.0)
    std_p_m = np.sqrt(sum_sq_p / dof_std_p)
    std_p_mm = std_p_m * 1000.0

    # 4. Linearly detrended residual standard deviation (N - 2 DOF)
    # Fit z(t) = a * t + b per pixel using OLS
    T_col = ts_arr[:, None]  # (K, 1)
    mean_t_p = np.sum(np.where(valid_p, T_col, 0.0), axis=0) / N_p  # (P,)
    dt_p = np.where(valid_p, T_col - mean_t_p[None, :], 0.0)        # (K, P)
    dz_p = np.where(valid_p, depth_p - mean_p[None, :], 0.0)        # (K, P)

    var_t_p = np.sum(dt_p ** 2, axis=0)                             # (P,)
    cov_tz_p = np.sum(dt_p * dz_p, axis=0)                          # (P,)

    has_var_t = var_t_p > 1e-12
    slope_p = np.where(has_var_t, cov_tz_p / np.maximum(var_t_p, 1e-12), 0.0)
    intercept_p = mean_p - slope_p * mean_t_p

    # Model predictions on valid observations
    pred_p = slope_p[None, :] * T_col + intercept_p[None, :]
    res_p = np.where(valid_p, depth_p - pred_p, 0.0)
    sum_res_sq_p = np.sum(res_p ** 2, axis=0)

    # Degrees of freedom: N - 2 for 2-parameter linear fit; fallback to N - 1 if slope undefined
    dof_detrend_p = np.where(has_var_t, np.maximum(N_p - 2.0, 1.0), np.maximum(N_p - 1.0, 1.0))
    detrended_std_p_m = np.sqrt(sum_res_sq_p / dof_detrend_p)
    detrended_std_p_mm = detrended_std_p_m * 1000.0

    # 5. Populate full 2D spatial maps if requested
    std_map_mm = None
    detrended_std_map_mm = None
    if return_maps:
        std_map_mm = np.full((H, W), np.nan, dtype=np.float32)
        std_map_mm[stable_mask] = std_p_mm.astype(np.float32)
        detrended_std_map_mm = np.full((H, W), np.nan, dtype=np.float32)
        detrended_std_map_mm[stable_mask] = detrended_std_p_mm.astype(np.float32)

    return SingleStreamNoiseMetrics(
        stream_name=stream_name,
        num_frames=K,
        num_total_pixels=active_pixel_count,
        num_stable_pixels=num_stable,
        stable_pixel_ratio_pct=stable_ratio_pct,
        mean_temporal_validity_pct=mean_validity_pct,
        mean_std_mm=float(np.mean(std_p_mm)),
        median_std_mm=float(np.median(std_p_mm)),
        p95_std_mm=float(np.percentile(std_p_mm, 95.0)),
        min_std_mm=float(np.min(std_p_mm)),
        max_std_mm=float(np.max(std_p_mm)),
        mean_detrended_std_mm=float(np.mean(detrended_std_p_mm)),
        median_detrended_std_mm=float(np.median(detrended_std_p_mm)),
        p95_detrended_std_mm=float(np.percentile(detrended_std_p_mm, 95.0)),
        min_detrended_std_mm=float(np.min(detrended_std_p_mm)),
        max_detrended_std_mm=float(np.max(detrended_std_p_mm)),
        stable_mask=stable_mask,
        std_map_mm=std_map_mm,
        detrended_std_map_mm=detrended_std_map_mm,
    )


def compare_raw_vs_filtered_noise(
    raw_depth_frames: Union[np.ndarray, List[np.ndarray]],
    filtered_depth_frames: Union[np.ndarray, List[np.ndarray]],
    timestamps: Optional[Union[np.ndarray, List[float]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    depth_min_m: float = 0.40,
    depth_max_m: float = 3.00,
    stable_threshold: float = 0.90,
    min_valid_samples: int = 3,
    motion_displacement_mm: Optional[float] = None,
    return_maps: bool = False,
) -> TemporalNoiseComparison:
    """
    Compare raw versus production-filtered temporal depth jitter.

    Executes three rigorous evaluation paths:
    1. Raw stream individual metrics (over raw stable mask)
    2. Filtered stream individual metrics (over filtered stable mask)
    3. Matched-population comparison (strictly over the intersection of both stable masks)

    Parameters
    ----------
    raw_depth_frames : Union[np.ndarray, List[np.ndarray]]
        Burst of raw (unfiltered) depth maps in metres.
    filtered_depth_frames : Union[np.ndarray, List[np.ndarray]]
        Burst of production-filtered depth maps in metres.
    timestamps : Optional[Union[np.ndarray, List[float]]]
        Burst timestamps in seconds.
    roi_mask : Optional[np.ndarray]
        Spatial region of interest mask.
    depth_min_m : float
        Minimum valid depth threshold in metres.
    depth_max_m : float
        Maximum valid depth threshold in metres.
    stable_threshold : float
        Stable pixel threshold ratio (default 0.90).
    min_valid_samples : int
        Minimum valid samples for detrending.
    motion_displacement_mm : Optional[float]
        Diagnostic physical motion measured across the burst.
    return_maps : bool
        Whether to retain 2D spatial maps in output dataclasses.

    Returns
    -------
    TemporalNoiseComparison
        Comparison report containing individual raw, individual filtered, and fair matched-population metrics.
    """
    # 1. Raw individual metrics
    raw_metrics = compute_temporal_depth_noise(
        depth_frames=raw_depth_frames,
        timestamps=timestamps,
        roi_mask=roi_mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        stable_threshold=stable_threshold,
        min_valid_samples=min_valid_samples,
        stream_name="raw",
        return_maps=return_maps,
    )

    # 2. Filtered individual metrics
    filtered_metrics = compute_temporal_depth_noise(
        depth_frames=filtered_depth_frames,
        timestamps=timestamps,
        roi_mask=roi_mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        stable_threshold=stable_threshold,
        min_valid_samples=min_valid_samples,
        stream_name="filtered",
        return_maps=return_maps,
    )

    # 3. Construct matched intersection mask
    raw_mask = raw_metrics.stable_mask
    filt_mask = filtered_metrics.stable_mask

    if raw_mask is not None and filt_mask is not None:
        matched_mask = raw_mask & filt_mask
    elif raw_mask is not None:
        matched_mask = raw_mask
    elif filt_mask is not None:
        matched_mask = filt_mask
    else:
        matched_mask = np.zeros_like(raw_depth_frames[0], dtype=bool)

    matched_pixel_count = int(np.sum(matched_mask))

    raw_retention_pct = (
        (matched_pixel_count / float(raw_metrics.num_stable_pixels) * 100.0)
        if raw_metrics.num_stable_pixels > 0
        else 0.0
    )
    filt_retention_pct = (
        (matched_pixel_count / float(filtered_metrics.num_stable_pixels) * 100.0)
        if filtered_metrics.num_stable_pixels > 0
        else 0.0
    )

    # 4. Evaluate matched-population metrics
    if matched_pixel_count == 0:
        matched_comp = MatchedPopulationComparison(
            matched_pixel_count=0,
            raw_stable_retention_pct=0.0,
            filtered_stable_retention_pct=0.0,
            status="empty_intersection",
        )
        return TemporalNoiseComparison(
            raw=raw_metrics,
            filtered=filtered_metrics,
            matched=matched_comp,
            ratio_filtered_to_raw_mean_jitter=None,
            jitter_reduction_pct=None,
            motion_centroid_displacement_mm=motion_displacement_mm,
        )

    # Evaluate raw and filtered on the exact matched mask
    raw_matched = compute_temporal_depth_noise(
        depth_frames=raw_depth_frames,
        timestamps=timestamps,
        roi_mask=matched_mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        stable_threshold=stable_threshold,
        min_valid_samples=min_valid_samples,
        stream_name="raw_matched",
        return_maps=False,
    )

    filt_matched = compute_temporal_depth_noise(
        depth_frames=filtered_depth_frames,
        timestamps=timestamps,
        roi_mask=matched_mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        stable_threshold=stable_threshold,
        min_valid_samples=min_valid_samples,
        stream_name="filtered_matched",
        return_maps=False,
    )

    matched_ratio: Optional[float] = None
    matched_reduction: Optional[float] = None
    status_str = "valid"

    if raw_matched.mean_detrended_std_mm is not None and filt_matched.mean_detrended_std_mm is not None:
        if raw_matched.mean_detrended_std_mm > 1e-3:
            matched_ratio = filt_matched.mean_detrended_std_mm / raw_matched.mean_detrended_std_mm
            matched_reduction = (1.0 - matched_ratio) * 100.0
        else:
            status_str = "zero_raw_jitter"

    matched_comp = MatchedPopulationComparison(
        matched_pixel_count=matched_pixel_count,
        raw_stable_retention_pct=raw_retention_pct,
        filtered_stable_retention_pct=filt_retention_pct,
        raw_mean_std_mm=raw_matched.mean_std_mm,
        raw_median_std_mm=raw_matched.median_std_mm,
        raw_mean_detrended_std_mm=raw_matched.mean_detrended_std_mm,
        raw_median_detrended_std_mm=raw_matched.median_detrended_std_mm,
        filtered_mean_std_mm=filt_matched.mean_std_mm,
        filtered_median_std_mm=filt_matched.median_std_mm,
        filtered_mean_detrended_std_mm=filt_matched.mean_detrended_std_mm,
        filtered_median_detrended_std_mm=filt_matched.median_detrended_std_mm,
        ratio_filtered_to_raw_mean_jitter=matched_ratio,
        denoising_reduction_pct=matched_reduction,
        status=status_str,
    )

    return TemporalNoiseComparison(
        raw=raw_metrics,
        filtered=filtered_metrics,
        matched=matched_comp,
        ratio_filtered_to_raw_mean_jitter=matched_ratio,
        jitter_reduction_pct=matched_reduction,
        motion_centroid_displacement_mm=motion_displacement_mm,
    )
