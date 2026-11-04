#!/usr/bin/env python3
"""
tools/realsense_hardware_smoke_test.py
======================================
Intel RealSense D455f Hardware Smoke Test and RGB-D Stream Validation Utility.

Validates the physical D455f sensor before live integration into the Anny
reconstruction pipeline:
- Device discovery, serial number, firmware version, USB connection type, sensors.
- Synchronized color and depth streaming, FPS, timestamps, frame drops.
- RGB-depth alignment and raster dimension matching.
- Stream intrinsics (fx, fy, cx, cy, distortion model, coefficients).
- Metric depth scale verification (0.001 m/unit).
- Comprehensive depth statistics (valid %, min, median, mean, std, max, zeros, NaNs, Infs).
- Optional manual reference-distance ROI sanity check (median error, MAD, temporal stability).
- Optional recording mode compatible with existing replay format.
- Side-by-side RGB | Depth visualization with interactive HUD (or headless mode).

Usage:
    # 1. Interactive live smoke test (press 'q' or ESC to exit):
    python tools/realsense_hardware_smoke_test.py

    # 2. Reference-distance sanity check at 1.5m:
    python tools/realsense_hardware_smoke_test.py --ref-distance 1.5

    # 3. Headless non-interactive 60-frame validation:
    python tools/realsense_hardware_smoke_test.py --headless --frames 60

    # 4. Record a 30-frame sequence for replay verification:
    python tools/realsense_hardware_smoke_test.py --record raw_captures/test_smoke_01 --frames 30
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

# Ensure repository paths are configured on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    PYREALSENSE_AVAILABLE = False

try:
    from config import CameraConfig
    from camera.realsense_capture import (
        CameraError,
        CameraNotStartedError,
        DeviceDisconnectedError,
        RealSenseCamera,
        StreamNegotiationError,
    )
except ImportError:
    from realsense_measure.config import CameraConfig
    from realsense_measure.camera.realsense_capture import (
        CameraError,
        CameraNotStartedError,
        DeviceDisconnectedError,
        RealSenseCamera,
        StreamNegotiationError,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realsense_smoke_test")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DepthStatistics:
    """Statistical summary of depth pixel distribution."""

    total_pixels: int
    valid_pixels: int
    valid_pct: float
    zero_pixels: int
    zero_pct: float
    nan_pixels: int
    inf_pixels: int
    out_of_range_pixels: int
    min_m: float | None = None
    median_m: float | None = None
    mean_m: float | None = None
    std_m: float | None = None
    max_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ROIStatistics:
    """Depth statistics within a specified Region of Interest."""

    roi: tuple[int, int, int, int]  # (x, y, w, h)
    total_pixels: int
    valid_pixels: int
    valid_pct: float
    median_m: float | None = None
    mean_m: float | None = None
    std_m: float | None = None
    mad_m: float | None = None  # Median Absolute Deviation
    ref_distance_m: float | None = None
    error_median_m: float | None = None
    error_median_mm: float | None = None
    error_mean_mm: float | None = None
    error_pct: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalStability:
    """Evaluation of temporal depth stability across consecutive frames."""

    num_samples: int
    temporal_std_m: float | None = None
    temporal_std_mm: float | None = None
    max_drift_mm: float | None = None
    is_stable: bool = True
    status_message: str = "Stable"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntrinsicsInfo:
    """Camera intrinsic parameters and lens distortion model."""

    stream_type: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs: list[float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Analytical Functions (Non-Hardware, Fully Testable)
# ---------------------------------------------------------------------------

def compute_depth_statistics(
    depth_m: np.ndarray,
    min_valid_m: float = 0.15,
    max_valid_m: float = 6.0,
) -> DepthStatistics:
    """
    Compute comprehensive pixel-level health and distance statistics.

    Parameters
    ----------
    depth_m:
        HxW float32 or float64 depth map in metres.
    min_valid_m:
        Minimum plausible physical distance (metres).
    max_valid_m:
        Maximum plausible physical distance (metres).

    Returns
    -------
    DepthStatistics
    """
    if depth_m is None or depth_m.size == 0:
        return DepthStatistics(
            total_pixels=0,
            valid_pixels=0,
            valid_pct=0.0,
            zero_pixels=0,
            zero_pct=0.0,
            nan_pixels=0,
            inf_pixels=0,
            out_of_range_pixels=0,
        )

    total_pixels = int(depth_m.size)
    nan_mask = np.isnan(depth_m)
    inf_mask = np.isinf(depth_m)
    nan_pixels = int(np.count_nonzero(nan_mask))
    inf_pixels = int(np.count_nonzero(inf_mask))

    finite_mask = ~(nan_mask | inf_mask)
    zero_mask = finite_mask & (depth_m == 0.0)
    zero_pixels = int(np.count_nonzero(zero_mask))
    zero_pct = float(100.0 * zero_pixels / total_pixels) if total_pixels > 0 else 0.0

    valid_mask = (
        finite_mask
        & (depth_m >= min_valid_m)
        & (depth_m <= max_valid_m)
    )
    valid_pixels = int(np.count_nonzero(valid_mask))
    valid_pct = float(100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Out of range: non-zero finite pixels strictly outside [min_valid_m, max_valid_m]
    out_of_range_mask = (
        finite_mask
        & (depth_m > 0.0)
        & ((depth_m < min_valid_m) | (depth_m > max_valid_m))
    )
    out_of_range_pixels = int(np.count_nonzero(out_of_range_mask))

    if valid_pixels > 0:
        valid_vals = depth_m[valid_mask]
        min_m = float(np.min(valid_vals))
        median_m = float(np.median(valid_vals))
        mean_m = float(np.mean(valid_vals))
        std_m = float(np.std(valid_vals))
        max_m = float(np.max(valid_vals))
    else:
        min_m = median_m = mean_m = std_m = max_m = None

    return DepthStatistics(
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        valid_pct=valid_pct,
        zero_pixels=zero_pixels,
        zero_pct=zero_pct,
        nan_pixels=nan_pixels,
        inf_pixels=inf_pixels,
        out_of_range_pixels=out_of_range_pixels,
        min_m=min_m,
        median_m=median_m,
        mean_m=mean_m,
        std_m=std_m,
        max_m=max_m,
    )


def compute_roi_statistics(
    depth_m: np.ndarray,
    roi: tuple[int, int, int, int],
    ref_distance_m: float | None = None,
    min_valid_m: float = 0.15,
    max_valid_m: float = 6.0,
) -> ROIStatistics:
    """
    Compute depth distribution statistics inside a defined bounding box.

    Parameters
    ----------
    depth_m:
        HxW depth image in metres.
    roi:
        (x, y, w, h) bounding box coordinates.
    ref_distance_m:
        Optional physical ground truth distance (metres).
    min_valid_m:
        Minimum plausible valid depth in metres.
    max_valid_m:
        Maximum plausible valid depth in metres.

    Returns
    -------
    ROIStatistics
    """
    h_img, w_img = depth_m.shape[:2]
    x, y, w, h = roi

    # Clamp ROI to image boundaries
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(x1 + 1, min(x + w, w_img))
    y2 = max(y1 + 1, min(y + h, h_img))
    clamped_roi = (x1, y1, x2 - x1, y2 - y1)

    roi_patch = depth_m[y1:y2, x1:x2]
    total_pixels = int(roi_patch.size)

    finite_mask = np.isfinite(roi_patch)
    valid_mask = (
        finite_mask
        & (roi_patch >= min_valid_m)
        & (roi_patch <= max_valid_m)
    )
    valid_pixels = int(np.count_nonzero(valid_mask))
    valid_pct = float(100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    if valid_pixels > 0:
        valid_vals = roi_patch[valid_mask]
        median_m = float(np.median(valid_vals))
        mean_m = float(np.mean(valid_vals))
        std_m = float(np.std(valid_vals))
        # Median Absolute Deviation (MAD) = median(|x - median(x)|)
        mad_m = float(np.median(np.abs(valid_vals - median_m)))
    else:
        median_m = mean_m = std_m = mad_m = None

    error_median_m = None
    error_median_mm = None
    error_mean_mm = None
    error_pct = None

    if ref_distance_m is not None and ref_distance_m > 0 and median_m is not None:
        error_median_m = median_m - ref_distance_m
        error_median_mm = error_median_m * 1000.0
        if mean_m is not None:
            error_mean_mm = (mean_m - ref_distance_m) * 1000.0
        error_pct = float(100.0 * abs(error_median_m) / ref_distance_m)

    return ROIStatistics(
        roi=clamped_roi,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        valid_pct=valid_pct,
        median_m=median_m,
        mean_m=mean_m,
        std_m=std_m,
        mad_m=mad_m,
        ref_distance_m=ref_distance_m,
        error_median_m=error_median_m,
        error_median_mm=error_median_mm,
        error_mean_mm=error_mean_mm,
        error_pct=error_pct,
    )


def evaluate_temporal_stability(
    roi_medians_m: list[float],
    max_std_mm: float = 6.0,
    max_drift_mm: float = 15.0,
) -> TemporalStability:
    """
    Evaluate temporal noise and drift of ROI median depth over time.

    Parameters
    ----------
    roi_medians_m:
        List of median depths in metres across consecutive frames.
    max_std_mm:
        Standard deviation threshold in mm for stability.
    max_drift_mm:
        Max absolute drift (peak-to-peak) threshold in mm.

    Returns
    -------
    TemporalStability
    """
    valid_medians = [m for m in roi_medians_m if m is not None and np.isfinite(m)]
    n = len(valid_medians)
    if n < 2:
        return TemporalStability(
            num_samples=n,
            is_stable=True,
            status_message="Insufficient samples to assess temporal stability",
        )

    arr = np.array(valid_medians, dtype=np.float64)
    std_m = float(np.std(arr))
    std_mm = std_m * 1000.0
    drift_mm = float(np.ptp(arr)) * 1000.0

    is_stable = bool((std_mm <= max_std_mm) and (drift_mm <= max_drift_mm))
    if is_stable:
        msg = f"Stable (Temporal jitter std: {std_mm:.2f} mm <= {max_std_mm:.1f} mm, drift: {drift_mm:.2f} mm)"
    else:
        reasons = []
        if std_mm > max_std_mm:
            reasons.append(f"jitter std {std_mm:.2f} mm > {max_std_mm:.1f} mm")
        if drift_mm > max_drift_mm:
            reasons.append(f"peak-to-peak drift {drift_mm:.2f} mm > {max_drift_mm:.1f} mm")
        msg = f"Unstable ({', '.join(reasons)})"

    return TemporalStability(
        num_samples=n,
        temporal_std_m=std_m,
        temporal_std_mm=std_mm,
        max_drift_mm=drift_mm,
        is_stable=is_stable,
        status_message=msg,
    )


def validate_stream_dimensions(
    color_shape: tuple[int, ...],
    depth_shape: tuple[int, ...],
) -> tuple[bool, str]:
    """
    Verify that aligned RGB and Depth arrays have identical 2D spatial dimensions.

    Parameters
    ----------
    color_shape:
        Color array shape (H, W, 3) or (H, W).
    depth_shape:
        Depth array shape (H, W).

    Returns
    -------
    (is_valid, message)
    """
    hc, wc = color_shape[:2]
    hd, wd = depth_shape[:2]
    if (hc, wc) == (hd, wd):
        return True, f"Dimensions match exactly: {wc}x{hc}"
    return False, f"Dimension mismatch! Color: {wc}x{hc}, Depth: {wd}x{hd}"


def get_default_center_roi(width: int, height: int, frac: float = 0.20) -> tuple[int, int, int, int]:
    """
    Calculate a centered rectangular ROI covering `frac` proportion of width/height.
    """
    w = max(16, int(width * frac))
    h = max(16, int(height * frac))
    x = (width - w) // 2
    y = (height - h) // 2
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# Recording & Replay Persistence
# ---------------------------------------------------------------------------

def save_smoke_test_recording(
    output_dir: Path | str,
    frames_data: list[dict[str, Any]],
    intrinsics: IntrinsicsInfo,
    device_info: dict[str, Any],
    camera_cfg: CameraConfig,
    overwrite: bool = False,
) -> Path:
    """
    Save an RGB-D capture sequence adhering to the existing replay format.

    Compatible with `realsense_measure.camera.recording.load_raw_capture()`.

    Saves:
      - frame_{i:02d}_color.png
      - frame_{i:02d}_depth.npy (metric float32 in metres)
      - frame_{i:02d}_raw_z16.npy (native uint16 sensor units, if available)
      - meta.json (intrinsics, camera_config, device_info, frame list)
    """
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{out_path}' exists and is not empty. "
                f"Pass --overwrite to permit overwriting."
            )

    out_path.mkdir(parents=True, exist_ok=True)

    intr_meta = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "model": intrinsics.model,
        "coeffs": intrinsics.coeffs,
    }

    cam_meta = {
        "width": camera_cfg.width,
        "height": camera_cfg.height,
        "fps": camera_cfg.fps,
        "use_decimation": camera_cfg.use_decimation,
        "decimation_magnitude": camera_cfg.decimation_magnitude,
        "use_spatial_filter": camera_cfg.use_spatial_filter,
        "use_temporal_filter": camera_cfg.use_temporal_filter,
        "use_hole_filling": camera_cfg.use_hole_filling,
        "depth_min_m": camera_cfg.depth_min_m,
        "depth_max_m": camera_cfg.depth_max_m,
        "rotate_90": getattr(camera_cfg, "rotate_90", None),
    }

    frames_meta = []
    for i, f in enumerate(frames_data):
        color_filename = f"frame_{i:02d}_color.png"
        depth_filename = f"frame_{i:02d}_depth.npy"
        raw_z16_filename = f"frame_{i:02d}_raw_z16.npy"

        # Save color
        cv2.imwrite(str(out_path / color_filename), f["color_bgr"])

        # Save metric depth (float32, metres)
        np.save(str(out_path / depth_filename), f["depth_m"].astype(np.float32))

        # Save native raw z16 if present
        if "raw_z16" in f and f["raw_z16"] is not None:
            np.save(str(out_path / raw_z16_filename), f["raw_z16"].astype(np.uint16))

        entry = {
            "frame_id": i,
            "order": i,
            "timestamp": f.get("timestamp", float(i)),
            "color_file": color_filename,
            "depth_file": depth_filename,
            "raw_z16_file": raw_z16_filename if "raw_z16" in f else None,
            "depth_stats": f.get("depth_stats"),
            "roi_stats": f.get("roi_stats"),
        }
        frames_meta.append(entry)

    meta_doc = {
        "target_name": "hardware_smoke_test",
        "capture_status": "completed",
        "alignment_state": "depth_aligned_to_color",
        "num_frames": len(frames_data),
        "intrinsics": intr_meta,
        "camera_config": cam_meta,
        "device_info": device_info,
        "has_background_ref": False,
        "frames": frames_meta,
    }

    with open(out_path / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta_doc, fp, indent=2)

    logger.info(f"Saved {len(frames_data)} frames and meta.json to: {out_path}/")
    return out_path


# ---------------------------------------------------------------------------
# Hardware Device Detection & Query
# ---------------------------------------------------------------------------

def detect_realsense_devices() -> list[dict[str, Any]]:
    """
    Detect all physically connected RealSense devices using pyrealsense2.

    Returns
    -------
    list of dicts containing name, serial, firmware, usb_type, and sensors.
    """
    if not PYREALSENSE_AVAILABLE:
        return []

    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        results = []
        for dev in devices:
            info = {
                "name": dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown",
                "serial_number": dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "Unknown",
                "firmware_version": dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "Unknown",
                "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown",
                "sensors": [s.get_info(rs.camera_info.name) for s in dev.query_sensors() if s.supports(rs.camera_info.name)],
            }
            results.append(info)
        return results
    except Exception as exc:
        logger.warning(f"Failed to query RealSense context: {exc}")
        return []


def query_stream_intrinsics(camera: RealSenseCamera) -> dict[str, IntrinsicsInfo]:
    """
    Retrieve intrinsics for active color and raw depth streams from active profile.
    """
    intr_dict: dict[str, IntrinsicsInfo] = {}
    if not PYREALSENSE_AVAILABLE or camera._profile is None:
        return intr_dict

    try:
        # Color stream intrinsics
        color_prof = camera._profile.get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = color_prof.get_intrinsics()
        intr_dict["color"] = IntrinsicsInfo(
            stream_type="color",
            width=c_intr.width,
            height=c_intr.height,
            fx=float(c_intr.fx),
            fy=float(c_intr.fy),
            cx=float(c_intr.ppx),
            cy=float(c_intr.ppy),
            model=str(c_intr.model),
            coeffs=[float(v) for v in c_intr.coeffs],
        )

        # Raw Depth stream intrinsics
        depth_prof = camera._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        d_intr = depth_prof.get_intrinsics()
        intr_dict["depth_raw"] = IntrinsicsInfo(
            stream_type="depth_raw",
            width=d_intr.width,
            height=d_intr.height,
            fx=float(d_intr.fx),
            fy=float(d_intr.fy),
            cx=float(d_intr.ppx),
            cy=float(d_intr.ppy),
            model=str(d_intr.model),
            coeffs=[float(v) for v in d_intr.coeffs],
        )
    except Exception as exc:
        logger.warning(f"Could not retrieve stream intrinsics: {exc}")

    return intr_dict


# ---------------------------------------------------------------------------
# HUD Rendering
# ---------------------------------------------------------------------------

def render_smoke_test_hud(
    color_bgr: np.ndarray,
    depth_color_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    depth_stats: DepthStatistics,
    roi_stats: ROIStatistics,
    device_info: dict[str, Any],
    intrinsics: IntrinsicsInfo | None,
    drop_count: int,
    recording_active: bool,
    ref_distance_m: float | None = None,
) -> np.ndarray:
    """
    Render a synchronized side-by-side display of RGB and Depth with informative HUD.
    """
    h_c, w_c = color_bgr.shape[:2]
    h_d, w_d = depth_color_bgr.shape[:2]

    # Resize depth visualization if shapes differ
    if (h_d, w_d) != (h_c, w_c):
        depth_vis = cv2.resize(depth_color_bgr, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
    else:
        depth_vis = depth_color_bgr.copy()

    rgb_vis = color_bgr.copy()

    # Draw ROI rectangle on both views
    rx, ry, rw, rh = roi_stats.roi
    box_color = (0, 255, 0) if roi_stats.valid_pct >= 80.0 else ((0, 255, 255) if roi_stats.valid_pct >= 40.0 else (0, 0, 255))
    cv2.rectangle(rgb_vis, (rx, ry), (rx + rw, ry + rh), box_color, 2)
    cv2.rectangle(depth_vis, (rx, ry), (rx + rw, ry + rh), box_color, 2)

    # Top info banner on RGB
    dev_name = device_info.get("name", "RealSense D455")
    usb_type = device_info.get("usb_type", "Unknown")
    cv2.putText(rgb_vis, f"{dev_name} (USB {usb_type})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(rgb_vis, f"{dev_name} (USB {usb_type})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    rec_str = " [RECORDING]" if recording_active else ""
    cv2.putText(rgb_vis, f"Frame: {frame_idx:04d} | FPS: {fps:.1f} | Drops: {drop_count}{rec_str}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(rgb_vis, f"Frame: {frame_idx:04d} | FPS: {fps:.1f} | Drops: {drop_count}{rec_str}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if not rec_str else (0, 0, 255), 1)

    if intrinsics:
        intr_str = f"Intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}"
        cv2.putText(rgb_vis, intr_str, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(rgb_vis, intr_str, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Depth panel stats
    cv2.putText(depth_vis, f"Depth Valid: {depth_stats.valid_pct:.1f}% (Zeros: {depth_stats.zero_pct:.1f}%)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3)
    cv2.putText(depth_vis, f"Depth Valid: {depth_stats.valid_pct:.1f}% (Zeros: {depth_stats.zero_pct:.1f}%)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    if depth_stats.median_m is not None:
        range_str = f"Range: [{depth_stats.min_m:.2f}m .. {depth_stats.max_m:.2f}m] | Median: {depth_stats.median_m:.3f}m"
        cv2.putText(depth_vis, range_str, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(depth_vis, range_str, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # ROI statistics overlay
    roi_text_y = ry + rh + 22 if (ry + rh + 40 < h_c) else ry - 10
    if roi_stats.median_m is not None:
        roi_str = f"ROI Depth: {roi_stats.median_m:.3f}m (std: {roi_stats.std_m*1000.0:.1f}mm, valid: {roi_stats.valid_pct:.0f}%)"
    else:
        roi_str = "ROI: No valid depth"
    cv2.putText(depth_vis, roi_str, (rx, roi_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
    cv2.putText(depth_vis, roi_str, (rx, roi_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 1)

    # If reference distance provided
    if ref_distance_m is not None and roi_stats.error_median_mm is not None:
        err_str = f"Ref: {ref_distance_m:.3f}m | Err: {roi_stats.error_median_mm:+.1f}mm ({roi_stats.error_pct:.1f}%)"
        cv2.putText(rgb_vis, err_str, (rx, roi_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(rgb_vis, err_str, (rx, roi_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    # Bottom helper prompt
    cv2.putText(rgb_vis, "Keys: [Q / ESC] Exit | [R] Record Toggle", (12, h_c - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(rgb_vis, "Keys: [Q / ESC] Exit | [R] Record Toggle", (12, h_c - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return np.hstack([rgb_vis, depth_vis])


# ---------------------------------------------------------------------------
# Smoke Test Runner
# ---------------------------------------------------------------------------

class HardwareSmokeTestRunner:
    """
    Executes the Intel RealSense D455f hardware smoke test.
    """

    def __init__(
        self,
        camera_cfg: CameraConfig,
        ref_distance_m: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        record_dir: Path | str | None = None,
        overwrite: bool = False,
        headless: bool = False,
        target_frames: int | None = None,
    ) -> None:
        self.cfg = camera_cfg
        self.ref_distance_m = ref_distance_m
        self.roi = roi
        self.record_dir = Path(record_dir) if record_dir else None
        self.overwrite = overwrite
        self.headless = headless
        self.target_frames = target_frames

        self.recorded_frames: list[dict[str, Any]] = []
        self.roi_medians_history: list[float] = []
        self.all_frame_stats: list[DepthStatistics] = []
        self.timestamps: list[float] = []
        self.drop_count: int = 0

    def run(self) -> dict[str, Any]:
        """
        Execute smoke test session and return structured summary dict.
        """
        print("=" * 72)
        print("  Intel RealSense D455f Hardware Smoke Test & Stream Validation")
        print("=" * 72)

        # 1. Device Discovery
        devices = detect_realsense_devices()
        if not devices:
            err_msg = (
                "[FATAL] No RealSense devices detected by SDK context.\n"
                "Checklist:\n"
                "  1. Ensure the D455f is plugged into a USB 3.0/3.2 port with an authorized cable.\n"
                "  2. Verify device visibility in Windows Device Manager or 'rs-enumerate-devices'.\n"
                "  3. Note: The smoke test will not silently fall back to synthetic data."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        dev_primary = devices[0]
        print(f"\n[DEVICE DETECTED]")
        print(f"  Model:           {dev_primary['name']}")
        print(f"  Serial Number:   {dev_primary['serial_number']}")
        print(f"  Firmware:        {dev_primary['firmware_version']}")
        print(f"  USB Type:        {dev_primary['usb_type']}")
        print(f"  Sensors:         {', '.join(dev_primary['sensors'])}")

        if dev_primary.get("usb_type", "").startswith("2"):
            logger.warning("  [WARNING] Camera is connected via USB 2.x! Bandwidth and stream stability will be degraded.")

        # 2. Camera Pipeline Initialization
        print(f"\n[STREAM CONFIGURATION]")
        print(f"  Requested:       {self.cfg.width}x{self.cfg.height} @ {self.cfg.fps} FPS")
        print(f"  Filters:         Decimation={self.cfg.use_decimation}, Spatial={self.cfg.use_spatial_filter}, Temporal={self.cfg.use_temporal_filter}, HoleFill={self.cfg.use_hole_filling}")

        camera = RealSenseCamera(self.cfg)
        try:
            camera.start()
        except Exception as exc:
            logger.error(f"[FATAL] Failed to negotiate or start camera pipeline: {exc}")
            raise StreamNegotiationError(f"Stream negotiation failed: {exc}") from exc

        # Retrieve detailed hardware options and intrinsics
        device_details = camera.get_device_info()
        depth_scale_m = camera.depth_scale
        intr_map = query_stream_intrinsics(camera)
        color_intr = intr_map.get("color")
        depth_raw_intr = intr_map.get("depth_raw")

        print(f"\n[STREAM INTRINSICS & ALIGNMENT]")
        if color_intr:
            print(f"  Color Camera:    {color_intr.width}x{color_intr.height} | fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}, cx={color_intr.cx:.2f}, cy={color_intr.cy:.2f}")
            print(f"  Distortion:      Model={color_intr.model}, Coeffs={color_intr.coeffs}")
        if depth_raw_intr:
            print(f"  Raw Depth Sensor: {depth_raw_intr.width}x{depth_raw_intr.height} | fx={depth_raw_intr.fx:.2f}, fy={depth_raw_intr.fy:.2f}, cx={depth_raw_intr.cx:.2f}, cy={depth_raw_intr.cy:.2f}")
        print(f"  Depth Scale:     {camera.depth_scale} m/unit")
        print(f"  Alignment:       Active (depth warped into color optical frame)")
        print(f"  Note:            Because depth is aligned to color, 3D backprojection MUST use color camera intrinsics.")

        if device_details.get("asic_temperature_celsius") is not None:
            print(f"  ASIC Temp:       {device_details['asic_temperature_celsius']:.1f} °C")
        if device_details.get("projector_temperature_celsius") is not None:
            print(f"  Projector Temp:  {device_details['projector_temperature_celsius']:.1f} °C")

        # Configure ROI
        active_roi = self.roi
        if active_roi is None:
            active_roi = get_default_center_roi(self.cfg.width, self.cfg.height, frac=0.20)
        print(f"\n[REGION OF INTEREST]")
        print(f"  ROI Bounding Box: (x={active_roi[0]}, y={active_roi[1]}, w={active_roi[2]}, h={active_roi[3]})")
        if self.ref_distance_m is not None:
            print(f"  Target Reference Distance: {self.ref_distance_m:.3f} m")
            print(f"  [PRELIMINARY SANITY CHECK ONLY - NOT A FORMAL CALIBRATION]")

        # Prepare GUI Window if interactive
        window_name = "Intel RealSense D455f Hardware Smoke Test (RGB | Depth)"
        gui_active = False
        if not self.headless:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 480)
                gui_active = True
            except Exception as e:
                logger.warning(f"Could not initialize GUI window ({e}); falling back to headless mode.")
                gui_active = False

        frame_count = 0
        t_start = time.time()
        last_frame_time = t_start
        fps = 0.0
        fps_frame_count = 0
        last_fps_time = t_start
        nominal_interval = 1.0 / float(self.cfg.fps)

        recording_active = bool(self.record_dir is not None)
        session_aborted = False

        print(f"\n[CAPTURE STREAMING ACTIVE]")
        if gui_active:
            print("  Press [Q] or [ESC] to exit cleanly.")
            print("  Press [R] to toggle recording.")
        else:
            print(f"  Running in headless mode for {self.target_frames or 60} frames...")

        try:
            while True:
                t_now = time.time()

                # Read aligned frames returning raw and filtered depth
                res = camera.read_with_raw_and_colorized()
                color_bgr, filtered_depth_m, raw_depth_m, raw_z16, depth_color_bgr, frame_meta = res

                if color_bgr is None or filtered_depth_m is None:
                    self.drop_count += 1
                    time.sleep(0.002)
                    continue

                # Dimension check
                valid_dims, dim_msg = validate_stream_dimensions(color_bgr.shape, filtered_depth_m.shape)
                if not valid_dims:
                    raise ValueError(dim_msg)

                frame_count += 1
                fps_frame_count += 1
                dt = t_now - last_frame_time
                self.timestamps.append(t_now)

                # Check timestamp discontinuity / frame drop
                if frame_count > 1 and dt > (1.8 * nominal_interval):
                    self.drop_count += 1

                last_frame_time = t_now

                if t_now - last_fps_time >= 1.0:
                    fps = fps_frame_count / (t_now - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = t_now

                # Analytical statistics
                d_stats = compute_depth_statistics(filtered_depth_m, min_valid_m=self.cfg.depth_min_m)
                roi_stats = compute_roi_statistics(
                    filtered_depth_m,
                    roi=active_roi,
                    ref_distance_m=self.ref_distance_m,
                    min_valid_m=self.cfg.depth_min_m,
                )

                self.all_frame_stats.append(d_stats)
                if roi_stats.median_m is not None:
                    self.roi_medians_history.append(roi_stats.median_m)

                # Recording buffer
                if recording_active:
                    frame_record = {
                        "frame_id": frame_count - 1,
                        "timestamp": t_now,
                        "color_bgr": color_bgr.copy(),
                        "depth_m": filtered_depth_m.copy(),
                        "raw_z16": raw_z16.copy() if raw_z16 is not None else None,
                        "depth_stats": d_stats.as_dict(),
                        "roi_stats": roi_stats.as_dict(),
                    }
                    self.recorded_frames.append(frame_record)

                # GUI Display
                if gui_active:
                    combined_vis = render_smoke_test_hud(
                        color_bgr=color_bgr,
                        depth_color_bgr=depth_color_bgr,
                        frame_idx=frame_count,
                        fps=fps,
                        depth_stats=d_stats,
                        roi_stats=roi_stats,
                        device_info=dev_primary,
                        intrinsics=color_intr,
                        drop_count=self.drop_count,
                        recording_active=recording_active,
                        ref_distance_m=self.ref_distance_m,
                    )
                    cv2.imshow(window_name, combined_vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        print("\n  User requested exit.")
                        break
                    elif key in (ord('r'), ord('R')):
                        recording_active = not recording_active
                        print(f"  Recording toggled: {'ON' if recording_active else 'OFF'}")

                # Target frame termination in headless or batch mode
                if self.target_frames is not None and frame_count >= self.target_frames:
                    print(f"\n  Target frame count ({self.target_frames}) reached.")
                    break

        except KeyboardInterrupt:
            print("\n  Capture interrupted by user (Ctrl+C).")
            session_aborted = True
        except Exception as exc:
            logger.error(f"Error during stream execution: {exc}")
            session_aborted = True
            raise
        finally:
            camera.stop()
            if gui_active:
                cv2.destroyAllWindows()
            print("  Camera pipeline closed cleanly.")

        # 3. Post-Capture Analysis & Compilation
        total_time = max(0.001, time.time() - t_start)
        avg_fps = frame_count / total_time

        # Temporal stability analysis
        temp_stability = evaluate_temporal_stability(self.roi_medians_history)

        # Average statistics across session
        if self.all_frame_stats:
            avg_valid_pct = float(np.mean([s.valid_pct for s in self.all_frame_stats]))
            avg_zero_pct = float(np.mean([s.zero_pct for s in self.all_frame_stats]))
            all_medians = [s.median_m for s in self.all_frame_stats if s.median_m is not None]
            overall_median = float(np.median(all_medians)) if all_medians else None
        else:
            avg_valid_pct = 0.0
            avg_zero_pct = 100.0
            overall_median = None

        final_roi_stats = roi_stats if 'roi_stats' in locals() else None

        summary: dict[str, Any] = {
            "device": dev_primary,
            "device_details": device_details,
            "stream_config": {
                "width": self.cfg.width,
                "height": self.cfg.height,
                "fps": self.cfg.fps,
                "total_frames_received": frame_count,
                "total_seconds": total_time,
                "average_fps": avg_fps,
                "dropped_frames": self.drop_count,
            },
            "intrinsics": {
                "color": color_intr.as_dict() if color_intr else None,
                "depth_raw": depth_raw_intr.as_dict() if depth_raw_intr else None,
                "depth_scale_m": depth_scale_m,
                "effective_projection_intrinsics": "color",
            },
            "depth_health": {
                "average_valid_pct": avg_valid_pct,
                "average_zero_pct": avg_zero_pct,
                "overall_median_depth_m": overall_median,
            },
            "roi_analysis": final_roi_stats.as_dict() if final_roi_stats else None,
            "temporal_stability": temp_stability.as_dict(),
            "session_status": "aborted" if session_aborted else "completed",
        }

        # Save recording if requested
        if self.record_dir and self.recorded_frames:
            rec_path = save_smoke_test_recording(
                output_dir=self.record_dir,
                frames_data=self.recorded_frames,
                intrinsics=color_intr or IntrinsicsInfo("color", self.cfg.width, self.cfg.height, 0, 0, 0, 0, "", []),
                device_info=dev_primary,
                camera_cfg=self.cfg,
                overwrite=self.overwrite,
            )
            summary["recording_saved_to"] = str(rec_path)

        # Print Final Summary Report
        print("\n" + "=" * 72)
        print("  HARDWARE SMOKE TEST FINAL REPORT")
        print("=" * 72)
        print(f"  Device:               {dev_primary['name']} (S/N: {dev_primary['serial_number']})")
        print(f"  USB Connection:       {dev_primary['usb_type']}")
        print(f"  Frames Received:      {frame_count} in {total_time:.2f}s ({avg_fps:.1f} FPS)")
        print(f"  Dropped Frames:       {self.drop_count}")
        print(f"  Average Depth Valid:  {avg_valid_pct:.1f}%")
        print(f"  Average Zero/Void:    {avg_zero_pct:.1f}%")
        if overall_median is not None:
            print(f"  Overall Median Depth: {overall_median:.3f} m")

        if final_roi_stats and final_roi_stats.median_m is not None:
            print(f"\n  [ROI Summary]")
            print(f"    Bounding Box:       {final_roi_stats.roi}")
            print(f"    Measured Median:    {final_roi_stats.median_m:.3f} m")
            print(f"    Measured MAD:       {final_roi_stats.mad_m*1000.0:.2f} mm")
            if self.ref_distance_m is not None:
                print(f"    Reference Distance: {self.ref_distance_m:.3f} m")
                print(f"    Median Error:       {final_roi_stats.error_median_mm:+.1f} mm ({final_roi_stats.error_pct:.2f}%)")
                print(f"    Sanity Check:       [PRELIMINARY CHECK ONLY - NOT A FORMAL CALIBRATION]")

        print(f"\n  [Temporal Stability]")
        print(f"    Status:             {'PASS' if temp_stability.is_stable else 'WARNING'}")
        print(f"    Details:            {temp_stability.status_message}")

        if "recording_saved_to" in summary:
            print(f"\n  [Recording]")
            print(f"    Saved:              {summary['recording_saved_to']}")

        print("=" * 72 + "\n")
        return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intel RealSense D455f Hardware Smoke Test & Stream Validation Utility"
    )
    parser.add_argument("--width", type=int, default=848, help="Stream width (default: 848)")
    parser.add_argument("--height", type=int, default=480, help="Stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Stream framerate (default: 30)")
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        choices=["848x480", "1280x720", "640x480"],
        help="Convenience resolution presets",
    )
    parser.add_argument("--disable-filters", action="store_true", help="Disable all post-processing filters")
    parser.add_argument("--no-decimation", action="store_true", help="Disable depth decimation filter")
    parser.add_argument("--frames", type=int, default=None, help="Target number of frames to run")
    parser.add_argument("--headless", "--no-display", dest="headless", action="store_true", help="Run without GUI preview window")
    parser.add_argument("--ref-distance", type=float, default=None, help="Reference distance (m) to flat target")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None, help="Explicit ROI rectangle")
    parser.add_argument("--record", type=str, default=None, help="Directory to save raw RGB-D capture")
    parser.add_argument("--overwrite", action="store_true", help="Permit overwriting existing recording directory")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save summary report JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    width = args.width
    height = args.height
    if args.resolution:
        w_str, h_str = args.resolution.split("x")
        width, height = int(w_str), int(h_str)

    cfg = CameraConfig(
        width=width,
        height=height,
        fps=args.fps,
        use_decimation=not (args.disable_filters or args.no_decimation),
        use_spatial_filter=not args.disable_filters,
        use_temporal_filter=not args.disable_filters,
        use_hole_filling=not args.disable_filters,
    )

    roi = tuple(args.roi) if args.roi is not None else None

    # In headless mode without --frames specified, default to 60 frames (~2 seconds)
    target_frames = args.frames
    if args.headless and target_frames is None:
        target_frames = 60

    runner = HardwareSmokeTestRunner(
        camera_cfg=cfg,
        ref_distance_m=args.ref_distance,
        roi=roi,
        record_dir=args.record,
        overwrite=args.overwrite,
        headless=args.headless,
        target_frames=target_frames,
    )

    try:
        summary = runner.run()
        if args.output_json:
            out_json = Path(args.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            print(f"Summary report written to: {out_json}")
        return 0
    except Exception as exc:
        print(f"\n[FAILURE] Smoke test terminated with error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
