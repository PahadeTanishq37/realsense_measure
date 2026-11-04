"""
tests/test_realsense_hardware_smoke_test.py
===========================================
Unit tests for the Intel RealSense D455f Hardware Smoke Test utility:
- Depth statistics calculation and invalid-depth handling (zeros, NaNs, Infs).
- ROI statistics, reference-distance error evaluation, and ROI boundary clamping.
- Temporal stability assessment (jitter standard deviation and drift).
- Stream dimension mismatch detection.
- Metadata serialization and format integrity.
- Recording completeness and load_raw_capture replay compatibility.
- Mocked end-to-end runner execution (fully offline).
- Physical camera hardware test (conditional on physical device connection).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from tools.realsense_hardware_smoke_test import (
    DepthStatistics,
    HardwareSmokeTestRunner,
    IntrinsicsInfo,
    ROIStatistics,
    compute_depth_statistics,
    compute_roi_statistics,
    detect_realsense_devices,
    evaluate_temporal_stability,
    get_default_center_roi,
    save_smoke_test_recording,
    validate_stream_dimensions,
)
from realsense_measure.config import CameraConfig
from realsense_measure.camera.recording import load_raw_capture


# ===========================================================================
# 1. Depth Statistics & Invalid-Depth Handling
# ===========================================================================

def test_depth_statistics_normal():
    """Verify depth statistics on a known synthesized depth array."""
    depth = np.full((100, 100), 1.5, dtype=np.float32)
    # 2000 zero pixels
    depth[:20, :] = 0.0
    # 500 NaN pixels
    depth[20:25, :] = np.nan
    # 500 Inf pixels
    depth[25:30, :] = np.inf
    # 200 out of range pixels (< 0.15m)
    depth[30:32, :] = 0.05

    stats = compute_depth_statistics(depth, min_valid_m=0.15, max_valid_m=6.0)

    assert stats.total_pixels == 10000
    assert stats.zero_pixels == 2000
    assert stats.zero_pct == pytest.approx(20.0)
    assert stats.nan_pixels == 500
    assert stats.inf_pixels == 500
    assert stats.out_of_range_pixels == 200

    # Valid pixels = 10000 - 2000 - 500 - 500 - 200 = 6800
    assert stats.valid_pixels == 6800
    assert stats.valid_pct == pytest.approx(68.0)
    assert stats.median_m == pytest.approx(1.5, abs=1e-3)
    assert stats.mean_m == pytest.approx(1.5, abs=1e-3)
    assert stats.std_m == pytest.approx(0.0, abs=1e-4)


def test_depth_statistics_all_zeros_and_nans():
    """Verify statistics handle all-zero or all-NaN arrays gracefully without crashing."""
    all_zeros = np.zeros((50, 50), dtype=np.float32)
    stats_zeros = compute_depth_statistics(all_zeros)
    assert stats_zeros.valid_pixels == 0
    assert stats_zeros.valid_pct == 0.0
    assert stats_zeros.zero_pixels == 2500
    assert stats_zeros.median_m is None
    assert stats_zeros.mean_m is None

    all_nans = np.full((50, 50), np.nan, dtype=np.float32)
    stats_nans = compute_depth_statistics(all_nans)
    assert stats_nans.valid_pixels == 0
    assert stats_nans.nan_pixels == 2500
    assert stats_nans.median_m is None

    empty = np.array([], dtype=np.float32)
    stats_empty = compute_depth_statistics(empty)
    assert stats_empty.total_pixels == 0
    assert stats_empty.valid_pixels == 0


# ===========================================================================
# 2. ROI Statistics & Reference Distance
# ===========================================================================

def test_roi_statistics_accuracy_and_reference_distance():
    """Verify ROI statistics and reference-distance error calculations."""
    depth = np.zeros((200, 200), dtype=np.float32)
    # Fill ROI region (50, 50, 60, 60) with values centered at 1.480m
    y1, y2 = 50, 110
    x1, x2 = 50, 110
    depth[y1:y2, x1:x2] = 1.480

    roi = (50, 50, 60, 60)
    roi_stats = compute_roi_statistics(depth, roi=roi, ref_distance_m=1.500)

    assert roi_stats.total_pixels == 3600
    assert roi_stats.valid_pixels == 3600
    assert roi_stats.valid_pct == 100.0
    assert roi_stats.median_m == pytest.approx(1.480, abs=1e-3)
    assert roi_stats.ref_distance_m == 1.500
    # Error: 1.480 - 1.500 = -0.020 m = -20.0 mm
    assert roi_stats.error_median_mm == pytest.approx(-20.0, abs=0.1)
    assert roi_stats.error_pct == pytest.approx(1.333, abs=0.01)


def test_roi_clamping_out_of_bounds():
    """Verify that an ROI extending beyond image boundaries is safely clamped."""
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    # Bounding box starts at 80, 80 and extends to 140, 140
    roi = (80, 80, 60, 60)
    roi_stats = compute_roi_statistics(depth, roi=roi)

    # Clamped width/height should be 20x20
    assert roi_stats.roi == (80, 80, 20, 20)
    assert roi_stats.total_pixels == 400
    assert roi_stats.valid_pct == 100.0
    assert roi_stats.median_m == pytest.approx(2.0, abs=1e-3)


def test_default_center_roi():
    """Verify default center ROI calculation."""
    roi = get_default_center_roi(width=848, height=480, frac=0.20)
    x, y, w, h = roi
    assert w == int(848 * 0.20)
    assert h == int(480 * 0.20)
    assert x == (848 - w) // 2
    assert y == (480 - h) // 2


# ===========================================================================
# 3. Temporal Stability
# ===========================================================================

def test_temporal_stability_stable_and_unstable():
    """Verify temporal stability categorization based on jitter and drift."""
    # Stable sequence (small random jitter around 1.500m, std ~ 1.5mm)
    stable_samples = [1.500, 1.501, 1.499, 1.500, 1.502, 1.498, 1.500, 1.501]
    stab_res = evaluate_temporal_stability(stable_samples, max_std_mm=5.0, max_drift_mm=10.0)
    assert stab_res.is_stable is True
    assert stab_res.temporal_std_mm < 2.0
    assert "Stable" in stab_res.status_message

    # Unstable sequence (large drift across frames)
    unstable_samples = [1.500, 1.520, 1.540, 1.560, 1.580, 1.600]
    unstab_res = evaluate_temporal_stability(unstable_samples, max_std_mm=5.0, max_drift_mm=10.0)
    assert unstab_res.is_stable is False
    assert "Unstable" in unstab_res.status_message


# ===========================================================================
# 4. Stream Dimension Validation
# ===========================================================================

def test_validate_stream_dimensions():
    """Verify RGB-Depth shape comparison logic."""
    # Matching
    ok, msg = validate_stream_dimensions((480, 848, 3), (480, 848))
    assert ok is True
    assert "848x480" in msg

    # Mismatch
    bad, bad_msg = validate_stream_dimensions((720, 1280, 3), (480, 848))
    assert bad is False
    assert "Dimension mismatch" in bad_msg


# ===========================================================================
# 5. Recording & Replay Compatibility
# ===========================================================================

def test_recording_and_load_raw_capture_compatibility(tmp_path: Path):
    """
    Verify that saved recordings are fully compatible with existing load_raw_capture().
    """
    intr = IntrinsicsInfo(
        stream_type="color",
        width=848,
        height=480,
        fx=428.5,
        fy=428.0,
        cx=422.0,
        cy=244.5,
        model="distortion.inverse_brown_conrady",
        coeffs=[-0.05, 0.06, 0.0, 0.0, 0.0],
    )
    cam_cfg = CameraConfig(width=848, height=480, fps=30)
    device_info = {
        "name": "RealSense D455F",
        "serial_number": "342622300088",
        "firmware_version": "5.17.3.10",
        "usb_type": "3.2",
    }

    # Synthesize 3 test frames
    frames_data = []
    for i in range(3):
        color = np.full((480, 848, 3), 100 + i * 20, dtype=np.uint8)
        depth_m = np.full((480, 848), 1.2 + i * 0.1, dtype=np.float32)
        raw_z16 = (depth_m * 1000.0).astype(np.uint16)
        frames_data.append({
            "timestamp": 1000.0 + float(i),
            "color_bgr": color,
            "depth_m": depth_m,
            "raw_z16": raw_z16,
        })

    rec_dir = tmp_path / "smoke_test_rec"
    out_path = save_smoke_test_recording(
        output_dir=rec_dir,
        frames_data=frames_data,
        intrinsics=intr,
        device_info=device_info,
        camera_cfg=cam_cfg,
        overwrite=False,
    )

    assert out_path.exists()
    assert (out_path / "meta.json").exists()
    assert (out_path / "frame_00_color.png").exists()
    assert (out_path / "frame_00_depth.npy").exists()
    assert (out_path / "frame_00_raw_z16.npy").exists()

    # Verify existing replay loader loads this without error
    loaded_frames = load_raw_capture(out_path)
    assert len(loaded_frames) == 3
    assert loaded_frames[0].color_bgr.shape == (480, 848, 3)
    assert loaded_frames[0].depth_m.shape == (480, 848)
    assert loaded_frames[0].timestamp == pytest.approx(1000.0)
    assert loaded_frames[0].pcd is not None

    # Overwrite protection test: saving without overwrite must raise FileExistsError
    with pytest.raises(FileExistsError):
        save_smoke_test_recording(
            output_dir=rec_dir,
            frames_data=frames_data,
            intrinsics=intr,
            device_info=device_info,
            camera_cfg=cam_cfg,
            overwrite=False,
        )


# ===========================================================================
# 6. Mocked End-to-End Runner Execution
# ===========================================================================

def test_runner_mocked_camera(tmp_path: Path):
    """
    Test the complete runner lifecycle with a mocked camera pipeline (offline).
    """
    mock_devs = [{
        "name": "Mock RealSense D455",
        "serial_number": "MOCK123456",
        "firmware_version": "5.17.0.0",
        "usb_type": "3.2",
        "sensors": ["Stereo Module", "RGB Camera"],
    }]

    mock_color = np.full((480, 848, 3), 120, dtype=np.uint8)
    mock_depth_m = np.full((480, 848), 1.5, dtype=np.float32)
    mock_raw_z16 = np.full((480, 848), 1500, dtype=np.uint16)
    mock_colorized = np.full((480, 848, 3), 200, dtype=np.uint8)

    mock_cam_instance = MagicMock()
    mock_cam_instance.depth_scale = 0.001
    mock_cam_instance._profile = MagicMock()
    mock_cam_instance.get_device_info.return_value = {
        "name": "Mock RealSense D455",
        "serial_number": "MOCK123456",
        "firmware_version": "5.17.0.0",
        "usb_type": "3.2",
        "sensors": ["Stereo Module", "RGB Camera"],
        "asic_temperature_celsius": 32.0,
        "projector_temperature_celsius": 30.0,
    }
    mock_cam_instance.read_with_raw_and_colorized.return_value = (
        mock_color,
        mock_depth_m,
        mock_depth_m,
        mock_raw_z16,
        mock_colorized,
        {"timestamp_ms": 1000.0, "frame_number": 1},
    )

    with patch("tools.realsense_hardware_smoke_test.detect_realsense_devices", return_value=mock_devs), \
         patch("tools.realsense_hardware_smoke_test.RealSenseCamera", return_value=mock_cam_instance), \
         patch("tools.realsense_hardware_smoke_test.query_stream_intrinsics") as mock_intr:

        mock_intr.return_value = {
            "color": IntrinsicsInfo("color", 848, 480, 428.0, 428.0, 422.0, 244.0, "inverse_brown_conrady", [0.0]*5),
            "depth_raw": IntrinsicsInfo("depth_raw", 848, 480, 433.0, 433.0, 426.0, 244.0, "brown_conrady", [0.0]*5),
        }

        cfg = CameraConfig(width=848, height=480, fps=30)
        rec_dir = tmp_path / "mock_recording"

        runner = HardwareSmokeTestRunner(
            camera_cfg=cfg,
            ref_distance_m=1.500,
            record_dir=rec_dir,
            headless=True,
            target_frames=5,
        )

        summary = runner.run()

        assert summary["session_status"] == "completed"
        assert summary["stream_config"]["total_frames_received"] == 5
        assert summary["depth_health"]["average_valid_pct"] == pytest.approx(100.0)
        assert summary["roi_analysis"]["median_m"] == pytest.approx(1.5, abs=1e-3)
        assert summary["roi_analysis"]["error_median_mm"] == pytest.approx(0.0, abs=1e-2)
        assert summary["temporal_stability"]["is_stable"] is True
        assert (rec_dir / "meta.json").exists()


# ===========================================================================
# 7. Physical Hardware Integration Test (Conditional)
# ===========================================================================

def is_physical_d455_connected() -> bool:
    """Check whether a physical RealSense camera is currently detectable."""
    try:
        devs = detect_realsense_devices()
        return len(devs) > 0
    except Exception:
        return False


@pytest.mark.skipif(not is_physical_d455_connected(), reason="Physical RealSense camera not detected")
def test_physical_camera_smoke_test():
    """
    Live hardware smoke test on physical Intel RealSense camera.
    Only executed if camera is physically connected to host system.
    """
    cfg = CameraConfig(width=848, height=480, fps=30)
    runner = HardwareSmokeTestRunner(
        camera_cfg=cfg,
        headless=True,
        target_frames=5,
    )
    summary = runner.run()

    assert summary["session_status"] == "completed"
    assert summary["stream_config"]["total_frames_received"] == 5
    assert summary["device"]["name"] is not None
    assert summary["intrinsics"]["depth_scale_m"] == pytest.approx(0.001, abs=1e-4)

