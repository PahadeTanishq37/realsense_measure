"""
tests/body/test_body_burst_capture.py
=====================================
Unit tests for body burst capture, settling countdown, per-pixel median depth,
and stillness motion tracking under simulated stationary and moving subjects.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from realsense_measure.body.capture_motion import (
    BurstMotionResult,
    compute_frame_centroids,
    evaluate_burst_stillness,
    extract_subject_mask,
)
from realsense_measure.camera.frame import RGBDFrame, compute_burst_median_frame
from realsense_measure.camera.recording import save_raw_capture
from realsense_measure.config import PipelineConfig


def _create_synthetic_body_frame(
    h: int = 120,
    w: int = 160,
    center_x: int = 80,
    center_y: int = 60,
    radius: int = 25,
    depth_val_m: float = 1.50,
    noise_sigma_m: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simulated person mask with metric depth and BGR color."""
    color_bgr = np.full((h, w, 3), 180, dtype=np.uint8)
    depth_m = np.zeros((h, w), dtype=np.float32)

    # Elliptical body silhouette
    yy, xx = np.ogrid[:h, :w]
    dist_sq = ((xx - center_x) / radius) ** 2 + ((yy - center_y) / (radius * 1.5)) ** 2
    body_mask = dist_sq <= 1.0

    color_bgr[body_mask] = [50, 120, 200]  # Foreground person color
    noise = np.random.normal(0, noise_sigma_m, size=(h, w)).astype(np.float32)
    depth_m[body_mask] = depth_val_m + noise[body_mask]

    return color_bgr, depth_m


def test_still_subject_within_10mm_threshold():
    """Verify a stationary subject has motion <= 10.0 mm and is not flagged."""
    np.random.seed(42)
    burst_colors = []
    burst_depths = []
    burst_timestamps = []

    # 15 frames over 4.0 seconds (simulated 30 fps sub-sampled)
    for i in range(15):
        t = 3.0 + i * (4.0 / 14.0)  # Starts at 4th second (t=3.0)
        c, d = _create_synthetic_body_frame(
            center_x=80,
            center_y=60,
            depth_val_m=1.50,
            noise_sigma_m=0.001,  # 1 mm noise
        )
        burst_colors.append(c)
        burst_depths.append(d)
        burst_timestamps.append(t)

    intrinsics = (160.0, 160.0, 80.0, 60.0)
    res: BurstMotionResult = evaluate_burst_stillness(
        burst_colors,
        burst_depths,
        burst_timestamps=burst_timestamps,
        intrinsics=intrinsics,
        threshold_mm=10.0,
    )

    assert res.frames_used == 15
    assert 3.9 <= res.real_elapsed_sec <= 4.1
    # Motion from 1 mm noise should be well under 10 mm
    assert res.motion_mm < 10.0
    assert res.is_flagged is False


def test_moving_subject_silhouette_lateral_displacement_flagged():
    """Verify lateral motion of 25 mm flags the burst capture."""
    np.random.seed(42)
    burst_colors = []
    burst_depths = []
    burst_timestamps = []

    # Subject sways laterally from x=80 to x=86 (+6 pixels at fx=160, z=1.5m -> ~56 mm)
    for i in range(15):
        t = 3.0 + i * (4.0 / 14.0)
        shift_x = int(np.round(6.0 * (i / 14.0)))
        c, d = _create_synthetic_body_frame(
            center_x=80 + shift_x,
            center_y=60,
            depth_val_m=1.50,
            noise_sigma_m=0.001,
        )
        burst_colors.append(c)
        burst_depths.append(d)
        burst_timestamps.append(t)

    intrinsics = (160.0, 160.0, 80.0, 60.0)
    res: BurstMotionResult = evaluate_burst_stillness(
        burst_colors,
        burst_depths,
        burst_timestamps=burst_timestamps,
        intrinsics=intrinsics,
        threshold_mm=10.0,
    )

    assert res.frames_used == 15
    assert res.motion_mm > 10.0
    assert res.silhouette_motion_mm > 10.0
    assert res.is_flagged is True


def test_moving_subject_depth_displacement_flagged():
    """Verify depth motion (swaying forward/backward by 20 mm) flags the burst capture."""
    np.random.seed(42)
    burst_colors = []
    burst_depths = []
    burst_timestamps = []

    # Subject moves in depth from 1.500m to 1.525m (+25 mm)
    for i in range(15):
        t = 3.0 + i * (4.0 / 14.0)
        z_shift = 0.025 * (i / 14.0)
        c, d = _create_synthetic_body_frame(
            center_x=80,
            center_y=60,
            depth_val_m=1.50 + z_shift,
            noise_sigma_m=0.001,
        )
        burst_colors.append(c)
        burst_depths.append(d)
        burst_timestamps.append(t)

    intrinsics = (160.0, 160.0, 80.0, 60.0)
    res: BurstMotionResult = evaluate_burst_stillness(
        burst_colors,
        burst_depths,
        burst_timestamps=burst_timestamps,
        intrinsics=intrinsics,
        threshold_mm=10.0,
    )

    assert res.frames_used == 15
    assert res.depth_motion_mm >= 20.0
    assert res.motion_mm > 10.0
    assert res.is_flagged is True


def test_compute_burst_median_frame_50_percent_valid():
    """Verify per-pixel median requires >= 50% valid pixels and rejects intermittent dropouts."""
    h, w = 40, 40
    burst_c = [np.full((h, w, 3), 100, dtype=np.uint8) for _ in range(10)]
    burst_d = [np.zeros((h, w), dtype=np.float32) for _ in range(10)]

    # Pixel (10, 10): 6 valid frames (60% valid >= 50%) -> should have valid median
    for f in range(6):
        burst_d[f][10, 10] = 1.50 + f * 0.01

    # Pixel (20, 20): only 4 valid frames (40% valid < 50%) -> should be set to 0.0
    for f in range(4):
        burst_d[f][20, 20] = 1.50 + f * 0.01

    med_c, med_d = compute_burst_median_frame(burst_c, burst_d, min_valid_frac=0.50)

    assert med_d[10, 10] > 0.0
    assert pytest.approx(med_d[10, 10], abs=1e-3) == 1.525
    assert med_d[20, 20] == 0.0


def test_save_raw_metadata_includes_burst_summary(tmp_path: Path):
    """Verify that save_raw_capture preserves burst summary metrics into meta.json."""
    c, d = _create_synthetic_body_frame()
    import open3d as o3d

    intr = o3d.camera.PinholeCameraIntrinsic(160, 120, 160.0, 160.0, 80.0, 60.0)
    frame = RGBDFrame(
        color_bgr=c,
        depth_m=d,
        frame_id=0,
        timestamp=3.0,
        pcd=o3d.geometry.PointCloud(),
        intrinsics=intr,
    )
    frame.quality_metrics = {
        "median_dist_m": 1.50,
        "burst_frames_used": 120,
        "burst_elapsed_sec": 4.005,
        "burst_motion_mm": 14.2,
        "burst_motion_flagged": True,
    }

    cfg = PipelineConfig()
    cfg.target.name = "body"
    out_dir = tmp_path / "raw_test"

    save_raw_capture([frame], out_dir, cfg)

    meta_file = out_dir / "meta.json"
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    frame_meta = doc["frames"][0]
    assert frame_meta["burst_frames_used"] == 120
    assert frame_meta["burst_elapsed_sec"] == 4.005
    assert frame_meta["burst_motion_mm"] == 14.2
    assert frame_meta["burst_motion_flagged"] is True


def test_body_capture_report_prints_burst_summary(tmp_path: Path, capsys):
    """Verify that tools/body_capture_report.py reads burst metrics and prints summary table."""
    from tools.body_capture_report import generate_body_report
    import open3d as o3d

    c, d = _create_synthetic_body_frame()
    intr = o3d.camera.PinholeCameraIntrinsic(160, 120, 160.0, 160.0, 80.0, 60.0)
    frame = RGBDFrame(
        color_bgr=c,
        depth_m=d,
        frame_id=0,
        timestamp=3.0,
        pcd=o3d.geometry.PointCloud(),
        intrinsics=intr,
    )
    frame.quality_metrics = {
        "median_dist_m": 1.50,
        "burst_frames_used": 120,
        "burst_elapsed_sec": 4.005,
        "burst_motion_mm": 14.2,
        "burst_motion_flagged": True,
    }

    cfg = PipelineConfig()
    cfg.target.name = "body"
    out_dir = tmp_path / "raw_report_test"
    save_raw_capture([frame], out_dir, cfg)

    # Run report generator
    res = generate_body_report(out_dir)
    assert len(res) == 1
    assert res[0]["burst_frames_used"] == 120
    assert res[0]["burst_motion_mm"] == 14.2
    assert res[0]["burst_motion_flagged"] is True

    captured = capsys.readouterr()
    assert "BURST CAPTURE & STILLNESS MOTION SUMMARY" in captured.out
    assert "120 frames" in captured.out
    assert "14.2 mm" in captured.out
    assert "FLAGGED" in captured.out

def test_hands_free_auto_sequencing(monkeypatch):
    """Verify that --target body automatically sequences through all yaw stops without user keypresses."""
    from unittest.mock import MagicMock
    import cv2
    from realsense_measure.pipeline import ScanPipeline
    import open3d as o3d

    # Mock cv2 GUI calls so test runs headlessly without opening desktop windows
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 0xFF)  # No keys pressed (255)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)

    cfg = PipelineConfig()
    cfg.target.name = "body"
    cfg.yaw_stops = 4
    cfg.buffer_seconds = 0.05   # Fast buffer for unit test
    cfg.burst_seconds = 0.05    # Fast burst for unit test

    pipeline = ScanPipeline(cfg)

    # Mock camera yielding synthetic frames
    c_syn, d_syn = _create_synthetic_body_frame(h=120, w=160)
    dc_syn = np.full((120, 160, 3), 50, dtype=np.uint8)

    mock_cam = MagicMock()
    mock_cam.read_with_colorized.return_value = (c_syn, d_syn, dc_syn)
    mock_cam.intrinsics = MagicMock(fx=160.0, fy=160.0, cx=80.0, cy=60.0)

    intr = o3d.camera.PinholeCameraIntrinsic(160, 120, 160.0, 160.0, 80.0, 60.0)
    def _create_rgbd(color_bgr, depth_m, frame_id, timestamp):
        return RGBDFrame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            frame_id=frame_id,
            timestamp=timestamp,
            pcd=o3d.geometry.PointCloud(),
            intrinsics=intr,
        )
    mock_cam.create_rgbd_frame.side_effect = _create_rgbd

    # Run auto-sequencing capture loop
    frames = pipeline._capture_frames_body(mock_cam)

    assert len(frames) == 4, f"Expected 4 frames for 4 yaw stops, got {len(frames)}"
    for i, f in enumerate(frames):
        assert f.frame_id == i
        assert "burst_frames_used" in f.quality_metrics
        assert "burst_elapsed_sec" in f.quality_metrics
        assert "burst_motion_mm" in f.quality_metrics
        assert "burst_motion_flagged" in f.quality_metrics
        assert f.quality_metrics["burst_motion_flagged"] is False

    # Check camera was stopped cleanly
    mock_cam.stop.assert_called_once()
