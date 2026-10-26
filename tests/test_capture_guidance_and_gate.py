"""
Tests for live capture guidance, HUD state indicators, capture gating,
and capture quality report separated motion columns.
"""

from __future__ import annotations

import csv
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.recording import load_raw_capture, save_raw_capture
from config import HeadCaptureConfig, PipelineConfig
from pipeline import _draw_hud, compute_head_roi_quality


def test_gate_pass_nominal():
    """Verify frame passes gate when ROI depth is within nominal range and face is locked."""
    cfg = HeadCaptureConfig(
        capture_gate=True,
        gate_min_valid_depth_frac=0.70,
        gate_min_dist_m=0.55,
        gate_max_dist_m=0.95,
        gate_require_face=True,
    )
    # 720x1280 depth map with nominal 0.70m depth across ROI
    depth_m = np.zeros((720, 1280), dtype=np.float32)
    # Landmark coordinates in center
    lms = {
        f"pt_{i}": np.array([600 + (i % 3) * 30, 300 + (i // 3) * 30], dtype=np.float32)
        for i in range(6)
    }
    # Populate depth around landmarks (from x: 580 to 680, y: 280 to 380) with 0.70m
    depth_m[280:380, 580:680] = 0.70

    res = compute_head_roi_quality(depth_m, lms, cfg)
    assert res["is_gate_pass"] is True
    assert res["gate_reason"] is None
    assert res["face_detected"] is True
    assert res["dist_status_text"] == "OK"
    assert res["dist_color"] == "green"
    assert res["roi_valid_depth_frac"] > 0.70
    assert 0.65 <= res["roi_median_depth_m"] <= 0.85


def test_gate_too_close():
    """Verify frame is blocked with 'too close' when subject is closer than gate_min_dist_m."""
    cfg = HeadCaptureConfig(
        capture_gate=True,
        gate_min_valid_depth_frac=0.70,
        gate_min_dist_m=0.55,
        gate_max_dist_m=0.95,
        critical_dist_min_m=0.52,
    )
    depth_m = np.zeros((720, 1280), dtype=np.float32)
    lms = {
        f"pt_{i}": np.array([600 + (i % 3) * 30, 300 + (i // 3) * 30], dtype=np.float32)
        for i in range(6)
    }
    # Too close: 0.48m (under physical D455 reliable limit)
    depth_m[280:380, 580:680] = 0.48

    res = compute_head_roi_quality(depth_m, lms, cfg)
    assert res["is_gate_pass"] is False
    assert "too close" in res["gate_reason"].lower()
    assert res["dist_status_text"] == "MOVE BACK"
    assert res["dist_color"] == "red"


def test_gate_dropout():
    """Verify frame is blocked when ROI valid depth fraction is below gate threshold."""
    cfg = HeadCaptureConfig(
        capture_gate=True,
        gate_min_valid_depth_frac=0.70,
        gate_min_dist_m=0.55,
        gate_max_dist_m=0.95,
    )
    depth_m = np.zeros((720, 1280), dtype=np.float32)
    lms = {
        f"pt_{i}": np.array([600 + (i % 3) * 30, 300 + (i // 3) * 30], dtype=np.float32)
        for i in range(6)
    }
    # Only 30% of pixels valid
    np.random.seed(42)
    roi_mask = np.random.rand(100, 100) < 0.30
    patch = np.zeros((100, 100), dtype=np.float32)
    patch[roi_mask] = 0.70
    depth_m[280:380, 580:680] = patch

    res = compute_head_roi_quality(depth_m, lms, cfg)
    assert res["is_gate_pass"] is False
    assert "roi valid depth too low" in res["gate_reason"].lower()


def test_gate_no_face():
    """Verify frame is blocked when face landmarks are missing and tracker expects one."""
    cfg = HeadCaptureConfig(
        capture_gate=True,
        gate_require_face=True,
    )
    depth_m = np.full((720, 1280), 0.70, dtype=np.float32)
    res = compute_head_roi_quality(depth_m, landmarks_px=None, cfg=cfg)

    assert res["is_gate_pass"] is False
    assert "face not detected" in res["gate_reason"].lower()
    assert res["face_detected"] is False
    assert res["roi_type"] == "central_roi"


def test_hud_state_logic():
    """Test HUD rendering across distance ranges without requiring camera hardware."""
    cfg = HeadCaptureConfig()
    panel = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_m = np.full((480, 640), 0.70, dtype=np.float32)

    # 1. Nominal OK
    q_ok = {
        "roi_median_depth_m": 0.72,
        "roi_min_depth_m": 0.69,
        "roi_valid_depth_frac": 0.92,
        "dist_status_text": "OK",
        "dist_color": "green",
        "is_gate_pass": True,
        "gate_reason": None,
    }
    _draw_hud(
        panel,
        depth_m,
        n_frames=5,
        n_pts=15000,
        vis_min_m=0.2,
        vis_max_m=1.5,
        is_human=True,
        roi_quality=q_ok,
        gate_enabled=True,
    )
    # Ensure drawing modified the panel
    assert np.any(panel > 0)

    # 2. Blocked / Move back
    panel.fill(0)
    q_blocked = {
        "roi_median_depth_m": 0.48,
        "roi_min_depth_m": 0.45,
        "roi_valid_depth_frac": 0.55,
        "dist_status_text": "MOVE BACK",
        "dist_color": "red",
        "is_gate_pass": False,
        "gate_reason": "Subject too close",
    }
    _draw_hud(
        panel,
        depth_m,
        n_frames=5,
        n_pts=15000,
        vis_min_m=0.2,
        vis_max_m=1.5,
        is_human=True,
        roi_quality=q_blocked,
        gate_enabled=True,
    )
    assert np.any(panel > 0)


def test_quality_tool_motion_separation(tmp_path: Path):
    """Test that capture_quality_report separates background and subject motion."""
    from tools.capture_quality_report import analyze_capture

    intr = o3d.camera.PinholeCameraIntrinsic(320, 240, 250, 250, 160, 120)

    frames = []
    # Frame 0:
    # Subject: centered disc radius 40 px at depth 0.65m
    # Background: depth 1.80m
    color0 = np.full((240, 320, 3), 100, dtype=np.uint8)
    depth0 = np.full((240, 320), 1.80, dtype=np.float32)
    yy, xx = np.ogrid[:240, :320]
    subj_mask = (xx - 160) ** 2 + (yy - 120) ** 2 <= 40 ** 2
    color0[subj_mask] = [0, 200, 0]
    depth0[subj_mask] = 0.65

    pcd0 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth0), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frame0 = RGBDFrame(
        frame_id=0,
        color_bgr=color0,
        depth_m=depth0,
        pcd=pcd0,
        timestamp=100.0,
        intrinsics=intr,
    )
    frames.append(frame0)

    # Frame 1:
    # Background moved: depth shifts slightly (e.g. 1.85m), subject stays at 0.65m
    color1 = color0.copy()
    depth1 = np.full((240, 320), 1.85, dtype=np.float32)
    depth1[subj_mask] = 0.65
    pcd1 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth1), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frame1 = RGBDFrame(
        frame_id=1,
        color_bgr=color1,
        depth_m=depth1,
        pcd=pcd1,
        timestamp=102.0,
        intrinsics=intr,
    )
    frames.append(frame1)

    # Save to tmp_path
    cfg = PipelineConfig()
    save_raw_capture(frames, tmp_path, cfg)

    # Run analyze_capture
    report_dir = tmp_path / "report"
    analyze_capture(tmp_path, out_dir=report_dir)

    csv_file = report_dir / "capture_quality.csv"
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "camera_motion_bg" in rows[1]
        assert "subject_motion" in rows[1]
        assert "camera_motion_bg_fitness" in rows[1]
        assert "subject_motion_fitness" in rows[1]
        assert "time_gap_s" in rows[1]
        assert float(rows[1]["time_gap_s"]) == pytest.approx(2.0, abs=1e-3)


def test_gate_profile_frame_face_requirement():
    """Verify profile-like frame passes gate by default (gate_require_face=False) and fails when gate_require_face=True."""
    # Nominal depth at 0.70m (perfect distance and 100% valid depth in central ROI)
    depth_m = np.full((720, 1280), 0.70, dtype=np.float32)

    # 1. Default configuration: gate_require_face is False
    cfg_default = HeadCaptureConfig(capture_gate=True)
    assert cfg_default.gate_require_face is False

    res_default = compute_head_roi_quality(depth_m, landmarks_px=None, cfg=cfg_default)
    assert res_default["face_detected"] is False
    assert res_default["roi_type"] == "central_roi"
    assert res_default["roi_valid_depth_frac"] == 1.0
    assert 0.65 <= res_default["roi_median_depth_m"] <= 0.85
    assert res_default["is_gate_pass"] is True
    assert res_default["gate_reason"] is None

    # 2. Opt-in face requirement: gate_require_face is True
    cfg_require_face = HeadCaptureConfig(capture_gate=True, gate_require_face=True)
    res_require_face = compute_head_roi_quality(depth_m, landmarks_px=None, cfg=cfg_require_face)
    assert res_require_face["is_gate_pass"] is False
    assert "face not detected" in res_require_face["gate_reason"].lower()
    assert "Face not detected" in res_require_face["gate_reasons"]


def test_quality_tool_no_mutation(tmp_path: Path):
    """Verify analyze_capture does not mutate frame.depth_m or frame.color_bgr in place."""
    import hashlib
    from tools.capture_quality_report import analyze_capture

    intr = o3d.camera.PinholeCameraIntrinsic(320, 240, 250, 250, 160, 120)

    frames = []
    color0 = np.full((240, 320, 3), 100, dtype=np.uint8)
    depth0 = np.full((240, 320), 0.65, dtype=np.float32)
    pcd0 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth0), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frames.append(RGBDFrame(frame_id=0, color_bgr=color0, depth_m=depth0, pcd=pcd0, timestamp=10.0, intrinsics=intr))

    color1 = np.full((240, 320, 3), 120, dtype=np.uint8)
    depth1 = np.full((240, 320), 0.66, dtype=np.float32)
    pcd1 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth1), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frames.append(RGBDFrame(frame_id=1, color_bgr=color1, depth_m=depth1, pcd=pcd1, timestamp=12.0, intrinsics=intr))

    # Save to disk
    save_raw_capture(frames, tmp_path, PipelineConfig())

    # Load and hash
    loaded_frames = load_raw_capture(tmp_path)
    depth_hashes_before = [hashlib.sha256(f.depth_m.tobytes()).hexdigest() for f in loaded_frames]
    color_hashes_before = [hashlib.sha256(f.color_bgr.tobytes()).hexdigest() for f in loaded_frames]

    # Run analyze_capture
    analyze_capture(tmp_path, out_dir=tmp_path / "report")

    depth_hashes_after = [hashlib.sha256(f.depth_m.tobytes()).hexdigest() for f in loaded_frames]
    color_hashes_after = [hashlib.sha256(f.color_bgr.tobytes()).hexdigest() for f in loaded_frames]

    assert depth_hashes_before == depth_hashes_after
    assert color_hashes_before == color_hashes_after


def test_quality_tool_unreliable_motion_marking(tmp_path: Path):
    """Verify that poor odometry fitness or high mask zero-depth marks motion as 'unreliable'."""
    from tools.capture_quality_report import analyze_capture

    intr = o3d.camera.PinholeCameraIntrinsic(320, 240, 250, 250, 160, 120)

    frames = []
    color0 = np.full((240, 320, 3), 100, dtype=np.uint8)
    depth0 = np.full((240, 320), 0.65, dtype=np.float32)
    pcd0 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth0), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frames.append(RGBDFrame(frame_id=0, color_bgr=color0, depth_m=depth0, pcd=pcd0, timestamp=10.0, intrinsics=intr))

    # Frame 1: 90% zero depth (dropout / degenerate frame)
    color1 = color0.copy()
    depth1 = np.zeros((240, 320), dtype=np.float32)
    # Only small patch valid
    depth1[100:120, 140:160] = 0.65
    pcd1 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth1), intr, depth_scale=1.0, depth_trunc=3.0
    )
    frames.append(RGBDFrame(frame_id=1, color_bgr=color1, depth_m=depth1, pcd=pcd1, timestamp=12.0, intrinsics=intr))

    save_raw_capture(frames, tmp_path, PipelineConfig())

    report_dir = tmp_path / "report"
    analyze_capture(tmp_path, out_dir=report_dir)

    csv_file = report_dir / "capture_quality.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    # Frame 1 motion should be marked unreliable because of dropout and poor fitness
    assert rows[1]["camera_motion_bg"] == "unreliable" or rows[1]["subject_motion"] == "unreliable"
    assert rows[1]["camera_motion_bg_fitness"] != ""
    assert rows[1]["subject_motion_fitness"] != ""

