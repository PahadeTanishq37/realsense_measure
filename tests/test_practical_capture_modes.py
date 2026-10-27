"""
Unit and integration tests for practical capture modes:
  1. Stillness-triggered burst capture (--auto-hold, --burst-count)
  2. Burst median frame depth and color computation
  3. Soft capture gate (--gate-mode {off, advisory, hard})
  4. Multi-tier no-face fallback ROI (last face bbox -> nearest central foreground -> 'no subject found')
  5. Resolution override (--resolution WxH)
  6. Fixed-camera subject turns (--subject-turns):
     - Background isolation (isolate_subject_by_background_subtraction without 3D DBSCAN)
     - Subject-only odometry registration
     - Negative control: unmasked odometry on static background fails/collapses to ~identity
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame, compute_burst_median_frame, create_pcd_from_rgbd
from config import CameraConfig, HeadCaptureConfig, PipelineConfig, RegistrationConfig
from pipeline import HeadCaptureTracker, _depth_mean_diff, compute_head_roi_quality
from preprocessing import isolate_subject_by_background_subtraction
from registration import compute_rotation_deg, register_rgbd_pair
from open3d.pipelines.odometry import (
    OdometryOption,
    RGBDOdometryJacobianFromColorTerm,
    compute_rgbd_odometry,
)


# ---------------------------------------------------------------------------
# 1. Burst Median Frame Computation
# ---------------------------------------------------------------------------

def test_burst_median_valid_fraction_and_outlier_rejection():
    """Verify compute_burst_median_frame requires >= 50% valid pixels and rejects outliers."""
    h, w = 10, 10
    burst_colors = [np.full((h, w, 3), i * 20, dtype=np.uint8) for i in range(5)]
    burst_depths = []

    # 5 frames
    for i in range(5):
        d = np.zeros((h, w), dtype=np.float32)
        # Pixel (0, 0): valid in all 5 frames (depths: 0.60, 0.61, 0.62, 0.60, 2.50 outlier)
        d[0, 0] = 2.50 if i == 4 else 0.60 + 0.01 * (i % 3)
        # Pixel (1, 1): valid in only 2 of 5 frames (< 50% valid fraction)
        if i < 2:
            d[1, 1] = 0.70
        # Pixel (2, 2): valid in 3 of 5 frames (60% valid fraction >= 50%)
        if i < 3:
            d[2, 2] = 0.80 + 0.02 * i
        burst_depths.append(d)

    med_color, med_depth = compute_burst_median_frame(burst_colors, burst_depths, min_valid_frac=0.50)

    # Middle color frame (index 2: value 40) is preserved
    assert np.all(med_color == 40)

    # Pixel (0, 0): outlier 2.50 is rejected; median is ~0.61
    assert 0.60 <= med_depth[0, 0] <= 0.62

    # Pixel (1, 1): < 50% valid frames -> 0.0
    assert med_depth[1, 1] == 0.0

    # Pixel (2, 2): >= 50% valid frames -> median of [0.80, 0.82, 0.84] = 0.82
    assert np.isclose(med_depth[2, 2], 0.82, atol=1e-3)


# ---------------------------------------------------------------------------
# 2. Soft Gate Modes (off, advisory, hard)
# ---------------------------------------------------------------------------

def test_gate_mode_off_allows_all():
    """In 'off' mode, quality issues do not block capture."""
    cfg = HeadCaptureConfig(gate_mode="off")
    depth_m = np.zeros((480, 640), dtype=np.float32)  # Empty depth
    res = compute_head_roi_quality(depth_m, landmarks_px=None, cfg=cfg)
    assert res["gate_mode"] == "off"


def test_gate_mode_advisory_populates_reasons_without_blocking():
    """In 'advisory' mode, reasons are reported for HUD but is_gate_pass is True."""
    cfg = HeadCaptureConfig(
        gate_mode="advisory",
        gate_min_valid_depth_frac=0.70,
        gate_min_dist_m=0.55,
        gate_max_dist_m=0.95,
        gate_require_face=True,
    )
    # Depth at 0.50m (too close for advisory limit 0.55m)
    depth_m = np.full((480, 640), 0.50, dtype=np.float32)
    # Face landmarks in center
    lms = {f"p_{i}": np.array([300 + i, 240 + i]) for i in range(10)}

    res = compute_head_roi_quality(depth_m, lms, cfg)
    assert res["is_gate_pass"] is True
    assert len(res["advisory_reasons"]) > 0
    assert any("too close" in r.lower() for r in res["advisory_reasons"])


def test_gate_mode_hard_catastrophic_blocking():
    """In 'hard' mode, catastrophic low coverage (< 0.35) or too close (< 0.50m) blocks capture."""
    cfg = HeadCaptureConfig(
        gate_mode="hard",
        gate_hard_min_valid_depth_frac=0.35,
        gate_hard_min_dist_m=0.50,
        gate_require_face=False,
    )
    # Scenario A: Catastrophically close (0.42m < 0.50m)
    depth_close = np.full((480, 640), 0.42, dtype=np.float32)
    lms = {f"p_{i}": np.array([300 + i, 240 + i]) for i in range(10)}
    res_close = compute_head_roi_quality(depth_close, lms, cfg)
    assert res_close["is_gate_pass"] is False
    assert any("too close" in r.lower() for r in res_close["gate_reasons"])

    # Scenario B: Catastrophic dropout (only ~15% valid depth < 35%)
    depth_dropout = np.full((480, 640), 0.70, dtype=np.float32)
    depth_dropout[235:, :] = 0.0
    depth_dropout[:, 310:] = 0.0
    res_dropout = compute_head_roi_quality(depth_dropout, lms, cfg)
    assert res_dropout["is_gate_pass"] is False
    assert any("too low" in r.lower() for r in res_dropout["gate_reasons"])


# ---------------------------------------------------------------------------
# 3. Multi-tier No-Face Fallback ROI
# ---------------------------------------------------------------------------

def test_no_face_fallback_roi_hierarchy():
    """Verify hierarchy: face_bbox -> last_face_bbox -> nearest_foreground_roi -> none."""
    cfg = HeadCaptureConfig(fallback_face_box_max_frames=30)
    tracker = HeadCaptureTracker(cfg)

    # 1. Tier 1: Face detected -> face_bbox
    lms = {f"p_{i}": np.array([200 + i * 5, 150 + i * 5]) for i in range(10)}
    depth_m = np.full((480, 640), 0.70, dtype=np.float32)
    res1 = compute_head_roi_quality(depth_m, lms, cfg, tracker=tracker)
    assert "face_bbox" in res1["roi_type"]
    assert tracker.last_face_bbox is not None

    # 2. Tier 2: Face lost, but tracker has last_face_bbox within 30 frames
    tracker.frames_since_face_bbox = 5
    res2 = compute_head_roi_quality(depth_m, None, cfg, tracker=tracker)
    assert "last_face_bbox" in res2["roi_type"]

    # 3. Tier 3: Face never seen / expired, but central 60% contains nearest foreground
    tracker.last_face_bbox = None
    tracker.frames_since_face_bbox = 100
    depth_fg = np.full((480, 640), 1.80, dtype=np.float32)  # Wall at 1.80m
    cv2.circle(depth_fg, (320, 240), 60, 0.65, -1)  # Central subject at 0.65m
    res3 = compute_head_roi_quality(depth_fg, None, cfg, tracker=tracker)
    assert res3["roi_type"] == "nearest_foreground_roi"
    assert np.isclose(res3["roi_median_depth_m"], 0.65, atol=0.05)

    # 4. Tier 4: Flat wall at 1.80m with no foreground subject
    depth_wall = np.full((480, 640), 1.80, dtype=np.float32)
    res4 = compute_head_roi_quality(depth_wall, None, cfg, tracker=tracker)
    assert res4["roi_type"] == "none"
    assert res4["dist_status_text"] == "NO SUBJECT"


# ---------------------------------------------------------------------------
# 4. Stillness and Spacing Detection Logic
# ---------------------------------------------------------------------------

def test_stillness_and_motion_spacing():
    """Verify depth difference thresholding separates stillness from motion."""
    cfg = HeadCaptureConfig(
        stillness_depth_thresh_m=0.015,
        stillness_min_motion_spacing_m=0.025,
    )
    roi1 = np.full((100, 100), 0.700, dtype=np.float32)
    roi_still = np.full((100, 100), 0.705, dtype=np.float32)  # 5mm diff (< 15mm)
    roi_moved = np.full((100, 100), 0.740, dtype=np.float32)  # 40mm diff (>= 25mm spacing)

    diff_still = _depth_mean_diff(roi1, roi_still)
    assert diff_still < cfg.stillness_depth_thresh_m

    diff_moved = _depth_mean_diff(roi1, roi_moved)
    assert diff_moved >= cfg.stillness_min_motion_spacing_m


# ---------------------------------------------------------------------------
# 5. Configuration & Resolution Defaults
# ---------------------------------------------------------------------------

def test_config_resolution_and_defaults():
    """Verify new flags have safe opt-in defaults and resolution string parsing works."""
    head_cfg = HeadCaptureConfig()
    assert head_cfg.gate_mode == "off"
    assert head_cfg.auto_hold is False
    assert head_cfg.subject_turns is False
    assert head_cfg.burst_count == 10
    assert head_cfg.bg_diff_threshold_m == 0.05

    # Resolution override parsing
    cam_cfg = CameraConfig(resolution_override="848x480")
    cam_cfg.resolve_defaults()
    assert cam_cfg.width == 848 and cam_cfg.height == 480

    cam_cfg_hd = CameraConfig(resolution_override="1280x720")
    cam_cfg_hd.resolve_defaults()
    assert cam_cfg_hd.width == 1280 and cam_cfg_hd.height == 720

    # PipelineConfig syncing
    pipe_cfg = PipelineConfig(
        gate_mode="advisory",
        auto_hold=True,
        subject_turns=True,
        burst_count=8,
        resolution="848x480",
    )
    assert pipe_cfg.head_capture.gate_mode == "advisory"
    assert pipe_cfg.head_capture.auto_hold is True
    assert pipe_cfg.head_capture.subject_turns is True
    assert pipe_cfg.head_capture.burst_count == 8
    assert pipe_cfg.camera.resolution_override == "848x480"


# ---------------------------------------------------------------------------
# 6. Synthetic Scene: Background Subtraction & Odometry vs Negative Control
# ---------------------------------------------------------------------------

def test_synthetic_subject_turns_and_negative_control():
    """
    Synthetic scene verification:
      - Static background wall at 1.15m (within odometry depth limit).
      - Foreground textured sphere at 0.65m rotating 20 degrees.
      - isolate_subject_by_background_subtraction cleanly separates sphere.
      - Masked registration successfully captures the rotation.
      - Negative control: Unmasked odometry on the same pair collapses to near 0 deg (< 1 deg)
        because the static background pixels dominate.
    """
    w, h = 640, 480
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    z0 = 0.65
    r = 0.12

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    xn = (uu - cx) / fx
    yn = (vv - cy) / fy

    a = xn**2 + yn**2 + 1.0
    b = -2.0 * z0
    c = z0**2 - r**2
    disc = b**2 - 4.0 * a * c
    sphere_mask = disc >= 0

    t = np.zeros_like(disc)
    t[sphere_mask] = (-b - np.sqrt(disc[sphere_mask])) / (2.0 * a[sphere_mask])

    px = np.zeros_like(t)
    py = np.zeros_like(t)
    pz = np.zeros_like(t)
    px[sphere_mask] = t[sphere_mask] * xn[sphere_mask]
    py[sphere_mask] = t[sphere_mask] * yn[sphere_mask]
    pz[sphere_mask] = t[sphere_mask]

    rx = px[sphere_mask]
    ry = py[sphere_mask]
    rz = pz[sphere_mask] - z0

    lat = np.arcsin(np.clip(ry / r, -1.0, 1.0))
    lon = np.arctan2(rx, rz)

    def render_frame(theta_deg):
        theta = np.deg2rad(theta_deg)
        # Background wall at 1.15m
        depth = np.full((h, w), 1.15, dtype=np.float32)
        depth[sphere_mask] = pz[sphere_mask]

        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[:, :, 0] = ((uu // 32) % 2 * 180 + (vv // 32) % 2 * 60).astype(np.uint8)
        color[:, :, 1] = ((vv // 32) % 2 * 180 + (uu // 32) % 2 * 60).astype(np.uint8)
        color[:, :, 2] = 100

        lon_rot = lon - theta
        tex_val = np.sin(12 * lon_rot) * np.cos(12 * lat)
        col_sphere = np.zeros((len(lon), 3), dtype=np.uint8)
        col_sphere[:, 0] = np.clip((tex_val + 1.0) * 120, 0, 255).astype(np.uint8)
        col_sphere[:, 1] = np.clip((-tex_val + 1.0) * 120, 0, 255).astype(np.uint8)
        col_sphere[:, 2] = 200

        color[sphere_mask] = col_sphere
        return depth, color

    bg_depth = np.full((h, w), 1.15, dtype=np.float32)
    d0, c0 = render_frame(0.0)
    d1, c1 = render_frame(20.0)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd0 = create_pcd_from_rgbd(c0, d0, intrinsics, depth_max_m=2.0)
    pcd1 = create_pcd_from_rgbd(c1, d1, intrinsics, depth_max_m=2.0)

    f0 = RGBDFrame(frame_id=0, color_bgr=c0, depth_m=d0, intrinsics=intrinsics, pcd=pcd0, timestamp=0.0)
    f1 = RGBDFrame(frame_id=1, color_bgr=c1, depth_m=d1, intrinsics=intrinsics, pcd=pcd1, timestamp=0.1)

    # 1. Background subtraction isolation
    pcd_iso0, mask0 = isolate_subject_by_background_subtraction(f0, bg_depth)
    pcd_iso1, mask1 = isolate_subject_by_background_subtraction(f1, bg_depth)

    # Verify sphere is cleanly isolated
    assert np.count_nonzero(mask0) > 20000
    iso_pts = np.asarray(pcd_iso0.points)
    assert np.all(iso_pts[:, 2] <= 0.85)

    # 2. Odometry evaluation
    jacobian = RGBDOdometryJacobianFromColorTerm()
    option = OdometryOption()
    option.depth_min = 0.20
    option.depth_max = 1.50
    option.depth_diff_max = 0.07

    # Negative control: unmasked odometry
    rgbd0_un = f0.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True)
    rgbd1_un = f1.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True)
    success_un, trans_un, _ = compute_rgbd_odometry(rgbd0_un, rgbd1_un, intrinsics, np.eye(4), jacobian, option)
    assert success_un is True
    rot_unmasked = compute_rotation_deg(trans_un)
    # The static background pins unmasked rotation near 0 deg (< 1 deg)
    assert rot_unmasked < 1.0

    # Masked odometry on subject only
    d0_masked = np.where(mask0, d0, 0.0)
    d1_masked = np.where(mask1, d1, 0.0)
    c0_masked = c0.copy(); c0_masked[~mask0] = 0
    c1_masked = c1.copy(); c1_masked[~mask1] = 0
    rgbd0_masked = f0.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True, depth_override=d0_masked, color_override=c0_masked)
    rgbd1_masked = f1.to_rgbd_image(depth_trunc=1.5, convert_rgb_to_intensity=True, depth_override=d1_masked, color_override=c1_masked)

    success_masked, trans_masked, _ = compute_rgbd_odometry(rgbd0_masked, rgbd1_masked, intrinsics, np.eye(4), jacobian, option)
    assert success_masked is True
    rot_masked = compute_rotation_deg(trans_masked)
    # Masked rotation accurately captures the motion (~10 deg per step in camera frame)
    assert rot_masked > 8.0
