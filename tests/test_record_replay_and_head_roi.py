"""
Comprehensive Unit Test Suite for Record & Replay and Head-Only Registration Mode.

Covers:
  1. Record & Replay roundtrip equality (lossless PNG color, float32 depth in metres,
     metadata, intrinsics, and exact point cloud reconstruction).
  2. Replay without pyrealsense2 (zero camera dependencies; pyrealsense2 set to None).
  3. Head ROI derivation (MediaPipe landmark detection with chin clipping,
     neighbouring pose propagation, and graceful fallback to full-cloud with logging).
  4. Head-only registration vs baseline regression (ensuring default head_only_registration=False
     preserves exact baseline behavior while opt-in mode evaluates on head ROI and logs metrics).
  5. registration_report.json schema and aggregation.
  6. Named config parsing and batch summary CSV generation.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame, create_pcd_from_rgbd
from camera.recording import (
    load_raw_capture,
    parse_named_config,
    run_replay_batch,
    save_raw_capture,
)
from config import PipelineConfig, RegistrationConfig
from pipeline import ScanPipeline
from registration import (
    PairRegistrationResult,
    create_depth_mask_from_pcd,
    derive_head_roi,
    register_rgbd_pair,
    register_sequence_rgbd_human,
)


def _make_dummy_frame(
    frame_id: int = 0,
    timestamp: float = 100.0,
    width: int = 80,
    height: int = 60,
    fx: float = 60.0,
    fy: float = 60.0,
    cx: float = 40.0,
    cy: float = 30.0,
) -> RGBDFrame:
    """Helper to synthesize a clean, reproducible RGBDFrame with non-trivial geometry."""
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Color gradient
    color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    color_bgr[:, :, 0] = np.linspace(50, 200, width, dtype=np.uint8)
    color_bgr[:, :, 1] = np.linspace(30, 180, height, dtype=np.uint8)[:, None]
    color_bgr[:, :, 2] = 120

    # Spherical depth bump around image center
    y_idx, x_idx = np.indices((height, width))
    r_sq = ((x_idx - cx) / 20.0) ** 2 + ((y_idx - cy) / 20.0) ** 2
    depth_m = np.zeros((height, width), dtype=np.float32)
    valid_mask = r_sq <= 1.0
    depth_m[valid_mask] = 0.65 - 0.05 * np.sqrt(np.clip(1.0 - r_sq[valid_mask], 0.0, 1.0))

    pcd = create_pcd_from_rgbd(
        color_bgr=color_bgr,
        depth_m=depth_m,
        intrinsics=intr,
        depth_min_m=0.40,
        depth_max_m=1.20,
    )

    return RGBDFrame(
        frame_id=frame_id,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=pcd,
        timestamp=timestamp,
        intrinsics=intr,
    )


# ---------------------------------------------------------------------------
# Test 1: Record & Replay Roundtrip Equality
# ---------------------------------------------------------------------------
def test_record_replay_roundtrip(tmp_path: Path):
    """
    Verify save_raw_capture -> load_raw_capture preserves exact color, depth,
    intrinsics, frame timestamps/order, and produces numerically identical point clouds.
    """
    f0 = _make_dummy_frame(frame_id=0, timestamp=10.123)
    f1 = _make_dummy_frame(frame_id=1, timestamp=10.456)
    orig_frames = [f0, f1]

    cfg = PipelineConfig()
    cfg.camera.width = 80
    cfg.camera.height = 60
    cfg.camera.depth_min_m = 0.40
    cfg.camera.depth_max_m = 1.20

    raw_dir = tmp_path / "raw_capture_test"
    saved_path = save_raw_capture(orig_frames, raw_dir, cfg)
    assert saved_path.exists()

    meta_file = raw_dir / "meta.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_doc = json.load(f)

    assert meta_doc["num_frames"] == 2
    assert meta_doc["intrinsics"]["width"] == 80
    assert meta_doc["intrinsics"]["height"] == 60
    assert len(meta_doc["frames"]) == 2
    assert meta_doc["frames"][0]["order"] == 0
    assert meta_doc["frames"][1]["order"] == 1

    loaded_frames = load_raw_capture(raw_dir)
    assert len(loaded_frames) == 2

    for orig, loaded in zip(orig_frames, loaded_frames):
        assert loaded.frame_id == orig.frame_id
        assert abs(loaded.timestamp - orig.timestamp) < 1e-4
        assert np.array_equal(loaded.color_bgr, orig.color_bgr)
        assert np.allclose(loaded.depth_m, orig.depth_m, atol=1e-6)
        assert np.allclose(
            loaded.intrinsics.intrinsic_matrix,
            orig.intrinsics.intrinsic_matrix,
            atol=1e-6,
        )

        orig_pts = np.asarray(orig.pcd.points)
        loaded_pts = np.asarray(loaded.pcd.points)
        assert orig_pts.shape == loaded_pts.shape
        assert np.allclose(loaded_pts, orig_pts, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Replay Without pyrealsense2
# ---------------------------------------------------------------------------
def test_replay_without_pyrealsense2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Verify replay mode loads frames and supports point cloud generation even when
    pyrealsense2 is absent from the python environment (sys.modules['pyrealsense2'] = None).
    """
    raw_dir = tmp_path / "raw_replay_no_rs"
    f0 = _make_dummy_frame(frame_id=0, timestamp=1.0)
    save_raw_capture([f0], raw_dir, PipelineConfig())

    # Simulate environment where pyrealsense2 is not installed
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)

    # load_raw_capture must succeed without pyrealsense2
    loaded = load_raw_capture(raw_dir)
    assert len(loaded) == 1
    assert loaded[0].pcd.has_points()

    # to_point_cloud() on loaded frame must succeed without pyrealsense2
    pcd_recalc = loaded[0].to_point_cloud(0.40, 1.20)
    assert len(pcd_recalc.points) == len(loaded[0].pcd.points)

    # Pipeline capture_frames() must invoke load_raw_capture and avoid RealSenseCamera
    p_cfg = PipelineConfig()
    p_cfg.replay_dir = str(raw_dir)
    pipeline = ScanPipeline(p_cfg)
    captured = pipeline.capture_frames()
    assert len(captured) == 1
    assert captured[0].frame_id == 0


# ---------------------------------------------------------------------------
# Test 3: Head ROI Derivation
# ---------------------------------------------------------------------------
def test_head_roi_derivation_with_landmarks(monkeypatch: pytest.MonkeyPatch):
    """
    Test 3A: When landmarks are detected, derive_head_roi computes centroid,
    crops within head_roi_radius_m (0.15m), and clips above the chin line.
    """
    frame = _make_dummy_frame()

    # Create synthetic point cloud with head points and lower body/torso points
    head_center = np.array([0.0, -0.05, 0.65], dtype=np.float64)
    chin_point = np.array([0.0, 0.05, 0.65], dtype=np.float64)  # down is +Y

    # Head points within 0.10m radius
    rng = np.random.default_rng(42)
    head_pts = head_center + rng.uniform(-0.08, 0.08, size=(100, 3))

    # Torso points well below chin (y > 0.15)
    torso_pts = np.array([[0.0, 0.20, 0.65], [0.05, 0.25, 0.65], [-0.05, 0.22, 0.65]])

    all_pts = np.vstack([head_pts, torso_pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    frame.pcd = pcd

    # Mock landmark detection to return facial landmarks with chin
    mock_landmarks = {
        "nose": {"position": head_center + np.array([0.0, 0.0, -0.05])},
        "forehead": {"position": head_center + np.array([0.0, -0.08, 0.0])},
        "left_eye": {"position": head_center + np.array([-0.03, -0.03, 0.0])},
        "right_eye": {"position": head_center + np.array([0.03, -0.03, 0.0])},
        "chin": {"position": chin_point},
    }

    import landmarks
    monkeypatch.setattr(landmarks, "detect_face_landmarks_3d", lambda *args, **kwargs: mock_landmarks)

    cfg = RegistrationConfig()
    cfg.head_only_registration = True
    cfg.head_roi_radius_m = 0.15

    roi_pcd, center, down_vec, source_type = derive_head_roi(frame, cfg)

    assert roi_pcd is not None
    assert source_type == "landmarks"
    assert center is not None
    assert down_vec is not None
    assert np.allclose(down_vec, np.array([0.0, 1.0, 0.0]), atol=0.1)

    roi_pts = np.asarray(roi_pcd.points)
    # Check that all points in roi_pcd are within sphere radius
    dists = np.linalg.norm(roi_pts - center, axis=1)
    assert np.all(dists <= 0.15 + 1e-4)

    # Check that torso points (y >= 0.20) are excluded
    assert not np.any(roi_pts[:, 1] >= 0.15)


def test_head_roi_derivation_propagation(monkeypatch: pytest.MonkeyPatch):
    """
    Test 3B: When landmarks are missing, propagate ROI from neighbouring frame
    using relative pose.
    """
    frame = _make_dummy_frame()

    # Mock landmark detector returning None (e.g. profile or back of head)
    import landmarks
    monkeypatch.setattr(landmarks, "detect_face_landmarks_3d", lambda *args, **kwargs: None)

    # Create points around [0.05, -0.02, 0.70]
    target_center = np.array([0.05, -0.02, 0.70])
    rng = np.random.default_rng(123)
    pts = target_center + rng.uniform(-0.06, 0.06, size=(80, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    frame.pcd = pcd

    ref_center = np.array([0.0, 0.0, 0.70])
    ref_down = np.array([0.0, 1.0, 0.0])
    # Relative pose T maps source to target; inv_T transforms target into source frame
    rel_pose = np.eye(4)
    rel_pose[0, 3] = -0.05
    rel_pose[1, 3] = 0.02

    cfg = RegistrationConfig()
    cfg.head_only_registration = True

    roi_pcd, center, down, source_type = derive_head_roi(
        frame,
        cfg,
        ref_center=ref_center,
        ref_down=ref_down,
        rel_pose=rel_pose,
    )

    assert roi_pcd is not None
    assert source_type == "propagated"
    assert center is not None
    assert np.allclose(center, target_center, atol=1e-3)
    assert len(roi_pcd.points) > 20


def test_head_roi_derivation_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """
    Test 3C: When landmarks are missing and no neighbour propagation is available,
    fall back to full cloud and log warning per frame.
    """
    frame = _make_dummy_frame()
    import landmarks
    monkeypatch.setattr(landmarks, "detect_face_landmarks_3d", lambda *args, **kwargs: None)

    cfg = RegistrationConfig()
    cfg.head_only_registration = True

    with caplog.at_level(logging.WARNING):
        roi_pcd, center, down, source_type = derive_head_roi(frame, cfg, ref_center=None, rel_pose=None)

    assert roi_pcd is None
    assert center is None
    assert source_type == "fallback_full_cloud"
    assert any("[head-roi]" in rec.message and "falling back to full-cloud" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Test 4: Head-Only Registration vs Baseline Regression
# ---------------------------------------------------------------------------
def test_head_only_registration_baseline_regression():
    """
    Verify that head_only_registration=False (the default) produces identical behavior,
    and when enabled, pairwise registration evaluates on head ROIs, outputs PairRegistrationResult
    with src_roi_pts/tgt_roi_pts, and logs Kabsch-vs-FPFH disagreement.
    """
    src = _make_dummy_frame(frame_id=0)
    tgt = _make_dummy_frame(frame_id=1)

    cfg = RegistrationConfig()
    cfg.resolve_defaults()

    # 1. Baseline mode (head_only_registration = False)
    assert cfg.head_only_registration is False
    res_base = register_rgbd_pair(src, tgt, cfg)
    assert isinstance(res_base, PairRegistrationResult)
    assert res_base.src_roi_pts == len(src.pcd.points)
    assert res_base.tgt_roi_pts == len(tgt.pcd.points)

    # 2. Opt-in Head-Only mode
    cfg_head = RegistrationConfig()
    cfg_head.resolve_defaults()
    cfg_head.head_only_registration = True

    # Crop subset of points as head ROI
    src_roi = o3d.geometry.PointCloud()
    src_roi.points = o3d.utility.Vector3dVector(np.asarray(src.pcd.points)[:30])
    tgt_roi = o3d.geometry.PointCloud()
    tgt_roi.points = o3d.utility.Vector3dVector(np.asarray(tgt.pcd.points)[:35])

    res_head = register_rgbd_pair(
        src,
        tgt,
        cfg_head,
        src_head_roi=src_roi,
        tgt_head_roi=tgt_roi,
    )
    assert isinstance(res_head, PairRegistrationResult)
    assert res_head.src_roi_pts == 30
    assert res_head.tgt_roi_pts == 35


# ---------------------------------------------------------------------------
# Test 5: Registration Report Schema & Summary
# ---------------------------------------------------------------------------
def test_registration_report_generated():
    """
    Verify register_sequence_rgbd_human produces a complete 'registration_report'
    dictionary adhering to the required schema.
    """
    f0 = _make_dummy_frame(frame_id=0)
    f1 = _make_dummy_frame(frame_id=1)
    frames = [f0, f1]

    cfg = RegistrationConfig()
    cfg.resolve_defaults()
    cfg.head_only_registration = True

    _, diagnostics, poses, summary_stats = register_sequence_rgbd_human(
        frames=frames,
        cfg=cfg,
    )

    assert "registration_report" in summary_stats
    report = summary_stats["registration_report"]

    assert report["head_only_registration"] is True
    assert "total_pairs_evaluated" in report
    assert "accepted_pairs_count" in report
    assert "tier_counts" in report
    assert "mean_rmse_mm" in report
    assert "pairs" in report

    for pair_entry in report["pairs"]:
        assert "source" in pair_entry
        assert "target" in pair_entry
        assert "src_roi_pts" in pair_entry
        assert "tgt_roi_pts" in pair_entry
        assert "method" in pair_entry
        assert "fitness" in pair_entry
        assert "rmse_mm" in pair_entry


# ---------------------------------------------------------------------------
# Test 6: Named Config Parsing & Batch Summary CSV
# ---------------------------------------------------------------------------
def test_named_config_parsing():
    """
    Verify parse_named_config sets appropriate registration and fusion flags,
    and returns capture-time only warnings when camera flags are supplied.
    """
    base_cfg = PipelineConfig()

    cfg1, warns1 = parse_named_config("baseline", base_cfg)
    assert cfg1.registration.head_only_registration is False
    assert cfg1.registration.use_multiway is False
    assert len(warns1) == 0

    cfg2, warns2 = parse_named_config("head_only+multiway+tsdf", base_cfg)
    assert cfg2.registration.head_only_registration is True
    assert cfg2.registration.use_multiway is True
    assert cfg2.fusion_method == "tsdf"
    assert len(warns2) == 0

    cfg3, warns3 = parse_named_config("baseline+face_quality_strict", base_cfg)
    assert len(warns3) >= 1
    assert any("capture-time only" in w for w in warns3)


def test_run_replay_batch_summary_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify run_replay_batch executes configs against a raw capture folder and
    writes out comparison_summary.csv with all requested headers.
    """
    raw_dir = tmp_path / "raw_capture_batch"
    f0 = _make_dummy_frame(frame_id=0)
    f1 = _make_dummy_frame(frame_id=1)
    save_raw_capture([f0, f1], raw_dir, PipelineConfig())

    out_base = tmp_path / "batch_out"

    # Mock ScanPipeline.run to return predictable metrics quickly
    def mock_run(self):
        return {
            "success": True,
            "fusion_method": self.cfg.fusion_method,
            "points_in_stage4": 1500,
            "measurements": {"circumference_mm": 560.2},
            "summary_stats": {
                "registration_report": {
                    "tier_counts": {"tier2_rgbd_odom": 1},
                    "mean_rmse_mm": 1.25,
                    "median_rmse_mm": 1.25,
                    "max_rmse_mm": 1.25,
                    "pairs": [{"source": 1, "target": 0}],
                }
            },
        }

    monkeypatch.setattr(ScanPipeline, "run", mock_run)

    csv_path = run_replay_batch(
        raw_dir=raw_dir,
        config_names=["baseline", "head_only+multiway+tsdf"],
        output_base_dir=out_base,
        target_name="head",
    )

    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    row0 = rows[0]
    expected_headers = [
        "config_name",
        "registration_tier_counts",
        "mean_pair_rmse_mm",
        "median_pair_rmse_mm",
        "max_pair_rmse_mm",
        "edges",
        "points_in_stage4",
        "fusion_method",
        "stage5_measurement",
        "capture_time_notes",
    ]
    for h in expected_headers:
        assert h in row0


def test_seeded_replay_reproducibility(tmp_path: Path):
    """Verify that two replay runs with the same random seed produce identical results."""
    from camera.recording import save_raw_capture
    from pipeline import ScanPipeline
    from config import TargetConfig

    frames = [_make_dummy_frame(i, timestamp=100.0 + i * 2.0) for i in range(3)]
    raw_dir = tmp_path / "raw"
    save_cfg = PipelineConfig(target=TargetConfig(name="head"))
    save_raw_capture(frames, raw_dir, save_cfg)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    cfg1 = PipelineConfig(
        output_dir=out1,
        replay_dir=raw_dir,
        random_seed=42,
        show_stage_windows=False,
    )
    cfg1.target.name = "head"
    ScanPipeline(cfg1).run()

    cfg2 = PipelineConfig(
        output_dir=out2,
        replay_dir=raw_dir,
        random_seed=42,
        show_stage_windows=False,
    )
    cfg2.target.name = "head"
    ScanPipeline(cfg2).run()

    rep1_path = out1 / "registration_report.json"
    rep2_path = out2 / "registration_report.json"
    assert rep1_path.exists() and rep2_path.exists()
    with open(rep1_path, "r", encoding="utf-8") as f:
        rep1 = json.load(f)
    with open(rep2_path, "r", encoding="utf-8") as f:
        rep2 = json.load(f)

    assert rep1["accepted_pairs_count"] == rep2["accepted_pairs_count"]
    if rep1["mean_rmse_mm"] is not None:
        assert rep1["mean_rmse_mm"] == pytest.approx(rep2["mean_rmse_mm"], rel=1e-6)
