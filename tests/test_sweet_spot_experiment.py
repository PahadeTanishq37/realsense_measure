"""
tests/test_sweet_spot_experiment.py
===================================
Comprehensive unit test suite for sweet-spot trial runner, fair matched-mask
noise evaluation, safe trial finalization, aborted replay protection, and legacy compatibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.config import PipelineConfig
from realsense_measure.camera.recording import load_raw_capture
from realsense_measure.camera.rotation import rotate_intrinsics, rotate_rgbd_arrays
from realsense_measure.diagnostics.temporal_noise import (
    MatchedPopulationComparison,
    SingleStreamNoiseMetrics,
    TemporalNoiseComparison,
    compare_raw_vs_filtered_noise,
    compute_temporal_depth_noise,
)
from tools.run_sweet_spot_experiment import (
    MockRealSenseExperimentCamera,
    extract_evaluation_mask,
    run_experiment,
)


def test_temporal_std_synthetic_stable_depth():
    """Verify sample standard deviation matches known Gaussian noise on a stable surface."""
    K, H, W = 30, 20, 20
    np.random.seed(100)
    true_sigma_m = 0.0025  # 2.5 mm
    depth_base = np.full((K, H, W), 1.20, dtype=np.float32)
    noise = np.random.normal(0.0, true_sigma_m, (K, H, W)).astype(np.float32)
    depth_frames = depth_base + noise

    metrics = compute_temporal_depth_noise(
        depth_frames=depth_frames,
        depth_min_m=0.40,
        depth_max_m=3.00,
        stable_threshold=0.90,
    )

    assert metrics.num_stable_pixels == H * W
    assert metrics.stable_pixel_ratio_pct == 100.0
    assert metrics.mean_std_mm is not None
    assert abs(metrics.mean_std_mm - 2.5) < 0.20
    assert metrics.mean_detrended_std_mm is not None
    assert abs(metrics.mean_detrended_std_mm - 2.5) < 0.20


def test_invalid_depth_handling():
    """Verify that zero, negative, out-of-range, NaN, and Inf depths are excluded from calculations."""
    K, H, W = 30, 10, 10
    depth_frames = np.full((K, H, W), 1.20, dtype=np.float32)

    depth_frames[:, 0, 0] = 0.0
    depth_frames[:, 0, 1] = np.nan
    depth_frames[:, 0, 2] = np.inf
    depth_frames[:, 0, 3] = 3.50
    depth_frames[:, 0, 4] = -1.0

    metrics = compute_temporal_depth_noise(
        depth_frames=depth_frames,
        depth_min_m=0.40,
        depth_max_m=3.00,
        stable_threshold=0.90,
    )

    assert metrics.num_stable_pixels == (H * W) - 5
    assert metrics.mean_std_mm is not None
    assert np.isfinite(metrics.mean_std_mm)
    assert np.isfinite(metrics.mean_detrended_std_mm)


def test_stable_pixel_90_percent_threshold():
    """Verify the 90% stability threshold correctly classifies pixels."""
    K, H, W = 30, 2, 2
    depth_frames = np.full((K, H, W), 1.20, dtype=np.float32)

    # Pixel (0, 0): valid in 28/30 frames = 93.3% -> STABLE
    depth_frames[0:2, 0, 0] = 0.0
    # Pixel (0, 1): valid in 26/30 frames = 86.7% -> UNSTABLE (< 90%)
    depth_frames[0:4, 0, 1] = 0.0
    # Pixel (1, 0): valid in 30/30 frames = 100.0% -> STABLE
    # Pixel (1, 1): valid in 2/30 frames -> UNSTABLE (< 3 valid samples)
    depth_frames[2:, 1, 1] = 0.0

    metrics = compute_temporal_depth_noise(
        depth_frames=depth_frames,
        depth_min_m=0.40,
        depth_max_m=3.00,
        stable_threshold=0.90,
    )

    assert metrics.num_stable_pixels == 2
    assert metrics.stable_pixel_ratio_pct == 50.0


def test_linear_trend_removal():
    """Verify linear detrending removes monotonic drift while standard deviation is inflated."""
    K, H, W = 30, 5, 5
    timestamps = np.linspace(0.0, 1.0, K)

    drift = (timestamps * 0.03)[:, None, None]  # 30 mm drift
    pure_drift_depth = np.full((K, H, W), 1.20, dtype=np.float32) + drift.astype(np.float32)

    metrics = compute_temporal_depth_noise(
        depth_frames=pure_drift_depth,
        timestamps=timestamps,
        depth_min_m=0.40,
        depth_max_m=3.00,
    )

    assert metrics.mean_std_mm > 8.0
    assert metrics.mean_detrended_std_mm < 1e-3

    # Add 2.0 mm noise
    np.random.seed(42)
    noise = np.random.normal(0.0, 0.002, (K, H, W)).astype(np.float32)
    drift_plus_noise = pure_drift_depth + noise

    metrics_noisy = compute_temporal_depth_noise(
        depth_frames=drift_plus_noise,
        timestamps=timestamps,
        depth_min_m=0.40,
        depth_max_m=3.00,
    )

    assert abs(metrics_noisy.mean_detrended_std_mm - 2.0) < 0.25
    assert metrics_noisy.mean_std_mm > 8.0


def test_timestamp_normalization_unix_vs_relative():
    """Verify Unix-epoch and relative timestamps yield mathematically identical detrended residuals."""
    K, H, W = 30, 5, 5
    np.random.seed(77)
    noise = np.random.normal(0.0, 0.003, (K, H, W)).astype(np.float32)
    depth_frames = np.full((K, H, W), 1.20, dtype=np.float32) + noise

    rel_ts = np.linspace(0.0, 1.0, K)
    epoch_ts = 1793556000.0 + rel_ts

    res_rel = compute_temporal_depth_noise(depth_frames, timestamps=rel_ts)
    res_epoch = compute_temporal_depth_noise(depth_frames, timestamps=epoch_ts)

    assert res_rel.mean_detrended_std_mm is not None
    assert res_epoch.mean_detrended_std_mm is not None
    diff = abs(res_rel.mean_detrended_std_mm - res_epoch.mean_detrended_std_mm)
    assert diff < 1e-5, f"Timestamp normalization discrepancy: {diff}"


def test_raw_and_filtered_streams_with_different_stable_masks():
    """Verify correct handling when raw and filtered streams possess different stable masks."""
    K, H, W = 30, 10, 10
    raw_depth = np.full((K, H, W), 1.20, dtype=np.float32)
    filt_depth = np.full((K, H, W), 1.20, dtype=np.float32)

    raw_depth[:, :, 0:2] = 0.0
    filt_depth[:, :, 9] = 0.0

    comp = compare_raw_vs_filtered_noise(
        raw_depth_frames=raw_depth,
        filtered_depth_frames=filt_depth,
    )

    assert comp.raw.num_stable_pixels == 80
    assert comp.filtered.num_stable_pixels == 90
    assert comp.matched.matched_pixel_count == 70
    assert comp.matched.raw_stable_retention_pct == (70 / 80.0) * 100.0
    assert comp.matched.filtered_stable_retention_pct == (70 / 90.0) * 100.0


def test_matched_population_equal_comparison():
    """Verify matched-population evaluates raw and filtered jitter on identical pixels."""
    K, H, W = 30, 10, 10
    np.random.seed(42)
    raw_noise = np.random.normal(0.0, 0.004, (K, H, W)).astype(np.float32)
    filt_noise = np.random.normal(0.0, 0.002, (K, H, W)).astype(np.float32)

    raw_depth = np.full((K, H, W), 1.20, dtype=np.float32) + raw_noise
    filt_depth = np.full((K, H, W), 1.20, dtype=np.float32) + filt_noise

    raw_depth[:, 0:3, :] = 0.0
    filt_depth[:, 7:10, :] = 0.0

    comp = compare_raw_vs_filtered_noise(
        raw_depth_frames=raw_depth,
        filtered_depth_frames=filt_depth,
    )

    assert comp.matched.matched_pixel_count == 40
    assert comp.matched.raw_mean_detrended_std_mm is not None
    assert comp.matched.filtered_mean_detrended_std_mm is not None

    assert abs(comp.matched.raw_mean_detrended_std_mm - 4.0) < 0.35
    assert abs(comp.matched.filtered_mean_detrended_std_mm - 2.0) < 0.25
    assert comp.matched.denoising_reduction_pct is not None
    assert abs(comp.matched.denoising_reduction_pct - 50.0) < 8.0


def test_empty_intersection_handling():
    """Verify graceful handling when raw and filtered stable masks have zero overlap."""
    K, H, W = 30, 10, 10
    raw_depth = np.full((K, H, W), 1.20, dtype=np.float32)
    filt_depth = np.full((K, H, W), 1.20, dtype=np.float32)

    raw_depth[:, :, 0:5] = 0.0
    filt_depth[:, :, 5:10] = 0.0

    comp = compare_raw_vs_filtered_noise(
        raw_depth_frames=raw_depth,
        filtered_depth_frames=filt_depth,
    )

    assert comp.matched.matched_pixel_count == 0
    assert comp.matched.status == "empty_intersection"
    assert comp.matched.raw_stable_retention_pct == 0.0
    assert comp.matched.filtered_stable_retention_pct == 0.0
    assert comp.matched.ratio_filtered_to_raw_mean_jitter is None
    assert comp.matched.denoising_reduction_pct is None

    d = comp.to_dict()
    assert d["matched"]["status"] == "empty_intersection"
    assert d["matched"]["matched_pixel_count"] == 0
    assert d["matched"]["ratio_filtered_to_raw_mean_jitter"] is None


def test_zero_raw_jitter_denominator():
    """Verify zero raw jitter denominator does not raise ZeroDivisionError."""
    K, H, W = 30, 4, 4
    raw_depth = np.full((K, H, W), 1.20, dtype=np.float32)
    filt_depth = np.full((K, H, W), 1.20, dtype=np.float32)

    comp = compare_raw_vs_filtered_noise(
        raw_depth_frames=raw_depth,
        filtered_depth_frames=filt_depth,
    )

    assert comp.matched.status == "zero_raw_jitter"
    assert comp.matched.ratio_filtered_to_raw_mean_jitter is None
    assert comp.matched.denoising_reduction_pct is None


def test_json_serialization_without_nan_or_inf():
    """Verify JSON serialization contains strictly finite numbers and nulls."""
    metrics = SingleStreamNoiseMetrics(
        stream_name="test",
        num_frames=30,
        num_total_pixels=100,
        num_stable_pixels=0,
        stable_pixel_ratio_pct=float("nan"),
        mean_temporal_validity_pct=float("inf"),
        mean_std_mm=float("nan"),
    )
    d = metrics.to_dict()
    assert d["stable_pixel_ratio_pct"] is None
    assert d["mean_temporal_validity_pct"] is None
    assert d["mean_std_mm"] is None
    json_str = json.dumps(d)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str


def test_successful_capture_and_persistence_completed(tmp_path: Path):
    """Prompt 2G: Verify full capture and verified output persistence marks trial_status == 'completed'."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=360.0,
        subject_id="SUBJ_SAFE_COMP",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=20.0,
        ambient_lux=400.0,
        room_temp_c=22.0,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    ret = run_experiment(args)
    assert ret == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "completed"
    assert meta["failure_stage"] is None
    assert meta["failure_reason"] is None
    assert meta["capture_performance"]["requested_frames"] == 5
    assert meta["capture_performance"]["captured_frames"] == 5


def test_full_frame_capture_but_frame_write_failure(tmp_path: Path):
    """Prompt 2G: Full frame capture with frame asset write failure marks trial_status == 'aborted'."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_FAIL_FRAME",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="frames",
    )

    ret = run_experiment(args)
    assert ret == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "persistence_frames"
    assert "Frame write failure" in meta["failure_reason"]


def test_full_frame_capture_but_meta_write_failure(tmp_path: Path):
    """Prompt 2G: Full frame capture with meta.json write failure marks trial_status == 'aborted'."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_FAIL_META",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="meta",
    )

    ret = run_experiment(args)
    assert ret == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "persistence_meta"


def test_partial_capture_aborted(tmp_path: Path):
    """Prompt 2G: Incomplete frame capture marks trial_status == 'aborted'."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=10,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_PARTIAL",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=4,  # only 4 of 10 captured
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "capture"
    assert "Incomplete capture" in meta["failure_reason"]


def test_trial_meta_write_failure_reported(tmp_path: Path):
    """Prompt 2G: Write failure of trial_meta.json logs critical error and updates session CSV."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_FAIL_TMETA",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="trial_meta",
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    csv_path = session_dir / "session_summary.csv"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["trial_status"] == "aborted"
        assert rows[0]["failure_stage"] == "persistence_trial_meta"


def test_aborted_trial_rejected_by_default_replay(tmp_path: Path):
    """Prompt 2G: Replay of an aborted trial raises RuntimeError by default."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=10,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_REPLAY_REJECT",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=3,  # Aborted trial
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Replay by default must reject aborted trials
    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)

    assert "Refusing to replay aborted trial" in str(exc_info.value)
    assert "allow_aborted=True" in str(exc_info.value)


def test_completed_trial_accepted_replay(tmp_path: Path):
    """Prompt 2G: Replay of a completed trial succeeds without error."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_REPLAY_ACCEPT",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    frames = load_raw_capture(trial_dir)
    assert len(frames) == 5
    assert frames[0].color_bgr.shape == (848, 480, 3)


def test_genuine_legacy_recording_accepted(tmp_path: Path):
    """Prompt 2K: Genuine legacy recording lacking trial_meta.json and modern artifacts is accepted."""
    legacy_dir = tmp_path / "legacy_trial"
    legacy_dir.mkdir()

    color_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(legacy_dir / "frame_0000_color.png"), color_img)

    depth_arr = np.ones((480, 640), dtype=np.float32) * 1.5
    np.save(str(legacy_dir / "frame_0000_depth.npy"), depth_arr)

    meta_doc = {
        "schema_version": "1.0",
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 385.0,
            "fy": 385.0,
            "cx": 320.0,
            "cy": 240.0,
        },
        "camera_config": {
            "depth_min_m": 0.40,
            "depth_max_m": 3.00,
        },
        "frames": [
            {
                "frame_id": 0,
                "timestamp": 1000.0,
                "color_file": "frame_0000_color.png",
                "depth_file": "frame_0000_depth.npy",
            }
        ],
    }
    with open(legacy_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)

    frames = load_raw_capture(legacy_dir)
    assert len(frames) == 1
    assert frames[0].color_bgr.shape == (480, 640, 3)


def test_modern_recording_missing_marker_rejected(tmp_path: Path):
    """Prompt 2K: Modern recording with missing trial_meta.json is rejected when modern artifacts exist."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_MODERN_MISSING",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Delete trial_meta.json to simulate an interrupted/corrupt modern trial lacking marker
    trial_meta_file = trial_dir / "trial_meta.json"
    if trial_meta_file.exists():
        trial_meta_file.unlink()

    assert not trial_meta_file.exists()
    assert (trial_dir / "noise_evaluation.json").exists()

    # Replay must be rejected because modern artifacts are detected
    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)
    assert "Corrupt or unfinalized modern recording" in str(exc_info.value)
    assert "noise_evaluation.json" in str(exc_info.value)


def test_modern_recording_filtered_file_detected_without_marker(tmp_path: Path):
    """Prompt 2K: Modern recording with filtered depth or depth_filtered_file is rejected without marker."""
    modern_dir = tmp_path / "interrupted_modern_trial"
    modern_dir.mkdir()

    color_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(modern_dir / "frame_0000_color.png"), color_img)
    depth_arr = np.ones((480, 640), dtype=np.float32) * 1.5
    np.save(str(modern_dir / "frame_0000_depth.npy"), depth_arr)
    np.save(str(modern_dir / "frame_0000_depth_filtered.npy"), depth_arr)

    meta_doc = {
        "schema_version": "1.0",
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 385.0,
            "fy": 385.0,
            "cx": 320.0,
            "cy": 240.0,
        },
        "camera_config": {"depth_min_m": 0.40, "depth_max_m": 3.00},
        "frames": [
            {
                "frame_id": 0,
                "timestamp": 1000.0,
                "color_file": "frame_0000_color.png",
                "depth_file": "frame_0000_depth.npy",
                "depth_filtered_file": "frame_0000_depth_filtered.npy",
            }
        ],
    }
    with open(modern_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)

    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(modern_dir)
    assert "Corrupt or unfinalized modern recording" in str(exc_info.value)
    assert "filtered depth file(s)" in str(exc_info.value)


def test_aborted_trial_debug_opt_in_replay(tmp_path: Path):
    """Prompt 2G: Explicit allow_aborted=True opt-in permits replaying aborted captures."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=10,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_DEBUG_OPTIN",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=4,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Passing allow_aborted=True allows loading the 4 frames for debugging
    frames = load_raw_capture(trial_dir, allow_aborted=True)
    assert len(frames) == 4


def test_unrecognized_trial_status_rejected_replay(tmp_path: Path):
    """Prompt 2G: Unrecognized or missing trial_status in trial_meta.json is rejected."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_UNRECOGNIZED",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Tamper with trial_meta.json to inject invalid status
    trial_meta_file = trial_dir / "trial_meta.json"
    with open(trial_meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["trial_status"] = "in_progress_unknown"
    with open(trial_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)

    assert "Unrecognized or missing trial_status" in str(exc_info.value)


def test_full_frame_capture_but_noise_write_failure(tmp_path: Path):
    """Prompt 2I: Full frame capture with noise_evaluation.json write failure marks trial_status == 'aborted'."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_FAIL_NOISE",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="noise",
    )

    ret = run_experiment(args)
    assert ret == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "persistence_noise"
    assert "noise_evaluation.json write failure" in meta["failure_reason"]

    # Replay by default must reject this aborted trial
    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)
    assert "Refusing to replay aborted trial" in str(exc_info.value)


def test_session_csv_write_failure_surfaced(tmp_path: Path):
    """Prompt 2I: Session summary CSV write failure surfaces non-zero return code (1)."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_FAIL_CSV",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="csv",
    )

    ret = run_experiment(args)
    assert ret == 1, "Expected run_experiment to return 1 when session CSV persistence fails"


def test_mid_capture_device_disconnect_aborted(tmp_path: Path):
    """Prompt 2I: Device disconnection mid-capture aborts trial and preserves partial frames."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=10,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_DISCONNECT",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=True,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "capture"
    assert "DeviceDisconnectedError" in meta["failure_reason"]
    assert meta["capture_performance"]["captured_frames"] == 2

    # Default replay rejects
    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)
    assert "Refusing to replay aborted trial" in str(exc_info.value)

    # Opt-in debug replay succeeds and loads partial frames
    frames = load_raw_capture(trial_dir, allow_aborted=True)
    assert len(frames) == 2


def test_initial_marker_protects_against_early_crash(tmp_path: Path, monkeypatch):
    """Prompt 2I: Immediate initial marker ensures directory is rejected if killed mid-capture."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_EARLY_CRASH",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    # Simulate hard process abort/exception right after directory creation and initial marker
    orig_capture = MockRealSenseExperimentCamera.capture_burst_pair
    def crash_capture(self, num_frames):
        raise SystemExit("Simulated unexpected process termination")
    monkeypatch.setattr(MockRealSenseExperimentCamera, "capture_burst_pair", crash_capture)

    with pytest.raises(SystemExit):
        run_experiment(args)

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Initial marker must exist on disk
    marker_path = trial_dir / "trial_meta.json"
    assert marker_path.exists()
    with open(marker_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "in_progress"

    # Replay by default must reject this crashed directory
    with pytest.raises(RuntimeError) as exc_info:
        load_raw_capture(trial_dir)
    assert "Refusing to replay aborted trial" in str(exc_info.value)


def test_corrupt_asset_fails_verification(tmp_path: Path, monkeypatch):
    """Prompt 2I: Empty or corrupt frame asset prevents trial from being marked completed."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=3,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_CORRUPT_ASSET",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    # Monkeypatch cv2.imwrite to write a 0-byte file for frame 0
    orig_imwrite = cv2.imwrite
    def bad_imwrite(path, img):
        Path(path).touch()  # Creates a 0-byte file
        return True
    monkeypatch.setattr(cv2, "imwrite", bad_imwrite)

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "verification"
    assert "Post-persistence verification failed" in meta["failure_reason"]


def test_body_capture_report_rejects_aborted_trial(tmp_path: Path):
    """Prompt 2I: Auxiliary body_capture_report tool rejects aborted trials by default."""
    from tools.body_capture_report import generate_body_report

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_REPORT_REJECT",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=2,  # Aborted trial
        mock_disconnect=False,
        mock_write_failure=None,
    )

    run_experiment(args)
    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # Must reject by default
    with pytest.raises(RuntimeError) as exc_info:
        generate_body_report(trial_dir)
    assert "Refusing to replay aborted trial" in str(exc_info.value)

    # Allow aborted opt-in succeeds
    res = generate_body_report(trial_dir, allow_aborted=True)
    assert len(res) == 2


def test_initial_marker_failure_aborts_immediately_without_capture(tmp_path: Path, monkeypatch):
    """Prompt 2K: Marker write failure immediately aborts trial before camera capture."""
    import tools.run_sweet_spot_experiment as runner

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_MARKER_FAIL",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure="initial_marker",
    )

    capture_called = []
    orig_mock_capture = runner.MockRealSenseExperimentCamera.capture_burst_pair

    def mock_capture_spy(*spy_args, **spy_kwargs):
        capture_called.append(True)
        return orig_mock_capture(*spy_args, **spy_kwargs)

    monkeypatch.setattr(runner.MockRealSenseExperimentCamera, "capture_burst_pair", mock_capture_spy)

    exit_code = run_experiment(args)
    assert exit_code == 1

    # Ensure capture was never called
    assert len(capture_called) == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    # No frames should have been persisted
    frame_files = list(trial_dir.glob("frame_*"))
    assert len(frame_files) == 0

    # No trial_meta.json should exist since the initial write failed
    assert not (trial_dir / "trial_meta.json").exists()

    # CSV summary must record failure_stage as persistence_initial_marker
    csv_file = session_dir / "session_summary.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_content = f.read()
    assert "persistence_initial_marker" in csv_content


def test_atomic_final_metadata_replacement_success(tmp_path: Path):
    """Prompt 2K: Successful trial finalization atomically replaces trial_meta.json and leaves no temp file."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_ATOMIC_OK",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    assert exit_code == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    meta_file = trial_dir / "trial_meta.json"
    temp_file = trial_dir / "trial_meta.json.tmp"

    assert meta_file.exists()
    assert not temp_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["trial_status"] == "completed"


def test_atomic_final_metadata_replacement_failure_preserves_initial_marker(tmp_path: Path, monkeypatch):
    """Prompt 2K: Failure during atomic replace cleans up temp file and preserves initial aborted marker on disk."""
    import os

    orig_replace = os.replace

    def mock_replace(src, dst):
        if "trial_meta.json.tmp" in str(src):
            raise OSError("Mock disk full during atomic replace")
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_ATOMIC_FAIL",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    assert exit_code == 1

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]

    meta_file = trial_dir / "trial_meta.json"
    temp_file = trial_dir / "trial_meta.json.tmp"

    # Temp file must be cleaned up
    assert not temp_file.exists()

    # The initial aborted marker must be intact on disk!
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "in_progress"


def test_depth_integrity_all_zero_rejected(tmp_path: Path, monkeypatch):
    """Prompt 2K: All-zero depth frames fail Stage 3 numerical integrity verification."""
    import tools.run_sweet_spot_experiment as runner

    orig_mock = runner.MockRealSenseExperimentCamera.capture_burst_pair

    def mock_all_zero(self, num_frames):
        colors, raw_depths, filt_depths, timestamps = orig_mock(self, num_frames)
        for i in range(len(raw_depths)):
            raw_depths[i] = np.zeros_like(raw_depths[i])
            filt_depths[i] = np.zeros_like(filt_depths[i])
        return colors, raw_depths, filt_depths, timestamps

    monkeypatch.setattr(runner.MockRealSenseExperimentCamera, "capture_burst_pair", mock_all_zero)

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_ZERO_DEPTH",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    # Verification failure sets trial to aborted

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]
    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "verification"
    assert "no valid depth samples" in meta["failure_reason"] or "below minimum threshold" in meta["failure_reason"]


def test_depth_integrity_all_nan_rejected(tmp_path: Path, monkeypatch):
    """Prompt 2K: All-NaN depth frames fail Stage 3 numerical integrity verification."""
    import tools.run_sweet_spot_experiment as runner

    orig_mock = runner.MockRealSenseExperimentCamera.capture_burst_pair

    def mock_all_nan(self, num_frames):
        colors, raw_depths, filt_depths, timestamps = orig_mock(self, num_frames)
        for i in range(len(raw_depths)):
            raw_depths[i] = np.full_like(raw_depths[i], np.nan)
            filt_depths[i] = np.full_like(filt_depths[i], np.nan)
        return colors, raw_depths, filt_depths, timestamps

    monkeypatch.setattr(runner.MockRealSenseExperimentCamera, "capture_burst_pair", mock_all_nan)

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_NAN_DEPTH",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    # Verification failure sets trial to aborted

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]
    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "aborted"
    assert meta["failure_stage"] == "verification"
    assert "no valid depth samples" in meta["failure_reason"] or "below minimum threshold" in meta["failure_reason"]


def test_depth_integrity_mixed_valid_and_nan_accepted(tmp_path: Path, monkeypatch):
    """Prompt 2K: Depth frames with legitimate NaNs (e.g. 50% background) pass verification when valid fraction >= 2%."""
    import tools.run_sweet_spot_experiment as runner

    orig_mock = runner.MockRealSenseExperimentCamera.capture_burst_pair

    def mock_half_nan(self, num_frames):
        colors, raw_depths, filt_depths, timestamps = orig_mock(self, num_frames)
        for i in range(len(raw_depths)):
            # Set left half to NaN
            raw_depths[i][:, : raw_depths[i].shape[1] // 2] = np.nan
            filt_depths[i][:, : filt_depths[i].shape[1] // 2] = np.nan
        return colors, raw_depths, filt_depths, timestamps

    monkeypatch.setattr(runner.MockRealSenseExperimentCamera, "capture_burst_pair", mock_half_nan)

    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_HALF_NAN",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    assert exit_code == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]
    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "completed"
    depth_int = meta.get("depth_numerical_integrity", {})
    assert depth_int["verified"] is True
    # Fraction of subject depth should be >= 2% threshold (approx 17% in half-masked mock)
    assert depth_int["raw_valid_fraction"] >= 0.02
    assert 0.10 <= depth_int["raw_valid_fraction"] <= 0.25
    assert 0.10 <= depth_int["filtered_valid_fraction"] <= 0.25


def test_depth_integrity_metrics_recorded_in_metadata(tmp_path: Path):
    """Prompt 2K: Standard mock capture passes verification and records depth integrity metrics in trial_meta.json."""
    args = argparse.Namespace(
        distance=1.20,
        elevation=0.0,
        azimuth=0.0,
        pose="a_pose",
        orientation="portrait_cw",
        burst_frames=5,
        repetitions=1,
        emitter_power=None,
        subject_id="SUBJ_DEPTH_METRICS",
        output_dir=str(tmp_path),
        static_reference=False,
        warmup_min=None,
        ambient_lux=None,
        room_temp_c=None,
        skip_prompt=True,
        mock_capture=True,
        mock_incomplete_frames=None,
        mock_disconnect=False,
        mock_write_failure=None,
    )

    exit_code = run_experiment(args)
    assert exit_code == 0

    session_dir = list(tmp_path.glob("SESSION_*"))[0]
    trial_dir = list(session_dir.glob("trial_01*"))[0]
    with open(trial_dir / "trial_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["trial_status"] == "completed"
    depth_int = meta.get("depth_numerical_integrity")
    assert depth_int is not None
    assert depth_int["verified"] is True
    assert depth_int["raw_valid_fraction"] > 0.02
    assert depth_int["filtered_valid_fraction"] > 0.02
    assert depth_int["raw_mean_depth_m"] > 0.0
    assert depth_int["filtered_mean_depth_m"] > 0.0
