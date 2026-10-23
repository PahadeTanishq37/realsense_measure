"""
Unit and integration tests for Face Quality Mode, new CLI flags, and Stage 4 Fusion Gate Diagnostics.

Covers:
  1. Face-quality mode config resolution (True vs False, target head/body vs box).
  2. RealSenseCamera spatial filter magnitude and stream fallback logic under face-quality mode.
  3. CLI flags (--face-quality, --multiway, --loop-closure, --tsdf) and defaults.
  4. Stage 4 Gate Report generation (JSON schema, threshold comparisons, per-frame records, one-line summary).
  5. Multiway and loop-closure execution in head mode with synthetic frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.realsense_capture import RealSenseCamera, StreamNegotiationError
from config import CameraConfig, PipelineConfig, RegistrationConfig, TargetConfig
from main import _build_parser
from pipeline import ScanPipeline


# ===========================================================================
# 1. Config Resolution Tests
# ===========================================================================

class TestFaceQualityConfigResolution:
    """Test resolution of CameraConfig settings under human_face_quality_mode."""

    def test_default_camera_config_is_byte_for_byte_identical(self):
        """When human_face_quality_mode is False, all camera defaults remain unchanged."""
        cam = CameraConfig()
        assert cam.human_face_quality_mode is False
        assert cam.use_decimation is True
        assert cam.use_hole_filling is True
        assert cam.width == 848
        assert cam.height == 480

        # Resolving for head or box while False leaves all values intact
        cam.resolve_defaults("head")
        assert cam.use_decimation is True
        assert cam.use_hole_filling is True
        assert cam.width == 848
        assert cam.height == 480

    def test_face_quality_enabled_with_box_target_does_not_override(self):
        """When human_face_quality_mode is True but target is box, resolution does not switch to 1280x720."""
        cam = CameraConfig(human_face_quality_mode=True)
        cam.resolve_defaults("box")
        assert cam.use_decimation is True
        assert cam.use_hole_filling is True
        assert cam.width == 848
        assert cam.height == 480

    def test_face_quality_enabled_with_head_target_resolves_hd_settings(self):
        """When human_face_quality_mode is True and target is head/body, resolution switches to 1280x720 and filters turn off."""
        cam_head = CameraConfig(human_face_quality_mode=True)
        cam_head.resolve_defaults("head")
        assert cam_head.use_decimation is False
        assert cam_head.use_hole_filling is False
        assert cam_head.width == 1280
        assert cam_head.height == 720

        cam_body = CameraConfig(human_face_quality_mode=True)
        cam_body.resolve_defaults("body")
        assert cam_body.use_decimation is False
        assert cam_body.use_hole_filling is False
        assert cam_body.width == 1280
        assert cam_body.height == 720

    def test_pipeline_config_post_init_resolves_camera_defaults(self):
        """PipelineConfig.__post_init__ automatically resolves camera settings based on target."""
        pipe_head = PipelineConfig(
            target=TargetConfig(name="head"),
            camera=CameraConfig(human_face_quality_mode=True),
        )
        assert pipe_head.camera.width == 1280
        assert pipe_head.camera.height == 720
        assert pipe_head.camera.use_decimation is False
        assert pipe_head.camera.use_hole_filling is False

        pipe_box = PipelineConfig(
            target=TargetConfig(name="box"),
            camera=CameraConfig(human_face_quality_mode=True),
        )
        assert pipe_box.camera.width == 848
        assert pipe_box.camera.height == 480
        assert pipe_box.camera.use_decimation is True
        assert pipe_box.camera.use_hole_filling is True


# ===========================================================================
# 2. Camera Capture Settings & Fallback Tests
# ===========================================================================

class TestRealSenseCaptureFaceQuality:
    """Test RealSenseCamera filter configuration and fallback when high resolution is unsupported."""

    def test_spatial_filter_magnitude_adjusted_in_face_quality_mode(self):
        """Spatial filter magnitude is lowered to 1.0 when human_face_quality_mode is True."""
        mock_rs = MagicMock()
        with patch("camera.realsense_capture.rs", mock_rs):
            cfg_default = CameraConfig(human_face_quality_mode=False)
            RealSenseCamera(cfg_default)
            # Find spatial filter call
            spatial_calls = mock_rs.spatial_filter.return_value.set_option.call_args_list
            assert any(call[0][1] == 2.0 for call in spatial_calls)

            mock_rs.reset_mock()
            cfg_quality = CameraConfig(human_face_quality_mode=True)
            RealSenseCamera(cfg_quality)
            quality_calls = mock_rs.spatial_filter.return_value.set_option.call_args_list
            assert any(call[0][1] == 1.0 for call in quality_calls)

    def test_stream_fallback_to_848x480_on_unsupported_1280x720(self, caplog):
        """If 1280x720 is unsupported, camera logs a warning and falls back to 848x480."""
        mock_rs = MagicMock()
        mock_rs.option.visual_preset = "visual_preset"
        mock_rs.option.emitter_enabled = "emitter_enabled"
        mock_rs.rs400_visual_preset.high_accuracy = 3

        mock_profile = MagicMock()
        depth_sensor = MagicMock()
        depth_sensor.get_depth_scale.return_value = 0.001
        depth_sensor.supports.return_value = True
        mock_profile.get_device.return_value.first_depth_sensor.return_value = depth_sensor

        stream_prof = MagicMock()
        intr_mock = MagicMock(width=848, height=480, fx=420.0, fy=420.0, ppx=424.0, ppy=240.0)
        stream_prof.get_intrinsics.return_value = intr_mock
        mock_profile.get_stream.return_value.as_video_stream_profile.return_value = stream_prof

        # First call fails (1280x720), second call succeeds (848x480)
        start_mock = MagicMock(side_effect=[RuntimeError("Unsupported 1280x720 format"), mock_profile])

        with patch("camera.realsense_capture.rs", mock_rs):
            cfg = CameraConfig(width=1280, height=720, human_face_quality_mode=True)
            cam = RealSenseCamera(cfg)
            cam.pipeline.start = start_mock

            cam.start()

            assert cam.cfg.width == 848
            assert cam.cfg.height == 480
            assert "Falling back to 848x480" in caplog.text
            # High Accuracy preset & emitter called
            assert depth_sensor.set_option.call_count >= 2


# ===========================================================================
# 3. CLI Flags Tests
# ===========================================================================

class TestCLIFlags:
    """Test CLI argument parsing for --face-quality, --multiway, --loop-closure, and --tsdf."""

    def test_default_cli_flags(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.face_quality is False
        assert args.multiway is False
        assert args.loop_closure is False
        assert args.tsdf is False
        assert args.fusion == "pointcloud"

    def test_cli_flags_toggle_properly(self):
        parser = _build_parser()
        args = parser.parse_args(["--face-quality", "--multiway", "--loop-closure", "--tsdf"])
        assert args.face_quality is True
        assert args.multiway is True
        assert args.loop_closure is True
        assert args.tsdf is True


# ===========================================================================
# 4. Fusion-Gate Diagnostics Report Tests
# ===========================================================================

class TestFusionGateReport:
    """Test Stage 4 Fusion-Gate Report generation and one-line summary."""

    def test_gate_report_generation(self, tmp_path, capsys):
        """Construct synthetic aligned frames and diagnostics to test all gate conditions."""
        out_dir = tmp_path / "scan_gate_test"
        scan_folder = tmp_path / "stage_wise_output" / "scan_test_99"
        out_dir.mkdir(parents=True)
        scan_folder.mkdir(parents=True)

        cfg = PipelineConfig(
            output_dir=out_dir,
            target=TargetConfig(name="head"),
            show_stage_windows=False,
            save_intermediate=False,
        )
        pipeline = ScanPipeline(cfg)
        pipeline.scan_folder = scan_folder

        # Create 6 synthetic point clouds
        pcd_template = o3d.geometry.PointCloud()
        pcd_template.points = o3d.utility.Vector3dVector(np.random.uniform(-0.1, 0.1, size=(50, 3)))
        aligned_frames = [pcd_template for _ in range(6)]

        # Diagnostic configurations:
        # Frame 0: origin reference (fused)
        # Frame 1: upper frame, fit 0.85, rmse 0.002 (fused)
        # Frame 2: upper frame, fit 0.65 (< min_face_fit 0.75) (excluded)
        # Frame 3: upper frame, rmse 0.0055 (> max_face_rmse 4.0mm) (excluded)
        # Frame 4: lower frame, fit 0.50 (< min_fit_fusion 0.60) (excluded)
        # Frame 5: rejected in Stage 3 (accepted=False) (excluded)
        diagnostics = [
            (1.0, 0.0, True),
            (0.85, 0.0020, True),
            (0.65, 0.0020, True),
            (0.80, 0.0055, True),
            (0.50, 0.0020, True),
            (0.20, 0.0100, False),
        ]

        pipeline.last_registration_stats = {
            "frame_methods": {0: "reference", 1: "colored_icp", 2: "colored_icp", 3: "colored_icp", 4: "fpfh_fallback", 5: "none"},
            "frame_scores": {0: "(origin)", 1: "fitness=0.85", 2: "fitness=0.65", 3: "fitness=0.80", 4: "fitness=0.50", 5: "none"},
            "residuals": {1: 2.1, 2: 2.2, 3: 2.3, 4: 2.4, 5: None},
            "fusion_eligible": {0: True, 1: True, 2: False, 3: False, 4: False, 5: False},
            "fusion_reasons": {
                2: "face-pair fitness 0.65 < 0.75 threshold",
                3: "face-pair rmse 5.5mm > 4.0mm threshold",
                4: "fitness 0.50 < 0.60 fusion threshold",
                5: "rejected in Stage 3",
            },
        }

        accepted_frames, accepted_indices = pipeline._filter_fusion_eligible_frames(
            aligned_frames, diagnostics
        )

        assert accepted_indices == [0, 1]
        assert len(accepted_frames) == 2

        # Check console output for one-line summary
        captured = capsys.readouterr()
        assert "Stage 4 Gate Summary: 5 frames registered, 2 fused" in captured.out

        # Verify JSON report was created in scan_folder and output_dir
        report_scan = scan_folder / "stage4_gate_report.json"
        report_out = out_dir / "stage4_gate_report.json"
        assert report_scan.exists()
        assert report_out.exists()

        with open(report_scan, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data["summary"]
        assert summary["frames_total"] == 6
        assert summary["frames_registered"] == 5
        assert summary["frames_fused"] == 2
        assert summary["frames_excluded"] == 4

        # Gate failures tallies
        excluded_gates = summary["excluded_by_gate"]
        assert excluded_gates["min_face_pair_fitness_for_fusion"] >= 1
        assert excluded_gates["max_face_pair_rmse_for_fusion_mm"] >= 1
        assert excluded_gates["min_fitness_for_fusion"] >= 1

        # Check per-frame entries
        frames_data = data["frames"]
        assert len(frames_data) == 6

        # Frame 0: fused
        assert frames_data[0]["final_decision"] == "fused"
        assert frames_data[0]["registration_method"] == "reference"

        # Frame 1: fused
        assert frames_data[1]["final_decision"] == "fused"
        assert frames_data[1]["fitness"] == 0.85

        # Frame 2: excluded by face fitness
        assert frames_data[2]["final_decision"] == "excluded"
        assert "min_face_pair_fitness_for_fusion" in frames_data[2]["failed_gates"]
        assert frames_data[2]["failed_gates"]["min_face_pair_fitness_for_fusion"]["actual"] == 0.65
        assert frames_data[2]["failed_gates"]["min_face_pair_fitness_for_fusion"]["threshold"] == 0.75

        # Frame 3: excluded by face rmse
        assert frames_data[3]["final_decision"] == "excluded"
        assert "max_face_pair_rmse_for_fusion_mm" in frames_data[3]["failed_gates"]

        # Frame 5: rejected in registration
        assert frames_data[5]["final_decision"] == "excluded"
        assert "Stage 3" in frames_data[5]["reason"]


# ===========================================================================
# 5. Multiway & Loop-Closure End-to-End Synthetic Head Tests
# ===========================================================================

class TestMultiwayAndLoopClosureHeadExecution:
    """Verify that multiway registration and loop-closure run without exceptions in head mode."""

    def test_loop_closure_with_rgbd_frames(self, tmp_path):
        """_check_pointcloud_loop_closure executes without error when isolated frames are RGBDFrame instances."""
        out_dir = tmp_path / "lc_test"
        out_dir.mkdir(parents=True)
        cfg = PipelineConfig(
            output_dir=out_dir,
            target=TargetConfig(name="head"),
            attempt_loop_closure=True,
        )
        pipeline = ScanPipeline(cfg)

        intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500, 500, 320, 240)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-0.1, 0.1, size=(50, 3)))

        frames = [
            RGBDFrame(
                frame_id=i,
                color_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
                depth_m=np.ones((10, 10), dtype=np.float32),
                pcd=pcd,
                timestamp=float(i),
                intrinsics=intr,
            )
            for i in range(10)
        ]
        diagnostics = [(0.90, 0.001, True) for _ in range(10)]

        # Should execute safely and inspect loop closure
        pipeline._check_pointcloud_loop_closure(frames, diagnostics)

    def test_register_pointcloud_with_multiway_unwraps_rgbd_frames(self, tmp_path):
        """_register_pointcloud unwraps RGBDFrame instances to PointCloud before calling register_sequence_multiway."""
        out_dir = tmp_path / "mw_test"
        out_dir.mkdir(parents=True)
        cfg = PipelineConfig(
            output_dir=out_dir,
            target=TargetConfig(name="head"),
            show_stage_windows=False,
            save_intermediate=False,
        )
        cfg.registration.use_multiway = True
        pipeline = ScanPipeline(cfg)

        intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 500, 500, 320, 240)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-0.1, 0.1, size=(30, 3)))

        frames = [
            RGBDFrame(
                frame_id=i,
                color_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
                depth_m=np.ones((10, 10), dtype=np.float32),
                pcd=pcd,
                timestamp=float(i),
                intrinsics=intr,
            )
            for i in range(3)
        ]

        aligned, diag, poses = pipeline._register_pointcloud(frames)
        assert len(aligned) == 3
        assert len(diag) == 3
