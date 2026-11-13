"""
tests/fullbody/test_stage27_adapter_ui_cli.py

Unit tests for Stage 27 Change 5:
- RealSenseCaptureAdapter lifecycle, warmup frame discarding, dropped frame retry,
  disconnect handling, decimation refusal.
- full_body_camera_config parameters.
- capture_ui render_frame pure headless HUD rendering.
- cli import isolation (no open3d or pyrealsense2 in sys.modules).
- cli capture --fake end-to-end and failure handling.
- cli check-camera --fake snapshot output.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from realsense_measure.config import CameraConfig
from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
from realsense_measure.fullbody.capture_controller import ControllerState
from realsense_measure.fullbody.capture_ui import render_frame
from realsense_measure.fullbody.cli import main
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.io import load_capture_session


def test_full_body_camera_config():
    """Verify full_body_camera_config sets required Stage 27 parameters."""
    cfg_cw = full_body_camera_config("cw", resolution="848x480", fps=30)
    assert cfg_cw.width == 848
    assert cfg_cw.height == 480
    assert cfg_cw.fps == 30
    assert cfg_cw.depth_max_m >= 2.6
    assert cfg_cw.depth_min_m == pytest.approx(0.30)
    assert cfg_cw.use_decimation is False
    assert cfg_cw.rotate_90 == "cw"

    cfg_ccw = full_body_camera_config("ccw")
    assert cfg_ccw.rotate_90 == "ccw"
    assert cfg_ccw.use_decimation is False

    with pytest.raises(ValueError, match="Invalid portrait_rotation"):
        full_body_camera_config("invalid")


def test_adapter_warmup_and_raw_depth_units():
    """Verify adapter drops warmup frames and returns uint16 raw depth with correct depth_scale."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001

    class MockIntr:
        width = 480
        height = 848
        def get_focal_length(self):
            return 425.0, 425.0
        def get_principal_point(self):
            return 240.0, 424.0

    mock_cam.intrinsics_o3d = MockIntr()
    mock_cam.get_device_info.return_value = {
        "name": "Mock D455",
        "serial_number": "MOCK_123",
        "firmware_version": "5.13.0",
    }

    call_count = [0]
    dummy_c = np.full((848, 480, 3), 100, dtype=np.uint8)
    dummy_filt = np.full((848, 480), 1.25, dtype=np.float32)
    dummy_raw_m = np.full((848, 480), 1.25, dtype=np.float32)
    dummy_z16 = np.full((848, 480), 1250, dtype=np.uint16)
    dummy_meta = {"timestamp_ms": 1000.0}

    def mock_read():
        call_count[0] += 1
        return dummy_c, dummy_filt, dummy_raw_m, dummy_z16, None, dummy_meta

    mock_cam.read_with_raw_and_colorized.side_effect = mock_read

    adapter = RealSenseCaptureAdapter(
        portrait_rotation="cw",
        warmup_frames=5,
        camera_factory=lambda cfg: mock_cam,
    )

    adapter.start()
    assert call_count[0] == 5  # 5 warmup frames discarded
    assert adapter.depth_units == "sensor_units"
    assert adapter.depth_scale == 0.001
    assert adapter.intrinsics["width"] == 480
    assert adapter.intrinsics["height"] == 848

    c, r, f, ts = adapter.read_raw_filtered()
    assert c is not None and r is not None and f is not None
    assert r.dtype == np.uint16
    assert f.dtype == np.float32
    assert ts == pytest.approx(1.0)
    assert call_count[0] == 6

    adapter.stop()
    mock_cam.stop.assert_called_once()


def test_adapter_dropped_frames_retried():
    """Verify adapter retries dropped frames (None) up to max_dropped_retries."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001
    mock_cam.intrinsics_o3d = None

    attempts = [0]
    dummy_c = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_z16 = np.zeros((10, 10), dtype=np.uint16)

    def mock_read():
        attempts[0] += 1
        if attempts[0] < 3:
            return None, None, None, None, None, None
        return dummy_c, None, None, dummy_z16, None, {"timestamp_ms": 2000.0}

    mock_cam.read_with_raw_and_colorized.side_effect = mock_read

    adapter = RealSenseCaptureAdapter(
        warmup_frames=0,
        max_dropped_retries=5,
        camera_factory=lambda cfg: mock_cam,
    )
    adapter.start()
    c, r, f, ts = adapter.read_raw_filtered()
    assert c is not None
    assert r is not None
    assert attempts[0] == 3
    adapter.stop()


def test_adapter_disconnect_raises_and_stops():
    """Verify hardware failure raises RuntimeError and ensures stop() runs."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001
    mock_cam.read_with_raw_and_colorized.side_effect = RuntimeError("USB device disconnected")

    adapter = RealSenseCaptureAdapter(
        warmup_frames=0,
        camera_factory=lambda cfg: mock_cam,
    )
    adapter.start()

    with pytest.raises(RuntimeError, match="RealSense camera disconnected"):
        adapter.read_raw_filtered()

    adapter.stop()
    assert mock_cam.stop.called


def test_adapter_refuses_decimation():
    """Verify adapter refuses to start if decimation is active."""
    cfg = CameraConfig(use_decimation=True)
    adapter = RealSenseCaptureAdapter(camera_config=cfg)
    with pytest.raises(RuntimeError, match="decimation is active"):
        adapter.start()


def test_render_frame_shapes_and_instructions():
    """Verify render_frame produces expected dimensions and differing turn text for CW vs CCW."""
    color = np.zeros((480, 270, 3), dtype=np.uint8)
    depth = np.full((480, 270), 1.25, dtype=np.float32)

    for st in ControllerState:
        snap = {
            "state": st.value,
            "stop_index": 2,
            "n_stops": 8,
            "turn_direction": "cw",
            "distance_m": 1.25,
            "valid_depth_pct": 85.0,
            "motion_mm": 3.2,
            "head_visible": True,
            "feet_visible": True,
            "burst_progress": 0.5,
            "last_rejection_reason": "Slight sway",
        }
        out_img = render_frame(snap, color, depth)
        assert out_img.shape == (480, 540, 3)

    # CW instruction vs CCW instruction
    snap_cw = {"state": "TURN_PROMPT", "turn_direction": "cw", "n_stops": 8}
    snap_ccw = {"state": "TURN_PROMPT", "turn_direction": "ccw", "n_stops": 8}

    img_cw = render_frame(snap_cw, color, depth)
    img_ccw = render_frame(snap_ccw, color, depth)

    # Images must not be identical because text says RIGHT vs LEFT
    assert not np.array_equal(img_cw, img_ccw)


def test_cli_import_isolation():
    """Verify importing cli does NOT import pyrealsense2 or open3d."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; from realsense_measure.fullbody import cli; "
        "assert 'pyrealsense2' not in sys.modules, 'pyrealsense2 imported!'; "
        "assert 'open3d' not in sys.modules, 'open3d imported!'; "
        "print('CLEAN_IMPORT')",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI import failed: {proc.stderr}"
    assert "CLEAN_IMPORT" in proc.stdout


def test_cli_capture_fake_end_to_end():
    """Verify python -m realsense_measure.fullbody capture --fake end-to-end creates finalized session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fake_capture"
        argv = [
            "capture",
            "--out", str(out_dir),
            "--subject-id", "SUBJ_CLI01",
            "--stops", "2",
            "--turn", "cw",
            "--portrait", "cw",
            "--fake",
            "--burst-frames", "3",
            "--headless",
        ]
        ret = main(argv)
        assert ret == 0
        assert (out_dir / "meta.json").exists()

        cfg = FullBodyConfig()
        stops = load_capture_session(out_dir, cfg)
        assert len(stops) == 2


def test_cli_capture_fake_failure_leaves_unfinalized(monkeypatch):
    """Verify capture failure or interruption leaves session unfinalized with data intact."""
    from realsense_measure.fullbody.capture_controller import CaptureController

    orig_step = CaptureController.step

    def failing_step(self, now):
        self._transition(ControllerState.ABORT, now)
        return ControllerState.ABORT

    monkeypatch.setattr(CaptureController, "step", failing_step)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "failing_capture"
        argv = [
            "capture",
            "--out", str(out_dir),
            "--subject-id", "SUBJ_FAIL",
            "--stops", "2",
            "--fake",
            "--headless",
        ]
        ret = main(argv)
        assert ret == 1
        assert not (out_dir / "meta.json").exists()


def test_cli_check_camera_fake():
    """Verify check-camera --fake creates snapshot images and metrics text without session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "camera_check_out"
        argv = [
            "check-camera",
            "--out", str(out_dir),
            "--portrait", "cw",
            "--fake",
        ]
        ret = main(argv)
        assert ret == 0

        chk_dir = out_dir / "check"
        assert (chk_dir / "check_color.png").exists()
        assert (chk_dir / "check_depth_colorized.png").exists()
        assert (chk_dir / "check_metrics.txt").exists()

        txt = (chk_dir / "check_metrics.txt").read_text(encoding="utf-8")
        assert "portrait_rotation: cw" in txt
        assert "depth_scale:" in txt

