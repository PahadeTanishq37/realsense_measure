"""
tests/body/test_physical_stage11_hardware_smoke_test.py
======================================================
Tests for Stage 11: Live RealSense Hardware Smoke Test & Full-Body Framing Trial.

Validates:
1. Hardware device probe safe fallback and data structures.
2. Live hardware smoke test mock execution and metric verification.
3. Collision prevention guarding against accidental overwrite of raw captures.
4. Optical-to-world coordinate transformation and pitch tilt compensation.
5. Framing trial protocol matrix covering standoffs (2.2m, 2.4m, 2.6m), heights (0.9m, 1.0m, 1.2m), and tilts.
6. Generation and schema integrity of all 9 required Stage 11 deliverables.
7. Preservation of historical raw_captures/body_capture_01.
8. save_raw_capture integration with collision guard.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from config import PipelineConfig
from realsense_measure.body.hardware_smoke_test import (
    CollisionError,
    check_capture_collision,
    DeviceInfo,
    HardwareProbeResult,
    HardwareDeviceProbe,
    LiveSmokeTestResult,
    LiveHardwareSmokeTester,
    CoordinateGeometryAuditor,
    FramingCondition,
    FramingTrialProtocol,
    Stage11DeliverablesGenerator,
)
from realsense_measure.body.adapter import (
    camera_to_world_matrix,
    world_to_camera_matrix,
    camera_to_world_points,
    world_to_camera_points,
)
from camera.recording import save_raw_capture
from camera.frame import RGBDFrame
import open3d as o3d


RAW_DIR = Path("raw_captures/body_capture_01")
STAGE11_DIR = Path("stage11_hardware_smoke_test_output")


# ---------------------------------------------------------------------------
# 1. Hardware Device Probe
# ---------------------------------------------------------------------------

def test_hardware_device_probe_returns_valid_result():
    """Verify probe returns a structured result without throwing exceptions."""
    res = HardwareDeviceProbe.probe()
    assert isinstance(res, HardwareProbeResult)
    assert isinstance(res.status, str)
    assert isinstance(res.notes, list)
    assert res.probe_timestamp != ""

    res_dict = res.to_dict()
    assert "pyrealsense2_available" in res_dict
    assert "device_connected" in res_dict
    assert "device_count" in res_dict


def test_hardware_device_probe_fallback_when_no_pyrealsense2():
    """Verify clean fallback when pyrealsense2 is not available."""
    with patch.dict("sys.modules", {"pyrealsense2": None}):
        # Mock import failure
        with patch("builtins.__import__", side_effect=ImportError("No module pyrealsense2")):
            res = HardwareDeviceProbe.probe()
            assert res.pyrealsense2_available is False
            assert res.device_connected is False
            assert res.status == "PYREALSENSE2_NOT_INSTALLED"


# ---------------------------------------------------------------------------
# 2. Live Smoke Tester (Mock & Live Fallback)
# ---------------------------------------------------------------------------

def test_hardware_smoke_tester_mock_execution():
    """Verify mock execution returns valid streaming simulation."""
    res = LiveHardwareSmokeTester.run_smoke_test(
        num_frames=10,
        target_resolution=(1280, 720),
        target_fps=30,
        is_mock=True,
    )
    assert res.is_mock is True
    assert res.stream_active is True
    assert res.frames_captured == 10
    assert res.depth_resolution == (1280, 720)
    assert res.color_resolution == (1280, 720)
    assert res.depth_scale_m == 0.001
    assert res.valid_depth_fraction > 0.80
    assert res.timestamp_monotonic is True
    assert "PASS" in res.status


# ---------------------------------------------------------------------------
# 3. Collision Prevention
# ---------------------------------------------------------------------------

def test_check_capture_collision_raises_on_existing_data(tmp_path):
    """Verify CollisionError is raised when trying to overwrite a non-empty capture folder."""
    capture_dir = tmp_path / "existing_capture"
    capture_dir.mkdir(parents=True)
    (capture_dir / "meta.json").write_text("{}", encoding="utf-8")

    # Should raise by default
    with pytest.raises(CollisionError) as exc_info:
        check_capture_collision(capture_dir, allow_overwrite=False)
    assert "Collision detected" in str(exc_info.value)

    # Should pass when allow_overwrite=True
    collided, msg = check_capture_collision(capture_dir, allow_overwrite=True)
    assert collided is True
    assert "allow_overwrite=True" in msg


def test_check_capture_collision_passes_clean_directory(tmp_path):
    """Verify clean directory produces no collision."""
    clean_dir = tmp_path / "fresh_capture"
    collided, msg = check_capture_collision(clean_dir, allow_overwrite=False)
    assert collided is False


def test_save_raw_capture_respects_collision_guard(tmp_path):
    """Verify save_raw_capture raises CollisionError if target has existing meta.json."""
    target_dir = tmp_path / "raw_session"
    target_dir.mkdir(parents=True)
    (target_dir / "meta.json").write_text("{}", encoding="utf-8")

    dummy_color = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_depth = np.ones((720, 1280), dtype=np.float32) * 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(1280, 720, 640.0, 640.0, 640.0, 360.0)
    frame = RGBDFrame(
        color_bgr=dummy_color,
        depth_m=dummy_depth,
        frame_id=0,
        timestamp=0.0,
        pcd=o3d.geometry.PointCloud(),
        intrinsics=intr,
    )

    cfg = PipelineConfig()
    cfg.allow_overwrite = False

    with pytest.raises(CollisionError):
        save_raw_capture([frame], target_dir, cfg)


# ---------------------------------------------------------------------------
# 4. Coordinate Geometry & Tilt Transformation
# ---------------------------------------------------------------------------

def test_coordinate_geometry_level_inversion():
    """Verify camera_to_world and world_to_camera are exact inverses at level pitch."""
    dist = 2.4
    T_c2w = camera_to_world_matrix(distance_m=dist, camera_height_m=1.0, pitch_deg=0.0)
    T_w2c = world_to_camera_matrix(distance_m=dist, camera_height_m=1.0, pitch_deg=0.0)

    # Identity check
    product = T_c2w @ T_w2c
    np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    # Point round-trip check
    pts_cam = np.array([[0.1, -0.2, 2.3], [-0.5, 0.4, 2.5]], dtype=np.float32)
    pts_world = camera_to_world_points(pts_cam, distance_m=dist, camera_height_m=1.0, pitch_deg=0.0)
    pts_cam_recovered = world_to_camera_points(pts_world, distance_m=dist, camera_height_m=1.0, pitch_deg=0.0)
    np.testing.assert_allclose(pts_cam, pts_cam_recovered, atol=1e-4)


def test_coordinate_geometry_pitch_tilt_downward():
    """Verify downward pitch tilts optical axis towards the floor."""
    dist = 2.5
    height = 1.0
    pitch_deg = 5.0

    T_tilt = camera_to_world_matrix(distance_m=dist, camera_height_m=height, pitch_deg=pitch_deg)
    T_level = camera_to_world_matrix(distance_m=dist, camera_height_m=height, pitch_deg=0.0)

    # Point along optical axis
    p_opt = np.array([[0.0, 0.0, dist]], dtype=np.float32)
    p_tilt_w = CoordinateGeometryAuditor.transform_camera_to_world(p_opt, dist, height, pitch_deg)
    p_level_w = CoordinateGeometryAuditor.transform_camera_to_world(p_opt, dist, height, 0.0)

    # Under downward tilt, optical center line hits a lower elevation Z
    assert p_tilt_w[0, 2] < p_level_w[0, 2]
    # Discrepancy should match dist * sin(pitch)
    expected_drop = dist * np.sin(np.radians(pitch_deg))
    actual_drop = p_level_w[0, 2] - p_tilt_w[0, 2]
    assert np.isclose(actual_drop, expected_drop, atol=1e-3)


def test_coordinate_geometry_audit_report():
    """Verify tilt audit accurately computes slant error and warnings."""
    audit = CoordinateGeometryAuditor.audit_tilt_effect_on_vertical_axis(
        distance_m=2.4, camera_height_m=1.0, pitch_deg=5.0
    )
    assert audit["standoff_m"] == 2.4
    assert audit["pitch_deg"] == 5.0
    assert audit["vertical_elevation_error_mm"] > 200.0  # ~209 mm at 2.4m
    assert "diagonal" in audit["diagnostic"].lower() or "slant" in audit["diagnostic"].lower()


# ---------------------------------------------------------------------------
# 5. Framing Trial Protocol
# ---------------------------------------------------------------------------

def test_framing_trial_protocol_matrix():
    """Verify matrix contains 18 conditions across designated parameter space."""
    matrix = FramingTrialProtocol.generate_full_trial_matrix()
    # 3 standoffs x 3 heights x 2 tilts = 18 conditions
    assert len(matrix) == 18

    # Check standoffs present
    dists = {c.standoff_m for c in matrix}
    assert dists == {2.2, 2.4, 2.6}

    # Check heights present
    heights = {c.camera_height_m for c in matrix}
    assert heights == {0.9, 1.0, 1.2}

    # Check tilts present
    tilts = {c.pitch_deg for c in matrix}
    assert tilts == {0.0, 5.0}

    # Verify at least some conditions provide full body coverage
    covering = [c for c in matrix if c.full_body_covered]
    assert len(covering) > 0


# ---------------------------------------------------------------------------
# 6. Deliverables Generator
# ---------------------------------------------------------------------------

def test_stage11_deliverables_generator_produces_all_files(tmp_path):
    """Verify all 9 Stage 11 deliverables are created and valid."""
    probe = HardwareProbeResult(
        pyrealsense2_available=True,
        device_connected=True,
        device_count=1,
        device_info=DeviceInfo(
            name="Intel RealSense D455",
            serial_number="TEST_SN_123",
            firmware_version="5.13.0.50",
            usb_type_descriptor="3.2",
            is_usb3=True,
            sensors=["Stereo Module", "RGB Camera"],
        ),
        status="HARDWARE_DETECTED_READY",
    )
    smoke = LiveSmokeTestResult(
        is_mock=True,
        device_connected=True,
        device_name="Intel RealSense D455",
        serial_number="TEST_SN_123",
        stream_active=True,
        frames_captured=15,
        depth_resolution=(1280, 720),
        color_resolution=(1280, 720),
        status="PASS_MOCK",
    )

    deliverables = Stage11DeliverablesGenerator.generate_all(tmp_path, smoke, probe)
    expected_keys = [
        "stage11_report",
        "checklist",
        "framing_protocol",
        "capture_commands",
        "expected_output",
        "hardware_status",
        "coordinate_audit",
        "known_issues",
        "test_results",
    ]
    for key in expected_keys:
        assert key in deliverables
        p = deliverables[key]
        assert p.exists()
        assert p.stat().st_size > 0

    # Validate JSON deliverables
    with open(deliverables["hardware_status"], "r", encoding="utf-8") as f:
        stat_doc = json.load(f)
        assert stat_doc["verdict"] == "PASS_MOCK"

    with open(deliverables["test_results"], "r", encoding="utf-8") as f:
        res_doc = json.load(f)
        assert res_doc["stage"] == 11
        assert res_doc["trial_conditions_evaluated"] == 18


# ---------------------------------------------------------------------------
# 7. Preservation of Historical Dataset
# ---------------------------------------------------------------------------

def test_historical_raw_capture_remains_untouched():
    """Verify raw_captures/body_capture_01 is preserved and untouched."""
    assert RAW_DIR.exists(), f"Historical directory missing: {RAW_DIR}"
    meta_path = RAW_DIR / "meta.json"
    assert meta_path.exists(), f"meta.json missing: {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 4 yaw stops
    assert meta["num_frames"] == 4
    for i in range(4):
        c_file = RAW_DIR / f"frame_{i:02d}_color.png"
        d_file = RAW_DIR / f"frame_{i:02d}_depth.npy"
        assert c_file.exists()
        assert d_file.exists()

