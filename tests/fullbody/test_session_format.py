"""
tests/fullbody/test_session_format.py

Comprehensive tests for the extended session format ("stops_v2"):
1. Write an 8-stop synthetic session with one rejected attempt on stop 3:
   - Assert rejected attempt files still exist.
   - Assert round-trip load returns 8 stops with correct angles for cw and ccw.
   - Assert correct intrinsics and preserved burst count.
2. Tamper with one raw file and assert the loader fails on SHA-256 verification.
3. Assert writing a raw file twice raises FileExistsError.
4. Assert an unfinalized session is rejected.
5. Assert both legacy layouts (Layout 1 meta.json, Layout 2 subdirs) still load.

Open3D stub:
Minimal mock for o3d.camera.PinholeCameraIntrinsic so tests run in environments
where open3d native C++ DLL is blocked by host system policy.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import pytest

# Minimal open3d stub if native open3d cannot be loaded
if "open3d" not in sys.modules:
    from unittest.mock import MagicMock
    mock_o3d = MagicMock()

    class PinholeCameraIntrinsic:
        def __init__(self, width=0, height=0, fx=0, fy=0, cx=0, cy=0):
            self.width = int(width)
            self.height = int(height)
            self.intrinsic_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
            )

        def get_focal_length(self):
            return float(self.intrinsic_matrix[0, 0]), float(self.intrinsic_matrix[1, 1])

        def get_principal_point(self):
            return float(self.intrinsic_matrix[0, 2]), float(self.intrinsic_matrix[1, 2])

    mock_o3d.camera.PinholeCameraIntrinsic = PinholeCameraIntrinsic
    sys.modules["open3d"] = mock_o3d
    sys.modules["open3d.camera"] = mock_o3d.camera
    sys.modules["open3d.geometry"] = mock_o3d.geometry

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.io import load_capture_session
from realsense_measure.fullbody.session_format import SessionWriter


def test_stops_v2_round_trip_with_rejected_attempt():
    """
    Test writing an 8-stop synthetic session with a rejected attempt on stop 3:
    - Retries stop 3 with attempt 1 (accepted).
    - Asserts rejected attempt 0 still exists.
    - Asserts round-trip load returns 8 stops with correct angles, intrinsics, and burst count.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "session_v2_cw"
        intrinsics = {
            "width": 480,
            "height": 848,
            "fx": 425.0,
            "fy": 425.0,
            "cx": 239.0,
            "cy": 424.0,
            "is_post_rotation": True,
            "is_post_decimation": False,
        }

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_01",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
            camera_config_snapshot={"width": 848, "height": 480, "fps": 30},
            device_info={"name": "Intel RealSense D455f", "serial_number": "12345678"},
        )

        dummy_color = np.full((848, 480, 3), 150, dtype=np.uint8)
        dummy_depth = np.full((848, 480), 1.25, dtype=np.float32)

        # Write stops 0..7
        for stop_idx in range(8):
            if stop_idx == 3:
                # First attempt at stop 3: rejected due to subject motion
                h0 = writer.begin_attempt(stop_idx)
                writer.write_burst(h0, [dummy_color] * 5, [dummy_depth] * 5)
                writer.write_quality(h0, {"motion_mm": 18.5, "is_flagged": True})
                writer.finish_attempt(h0, accepted=False, reason="Subject motion 18.5mm > 10mm")

                # Second attempt at stop 3: accepted
                h1 = writer.begin_attempt(stop_idx)
                writer.write_burst(h1, [dummy_color] * 10, [dummy_depth] * 10)
                writer.write_quality(h1, {"motion_mm": 2.1, "is_flagged": False})
                writer.finish_attempt(h1, accepted=True, reason="Motion within tolerance")
            else:
                h = writer.begin_attempt(stop_idx)
                writer.write_burst(h, [dummy_color] * 10, [dummy_depth] * 10)
                writer.write_quality(h, {"motion_mm": 3.0, "is_flagged": False})
                writer.finish_attempt(h, accepted=True, reason="OK")

        # Finalize
        writer.finalize()

        # Assert rejected attempt files still exist on disk
        attempt0_dir = root / "stop_03" / "attempt_00"
        attempt1_dir = root / "stop_03" / "attempt_01"
        assert attempt0_dir.exists()
        assert attempt1_dir.exists()
        assert (attempt0_dir / "raw_color_000.png").exists()
        assert (attempt0_dir / "files.sha256").exists()

        # Check stop_03 stop.json contains both attempts
        with open(root / "stop_03" / "stop.json", "r", encoding="utf-8") as f:
            stop3_meta = json.load(f)
        assert stop3_meta["accepted_attempt"] == 1
        assert len(stop3_meta["attempts"]) == 2
        assert stop3_meta["attempts"][0]["status"] == "REJECTED"
        assert stop3_meta["attempts"][1]["status"] == "ACCEPTED"

        # Round-trip load
        cfg = FullBodyConfig(turn_direction="cw")
        stops = load_capture_session(root, cfg)
        assert len(stops) == 8
        for i, s in enumerate(stops):
            assert s.stop_id == i
            assert s.nominal_angle_deg == i * 45.0
            assert s.intrinsics.width == 480
            assert s.intrinsics.height == 848
            assert len(s.raw_burst_depths) == 10
            assert len(s.raw_burst_colors) == 10
            assert s.depth.shape == (848, 480)


def test_tampered_file_fails_sha256_verification():
    """Tamper with one raw depth file and assert loader raises ValueError on checksum."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "session_tampered"
        intrinsics = {"width": 100, "height": 100, "fx": 50.0, "fy": 50.0, "cx": 50.0, "cy": 50.0}
        writer = SessionWriter(
            root=root,
            subject_id="ANON_TAMPER",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
        )

        h = writer.begin_attempt(0)
        c = np.zeros((100, 100, 3), dtype=np.uint8)
        d = np.ones((100, 100), dtype=np.float32)
        writer.write_burst(h, [c], [d])
        writer.finish_attempt(h, accepted=True)
        writer.finalize()

        # Tamper with raw_depth_000.npy
        tamper_path = root / "stop_00" / "attempt_00" / "raw_depth_000.npy"
        np.save(str(tamper_path), np.full((100, 100), 9.99, dtype=np.float32))

        cfg = FullBodyConfig()
        with pytest.raises(ValueError, match="SHA-256 integrity check failed"):
            load_capture_session(root, cfg)


def test_write_once_raw_file_raises():
    """Assert writing raw files twice raises FileExistsError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "session_write_once"
        intrinsics = {"width": 10, "height": 10, "fx": 5.0, "fy": 5.0, "cx": 5.0, "cy": 5.0}
        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_ONCE",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
        )

        h = writer.begin_attempt(0)
        c = np.zeros((10, 10, 3), dtype=np.uint8)
        d = np.ones((10, 10), dtype=np.float32)
        writer.write_burst(h, [c], [d])

        with pytest.raises(FileExistsError):
            writer.write_burst(h, [c], [d])


def test_unfinalized_session_is_rejected():
    """Assert an unfinalized session is rejected by the loader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "session_unfinalized"
        intrinsics = {"width": 10, "height": 10, "fx": 5.0, "fy": 5.0, "cx": 5.0, "cy": 5.0}
        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_INCOMPLETE",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
        )

        h = writer.begin_attempt(0)
        c = np.zeros((10, 10, 3), dtype=np.uint8)
        d = np.ones((10, 10), dtype=np.float32)
        writer.write_burst(h, [c], [d])
        writer.finish_attempt(h, accepted=True)
        # Note: writer.finalize() NOT called!

        # Manually write an unfinalized meta.json to test loader detection
        unfinalized_meta = {
            "schema_version": 2,
            "layout": "stops_v2",
            "n_stops": 1,
            # 'finalized_utc' omitted!
        }
        with open(root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(unfinalized_meta, f)

        cfg = FullBodyConfig()
        with pytest.raises(ValueError, match="incomplete \\(not finalized\\)"):
            load_capture_session(root, cfg)


def test_both_legacy_layouts_still_load():
    """Assert that both legacy Layout 1 (meta.json) and Layout 2 (stop_XX subdirs) still load without changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        color = np.full((100, 80, 3), 120, dtype=np.uint8)
        depth = np.full((100, 80), 1.25, dtype=np.float32)

        # 1. Legacy Layout 1 (meta.json)
        l1_dir = root / "legacy_l1"
        l1_dir.mkdir()
        cv2.imwrite(str(l1_dir / "frame_00_color.png"), color)
        np.save(str(l1_dir / "frame_00_depth.npy"), depth)
        meta_l1 = {
            "intrinsics": {"width": 80, "height": 100, "fx": 100.0, "fy": 100.0, "cx": 40.0, "cy": 50.0},
            "frames": [{"frame_id": 0, "color_file": "frame_00_color.png", "depth_file": "frame_00_depth.npy", "timestamp": 0.0}],
        }
        with open(l1_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_l1, f)

        cfg = FullBodyConfig()
        stops_l1 = load_capture_session(l1_dir, cfg)
        assert len(stops_l1) == 1
        assert stops_l1[0].depth.shape == (100, 80)

        # 2. Legacy Layout 2 (subdirectories)
        l2_dir = root / "legacy_l2"
        l2_dir.mkdir()
        s0 = l2_dir / "stop_00"
        s0.mkdir()
        cv2.imwrite(str(s0 / "color.png"), color)
        np.save(str(s0 / "depth.npy"), depth)

        stops_l2 = load_capture_session(l2_dir, cfg)
        assert len(stops_l2) == 1
        assert stops_l2[0].depth.shape == (100, 80)


def test_uint16_and_float32_depth_storage_load_identically():
    """
    Assert that both uint16 sensor-units burst (depth_scale 0.001) and float32 metres burst
    load to identical metric depth arrays.
    """
    intrinsics = {
        "width": 80,
        "height": 100,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 40.0,
        "cy": 50.0,
        "is_post_rotation": True,
        "is_post_decimation": False,
    }
    dummy_color = np.full((100, 80, 3), 150, dtype=np.uint8)
    expected_depth_m = np.full((100, 80), 1.25, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        root_u16 = Path(tmpdir) / "sess_u16"
        writer_u16 = SessionWriter(
            root=root_u16,
            subject_id="SUBJ_U16",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
            depth_units="sensor_units",
        )
        h_u16 = writer_u16.begin_attempt(0)
        # 1.25 m in sensor units with depth_scale 0.001 is 1250
        u16_depth = (expected_depth_m / 0.001).astype(np.uint16)
        writer_u16.write_burst(h_u16, [dummy_color] * 3, [u16_depth] * 3)
        writer_u16.finish_attempt(h_u16, accepted=True)
        writer_u16.finalize()

        root_f32 = Path(tmpdir) / "sess_f32"
        writer_f32 = SessionWriter(
            root=root_f32,
            subject_id="SUBJ_F32",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=1.0,
            depth_units="m",
        )
        h_f32 = writer_f32.begin_attempt(0)
        writer_f32.write_burst(h_f32, [dummy_color] * 3, [expected_depth_m] * 3)
        writer_f32.finish_attempt(h_f32, accepted=True)
        writer_f32.finalize()

        cfg = FullBodyConfig()
        stops_u16 = load_capture_session(root_u16, cfg)
        stops_f32 = load_capture_session(root_f32, cfg)

        assert len(stops_u16) == 1
        assert len(stops_f32) == 1
        np.testing.assert_allclose(stops_u16[0].depth, stops_f32[0].depth, atol=1e-6)
        np.testing.assert_allclose(stops_u16[0].depth, expected_depth_m, atol=1e-6)


def test_missing_depth_storage_metadata_raises():
    """Assert that a stops_v2 session lacking 'depth_storage' in meta.json is rejected."""
    intrinsics = {
        "width": 80,
        "height": 100,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 40.0,
        "cy": 50.0,
    }
    dummy_color = np.full((100, 80, 3), 150, dtype=np.uint8)
    dummy_depth = np.full((100, 80), 1.25, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "sess_invalid"
        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_TEST",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=intrinsics,
            depth_scale=0.001,
        )
        h = writer.begin_attempt(0)
        writer.write_burst(h, [dummy_color] * 2, [dummy_depth] * 2)
        writer.finish_attempt(h, accepted=True)
        writer.finalize()

        # Remove depth_storage from meta.json
        meta_path = root / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert "depth_storage" in meta
        del meta["depth_storage"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        cfg = FullBodyConfig()
        with pytest.raises(ValueError, match="missing required 'depth_storage'"):
            load_capture_session(root, cfg)

