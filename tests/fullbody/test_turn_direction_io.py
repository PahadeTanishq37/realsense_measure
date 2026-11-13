"""
tests/fullbody/test_turn_direction_io.py

Numpy-only unit tests for turn direction, canonical yaw conventions, and metadata persistence:
- (a) cw and ccw sequences for N=8 and N=4 differ and are exact mirror images about 180 deg.
- (b) stop 0 is 180 deg for both.
- (c) io.py loads the same stored nominal angles for both directions.
- (d) synthetic check: two yaw rotation matrices for cw and ccw at stop 2 are inverses about the vertical axis.
- Metadata override and warning checks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from realsense_measure.camera.recording import save_raw_capture
from realsense_measure.config import PipelineConfig
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.io import _load_meta_json_session, _load_subdirs_session, load_capture_session
from realsense_measure.fullbody.orientation import canonical_yaw_deg, nominal_angle_deg


@pytest.mark.parametrize("n_stops", [8, 4])
def test_cw_ccw_sequences_mirror_images_about_180(n_stops: int) -> None:
    """
    Assert that:
    (a) cw and ccw sequences for N=8 and N=4 differ and are exact mirror images about 180 deg.
    (b) stop 0 is 180 deg for both.
    """
    yaws_cw = [canonical_yaw_deg(nominal_angle_deg(i, n_stops), "cw") for i in range(n_stops)]
    yaws_ccw = [canonical_yaw_deg(nominal_angle_deg(i, n_stops), "ccw") for i in range(n_stops)]

    # (b) stop 0 is 180 for both
    assert yaws_cw[0] == 180.0
    assert yaws_ccw[0] == 180.0

    # (a) cw and ccw differ
    assert yaws_cw != yaws_ccw

    # Verify exact expected numbers
    if n_stops == 8:
        assert yaws_cw == [180.0, 135.0, 90.0, 45.0, 0.0, 315.0, 270.0, 225.0]
        assert yaws_ccw == [180.0, 225.0, 270.0, 315.0, 0.0, 45.0, 90.0, 135.0]
    elif n_stops == 4:
        assert yaws_cw == [180.0, 90.0, 0.0, 270.0]
        assert yaws_ccw == [180.0, 270.0, 0.0, 90.0]

    # Mirror image about 180:
    # (yaw_cw - 180) == -(yaw_ccw - 180) (modulo 360), i.e. (yaw_cw + yaw_ccw) % 360 == 0
    for i in range(n_stops):
        sum_mod = (yaws_cw[i] + yaws_ccw[i]) % 360.0
        assert np.isclose(sum_mod, 0.0, atol=1e-6) or np.isclose(sum_mod, 360.0, atol=1e-6), (
            f"Stop {i} not mirror: cw={yaws_cw[i]}, ccw={yaws_ccw[i]}"
        )


def test_yaw_rotation_matrices_inverses_at_stop_2() -> None:
    """
    (d) Synthetic check: build two yaw rotation matrices from the helper for cw and ccw
    at stop 2 and assert they are inverses of each other about the vertical axis.
    """
    # At stop 2 of 8:
    nom = nominal_angle_deg(2, 8)  # 90 deg
    yaw_cw = canonical_yaw_deg(nom, "cw")    # 90 deg
    yaw_ccw = canonical_yaw_deg(nom, "ccw")  # 270 deg

    def _rot_y(deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ], dtype=np.float64)

    R_cw = _rot_y(yaw_cw)
    R_ccw = _rot_y(yaw_ccw)

    # R_cw @ R_ccw should be identity because 90 + 270 = 360 = 0 deg
    prod = R_cw @ R_ccw
    eye = np.eye(3, dtype=np.float64)
    assert np.allclose(prod, eye, atol=1e-6), f"Product R_cw @ R_ccw is not identity:\n{prod}"
    assert np.allclose(R_cw.T, R_ccw, atol=1e-6), "R_cw transpose is not R_ccw"


def test_io_loaders_store_identical_nominal_angles_for_both_directions() -> None:
    """
    (c) io.py loads the same stored angles for both directions.
    Both _load_meta_json_session and _load_subdirs_session return
    nominal_angle_deg = [0, 45, 90, 135, 180, 225, 270, 315] regardless of turn_direction.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.ones((480, 848), dtype=np.float32)

        dir_meta = root / "session_meta"
        dir_meta.mkdir()
        dir_subdirs = root / "session_subdirs"
        dir_subdirs.mkdir()

        frames_meta = []
        for i in range(8):
            c_name = f"frame_{i:02d}_color.png"
            d_name = f"frame_{i:02d}_depth.npy"
            cv2.imwrite(str(dir_meta / c_name), color)
            np.save(str(dir_meta / d_name), depth)
            frames_meta.append({
                "frame_id": i,
                "color_file": c_name,
                "depth_file": d_name,
                "timestamp": float(i),
            })

            s_dir = dir_subdirs / f"stop_{i:02d}"
            s_dir.mkdir()
            cv2.imwrite(str(s_dir / "color.png"), color)
            np.save(str(s_dir / "depth.npy"), depth)

        meta_doc = {
            "intrinsics": {"width": 848, "height": 480, "fx": 425.0, "fy": 425.0, "cx": 424.0, "cy": 240.0},
            "frames": frames_meta,
        }
        with open(dir_meta / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_doc, f)

        expected_nominals = [i * 45.0 for i in range(8)]

        # Check CW
        cfg_cw = FullBodyConfig(turn_direction="cw")
        stops_meta_cw = load_capture_session(dir_meta, cfg_cw)
        stops_sub_cw = load_capture_session(dir_subdirs, cfg_cw)
        assert [s.nominal_angle_deg for s in stops_meta_cw] == expected_nominals
        assert [s.nominal_angle_deg for s in stops_sub_cw] == expected_nominals

        # Check CCW: stored nominal_angle_deg is STILL i * 45.0!
        cfg_ccw = FullBodyConfig(turn_direction="ccw")
        stops_meta_ccw = load_capture_session(dir_meta, cfg_ccw)
        stops_sub_ccw = load_capture_session(dir_subdirs, cfg_ccw)
        assert [s.nominal_angle_deg for s in stops_meta_ccw] == expected_nominals
        assert [s.nominal_angle_deg for s in stops_sub_ccw] == expected_nominals


def test_meta_json_overrides_cfg_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """
    Verify that when meta.json contains turn_direction and camera_portrait_rotation,
    they override cfg and emit a WARNING naming both values if they disagree.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cap_dir = Path(tmpdir)
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.ones((480, 848), dtype=np.float32)
        cv2.imwrite(str(cap_dir / "frame_00_color.png"), color)
        np.save(str(cap_dir / "frame_00_depth.npy"), depth)

        meta = {
            "intrinsics": {"width": 848, "height": 480, "fx": 425.0, "fy": 425.0, "cx": 424.0, "cy": 240.0},
            "turn_direction": "ccw",
            "camera_portrait_rotation": "ccw",
            "frames": [{"frame_id": 0, "color_file": "frame_00_color.png", "depth_file": "frame_00_depth.npy", "timestamp": 0.0}],
        }
        with open(cap_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        cfg = FullBodyConfig(turn_direction="cw", camera_portrait_rotation="cw")
        with caplog.at_level(logging.WARNING):
            stops = load_capture_session(cap_dir, cfg)

        assert len(stops) == 1
        assert any("turn_direction 'ccw' disagrees with cfg.turn_direction 'cw'" in rec.message for rec in caplog.records)
        assert any("camera_portrait_rotation 'ccw' disagrees with cfg.camera_portrait_rotation 'cw'" in rec.message for rec in caplog.records)


def test_save_raw_capture_persists_orientation_fields() -> None:
    """
    Verify that save_raw_capture writes turn_direction, camera_portrait_rotation,
    depth_scale, image_width, image_height, schema_version to meta.json.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        class MockFrame:
            color_bgr = np.zeros((480, 848, 3), dtype=np.uint8)
            depth_m = np.ones((480, 848), dtype=np.float32)
            frame_id = 0
            timestamp = 0.0
            class intrinsics:
                width = 848
                height = 480
                intrinsic_matrix = np.array([[425.0, 0, 424.0], [0, 425.0, 240.0], [0, 0, 1]])

        cfg = PipelineConfig()
        save_raw_capture(
            [MockFrame()],
            out_dir,
            cfg,
            turn_direction="ccw",
            camera_portrait_rotation="cw",
            depth_scale=0.001,
            image_width=848,
            image_height=480,
            schema_version="2.0.0",
        )

        with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["turn_direction"] == "ccw"
        assert meta["camera_portrait_rotation"] == "cw"
        assert meta["depth_scale"] == 0.001
        assert meta["image_width"] == 848
        assert meta["image_height"] == 480
        assert meta["schema_version"] == "2.0.0"
