"""
tests/fullbody/test_capture_controller.py

Unit tests for Stage 27 CaptureController:
1. Clean 8-stop run reaches DONE and writes valid stops_v2 session that loads via _load_stops_v2.
2. Swaying subject rejected on stop 3, retried, and accepted.
3. Head clipped rejected with reason mentioning "head".
4. Subject too far rejected before burst begins.
5. Max retakes exceeded -> ABORT, session not finalized.
6. CW vs CCW nominal angles stored correctly.
7. Monkeypatch time.sleep and builtins.input: controller never calls either.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pytest

from realsense_measure.fullbody.capture_controller import (
    CaptureController,
    CaptureThresholds,
    ControllerState,
)
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.io import _load_stops_v2, load_capture_session
from realsense_measure.fullbody.session_format import SessionWriter
from tests.fullbody.fake_camera import FakeCamera


def test_clean_8_stop_run_reaches_done_and_loads():
    """Clean 8-stop run reaches DONE, writes valid stops_v2 session that loads via _load_stops_v2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "clean_8_stop"
        cam = FakeCamera(width=240, height=320, depth_scale=0.001, depth_units="sensor_units")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_CLEAN",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=5,
            stillness_frames=3,
            max_retakes=3,
            turn_prompt_timeout_s=0.1,
            position_check_timeout_s=1.0,
            burst_timeout_s=1.0,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        now = 0.0
        max_steps = 2000
        for _ in range(max_steps):
            state = controller.step(now)
            now += 0.033
            if state == ControllerState.DONE:
                break

        assert controller.state == ControllerState.DONE
        assert (root / "meta.json").exists()

        cfg = FullBodyConfig()
        stops = load_capture_session(root, cfg)
        assert len(stops) == 8
        for i, s in enumerate(stops):
            assert s.stop_id == i
            assert s.depth.shape == (320, 240)


def test_swaying_subject_rejected_then_accepted():
    """Swaying subject rejected on attempt 0, retried without sway, and accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "sway_retry"
        cam = FakeCamera(width=240, height=320, depth_scale=0.001, depth_units="sensor_units")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_SWAY",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=5,
            stillness_frames=3,
            max_retakes=3,
            motion_limit_mm=5.0,
            turn_prompt_timeout_s=0.1,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        # Trigger sway during first burst
        sway_injected = False
        sway_cleared = False
        now = 0.0

        for _ in range(1000):
            if controller.state == ControllerState.CAPTURE_BURST and not sway_injected:
                cam.set_fault("swaying")
                sway_injected = True

            if controller.state == ControllerState.RETAKE and not sway_cleared:
                # Assert attempt 0 was rejected due to motion
                stop_json = root / "stop_00" / "stop.json"
                with open(stop_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                assert len(meta["attempts"]) == 1
                assert meta["attempts"][0]["status"] == "REJECTED"
                assert "motion" in meta["attempts"][0]["reason"].lower()

                # Clear fault so attempt 1 succeeds
                cam.set_fault(None)
                sway_cleared = True

            state = controller.step(now)
            now += 0.033
            if state == ControllerState.DONE:
                break

        assert controller.state == ControllerState.DONE
        # Check stop.json has 2 attempts, accepted attempt is 1
        stop_json = root / "stop_00" / "stop.json"
        with open(stop_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert len(meta["attempts"]) == 2
        assert meta["accepted_attempt"] == 1
        assert meta["attempts"][1]["status"] == "ACCEPTED"


def test_head_clipped_rejected_with_reason_mentioning_head():
    """Head clipped rejected with failure reason mentioning 'head'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "head_clipped"
        cam = FakeCamera(width=240, height=320, depth_scale=0.001, depth_units="sensor_units")
        cam.set_fault("head_clipped")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_HEAD",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=5,
            stillness_frames=3,
            max_retakes=1,
            turn_prompt_timeout_s=0.1,
            position_check_timeout_s=0.2,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        # 1. In position check, it stays blocked and does not proceed to burst
        now = 0.0
        for _ in range(30):
            controller.step(now)
            now += 0.033
            assert controller.state in (ControllerState.WAIT_FOR_SUBJECT, ControllerState.POSITION_CHECK)

        # 2. If burst is evaluated with head clipped, quality_fn flags "head"
        h, w = 320, 240
        burst_c = [np.full((h, w, 3), 120, dtype=np.uint8)] * 5
        burst_d = [np.full((h, w), 1.25, dtype=np.float32)] * 5
        # Set top row body pixels to trigger head clip
        for d in burst_d:
            d[0:5, :] = 1.25

        q_report = controller.quality_fn(burst_c, burst_d, 1.0, "m", thresholds)
        assert q_report["is_flagged"] is True
        assert any("head" in r.lower() for r in q_report["reasons"])


def test_subject_too_far_rejected_before_burst_begins():
    """Subject too far is rejected before burst begins (blocked in POSITION_CHECK/WAIT_FOR_SUBJECT)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "too_far"
        cam = FakeCamera(width=240, height=320, depth_scale=0.001, depth_units="sensor_units")
        cam.set_fault("too_far")  # Distance = 2.50 m (threshold is max 1.50 m)
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_FAR",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=5,
            stillness_frames=3,
            position_check_timeout_s=0.2,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        now = 0.0
        for _ in range(50):
            state = controller.step(now)
            now += 0.033
            # Must never proceed to TURN_PROMPT, WAIT_STILL, or CAPTURE_BURST
            assert state in (ControllerState.WAIT_FOR_SUBJECT, ControllerState.POSITION_CHECK)


def test_max_retakes_exceeded_aborts_and_not_finalized():
    """When retakes exceed max_retakes, controller transitions to ABORT and session is not finalized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "max_retakes_abort"
        cam = FakeCamera(width=240, height=320, depth_scale=0.001, depth_units="sensor_units")
        cam.set_fault("swaying")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_ABORT",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=4,
            stillness_frames=2,
            max_retakes=2,  # Allowed attempts: attempt 0 (first), 1 (retake 1), 2 (retake 2) -> then abort
            motion_limit_mm=5.0,
            turn_prompt_timeout_s=0.05,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        now = 0.0
        for _ in range(500):
            state = controller.step(now)
            now += 0.033
            if state == ControllerState.ABORT:
                break

        assert controller.state == ControllerState.ABORT
        assert not (root / "meta.json").exists()

        stop_json = root / "stop_00" / "stop.json"
        with open(stop_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["accepted_attempt"] is None
        assert len(meta["attempts"]) == 3  # attempt 0, 1, 2


def test_cw_vs_ccw_nominal_angles_stored_correctly():
    """Verify cw and ccw nominal angles are recorded in stop.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cam = FakeCamera(width=100, height=100, depth_scale=0.001, depth_units="sensor_units")
        cam.start()

        for turn_dir in ("cw", "ccw"):
            root = Path(tmpdir) / f"session_{turn_dir}"
            writer = SessionWriter(
                root=root,
                subject_id=f"SUBJ_{turn_dir.upper()}",
                n_stops=4,
                turn_direction=turn_dir,
                camera_portrait_rotation="cw",
                intrinsics=cam.intrinsics,
                depth_scale=cam.depth_scale,
                depth_units=cam.depth_units,
            )
            # Check initialized stop.json angles
            expected_nominal = [0.0, 90.0, 180.0, 270.0]
            for i, exp_ang in enumerate(expected_nominal):
                stop_json = root / f"stop_{i:02d}" / "stop.json"
                with open(stop_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                assert data["nominal_angle_deg"] == pytest.approx(exp_ang)


def test_controller_never_calls_sleep_or_input(monkeypatch):
    """Monkeypatch time.sleep and builtins.input: assert controller never calls either."""
    def forbidden_sleep(*args, **kwargs):
        raise AssertionError("Controller must never call time.sleep()!")

    def forbidden_input(*args, **kwargs):
        raise AssertionError("Controller must never call input()!")

    monkeypatch.setattr(time, "sleep", forbidden_sleep)
    monkeypatch.setattr(builtins, "input", forbidden_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "no_sleep_test"
        cam = FakeCamera(width=100, height=100, depth_scale=0.001, depth_units="sensor_units")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="SUBJ_NOSLEEP",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="cw",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        thresholds = CaptureThresholds(
            burst_frames=3,
            stillness_frames=2,
            turn_prompt_timeout_s=0.05,
        )

        controller = CaptureController(camera=cam, writer=writer, thresholds=thresholds)

        now = 0.0
        for _ in range(100):
            controller.step(now)
            now += 0.033
            if controller.state == ControllerState.DONE:
                break

        assert controller.state == ControllerState.DONE


def test_burst_with_fewer_than_burst_frames_rejected():
    """Verify EVALUATE rejects a burst with fewer than burst_frames frames with explicit reason."""
    th = CaptureThresholds(burst_frames=10)
    c_frames = [np.full((100, 100, 3), 100, dtype=np.uint8)] * 3
    d_frames = [np.full((100, 100), 1.25, dtype=np.float32)] * 3

    from realsense_measure.fullbody.capture_controller import default_quality_fn

    report = default_quality_fn(c_frames, d_frames, depth_scale=1.0, depth_units="m", thresholds=th)
    assert report["is_flagged"] is True
    assert any("burst_frames" in r for r in report["reasons"])
