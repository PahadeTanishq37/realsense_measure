"""
Unit tests for RealSense camera lifecycle hardening:
1. Disconnect vs Timeout detection in read() / read_with_colorized().
2. Failed stream negotiation handling in start() and clean attribute state.
3. Guards in methods depending on depth_scale and intrinsics_o3d.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from camera.realsense_capture import (
    CameraError,
    CameraNotStartedError,
    DEVICE_DISCONNECTED,
    DeviceDisconnectedError,
    RealSenseCamera,
    StreamNegotiationError,
)
from config import CameraConfig


def test_exception_hierarchy_and_sentinel():
    """Verify exceptions inherit from CameraError and RuntimeError, and sentinel exists."""
    assert issubclass(DeviceDisconnectedError, CameraError)
    assert issubclass(DeviceDisconnectedError, RuntimeError)
    assert issubclass(StreamNegotiationError, CameraError)
    assert issubclass(StreamNegotiationError, RuntimeError)
    assert issubclass(CameraNotStartedError, CameraError)
    assert issubclass(CameraNotStartedError, RuntimeError)
    assert DEVICE_DISCONNECTED is not None


def test_unstarted_guards():
    """Verify methods raise CameraNotStartedError if camera is not started."""
    cam = RealSenseCamera(CameraConfig())

    # read() guard
    with pytest.raises(CameraNotStartedError):
        cam.read()

    # read_with_colorized() guard
    with pytest.raises(CameraNotStartedError):
        cam.read_with_colorized()

    # to_point_cloud() guard
    fake_color = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_depth = np.zeros((480, 640), dtype=np.float32)
    with pytest.raises(CameraNotStartedError):
        cam.to_point_cloud(fake_color, fake_depth)

    # create_rgbd_frame() guard
    with pytest.raises(CameraNotStartedError):
        cam.create_rgbd_frame(fake_color, fake_depth)


def test_stream_negotiation_failure_raises_and_cleans_state():
    """Verify failed pipeline.start raises StreamNegotiationError and leaves attributes as None."""
    cam = RealSenseCamera(CameraConfig())
    cam.pipeline = MagicMock()
    cam.pipeline.start.side_effect = RuntimeError("Couldn't resolve requests: unsupported resolution 1280x720 @ 90fps")

    with pytest.raises(StreamNegotiationError) as exc_info:
        cam.start()

    assert "Failed to negotiate and start RealSense stream" in str(exc_info.value)
    assert cam._profile is None
    assert cam.depth_scale is None
    assert cam.intrinsics_o3d is None


def test_normal_timeout_returns_none():
    """Verify normal frame timeout on a connected device returns (None, None) for retry."""
    cam = RealSenseCamera(CameraConfig())
    cam._profile = MagicMock()
    cam.depth_scale = 0.001
    cam.pipeline = MagicMock()
    cam.pipeline.wait_for_frames.side_effect = RuntimeError("Frame didn't arrive within 2000")

    # Mock is_device_connected to True (device still plugged in)
    with patch.object(cam, "is_device_connected", return_value=True):
        color, depth = cam.read()
        assert color is None
        assert depth is None

        c, d, vis = cam.read_with_colorized()
        assert c is None
        assert d is None
        assert vis is None


def test_disconnect_message_raises_device_disconnected():
    """Verify explicit disconnect error messages raise DeviceDisconnectedError."""
    cam = RealSenseCamera(CameraConfig())
    cam._profile = MagicMock()
    cam.depth_scale = 0.001
    cam.pipeline = MagicMock()
    cam.pipeline.wait_for_frames.side_effect = RuntimeError("Notification: Device disconnected")

    with pytest.raises(DeviceDisconnectedError):
        cam.read()

    cam.pipeline.wait_for_frames.side_effect = RuntimeError("backend-error: failed to set power state")
    with pytest.raises(DeviceDisconnectedError):
        cam.read_with_colorized()


def test_timeout_with_missing_device_raises_device_disconnected():
    """Verify that if wait_for_frames times out and device is no longer present, it raises DeviceDisconnectedError."""
    cam = RealSenseCamera(CameraConfig())
    cam._profile = MagicMock()
    cam.depth_scale = 0.001
    cam.pipeline = MagicMock()
    cam.pipeline.wait_for_frames.side_effect = RuntimeError("Frame didn't arrive within 2000")

    with patch.object(cam, "is_device_connected", return_value=False):
        with pytest.raises(DeviceDisconnectedError):
            cam.read()


def test_stop_resets_attributes_and_is_idempotent():
    """Verify stop() resets calibration attributes to None and can be called repeatedly."""
    cam = RealSenseCamera(CameraConfig())
    cam._profile = MagicMock()
    cam.depth_scale = 0.001
    cam.intrinsics_o3d = MagicMock()
    mock_pipeline = MagicMock()
    cam.pipeline = mock_pipeline

    cam.stop()
    assert cam._profile is None
    assert cam.depth_scale is None
    assert cam.intrinsics_o3d is None
    assert mock_pipeline.stop.call_count == 1

    # Calling stop again does not re-invoke pipeline.stop or crash
    cam.stop()
    assert mock_pipeline.stop.call_count == 1
