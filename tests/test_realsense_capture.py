"""
Unit tests for camera/realsense_capture.py covering:
1. The decimation/shape-mismatch resize path in read() and read_with_colorized().
2. The post-processing disparity-filter chain in _filter_depth().
3. Disconnect-vs-timeout handling added in fix #2 (mock pyrealsense2's pipeline/frames
   to simulate both a timeout and a disconnect exception, and assert the camera
   responds differently to each).
"""

from unittest.mock import MagicMock, patch
import cv2
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
from config import CameraConfig, DepthVisConfig


# ===========================================================================
# Helper Fixtures & Builders
# ===========================================================================

@pytest.fixture
def base_camera():
    """Create a RealSenseCamera instance with mocked pipeline and started state."""
    cfg = CameraConfig(
        width=848,
        height=480,
        fps=30,
        use_decimation=True,
        decimation_magnitude=2,
        use_spatial_filter=True,
        use_temporal_filter=True,
        use_hole_filling=True,
    )
    cam = RealSenseCamera(cfg)
    cam._profile = MagicMock()
    cam.depth_scale = 0.001  # 1mm per unit
    cam.pipeline = MagicMock()
    cam.align = MagicMock()
    return cam


def build_mock_frameset(
    color_shape: tuple[int, int, int] = (480, 848, 3),
    depth_shape: tuple[int, int] = (240, 424),
    depth_val: int = 1500,
):
    """
    Build a mock aligned frameset returning specific numpy buffers for
    color and depth frames.
    """
    mock_frames = MagicMock(name="composite_frames")
    mock_aligned = MagicMock(name="aligned_frames")

    mock_depth_frame = MagicMock(name="depth_frame")
    depth_array = np.full(depth_shape, depth_val, dtype=np.uint16)
    mock_depth_frame.get_data.return_value = depth_array

    mock_color_frame = MagicMock(name="color_frame")
    color_array = np.zeros(color_shape, dtype=np.uint8)
    # Paint top-left pixel distinctively to verify data integrity
    color_array[0, 0] = [10, 20, 30]
    mock_color_frame.get_data.return_value = color_array

    mock_aligned.get_depth_frame.return_value = mock_depth_frame
    mock_aligned.get_color_frame.return_value = mock_color_frame

    return mock_frames, mock_aligned, mock_depth_frame, mock_color_frame


# ===========================================================================
# 1. Decimation / Shape-Mismatch Resize Path Tests
# ===========================================================================

class TestDecimationShapeMismatchResize:
    """Tests covering the decimation / shape-mismatch resize path."""

    def test_read_resizes_decimated_depth_to_color_shape(self, base_camera):
        """
        When decimation shrinks depth resolution (e.g. 240x424 vs 480x848 color),
        read() must resize depth_m to match color_bgr's spatial dimensions using
        nearest-neighbor interpolation.
        """
        cam = base_camera
        mock_frames, mock_aligned, mock_depth, mock_color = build_mock_frameset(
            color_shape=(480, 848, 3),
            depth_shape=(240, 424),
            depth_val=1250,
        )

        cam.pipeline.wait_for_frames.return_value = mock_frames
        cam.align.process.return_value = mock_aligned
        # Bypass filter chain to isolate resize logic on mock_depth
        cam._filter_depth = MagicMock(return_value=mock_depth)

        color_bgr, depth_m = cam.read()

        assert color_bgr is not None
        assert depth_m is not None
        assert color_bgr.shape == (480, 848, 3)
        # Depth must be resized up to match color resolution
        assert depth_m.shape == (480, 848)
        assert depth_m.dtype == np.float32
        # Converted via depth_scale: 1250 * 0.001 = 1.25m
        np.testing.assert_allclose(depth_m, 1.25, rtol=1e-5)

    def test_read_nearest_neighbor_interpolation_fidelity(self, base_camera):
        """
        Verify that decimation upsampling uses INTER_NEAREST without blurring
        sharp step boundaries.
        """
        cam = base_camera
        # 2x2 depth array with 4 distinct quadrant values
        depth_data = np.array([
            [1000, 2000],
            [3000, 4000],
        ], dtype=np.uint16)

        mock_depth = MagicMock(name="depth_frame")
        mock_depth.get_data.return_value = depth_data

        mock_color = MagicMock(name="color_frame")
        # 4x4 color array
        mock_color.get_data.return_value = np.zeros((4, 4, 3), dtype=np.uint8)

        mock_aligned = MagicMock()
        mock_aligned.get_depth_frame.return_value = mock_depth
        mock_aligned.get_color_frame.return_value = mock_color

        cam.pipeline.wait_for_frames.return_value = MagicMock()
        cam.align.process.return_value = mock_aligned
        cam._filter_depth = MagicMock(return_value=mock_depth)

        _, depth_m = cam.read()

        assert depth_m.shape == (4, 4)
        # Each quadrant should be expanded exactly 2x2
        expected = np.array([
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ], dtype=np.float32)
        np.testing.assert_allclose(depth_m, expected, rtol=1e-5)

    def test_read_with_matching_shapes_does_not_resize(self, base_camera):
        """
        When depth and color shapes already match (no decimation), arrays are
        returned directly without invoking cv2.resize.
        """
        cam = base_camera
        mock_frames, mock_aligned, mock_depth, mock_color = build_mock_frameset(
            color_shape=(480, 848, 3),
            depth_shape=(480, 848),
            depth_val=800,
        )

        cam.pipeline.wait_for_frames.return_value = mock_frames
        cam.align.process.return_value = mock_aligned
        cam._filter_depth = MagicMock(return_value=mock_depth)

        with patch("cv2.resize") as mock_resize:
            color_bgr, depth_m = cam.read()
            mock_resize.assert_not_called()

        assert color_bgr.shape == (480, 848, 3)
        assert depth_m.shape == (480, 848)
        assert depth_m[0, 0] == pytest.approx(0.8)

    def test_read_with_colorized_resizes_both_depth_and_colorized_images(self, base_camera):
        """
        In read_with_colorized(), both raw depth_m AND depth_color_bgr preview
        must be resized up to match color_bgr shape when decimation is active.
        """
        cam = base_camera
        mock_frames, mock_aligned, mock_depth, mock_color = build_mock_frameset(
            color_shape=(480, 848, 3),
            depth_shape=(240, 424),
            depth_val=2000,
        )

        cam.pipeline.wait_for_frames.return_value = mock_frames
        cam.align.process.return_value = mock_aligned
        cam._filter_depth = MagicMock(return_value=mock_depth)

        # Mock colorize_depth returning preview at the decimated resolution (240x424x3)
        decimated_preview = np.ones((240, 424, 3), dtype=np.uint8) * 128
        cam.colorize_depth = MagicMock(return_value=decimated_preview)

        color_bgr, depth_m, depth_color_bgr = cam.read_with_colorized()

        assert color_bgr.shape == (480, 848, 3)
        assert depth_m.shape == (480, 848)
        assert depth_color_bgr.shape == (480, 848, 3)
        cam.colorize_depth.assert_called_once_with(mock_depth)


# ===========================================================================
# 2. Disparity-Filter Chain Tests
# ===========================================================================

class TestDisparityFilterChain:
    """Tests covering the post-processing filter chain in _filter_depth()."""

    def test_filter_chain_full_order(self):
        """
        Verify the exact filter execution order:
          decimation -> to_disparity -> spatial -> temporal -> to_depth -> hole_filling
        """
        cfg = CameraConfig(
            use_decimation=True,
            use_spatial_filter=True,
            use_temporal_filter=True,
            use_hole_filling=True,
        )
        cam = RealSenseCamera(cfg)

        f_raw = MagicMock(name="f_raw")
        f_dec = MagicMock(name="f_dec")
        f_disp = MagicMock(name="f_disp")
        f_spat = MagicMock(name="f_spat")
        f_temp = MagicMock(name="f_temp")
        f_depth = MagicMock(name="f_depth")
        f_hole = MagicMock(name="f_hole")

        cam._decimation = MagicMock()
        cam._decimation.process.return_value = f_dec
        cam._to_disparity = MagicMock()
        cam._to_disparity.process.return_value = f_disp
        cam._spatial = MagicMock()
        cam._spatial.process.return_value = f_spat
        cam._temporal = MagicMock()
        cam._temporal.process.return_value = f_temp
        cam._to_depth = MagicMock()
        cam._to_depth.process.return_value = f_depth
        cam._hole_filling = MagicMock()
        cam._hole_filling.process.return_value = f_hole

        result = cam._filter_depth(f_raw)

        # Output must be the final filter product
        assert result is f_hole

        # Sequential processing assertions
        cam._decimation.process.assert_called_once_with(f_raw)
        cam._to_disparity.process.assert_called_once_with(f_dec)
        cam._spatial.process.assert_called_once_with(f_disp)
        cam._temporal.process.assert_called_once_with(f_spat)
        cam._to_depth.process.assert_called_once_with(f_temp)
        cam._hole_filling.process.assert_called_once_with(f_depth)

    def test_filter_chain_bypasses_decimation_when_disabled(self):
        """When use_decimation=False, raw frame passes straight to to_disparity."""
        cfg = CameraConfig(
            use_decimation=False,
            use_spatial_filter=True,
            use_temporal_filter=True,
            use_hole_filling=True,
        )
        cam = RealSenseCamera(cfg)

        f_raw = MagicMock(name="f_raw")
        f_disp = MagicMock(name="f_disp")

        cam._decimation = MagicMock()
        cam._to_disparity = MagicMock()
        cam._to_disparity.process.return_value = f_disp
        cam._spatial = MagicMock()
        cam._spatial.process.return_value = f_disp
        cam._temporal = MagicMock()
        cam._temporal.process.return_value = f_disp
        cam._to_depth = MagicMock()
        cam._to_depth.process.return_value = f_raw
        cam._hole_filling = MagicMock()
        cam._hole_filling.process.return_value = f_raw

        cam._filter_depth(f_raw)

        cam._decimation.process.assert_not_called()
        cam._to_disparity.process.assert_called_once_with(f_raw)

    def test_filter_chain_disparity_brackets_spatial_and_temporal(self):
        """
        Disparity transform (depth -> disparity and disparity -> depth) always
        brackets spatial/temporal filtering even if one or both are disabled.
        """
        cfg = CameraConfig(
            use_decimation=False,
            use_spatial_filter=False,
            use_temporal_filter=False,
            use_hole_filling=False,
        )
        cam = RealSenseCamera(cfg)

        f_raw = MagicMock(name="f_raw")
        f_disp = MagicMock(name="f_disp")
        f_depth = MagicMock(name="f_depth")

        cam._to_disparity = MagicMock()
        cam._to_disparity.process.return_value = f_disp
        cam._spatial = MagicMock()
        cam._temporal = MagicMock()
        cam._to_depth = MagicMock()
        cam._to_depth.process.return_value = f_depth
        cam._hole_filling = MagicMock()

        result = cam._filter_depth(f_raw)

        assert result is f_depth
        cam._to_disparity.process.assert_called_once_with(f_raw)
        cam._spatial.process.assert_not_called()
        cam._temporal.process.assert_not_called()
        # to_depth received the disparity frame untouched
        cam._to_depth.process.assert_called_once_with(f_disp)
        cam._hole_filling.process.assert_not_called()

    def test_filter_chain_selective_spatial_only(self):
        """When only spatial is enabled, temporal is bypassed inside disparity bracket."""
        cfg = CameraConfig(
            use_decimation=False,
            use_spatial_filter=True,
            use_temporal_filter=False,
            use_hole_filling=False,
        )
        cam = RealSenseCamera(cfg)

        f_raw = MagicMock(name="f_raw")
        f_disp = MagicMock(name="f_disp")
        f_spat = MagicMock(name="f_spat")
        f_depth = MagicMock(name="f_depth")

        cam._to_disparity = MagicMock()
        cam._to_disparity.process.return_value = f_disp
        cam._spatial = MagicMock()
        cam._spatial.process.return_value = f_spat
        cam._temporal = MagicMock()
        cam._to_depth = MagicMock()
        cam._to_depth.process.return_value = f_depth

        result = cam._filter_depth(f_raw)

        assert result is f_depth
        cam._spatial.process.assert_called_once_with(f_disp)
        cam._temporal.process.assert_not_called()
        cam._to_depth.process.assert_called_once_with(f_spat)


# ===========================================================================
# 3. Disconnect vs Timeout Handling Tests (Fix #2)
# ===========================================================================

class TestDisconnectVsTimeoutHandling:
    """
    Tests covering fix #2: distinguishing normal frame timeouts from genuine
    device disconnects.
    """

    def test_timeout_and_disconnect_produce_different_responses(self, base_camera):
        """
        CORE CONTRAST ASSERTION:
        Verify the camera responds differently to a timeout vs a disconnect exception.
        - Timeout on connected device: Gracefully returns (None, None) without raising.
        - Disconnect exception: Raises DeviceDisconnectedError.
        """
        cam = base_camera

        # 1. Simulate TIMEOUT on connected device
        cam.pipeline.wait_for_frames.side_effect = RuntimeError(
            "Frame didn't arrive within 2000"
        )
        with patch.object(cam, "is_device_connected", return_value=True):
            color_res, depth_res = cam.read()
            # Assert graceful dropped-frame response (no exception)
            assert color_res is None
            assert depth_res is None

        # 2. Simulate DISCONNECT exception
        cam.pipeline.wait_for_frames.side_effect = RuntimeError(
            "Notification: Device disconnected"
        )
        with patch.object(cam, "is_device_connected", return_value=False):
            # Assert fatal exception response
            with pytest.raises(DeviceDisconnectedError) as exc_info:
                cam.read()
            assert "RealSense camera disconnected" in str(exc_info.value)

    def test_read_with_colorized_timeout_vs_disconnect(self, base_camera):
        """
        Verify read_with_colorized() similarly responds differently to timeout vs disconnect:
        - Timeout on connected device: Returns (None, None, None).
        - Disconnect: Raises DeviceDisconnectedError.
        """
        cam = base_camera

        # Timeout scenario
        cam.pipeline.wait_for_frames.side_effect = RuntimeError("wait_for_frames timed out")
        with patch.object(cam, "is_device_connected", return_value=True):
            c, d, vis = cam.read_with_colorized()
            assert c is None
            assert d is None
            assert vis is None

        # Disconnect scenario
        cam.pipeline.wait_for_frames.side_effect = RuntimeError("No device connected")
        with pytest.raises(DeviceDisconnectedError):
            cam.read_with_colorized()

    def test_timeout_with_disconnected_device_detected_as_disconnect(self, base_camera):
        """
        When wait_for_frames times out because the USB cable was pulled,
        is_device_connected() returns False, and camera escalates to DeviceDisconnectedError.
        """
        cam = base_camera
        cam.pipeline.wait_for_frames.side_effect = RuntimeError(
            "Frame didn't arrive within 5000"
        )

        with patch.object(cam, "is_device_connected", return_value=False):
            with pytest.raises(DeviceDisconnectedError) as exc_info:
                cam.read()
            assert "timeout with no device connected" in str(exc_info.value)

    @pytest.mark.parametrize("error_msg", [
        "Device disconnected",
        "device not connected",
        "no device",
        "backend-error: failed to set power",
        "hardware error: ASIC failure",
        "resource busy",
        "resource unavailable",
        "input/output error",
        "failed to set power",
        "access denied",
        "device removed",
        "broken pipe",
    ])
    def test_hardware_and_disconnect_exceptions_raise_device_disconnected(
        self, base_camera, error_msg
    ):
        """
        Any explicit RealSense SDK hardware/disconnection error strings must
        immediately raise DeviceDisconnectedError.
        """
        cam = base_camera
        cam.pipeline.wait_for_frames.side_effect = RuntimeError(f"RealSense error: {error_msg}")

        with pytest.raises(DeviceDisconnectedError) as exc_info:
            cam._wait_for_frames()
        assert "RealSense camera disconnected" in str(exc_info.value)

    def test_wait_for_frames_unstarted_raises_camera_not_started(self):
        """_wait_for_frames() raises CameraNotStartedError if camera was never started."""
        cam = RealSenseCamera(CameraConfig())
        cam._profile = None
        with pytest.raises(CameraNotStartedError):
            cam._wait_for_frames()

    def test_is_device_connected_detection_logic(self):
        """Verify is_device_connected checks profile device first, then context."""
        cam = RealSenseCamera(CameraConfig())

        # 1. Profile device query succeeds
        mock_profile = MagicMock()
        mock_dev = MagicMock()
        mock_dev.get_info.return_value = "Intel RealSense D455"
        mock_profile.get_device.return_value = mock_dev
        cam._profile = mock_profile

        assert cam.is_device_connected() is True

        # 2. Profile device query fails, but context has devices
        mock_dev.get_info.side_effect = RuntimeError("device communication lost")
        with patch("pyrealsense2.context") as mock_ctx_cls:
            mock_ctx = MagicMock()
            mock_ctx.query_devices.return_value = [MagicMock()]
            mock_ctx_cls.return_value = mock_ctx

            assert cam.is_device_connected() is True

        # 3. Profile device query fails and context has no devices
        with patch("pyrealsense2.context") as mock_ctx_cls:
            mock_ctx = MagicMock()
            mock_ctx.query_devices.return_value = []
            mock_ctx_cls.return_value = mock_ctx

            assert cam.is_device_connected() is False
