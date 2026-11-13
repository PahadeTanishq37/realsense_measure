"""
realsense_measure/fullbody/capture_camera.py

RealSense camera adapter and configuration for full-body capture.
Implements the CaptureCamera protocol.
Guaranteed to import without loading pyrealsense2 or open3d native libraries.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from realsense_measure.config import CameraConfig
from realsense_measure.fullbody.capture_controller import CaptureCamera

logger = logging.getLogger(__name__)


def full_body_camera_config(
    portrait_rotation: str,
    resolution: str = "848x480",
    fps: int = 30,
) -> CameraConfig:
    """
    Construct a CameraConfig tailored for full-body 8-stop turntable capture.

    Design rationale (docs/STAGE27_CAPTURE_DESIGN.md item 2):
    - depth_max_m = 3.0: Subject stands on the turntable at ~1.1-1.5 m (nominal ~1.3 m).
      Setting depth_max_m to 3.0 m retains full subject silhouette, extended arms, and surrounding
      turntable/floor context without clipping distant limbs or ground plane references.
    - depth_min_m = 0.3: At 848x480, the RealSense D455 physical minimum range is ~0.35 m.
      Setting depth_min_m to 0.3 m prevents near-field clipping if a subject leans toward camera.
    - use_decimation = False: Decimation reduces spatial resolution by half (424x240) and degrades
      fine anatomical contours and circumference estimation. Keeping decimation OFF preserves
      full native resolution (848x480) and 1:1 pixel correspondence.
    - rotate_90 = portrait_rotation ('cw' or 'ccw'): Required for portrait tripod mounting to
      orient the camera's wide FOV vertically, covering a full standing humanoid from crown to floor.
    - fps = 30: Standard frame rate providing good temporal integration and SNR during burst capture.
    """
    rot = str(portrait_rotation).lower()
    if rot not in ("cw", "ccw"):
        raise ValueError(f"Invalid portrait_rotation '{portrait_rotation}'. Must be 'cw' or 'ccw'.")

    # Parse resolution string (e.g. "848x480")
    if "x" in resolution:
        w_str, h_str = resolution.lower().split("x")
        w, h = int(w_str), int(h_str)
    else:
        w, h = 848, 480

    cfg = CameraConfig(
        width=w,
        height=h,
        fps=fps,
        depth_min_m=0.30,
        depth_max_m=3.00,
        use_decimation=False,  # Strictly OFF to preserve full resolution geometry
        rotate_90=rot,
    )
    # Ensure preview visualizer covers full range
    cfg.depth_vis.visual_min_m = 0.30
    cfg.depth_vis.visual_max_m = 3.00
    return cfg


class RealSenseCaptureAdapter:
    """
    Adapter implementing the CaptureCamera protocol using RealSenseCamera.
    Loads pyrealsense2 / RealSenseCamera lazily on start() to prevent import-time side effects.
    """

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        portrait_rotation: str = "cw",
        warmup_frames: int = 30,
        max_dropped_retries: int = 5,
        camera_factory: Optional[Callable[..., Any]] = None,
    ):
        self.camera_config = camera_config or full_body_camera_config(portrait_rotation=portrait_rotation)
        self.portrait_rotation = portrait_rotation
        self.warmup_frames = int(warmup_frames)
        self.max_dropped_retries = int(max_dropped_retries)
        self._camera_factory = camera_factory

        self._camera: Any = None
        self._is_started = False
        self._intrinsics: Dict[str, Any] = {}
        self._device_info: Dict[str, Any] = {}
        self._stream_profile_str: str = ""

    def start(self) -> None:
        """Start the camera pipeline, drop warm-up frames, and cache intrinsics."""
        if self._is_started:
            return

        # Refuse to start if decimation is active and intrinsics are not scaled
        if self.camera_config.use_decimation:
            raise RuntimeError(
                "RealSenseCaptureAdapter: decimation is active. Full-body capture requires decimation OFF."
            )

        # Lazy instantiation of RealSenseCamera
        if self._camera_factory is not None:
            self._camera = self._camera_factory(self.camera_config)
        else:
            try:
                from realsense_measure.camera.realsense_capture import RealSenseCamera
                self._camera = RealSenseCamera(cfg=self.camera_config)
            except Exception as exc:
                raise RuntimeError(f"Failed to import/instantiate RealSenseCamera: {exc}") from exc

        try:
            self._camera.start()
            self._is_started = True

            # Query device info
            try:
                info = self._camera.get_device_info()
            except Exception:
                info = {}

            self._device_info = {
                "name": info.get("name", "Intel RealSense D455"),
                "serial_number": info.get("serial_number", "UNKNOWN"),
                "firmware_version": info.get("firmware_version", "UNKNOWN"),
                "usb_type": info.get("usb_type", "3.x"),
            }

            # Cache post-rotation intrinsics
            intr_o3d = getattr(self._camera, "intrinsics_o3d", None)
            has_valid_intr = False
            if intr_o3d is not None and hasattr(intr_o3d, "get_focal_length"):
                try:
                    fx, fy = intr_o3d.get_focal_length()
                    cx, cy = intr_o3d.get_principal_point()
                    w, h = int(intr_o3d.width), int(intr_o3d.height)
                    has_valid_intr = True
                except Exception:
                    has_valid_intr = False

            if not has_valid_intr:
                # Fallback to configured rotated dimensions
                if self.camera_config.rotate_90 in ("cw", "ccw"):
                    w, h = self.camera_config.height, self.camera_config.width
                else:
                    w, h = self.camera_config.width, self.camera_config.height
                fx = fy = float(max(w, h))
                cx, cy = w / 2.0, h / 2.0

            self._intrinsics = {
                "width": int(w),
                "height": int(h),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "is_post_rotation": bool(self.camera_config.rotate_90 in ("cw", "ccw")),
                "is_post_decimation": False,
            }

            self._stream_profile_str = f"{self.camera_config.width}x{self.camera_config.height}@{self.camera_config.fps}fps"

            # Discard warm-up frames for auto-exposure, gain, and temporal stabilization
            logger.info(f"Discarding {self.warmup_frames} warm-up frames ...")
            for _ in range(self.warmup_frames):
                self._camera.read_with_raw_and_colorized()

        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Safely stop camera in a finally block."""
        try:
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception as exc:
                    logger.warning(f"Error while stopping RealSenseCamera: {exc}")
        finally:
            self._camera = None
            self._is_started = False

    def __enter__(self) -> "RealSenseCaptureAdapter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @property
    def intrinsics(self) -> Dict[str, Any]:
        return dict(self._intrinsics)

    @property
    def depth_scale(self) -> float:
        if self._camera is not None and getattr(self._camera, "depth_scale", None) is not None:
            return float(self._camera.depth_scale)
        return 0.001

    @property
    def depth_units(self) -> str:
        # Raw depth is stored in native uint16 sensor units
        return "sensor_units"

    def device_info(self) -> Dict[str, Any]:
        info = dict(self._device_info)
        info["depth_scale"] = self.depth_scale
        info["stream_profile"] = self._stream_profile_str
        info["intrinsics"] = self.intrinsics
        return info

    def read_raw_filtered(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Capture one frame pair.
        Returns (color_bgr, raw_depth_z16, filtered_depth_m, timestamp_s).
        Retries dropped frames up to max_dropped_retries.
        Raises RuntimeError on device disconnect.
        """
        if not self._is_started or self._camera is None:
            raise RuntimeError("RealSenseCaptureAdapter is not started. Call start() before reading.")

        for attempt in range(self.max_dropped_retries + 1):
            try:
                (
                    color_bgr,
                    depth_filtered_m,
                    depth_raw_m,
                    raw_z16,
                    depth_color_bgr,
                    frame_meta,
                ) = self._camera.read_with_raw_and_colorized()
            except Exception as exc:
                # Disconnect / fatal hardware error detected
                logger.error(f"RealSense hardware read failure: {exc}")
                raise RuntimeError(f"RealSense camera disconnected or hardware failed: {exc}") from exc

            if color_bgr is not None and raw_z16 is not None:
                ts_ms = frame_meta.get("timestamp_ms", 0.0) if frame_meta else 0.0
                ts_s = float(ts_ms) / 1000.0 if ts_ms > 0 else time.time()
                return color_bgr, raw_z16, depth_filtered_m, ts_s

            logger.debug(f"Dropped frame encountered on attempt {attempt + 1}/{self.max_dropped_retries + 1}")

        # All retries returned None (normal dropped frame)
        return None, None, None, time.time()

