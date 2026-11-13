"""
tests/fullbody/fake_camera.py

Numpy-only synthetic camera generating humanoid-like depth blobs.
Implements the CaptureCamera protocol with scriptable faults:
- swaying (lateral motion > 10 mm)
- depth_dropout (valid fraction < 0.30)
- head_clipped (body pixels touching top image boundary)
- too_far (subject distance > 1.5 m)
- too_close (subject distance < 1.1 m)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


class FakeCamera:
    """Synthetic camera implementing CaptureCamera protocol."""

    def __init__(
        self,
        width: int = 480,
        height: int = 848,
        depth_scale: float = 0.001,
        depth_units: str = "sensor_units",
        base_depth_m: float = 1.25,
        fps: float = 30.0,
    ):
        self.width = width
        self.height = height
        self._depth_scale = float(depth_scale)
        self._depth_units = str(depth_units)
        self.base_depth_m = float(base_depth_m)
        self.fps = float(fps)

        self.is_running = False
        self.frame_id = 0
        self.fault: Optional[str] = None

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    @property
    def intrinsics(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": 425.0,
            "fy": 425.0,
            "cx": self.width / 2.0,
            "cy": self.height / 2.0,
            "is_post_rotation": True,
            "is_post_decimation": False,
        }

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def depth_units(self) -> str:
        return self._depth_units

    def device_info(self) -> Dict[str, Any]:
        return {
            "name": "Synthetic FakeCamera D455f",
            "serial_number": "SIM_001",
            "firmware_version": "9.9.9",
        }

    def set_fault(self, fault: Optional[str]) -> None:
        self.fault = fault

    def read_raw_filtered(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        if not self.is_running:
            return None, None, None, 0.0

        ts = float(self.frame_id) / self.fps
        self.frame_id += 1

        h, w = self.height, self.width
        # Grid coordinates
        y_coords, x_coords = np.ogrid[:h, :w]

        # Base body center
        cx = w / 2.0
        cy = h / 2.0
        current_depth_m = self.base_depth_m

        # Handle faults
        if self.fault == "too_far":
            current_depth_m = 2.50
        elif self.fault == "too_close":
            current_depth_m = 0.70

        if self.fault == "swaying":
            # Significant sway: +/- 40 pixels lateral shift oscillating at 1 Hz
            cx += 40.0 * math.sin(2.0 * math.pi * 1.0 * ts)

        # Body ellipsoid: torso + head + limbs
        # Torso center
        torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - cy) / (h * 0.35)) ** 2) <= 1.0

        # Head center
        head_cy = cy - h * 0.38
        if self.fault == "head_clipped":
            # Shift body and head up so head hits row 0
            head_cy = 10.0
            torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - (cy - h * 0.25)) / (h * 0.35)) ** 2) <= 1.0

        head_mask = (((x_coords - cx) / (w * 0.12)) ** 2 + ((y_coords - head_cy) / (h * 0.10)) ** 2) <= 1.0

        body_mask = torso_mask | head_mask

        # Depth buffer in metres
        depth_m = np.zeros((h, w), dtype=np.float32)
        depth_m[body_mask] = current_depth_m

        if self.fault == "depth_dropout":
            # Severe dropout: zero out 85% of pixels
            random_drop = np.random.rand(h, w) > 0.15
            depth_m[random_drop] = 0.0

        # Convert to storage format
        if self.depth_units == "sensor_units":
            raw_depth = (depth_m / self.depth_scale).astype(np.uint16)
        else:
            raw_depth = depth_m.astype(np.float32)

        # Color buffer
        color_bgr = np.full((h, w, 3), 40, dtype=np.uint8)
        color_bgr[body_mask] = [180, 150, 120]

        filtered_depth_m = depth_m.copy()

        return color_bgr, raw_depth, filtered_depth_m, ts

