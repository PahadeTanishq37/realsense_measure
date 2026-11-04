#!/usr/bin/env python3
"""
tools/run_sweet_spot_experiment.py
==================================
Standalone trial runner for controlled Intel RealSense D455f sweet-spot experiments.

Executes calibrated capture sweeps across distance, elevation, azimuth, pose,
and camera orientation (portrait vs landscape).

Features:
- Dual-stream depth capture: synchronous raw depth and production-filtered depth
- Three-way noise evaluation: raw individual, filtered individual, and fair matched-population
- Safe trial finalization lifecycle: trial_status begins as 'aborted' and is marked
  'completed' ONLY after successful capture, full output persistence, and verification.
- Replay protection: aborted trials are rejected during replay unless explicitly opted in.
- Preserves partial frame data and writes trial metadata on interruption or output failure.
- Portrait rotation (CW / CCW) applied consistently to RGB, depth arrays, and intrinsics.
- Clear separation between physical tripod mounting and software image rotation.
- Live operator prompt before every repetition to guide subject positioning.
- Strict device option inspection (emitter power, ASIC and projector temperatures).
- Comprehensive versioned trial_meta.json adhering to the Prompt 2G metadata schema.
- Temporal depth noise and jitter evaluation via realsense_measure.diagnostics.temporal_noise.
- Session summary CSV with matched population denoising metrics and failure diagnostics.
- Safe abort handling on frame drops, device disconnection, or persistence failure.

Usage:
    python tools/run_sweet_spot_experiment.py --distance 1.2 --azimuth 0.0 --pose a_pose --orientation portrait_cw
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    PYREALSENSE_AVAILABLE = False

from realsense_measure.config import CameraConfig, PipelineConfig
from realsense_measure.camera.frame import RGBDFrame
from realsense_measure.camera.realsense_capture import (
    CameraNotStartedError,
    DeviceDisconnectedError,
    RealSenseCamera,
    StreamNegotiationError,
)
from realsense_measure.camera.rotation import rotate_intrinsics, rotate_rgbd_arrays
from realsense_measure.diagnostics.temporal_noise import (
    MatchedPopulationComparison,
    TemporalNoiseComparison,
    compare_raw_vs_filtered_noise,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweet_spot_runner")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intel RealSense D455f Sweet-Spot Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--distance", type=float, default=1.20, help="Target distance in metres (camera to CoR)")
    parser.add_argument("--elevation", type=float, default=0.0, help="Camera elevation angle in degrees (+ = downward)")
    parser.add_argument("--azimuth", type=float, default=0.0, help="Subject azimuth angle in degrees (0 = frontal)")
    parser.add_argument("--pose", type=str, default="a_pose", help="Subject pose name (a_pose, t_pose, neutral, natural)")
    parser.add_argument(
        "--orientation",
        type=str,
        default="portrait_cw",
        choices=["landscape", "portrait_cw", "portrait_ccw"],
        help="Camera mounting orientation and matching image rotation",
    )
    parser.add_argument("--burst-frames", type=int, default=30, help="Number of frames per burst capture")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repeat trials for this condition")
    parser.add_argument("--emitter-power", type=float, default=None, help="Target IR projector power in mW (e.g. 360.0, 240.0)")
    parser.add_argument("--subject-id", type=str, default="SUBJ01", help="Subject identifier or 'STATIC_REF'")
    parser.add_argument("--output-dir", type=str, default="experiments/sweet_spot", help="Base output directory")
    parser.add_argument("--static-reference", action="store_true", help="Capture mode for static planar reference target")
    parser.add_argument("--warmup-min", type=float, default=None, help="Manually logged camera warm-up time in minutes")
    parser.add_argument("--ambient-lux", type=float, default=None, help="Manually logged room illuminance in lux")
    parser.add_argument("--room-temp-c", type=float, default=None, help="Manually logged ambient room temperature in Celsius")
    parser.add_argument("--skip-prompt", action="store_true", help="Skip operator interactive prompt before trials")
    parser.add_argument("--mock-capture", action="store_true", help="Simulate camera captures without physical RealSense")
    parser.add_argument("--mock-incomplete-frames", type=int, default=None, help="Simulate premature cutoff after N frames for testing abort handling")
    parser.add_argument("--mock-disconnect", action="store_true", help="Simulate DeviceDisconnectedError mid-capture")
    parser.add_argument(
        "--mock-write-failure",
        type=str,
        default=None,
        choices=["initial_marker", "frames", "meta", "noise", "trial_meta", "csv"],
        help="Simulate file persistence failure at specific pipeline stage for testing safe finalization",
    )
    return parser.parse_args(argv)


class MockRealSenseExperimentCamera:
    """Mock camera for offline verification, regression testing, and CI without hardware."""
    def __init__(
        self,
        cfg: PipelineConfig,
        distance_m: float = 1.20,
        incomplete_frames: Optional[int] = None,
        simulate_disconnect: bool = False,
    ):
        self.cfg = cfg
        self.distance_m = distance_m
        self.incomplete_frames = incomplete_frames
        self.simulate_disconnect = simulate_disconnect
        self.width = cfg.camera.width
        self.height = cfg.camera.height
        self.fps = cfg.camera.fps
        self.depth_scale = 0.001
        self._is_started = False
        fx = 615.0
        fy = 616.0
        cx = float(self.width) / 2.0
        cy = float(self.height) / 2.0
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        if self.cfg.camera.rotate_90 in ("cw", "ccw"):
            self.intrinsics_o3d = rotate_intrinsics(self.intrinsics_o3d, self.cfg.camera.rotate_90)

    def start(self):
        self._is_started = True
        return self

    def stop(self):
        self._is_started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def query_device_metadata(self) -> Dict[str, Any]:
        return {
            "device_name": "Mock RealSense D455f",
            "serial_number": "MOCK_SERIAL_12345",
            "firmware_version": "5.15.1.0_MOCK",
            "asic_temperature_celsius": 42.0,
            "projector_temperature_celsius": 39.5,
            "temperature_status": "mock_readout",
            "emitter_enabled": True,
            "emitter_power_mw": 360.0,
            "emitter_status": "mock_readout",
        }

    def capture_burst_pair(
        self,
        num_frames: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
        """Simulate aligned color, raw depth, filtered depth, and timestamps."""
        color_frames = []
        raw_depth_frames = []
        filtered_depth_frames = []
        timestamps = []

        base_time = time.time()
        rot = self.cfg.camera.rotate_90

        w, h = self.width, self.height
        target_count = num_frames
        if self.incomplete_frames is not None:
            target_count = min(num_frames, self.incomplete_frames)

        np.random.seed(42)
        for i in range(target_count):
            if self.simulate_disconnect and i >= 2:
                self.partial_burst = (color_frames, raw_depth_frames, filtered_depth_frames, timestamps)
                raise DeviceDisconnectedError("Simulated device disconnection mid-capture.")

            ts = base_time + (i / float(self.fps))

            # Base synthetic human/planar target depth map
            drift_m = (i * 0.0005)
            raw_noise = np.random.normal(0.0, 0.002, (h, w)).astype(np.float32)
            filtered_noise = np.random.normal(0.0, 0.001, (h, w)).astype(np.float32)

            raw_d = np.zeros((h, w), dtype=np.float32)
            filt_d = np.zeros((h, w), dtype=np.float32)

            # Central rectangular subject/target
            y1, y2 = int(h * 0.15), int(h * 0.85)
            x1, x2 = int(w * 0.25), int(w * 0.75)
            target_depth = self.distance_m + drift_m

            raw_d[y1:y2, x1:x2] = target_depth + raw_noise[y1:y2, x1:x2]
            filt_d[y1:y2, x1:x2] = target_depth + filtered_noise[y1:y2, x1:x2]

            # In the raw stream, simulate a small boundary void (5% width) to verify mask intersection
            raw_d[y1:y2, x1 : x1 + 10] = 0.0

            # Color frame
            color = np.full((h, w, 3), 120, dtype=np.uint8)
            color[y1:y2, x1:x2] = [180, 140, 100]

            if rot in ("cw", "ccw"):
                color, raw_d = rotate_rgbd_arrays(color, raw_d, rot)
                _, filt_d = rotate_rgbd_arrays(color, filt_d, rot)

            color_frames.append(color)
            raw_depth_frames.append(raw_d)
            filtered_depth_frames.append(filt_d)
            timestamps.append(ts)

        return color_frames, raw_depth_frames, filtered_depth_frames, timestamps


def inspect_device_controls(
    camera: RealSenseCamera,
    target_emitter_power: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query and optionally configure hardware controls (emitter power, temperatures, firmware).
    Strictly checks whether options are supported by the physical device before querying.
    """
    meta: Dict[str, Any] = {
        "device_name": None,
        "serial_number": None,
        "firmware_version": None,
        "asic_temperature_celsius": None,
        "projector_temperature_celsius": None,
        "temperature_status": "unavailable",
        "emitter_enabled": None,
        "emitter_power_mw": None,
        "emitter_status": "unavailable",
    }

    if not PYREALSENSE_AVAILABLE or camera._profile is None:
        return meta

    try:
        device = camera._profile.get_device()
        if device.supports(rs.camera_info.name):
            meta["device_name"] = device.get_info(rs.camera_info.name)
        if device.supports(rs.camera_info.serial_number):
            meta["serial_number"] = device.get_info(rs.camera_info.serial_number)
        if device.supports(rs.camera_info.firmware_version):
            meta["firmware_version"] = device.get_info(rs.camera_info.firmware_version)

        # Depth sensor controls
        depth_sensor = device.first_depth_sensor()
        if depth_sensor is not None:
            # Emitter enable
            if depth_sensor.supports(rs.option.emitter_enabled):
                meta["emitter_enabled"] = bool(depth_sensor.get_option(rs.option.emitter_enabled) > 0.5)

            # Laser power
            if depth_sensor.supports(rs.option.laser_power):
                if target_emitter_power is not None:
                    try:
                        p_range = depth_sensor.get_option_range(rs.option.laser_power)
                        clamped_power = float(np.clip(target_emitter_power, p_range.min, p_range.max))
                        depth_sensor.set_option(rs.option.laser_power, clamped_power)
                        logger.info(f"  [camera] Set emitter laser_power to {clamped_power:.1f} mW (range: [{p_range.min}, {p_range.max}])")
                    except Exception as e:
                        logger.warning(f"  [camera] Failed to set laser_power to {target_emitter_power}: {e}")

                meta["emitter_power_mw"] = float(depth_sensor.get_option(rs.option.laser_power))
                meta["emitter_status"] = "measured_from_device"
            else:
                meta["emitter_status"] = "unsupported_by_device"
                if target_emitter_power is not None:
                    logger.warning("  [camera] Requested --emitter-power, but rs.option.laser_power is unsupported on this device.")

            # Temperatures
            asic_val = None
            proj_val = None
            if depth_sensor.supports(rs.option.asic_temperature):
                asic_val = float(depth_sensor.get_option(rs.option.asic_temperature))
            if depth_sensor.supports(rs.option.projector_temperature):
                proj_val = float(depth_sensor.get_option(rs.option.projector_temperature))

            meta["asic_temperature_celsius"] = asic_val
            meta["projector_temperature_celsius"] = proj_val
            if asic_val is not None or proj_val is not None:
                meta["temperature_status"] = "measured_from_device"
            else:
                meta["temperature_status"] = "unsupported_by_device"

    except Exception as exc:
        logger.warning(f"  [camera] Metadata inspection encountered error: {exc}")

    return meta


def capture_dual_stream_burst(
    camera: RealSenseCamera,
    burst_frames: int,
    rotate_90: Optional[str] = None,
    timeout_ms: int = 2000,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Capture a burst of frames, extracting synchronous raw depth and filtered depth pairs.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]
        (color_frames, raw_depth_frames, filtered_depth_frames, timestamps_sec)
    """
    if camera._profile is None or camera.depth_scale is None:
        raise CameraNotStartedError("RealSense camera is not started. Call start() before capturing burst.")

    color_frames: List[np.ndarray] = []
    raw_depth_frames: List[np.ndarray] = []
    filtered_depth_frames: List[np.ndarray] = []
    timestamps: List[float] = []

    consecutive_drops = 0
    max_drops = 10

    for i in range(burst_frames):
        frames = camera._wait_for_frames(timeout_ms=timeout_ms)
        if frames is None:
            consecutive_drops += 1
            if consecutive_drops >= max_drops:
                camera.partial_burst = (color_frames, raw_depth_frames, filtered_depth_frames, timestamps)
                raise DeviceDisconnectedError(f"RealSense camera dropped {consecutive_drops} consecutive frames. Aborting trial.")
            continue
        consecutive_drops = 0

        aligned = camera.align.process(frames)
        depth_frame_raw = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame_raw or not color_frame:
            continue

        # Extract timestamp in seconds
        ts_sec = float(depth_frame_raw.get_timestamp()) / 1000.0

        # Raw metric depth (unfiltered)
        raw_depth_m = (
            np.asanyarray(depth_frame_raw.get_data()).astype(np.float32)
            * camera.depth_scale
        )
        color_bgr = np.asanyarray(color_frame.get_data())

        # Production-filtered metric depth
        filtered_frame = camera._filter_depth(depth_frame_raw)
        filtered_depth_m = (
            np.asanyarray(filtered_frame.get_data()).astype(np.float32)
            * camera.depth_scale
        )

        # Handle potential decimation resize on filtered depth
        if filtered_depth_m.shape[:2] != color_bgr.shape[:2]:
            h, w = color_bgr.shape[:2]
            filtered_depth_m = cv2.resize(filtered_depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # Consistently rotate all arrays if portrait mode is active
        if rotate_90 in ("cw", "ccw"):
            color_bgr, raw_depth_m = rotate_rgbd_arrays(color_bgr, raw_depth_m, rotate_90)
            _, filtered_depth_m = rotate_rgbd_arrays(color_bgr, filtered_depth_m, rotate_90)

        color_frames.append(color_bgr)
        raw_depth_frames.append(raw_depth_m)
        filtered_depth_frames.append(filtered_depth_m)
        timestamps.append(ts_sec)

    return color_frames, raw_depth_frames, filtered_depth_frames, timestamps


def extract_evaluation_mask(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    is_static_ref: bool,
    target_distance_m: float,
) -> np.ndarray:
    """
    Extract spatial region of interest mask for noise evaluation.
    For human subjects: uses person segmentation or depth foreground.
    For static reference targets: uses planar depth window around target distance.
    """
    h, w = depth_m.shape[:2]
    if is_static_ref:
        d_min = max(0.20, target_distance_m - 0.25)
        d_max = target_distance_m + 0.25
        in_depth = (depth_m >= d_min) & (depth_m <= d_max)

        y_margin = int(h * 0.20)
        x_margin = int(w * 0.20)
        spatial_box = np.zeros((h, w), dtype=bool)
        spatial_box[y_margin : h - y_margin, x_margin : w - x_margin] = True
        mask = in_depth & spatial_box
        if np.sum(mask) >= 50:
            return mask
        return in_depth

    # Human subject: try MediaPipe or fallback connected component
    try:
        from realsense_measure.body.capture_motion import extract_subject_mask
        mask = extract_subject_mask(color_bgr, depth_m)
        if mask is not None and np.sum(mask) >= 50:
            return mask
    except Exception:
        pass

    # Connected component fallback
    valid = (depth_m >= 0.40) & (depth_m <= 3.00)
    if np.any(valid):
        med_z = float(np.median(depth_m[valid]))
        slice_m = (depth_m >= max(0.40, med_z - 0.40)) & (depth_m <= med_z + 0.40)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(slice_m.astype(np.uint8), connectivity=8)
        if n_lbl > 1:
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[largest, cv2.CC_STAT_AREA] >= 100:
                return (lbls == largest)

    return (depth_m > 0.0)


def prompt_operator(
    repetition: int,
    total_repetitions: int,
    distance_m: float,
    elevation_deg: float,
    azimuth_deg: float,
    pose: str,
    orientation: str,
    subject_id: str,
    is_static_ref: bool,
    skip_prompt: bool,
) -> bool:
    """Display interactive operator guidance before each trial repetition."""
    if skip_prompt:
        return True

    header = f"=== TRIAL SETUP PROMPT (Repetition {repetition} of {total_repetitions}) ==="
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"  Target Type:  {'STATIC PLANAR REFERENCE TARGET' if is_static_ref else 'HUMAN SUBJECT'}")
    print(f"  Subject ID:   {subject_id}")
    print(f"  Distance:     {distance_m:.2f} m (Camera front to floor Center of Rotation)")
    print(f"  Elevation:    {elevation_deg:+.1f}° on tripod head")
    print(f"  Azimuth:      {azimuth_deg:+.1f}° on radial floor protractor")
    print(f"  Pose:         {pose}")
    print(f"  Mounting:     {orientation}")
    print("----------------------------------------------------------------------")
    print("  CHECKLIST:")
    print("  [1] Confirm physical camera tripod is at requested elevation and orientation.")
    print("  [2] Confirm subject's ankles are centered over the floor Center of Rotation (CoR).")
    print(f"  [3] Confirm subject's stance aligns with {azimuth_deg:+.1f}° azimuth line.")
    print(f"  [4] Subject assumes '{pose}' stance and holds breath at resting expiration.")
    print("----------------------------------------------------------------------")
    try:
        user_input = input("  Press [ENTER] to execute burst capture (or 'q' to abort): ").strip().lower()
        if user_input == "q":
            logger.info("Operator requested abort.")
            return False
    except (EOFError, KeyboardInterrupt):
        return False
    return True


def run_experiment(args: argparse.Namespace) -> int:
    """Execute complete sweet-spot trial session according to Prompt 2G specifications."""
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = f"SESSION_{timestamp_str}_{args.subject_id}"
    base_out = Path(args.output_dir) / session_id
    base_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Sweet-Spot Experiment Session: {session_id}")
    logger.info(f"Output directory: {base_out}")

    # Determine software rotation vs physical mounting
    if args.orientation == "portrait_cw":
        rotate_90 = "cw"
        physical_mount = "portrait_90_cw"
        software_rot = "cw"
    elif args.orientation == "portrait_ccw":
        rotate_90 = "ccw"
        physical_mount = "portrait_90_ccw"
        software_rot = "ccw"
    else:
        rotate_90 = None
        physical_mount = "landscape"
        software_rot = "none"

    # Configure PipelineConfig
    cfg = PipelineConfig()
    cfg.target.name = "body"
    cfg.camera.depth_min_m = 0.40
    cfg.camera.depth_max_m = 3.00
    cfg.camera.rotate_90 = rotate_90

    # Initialize CSV summary with failure stage and reason columns
    csv_path = base_out / "session_summary.csv"
    csv_headers = [
        "trial_id",
        "repetition",
        "trial_status",
        "failure_stage",
        "failure_reason",
        "is_static_ref",
        "distance_m",
        "elevation_deg",
        "azimuth_deg",
        "pose",
        "orientation",
        "requested_frames",
        "captured_frames",
        "raw_mean_jitter_mm",
        "raw_median_jitter_mm",
        "raw_detrended_mean_jitter_mm",
        "filtered_mean_jitter_mm",
        "filtered_median_jitter_mm",
        "filtered_detrended_mean_jitter_mm",
        "matched_stable_pixel_count",
        "raw_detrended_jitter_mm_matched",
        "filtered_detrended_jitter_mm_matched",
        "matched_population_denoising_reduction_percent",
        "jitter_reduction_pct",
        "stable_pixels",
        "stable_ratio_pct",
        "temporal_validity_pct",
        "motion_displacement_mm",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # Initialize camera (mock or physical)
    camera = None
    if args.mock_capture or not PYREALSENSE_AVAILABLE:
        if not args.mock_capture:
            logger.info("pyrealsense2 not available; automatically using MockRealSenseExperimentCamera.")
        camera = MockRealSenseExperimentCamera(
            cfg=cfg,
            distance_m=args.distance,
            incomplete_frames=getattr(args, "mock_incomplete_frames", None),
            simulate_disconnect=getattr(args, "mock_disconnect", False),
        )
    else:
        try:
            camera = RealSenseCamera(cfg)
            camera.start()
        except Exception as exc:
            logger.error(f"Failed to start physical RealSense camera: {exc}")
            return 1

    try:
        # Query device metadata
        if isinstance(camera, MockRealSenseExperimentCamera):
            device_meta = camera.query_device_metadata()
            intr_o3d = camera.intrinsics_o3d
        else:
            device_meta = inspect_device_controls(camera, target_emitter_power=args.emitter_power)
            intr_o3d = camera.intrinsics_o3d

        intr_mat = intr_o3d.intrinsic_matrix
        intr_meta = {
            "width": intr_o3d.width,
            "height": intr_o3d.height,
            "fx": float(intr_mat[0, 0]),
            "fy": float(intr_mat[1, 1]),
            "cx": float(intr_mat[0, 2]),
            "cy": float(intr_mat[1, 2]),
        }

        # Session metadata
        session_meta = {
            "session_id": session_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subject_id": args.subject_id,
            "is_static_reference": args.static_reference,
            "condition": {
                "distance_m": args.distance,
                "elevation_deg": args.elevation,
                "azimuth_deg": args.azimuth,
                "pose": args.pose,
                "orientation": args.orientation,
                "physical_mounting": physical_mount,
                "software_rotation": software_rot,
                "emitter_power_mw_requested": args.emitter_power,
            },
            "environment": {
                "warmup_min": args.warmup_min,
                "ambient_lux": args.ambient_lux,
                "room_temp_celsius": args.room_temp_c,
            },
            "device": device_meta,
            "repetitions": args.repetitions,
            "burst_frames": args.burst_frames,
        }
        with open(base_out / "session_meta.json", "w", encoding="utf-8") as f:
            json.dump(session_meta, f, indent=2)

        has_session_error = False
        session_error_message: Optional[str] = None

        # Loop through repetitions
        for rep in range(1, args.repetitions + 1):
            should_proceed = prompt_operator(
                repetition=rep,
                total_repetitions=args.repetitions,
                distance_m=args.distance,
                elevation_deg=args.elevation,
                azimuth_deg=args.azimuth,
                pose=args.pose,
                orientation=args.orientation,
                subject_id=args.subject_id,
                is_static_ref=args.static_reference,
                skip_prompt=args.skip_prompt,
            )
            if not should_proceed:
                logger.warning(f"Trial repetition {rep} aborted by operator.")
                break

            trial_name = (
                f"trial_{rep:02d}_D{int(args.distance * 100):03d}"
                f"_E{int(abs(args.elevation)):02d}{'n' if args.elevation < 0 else 'p'}"
                f"_A{int(abs(args.azimuth)):03d}{'n' if args.azimuth < 0 else 'p'}"
                f"_{args.pose.upper()}_{args.orientation.upper()}"
            )
            trial_dir = base_out / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n---> Executing Repetition {rep}/{args.repetitions}: {trial_name}")

            # -------------------------------------------------------------
            # SAFE TRIAL FINALIZATION LIFECYCLE
            # Rule 1: A trial begins in an explicitly non-completed state on disk.
            # Write initial aborted marker immediately before capture or persistence.
            # -------------------------------------------------------------
            trial_status = "aborted"
            failure_stage: Optional[str] = "in_progress"
            failure_reason: Optional[str] = "Trial has not completed finalization"
            all_verified: bool = False
            raw_valid_fractions: List[float] = []
            filtered_valid_fractions: List[float] = []
            raw_mean_depths: List[float] = []
            filtered_mean_depths: List[float] = []

            trial_meta_path = trial_dir / "trial_meta.json"
            initial_meta_doc = {
                "schema_version": "2.2",
                "trial_id": trial_name,
                "session_id": session_id,
                "trial_status": "aborted",
                "failure_stage": "in_progress",
                "failure_reason": "Trial has not completed finalization",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repetition_index": rep,
                "total_repetitions": args.repetitions,
                "is_static_reference": args.static_reference,
                "subject": {
                    "subject_id": args.subject_id,
                    "requested_pose": args.pose,
                    "distance_reference": "center_of_rotation_ankles",
                    "distance_reference_status": (
                        "measured_to_planar_target" if args.static_reference else "aligned_to_floor_grid"
                    ),
                },
                "geometry": {
                    "target_distance_m": args.distance,
                    "elevation_deg": args.elevation,
                    "azimuth_deg": args.azimuth,
                    "physical_mounting_orientation": physical_mount,
                    "software_image_rotation": software_rot,
                    "distance_status": "target_nominal",
                    "elevation_status": "manually_set_on_tripod",
                    "azimuth_status": "manually_aligned_to_floor_grid",
                },
                "stream": {
                    "requested_resolution": [cfg.camera.width, cfg.camera.height],
                    "effective_resolution": [intr_o3d.width, intr_o3d.height],
                    "fps": cfg.camera.fps,
                    "intrinsics": intr_meta,
                },
                "camera_hardware": device_meta,
                "emitter_controls": {
                    "emitter_enabled": device_meta.get("emitter_enabled"),
                    "emitter_power_mw": device_meta.get("emitter_power_mw"),
                    "emitter_status": device_meta.get("emitter_status"),
                },
                "filter_configuration": {
                    "raw_stream": {
                        "use_spatial_filter": False,
                        "use_temporal_filter": False,
                        "use_hole_filling": False,
                        "use_decimation": False,
                    },
                    "filtered_stream": {
                        "use_spatial_filter": cfg.camera.use_spatial_filter,
                        "use_temporal_filter": cfg.camera.use_temporal_filter,
                        "use_hole_filling": cfg.camera.use_hole_filling,
                        "use_decimation": cfg.camera.use_decimation,
                    },
                },
                "capture_performance": {
                    "requested_frames": args.burst_frames,
                    "captured_frames": 0,
                    "dropped_frames": args.burst_frames,
                    "capture_duration_sec": 0.0,
                    "capture_error": "Trial interrupted before or during capture",
                },
                "environment": {
                    "warmup_duration_min": args.warmup_min,
                    "warmup_status": "manually_entered" if args.warmup_min is not None else "unavailable",
                    "ambient_lux": args.ambient_lux,
                    "ambient_lux_status": "manually_entered" if args.ambient_lux is not None else "unavailable",
                    "room_temp_celsius": args.room_temp_c,
                    "room_temp_status": "manually_entered" if args.room_temp_c is not None else "unavailable",
                },
            }
            mock_write_fail = getattr(args, "mock_write_failure", None)
            initial_marker_written = False
            try:
                if mock_write_fail == "initial_marker":
                    raise IOError("Simulated disk error during initial trial marker write.")

                with open(trial_meta_path, "w", encoding="utf-8") as f:
                    json.dump(initial_meta_doc, f, indent=2)
                if trial_meta_path.exists() and trial_meta_path.stat().st_size > 0:
                    initial_marker_written = True
                else:
                    raise IOError("Initial trial marker was not created or is 0 bytes on disk.")
            except Exception as init_err:
                logger.error(f"CRITICAL: Failed to write initial trial marker for {trial_name}: {init_err}")
                failure_stage = "persistence_initial_marker"
                failure_reason = f"Initial trial marker write failure: {init_err}"
                has_session_error = True
                session_error_message = failure_reason

            if not initial_marker_written:
                # Rule 1 Hardening: Abort immediately without camera capture or frame persistence
                csv_row = [
                    trial_name,
                    rep,
                    "aborted",
                    failure_stage,
                    failure_reason,
                    args.static_reference,
                    args.distance,
                    args.elevation,
                    args.azimuth,
                    args.pose,
                    args.orientation,
                    args.burst_frames,
                    0,
                    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                ]
                try:
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_row)
                except Exception as csv_err:
                    logger.error(f"CRITICAL: Failed to append to session summary CSV: {csv_err}")

                print("  --- TRIAL METRICS ---")
                print("  Trial Status:       ABORTED")
                print(f"  Failure Stage:      {failure_stage}")
                print(f"  Failure Reason:     {failure_reason}")
                print(f"  Frames Captured:    0 / {args.burst_frames}")
                print(f"  Trial Saved:        {trial_dir} (Metadata persisted: False)")
                continue

            colors: List[np.ndarray] = []
            raw_depths: List[np.ndarray] = []
            filt_depths: List[np.ndarray] = []
            timestamps: List[float] = []
            t_start = time.time()
            capture_error: Optional[str] = None

            # Stage 1: Burst Capture
            try:
                if isinstance(camera, MockRealSenseExperimentCamera):
                    colors, raw_depths, filt_depths, timestamps = camera.capture_burst_pair(args.burst_frames)
                else:
                    colors, raw_depths, filt_depths, timestamps = capture_dual_stream_burst(
                        camera=camera,
                        burst_frames=args.burst_frames,
                        rotate_90=rotate_90,
                    )
            except DeviceDisconnectedError as disc_err:
                capture_error = f"DeviceDisconnectedError: {disc_err}"
                failure_stage = "capture"
                failure_reason = capture_error
                if hasattr(camera, "partial_burst") and camera.partial_burst is not None:
                    colors, raw_depths, filt_depths, timestamps = camera.partial_burst
                logger.error(f"Device disconnected during trial {trial_name}: {disc_err}")
            except Exception as gen_err:
                capture_error = f"Exception: {gen_err}"
                failure_stage = "capture"
                failure_reason = capture_error
                logger.error(f"Unexpected capture error during trial {trial_name}: {gen_err}")

            t_elapsed = time.time() - t_start

            if capture_error is None:
                if len(colors) < args.burst_frames:
                    failure_stage = "capture"
                    failure_reason = f"Incomplete capture: acquired {len(colors)} of {args.burst_frames} requested frames"
                    logger.warning(f"  [trial-lifecycle] {failure_reason}")
                else:
                    failure_stage = None
                    failure_reason = None

            # Stage 2: Output Persistence (Rule 2 & 3: Retain partial data and check writes)
            frames_meta = []
            mock_write_fail = getattr(args, "mock_write_failure", None)

            # 2a: Persist frame assets
            if len(colors) > 0:
                try:
                    if mock_write_fail == "frames":
                        raise IOError("Simulated disk error during frame asset write.")

                    for idx in range(len(colors)):
                        c_name = f"frame_{idx:02d}_color.png"
                        d_name = f"frame_{idx:02d}_depth.npy"
                        df_name = f"frame_{idx:02d}_depth_filtered.npy"

                        c_path = trial_dir / c_name
                        d_path = trial_dir / d_name
                        df_path = trial_dir / df_name

                        ok_c = cv2.imwrite(str(c_path), colors[idx])
                        if not ok_c or not c_path.exists():
                            raise IOError(f"Failed to write RGB image: {c_path}")

                        np.save(str(d_path), raw_depths[idx].astype(np.float32))
                        if not d_path.exists():
                            raise IOError(f"Failed to write raw depth array: {d_path}")

                        np.save(str(df_path), filt_depths[idx].astype(np.float32))
                        if not df_path.exists():
                            raise IOError(f"Failed to write filtered depth array: {df_path}")

                        frames_meta.append({
                            "frame_id": idx,
                            "order": idx,
                            "timestamp": timestamps[idx],
                            "color_file": c_name,
                            "depth_file": d_name,
                            "depth_filtered_file": df_name,
                        })
                except Exception as frame_err:
                    if failure_reason is None:
                        failure_stage = "persistence_frames"
                        failure_reason = f"Frame write failure: {frame_err}"
                    logger.error(f"Error persisting frame assets for {trial_name}: {frame_err}")

            # 2b: Persist replay meta.json
            cam_meta = {
                "width": cfg.camera.width,
                "height": cfg.camera.height,
                "fps": cfg.camera.fps,
                "use_decimation": cfg.camera.use_decimation,
                "use_spatial_filter": cfg.camera.use_spatial_filter,
                "use_temporal_filter": cfg.camera.use_temporal_filter,
                "use_hole_filling": cfg.camera.use_hole_filling,
                "depth_min_m": cfg.camera.depth_min_m,
                "depth_max_m": cfg.camera.depth_max_m,
                "rotate_90": rotate_90,
            }
            replay_meta = {
                "intrinsics": intr_meta,
                "camera_config": cam_meta,
                "target_name": cfg.target.name,
                "num_frames": len(colors),
                "has_background_ref": False,
                "frames": frames_meta,
            }
            try:
                if mock_write_fail == "meta":
                    raise IOError("Simulated disk error during meta.json write.")

                meta_json_path = trial_dir / "meta.json"
                with open(meta_json_path, "w", encoding="utf-8") as f:
                    json.dump(replay_meta, f, indent=2)
                if not meta_json_path.exists():
                    raise IOError("meta.json file was not created on disk.")
            except Exception as meta_err:
                if failure_reason is None:
                    failure_stage = "persistence_meta"
                    failure_reason = f"meta.json write failure: {meta_err}"
                logger.error(f"Error persisting meta.json for {trial_name}: {meta_err}")

            # 2c: Evaluate and persist noise_evaluation.json
            noise_comp: Optional[TemporalNoiseComparison] = None
            try:
                if mock_write_fail == "noise":
                    raise IOError("Simulated disk error during noise_evaluation.json write.")

                if len(colors) >= 3:
                    mid_idx = len(colors) // 2
                    eval_mask = extract_evaluation_mask(
                        color_bgr=colors[mid_idx],
                        depth_m=raw_depths[mid_idx],
                        is_static_ref=args.static_reference,
                        target_distance_m=args.distance,
                    )
                    noise_comp = compare_raw_vs_filtered_noise(
                        raw_depth_frames=raw_depths,
                        filtered_depth_frames=filt_depths,
                        timestamps=timestamps,
                        roi_mask=eval_mask,
                        depth_min_m=cfg.camera.depth_min_m,
                        depth_max_m=cfg.camera.depth_max_m,
                        stable_threshold=0.90,
                        min_valid_samples=3,
                    )
                    noise_json_path = trial_dir / "noise_evaluation.json"
                    with open(noise_json_path, "w", encoding="utf-8") as f:
                        json.dump(noise_comp.to_dict(), f, indent=2)
                else:
                    noise_json_path = trial_dir / "noise_evaluation.json"
                    with open(noise_json_path, "w", encoding="utf-8") as f:
                        json.dump({"status": "insufficient_frames", "frames_captured": len(colors)}, f, indent=2)

                if not (trial_dir / "noise_evaluation.json").exists():
                    raise IOError("noise_evaluation.json was not created on disk.")
            except Exception as noise_err:
                if failure_reason is None:
                    failure_stage = "persistence_noise"
                    failure_reason = f"noise_evaluation.json write failure: {noise_err}"
                logger.error(f"Error persisting noise_evaluation.json for {trial_name}: {noise_err}")

            # Stage 3: Verification (Rule 4 & 5: Only mark completed after verification passes)
            if failure_reason is None and len(colors) == args.burst_frames:
                all_verified = True
                expected_h = intr_o3d.height
                expected_w = intr_o3d.width

                def _is_valid_regular_file(p: Path) -> bool:
                    return p.exists() and p.is_file() and p.stat().st_size > 0

                try:
                    # Check replay meta.json
                    meta_p = trial_dir / "meta.json"
                    if not _is_valid_regular_file(meta_p):
                        all_verified = False
                    else:
                        with open(meta_p, "r", encoding="utf-8") as f:
                            loaded_meta = json.load(f)
                        if not isinstance(loaded_meta, dict) or "frames" not in loaded_meta:
                            all_verified = False

                    # Check noise_evaluation.json
                    noise_p = trial_dir / "noise_evaluation.json"
                    if not _is_valid_regular_file(noise_p):
                        all_verified = False
                    else:
                        with open(noise_p, "r", encoding="utf-8") as f:
                            loaded_noise = json.load(f)
                        if not isinstance(loaded_noise, dict):
                            all_verified = False

                    # Check all frame assets (PNG decodable, NPY loadable float32, numerical integrity)
                    raw_valid_fractions.clear()
                    filtered_valid_fractions.clear()
                    raw_mean_depths.clear()
                    filtered_mean_depths.clear()

                    def _check_depth_numerical_integrity(arr: np.ndarray, stream_name: str) -> tuple[bool, float, float, str]:
                        if arr is None or arr.dtype != np.float32:
                            return False, 0.0, 0.0, f"{stream_name} depth is not float32"
                        if arr.shape[:2] != (expected_h, expected_w):
                            return False, 0.0, 0.0, f"{stream_name} depth shape {arr.shape[:2]} != expected {(expected_h, expected_w)}"
                        if np.isneginf(arr).any() or np.isposinf(arr).any():
                            return False, 0.0, 0.0, f"{stream_name} depth contains infinite values"

                        # Valid finite depth samples in plausible sensor range [0.10, 10.00] m
                        valid_mask = np.isfinite(arr) & (arr >= 0.10) & (arr <= 10.00)
                        valid_count = int(np.count_nonzero(valid_mask))
                        total_count = int(arr.size)
                        valid_frac = float(valid_count) / float(total_count) if total_count > 0 else 0.0
                        mean_depth = float(np.mean(arr[valid_mask])) if valid_count > 0 else 0.0

                        if valid_count == 0:
                            return False, 0.0, 0.0, f"{stream_name} depth contains no valid depth samples (all zeros or all NaNs)"

                        # Documented minimum threshold: 2% (approx 8,140 pixels at 848x480)
                        # ensures target subject or planar reference target is detected
                        MIN_VALID_DEPTH_FRACTION = 0.02
                        if valid_frac < MIN_VALID_DEPTH_FRACTION:
                            return False, valid_frac, mean_depth, (
                                f"{stream_name} depth valid fraction ({valid_frac:.2%}) "
                                f"below minimum threshold ({MIN_VALID_DEPTH_FRACTION:.2%})"
                            )

                        return True, valid_frac, mean_depth, "ok"

                    verif_failure_detail = None
                    for idx in range(args.burst_frames):
                        c_p = trial_dir / f"frame_{idx:02d}_color.png"
                        d_p = trial_dir / f"frame_{idx:02d}_depth.npy"
                        df_p = trial_dir / f"frame_{idx:02d}_depth_filtered.npy"

                        if not (_is_valid_regular_file(c_p) and _is_valid_regular_file(d_p) and _is_valid_regular_file(df_p)):
                            all_verified = False
                            verif_failure_detail = f"Frame {idx} files missing, not regular files, or 0 bytes"
                            break

                        c_img = cv2.imread(str(c_p), cv2.IMREAD_UNCHANGED)
                        if c_img is None or c_img.shape[:2] != (expected_h, expected_w):
                            all_verified = False
                            verif_failure_detail = f"Frame {idx} color image decode failure or dimension mismatch"
                            break

                        d_arr = np.load(str(d_p))
                        ok_raw, r_frac, r_mean, err_raw = _check_depth_numerical_integrity(d_arr, f"frame_{idx:02d}_depth.npy")
                        if not ok_raw:
                            all_verified = False
                            verif_failure_detail = err_raw
                            break
                        raw_valid_fractions.append(r_frac)
                        raw_mean_depths.append(r_mean)

                        df_arr = np.load(str(df_p))
                        ok_filt, f_frac, f_mean, err_filt = _check_depth_numerical_integrity(df_arr, f"frame_{idx:02d}_depth_filtered.npy")
                        if not ok_filt:
                            all_verified = False
                            verif_failure_detail = err_filt
                            break
                        filtered_valid_fractions.append(f_frac)
                        filtered_mean_depths.append(f_mean)
                except Exception as verif_err:
                    logger.error(f"Asset verification error in {trial_name}: {verif_err}")
                    all_verified = False
                    verif_failure_detail = str(verif_err)

                if all_verified:
                    trial_status = "completed"
                    failure_stage = None
                    failure_reason = None
                else:
                    trial_status = "aborted"
                    failure_stage = "verification"
                    failure_reason = f"Post-persistence verification failed: {verif_failure_detail or 'one or more required output files are missing, empty, or unreadable.'}"
            else:
                trial_status = "aborted"

            # Stage 4: Persist trial_meta.json (Rule 7: Atomic safe replacement)
            trial_meta_doc = {
                "schema_version": "2.2",
                "trial_id": trial_name,
                "session_id": session_id,
                "trial_status": trial_status,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repetition_index": rep,
                "total_repetitions": args.repetitions,
                "is_static_reference": args.static_reference,
                "depth_numerical_integrity": {
                    "verified": all_verified,
                    "status": "verified" if all_verified else "failed",
                    "raw_valid_fraction": (
                        round(float(np.mean(raw_valid_fractions)), 4) if raw_valid_fractions else 0.0
                    ),
                    "filtered_valid_fraction": (
                        round(float(np.mean(filtered_valid_fractions)), 4) if filtered_valid_fractions else 0.0
                    ),
                    "raw_mean_valid_depth_fraction": (
                        round(float(np.mean(raw_valid_fractions)), 4) if raw_valid_fractions else 0.0
                    ),
                    "filtered_mean_valid_depth_fraction": (
                        round(float(np.mean(filtered_valid_fractions)), 4) if filtered_valid_fractions else 0.0
                    ),
                    "raw_mean_depth_m": (
                        round(float(np.mean(raw_mean_depths)), 4) if raw_mean_depths else 0.0
                    ),
                    "filtered_mean_depth_m": (
                        round(float(np.mean(filtered_mean_depths)), 4) if filtered_mean_depths else 0.0
                    ),
                    "min_valid_depth_fraction_threshold": 0.02,
                },
                "subject": {
                    "subject_id": args.subject_id,
                    "requested_pose": args.pose,
                    "distance_reference": "center_of_rotation_ankles",
                    "distance_reference_status": (
                        "measured_to_planar_target" if args.static_reference else "aligned_to_floor_grid"
                    ),
                },
                "geometry": {
                    "target_distance_m": args.distance,
                    "elevation_deg": args.elevation,
                    "azimuth_deg": args.azimuth,
                    "physical_mounting_orientation": physical_mount,
                    "software_image_rotation": software_rot,
                    "distance_status": "target_nominal",
                    "elevation_status": "manually_set_on_tripod",
                    "azimuth_status": "manually_aligned_to_floor_grid",
                },
                "stream": {
                    "requested_resolution": [cfg.camera.width, cfg.camera.height],
                    "effective_resolution": [intr_o3d.width, intr_o3d.height],
                    "fps": cfg.camera.fps,
                    "intrinsics": intr_meta,
                },
                "camera_hardware": device_meta,
                "emitter_controls": {
                    "emitter_enabled": device_meta.get("emitter_enabled"),
                    "emitter_power_mw": device_meta.get("emitter_power_mw"),
                    "emitter_status": device_meta.get("emitter_status"),
                },
                "filter_configuration": {
                    "raw_stream": {
                        "use_spatial_filter": False,
                        "use_temporal_filter": False,
                        "use_hole_filling": False,
                        "use_decimation": False,
                    },
                    "filtered_stream": {
                        "use_spatial_filter": cfg.camera.use_spatial_filter,
                        "use_temporal_filter": cfg.camera.use_temporal_filter,
                        "use_hole_filling": cfg.camera.use_hole_filling,
                        "use_decimation": cfg.camera.use_decimation,
                    },
                },
                "capture_performance": {
                    "requested_frames": args.burst_frames,
                    "captured_frames": len(colors),
                    "dropped_frames": args.burst_frames - len(colors),
                    "capture_duration_sec": round(t_elapsed, 3),
                    "capture_error": capture_error,
                },
                "environment": {
                    "warmup_duration_min": args.warmup_min,
                    "warmup_status": "manually_entered" if args.warmup_min is not None else "unavailable",
                    "ambient_lux": args.ambient_lux,
                    "ambient_lux_status": "manually_entered" if args.ambient_lux is not None else "unavailable",
                    "room_temp_celsius": args.room_temp_c,
                    "room_temp_status": "manually_entered" if args.room_temp_c is not None else "unavailable",
                },
            }

            trial_meta_path = trial_dir / "trial_meta.json"
            trial_meta_tmp = trial_dir / "trial_meta.json.tmp"
            trial_meta_persisted = False
            try:
                if mock_write_fail == "trial_meta":
                    raise IOError("Simulated disk error during trial_meta.json write.")

                with open(trial_meta_tmp, "w", encoding="utf-8") as f:
                    json.dump(trial_meta_doc, f, indent=2)
                    f.flush()

                import os
                os.replace(trial_meta_tmp, trial_meta_path)
                if trial_meta_path.exists() and trial_meta_path.stat().st_size > 0:
                    trial_meta_persisted = True
                else:
                    raise IOError("Atomic replace failed to create non-empty trial_meta.json.")
            except Exception as tmeta_err:
                logger.error(f"CRITICAL: Failed to write trial_meta.json for {trial_name}: {tmeta_err}")
                trial_status = "aborted"
                if failure_reason is None:
                    failure_stage = "persistence_trial_meta"
                    failure_reason = f"trial_meta.json write failure: {tmeta_err}"
                has_session_error = True
                session_error_message = failure_reason
                try:
                    if trial_meta_tmp.exists():
                        trial_meta_tmp.unlink()
                except Exception:
                    pass

            # Stage 5: Append to session_summary.csv
            matched = noise_comp.matched if noise_comp else None
            csv_row = [
                trial_name,
                rep,
                trial_status,
                failure_stage or "",
                failure_reason or "",
                args.static_reference,
                args.distance,
                args.elevation,
                args.azimuth,
                args.pose,
                args.orientation,
                args.burst_frames,
                len(colors),
                noise_comp.raw.mean_std_mm if noise_comp else "",
                noise_comp.raw.median_std_mm if noise_comp else "",
                noise_comp.raw.mean_detrended_std_mm if noise_comp else "",
                noise_comp.filtered.mean_std_mm if noise_comp else "",
                noise_comp.filtered.median_std_mm if noise_comp else "",
                noise_comp.filtered.mean_detrended_std_mm if noise_comp else "",
                matched.matched_pixel_count if matched else "",
                matched.raw_mean_detrended_std_mm if matched else "",
                matched.filtered_mean_detrended_std_mm if matched else "",
                matched.denoising_reduction_pct if matched else "",
                noise_comp.jitter_reduction_pct if noise_comp else "",
                noise_comp.raw.num_stable_pixels if noise_comp else "",
                noise_comp.raw.stable_pixel_ratio_pct if noise_comp else "",
                noise_comp.raw.mean_temporal_validity_pct if noise_comp else "",
                noise_comp.motion_centroid_displacement_mm if noise_comp else "",
            ]
            try:
                if mock_write_fail == "csv":
                    raise IOError("Simulated disk error during CSV write.")
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)
            except Exception as csv_err:
                logger.error(f"CRITICAL: Failed to append to session summary CSV: {csv_err}")
                has_session_error = True
                session_error_message = f"CSV append failure: {csv_err}"

            # Console output
            print("  --- TRIAL METRICS ---")
            print(f"  Trial Status:       {trial_status.upper()}")
            if failure_reason:
                print(f"  Failure Stage:      {failure_stage}")
                print(f"  Failure Reason:     {failure_reason}")
            print(f"  Frames Captured:    {len(colors)} / {args.burst_frames}")
            if noise_comp and matched:
                print(f"  Matched Pixels:     {matched.matched_pixel_count} (Retention: Raw {matched.raw_stable_retention_pct:.1f}%, Filt {matched.filtered_stable_retention_pct:.1f}%)")
                print(f"  Raw Jitter (Match): {matched.raw_mean_detrended_std_mm or 0.0:.2f} mm")
                print(f"  Filt Jitter (Match):{matched.filtered_mean_detrended_std_mm or 0.0:.2f} mm")
                if matched.denoising_reduction_pct is not None:
                    print(f"  Denoising Efficacy: {matched.denoising_reduction_pct:.1f}%")
            print(f"  Trial Saved:        {trial_dir} (Metadata persisted: {trial_meta_persisted})")

            if capture_error:
                logger.error(f"Terminating session early due to capture error: {capture_error}")
                break

    finally:
        if camera is not None:
            camera.stop()

    if has_session_error:
        logger.error(f"\nExperiment Session Completed WITH ERRORS: {session_error_message}")
        return 1

    logger.info(f"\nExperiment Session Complete. Results saved in: {base_out}")
    logger.info(f"Summary CSV: {csv_path}")
    return 0


def main():
    args = parse_args()
    sys.exit(run_experiment(args))


if __name__ == "__main__":
    main()
