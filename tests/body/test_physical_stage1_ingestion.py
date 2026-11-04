"""
tests/body/test_physical_stage1_ingestion.py
============================================
Physical RGB-D ingestion and geometry validation tests for Stage 1.

Validates the real physical dataset recorded with the Intel RealSense D455f:
- raw_captures/body_capture_01
Ensures dataset loading, calibration integrity, coordinate transformation fidelity,
and inter-stop stability without running articulated fitting or anthropometry.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from camera.recording import load_raw_capture
from realsense_measure.body.adapter import (
    R_CAM_TO_WORLD,
    R_WORLD_TO_CAM,
    camera_to_world_points,
    world_to_camera_points,
    frame_to_body_stop,
    extract_intrinsics_dict,
)


PHYSICAL_DATASET_DIR = Path("raw_captures/body_capture_01")
AUDIT_OUTPUT_DIR = Path("stage1_audit_output")
CALIBRATED_DISTANCE_M = 2.142


@pytest.mark.skipif(
    not PHYSICAL_DATASET_DIR.exists(),
    reason="Physical capture raw_captures/body_capture_01 not present"
)
class TestPhysicalStage1Ingestion:

    def test_physical_dataset_loading(self):
        """Verify load_raw_capture loads the physical dataset with exact dimensions and calibration."""
        frames = load_raw_capture(PHYSICAL_DATASET_DIR)
        assert len(frames) == 4, f"Expected 4 stops, got {len(frames)}"

        meta_file = PHYSICAL_DATASET_DIR / "meta.json"
        assert meta_file.exists(), "meta.json must be present in raw capture folder"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("target_name") == "body"

        for i, frame in enumerate(frames):
            assert frame.color_bgr.shape == (720, 1280, 3)
            assert frame.depth_m.shape == (720, 1280)
            assert frame.depth_m.dtype == np.float32
            assert frame.intrinsics is not None

            # Verify depth range has valid physical meters
            valid_depth = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0.1)]
            assert len(valid_depth) > 10000
            assert 0.4 <= np.min(valid_depth) <= 1.0
            assert np.max(valid_depth) >= 2.0

    def test_physical_adapter_conversion(self):
        """Verify frame_to_body_stop converts physical frames to valid AdaptedStopData."""
        frames = load_raw_capture(PHYSICAL_DATASET_DIR)
        nominal_yaws = [0.0, 90.0, 180.0, 270.0]

        for i, (frame, yaw) in enumerate(zip(frames, nominal_yaws)):
            stop = frame_to_body_stop(
                frame=frame,
                nominal_yaw_deg=yaw,
                distance_m=CALIBRATED_DISTANCE_M,
                stop_idx=i,
            )

            # Contract checks
            assert stop.point_cloud.shape[1] == 3
            assert stop.point_cloud_world.shape[1] == 3
            assert len(stop.point_cloud) == len(stop.point_cloud_world)
            assert len(stop.point_cloud) > 20000, f"Stop {i} point cloud unexpectedly sparse"
            assert np.all(np.isfinite(stop.point_cloud))
            assert np.all(np.isfinite(stop.point_cloud_world))

            # Foreground mask checks
            assert stop.foreground_mask.shape == (720, 1280)
            assert np.count_nonzero(stop.foreground_mask) > 10000
            assert stop.nominal_yaw_deg == yaw
            assert np.isclose(stop.intrinsics["distance_m"], CALIBRATED_DISTANCE_M)
            assert stop.stop_idx == i

    def test_physical_coordinate_transform_roundtrip(self):
        """Verify round-trip camera-to-world and world-to-camera on real physical points."""
        frames = load_raw_capture(PHYSICAL_DATASET_DIR)
        frame0 = frames[0]
        stop0 = frame_to_body_stop(frame0, nominal_yaw_deg=0.0, distance_m=CALIBRATED_DISTANCE_M)

        pts_cam = stop0.point_cloud
        pts_world = stop0.point_cloud_world

        # Direct transformation verification
        pts_cam_reprojected = world_to_camera_points(pts_world, distance_m=CALIBRATED_DISTANCE_M)
        diff = np.max(np.abs(pts_cam - pts_cam_reprojected))
        assert diff < 1e-6, f"Roundtrip error {diff} exceeded numerical tolerance"

        # Determinant and rotation orthogonality
        det_cam2world = np.linalg.det(R_CAM_TO_WORLD)
        det_world2cam = np.linalg.det(R_WORLD_TO_CAM)
        assert np.isclose(det_cam2world, 1.0)
        assert np.isclose(det_world2cam, 1.0)
        assert np.allclose(R_CAM_TO_WORLD.T @ R_CAM_TO_WORLD, np.eye(3))

    def test_physical_inter_stop_stability(self):
        """Verify subject stance stability across the 4 physical stops."""
        frames = load_raw_capture(PHYSICAL_DATASET_DIR)
        nominal_yaws = [0.0, 90.0, 180.0, 270.0]

        distances = []
        for frame, yaw in zip(frames, nominal_yaws):
            # Compute median depth inside typical human bounds [0.4, 3.0]
            d = frame.depth_m
            valid_d = d[(d >= 0.40) & (d <= 3.00) & np.isfinite(d)]
            distances.append(float(np.median(valid_d)))

        std_distance = np.std(distances)
        # Check that stance variance is small (< 30 mm)
        assert std_distance < 0.030, f"Subject stance distance drift {std_distance*1000:.1f} mm exceeds 30 mm"

    def test_stage1_audit_summary_file(self):
        """Verify the generated stage1_geometry_audit_summary.json exists and is valid."""
        summary_path = AUDIT_OUTPUT_DIR / "stage1_geometry_audit_summary.json"
        if not summary_path.exists():
            pytest.skip("Audit tool has not been run yet")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["num_stops_audited"] == 4
        assert len(summary["stops"]) == 4
        assert summary["inter_stop_consistency"]["centroid_drift_xy_mm"] < 50.0
        assert summary["coordinate_frame_verification"]["is_right_handed_so3"] is True
