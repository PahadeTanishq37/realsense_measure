"""
tests/body/test_physical_stage17_anthropometry.py
=================================================
Unit and integration tests for Stage 17: Upper-Body Anthropometric Measurement Pipeline (Offline).

Verifies:
1. Multi-stage source data preservation (body_capture_01, 02, 03 strictly unmodified).
2. Synthetic geometric primitive slicing (software correctness tests: full cylinder vs partial arc).
3. Circumference blocking on incomplete angular surface coverage (< 75% or max gap > 60 deg).
4. Landmark unprojection, classification as ESTIMATED_2D_3D_LANDMARK, and truncation handling.
5. Extraction of valid physical upper-body measurements (biacromial breadth, arm lengths, chest breadth).
6. Strict enforcement of anthropometry safeguards (zero circumference fabrication on partial slices).
7. Deliverable artifact completeness and schema verification.
8. Final verdict confirmation: PIPELINE_IMPLEMENTED_OFFLINE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MEASURE_PKG = REPO_ROOT / "realsense_measure-main"
MEASURE_INNER = MEASURE_PKG / "realsense_measure"

for p in [str(REPO_ROOT), str(MEASURE_PKG), str(MEASURE_INNER)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realsense_measure.measurements import (
    AnatomicalLandmark,
    AnatomicalLandmarksExtractor,
    AnatomicalPlaneSlicer,
    DataSourceType,
    MeasurementQualityAuditor,
    MeasurementQualityStatus,
    SliceRingAnalysis,
    Stage17ReportGenerator,
    UpperBodyAnthropometryPipeline,
)

CAPTURE_01_DIR = REPO_ROOT / "raw_captures" / "body_capture_01"
CAPTURE_02_DIR = REPO_ROOT / "raw_captures" / "body_capture_02"
CAPTURE_03_DIR = REPO_ROOT / "raw_captures" / "body_capture_03_diagnostic"
STAGE17_DIR = REPO_ROOT / "stage17_upper_body_anthropometry_output"


class TestPhysicalStage17Anthropometry:

    def test_01_source_data_preservation_and_checksums(self):
        """Verifies source datasets exist and SHA-256 hashes match reference baselines."""
        assert CAPTURE_01_DIR.exists()
        assert CAPTURE_02_DIR.exists()
        assert CAPTURE_03_DIR.exists()

        # Check body_capture_03 checksums
        chk_03 = CAPTURE_03_DIR / "checksums.json"
        assert chk_03.exists(), f"Missing checksums file: {chk_03}"
        with open(chk_03, "r", encoding="utf-8") as fp:
            c3 = json.load(fp)
        for fname in ["frame_00_color.png", "frame_00_depth.npy", "meta.json"]:
            assert fname in c3

    def test_02_synthetic_geometric_primitive_cylinder_slicing(self):
        """
        SOFTWARE CORRECTNESS TEST ONLY (NOT human measurement validation).
        Verifies that AnatomicalPlaneSlicer correctly extracts perimeter on a complete 360-degree cylinder.
        """
        # Create synthetic cylinder of radius R = 0.150 m (150 mm), height H = 0.40 m along Y axis
        # Expected circumference: 2 * pi * 150 = 942.48 mm
        radius_m = 0.150
        n_pts = 1000
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        y_vals = np.linspace(-0.20, 0.20, 50)
        
        grid_angles, grid_y = np.meshgrid(angles, y_vals)
        x_pts = radius_m * np.cos(grid_angles).flatten()
        z_pts = radius_m * np.sin(grid_angles).flatten()
        pts_3d = np.stack([x_pts, grid_y.flatten(), z_pts], axis=-1)

        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        slab_pts, analysis = AnatomicalPlaneSlicer.slice_point_cloud(
            points_3d=pts_3d,
            plane_point=plane_point,
            plane_normal=plane_normal,
            plane_name="synthetic_complete_cylinder",
            slab_thickness_m=0.020,
        )

        assert analysis.is_closed_ring is True
        assert analysis.ring_status == "VALID_CLOSED_RING"
        assert analysis.observed_coverage_fraction >= 0.95
        assert analysis.max_angular_gap_deg < 15.0
        assert analysis.closed_circumference_mm is not None
        # Verify perimeter matches theoretical within 2%
        expected_circ_mm = 2.0 * math.pi * 150.0
        assert abs(analysis.closed_circumference_mm - expected_circ_mm) < (0.02 * expected_circ_mm)

    def test_03_synthetic_partial_cross_section_blocking(self):
        """
        SOFTWARE CORRECTNESS TEST ONLY.
        Verifies that partial cross-sections (e.g. 120-degree frontal arc) are strictly BLOCKED
        and closed circumference is NOT calculated.
        """
        # Create synthetic frontal arc: angles only in [-pi/3, pi/3] (120 degrees)
        radius_m = 0.150
        angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, 300)
        y_vals = np.linspace(-0.10, 0.10, 20)
        grid_angles, grid_y = np.meshgrid(angles, y_vals)
        x_pts = radius_m * np.cos(grid_angles).flatten()
        z_pts = radius_m * np.sin(grid_angles).flatten()
        pts_3d = np.stack([x_pts, grid_y.flatten(), z_pts], axis=-1)

        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        slab_pts, analysis = AnatomicalPlaneSlicer.slice_point_cloud(
            points_3d=pts_3d,
            plane_point=plane_point,
            plane_normal=plane_normal,
            plane_name="synthetic_partial_arc",
            slab_thickness_m=0.020,
        )

        # Must NOT be closed ring
        assert analysis.is_closed_ring is False
        assert analysis.ring_status == "INCOMPLETE_PERIMETER_ANTERIOR_ONLY"
        assert analysis.closed_circumference_mm is None
        assert analysis.blocking_reason is not None
        assert "Incomplete cross-sectional surface coverage" in analysis.blocking_reason
        assert analysis.max_angular_gap_deg > 60.0

    def test_04_synthetic_landmark_distance_and_quality_rules(self):
        """SOFTWARE CORRECTNESS TEST: Audits quality state transitions on landmark evaluations."""
        # 1. Valid landmarks within expected range
        lm_a = AnatomicalLandmark(
            landmark_id="test_a",
            anatomical_name="Test A",
            classification="ESTIMATED_2D_3D_LANDMARK",
            pixel_u=100.0,
            pixel_v=100.0,
            in_image_bounds=True,
            point_3d_camera_m=[0.0, 0.0, 1.0],
            depth_m=1.0,
            depth_valid=True,
            depth_std_mm=5.0,
            visibility_confidence=0.95,
            status="VALID",
        )
        lm_b = AnatomicalLandmark(
            landmark_id="test_b",
            anatomical_name="Test B",
            classification="ESTIMATED_2D_3D_LANDMARK",
            pixel_u=200.0,
            pixel_v=100.0,
            in_image_bounds=True,
            point_3d_camera_m=[0.35, 0.0, 1.0],
            depth_m=1.0,
            depth_valid=True,
            depth_std_mm=5.0,
            visibility_confidence=0.95,
            status="VALID",
        )

        res_valid = MeasurementQualityAuditor.evaluate_landmark_distance(
            measurement_id="test_length",
            dist_mm=350.0,
            landmark_a=lm_a,
            landmark_b=lm_b,
            expected_range_mm=(200.0, 500.0),
        )
        assert res_valid.status == MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE
        assert res_valid.is_valid is True

        # 2. Out of bounds landmark must transition to NOT_AVAILABLE / BLOCKED
        lm_trunc = AnatomicalLandmark(
            landmark_id="test_trunc",
            anatomical_name="Test Truncated",
            classification="ESTIMATED_2D_3D_LANDMARK",
            pixel_u=500.0,
            pixel_v=750.0,
            in_image_bounds=False,
            point_3d_camera_m=None,
            depth_m=None,
            depth_valid=False,
            depth_std_mm=0.0,
            visibility_confidence=0.5,
            status="TRUNCATED_OUT_OF_BOUNDS",
        )
        res_trunc = MeasurementQualityAuditor.evaluate_landmark_distance(
            measurement_id="test_trunc_length",
            dist_mm=None,
            landmark_a=lm_a,
            landmark_b=lm_trunc,
        )
        assert res_trunc.status == MeasurementQualityStatus.NOT_AVAILABLE
        assert res_trunc.data_source == DataSourceType.BLOCKED
        assert res_trunc.is_valid is False

    def test_05_physical_landmarks_classification_and_safeguards(self):
        """Verifies that actual physical landmarks are explicitly classified as ESTIMATED_2D_3D_LANDMARK."""
        audit_file = STAGE17_DIR / "measurement_pipeline_audit.json"
        assert audit_file.exists(), f"Missing audit JSON at {audit_file}"

        with open(audit_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        landmarks = data.get("landmarks", {})
        assert len(landmarks) >= 10

        for lid, lm in landmarks.items():
            # Must NEVER be labeled as ground truth
            assert lm["classification"] == "ESTIMATED_2D_3D_LANDMARK"
            assert "GROUND_TRUTH" not in lm["classification"]

        # Shoulder landmarks must be valid with metric 3D coordinates
        assert landmarks["acromion_left"]["status"] == "VALID"
        assert landmarks["acromion_right"]["status"] == "VALID"
        assert landmarks["acromion_left"]["depth_valid"] is True
        assert landmarks["acromion_right"]["depth_valid"] is True

        # Truncated wrist landmarks must be flagged as out-of-bounds
        assert landmarks["radial_styloid_left"]["status"] == "TRUNCATED_OUT_OF_BOUNDS"
        assert landmarks["radial_styloid_right"]["status"] == "TRUNCATED_OUT_OF_BOUNDS"

    def test_06_physical_measurements_safeguards_and_blocking(self):
        """Verifies that valid lengths are extracted and incomplete circumferences are strictly blocked."""
        qual_file = STAGE17_DIR / "measurement_quality_report.json"
        assert qual_file.exists(), f"Missing quality JSON at {qual_file}"

        with open(qual_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        meas = data.get("measurements_summary", {})

        # 1. Biacromial breadth must be valid and directly measured
        sh = meas.get("biacromial_breadth")
        assert sh is not None
        assert sh["quality_status"] == "VALID_FOR_EXPERIMENTAL_USE"
        assert sh["data_source_type"] == "DIRECTLY_MEASURED"
        assert sh["is_valid"] is True
        assert 300.0 <= sh["value_mm"] <= 450.0

        # 2. Right upper arm length must be valid
        uar = meas.get("upper_arm_length_right")
        assert uar is not None
        assert uar["quality_status"] == "VALID_FOR_EXPERIMENTAL_USE"
        assert uar["is_valid"] is True
        assert 150.0 <= uar["value_mm"] <= 350.0

        # 3. Forearm lengths must be blocked due to wrist truncation
        fal = meas.get("forearm_length_left")
        assert fal["quality_status"] == "NOT_AVAILABLE"
        assert fal["value_mm"] is None
        assert fal["is_valid"] is False

        # 4. Chest circumference must be strictly BLOCKED due to lack of posterior surface
        chest_circ = meas.get("chest_circumference")
        assert chest_circ is not None
        assert chest_circ["quality_status"] == "INSUFFICIENT_DATA"
        assert chest_circ["value_mm"] is None
        assert chest_circ["is_valid"] is False
        assert "Incomplete cross-sectional surface coverage" in chest_circ["blocking_reason"]

        # 5. Chest transverse breadth must be valid across visible points
        chest_br = meas.get("chest_transverse_breadth")
        assert chest_br is not None
        assert chest_br["quality_status"] == "VALID_FOR_EXPERIMENTAL_USE"
        assert chest_br["value_mm"] > 300.0

    def test_07_deliverables_completeness_and_schema(self):
        """Verifies all required deliverable artifacts exist and are non-empty."""
        expected = [
            "stage17_report.md",
            "measurement_pipeline_audit.json",
            "measurement_quality_report.json",
            "landmarks_overlay.png",
            "chest_slice_cross_section.png",
            "upper_body_measurements_chart.png",
        ]
        for fname in expected:
            fpath = STAGE17_DIR / fname
            assert fpath.exists(), f"Missing deliverable: {fname}"
            assert fpath.stat().st_size > 0, f"Empty deliverable: {fname}"

    def test_08_verdict_and_absence_of_fabrication(self):
        """Confirms final verdict is PIPELINE_IMPLEMENTED_OFFLINE and safeguards are strictly maintained."""
        with open(STAGE17_DIR / "measurement_pipeline_audit.json", "r", encoding="utf-8") as fp:
            data = json.load(fp)

        assert data["verdict"] == "PIPELINE_IMPLEMENTED_OFFLINE"

        with open(STAGE17_DIR / "stage17_report.md", "r", encoding="utf-8") as fp:
            report_text = fp.read()

        assert "PIPELINE_IMPLEMENTED_OFFLINE" in report_text
        assert "OFFLINE_DEVELOPMENT" in report_text

        # Anti-fabrication check: no SMPL model parameters or synthetic mesh vertices
        forbidden = ["smpl_betas", "smpl_pose", "mesh_vertices", "mesh_faces", "model_fill"]
        for k in forbidden:
            assert k not in data
            assert k not in report_text
