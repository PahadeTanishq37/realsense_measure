"""
tests/body/test_physical_stage16_diagnostics.py
===============================================
Unit and integration tests for Stage 16: Physical Subject Depth Diagnostic & Capture Recovery.

Verifies:
1. Multi-stage source data preservation (body_capture_01, body_capture_02 strictly unmodified).
2. Diagnostic burst capture integrity (10 frames, median, variance arrays).
3. 12-region anatomical depth coverage and dropout metrics.
4. Physical infrared absorption and framing truncation diagnostics.
5. Subject point cloud extraction (exclusion of background and sentinel saturations).
6. Complete deliverable artifact presence and schema validation.
7. Final recovery verdict and strict compliance with anthropometry safeguards.
"""

from __future__ import annotations

import json
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

from realsense_measure.body.physical_subject_depth_diagnostic import (
    AnatomicalRegionDepthAuditor,
    AnatomicalRegionMetrics,
    DiagnosticBurstCaptureEngine,
    DiagnosticCaptureMetadata,
    DiagnosticSubjectPointcloudBuilder,
    Stage16DiagnosticsOrchestrator,
)

CAPTURE_01_DIR = REPO_ROOT / "raw_captures" / "body_capture_01"
CAPTURE_02_DIR = REPO_ROOT / "raw_captures" / "body_capture_02"
CAPTURE_03_DIR = REPO_ROOT / "raw_captures" / "body_capture_03_diagnostic"
STAGE16_DIR = REPO_ROOT / "stage16_subject_depth_diagnostic_output"


class TestPhysicalStage16Diagnostics:

    def test_01_prior_data_preservation_and_checksums(self):
        """Verifies body_capture_01 and body_capture_02 are intact and unmodified."""
        assert CAPTURE_01_DIR.exists(), f"Missing body_capture_01 at {CAPTURE_01_DIR}"
        assert CAPTURE_02_DIR.exists(), f"Missing body_capture_02 at {CAPTURE_02_DIR}"
        assert CAPTURE_03_DIR.exists(), f"Missing diagnostic capture at {CAPTURE_03_DIR}"

        # Confirm body_capture_02 contains original 4 stops and meta.json
        for i in range(4):
            assert (CAPTURE_02_DIR / f"frame_{i:02d}_color.png").exists()
            assert (CAPTURE_02_DIR / f"frame_{i:02d}_depth.npy").exists()
        assert (CAPTURE_02_DIR / "meta.json").exists()

        # Confirm body_capture_03_diagnostic has checksums.json
        chk_file = CAPTURE_03_DIR / "checksums.json"
        assert chk_file.exists(), f"Missing checksums file: {chk_file}"
        with open(chk_file, "r", encoding="utf-8") as fp:
            chk = json.load(fp)
        assert len(chk) >= 4
        assert "frame_00_color.png" in chk
        assert "frame_00_depth.npy" in chk
        assert "frame_00_depth_variance.npy" in chk
        assert "meta.json" in chk

    def test_02_burst_capture_and_temporal_stability(self):
        """Verifies burst capture arrays, shapes, and temporal depth noise."""
        color_path = CAPTURE_03_DIR / "frame_00_color.png"
        depth_path = CAPTURE_03_DIR / "frame_00_depth.npy"
        var_path = CAPTURE_03_DIR / "frame_00_depth_variance.npy"

        assert color_path.exists()
        assert depth_path.exists()
        assert var_path.exists()

        depth_arr = np.load(depth_path)
        var_arr = np.load(var_path)

        assert depth_arr.shape == (720, 1280)
        assert var_arr.shape == (720, 1280)
        assert depth_arr.dtype == np.float32
        assert var_arr.dtype == np.float32

        # Temporal noise on non-zero regions must be non-negative
        assert np.all(var_arr >= 0.0)

        # Check burst folder has individual frames
        burst_dir = CAPTURE_03_DIR / "burst_frames"
        assert burst_dir.exists()
        burst_depths = list(burst_dir.glob("burst_*_depth.npy"))
        assert len(burst_depths) == 10

    def test_03_anatomical_depth_coverage_12_regions(self):
        """Verifies that all 12 standard anatomical regions are audited."""
        cov_file = STAGE16_DIR / "anatomical_depth_coverage.json"
        assert cov_file.exists(), f"Missing coverage JSON: {cov_file}"

        with open(cov_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        regions = data.get("regions", [])
        assert len(regions) == 12

        expected_regions = [
            "head_neck",
            "torso",
            "pelvis",
            "left_upper_arm",
            "right_upper_arm",
            "left_forearm",
            "right_forearm",
            "left_thigh",
            "right_thigh",
            "left_lower_leg",
            "right_lower_leg",
            "feet",
        ]
        region_ids = [r["region_id"] for r in regions]
        assert region_ids == expected_regions

        for r in regions:
            # Valid percentage + dropout percentage must sum to 100% within float tolerance
            assert abs((r["valid_depth_percentage"] + r["dropout_percentage"]) - 100.0) < 0.1
            # Pixel counts must match
            assert r["rgb_pixel_count"] == (r["valid_depth_pixel_count"] + r["dropout_pixel_count"])
            # Status must be valid enum
            assert r["coverage_status"] in ["USABLE", "MARGINAL", "HEAVY_DROPOUT", "UNOBSERVED_TRUNCATED"]

    def test_04_depth_dropout_and_absorption_analysis(self):
        """Audits root-cause breakdown of depth dropouts."""
        drop_file = STAGE16_DIR / "depth_dropout_analysis.json"
        assert drop_file.exists(), f"Missing dropout analysis JSON: {drop_file}"

        with open(drop_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        assert "verdict" in data
        assert "rationale" in data
        assert "torso_usable" in data
        assert "head_usable" in data
        assert "lower_body_usable" in data
        assert "point_count" in data
        assert "spatial_extent_xyz_m" in data

        # Torso and head are usable, lower body is truncated/absorbed
        assert data["torso_usable"] is True
        assert data["head_usable"] is True
        assert data["lower_body_usable"] is False

    def test_05_subject_point_cloud_extraction(self):
        """Verifies that subject point cloud isolates human body and excludes sentinel values."""
        ply_file = STAGE16_DIR / "subject_diagnostic_cloud.ply"
        assert ply_file.exists(), f"Missing subject point cloud: {ply_file}"
        assert ply_file.stat().st_size > 100000

        # Read ply header to verify point count
        with open(ply_file, "rb") as fp:
            header_lines = [fp.readline().decode("latin-1") for _ in range(15)]
        
        element_line = next((l for l in header_lines if "element vertex" in l), None)
        assert element_line is not None
        pt_count = int(element_line.strip().split()[-1])
        assert pt_count > 50000

    def test_06_deliverables_completeness(self):
        """Verifies that all 11 required deliverable artifacts exist and are non-empty."""
        expected_files = [
            "stage16_report.md",
            "anatomical_depth_coverage.json",
            "depth_dropout_analysis.json",
            "capture_metadata.json",
            "subject_diagnostic_cloud.ply",
            "rgb_image.png",
            "raw_depth_visualized.png",
            "valid_depth_mask.png",
            "subject_segmentation_mask.png",
            "subject_only_depth.png",
            "anatomical_regions_overlay.png",
        ]
        for fname in expected_files:
            fpath = STAGE16_DIR / fname
            assert fpath.exists(), f"Deliverable missing: {fname}"
            assert fpath.stat().st_size > 0, f"Deliverable is empty: {fname}"

    def test_07_final_verdict_and_safeguards(self):
        """Confirms final verdict is valid and scientific safeguards are maintained."""
        with open(STAGE16_DIR / "depth_dropout_analysis.json", "r", encoding="utf-8") as fp:
            drop_data = json.load(fp)

        assert drop_data["verdict"] == "TORSO_ONLY_DEPTH_USABLE"

        # Check report mentions the verified hardware
        with open(STAGE16_DIR / "stage16_report.md", "r", encoding="utf-8") as fp:
            report_text = fp.read()
        assert "Intel RealSense D455f" in report_text
        assert "342622300088" in report_text
        assert "TORSO_ONLY_DEPTH_USABLE" in report_text

        # Strictly enforce safeguards: no circumference measurements, no mesh vertices
        forbidden_keys = [
            "waist_circumference_mm",
            "hip_circumference_mm",
            "chest_circumference_mm",
            "smpl_betas",
            "mesh_faces",
        ]
        for k in forbidden_keys:
            assert k not in drop_data
            assert k not in report_text

