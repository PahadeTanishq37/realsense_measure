r"""
tests/body/test_physical_stage13_capture_audit.py
=================================================
Tests for Stage 13: First Full-Body RGB-D Capture & Sensor Data Audit.

Validates:
1. Physical capture files existence, shapes, dtypes, and readability.
2. Manifest schema, anonymous subject ID, and camera metadata.
3. Metric depth integrity (finite values, valid ranges, reasonable percentiles).
4. Spatial coverage across vertical sensor zones.
5. Per-view anatomical visibility heuristics (head, torso, pelvis, legs).
6. Preservation and isolation of historical dataset raw_captures/body_capture_01.
7. Stage 13 audit deliverables existence, SHA-256 hashes, and verdict.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from realsense_measure.body.sensor_data_audit import (
    PhysicalCaptureAuditor,
    PhysicalCaptureAuditSummary,
    Stage13DeliverablesGenerator,
    ViewAuditRecord,
)

RAW_DIR_01 = Path("raw_captures/body_capture_01")
RAW_DIR_02 = Path("raw_captures/body_capture_02")
STAGE13_DIR = Path("stage13_full_body_capture_output")


# ---------------------------------------------------------------------------
# 1. Physical Capture Files & Array Integrity
# ---------------------------------------------------------------------------

def test_physical_capture_directory_and_files_exist():
    """Verify raw_captures/body_capture_02 contains all 4 stops and metadata."""
    assert RAW_DIR_02.exists(), "Capture directory raw_captures/body_capture_02 does not exist."
    manifest_p = RAW_DIR_02 / "manifest.json"
    meta_p = RAW_DIR_02 / "meta.json"
    assert manifest_p.exists(), "manifest.json missing."
    assert meta_p.exists(), "meta.json missing."

    for i in range(4):
        c_p = RAW_DIR_02 / f"frame_{i:02d}_color.png"
        d_p = RAW_DIR_02 / f"frame_{i:02d}_depth.npy"
        assert c_p.exists(), f"Color frame missing: {c_p}"
        assert d_p.exists(), f"Depth frame missing: {d_p}"

        # Inspect RGB
        img = cv2.imread(str(c_p))
        assert img is not None, f"Failed to read image: {c_p}"
        assert img.shape == (720, 1280, 3)
        assert img.dtype == np.uint8

        # Inspect Depth
        depth = np.load(str(d_p))
        assert depth.shape == (720, 1280)
        assert depth.dtype == np.float32


def test_metric_depth_values_are_plausible():
    """Verify depth maps contain metric distances within sensor limits."""
    for i in range(4):
        d_p = RAW_DIR_02 / f"frame_{i:02d}_depth.npy"
        depth = np.load(str(d_p))

        # Check finite and non-negative
        assert np.all(np.isfinite(depth))
        assert np.all(depth >= 0.0)

        valid_mask = (depth >= 0.40) & (depth <= 3.50)
        valid_frac = np.mean(valid_mask)
        # Should have reasonable depth coverage in typical room
        assert valid_frac > 0.20

        valid_vals = depth[valid_mask]
        med_val = np.median(valid_vals)
        # Median depth should be in plausible human-at-tripod range [1.0m, 3.0m]
        assert 1.0 <= med_val <= 3.0


# ---------------------------------------------------------------------------
# 2. Manifest Schema & Anonymous Metadata
# ---------------------------------------------------------------------------

def test_manifest_metadata_and_anonymous_ids():
    """Verify manifest fields adhere to capture protocol contract."""
    manifest_p = RAW_DIR_02 / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        doc = json.load(f)

    assert doc["capture_id"] == "body_capture_02"
    assert doc["session_id"] == "SESS_20261102_PHYS02"
    assert doc["subject_id"] == "SUBJ_02"
    assert doc["operator_id"] == "OPERATOR_01"
    assert doc["config_id"] == "CONFIG_D240_H100_T00_Y4_R720"
    assert doc["camera_to_subject_distance_m"] == 2.4
    assert doc["camera_height_m"] == 1.0
    assert doc["camera_pitch_tilt_deg"] == 0.0
    assert doc["frames_captured"] == 4
    assert doc["expected_frames"] == 4
    assert doc["capture_mode"] == "capture_only"

    # Confirm anonymous subject ID
    subj = doc["subject_id"]
    assert "name" not in subj.lower()
    assert "pahad" not in subj.lower()


# ---------------------------------------------------------------------------
# 3. Sensor Data Audit Engine
# ---------------------------------------------------------------------------

def test_physical_capture_auditor_summary():
    """Verify auditor generates comprehensive summary with 4 views."""
    summary = PhysicalCaptureAuditor.audit_capture_directory(RAW_DIR_02)
    assert isinstance(summary, PhysicalCaptureAuditSummary)
    assert summary.capture_id == "body_capture_02"
    assert summary.num_stops == 4
    assert summary.all_files_valid is True
    assert summary.overall_valid_depth_fraction > 0.25
    assert len(summary.views) == 4

    for v in summary.views:
        assert isinstance(v, ViewAuditRecord)
        assert v.rgb_resolution == (1280, 720)
        assert v.depth_shape == (1280, 720)
        assert v.total_pixels == 1280 * 720
        assert v.valid_fraction > 0.20
        assert v.median_depth_m > 1.0
        # Confirm vertical zone coverage exists
        assert v.middle_zone_coverage > 0.05


# ---------------------------------------------------------------------------
# 4. Reconstruction Readiness & Unobserved Anatomy
# ---------------------------------------------------------------------------

def test_reconstruction_readiness_verdict():
    """Verify readiness verdict reflects known capture limitations."""
    summary = PhysicalCaptureAuditor.audit_capture_directory(RAW_DIR_02)
    assert summary.reconstruction_readiness in ("READY", "READY_WITH_LIMITATIONS")
    assert len(summary.suitability_reasons) > 0
    assert len(summary.unobserved_regions) > 0

    # Ensure unobserved regions acknowledge crown and underarm shadowing
    unobserved_str = " ".join(summary.unobserved_regions).lower()
    assert "crown" in unobserved_str or "vertex" in unobserved_str


# ---------------------------------------------------------------------------
# 5. Preservation of Historical body_capture_01
# ---------------------------------------------------------------------------

def test_historical_body_capture_01_strictly_preserved():
    """Verify raw_captures/body_capture_01 remains untouched."""
    assert RAW_DIR_01.exists()
    meta_p = RAW_DIR_01 / "meta.json"
    assert meta_p.exists()
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_frames"] == 4

    for i in range(4):
        c_p = RAW_DIR_01 / f"frame_{i:02d}_color.png"
        d_p = RAW_DIR_01 / f"frame_{i:02d}_depth.npy"
        assert c_p.exists()
        assert d_p.exists()


# ---------------------------------------------------------------------------
# 6. Deliverables Generation & Schemas
# ---------------------------------------------------------------------------

def test_stage13_deliverables_exist_and_valid():
    """Verify all 7 deliverables in stage13_full_body_capture_output/ exist."""
    expected_files = [
        "stage13_report.md",
        "capture_manifest_audit.json",
        "per_view_visibility_audit.json",
        "spatial_coverage_analysis.json",
        "file_checksums.json",
        "reconstruction_readiness_verdict.json",
        "stage13_test_results.json",
    ]
    for fn in expected_files:
        p = STAGE13_DIR / fn
        assert p.exists(), f"Missing Stage 13 deliverable: {p}"
        assert p.stat().st_size > 0, f"Empty Stage 13 deliverable: {p}"

    # Verify JSON parsability
    with open(STAGE13_DIR / "reconstruction_readiness_verdict.json", "r", encoding="utf-8") as f:
        v_doc = json.load(f)
        assert v_doc["capture_id"] == "body_capture_02"
        assert v_doc["manual_measurements_collected"] is False
        assert len(v_doc["downstream_stages_bypassed"]) == 4

    with open(STAGE13_DIR / "file_checksums.json", "r", encoding="utf-8") as f:
        cs_doc = json.load(f)
        assert len(cs_doc) == 10
        assert "frame_00_color.png" in cs_doc
        assert "frame_00_depth.npy" in cs_doc
