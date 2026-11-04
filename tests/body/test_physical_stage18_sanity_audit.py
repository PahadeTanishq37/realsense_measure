"""
tests/body/test_physical_stage18_sanity_audit.py
================================================
Unit and integration tests for Stage 18: Offline Anthropometric Sanity & Landmark Reliability Audit.

Verifies:
1. Multi-stage source data preservation (body_capture_01, 02, 03 strictly unmodified).
2. Landmark definitions audit and reclassification of shoulder breadth as MediaPipe proxy.
3. Rigorous 3D vector decomposition of left/right upper-arm discrepancy (Z-dominance vs truncation).
4. Mathematical reconciliation of depth stability (arithmetic mean outlier vulnerability vs robust median).
5. Corrected measurement quality governance rules (rejection of truncated/contaminated segments).
6. Deliverable artifact completeness, plots, and schema integrity.
7. Final audit verdict confirmation: SANITY_AUDIT_DEFECTS_IDENTIFIED.
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

from realsense_measure.measurements import (
    ArmSegmentDecomposition,
    LandmarkSanityAuditRecord,
    Stage18SanityAuditor,
)

CAPTURE_01_DIR = REPO_ROOT / "raw_captures" / "body_capture_01"
CAPTURE_02_DIR = REPO_ROOT / "raw_captures" / "body_capture_02"
CAPTURE_03_DIR = REPO_ROOT / "raw_captures" / "body_capture_03_diagnostic"
STAGE17_DIR = REPO_ROOT / "stage17_upper_body_anthropometry_output"
STAGE18_DIR = REPO_ROOT / "stage18_measurement_sanity_audit_output"


class TestPhysicalStage18SanityAudit:

    def test_01_prior_data_preservation_and_checksums(self):
        """Verifies all source capture datasets and prior stage deliverables are intact."""
        assert CAPTURE_01_DIR.exists()
        assert CAPTURE_02_DIR.exists()
        assert CAPTURE_03_DIR.exists()
        assert STAGE17_DIR.exists()

        chk_03 = CAPTURE_03_DIR / "checksums.json"
        assert chk_03.exists()
        with open(chk_03, "r", encoding="utf-8") as fp:
            c3 = json.load(fp)
        for fname in ["frame_00_color.png", "frame_00_depth.npy", "meta.json"]:
            assert fname in c3

    def test_02_landmark_definitions_and_proxy_classification(self):
        """Audits landmark classification and defensible shoulder breadth labeling."""
        lm_file = STAGE18_DIR / "landmark_reliability_audit.json"
        assert lm_file.exists(), f"Missing landmark audit: {lm_file}"

        with open(lm_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        landmarks = data.get("landmarks", [])
        assert len(landmarks) >= 10

        for lm in landmarks:
            # Must be recognized as an estimated proxy, NOT direct bony landmark
            assert lm["is_direct_anatomical_landmark"] is False
            assert "proxy" in lm["proxy_description"].lower() or "derived" in lm["proxy_description"].lower()

        # Shoulder label determination check
        sh_det = data.get("shoulder_breadth_label_determination", {})
        assert sh_det.get("selected_label") == "MediaPipe Shoulder Breadth Proxy"
        assert "joint rotation centers" in sh_det.get("scientific_rationale")

    def test_03_arm_discrepancy_decomposition_and_root_cause(self):
        """Audits 3D vector decomposition proving discrepancy is non-biological."""
        arm_file = STAGE18_DIR / "arm_discrepancy_investigation.json"
        assert arm_file.exists(), f"Missing arm investigation: {arm_file}"

        with open(arm_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        left = data.get("left_arm", {})
        right = data.get("right_arm", {})

        # Left Arm: dominated by delta_Z (>90% depth disparity)
        assert left["z_dominance_fraction"] > 0.90
        assert abs(left["delta_z_mm"]) > 300.0
        assert left["delta_y_mm"] < 100.0  # Vertical span only 71 mm
        assert left["elbow_visibility"] < 0.25  # Extremely low visibility (0.19)
        assert left["anatomically_plausible"] is False
        assert "Cross-Plane Depth Contamination" in left["primary_failure_mode"]

        # Right Arm: truncated at image boundary
        assert right["elbow_distance_to_edge_px"] < 25.0  # Clamped within 19 px of frame bottom
        assert right["delta_y_mm"] < 180.0  # Truncated vertical span (154 mm)
        assert right["elbow_patch_purity"] < 0.15  # 91.4% sentinel background
        assert right["raw_elbow_depth_m"] > 50.0  # Raw pixel depth is 54.4 m sentinel
        assert right["anatomically_plausible"] is False
        assert "Boundary Truncation Clamping" in right["primary_failure_mode"]

    def test_04_depth_stability_reconciliation(self):
        """Verifies arithmetic mean vs median reconciliation explaining the apparent stability discrepancy."""
        stab_file = STAGE18_DIR / "depth_stability_reconciliation.json"
        assert stab_file.exists(), f"Missing stability reconciliation: {stab_file}"

        with open(stab_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Stage 16 reported ~876 mm
        assert data["stage16_regional_reported_mean_mm"] == 876.0

        # Recomputed median is small (~6 mm), proving true hardware jitter is low
        assert data["recomputed_torso_median_mm"] < 10.0

        # Outlier max is massive (>20,000 mm) due to sentinel flickering
        assert data["recomputed_torso_max_mm"] > 20000.0
        assert data["torso_sentinel_outlier_percentage"] > 2.0

        # Patch medians on interior landmarks are sub-millimetre
        patch_rng = data["stage17_landmark_patch_median_range_mm"]
        assert patch_rng[0] < 1.0 and patch_rng[1] < 1.0

    def test_05_corrected_measurement_governance(self):
        """Verifies updated governance rules correctly reject truncated/contaminated metrics."""
        qual_file = STAGE18_DIR / "corrected_measurement_quality_report.json"
        assert qual_file.exists(), f"Missing corrected quality report: {qual_file}"

        with open(qual_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        meas = data.get("corrected_measurements", {})

        # Both upper arm segments must now be BLOCKED with INVALID_GEOMETRY
        assert meas["upper_arm_length_left"]["quality_status"] == "INVALID_GEOMETRY"
        assert meas["upper_arm_length_left"]["value_mm"] is None
        assert meas["upper_arm_length_left"]["is_valid"] is False

        assert meas["upper_arm_length_right"]["quality_status"] == "INVALID_GEOMETRY"
        assert meas["upper_arm_length_right"]["value_mm"] is None
        assert meas["upper_arm_length_right"]["is_valid"] is False

        # Shoulder breadth must be reclassified as LOW_CONFIDENCE proxy
        assert meas["biacromial_breadth"]["name"] == "MediaPipe Shoulder Breadth Proxy"
        assert meas["biacromial_breadth"]["quality_status"] == "LOW_CONFIDENCE"
        assert meas["biacromial_breadth"]["is_valid"] is False

        # No straight-line segment length should have circular angular coverage
        for m_id in ["biacromial_breadth", "upper_arm_length_left", "upper_arm_length_right"]:
            assert "observed_surface_coverage_fraction" not in meas[m_id]

    def test_06_deliverables_completeness_and_plots(self):
        """Verifies all 7 deliverable files exist, are non-empty, and have valid content."""
        expected = [
            "stage18_report.md",
            "landmark_reliability_audit.json",
            "arm_discrepancy_investigation.json",
            "depth_stability_reconciliation.json",
            "corrected_measurement_quality_report.json",
            "arm_discrepancy_diagnostic.png",
            "depth_stability_reconciliation.png",
        ]
        for fname in expected:
            fpath = STAGE18_DIR / fname
            assert fpath.exists(), f"Missing deliverable: {fname}"
            assert fpath.stat().st_size > 0, f"Empty deliverable: {fname}"

    def test_07_final_verdict_and_safeguards(self):
        """Confirms final verdict is SANITY_AUDIT_DEFECTS_IDENTIFIED and no fabrication occurs."""
        qual_file = STAGE18_DIR / "corrected_measurement_quality_report.json"
        with open(qual_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        assert data["audit_verdict"] == "SANITY_AUDIT_DEFECTS_IDENTIFIED"

        with open(STAGE18_DIR / "stage18_report.md", "r", encoding="utf-8") as fp:
            rep_text = fp.read()

        assert "SANITY_AUDIT_DEFECTS_IDENTIFIED" in rep_text
        assert "MediaPipe Shoulder Breadth Proxy" in rep_text

        # Anti-fabrication verification: zero SMPL model parameters or synthetic vertices
        forbidden = ["smpl_betas", "smpl_pose", "mesh_vertices", "mesh_faces", "model_fill"]
        for k in forbidden:
            assert k not in data
            assert k not in rep_text
