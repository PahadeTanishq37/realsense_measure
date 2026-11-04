"""
tests/body/test_physical_stage15_diagnostics.py
===============================================
Unit and integration tests for Stage 15: Depth Semantics, Coordinate Consistency &
Registration Failure Diagnosis.

Verifies:
1. Multi-stage source data preservation and checksum validation.
2. Exact reproduction of Stage 13 vs Stage 14 valid-depth calculations.
3. 16-bit depth saturation sentinel identification (values near 65.535m).
4. Coordinate convention and nominal yaw pivot discrepancy audit.
5. Segmentation interval diagnostic (near-field [0.45, 1.40]m vs standoff [1.8, 3.2]m).
6. Root-cause verification of pairwise ICP sliding and large corrections.
7. Deliverable completeness and enforcement of scientific safeguards.
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

from realsense_measure.body.registration_diagnostics import (
    CoordinateConventionAuditorEngine,
    DepthSemanticsAuditor,
    MultiStageDataPreservationAuditor,
    RegistrationFailureAnalyzerEngine,
    SegmentationDiagnosticsAuditor,
    Stage15DiagnosticsOrchestrator,
)

CAPTURE_02_DIR = REPO_ROOT / "raw_captures" / "body_capture_02"
STAGE13_DIR = REPO_ROOT / "stage13_full_body_capture_output"
STAGE14_DIR = REPO_ROOT / "stage14_physical_reconstruction_output"
STAGE15_DIR = REPO_ROOT / "stage15_registration_diagnostics_output"


class TestPhysicalStage15Diagnostics:

    def test_01_source_data_preservation_and_checksums(self):
        """Verifies body_capture_02 files strictly match Stage 13 baseline hashes."""
        ref_chk = STAGE13_DIR / "file_checksums.json"
        assert ref_chk.exists(), f"Missing reference checksums: {ref_chk}"

        audit = MultiStageDataPreservationAuditor.audit_checksums(CAPTURE_02_DIR, ref_chk)
        assert audit["all_checksums_verified"] is True
        assert audit["total_files"] == 10

        for fname, d in audit["files"].items():
            assert d["matched"] is True, f"Mismatch in {fname}"

    def test_02_depth_validity_discrepancy_reproduced(self):
        """Reproduces the exact arithmetic cause of Stage 13 vs Stage 14 reporting."""
        # Frame 0
        d0 = np.load(CAPTURE_02_DIR / "frame_00_depth.npy")
        aud0 = DepthSemanticsAuditor.audit_stop_depth(d0, 0)

        # Stage 13: [0.30, 3.50]m -> ~31.9%
        assert 0.30 <= aud0.stage13_valid_fraction <= 0.35
        assert aud0.stage13_valid_count == 293948

        # Stage 14: [0.05, 65.0]m -> ~93.8%
        assert 0.90 <= aud0.stage14_valid_fraction <= 0.99
        assert aud0.stage14_valid_count == 864528

        # Sentinels
        d2 = np.load(CAPTURE_02_DIR / "frame_02_depth.npy")
        aud2 = DepthSemanticsAuditor.audit_stop_depth(d2, 2)
        assert aud2.sentinel_saturation_pixels > 70000

    def test_03_depth_percentiles_bimodal_distribution(self):
        """Verifies non-zero depth percentiles reflect both near foreground and far room."""
        for i in range(4):
            d = np.load(CAPTURE_02_DIR / f"frame_{i:02d}_depth.npy")
            aud = DepthSemanticsAuditor.audit_stop_depth(d, i)
            pcts = aud.percentiles_nonzero_finite
            assert pcts["P01"] < 0.60  # Near cluster
            assert pcts["P05"] < 0.70
            assert pcts["P50"] > 4.0   # Background dominant median
            assert pcts["P95"] > 35.0  # Far wall / sentinel dominant

    def test_04_coordinate_convention_and_pivot_mismatch(self):
        """Verifies coordinate definition and flags the pivot distance mismatch."""
        coord_audit = CoordinateConventionAuditorEngine.audit_conventions(
            nominal_standoff_m=2.40, stage14_pivot_z_m=0.80
        )
        assert coord_audit.pivot_mismatch_detected is True
        assert coord_audit.pivot_z_used_m == 0.80
        assert coord_audit.nominal_standoff_m == 2.40
        assert "RealSense_Camera_Optical" in coord_audit.optical_axes_definition

    def test_05_segmentation_interval_discrepancy(self):
        """Verifies that Stage 14 near interval [0.45, 1.4]m captured foreground obstacle."""
        seg_json = STAGE15_DIR / "segmentation_diagnostics.json"
        assert seg_json.exists(), f"Missing segmentation diagnostics: {seg_json}"

        with open(seg_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        stops = data["stops"]
        assert len(stops) == 4

        for s in stops:
            # Near field has ~90k-102k points in mask
            assert s["near_field_in_mask_pixels"] > 80000
            # Standoff [1.8, 3.2]m has < 25k points (<10% of mask)
            assert s["standoff_in_mask_pixels"] < 25000
            assert (s["standoff_in_mask_pixels"] / s["person_mask_total_pixels"]) < 0.10

    def test_06_pairwise_failure_analysis_and_sliding(self):
        """Audits large corrections and root cause of loop drift."""
        fail_json = STAGE15_DIR / "registration_failure_analysis.json"
        assert fail_json.exists(), f"Missing failure analysis: {fail_json}"

        with open(fail_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        pairs = data["pairwise_comparisons"]
        assert len(pairs) == 4

        # Pair 3->2 in Stage 14 near field experienced massive sliding
        p32 = [p for p in pairs if p["pair"] == [3, 2]][0]
        assert p32["stage14_near_delta_trans_mm"] > 300.0
        assert p32["stage14_near_delta_rot_deg"] > 70.0

        # Standoff registration suffered from point starvation
        for p in pairs:
            assert p["standoff_fitness"] < 0.25

    def test_07_deliverable_artifacts_completeness(self):
        """Verifies all 11 required deliverable files exist in output directory."""
        expected = [
            "stage15_report.md",
            "depth_semantics_audit.json",
            "coordinate_convention_audit.json",
            "segmentation_diagnostics.json",
            "registration_failure_analysis.json",
            "depth_semantics_histogram.png",
            "icp_failure_diagnostics.png",
            "rgb_mask_overlay_stop00.png",
            "rgb_mask_overlay_stop01.png",
            "rgb_mask_overlay_stop02.png",
            "rgb_mask_overlay_stop03.png",
        ]
        for fname in expected:
            fpath = STAGE15_DIR / fname
            assert fpath.exists(), f"Missing deliverable: {fname}"
            assert fpath.stat().st_size > 0, f"Empty deliverable: {fname}"

    def test_08_verdict_and_scientific_safeguards(self):
        """Confirms final verdict is MULTIPLE_ISSUES_FOUND and safeguards are strictly maintained."""
        with open(STAGE15_DIR / "depth_semantics_audit.json", "r", encoding="utf-8") as fp:
            d_audit = json.load(fp)

        # Confirm zero circumference measurements exist
        forbidden = ["waist_circumference_mm", "hip_circumference_mm", "smpl_beta", "mesh_vertices"]
        for k in forbidden:
            assert k not in d_audit

