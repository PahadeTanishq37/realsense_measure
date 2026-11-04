"""
tests/body/test_physical_stage14_reconstruction.py
==================================================
Unit and integration tests for Stage 14: Physical Multi-View Registration &
Reconstruction Audit.

Verifies:
1. Source data integrity & SHA-256 verification against Stage 13 baseline.
2. Strict preservation of historical body_capture_01 dataset.
3. Pre-registration input audit (metric depth float32, scaling, dimensions).
4. Subject point-cloud extraction (multi-stage segmentation, bounds, density).
5. Mathematical consistency of nominal yaw SE(3) initialization.
6. Pairwise Point-to-Plane ICP convergence and residual metric bounds.
7. Multi-view closed-loop trajectory drift quantification.
8. Artifact completeness (all JSON deliverables, PLY clouds, visualizations).
9. Scientific safeguards enforcement (no circumference slicing, no mesh fitting).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import pytest

# Ensure repository paths are configured
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MEASURE_PKG = REPO_ROOT / "realsense_measure-main"
MEASURE_INNER = MEASURE_PKG / "realsense_measure"

for p in [str(REPO_ROOT), str(MEASURE_PKG), str(MEASURE_INNER)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realsense_measure.body.physical_reconstruction import (
    InputAuditStopRecord,
    LoopClosureRecord,
    MultiViewRegistrationEngine,
    PairwiseRegistrationRecord,
    PerViewCloudAuditRecord,
    PhysicalInputIntegrityAuditor,
    ReconstructionQualityRecord,
    SourceDataIntegrityAuditor,
    Stage14ReconstructionOrchestrator,
    SubjectCloudExtractor,
)

CAPTURE_02_DIR = REPO_ROOT / "raw_captures" / "body_capture_02"
CAPTURE_01_DIR = REPO_ROOT / "raw_captures" / "body_capture_01"
STAGE13_OUT_DIR = REPO_ROOT / "stage13_full_body_capture_output"
STAGE14_OUT_DIR = REPO_ROOT / "stage14_physical_reconstruction_output"
REF_CHECKSUMS_PATH = STAGE13_OUT_DIR / "file_checksums.json"


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestPhysicalStage14Reconstruction:

    def test_01_source_data_checksums_intact(self):
        """Validates that body_capture_02 files strictly match Stage 13 reference hashes."""
        assert CAPTURE_02_DIR.exists(), f"Capture directory missing: {CAPTURE_02_DIR}"
        assert REF_CHECKSUMS_PATH.exists(), f"Reference checksums missing: {REF_CHECKSUMS_PATH}"

        chk_result = SourceDataIntegrityAuditor.verify_capture_checksums(
            CAPTURE_02_DIR, REF_CHECKSUMS_PATH
        )

        assert chk_result["all_checksums_verified"] is True
        assert chk_result["total_files_audited"] == 10

        for fname, details in chk_result["files"].items():
            assert details["matched"] is True, f"Checksum mismatch for {fname}"
            assert details["actual_sha256"] == details["expected_sha256"]

    def test_02_historical_capture_01_preserved(self):
        """Verifies that historical body_capture_01 was untouched and preserved."""
        assert CAPTURE_01_DIR.exists(), "Historical body_capture_01 missing!"
        files_01 = list(CAPTURE_01_DIR.glob("*"))
        assert len(files_01) > 0, "body_capture_01 directory is unexpectedly empty!"

    def test_03_input_array_properties_and_scaling(self):
        """Audits that saved depth arrays are metric float32 and dimensions match RGB."""
        auditor = PhysicalInputIntegrityAuditor(CAPTURE_02_DIR)
        stops = auditor.audit_all_stops()

        assert len(stops) == 4, f"Expected 4 stops, found {len(stops)}"

        for s in stops:
            assert s.color_shape == (720, 1280, 3)
            assert s.depth_shape == (720, 1280)
            assert s.depth_dtype == "float32"
            assert s.is_depth_float32_meters is True
            assert s.valid_depth_fraction > 0.20
            assert s.nan_pixels == 0
            assert s.infinite_pixels == 0
            assert s.min_depth_m > 0.40
            assert s.max_depth_m < 70.0
            # Confirm presence of foreground human subject
            assert s.foreground_pixels_045_140m > 50000
            # Confirm presence of background scene
            assert s.background_pixels_gt_180m > 100000

    def test_04_nominal_yaw_initialization_se3_properties(self):
        """Validates that initial nominal yaw SE(3) matrices have exact SO(3) properties."""
        test_yaws = [(0.0, 0.0), (90.0, 0.0), (180.0, 90.0), (270.0, 180.0), (0.0, 270.0)]
        for src_yaw, tgt_yaw in test_yaws:
            T = MultiViewRegistrationEngine.get_initial_nominal_transform(
                src_yaw, tgt_yaw, pivot_z_m=0.80
            )
            assert T.shape == (4, 4)
            assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])

            R = T[0:3, 0:3]
            # Determinant must be +1 (proper rotation, no reflection)
            det = np.linalg.det(R)
            assert np.isclose(det, 1.0, atol=1e-6)

            # Orthogonality: R.T @ R = I
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)

            # Rotation must be around Y axis (camera vertical)
            delta_rad = np.radians(src_yaw - tgt_yaw)
            expected_R = np.array([
                [np.cos(delta_rad), 0.0, np.sin(delta_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(delta_rad), 0.0, np.cos(delta_rad)]
            ])
            assert np.allclose(R, expected_R, atol=1e-6)

    def test_05_subject_cloud_extraction_stages(self):
        """Audits multi-stage subject point-cloud extraction filtering."""
        auditor = PhysicalInputIntegrityAuditor(CAPTURE_02_DIR)
        stops = auditor.audit_all_stops()
        extractor = SubjectCloudExtractor(auditor.intrinsics, voxel_size_m=0.01)

        import cv2
        c_img = cv2.imread(str(CAPTURE_02_DIR / stops[0].color_filename))
        d_arr = np.load(str(CAPTURE_02_DIR / stops[0].depth_filename))

        raw_cl, sub_cl, audit = extractor.extract_clouds_for_view(
            c_img, d_arr, stop_index=0, nominal_yaw_deg=0.0
        )

        assert audit.raw_valid_points > 100000
        assert audit.subject_segmented_points > 50000
        assert audit.outlier_filtered_points <= audit.subject_segmented_points
        assert audit.downsampled_points > 1000
        assert audit.subject_extent_xyz_m[0] > 0.4  # Lateral torso/arm width > 40 cm
        assert audit.subject_extent_xyz_m[1] > 0.5  # Vertical torso extent > 50 cm
        assert audit.subject_extent_xyz_m[2] < 1.0  # Anteroposterior thickness < 1 m

    def test_06_pairwise_icp_metrics(self):
        """Verifies pairwise ICP produces plausible fitness and inlier RMSE."""
        # Load exported registered results
        reg_json_path = STAGE14_OUT_DIR / "registration_results.json"
        assert reg_json_path.exists(), f"Registration results missing: {reg_json_path}"

        with open(reg_json_path, "r", encoding="utf-8") as fp:
            reg_data = json.load(fp)

        adj_pairs = reg_data["adjacent_pairwise_registrations"]
        assert len(adj_pairs) == 4

        for pair in adj_pairs:
            assert pair["refined_fitness"] > 0.10, f"Pair {pair['source_index']}->{pair['target_index']} fitness too low"
            assert 20.0 <= pair["refined_inlier_rmse_mm"] <= 45.0
            assert pair["refined_correspondences"] > 300
            assert "median_mm" in pair["residual_distribution_mm"]
            assert pair["residual_distribution_mm"]["median_mm"] <= 35.0

    def test_07_closed_loop_drift_quantification(self):
        """Audits that multi-view loop closure drift is accurately quantified."""
        reg_json_path = STAGE14_OUT_DIR / "registration_results.json"
        with open(reg_json_path, "r", encoding="utf-8") as fp:
            reg_data = json.load(fp)

        loop = reg_data["loop_closure_drift"]
        # Drift must be quantified and positive
        assert loop["loop_translation_drift_mm"] > 50.0  # Captures real biomechanical subject shift
        assert loop["loop_rotation_drift_deg"] > 5.0
        assert loop["mean_drift_per_stop_mm"] > 10.0
        assert loop["is_closed_loop_consistent"] is False  # Rigid closed loop is not consistent

    def test_08_reconstruction_artifacts_completeness(self):
        """Verifies all 18 required deliverable files exist in output directory."""
        expected_files = [
            "stage14_report.md",
            "input_integrity_audit.json",
            "per_view_cloud_audit.json",
            "registration_results.json",
            "reconstruction_quality.json",
            "alignment_initial_vs_registered.png",
            "pairwise_residuals.png",
            "combined_reconstructed_cloud.ply",
            "view_00_raw.ply", "view_01_raw.ply", "view_02_raw.ply", "view_03_raw.ply",
            "view_00_subject.ply", "view_01_subject.ply", "view_02_subject.ply", "view_03_subject.ply",
            "view_00_registered.ply", "view_01_registered.ply", "view_02_registered.ply", "view_03_registered.ply",
        ]
        for fname in expected_files:
            fpath = STAGE14_OUT_DIR / fname
            assert fpath.exists(), f"Missing required deliverable: {fname}"
            assert fpath.stat().st_size > 0, f"File is empty: {fname}"

    def test_09_verdict_and_scientific_safeguards(self):
        """Confirms final verdict is PARTIAL_REGISTRATION and safeguards are maintained."""
        quality_path = STAGE14_OUT_DIR / "reconstruction_quality.json"
        with open(quality_path, "r", encoding="utf-8") as fp:
            quality = json.load(fp)

        assert quality["verdict"] == "PARTIAL_REGISTRATION"
        assert len(quality["verdict_rationale"]) > 50
        assert quality["scientific_safeguards_confirmed"] is True

        # Confirm NO anthropometric circumference keys exist
        forbidden_keys = [
            "waist_circumference_mm", "hip_circumference_mm",
            "chest_circumference_mm", "thigh_circumference_mm",
            "smpl_beta", "mesh_vertices"
        ]
        for k in forbidden_keys:
            assert k not in quality, f"Safeguard violation: found forbidden key {k}"

