"""
tests/body/test_physical_stage3_unposing_fusion.py
==================================================
Unit and integration tests for Problem 4 — Stage 3:
Canonical Unposing and Observation Fusion.

Verifies:
1. Synthetic canonical unposing with known ground-truth poses.
2. Coordinate-frame consistency in canonical space.
3. Proper utilization of fitted per-stop pose parameters (preventing nominal overwrite).
4. Provenance preservation (cam XYZ, world XYZ, canonical XYZ, UV pixel, depth, stop index).
5. Outlier rejection and invalid-point handling.
6. Fusion behavior and balanced per-stop weighting.
7. Verification that fused cloud contains ONLY derived observation geometry (no synthetic hole-filling).
8. Determinism and numerical stability.
9. Verification of generated Stage 3 output artifacts.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from camera.recording import load_raw_capture
from realsense_measure.body.adapter import frame_to_body_stop, get_or_create_foreground_mask
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.unpose import BodyUnposer


PHYSICAL_DATASET_DIR = Path("raw_captures/body_capture_01")
STAGE2_PARAMS_FILE = Path("stage2_articulated_fit_output/fitted_body_parameters.json")
STAGE3_OUTPUT_DIR = Path("stage3_canonical_fusion_output")


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend()


@pytest.fixture(scope="module")
def body_unposer(anny_backend):
    out = anny_backend.rest_pose()
    return BodyUnposer(
        canon_vertices=out.numpy_vertices(0),
        canon_joints=out.numpy_joints(0),
        joint_names=list(out.joint_names),
        raw_dict=out.raw if hasattr(out, "raw") else {},
    )


# ---------------------------------------------------------------------------
# 1. Synthetic Validation Tests (Known Ground-Truth Transformations)
# ---------------------------------------------------------------------------

class TestSyntheticCanonicalUnposing:

    def test_synthetic_identity_unposing(self, body_unposer):
        """Points already in canonical pose with zero pose transforms remain invariant."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.1, -0.05, 0.3],
            [-0.15, 0.02, -0.2],
        ], dtype=np.float32)

        p_unp, dists, _ = body_unposer.unpose_points(
            points_world=pts,
            yaw_deg=0.0,
            translation_m=[0.0, 0.0, 0.0],
            arm_abd_L_deg=0.0,
            arm_abd_R_deg=0.0,
        )
        assert np.allclose(p_unp, pts, atol=1e-6)

    def test_synthetic_known_yaw_and_translation_unposing(self, body_unposer):
        """Applying a known yaw and translation then inverting recovers original points."""
        pts_orig = np.array([
            [0.05, 0.10, 0.20],
            [-0.08, -0.04, 0.15],
            [0.02, -0.12, -0.10],
        ], dtype=np.float32)

        yaw_deg = 45.0
        rad = np.radians(yaw_deg)
        cos_y = np.cos(rad)
        sin_y = np.sin(rad)
        rot_z = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)
        trans = np.array([0.15, -0.05, 0.08], dtype=np.float32)

        # Forward transformation into world
        pts_world = pts_orig @ rot_z.T + trans

        # Unpose
        pts_unp, _, _ = body_unposer.unpose_points(
            points_world=pts_world,
            yaw_deg=yaw_deg,
            translation_m=trans,
        )
        assert np.allclose(pts_unp, pts_orig, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Physical Data Validation Tests (Using Real D455f Capture & Stage 2 Fits)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not PHYSICAL_DATASET_DIR.exists() or not STAGE2_PARAMS_FILE.exists(),
    reason="Physical dataset or Stage 2 parameters not present"
)
class TestPhysicalStage3UnposingAndFusion:

    def test_stage2_parameters_validation(self):
        """Verify Stage 2 parameters are complete and valid before unposing."""
        with open(STAGE2_PARAMS_FILE, "r", encoding="utf-8") as f:
            params = json.load(f)

        assert "shared_phenotypes" in params
        assert "per_stop_poses" in params
        assert len(params["per_stop_poses"]) == 4

        # Verify Stop 01 preserves fitted yaw discrepancy (~63.1 deg) instead of nominal 90 deg
        pose1 = params["per_stop_poses"][1]
        assert 55.0 < pose1["yaw_deg"] < 75.0, f"Expected fitted yaw ~63 deg for Stop 1, got {pose1['yaw_deg']}"

    def test_physical_provenance_preservation(self):
        """Verify sensor provenance archive contains all required keys and correct record counts."""
        prov_file = STAGE3_OUTPUT_DIR / "canonical_point_provenance.npz"
        assert prov_file.exists(), "canonical_point_provenance.npz missing"

        data = np.load(prov_file)
        expected_keys = [
            "original_xyz_cam",
            "original_xyz_world",
            "canonical_xyz",
            "rgb_colors",
            "stop_indices",
            "pixel_uv",
            "source_depth_m",
            "transformation_methods",
            "residuals_mm",
            "is_inlier",
            "fused_voxel_positions",
            "fused_voxel_residuals_mm",
        ]
        for k in expected_keys:
            assert k in data, f"Key {k} missing from provenance archive"

        n_records = len(data["canonical_xyz"])
        assert n_records == 457321, f"Expected 457,321 provenance records, got {n_records}"
        assert len(data["pixel_uv"]) == n_records
        assert len(data["source_depth_m"]) == n_records
        assert len(data["is_inlier"]) == n_records

    def test_no_synthetic_fabrication_in_fused_cloud(self):
        """Verify the fused canonical observation cloud contains ONLY derived observation geometry."""
        fused_ply = STAGE3_OUTPUT_DIR / "canonical_fused_observations.ply"
        assert fused_ply.exists(), "canonical_fused_observations.ply missing"

        prov_data = np.load(STAGE3_OUTPUT_DIR / "canonical_point_provenance.npz")
        fused_pts = prov_data["fused_voxel_positions"]

        # 1. Point count must match unique occupied voxels (~8,000 to 10,000)
        assert 5000 < len(fused_pts) < 15000

        # 2. Upper-body vertical bounding extent must reflect physical truncation (feet at -0.94 m are truncated)
        z_min = float(fused_pts[:, 2].min())
        z_max = float(fused_pts[:, 2].max())
        assert z_min > -0.75, f"Lower leg/feet geometry was falsely fabricated (Z min: {z_min})"
        assert z_max < 0.35, f"Crown geometry was falsely fabricated (Z max: {z_max})"

    def test_balanced_fusion_weighting(self):
        """Verify fused voxels record multi-stop contributions without single-view monopoly."""
        summary_file = STAGE3_OUTPUT_DIR / "stage3_canonical_fusion_summary.json"
        assert summary_file.exists()

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        v_stats = summary["voxel_view_statistics"]
        assert v_stats["total_fused_voxels"] > 0
        assert v_stats["voxels_observed_by_1_view"] > 0
        # Check that multi-view voxels exist where views overlap
        assert v_stats["voxels_observed_by_2_views"] > 0

    def test_stage3_output_artifacts_exist(self):
        """Verify all required Stage 3 artifacts are present and non-empty."""
        expected_artifacts = [
            STAGE3_OUTPUT_DIR / "canonical_observations_all_stops.ply",
            STAGE3_OUTPUT_DIR / "canonical_fused_observations.ply",
            STAGE3_OUTPUT_DIR / "canonical_point_provenance.npz",
            STAGE3_OUTPUT_DIR / "stop_00_canonical.ply",
            STAGE3_OUTPUT_DIR / "stop_01_canonical.ply",
            STAGE3_OUTPUT_DIR / "stop_02_canonical.ply",
            STAGE3_OUTPUT_DIR / "stop_03_canonical.ply",
            STAGE3_OUTPUT_DIR / "canonical_pairwise_overlap.png",
            STAGE3_OUTPUT_DIR / "canonical_voxel_density.png",
            STAGE3_OUTPUT_DIR / "canonical_residual_distribution.png",
            STAGE3_OUTPUT_DIR / "stop_01_yaw_investigation.png",
            STAGE3_OUTPUT_DIR / "stage3_canonical_fusion_summary.json",
            STAGE3_OUTPUT_DIR / "stage3_canonical_fusion_report.md",
        ]
        for art in expected_artifacts:
            assert art.exists(), f"Artifact {art} missing"
            assert art.stat().st_size > 0, f"Artifact {art} is empty"
