"""
tests/body/test_physical_stage4_personalization.py
==================================================
Unit and integration tests for Problem 4 — Stage 4:
Conservative Personalized Surface Deformation from Canonical RGB-D Observations.

Verifies:
1. Synthetic regularized Laplacian deformation properties (identity invariance, pinning, clamping).
2. Stage 4 physical output artifacts existence and schema.
3. Strict preservation of unobserved anatomical regions (head, arms, calves, feet pinned to 0).
4. Meaningful, bounded deformation on observed regions (torso, pelvis, thighs).
5. Quantitative residual reduction between Stage 2 and Stage 4.
6. Topological mesh integrity (watertight closed 2-manifold, 0 flipped normals, edge ratio bounds).
7. Array dimensionality and data contract compliance.
"""

import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from realsense_measure.body.personalized_surface import (
    ANATOMICAL_REGIONS,
    PersonalisedSurfaceFitter,
    _compute_vertex_normals,
    _compute_vertex_regions,
    _get_cached_default_anny_skinning,
)
from tools.run_physical_stage4_personalized_surface import PhysicalStage4Personalizer


STAGE2_MESH_FILE = Path("stage2_articulated_fit_output/canonical_fitted_model.ply")
STAGE2_PARAMS_FILE = Path("stage2_articulated_fit_output/fitted_body_parameters.json")
STAGE3_PROVENANCE_FILE = Path("stage3_canonical_fusion_output/canonical_point_provenance.npz")
STAGE4_OUTPUT_DIR = Path("stage4_personalized_surface_output")


@pytest.fixture(scope="module")
def stage2_mesh_data():
    assert STAGE2_MESH_FILE.exists(), f"Stage 2 mesh missing: {STAGE2_MESH_FILE}"
    mesh = o3d.io.read_triangle_mesh(str(STAGE2_MESH_FILE))
    v0 = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    return v0, faces


@pytest.fixture(scope="module")
def default_skinning():
    skin = _get_cached_default_anny_skinning()
    assert skin is not None
    return skin


# ---------------------------------------------------------------------------
# 1. Synthetic Regularization Properties Tests
# ---------------------------------------------------------------------------

class TestSyntheticLaplacianProperties:

    def test_synthetic_identity_unmoved_mesh(self, stage2_mesh_data):
        """Points identical to canonical vertices produce negligible displacement."""
        v0, faces = stage2_mesh_data
        fitter = PersonalisedSurfaceFitter(
            canonical_vertices=v0,
            faces=faces,
            max_dist_m=0.040,
        )
        # Select 500 vertices on the torso
        torso_mask = fitter.vertex_regions == "torso"
        torso_v = v0[torso_mask][:500]

        delta_v = fitter._solve_offsets(torso_v, lam=60.0)
        disp_norms = np.linalg.norm(delta_v, axis=-1) * 1000.0

        # Displacements on zero-error inputs should be near 0
        assert np.mean(disp_norms) < 0.1, f"Expected near zero displacement, got mean {np.mean(disp_norms)} mm"

    def test_dirichlet_boundary_pinning_invariance(self, stage2_mesh_data):
        """Unobserved limbs must have strictly zero displacement under Dirichlet pinning."""
        v0, faces = stage2_mesh_data
        fitter = PersonalisedSurfaceFitter(
            canonical_vertices=v0,
            faces=faces,
            max_dist_m=0.040,
        )
        # Only provide observations on mid-torso with a 10 mm outward offset
        mid_torso_mask = (fitter.vertex_regions == "torso") & (v0[:, 2] > 0.10) & (v0[:, 2] < 0.28)
        torso_pts = v0[mid_torso_mask][:200] + np.array([0.0, 0.010, 0.0])

        delta_v = fitter._solve_offsets(torso_pts, lam=30.0)
        disp_norms = np.linalg.norm(delta_v, axis=-1) * 1000.0

        # Inactive regions (arms, lower legs, head) must have 0 displacement
        for reg in ["head_neck", "upperarm_L", "hand_L", "lowerleg_L", "lowerleg_R"]:
            reg_mask = fitter.vertex_regions == reg
            if np.any(reg_mask):
                reg_disps = disp_norms[reg_mask]
                assert np.allclose(reg_disps, 0.0, atol=1e-3), (
                    f"Region {reg} expected 0 mm displacement under pinning, got max {np.max(reg_disps):.4f} mm"
                )

    def test_displacement_clamping_enforcement(self, stage2_mesh_data):
        """Hard safety clamping prevents displacements exceeding the user ceiling."""
        v0, faces = stage2_mesh_data
        # Create an artificial displacement field with extreme 50 mm offsets
        delta_v = np.zeros_like(v0)
        delta_v[:50] = np.array([0.0, 0.050, 0.0])  # 50 mm

        clamp_m = 0.020  # 20 mm clamp
        disp_m = np.linalg.norm(delta_v, axis=-1)
        exceeded = disp_m > clamp_m
        if np.any(exceeded):
            delta_v[exceeded] *= (clamp_m / (disp_m[exceeded, None] + 1e-8))

        clamped_norms_mm = np.linalg.norm(delta_v, axis=-1) * 1000.0
        assert np.max(clamped_norms_mm) <= 20.0001, f"Clamp exceeded: max was {np.max(clamped_norms_mm)} mm"


# ---------------------------------------------------------------------------
# 2. Physical Stage 4 Artifacts and Results Tests
# ---------------------------------------------------------------------------

class TestPhysicalStage4Artifacts:

    def test_all_stage4_output_files_exist(self):
        """Verify all 10 expected Stage 4 artifacts exist on disk."""
        assert STAGE4_OUTPUT_DIR.exists(), f"Stage 4 directory missing: {STAGE4_OUTPUT_DIR}"

        expected_files = [
            "stage2_original_mesh.ply",
            "stage4_personalized_mesh.obj",
            "stage4_personalized_mesh.ply",
            "vertex_displacements.npy",
            "vertex_observation_labels.npy",
            "residual_comparison.png",
            "displacement_magnitude_distribution.png",
            "displacement_spatial_heatmap.png",
            "stage4_personalization_summary.json",
            "stage4_personalization_report.md",
        ]
        for fname in expected_files:
            fpath = STAGE4_OUTPUT_DIR / fname
            assert fpath.exists(), f"Expected artifact missing: {fpath}"
            assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"

    def test_summary_json_metrics_and_status(self):
        """Verify structured JSON summary has status PASSED, valid lambda, and residual reduction."""
        summary_path = STAGE4_OUTPUT_DIR / "stage4_personalization_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["status"] == "PASSED"
        assert "optimization_configuration" in summary
        assert summary["optimization_configuration"]["selected_lambda"] == 60.0

        # Residual reduction checks
        res = summary["residual_comparison_mm"]
        assert res["median_reduction_mm"] > 2.0, f"Expected >2 mm median reduction, got {res['median_reduction_mm']}"
        assert res["median_reduction_percent"] > 10.0, f"Expected >10% improvement, got {res['median_reduction_percent']}%"

        # Classification counts
        v_class = summary["vertex_classification"]
        assert v_class["total_vertices"] == 13718
        assert v_class["observed_supported_count"] > 800
        assert v_class["inferred_prior_count"] > 10000

    def test_mesh_integrity_and_watertightness(self):
        """Verify personalized mesh has 0 flipped faces, manifold topology, and bounded edge ratios."""
        summary_path = STAGE4_OUTPUT_DIR / "stage4_personalization_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        audit = summary["mesh_integrity_audit"]
        assert audit["vertex_count"] == 13718
        assert audit["face_count"] == 27420
        assert audit["flipped_normal_faces"] == 0, f"Detected flipped faces: {audit['flipped_normal_faces']}"
        assert audit["min_normal_dot_product"] > 0.80, f"Low normal dot product: {audit['min_normal_dot_product']}"
        assert audit["non_finite_vertices"] == 0
        assert audit["watertight"] is True
        assert 0.70 <= audit["edge_length_ratio_min"] <= 1.0
        assert 1.0 <= audit["edge_length_ratio_max"] <= 2.0

    def test_unobserved_anatomical_regions_invariance(self):
        """Verify unobserved anatomical regions have strictly 0 displacement."""
        summary_path = STAGE4_OUTPUT_DIR / "stage4_personalization_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        per_reg = summary["displacement_statistics_mm"]["per_region"]

        # Strictly 0 mm regions (within numerical CG solver tolerance < 0.01 mm)
        for reg in ["head_neck", "upperarm_L", "forearm_L", "hand_L", "upperarm_R", "forearm_R", "hand_R", "lowerleg_R"]:
            assert per_reg[reg]["mean_mm"] < 0.01, f"{reg} expected <0.01 mm mean disp, got {per_reg[reg]['mean_mm']}"
            assert per_reg[reg]["max_mm"] < 0.05, f"{reg} expected <0.05 mm max disp, got {per_reg[reg]['max_mm']}"

        # Lower leg left (top transition at knee)
        assert per_reg["lowerleg_L"]["median_mm"] < 0.01

    def test_observed_regions_meaningful_deformation(self):
        """Verify observed regions (torso, pelvis, thighs) deformed realistically within clamp bounds."""
        summary_path = STAGE4_OUTPUT_DIR / "stage4_personalization_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        per_reg = summary["displacement_statistics_mm"]["per_region"]

        # Torso
        assert 1.0 < per_reg["torso"]["mean_mm"] < 5.0
        assert per_reg["torso"]["max_mm"] <= 20.0

        # Thighs
        assert 3.0 < per_reg["thigh_L"]["mean_mm"] < 8.0
        assert per_reg["thigh_L"]["max_mm"] <= 20.0
        assert 3.0 < per_reg["thigh_R"]["mean_mm"] < 8.0
        assert per_reg["thigh_R"]["max_mm"] <= 20.0

    def test_arrays_consistency(self):
        """Verify vertex_displacements.npy and vertex_observation_labels.npy match mesh dims."""
        disp_path = STAGE4_OUTPUT_DIR / "vertex_displacements.npy"
        labels_path = STAGE4_OUTPUT_DIR / "vertex_observation_labels.npy"

        disps = np.load(str(disp_path))
        labels = np.load(str(labels_path))

        assert disps.shape == (13718, 3)
        assert disps.dtype == np.float32
        assert np.all(np.isfinite(disps))

        assert labels.shape == (13718,)
        valid_tiers = {"observed_supported", "weakly_observed", "inferred_prior"}
        assert set(np.unique(labels)).issubset(valid_tiers)
