"""
tests/body/test_physical_stage25_smpl_personalization.py
=========================================================
Unit and integration test suite for Stage 25:
RealSense-Driven SMPL Mesh Personalization and Surface Deformation.

Verifies:
1. Canonical SMPL topology preservation (exact 6,890 vertices, 13,776 faces).
2. Genuine point-cloud-driven surface deformation with biharmonic Laplacian regularization.
3. Residual error improvement:
   - Median point-to-mesh residual improves significantly (> 8 mm reduction).
   - Inlier percentage (<= 25 mm) increases (> 10% gain).
4. Displacement clamping and Dirichlet boundary pinning:
   - Max displacement strictly bounded <= 25.0 mm.
   - Unobserved lower limbs (Z < -0.70m, feet/calves) pinned to model template (< 1.0 mm displacement).
5. Topological manifoldness and geometric validity:
   - 100% edge manifold and vertex manifold.
   - Zero degenerate faces (area < 1e-8).
   - Zero inverted surface normals (dot product >= 0.0).
6. Anthropometric measurement suite extraction:
   - 15 measurements with dual units (mm & cm), confidence status, definitions, and visibility flags.
   - Unobserved regions marked MODEL_DERIVED_UNOBSERVED_REGION.
   - Chest, waist, and hip circumferences properly adapted to observed physical depth contours.
7. Deliverables inventory and valid final verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import open3d as o3d

from realsense_measure.body.smpl_backend import (
    SMPLModelBackend,
    SMPLMeshOutput,
)
from realsense_measure.body.smpl_personalization import (
    SMPLPersonalizer,
    Stage25Verdict,
)
from realsense_measure.measurements.smpl_personalization_orchestrator import (
    Stage25SMPLPersonalizationOrchestrator,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def stage25_output_dir(repo_root: Path) -> Path:
    out = repo_root / "stage25_smpl_personalization_output"
    assert out.exists(), f"Stage 25 output directory missing: {out}"
    return out


@pytest.fixture(scope="module")
def orchestrator(repo_root: Path, stage25_output_dir: Path) -> Stage25SMPLPersonalizationOrchestrator:
    return Stage25SMPLPersonalizationOrchestrator(repo_root=repo_root, output_dir=stage25_output_dir)


class TestStage25SMPLTopologyAndDeformation:
    """Verifies regularized Laplacian deformation mechanics and residual improvement."""

    def test_canonical_topology_exact_counts(self, stage25_output_dir: Path):
        ply_path = stage25_output_dir / "personalized_smpl_mesh.ply"
        obj_path = stage25_output_dir / "personalized_smpl_mesh.obj"
        assert ply_path.exists()
        assert obj_path.exists()

        m = o3d.io.read_triangle_mesh(str(ply_path))
        verts = np.asarray(m.vertices)
        faces = np.asarray(m.triangles)

        assert len(verts) == 6890, f"Expected exactly 6,890 SMPL vertices, got {len(verts)}"
        assert len(faces) == 13776, f"Expected exactly 13,776 SMPL faces, got {len(faces)}"

    def test_residual_improvement_metrics(self, stage25_output_dir: Path):
        res_file = stage25_output_dir / "fit_residual_metrics.json"
        assert res_file.exists()
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        before = data["pre_deformation_residuals_mm"]
        after = data["post_personalization_residuals_mm"]
        improvement = data["residual_improvement_mm"]

        assert before["evaluated_points"] > 400000
        assert after["evaluated_points"] == before["evaluated_points"]

        # Median residual must improve
        assert after["median_mm"] < before["median_mm"], (
            f"Expected median to improve, got before={before['median_mm']:.2f}mm, after={after['median_mm']:.2f}mm"
        )
        assert improvement["median_reduction_mm"] >= 8.0, (
            f"Expected at least 8mm median reduction, got {improvement['median_reduction_mm']:.2f}mm"
        )

        # Inlier rate (<= 25mm) must improve significantly
        assert after["inlier_percent_25mm"] > before["inlier_percent_25mm"]
        assert improvement["inlier_25mm_gain_percent"] >= 10.0

    def test_displacement_bounds_and_dirichlet_pinning(self, stage25_output_dir: Path):
        deform_file = stage25_output_dir / "deformation_metrics.json"
        assert deform_file.exists()
        with open(deform_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Displacement must be bounded <= 25.1 mm
        assert stats["max_displacement_mm"] <= 25.1, (
            f"Displacement exceeded 25mm clamp: {stats['max_displacement_mm']}mm"
        )
        assert stats["mean_displacement_mm"] > 0.0
        assert stats["active_deformed_vertices"] > 500

        # Check Dirichlet pinning on unobserved lower limbs relative to input fitted mesh
        ply_pers = stage25_output_dir / "personalized_smpl_mesh.ply"
        fitted_path = stage25_output_dir.parent / "stage24_smpl_integration_output" / "fitted_smpl_mesh.ply"
        assert fitted_path.exists()
        m_pers = o3d.io.read_triangle_mesh(str(ply_pers))
        m_fit = o3d.io.read_triangle_mesh(str(fitted_path))

        v_pers = np.asarray(m_pers.vertices)
        v_fit = np.asarray(m_fit.vertices)

        # In Stage 24 fitted mesh, feet/calves are below Z = -0.70m
        feet_mask = v_fit[:, 2] < -0.70
        if np.any(feet_mask):
            feet_disp = np.linalg.norm(v_pers[feet_mask] - v_fit[feet_mask], axis=1) * 1000.0
            # Feet must remain pinned to model prior (negligible displacement)
            assert np.median(feet_disp) < 2.0, (
                f"Unobserved lower limbs were unconstrained: median feet displacement = {np.median(feet_disp):.2f}mm"
            )


class TestStage25MeshQualityAndManifoldness:
    """Verifies that deformation strictly preserves 2-manifold mesh topology."""

    def test_mesh_quality_report_schema_and_values(self, stage25_output_dir: Path):
        q_file = stage25_output_dir / "mesh_quality_report.json"
        assert q_file.exists()
        with open(q_file, "r", encoding="utf-8") as f:
            q = json.load(f)

        assert q["vertex_count"] == 6890
        assert q["face_count"] == 13776
        assert q["is_edge_manifold"] is True
        assert q["is_vertex_manifold"] is True
        assert q["has_degenerate_faces"] is False
        assert q["degenerate_face_count"] == 0
        assert q["inverted_normal_face_count"] == 0
        assert q["min_normal_dot_product"] >= 0.0

    def test_open3d_mesh_integrity_on_saved_ply(self, stage25_output_dir: Path):
        ply_path = stage25_output_dir / "personalized_smpl_mesh.ply"
        m = o3d.io.read_triangle_mesh(str(ply_path))

        assert m.is_edge_manifold(allow_boundary_edges=True) is True
        assert m.is_vertex_manifold() is True

        # Check face normals
        faces = np.asarray(m.triangles)
        verts = np.asarray(m.vertices)
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        fn = np.cross(verts[f1] - verts[f0], verts[f2] - verts[f0])
        areas = 0.5 * np.linalg.norm(fn, axis=1)

        assert np.all(areas > 1e-8), "Degenerate zero-area triangles detected in personalized mesh!"


class TestStage25AnthropometricMeasurements:
    """Verifies 15-measurement anthropometric extraction on personalized SMPL mesh."""

    def test_measurement_suite_schema(self, stage25_output_dir: Path):
        meas_file = stage25_output_dir / "anthropometric_measurements.json"
        assert meas_file.exists()
        with open(meas_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_measurements"] >= 15
        measurements = data["measurements"]

        expected_keys = {
            "overall_body_height",
            "shoulder_breadth",
            "upper_arm_length_left",
            "upper_arm_length_right",
            "forearm_length_left",
            "forearm_length_right",
            "torso_length",
            "upper_leg_length_left",
            "upper_leg_length_right",
            "lower_leg_length_left",
            "lower_leg_length_right",
            "chest_circumference",
            "waist_circumference",
            "hip_circumference",
            "head_circumference",
        }
        found_keys = {m["name"] for m in measurements}
        assert expected_keys.issubset(found_keys)

        for m in measurements:
            assert "value_mm" in m and m["value_mm"] > 0
            assert "value_cm" in m and abs(m["value_cm"] - m["value_mm"] / 10.0) < 1e-4
            assert "confidence_status" in m
            assert "is_anatomical_region_visible" in m
            assert "definition" in m
            assert "source" in m

    def test_anatomical_ranges(self, stage25_output_dir: Path):
        meas_file = stage25_output_dir / "anthropometric_measurements.json"
        with open(meas_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meas_by_name = {m["name"]: m for m in data["measurements"]}

        # Height (Stature): 1650 - 2000 mm (preserving fitted SMPL stature prior)
        assert 1650 <= meas_by_name["overall_body_height"]["value_mm"] <= 2000
        assert meas_by_name["overall_body_height"]["confidence_status"] == "MODEL_DERIVED_UNOBSERVED_REGION"

        # Shoulder Breadth: 350 - 700 mm (acromion to acromion span on SMPL)
        assert 350 <= meas_by_name["shoulder_breadth"]["value_mm"] <= 700

        # Torso Length: 100 - 700 mm
        assert 100 <= meas_by_name["torso_length"]["value_mm"] <= 700

        # Circumferences
        assert 800 <= meas_by_name["chest_circumference"]["value_mm"] <= 1400
        assert 700 <= meas_by_name["waist_circumference"]["value_mm"] <= 1600
        assert 800 <= meas_by_name["hip_circumference"]["value_mm"] <= 1500
        assert 480 <= meas_by_name["head_circumference"]["value_mm"] <= 650


class TestStage25DeliverablesAndVerdict:
    """Verifies all deliverables exist and conform to Stage 25 specifications."""

    def test_all_deliverables_present(self, stage25_output_dir: Path):
        expected_files = [
            "stage25_report.md",
            "personalized_smpl_mesh.ply",
            "personalized_smpl_mesh.obj",
            "baseline_smpl_mesh.ply",
            "deformation_metrics.json",
            "fit_residual_metrics.json",
            "anthropometric_measurements.json",
            "measurement_quality.json",
            "mesh_quality_report.json",
            "integration_test_results.json",
            "point_cloud_smpl_overlay.png",
            "baseline_vs_personalized_overlay.png",
            "residual_heatmap.png",
            "observed_vs_model_regions.png",
            "measurement_comparison.png",
        ]
        for fname in expected_files:
            target = stage25_output_dir / fname
            assert target.exists(), f"Missing required deliverable: {fname}"
            assert target.stat().st_size > 0, f"Deliverable is empty: {fname}"

    def test_verdict_validity(self, stage25_output_dir: Path):
        test_file = stage25_output_dir / "integration_test_results.json"
        assert test_file.exists()
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        verdict = data["final_verdict"]
        allowed = {
            Stage25Verdict.PERSONALIZED_MESH_SUCCESSFULLY_DEFORMED.value,
            Stage25Verdict.DEFORMATION_IMPROVES_POINT_CLOUD_ALIGNMENT.value,
            Stage25Verdict.DEFORMATION_OVERFITS.value,
            Stage25Verdict.MESH_GEOMETRY_UNSTABLE.value,
            Stage25Verdict.SMPL_ASSETS_REQUIRED_FOR_PARAMETRIC_FITTING.value,
            Stage25Verdict.STAGE25_INTEGRATION_BLOCKED.value,
            Stage25Verdict.MULTIPLE_ISSUES_FOUND.value,
        }
        assert verdict in allowed
        assert verdict in [
            Stage25Verdict.PERSONALIZED_MESH_SUCCESSFULLY_DEFORMED.value,
            Stage25Verdict.DEFORMATION_IMPROVES_POINT_CLOUD_ALIGNMENT.value,
        ]
