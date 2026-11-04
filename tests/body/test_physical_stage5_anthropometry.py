"""
tests/body/test_physical_stage5_anthropometry.py
================================================
Unit and integration tests for Problem 4 — Stage 5:
Dual-Mesh Anthropometric Measurement and Observed-Ring Diagnostics.

Verifies:
1. Synthetic geometric accuracy:
     - Analytic circle / cylinder circumference (2*pi*R).
     - Analytic ellipse circumference (Ramanujan formula).
     - Planar slicing coordinate precision (|z - z0| < 1e-6).
     - Component separation (preventing disconnected limb contour bridging).
     - Degenerate / open contour rejection (is_contour_closed).
2. Physical Stage 5 output artifacts existence and schema.
3. Dual-mesh numerical consistency (diff_mm = Stage 4 - Stage 2).
4. Unobserved limb invariance (arms, head, stature have 0.0 mm diff).
5. Meaningful physical deformation on observed rings (hip, thighs, waist).
6. Observed-ring coverage diagnostics and honest confidence tier classification.
"""

import csv
import json
import math
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.body.measure import (
    _loop_perimeter,
    _slice_mesh_at_z,
    _slice_mesh_plane,
    is_contour_closed,
)
from tools.run_physical_stage5_dual_mesh_anthropometry import (
    PhysicalStage5DualMeshAnthropometry,
    RingCoverageDiagnostic,
)


STAGE5_OUTPUT_DIR = Path("stage5_anthropometric_output")


# ---------------------------------------------------------------------------
# 1. Synthetic Geometric & Numerical Accuracy Tests
# ---------------------------------------------------------------------------

class TestSyntheticGeometricAccuracy:

    def test_synthetic_cylinder_circumference(self):
        """Verify planar slice of synthetic cylinder matches analytic 2*pi*R."""
        radius = 0.150  # 150 mm radius -> perimeter = 300*pi ≈ 942.4778 mm
        height = 0.500  # 500 mm height
        n_radial = 120  # 120-gon approximation

        theta = np.linspace(0, 2.0 * np.pi, n_radial, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Build two rings of vertices at z=0 and z=height
        verts = np.vstack([
            np.stack([x, y, np.zeros_like(theta)], axis=1),
            np.stack([x, y, np.full_like(theta, height)], axis=1),
        ])
        faces = []
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            faces.append([i, i_next, i + n_radial])
            faces.append([i_next, i_next + n_radial, i + n_radial])
        faces = np.array(faces, dtype=np.int64)

        # Slice cylinder at midpoint z=0.25 m
        loops = _slice_mesh_at_z(verts, faces, 0.25)
        assert len(loops) == 1, f"Expected exactly 1 loop, got {len(loops)}"

        loop = loops[0]
        closed, reason = is_contour_closed(loop)
        assert closed is True, f"Cylinder loop failed closure test: {reason}"

        measured_p_mm = _loop_perimeter(loop) * 1000.0
        expected_p_mm = 2.0 * np.pi * radius * 1000.0  # 942.4778 mm

        # 120-gon polygon discretization error is ~0.04%
        rel_error = abs(measured_p_mm - expected_p_mm) / expected_p_mm
        assert rel_error < 0.001, f"Cylinder error {rel_error*100:.3f}% exceeded 0.1% tolerance"

    def test_synthetic_ellipse_circumference(self):
        """Verify planar slice of synthetic ellipse matches Ramanujan approximation."""
        a = 0.200  # 200 mm semi-major
        b = 0.120  # 120 mm semi-minor
        height = 0.400
        n_radial = 120

        # Ramanujan's formula: P ≈ pi*(a+b)*(1 + 3h / (10 + sqrt(4 - 3h)))
        h = ((a - b) ** 2) / ((a + b) ** 2)
        expected_p_mm = float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))) * 1000.0)

        theta = np.linspace(0, 2.0 * np.pi, n_radial, endpoint=False)
        x = a * np.cos(theta)
        y = b * np.sin(theta)

        verts = np.vstack([
            np.stack([x, y, np.zeros_like(theta)], axis=1),
            np.stack([x, y, np.full_like(theta, height)], axis=1),
        ])
        faces = []
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            faces.append([i, i_next, i + n_radial])
            faces.append([i_next, i_next + n_radial, i + n_radial])
        faces = np.array(faces, dtype=np.int64)

        loops = _slice_mesh_at_z(verts, faces, 0.20)
        assert len(loops) == 1
        loop = loops[0]
        measured_p_mm = _loop_perimeter(loop) * 1000.0

        rel_error = abs(measured_p_mm - expected_p_mm) / expected_p_mm
        assert rel_error < 0.002, f"Ellipse error {rel_error*100:.3f}% exceeded 0.2% tolerance"

    def test_planar_slicing_coordinate_precision(self):
        """Verify all vertices on a horizontal slice plane have |Z - Z0| < 1e-6."""
        # Simple cube mesh sliced at Z=0.5
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            [4, 5, 6], [4, 6, 7], [3, 2, 1], [3, 1, 0],
        ], dtype=np.int64)

        z_plane = 0.50
        loops = _slice_mesh_at_z(verts, faces, z_plane)
        assert len(loops) == 1
        loop = loops[0]
        assert np.allclose(loop[:, 2], z_plane, atol=1e-6), "Planar slice vertices deviated from Z0"

    def test_disconnected_component_separation(self):
        """Verify that slicing through two separate limbs preserves two disjoint loops without bridging."""
        # Create two disjoint cylinders: Left (X=+0.15) and Right (X=-0.15)
        radius = 0.05
        height = 0.40
        n_pts = 30
        theta = np.linspace(0, 2.0 * np.pi, n_pts, endpoint=False)

        x_L = 0.15 + radius * np.cos(theta)
        x_R = -0.15 + radius * np.cos(theta)
        y = radius * np.sin(theta)

        v_L = np.vstack([
            np.stack([x_L, y, np.zeros_like(theta)], axis=1),
            np.stack([x_L, y, np.full_like(theta, height)], axis=1),
        ])
        v_R = np.vstack([
            np.stack([x_R, y, np.zeros_like(theta)], axis=1),
            np.stack([x_R, y, np.full_like(theta, height)], axis=1),
        ])
        verts = np.vstack([v_L, v_R])

        f_L = []
        for i in range(n_pts):
            i_next = (i + 1) % n_pts
            f_L.append([i, i_next, i + n_pts])
            f_L.append([i_next, i_next + n_pts, i + n_pts])

        offset = 2 * n_pts
        f_R = [[f[0] + offset, f[1] + offset, f[2] + offset] for f in f_L]
        faces = np.vstack([f_L, f_R])

        loops = _slice_mesh_at_z(verts, faces, 0.20)
        assert len(loops) == 2, f"Expected 2 separate disjoint loops, got {len(loops)}"

        centers = [np.mean(lp[:, 0]) for lp in loops]
        assert any(c > 0.10 for c in centers), "Left cylinder missing"
        assert any(c < -0.10 for c in centers), "Right cylinder missing"

    def test_degenerate_contour_rejection(self):
        """Verify is_contour_closed rejects open arcs, collapsed geometries, and hairpins."""
        # Open arc (endpoints far apart)
        theta = np.linspace(0, np.pi, 20)  # semi-circle only
        open_arc = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        closed, reason = is_contour_closed(open_arc)
        assert closed is False
        assert "Endpoint gap" in reason or "fewer" in reason

        # Collapsed line segment
        line = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.1], [0, 0, 0]], dtype=np.float64)
        closed_line, reason_line = is_contour_closed(line)
        assert closed_line is False


# ---------------------------------------------------------------------------
# 2. Physical Stage 5 Artifacts and Consistency Tests
# ---------------------------------------------------------------------------

class TestPhysicalStage5ArtifactsAndConsistency:

    def test_all_stage5_output_files_exist(self):
        """Verify all 10 expected Stage 5 output artifacts exist on disk."""
        assert STAGE5_OUTPUT_DIR.exists(), f"Stage 5 directory missing: {STAGE5_OUTPUT_DIR}"

        expected_files = [
            "dual_mesh_measurement_comparison.csv",
            "stage2_measurements.json",
            "stage4_measurements.json",
            "ring_coverage_diagnostics.json",
            "measurement_rings_stage2.ply",
            "measurement_rings_stage4.ply",
            "ring_coverage_visualization.png",
            "measurement_comparison.png",
            "stage5_summary.json",
            "stage5_anthropometric_report.md",
        ]
        for fname in expected_files:
            fpath = STAGE5_OUTPUT_DIR / fname
            assert fpath.exists(), f"Expected artifact missing: {fpath}"
            assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"

    def test_summary_json_schema_and_metrics(self):
        """Verify structured summary JSON has PASSED status, 19 metrics, and valid tier counts."""
        sum_path = STAGE5_OUTPUT_DIR / "stage5_summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["status"] == "PASSED"
        assert summary["total_metrics_evaluated"] == 19
        assert summary["confidence_tier_counts"]["observation_supported"] == 3
        assert summary["confidence_tier_counts"]["partially_observed"] == 3
        assert summary["confidence_tier_counts"]["model_inferred"] == 12
        assert summary["confidence_tier_counts"]["invalid"] == 1

    def test_dual_mesh_difference_consistency(self):
        """Verify diff_mm = stage4_value_mm - stage2_value_mm across all measurements."""
        csv_path = STAGE5_OUTPUT_DIR / "dual_mesh_measurement_comparison.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s2 = float(row["stage2_value_mm"]) if row["stage2_value_mm"] != "nan" else float("nan")
                s4 = float(row["stage4_value_mm"]) if row["stage4_value_mm"] != "nan" else float("nan")
                diff = float(row["diff_mm"]) if row["diff_mm"] != "nan" else float("nan")

                if not math.isnan(s2) and not math.isnan(s4):
                    expected_diff = s4 - s2
                    assert math.isclose(diff, expected_diff, abs_tol=0.02), (
                        f"Metric {row['measurement_name']} diff inconsistent: {diff} vs {expected_diff}"
                    )

    def test_unobserved_regions_prior_invariance(self):
        """Verify unobserved prior-pinned limbs have exactly 0 mm difference between Stage 2 and Stage 4."""
        sum_path = STAGE5_OUTPUT_DIR / "stage5_summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        comp = summary["key_measurements_comparison"]

        prior_pinned_metrics = [
            "upper_arm_circumference_left",
            "upper_arm_circumference_right",
            "forearm_circumference_left",
            "forearm_circumference_right",
            "shoulder_breadth",
            "stature",
            "head_circumference",
            "shoulder_elbow_length",
            "shoulder_wrist_length",
        ]
        for name in prior_pinned_metrics:
            m = comp[name]
            assert abs(m["diff_mm"]) < 0.05, f"{name} expected 0 mm diff, got {m['diff_mm']} mm"
            assert m["status"] == "model_inferred"

    def test_observed_regions_meaningful_physical_deformation(self):
        """Verify observed rings (hip, thighs, waist) have non-zero, plausible deltas."""
        sum_path = STAGE5_OUTPUT_DIR / "stage5_summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        comp = summary["key_measurements_comparison"]

        # Hip circumference: high observation support
        hip = comp["hip_circumference"]
        assert hip["status"] == "observation_supported"
        assert hip["observed_coverage_percent"] > 90.0
        assert -15.0 < hip["diff_mm"] < -5.0  # -10.6 mm

        # Thighs: high observation support
        thigh_l = comp["mid_thigh_circumference_left"]
        assert thigh_l["status"] == "observation_supported"
        assert thigh_l["observed_coverage_percent"] > 75.0
        assert -20.0 < thigh_l["diff_mm"] < -5.0  # -13.6 mm

        thigh_r = comp["mid_thigh_circumference_right"]
        assert thigh_r["status"] == "observation_supported"
        assert thigh_r["observed_coverage_percent"] > 75.0
        assert 5.0 < thigh_r["diff_mm"] < 20.0  # +12.2 mm

        # Waist: partially observed
        waist = comp["waist_circumference"]
        assert waist["status"] == "partially_observed"
        assert 30.0 <= waist["observed_coverage_percent"] <= 50.0
        assert -12.0 < waist["diff_mm"] < -4.0  # -8.1 mm

    def test_neck_base_marked_invalid(self):
        """Verify neck base circumference is explicitly flagged as invalid due to shoulder slope intersection."""
        sum_path = STAGE5_OUTPUT_DIR / "stage5_summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        nb = summary["key_measurements_comparison"]["neck_base_circumference"]
        assert nb["status"] == "invalid"
        assert nb["stage4_mm"] > 800.0  # Inflated perimeter including biacromial shelf

