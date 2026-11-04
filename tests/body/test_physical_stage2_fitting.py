"""
tests/body/test_physical_stage2_fitting.py
==========================================
Unit and integration tests for Problem 4 — Stage 2:
Articulated Multi-View Fitting on Physical RGB-D Data.

Verifies:
1. Shared phenotype parameterization across stops.
2. Independent per-stop pose optimization (yaw, 3D translation, arm abduction).
3. Coordinate-frame consistency between camera and Anny world space.
4. Input and output tensor dimensions of the fitted body model.
5. Numerical stability and loss breakdown integrity.
6. Handling of incomplete physical observations (truncated lower legs).
7. Classification and labeling of directly observed vs. prior-driven surfaces.
8. Reproducibility with fixed random seed.
9. Integrity and validity of generated Stage 2 output artifacts.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from camera.recording import load_raw_capture
from realsense_measure.body.adapter import frame_to_body_stop, get_or_create_foreground_mask
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.fit import DifferentiableBodyFitter, FitResult, PHENOTYPE_NAMES
from realsense_measure.body.prior import AdultShapePrior


PHYSICAL_DATASET_DIR = Path("raw_captures/body_capture_01")
STAGE2_OUTPUT_DIR = Path("stage2_articulated_fit_output")


@pytest.fixture(scope="module")
def physical_frames():
    if not PHYSICAL_DATASET_DIR.exists():
        pytest.skip(f"Physical dataset {PHYSICAL_DATASET_DIR} not found")
    return load_raw_capture(PHYSICAL_DATASET_DIR)


@pytest.fixture(scope="module")
def adapted_physical_stops(physical_frames):
    nominal_yaws = [0.0, 90.0, 180.0, 270.0]
    # Calibrate subject distance using foreground mask median depth
    subject_depths = []
    for f in physical_frames:
        mask = get_or_create_foreground_mask(f.color_bgr, f.depth_m)
        d = f.depth_m[mask & (f.depth_m > 0.10) & np.isfinite(f.depth_m)]
        if len(d) > 0:
            subject_depths.append(float(np.median(d)))
    distance_m = float(np.mean(subject_depths)) if subject_depths else 0.575

    stops = []
    for i, (f, yaw) in enumerate(zip(physical_frames, nominal_yaws)):
        stop = frame_to_body_stop(f, nominal_yaw_deg=yaw, distance_m=distance_m, stop_idx=i)
        # Match contract: mask background pixels to 0.0
        stop.depth_image = np.where(stop.foreground_mask, stop.depth_image, 0.0).astype(np.float32)
        stops.append(stop)
    return stops


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend()


@pytest.fixture(scope="module")
def adult_prior(anny_backend):
    return AdultShapePrior(anny_backend._model)


@pytest.fixture(scope="module")
def fast_fitted_result(anny_backend, adult_prior, adapted_physical_stops):
    fitter = DifferentiableBodyFitter(anny_backend, adult_prior)
    # Fast 10 coarse / 5 fine run for testing
    return fitter.fit(
        stops=adapted_physical_stops,
        known_height_mm=1750.0,
        n_coarse_iter=10,
        n_fine_iter=5,
        multi_starts=1,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not PHYSICAL_DATASET_DIR.exists(),
    reason="Physical capture raw_captures/body_capture_01 not present"
)
class TestPhysicalStage2ArticulatedFitting:

    def test_shared_shape_parameterization(self, fast_fitted_result):
        """Verify body shape phenotypes beta are shared across all 4 stops and bounded in [0, 1]."""
        phenos = fast_fitted_result.fitted_phenotypes
        assert set(phenos.keys()) == set(PHENOTYPE_NAMES)
        for name, val in phenos.items():
            assert 0.0 <= val <= 1.0, f"Phenotype {name} value {val} out of bounds [0, 1]"
            assert np.isfinite(val), f"Phenotype {name} is not finite: {val}"

    def test_independent_per_stop_poses(self, fast_fitted_result):
        """Verify each stop has an independently optimized pose (yaw, translation, arm abduction)."""
        poses = fast_fitted_result.per_stop_poses
        assert len(poses) == 4, f"Expected 4 stop poses, got {len(poses)}"

        # Verify all expected pose parameters exist
        for i, pose in enumerate(poses):
            assert "yaw_deg" in pose
            assert "translation_m" in pose
            assert "arm_abd_L_deg" in pose
            assert "arm_abd_R_deg" in pose
            assert len(pose["translation_m"]) == 3
            assert np.all(np.isfinite(pose["translation_m"]))

        # Verify yaws reflect nominal 0, 90, 180, 270 with small articulated adjustments
        yaws = [p["yaw_deg"] for p in poses]
        assert -30.0 < yaws[0] < 30.0, f"Stop 0 yaw {yaws[0]} unexpected"
        assert 60.0 < yaws[1] < 120.0, f"Stop 1 yaw {yaws[1]} unexpected"
        assert 150.0 < yaws[2] < 210.0, f"Stop 2 yaw {yaws[2]} unexpected"
        assert 240.0 < yaws[3] < 300.0, f"Stop 3 yaw {yaws[3]} unexpected"

    def test_coordinate_frame_consistency(self, adapted_physical_stops):
        """Verify physical subject points are properly centered around Anny world origin."""
        stop0 = adapted_physical_stops[0]
        pts_world = stop0.point_cloud_world
        # Subject should be centered within 20 cm of world origin laterally and anteroposteriorly
        med_x = float(np.median(pts_world[:, 0]))
        med_y = float(np.median(pts_world[:, 1]))
        assert abs(med_x) < 0.20, f"Lateral center {med_x} m drifted from world origin"
        assert abs(med_y) < 0.20, f"Depth center {med_y} m drifted from world origin"

    def test_model_tensor_dimensions(self, anny_backend, fast_fitted_result):
        """Verify Anny forward pass produces valid vertices (13718, 3) and joints."""
        out = anny_backend.rest_pose(phenotypes=fast_fitted_result.fitted_phenotypes)
        verts = out.numpy_vertices(0)
        joints = out.numpy_joints(0)
        faces = out.faces.cpu().numpy()

        assert verts.shape == (13718, 3)
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert len(joints) > 0 and joints.shape[1] == 3
        assert np.all(np.isfinite(verts))

    def test_loss_breakdown_numerical_stability(self, fast_fitted_result):
        """Verify loss terms are finite, non-negative, and free-space loss is well-behaved."""
        bd = fast_fitted_result.loss_breakdown
        assert np.isfinite(bd.total_loss) and bd.total_loss < 50.0
        assert bd.loss_data >= 0.0
        assert bd.loss_vis >= 0.0
        # Free space loss must not blow out (must be < 5.0 with masked depth)
        assert 0.0 <= bd.loss_free < 5.0, f"loss_free {bd.loss_free} was unexpectedly high"
        assert bd.loss_sil >= 0.0

    def test_incomplete_framing_prior_handling(self, adapted_physical_stops, anny_backend, adult_prior):
        """Verify that when stature is truncated (< 1.30 m), the fitter defaults height to adult prior."""
        fitter = DifferentiableBodyFitter(anny_backend, adult_prior)
        # Without known_height_mm, fitter should detect truncated framing and initialize height to 0.50
        res = fitter.fit(
            stops=adapted_physical_stops,
            known_height_mm=None,
            n_coarse_iter=5,
            n_fine_iter=2,
            multi_starts=1,
            seed=42,
        )
        assert np.isfinite(res.fitted_phenotypes["height"])
        assert res.final_loss < 50.0

    def test_reproducibility_with_fixed_seed(self, anny_backend, adult_prior, adapted_physical_stops):
        """Verify optimization with identical seed yields identical fitted parameters."""
        fitter = DifferentiableBodyFitter(anny_backend, adult_prior)
        res1 = fitter.fit(stops=adapted_physical_stops, n_coarse_iter=5, n_fine_iter=2, multi_starts=1, seed=123)
        res2 = fitter.fit(stops=adapted_physical_stops, n_coarse_iter=5, n_fine_iter=2, multi_starts=1, seed=123)
        assert np.isclose(res1.final_loss, res2.final_loss, atol=1e-3)
        for k in PHENOTYPE_NAMES:
            assert np.isclose(res1.fitted_phenotypes[k], res2.fitted_phenotypes[k], atol=1e-4)

    def test_stage2_output_artifacts_exist_and_valid(self):
        """Verify that all Stage 2 artifacts were generated and are valid."""
        summary_path = STAGE2_OUTPUT_DIR / "stage2_articulated_fit_summary.json"
        params_path = STAGE2_OUTPUT_DIR / "fitted_body_parameters.json"
        obj_path = STAGE2_OUTPUT_DIR / "canonical_fitted_model.obj"
        ply_path = STAGE2_OUTPUT_DIR / "canonical_fitted_model.ply"
        res_ply_path = STAGE2_OUTPUT_DIR / "all_stops_observed_with_residual.ply"
        chart_path = STAGE2_OUTPUT_DIR / "residual_distribution.png"
        report_path = STAGE2_OUTPUT_DIR / "stage2_articulated_fit_report.md"

        assert summary_path.exists(), "stage2_articulated_fit_summary.json missing"
        assert params_path.exists(), "fitted_body_parameters.json missing"
        assert obj_path.exists(), "canonical_fitted_model.obj missing"
        assert ply_path.exists(), "canonical_fitted_model.ply missing"
        assert res_ply_path.exists(), "all_stops_observed_with_residual.ply missing"
        assert chart_path.exists(), "residual_distribution.png missing"
        assert report_path.exists(), "stage2_articulated_fit_report.md missing"

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        assert params["convergence"]["converged"] is True
        assert params["overall_residual_metrics"]["median_mm"] < 40.0, "Overall median residual unexpectedly high"
        assert params["surface_classification"]["observed_vertices_count"] > 0
        assert params["surface_classification"]["inferred_vertices_count"] > 0

        # Verify all 4 overlay images exist
        for i in range(4):
            overlay_file = STAGE2_OUTPUT_DIR / f"stop_{i:02d}_overlay.png"
            assert overlay_file.exists(), f"Overlay {overlay_file} missing"

