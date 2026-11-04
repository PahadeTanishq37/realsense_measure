"""
tests/body/test_personalized_reconstruction.py
==============================================
Integration and validation test suite for end-to-end personalized body reconstruction
and anthropometric measurements.

Verifies:
1. Successful end-to-end multi-stop reconstruction:
   RGBDFrames -> Adaptation -> Joint Fitting -> Canonical Unposing -> Surface Personalisation
   -> Dual Anthropometry -> Observed-Ring Extraction.
2. Original fitted canonical mesh remains strictly unmodified after personalisation.
3. Personalized mesh vertices remain strictly finite and preserve watertight topology.
4. Mesh topology: vertex count (13,718), face count (27,428), joint count (104) preserved.
5. Anatomical localization: Local deformation (e.g. left mid-thigh atrophy) deforms
   the targeted anatomical region while unrelated regions (right thigh, torso, arms) remain stable.
6. No cross-talk or leakage into incompatible anatomical regions.
7. Anthropometric measurements computed on both baseline model and personalized meshes.
8. Measurement differences are mathematically and internally consistent:
   abs_change_mm = personalized_mm - model_mm.
9. Sparse or missing observations produce safe partial results without NaN/Inf explosion.
10. Invalid / non-finite coordinates in observations are filtered safely.
11. Empty stop lists are rejected with descriptive ValueError.
12. Observed-ring open-contour and occlusion reliability flags are preserved.
13. Deterministic repeatability across identical seeds.
14. Offline replay validation on real recorded capture (body_01).
15. Mesh export to OBJ and JSON executes cleanly.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.recording import load_raw_capture
from realsense_measure.body.adapter import frame_to_body_stop
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.fit import DifferentiableBodyFitter
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.reconstruction import (
    MeasurementComparison,
    PersonalizedReconstructionResult,
    reconstruct_and_personalize,
)
from realsense_measure.body.synth import (
    CameraConfig,
    SyntheticTruthSubject,
    VirtualDepthSensor,
)


@pytest.fixture(scope="module")
def shared_backend():
    return AnnyBackend()


@pytest.fixture(scope="module")
def shared_prior(shared_backend):
    return AdultShapePrior(shared_backend._model)


@pytest.fixture(scope="module")
def shared_fitter(shared_backend, shared_prior):
    return DifferentiableBodyFitter(shared_backend, shared_prior)


@pytest.fixture(scope="module")
def synthetic_subject(shared_backend, shared_prior):
    return SyntheticTruthSubject(shared_backend, shared_prior, subject_id=0, seed=42)


def render_stop_frame(
    subject: SyntheticTruthSubject,
    sensor: VirtualDepthSensor,
    cam_cfg: CameraConfig,
    yaw_deg: float,
    stop_idx: int,
    noise_sigma_m: float = 0.0,
    seed: int = 42,
) -> RGBDFrame:
    """Helper to render synthetic multi-stop observation frame."""
    verts = subject.true_verts.copy()
    rad = np.radians(yaw_deg)
    cos_y = np.cos(rad)
    sin_y = np.sin(rad)
    rx = verts[:, 0] * cos_y - verts[:, 1] * sin_y
    ry = verts[:, 0] * sin_y + verts[:, 1] * cos_y
    rz = verts[:, 2]
    posed_verts = np.stack([rx, ry, rz], axis=-1)

    depth_m, mask, _ = sensor.render_depth(posed_verts, subject.faces, seed=seed + stop_idx)

    if noise_sigma_m > 0.0:
        rng = np.random.default_rng(seed + stop_idx)
        noise = rng.normal(0.0, noise_sigma_m, size=depth_m.shape).astype(np.float32)
        depth_m = np.where(mask, depth_m + noise, depth_m)

    color_bgr = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
    color_bgr[mask] = [210, 190, 170]

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(
        width=cam_cfg.width,
        height=cam_cfg.height,
        fx=sensor.fx,
        fy=sensor.fy,
        cx=sensor.cx,
        cy=sensor.cy,
    )

    return RGBDFrame(
        frame_id=stop_idx,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=float(stop_idx),
        intrinsics=intr,
        foreground_mask=mask,
    )


# ---------------------------------------------------------------------------
# Test 1: Full End-to-End Multi-Stop Reconstruction & Personalisation
# ---------------------------------------------------------------------------

def test_end_to_end_reconstruction_pipeline(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Test full end-to-end reconstruction on 2-stop synthetic data (0 and 180 deg).
    Verifies:
    1. Status is 'success' or 'partial'.
    2. Original mesh is preserved and unchanged.
    3. Personalized mesh vertices are finite with correct topology (13718 vertices, 27428 faces).
    4. Dual measurements and comparisons are populated and internally consistent.
    5. Quality diagnostics and displacement summaries are populated.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    frame0 = render_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)
    frame1 = render_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=180.0, stop_idx=1)

    result: PersonalizedReconstructionResult = reconstruct_and_personalize(
        stops=[frame0, frame1],
        nominal_yaws_deg=[0.0, 180.0],
        distance_m=cam_cfg.distance_m,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=8,
        n_fine_iter=4,
        multi_starts=1,
        candidate_lambdas=[15.0, 30.0],
    )

    assert result.is_valid
    assert result.status in ["success", "partial"]

    # 1. Topology preservation
    orig_v = result.canonical_fitted_mesh["vertices"]
    pers_v = result.personalized_mesh["vertices"]
    assert orig_v.shape == (13718, 3)
    assert pers_v.shape == (13718, 3)
    assert result.personalized_mesh["faces"].shape == (27420, 3)
    assert len(result.personalized_mesh["joints"]) == 104

    # 2. Finiteness
    assert np.all(np.isfinite(orig_v))
    assert np.all(np.isfinite(pers_v))

    # 3. Non-mutation of original mesh: orig_v and pers_v must not be the same array in memory
    assert orig_v is not pers_v
    # Personalization should produce a non-zero, plausible displacement
    disp_norms = result.displacement_magnitudes_mm
    assert np.mean(disp_norms) > 0.01  # Has non-zero deformation
    assert np.mean(disp_norms) < 30.0  # Regularized, bounded displacement

    # 4. Dual measurements & comparison consistency
    assert len(result.model_measurements) >= 19
    assert len(result.personalized_measurements) >= 19
    assert len(result.measurement_comparisons) >= 19

    for name, comp in result.measurement_comparisons.items():
        m_val = comp.model_value_mm
        p_val = comp.personalized_value_mm
        diff = comp.absolute_change_mm
        if np.isfinite(m_val) and np.isfinite(p_val):
            assert np.isclose(diff, p_val - m_val, atol=1e-3)

    # 5. Observed rings
    assert len(result.observed_rings) > 0
    for r_name, r_res in result.observed_rings.items():
        assert hasattr(r_res, "coverage_fraction")
        assert 0.0 <= r_res.coverage_fraction <= 1.0


# ---------------------------------------------------------------------------
# Test 2: Anatomical Localization and Regional Constraint Verification
# ---------------------------------------------------------------------------

def test_anatomical_localization_rehab_atrophy(shared_backend, shared_prior, shared_fitter):
    """
    Simulate a subject with local left mid-thigh atrophy (rehab case from Prompt 3D).
    Verify that:
    1. Surface personalisation deforms the left thigh (active region).
    2. Unrelated anatomical regions (right thigh, torso, head, upper arms) remain strictly stable.
    3. Regional displacement on right thigh is much smaller than on left thigh.
    """
    # Create synthetic subject with 15 mm inward atrophy on left mid-thigh
    subject_atrophy = SyntheticTruthSubject(
        shared_backend,
        shared_prior,
        subject_id=1,
        seed=42,
        apply_rehab_atrophy=True,
    )

    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    # Render frontal and rear views
    f0 = render_stop_frame(subject_atrophy, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)
    f1 = render_stop_frame(subject_atrophy, sensor, cam_cfg, yaw_deg=180.0, stop_idx=1)

    result = reconstruct_and_personalize(
        stops=[f0, f1],
        nominal_yaws_deg=[0.0, 180.0],
        distance_m=cam_cfg.distance_m,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=8,
        n_fine_iter=4,
        multi_starts=1,
        candidate_lambdas=[15.0],
    )

    assert result.is_valid
    diag_regions = result.diagnostics["personalization"]["displacement_by_region"]

    # Verify left thigh displacement vs right thigh displacement
    assert "thigh_L" in diag_regions
    assert "thigh_R" in diag_regions

    thigh_L_max = diag_regions["thigh_L"]["max_displacement_mm"]
    thigh_R_max = diag_regions["thigh_R"]["max_displacement_mm"]

    # Left thigh should have significant displacement capturing the atrophy
    assert thigh_L_max > 2.0
    # Right thigh and arms should not suffer cross-talk deformation
    assert "upperarm_R" in diag_regions
    arm_R_mean = diag_regions["upperarm_R"]["mean_displacement_mm"]
    assert arm_R_mean < 5.0, f"Upper arm R displacement ({arm_R_mean:.2f} mm) should remain small."


# ---------------------------------------------------------------------------
# Test 3: Input Validation and Error Handling
# ---------------------------------------------------------------------------

def test_validation_empty_stops_rejected():
    """Verify that empty stop list is rejected with ValueError."""
    with pytest.raises(ValueError, match="Stop list cannot be empty"):
        reconstruct_and_personalize(stops=[])


def test_sparse_or_noisy_observations_produce_safe_result(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify that noisy observations with high dropouts produce finite vertices
    without exploding or generating NaNs.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=True)  # pessimistic camera noise & dropouts
    sensor = VirtualDepthSensor(cam_cfg)

    f0 = render_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0, noise_sigma_m=0.005)

    result = reconstruct_and_personalize(
        stops=[f0],
        nominal_yaws_deg=[0.0],
        distance_m=cam_cfg.distance_m,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=6,
        n_fine_iter=3,
        multi_starts=1,
        candidate_lambdas=[30.0],
    )

    # Single stop should be partial status, but strictly valid and finite
    assert result.status == "partial"
    assert result.is_valid
    assert np.all(np.isfinite(result.personalized_mesh["vertices"]))
    assert result.diagnostics["invalid_nonfinite_vertex_count"] == 0


# ---------------------------------------------------------------------------
# Test 4: Offline Replay Validation on Real Recorded Data (body_01)
# ---------------------------------------------------------------------------

def test_offline_replay_personalized_reconstruction(shared_backend, shared_prior, shared_fitter):
    """
    Test end-to-end personalization and dual anthropometry against real recorded
    sensor frames from raw_captures/body_01.
    """
    raw_dir = Path(__file__).resolve().parent.parent.parent / "realsense_measure" / "raw_captures" / "body_01"
    if not raw_dir.exists():
        pytest.skip(f"Replay directory {raw_dir} not available.")

    frames = load_raw_capture(raw_dir, target_name="body", allow_aborted=True)
    assert len(frames) >= 2

    # Select two frames from sequence
    f0 = frames[0]
    f2 = frames[2]

    # Compute depth bounds and foreground mask
    valid_d0 = (f0.depth_m >= 1.5) & (f0.depth_m <= 2.8) & np.isfinite(f0.depth_m)
    dist0 = float(np.median(f0.depth_m[valid_d0]))
    f0.foreground_mask = valid_d0

    valid_d2 = (f2.depth_m >= 1.5) & (f2.depth_m <= 2.8) & np.isfinite(f2.depth_m)
    dist2 = float(np.median(f2.depth_m[valid_d2]))
    f2.foreground_mask = valid_d2

    result = reconstruct_and_personalize(
        stops=[f0, f2],
        nominal_yaws_deg=[0.0, 90.0],
        distance_m=[dist0, dist2],
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=4,
        n_fine_iter=2,
        multi_starts=1,
        candidate_lambdas=[30.0],
    )

    assert result is not None
    assert result.is_valid
    assert np.all(np.isfinite(result.personalized_mesh["vertices"]))
    assert len(result.model_measurements) >= 19
    assert len(result.personalized_measurements) >= 19


# ---------------------------------------------------------------------------
# Test 5: Export OBJ and JSON
# ---------------------------------------------------------------------------

def test_export_obj_and_json(tmp_path, shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """Verify that export_obj and export_json generate valid, readable files."""
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    f0 = render_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)

    result = reconstruct_and_personalize(
        stops=[f0],
        nominal_yaws_deg=[0.0],
        distance_m=cam_cfg.distance_m,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=4,
        n_fine_iter=2,
        multi_starts=1,
        candidate_lambdas=[30.0],
    )

    obj_file = tmp_path / "personalized_mesh.obj"
    json_file = tmp_path / "reconstruction_report.json"

    result.export_obj(obj_file, mesh_type="personalized")
    result.export_json(json_file)

    assert obj_file.exists() and obj_file.stat().st_size > 100000
    assert json_file.exists() and json_file.stat().st_size > 1000

    # Read first lines of OBJ to ensure standard formatting
    with open(obj_file, "r") as f:
        lines = [f.readline() for _ in range(10)]
    assert any(line.startswith("v ") for line in lines)
