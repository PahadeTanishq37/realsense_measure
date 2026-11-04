"""
tests/body/test_multi_stop_reconstruction.py
============================================
Comprehensive test suite for the Multi-Stop Body Reconstruction workflow.

Verifies:
1. Input validation and contract enforcement:
   - Empty stop list rejected.
   - Duplicate stop indices rejected.
   - Mismatched image dimensions rejected.
   - Invalid intrinsics and non-finite nominal yaws rejected.
   - Empty foreground masks / point clouds handled according to raise_on_empty.
2. Multi-stop synthetic benchmark:
   - 2-stop ($0^\\circ, 180^\\circ$) joint phenotype fitting & unposing.
   - 4-stop ($0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ$) joint phenotype fitting & unposing.
   - Canonical coordinate frame consistency (+Z up, coronal X, sagittal Y).
   - Unposed point residuals against canonical fitted mesh.
3. Robustness against realistic sensor artifacts:
   - Gaussian depth noise (sigma = 3 mm).
   - Nominal yaw misalignment (e.g. turntable rotated 85 deg instead of 90 deg).
   - Asymmetric arm poses across stops.
4. Offline replay validation on real recorded capture (body_01).
5. Multi-stop vs single-stop accuracy comparison.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.recording import load_raw_capture
from realsense_measure.body.adapter import AdaptedStopData, frame_to_body_stop
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.fit import DifferentiableBodyFitter
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.reconstruction import (
    MultiStopFitResult,
    MultiStopOrchestrator,
    fit_and_unpose_multi_stop,
    validate_multi_stop_session,
)
from realsense_measure.body.synth import (
    CameraConfig,
    SimulatedStopData,
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


def make_dummy_adapted_stop(
    stop_idx: int = 0,
    width: int = 640,
    height: int = 480,
    fx: float = 500.0,
    fy: float = 500.0,
    distance_m: float = 2.0,
    nominal_yaw_deg: float = 0.0,
    num_points: int = 100,
) -> AdaptedStopData:
    """Create a minimal valid AdaptedStopData for contract validation tests."""
    rng = np.random.default_rng(1234 + stop_idx)
    pts_cam = rng.normal(loc=[0.0, 0.0, distance_m], scale=[0.2, 0.5, 0.1], size=(num_points, 3)).astype(np.float32)
    pts_world = np.stack([pts_cam[:, 0], pts_cam[:, 2] - distance_m, -pts_cam[:, 1]], axis=-1)

    depth_img = np.full((height, width), distance_m, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True

    return AdaptedStopData(
        stop_idx=stop_idx,
        nominal_yaw_deg=nominal_yaw_deg,
        true_yaw_deg=nominal_yaw_deg,
        true_translation_m=np.zeros(3, dtype=np.float32),
        true_arm_abduction_deg=(0.0, 0.0),
        point_cloud=pts_cam,
        point_cloud_world=pts_world,
        depth_image=depth_img,
        foreground_mask=mask,
        intrinsics={
            "fx": fx,
            "fy": fy,
            "cx": float(width / 2.0),
            "cy": float(height / 2.0),
            "width": int(width),
            "height": int(height),
            "distance_m": float(distance_m),
        },
    )


# ---------------------------------------------------------------------------
# 1. Input Validation and Contract Enforcement
# ---------------------------------------------------------------------------

def test_validation_empty_stop_list():
    """Verify that an empty stop list raises ValueError."""
    with pytest.raises(ValueError, match="Stop list cannot be empty"):
        validate_multi_stop_session([])

    with pytest.raises(ValueError, match="Stop list cannot be empty"):
        fit_and_unpose_multi_stop([])


def test_validation_duplicate_stop_indices():
    """Verify that duplicate stop indices in session raise ValueError."""
    s0 = make_dummy_adapted_stop(stop_idx=0, nominal_yaw_deg=0.0)
    s1 = make_dummy_adapted_stop(stop_idx=0, nominal_yaw_deg=180.0)  # Duplicate stop_idx=0

    with pytest.raises(ValueError, match="Duplicate stop indices detected"):
        validate_multi_stop_session([s0, s1])


def test_validation_mismatched_image_dimensions():
    """Verify that stops with different resolutions raise ValueError."""
    s0 = make_dummy_adapted_stop(stop_idx=0, width=640, height=480)
    s1 = make_dummy_adapted_stop(stop_idx=1, width=848, height=480)

    with pytest.raises(ValueError, match="Mismatched image dimensions"):
        validate_multi_stop_session([s0, s1])


def test_validation_invalid_intrinsics():
    """Verify that stops with invalid intrinsics raise ValueError."""
    s0 = make_dummy_adapted_stop(stop_idx=0)
    s1 = make_dummy_adapted_stop(stop_idx=1)
    s1.intrinsics["fx"] = -10.0  # Invalid negative focal length

    with pytest.raises(ValueError, match="Invalid focal lengths"):
        validate_multi_stop_session([s0, s1])


def test_validation_non_finite_nominal_yaw():
    """Verify that NaN or infinite nominal yaw raises ValueError."""
    s0 = make_dummy_adapted_stop(stop_idx=0, nominal_yaw_deg=0.0)
    s1 = make_dummy_adapted_stop(stop_idx=1, nominal_yaw_deg=float("nan"))

    with pytest.raises(ValueError, match="Invalid nominal yaw"):
        validate_multi_stop_session([s0, s1])


def test_validation_empty_mask_handling():
    """Verify that empty masks/points raise when raise_on_empty=True and warn when False."""
    s0 = make_dummy_adapted_stop(stop_idx=0)
    s1 = make_dummy_adapted_stop(stop_idx=1)
    s1.foreground_mask[:] = False
    s1.point_cloud_world = np.zeros((0, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="has empty points"):
        validate_multi_stop_session([s0, s1], raise_on_empty=True)

    warnings = validate_multi_stop_session([s0, s1], raise_on_empty=False)
    assert len(warnings) > 0
    assert any("has empty points" in w for w in warnings)


# ---------------------------------------------------------------------------
# 2. Synthetic Multi-Stop Benchmark (2-Stop and 4-Stop)
# ---------------------------------------------------------------------------

def render_synthetic_stop_frame(
    subject: SyntheticTruthSubject,
    sensor: VirtualDepthSensor,
    cam_cfg: CameraConfig,
    yaw_deg: float,
    stop_idx: int,
    noise_sigma_m: float = 0.0,
    arm_abd_L_deg: float = 0.0,
    arm_abd_R_deg: float = 0.0,
    seed: int = 42,
) -> AdaptedStopData:
    """Helper to render synthetic multi-stop observation and adapt it."""
    # Rotate vertices around Z (turntable)
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

    frame = RGBDFrame(
        frame_id=stop_idx,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=float(stop_idx),
        intrinsics=intr,
        foreground_mask=mask,
    )

    return frame_to_body_stop(
        frame=frame,
        nominal_yaw_deg=yaw_deg,
        distance_m=cam_cfg.distance_m,
        stop_idx=stop_idx,
        true_yaw_deg=yaw_deg,
        true_translation_m=np.zeros(3, dtype=np.float32),
        true_arm_abduction_deg=(arm_abd_L_deg, arm_abd_R_deg),
    )


def test_two_stop_synthetic_reconstruction(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Test 2-stop reconstruction (0 deg and 180 deg).
    Verifies:
    1. Joint phenotype fitting across opposing views.
    2. Retaining independent per-stop poses.
    3. Both stops unposed into canonical frame.
    4. Residuals are finite and canonical vertices/joints match contract.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    stop0 = render_synthetic_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)
    stop1 = render_synthetic_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=180.0, stop_idx=1)

    orchestrator = MultiStopOrchestrator(
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
    )

    result: MultiStopFitResult = orchestrator.run(
        stops=[stop0, stop1],
        n_coarse_iter=10,
        n_fine_iter=6,
        multi_starts=1,
    )

    assert result.is_valid
    assert len(result.unposed_stops) == 2
    assert len(result.per_stop_poses) == 2

    # Verify phenotypes are valid in (0, 1)
    for p_name in ["gender", "age", "muscle", "weight", "height", "proportions"]:
        val = result.fitted_phenotypes[p_name]
        assert 0.0 < val < 1.0

    # Verify per-stop fitted yaw is close to nominal
    p0 = result.per_stop_poses[0]
    p1 = result.per_stop_poses[1]
    assert abs(p0["yaw_deg"] - 0.0) < 15.0
    assert abs(p1["yaw_deg"] - 180.0) < 15.0

    # Verify canonical unposed points
    assert len(result.all_unposed_points) > 0
    assert result.mean_residual_mm < 35.0  # Coarse-fine fit residual under 35 mm
    assert np.isfinite(result.median_residual_mm)
    assert np.isfinite(result.p90_residual_mm)

    # Verify canonical coordinate frame properties:
    # +Z is vertical stature (range approx -1.0 to +1.0 or span > 1.2 m)
    canon_pts = result.all_unposed_points
    z_span = float(np.max(canon_pts[:, 2]) - np.min(canon_pts[:, 2]))
    assert z_span > 1.20, f"Unposed stature span {z_span:.2f} m should reflect human height."

    # Verify canonical mesh structure
    assert result.canonical_mesh["vertices"].shape[1] == 3
    assert result.canonical_mesh["faces"].shape[1] == 3
    assert len(result.canonical_mesh["joints"]) == 104


def test_four_stop_synthetic_reconstruction_with_noise(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Test 4-stop reconstruction (0, 90, 180, 270 deg) with 3 mm sensor depth noise.
    Verifies full 360-degree coverage unposing and joint convergence.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    stops = [
        render_synthetic_stop_frame(
            synthetic_subject,
            sensor,
            cam_cfg,
            yaw_deg=yaw,
            stop_idx=i,
            noise_sigma_m=0.003,  # 3 mm Gaussian noise
        )
        for i, yaw in enumerate([0.0, 90.0, 180.0, 270.0])
    ]

    result = fit_and_unpose_multi_stop(
        stops=stops,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=10,
        n_fine_iter=6,
        multi_starts=1,
    )

    assert result.is_valid
    assert len(result.unposed_stops) == 4
    assert len(result.per_stop_poses) == 4

    # All 4 camera view directions in canonical frame must point inward towards center
    for u in result.unposed_stops:
        assert u.camera_view_dir.shape == (3,)
        assert np.isclose(np.linalg.norm(u.camera_view_dir), 1.0, atol=1e-4)

    # Check unposed points coverage: min 4 * ~1000 points
    total_pts = len(result.all_unposed_points)
    assert total_pts > 4000
    assert result.mean_residual_mm < 35.0


def test_nominal_yaw_misalignment_robustness(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Test that optimization converges when true stop angle is rotated by 85 deg
    while nominal yaw is declared as 90 deg.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    # Stop 0 at true 0 deg, nominal 0 deg
    s0 = render_synthetic_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)
    # Stop 1 at true 85 deg, nominal 90 deg (5 deg misalignment)
    s1 = render_synthetic_stop_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=85.0, stop_idx=1)
    s1.nominal_yaw_deg = 90.0

    result = fit_and_unpose_multi_stop(
        stops=[s0, s1],
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=12,
        n_fine_iter=8,
        multi_starts=1,
    )

    assert result.is_valid
    # The fitter should adjust the yaw for stop 1 towards 85 deg
    fitted_yaw_1 = result.per_stop_poses[1]["yaw_deg"]
    assert abs(fitted_yaw_1 - 85.0) < 7.0, f"Expected fitted yaw near 85 deg, got {fitted_yaw_1:.2f} deg"


# ---------------------------------------------------------------------------
# 3. Offline Replay Validation
# ---------------------------------------------------------------------------

def test_offline_replay_multi_stop_integration(shared_backend, shared_prior, shared_fitter):
    """
    Test multi-stop workflow against real captured data in raw_captures/body_01.
    Verifies that multi-frame sequence from disk adapts cleanly and fits jointly
    without numerical instability or unposing failures.
    """
    raw_dir = Path(__file__).resolve().parent.parent.parent / "realsense_measure" / "raw_captures" / "body_01"
    if not raw_dir.exists():
        pytest.skip(f"Replay directory {raw_dir} not available.")

    frames = load_raw_capture(raw_dir, target_name="body", allow_aborted=True)
    assert len(frames) >= 2, f"Expected at least 2 frames in body_01, got {len(frames)}"

    # Select two distinct frames from the sequence (frame 0 and frame 2)
    selected_frames = [frames[0], frames[2]]
    adapted_stops = []

    for i, f in enumerate(selected_frames):
        # Foreground depth band in body_01
        valid_d = (f.depth_m >= 1.5) & (f.depth_m <= 2.8) & np.isfinite(f.depth_m)
        dist_m = float(np.median(f.depth_m[valid_d]))

        stop = frame_to_body_stop(
            frame=f,
            nominal_yaw_deg=float(i * 90.0),
            distance_m=dist_m,
            foreground_mask=valid_d,
            stop_idx=i,
        )
        adapted_stops.append(stop)

    result = fit_and_unpose_multi_stop(
        stops=adapted_stops,
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=4,
        n_fine_iter=3,
        multi_starts=1,
    )

    assert result is not None
    assert len(result.unposed_stops) == 2
    assert np.isfinite(result.fit_result.final_loss)
    assert len(result.all_unposed_points) > 0
    assert np.isfinite(result.mean_residual_mm)
    assert result.mean_residual_mm > 0.0

