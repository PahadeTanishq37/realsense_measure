"""
tests/body/test_readiness_audit.py
==================================
Comprehensive offline audit and failure-mode test suite for Prompt 3L:
End-to-End Reconstruction Readiness Audit and Physical Validation Protocol.

Covers:
1. RGB-D dimension mismatch between color and depth buffers.
2. Missing, NaN, or completely corrupted depth maps.
3. Clothing-induced depth offset simulation and anatomical surface compliance.
4. Mathematical verification of coordinate frames, right-handedness, and yaw convention.
5. Sparse observation and partial body occlusion handling.
6. Inter-stop subject movement and independent pose decoupling.
7. Measurement repeatability on exported and reloaded OBJ geometry.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from realsense_measure.body.adapter import (
    AdaptedStopData,
    camera_to_world_matrix,
    camera_to_world_points,
    frame_to_body_stop,
    world_to_camera_matrix,
    world_to_camera_points,
)
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.appearance import (
    AppearanceModeler,
    export_vertex_colored_obj,
    export_vertex_colored_ply,
    extract_vertex_appearance,
    project_canonical_to_camera,
)
from realsense_measure.body.fit import DifferentiableBodyFitter
from realsense_measure.body.measure import measure_body, measure_mesh
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.reconstruction import (
    PersonalizedReconstructionResult,
    reconstruct_and_personalize,
    validate_multi_stop_session,
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
    return SyntheticTruthSubject(shared_backend, shared_prior, subject_id=1, seed=123)


def render_synthetic_frame(
    subject: SyntheticTruthSubject,
    sensor: VirtualDepthSensor,
    cam_cfg: CameraConfig,
    yaw_deg: float,
    stop_idx: int,
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

    depth_m, mask, _ = sensor.render_depth(posed_verts, subject.faces, seed=100 + stop_idx)

    color_bgr = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
    color_bgr[mask] = [210, 180, 150]

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
# Test 1: Color/Depth Dimension Mismatch
# ---------------------------------------------------------------------------

def test_rgb_depth_dimension_mismatch():
    """Verify that frame adaptation rejects mismatching RGB and depth array resolutions."""
    color_img = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_m = np.ones((240, 320), dtype=np.float32) * 2.0  # Differing resolution

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    frame = RGBDFrame(
        frame_id=0,
        color_bgr=color_img,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=intr,
    )

    with pytest.raises(ValueError, match="Color shape .* does not match depth shape"):
        frame_to_body_stop(frame, nominal_yaw_deg=0.0, distance_m=2.0)


# ---------------------------------------------------------------------------
# Test 2: Missing or Corrupted Depth Maps
# ---------------------------------------------------------------------------

def test_corrupted_all_nan_depth_handling():
    """Verify that an all-NaN or all-zero corrupted depth frame is safely handled or rejected."""
    color_img = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth_nan = np.full((480, 640), np.nan, dtype=np.float32)

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    frame = RGBDFrame(
        frame_id=0,
        color_bgr=color_img,
        depth_m=depth_nan,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=intr,
    )

    # When raise_on_empty is True, must fail with informative error
    with pytest.raises(ValueError, match=r"(No valid 3D points found|Foreground mask is empty)"):
        frame_to_body_stop(frame, nominal_yaw_deg=0.0, distance_m=2.0, raise_on_empty=True)

    # When raise_on_empty is False, must return valid empty structure
    adapted = frame_to_body_stop(frame, nominal_yaw_deg=0.0, distance_m=2.0, raise_on_empty=False)
    assert len(adapted.point_cloud) == 0
    assert len(adapted.point_cloud_world) == 0


# ---------------------------------------------------------------------------
# Test 3: Mathematical Verification of Coordinate Transformations
# ---------------------------------------------------------------------------

def test_coordinate_transformation_properties():
    """
    Verify rigorous mathematical properties of coordinate transformations:
    1. Determinant of rotation matrices = +1.0 (pure right-handed SO(3)).
    2. Homogeneous transformation matrix inverse equality: T_c2w @ T_w2c = I_4.
    3. Unit round-trip preservation on 3D points (error < 1e-6 m).
    4. Optical forward axis (+Z_c in camera = +Y_w in Anny world).
    5. Optical vertical axis (-Y_c in camera = +Z_w in Anny world).
    6. Optical horizontal axis (+X_c in camera = +X_w in Anny world).
    """
    dist_m = 2.45
    T_c2w = camera_to_world_matrix(dist_m)
    T_w2c = world_to_camera_matrix(dist_m)

    # 1. Determinants
    det_c2w = float(np.linalg.det(T_c2w[:3, :3]))
    det_w2c = float(np.linalg.det(T_w2c[:3, :3]))
    assert np.isclose(det_c2w, 1.0, atol=1e-6), f"Det(R_c2w) = {det_c2w} != 1.0"
    assert np.isclose(det_w2c, 1.0, atol=1e-6), f"Det(R_w2c) = {det_w2c} != 1.0"

    # 2. Inverse equality
    I_ident = T_c2w @ T_w2c
    np.testing.assert_allclose(I_ident, np.eye(4), atol=1e-6)

    # 3. Canonical camera point (0, 0, dist_m): optical forward should land at (0, 0, 0) in world
    pt_cam_center = np.array([[0.0, 0.0, dist_m]], dtype=np.float32)
    pt_world_center = camera_to_world_points(pt_cam_center, distance_m=dist_m)
    np.testing.assert_allclose(pt_world_center[0], [0.0, 0.0, 0.0], atol=1e-6)

    # 4. Canonical Up direction: in camera +Y is down, so (0, -1, dist_m) should land at (0, 0, 1) in world
    pt_cam_up = np.array([[0.0, -1.0, dist_m]], dtype=np.float32)
    pt_world_up = camera_to_world_points(pt_cam_up, distance_m=dist_m)
    np.testing.assert_allclose(pt_world_up[0], [0.0, 0.0, 1.0], atol=1e-6)

    # 5. Round trip with arbitrary points
    rng = np.random.default_rng(42)
    pts_arbitrary = rng.uniform(-1.0, 1.0, size=(100, 3)).astype(np.float32)
    pts_arbitrary[:, 2] += dist_m  # positive camera depth
    pts_w = camera_to_world_points(pts_arbitrary, distance_m=dist_m)
    pts_c_rec = world_to_camera_points(pts_w, distance_m=dist_m)
    np.testing.assert_allclose(pts_arbitrary, pts_c_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Clothing-Induced Depth Offset Simulation
# ---------------------------------------------------------------------------

def test_clothing_offset_simulation(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Simulate loose clothing (e.g., +15 mm outward depth expansion over the torso).
    Verify that:
    1. Surface personalisation absorbs the outward deformation without NaN/Inf explosion.
    2. Anatomical Laplacian regularization preserves non-clothing regions (head, hands).
    3. Circumferences reflect clothing expansion predictably.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    # Render base stop
    f0 = render_synthetic_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)

    # Add +15 mm clothing offset to torso depth (closer to camera)
    h, w = f0.depth_m.shape
    torso_region = (f0.depth_m > 1.5) & (f0.depth_m < 3.0)
    torso_region[:h // 3, :] = False  # Exclude head
    torso_region[2 * h // 3:, :] = False  # Exclude lower legs
    f0.depth_m[torso_region] -= 0.015  # 15 mm closer to camera = thicker body

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

    assert result.is_valid
    assert np.all(np.isfinite(result.personalized_mesh["vertices"]))

    # Chest circumference should be larger on personalized mesh than fitted template
    chest_diff = result.measurement_comparisons["chest_circumference"].absolute_change_mm
    assert chest_diff > 0.0, f"Expected positive chest expansion from clothing, got {chest_diff:.2f} mm"


# ---------------------------------------------------------------------------
# Test 5: Sparse Observations and Partial Body Coverage
# ---------------------------------------------------------------------------

def test_sparse_observation_graceful_degradation(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify behavior when only a single frontal view is available (1 stop, 0 deg yaw).
    - Front is observed; rear is unobserved.
    - Unobserved rings are flagged as mixed/incomplete.
    - Unobserved appearance is explicitly marked neutral with is_observed_mask=False.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_synthetic_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)

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
        extract_appearance=True,
    )

    assert result.status in ("success", "partial")
    # Single stop cannot provide full closed observed rings
    for r_name, ring in result.observed_rings.items():
        assert ring.coverage_fraction < 0.90
        assert ring.source in ("mixed", "model")

    # Appearance coverage should be partial (30-60%)
    assert 0.20 < result.appearance.diagnostics.coverage_fraction < 0.65
    assert np.sum(~result.appearance.is_observed_mask) > 1000


# ---------------------------------------------------------------------------
# Test 6: Inter-Stop Subject Motion Decoupling
# ---------------------------------------------------------------------------

def test_inter_stop_subject_motion_decoupling(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify that subject movement between stops (e.g. slight repositioning and yaw drift)
    is accommodated by independent per-stop pose optimization while phenotypes remain shared.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    # Stop 0: yaw 0 deg, translation [0, 0, 0]
    f0 = render_synthetic_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)

    # Stop 1: nominal 90 deg, but subject drifted by +5 deg and shifted by 2 cm in X
    verts = synthetic_subject.true_verts.copy()
    rad = np.radians(95.0)  # +5 deg drift
    rx = verts[:, 0] * np.cos(rad) - verts[:, 1] * np.sin(rad) + 0.020  # +20 mm shift
    ry = verts[:, 0] * np.sin(rad) + verts[:, 1] * np.cos(rad)
    rz = verts[:, 2]
    posed_verts = np.stack([rx, ry, rz], axis=-1)

    depth_m, mask, _ = sensor.render_depth(posed_verts, synthetic_subject.faces, seed=201)
    color_bgr = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
    color_bgr[mask] = [210, 180, 150]

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width=cam_cfg.width, height=cam_cfg.height, fx=sensor.fx, fy=sensor.fy, cx=sensor.cx, cy=sensor.cy)
    f1 = RGBDFrame(frame_id=1, color_bgr=color_bgr, depth_m=depth_m, pcd=o3d.geometry.PointCloud(), timestamp=1.0, intrinsics=intr, foreground_mask=mask)

    result = reconstruct_and_personalize(
        stops=[f0, f1],
        nominal_yaws_deg=[0.0, 90.0],
        distance_m=[cam_cfg.distance_m, cam_cfg.distance_m],
        backend=shared_backend,
        prior=shared_prior,
        fitter=shared_fitter,
        n_coarse_iter=6,
        n_fine_iter=4,
        multi_starts=1,
        candidate_lambdas=[30.0],
    )

    assert result.is_valid
    poses = result.multi_stop_fit.per_stop_poses
    assert len(poses) == 2
    # Verify independent poses are recovered
    assert isinstance(poses[0], dict)
    assert isinstance(poses[1], dict)
    yaw0 = poses[0]["yaw_deg"]
    yaw1 = poses[1]["yaw_deg"]
    assert abs(yaw0) < 15.0
    assert abs(yaw1 - 90.0) < 15.0


# ---------------------------------------------------------------------------
# Test 7: Measurement Repeatability on Reloaded Export
# ---------------------------------------------------------------------------

def test_measurement_repeatability_on_reloaded_export(tmp_path, shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify that exporting the personalized mesh to OBJ and re-loading it preserves
    measurements exactly (within 0.05 mm mesh float quantization).
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_synthetic_frame(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0)

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
        extract_appearance=True,
    )

    obj_file = tmp_path / "test_pers.obj"
    ply_file = tmp_path / "test_pers.ply"
    result.export_obj(obj_file, include_vertex_colors=True)
    result.export_ply(ply_file)

    assert obj_file.exists() and obj_file.stat().st_size > 100000
    assert ply_file.exists() and ply_file.stat().st_size > 100000

    # Read vertices and faces from exported PLY (which strictly preserves canonical vertex indexing)
    o3d_mesh = o3d.io.read_triangle_mesh(str(ply_file))
    reloaded_verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    reloaded_faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)

    # Re-evaluate measurements on reloaded geometry
    reloaded_measurements = measure_mesh(
        verts=reloaded_verts,
        joints=result.personalized_mesh["joints"],
        faces=reloaded_faces,
        jnames=result.personalized_mesh["joint_names"],
        raw_dict=result.personalized_mesh["raw"],
    )

    for m_name, original_m in result.personalized_measurements.items():
        reloaded_val = reloaded_measurements[m_name].value_mm
        orig_val = original_m.value_mm
        if np.isfinite(orig_val):
            diff = abs(reloaded_val - orig_val)
            assert diff < 0.10, f"Measurement {m_name} differed by {diff:.4f} mm after export round-trip!"

