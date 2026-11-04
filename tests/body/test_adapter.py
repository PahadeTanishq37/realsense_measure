"""
tests/body/test_adapter.py
==========================
Comprehensive unit and integration tests for RGB-D-to-Anny Stop Adapter.

Verifies:
1. Valid synthetic RGB-D frame converts successfully.
2. Output fields match the exact SimulatedStopData contract.
3. Camera-to-world transformation produces expected coordinates and preserves handedness.
4. Round-trip transformation recovers the original points within numerical tolerance.
5. RGB/depth shape mismatch is rejected.
6. Empty foreground mask is handled explicitly.
7. Invalid or missing intrinsics/parameters are rejected.
8. Invalid depth values (NaN, Inf, 0, negative) do not produce NaN/Inf coordinates.
9. Indexed point-cloud source-pixel provenance remains available.
10. A saved replay frame can be adapted without physical camera access.
11. Adapted stop is directly accepted by DifferentiableBodyFitter.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.recording import load_raw_capture
from realsense_measure.body.adapter import (
    AdaptedStopData,
    R_CAM_TO_WORLD,
    R_WORLD_TO_CAM,
    camera_to_world_matrix,
    camera_to_world_points,
    extract_intrinsics_dict,
    frame_to_body_stop,
    world_to_camera_matrix,
    world_to_camera_points,
)
from realsense_measure.body.synth import SimulatedStopData


@pytest.fixture
def synthetic_pinhole_intrinsics():
    """Deterministic pinhole intrinsics for 640x480 frame."""
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    return intr


@pytest.fixture
def synthetic_rgbd_frame(synthetic_pinhole_intrinsics):
    """
    Synthetic RGB-D frame representing a subject at 1.5 m distance.
    Creates a central rectangular foreground blob [160:320, 240:400] at 1.5 m.
    """
    h, w = 480, 640
    color_bgr = np.full((h, w, 3), 128, dtype=np.uint8)
    depth_m = np.zeros((h, w), dtype=np.float32)

    # Subject box in center
    depth_m[160:320, 240:400] = 1.50
    color_bgr[160:320, 240:400] = [180, 160, 140]

    mask = np.zeros((h, w), dtype=bool)
    mask[160:320, 240:400] = True

    pcd = o3d.geometry.PointCloud()

    return RGBDFrame(
        frame_id=1,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=pcd,
        timestamp=12345.678,
        intrinsics=synthetic_pinhole_intrinsics,
        foreground_mask=mask,
    )


# ---------------------------------------------------------------------------
# Test 1: Valid synthetic RGB-D frame converts successfully
# ---------------------------------------------------------------------------

def test_valid_synthetic_frame_conversion(synthetic_rgbd_frame):
    stop = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=90.0,
        distance_m=1.60,
        stop_idx=2,
    )

    assert isinstance(stop, SimulatedStopData)
    assert isinstance(stop, AdaptedStopData)
    assert stop.stop_idx == 2
    assert stop.nominal_yaw_deg == 90.0
    assert len(stop.point_cloud) > 0
    assert len(stop.point_cloud_world) == len(stop.point_cloud)


# ---------------------------------------------------------------------------
# Test 2: Output fields match exact SimulatedStopData contract
# ---------------------------------------------------------------------------

def test_output_fields_match_simulated_stop_data_contract(synthetic_rgbd_frame):
    stop = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=180.0,
        distance_m=1.50,
        stop_idx=0,
    )

    # Verify all required dataclass fields are present and correctly typed
    assert hasattr(stop, "stop_idx") and isinstance(stop.stop_idx, int)
    assert hasattr(stop, "nominal_yaw_deg") and isinstance(stop.nominal_yaw_deg, float)
    assert hasattr(stop, "true_yaw_deg") and isinstance(stop.true_yaw_deg, float)
    assert hasattr(stop, "true_translation_m") and isinstance(stop.true_translation_m, np.ndarray)
    assert stop.true_translation_m.shape == (3,)
    assert hasattr(stop, "true_arm_abduction_deg") and isinstance(stop.true_arm_abduction_deg, tuple)
    assert len(stop.true_arm_abduction_deg) == 2

    assert hasattr(stop, "depth_image") and isinstance(stop.depth_image, np.ndarray)
    assert stop.depth_image.shape == (480, 640)
    assert stop.depth_image.dtype == np.float32

    assert hasattr(stop, "foreground_mask") and isinstance(stop.foreground_mask, np.ndarray)
    assert stop.foreground_mask.shape == (480, 640)
    assert stop.foreground_mask.dtype == bool

    assert hasattr(stop, "point_cloud") and isinstance(stop.point_cloud, np.ndarray)
    assert stop.point_cloud.ndim == 2 and stop.point_cloud.shape[1] == 3
    assert stop.point_cloud.dtype == np.float32

    assert hasattr(stop, "point_cloud_world") and isinstance(stop.point_cloud_world, np.ndarray)
    assert stop.point_cloud_world.ndim == 2 and stop.point_cloud_world.shape[1] == 3
    assert stop.point_cloud_world.dtype == np.float32

    assert hasattr(stop, "intrinsics") and isinstance(stop.intrinsics, dict)
    for k in ["fx", "fy", "cx", "cy", "width", "height", "distance_m"]:
        assert k in stop.intrinsics
        assert np.isfinite(stop.intrinsics[k])
    assert stop.intrinsics["distance_m"] == 1.50


# ---------------------------------------------------------------------------
# Test 3: Camera-to-world transformation produces expected coordinates
# ---------------------------------------------------------------------------

def test_camera_to_world_expected_coordinates():
    distance_m = 1.60

    # Test Case 1: Optical axis at depth d -> Origin in world (0, 0, 0)
    pt_cam = np.array([[0.0, 0.0, 1.60]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.0, 0.0, 0.0], atol=1e-5)

    # Test Case 2: Above origin: -Y_cam (top of image) -> +Z_world (upwards stature)
    pt_cam = np.array([[0.0, -0.45, 1.60]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.0, 0.0, 0.45], atol=1e-5)

    # Test Case 3: Below origin: +Y_cam (bottom of image) -> -Z_world (downwards to floor)
    pt_cam = np.array([[0.0, 0.45, 1.60]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.0, 0.0, -0.45], atol=1e-5)

    # Test Case 4: Left of origin: +X_cam (right in image) -> +X_world (anatomical lateral left)
    pt_cam = np.array([[0.35, 0.0, 1.60]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.35, 0.0, 0.0], atol=1e-5)

    # Test Case 5: Right of origin: -X_cam (left in image) -> -X_world (anatomical lateral right)
    pt_cam = np.array([[-0.35, 0.0, 1.60]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [-0.35, 0.0, 0.0], atol=1e-5)

    # Test Case 6: Nearer to camera: Z_c = 1.20 m < 1.60 m -> -Y_world (towards camera)
    pt_cam = np.array([[0.0, 0.0, 1.20]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.0, -0.40, 0.0], atol=1e-5)

    # Test Case 7: Farther from camera: Z_c = 2.10 m > 1.60 m -> +Y_world (away from camera)
    pt_cam = np.array([[0.0, 0.0, 2.10]], dtype=np.float32)
    pt_world = camera_to_world_points(pt_cam, distance_m=distance_m)
    assert np.allclose(pt_world[0], [0.0, 0.50, 0.0], atol=1e-5)

    # Verify rotation matrix properties:
    det_c2w = float(np.linalg.det(R_CAM_TO_WORLD))
    assert np.isclose(det_c2w, 1.0, atol=1e-6), "R_CAM_TO_WORLD must be a proper rotation (det=+1)"

    det_w2c = float(np.linalg.det(R_WORLD_TO_CAM))
    assert np.isclose(det_w2c, 1.0, atol=1e-6), "R_WORLD_TO_CAM must be a proper rotation (det=+1)"

    assert np.allclose(R_CAM_TO_WORLD @ R_WORLD_TO_CAM, np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Round-trip transformation recovers original points
# ---------------------------------------------------------------------------

def test_round_trip_transformation():
    rng = np.random.default_rng(42)
    distance_m = 1.75

    # Random camera-frame points
    pts_cam = rng.uniform(-1.0, 1.0, size=(100, 3)).astype(np.float32)
    pts_cam[:, 2] = rng.uniform(0.5, 3.0, size=100)  # Positive depth

    # Round trip: Camera -> World -> Camera
    pts_world = camera_to_world_points(pts_cam, distance_m=distance_m)
    pts_cam_recovered = world_to_camera_points(pts_world, distance_m=distance_m)
    assert np.allclose(pts_cam_recovered, pts_cam, atol=1e-5)

    # Round trip: World -> Camera -> World
    pts_world_recovered = camera_to_world_points(pts_cam_recovered, distance_m=distance_m)
    assert np.allclose(pts_world_recovered, pts_world, atol=1e-5)

    # Matrix consistency check
    T_c2w = camera_to_world_matrix(distance_m)
    T_w2c = world_to_camera_matrix(distance_m)
    assert np.allclose(T_c2w @ T_w2c, np.eye(4), atol=1e-5)

    # Homogeneous multiplication check
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)])
    pts_world_h = (pts_h @ T_c2w.T)[:, :3]
    assert np.allclose(pts_world_h, pts_world, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: RGB / depth shape mismatch is rejected
# ---------------------------------------------------------------------------

def test_rgb_depth_shape_mismatch_rejected(synthetic_pinhole_intrinsics):
    # Color 480x640, Depth 400x600
    color_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_m = np.zeros((400, 600), dtype=np.float32)

    frame = RGBDFrame(
        frame_id=0,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=synthetic_pinhole_intrinsics,
    )

    with pytest.raises(ValueError, match="does not match depth shape"):
        frame_to_body_stop(frame, nominal_yaw_deg=0.0, distance_m=1.50)


# ---------------------------------------------------------------------------
# Test 6: Empty foreground mask is handled explicitly
# ---------------------------------------------------------------------------

def test_empty_foreground_mask_handled_explicitly(synthetic_rgbd_frame):
    h, w = synthetic_rgbd_frame.depth_m.shape
    empty_mask = np.zeros((h, w), dtype=bool)

    # Default raise_on_empty=True -> ValueError
    with pytest.raises(ValueError, match="Foreground mask is empty"):
        frame_to_body_stop(
            synthetic_rgbd_frame,
            nominal_yaw_deg=0.0,
            distance_m=1.50,
            foreground_mask=empty_mask,
            raise_on_empty=True,
        )

    # Explicit raise_on_empty=False -> Returns empty point cloud without crashing
    stop = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=0.0,
        distance_m=1.50,
        foreground_mask=empty_mask,
        raise_on_empty=False,
    )
    assert len(stop.point_cloud) == 0
    assert len(stop.point_cloud_world) == 0
    assert stop.foreground_mask.shape == (h, w)


# ---------------------------------------------------------------------------
# Test 7: Invalid or missing intrinsics / parameters are rejected
# ---------------------------------------------------------------------------

def test_invalid_or_missing_intrinsics_rejected(synthetic_rgbd_frame):
    # 1. Intrinsics is None
    frame_no_intr = RGBDFrame(
        frame_id=0,
        color_bgr=synthetic_rgbd_frame.color_bgr,
        depth_m=synthetic_rgbd_frame.depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=None,
    )
    with pytest.raises(ValueError, match="Camera intrinsics cannot be None"):
        frame_to_body_stop(frame_no_intr, nominal_yaw_deg=0.0, distance_m=1.50)

    # 2. Negative focal length
    with pytest.raises(ValueError, match="Invalid focal lengths"):
        extract_intrinsics_dict(
            {"fx": -500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
            distance_m=1.50,
        )

    # 3. Invalid distance_m (<= 0.10 m or NaN)
    with pytest.raises(ValueError, match="distance_m"):
        frame_to_body_stop(synthetic_rgbd_frame, nominal_yaw_deg=0.0, distance_m=-1.0)
    with pytest.raises(ValueError, match="distance_m"):
        frame_to_body_stop(synthetic_rgbd_frame, nominal_yaw_deg=0.0, distance_m=float("nan"))

    # 4. Non-finite yaw
    with pytest.raises(ValueError, match="nominal_yaw_deg"):
        frame_to_body_stop(synthetic_rgbd_frame, nominal_yaw_deg=float("inf"), distance_m=1.50)


# ---------------------------------------------------------------------------
# Test 8: Invalid depth values do not produce NaN / Inf coordinates
# ---------------------------------------------------------------------------

def test_invalid_depth_values_no_nan_or_inf(synthetic_pinhole_intrinsics):
    h, w = 480, 640
    color_bgr = np.full((h, w, 3), 100, dtype=np.uint8)
    depth_m = np.zeros((h, w), dtype=np.float32)

    # Inject extreme / corrupt depth values
    depth_m[10:30, 10:30] = float("nan")
    depth_m[30:50, 10:30] = float("inf")
    depth_m[50:70, 10:30] = -5.0
    depth_m[70:90, 10:30] = 0.0
    depth_m[90:110, 10:30] = 100.0  # Beyond max_depth

    # Valid subject region
    depth_m[160:260, 240:340] = 1.45
    mask = np.ones((h, w), dtype=bool)

    frame = RGBDFrame(
        frame_id=0,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=synthetic_pinhole_intrinsics,
        foreground_mask=mask,
    )

    stop = frame_to_body_stop(frame, nominal_yaw_deg=0.0, distance_m=1.50, depth_max_m=3.0)

    assert len(stop.point_cloud) > 0
    assert np.all(np.isfinite(stop.point_cloud))
    assert np.all(np.isfinite(stop.point_cloud_world))
    assert not np.any(np.isnan(stop.point_cloud))
    assert not np.any(np.isinf(stop.point_cloud))


# ---------------------------------------------------------------------------
# Test 9: Indexed point-cloud source-pixel provenance remains available
# ---------------------------------------------------------------------------

def test_indexed_point_cloud_provenance_preserved(synthetic_rgbd_frame):
    stop = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=45.0,
        distance_m=1.60,
        stop_idx=3,
    )

    assert hasattr(stop, "indexed_pcd")
    assert stop.indexed_pcd is not None
    assert len(stop.indexed_pcd) == len(stop.point_cloud)

    # Check provenance fields
    assert hasattr(stop.indexed_pcd, "source_pixel_uv")
    uv = stop.indexed_pcd.source_pixel_uv
    assert uv.shape == (len(stop.point_cloud), 2)

    # All foreground pixels were in [160:320, 240:400]
    u = uv[:, 0]
    v = uv[:, 1]
    assert np.all((u >= 240) & (u < 400))
    assert np.all((v >= 160) & (v < 320))

    assert hasattr(stop.indexed_pcd, "source_frame_id")
    assert np.all(stop.indexed_pcd.source_frame_id == synthetic_rgbd_frame.frame_id)


# ---------------------------------------------------------------------------
# Test 10: Saved replay frame can be adapted without physical camera access
# ---------------------------------------------------------------------------

def test_saved_replay_frame_adapted_without_camera():
    raw_dir = Path(__file__).resolve().parent.parent.parent / "realsense_measure" / "raw_captures" / "body_01"
    if not raw_dir.exists():
        pytest.skip(f"Replay fixture directory {raw_dir} not present.")

    frames = load_raw_capture(raw_dir, target_name="body", allow_aborted=True)
    assert len(frames) > 0, "Failed to load replay frames"

    rep_frame = frames[0]

    # Provide a simple mask if segmentation model is not locally cached
    h, w = rep_frame.depth_m.shape
    valid_depth_mask = (rep_frame.depth_m >= 1.0) & (rep_frame.depth_m <= 2.5) & np.isfinite(rep_frame.depth_m)

    stop = frame_to_body_stop(
        rep_frame,
        nominal_yaw_deg=0.0,
        distance_m=1.60,
        foreground_mask=valid_depth_mask,
        stop_idx=0,
    )

    assert isinstance(stop, SimulatedStopData)
    assert stop.depth_image.shape == (h, w)
    assert stop.intrinsics["width"] == float(w)
    assert stop.intrinsics["height"] == float(h)
    assert len(stop.point_cloud) > 0
    assert len(stop.point_cloud_world) == len(stop.point_cloud)
    assert np.all(np.isfinite(stop.point_cloud_world))


# ---------------------------------------------------------------------------
# Test 11: Adapted stop is directly accepted by DifferentiableBodyFitter
# ---------------------------------------------------------------------------

def test_adapted_stop_compatibility_with_differentiable_fitter(synthetic_rgbd_frame):
    from realsense_measure.body.anny_backend import AnnyBackend
    from realsense_measure.body.fit import DifferentiableBodyFitter
    from realsense_measure.body.prior import AdultShapePrior

    backend = AnnyBackend()
    prior = AdultShapePrior(backend._model)
    fitter = DifferentiableBodyFitter(backend, prior)

    stop = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=0.0,
        distance_m=1.50,
        stop_idx=0,
    )

    # Run quick 2-iteration coarse fit to confirm DifferentiableBodyFitter consumes the adapted stop
    res = fitter.fit(
        [stop],
        n_coarse_iter=2,
        n_fine_iter=1,
        multi_starts=1,
    )

    assert res is not None
    assert "height" in res.fitted_phenotypes
    assert len(res.per_stop_poses) == 1
    assert "yaw_deg" in res.per_stop_poses[0]


# ---------------------------------------------------------------------------
# Test 12: Ground truth semantics unobserved by default
# ---------------------------------------------------------------------------

def test_ground_truth_semantics_unobserved_by_default(synthetic_rgbd_frame):
    """
    Ensure real/replayed captures do not masquerade nominal yaw as measured ground truth.
    When true pose is unobserved, true_* fields must be NaN to prevent contamination.
    """
    stop_real = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=90.0,
        distance_m=1.60,
    )

    # Nominal yaw is recorded for initialization
    assert stop_real.nominal_yaw_deg == 90.0
    # Ground truth must be explicit NaN for real captures
    assert np.isnan(stop_real.true_yaw_deg)
    assert np.all(np.isnan(stop_real.true_translation_m))
    assert np.isnan(stop_real.true_arm_abduction_deg[0])
    assert np.isnan(stop_real.true_arm_abduction_deg[1])

    # In synthetic benchmarking, caller can explicitly provide known ground truth
    stop_synth = frame_to_body_stop(
        synthetic_rgbd_frame,
        nominal_yaw_deg=90.0,
        distance_m=1.60,
        true_yaw_deg=92.5,
        true_translation_m=np.array([0.01, -0.02, 0.005], dtype=np.float32),
        true_arm_abduction_deg=(5.0, -3.0),
    )
    assert stop_synth.nominal_yaw_deg == 90.0
    assert stop_synth.true_yaw_deg == 92.5
    assert np.allclose(stop_synth.true_translation_m, [0.01, -0.02, 0.005])
    assert stop_synth.true_arm_abduction_deg == (5.0, -3.0)


# ---------------------------------------------------------------------------
# Test 13: Single-stop synthetic raycast fit validation
# ---------------------------------------------------------------------------

def test_single_stop_synthetic_raycast_fit_validation():
    """
    Full single-stop integration:
    1. Generates a known synthetic Anny subject.
    2. Renders synthetic RGB-D depth using Open3D RaycastingScene.
    3. Converts observation through frame_to_body_stop().
    4. Runs DifferentiableBodyFitter.fit().
    5. Validates finite loss, valid phenotypes, and estimated pose.
    """
    from realsense_measure.body.anny_backend import AnnyBackend
    from realsense_measure.body.fit import DifferentiableBodyFitter
    from realsense_measure.body.prior import AdultShapePrior
    from realsense_measure.body.synth import CameraConfig, SyntheticTruthSubject, VirtualDepthSensor

    backend = AnnyBackend()
    prior = AdultShapePrior(backend._model)
    subject = SyntheticTruthSubject(backend, prior, subject_id=0, seed=42)

    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    # Render raycast depth for the frontal stop (yaw 0)
    depth_m, mask, _ = sensor.render_depth(subject.true_verts, subject.faces, seed=42)

    color_bgr = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
    color_bgr[mask] = [200, 180, 160]

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
        frame_id=0,
        color_bgr=color_bgr,
        depth_m=depth_m,
        pcd=o3d.geometry.PointCloud(),
        timestamp=0.0,
        intrinsics=intr,
        foreground_mask=mask,
    )

    # Adapt through frame_to_body_stop
    stop = frame_to_body_stop(
        frame,
        nominal_yaw_deg=0.0,
        distance_m=cam_cfg.distance_m,
        stop_idx=0,
    )

    fitter = DifferentiableBodyFitter(backend, prior)
    res = fitter.fit(
        [stop],
        n_coarse_iter=6,
        n_fine_iter=4,
        multi_starts=1,
    )

    assert res is not None
    assert np.isfinite(res.final_loss)
    assert res.final_loss < 10.0

    # Verify all 6 phenotypes are valid in (0, 1)
    for p_name in ["gender", "age", "muscle", "weight", "height", "proportions"]:
        assert p_name in res.fitted_phenotypes
        val = res.fitted_phenotypes[p_name]
        assert 0.0 < val < 1.0

    # Verify estimated pose parameters
    assert len(res.per_stop_poses) == 1
    p0 = res.per_stop_poses[0]
    assert "yaw_deg" in p0 and np.isfinite(p0["yaw_deg"])
    assert "translation_m" in p0 and len(p0["translation_m"]) == 3
    assert "arm_abd_L_deg" in p0 and np.isfinite(p0["arm_abd_L_deg"])
    assert "arm_abd_R_deg" in p0 and np.isfinite(p0["arm_abd_R_deg"])

    # Verify loss breakdown is populated
    bd = res.loss_breakdown.as_dict()
    assert "loss_data" in bd and np.isfinite(bd["loss_data"])
    assert "loss_sil" in bd and np.isfinite(bd["loss_sil"])
    assert "loss_prior" in bd and np.isfinite(bd["loss_prior"])


# ---------------------------------------------------------------------------
# Test 14: Single-stop replay capture fit validation
# ---------------------------------------------------------------------------

def test_single_stop_replay_fitter_integration():
    """
    Validate that a real recorded capture from disk (body_01) can be adapted
    and processed by DifferentiableBodyFitter without physical camera hardware.
    """
    from realsense_measure.body.anny_backend import AnnyBackend
    from realsense_measure.body.fit import DifferentiableBodyFitter
    from realsense_measure.body.prior import AdultShapePrior

    raw_dir = Path(__file__).resolve().parent.parent.parent / "realsense_measure" / "raw_captures" / "body_01"
    if not raw_dir.exists():
        pytest.skip(f"Replay fixture directory {raw_dir} not found.")

    frames = load_raw_capture(raw_dir, target_name="body", allow_aborted=True)
    assert len(frames) > 0

    f0 = frames[0]

    # Subject depth band in body_01 is 1.5 - 2.8 m
    valid_d = (f0.depth_m >= 1.5) & (f0.depth_m <= 2.8) & np.isfinite(f0.depth_m)
    assert np.any(valid_d)
    subj_dist_m = float(np.median(f0.depth_m[valid_d]))  # ~2.33 m

    stop = frame_to_body_stop(
        f0,
        nominal_yaw_deg=0.0,
        distance_m=subj_dist_m,
        foreground_mask=valid_d,
        stop_idx=0,
    )

    backend = AnnyBackend()
    prior = AdultShapePrior(backend._model)
    fitter = DifferentiableBodyFitter(backend, prior)

    res = fitter.fit(
        [stop],
        n_coarse_iter=3,
        n_fine_iter=2,
        multi_starts=1,
    )

    assert res is not None
    assert np.isfinite(res.final_loss)
    assert len(res.fitted_phenotypes) == 6
    assert len(res.per_stop_poses) == 1


