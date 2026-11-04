"""
tests/body/test_appearance.py
=============================
Validation and test suite for Prompt 3K: Appearance Modeling and Visual Identity Layer.

Verifies:
1. Synthetic RGB frame projection and color sampling with known solid colors.
2. Backface culling (>80 deg normal-to-ray rejected as backface).
3. Out-of-frame rejection for vertices outside image bounds.
4. Z-buffer depth occlusion testing (foreground obstacles reject vertices as occluded).
5. Multi-stop cosine-weighted color aggregation.
6. Unobserved vertices marked explicitly (is_observed_mask=False, neutral grey 128,128,128).
7. Appearance quality diagnostics completeness (coverage, multi-view counts, rejection reasons).
8. Strict geometry invariance (vertices, faces, joints, phenotypes identical with/without appearance).
9. Strict anthropometry invariance (measurements identical with/without appearance).
10. Export to PLY with vertex colors and Open3D reload integrity.
11. Export to OBJ with vertex colors (v x y z r g b) and without vertex colors (v x y z).
12. Deterministic repeatability across identical runs.
13. Validation on real offline replay capture (body_01).
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from camera.frame import RGBDFrame
from camera.recording import load_raw_capture
from realsense_measure.body.adapter import AdaptedStopData, frame_to_body_stop
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.appearance import (
    AppearanceModeler,
    AppearanceQualityDiagnostics,
    VertexAppearanceResult,
    compute_mesh_vertex_normals,
    export_vertex_colored_obj,
    export_vertex_colored_ply,
    extract_vertex_appearance,
    project_canonical_to_camera,
)
from realsense_measure.body.fit import DifferentiableBodyFitter
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.reconstruction import (
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


def render_stop_frame_with_color(
    subject: SyntheticTruthSubject,
    sensor: VirtualDepthSensor,
    cam_cfg: CameraConfig,
    yaw_deg: float,
    stop_idx: int,
    rgb_color: np.ndarray,
) -> RGBDFrame:
    """Render a synthetic stop and assign a known uniform or patterned RGB image (BGR in memory)."""
    verts = subject.true_verts.copy()
    rad = np.radians(yaw_deg)
    cos_y = np.cos(rad)
    sin_y = np.sin(rad)
    rx = verts[:, 0] * cos_y - verts[:, 1] * sin_y
    ry = verts[:, 0] * sin_y + verts[:, 1] * cos_y
    rz = verts[:, 2]
    posed_verts = np.stack([rx, ry, rz], axis=-1)

    depth_m, mask, _ = sensor.render_depth(posed_verts, subject.faces, seed=42 + stop_idx)

    if rgb_color.ndim == 1:
        bgr = rgb_color[[2, 1, 0]]
        color_bgr = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
        color_bgr[mask] = bgr
    else:
        color_bgr = rgb_color[:, :, [2, 1, 0]].astype(np.uint8)

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
# Test 1: Synthetic RGB Projection & Sampling with Known Solid Color
# ---------------------------------------------------------------------------

def test_synthetic_rgb_projection_and_sampling(shared_backend, synthetic_subject):
    """
    Verify that visible front-facing mesh vertices project to the correct camera pixels
    and sample the synthetic ground-truth color (e.g. pure Red [255, 0, 0]).
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    pure_red = np.array([255, 0, 0], dtype=np.uint8)
    frame0 = render_stop_frame_with_color(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0, rgb_color=pure_red)

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    modeler = AppearanceModeler(min_cos_angle=0.1736, depth_margin_m=0.025)
    app = modeler.extract_appearance([frame0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    assert app.diagnostics.total_vertices == len(verts)
    assert app.diagnostics.observed_vertex_count > 0
    assert app.diagnostics.unobserved_vertex_count > 0
    assert 0.20 < app.diagnostics.coverage_fraction < 0.65

    obs_indices = np.where(app.is_observed_mask)[0]
    obs_colors = app.vertex_colors_uint8[obs_indices]

    # All observed vertices should be pure red [255, 0, 0]
    np.testing.assert_allclose(obs_colors[:, 0], 255)
    np.testing.assert_allclose(obs_colors[:, 1], 0)
    np.testing.assert_allclose(obs_colors[:, 2], 0)


# ---------------------------------------------------------------------------
# Test 2: Backface Culling
# ---------------------------------------------------------------------------

def test_backface_culling(shared_backend, synthetic_subject):
    """
    Verify backface culling: vertices with normals facing away from camera
    (cos_angle < threshold) must be culled and logged as 'backface'.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    frame0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([100, 150, 200], dtype=np.uint8),
    )

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    modeler = AppearanceModeler()
    app = modeler.extract_appearance([frame0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    reasons = app.diagnostics.per_stop_rejection_reasons[0]
    assert "backface" in reasons
    assert reasons["backface"] > 1000  # Back side of body must be rejected as backface


# ---------------------------------------------------------------------------
# Test 3: Out of Frame Rejection
# ---------------------------------------------------------------------------

def test_out_of_frame_rejection():
    """Verify that vertices projecting outside camera viewport are rejected with reason 'out_of_frame'."""
    verts = np.array([
        [10.0, 0.0, 0.0],   # far right (x=10m)
        [-10.0, 0.0, 0.0],  # far left (x=-10m)
        [0.0, 0.0, 10.0],   # far top (z_canon=10m)
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.array([[0, -1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float32)

    pose = {"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}
    intrinsics = {
        "width": 640,
        "height": 480,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    proj = project_canonical_to_camera(verts, normals, pose, distance_m=2.0, intrinsics=intrinsics)

    assert not np.any(proj["is_in_frame"])
    assert not np.any(proj["is_visible"])


# ---------------------------------------------------------------------------
# Test 4: Z-Buffer Occlusion Testing
# ---------------------------------------------------------------------------

def test_depth_occlusion_rejection(shared_backend, synthetic_subject):
    """
    Verify Z-buffer depth occlusion testing:
    If depth buffer contains a foreground object closer to camera than the body mesh,
    the occluded vertices must be rejected with reason 'occluded'.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    frame0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([200, 200, 200], dtype=np.uint8),
    )

    # Insert an artificial foreground occluder at depth 0.8m over center of frame
    h, w = frame0.depth_m.shape
    frame0.depth_m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.8

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    modeler = AppearanceModeler(depth_margin_m=0.025)
    app = modeler.extract_appearance([frame0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    reasons = app.diagnostics.per_stop_rejection_reasons[0]
    assert "occluded" in reasons
    assert reasons["occluded"] > 100


# ---------------------------------------------------------------------------
# Test 5: Multi-Stop Cosine-Weighted Aggregation
# ---------------------------------------------------------------------------

def test_multi_stop_cosine_weighted_aggregation(shared_backend, synthetic_subject):
    """
    Verify multi-stop aggregation:
    Stop 0 (yaw 0) has pure Red [255, 0, 0].
    Stop 1 (yaw 45) has pure Green [0, 255, 0].
    Overlapping vertices must have blended colors weighted by view angle cosine.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)

    red = np.array([255, 0, 0], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)

    frame0 = render_stop_frame_with_color(synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0, rgb_color=red)
    frame1 = render_stop_frame_with_color(synthetic_subject, sensor, cam_cfg, yaw_deg=45.0, stop_idx=1, rgb_color=green)

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [
        {"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0},
        {"yaw_deg": 45.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0},
    ]

    modeler = AppearanceModeler()
    app = modeler.extract_appearance([frame0, frame1], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    assert app.diagnostics.multi_view_observed_count > 0
    multi_view_mask = (app.observation_count_per_vertex >= 2)
    multi_colors = app.vertex_colors_uint8[multi_view_mask]

    # Multi-view vertices must contain BOTH red and green components, and zero blue
    assert np.all(multi_colors[:, 0] > 0)
    assert np.all(multi_colors[:, 1] > 0)
    np.testing.assert_allclose(multi_colors[:, 2], 0)

    # Multi-view color standard deviation should be tracked
    assert app.diagnostics.mean_color_std_rgb is not None
    assert app.diagnostics.mean_color_std_rgb[0] > 0.0


# ---------------------------------------------------------------------------
# Test 6: Unobserved Vertices Marked Explicitly
# ---------------------------------------------------------------------------

def test_unobserved_neutral_marking(shared_backend, synthetic_subject):
    """
    Verify that unobserved vertices are explicitly marked (is_observed_mask=False)
    and given the fallback neutral grey color [128, 128, 128].
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    frame0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([255, 255, 0], dtype=np.uint8),
    )

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    app = extract_vertex_appearance([frame0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    unobs_mask = ~app.is_observed_mask
    assert np.sum(unobs_mask) > 0
    unobs_colors = app.vertex_colors_uint8[unobs_mask]

    np.testing.assert_allclose(unobs_colors[:, 0], 128)
    np.testing.assert_allclose(unobs_colors[:, 1], 128)
    np.testing.assert_allclose(unobs_colors[:, 2], 128)


# ---------------------------------------------------------------------------
# Test 7: Appearance Quality Diagnostics
# ---------------------------------------------------------------------------

def test_appearance_quality_diagnostics(shared_backend, synthetic_subject):
    """Verify completeness and sanity of all diagnostics fields in AppearanceQualityDiagnostics."""
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    frame0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([50, 100, 150], dtype=np.uint8),
    )

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    app = extract_vertex_appearance([frame0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)
    d = app.diagnostics

    assert d.total_vertices == len(verts)
    assert d.observed_vertex_count + d.unobserved_vertex_count == d.total_vertices
    assert np.isclose(d.coverage_fraction, d.observed_vertex_count / d.total_vertices)
    assert np.isclose(d.coverage_percentage, d.coverage_fraction * 100.0)
    assert d.single_view_observed_count == d.observed_vertex_count
    assert d.multi_view_observed_count == 0  # Single view input
    assert len(d.per_stop_sampled_vertices) == 1
    assert len(d.per_stop_rejected_vertices) == 1
    assert len(d.per_stop_rejection_reasons) == 1

    d_dict = d.as_dict()
    assert isinstance(d_dict, dict)
    assert "coverage_percentage" in d_dict
    assert "rejection_reasons" in d_dict["per_stop"][0]


# ---------------------------------------------------------------------------
# Test 8: Strict Geometry Invariance
# ---------------------------------------------------------------------------

def test_strict_geometry_invariance(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    CRITICAL: Verify that enabling appearance modeling does NOT alter mesh geometry:
    vertices, faces, joints, and fitted phenotypes must be 100% bitwise identical.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([255, 128, 64], dtype=np.uint8),
    )

    # Run without appearance
    res_no_app = reconstruct_and_personalize(
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
        extract_appearance=False,
    )

    # Run with appearance
    res_with_app = reconstruct_and_personalize(
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

    assert res_no_app.appearance is None
    assert res_with_app.appearance is not None

    # Geometry invariance checks
    np.testing.assert_array_equal(
        res_no_app.personalized_mesh["vertices"],
        res_with_app.personalized_mesh["vertices"],
        err_msg="Personalized vertices modified by appearance extraction!",
    )
    np.testing.assert_array_equal(
        res_no_app.canonical_fitted_mesh["vertices"],
        res_with_app.canonical_fitted_mesh["vertices"],
        err_msg="Canonical fitted vertices modified by appearance extraction!",
    )
    np.testing.assert_array_equal(
        res_no_app.personalized_mesh["faces"],
        res_with_app.personalized_mesh["faces"],
    )
    np.testing.assert_array_equal(
        res_no_app.personalized_mesh["joints"],
        res_with_app.personalized_mesh["joints"],
    )

    # Phenotype invariance
    for k in res_no_app.multi_stop_fit.fitted_phenotypes:
        assert res_no_app.multi_stop_fit.fitted_phenotypes[k] == res_with_app.multi_stop_fit.fitted_phenotypes[k]


# ---------------------------------------------------------------------------
# Test 9: Strict Anthropometry Invariance
# ---------------------------------------------------------------------------

def test_strict_anthropometry_invariance(shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    CRITICAL: Verify that anthropometric measurement outputs are strictly invariant
    whether appearance extraction is enabled or disabled.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([200, 100, 50], dtype=np.uint8),
    )

    res_no_app = reconstruct_and_personalize(
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
        extract_appearance=False,
    )

    res_with_app = reconstruct_and_personalize(
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

    for m_name in res_no_app.model_measurements:
        v_no = res_no_app.model_measurements[m_name].value_mm
        v_with = res_with_app.model_measurements[m_name].value_mm
        if np.isnan(v_no):
            assert np.isnan(v_with)
        else:
            assert v_no == v_with

    for m_name in res_no_app.personalized_measurements:
        v_no = res_no_app.personalized_measurements[m_name].value_mm
        v_with = res_with_app.personalized_measurements[m_name].value_mm
        if np.isnan(v_no):
            assert np.isnan(v_with)
        else:
            assert v_no == v_with


# ---------------------------------------------------------------------------
# Test 10: Export PLY with Vertex Colors and Reload Integrity
# ---------------------------------------------------------------------------

def test_export_ply_with_colors_and_reload(tmp_path, shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify export to Stanford PLY with vertex colors and verify reload via Open3D:
    - has_vertex_colors() == True
    - vertex and face counts match Anny mesh
    - color coordinates are valid.
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([0, 200, 100], dtype=np.uint8),
    )

    res = reconstruct_and_personalize(
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

    ply_path = tmp_path / "body_colored.ply"
    res.export_ply(ply_path, mesh_type="personalized")

    assert ply_path.exists()
    assert ply_path.stat().st_size > 100000

    mesh_o3d = o3d.io.read_triangle_mesh(str(ply_path))
    assert len(mesh_o3d.vertices) == 13718
    assert len(mesh_o3d.triangles) == 27420
    assert mesh_o3d.has_vertex_colors()

    colors = np.asarray(mesh_o3d.vertex_colors)
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0)


# ---------------------------------------------------------------------------
# Test 11: Export OBJ with and without Vertex Colors
# ---------------------------------------------------------------------------

def test_export_obj_with_and_without_colors(tmp_path, shared_backend, shared_prior, shared_fitter, synthetic_subject):
    """
    Verify export to OBJ:
    - With include_vertex_colors=True: produces lines `v x y z r g b` (6 floats)
    - With include_vertex_colors=False: produces standard lines `v x y z` (3 floats)
    """
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([255, 0, 128], dtype=np.uint8),
    )

    res = reconstruct_and_personalize(
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

    # 1. Standard OBJ
    obj_plain = tmp_path / "plain.obj"
    res.export_obj(obj_plain, include_vertex_colors=False)
    with open(obj_plain, "r") as f:
        v_lines = [line.strip().split() for line in f if line.startswith("v ")]
    assert len(v_lines) == 13718
    assert len(v_lines[0]) == 4  # ['v', 'x', 'y', 'z']

    # 2. Vertex-colored OBJ
    obj_colored = tmp_path / "colored.obj"
    res.export_obj(obj_colored, include_vertex_colors=True)
    with open(obj_colored, "r") as f:
        v_col_lines = [line.strip().split() for line in f if line.startswith("v ")]
    assert len(v_col_lines) == 13718
    assert len(v_col_lines[0]) == 7  # ['v', 'x', 'y', 'z', 'r', 'g', 'b']

    o3d_obj = o3d.io.read_triangle_mesh(str(obj_colored))
    assert len(o3d_obj.vertices) == 13718
    assert o3d_obj.has_vertex_colors()


# ---------------------------------------------------------------------------
# Test 12: Deterministic Repeatability
# ---------------------------------------------------------------------------

def test_deterministic_repeatability(shared_backend, synthetic_subject):
    """Verify that appearance modeling is strictly deterministic across repeated invocations."""
    cam_cfg = CameraConfig.portrait(pessimistic=False)
    sensor = VirtualDepthSensor(cam_cfg)
    f0 = render_stop_frame_with_color(
        synthetic_subject, sensor, cam_cfg, yaw_deg=0.0, stop_idx=0,
        rgb_color=np.array([123, 45, 67], dtype=np.uint8),
    )

    verts = synthetic_subject.true_verts
    faces = synthetic_subject.faces

    poses = [{"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0], "arm_abd_L_deg": 0.0, "arm_abd_R_deg": 0.0}]
    app1 = extract_vertex_appearance([f0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)
    app2 = extract_vertex_appearance([f0], verts, faces, per_stop_poses=poses, distance_m=cam_cfg.distance_m)

    np.testing.assert_array_equal(app1.vertex_colors_uint8, app2.vertex_colors_uint8)
    np.testing.assert_array_equal(app1.is_observed_mask, app2.is_observed_mask)
    np.testing.assert_array_equal(app1.observation_count_per_vertex, app2.observation_count_per_vertex)
    assert app1.diagnostics.as_dict() == app2.diagnostics.as_dict()


# ---------------------------------------------------------------------------
# Test 13: Offline Replay Validation on Real Recorded Capture (body_01)
# ---------------------------------------------------------------------------

def test_offline_replay_validation_body_01(shared_backend, shared_prior, shared_fitter):
    """
    Verify appearance modeling against real recorded RGB-D camera frames from body_01.
    """
    raw_dir = Path(__file__).resolve().parent.parent.parent / "realsense_measure" / "raw_captures" / "body_01"
    if not raw_dir.exists():
        pytest.skip(f"Replay directory {raw_dir} not available.")

    frames = load_raw_capture(raw_dir, target_name="body", allow_aborted=True)
    assert len(frames) >= 2

    f0 = frames[0]
    f2 = frames[2]

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
        extract_appearance=True,
    )

    assert result.appearance is not None
    assert result.appearance.diagnostics.observed_vertex_count > 0
    assert result.appearance.colors_uint8.shape == (13718, 3)
    assert np.all(np.isfinite(result.appearance.colors_rgb))

    d = result.as_dict()
    assert "appearance" in d
    assert d["appearance"]["observed_vertices"] > 0
