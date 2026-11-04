"""
tests/body/test_constrained_personalization.py
===============================================
Comprehensive numerical validation of anatomically constrained surface personalisation
in realsense_measure/body/personalized_surface.py.

Validates the elimination of cross-region Laplacian deformation leakage:
- Test A: Left mid-thigh deformation (target responds, right thigh/hip/waist/chest <= 1.0 mm).
- Test B: Right upper-arm deformation (target responds, left arm/torso/pelvis <= 1.0 mm).
- Test C: Torso waist deformation (waist responds, chest/hip/limbs <= 1.0 mm).
- Test D: Adjacent anatomical boundary transitions (thigh-pelvis, arm-shoulder, neck-torso).
- Mesh Integrity: Zero NaNs/infs, zero degenerate faces, zero inverted faces, aspect ratios stable.
"""

import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import SyntheticTruthSubject
from realsense_measure.body.personalized_surface import PersonalisedSurfaceFitter
from realsense_measure.body.measure import measure_mesh


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend(verbose=False)


@pytest.fixture(scope="module")
def adult_prior():
    return AdultShapePrior()


@pytest.fixture(scope="module")
def truth_subject(anny_backend, adult_prior):
    return SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)


@pytest.fixture(scope="module")
def fitter(truth_subject, anny_backend):
    return PersonalisedSurfaceFitter(
        canonical_vertices=truth_subject.true_verts,
        faces=truth_subject.faces,
        raw_dict=truth_subject.raw_dict,
        canonical_joints=truth_subject.true_joints,
        joint_names=truth_subject.joint_names,
    )


@pytest.fixture(scope="module")
def baseline_measurements(truth_subject):
    return measure_mesh(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.faces,
        truth_subject.joint_names,
        truth_subject.raw_dict,
    )


def compute_face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return fn / (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-12)


def compute_face_aspect_ratios(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=-1)
    b = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=-1)
    c = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=-1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12))
    inradius = area / (s + 1e-12)
    max_edge = np.maximum(np.maximum(a, b), c)
    return max_edge / (2.0 * np.sqrt(3.0) * inradius + 1e-12)


# ---------------------------------------------------------------------------
# Test A — Left Mid-Thigh Deformation
# ---------------------------------------------------------------------------

def test_a_left_thigh_deformation(fitter, truth_subject, baseline_measurements):
    """
    Test A: Apply a 15 mm inward displacement to the left mid-thigh.
    Verify:
    1. Left mid-thigh circumference responds strongly (|diff| > 15 mm).
    2. Right mid-thigh circumference change <= 1.0 mm.
    3. Hip circumference change <= 1.0 mm.
    4. Waist circumference change <= 1.0 mm.
    5. Chest circumference change <= 1.0 mm.
    6. Max delta_v on right thigh and pelvis is strictly bounded.
    """
    thigh_mask = (fitter.vertex_regions == "thigh_L")
    thigh_z = fitter.v0[thigh_mask, 2]
    z_mid = (np.max(thigh_z) + np.min(thigh_z)) / 2.0
    band_mask = thigh_mask & (np.abs(fitter.v0[:, 2] - z_mid) < 0.03)
    pts = fitter.v0[band_mask].copy()
    normals = fitter.vertex_normals[band_mask]

    displaced_pts = pts - 0.015 * normals
    regs = np.array(["thigh_L"] * len(pts))

    delta_v = fitter._solve_offsets(
        displaced_pts,
        lam=15.0,
        point_normals=normals,
        point_regions=regs,
    )
    v_def = fitter.v0 + delta_v
    def_m = measure_mesh(v_def, truth_subject.true_joints, truth_subject.faces, truth_subject.joint_names, truth_subject.raw_dict)

    diff_thigh_L = abs(def_m["mid_thigh_circumference_left"].value_mm - baseline_measurements["mid_thigh_circumference_left"].value_mm)
    diff_thigh_R = abs(def_m["mid_thigh_circumference_right"].value_mm - baseline_measurements["mid_thigh_circumference_right"].value_mm)
    diff_hip = abs(def_m["hip_circumference"].value_mm - baseline_measurements["hip_circumference"].value_mm)
    diff_waist = abs(def_m["waist_circumference"].value_mm - baseline_measurements["waist_circumference"].value_mm)
    diff_chest = abs(def_m["chest_circumference"].value_mm - baseline_measurements["chest_circumference"].value_mm)

    # Target response
    assert diff_thigh_L > 15.0, f"Left thigh circumference must respond strongly, got {diff_thigh_L:.2f} mm"

    # Cross-region leakage isolation (Quantitative Acceptance: <= 1.0 mm)
    assert diff_thigh_R <= 1.0, f"Right thigh leaked: {diff_thigh_R:.2f} mm > 1.0 mm"
    assert diff_hip <= 1.0, f"Hip leaked: {diff_hip:.2f} mm > 1.0 mm"
    assert diff_waist <= 1.0, f"Waist leaked: {diff_waist:.2f} mm > 1.0 mm"
    assert diff_chest <= 1.0, f"Chest leaked: {diff_chest:.2f} mm > 1.0 mm"

    # Strict vertex displacement checks on unrelated regions
    thigh_R_displ = np.max(np.linalg.norm(delta_v[fitter.vertex_regions == "thigh_R"], axis=1)) * 1000.0
    pelvis_displ = np.max(np.linalg.norm(delta_v[fitter.vertex_regions == "pelvis"], axis=1)) * 1000.0
    assert thigh_R_displ <= 0.1, f"Right thigh vertex displacement {thigh_R_displ:.3f} mm > 0.1 mm"
    assert pelvis_displ <= 0.1, f"Pelvis vertex displacement {pelvis_displ:.3f} mm > 0.1 mm"


# ---------------------------------------------------------------------------
# Test B — Right Arm Deformation
# ---------------------------------------------------------------------------

def test_b_right_arm_deformation(fitter, truth_subject, baseline_measurements):
    """
    Test B: Apply a local 15 mm displacement to the right upper arm.
    Verify:
    1. Right upper-arm circumference responds strongly (|diff| > 15 mm).
    2. Left upper arm remains stable (diff <= 1.0 mm).
    3. Torso circumferences (chest, waist) remain stable (diff <= 1.0 mm).
    4. Pelvis and hip remain stable (diff <= 1.0 mm).
    """
    arm_mask = (fitter.vertex_regions == "upperarm_R")
    arm_z = fitter.v0[arm_mask, 2]
    z_mid_arm = (np.max(arm_z) + np.min(arm_z)) / 2.0
    band_mask = arm_mask & (np.abs(fitter.v0[:, 2] - z_mid_arm) < 0.03)
    pts = fitter.v0[band_mask].copy()
    normals = fitter.vertex_normals[band_mask]

    displaced_pts = pts - 0.015 * normals
    regs = np.array(["upperarm_R"] * len(pts))

    delta_v = fitter._solve_offsets(
        displaced_pts,
        lam=15.0,
        point_normals=normals,
        point_regions=regs,
    )
    v_def = fitter.v0 + delta_v
    def_m = measure_mesh(v_def, truth_subject.true_joints, truth_subject.faces, truth_subject.joint_names, truth_subject.raw_dict)

    diff_arm_R = abs(def_m["upper_arm_circumference_right"].value_mm - baseline_measurements["upper_arm_circumference_right"].value_mm)
    diff_arm_L = abs(def_m["upper_arm_circumference_left"].value_mm - baseline_measurements["upper_arm_circumference_left"].value_mm)
    diff_chest = abs(def_m["chest_circumference"].value_mm - baseline_measurements["chest_circumference"].value_mm)
    diff_waist = abs(def_m["waist_circumference"].value_mm - baseline_measurements["waist_circumference"].value_mm)
    diff_hip = abs(def_m["hip_circumference"].value_mm - baseline_measurements["hip_circumference"].value_mm)

    assert diff_arm_R > 15.0, f"Right upper arm circumference must respond, got {diff_arm_R:.2f} mm"
    assert diff_arm_L <= 1.0, f"Left upper arm leaked: {diff_arm_L:.2f} mm > 1.0 mm"
    assert diff_chest <= 1.0, f"Chest leaked: {diff_chest:.2f} mm > 1.0 mm"
    assert diff_waist <= 1.0, f"Waist leaked: {diff_waist:.2f} mm > 1.0 mm"
    assert diff_hip <= 1.0, f"Hip leaked: {diff_hip:.2f} mm > 1.0 mm"


# ---------------------------------------------------------------------------
# Test C — Torso Waist Deformation
# ---------------------------------------------------------------------------

def test_c_torso_waist_deformation(fitter, truth_subject, baseline_measurements):
    """
    Test C: Apply a local 15 mm displacement to the waist region.
    Verify:
    1. Waist responds strongly (|diff| > 15 mm).
    2. Chest and hip do not experience unintended displacement (diff <= 1.0 mm).
    3. Limb measurements (arms, thighs) remain stable (diff <= 1.0 mm).
    """
    sp04 = truth_subject.true_joints[truth_subject.joint_names.index("spine04")]
    sp02 = truth_subject.true_joints[truth_subject.joint_names.index("spine02")]
    waist_z = (sp04[2] + sp02[2]) / 2.0
    torso_mask = (fitter.vertex_regions == "torso")
    waist_band = torso_mask & (np.abs(fitter.v0[:, 2] - waist_z) < 0.03)
    pts = fitter.v0[waist_band].copy()
    normals = fitter.vertex_normals[waist_band]

    displaced_pts = pts - 0.015 * normals
    regs = np.array(["torso"] * len(pts))

    delta_v = fitter._solve_offsets(
        displaced_pts,
        lam=15.0,
        point_normals=normals,
        point_regions=regs,
    )
    v_def = fitter.v0 + delta_v
    def_m = measure_mesh(v_def, truth_subject.true_joints, truth_subject.faces, truth_subject.joint_names, truth_subject.raw_dict)

    diff_waist = abs(def_m["waist_circumference"].value_mm - baseline_measurements["waist_circumference"].value_mm)
    diff_chest = abs(def_m["chest_circumference"].value_mm - baseline_measurements["chest_circumference"].value_mm)
    diff_hip = abs(def_m["hip_circumference"].value_mm - baseline_measurements["hip_circumference"].value_mm)
    diff_thigh_L = abs(def_m["mid_thigh_circumference_left"].value_mm - baseline_measurements["mid_thigh_circumference_left"].value_mm)
    diff_arm_R = abs(def_m["upper_arm_circumference_right"].value_mm - baseline_measurements["upper_arm_circumference_right"].value_mm)

    assert diff_waist > 15.0, f"Waist circumference must respond, got {diff_waist:.2f} mm"
    assert diff_chest <= 1.0, f"Chest leaked: {diff_chest:.2f} mm > 1.0 mm"
    assert diff_hip <= 1.0, f"Hip leaked: {diff_hip:.2f} mm > 1.0 mm"
    assert diff_thigh_L <= 1.0, f"Thigh leaked: {diff_thigh_L:.2f} mm > 1.0 mm"
    assert diff_arm_R <= 1.0, f"Arm leaked: {diff_arm_R:.2f} mm > 1.0 mm"


# ---------------------------------------------------------------------------
# Test D — Adjacent Anatomical Transitions & Mesh Integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reg_a, reg_b", [
    ("thigh_L", "pelvis"),
    ("upperarm_R", "torso"),
    ("head_neck", "torso"),
])
def test_d_adjacent_anatomical_transitions(fitter, reg_a, reg_b):
    """
    Test D: Apply deformation immediately adjacent to anatomical region boundaries:
    - Thigh-pelvis boundary
    - Upper-arm-shoulder boundary
    - Torso-neck boundary

    Verify:
    1. No NaNs or infinities.
    2. No degenerate faces (area > 1e-8 m^2).
    3. No inverted faces (cos(theta) between deformed and original face normal > 0).
    4. Stable triangle aspect ratios (max aspect ratio does not blow up).
    5. Clean continuous transition without tearing or staircasing.
    """
    boundary_faces = [
        f for f in fitter.faces
        if reg_a in [fitter.vertex_regions[v] for v in f] and reg_b in [fitter.vertex_regions[v] for v in f]
    ]
    assert len(boundary_faces) > 0, f"Expected boundary faces between {reg_a} and {reg_b}"
    boundary_faces = np.array(boundary_faces)
    boundary_verts = np.unique(boundary_faces)
    a_verts = [v for v in boundary_verts if fitter.vertex_regions[v] == reg_a]
    assert len(a_verts) > 0

    pts = fitter.v0[a_verts].copy()
    normals = fitter.vertex_normals[a_verts]
    displaced_pts = pts - 0.015 * normals

    delta_v = fitter._solve_offsets(
        displaced_pts,
        lam=15.0,
        point_normals=normals,
        point_regions=[reg_a] * len(pts),
    )
    v_def = fitter.v0 + delta_v

    # 1. No NaN or inf
    assert not np.any(np.isnan(v_def)), "Deformed vertices contain NaN"
    assert not np.any(np.isinf(v_def)), "Deformed vertices contain Inf"

    # 2. Triangle areas & degeneracy
    v0_areas = 0.5 * np.linalg.norm(
        np.cross(fitter.v0[fitter.faces[:, 1]] - fitter.v0[fitter.faces[:, 0]], fitter.v0[fitter.faces[:, 2]] - fitter.v0[fitter.faces[:, 0]]),
        axis=-1,
    )
    vdef_areas = 0.5 * np.linalg.norm(
        np.cross(v_def[fitter.faces[:, 1]] - v_def[fitter.faces[:, 0]], v_def[fitter.faces[:, 2]] - v_def[fitter.faces[:, 0]]),
        axis=-1,
    )
    assert np.all(vdef_areas > 1e-8), "Degenerate faces detected after deformation"

    # 3. Inverted faces check
    orig_fn = compute_face_normals(fitter.v0, fitter.faces)
    new_fn = compute_face_normals(v_def, fitter.faces)
    cos_alignment = np.sum(orig_fn * new_fn, axis=-1)
    assert np.all(cos_alignment > 0.0), f"Inverted faces detected at {reg_a}-{reg_b} transition"
    assert np.min(cos_alignment[boundary_faces]) > 0.99, "Boundary faces suffered severe normal twisting"

    # 4. Aspect ratio preservation
    orig_ar = compute_face_aspect_ratios(fitter.v0, fitter.faces)
    new_ar = compute_face_aspect_ratios(v_def, fitter.faces)
    max_ar_orig = np.max(orig_ar[boundary_faces])
    max_ar_new = np.max(new_ar[boundary_faces])
    assert max_ar_new < max_ar_orig * 1.10, (
        f"Aspect ratio blew up across {reg_a}-{reg_b} boundary: orig={max_ar_orig:.2f}, new={max_ar_new:.2f}"
    )

