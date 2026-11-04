"""
tests/body/test_measurement_engine_completeness.py
===================================================
Comprehensive tests for anthropometric measurement engine completeness and
open-contour reliability in realsense_measure/body/measure.py:

1. Shoulder breadth integration:
   - Present in output schema
   - Accurate on symmetric synthetic body
   - Responds linearly to synthetic shoulder lateral displacement
   - Safe fallback to NaN on missing shoulder landmarks/joints

2. Bilateral limb-length measurements:
   - shoulder_to_elbow_length_left and shoulder_to_elbow_length_right
   - shoulder_to_wrist_length_left and shoulder_to_wrist_length_right
   - Legacy aliases retained (shoulder_elbow_length, shoulder_wrist_length)
   - Independent verification on asymmetric limb geometry
   - Independent fault tolerance (missing right landmarks do not invalidate left)

3. Inseam measurement:
   - Midline crotch landmark to distal leg path verified and documented

4. Contour-closure validation (all 8 Prompt 3E cases):
   - Fully closed circle
   - Fully closed ellipse
   - Open arc
   - Contour with small gap
   - Contour with large gap
   - Empty contour
   - Duplicate/invalid points
   - Hairpin/omega contour with close endpoints
   - Rejection of silent chord-bridging
"""

import math
import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import SyntheticTruthSubject
from realsense_measure.body.measure import (
    Measurement,
    measure_body,
    measure_mesh,
    is_contour_closed,
    _loop_perimeter,
    LandmarkHeights,
)


@pytest.fixture(scope="module")
def anny_backend():
    return AnnyBackend(verbose=False)


@pytest.fixture(scope="module")
def adult_prior():
    return AdultShapePrior()


@pytest.fixture(scope="module")
def truth_subject(anny_backend, adult_prior):
    return SyntheticTruthSubject(anny_backend, adult_prior, subject_id=0, seed=1000)


# ===========================================================================
# 1. Shoulder Breadth Integration Tests
# ===========================================================================

def test_shoulder_breadth_present_in_output(truth_subject):
    """Verify shoulder_breadth is included in measure_body output schema and is valid."""
    meas = measure_mesh(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.faces,
        truth_subject.joint_names,
        truth_subject.raw_dict,
    )
    assert "shoulder_breadth" in meas, "shoulder_breadth must be in output"
    sb = meas["shoulder_breadth"]
    assert sb.is_valid(), f"shoulder_breadth should be valid, got {sb.value_mm}"
    assert 300.0 <= sb.value_mm <= 500.0, f"Plausible shoulder breadth expected, got {sb.value_mm:.1f} mm"
    assert sb.method == "3d_biacromial_distance"


def test_shoulder_breadth_symmetric_synthetic_body(anny_backend):
    """Verify that a symmetric rest pose produces symmetric acromion landmarks."""
    body = anny_backend.rest_pose()
    verts = body.vertices.cpu().numpy() if hasattr(body.vertices, "cpu") else np.asarray(body.vertices)
    joints = body.joints.cpu().numpy() if hasattr(body.joints, "cpu") else np.asarray(body.joints)
    faces = body.faces.cpu().numpy() if hasattr(body.faces, "cpu") else np.asarray(body.faces)
    jnames = list(body.raw["bone_labels"])

    lm = LandmarkHeights(joints, jnames, verts)
    ac_l = lm.acromion_landmark("L")
    ac_r = lm.acromion_landmark("R")

    assert ac_l is not None and ac_r is not None
    # In rest pose, lateral X should be symmetric within 2 mm
    assert abs(ac_l[0] - abs(ac_r[0])) < 0.005, f"Acromion lateral X asymmetry: {ac_l[0]} vs {ac_r[0]}"
    # Height Z should be identical within 2 mm
    assert abs(ac_l[2] - ac_r[2]) < 0.005, f"Acromion height Z asymmetry: {ac_l[2]} vs {ac_r[2]}"


def test_shoulder_breadth_sensitivity_to_shoulder_displacement(truth_subject):
    """Verify that a known lateral displacement of shoulder joints/mesh produces expected breadth change."""
    verts_mod = truth_subject.true_verts.copy()
    joints_mod = truth_subject.true_joints.copy()
    jnames = truth_subject.joint_names

    base_meas = measure_mesh(verts_mod, joints_mod, truth_subject.faces, jnames, truth_subject.raw_dict)
    base_sb = base_meas["shoulder_breadth"].value_mm

    # Shift left shoulder outward by +20 mm and right shoulder outward by -20 mm
    shift = 0.020  # metres
    up_l_idx = jnames.index("upperarm01.L")
    up_r_idx = jnames.index("upperarm01.R")
    joints_mod[up_l_idx, 0] += shift
    joints_mod[up_r_idx, 0] -= shift

    # Shift nearby mesh vertices
    dist_l = np.linalg.norm(verts_mod - truth_subject.true_joints[up_l_idx], axis=1)
    dist_r = np.linalg.norm(verts_mod - truth_subject.true_joints[up_r_idx], axis=1)
    verts_mod[dist_l < 0.080, 0] += shift
    verts_mod[dist_r < 0.080, 0] -= shift

    mod_meas = measure_mesh(verts_mod, joints_mod, truth_subject.faces, jnames, truth_subject.raw_dict)
    mod_sb = mod_meas["shoulder_breadth"].value_mm

    expected_change = 2.0 * shift * 1000.0  # +40 mm
    actual_change = mod_sb - base_sb
    assert abs(actual_change - expected_change) < 3.0, (
        f"Expected shoulder breadth change ~{expected_change:.1f} mm, got {actual_change:.1f} mm"
    )


def test_shoulder_breadth_missing_geometry_fallback():
    """Verify that missing shoulder joints or geometry cleanly returns NaN without crashing."""
    # Mesh with only lower body vertices (no shoulders) and empty joints
    lower_verts = np.array([
        [0.0, 0.0, -0.5],
        [0.1, 0.0, -0.5],
        [0.0, 0.1, -0.5],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    meas = measure_mesh(lower_verts, np.zeros((0, 3)), faces, [], {})
    assert "shoulder_breadth" in meas
    sb = meas["shoulder_breadth"]
    assert not sb.is_valid(), "shoulder_breadth must be invalid when shoulder geometry is missing"
    assert math.isnan(sb.value_mm)


# ===========================================================================
# 2. Bilateral Limb-Length Measurements Tests
# ===========================================================================

def test_bilateral_limb_lengths_present_in_output(truth_subject):
    """Verify bilateral keys and legacy aliases are all present in the output schema."""
    meas = measure_mesh(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.faces,
        truth_subject.joint_names,
        truth_subject.raw_dict,
    )

    required_keys = [
        "shoulder_to_elbow_length_left",
        "shoulder_to_elbow_length_right",
        "shoulder_elbow_length",  # legacy alias (left)
        "shoulder_to_wrist_length_left",
        "shoulder_to_wrist_length_right",
        "shoulder_wrist_length",  # legacy alias (left)
    ]
    for k in required_keys:
        assert k in meas, f"Missing key {k} in measure_mesh output"
        assert meas[k].is_valid(), f"{k} should be valid"

    # Verify legacy alias equality
    assert meas["shoulder_elbow_length"].value_mm == meas["shoulder_to_elbow_length_left"].value_mm
    assert meas["shoulder_wrist_length"].value_mm == meas["shoulder_to_wrist_length_left"].value_mm


def test_bilateral_limb_lengths_asymmetric_synthetic_mesh(truth_subject):
    """Verify independent left and right limb length measurement on an intentionally asymmetric body."""
    verts = truth_subject.true_verts.copy()
    joints = truth_subject.true_joints.copy()
    jnames = truth_subject.joint_names

    base_meas = measure_mesh(verts, truth_subject.true_joints, truth_subject.faces, jnames, truth_subject.raw_dict)
    base_left_sw = base_meas["shoulder_to_wrist_length_left"].value_mm
    base_right_sw = base_meas["shoulder_to_wrist_length_right"].value_mm

    # Lengthen right forearm by moving right wrist 50 mm downward along arm
    wr_r_idx = jnames.index("wrist.R")
    el_r_idx = jnames.index("lowerarm01.R")
    arm_dir = joints[wr_r_idx] - joints[el_r_idx]
    arm_dir /= np.linalg.norm(arm_dir)

    extension = 0.050  # 50 mm
    joints[wr_r_idx] += arm_dir * extension

    mod_meas = measure_mesh(verts, joints, truth_subject.faces, jnames, truth_subject.raw_dict)

    mod_left_sw = mod_meas["shoulder_to_wrist_length_left"].value_mm
    mod_right_sw = mod_meas["shoulder_to_wrist_length_right"].value_mm

    # Left arm should be unchanged
    assert abs(mod_left_sw - base_left_sw) < 0.1, "Left arm length must not change when right arm is extended"
    # Right arm should be extended by exactly 50 mm
    right_extension = mod_right_sw - base_right_sw
    assert abs(right_extension - 50.0) < 1.0, (
        f"Right arm should be extended by 50 mm, got {right_extension:.2f} mm"
    )


def test_bilateral_limb_lengths_missing_landmarks_independent(truth_subject):
    """Verify that missing right-side joints do not invalidate valid left-side measurements."""
    joints_partial = truth_subject.true_joints.copy()
    jnames = list(truth_subject.joint_names)

    # Invalidate right elbow and wrist by removing them from joint_names
    jnames[jnames.index("lowerarm01.R")] = "deleted_elbow_R"
    jnames[jnames.index("wrist.R")] = "deleted_wrist_R"

    meas = measure_mesh(truth_subject.true_verts, joints_partial, truth_subject.faces, jnames, truth_subject.raw_dict)

    # Left side must remain valid
    assert meas["shoulder_to_elbow_length_left"].is_valid()
    assert meas["shoulder_to_wrist_length_left"].is_valid()
    assert meas["shoulder_elbow_length"].is_valid()
    assert meas["shoulder_wrist_length"].is_valid()

    # Right side must be invalid (NaN) without crashing
    assert not meas["shoulder_to_elbow_length_right"].is_valid()
    assert not meas["shoulder_to_wrist_length_right"].is_valid()
    assert math.isnan(meas["shoulder_to_elbow_length_right"].value_mm)
    assert math.isnan(meas["shoulder_to_wrist_length_right"].value_mm)


# ===========================================================================
# 3. Inseam Measurement Definition Tests
# ===========================================================================

def test_inseam_definition_and_validity(truth_subject):
    """Verify that inseam is present as a single midline-to-ankle measurement and is valid."""
    meas = measure_mesh(
        truth_subject.true_verts,
        truth_subject.true_joints,
        truth_subject.faces,
        truth_subject.joint_names,
        truth_subject.raw_dict,
    )
    assert "inseam" in meas
    ins = meas["inseam"]
    assert ins.is_valid()
    assert 600.0 <= ins.value_mm <= 950.0, f"Plausible adult inseam expected, got {ins.value_mm:.1f} mm"
    assert "crotch" in ins.notes.lower()
    assert ins.method == "3d_crotch_knee_ankle_chain"


# ===========================================================================
# 4. Contour-Closure Validation Tests (All 8 Prompt 3E Scenarios)
# ===========================================================================

def test_contour_closure_case1_closed_circle():
    """Case 1: A fully closed circle must be valid and report accurate circumference."""
    r = 0.10  # 100 mm radius -> perimeter = 2 * pi * 0.10 = 0.628318 m
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([r * np.cos(t), r * np.sin(t), np.zeros(64)], axis=1)

    closed, reason = is_contour_closed(circle)
    assert closed, f"Fully closed circle rejected: {reason}"

    perim_m = _loop_perimeter(circle, require_closed=True)
    assert not math.isnan(perim_m)
    expected = 2 * math.pi * r
    rel_err = abs(perim_m - expected) / expected
    assert rel_err < 0.005, f"Circle perimeter error {rel_err*100:.3f}%"


def test_contour_closure_case2_closed_ellipse():
    """Case 2: A fully closed ellipse must be valid and report accurate circumference."""
    a, b = 0.15, 0.08  # semi-major, semi-minor (metres)
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    ellipse = np.stack([a * np.cos(t), b * np.sin(t), np.zeros(128)], axis=1)

    closed, reason = is_contour_closed(ellipse)
    assert closed, f"Fully closed ellipse rejected: {reason}"

    perim_m = _loop_perimeter(ellipse, require_closed=True)
    assert not math.isnan(perim_m)
    # Ramanujan's formula
    h = ((a - b) / (a + b)) ** 2
    expected = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    rel_err = abs(perim_m - expected) / expected
    assert rel_err < 0.005, f"Ellipse perimeter error {rel_err*100:.3f}%"


def test_contour_closure_case3_open_arc():
    """Case 3: An open arc (180 deg) must be rejected and return NaN perimeter."""
    r = 0.10
    t = np.linspace(0, np.pi, 32)
    arc = np.stack([r * np.cos(t), r * np.sin(t), np.zeros(32)], axis=1)

    closed, reason = is_contour_closed(arc)
    assert not closed, "Open arc must NOT be reported as closed"
    perim = _loop_perimeter(arc, require_closed=True)
    assert math.isnan(perim), "Open arc must return NaN perimeter when closed is required"


def test_contour_closure_case4_small_gap():
    """Case 4: A contour with a small gap (missing sector) must be rejected."""
    r = 0.10
    # 360 - 25 degrees = 335 degree arc
    t = np.linspace(0, 2 * np.pi - math.radians(25), 55)
    small_gap_contour = np.stack([r * np.cos(t), r * np.sin(t), np.zeros(55)], axis=1)

    closed, reason = is_contour_closed(small_gap_contour)
    assert not closed, "Contour with small gap must be rejected"
    perim = _loop_perimeter(small_gap_contour, require_closed=True)
    assert math.isnan(perim)


def test_contour_closure_case5_large_gap():
    """Case 5: A contour with a large gap (missing 90 deg quarter) must be rejected."""
    r = 0.10
    t = np.linspace(0, 1.5 * np.pi, 45)
    large_gap_contour = np.stack([r * np.cos(t), r * np.sin(t), np.zeros(45)], axis=1)

    closed, reason = is_contour_closed(large_gap_contour)
    assert not closed, "Contour with large gap must be rejected"
    perim = _loop_perimeter(large_gap_contour, require_closed=True)
    assert math.isnan(perim)


def test_contour_closure_case6_empty_contour():
    """Case 6: An empty contour must be rejected safely."""
    empty = np.zeros((0, 3), dtype=np.float64)
    closed, reason = is_contour_closed(empty)
    assert not closed, "Empty contour must be rejected"
    perim = _loop_perimeter(empty, require_closed=True)
    assert math.isnan(perim)

    none_closed, _ = is_contour_closed(None)
    assert not none_closed
    assert math.isnan(_loop_perimeter(None, require_closed=True))


def test_contour_closure_case7_duplicate_invalid_points():
    """Case 7: A contour containing degenerate duplicates or NaN/Inf must be rejected."""
    # Degenerate: all points identical
    all_same = np.full((10, 3), 5.0)
    closed, reason = is_contour_closed(all_same)
    assert not closed, "Degenerate all-identical points must be rejected"
    assert math.isnan(_loop_perimeter(all_same, require_closed=True))

    # Points with NaN
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    with_nan = np.stack([np.cos(t), np.sin(t), np.zeros(50)], axis=1)
    with_nan[10, 0] = float("nan")
    closed_nan, reason_nan = is_contour_closed(with_nan)
    assert not closed_nan, "Contour with NaN must be rejected"
    assert math.isnan(_loop_perimeter(with_nan, require_closed=True))


def test_contour_closure_case8_hairpin_close_endpoints():
    """
    Case 8: A contour whose endpoints are physically close (2 mm) but whose
    geometry is an open fold / hairpin must be detected as unclosed.
    """
    # Hairpin going up 200 mm, turning 180 deg, coming down 200 mm with only 2 mm spacing
    x_hp = np.concatenate([np.zeros(25), np.linspace(0, 0.002, 10), np.full(25, 0.002)])
    y_hp = np.concatenate([np.linspace(0, 0.20, 25), np.full(10, 0.20), np.linspace(0.20, 0, 25)])
    hairpin = np.stack([x_hp, y_hp, np.zeros(60)], axis=1)

    # Endpoints are physically 2 mm apart:
    endpoint_dist = np.linalg.norm(hairpin[-1] - hairpin[0])
    assert endpoint_dist < 0.003, f"Endpoints should be close (< 3 mm), got {endpoint_dist*1000:.1f} mm"

    closed, reason = is_contour_closed(hairpin)
    assert not closed, f"Hairpin fold with close endpoints must be rejected as unclosed, got: {reason}"
    perim = _loop_perimeter(hairpin, require_closed=True)
    assert math.isnan(perim), "Hairpin must return NaN perimeter"


def test_loop_perimeter_does_not_silently_close_open_contour():
    """Verify that _loop_perimeter does not bridge an open arc with an artificial chord."""
    r = 0.10
    # Half-circle: arc length = pi * r = 0.314159 m
    # Straight-line chord across ends = 2 * r = 0.200 m
    # Silently closing would yield: 0.314159 + 0.200 = 0.514159 m!
    t = np.linspace(0, np.pi, 50)
    half_circle = np.stack([r * np.cos(t), r * np.sin(t), np.zeros(50)], axis=1)

    # Closed=True must return NaN rather than silent 0.514 m
    perim_closed = _loop_perimeter(half_circle, require_closed=True)
    assert math.isnan(perim_closed), "Must not return a finite perimeter for an unclosed arc"

    # Closed=False (open path length) must return the arc length (0.314 m), NOT chord-augmented (0.514 m)
    open_len = _loop_perimeter(half_circle, require_closed=False)
    expected_arc = math.pi * r
    assert abs(open_len - expected_arc) < 0.005, (
        f"Open path length should be arc length ~{expected_arc:.3f} m, got {open_len:.3f} m"
    )
