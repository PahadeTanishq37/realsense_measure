"""
tests/body/test_measure_geometry.py
=====================================
Unit tests for body/measure.py geometry helpers.

Tests are pure-geometry (no anny model required) so they run instantly
and without network access.

Coverage
--------
1. Ellipse cross-section perimeter within 0.5% of analytic value
2. Cylinder cross-section perimeter within 0.5% of analytic value
3. Box cross-section perimeter within 0.5% of analytic value
4. Left/right limb measures differ for an artificially asymmetric mesh
5. Torso loop selection excludes arm cross-sections
6. NaN returned when plane misses the mesh entirely
7. _chain_segments produces correct loop from clean segments
8. Measurement dataclass: source defaults to "model", is_valid works
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


from realsense_measure.body.measure import (
    Measurement,
    _chain_segments,
    _circumference_at_z,
    _loop_perimeter,
    _slice_mesh_at_z,
    _select_torso_loop,
    measure_body,
)


# ---------------------------------------------------------------------------
# Synthetic mesh helpers
# ---------------------------------------------------------------------------

def _make_cylinder_mesh(
    radius: float = 0.10,
    height: float = 0.40,
    n_sides: int = 128,
    z_bottom: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a closed cylinder mesh aligned with the Z axis.

    Vertices on two rings (bottom + top) plus the two centre caps.
    Triangulated with n_sides quads per ring band + n_sides triangles per cap.

    Coordinate system: +Z up (matching anny convention).
    Returns (vertices (V,3), faces (F,3)).
    """
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    z_bot = z_bottom
    z_top = z_bottom + height

    # rings
    bot_ring = np.stack([xs, ys, np.full(n_sides, z_bot)], axis=1)
    top_ring = np.stack([xs, ys, np.full(n_sides, z_top)], axis=1)

    # caps
    bot_center = np.array([[0.0, 0.0, z_bot]])
    top_center = np.array([[0.0, 0.0, z_top]])

    verts = np.vstack([bot_ring, top_ring, bot_center, top_center])
    # indices: bottom ring = 0..n-1, top ring = n..2n-1
    # bot_center = 2n, top_center = 2n+1
    n = n_sides
    faces = []

    # Side quads (two triangles per quad)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + i])
        faces.append([j, n + j, n + i])

    # Bottom cap
    bc = 2 * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([bc, j, i])

    # Top cap
    tc = 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, n + i, n + j])

    return verts, np.array(faces, dtype=np.int64)


def _make_ellipse_cylinder_mesh(
    a: float = 0.12,    # semi-axis along X
    b: float = 0.08,    # semi-axis along Y
    height: float = 0.30,
    n_sides: int = 256,
    z_bottom: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder with elliptical cross-section (a × b)."""
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
    xs = a * np.cos(angles)
    ys = b * np.sin(angles)

    z_bot = z_bottom
    z_top = z_bottom + height

    bot_ring = np.stack([xs, ys, np.full(n_sides, z_bot)], axis=1)
    top_ring = np.stack([xs, ys, np.full(n_sides, z_top)], axis=1)
    bot_center = np.array([[0.0, 0.0, z_bot]])
    top_center = np.array([[0.0, 0.0, z_top]])

    verts = np.vstack([bot_ring, top_ring, bot_center, top_center])
    n = n_sides
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + i])
        faces.append([j, n + j, n + i])
    bc = 2 * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([bc, j, i])
    tc = 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, n + i, n + j])

    return verts, np.array(faces, dtype=np.int64)


def _make_box_mesh(
    lx: float = 0.20,   # half-length in X
    ly: float = 0.15,   # half-length in Y
    height: float = 0.30,
    z_bottom: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned rectangular box.  Cross-section perimeter = 4*(lx+ly).
    The box extends from (-lx,-ly) to (+lx,+ly) in XY.
    """
    z_bot = z_bottom
    z_top = z_bottom + height

    corners_xy = [
        (lx, ly), (-lx, ly), (-lx, -ly), (lx, -ly)
    ]
    bot = np.array([[x, y, z_bot] for x, y in corners_xy])
    top = np.array([[x, y, z_top] for x, y in corners_xy])
    bc = np.array([[0.0, 0.0, z_bot]])
    tc = np.array([[0.0, 0.0, z_top]])

    verts = np.vstack([bot, top, bc, tc])
    n = 4

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + i])
        faces.append([j, n + j, n + i])
    bc_idx = 2 * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([bc_idx, j, i])
    tc_idx = 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc_idx, n + i, n + j])

    return verts, np.array(faces, dtype=np.int64)


def _analytic_ellipse_perimeter(a: float, b: float, n_terms: int = 100) -> float:
    """
    Ramanujan approximation for ellipse perimeter (accurate to < 1e-7 for
    normal b/a ratios).
    """
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoopGeometry:
    """Tests for low-level loop geometry helpers."""

    def test_loop_perimeter_square(self):
        """Unit square perimeter = 4."""
        pts = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0]])
        p = _loop_perimeter(pts)
        assert abs(p - 4.0) < 1e-10, f"Expected 4.0 got {p}"

    def test_chain_segments_triangle(self):
        """Three segments chain into a triangle loop."""
        segs = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.5, 1.0, 0.0])),
            (np.array([0.5, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        ]
        loops = _chain_segments(segs, tol=1e-9)
        assert len(loops) >= 1
        p = _loop_perimeter(loops[0])
        # Isosceles triangle: base 1.0, two sides each sqrt(0.5^2 + 1.0^2) = sqrt(1.25) = 1.118
        expected = 1.0 + 2.0 * math.sqrt(0.5**2 + 1.0**2)
        assert abs(p - expected) < 0.01, f"Expected ~{expected:.3f} got {p:.3f}"


class TestCircumferenceSyntheticMeshes:
    """
    Verify circumference values against analytic formulae.
    Tolerance: ≤ 0.5% relative error.
    """

    TOL = 0.005   # 0.5%

    def test_cylinder_circumference(self):
        """Circular cylinder: perimeter at mid-height = 2π r."""
        r = 0.10   # 100 mm radius
        verts, faces = _make_cylinder_mesh(radius=r, height=0.40,
                                           n_sides=256, z_bottom=0.0)
        z_mid = 0.20
        circ_m = _circumference_at_z(verts, faces, z_mid) / 1000.0   # mm → m
        analytic = 2 * math.pi * r
        rel_err = abs(circ_m - analytic) / analytic
        assert rel_err < self.TOL, (
            f"Cylinder circ: got {circ_m*1000:.2f} mm, "
            f"expected {analytic*1000:.2f} mm, "
            f"rel_err={rel_err:.4f}"
        )

    def test_ellipse_circumference(self):
        """Elliptical cylinder: perimeter within 0.5% of Ramanujan formula."""
        a, b = 0.12, 0.08
        verts, faces = _make_ellipse_cylinder_mesh(a=a, b=b, height=0.30,
                                                   n_sides=512, z_bottom=0.0)
        z_mid = 0.15
        circ_m = _circumference_at_z(verts, faces, z_mid) / 1000.0
        analytic = _analytic_ellipse_perimeter(a, b)
        rel_err = abs(circ_m - analytic) / analytic
        assert rel_err < self.TOL, (
            f"Ellipse circ: got {circ_m*1000:.2f} mm, "
            f"expected {analytic*1000:.2f} mm, "
            f"rel_err={rel_err:.4f}"
        )

    def test_box_circumference(self):
        """Rectangular box cross-section: perimeter = 4*(lx+ly)."""
        lx, ly = 0.20, 0.15
        verts, faces = _make_box_mesh(lx=lx, ly=ly, height=0.30, z_bottom=0.0)
        z_mid = 0.15
        circ_m = _circumference_at_z(verts, faces, z_mid) / 1000.0
        analytic = 4 * (lx + ly)   # sides only, no diagonals
        # Note: a box mesh has only diagonal triangles on each face;
        # the cross-section at z_mid passes through the flat portions,
        # giving a rectangular loop of perimeter 4*(2*lx + 2*ly)/2 = 4*(lx+ly)
        # But with only 4 vertices per cap, the cross-section goes through
        # the quad edges:
        # actual perimeter = 4*(lx+ly) since each edge goes corner to corner
        rel_err = abs(circ_m - analytic) / analytic
        assert rel_err < self.TOL, (
            f"Box circ: got {circ_m*1000:.2f} mm, "
            f"expected {analytic*1000:.2f} mm, "
            f"rel_err={rel_err:.4f}"
        )


class TestLeftRightAsymmetry:
    """
    Verify that left and right limb circumferences differ when the mesh
    is explicitly asymmetric.
    """

    def test_left_right_differ_for_asymmetric_mesh(self):
        """
        Build two cylinders with different radii, one on each side of X=0.
        Left cylinder: radius=0.08 m  (X < 0)
        Right cylinder: radius=0.12 m (X > 0)

        The circumference at mid-height should differ by ≥ 20% and match
        their analytic values within 1%.
        """
        r_left  = 0.08
        r_right = 0.12
        height = 0.40
        n = 256

        # Left cylinder centred at X = -0.20
        def _cyl(cx, cy, radius, n_sides, z_bot=0.0, z_top=0.4):
            angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
            xs = cx + radius * np.cos(angles)
            ys = cy + radius * np.sin(angles)
            bot = np.stack([xs, ys, np.full(n_sides, z_bot)], axis=1)
            top = np.stack([xs, ys, np.full(n_sides, z_top)], axis=1)
            bc = np.array([[cx, cy, z_bot]])
            tc = np.array([[cx, cy, z_top]])
            verts = np.vstack([bot, top, bc, tc])
            ns = n_sides
            faces = []
            for i in range(ns):
                j = (i + 1) % ns
                faces.append([i, j, ns + i])
                faces.append([j, ns + j, ns + i])
            bc_i = 2 * ns
            for i in range(ns):
                j = (i + 1) % ns
                faces.append([bc_i, j, i])
            tc_i = 2 * ns + 1
            for i in range(ns):
                j = (i + 1) % ns
                faces.append([tc_i, ns + i, ns + j])
            return verts, np.array(faces, dtype=np.int64)

        v_l, f_l = _cyl(-0.25, 0.0, r_left,  n)
        v_r, f_r = _cyl(+0.25, 0.0, r_right, n)

        # Merge the two meshes
        n_l = len(v_l)
        verts = np.vstack([v_l, v_r])
        faces = np.vstack([f_l, f_r + n_l])

        z_mid = 0.20

        circ_left  = _circumference_at_z(verts, faces, z_mid, side="left")
        circ_right = _circumference_at_z(verts, faces, z_mid, side="right")

        expected_left  = 2 * math.pi * r_left  * 1000.0   # mm
        expected_right = 2 * math.pi * r_right * 1000.0   # mm

        rel_err_l = abs(circ_left  - expected_left)  / expected_left
        rel_err_r = abs(circ_right - expected_right) / expected_right

        assert rel_err_l < 0.01, (
            f"Left circ: got {circ_left:.2f} mm, expected {expected_left:.2f} mm"
        )
        assert rel_err_r < 0.01, (
            f"Right circ: got {circ_right:.2f} mm, expected {expected_right:.2f} mm"
        )

        # The two sides must differ
        assert abs(circ_left - circ_right) > 50.0, (
            f"Expected L/R to differ by >50 mm; got {circ_left:.2f} vs {circ_right:.2f}"
        )


class TestTorsoLoopSelection:
    """
    Verify that the torso loop is correctly selected when arms are present.
    """

    def test_torso_larger_than_arms(self):
        """
        Build a large central cylinder (torso) flanked by two small ones (arms).
        At the same Z height, circumference with torso=True should return
        the largest-area loop (central cylinder).
        """
        n = 128
        r_torso = 0.20
        r_arm   = 0.04

        def _ring(cx, cy, r, z=0.0):
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            return np.stack(
                [cx + r * np.cos(angles), cy + r * np.sin(angles), np.full(n, z)],
                axis=1,
            )

        z_bot, z_top = 0.0, 0.30
        v_torso_bot = _ring(0.0,   0.0, r_torso, z_bot)
        v_torso_top = _ring(0.0,   0.0, r_torso, z_top)
        v_arm1_bot  = _ring(-0.4,  0.0, r_arm,   z_bot)
        v_arm1_top  = _ring(-0.4,  0.0, r_arm,   z_top)
        v_arm2_bot  = _ring(+0.4,  0.0, r_arm,   z_bot)
        v_arm2_top  = _ring(+0.4,  0.0, r_arm,   z_top)

        def _make_sides(bot, top, offset=0):
            ns = len(bot)
            fs = []
            for i in range(ns):
                j = (i + 1) % ns
                fs.append([offset + i, offset + j, offset + ns + i])
                fs.append([offset + j, offset + ns + j, offset + ns + i])
            return fs

        v_all = np.vstack([
            v_torso_bot, v_torso_top,     # 0..2n-1
            v_arm1_bot,  v_arm1_top,      # 2n..4n-1
            v_arm2_bot,  v_arm2_top,      # 4n..6n-1
        ])
        f_all = np.array(
            _make_sides(v_torso_bot, v_torso_top, 0) +
            _make_sides(v_arm1_bot,  v_arm1_top,  2 * n) +
            _make_sides(v_arm2_bot,  v_arm2_top,  4 * n),
            dtype=np.int64,
        )

        z_mid = 0.15
        # With torso=True, should return the large cylinder's circumference
        torso_circ = _circumference_at_z(v_all, f_all, z_mid, torso=True) / 1000.0
        expected   = 2 * math.pi * r_torso
        rel_err    = abs(torso_circ - expected) / expected

        assert rel_err < 0.01, (
            f"Torso circ: got {torso_circ*1000:.2f} mm, "
            f"expected {expected*1000:.2f} mm, rel_err={rel_err:.4f}"
        )


class TestMissingPlane:
    """Edge cases where the slice plane misses the mesh."""

    def test_plane_above_mesh_returns_nan(self):
        verts, faces = _make_cylinder_mesh(radius=0.1, height=0.3, z_bottom=0.0)
        result = _circumference_at_z(verts, faces, z_plane=0.5)
        assert math.isnan(result), f"Expected NaN for plane above mesh, got {result}"

    def test_plane_below_mesh_returns_nan(self):
        verts, faces = _make_cylinder_mesh(radius=0.1, height=0.3, z_bottom=1.0)
        result = _circumference_at_z(verts, faces, z_plane=0.5)
        assert math.isnan(result), f"Expected NaN for plane below mesh, got {result}"


class TestMeasurementDataclass:
    """Unit tests for the Measurement dataclass."""

    def test_source_default(self):
        m = Measurement(name="test", value_mm=100.0, method="test_method")
        assert m.source == "model"

    def test_is_valid_normal(self):
        m = Measurement(name="test", value_mm=250.0, method="x")
        assert m.is_valid()

    def test_is_valid_nan(self):
        m = Measurement(name="test", value_mm=float("nan"), method="x")
        assert not m.is_valid()

    def test_repr(self):
        m = Measurement(name="chest", value_mm=950.5, method="surface_path")
        r = repr(m)
        assert "chest" in r
        assert "950.5" in r


class TestMeasureBodySyntheticHumanoid:
    """
    Smoke test: build a very simple pseudo-humanoid mesh and call
    measure_body.  Checks that the function returns the expected set of
    measurements without crashing, and that torso measurements are
    plausibly larger than arm measurements.
    """

    @pytest.fixture(scope="class")
    def synthetic_body_output(self):
        """
        Build a minimal pseudo-humanoid BodyOutput from scratch.
        Torso: large cylinder on Z axis, radius=0.20 m, Z=[-0.80, 0.20]
        Head:  small sphere above, radius=0.10 m, Z=[0.20, 0.50]
        Left arm:  thin cylinder at X=-0.30, radius=0.04, Z=[-0.30, 0.10]
        Right arm: thin cylinder at X=+0.30, radius=0.04, Z=[-0.30, 0.10]
        Left leg:  cylinder at X=-0.10, radius=0.07, Z=[-1.40, -0.80]
        Right leg: cylinder at X=+0.10, radius=0.07, Z=[-1.40, -0.80]
        """
        from realsense_measure.body.model import BodyOutput

        def _cyl(cx, cy, r, z_bot, z_top, n=64):
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            xs = cx + r * np.cos(angles)
            ys = cy + r * np.sin(angles)
            bot = np.stack([xs, ys, np.full(n, z_bot)], axis=1)
            top = np.stack([xs, ys, np.full(n, z_top)], axis=1)
            bc  = np.array([[cx, cy, z_bot]])
            tc  = np.array([[cx, cy, z_top]])
            v = np.vstack([bot, top, bc, tc]).astype(np.float32)
            fs = []
            for i in range(n):
                j = (i + 1) % n
                fs.append([i, j, n + i])
                fs.append([j, n + j, n + i])
            bc_i, tc_i = 2 * n, 2 * n + 1
            for i in range(n):
                j = (i + 1) % n
                fs.append([bc_i, j, i])
                fs.append([tc_i, n + i, n + j])
            return v, np.array(fs, dtype=np.int64)

        parts = [
            _cyl(0.0,   0.0,  0.20, -0.80,  0.20, n=128),  # torso
            _cyl(0.0,   0.0,  0.10,  0.20,  0.50, n=64),   # head
            _cyl(-0.30, 0.0,  0.04, -0.30,  0.10, n=64),   # left arm
            _cyl(+0.30, 0.0,  0.04, -0.30,  0.10, n=64),   # right arm
            _cyl(-0.10, 0.0,  0.07, -1.40, -0.80, n=64),   # left leg
            _cyl(+0.10, 0.0,  0.07, -1.40, -0.80, n=64),   # right leg
        ]

        offset = 0
        all_verts = []
        all_faces = []
        for vp, fp in parts:
            all_verts.append(vp)
            all_faces.append(fp + offset)
            offset += len(vp)

        verts_np = np.vstack(all_verts).astype(np.float32)
        faces_np = np.vstack(all_faces).astype(np.int64)

        # Fake joint positions (J=10, 3) — enough for fallback stature calc
        # Use approximate anny-like bone names
        bone_names = [
            "root", "neck01", "head",
            "upperleg01.L", "lowerleg01.L", "foot.L",
            "upperleg01.R", "lowerleg01.R", "foot.R",
            "upperarm01.L",
        ]
        joints_np = np.array([
            [0.0,  0.0,  -0.80],  # root (pelvis)
            [0.0,  0.0,   0.10],  # neck01
            [0.0,  0.0,   0.35],  # head
            [-0.10, 0.0, -0.80],  # upperleg01.L
            [-0.10, 0.0, -1.10],  # lowerleg01.L
            [-0.10, 0.0, -1.35],  # foot.L
            [+0.10, 0.0, -0.80],  # upperleg01.R
            [+0.10, 0.0, -1.10],  # lowerleg01.R
            [+0.10, 0.0, -1.35],  # foot.R
            [-0.30, 0.0,   0.00], # upperarm01.L
        ], dtype=np.float32)

        body_out = BodyOutput(
            vertices=verts_np[np.newaxis, ...],
            faces=faces_np,
            joints=joints_np[np.newaxis, ...],
            joint_names=bone_names,
            phenotypes={"test": True},
            raw={},
        )
        return body_out

    def test_measure_body_returns_all_keys(self, synthetic_body_output):
        meas = measure_body(synthetic_body_output)
        expected_keys = {
            "stature",
            "head_circumference",
            "neck_base_circumference",
            "chest_circumference",
            "waist_circumference",
            "hip_circumference",
            "crotch_height",
            "ankle_circumference",
            "shoulder_elbow_length",
            "shoulder_wrist_length",
            "inseam",
            "upper_arm_circumference_left",
            "upper_arm_circumference_right",
            "forearm_circumference_left",
            "forearm_circumference_right",
            "mid_thigh_circumference_left",
            "mid_thigh_circumference_right",
            "max_calf_circumference_left",
            "max_calf_circumference_right",
        }
        assert expected_keys.issubset(set(meas.keys())), (
            f"Missing keys: {expected_keys - set(meas.keys())}"
        )

    def test_source_is_model(self, synthetic_body_output):
        meas = measure_body(synthetic_body_output)
        for m in meas.values():
            assert m.source == "model", f"{m.name} has source={m.source!r}"

    def test_stature_plausible(self, synthetic_body_output):
        meas = measure_body(synthetic_body_output)
        s = meas["stature"].value_mm
        assert s > 1500 and s < 2500, f"Stature {s:.1f} mm implausible for synthetic body"

    def test_chest_larger_than_arm(self, synthetic_body_output):
        meas = measure_body(synthetic_body_output)
        chest = meas["chest_circumference"].value_mm
        arm_l = meas["upper_arm_circumference_left"].value_mm
        arm_r = meas["upper_arm_circumference_right"].value_mm
        if not math.isnan(chest) and not math.isnan(arm_l):
            assert chest > arm_l, f"Chest {chest:.1f} should be > arm_L {arm_l:.1f}"
        if not math.isnan(chest) and not math.isnan(arm_r):
            assert chest > arm_r, f"Chest {chest:.1f} should be > arm_R {arm_r:.1f}"

