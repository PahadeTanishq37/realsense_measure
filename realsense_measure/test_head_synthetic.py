"""
Synthetic test suite for 3D Human Head Geometric Envelope Estimator.

Covers:
  1. Known complete synthetic geometry (W=160, D=200, H=240 mm).
  2. Rotation and translation invariance (arbitrary yaw, pitch, roll, and shift).
  3. Missing posterior skull / hair absorption (status: unverified).
  4. Missing lateral or superior surface (status: unverified).
  5. Head connected to neck and shoulders (neck-cut isolates head envelope).
  6. Sparse or planar geometry (status: unverified).
  7. Outliers and non-finite (NaN / Inf) points (defensive filtering).
  8. Ambiguous orientation (symmetric sphere with no facial relief -> unverified).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from targets.head import HeadTarget


def create_synthetic_head(
    w: float = 0.16,
    d: float = 0.20,
    h: float = 0.24,
    n_pts: int = 4000,
    with_nose: bool = True,
) -> o3d.geometry.PointCloud:
    """Create a canonical synthetic human head point cloud with facial relief."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    v = np.asarray(mesh.vertices)
    v[:, 0] *= w / 0.20  # lateral (breadth)
    v[:, 1] *= h / 0.20  # longitudinal (height)
    v[:, 2] *= d / 0.20  # anterior-posterior (length)
    mesh.vertices = o3d.utility.Vector3dVector(v)

    if with_nose:
        nose = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.035)
        R_nose = nose.get_rotation_matrix_from_xyz((np.pi / 2.0, 0, 0))
        nose.rotate(R_nose, center=(0, 0, 0))
        nose.translate((0.0, -0.015, d / 2.0))
        mesh = mesh + nose

    pcd = mesh.sample_points_poisson_disk(number_of_points=n_pts)
    return pcd


def create_head_with_neck_shoulders(n_pts: int = 8000) -> o3d.geometry.PointCloud:
    """Create a synthetic head attached to neck and broad shoulders."""
    head_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    v = np.asarray(head_mesh.vertices)
    v[:, 0] *= 0.8  # 160 mm
    v[:, 1] *= 1.2  # 240 mm (y from -0.12 to +0.12)
    v[:, 2] *= 1.0  # 200 mm
    head_mesh.vertices = o3d.utility.Vector3dVector(v)

    nose = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.035)
    R_nose = nose.get_rotation_matrix_from_xyz((np.pi / 2.0, 0, 0))
    nose.rotate(R_nose, center=(0, 0, 0))
    nose.translate((0.0, -0.015, 0.10))
    head_mesh = head_mesh + nose

    # Neck cylinder
    neck_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.045, height=0.08)
    neck_mesh.translate((0.0, -0.16, 0.0))

    # Shoulders box
    shoulders_mesh = o3d.geometry.TriangleMesh.create_box(width=0.40, height=0.12, depth=0.18)
    shoulders_mesh.translate((-0.20, -0.32, -0.09))

    combined = head_mesh + neck_mesh + shoulders_mesh
    pcd = combined.sample_points_poisson_disk(number_of_points=n_pts)
    return pcd


def run_all_tests() -> bool:
    np.random.seed(42)
    target = HeadTarget()

    print("=" * 115)
    print("  3D Human Head Geometric Envelope Estimator -- Acceptance Suite")
    print("=" * 115)
    print(f"{'Test Case':<36} {'Expected':<12} {'Actual':<12} {'Breadth':<9} {'Length':<9} {'Height':<9} {'Volume':<10} {'Result'}")
    print("-" * 115)

    all_passed = True
    tests = []

    # 1. Complete Synthetic Geometry
    pcd_1 = create_synthetic_head(w=0.16, d=0.20, h=0.24)
    res_1 = target.measure(target.segment(pcd_1))
    tests.append((
        "1. Complete synthetic head",
        res_1,
        "validated",
        True,
        (150.0, 170.0),  # breadth mm
        (190.0, 240.0),  # length mm (with nose)
        (225.0, 255.0),  # height mm
    ))

    # 2. Rotation and Translation Invariance
    pcd_2 = create_synthetic_head(w=0.16, d=0.20, h=0.24)
    R_rot = o3d.geometry.get_rotation_matrix_from_xyz((0.45, -0.60, 0.35))
    pcd_2.rotate(R_rot, center=(0, 0, 0))
    pcd_2.translate((0.15, -0.30, 0.75))
    res_2 = target.measure(target.segment(pcd_2))
    tests.append((
        "2. Arbitrary 3D rotation & shift",
        res_2,
        "validated",
        True,
        (150.0, 170.0),
        (190.0, 240.0),
        (225.0, 255.0),
    ))

    # 3. Missing Posterior Surface (Hair Absorption)
    pcd_3_full = create_synthetic_head(w=0.16, d=0.20, h=0.24)
    pts_3 = np.asarray(pcd_3_full.points)
    pts_3_no_post = pts_3[pts_3[:, 2] > -0.02]  # remove back of head
    pcd_3 = o3d.geometry.PointCloud()
    pcd_3.points = o3d.utility.Vector3dVector(pts_3_no_post)
    res_3 = target.measure(target.segment(pcd_3))
    tests.append((
        "3. Missing posterior surface",
        res_3,
        "unverified",
        False,
        None,
        None,
        None,
    ))

    # 4. Missing Lateral Surface
    pcd_4_full = create_synthetic_head(w=0.16, d=0.20, h=0.24)
    pts_4 = np.asarray(pcd_4_full.points)
    pts_4_no_lat = pts_4[pts_4[:, 0] > -0.03]  # remove left ear/lateral face
    pcd_4 = o3d.geometry.PointCloud()
    pcd_4.points = o3d.utility.Vector3dVector(pts_4_no_lat)
    res_4 = target.measure(target.segment(pcd_4))
    tests.append((
        "4. Missing lateral surface",
        res_4,
        "unverified",
        False,
        None,
        None,
        None,
    ))

    # 5. Head Connected to Neck and Shoulders
    pcd_5 = create_head_with_neck_shoulders()
    res_5 = target.measure(target.segment(pcd_5))
    tests.append((
        "5. Head + neck + shoulders",
        res_5,
        "validated",
        True,
        (150.0, 170.0),
        (190.0, 240.0),
        (220.0, 260.0),  # height should be head only (< 280 mm), not torso (> 450 mm)
    ))

    # 6. Sparse or Planar Geometry
    pcd_6 = o3d.geometry.PointCloud()
    pcd_6.points = o3d.utility.Vector3dVector(np.random.uniform(-0.1, 0.1, size=(80, 3)))
    res_6 = target.measure(target.segment(pcd_6))
    tests.append((
        "6. Sparse point cloud (< 200 pts)",
        res_6,
        "unverified",
        False,
        None,
        None,
        None,
    ))

    # 7. Outliers and Non-Finite Points
    pcd_7 = create_synthetic_head(w=0.16, d=0.20, h=0.24)
    pts_7 = np.asarray(pcd_7.points)
    # Add non-finite coordinates + random spatial outliers
    nan_pts = np.array([[np.nan, np.nan, np.nan], [np.inf, -np.inf, 0.0]])
    outlier_pts = np.random.uniform(-0.5, 0.5, size=(40, 3))
    pts_7_polluted = np.vstack([pts_7, nan_pts, outlier_pts])
    pcd_7_polluted = o3d.geometry.PointCloud()
    pcd_7_polluted.points = o3d.utility.Vector3dVector(pts_7_polluted)
    res_7 = target.measure(target.segment(pcd_7_polluted))
    tests.append((
        "7. Outliers & non-finite coords",
        res_7,
        "validated",
        True,
        (150.0, 170.0),
        (190.0, 240.0),
        (225.0, 255.0),
    ))

    # 8. Ambiguous Orientation (Pure Featureless Sphere)
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    pcd_8 = sphere_mesh.sample_points_poisson_disk(number_of_points=3000)
    res_8 = target.measure(target.segment(pcd_8))
    tests.append((
        "8. Ambiguous orientation (sphere)",
        res_8,
        "unverified",
        False,
        None,
        None,
        None,
    ))

    for name, res, exp_status, exp_vol_valid, b_range, l_range, h_range in tests:
        act_status = res.get("status", "unverified")
        env = res.get("cranial_envelope_3d", {})
        b_mm = env.get("breadth_mm", 0.0)
        l_mm = env.get("length_mm", 0.0)
        h_mm = env.get("height_mm", 0.0)
        act_vol_valid = (res.get("oriented_bbox", {}).get("volume_m3") is not None)

        status_ok = (act_status == exp_status)
        vol_ok = (act_vol_valid == exp_vol_valid)
        dim_ok = True
        if exp_status == "validated" and b_range and l_range and h_range:
            dim_ok = (
                b_range[0] <= b_mm <= b_range[1]
                and l_range[0] <= l_mm <= l_range[1]
                and h_range[0] <= h_mm <= h_range[1]
            )

        passed = status_ok and vol_ok and dim_ok
        if not passed:
            all_passed = False
        res_label = "PASS [OK]" if passed else "FAIL [X]"

        b_str = f"{b_mm:.1f} mm" if b_mm > 0 else "--"
        l_str = f"{l_mm:.1f} mm" if l_mm > 0 else "--"
        h_str = f"{h_mm:.1f} mm" if h_mm > 0 else "--"
        vol_str = "Valid" if act_vol_valid else "None"

        print(f"{name:<36} {exp_status:<12} {act_status:<12} {b_str:<9} {l_str:<9} {h_str:<9} {vol_str:<10} {res_label}")

    print("=" * 115)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nALL 8 HEAD SYNTHETIC TESTS PASSED SUCCESSFULLY.")
    else:
        print("\nSOME HEAD SYNTHETIC TESTS FAILED.")
        exit(1)
