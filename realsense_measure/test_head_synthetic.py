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
    np.random.seed(42)
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


def test_neck_region_preservation_isolate_register_fuse() -> bool:
    print("\n" + "=" * 115)
    print("  Testing Neck-to-Head Preservation through Isolate -> Register -> Fuse Pipeline")
    print("=" * 115)
    import copy
    from config import PreprocessConfig, RegistrationConfig, TargetConfig
    from preprocessing import isolate_object
    from registration import register_frame_pair
    from reconstruction import fuse_point_clouds

    np.random.seed(42)
    # 1. Build head+neck+shoulders mesh with intentionally sparse/noisy neck band
    # to simulate sensor dropouts at the chin-to-neck transition.
    pcd_full = create_head_with_neck_shoulders(n_pts=8000)
    pts = np.asarray(pcd_full.points)

    # Induce an intentionally sparse/noisy neck dropout zone (y in [-0.17, -0.13])
    neck_mask = (pts[:, 1] >= -0.17) & (pts[:, 1] <= -0.13)
    drop_indices = np.where(neck_mask)[0]
    # Keep 70% of points in this weak-spot band and add noise, simulating real sensor dropout
    keep_drop = np.random.choice(drop_indices, size=int(0.70 * len(drop_indices)), replace=False)
    remove_mask = np.zeros(len(pts), dtype=bool)
    remove_mask[drop_indices] = True
    remove_mask[keep_drop] = False
    retained_indices = np.where(~remove_mask)[0]

    pcd_f0 = pcd_full.select_by_index(retained_indices)
    # Add subtle sensor noise in the neck transition
    f0_pts = np.asarray(pcd_f0.points)
    f0_neck_mask = (f0_pts[:, 1] >= -0.17) & (f0_pts[:, 1] <= -0.13)
    f0_pts[f0_neck_mask] += np.random.normal(0, 0.002, size=(np.sum(f0_neck_mask), 3))
    pcd_f0.points = o3d.utility.Vector3dVector(f0_pts)

    # Create a second frame slightly rotated (3 degrees around Y) to test registration
    R_step = o3d.geometry.get_rotation_matrix_from_xyz((0, np.deg2rad(3.0), 0))
    pcd_f1 = copy.deepcopy(pcd_f0).rotate(R_step, center=(0, 0, 0))

    cfg_prep = PreprocessConfig()
    cfg_reg = RegistrationConfig()
    target_cfg = TargetConfig(name="head")

    # Step 1: Preprocessing (isolate_object)
    iso_f0 = isolate_object(pcd_f0, cfg_prep, target_cfg=target_cfg)
    iso_f1 = isolate_object(pcd_f1, cfg_prep, target_cfg=target_cfg)

    assert iso_f0.has_points(), "Frame 0 failed object isolation"
    assert iso_f1.has_points(), "Frame 1 failed object isolation"

    # Step 2: Registration
    T_10, fit, rmse = register_frame_pair(iso_f1, iso_f0, cfg_reg)
    assert fit >= cfg_reg.min_accept_fitness, f"Registration failed (fitness={fit:.2f})"
    iso_f1_trans = copy.deepcopy(iso_f1).transform(T_10)

    # Step 3: Fusion
    fused = fuse_point_clouds([iso_f0, iso_f1_trans], cfg_prep, skip_final_clustering=True, skip_outlier_removal=True)
    fused_pts = np.asarray(fused.points)
    assert len(fused_pts) > 0, "Fused point cloud is empty"

    # Evaluate point density across bands
    # Head region (inferior cranium): y in [-0.10, -0.04]
    # Neck region: y in [-0.18, -0.12]
    # Shoulder region: y in [-0.26, -0.20]
    head_band = (fused_pts[:, 1] >= -0.10) & (fused_pts[:, 1] <= -0.04)
    neck_band = (fused_pts[:, 1] >= -0.18) & (fused_pts[:, 1] <= -0.12)
    shoulder_band = (fused_pts[:, 1] >= -0.26) & (fused_pts[:, 1] <= -0.20)

    n_head = int(np.sum(head_band))
    n_neck = int(np.sum(neck_band))
    n_shoulders = int(np.sum(shoulder_band))

    # Cross-sectional surface area estimates:
    # Neck cylinder radius ~0.045, height 0.06 -> surface ~ 0.017 m^2
    # Head band radius ~0.09, height 0.06 -> surface ~ 0.034 m^2 (~2x larger area)
    density_head = n_head / 0.034
    density_neck = n_neck / 0.017
    density_shoulder = n_shoulders / 0.045

    print(f"  Head band points:      {n_head} (area-density: {density_head:.0f} pts/m^2)")
    print(f"  Neck band points:      {n_neck} (area-density: {density_neck:.0f} pts/m^2)")
    print(f"  Shoulder band points:  {n_shoulders} (area-density: {density_shoulder:.0f} pts/m^2)")

    # Assertions:
    # 1. Neck and shoulders must NOT be zero/stripped
    assert n_neck > 50, f"Neck region was clipped! Only {n_neck} points found in neck band."
    assert n_shoulders > 50, f"Shoulder region was clipped! Only {n_shoulders} points found in shoulder band."

    # 2. Reasonable density: neck density should be at least 50% of the adjacent head density
    density_ratio = density_neck / density_head
    print(f"  Neck-to-head density ratio: {density_ratio:.2f} (threshold: >= 0.50)")
    assert density_ratio >= 0.50, f"Neck region too sparse: density ratio {density_ratio:.2f} < 0.50"

    print("  PASS: Neck-to-head bridge preserved with healthy point density throughout.")
    return True


if __name__ == "__main__":
    success_envelope = run_all_tests()
    success_neck = test_neck_region_preservation_isolate_register_fuse()
    if success_envelope and success_neck:
        print("\nALL HEAD SYNTHETIC TESTS (ENVELOPE SUITE + NECK PRESERVATION) PASSED SUCCESSFULLY.")
    else:
        print("\nSOME HEAD SYNTHETIC TESTS FAILED.")
        exit(1)
