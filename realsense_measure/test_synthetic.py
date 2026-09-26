"""
Synthetic end-to-end test for the realsense_measure pipeline.

Validates the full isolate → register → fuse → measure chain WITHOUT a real
camera or any display window.  All interaction is via printed text.

Run with:
    python test_synthetic.py

A PASS means the pipeline recovers all three box dimensions to within 15 mm
of ground truth on simulated, noisy, partially-visible point clouds — a
realistic upper bound for accumulated error from sensor noise + registration.
"""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

from config import PreprocessConfig, RegistrationConfig
from preprocessing import isolate_object
from reconstruction import fuse_point_clouds
from registration import register_sequence
from targets.box import BoxTarget

# ---------------------------------------------------------------------------
# Ground-truth box dimensions (metres)
# ---------------------------------------------------------------------------
TRUE_L, TRUE_W, TRUE_H = 0.30, 0.20, 0.12


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def make_box_mesh(l: float, w: float, h: float) -> o3d.geometry.TriangleMesh:
    """
    Create a solid box mesh of size (l × w × h) centred at the origin.

    Open3D's create_box places one corner at the origin; we shift by half
    the extents so the box is symmetric around (0, 0, 0).
    """
    mesh = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
    mesh.translate((-l / 2.0, -w / 2.0, -h / 2.0))
    mesh.compute_vertex_normals()
    return mesh


def make_table_plane(size: float = 1.2) -> o3d.geometry.TriangleMesh:
    """
    Create a thin flat box representing a background table surface.

    Centred in X/Y, sitting at z ∈ [-0.01, 0] so its top face is flush
    with z = 0.  The box rests on top of this plane.
    """
    thickness = 0.01
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness)
    mesh.translate((-size / 2.0, -size / 2.0, -thickness))
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# View sampler
# ---------------------------------------------------------------------------

def sample_view(
    box_mesh: o3d.geometry.TriangleMesh,
    table_mesh: o3d.geometry.TriangleMesh,
    R: np.ndarray,
    t: np.ndarray,
    n_points: int = 20_000,
    noise_std: float = 0.0015,
) -> o3d.geometry.PointCloud:
    """
    Simulate one depth-camera frame of the rotated/translated box + table.

    Steps:
      1. Rotate and translate a copy of the box mesh.
      2. Sample the surface using Poisson-disk sampling (carries normals).
      3. Keep only points whose surface normal faces the camera (+Z direction),
         simulating the one-sided visibility of a real depth sensor.
      4. Add Gaussian sensor noise to the surviving box points.
      5. Add ~4 000 noisy table points — the isolation step must remove these,
         just as it would real background on a live camera.

    Parameters
    ----------
    box_mesh, table_mesh:
        Canonical meshes (not mutated).
    R:
        3×3 rotation matrix applied to the box.
    t:
        3-element translation vector applied after rotation.
    n_points:
        Surface samples taken from the rotated box before visibility filtering.
    noise_std:
        Standard deviation (metres) of Gaussian position noise — 1.5 mm
        matches typical RealSense D455 depth noise at ~0.5 m range.

    Returns
    -------
    o3d.geometry.PointCloud
        Combined box (visible face) + table background points.
    """
    # --- rotate and translate a copy of the box --------------------------
    box_copy = copy.deepcopy(box_mesh)
    box_copy.rotate(R, center=(0.0, 0.0, 0.0))
    box_copy.translate(t)
    box_copy.compute_vertex_normals()

    # Poisson-disk sampling preserves normals (uniformly random sampling does
    # not reliably propagate them through the sample step in all O3D versions).
    box_pcd = box_copy.sample_points_poisson_disk(number_of_points=n_points)

    # --- visibility filter: keep only points facing the camera (+Z) -------
    # A real depth sensor sees only surfaces whose outward normal has a
    # positive Z component (i.e., facing the camera which looks down -Z).
    normals = np.asarray(box_pcd.normals)
    points = np.asarray(box_pcd.points)

    visible_mask = normals[:, 2] > 0.05   # Z-component threshold
    vis_points = points[visible_mask]

    # --- add sensor noise -------------------------------------------------
    vis_points = vis_points + np.random.normal(0.0, noise_std, vis_points.shape)

    # --- table background points ------------------------------------------
    table_pcd = table_mesh.sample_points_uniformly(number_of_points=4_000)
    table_pts = np.asarray(table_pcd.points)
    table_pts = table_pts + np.random.normal(0.0, noise_std, table_pts.shape)

    # --- combine and return -----------------------------------------------
    all_pts = np.vstack([vis_points, table_pts])
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(all_pts)
    return combined


# ---------------------------------------------------------------------------
# Random rotation
# ---------------------------------------------------------------------------

def random_rotation() -> np.ndarray:
    """
    Return a random 3×3 rotation matrix with rotation angle in [-1.3, 1.3] rad.

    WHY THE RANGE MUST BE WIDE: if consecutive frames differ only by small
    rotations (say < 0.3 rad), the camera keeps seeing roughly the same face
    or pair of faces of the box.  The dimension aligned with the camera's
    viewing axis (+Z) never gets revealed in a clean face-on view, so DBSCAN
    isolation and the OBB fit both underestimate that axis — sometimes by
    several centimetres.  Using wide-angle rotations forces genuinely
    different faces to be exposed in each frame, which is also good practice
    when capturing a real object: show the front, then the side, then the top,
    not just wobble it slightly.
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-1.3, 1.3)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main() -> None:
    # Seed numpy for reproducibility.
    # NOTE: Open3D's RANSAC uses its own internal random state that we cannot
    # control from Python, so exact fitness/rmse numbers will vary slightly
    # from run to run even with the same numpy seed.  Per-run variation of a
    # few mm in the final measurement is expected and fine — the 15 mm
    # tolerance absorbs it.
    np.random.seed(7)

    print("=" * 60)
    print("  realsense_measure -- synthetic end-to-end test")
    print(f"  Ground truth:  L={TRUE_L*1000:.1f} mm  "
          f"W={TRUE_W*1000:.1f} mm  H={TRUE_H*1000:.1f} mm")
    print("=" * 60)

    # ---- build scene meshes ----------------------------------------------
    box_mesh   = make_box_mesh(TRUE_L, TRUE_W, TRUE_H)
    table_mesh = make_table_plane(size=1.2)

    # ---- generate frames -------------------------------------------------
    # Four canonical orientations that guarantee each face of the box is
    # visible in at least one frame, plus random extra frames for robustness.
    gra = o3d.geometry.get_rotation_matrix_from_axis_angle

    canonical_rotations = [
        np.eye(3),                                     # front face up
        gra(np.array([1.2, 0.0, 0.0])),               # tilted around X
        gra(np.array([0.0, 1.2, 0.0])),               # tilted around Y
        gra(np.array([1.2, 1.0, 0.0])),               # combined tilt
    ]
    extra_rotations = [random_rotation() for _ in range(4)]
    all_rotations = canonical_rotations + extra_rotations   # 8 frames total

    def rand_t() -> np.ndarray:
        """Small random XY shift + fixed Z lift to put the box above the table."""
        return np.array([
            np.random.uniform(-0.03, 0.03),
            np.random.uniform(-0.03, 0.03),
            0.10,
        ])

    raw_frames = [
        sample_view(box_mesh, table_mesh, R, rand_t())
        for R in all_rotations
    ]

    # ---- preprocessing: isolate the object per frame ---------------------
    pre_cfg = PreprocessConfig()
    print(f"\n[1/4] Isolating object in {len(raw_frames)} frames ...")
    isolated_frames: list[o3d.geometry.PointCloud] = []
    for i, frame in enumerate(raw_frames):
        n_before = len(frame.points)
        iso = isolate_object(frame, pre_cfg)
        n_after = len(iso.points)
        print(f"  frame {i}: {n_before:>6} pts  ->  {n_after:>5} pts after isolation")
        isolated_frames.append(iso)

    # ---- registration: align all frames into one coordinate system -------
    reg_cfg = RegistrationConfig()
    print(f"\n[2/4] Registering sequence (growing-reference) ...")
    aligned_frames, diagnostics = register_sequence(isolated_frames, reg_cfg)
    for i, (fitness, rmse) in enumerate(diagnostics):
        print(f"  frame {i}: fitness={fitness:.4f}  rmse={rmse*1000:.3f} mm")

    # ---- fusion: merge all aligned frames --------------------------------
    print(f"\n[3/4] Fusing {len(aligned_frames)} aligned frames ...")
    fused = fuse_point_clouds(aligned_frames, pre_cfg)
    print(f"  fused cloud: {len(fused.points)} points")

    # ---- measurement -----------------------------------------------------
    print("\n[4/4] Segmenting and measuring ...")
    target = BoxTarget(preprocess_cfg=pre_cfg)
    segmented = target.segment(fused)
    print(f"  segmented cloud: {len(segmented.points)} points")

    result = target.measure(segmented)
    obb_info = result["oriented_bbox"]

    # Recovered dims are already sorted longest->shortest by BoxTarget.measure
    rec_dims  = np.array([obb_info["length_m"], obb_info["width_m"], obb_info["height_m"]])
    true_dims = np.sort([TRUE_L, TRUE_W, TRUE_H])[::-1]   # match same ordering

    errors_m  = np.abs(rec_dims - true_dims)
    errors_mm = errors_m * 1000.0

    print("\n  Dimension comparison (sorted L->W->H):")
    print(f"  {'Axis':<8} {'Ground truth':>14} {'Recovered':>12} {'Error':>10}")
    print(f"  {'-'*46}")
    labels = ["Length", "Width ", "Height"]
    for lbl, gt, rc, err in zip(labels, true_dims, rec_dims, errors_mm):
        print(f"  {lbl:<8} {gt*1000:>11.1f} mm  {rc*1000:>9.1f} mm  {err:>7.2f} mm")

    print(f"\n  Volume: {obb_info['volume_m3']*1e6:.1f} cm^3  "
          f"(true: {TRUE_L*TRUE_W*TRUE_H*1e6:.1f} cm^3)")

    # ---- assertion -------------------------------------------------------
    # 15 mm tolerance reflects realistic accumulated error from simulated
    # sensor noise (1.5 mm sigma) + registration residuals, not an exact match.
    TOLERANCE_M = 0.015
    passed = bool(np.all(errors_m < TOLERANCE_M))

    print()
    if passed:
        print("RESULT: PASS  [OK]  (all axes within 15 mm of ground truth)")
    else:
        failing = [
            f"{lbl.strip()} error={err:.2f} mm"
            for lbl, err in zip(labels, errors_mm)
            if err / 1000.0 >= TOLERANCE_M
        ]
        print(f"RESULT: FAIL  [X]  -- axis error(s) exceed 15 mm: {', '.join(failing)}")

    print("=" * 60)

    assert passed, (
        f"Synthetic test FAILED: per-axis errors (mm) = {errors_mm.tolist()}"
    )


if __name__ == "__main__":
    main()
