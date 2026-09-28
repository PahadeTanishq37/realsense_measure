"""
Regression test for the registration stage (no camera needed).

Builds one asymmetric cloud, makes frames from it with small KNOWN motions,
and checks that every registration path recovers them.  Also checks that a
junk frame (unrelated flat plane) is REJECTED by the gating instead of
corrupting the result.

Run from the project folder:   python test_registration_regression.py
"""
import contextlib
import copy
import io
import sys

import numpy as np
import open3d as o3d

from config import RegistrationConfig
import registration as reg

TOL_MM = 3.0


def make_asymmetric_cloud() -> o3d.geometry.PointCloud:
    a = o3d.geometry.TriangleMesh.create_box(0.30, 0.20, 0.12)
    b = o3d.geometry.TriangleMesh.create_box(0.10, 0.08, 0.10)
    b.translate([0.30, 0.0, 0.0])
    c = o3d.geometry.TriangleMesh.create_box(0.06, 0.06, 0.15)
    c.translate([0.05, 0.20, 0.0])
    mesh = a + b + c
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(8000).voxel_down_sample(0.004)


def make_junk_plane() -> o3d.geometry.PointCloud:
    pts = np.random.uniform(-0.3, 0.3, size=(3000, 3))
    pts[:, 2] = 0.0
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    return p


def make_frames(base, n=6, with_junk_at=None):
    """frame_k = base moved by the inverse of a known cumulative pose G_k."""
    frames, Gk = [], np.eye(4)
    axis = np.array([0.3, 1.0, 0.2]); axis /= np.linalg.norm(axis)
    for k in range(n):
        if k > 0:
            step = np.eye(4)
            step[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(5.0))
            step[:3, 3] = [0.005, -0.003, 0.002]
            Gk = Gk @ step
        frames.append(copy.deepcopy(base).transform(np.linalg.inv(Gk)))
    if with_junk_at is not None:
        frames.insert(with_junk_at, make_junk_plane())
    return frames


def mean_nn_mm(a, ref):
    return float(np.mean(np.asarray(a.compute_point_cloud_distance(ref))) * 1000)


def run(fn, frames, cfg):
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(frames, cfg)


def main() -> int:
    np.random.seed(1)
    base = make_asymmetric_cloud()
    cfg = RegistrationConfig(ransac_max_iter=100_000)
    failures = []

    paths = [("sequential", reg.register_sequence),
             ("multiway", reg.register_sequence_multiway)]

    # 1) small known motions must be recovered by every path
    frames = make_frames(base)
    for name, fn in paths:
        aligned, diag = run(fn, frames, cfg)
        errs = [mean_nn_mm(a, aligned[0]) for a in aligned[1:]]
        ok = max(errs) < TOL_MM and all(d[2] for d in diag)
        print(f"[known motion] {name:10s} errors(mm)={np.round(errs, 1)}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"known motion / {name}")

    # 2) a junk frame must be rejected and must not damage the others
    junk_at = 3
    frames = make_frames(base, with_junk_at=junk_at)
    for name, fn in paths:
        aligned, diag = run(fn, frames, cfg)
        rejected = not diag[junk_at][2]
        good = [i for i in range(1, len(frames)) if i != junk_at and diag[i][2]]
        errs = [mean_nn_mm(aligned[i], aligned[0]) for i in good]
        ok = rejected and len(good) == len(frames) - 2 and max(errs) < TOL_MM
        print(f"[junk frame  ] {name:10s} junk rejected={rejected}  others ok={max(errs) < TOL_MM if errs else False}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"junk frame / {name}")

    print("\nRESULT:", "PASS" if not failures else f"FAIL -> {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
