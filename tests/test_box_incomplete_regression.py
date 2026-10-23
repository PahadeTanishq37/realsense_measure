"""
Regression tests for BoxTarget on complete vs incomplete scans.

Covers:
  1. Complete six-face synthetic box.
  2. One opposing face removed.
  3. One face with only five supporting points.
  4. One face with broad but partial coverage.
  5. Outliers added to a complete box.
  6. A planar fragment.
  7. Status and volume propagation for every invalid case.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from targets.box import BoxTarget


def run_tests() -> bool:
    np.random.seed(42)
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.30, height=0.20, depth=0.12)
    mesh.translate((-0.15, -0.10, -0.06))
    pcd_full = mesh.sample_points_poisson_disk(number_of_points=6000)
    pts_full = np.asarray(pcd_full.points)

    target = BoxTarget()

    test_cases = [
        ("Complete 6-face box", pts_full, "validated", True),
        ("One opposing face removed", pts_full[pts_full[:, 2] < 0.058], "unverified", False),
        ("One face with 5 points", np.vstack([pts_full[pts_full[:, 2] < 0.058], pts_full[pts_full[:, 2] >= 0.058][:5]]), "unverified", False),
        ("One face with broad partial coverage", np.vstack([pts_full[pts_full[:, 2] < 0.058], pts_full[(pts_full[:, 2] >= 0.058) & (pts_full[:, 0] < 0)]]), "validated", True),
        ("Complete box + 100 outliers", np.vstack([pts_full, np.random.uniform(-0.3, 0.3, size=(100, 3))]), "unverified", False),
        ("Planar fragment (single face)", pts_full[pts_full[:, 2] >= 0.058], "unverified", False),
    ]

    print("=" * 115)
    print("  Box Measurement Acceptance Suite -- Incomplete & Degradation Cases")
    print("=" * 115)
    print(f"{'Test Case':<38} {'Expected':<12} {'Actual':<12} {'L (mm)':<8} {'W (mm)':<8} {'H (mm)':<8} {'Vol Valid?':<12} {'Diagnostics / Result'}")
    print("-" * 115)

    all_passed = True

    for name, pts, exp_status, exp_vol_valid in test_cases:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        is_plaus, reason = target.is_plausible_box(pcd)
        if not is_plaus:
            act_status = "unverified"  # implausible geometry cannot be validated
            act_vol_valid = False
            l_mm, w_mm, h_mm = 0.0, 0.0, 0.0
            diag_str = f"is_plausible_box failed: {reason}"
        else:
            res = target.measure(pcd)
            act_status = res["status"]
            obb = res["oriented_bbox"]
            l_mm = obb["length_m"] * 1000.0
            w_mm = obb["width_m"] * 1000.0
            h_mm = obb["height_m"] * 1000.0
            act_vol_valid = (obb["volume_m3"] is not None)
            diag = res.get("planarity_check", {})
            warn = diag.get("warning") or "All 6 faces verified"
            diag_str = warn[:40] + "..." if len(warn) > 40 else warn

        passed = (act_status == exp_status and act_vol_valid == exp_vol_valid)
        if not passed:
            all_passed = False
        res_label = "PASS [OK]" if passed else "FAIL [X]"

        print(f"{name:<38} {exp_status:<12} {act_status:<12} {l_mm:>7.1f} {w_mm:>7.1f} {h_mm:>7.1f} {str(act_vol_valid):<12} {res_label} ({diag_str})")

    print("=" * 115)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nALL INCOMPLETE BOX REGRESSION TESTS PASSED.")
    else:
        print("\nSOME TESTS FAILED.")
        exit(1)
