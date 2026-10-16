"""
Deterministic Offline Measurement Validation Suite.

Covers:
  1. Reference-dimension comparison (--reference-dims):
     - Production error calculation on known example (303, 198, 100 vs 300, 200, 100 mm).
     - Zero-error case.
     - Zero reference-dimension handling (division by zero guard).
     - Dimension ordering invariance (descending sorting L -> W -> H).
     - HeadTarget vs BoxTarget branching verification.
  2. Arbitrarily rotated synthetic box measurement:
     - Euler rotation (37 deg, 53 deg, 19 deg).
     - 1.5 mm Gaussian noise with fixed seed.
     - Six-face support, planarity RMS <= 4.0 mm, dimension error within justified tolerance (< 8.0 mm).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from config import PipelineConfig, TargetConfig
from pipeline import ScanPipeline
from targets.box import BoxTarget
from test_head_synthetic import create_synthetic_head


def _make_box_pcd(l: float, w: float, h: float, n_pts: int = 6000) -> o3d.geometry.PointCloud:
    """Create a clean centered box point cloud of size (l x w x h) in metres."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
    mesh.translate((-l / 2.0, -w / 2.0, -h / 2.0))
    return mesh.sample_points_poisson_disk(number_of_points=n_pts)


# ===========================================================================
# Task 1: Reference-Dimension Comparison Tests
# ===========================================================================

def test_reference_dims_error_calculation(tmp_path: Path):
    """
    Test production --reference-dims calculation on the standard test case:
      Reference: [300, 200, 100] mm -> [0.300, 0.200, 0.100] m
      Measured:  [303, 198, 100] mm -> [0.303, 0.198, 0.100] m
    Expected production calculation:
      - Dimension ordering: length (303 vs 300), width (198 vs 200), height (100 vs 100)
      - error_mm uses abs(m - r) * 1000.0: [3.0, 2.0, 0.0] mm
      - error_pct uses abs(m - r) / r * 100.0: [1.0, 1.0, 0.0] %
    Also verifies the signed discrepancies (m - r): [+3.0, -2.0, 0.0] mm.
    """
    pcd = _make_box_pcd(0.303, 0.198, 0.100, n_pts=6000)

    cfg = PipelineConfig(
        target=TargetConfig(name="box"),
        reference_dims_m=[0.300, 0.200, 0.100],
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=tmp_path,
    )
    pipeline = ScanPipeline(cfg)
    result = pipeline.measure(pcd)

    assert "reference_comparison" in result, "reference_comparison missing from result"
    ref_comp = result["reference_comparison"]

    assert set(ref_comp.keys()) == {"length", "width", "height"}

    # Length: 303.0 measured vs 300.0 reference
    l_info = ref_comp["length"]
    assert np.isclose(l_info["measured_mm"], 303.0, atol=0.2)
    assert np.isclose(l_info["reference_mm"], 300.0, atol=0.1)
    assert np.isclose(l_info["error_mm"], 3.0, atol=0.2)
    assert np.isclose(l_info["error_pct"], 1.0, atol=0.1)
    # Signed discrepancy check
    signed_err_l = l_info["measured_mm"] - l_info["reference_mm"]
    assert np.isclose(signed_err_l, 3.0, atol=0.2)

    # Width: 198.0 measured vs 200.0 reference
    w_info = ref_comp["width"]
    assert np.isclose(w_info["measured_mm"], 198.0, atol=0.2)
    assert np.isclose(w_info["reference_mm"], 200.0, atol=0.1)
    # Production uses abs(m - r)
    assert np.isclose(w_info["error_mm"], 2.0, atol=0.2)
    assert np.isclose(w_info["error_pct"], 1.0, atol=0.1)
    # Signed discrepancy check
    signed_err_w = w_info["measured_mm"] - w_info["reference_mm"]
    assert np.isclose(signed_err_w, -2.0, atol=0.2)

    # Height: 100.0 measured vs 100.0 reference
    h_info = ref_comp["height"]
    assert np.isclose(h_info["measured_mm"], 100.0, atol=0.2)
    assert np.isclose(h_info["reference_mm"], 100.0, atol=0.1)
    assert np.isclose(h_info["error_mm"], 0.0, atol=0.2)
    assert np.isclose(h_info["error_pct"], 0.0, atol=0.1)
    signed_err_h = h_info["measured_mm"] - h_info["reference_mm"]
    assert np.isclose(signed_err_h, 0.0, atol=0.2)


def test_reference_dims_zero_error(tmp_path: Path):
    """Test the exact zero-error case where measured matches reference exactly."""
    pcd = _make_box_pcd(0.300, 0.200, 0.100, n_pts=6000)

    cfg = PipelineConfig(
        target=TargetConfig(name="box"),
        reference_dims_m=[0.300, 0.200, 0.100],
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=tmp_path,
    )
    pipeline = ScanPipeline(cfg)
    result = pipeline.measure(pcd)

    ref_comp = result["reference_comparison"]
    for axis in ["length", "width", "height"]:
        assert np.isclose(ref_comp[axis]["error_mm"], 0.0, atol=0.2), f"Non-zero error_mm on {axis}"
        assert np.isclose(ref_comp[axis]["error_pct"], 0.0, atol=0.1), f"Non-zero error_pct on {axis}"


def test_reference_dims_zero_reference_value(tmp_path: Path):
    """
    Test zero reference dimension handling.
    The implementation protects against division by zero via:
      err_pct = (abs(m_val - r_val) / r_val * 100.0) if r_val > 0 else 0.0
    Confirms zero reference value produces error_pct == 0.0 without ZeroDivisionError.
    """
    pcd = _make_box_pcd(0.300, 0.200, 0.100, n_pts=6000)

    cfg = PipelineConfig(
        target=TargetConfig(name="box"),
        reference_dims_m=[0.300, 0.200, 0.0],
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=tmp_path,
    )
    pipeline = ScanPipeline(cfg)
    result = pipeline.measure(pcd)

    ref_comp = result["reference_comparison"]
    assert np.isclose(ref_comp["height"]["reference_mm"], 0.0)
    assert np.isclose(ref_comp["height"]["error_pct"], 0.0)
    assert ref_comp["height"]["error_mm"] > 0.0  # measured - 0.0


def test_reference_dims_ordering_invariance(tmp_path: Path):
    """
    Test that reference dimensions provided in arbitrary order are sorted descending
    to match the descending ordering of measured dimensions (length >= width >= height).
    """
    pcd = _make_box_pcd(0.303, 0.198, 0.100, n_pts=6000)

    # Provided in ascending order: [100, 200, 300] mm
    cfg = PipelineConfig(
        target=TargetConfig(name="box"),
        reference_dims_m=[0.100, 0.200, 0.300],
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=tmp_path,
    )
    pipeline = ScanPipeline(cfg)
    result = pipeline.measure(pcd)

    ref_comp = result["reference_comparison"]
    assert np.isclose(ref_comp["length"]["reference_mm"], 300.0, atol=0.1)
    assert np.isclose(ref_comp["width"]["reference_mm"], 200.0, atol=0.1)
    assert np.isclose(ref_comp["height"]["reference_mm"], 100.0, atol=0.1)


def test_reference_dims_head_target_branching(tmp_path: Path):
    """
    Verify HeadTarget behavior with reference_dims_m configured.
    In pipeline.py lines 1255-1322, reference_comparison is implemented under the
    'else' block (box target). For head target, reference_comparison is currently
    skipped and remains absent from result.
    """
    head_pcd = create_synthetic_head(w=0.16, d=0.20, h=0.24, n_pts=5000)

    cfg = PipelineConfig(
        target=TargetConfig(name="head"),
        reference_dims_m=[0.24, 0.20, 0.16],
        show_stage_windows=False,
        save_intermediate=False,
        output_dir=tmp_path,
    )
    pipeline = ScanPipeline(cfg)
    result = pipeline.measure(head_pcd)

    assert result["target"] == "head"
    # Documented finding: HeadTarget path does not execute reference_comparison in current pipeline.py
    assert "reference_comparison" not in result


# ===========================================================================
# Task 2: Arbitrarily Rotated Synthetic Box Tests
# ===========================================================================

def test_arbitrarily_rotated_synthetic_box():
    """
    Test BoxTarget measurement on an arbitrarily rotated synthetic box:
      True dimensions: 300 x 200 x 100 mm (0.30 x 0.20 x 0.10 m)
      Euler rotation angles: 37 deg, 53 deg, 19 deg (XYZ convention)
      Sensor noise: 1.5 mm Gaussian position noise (fixed seed=42)

    Verifies:
      1. Correct dimension order: length >= width >= height.
      2. Status is 'validated' only if all 6 faces are supported and max planarity RMS <= 4.0 mm.
      3. Dimension errors are checked against a statistically justified tolerance:
         - Planarity RMS threshold (4.0 mm) measures 1-sided point scatter from a single plane.
         - Caliper dimension error measures two opposing faces with trim_percentile=0.5%.
           For sigma=1.5 mm, the 99% noise spread across both opposing faces is ~2 * 2.58 * sigma = ~7.7 mm.
         - Tolerance of 8.0 mm (0.008 m) is justified and well within test_synthetic.py's 15.0 mm bound.
      4. Volume is validated (volume_m3 is not None) and within 15% of true volume.
    """
    np.random.seed(42)

    l_true, w_true, h_true = 0.30, 0.20, 0.10
    true_dims_mm = np.array([300.0, 200.0, 100.0])

    mesh = o3d.geometry.TriangleMesh.create_box(width=l_true, height=w_true, depth=h_true)
    mesh.translate((-l_true / 2.0, -w_true / 2.0, -h_true / 2.0))

    # Apply Euler rotation: 37 deg, 53 deg, 19 deg
    R = o3d.geometry.get_rotation_matrix_from_xyz((
        np.radians(37.0),
        np.radians(53.0),
        np.radians(19.0),
    ))
    mesh.rotate(R, center=(0, 0, 0))

    # Sample surface and add 1.5 mm Gaussian noise
    pcd = mesh.sample_points_poisson_disk(number_of_points=8000)
    pts = np.asarray(pcd.points)
    noise = np.random.normal(0, 0.0015, size=pts.shape)
    pcd.points = o3d.utility.Vector3dVector(pts + noise)

    target = BoxTarget()
    result = target.measure(pcd)

    # 1. Quality validation
    assert result["status"] == "validated", f"Expected 'validated', got {result['status']}"
    planarity = result["planarity_check"]
    assert planarity["all_faces_supported"] is True, "Not all 6 faces were supported"
    assert planarity["max_rms_mm"] <= 4.0, f"Max face RMS {planarity['max_rms_mm']} exceeded 4.0 mm threshold"

    # 2. Correct dimension ordering
    obb = result["oriented_bbox"]
    meas_l = obb["length_m"] * 1000.0
    meas_w = obb["width_m"] * 1000.0
    meas_h = obb["height_m"] * 1000.0
    assert meas_l >= meas_w >= meas_h, f"Dimensions not sorted descending: {meas_l}, {meas_w}, {meas_h}"

    # 3. Dimension error evaluation
    meas_dims_mm = np.array([meas_l, meas_w, meas_h])
    errors_mm = np.abs(meas_dims_mm - true_dims_mm)
    print(f"\nRotated Box Measured Dims: L={meas_l:.1f}, W={meas_w:.1f}, H={meas_h:.1f} mm")
    print(f"Dimension Errors (mm):      L={errors_mm[0]:.1f}, W={errors_mm[1]:.1f}, H={errors_mm[2]:.1f} mm")
    print(f"Planarity Max RMS:          {planarity['max_rms_mm']:.2f} mm (limit: 4.0 mm)")

    # Justified 8.0 mm tolerance for 2-sided noisy boundary caliper measurement (1.5 mm sigma)
    DIM_TOLERANCE_MM = 8.0
    assert np.all(errors_mm < DIM_TOLERANCE_MM), (
        f"Dimension error exceeded {DIM_TOLERANCE_MM} mm tolerance: {errors_mm.tolist()}"
    )

    # 4. Volume check
    assert obb["volume_m3"] is not None, "volume_m3 must not be None for validated box"
    true_vol_m3 = l_true * w_true * h_true  # 0.006 m^3
    vol_err_pct = abs(obb["volume_m3"] - true_vol_m3) / true_vol_m3 * 100.0
    assert vol_err_pct < 15.0, f"Volume error {vol_err_pct:.1f}% exceeded 15% tolerance"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
