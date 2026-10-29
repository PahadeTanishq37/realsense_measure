"""
Unit tests for visualizer robust bounding box framing helper.
Verifies robust box computation rejects outliers without GUI.
"""

import numpy as np
import open3d as o3d
import pytest
from visualizer import compute_robust_bounding_box, frame_view


def test_robust_bounding_box_outlier_rejection():
    """
    Requirement 5:
    A cloud of 10k points inside a 0.25 m head box plus 5 points 3 m away
    must give a centre within 2 cm of the head centre and an extent < 0.4 m.
    """
    np.random.seed(42)
    # Head centre at (0.1, -0.2, 0.5) to test arbitrary center
    head_center = np.array([0.1, -0.2, 0.5])
    half_box = 0.25 / 2.0  # 0.125 m

    # 10,000 points uniformly distributed in a 0.25m box
    head_points = np.random.uniform(
        low=head_center - half_box,
        high=head_center + half_box,
        size=(10000, 3),
    )

    # 5 outlier points 3 m away in various directions
    outliers = np.array([
        head_center + [3.0, 0.0, 0.0],
        head_center + [0.0, 3.0, 0.0],
        head_center + [0.0, 0.0, 3.0],
        head_center + [-3.0, 2.0, 1.0],
        head_center + [2.0, -3.0, 2.0],
    ])

    all_points = np.vstack([head_points, outliers])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    centre, max_extent, p1, p99 = compute_robust_bounding_box([pcd])

    centre_err = np.linalg.norm(centre - head_center)
    print(f"\nCentre error: {centre_err * 100:.2f} cm (must be < 2.0 cm)")
    print(f"Max extent:   {max_extent:.4f} m (must be < 0.4 m)")

    # Assertions per requirement 5
    assert centre_err < 0.02, f"Centre error {centre_err:.4f} m exceeded 2 cm limit"
    assert max_extent < 0.4, f"Max extent {max_extent:.4f} m exceeded 0.4 m limit"

    # Also test frame_view helper non-GUI interface
    c, ext, zoom = frame_view([pcd])
    assert np.allclose(c, centre)
    assert np.isclose(ext, max_extent)
    assert 0.1 <= zoom <= 1.0


def test_robust_bounding_box_mesh_and_lineset():
    """Verify compute_robust_bounding_box handles TriangleMesh and LineSet seamlessly."""
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.25, depth=0.2)
    centre, max_extent, p1, p99 = compute_robust_bounding_box(box_mesh)
    assert max_extent <= 0.35
    assert np.all(centre >= 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
