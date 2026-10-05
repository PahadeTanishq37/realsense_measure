"""
Regression tests for Stage 2 head-cluster selection in extract_plausible_cluster().

Validates the fix for scan_test_35 where depth shadows around the chin/jawline
separated the head from the torso, causing extract_plausible_cluster() to mistakenly
select the larger chest/torso cluster instead of the head cluster.

Tests:
  1. Connected head + torso remains accepted as one unified cluster.
  2. Disconnected head (~2000 pts) and torso (~4500 pts) with last_centroid closer
     to the torso correctly selects the HEAD cluster (reproducing scan_test_35 failure).
  3. Small unrelated noise cluster above the head does not hijack head selection.
  4. Non-head targets (e.g. TargetConfig(name="box")) retain existing behavior unchanged.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from config import PreprocessConfig, TargetConfig
from preprocessing import extract_plausible_cluster


def _make_head_cluster(
    center_y: float = -0.05,
    center_z: float = 0.50,
    step: float = 0.0035,
) -> o3d.geometry.PointCloud:
    """Create a synthetic head point cloud (~2000 points, elliptical front surface)."""
    gx = np.arange(-0.08, 0.08, step)
    gy = np.arange(-0.10, 0.10, step)
    X, Y = np.meshgrid(gx, gy)
    mask = (X / 0.08) ** 2 + (Y / 0.10) ** 2 <= 1.0
    X = X[mask]
    Y = Y[mask]
    Z = np.sqrt(np.maximum(0, 0.09**2 - X**2 - Y**2)) * 0.5 + center_z
    pts = np.column_stack([X, Y + center_y, Z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _make_torso_cluster(
    center_y: float = 0.18,
    center_z: float = 0.50,
    step: float = 0.0035,
) -> o3d.geometry.PointCloud:
    """Create a synthetic torso/chest point cloud (~4100 points, wide surface)."""
    gx = np.arange(-0.18, 0.18, step)
    gy = np.arange(-0.07, 0.07, step)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.flatten(), (Y + center_y).flatten(), np.full_like(X.flatten(), center_z)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _make_box_cluster(
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.50,
    extent: float = 0.16,
    step: float = 0.0035,
) -> o3d.geometry.PointCloud:
    """Create a planar box surface for non-head target tests."""
    half = extent / 2.0
    gx = np.arange(-half, half, step)
    gy = np.arange(-half, half, step)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([(X + center_x).flatten(), (Y + center_y).flatten(), np.full_like(X.flatten(), center_z)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ===========================================================================
# Test 1: Connected head + torso remains accepted as one unified cluster
# ===========================================================================
def test_connected_head_torso_remains_accepted_as_one_cluster():
    cfg = PreprocessConfig()
    cfg.cluster_eps_m = 0.02
    target_cfg = TargetConfig(name="head")

    # Head (Y = -0.05)
    head_pcd = _make_head_cluster(center_y=-0.05, center_z=0.50)
    # Torso (Y = +0.18)
    torso_pcd = _make_torso_cluster(center_y=0.18, center_z=0.50)
    # Neck connector bridging head and torso (spanning Y from +0.04 to +0.12)
    nx = np.arange(-0.04, 0.04, 0.0035)
    ny = np.arange(0.04, 0.12, 0.0035)
    nX, nY = np.meshgrid(nx, ny)
    neck_pts = np.column_stack([nX.flatten(), nY.flatten(), np.full_like(nX.flatten(), 0.50)])
    neck_pcd = o3d.geometry.PointCloud()
    neck_pcd.points = o3d.utility.Vector3dVector(neck_pts)

    connected = head_pcd + neck_pcd + torso_pcd

    selected = extract_plausible_cluster(connected, cfg, target_cfg=target_cfg)

    # When connected, the unified cluster spans from the cranium to the chest
    assert len(selected.points) >= 6000, f"Expected unified cluster, got {len(selected.points)} pts"
    pts = np.asarray(selected.points)
    assert pts[:, 1].min() < -0.10, "Upper head missing from connected cluster"
    assert pts[:, 1].max() > +0.20, "Lower torso missing from connected cluster"


# ===========================================================================
# Test 2: Disconnected head vs larger torso (scan_test_35 exact failure)
# ===========================================================================
def test_head_cluster_selected_when_disconnected_from_larger_torso():
    cfg = PreprocessConfig()
    cfg.cluster_eps_m = 0.02
    target_cfg = TargetConfig(name="head")

    # Frame B: jawline shadow separates head from torso
    # Cluster A = Head/face (~2046 pts, centroid Y ~ -0.05)
    head_pcd = _make_head_cluster(center_y=-0.05, center_z=0.50)

    # Cluster B = Torso/chest (~4120 pts, centroid Y ~ +0.18)
    torso_pcd = _make_torso_cluster(center_y=0.18, center_z=0.50)

    # Combined scene (head and torso are separate DBSCAN clusters with gap > 0.04m)
    scene = head_pcd + torso_pcd

    # Intentionally position last_centroid close to the torso (e.g. Y = +0.16)
    # to simulate the downward centroid drift that caused the scan_test_35 lockup
    last_centroid = np.array([0.0, 0.16, 0.50])

    selected = extract_plausible_cluster(scene, cfg, target_cfg=target_cfg, last_centroid=last_centroid)

    # The algorithm must select the HEAD cluster, NOT the larger torso cluster
    pts = np.asarray(selected.points)
    centroid_y = pts[:, 1].mean()

    assert centroid_y < 0.05, (
        f"Selected cluster centroid Y = {centroid_y:.3f}m corresponds to the torso! "
        f"Expected head cluster with Y < 0.05m."
    )
    assert len(pts) == pytest.approx(2046, abs=100), (
        f"Selected point count {len(pts)} does not match head cluster (~2046 pts)."
    )


# ===========================================================================
# Test 3: Small unrelated noise cluster above head does not hijack selection
# ===========================================================================
def test_small_unrelated_noise_cluster_not_selected_as_head():
    cfg = PreprocessConfig()
    cfg.cluster_eps_m = 0.02
    target_cfg = TargetConfig(name="head")

    # Head cluster: ~2046 pts, centroid Y = -0.05
    head_pcd = _make_head_cluster(center_y=-0.05, center_z=0.50)

    # Torso cluster: ~4120 pts, centroid Y = +0.18
    torso_pcd = _make_torso_cluster(center_y=0.18, center_z=0.50)

    # Small noise cluster above the head (e.g. ceiling/lamp fragment, ~228 pts, Y = -0.28)
    # Extent is 0.13m (just passing expected_min_size_m=0.12m), but point count is small
    nx = np.arange(-0.065, 0.065, 0.0035)
    ny = np.arange(-0.01, 0.01, 0.0035)
    nX, nY = np.meshgrid(nx, ny)
    noise_pts = np.column_stack([nX.flatten(), (nY - 0.28).flatten(), np.full_like(nX.flatten(), 0.50)])
    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.points = o3d.utility.Vector3dVector(noise_pts)

    scene = head_pcd + torso_pcd + noise_pcd

    selected = extract_plausible_cluster(scene, cfg, target_cfg=target_cfg)

    pts = np.asarray(selected.points)
    centroid_y = pts[:, 1].mean()

    # Must select the HEAD cluster (~2046 pts, Y ~ -0.05), NOT the noise cluster (Y = -0.28)
    # and NOT the torso cluster (Y = +0.18)
    assert -0.12 < centroid_y < 0.05, (
        f"Expected head cluster, got centroid Y = {centroid_y:.3f}m"
    )
    assert len(pts) == pytest.approx(2046, abs=100), (
        f"Expected ~2046 head points, got {len(pts)} points."
    )


# ===========================================================================
# Test 4: Box / non-head target behavior is completely unchanged
# ===========================================================================
def test_box_object_cluster_selection_unchanged():
    cfg = PreprocessConfig()
    cfg.cluster_eps_m = 0.02
    box_target_cfg = TargetConfig(name="box")

    # Two box-like clusters at different X positions:
    # Box 1 at X = 0.0, ~2100 pts
    box1 = _make_box_cluster(center_x=0.0, center_y=0.0, center_z=0.50, extent=0.16)

    # Box 2 at X = 0.35, ~2100 pts
    box2 = _make_box_cluster(center_x=0.35, center_y=0.0, center_z=0.50, extent=0.16)

    scene = box1 + box2

    # If last_centroid was tracking Box 2 (at X = 0.35):
    last_centroid = np.array([0.35, 0.0, 0.50])
    selected = extract_plausible_cluster(scene, cfg, target_cfg=box_target_cfg, last_centroid=last_centroid)

    # Must select Box 2 via proximity to last_centroid (original box behavior preserved)
    pts = np.asarray(selected.points)
    assert pts[:, 0].mean() > 0.20, "Box target failed to follow last_centroid tracking"

