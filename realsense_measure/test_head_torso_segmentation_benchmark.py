"""
Offline Head-Torso Segmentation Benchmark and Diagnostic Suite.

Covers:
  1. Synthetic Labeled Geometry Generator:
     - Head (ellipsoid + anterior nasal cone), neck (cylinder), slouched torso (wedge).
     - Deterministic Poisson-disk surface sampling (not solid volume).
     - Immutable semantic labels (0: Head, 1: Neck, 2: Torso).
     - Global pitch transformation with exact coordinate inversion.
  2. Segmentation Methods Comparison:
     - Method A: Production PCA Slicing (actual current implementation HeadTarget().segment).
     - Method C: Naive Fixed Camera-Y Slicing (baseline control).
     - Method B: Multi-Axis Conical Constriction Search Prototype (test-only helper).
  3. Strict Count Reconciliation & Metrics:
     - Surviving = Retained Head + Retained Neck + Retained Torso.
     - Head recall, Torso rejection, Head loss, Subject-local cut plane position.
  4. Independent Measurement Validation:
     - Stage B: Complete synthetic cranial geometry measurement.
     - Stage C: Posterior-truncated geometry invariance (must remain unverified, volume None).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from targets.head import HeadTarget


# ===========================================================================
# 1. Deterministic Synthetic Labeled Generator
# ===========================================================================

def create_synthetic_labeled_head_torso(
    pitch_deg: float = 0.0,
    slouch_deg: float = 25.0,
    n_head: int = 4500,
    n_neck: int = 1500,
    n_torso: int = 4000,
    truncate_posterior: bool = False,
    seed: int = 42,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Construct a deterministic synthetic human upper-body point cloud with
    immutable ground-truth semantic labels.

    Coordinate Convention (Subject-Local Frame P_local):
      +x_s: Right Lateral (breadth axis)
      +y_s: Superior / Vertex (height axis)
      +z_s: Anterior / Face (length/depth axis)

    Anatomical Specifications:
      - Head: Ellipsoid (width 160 mm, height 220 mm, depth 200 mm) centered at (0, 0, 0).
              Nasal cone (radius 20 mm, height 35 mm) pointing anteriorly along +z_s.
              Base ellipsoid span: z_s in [-100.0, +100.0] mm.
              Total geometric extent with nose: z_s in [-100.0, +135.0] mm (235.0 mm).
              Head boundary: y_s in [-0.110, +0.110] m.
      - Neck: Cylinder (radius 45 mm, height 70 mm) centered on y_s axis.
              Spans y_s in [-0.180, -0.110] m.
              Mandible boundary: y_s = -0.110 m. Clavicle boundary: y_s = -0.180 m.
      - Torso: Trapezoidal box (width 380 mm, depth 220 mm, height 250 mm).
               Spans y_s in [-0.430, -0.180] m before slouch.
               Pitched forward by slouch_deg around neck base (0, -0.180, 0).

    Sampling Method:
      Poisson-disk sampling on triangular surface meshes.
      NOTE: This is explicitly hollow 2D surface sampling, NOT solid volume sampling,
      matching optical RGB-D sensor reflections.

    Returns
    -------
    pcd_cam:
        PointCloud in camera frame after pitch rotation.
    pts_local:
        (N, 3) original subject-local coordinates.
    labels:
        (N,) integer semantic labels: 0=Head, 1=Neck, 2=Torso.
    R_pitch:
        (3, 3) rotation matrix mapping P_local -> P_cam.
    meta:
        Metadata dict containing nominal dimensions and counts.
    """
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    # 1. Head mesh
    head_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    vh = np.asarray(head_mesh.vertices)
    vh[:, 0] *= 0.16 / 0.20  # width 160 mm
    vh[:, 1] *= 0.22 / 0.20  # height 220 mm
    vh[:, 2] *= 0.20 / 0.20  # depth 200 mm
    head_mesh.vertices = o3d.utility.Vector3dVector(vh)

    # Nose cone: apex at +Z = 0.035, base at 0.0. Translated so base is at z=0.10, apex at z=0.135
    nose = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.035)
    nose.translate((0.0, 0.0, 0.10))
    head_mesh = head_mesh + nose

    # 2. Neck mesh: rotated by 90 deg around X so longitudinal axis aligns with subject-local Y
    neck_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.045, height=0.070)
    R_neck_x = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2.0, 0.0, 0.0))
    neck_mesh.rotate(R_neck_x, center=(0.0, 0.0, 0.0))
    neck_mesh.translate((0.0, -0.145, 0.0))  # y_s in [-0.180, -0.110] m

    # 3. Torso mesh: anchored at cervicothoracic junction at neck base (z = -0.045 m, y = -0.180 m)
    # Rotating forward by slouch_deg ensures all torso points rotate downward (y_s <= -0.180 m)
    # with exact touching contact at the base-of-neck boundary plane (zero volumetric penetration).
    torso_mesh = o3d.geometry.TriangleMesh.create_box(width=0.380, height=0.250, depth=0.220)
    torso_mesh.translate((-0.190, -0.430, -0.045))
    R_slouch = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(slouch_deg), 0.0, 0.0))
    torso_mesh.rotate(R_slouch, center=(0.0, -0.180, -0.045))

    # Surface sampling
    pcd_head = head_mesh.sample_points_poisson_disk(number_of_points=n_head)
    pcd_neck = neck_mesh.sample_points_poisson_disk(number_of_points=n_neck)
    pcd_torso = torso_mesh.sample_points_poisson_disk(number_of_points=n_torso)

    pts_h = np.asarray(pcd_head.points)
    pts_n = np.asarray(pcd_neck.points)
    pts_t = np.asarray(pcd_torso.points)

    if truncate_posterior:
        # Truncate posterior cranium (z_s < -0.010 m) to test partial view
        mask_h = pts_h[:, 2] >= -0.010
        pts_h = pts_h[mask_h]

    # Immutable semantic labels
    labels_h = np.zeros(len(pts_h), dtype=int)
    labels_n = np.ones(len(pts_n), dtype=int)
    labels_t = np.full(len(pts_t), 2, dtype=int)

    pts_local = np.vstack([pts_h, pts_n, pts_t])
    labels = np.concatenate([labels_h, labels_n, labels_t])

    # Global camera pitch transform
    R_pitch = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(pitch_deg), 0.0, 0.0))
    pts_cam = (R_pitch @ pts_local.T).T

    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pts_cam)

    meta = {
        "n_head": len(pts_h),
        "n_neck": len(pts_n),
        "n_torso": len(pts_t),
        "total_points": len(pts_local),
        "nominal_breadth_mm": 160.0,
        "nominal_height_mm": 220.0,
        "nominal_ellipsoid_depth_mm": 200.0,
        "nominal_geometric_depth_with_nose_mm": 235.0,
        "pitch_deg": pitch_deg,
        "slouch_deg": slouch_deg,
        "truncate_posterior": truncate_posterior,
        "meshes": {
            "head": head_mesh,
            "neck": neck_mesh,
            "torso": torso_mesh,
        },
    }

    return pcd_cam, pts_local, labels, R_pitch, meta


# ===========================================================================
# 2. Test-Only Segmentation Prototypes
# ===========================================================================

def slice_camera_y_baseline(pts_cam: np.ndarray) -> np.ndarray:
    """
    Baseline Control: Slice horizontally along pure camera vertical (+Y).
    Identifies 1D cross-sectional area minimum along camera Y.
    """
    y = pts_cam[:, 1]
    y_min, y_max = np.min(y), np.max(y)
    bins = np.linspace(y_min, y_max, 35)
    areas: list[float] = []
    for i in range(len(bins) - 1):
        m = (y >= bins[i]) & (y < bins[i + 1])
        if np.sum(m) < 10:
            areas.append(0.0)
        else:
            p = pts_cam[m]
            areas.append(float(np.ptp(p[:, 0]) * np.ptp(p[:, 2])))

    cand_idx = -1
    min_a = float("inf")
    for i in range(5, 25):
        if 0 < i < len(areas) - 1 and areas[i] < areas[i - 1] and areas[i] <= areas[i + 1]:
            if areas[i] < min_a:
                min_a = areas[i]
                cand_idx = i

    if cand_idx != -1 and min_a > 0:
        cut_y = (bins[cand_idx] + bins[cand_idx + 1]) / 2.0
        return pts_cam[y >= cut_y]
    return pts_cam


def multi_axis_constriction_search_prototype(
    pts_cam: np.ndarray,
    max_cone_deg: float = 25.0,
) -> np.ndarray:
    """
    Test-Only Prototype: Multi-Axis Conical Constriction Search.
    Evaluates transverse cross-sectional bounding areas over a conical bundle of candidate axes.
    """
    best_axis = None
    best_cut_val = None
    best_score = -1.0

    base_axis = np.array([0.0, 1.0, 0.0])
    candidate_axes = []
    for pitch in np.linspace(-max_cone_deg, max_cone_deg, 11):
        for roll in np.linspace(-15.0, 15.0, 5):
            R = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(pitch), 0.0, np.radians(roll)))
            candidate_axes.append(R @ base_axis)

    for axis in candidate_axes:
        proj = pts_cam @ axis
        p_min, p_max = np.min(proj), np.max(proj)
        bins = np.linspace(p_min, p_max, 35)
        areas: list[float] = []
        for i in range(len(bins) - 1):
            m = (proj >= bins[i]) & (proj < bins[i + 1])
            if np.sum(m) < 10:
                areas.append(0.0)
            else:
                p = pts_cam[m]
                orth = p - (p @ axis[:, None]) * axis
                areas.append(float(np.ptp(orth[:, 0]) * np.ptp(orth[:, 2])))

        for i in range(5, 25):
            if 0 < i < len(areas) - 1 and areas[i] < areas[i - 1] and areas[i] <= areas[i + 1]:
                a_min = areas[i]
                if a_min > 0:
                    head_mass = max(areas[i:]) if len(areas[i:]) else 0.0
                    torso_mass = max(areas[:i]) if len(areas[:i]) else 0.0
                    if head_mass >= 1.5 * a_min and torso_mass >= 1.3 * a_min:
                        score = (head_mass * torso_mass) / (a_min ** 2)
                        if score > best_score:
                            best_score = score
                            best_axis = axis
                            best_cut_val = (bins[i] + bins[i + 1]) / 2.0

    if best_axis is not None and best_cut_val is not None:
        return pts_cam[pts_cam @ best_axis >= best_cut_val]
    return pts_cam


# ===========================================================================
# 3. Test Cases & Reconciled Metrics
# ===========================================================================

def test_synthetic_generator_label_reconciliation():
    """
    P0 Verification:
      - Coordinate bounds match exact mathematical definition.
      - Point counts reconcile: N_total = N_head + N_neck + N_torso.
      - Poisson-disk sampling is explicitly surface sampling.
    """
    pcd_cam, pts_local, labels, R_pitch, meta = create_synthetic_labeled_head_torso(
        pitch_deg=0.0, slouch_deg=25.0, seed=42
    )

    N_h, N_n, N_t = meta["n_head"], meta["n_neck"], meta["n_torso"]
    assert N_h == 4500, f"Expected 4500 head points, got {N_h}"
    assert N_n == 1500, f"Expected 1500 neck points, got {N_n}"
    assert N_t == 4000, f"Expected 4000 torso points, got {N_t}"
    assert len(pts_local) == N_h + N_n + N_t == 10000

    pts_h = pts_local[labels == 0]
    pts_n = pts_local[labels == 1]
    pts_t = pts_local[labels == 2]

    # ------------------------------------------------------------------
    # 1. Analytical Mesh Geometry Bounds
    # ------------------------------------------------------------------
    meshes = meta["meshes"]
    vh = np.asarray(meshes["head"].vertices)
    vn = np.asarray(meshes["neck"].vertices)
    vt = np.asarray(meshes["torso"].vertices)

    # Head mesh analytical bounds (breadth 160 mm, height 220 mm, depth 235 mm with nose cone)
    assert np.isclose(np.ptp(vh[:, 0]) * 1000.0, 160.0, atol=1e-3)
    assert np.isclose(np.ptp(vh[:, 1]) * 1000.0, 220.0, atol=1e-3)
    assert np.isclose(np.min(vh[:, 1]), -0.110, atol=1e-3)
    assert np.isclose(np.max(vh[:, 1]), 0.110, atol=1e-3)
    assert np.isclose(np.ptp(vh[:, 2]) * 1000.0, 235.0, atol=1e-3)

    # Neck mesh analytical bounds (height 70 mm, aligned with subject Y axis)
    assert np.isclose(np.min(vn[:, 1]), -0.180, atol=1e-3)
    assert np.isclose(np.max(vn[:, 1]), -0.110, atol=1e-3)
    assert np.isclose(np.ptp(vn[:, 0]) * 1000.0, 90.0, atol=1e-3)
    assert np.isclose(np.ptp(vn[:, 2]) * 1000.0, 90.0, atol=1e-3)

    # Torso mesh analytical bounds (top face anchored at y_s = -0.180 m)
    assert np.isclose(np.max(vt[:, 1]), -0.180, atol=1e-3)
    assert np.min(vt[:, 1]) >= -0.500

    # ------------------------------------------------------------------
    # 2. Continuous Mesh-Surface Interface & Penetration Tolerance
    # ------------------------------------------------------------------
    # Interface tolerance: zero volumetric penetration beyond numerical epsilon (50 um)
    interface_tolerance_m = 5e-5

    # Head-to-neck boundary plane (y_s = -0.110 m):
    head_min_y = np.min(vh[:, 1])
    neck_max_y = np.max(vn[:, 1])
    assert abs(head_min_y - (-0.110)) <= interface_tolerance_m
    assert abs(neck_max_y - (-0.110)) <= interface_tolerance_m
    # Penetration check: neck must not penetrate cranium beyond declared tolerance
    assert (neck_max_y - head_min_y) <= interface_tolerance_m

    # Neck-to-torso boundary plane (y_s = -0.180 m):
    neck_min_y = np.min(vn[:, 1])
    torso_max_y = np.max(vt[:, 1])
    assert abs(neck_min_y - (-0.180)) <= interface_tolerance_m
    assert abs(torso_max_y - (-0.180)) <= interface_tolerance_m
    # Penetration check: torso must not penetrate neck beyond declared tolerance
    assert (torso_max_y - neck_min_y) <= interface_tolerance_m

    # ------------------------------------------------------------------
    # 3. Sampled-Point Discrete Surface Distance (Separate from Mesh Contact)
    # ------------------------------------------------------------------
    # Discrete Poisson-disk samples on 2D surfaces have non-zero spacing.
    # We verify discrete inter-sample distances and DBSCAN connectivity (eps = 25 mm).
    tree_n = cKDTree(pts_n)
    dists_hn, _ = tree_n.query(pts_h)
    dists_tn, _ = tree_n.query(pts_t)
    min_dist_hn_mm = np.min(dists_hn) * 1000.0
    min_dist_tn_mm = np.min(dists_tn) * 1000.0

    # Non-zero point spacing due to discrete sampling (distinct from continuous mesh contact)
    assert min_dist_hn_mm > 0.1
    assert min_dist_tn_mm > 0.1
    # Connectivity: must be within DBSCAN cluster_eps (25 mm)
    assert min_dist_hn_mm < 25.0, f"Head disconnected from neck (gap {min_dist_hn_mm:.1f} mm >= 25 mm)"
    assert min_dist_tn_mm < 25.0, f"Torso disconnected from neck (gap {min_dist_tn_mm:.1f} mm >= 25 mm)"


def test_segmentation_benchmark_full_reconciliation():
    """
    Execute benchmark across Methods A, B, C and pitch angles 0°, 10°, 20°, 30°.
    Assert exact point-count reconciliation:
      Surviving = RetHead + RetNeck + RetTorso.
    """
    target = HeadTarget()

    angles = [0.0, 10.0, 20.0, 30.0]
    for pitch in angles:
        pcd_cam, pts_local, labels, R_pitch, meta = create_synthetic_labeled_head_torso(
            pitch_deg=pitch, slouch_deg=25.0, seed=42
        )
        tree = cKDTree(pts_local)
        N_h, N_n, N_t = meta["n_head"], meta["n_neck"], meta["n_torso"]

        # ------------------------------------------------------------------
        # Method A: Production PCA Slicing (Current Source Code)
        # ------------------------------------------------------------------
        seg_pca = target.segment(pcd_cam)
        pts_pca_cam = np.asarray(seg_pca.points)
        pts_pca_local = (R_pitch.T @ pts_pca_cam.T).T

        dists, idx_pca = tree.query(pts_pca_local)
        assert np.all(dists < 1e-12), "PCA points altered coordinates!"
        assert len(np.unique(idx_pca)) == len(idx_pca), "Bijective point index mapping violated"
        surv_l_pca = labels[idx_pca]

        k0_pca = int(np.sum(surv_l_pca == 0))
        k1_pca = int(np.sum(surv_l_pca == 1))
        k2_pca = int(np.sum(surv_l_pca == 2))
        # P0 Identity Reconciliation
        assert len(pts_pca_cam) == k0_pca + k1_pca + k2_pca

        rec_pca = k0_pca / N_h
        rej_pca = (N_t - k2_pca) / N_t
        loss_pca = (N_h - k0_pca) / N_h
        assert np.isfinite(rec_pca) and 0.0 <= rec_pca <= 1.0
        assert np.isfinite(rej_pca) and 0.0 <= rej_pca <= 1.0
        assert np.isfinite(loss_pca) and 0.0 <= loss_pca <= 1.0
        assert np.isclose(loss_pca, 1.0 - rec_pca, atol=1e-12)
        # Stable regression target for Method A: Production PCA covariance tilts by ~37 deg
        # due to slouched torso; orientation heuristic flips v_long, cutting off 100% of the cranium.
        assert rec_pca <= 0.05, f"Method A unexpectedly isolated cranium (recall {rec_pca:.2f} > 0.05)"
        assert rej_pca >= 0.90, f"Method A failed torso rejection target (rej {rej_pca:.2f} < 0.90)"

        # ------------------------------------------------------------------
        # Method C: Baseline Camera-Y Slicing
        # ------------------------------------------------------------------
        pts_cam = np.asarray(pcd_cam.points)
        pts_y_cam = slice_camera_y_baseline(pts_cam)
        pts_y_local = (R_pitch.T @ pts_y_cam.T).T

        dists_y, idx_y = tree.query(pts_y_local)
        assert np.all(dists_y < 1e-12)
        assert len(np.unique(idx_y)) == len(idx_y)
        surv_l_y = labels[idx_y]

        k0_y = int(np.sum(surv_l_y == 0))
        k1_y = int(np.sum(surv_l_y == 1))
        k2_y = int(np.sum(surv_l_y == 2))
        assert len(pts_y_cam) == k0_y + k1_y + k2_y

        rec_y = k0_y / N_h
        rej_y = (N_t - k2_y) / N_t
        loss_y = (N_h - k0_y) / N_h
        assert np.isfinite(rec_y) and 0.0 <= rec_y <= 1.0
        assert np.isfinite(rej_y) and 0.0 <= rej_y <= 1.0
        assert np.isfinite(loss_y) and 0.0 <= loss_y <= 1.0
        assert np.isclose(loss_y, 1.0 - rec_y, atol=1e-12)
        # Experimental target for Method C: Upright 70 mm neck provides clearance for horizontal cut
        assert rec_y >= 0.99, f"Method C head recall target failed (recall {rec_y:.2f} < 0.99)"
        assert rej_y >= 0.99, f"Method C torso rejection target failed (rej {rej_y:.2f} < 0.99)"

        # ------------------------------------------------------------------
        # Method B: Multi-Axis Prototype (25 deg search cone)
        # ------------------------------------------------------------------
        pts_m_cam = multi_axis_constriction_search_prototype(pts_cam, max_cone_deg=25.0)
        pts_m_local = (R_pitch.T @ pts_m_cam.T).T

        dists_m, idx_m = tree.query(pts_m_local)
        assert np.all(dists_m < 1e-12)
        assert len(np.unique(idx_m)) == len(idx_m)
        surv_l_m = labels[idx_m]

        k0_m = int(np.sum(surv_l_m == 0))
        k1_m = int(np.sum(surv_l_m == 1))
        k2_m = int(np.sum(surv_l_m == 2))
        assert len(pts_m_cam) == k0_m + k1_m + k2_m

        rec_m = k0_m / N_h
        rej_m = (N_t - k2_m) / N_t
        loss_m = (N_h - k0_m) / N_h
        assert np.isfinite(rec_m) and 0.0 <= rec_m <= 1.0
        assert np.isfinite(rej_m) and 0.0 <= rej_m <= 1.0
        assert np.isfinite(loss_m) and 0.0 <= loss_m <= 1.0
        assert np.isclose(loss_m, 1.0 - rec_m, atol=1e-12)
        # Experimental target for Method B: Conical constriction search locates neck constriction plane
        assert rec_m >= 0.99, f"Method B head recall target failed (recall {rec_m:.2f} < 0.99)"
        assert rej_m >= 0.99, f"Method B torso rejection target failed (rej {rej_m:.2f} < 0.99)"

        # Print reconciled counts for audit
        print(f"\n[Pitch {pitch:4.1f}° Evaluation]")
        print(f"  Method A (Production PCA): Surv={len(pts_pca_cam):5d} | RetHead={k0_pca:5d} ({rec_pca*100:5.1f}%) | RetTorso={k2_pca:5d} | TorsoRej={rej_pca*100:5.1f}% | HeadLoss={loss_pca*100:5.1f}%")
        print(f"  Method C (Camera-Y):       Surv={len(pts_y_cam):5d} | RetHead={k0_y:5d} ({rec_y*100:5.1f}%) | RetTorso={k2_y:5d} | TorsoRej={rej_y*100:5.1f}% | HeadLoss={loss_y*100:5.1f}%")
        print(f"  Method B (Multi-Axis 25°): Surv={len(pts_m_cam):5d} | RetHead={k0_m:5d} ({rec_m*100:5.1f}%) | RetTorso={k2_m:5d} | TorsoRej={rej_m*100:5.1f}% | HeadLoss={loss_m*100:5.1f}%")


def test_measurement_validator_complete_synthetic_head():
    """
    P2 Evaluation: Test production HeadTarget().measure() on cleanly segmented complete head.
    Documents exact geometric ground truth vs trimmed percentile caliper measurement.
    """
    pcd_cam, pts_local, labels, R_pitch, meta = create_synthetic_labeled_head_torso(
        pitch_deg=0.0, slouch_deg=25.0, seed=42
    )
    # Extract only head points (Label 0)
    pts_head = pts_local[labels == 0]
    pcd_head = o3d.geometry.PointCloud()
    pcd_head.points = o3d.utility.Vector3dVector(pts_head)

    target = HeadTarget()
    result = target.measure(pcd_head)

    assert result["status"] == "validated", f"Expected 'validated', got {result['status']}"
    assert result["oriented_bbox"]["volume_m3"] is not None
    assert result["symmetry_residual_mm"] <= 12.0, f"Expected symmetry <= 12.0 mm, got {result['symmetry_residual_mm']}"
    assert result["diagnostics"] == [], f"Expected zero diagnostics on complete head, got {result['diagnostics']}"

    dims = result["cranial_envelope_3d"]
    # Breadth: nominal 160 mm, measured ~157.9 mm (error ~2.1 mm)
    assert np.isclose(dims["breadth_mm"], 160.0, atol=4.0)
    # Height: nominal 220 mm, measured ~216.4 mm (error ~3.6 mm)
    assert np.isclose(dims["height_mm"], 220.0, atol=5.0)
    # Length: robust 0.5% trimmed caliper measures ~215.7 mm (between 200 mm ellipsoid and 235 mm nose apex)
    assert 205.0 <= dims["length_mm"] <= 225.0


def test_measurement_validator_partial_synthetic_head():
    """
    P2 Evaluation: Truncate posterior cranial support and verify production validator
    strictly marks measurement 'unverified' with volume_m3 = None.
    """
    pcd_cam, pts_local, labels, R_pitch, meta = create_synthetic_labeled_head_torso(
        pitch_deg=0.0, slouch_deg=25.0, truncate_posterior=True, seed=42
    )
    pts_head = pts_local[labels == 0]
    pcd_head = o3d.geometry.PointCloud()
    pcd_head.points = o3d.utility.Vector3dVector(pts_head)

    target = HeadTarget()
    result = target.measure(pcd_head)

    assert result["status"] == "unverified", f"Expected 'unverified', got {result['status']}"
    assert result["oriented_bbox"]["volume_m3"] is None

    # Verify exact failure diagnostic and threshold from active production targets/head.py:274-276
    assert result["symmetry_residual_mm"] > 12.0, f"Expected symmetry > 12.0 mm, got {result['symmetry_residual_mm']}"
    diags = result.get("diagnostics", [])
    assert any("symmetry residual" in d.lower() and "12.0" in d for d in diags), f"Missing symmetry diagnostic: {diags}"
    assert any("anterior-posterior" in d.lower() or "facial relief" in d.lower() for d in diags), f"Missing A-P relief diagnostic: {diags}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
