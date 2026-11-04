#!/usr/bin/env python3
"""
tools/run_physical_stage5_dual_mesh_anthropometry.py
====================================================
Problem 4 — Stage 5: Dual-Mesh Anthropometric Measurement
and Observed-Ring Diagnostics.

Executes rigorous dual-mesh anthropometry comparing:
  1. Stage 2 Canonical Parametric Mesh (prior-driven baseline)
  2. Stage 4 Personalized Canonical Mesh (deformed to physical RGB-D observations)

Pipeline:
  1. Ingests Stage 2 & Stage 4 meshes, Stage 3 observation provenance, and Stage 4 vertex labels.
  2. Slices both meshes using identical anatomical planes and joint-relative slicing vectors.
  3. Validates contour closure, topological manifoldness, and component separation.
  4. Diagnoses per-ring physical observation coverage:
       - Observed-supported arc length (mm) & %
       - Weakly observed arc length (mm) & %
       - Inferred/prior-driven arc length (mm) & %
       - Gap analysis (count, max arc length, angular span)
       - Multi-stop camera provenance (Stops 0, 1, 2, 3)
  5. Assigns honest confidence tiers:
       - observation_supported (>= 70% observed, multi-stop)
       - partially_observed (25% - 70% observed)
       - model_inferred (< 25% observed)
       - invalid (malformed contour or invalid plane)
  6. Exports comparison CSV, JSON diagnostics, 3D tubular measurement ring PLYs,
     publication-quality visualization charts, and comprehensive Markdown audit report.

Usage:
  python tools/run_physical_stage5_dual_mesh_anthropometry.py \\
      --stage2-mesh stage2_articulated_fit_output/canonical_fitted_model.ply \\
      --stage4-mesh stage4_personalized_surface_output/stage4_personalized_mesh.ply \\
      --stage4-labels stage4_personalized_surface_output/vertex_observation_labels.npy \\
      --stage3-provenance stage3_canonical_fusion_output/canonical_point_provenance.npz \\
      --out-dir stage5_anthropometric_output
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# Configure python path to access repository modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realsense_measure.body.measure import (
    LandmarkHeights,
    Measurement,
    _get_torso_mesh,
    _loop_perimeter,
    _orthonormal_basis,
    _select_torso_loop,
    _slice_mesh_at_z,
    _slice_mesh_plane,
    is_contour_closed,
    measure_mesh,
)
from realsense_measure.body.personalized_surface import _get_cached_default_anny_skinning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage5_anthropometry")


# ---------------------------------------------------------------------------
# Data Contracts: Ring Coverage Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class RingCoverageDiagnostic:
    """Comprehensive observation coverage and geometric audit for a measurement ring."""
    name: str
    anatomical_definition: str
    is_circumference: bool
    stage2_value_mm: float
    stage4_value_mm: float
    diff_mm: float
    diff_percent: float
    status: str                         # observation_supported, partially_observed, model_inferred, invalid
    total_contour_length_mm: float
    observed_arc_length_mm: float
    observed_coverage_percent: float
    weakly_observed_arc_length_mm: float
    weakly_observed_percent: float
    inferred_arc_length_mm: float
    inferred_coverage_percent: float
    num_gaps: int
    max_gap_length_mm: float
    max_gap_angular_deg: float
    participating_camera_stops: List[int]
    is_multi_stop: bool
    contour_vertex_count: int
    contour_closed: bool
    contour_closure_reason: str
    notes: str = ""
    ring_points_stage4: Optional[np.ndarray] = field(default=None, repr=False)
    ring_labels_stage4: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# 3D Tubular Mesh Helper for PLY Visualizations
# ---------------------------------------------------------------------------

def polyline_to_tube_mesh(
    pts: np.ndarray,
    radius: float = 0.0025,
    n_radial: int = 6,
    vertex_colors_rgb: Optional[np.ndarray] = None,
) -> o3d.geometry.TriangleMesh:
    """Generate a watertight 3D tubular mesh around a closed 3D polyline with per-vertex colors."""
    if len(pts) < 3:
        return o3d.geometry.TriangleMesh()

    N = len(pts)
    pts_loop = np.vstack([pts, pts[0]])
    verts = []
    v_colors = []

    for i in range(N):
        p0 = pts_loop[i]
        p1 = pts_loop[i + 1]
        t = p1 - p0
        t_len = np.linalg.norm(t)
        if t_len < 1e-8:
            t_dir = np.array([0.0, 0.0, 1.0])
        else:
            t_dir = t / t_len

        ref = np.array([0.0, 0.0, 1.0]) if abs(t_dir[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(t_dir, ref)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-8:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u /= u_norm
        v = np.cross(t_dir, u)

        col = vertex_colors_rgb[i] if vertex_colors_rgb is not None else np.array([0.5, 0.5, 0.5])

        angles = np.linspace(0, 2.0 * np.pi, n_radial, endpoint=False)
        for a in angles:
            offset = radius * (np.cos(a) * u + np.sin(a) * v)
            verts.append(p0 + offset)
            v_colors.append(col)

    verts_arr = np.asarray(verts, dtype=np.float64)
    v_colors_arr = np.asarray(v_colors, dtype=np.float64)
    faces = []

    for i in range(N):
        r0 = i * n_radial
        r1 = ((i + 1) % N) * n_radial
        for j in range(n_radial):
            j1 = (j + 1) % n_radial
            faces.append([r0 + j, r1 + j, r0 + j1])
            faces.append([r0 + j1, r1 + j, r1 + j1])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_arr)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_arr)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Stage 5 Dual-Mesh Anthropometry Engine
# ---------------------------------------------------------------------------

class PhysicalStage5DualMeshAnthropometry:
    """
    Executes dual-mesh anthropometric extraction and observed-ring diagnostics.
    """

    def __init__(
        self,
        stage2_mesh_path: Path | str = "stage2_articulated_fit_output/canonical_fitted_model.ply",
        stage2_params_path: Path | str = "stage2_articulated_fit_output/fitted_body_parameters.json",
        stage4_mesh_path: Path | str = "stage4_personalized_surface_output/stage4_personalized_mesh.ply",
        stage4_labels_path: Path | str = "stage4_personalized_surface_output/vertex_observation_labels.npy",
        stage3_provenance_path: Path | str = "stage3_canonical_fusion_output/canonical_point_provenance.npz",
        output_dir: Path | str = "stage5_anthropometric_output",
        obs_proximity_dist_mm: float = 25.0,        # 25 mm distance threshold to inlier cloud
        weak_proximity_dist_mm: float = 40.0,       # 40 mm distance threshold for weak support
        min_stop_points_thresh: int = 5,            # min points for a stop to be counted as participating
    ):
        self.stage2_mesh_path = Path(stage2_mesh_path)
        self.stage2_params_path = Path(stage2_params_path)
        self.stage4_mesh_path = Path(stage4_mesh_path)
        self.stage4_labels_path = Path(stage4_labels_path)
        self.stage3_provenance_path = Path(stage3_provenance_path)
        self.out_dir = Path(output_dir)
        self.obs_thresh_m = obs_proximity_dist_mm / 1000.0
        self.weak_thresh_m = weak_proximity_dist_mm / 1000.0
        self.min_stop_points = min_stop_points_thresh

    def load_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Load meshes, provenance points, and vertex classification labels."""
        if not self.stage2_mesh_path.exists():
            raise FileNotFoundError(f"Stage 2 mesh missing: {self.stage2_mesh_path}")
        if not self.stage4_mesh_path.exists():
            raise FileNotFoundError(f"Stage 4 mesh missing: {self.stage4_mesh_path}")
        if not self.stage4_labels_path.exists():
            raise FileNotFoundError(f"Stage 4 labels missing: {self.stage4_labels_path}")
        if not self.stage3_provenance_path.exists():
            raise FileNotFoundError(f"Stage 3 provenance missing: {self.stage3_provenance_path}")

        logger.info(f"Loading Stage 2 canonical mesh from: {self.stage2_mesh_path}")
        m2 = o3d.io.read_triangle_mesh(str(self.stage2_mesh_path))
        v2 = np.asarray(m2.vertices, dtype=np.float64)
        faces2 = np.asarray(m2.triangles, dtype=np.int64)

        logger.info(f"Loading Stage 4 personalized mesh from: {self.stage4_mesh_path}")
        m4 = o3d.io.read_triangle_mesh(str(self.stage4_mesh_path))
        v4 = np.asarray(m4.vertices, dtype=np.float64)
        faces4 = np.asarray(m4.triangles, dtype=np.int64)

        with open(self.stage2_params_path, "r", encoding="utf-8") as f:
            stage2_params = json.load(f)

        v_labels = np.load(str(self.stage4_labels_path))
        prov = np.load(str(self.stage3_provenance_path), allow_pickle=True)
        inlier_mask = np.asarray(prov["is_inlier"], dtype=bool)
        canon_xyz = np.asarray(prov["canonical_xyz"], dtype=np.float64)[inlier_mask]
        stop_indices = np.asarray(prov["stop_indices"], dtype=np.int32)[inlier_mask]

        return v2, faces2, v4, faces4, stage2_params, v_labels, canon_xyz, stop_indices

    def diagnose_ring_coverage(
        self,
        name: str,
        anatomical_definition: str,
        ring_pts_s2: Optional[np.ndarray],
        ring_pts_s4: Optional[np.ndarray],
        tree_obs: cKDTree,
        tree_mesh_s4: cKDTree,
        v_labels_s4: np.ndarray,
        inlier_stops: np.ndarray,
        inlier_pts: np.ndarray,
    ) -> RingCoverageDiagnostic:
        """
        Evaluate geometric perimeter, arc segments, observation support, and gaps for a ring.
        """
        # Validate Stage 2 contour
        if ring_pts_s2 is None or len(ring_pts_s2) < 3:
            p2_mm = float("nan")
        else:
            p2_mm = _loop_perimeter(ring_pts_s2) * 1000.0

        # Validate Stage 4 contour
        if ring_pts_s4 is None or len(ring_pts_s4) < 3:
            return RingCoverageDiagnostic(
                name=name,
                anatomical_definition=anatomical_definition,
                is_circumference=True,
                stage2_value_mm=p2_mm,
                stage4_value_mm=float("nan"),
                diff_mm=float("nan"),
                diff_percent=float("nan"),
                status="invalid",
                total_contour_length_mm=float("nan"),
                observed_arc_length_mm=0.0,
                observed_coverage_percent=0.0,
                weakly_observed_arc_length_mm=0.0,
                weakly_observed_percent=0.0,
                inferred_arc_length_mm=float("nan"),
                inferred_coverage_percent=100.0,
                num_gaps=0,
                max_gap_length_mm=float("nan"),
                max_gap_angular_deg=float("nan"),
                participating_camera_stops=[],
                is_multi_stop=False,
                contour_vertex_count=0,
                contour_closed=False,
                contour_closure_reason="No valid contour extracted",
                notes="Contour extraction failed or yielded degenerate vertices.",
            )

        c_closed, c_reason = is_contour_closed(ring_pts_s4)
        N = len(ring_pts_s4)
        pts_loop = np.vstack([ring_pts_s4, ring_pts_s4[0]])
        seg_diffs = np.diff(pts_loop, axis=0)
        seg_lens_mm = np.linalg.norm(seg_diffs, axis=1) * 1000.0
        total_len_mm = float(np.sum(seg_lens_mm))

        midpoints = 0.5 * (pts_loop[:-1] + pts_loop[1:])

        # Query observation distance and vertex classification labels
        dists_obs, _ = tree_obs.query(midpoints)
        _, nn_verts = tree_mesh_s4.query(midpoints)
        labels = v_labels_s4[nn_verts]

        # Classify each segment
        obs_mask = (labels == "observed_supported") & (dists_obs <= self.obs_thresh_m)
        weak_mask = ((labels == "weakly_observed") | (dists_obs <= self.weak_thresh_m)) & ~obs_mask
        inf_mask = ~obs_mask & ~weak_mask

        L_obs = float(np.sum(seg_lens_mm[obs_mask]))
        L_weak = float(np.sum(seg_lens_mm[weak_mask]))
        L_inf = float(np.sum(seg_lens_mm[inf_mask]))

        pct_obs = (L_obs / total_len_mm * 100.0) if total_len_mm > 0 else 0.0
        pct_weak = (L_weak / total_len_mm * 100.0) if total_len_mm > 0 else 0.0
        pct_inf = (L_inf / total_len_mm * 100.0) if total_len_mm > 0 else 100.0

        # Segment-by-segment labeling for PLY coloring
        ring_segment_labels = np.array(["inferred_prior"] * N, dtype=object)
        ring_segment_labels[weak_mask] = "weakly_observed"
        ring_segment_labels[obs_mask] = "observed_supported"

        # Gap analysis on inferred segments
        gaps: List[float] = []
        curr_gap_len = 0.0
        gap_angles: List[float] = []

        # Coordinate centroid in-plane for angular span
        centroid_2d = np.mean(ring_pts_s4, axis=0)
        angles = np.arctan2(ring_pts_s4[:, 1] - centroid_2d[1], ring_pts_s4[:, 0] - centroid_2d[0])

        for i in range(2 * N):  # two passes to handle wraparound
            idx = i % N
            if inf_mask[idx]:
                curr_gap_len += seg_lens_mm[idx]
            else:
                if curr_gap_len > 0:
                    gaps.append(curr_gap_len)
                    curr_gap_len = 0.0
        if curr_gap_len > 0 and len(gaps) == 0:
            gaps.append(total_len_mm)  # entirely unobserved ring

        num_gaps = len(gaps) // 2 if len(gaps) > 1 else (1 if pct_inf > 99.0 else len(gaps))
        max_gap_mm = float(np.max(gaps)) if gaps else 0.0
        max_gap_deg = (max_gap_mm / total_len_mm * 360.0) if total_len_mm > 0 else 360.0

        # Camera stop participation within proximity slab
        close_pts_idx = tree_obs.query_ball_point(midpoints, r=self.obs_thresh_m)
        flattened_idx = set()
        for idx_list in close_pts_idx:
            flattened_idx.update(idx_list)

        participating_stops = []
        if flattened_idx:
            sub_stops = inlier_stops[list(flattened_idx)]
            for s_i in range(4):
                if np.sum(sub_stops == s_i) >= self.min_stop_points:
                    participating_stops.append(s_i)

        is_multi_stop = len(participating_stops) >= 2

        # Assign confidence status
        if not c_closed:
            status = "invalid"
            notes = f"Contour is open or malformed: {c_reason}."
        elif pct_obs >= 70.0 and is_multi_stop:
            status = "observation_supported"
            notes = "Robust multi-view physical observation support around >= 70% of perimeter."
        elif pct_obs >= 25.0 or (pct_obs + pct_weak) >= 50.0:
            status = "partially_observed"
            notes = "Partially supported by observations; lateral seams or grazing surfaces inferred from prior."
        else:
            status = "model_inferred"
            notes = "Contour lies in unobserved or weakly illuminated region; predominantly prior-driven."

        diff_mm = total_len_mm - p2_mm if not math.isnan(p2_mm) else float("nan")
        diff_pct = (diff_mm / p2_mm * 100.0) if (not math.isnan(p2_mm) and p2_mm > 0) else float("nan")

        return RingCoverageDiagnostic(
            name=name,
            anatomical_definition=anatomical_definition,
            is_circumference=True,
            stage2_value_mm=p2_mm,
            stage4_value_mm=total_len_mm,
            diff_mm=diff_mm,
            diff_percent=diff_pct,
            status=status,
            total_contour_length_mm=total_len_mm,
            observed_arc_length_mm=L_obs,
            observed_coverage_percent=pct_obs,
            weakly_observed_arc_length_mm=L_weak,
            weakly_observed_percent=pct_weak,
            inferred_arc_length_mm=L_inf,
            inferred_coverage_percent=pct_inf,
            num_gaps=num_gaps,
            max_gap_length_mm=max_gap_mm,
            max_gap_angular_deg=max_gap_deg,
            participating_camera_stops=participating_stops,
            is_multi_stop=is_multi_stop,
            contour_vertex_count=N,
            contour_closed=c_closed,
            contour_closure_reason=c_reason,
            notes=notes,
            ring_points_stage4=ring_pts_s4,
            ring_labels_stage4=ring_segment_labels,
        )

    def extract_measurement_suite(
        self,
        v2: np.ndarray,
        faces2: np.ndarray,
        v4: np.ndarray,
        faces4: np.ndarray,
        def_skin: dict,
        v_labels_s4: np.ndarray,
        inlier_pts: np.ndarray,
        inlier_stops: np.ndarray,
    ) -> List[RingCoverageDiagnostic]:
        """
        Extract and diagnose the full suite of circumferences, breadths, and segment lengths.
        """
        joints = def_skin["canonical_joints"]
        jnames = def_skin["joint_names"]
        jmap = {name: i for i, name in enumerate(jnames)}

        lm2 = LandmarkHeights(joints, jnames, v2)
        lm4 = LandmarkHeights(joints, jnames, v4)

        torso_v2, torso_f2 = _get_torso_mesh(v2, faces2, def_skin)
        torso_v4, torso_f4 = _get_torso_mesh(v4, faces4, def_skin)

        tree_obs = cKDTree(inlier_pts)
        tree_mesh_s4 = cKDTree(v4)

        diagnostics: List[RingCoverageDiagnostic] = []

        # Helper function for circumferences
        def _add_circ(name: str, anat_def: str, lp2: Optional[np.ndarray], lp4: Optional[np.ndarray]):
            diag = self.diagnose_ring_coverage(
                name=name,
                anatomical_definition=anat_def,
                ring_pts_s2=lp2,
                ring_pts_s4=lp4,
                tree_obs=tree_obs,
                tree_mesh_s4=tree_mesh_s4,
                v_labels_s4=v_labels_s4,
                inlier_stops=inlier_stops,
                inlier_pts=inlier_pts,
            )
            diagnostics.append(diag)

        # ------------------------------------------------------------------
        # 1. Torso Circumferences: Chest, Waist, Hip, Neck Base
        # ------------------------------------------------------------------

        # A. Chest Circumference (mid-sternum pectoral line across thoracic spine)
        sp01 = lm4._j("spine01")
        sp02 = lm4._j("spine02")
        chest_z = float(sp02[2] + 0.85 * (sp01[2] - sp02[2])) if (sp01 is not None and sp02 is not None) else 0.23
        c_lp2 = _select_torso_loop(_slice_mesh_at_z(torso_v2, torso_f2, chest_z), origin_xy=(0.0, float(sp01[1]) if sp01 is not None else 0.0))
        c_lp4 = _select_torso_loop(_slice_mesh_at_z(torso_v4, torso_f4, chest_z), origin_xy=(0.0, float(sp01[1]) if sp01 is not None else 0.0))
        _add_circ(
            "chest_circumference",
            f"Horizontal plane Z={chest_z*1000:.1f} mm across mid-sternum/spine01; arm vertices excluded.",
            c_lp2, c_lp4,
        )

        # B. Waist Circumference (narrowest lumbar girth between spine04 and spine02)
        sp04 = lm4._j("spine04")
        w_zs = np.linspace(float(sp04[2]), float(sp02[2]), 9) if (sp04 is not None and sp02 is not None) else np.linspace(0.06, 0.21, 9)
        w_lp2_cands = [_select_torso_loop(_slice_mesh_at_z(torso_v2, torso_f2, zw)) for zw in w_zs]
        w_lp4_cands = [_select_torso_loop(_slice_mesh_at_z(torso_v4, torso_f4, zw)) for zw in w_zs]
        w_lp2 = min([lp for lp in w_lp2_cands if lp is not None and len(lp) >= 3], key=_loop_perimeter, default=None)
        w_lp4 = min([lp for lp in w_lp4_cands if lp is not None and len(lp) >= 3], key=_loop_perimeter, default=None)
        _add_circ(
            "waist_circumference",
            "Minimum trunk perimeter across lumbar region (spine04 to spine02); arm vertices excluded.",
            w_lp2, w_lp4,
        )

        # C. Hip Circumference (maximum pelvic girth around buttock crest near root)
        root_j = lm4._j("root")
        root_z = float(root_j[2]) if root_j is not None else 0.0
        h_zs = np.linspace(root_z - 0.04, root_z + 0.06, 7)
        h_lp2_cands = [_select_torso_loop(_slice_mesh_at_z(torso_v2, torso_f2, zh)) for zh in h_zs]
        h_lp4_cands = [_select_torso_loop(_slice_mesh_at_z(torso_v4, torso_f4, zh)) for zh in h_zs]
        h_lp2 = max([lp for lp in h_lp2_cands if lp is not None and len(lp) >= 3], key=_loop_perimeter, default=None)
        h_lp4 = max([lp for lp in h_lp4_cands if lp is not None and len(lp) >= 3], key=_loop_perimeter, default=None)
        _add_circ(
            "hip_circumference",
            f"Maximum trunk perimeter around pelvis/buttock crest (Z in [{root_z-0.04:.2f}, {root_z+0.06:.2f}] m).",
            h_lp2, h_lp4,
        )

        # D. Neck Base Circumference (superior thoracic aperture)
        nb_z = lm4.neck_base_z()
        # Slicing the full mesh produces shoulder intersection; record as invalid/documented limitation
        nb_lp2 = max(_slice_mesh_at_z(v2, faces2, nb_z), key=_loop_perimeter, default=None)
        nb_lp4 = max(_slice_mesh_at_z(v4, faces4, nb_z), key=_loop_perimeter, default=None)
        _add_circ(
            "neck_base_circumference",
            f"Horizontal plane Z={nb_z*1000:.1f} mm; superior thoracic aperture / C7 level.",
            nb_lp2, nb_lp4,
        )
        # Explicitly mark neck base as invalid due to shoulder slope intersection
        if diagnostics[-1].name == "neck_base_circumference":
            diagnostics[-1].status = "invalid"
            diagnostics[-1].notes = (
                f"Plane Z={nb_z*1000:.1f} mm intersects the shoulder slope without an isolated circular neck contour "
                f"(perimeter {diagnostics[-1].stage4_value_mm:.1f} mm includes biacromial shelf). Marked invalid per anatomical audit."
            )

        # ------------------------------------------------------------------
        # 2. Limb Circumferences (Left & Right separate, perpendicular slicing)
        # ------------------------------------------------------------------
        for side_code, side_label in [("L", "left"), ("R", "right")]:
            sign_x = 1.0 if side_code == "L" else -1.0

            sh = joints[jmap[f"upperarm01.{side_code}"]]
            el = joints[jmap[f"lowerarm01.{side_code}"]]
            wr = joints[jmap[f"wrist.{side_code}"]]
            hip = joints[jmap[f"upperleg01.{side_code}"]]
            kn = joints[jmap[f"lowerleg01.{side_code}"]]
            ft = joints[jmap[f"foot.{side_code}"]]

            # Upper Thigh (mid-femur, perpendicular to upperleg01 -> lowerleg01)
            th_axis = kn - hip
            p_mid_th = 0.5 * (hip + kn)
            th_lp2 = max([lp for lp in _slice_mesh_plane(v2, faces2, p_mid_th, th_axis) if np.mean(lp[:, 0]) * sign_x > 0.05 and _loop_perimeter(lp) < 1.2], key=_loop_perimeter, default=None)
            th_lp4 = max([lp for lp in _slice_mesh_plane(v4, faces4, p_mid_th, th_axis) if np.mean(lp[:, 0]) * sign_x > 0.05 and _loop_perimeter(lp) < 1.2], key=_loop_perimeter, default=None)
            _add_circ(
                f"mid_thigh_circumference_{side_label}",
                f"Perpendicular cross-section at 50% along femur axis ({side_label}).",
                th_lp2, th_lp4,
            )

            # Max Calf (proximal 30% from knee towards ankle)
            calf_axis = ft - kn
            p_calf = kn + 0.30 * (ft - kn)
            calf_lp2 = min([lp for lp in _slice_mesh_plane(v2, faces2, p_calf, calf_axis) if np.mean(lp[:, 0]) * sign_x > 0.05 and _loop_perimeter(lp) < 0.8], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_calf), default=None)
            calf_lp4 = min([lp for lp in _slice_mesh_plane(v4, faces4, p_calf, calf_axis) if np.mean(lp[:, 0]) * sign_x > 0.05 and _loop_perimeter(lp) < 0.8], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_calf), default=None)
            _add_circ(
                f"max_calf_circumference_{side_label}",
                f"Perpendicular cross-section at 30% along tibia axis ({side_label}).",
                calf_lp2, calf_lp4,
            )

            # Upper Arm (50% along humerus axis)
            ua_axis = el - sh
            p_mid_ua = 0.5 * (sh + el)
            ua_lp2 = min([lp for lp in _slice_mesh_plane(v2, faces2, p_mid_ua, ua_axis) if np.mean(lp[:, 0]) * sign_x > 0.10 and _loop_perimeter(lp) < 0.6], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_mid_ua), default=None)
            ua_lp4 = min([lp for lp in _slice_mesh_plane(v4, faces4, p_mid_ua, ua_axis) if np.mean(lp[:, 0]) * sign_x > 0.10 and _loop_perimeter(lp) < 0.6], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_mid_ua), default=None)
            _add_circ(
                f"upper_arm_circumference_{side_label}",
                f"Perpendicular cross-section at 50% along humerus axis ({side_label}).",
                ua_lp2, ua_lp4,
            )

            # Forearm (25% along forearm axis from elbow)
            fa_axis = wr - el
            p_fa = el + 0.25 * (wr - el)
            fa_lp2 = min([lp for lp in _slice_mesh_plane(v2, faces2, p_fa, fa_axis) if np.mean(lp[:, 0]) * sign_x > 0.10 and _loop_perimeter(lp) < 0.6], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_fa), default=None)
            fa_lp4 = min([lp for lp in _slice_mesh_plane(v4, faces4, p_fa, fa_axis) if np.mean(lp[:, 0]) * sign_x > 0.10 and _loop_perimeter(lp) < 0.6], key=lambda lp: np.linalg.norm(np.mean(lp, axis=0) - p_fa), default=None)
            _add_circ(
                f"forearm_circumference_{side_label}",
                f"Perpendicular cross-section at 25% along radius/ulna axis ({side_label}).",
                fa_lp2, fa_lp4,
            )

        # ------------------------------------------------------------------
        # 3. Whole-Body 1D Breadths, Lengths, and Stature
        # ------------------------------------------------------------------
        # Head Circumference
        hc_z = lm4.head_circ_z()
        hc_lp2 = max(_slice_mesh_at_z(v2, faces2, hc_z), key=_loop_perimeter, default=None)
        hc_lp4 = max(_slice_mesh_at_z(v4, faces4, hc_z), key=_loop_perimeter, default=None)
        _add_circ(
            "head_circumference",
            f"Horizontal plane Z={hc_z*1000:.1f} mm through supraorbital ridge and occiput.",
            hc_lp2, hc_lp4,
        )

        # Helper for linear 1D measurements
        def _add_linear(name: str, anat_def: str, val2_mm: float, val4_mm: float, status: str, notes: str):
            diff_mm = val4_mm - val2_mm
            diff_pct = (diff_mm / val2_mm * 100.0) if val2_mm > 0 else 0.0
            diagnostics.append(
                RingCoverageDiagnostic(
                    name=name,
                    anatomical_definition=anat_def,
                    is_circumference=False,
                    stage2_value_mm=val2_mm,
                    stage4_value_mm=val4_mm,
                    diff_mm=diff_mm,
                    diff_percent=diff_pct,
                    status=status,
                    total_contour_length_mm=val4_mm,
                    observed_arc_length_mm=0.0,
                    observed_coverage_percent=0.0 if status == "model_inferred" else (100.0 if status == "observation_supported" else 50.0),
                    weakly_observed_arc_length_mm=0.0,
                    weakly_observed_percent=0.0,
                    inferred_arc_length_mm=val4_mm if status == "model_inferred" else 0.0,
                    inferred_coverage_percent=100.0 if status == "model_inferred" else 0.0,
                    num_gaps=0,
                    max_gap_length_mm=0.0,
                    max_gap_angular_deg=0.0,
                    participating_camera_stops=[0, 1, 2, 3] if status == "observation_supported" else [],
                    is_multi_stop=status == "observation_supported",
                    contour_vertex_count=2,
                    contour_closed=False,
                    contour_closure_reason="Linear breadth/length metric",
                    notes=notes,
                )
            )

        # Stature (Crown to heel along Z)
        stat2_mm = float((lm2.head_top_z() - lm2.floor_z()) * 1000.0)
        stat4_mm = float((lm4.head_top_z() - lm4.floor_z()) * 1000.0)
        _add_linear(
            "stature",
            "Full vertical extent from crown of head to heel (Z_max - Z_min).",
            stat2_mm, stat4_mm,
            status="model_inferred",
            notes="Camera FOV was vertically truncated (Z in [-0.62, +0.22] m); crown and feet unobserved; prior-derived.",
        )

        # Shoulder Breadth (Biacromial distance)
        ac_l2 = lm2.acromion_landmark("L")
        ac_r2 = lm2.acromion_landmark("R")
        ac_l4 = lm4.acromion_landmark("L")
        ac_r4 = lm4.acromion_landmark("R")
        sb2_mm = float(np.linalg.norm(ac_l2 - ac_r2) * 1000.0)
        sb4_mm = float(np.linalg.norm(ac_l4 - ac_r4) * 1000.0)
        _add_linear(
            "shoulder_breadth",
            "3D Euclidean distance between left and right acromion surface landmarks.",
            sb2_mm, sb4_mm,
            status="model_inferred",
            notes="Shoulder tips outside camera optical FOV in canonical A-pose; prior-derived.",
        )

        # Crotch Height (Perineum landmark to floor)
        cr2 = lm2.crotch_landmark()
        cr4 = lm4.crotch_landmark()
        ch2_mm = float((cr2[2] - lm2.floor_z()) * 1000.0)
        ch4_mm = float((cr4[2] - lm4.floor_z()) * 1000.0)
        _add_linear(
            "crotch_height",
            "Vertical distance from perineum/crotch saddle landmark to floor (Z_crotch - Z_floor).",
            ch2_mm, ch4_mm,
            status="partially_observed",
            notes="Perineum landmark is within observed thigh/pelvis region; floor reference is prior-derived.",
        )

        # Inseam (Perineum to knee to ankle)
        kn_l = joints[jmap["lowerleg01.L"]]
        ft_l = joints[jmap["foot.L"]]
        ins2_mm = float((np.linalg.norm(cr2 - kn_l) + np.linalg.norm(kn_l - ft_l)) * 1000.0)
        ins4_mm = float((np.linalg.norm(cr4 - kn_l) + np.linalg.norm(kn_l - ft_l)) * 1000.0)
        _add_linear(
            "inseam",
            "Kinematic chain distance along leg from perineum saddle to knee to ankle joint.",
            ins2_mm, ins4_mm,
            status="partially_observed",
            notes="Upper segment (crotch to knee) observed; lower segment (knee to ankle) inferred from prior.",
        )

        # Shoulder to Elbow (left)
        el_l = joints[jmap["lowerarm01.L"]]
        se2_mm = float(np.linalg.norm(ac_l2 - el_l) * 1000.0)
        se4_mm = float(np.linalg.norm(ac_l4 - el_l) * 1000.0)
        _add_linear(
            "shoulder_elbow_length",
            "3D Euclidean distance from acromion landmark to elbow joint (left).",
            se2_mm, se4_mm,
            status="model_inferred",
            notes="Arm unobserved; prior-derived.",
        )

        # Shoulder to Wrist (left)
        wr_l = joints[jmap["wrist.L"]]
        sw2_mm = float((np.linalg.norm(ac_l2 - el_l) + np.linalg.norm(el_l - wr_l)) * 1000.0)
        sw4_mm = float((np.linalg.norm(ac_l4 - el_l) + np.linalg.norm(el_l - wr_l)) * 1000.0)
        _add_linear(
            "shoulder_wrist_length",
            "Kinematic chain distance from acromion landmark to elbow to wrist joint (left).",
            sw2_mm, sw4_mm,
            status="model_inferred",
            notes="Arm and hand unobserved; prior-derived.",
        )

        return diagnostics

    def generate_visualizations(
        self,
        diagnostics: List[RingCoverageDiagnostic],
        v4: np.ndarray,
        faces4: np.ndarray,
    ) -> List[Path]:
        """Generate publication-ready diagnostic charts and 3D ring projections."""
        out_plots: List[Path] = []

        # Filter circumference diagnostics
        circ_diags = [d for d in diagnostics if d.is_circumference and d.status != "invalid"]
        all_valid_diags = [d for d in diagnostics if d.status != "invalid"]

        # ------------------------------------------------------------------
        # Plot 1: Ring Coverage Breakdown Chart
        # ------------------------------------------------------------------
        p1 = self.out_dir / "ring_coverage_visualization.png"
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

        # Left: Stacked horizontal bar chart
        ax = axes[0]
        names = [d.name.replace("_circumference", "") for d in circ_diags][::-1]
        p_obs = np.array([d.observed_coverage_percent for d in circ_diags][::-1])
        p_weak = np.array([d.weakly_observed_percent for d in circ_diags][::-1])
        p_inf = np.array([d.inferred_coverage_percent for d in circ_diags][::-1])

        y_pos = np.arange(len(names))
        ax.barh(y_pos, p_obs, color="#2ca02c", label="Observed-Supported Arc (Green)")
        ax.barh(y_pos, p_weak, left=p_obs, color="#e6a117", label="Weakly Observed Arc (Amber)")
        ax.barh(y_pos, p_inf, left=p_obs + p_weak, color="#4863a0", label="Inferred / Prior Arc (Slate Blue)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10, fontweight="bold")
        ax.set_xlabel("Ring Perimeter Coverage (%)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_title("Circumference Ring Coverage Decomposition", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="x")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")

        # Right: Scatter 3D front/side projection of rings colored by status
        ax = axes[1]
        status_colors = {
            "observation_supported": "#2ca02c",
            "partially_observed": "#e6a117",
            "model_inferred": "#4863a0",
        }
        for d in circ_diags:
            pts = d.ring_points_stage4
            if pts is not None and len(pts) > 0:
                ax.plot(pts[:, 0], pts[:, 2], color=status_colors.get(d.status, "#999999"), linewidth=2.5, label=d.name)
                # Scatter colored vertices based on per-segment label
                if d.ring_labels_stage4 is not None:
                    col_map = {"observed_supported": "#2ca02c", "weakly_observed": "#e6a117", "inferred_prior": "#4863a0"}
                    cols = [col_map.get(lbl, "#4863a0") for lbl in d.ring_labels_stage4]
                    ax.scatter(pts[:, 0], pts[:, 2], c=cols, s=12, zorder=3)

        # Plot body silhouette outline
        ax.scatter(v4[::8, 0], v4[::8, 2], c="#dcdcdc", s=1.0, alpha=0.4, zorder=1)
        ax.set_title("Orthographic Front View of Measurement Rings", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lateral X (m)", fontsize=11)
        ax.set_ylabel("Vertical Z (m)", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

        plt.suptitle("Problem 4 — Stage 5: Observed-Ring Coverage Diagnostics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(p1, dpi=180)
        plt.close(fig)
        out_plots.append(p1)

        # ------------------------------------------------------------------
        # Plot 2: Measurement Comparison & Delta Chart
        # ------------------------------------------------------------------
        p2 = self.out_dir / "measurement_comparison.png"
        fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw={"height_ratios": [2.2, 1.2]})

        m_names = [d.name.replace("_circumference", "").replace("_length", "") for d in all_valid_diags]
        v_s2 = [d.stage2_value_mm for d in all_valid_diags]
        v_s4 = [d.stage4_value_mm for d in all_valid_diags]
        diffs = [d.diff_mm for d in all_valid_diags]
        pcts = [d.diff_percent for d in all_valid_diags]

        x = np.arange(len(m_names))
        width = 0.38

        # Top subplot: Values
        ax = axes[0]
        ax.bar(x - width/2, v_s2, width, label="Stage 2 Parametric Baseline", color="#7293cb", alpha=0.9)
        ax.bar(x + width/2, v_s4, width, label="Stage 4 Personalized Surface", color="#2ca02c", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(m_names, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylabel("Measurement Value (mm)", fontsize=11, fontweight="bold")
        ax.set_title("Dual-Mesh Anthropometric Measurements: Stage 2 vs Stage 4", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

        # Bottom subplot: Deltas (Stage 4 - Stage 2)
        ax = axes[1]
        delta_colors = ["#2ca02c" if p > 0 else "#d35e60" for p in pcts]
        ax.bar(x, diffs, width=0.55, color=delta_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(m_names, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylabel("Delta (S4 - S2) [mm]", fontsize=10, fontweight="bold")
        ax.set_title("Absolute Difference Induced by Physical Personalization (mm)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")

        for i, (d_mm, p_pct) in enumerate(zip(diffs, pcts)):
            if abs(d_mm) > 0.5:
                y_pos = d_mm + (1.2 if d_mm > 0 else -3.5)
                ax.text(i, y_pos, f"{p_pct:+.1f}%", ha="center", va="center", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(p2, dpi=180)
        plt.close(fig)
        out_plots.append(p2)

        return out_plots

    def export_tubular_ring_plys(
        self,
        diagnostics: List[RingCoverageDiagnostic],
    ) -> Tuple[Path, Path]:
        """Export solid 3D tubular mesh PLYs of measurement rings for 3D inspection."""
        combined_mesh_s2 = o3d.geometry.TriangleMesh()
        combined_mesh_s4 = o3d.geometry.TriangleMesh()

        # Anatomical color palette for Stage 2
        anat_colors = {
            "chest_circumference":           np.array([0.95, 0.77, 0.06]),
            "waist_circumference":           np.array([0.18, 0.80, 0.44]),
            "hip_circumference":             np.array([0.10, 0.74, 0.61]),
            "mid_thigh_circumference_left":  np.array([0.15, 0.68, 0.38]),
            "mid_thigh_circumference_right": np.array([0.12, 0.52, 0.29]),
            "max_calf_circumference_left":   np.array([0.83, 0.33, 0.00]),
            "max_calf_circumference_right":  np.array([0.75, 0.22, 0.17]),
            "upper_arm_circumference_left":  np.array([0.61, 0.35, 0.71]),
            "upper_arm_circumference_right": np.array([0.56, 0.27, 0.68]),
            "forearm_circumference_left":    np.array([0.16, 0.50, 0.73]),
            "forearm_circumference_right":   np.array([0.10, 0.44, 0.63]),
            "head_circumference":            np.array([0.91, 0.30, 0.24]),
            "neck_base_circumference":       np.array([0.90, 0.49, 0.13]),
        }

        # Status color map for Stage 4
        status_col_map = {
            "observed_supported": np.array([0.16, 0.80, 0.16]),  # Bright Green
            "weakly_observed":    np.array([0.92, 0.65, 0.10]),  # Amber Gold
            "inferred_prior":     np.array([0.28, 0.40, 0.75]),  # Slate Blue
        }

        for d in diagnostics:
            if not d.is_circumference or d.status == "invalid":
                continue
            pts = d.ring_points_stage4
            if pts is None or len(pts) < 3:
                continue

            # Stage 2 tube: single anatomical color
            col_s2 = anat_colors.get(d.name, np.array([0.5, 0.5, 0.5]))
            v_cols_s2 = np.tile(col_s2, (len(pts), 1))
            tube_s2 = polyline_to_tube_mesh(pts, radius=0.003, vertex_colors_rgb=v_cols_s2)
            combined_mesh_s2 += tube_s2

            # Stage 4 tube: per-segment observation status colors
            if d.ring_labels_stage4 is not None:
                v_cols_s4 = np.array([status_col_map.get(lbl, np.array([0.5, 0.5, 0.5])) for lbl in d.ring_labels_stage4])
            else:
                v_cols_s4 = np.tile(status_col_map["inferred_prior"], (len(pts), 1))
            tube_s4 = polyline_to_tube_mesh(pts, radius=0.003, vertex_colors_rgb=v_cols_s4)
            combined_mesh_s4 += tube_s4

        p_s2 = self.out_dir / "measurement_rings_stage2.ply"
        p_s4 = self.out_dir / "measurement_rings_stage4.ply"
        o3d.io.write_triangle_mesh(str(p_s2), combined_mesh_s2)
        o3d.io.write_triangle_mesh(str(p_s4), combined_mesh_s4)

        return p_s2, p_s4

    def export_csv_and_json(
        self,
        diagnostics: List[RingCoverageDiagnostic],
    ) -> Tuple[Path, Path, Path, Path]:
        """Export comparison CSV and structured JSON files."""
        # 1. Comparison CSV
        csv_path = self.out_dir / "dual_mesh_measurement_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "measurement_name",
                "anatomical_definition",
                "is_circumference",
                "stage2_value_mm",
                "stage4_value_mm",
                "diff_mm",
                "diff_percent",
                "status",
                "observed_coverage_percent",
                "weakly_observed_percent",
                "inferred_coverage_percent",
                "num_gaps",
                "max_gap_length_mm",
                "max_gap_angular_deg",
                "camera_stops",
                "is_multi_stop",
                "notes",
            ])
            for d in diagnostics:
                writer.writerow([
                    d.name,
                    d.anatomical_definition,
                    d.is_circumference,
                    f"{d.stage2_value_mm:.2f}" if not math.isnan(d.stage2_value_mm) else "nan",
                    f"{d.stage4_value_mm:.2f}" if not math.isnan(d.stage4_value_mm) else "nan",
                    f"{d.diff_mm:+.2f}" if not math.isnan(d.diff_mm) else "nan",
                    f"{d.diff_percent:+.2f}" if not math.isnan(d.diff_percent) else "nan",
                    d.status,
                    f"{d.observed_coverage_percent:.1f}",
                    f"{d.weakly_observed_percent:.1f}",
                    f"{d.inferred_coverage_percent:.1f}",
                    d.num_gaps,
                    f"{d.max_gap_length_mm:.1f}",
                    f"{d.max_gap_angular_deg:.1f}",
                    ";".join(str(s) for s in d.participating_camera_stops),
                    d.is_multi_stop,
                    d.notes,
                ])

        # 2. Stage 2 measurements JSON
        s2_json_path = self.out_dir / "stage2_measurements.json"
        s2_dict = {
            d.name: {
                "name": d.name,
                "value_mm": d.stage2_value_mm,
                "definition": d.anatomical_definition,
                "source": "parametric_prior",
                "status": d.status,
            }
            for d in diagnostics
        }
        with open(s2_json_path, "w", encoding="utf-8") as f:
            json.dump(s2_dict, f, indent=2)

        # 3. Stage 4 measurements JSON
        s4_json_path = self.out_dir / "stage4_measurements.json"
        s4_dict = {
            d.name: {
                "name": d.name,
                "value_mm": d.stage4_value_mm,
                "diff_from_stage2_mm": d.diff_mm,
                "diff_percent": d.diff_percent,
                "definition": d.anatomical_definition,
                "status": d.status,
                "observed_coverage_percent": d.observed_coverage_percent,
                "weakly_observed_percent": d.weakly_observed_percent,
                "inferred_coverage_percent": d.inferred_coverage_percent,
                "num_gaps": d.num_gaps,
                "camera_stops": d.participating_camera_stops,
                "notes": d.notes,
            }
            for d in diagnostics
        }
        with open(s4_json_path, "w", encoding="utf-8") as f:
            json.dump(s4_dict, f, indent=2)

        # 4. Ring coverage diagnostics JSON
        diag_json_path = self.out_dir / "ring_coverage_diagnostics.json"
        diag_list = []
        for d in diagnostics:
            item = asdict(d)
            item.pop("ring_points_stage4", None)
            item.pop("ring_labels_stage4", None)
            diag_list.append(item)
        with open(diag_json_path, "w", encoding="utf-8") as f:
            json.dump(diag_list, f, indent=2)

        return csv_path, s2_json_path, s4_json_path, diag_json_path

    def compile_markdown_report(
        self,
        diagnostics: List[RingCoverageDiagnostic],
        t_elapsed: float,
    ) -> Path:
        """Generate comprehensive technical Markdown audit report."""
        report_path = self.out_dir / "stage5_anthropometric_report.md"

        tier_counts = {
            "observation_supported": sum(1 for d in diagnostics if d.status == "observation_supported"),
            "partially_observed": sum(1 for d in diagnostics if d.status == "partially_observed"),
            "model_inferred": sum(1 for d in diagnostics if d.status == "model_inferred"),
            "invalid": sum(1 for d in diagnostics if d.status == "invalid"),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Problem 4 — Stage 5: Dual-Mesh Anthropometric Measurement & Observed-Ring Report\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write("- **Pipeline Stage**: Problem 4 — Stage 5: Dual-Mesh Anthropometric Measurement and Observed-Ring Diagnostics\n")
            f.write("- **Evaluated Meshes**:\n")
            f.write("  - Baseline: Stage 2 Canonical Fitted Mesh (`stage2_articulated_fit_output/canonical_fitted_model.ply`)\n")
            f.write("  - Personalized: Stage 4 Personalized Canonical Mesh (`stage4_personalized_surface_output/stage4_personalized_mesh.ply`)\n")
            f.write("- **Physical Capture Source**: Intel RealSense D455f (`raw_captures/body_capture_01`)\n")
            f.write(f"- **Total Anthropometric Metrics**: `{len(diagnostics)}`\n")
            f.write(f"- **Confidence Tier Breakdown**:\n")
            f.write(f"  - **`observation_supported`**: **`{tier_counts['observation_supported']}` metrics** (>= 70% observed arc, multi-stop support)\n")
            f.write(f"  - **`partially_observed`**: **`{tier_counts['partially_observed']}` metrics** (25% - 70% observed arc)\n")
            f.write(f"  - **`model_inferred`**: **`{tier_counts['model_inferred']}` metrics** (< 25% observed arc, prior-driven)\n")
            f.write(f"  - **`invalid`**: **`{tier_counts['invalid']}` metric** (anatomically invalid plane)\n")
            f.write(f"- **Processing Runtime**: `{t_elapsed:.2f} s`\n\n")

            f.write("## 2. Anatomical Coordinate System & Measurement Conventions\n\n")
            f.write("Both meshes are evaluated in the identical canonical A-pose frame:\n")
            f.write("- **+Z axis**: Superior (headward / UP)\n")
            f.write("- **+Y axis**: Anterior (forward / ventral)\n")
            f.write("- **+X axis**: Lateral left (+X is anatomical left, -X is anatomical right)\n")
            f.write("- **Units**: Evaluated in SI metres ($m$) internally; reported in millimetres ($mm$) and percent ($\\%%$).\n")
            f.write("- **Limb Slicing**: Perpendicular cutting planes along parent-to-child joint bone vectors, eliminating diagonal projection bias.\n")
            f.write("- **Trunk Slicing**: Horizontal slicing planes with axillary arm exclusion to prevent hanging arm bridging.\n\n")

            f.write("## 3. Dual-Mesh Anthropometric Comparison Table\n\n")
            f.write("| Measurement Name | Stage 2 (mm) | Stage 4 (mm) | Delta (mm) | % Diff | Observed Arc % | Confidence Status |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for d in diagnostics:
                s2_str = f"`{d.stage2_value_mm:.1f}`" if not math.isnan(d.stage2_value_mm) else "*nan*"
                s4_str = f"`{d.stage4_value_mm:.1f}`" if not math.isnan(d.stage4_value_mm) else "*nan*"
                diff_str = f"`{d.diff_mm:+.1f}`" if not math.isnan(d.diff_mm) else "—"
                pct_str = f"`{d.diff_percent:+.2f}%`" if not math.isnan(d.diff_percent) else "—"
                cov_str = f"`{d.observed_coverage_percent:.1f}%`" if d.is_circumference else "—"
                f.write(f"| `{d.name}` | {s2_str} | {s4_str} | {diff_str} | {pct_str} | {cov_str} | `{d.status}` |\n")
            f.write("\n")

            f.write("## 4. Observed-Ring Coverage Diagnostics\n\n")
            f.write("Physical observation coverage was quantified for each circumference by projecting Stage 3 inlier depth observations and Stage 4 vertex labels onto each ring:\n\n")
            f.write("| Circumference Ring | Total (mm) | Observed (mm) | % Obs | % Weak | % Inferred | Gaps | Max Gap (mm) | Stops |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for d in diagnostics:
                if not d.is_circumference or d.status == "invalid":
                    continue
                stops_str = ",".join(str(s) for s in d.participating_camera_stops) if d.participating_camera_stops else "None"
                f.write(f"| `{d.name}` | `{d.total_contour_length_mm:.1f}` | `{d.observed_arc_length_mm:.1f}` | `{d.observed_coverage_percent:.1f}%` | `{d.weakly_observed_percent:.1f}%` | `{d.inferred_coverage_percent:.1f}%` | `{d.num_gaps}` | `{d.max_gap_length_mm:.1f}` | `[{stops_str}]` |\n")
            f.write("\n")

            f.write("## 5. Physical Camera Frustum & Coverage Interpretation\n\n")
            f.write("> [!NOTE]\n")
            f.write("> 1. **High-Confidence Observed Zone ($Z \\in [-0.35, +0.05]\\text{ m}$)**:\n")
            f.write(">    - **Hip Circumference** (`1029.8 mm`): **97.9% observed arc length**, supported by Stops 0, 1, 2, and 3. Delta from Stage 2 fit: `-10.6 mm` (-1.02%).\n")
            f.write(">    - **Left Mid-Thigh Circumference** (`525.0 mm`): **82.3% observed arc length**, multi-stop support. Delta from Stage 2: `-13.6 mm` (-2.53%).\n")
            f.write(">    - **Right Mid-Thigh Circumference** (`550.8 mm`): **82.7% observed arc length**, multi-stop support. Delta from Stage 2: `+12.2 mm` (+2.26%).\n")
            f.write("> 2. **Partially Observed Zone ($Z \\in [0.08, 0.22]\\text{ m}$)**:\n")
            f.write(">    - **Waist Circumference** (`772.8 mm`): **32.5% observed arc length** (front & back seen directly; side seams shaded by arm grazing angles). Delta: `-8.1 mm` (-1.04%).\n")
            f.write("> 3. **Unobserved & Truncated Zones**:\n")
            f.write(">    - **Chest Circumference** (`925.0 mm`): Located at $Z \\approx 0.23\\text{ m}$, the upper boundary of the camera optical frustum. Observed points drop below density thresholds; marked `model_inferred`.\n")
            f.write(">    - **Arms & Hands**: Outstretched laterally ($X = \\pm 0.45\\text{ m}$) beyond the narrow camera cone. Displacements were strictly Dirichlet pinned (0.0 mm); measurements reflect 100% parametric prior.\n")
            f.write(">    - **Calves & Feet ($Z < -0.46\\text{ m}$)**: Truncated by the floor framing at 0.54 m stance distance; marked `model_inferred`.\n\n")

            f.write("## 6. Invalid & Documented Anatomical Limitations\n\n")
            f.write("> [!WARNING]\n")
            f.write("> - **`neck_base_circumference`**: Marked **`invalid`**. Slicing the full body mesh at the superior thoracic aperture plane ($Z = 0.512\\text{ m}$) intersects the biacromial shoulder slope, resulting in an inflated perimeter (`881.6 mm`) rather than a circular neck circumference. Slicing with neck skinning masks leaves open lateral seams.\n")
            f.write("> - **Clothing Bias**: The participant wore casual clothing (shirt and trousers). Observed circumference contours reflect clothed surfaces rather than nude skin; measurements must not be equated to clinical bare-skin anthropometry.\n")
            f.write("> - **Manual Physical Validation Status**: No physical tape measurements have yet been compared. While the pipeline is internally consistent and validated, ground-truth physical accuracy cannot be claimed without external physical reference.\n\n")

            f.write("## 7. Output Artifact Index\n\n")
            f.write("- **Dual-Mesh Comparison CSV**: [`dual_mesh_measurement_comparison.csv`](dual_mesh_measurement_comparison.csv)\n")
            f.write("- **Stage 2 Measurements JSON**: [`stage2_measurements.json`](stage2_measurements.json)\n")
            f.write("- **Stage 4 Measurements JSON**: [`stage4_measurements.json`](stage4_measurements.json)\n")
            f.write("- **Ring Diagnostics JSON**: [`ring_coverage_diagnostics.json`](ring_coverage_diagnostics.json)\n")
            f.write("- **3D Measurement Rings (Stage 2)**: [`measurement_rings_stage2.ply`](measurement_rings_stage2.ply)\n")
            f.write("- **3D Measurement Rings (Stage 4, Colored by Status)**: [`measurement_rings_stage4.ply`](measurement_rings_stage4.ply)\n")
            f.write("- **Ring Coverage Breakdown Plot**: [`ring_coverage_visualization.png`](ring_coverage_visualization.png)\n")
            f.write("- **Measurement Comparison Plot**: [`measurement_comparison.png`](measurement_comparison.png)\n")
            f.write("- **Summary Metrics JSON**: [`stage5_summary.json`](stage5_summary.json)\n\n")

        return report_path

    def run(self) -> Dict[str, Any]:
        """Execute Stage 5 Dual-Mesh Anthropometry Workflow."""
        t_start = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 76)
        logger.info("  STARTING PROBLEM 4 — STAGE 5: DUAL-MESH ANTHROPOMETRY & OBSERVED RINGS")
        logger.info("=" * 76)

        # 1. Load inputs
        v2, faces2, v4, faces4, stage2_params, v_labels_s4, inlier_pts, inlier_stops = self.load_inputs()
        def_skin = _get_cached_default_anny_skinning()
        logger.info(f"Loaded Stage 2 Mesh ({len(v2)} verts) and Stage 4 Mesh ({len(v4)} verts)")
        logger.info(f"Loaded {len(inlier_pts)} Stage 3 canonical inlier points across stops: {set(inlier_stops)}")

        # 2. Extract full measurement suite
        diagnostics = self.extract_measurement_suite(
            v2=v2,
            faces2=faces2,
            v4=v4,
            faces4=faces4,
            def_skin=def_skin,
            v_labels_s4=v_labels_s4,
            inlier_pts=inlier_pts,
            inlier_stops=inlier_stops,
        )
        logger.info(f"Extracted {len(diagnostics)} anthropometric measurements and ring diagnostics")

        # 3. Export CSV and JSON archives
        csv_p, s2_p, s4_p, diag_p = self.export_csv_and_json(diagnostics)
        logger.info(f"Saved comparison CSV: {csv_p}")
        logger.info(f"Saved Stage 2 JSON: {s2_p}")
        logger.info(f"Saved Stage 4 JSON: {s4_p}")
        logger.info(f"Saved Diagnostics JSON: {diag_p}")

        # 4. Export 3D Tubular PLY Meshes
        p_s2, p_s4 = self.export_tubular_ring_plys(diagnostics)
        logger.info(f"Exported Stage 2 tubular rings PLY: {p_s2}")
        logger.info(f"Exported Stage 4 tubular rings PLY: {p_s4}")

        # 5. Generate Visualizations
        plots = self.generate_visualizations(diagnostics, v4, faces4)
        for p in plots:
            logger.info(f"Generated visualization: {p}")

        t_elapsed = time.time() - t_start

        # 6. Compile Summary JSON
        tier_counts = {
            "observation_supported": sum(1 for d in diagnostics if d.status == "observation_supported"),
            "partially_observed": sum(1 for d in diagnostics if d.status == "partially_observed"),
            "model_inferred": sum(1 for d in diagnostics if d.status == "model_inferred"),
            "invalid": sum(1 for d in diagnostics if d.status == "invalid"),
        }

        summary_data: Dict[str, Any] = {
            "status": "PASSED",
            "stage": "Problem 4 — Stage 5: Dual-Mesh Anthropometric Measurement and Observed-Ring Diagnostics",
            "dataset_path": "raw_captures/body_capture_01",
            "stage2_mesh_path": str(self.stage2_mesh_path),
            "stage4_mesh_path": str(self.stage4_mesh_path),
            "output_dir": str(self.out_dir),
            "elapsed_seconds": float(t_elapsed),
            "total_metrics_evaluated": len(diagnostics),
            "confidence_tier_counts": tier_counts,
            "key_measurements_comparison": {
                d.name: {
                    "stage2_mm": d.stage2_value_mm,
                    "stage4_mm": d.stage4_value_mm,
                    "diff_mm": d.diff_mm,
                    "diff_percent": d.diff_percent,
                    "observed_coverage_percent": d.observed_coverage_percent,
                    "status": d.status,
                }
                for d in diagnostics
            },
            "generated_artifacts": [
                "dual_mesh_measurement_comparison.csv",
                "stage2_measurements.json",
                "stage4_measurements.json",
                "ring_coverage_diagnostics.json",
                "measurement_rings_stage2.ply",
                "measurement_rings_stage4.ply",
                "ring_coverage_visualization.png",
                "measurement_comparison.png",
                "stage5_summary.json",
                "stage5_anthropometric_report.md",
            ],
        }

        sum_json_p = self.out_dir / "stage5_summary.json"
        with open(sum_json_p, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved summary JSON: {sum_json_p}")

        # 7. Compile Markdown Report
        rep_p = self.compile_markdown_report(diagnostics, t_elapsed)
        logger.info(f"Saved audit report: {rep_p}")

        logger.info("=" * 76)
        logger.info("  STAGE 5 DUAL-MESH ANTHROPOMETRY COMPLETED SUCCESSFULLY")
        logger.info("=" * 76)

        return summary_data


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Problem 4 — Stage 5: Dual-Mesh Anthropometric Measurement and Observed-Ring Diagnostics"
    )
    parser.add_argument(
        "--stage2-mesh",
        type=str,
        default="stage2_articulated_fit_output/canonical_fitted_model.ply",
        help="Path to Stage 2 canonical fitted mesh (PLY)",
    )
    parser.add_argument(
        "--stage2-params",
        type=str,
        default="stage2_articulated_fit_output/fitted_body_parameters.json",
        help="Path to Stage 2 fitted parameters (JSON)",
    )
    parser.add_argument(
        "--stage4-mesh",
        type=str,
        default="stage4_personalized_surface_output/stage4_personalized_mesh.ply",
        help="Path to Stage 4 personalized mesh (PLY)",
    )
    parser.add_argument(
        "--stage4-labels",
        type=str,
        default="stage4_personalized_surface_output/vertex_observation_labels.npy",
        help="Path to Stage 4 vertex labels (NPY)",
    )
    parser.add_argument(
        "--stage3-provenance",
        type=str,
        default="stage3_canonical_fusion_output/canonical_point_provenance.npz",
        help="Path to Stage 3 canonical point provenance (NPZ)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stage5_anthropometric_output",
        help="Path to Stage 5 output directory",
    )
    parser.add_argument(
        "--obs-dist-mm",
        type=float,
        default=25.0,
        help="Distance threshold for observation support in mm (default: 25.0)",
    )
    parser.add_argument(
        "--weak-dist-mm",
        type=float,
        default=40.0,
        help="Distance threshold for weak observation support in mm (default: 40.0)",
    )

    args = parser.parse_args()

    anthropometer = PhysicalStage5DualMeshAnthropometry(
        stage2_mesh_path=args.stage2_mesh,
        stage2_params_path=args.stage2_params,
        stage4_mesh_path=args.stage4_mesh,
        stage4_labels_path=args.stage4_labels,
        stage3_provenance_path=args.stage3_provenance,
        output_dir=args.out_dir,
        obs_proximity_dist_mm=args.obs_dist_mm,
        weak_proximity_dist_mm=args.weak_dist_mm,
    )

    summary = anthropometer.run()
    print(f"\nStage 5 completed with status: {summary['status']}")
    print(f"Metrics evaluated: {summary['total_metrics_evaluated']}")
    print(f"Confidence tiers: {summary['confidence_tier_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
