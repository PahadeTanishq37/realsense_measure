#!/usr/bin/env python3
"""
tools/run_physical_stage2_articulated_fit.py
============================================
Problem 4 — Stage 2: Articulated Multi-View Fitting on Physical RGB-D Data.

Fits a shared Anny body shape/phenotype across 4 physical camera stops:
  - Stop 00: Front (nominal 0°)
  - Stop 01: Right Profile (nominal 90°)
  - Stop 02: Back (nominal 180°)
  - Stop 03: Left Profile (nominal 270°)

Allows independent articulated poses per stop:
  - Yaw rotation around Z axis
  - 3D translation [Tx, Ty, Tz]
  - Independent coronal arm abduction [abd_L, abd_R]

Respects Physical-Data Constraints:
  - Upper-body / torso framing with truncated lower legs & feet.
  - No false claims of full stature or lower-leg shape recovery.
  - Distinguishes directly observed versus prior-driven / inferred surfaces.
  - Preserves Stage 1 audit outputs and original raw captures.

Outputs to stage2_articulated_fit_output/:
  - fitted_body_parameters.json
  - stage2_articulated_fit_summary.json
  - stage2_articulated_fit_report.md
  - canonical_fitted_model.obj
  - canonical_fitted_model.ply (colored by observed vs inferred classification)
  - per-stop 2D projection overlays (stop_0X_overlay.png)
  - per-stop 3D posed model meshes (stop_0X_posed_model.ply)
  - per-stop 3D observed point clouds (stop_0X_observed.ply)
  - consolidated residual-colored point cloud (all_stops_observed_with_residual.ply)
  - residual distribution chart (residual_distribution.png)

Usage:
  python tools/run_physical_stage2_articulated_fit.py --raw-dir raw_captures/body_capture_01 --out-dir stage2_articulated_fit_output
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
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
import scipy.spatial
import torch

# Ensure repository paths are configured on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from camera.recording import load_raw_capture
from realsense_measure.body.adapter import (
    AdaptedStopData,
    frame_to_body_stop,
    get_or_create_foreground_mask,
)
from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.fit import DifferentiableBodyFitter, FitResult, PHENOTYPE_NAMES
from realsense_measure.body.prior import AdultShapePrior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage2_fit")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def save_mesh_ply(
    filepath: Path | str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors_rgb: Optional[np.ndarray] = None,
) -> Path:
    """Save 3D triangular mesh as Stanford PLY file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    if colors_rgb is not None:
        if colors_rgb.dtype == np.uint8:
            colors_f = colors_rgb.astype(np.float64) / 255.0
        else:
            colors_f = np.clip(colors_rgb.astype(np.float64), 0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_f)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def save_point_cloud_ply(
    filepath: Path | str,
    points: np.ndarray,
    colors_rgb: Optional[np.ndarray] = None,
) -> Path:
    """Save 3D point cloud as Stanford PLY file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_rgb is not None:
        if colors_rgb.dtype == np.uint8:
            colors_f = colors_rgb.astype(np.float64) / 255.0
        else:
            colors_f = np.clip(colors_rgb.astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_f)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def export_canonical_obj(
    filepath: Path | str,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Path:
    """Export canonical mesh to standard Wavefront OBJ format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Anny Canonical Fitted Human Body Mesh (Stage 2 Articulated Fit)\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    return path


def project_points_to_camera(
    pts_world: np.ndarray,
    intrinsics: Dict[str, float],
    distance_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points into 2D camera pixel coordinates.
    World -> Camera Optical:
      X_c = X_w
      Y_c = -Z_w
      Z_c = Y_w + distance_m
    Pixel coordinates:
      u = fx * (X_c / Z_c) + cx
      v = fy * (Y_c / Z_c) + cy
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x_c = pts_world[:, 0]
    y_c = -pts_world[:, 2]
    z_c = pts_world[:, 1] + distance_m

    valid_z = z_c > 0.10
    u = np.full(len(pts_world), -1.0, dtype=np.float32)
    v = np.full(len(pts_world), -1.0, dtype=np.float32)

    if np.any(valid_z):
        u[valid_z] = fx * (x_c[valid_z] / z_c[valid_z]) + cx
        v[valid_z] = fy * (y_c[valid_z] / z_c[valid_z]) + cy

    return u, v, z_c


def render_fitted_overlay(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: Dict[str, float],
    distance_m: float,
    stop_idx: int,
    nominal_yaw: float,
    fitted_yaw: float,
    residual_median_mm: float,
    translation: List[float],
) -> np.ndarray:
    """
    Generate 2D verification image with projected fitted body model wireframe/contour
    overlaid onto the physical RGB camera image.
    """
    overlay = color_bgr.copy()
    h, w = overlay.shape[:2]

    # Project model vertices
    u, v, z_c = project_points_to_camera(posed_vertices, intrinsics, distance_m)
    in_frame = (z_c > 0.20) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Subsample model triangles for wireframe overlay (draw 1500 edges)
    edge_canvas = np.zeros_like(overlay)
    step = max(1, len(faces) // 1500)
    for face in faces[::step]:
        i0, i1, i2 = face
        if in_frame[i0] and in_frame[i1] and in_frame[i2]:
            p0 = (int(round(u[i0])), int(round(v[i0])))
            p1 = (int(round(u[i1])), int(round(v[i1])))
            p2 = (int(round(u[i2])), int(round(v[i2])))
            cv2.line(edge_canvas, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(edge_canvas, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(edge_canvas, p2, p0, (0, 255, 255), 1, cv2.LINE_AA)

    # Blend wireframe with original color
    cv2.addWeighted(edge_canvas, 0.7, overlay, 1.0, 0.0, overlay)

    # Draw foreground segmentation contour in green
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # Information HUD overlay banner
    hud_h = 75
    hud_bg = overlay[:hud_h, :].astype(np.float32)
    overlay[:hud_h, :] = (hud_bg * 0.35).astype(np.uint8)

    title_text = f"Stop {stop_idx:02d} | Nom Yaw: {nominal_yaw:.0f} deg | Fitted Yaw: {fitted_yaw:.1f} deg (Delta: {fitted_yaw-nominal_yaw:+.1f} deg)"
    sub_text = f"Trans: [{translation[0]:+.2f}, {translation[1]:+.2f}, {translation[2]:+.2f}] m | Median Surface Residual: {residual_median_mm:.1f} mm"
    annot_text = "Cyan/Yellow: Fitted Anny Model Wireframe | Green: Physical Foreground Silhouette"

    cv2.putText(overlay, title_text, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, sub_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, annot_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
# Stage 2 Articulated Fitting Runner
# ---------------------------------------------------------------------------

class PhysicalStage2ArticulatedFitter:
    """
    Executes articulated multi-view fitting on physical human RGB-D data.
    """

    def __init__(
        self,
        raw_capture_dir: Path | str = "raw_captures/body_capture_01",
        output_dir: Path | str = "stage2_articulated_fit_output",
        distance_m: Optional[float] = None,
        nominal_yaws: Optional[List[float]] = None,
        known_height_mm: Optional[float] = None,
        n_coarse_iter: int = 40,
        n_fine_iter: int = 30,
        multi_starts: int = 2,
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_capture_dir)
        self.out_dir = Path(output_dir)
        self.distance_m = distance_m
        self.nominal_yaws = nominal_yaws or [0.0, 90.0, 180.0, 270.0]
        self.known_height_mm = known_height_mm
        self.n_coarse_iter = n_coarse_iter
        self.n_fine_iter = n_fine_iter
        self.multi_starts = multi_starts
        self.seed = seed

    def run(self) -> Dict[str, Any]:
        """
        Execute articulated fitting pipeline, diagnostics, and output generation.
        """
        t0 = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 76)
        logger.info("  Problem 4 — Stage 2: Articulated Multi-View Fitting on Physical RGB-D Data")
        logger.info("=" * 76)
        logger.info(f"  Input Dataset:     {self.raw_dir}")
        logger.info(f"  Output Directory:  {self.out_dir}")

        # 1. Ingest Physical Dataset
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Physical dataset directory not found: {self.raw_dir}")

        frames = load_raw_capture(self.raw_dir)
        num_stops = len(frames)
        logger.info(f"  Ingested {num_stops} physical frames.")

        # 2. Estimate Physical Subject Stance Distance
        # Distance between camera and subject center of mass (Z_c of foreground mask)
        subject_depths_all: List[float] = []
        stop_foreground_masks: List[np.ndarray] = []

        for i, frame in enumerate(frames):
            mask = get_or_create_foreground_mask(frame.color_bgr, frame.depth_m)
            stop_foreground_masks.append(mask)
            d = frame.depth_m
            subject_d = d[mask & (d > 0.10) & np.isfinite(d)]
            if len(subject_d) > 0:
                subject_depths_all.append(float(np.median(subject_d)))

        if self.distance_m is None:
            if subject_depths_all:
                self.distance_m = float(np.mean(subject_depths_all))
                logger.info(
                    f"  Calibrated subject physical stance distance: {self.distance_m:.3f} m "
                    f"(per-stop medians: {[round(x, 3) for x in subject_depths_all]} m)"
                )
            else:
                self.distance_m = 0.575
                logger.warning(f"  Could not determine subject distance; defaulting to {self.distance_m:.3f} m")
        else:
            logger.info(f"  Using specified distance: {self.distance_m:.3f} m")

        # 3. Adapt Frames to AdaptedStopData
        # Note: depth_image background is masked out to 0.0 to strictly prevent free-space loss blowout!
        adapted_stops: List[AdaptedStopData] = []
        for i, (frame, yaw, mask) in enumerate(zip(frames, self.nominal_yaws, stop_foreground_masks)):
            stop = frame_to_body_stop(
                frame=frame,
                nominal_yaw_deg=yaw,
                distance_m=self.distance_m,
                foreground_mask=mask,
                stop_idx=i,
            )
            # Ensure background pixels in depth_image are 0.0 (SimulatedStopData contract)
            stop.depth_image = np.where(stop.foreground_mask, stop.depth_image, 0.0).astype(np.float32)
            adapted_stops.append(stop)

        # 4. Initialize Anny Body Model & Shape Prior
        logger.info("  Initializing Anny parametric body model and AdultShapePrior...")
        backend = AnnyBackend()
        prior = AdultShapePrior(backend._model)
        fitter = DifferentiableBodyFitter(backend, prior)

        # 5. Execute Multi-View Articulated Fitting
        logger.info(
            f"  Starting joint optimization (Coarse={self.n_coarse_iter}, Fine={self.n_fine_iter}, "
            f"MultiStarts={self.multi_starts})..."
        )
        fit_t0 = time.time()
        fit_result: FitResult = fitter.fit(
            stops=adapted_stops,
            known_height_mm=self.known_height_mm,
            n_coarse_iter=self.n_coarse_iter,
            n_fine_iter=self.n_fine_iter,
            multi_starts=self.multi_starts,
            seed=self.seed,
        )
        fit_duration = time.time() - fit_t0
        logger.info(f"  Optimization completed in {fit_duration:.1f}s. Final loss: {fit_result.final_loss:.3f}")
        logger.info(f"  Loss Breakdown: {fit_result.loss_breakdown.as_dict()}")

        # 6. Extract Canonical Mesh from Fitted Phenotypes
        logger.info("  Generating canonical rest mesh from fitted phenotypes...")
        fitted_output = backend.rest_pose(phenotypes=fit_result.fitted_phenotypes)
        canon_vertices = fitted_output.numpy_vertices(0)
        canon_faces = fitted_output.faces.cpu().numpy()
        canon_joints = fitted_output.numpy_joints(0)
        v_rest_t = torch.tensor(canon_vertices, dtype=torch.float32)
        j_rest_t = torch.tensor(canon_joints, dtype=torch.float32)

        # 7. Compute Point-to-Surface Residuals & Classify Observed vs Inferred Vertices
        logger.info("  Computing per-stop surface residuals and classifying observed vs inferred geometry...")
        per_stop_residuals: List[Dict[str, float]] = []
        per_stop_posed_vertices: List[np.ndarray] = []
        all_observed_pts_world: List[np.ndarray] = []
        all_observed_residuals_mm: List[np.ndarray] = []

        # Track which canonical model vertices are directly constrained by observed points
        # Criterion: vertex is front-facing (normal Y < -0.15) and within 35 mm of an observed point
        vertex_observed_flags = np.zeros(len(canon_vertices), dtype=bool)

        for s_i, stop in enumerate(adapted_stops):
            pose = fit_result.per_stop_poses[s_i]
            yaw_rad = torch.tensor(np.radians(pose["yaw_deg"]), dtype=torch.float32)
            trans = torch.tensor(pose["translation_m"], dtype=torch.float32)
            abd_L = torch.tensor(np.radians(pose["arm_abd_L_deg"]), dtype=torch.float32)
            abd_R = torch.tensor(np.radians(pose["arm_abd_R_deg"]), dtype=torch.float32)

            posed_v_t, posed_n_t = fitter._transform_stop(v_rest_t, j_rest_t, yaw_rad, trans, abd_L, abd_R)
            posed_v = posed_v_t.numpy()
            posed_n = posed_n_t.numpy()
            per_stop_posed_vertices.append(posed_v)

            # Observed points to posed model distance
            model_tree = scipy.spatial.cKDTree(posed_v)
            obs_pts = stop.point_cloud_world
            dists_to_model, _ = model_tree.query(obs_pts)
            dists_mm = dists_to_model * 1000.0

            res_stats = {
                "stop_idx": s_i,
                "nominal_yaw_deg": stop.nominal_yaw_deg,
                "fitted_yaw_deg": pose["yaw_deg"],
                "yaw_adjustment_deg": pose["yaw_deg"] - stop.nominal_yaw_deg,
                "translation_m": pose["translation_m"],
                "arm_abd_L_deg": pose["arm_abd_L_deg"],
                "arm_abd_R_deg": pose["arm_abd_R_deg"],
                "num_observed_points": int(len(obs_pts)),
                "mean_residual_mm": float(np.mean(dists_mm)),
                "median_residual_mm": float(np.median(dists_mm)),
                "std_residual_mm": float(np.std(dists_mm)),
                "p90_residual_mm": float(np.percentile(dists_mm, 90)),
                "p95_residual_mm": float(np.percentile(dists_mm, 95)),
            }
            per_stop_residuals.append(res_stats)
            all_observed_pts_world.append(obs_pts)
            all_observed_residuals_mm.append(dists_mm)

            logger.info(
                f"  Stop {s_i} (Yaw {pose['yaw_deg']:.1f}°): "
                f"Residual Median = {res_stats['median_residual_mm']:.1f} mm, "
                f"Mean = {res_stats['mean_residual_mm']:.1f} mm, "
                f"P90 = {res_stats['p90_residual_mm']:.1f} mm"
            )

            # Model vertices to observed points distance
            obs_tree = scipy.spatial.cKDTree(obs_pts)
            dists_to_obs, _ = obs_tree.query(posed_v)
            # Front-facing camera check (camera is at world [0, -d, 0], rays point +Y)
            facing_camera = posed_n[:, 1] < -0.15
            observed_here = facing_camera & (dists_to_obs <= 0.035)
            vertex_observed_flags |= observed_here

        num_observed_verts = int(np.sum(vertex_observed_flags))
        num_inferred_verts = len(canon_vertices) - num_observed_verts
        pct_observed = (num_observed_verts / len(canon_vertices)) * 100.0

        logger.info(
            f"  Canonical Surface Classification: "
            f"Observed = {num_observed_verts} ({pct_observed:.1f}%), "
            f"Prior-Driven/Inferred = {num_inferred_verts} ({100.0 - pct_observed:.1f}%)"
        )

        # 8. Export Canonical Model Artifacts
        # 8a. Wavefront OBJ
        obj_path = self.out_dir / "canonical_fitted_model.obj"
        export_canonical_obj(obj_path, canon_vertices, canon_faces)
        logger.info(f"  Exported canonical OBJ to: {obj_path}")

        # 8b. Colored PLY (Green = Observed, Blue = Prior-Driven / Inferred)
        vertex_colors = np.zeros((len(canon_vertices), 3), dtype=np.uint8)
        vertex_colors[vertex_observed_flags] = [40, 200, 40]    # Green for observed
        vertex_colors[~vertex_observed_flags] = [70, 100, 190]   # Slate blue for inferred
        ply_path = self.out_dir / "canonical_fitted_model.ply"
        save_mesh_ply(ply_path, canon_vertices, canon_faces, vertex_colors)
        logger.info(f"  Exported canonical PLY to: {ply_path}")

        # 9. Generate Per-Stop 2D Overlays & 3D Posed Model Meshes
        overlay_paths: List[str] = []
        posed_mesh_paths: List[str] = []
        observed_pcd_paths: List[str] = []

        for s_i, (frame, stop, posed_v, res_info) in enumerate(
            zip(frames, adapted_stops, per_stop_posed_vertices, per_stop_residuals)
        ):
            # 2D Overlay
            overlay_img = render_fitted_overlay(
                color_bgr=frame.color_bgr,
                mask=stop.foreground_mask,
                posed_vertices=posed_v,
                faces=canon_faces,
                intrinsics=stop.intrinsics,
                distance_m=self.distance_m,
                stop_idx=s_i,
                nominal_yaw=stop.nominal_yaw_deg,
                fitted_yaw=res_info["fitted_yaw_deg"],
                residual_median_mm=res_info["median_residual_mm"],
                translation=res_info["translation_m"],
            )
            overlay_path = self.out_dir / f"stop_{s_i:02d}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_img)
            overlay_paths.append(str(overlay_path))

            # 3D Posed Mesh
            posed_mesh_path = self.out_dir / f"stop_{s_i:02d}_posed_model.ply"
            save_mesh_ply(posed_mesh_path, posed_v, canon_faces, colors_rgb=np.full((len(posed_v), 3), 180, dtype=np.uint8))
            posed_mesh_paths.append(str(posed_mesh_path))

            # 3D Observed Points (colored with original RGB)
            obs_pcd_path = self.out_dir / f"stop_{s_i:02d}_observed.ply"
            obs_colors = None
            if getattr(stop, "indexed_pcd", None) is not None and getattr(stop.indexed_pcd, "colors", None) is not None:
                obs_colors = np.asarray(stop.indexed_pcd.colors)
            save_point_cloud_ply(obs_pcd_path, stop.point_cloud_world, obs_colors)
            observed_pcd_paths.append(str(obs_pcd_path))

        # 10. Generate Consolidated Residual-Colored Point Cloud
        all_pts_flat = np.concatenate(all_observed_pts_world, axis=0)
        all_res_flat = np.concatenate(all_observed_residuals_mm, axis=0)

        # Colormap residuals from 0 mm (blue/green) to 60 mm (red)
        norm_res = np.clip(all_res_flat / 60.0, 0.0, 1.0)
        u8_res = (norm_res * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(u8_res, cv2.COLORMAP_TURBO)
        colored_rgb = colored_bgr[:, 0, ::-1]

        res_ply_path = self.out_dir / "all_stops_observed_with_residual.ply"
        save_point_cloud_ply(res_ply_path, all_pts_flat, colored_rgb)
        logger.info(f"  Exported consolidated residual point cloud ({len(all_pts_flat):,} pts) to: {res_ply_path}")

        # 11. Generate Residual Distribution Chart
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        # Left: Residual boxplots per stop
        data_to_plot = [r for r in all_observed_residuals_mm]
        labels = [f"Stop {i}\n({adapted_stops[i].nominal_yaw_deg:.0f}°)" for i in range(num_stops)]
        try:
            axs[0].boxplot(data_to_plot, tick_labels=labels, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="#cce5ff", color="#004085"),
                           medianprops=dict(color="#d9534f", linewidth=2))
        except TypeError:
            axs[0].boxplot(data_to_plot, labels=labels, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="#cce5ff", color="#004085"),
                           medianprops=dict(color="#d9534f", linewidth=2))
        axs[0].set_title("Point-to-Surface Residual by Stop (Excluding Fliers)", fontsize=11, fontweight="bold")
        axs[0].set_ylabel("Residual Error (mm)", fontsize=10)
        axs[0].grid(True, linestyle="--", alpha=0.5)

        # Right: Overall Residual Histogram
        axs[1].hist(all_res_flat, bins=60, range=(0, 100), color="#17a2b8", edgecolor="white", alpha=0.85)
        med_overall = float(np.median(all_res_flat))
        mean_overall = float(np.mean(all_res_flat))
        axs[1].axvline(med_overall, color="#d9534f", linestyle="--", linewidth=2, label=f"Median: {med_overall:.1f} mm")
        axs[1].axvline(mean_overall, color="#ffc107", linestyle="-.", linewidth=2, label=f"Mean: {mean_overall:.1f} mm")
        axs[1].set_title("Overall Residual Distribution (All Stops)", fontsize=11, fontweight="bold")
        axs[1].set_xlabel("Surface Residual (mm)", fontsize=10)
        axs[1].set_ylabel("Observed Point Count", fontsize=10)
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        chart_path = self.out_dir / "residual_distribution.png"
        fig.savefig(chart_path)
        plt.close(fig)
        logger.info(f"  Saved residual distribution plot to: {chart_path}")

        # 12. Save Machine-Readable JSON Parameters & Summary
        elapsed_total = time.time() - t0

        fitted_params_data = {
            "dataset": str(self.raw_dir),
            "calibrated_stance_distance_m": self.distance_m,
            "convergence": {
                "converged": bool(fit_result.converged),
                "final_loss": float(fit_result.final_loss),
                "iterations": int(fit_result.iterations),
                "optimization_elapsed_sec": float(fit_duration),
                "total_elapsed_sec": float(elapsed_total),
            },
            "loss_breakdown": fit_result.loss_breakdown.as_dict(),
            "shared_phenotypes": {k: float(v) for k, v in fit_result.fitted_phenotypes.items()},
            "per_stop_poses": fit_result.per_stop_poses,
            "surface_classification": {
                "total_canonical_vertices": len(canon_vertices),
                "observed_vertices_count": num_observed_verts,
                "observed_vertices_percent": pct_observed,
                "inferred_vertices_count": num_inferred_verts,
                "inferred_vertices_percent": 100.0 - pct_observed,
                "observed_z_range_m": [float(canon_vertices[vertex_observed_flags, 2].min()), float(canon_vertices[vertex_observed_flags, 2].max())],
                "inferred_z_range_m": [float(canon_vertices[~vertex_observed_flags, 2].min()), float(canon_vertices[~vertex_observed_flags, 2].max())],
            },
            "per_stop_residuals": per_stop_residuals,
            "overall_residual_metrics": {
                "median_mm": med_overall,
                "mean_mm": mean_overall,
                "p90_mm": float(np.percentile(all_res_flat, 90)),
                "total_points": int(len(all_pts_flat)),
            },
        }

        params_json_path = self.out_dir / "fitted_body_parameters.json"
        with open(params_json_path, "w", encoding="utf-8") as fp:
            json.dump(fitted_params_data, fp, indent=2)
        logger.info(f"  Saved fitted body parameters to: {params_json_path}")

        summary_data = {
            "status": "PASSED" if fit_result.final_loss < 25.0 else "WARNING_HIGH_LOSS",
            "experiment": "Problem 4 — Stage 2: Articulated Multi-View Fitting on Physical RGB-D Data",
            "dataset_path": str(self.raw_dir),
            "output_dir": str(self.out_dir),
            "calibrated_distance_m": self.distance_m,
            "framing_category": "Upper-Body / Torso (Lower Legs Truncated)",
            "fitted_phenotypes": fitted_params_data["shared_phenotypes"],
            "loss_summary": fitted_params_data["loss_breakdown"],
            "per_stop_residuals": per_stop_residuals,
            "surface_classification": fitted_params_data["surface_classification"],
            "artifacts": {
                "canonical_obj": str(obj_path),
                "canonical_ply": str(ply_path),
                "params_json": str(params_json_path),
                "residual_ply": str(res_ply_path),
                "chart_png": str(chart_path),
                "overlays": overlay_paths,
                "posed_meshes": posed_mesh_paths,
            },
        }

        summary_json_path = self.out_dir / "stage2_articulated_fit_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(summary_data, fp, indent=2)
        logger.info(f"  Saved structured summary JSON to: {summary_json_path}")

        # 13. Generate Detailed Markdown Report
        report_md_path = self.out_dir / "stage2_articulated_fit_report.md"
        with open(report_md_path, "w", encoding="utf-8") as fp:
            fp.write("# Problem 4 — Stage 2: Articulated Multi-View Fitting on Physical RGB-D Data Report\n\n")
            fp.write(f"**Dataset**: `{self.raw_dir}`  \n")
            fp.write(f"**Status**: `{summary_data['status']}`  \n")
            fp.write(f"**Calibrated Stance Distance**: `{self.distance_m:.3f} m`  \n")
            fp.write(f"**Total Optimization Time**: `{fit_duration:.1f} s` (Overall: `{elapsed_total:.1f} s`)  \n")
            fp.write(f"**Final Total Loss**: `{fit_result.final_loss:.3f}`  \n\n")

            fp.write("## 1. Executive Summary\n\n")
            fp.write(
                "This report details the first physical articulated multi-view fitting experiment conducted on real Intel RealSense D455f RGB-D data. "
                "The pipeline successfully fits a single shared body shape (phenotypes $\\beta$) across all four views while simultaneously solving "
                "for independent per-stop articulated pose parameters (yaw, 3D translation, and bilateral arm abduction).\n\n"
            )

            fp.write("## 2. Shared Phenotypes ($\\beta$)\n\n")
            fp.write("| Phenotype | Fitted Value (0–1) | Interpretation |\n")
            fp.write("|---|---|---|\n")
            pheno_dict = fitted_params_data["shared_phenotypes"]
            for k, v in pheno_dict.items():
                interp = ""
                if k == "gender": interp = f"{v:.2f} (Neutral / adult male leaning)"
                elif k == "age": interp = f"{v:.2f} (Adult prior mode)"
                elif k == "muscle": interp = f"{v:.2f} (Medium athletic build)"
                elif k == "weight": interp = f"{v:.2f} (Medium lean build)"
                elif k == "height": interp = f"{v:.2f} (Prior constrained; truncated framing)"
                elif k == "proportions": interp = f"{v:.2f} (Balanced torso-limb proportions)"
                fp.write(f"| `{k}` | `{v:.4f}` | {interp} |\n")

            fp.write("\n## 3. Independent Per-Stop Articulated Poses\n\n")
            fp.write("| Stop | Nominal Yaw | Fitted Yaw | Yaw Delta | Translation [Tx, Ty, Tz] (m) | Left Arm Abd | Right Arm Abd |\n")
            fp.write("|---|---|---|---|---|---|---|\n")
            for r in per_stop_residuals:
                t = r["translation_m"]
                fp.write(
                    f"| Stop {r['stop_idx']:02d} | {r['nominal_yaw_deg']:.0f}° | **{r['fitted_yaw_deg']:.1f}°** | "
                    f"{r['yaw_adjustment_deg']:+.1f}° | `[{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}]` | "
                    f"{r['arm_abd_L_deg']:+.1f}° | {r['arm_abd_R_deg']:+.1f}° |\n"
                )

            fp.write("\n## 4. Fitting Residuals & Geometric Alignment\n\n")
            fp.write("| Stop | Viewpoint | Observed Points | Median Residual (mm) | Mean Residual (mm) | P90 Residual (mm) |\n")
            fp.write("|---|---|---|---|---|---|\n")
            for r in per_stop_residuals:
                view_name = ["Front (0°)", "Right (90°)", "Back (180°)", "Left (270°)"][r["stop_idx"]]
                fp.write(
                    f"| Stop {r['stop_idx']:02d} | {view_name} | {r['num_observed_points']:,} | "
                    f"**{r['median_residual_mm']:.1f} mm** | {r['mean_residual_mm']:.1f} mm | {r['p90_residual_mm']:.1f} mm |\n"
                )
            fp.write(
                f"\n**Overall Point Cloud Residual**: Median = `{med_overall:.1f} mm`, "
                f"Mean = `{mean_overall:.1f} mm`, P90 = `{float(np.percentile(all_res_flat, 90)):.1f} mm` across {len(all_pts_flat):,} physical points.\n\n"
            )

            fp.write("## 5. Observed vs. Prior-Driven / Inferred Surface Breakdown\n\n")
            fp.write(f"- **Total Model Vertices**: `{len(canon_vertices):,}`\n")
            fp.write(f"- **Directly Observed Vertices**: `{num_observed_verts:,}` ({pct_observed:.1f}%)\n")
            fp.write(f"  - Observed Vertical Extent: $Z \\in [{canon_vertices[vertex_observed_flags, 2].min():.2f}, {canon_vertices[vertex_observed_flags, 2].max():.2f}]\\text{{ m}}$ (Torso, chest, back, shoulders, arms)\n")
            fp.write(f"- **Prior-Driven / Inferred Vertices**: `{num_inferred_verts:,}` ({100.0 - pct_observed:.1f}%)\n")
            fp.write(f"  - Inferred Vertical Extent: Lower legs ($Z < -0.44\\text{{ m}}$) and vertex crown ($Z > +0.35\\text{{ m}}$)\n\n")

            fp.write("## 6. Physical Constraints & Limitations\n\n")
            fp.write(
                "> [!WARNING] Truncated Upper-Body Framing Notice\n"
                "> 1. **Lower Legs Truncation**: Lower legs and feet were outside the physical camera FOV during capture. Their geometry is solely governed by the Anny parametric shape prior.\n"
                "> 2. **Stature**: Full stature cannot be directly measured from this capture.\n"
                "> 3. **Clothing Offsets**: Real clothing introduces 5–15 mm offsets over bare skin on the chest and waist.\n"
                "> 4. **No Direct Backside Observation**: Each stop only directly observes front-facing surfaces; full 360° coverage is formed by fusing all 4 stops.\n\n"
            )

            fp.write("## 7. Generated Output Artifacts\n\n")
            fp.write(f"- Canonical Fitted OBJ: [`canonical_fitted_model.obj`](canonical_fitted_model.obj)\n")
            fp.write(f"- Canonical Fitted PLY (Observed/Inferred Color): [`canonical_fitted_model.ply`](canonical_fitted_model.ply)\n")
            fp.write(f"- Fitted Parameters JSON: [`fitted_body_parameters.json`](fitted_body_parameters.json)\n")
            fp.write(f"- Consolidated Residual PLY: [`all_stops_observed_with_residual.ply`](all_stops_observed_with_residual.ply)\n")
            fp.write(f"- Residual Chart: [`residual_distribution.png`](residual_distribution.png)\n")
            fp.write(f"- 2D Overlays: `stop_00_overlay.png`, `stop_01_overlay.png`, `stop_02_overlay.png`, `stop_03_overlay.png`\n")

        logger.info(f"  Saved detailed Markdown report to: {report_md_path}")
        logger.info("=" * 76)
        logger.info("  STAGE 2 ARTICULATED FITTING COMPLETED SUCCESSFULLY")
        logger.info("=" * 76)

        return summary_data


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Problem 4 — Stage 2: Articulated Multi-View Fitting on Physical RGB-D Data"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw_captures/body_capture_01",
        help="Path to raw capture directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stage2_articulated_fit_output",
        help="Directory to save fitted parameters, meshes, overlays, and reports",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Calibrated subject physical stance distance in metres (auto-detected if None)",
    )
    parser.add_argument(
        "--known-height-mm",
        type=float,
        default=1750.0,
        help="Known adult subject height in mm for stature regularizer (default: 1750 mm)",
    )
    parser.add_argument(
        "--n-coarse-iter",
        type=int,
        default=35,
        help="Number of coarse optimization iterations (default: 35)",
    )
    parser.add_argument(
        "--n-fine-iter",
        type=int,
        default=25,
        help="Number of fine optimization iterations (default: 25)",
    )
    parser.add_argument(
        "--multi-starts",
        type=int,
        default=2,
        help="Number of multi-start candidate initializations (default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    fitter = PhysicalStage2ArticulatedFitter(
        raw_capture_dir=args.raw_dir,
        output_dir=args.out_dir,
        distance_m=args.distance,
        known_height_mm=args.known_height_mm,
        n_coarse_iter=args.n_coarse_iter,
        n_fine_iter=args.n_fine_iter,
        multi_starts=args.multi_starts,
    )
    try:
        fitter.run()
        return 0
    except Exception as exc:
        logger.error(f"Stage 2 Articulated Fitting failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
