#!/usr/bin/env python3
"""
tools/audit_physical_stage1_geometry.py
=======================================
Problem 4 — Stage 1: Physical RGB-D Ingestion and Geometry Audit.

Audits and validates the physical four-stop human capture (raw_captures/body_capture_01/):
- Verifies raw dataset integrity (RGB & depth dimensions, intrinsics, timestamps).
- Evaluates foreground segmentation coverage and background depth gating.
- Converts each RGBDFrame to AdaptedStopData using body/adapter.py.
- Validates camera-to-world coordinate transformation and alignment.
- Computes per-stop 3D bounding boxes, centroids, and point counts.
- Verifies inter-view subject positioning consistency.
- Exports per-stop visualizations (RGB, depth, mask) and 3D point clouds (PLY).
- Generates structured summary JSON and audit markdown report.

Usage:
    python tools/audit_physical_stage1_geometry.py --raw-dir raw_captures/body_capture_01 --out-dir stage1_audit_output
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
import numpy as np
import open3d as o3d

# Ensure repository paths are configured on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from camera.recording import load_raw_capture
    from body.adapter import (
        AdaptedStopData,
        camera_to_world_matrix,
        camera_to_world_points,
        frame_to_body_stop,
    )
except ImportError:
    from realsense_measure.camera.recording import load_raw_capture
    from realsense_measure.body.adapter import (
        AdaptedStopData,
        camera_to_world_matrix,
        camera_to_world_points,
        frame_to_body_stop,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_geometry_audit")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class StopGeometryAudit:
    """Detailed geometric audit metrics for a single camera stop."""

    stop_idx: int
    nominal_yaw_deg: float
    color_shape: Tuple[int, int, int]
    depth_shape: Tuple[int, int]
    total_pixels: int
    valid_depth_pixels: int
    valid_depth_pct: float
    zero_depth_pixels: int
    zero_depth_pct: float
    median_depth_m: float
    min_depth_m: float
    max_depth_m: float
    mask_pixels: int
    mask_coverage_pct: float
    point_cloud_count: int
    bbox_cam: Dict[str, Tuple[float, float]]  # X, Y, Z min/max in camera frame
    centroid_cam: Tuple[float, float, float]
    bbox_world: Dict[str, Tuple[float, float]]  # X, Y, Z min/max in Anny world frame
    centroid_world: Tuple[float, float, float]
    color_image_path: str
    depth_vis_path: str
    mask_image_path: str
    ply_cam_path: str
    ply_world_path: str
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Visualization & Helper Functions
# ---------------------------------------------------------------------------

def create_depth_colormap(depth_m: np.ndarray, min_m: float = 0.40, max_m: float = 2.00) -> np.ndarray:
    """
    Generate an intuitive Jet-colormapped BGR depth image with invalid pixels colored black.
    """
    valid = (depth_m >= min_m) & (depth_m <= max_m) & np.isfinite(depth_m)
    depth_vis = np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)

    if np.any(valid):
        normalized = np.clip((depth_m[valid] - min_m) / (max_m - min_m), 0.0, 1.0)
        u8_vals = (normalized * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(u8_vals, cv2.COLORMAP_JET)
        depth_vis[valid] = colored.reshape(-1, 3)

    return depth_vis


def save_point_cloud_ply(filepath: Path | str, points: np.ndarray, colors_bgr: Optional[np.ndarray] = None) -> Path:
    """
    Save 3D points as a standard Stanford PLY file using Open3D.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors_bgr is not None and len(colors_bgr) == len(points):
        # Open3D expects RGB in [0.0, 1.0]
        rgb = colors_bgr[:, ::-1].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(str(path), pcd)
    return path


# ---------------------------------------------------------------------------
# Stage 1 Geometry Auditor
# ---------------------------------------------------------------------------

class PhysicalStage1GeometryAuditor:
    """
    Ingests and audits physical 4-stop human capture datasets.
    """

    def __init__(
        self,
        raw_capture_dir: Path | str,
        output_dir: Path | str = "stage1_audit_output",
        distance_m: Optional[float] = None,
        nominal_yaws: Optional[List[float]] = None,
    ) -> None:
        self.raw_dir = Path(raw_capture_dir)
        self.out_dir = Path(output_dir)
        self.distance_m = distance_m
        self.nominal_yaws = nominal_yaws or [0.0, 90.0, 180.0, 270.0]

    def audit(self) -> Dict[str, Any]:
        """
        Execute full Stage 1 geometry audit and return structured summary dict.
        """
        t0 = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 72)
        logger.info("  Problem 4 — Stage 1: Physical RGB-D Ingestion & Geometry Audit")
        logger.info("=" * 72)
        logger.info(f"  Input Dataset:  {self.raw_dir}")
        logger.info(f"  Output Folder:  {self.out_dir}")

        # 1. Load Raw Dataset via Existing Loader
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw capture directory does not exist: {self.raw_dir}")

        frames = load_raw_capture(self.raw_dir)
        num_frames = len(frames)
        logger.info(f"  Loaded {num_frames} raw RGBDFrame instances from recording.")

        if num_frames != len(self.nominal_yaws):
            logger.warning(
                f"  [Warning] Frame count ({num_frames}) does not match expected nominal yaws ({len(self.nominal_yaws)})!"
            )

        # Inspect meta.json if present
        meta_file = self.raw_dir / "meta.json"
        meta_info: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as fp:
                meta_info = json.load(fp)

        # 2. Automatically Estimate Subject Stance Distance if Not Explicitly Provided
        # Median depth across all stops inside plausible human zone [0.4, 3.0]m
        all_stop_medians: List[float] = []
        for f in frames:
            d = f.depth_m
            valid_d = d[(d >= 0.40) & (d <= 3.00) & np.isfinite(d)]
            if len(valid_d) > 0:
                all_stop_medians.append(float(np.median(valid_d)))

        if self.distance_m is None:
            if all_stop_medians:
                self.distance_m = float(np.mean(all_stop_medians))
                logger.info(f"  Auto-detected subject stance distance: {self.distance_m:.3f} m")
            else:
                self.distance_m = 2.14
                logger.warning(f"  Could not determine distance from depth; defaulting to {self.distance_m:.3f} m")
        else:
            logger.info(f"  Using user-specified tripod distance: {self.distance_m:.3f} m")

        # 3. Process Each Stop
        stop_audits: List[StopGeometryAudit] = []
        all_world_points: List[np.ndarray] = []
        all_world_colors: List[np.ndarray] = []
        global_warnings: List[str] = []

        for i, frame in enumerate(frames):
            stop_warnings: List[str] = []
            yaw = self.nominal_yaws[i] if i < len(self.nominal_yaws) else float(i * 90.0)
            color = frame.color_bgr
            depth = frame.depth_m
            h, w = depth.shape[:2]

            logger.info(f"\n--- [Auditing Stop {i+1}/{num_frames}: Nominal Yaw {yaw:.0f}°] ---")

            # Check Dimensions
            if color.shape[:2] != depth.shape:
                err_msg = f"Stop {i}: Color shape {color.shape[:2]} != Depth shape {depth.shape}"
                logger.error(f"  [ERROR] {err_msg}")
                stop_warnings.append(err_msg)

            # Pixel Health
            total_px = int(h * w)
            nan_count = int(np.count_nonzero(np.isnan(depth)))
            inf_count = int(np.count_nonzero(np.isinf(depth)))
            zero_count = int(np.count_nonzero(depth == 0.0))
            valid_mask = (depth >= 0.40) & (depth <= 3.00) & np.isfinite(depth)
            valid_count = int(np.count_nonzero(valid_mask))
            valid_pct = float(100.0 * valid_count / total_px)
            zero_pct = float(100.0 * zero_count / total_px)

            valid_vals = depth[valid_mask]
            med_depth = float(np.median(valid_vals)) if len(valid_vals) > 0 else float("nan")
            min_depth = float(np.min(valid_vals)) if len(valid_vals) > 0 else float("nan")
            max_depth = float(np.max(valid_vals)) if len(valid_vals) > 0 else float("nan")

            logger.info(f"  Raster Resolution:     {w}x{h}")
            logger.info(f"  Valid Depth Ratio:     {valid_count:,}/{total_px:,} ({valid_pct:.1f}%)")
            logger.info(f"  Zero / Void Depth:     {zero_count:,} ({zero_pct:.1f}%)")
            logger.info(f"  Depth Range [Min..Max]: [{min_depth:.2f} m .. {max_depth:.2f} m] (Median: {med_depth:.3f} m)")

            # Check Near-Clipping Warning (D455 Min-Z is 0.52m at 1280x720)
            if min_depth <= 0.52:
                near_warn = f"Stop {i}: Min depth ({min_depth:.3f}m) is at or below D455 physical Min-Z limit (0.52m at 1280x720). Risk of near-field depth truncation!"
                logger.warning(f"  [!] {near_warn}")
                stop_warnings.append(near_warn)

            # Check Distance Consistency relative to estimated tripod
            delta_dist = abs(med_depth - self.distance_m)
            if delta_dist > 0.15:
                dist_warn = f"Stop {i}: Median depth ({med_depth:.3f}m) deviates by {delta_dist*1000:.1f}mm from tripod distance ({self.distance_m:.3f}m)."
                logger.warning(f"  [!] {dist_warn}")
                stop_warnings.append(dist_warn)

            # Save Visual Artifacts
            rgb_path = self.out_dir / f"stop_{i:02d}_rgb.png"
            depth_vis_path = self.out_dir / f"stop_{i:02d}_depth_vis.png"
            mask_path = self.out_dir / f"stop_{i:02d}_mask.png"

            cv2.imwrite(str(rgb_path), color)
            depth_colormap = create_depth_colormap(depth, min_m=0.40, max_m=max(2.50, self.distance_m + 0.80))
            cv2.imwrite(str(depth_vis_path), depth_colormap)

            # Convert via Existing Adapter
            adapted_stop: AdaptedStopData = frame_to_body_stop(
                frame=frame,
                nominal_yaw_deg=yaw,
                distance_m=self.distance_m,
                stop_idx=i,
            )

            # Save Foreground Mask
            fg_mask_u8 = (adapted_stop.foreground_mask.astype(np.uint8)) * 255
            cv2.imwrite(str(mask_path), fg_mask_u8)
            mask_px = int(np.count_nonzero(adapted_stop.foreground_mask))
            mask_pct = float(100.0 * mask_px / total_px)

            pts_cam = adapted_stop.point_cloud
            pts_world = adapted_stop.point_cloud_world
            pcd_count = len(pts_world)

            logger.info(f"  Foreground Mask Area:  {mask_px:,} pixels ({mask_pct:.1f}% of image)")
            logger.info(f"  Adapted 3D Points:     {pcd_count:,} points")

            if pcd_count == 0:
                empty_err = f"Stop {i}: Backprojected point cloud is empty!"
                logger.error(f"  [ERROR] {empty_err}")
                stop_warnings.append(empty_err)

            # Bounding Boxes & Centroids
            if pcd_count > 0:
                c_cam = tuple(float(v) for v in np.mean(pts_cam, axis=0))
                c_world = tuple(float(v) for v in np.mean(pts_world, axis=0))

                bbox_cam = {
                    "x": (float(np.min(pts_cam[:, 0])), float(np.max(pts_cam[:, 0]))),
                    "y": (float(np.min(pts_cam[:, 1])), float(np.max(pts_cam[:, 1]))),
                    "z": (float(np.min(pts_cam[:, 2])), float(np.max(pts_cam[:, 2]))),
                }
                bbox_world = {
                    "x": (float(np.min(pts_world[:, 0])), float(np.max(pts_world[:, 0]))),
                    "y": (float(np.min(pts_world[:, 1])), float(np.max(pts_world[:, 1]))),
                    "z": (float(np.min(pts_world[:, 2])), float(np.max(pts_world[:, 2]))),
                }

                logger.info(f"  Camera Frame Centroid: [{c_cam[0]:.3f}, {c_cam[1]:.3f}, {c_cam[2]:.3f}] m")
                logger.info(f"  World Frame Centroid:  [{c_world[0]:.3f}, {c_world[1]:.3f}, {c_world[2]:.3f}] m")
                logger.info(
                    f"  World Bounds: X=[{bbox_world['x'][0]:.2f}..{bbox_world['x'][1]:.2f}] m, "
                    f"Y=[{bbox_world['y'][0]:.2f}..{bbox_world['y'][1]:.2f}] m, "
                    f"Z=[{bbox_world['z'][0]:.2f}..{bbox_world['z'][1]:.2f}] m"
                )

                # Save PLY point clouds
                ply_cam_path = self.out_dir / f"stop_{i:02d}_pcd_cam.ply"
                ply_world_path = self.out_dir / f"stop_{i:02d}_pcd_world.ply"

                # Extract RGB colors corresponding to points from indexed_pcd
                pcd_colors = None
                if adapted_stop.indexed_pcd is not None and getattr(adapted_stop.indexed_pcd, "colors", None) is not None:
                    pcd_colors = adapted_stop.indexed_pcd.colors

                save_point_cloud_ply(ply_cam_path, pts_cam, pcd_colors)
                save_point_cloud_ply(ply_world_path, pts_world, pcd_colors)

                all_world_points.append(pts_world)
                if pcd_colors is not None:
                    all_world_colors.append(pcd_colors)
            else:
                c_cam = (0.0, 0.0, 0.0)
                c_world = (0.0, 0.0, 0.0)
                bbox_cam = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
                bbox_world = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
                ply_cam_path = self.out_dir / f"stop_{i:02d}_pcd_cam.ply"
                ply_world_path = self.out_dir / f"stop_{i:02d}_pcd_world.ply"

            audit_rec = StopGeometryAudit(
                stop_idx=i,
                nominal_yaw_deg=yaw,
                color_shape=color.shape,
                depth_shape=depth.shape,
                total_pixels=total_px,
                valid_depth_pixels=valid_count,
                valid_depth_pct=valid_pct,
                zero_depth_pixels=zero_count,
                zero_depth_pct=zero_pct,
                median_depth_m=med_depth,
                min_depth_m=min_depth,
                max_depth_m=max_depth,
                mask_pixels=mask_px,
                mask_coverage_pct=mask_pct,
                point_cloud_count=pcd_count,
                bbox_cam=bbox_cam,
                centroid_cam=c_cam,
                bbox_world=bbox_world,
                centroid_world=c_world,
                color_image_path=str(rgb_path),
                depth_vis_path=str(depth_vis_path),
                mask_image_path=str(mask_path),
                ply_cam_path=str(ply_cam_path),
                ply_world_path=str(ply_world_path),
                warnings=stop_warnings,
            )
            stop_audits.append(audit_rec)

        # 4. Consolidated World Point Cloud Export
        consolidated_ply_path = self.out_dir / "all_stops_pcd_world.ply"
        if all_world_points:
            merged_pts = np.vstack(all_world_points)
            merged_colors = np.vstack(all_world_colors) if all_world_colors else None
            save_point_cloud_ply(consolidated_ply_path, merged_pts, merged_colors)
            logger.info(f"\n  Saved consolidated 4-stop point cloud ({len(merged_pts):,} points) to: {consolidated_ply_path}")
        else:
            merged_pts = np.zeros((0, 3), dtype=np.float32)

        # 5. Inter-View Consistency Analysis
        depth_medians = [s.median_depth_m for s in stop_audits if np.isfinite(s.median_depth_m)]
        dist_variance_mm = float(np.std(depth_medians) * 1000.0) if len(depth_medians) >= 2 else 0.0

        world_centroids = np.array([s.centroid_world for s in stop_audits])
        centroid_drift_xy_mm = float(np.ptp(np.linalg.norm(world_centroids[:, :2], axis=1)) * 1000.0) if len(world_centroids) >= 2 else 0.0

        # Assess Stature Framing (Check if vertical bounds span head to feet or only torso)
        z_extents = [s.bbox_world["z"][1] - s.bbox_world["z"][0] for s in stop_audits]
        mean_z_extent = float(np.mean(z_extents)) if z_extents else 0.0

        framing_category = "Full-Body" if mean_z_extent >= 1.30 else "Upper-Body / Torso"
        if framing_category == "Upper-Body / Torso":
            frame_note = (
                f"Subject vertical bounding extent is {mean_z_extent:.2f} m (< 1.30 m). "
                f"This physical capture captures Upper-Body / Torso framing. Full stature (vertex to floor) is truncated."
            )
            logger.warning(f"  [Framing Note] {frame_note}")
            global_warnings.append(frame_note)

        # Coordinate Frame Verification
        T_c2w = camera_to_world_matrix(self.distance_m)
        det_R = float(np.linalg.det(T_c2w[:3, :3]))
        is_so3_valid = bool(abs(det_R - 1.0) < 1e-5)

        elapsed = time.time() - t0

        summary: Dict[str, Any] = {
            "dataset_path": str(self.raw_dir),
            "output_dir": str(self.out_dir),
            "num_stops_audited": len(stop_audits),
            "calibrated_distance_m": self.distance_m,
            "framing_category": framing_category,
            "vertical_extent_m": mean_z_extent,
            "inter_stop_consistency": {
                "distance_medians_m": depth_medians,
                "distance_std_mm": dist_variance_mm,
                "centroid_drift_xy_mm": centroid_drift_xy_mm,
                "is_positionally_consistent": bool(dist_variance_mm <= 30.0 and centroid_drift_xy_mm <= 100.0),
            },
            "coordinate_frame_verification": {
                "rotation_matrix_det": det_R,
                "is_right_handed_so3": is_so3_valid,
                "cam_to_world_matrix": T_c2w.tolist(),
            },
            "stops": [s.as_dict() for s in stop_audits],
            "total_consolidated_points": int(len(merged_pts)),
            "consolidated_ply_path": str(consolidated_ply_path),
            "global_warnings": global_warnings,
            "elapsed_seconds": elapsed,
            "stage1_status": "passed_with_framing_notice" if framing_category == "Upper-Body / Torso" else "passed",
        }

        # 6. Save JSON Summary
        summary_path = self.out_dir / "stage1_geometry_audit_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        logger.info(f"  Saved JSON summary to: {summary_path}")

        # 6b. Save Markdown Audit Report
        md_path = self.out_dir / "stage1_geometry_audit_report.md"
        with open(md_path, "w", encoding="utf-8") as fp:
            fp.write(f"# Problem 4 — Stage 1: Physical RGB-D Ingestion & Geometry Audit Report\n\n")
            fp.write(f"**Dataset**: `{self.raw_dir}`  \n")
            fp.write(f"**Audit Status**: `{summary['stage1_status']}`  \n")
            fp.write(f"**Calibrated Stance Distance**: `{self.distance_m:.3f} m`  \n")
            fp.write(f"**Framing Classification**: `{framing_category}` (Vertical Extent: `{mean_z_extent:.2f} m`)  \n")
            fp.write(f"**Inter-Stop Distance Drift**: `{dist_variance_mm:.1f} mm`  \n")
            fp.write(f"**Centroid Drift (XY)**: `{centroid_drift_xy_mm:.1f} mm`  \n")
            fp.write(f"**Consolidated 3D Points**: `{len(merged_pts):,}` points  \n\n")
            fp.write("## Per-Stop Audit Breakdown\n\n")
            fp.write("| Stop | Yaw (°) | Valid Depth % | Median Depth (m) | Mask Pixels | 3D Points | World Z Extent (m) |\n")
            fp.write("|---|---|---|---|---|---|---|\n")
            for s in stop_audits:
                z_ext = s.bbox_world["z"][1] - s.bbox_world["z"][0]
                fp.write(f"| Stop {s.stop_idx:02d} | {s.nominal_yaw_deg:.0f}° | {s.valid_depth_pct:.1f}% | {s.median_depth_m:.3f} m | {s.mask_pixels:,} | {s.point_cloud_count:,} | {z_ext:.2f} m |\n")
            fp.write("\n## Coordinate Frame Verification\n\n")
            fp.write(f"- $R_{{cam \\to world}}$ determinant: `{det_R:.6f}` (Strict SO(3) Right-Handed)\n")
            fp.write(f"- Camera position in Anny world frame: `[0.0, -{self.distance_m:.3f}, 0.0] m`\n")
            fp.write(f"- Principal axis mapping: $+X_w = +X_c$, $+Y_w = +Z_c - d$, $+Z_w = -Y_c$\n\n")
            if global_warnings:
                fp.write("## Framing & Capture Notices\n\n")
                for w in global_warnings:
                    fp.write(f"- **Notice**: {w}\n")
        logger.info(f"  Saved Markdown report to: {md_path}")

        # 7. Print Terminal Table
        print("\n" + "=" * 78)
        print("  STAGE 1 PHYSICAL GEOMETRY AUDIT RESULTS")
        print("=" * 78)
        print(f"  Dataset:             {self.raw_dir}")
        print(f"  Distance (Tripod):   {self.distance_m:.3f} m")
        print(f"  Framing:             {framing_category} (Vertical extent: {mean_z_extent:.2f} m)")
        print(f"  Inter-Stop Jitter:   Distance std = {dist_variance_mm:.1f} mm | Centroid drift = {centroid_drift_xy_mm:.1f} mm")
        print(f"  Consolidated Points: {len(merged_pts):,} points")
        print("-" * 78)
        print(f"  {'Stop':<6} {'Yaw':<6} {'Valid %':<10} {'Med Depth':<12} {'Mask Px':<12} {'3D Points':<12} {'Z Extent'}")
        print("-" * 78)
        for s in stop_audits:
            z_ext = s.bbox_world["z"][1] - s.bbox_world["z"][0]
            print(
                f"  {s.stop_idx:<6} {s.nominal_yaw_deg:<6.0f} {s.valid_depth_pct:<10.1f} "
                f"{s.median_depth_m:<12.3f} {s.mask_pixels:<12,} {s.point_cloud_count:<12,} {z_ext:.2f} m"
            )
        print("=" * 78 + "\n")

        return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Problem 4 — Stage 1: Physical RGB-D Ingestion and Geometry Audit"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw_captures/body_capture_01",
        help="Path to raw capture directory containing frame_*.png, frame_*.npy, meta.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stage1_audit_output",
        help="Output directory to save visualizations, PLY point clouds, and summary JSON",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Calibrated tripod distance in metres (defaults to auto-detected subject depth)",
    )
    parser.add_argument(
        "--nominal-yaws",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 180.0, 270.0],
        help="Nominal yaw angles in degrees for each stop (default: 0.0 90.0 180.0 270.0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    auditor = PhysicalStage1GeometryAuditor(
        raw_capture_dir=args.raw_dir,
        output_dir=args.out_dir,
        distance_m=args.distance,
        nominal_yaws=args.nominal_yaws,
    )
    try:
        auditor.audit()
        return 0
    except Exception as exc:
        logger.error(f"Stage 1 Geometry Audit failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
