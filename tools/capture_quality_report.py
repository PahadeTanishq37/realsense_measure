"""
capture_quality_report.py - Standalone capture quality diagnostic tool for recorded RGB-D scans.

Analyzes raw capture recordings without live camera dependencies.
Computes per-frame depth completeness, range-binned valid depth pixel counts,
person segmentation mask statistics, facial landmark detections, inter-frame time gaps,
separated background/subject motion estimation, and odometry fitness metrics.

Guarantees:
  - Per-frame statistics are computed FIRST from untouched raw arrays before any motion processing.
  - No arrays (depth_m, color_bgr, masks) are mutated in place (explicit copies used for odometry).
  - Unreliable motion (poor odometry fitness < 0.50 or subject mask zero-depth > 50%) is marked 'unreliable'.
  - Outputs per-frame odometry fitness in CSV and console summary table.

Generates:
  - CSV report
  - Clean console summary table
  - Diagnostic side-by-side PNGs for flagged frames
  - Diagnostic reason summary
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

# Ensure repository root and package are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(PACKAGE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from camera.recording import load_raw_capture
except ImportError:
    from realsense_measure.camera.recording import load_raw_capture

try:
    from segmentation import get_person_mask
    SEGMENTATION_AVAILABLE = True
except Exception as e:
    SEGMENTATION_AVAILABLE = False
    _seg_err = e

try:
    import landmarks
    LANDMARKS_AVAILABLE = True
except Exception as e:
    LANDMARKS_AVAILABLE = False
    _landmarks_err = e

try:
    from registration import compute_rotation_deg, compute_translation_mm
except Exception:
    def compute_translation_mm(T: np.ndarray) -> float:
        return float(np.linalg.norm(T[:3, 3]) * 1000.0)

    def compute_rotation_deg(R: np.ndarray) -> float:
        trace = float(np.trace(R[:3, :3]))
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cos_theta)))


def colorize_depth(depth_m: np.ndarray, min_d: float = 0.2, max_d: float = 1.2) -> np.ndarray:
    """Produce a colorized depth map where zero depth is pure black and valid values map via TURBO."""
    valid_mask = (depth_m > 0) & np.isfinite(depth_m)
    clipped = np.clip(depth_m, min_d, max_d)
    norm = ((clipped - min_d) / max(max_d - min_d, 1e-6) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = [0, 0, 0]
    return colored


def overlay_mask_outline(
    color_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Overlay green boundary contour of person mask onto color image."""
    annotated = color_bgr.copy()
    if mask is not None and np.any(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)
    return annotated


def evaluate_odometry_fitness(
    rgbd_prev: o3d.geometry.RGBDImage,
    rgbd_curr: o3d.geometry.RGBDImage,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    trans: np.ndarray,
    voxel_size_m: float = 0.01,
    max_corresp_m: float = 0.05,
) -> float:
    """Evaluate inlier correspondence ratio (fitness) of pairwise RGB-D odometry transformation."""
    try:
        pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_prev, intrinsics)
        pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_curr, intrinsics)
        if len(pcd_prev.points) == 0 or len(pcd_curr.points) == 0:
            return 0.0
        pcd_prev = pcd_prev.voxel_down_sample(voxel_size_m)
        pcd_curr = pcd_curr.voxel_down_sample(voxel_size_m)
        eval_res = o3d.pipelines.registration.evaluate_registration(
            pcd_prev, pcd_curr, max_corresp_m, trans
        )
        return float(eval_res.fitness)
    except Exception:
        return 0.0


def analyze_capture(
    recording_dir: Path | str,
    out_dir: Path | str,
    csv_name: str = "capture_quality.csv",
    max_mask_zero_depth_frac: float = 0.3,
    min_valid_person_pixels: int = 3000,
    range_near_min: float = 0.20,
    range_subject_min: float = 0.40,
    range_subject_max: float = 0.80,
    min_subject_mask_pixels: int = 50000,
    d455_min_range_m: float = 0.50,
    max_motion_trans_mm: float = 120.0,
    max_motion_rot_deg: float = 15.0,
    min_odometry_fitness: float = 0.50,
    max_mask_zero_depth_for_motion: float = 0.50,
) -> None:
    recording_dir = Path(recording_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw capture from: {recording_dir.resolve()}")
    frames = load_raw_capture(recording_dir)
    print(f"Loaded {len(frames)} frames.")

    if not SEGMENTATION_AVAILABLE:
        print("[WARNING] segmentation.get_person_mask is unavailable. Person mask columns will be blank.")
    if not LANDMARKS_AVAILABLE:
        print("[WARNING] landmarks module is unavailable. Face detection columns will be blank/no.")

    # =========================================================================
    # STEP 1: Compute all per-frame statistics FIRST from untouched raw arrays.
    # We guarantee that per-frame stats (depth zero fractions, range counts,
    # person mask stats, landmark detections) are evaluated before any motion
    # or odometry processing is executed.
    # =========================================================================
    per_frame_stats: List[Dict[str, Any]] = []

    for i, frame in enumerate(frames):
        # Work on pristine, untouched raw arrays
        depth = frame.depth_m
        total_pixels = depth.size
        zero_depth_pixels = int(np.sum(depth == 0))
        whole_zero_depth_frac = float(zero_depth_pixels / total_pixels)

        # Depth range counts
        valid_0_40_0_80 = int(np.sum((depth >= range_subject_min) & (depth <= range_subject_max)))
        valid_0_20_0_40 = int(np.sum((depth >= range_near_min) & (depth < range_subject_min)))
        valid_gt_0_80 = int(np.sum(depth > range_subject_max))

        # Person mask
        mask: Optional[np.ndarray] = None
        mask_pixel_count: Optional[int] = None
        mask_zero_depth_frac: Optional[float] = None
        valid_person_pixels: Optional[int] = None
        median_person_depth: Optional[float] = None
        min_person_depth: Optional[float] = None

        if SEGMENTATION_AVAILABLE:
            try:
                # Pass a copy of color_bgr to protect input frame
                mask = get_person_mask(frame.color_bgr.copy())
            except Exception:
                mask = None

            if mask is not None:
                mask_pixel_count = int(np.sum(mask))
                if mask_pixel_count > 0:
                    person_depths = depth[mask]
                    person_zeros = int(np.sum(person_depths == 0))
                    mask_zero_depth_frac = float(person_zeros / mask_pixel_count)
                    valid_depths = person_depths[person_depths > 0]
                    valid_person_pixels = int(len(valid_depths))
                    if valid_person_pixels > 0:
                        median_person_depth = float(np.median(valid_depths))
                        min_person_depth = float(np.min(valid_depths))
                    else:
                        median_person_depth = None
                        min_person_depth = None
                else:
                    mask_zero_depth_frac = 0.0
                    valid_person_pixels = 0
            else:
                mask_pixel_count = None
                mask_zero_depth_frac = None
                valid_person_pixels = None

        # Landmarks
        face_detected_str = "no"
        num_landmarks = 0
        if LANDMARKS_AVAILABLE and hasattr(frame, "intrinsics"):
            try:
                lms_res = landmarks.detect_face_landmarks_3d(
                    frame.color_bgr.copy(), frame.depth_m.copy(), frame.intrinsics, quiet=True
                )
                if lms_res is not None:
                    face_detected_str = "yes"
                    num_landmarks = len(lms_res)
            except Exception:
                face_detected_str = "no"
                num_landmarks = 0

        per_frame_stats.append({
            "frame": frame,
            "depth_zero_frac": whole_zero_depth_frac,
            "valid_0_40_0_80": valid_0_40_0_80,
            "valid_0_20_0_40": valid_0_20_0_40,
            "valid_gt_0_80": valid_gt_0_80,
            "mask": mask,
            "mask_pixel_count": mask_pixel_count,
            "mask_zero_depth_frac": mask_zero_depth_frac,
            "valid_person_pixels": valid_person_pixels,
            "median_person_depth": median_person_depth,
            "min_person_depth": min_person_depth,
            "face_detected_str": face_detected_str,
            "num_landmarks": num_landmarks,
        })

    # =========================================================================
    # STEP 2: Pairwise inter-frame motion estimation with fitness evaluation.
    # Explicitly work on copies of depth and color arrays to ensure zero in-place mutation.
    # If odometry fitness is poor (< min_odometry_fitness) or subject mask zero-depth > 50%,
    # mark the motion value as 'unreliable'.
    # =========================================================================
    rows: List[Dict[str, Any]] = []
    bad_frames_info: List[Dict[str, Any]] = []

    for i in range(len(frames)):
        stats = per_frame_stats[i]
        frame = stats["frame"]
        mask = stats["mask"]
        mask_pixel_count = stats["mask_pixel_count"]
        mask_zero_depth_frac = stats["mask_zero_depth_frac"]
        valid_person_pixels = stats["valid_person_pixels"]
        median_person_depth = stats["median_person_depth"]
        min_person_depth = stats["min_person_depth"]
        face_detected_str = stats["face_detected_str"]
        num_landmarks = stats["num_landmarks"]
        whole_zero_depth_frac = stats["depth_zero_frac"]
        valid_0_40_0_80 = stats["valid_0_40_0_80"]
        valid_0_20_0_40 = stats["valid_0_20_0_40"]
        valid_gt_0_80 = stats["valid_gt_0_80"]

        time_gap_s: Optional[float] = None
        cam_motion_bg_str: str = "N/A"
        subj_motion_str: str = "N/A"
        cam_fit_bg: Optional[float] = None
        subj_fit: Optional[float] = None
        cam_trans_bg_mm: Optional[float] = None
        cam_rot_bg_deg: Optional[float] = None
        subj_trans_mm: Optional[float] = None
        subj_rot_deg: Optional[float] = None

        if i > 0:
            prev_stats = per_frame_stats[i - 1]
            prev_frame = prev_stats["frame"]
            prev_mask = prev_stats["mask"]
            time_gap_s = float(frame.timestamp - prev_frame.timestamp)

            # (a) Background-only motion (camera motion relative to static background)
            # Create fresh copies to guarantee no in-place mutation of frame arrays
            d_bg_prev = prev_frame.depth_m.copy()
            d_bg_curr = frame.depth_m.copy()
            if prev_mask is not None:
                d_bg_prev[prev_mask] = 0.0
            if mask is not None:
                d_bg_curr[mask] = 0.0

            try:
                rgbd_bg_prev = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(prev_frame.color_bgr.copy()),
                    o3d.geometry.Image(d_bg_prev),
                    depth_scale=1.0,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False,
                )
                rgbd_bg_curr = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(frame.color_bgr.copy()),
                    o3d.geometry.Image(d_bg_curr),
                    depth_scale=1.0,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False,
                )
                s_bg, trans_bg, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_bg_prev, rgbd_bg_curr, frame.intrinsics
                )
                if s_bg:
                    cam_fit_bg = evaluate_odometry_fitness(rgbd_bg_prev, rgbd_bg_curr, frame.intrinsics, trans_bg)
                    cam_trans_bg_mm = compute_translation_mm(trans_bg)
                    cam_rot_bg_deg = compute_rotation_deg(trans_bg[:3, :3])
                    if cam_fit_bg < min_odometry_fitness:
                        cam_motion_bg_str = "unreliable"
                    else:
                        cam_motion_bg_str = f"{cam_trans_bg_mm:.1f}mm, {cam_rot_bg_deg:.1f}deg"
                else:
                    cam_fit_bg = 0.0
                    cam_motion_bg_str = "unreliable"
            except Exception:
                cam_fit_bg = 0.0
                cam_motion_bg_str = "unreliable"

            # (b) Subject-mask-only motion (subject motion relative to camera)
            # Create fresh copies to guarantee no in-place mutation of frame arrays
            d_sub_prev = prev_frame.depth_m.copy()
            d_sub_curr = frame.depth_m.copy()
            if prev_mask is not None:
                d_sub_prev[~prev_mask] = 0.0
            if mask is not None:
                d_sub_curr[~mask] = 0.0

            try:
                rgbd_sub_prev = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(prev_frame.color_bgr.copy()),
                    o3d.geometry.Image(d_sub_prev),
                    depth_scale=1.0,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False,
                )
                rgbd_sub_curr = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(frame.color_bgr.copy()),
                    o3d.geometry.Image(d_sub_curr),
                    depth_scale=1.0,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False,
                )
                s_sub, trans_sub, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_sub_prev, rgbd_sub_curr, frame.intrinsics
                )
                if s_sub:
                    subj_fit = evaluate_odometry_fitness(rgbd_sub_prev, rgbd_sub_curr, frame.intrinsics, trans_sub)
                    subj_trans_mm = compute_translation_mm(trans_sub)
                    subj_rot_deg = compute_rotation_deg(trans_sub[:3, :3])
                    # Poor odometry fitness or >50% zero-depth in subject mask marks motion unreliable
                    has_high_mask_zeros = (mask_zero_depth_frac is not None and mask_zero_depth_frac > max_mask_zero_depth_for_motion)
                    if subj_fit < min_odometry_fitness or has_high_mask_zeros:
                        subj_motion_str = "unreliable"
                    else:
                        subj_motion_str = f"{subj_trans_mm:.1f}mm, {subj_rot_deg:.1f}deg"
                else:
                    subj_fit = 0.0
                    subj_motion_str = "unreliable"
            except Exception:
                subj_fit = 0.0
                subj_motion_str = "unreliable"

        # Check diagnostic save criteria
        flag_high_zero_frac = (mask_zero_depth_frac is not None and mask_zero_depth_frac > max_mask_zero_depth_frac)
        flag_low_valid_person = (valid_person_pixels is not None and valid_person_pixels < min_valid_person_pixels)
        should_save_png = flag_high_zero_frac or flag_low_valid_person

        saved_png_path = ""
        if should_save_png:
            color_annotated = overlay_mask_outline(frame.color_bgr, mask)
            depth_color = colorize_depth(frame.depth_m)
            side_by_side = np.hstack([color_annotated, depth_color])

            banner = np.zeros((40, side_by_side.shape[1], 3), dtype=np.uint8)
            info_txt = (
                f"Frame {frame.frame_id:02d} | Mask Px: {mask_pixel_count or 0} | "
                f"Mask 0-Depth: {(mask_zero_depth_frac or 0.0):.1%} | "
                f"Valid Person Px: {valid_person_pixels or 0} | Face: {face_detected_str} ({num_landmarks})"
            )
            cv2.putText(banner, info_txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            diag_img = np.vstack([banner, side_by_side])

            png_name = f"frame_{frame.frame_id:04d}_diagnostic.png"
            out_png = out_dir / png_name
            cv2.imwrite(str(out_png), diag_img)
            saved_png_path = str(out_png.name)

        # Reasons for looking bad
        reasons = []
        if mask_pixel_count is not None and mask_pixel_count < min_subject_mask_pixels:
            reasons.append("subject out of frame")

        if flag_high_zero_frac or flag_low_valid_person:
            if min_person_depth is not None and min_person_depth <= (d455_min_range_m + 0.01):
                reasons.append("too close")
            elif median_person_depth is not None and median_person_depth < range_subject_min:
                reasons.append("too close")
            elif median_person_depth is not None and median_person_depth > range_subject_max:
                reasons.append("too far")
            elif (cam_trans_bg_mm is not None and cam_trans_bg_mm > max_motion_trans_mm and cam_motion_bg_str != "unreliable") or \
                 (cam_rot_bg_deg is not None and cam_rot_bg_deg > max_motion_rot_deg and cam_motion_bg_str != "unreliable"):
                reasons.append("rapid camera motion")
            elif (subj_trans_mm is not None and subj_trans_mm > max_motion_trans_mm and subj_motion_str != "unreliable") or \
                 (subj_rot_deg is not None and subj_rot_deg > max_motion_rot_deg and subj_motion_str != "unreliable"):
                reasons.append("rapid subject motion")
            else:
                reasons.append("motion blur/low valid depth")

        if face_detected_str == "no":
            reasons.append("face not detected")

        if reasons:
            bad_frames_info.append({
                "frame_id": frame.frame_id,
                "reasons": reasons,
                "zero_frac": mask_zero_depth_frac,
                "valid_person": valid_person_pixels,
                "median_depth": median_person_depth,
                "min_depth": min_person_depth,
                "face": face_detected_str,
                "cam_motion_bg": cam_motion_bg_str,
                "subj_motion": subj_motion_str,
            })

        rows.append({
            "frame_id": frame.frame_id,
            "timestamp": f"{frame.timestamp:.4f}",
            "depth_zero_frac": f"{whole_zero_depth_frac:.4f}",
            "valid_depth_0.40_0.80m": valid_0_40_0_80,
            "valid_depth_0.20_0.40m": valid_0_20_0_40,
            "valid_depth_gt_0.80m": valid_gt_0_80,
            "person_mask_pixel_count": mask_pixel_count if mask_pixel_count is not None else "",
            "person_mask_zero_depth_frac": f"{mask_zero_depth_frac:.4f}" if mask_zero_depth_frac is not None else "",
            "person_median_depth_m": f"{median_person_depth:.4f}" if median_person_depth is not None else "",
            "person_min_depth_m": f"{min_person_depth:.4f}" if min_person_depth is not None else "",
            "face_detected": face_detected_str,
            "num_landmarks": num_landmarks,
            "time_gap_s": f"{time_gap_s:.4f}" if time_gap_s is not None else "",
            "camera_motion_bg": cam_motion_bg_str,
            "camera_motion_bg_fitness": f"{cam_fit_bg:.4f}" if cam_fit_bg is not None else "",
            "subject_motion": subj_motion_str,
            "subject_motion_fitness": f"{subj_fit:.4f}" if subj_fit is not None else "",
            "diagnostic_png": saved_png_path,
        })

    # Write CSV
    csv_path = out_dir / csv_name
    fieldnames = [
        "frame_id", "timestamp", "depth_zero_frac",
        "valid_depth_0.40_0.80m", "valid_depth_0.20_0.40m", "valid_depth_gt_0.80m",
        "person_mask_pixel_count", "person_mask_zero_depth_frac",
        "person_median_depth_m", "person_min_depth_m",
        "face_detected", "num_landmarks",
        "time_gap_s",
        "camera_motion_bg", "camera_motion_bg_fitness",
        "subject_motion", "subject_motion_fitness",
        "diagnostic_png"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV report to: {csv_path.resolve()}")

    # Print Console Table
    header_fmt = (
        "{:<5} | {:<12} | {:<8} | {:<8} | {:<8} | {:<7} | {:<7} | {:<9} | {:<8} | {:<8} | {:<4} | {:<3} | {:<6} | {:<18} | {:<7} | {:<18} | {:<7}"
    )
    row_fmt = (
        "{:<5} | {:<12} | {:<8} | {:<8} | {:<8} | {:<7} | {:<7} | {:<9} | {:<8} | {:<8} | {:<4} | {:<3} | {:<6} | {:<18} | {:<7} | {:<18} | {:<7}"
    )
    print("\n" + "=" * 185)
    print("CAPTURE QUALITY REPORT")
    print("=" * 185)
    print(header_fmt.format(
        "Frame", "Time(s)", "All0Frac", "0.4-0.8m", "0.2-0.4m", ">0.8m",
        "MaskPx", "Mask0Frac", "MedDepth", "MinDepth", "Face", "Lms", "dt(s)",
        "Cam(BG)", "CamFit", "Subj", "SubjFit"
    ))
    print("-" * 185)
    for r in rows:
        cam_fit_disp = f"{float(r['camera_motion_bg_fitness']):.2f}" if r["camera_motion_bg_fitness"] else "-"
        sub_fit_disp = f"{float(r['subject_motion_fitness']):.2f}" if r["subject_motion_fitness"] else "-"
        print(row_fmt.format(
            r["frame_id"],
            r["timestamp"],
            r["depth_zero_frac"][:6],
            r["valid_depth_0.40_0.80m"],
            r["valid_depth_0.20_0.40m"],
            r["valid_depth_gt_0.80m"],
            r["person_mask_pixel_count"] or "-",
            (r["person_mask_zero_depth_frac"][:6] if r["person_mask_zero_depth_frac"] else "-"),
            (r["person_median_depth_m"][:6] if r["person_median_depth_m"] else "-"),
            (r["person_min_depth_m"][:6] if r["person_min_depth_m"] else "-"),
            r["face_detected"],
            r["num_landmarks"],
            (r["time_gap_s"][:5] if r["time_gap_s"] else "-"),
            r["camera_motion_bg"],
            cam_fit_disp,
            r["subject_motion"],
            sub_fit_disp,
        ))
    print("=" * 185)

    # Print Quality Summary
    print("\n" + "=" * 85)
    print("QUALITY SUMMARY & DIAGNOSTIC FINDINGS")
    print("=" * 85)
    print(f"Thresholds applied:")
    print(f"  --max-mask-zero-depth-frac       : {max_mask_zero_depth_frac}")
    print(f"  --min-valid-person-pixels        : {min_valid_person_pixels}")
    print(f"  --range-subject-min              : {range_subject_min:.2f} m")
    print(f"  --range-subject-max              : {range_subject_max:.2f} m")
    print(f"  --d455-min-range-m               : {d455_min_range_m:.2f} m")
    print(f"  --max-motion-trans-mm            : {max_motion_trans_mm:.1f} mm")
    print(f"  --max-motion-rot-deg             : {max_motion_rot_deg:.1f} deg")
    print(f"  --min-odometry-fitness           : {min_odometry_fitness:.2f}")
    print(f"  --max-mask-zero-depth-for-motion : {max_mask_zero_depth_for_motion:.2f}\n")

    if not bad_frames_info:
        print("All frames passed capture quality checks!")
    else:
        print(f"Flagged {len(bad_frames_info)} frames with potential quality issues:")
        for bf in bad_frames_info:
            fid = bf["frame_id"]
            reasons_str = ", ".join(bf["reasons"])
            z_frac_str = f"zero_frac={bf['zero_frac']:.1%}" if bf["zero_frac"] is not None else "no_mask"
            val_px_str = f"valid_px={bf['valid_person']}" if bf["valid_person"] is not None else ""
            med_d_str = f"med_d={bf['median_depth']:.3f}m" if bf["median_depth"] is not None else ""
            min_d_str = f"min_d={bf['min_depth']:.3f}m" if bf["min_depth"] is not None else ""
            face_str = f"face={bf['face']}"
            mot_str = f"cam_bg={bf['cam_motion_bg']}, subj={bf['subj_motion']}"
            print(f"  - Frame {fid:02d}: [{reasons_str}] ({z_frac_str}, {val_px_str}, {med_d_str}, {min_d_str}, {face_str}, {mot_str})")

    print(f"\nDiagnostic side-by-side images saved to: {out_dir.resolve()}")
    print("=" * 85 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Quality Diagnostic Report for recorded RealSense scans."
    )
    parser.add_argument(
        "recording_dir",
        type=str,
        help="Path to directory containing recorded raw capture (with meta.json, PNGs, and NPYs).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="capture_quality_report",
        help="Directory where CSV report and diagnostic images will be saved.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="capture_quality.csv",
        help="Name of output CSV file.",
    )
    parser.add_argument(
        "--max-mask-zero-depth-frac",
        type=float,
        default=0.30,
        help="Threshold of zero-depth fraction in person mask above which frame is flagged (default: 0.30).",
    )
    parser.add_argument(
        "--min-valid-person-pixels",
        type=int,
        default=3000,
        help="Threshold of valid depth pixels in person mask below which frame is flagged (default: 3000).",
    )
    parser.add_argument(
        "--range-near-min",
        type=float,
        default=0.20,
        help="Near depth boundary in meters (default: 0.20).",
    )
    parser.add_argument(
        "--range-subject-min",
        type=float,
        default=0.40,
        help="Subject depth range minimum in meters (default: 0.40).",
    )
    parser.add_argument(
        "--range-subject-max",
        type=float,
        default=0.80,
        help="Subject depth range maximum in meters (default: 0.80).",
    )
    parser.add_argument(
        "--min-subject-mask-pixels",
        type=int,
        default=50000,
        help="Minimum mask pixel count before flagging subject as out of frame (default: 50000).",
    )
    parser.add_argument(
        "--d455-min-range-m",
        type=float,
        default=0.50,
        help="D455 minimum range cutoff in meters below which depth drops out (default: 0.50).",
    )
    parser.add_argument(
        "--max-motion-trans-mm",
        type=float,
        default=120.0,
        help="Translation threshold above which inter-frame motion is flagged as high (default: 120.0 mm).",
    )
    parser.add_argument(
        "--max-motion-rot-deg",
        type=float,
        default=15.0,
        help="Rotation threshold above which inter-frame motion is flagged as high (default: 15.0 deg).",
    )
    parser.add_argument(
        "--min-odometry-fitness",
        type=float,
        default=0.50,
        help="Minimum inlier correspondence fitness ratio below which odometry is marked unreliable (default: 0.50).",
    )
    parser.add_argument(
        "--max-mask-zero-depth-for-motion",
        type=float,
        default=0.50,
        help="Maximum zero-depth fraction in subject mask above which subject motion is marked unreliable (default: 0.50).",
    )

    args = parser.parse_args()

    analyze_capture(
        recording_dir=args.recording_dir,
        out_dir=args.out_dir,
        csv_name=args.csv_name,
        max_mask_zero_depth_frac=args.max_mask_zero_depth_frac,
        min_valid_person_pixels=args.min_valid_person_pixels,
        range_near_min=args.range_near_min,
        range_subject_min=args.range_subject_min,
        range_subject_max=args.range_subject_max,
        min_subject_mask_pixels=args.min_subject_mask_pixels,
        d455_min_range_m=args.d455_min_range_m,
        max_motion_trans_mm=args.max_motion_trans_mm,
        max_motion_rot_deg=args.max_motion_rot_deg,
        min_odometry_fitness=args.min_odometry_fitness,
        max_mask_zero_depth_for_motion=args.max_mask_zero_depth_for_motion,
    )


if __name__ == "__main__":
    main()
