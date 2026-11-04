"""
realsense_measure/fullbody/cli.py

Command-line interface for the model-free full-body scanning package.
Subcommands:
  scan        Run full acquisition/replay, alignment, fusion, and measurement.
  measure     Extract anthropometric measurements from an existing fused scan.
  validate    Compare scan measurements against physical tape measurements.
  sweet-spot  Evaluate capture geometry and report under-covered anatomical regions.
  review      Open 3D interactive viewer for fused mesh and point cloud.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.alignment import align_stops
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.confidence import compute_confidence_map
from realsense_measure.fullbody.floor import fit_floor_plane, verify_floor_consistency
from realsense_measure.fullbody.fusion import fuse_aligned_stops
from realsense_measure.fullbody.io import load_capture_session, load_scan_result, save_scan_result
from realsense_measure.fullbody.keypoints import MediaPipePoseDetector, lift_keypoints_to_3d
from realsense_measure.fullbody.measure_scan import measure_fused_scan
from realsense_measure.fullbody.sweet_spot import evaluate_capture_configuration
from realsense_measure.fullbody.types import ScanResult
from realsense_measure.fullbody.validate import compare_with_tape, format_validation_table, load_tape_measurements

logger = logging.getLogger("realsense_measure.fullbody")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m realsense_measure.fullbody",
        description="Model-free metric full-body scanning and anthropometry.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. scan
    p_scan = subparsers.add_parser("scan", help="Run full-body alignment, fusion, and measurement.")
    p_scan.add_argument("--capture", type=str, required=True, help="Path to raw capture directory.")
    p_scan.add_argument("--out", type=str, default="scan_output/fullbody", help="Output directory.")
    p_scan.add_argument("--stops", type=int, default=8, help="Number of stops (default: 8).")
    p_scan.add_argument("--voxel", type=float, default=0.005, help="TSDF voxel length in metres.")
    p_scan.add_argument("--no-viz", action="store_true", help="Run without opening GUI windows.")
    p_scan.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    # 2. measure
    p_meas = subparsers.add_parser("measure", help="Extract measurements from an existing fused scan.")
    p_meas.add_argument("--scan", type=str, required=True, help="Directory containing fused.ply or scan_result.json.")

    # 3. validate
    p_val = subparsers.add_parser("validate", help="Compare scan measurements with physical tape values.")
    p_val.add_argument("--scan", type=str, required=True, help="Scan output directory.")
    p_val.add_argument("--tape", type=str, default="data/tape_measurements.json", help="Tape measurements JSON.")

    # 4. sweet-spot
    p_sw = subparsers.add_parser("sweet-spot", help="Evaluate capture geometry information score.")
    p_sw.add_argument("--scan", type=str, required=True, help="Scan directory or capture path.")

    # 5. review
    p_rev = subparsers.add_parser("review", help="Review fused 3D assets in Open3D.")
    p_rev.add_argument("--scan", type=str, required=True, help="Scan output directory.")

    # 6. keypoints-check
    p_kpt = subparsers.add_parser(
        "keypoints-check",
        help="Run MediaPipe Pose on color image, overlay 33 joints, and print confidence.",
    )
    p_kpt.add_argument("color", type=str, help="Path to input color image (PNG/JPG).")
    p_kpt.add_argument("depth", type=str, nargs="?", default=None, help="Optional path to depth array (.npy).")
    p_kpt.add_argument("--out", type=str, default=None, help="Optional output image path for skeleton overlay.")

    return parser


def run_pipeline(capture_dir: Path, out_dir: Path, cfg: FullBodyConfig) -> ScanResult:
    """Execute the complete full-body reconstruction and measurement pipeline."""
    logger.info(f"Loading capture session from {capture_dir} ...")
    stops = load_capture_session(capture_dir, cfg)

    detector = MediaPipePoseDetector()
    detected_floors = []

    logger.info(f"Processing {len(stops)} stops (keypoint detection, floor fitting, confidence map) ...")
    for s in stops:
        # 1. 2D keypoints and 3D lifting
        pts_2d, conf_2d, vis_2d = detector.detect(s.color)
        s.keypoints_camera = lift_keypoints_to_3d(pts_2d, conf_2d, vis_2d, s.depth, s.intrinsics, cfg)

        # 2. Floor plane
        floor = fit_floor_plane(s.depth, s.intrinsics, cfg)
        if floor is not None:
            s.floor_plane = floor
            detected_floors.append(floor)

        # 3. Confidence map
        conf_map, filtered_depth = compute_confidence_map(
            s.depth, s.intrinsics, cfg, depth_std_m=s.depth_std, valid_counts=s.valid_counts
        )
        s.confidence_map = conf_map

        # 4. Backproject filtered depth to camera-frame point cloud
        h, w = filtered_depth.shape[:2]
        fx, fy = s.intrinsics.get_focal_length()
        cx, cy = s.intrinsics.get_principal_point()
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        val = filtered_depth > 0
        z_v = filtered_depth[val]
        x_v = (u_grid[val] - cx) * z_v / fx
        y_v = (v_grid[val] - cy) * z_v / fy
        pts_3d = np.stack([x_v, y_v, z_v], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        # Assign colors
        color_rgb = cv2.cvtColor(s.color, cv2.COLOR_BGR2RGB)
        pcd.colors = o3d.utility.Vector3dVector(color_rgb[val] / 255.0)
        s.pcd = pcd

    # Consensus floor
    if detected_floors:
        _, consensus_floor = verify_floor_consistency(detected_floors, cfg)
    else:
        logger.warning("No floor planes detected, falling back to nominal horizontal floor.")
        from realsense_measure.fullbody.types import FloorPlane
        consensus_floor = FloorPlane(
            coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
            normal=np.array([0.0, -1.0, 0.0]),
            point_on_plane=np.array([0.0, 0.90, 1.30]),
            gravity_R_cam=np.eye(3),
        )

    # Multi-stop alignment
    logger.info("Aligning stops into canonical ISB subject frame ...")
    aligned = align_stops(stops, consensus_floor, cfg)

    # Volumetric fusion
    logger.info("Executing TSDF volumetric fusion ...")
    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)

    # Anthropometric measurements
    logger.info("Extracting tape-style anthropometric measurements ...")
    ref_kpts = aligned[0].keypoints_canonical if aligned else None
    measurements = measure_fused_scan(fused_pcd, ref_kpts, cfg, hole_map=hole_map)

    scan_res = ScanResult(
        subject_id="subject_01",
        mesh=mesh,
        pcd=fused_pcd,
        aligned_stops=aligned,
        measurements=measurements,
        hole_map=hole_map,
    )

    save_scan_result(scan_res, out_dir)
    return scan_res


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.subcommand == "scan":
        cfg = FullBodyConfig()
        cfg.output_dir = Path(args.out)
        cfg.num_stops = args.stops
        cfg.tsdf_voxel_length_m = args.voxel
        cfg.verbose = args.verbose

        result = run_pipeline(Path(args.capture), Path(args.out), cfg)
        print("\n" + result.summary_table() + "\n")
        return 0

    elif args.subcommand == "measure":
        p = Path(args.scan)
        scan = load_scan_result(p)
        print("\n" + scan.summary_table() + "\n")
        return 0

    elif args.subcommand == "validate":
        scan = load_scan_result(Path(args.scan))
        tape_vals, rep_floor, subj = load_tape_measurements(Path(args.tape))
        report = compare_with_tape(scan.measurements, tape_vals, rep_floor)
        print("\n" + format_validation_table(report) + "\n")
        return 0

    elif args.subcommand == "sweet-spot":
        scan = load_scan_result(Path(args.scan))
        cfg = FullBodyConfig()
        if scan.pcd is not None:
            sw_rep = evaluate_capture_configuration(scan.pcd, None, cfg)
            print(f"\nConfiguration Score: {sw_rep.total_score:.1f}")
            print(f"Per-part scores: {sw_rep.per_part_scores}")
            if sw_rep.under_covered_parts:
                print(f"Under-covered parts: {sw_rep.under_covered_parts}")
            if sw_rep.recommendations:
                print(f"Recommendations: {sw_rep.recommendations}\n")
        return 0

    elif args.subcommand == "review":
        scan = load_scan_result(Path(args.scan))
        geoms = []
        if scan.mesh is not None and len(scan.mesh.vertices) > 0:
            geoms.append(scan.mesh)
        elif scan.pcd is not None and len(scan.pcd.points) > 0:
            geoms.append(scan.pcd)
        if geoms:
            o3d.visualization.draw_geometries(geoms, window_name=f"Full-Body Scan: {args.scan}")
        else:
            print("No 3D mesh or point cloud found to review.")
        return 0

    elif args.subcommand == "keypoints-check":
        c_path = Path(args.color)
        d_path = Path(args.depth) if args.depth else None
        o_path = Path(args.out) if args.out else None
        return _run_keypoints_check(c_path, d_path, o_path)

    return 0


def _run_keypoints_check(color_path: Path, depth_path: Optional[Path], out_path: Optional[Path]) -> int:
    """Execute MediaPipe Pose detection on a single frame, print confidence table, and write overlay."""
    color = cv2.imread(str(color_path))
    if color is None:
        print(f"Error: Could not read color image from {color_path}", file=sys.stderr)
        return 1
    h, w = color.shape[:2]

    try:
        import mediapipe as mp
    except ImportError:
        print("Error: mediapipe is not installed.", file=sys.stderr)
        return 1

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        print("Error: MediaPipe solutions.pose is not available in installed mediapipe build.", file=sys.stderr)
        return 1

    mp_pose_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

    if not res.pose_landmarks:
        print("No human pose detected by MediaPipe in image.", file=sys.stderr)
        return 1

    depth = np.load(str(depth_path)) if depth_path and depth_path.exists() else None

    print("\n" + "=" * 82)
    print(f"MEDIAPIPE POSE 33-LANDMARK AUDIT: {color_path.name}")
    print("=" * 82)
    if depth is not None:
        print(f"{'Idx':<4} | {'Landmark':<20} | {'u (px)':>7} | {'v (px)':>7} | {'Conf':>6} | {'Vis':>6} | {'Depth (m)':>9}")
    else:
        print(f"{'Idx':<4} | {'Landmark':<20} | {'u (px)':>7} | {'v (px)':>7} | {'Conf':>6} | {'Vis':>6}")
    print("-" * 82)

    vis_img = color.copy()
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        name = mp_pose_names[idx] if idx < len(mp_pose_names) else f"joint_{idx}"
        u = float(lm.x * (w - 1))
        v = float(lm.y * (h - 1))
        conf = float(lm.presence if hasattr(lm, "presence") and lm.presence > 0 else (1.0 if lm.visibility > 0.5 else lm.visibility))
        vis = float(lm.visibility)

        if depth is not None:
            iu = int(np.clip(round(u), 0, w - 1))
            iv = int(np.clip(round(v), 0, h - 1))
            zv = float(depth[iv, iu])
            d_str = f"{zv:.3f}" if zv > 0 else "0.000"
            print(f"{idx:<4} | {name:<20} | {u:>7.1f} | {v:>7.1f} | {conf:>6.2f} | {vis:>6.2f} | {d_str:>9}")
        else:
            print(f"{idx:<4} | {name:<20} | {u:>7.1f} | {v:>7.1f} | {conf:>6.2f} | {vis:>6.2f}")

        cv2.circle(vis_img, (int(round(u)), int(round(v))), 4, (0, 255, 0), -1)
        cv2.putText(vis_img, str(idx), (int(round(u)) + 5, int(round(v)) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if hasattr(mp.solutions.pose, "POSE_CONNECTIONS"):
        for a, b in mp.solutions.pose.POSE_CONNECTIONS:
            lm_a = res.pose_landmarks.landmark[a]
            lm_b = res.pose_landmarks.landmark[b]
            pt_a = (int(round(lm_a.x * (w - 1))), int(round(lm_a.y * (h - 1))))
            pt_b = (int(round(lm_b.x * (w - 1))), int(round(lm_b.y * (h - 1))))
            cv2.line(vis_img, pt_a, pt_b, (0, 200, 255), 2)

    print("=" * 82 + "\n")

    out_file = out_path if out_path else color_path.parent / f"{color_path.stem}_keypoints.png"
    cv2.imwrite(str(out_file), vis_img)
    print(f"Saved annotated keypoint overlay to: {out_file}")
    return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

