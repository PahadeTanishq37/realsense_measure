"""
realsense_measure/fullbody/cli.py

Command-line interface for the model-free full-body scanning package.
Subcommands:
  capture       Live eight-stop capture controller with operator preview.
  check-camera  Verify RealSense orientation, FOV, and depth sanity (no session).
  scan          Run full acquisition/replay, alignment, fusion, and measurement.
  measure       Extract anthropometric measurements from an existing fused scan.
  validate      Compare scan measurements against physical tape measurements.
  sweet-spot    Evaluate capture geometry and report under-covered anatomical regions.
  review        Open 3D interactive viewer for fused mesh and point cloud.
  keypoints-check Audit MediaPipe Pose detection on a single frame.

Strictly lazy imports so importing this module never pulls in Open3D or pyrealsense2.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("realsense_measure.fullbody")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m realsense_measure.fullbody",
        description="Model-free metric full-body scanning and anthropometry.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. capture (Stage 27 live capture controller)
    p_cap = subparsers.add_parser("capture", help="Live eight-stop capture controller with operator preview.")
    p_cap.add_argument("--out", type=str, required=True, help="Destination directory for the capture session.")
    p_cap.add_argument("--subject-id", type=str, required=True, help="Anonymous subject ID (e.g. SUBJ_01, TEST_42).")
    p_cap.add_argument("--stops", type=int, default=8, help="Number of stops (default: 8).")
    p_cap.add_argument("--turn", type=str, choices=["cw", "ccw"], default="cw", help="Turn direction ('cw' or 'ccw').")
    p_cap.add_argument("--portrait", type=str, choices=["cw", "ccw"], default="cw", help="Portrait camera orientation ('cw' or 'ccw').")
    p_cap.add_argument("--fake", action="store_true", help="Use synthetic FakeCamera instead of hardware RealSense.")
    p_cap.add_argument("--burst-frames", type=int, default=30, help="Frames per stop burst (default: 30).")
    p_cap.add_argument("--headless", action="store_true", help="Run without opening GUI preview window.")
    p_cap.add_argument("--dist-min", type=float, default=1.10, help="Minimum subject distance in metres (default: 1.10).")
    p_cap.add_argument("--dist-max", type=float, default=1.50, help="Maximum subject distance in metres (default: 1.50).")
    p_cap.add_argument("--no-head-feet-check", action="store_true", help="Disable strict head and feet unclipped check.")
    p_cap.add_argument("--turn-prompt-timeout", type=float, default=10.0, help="Turn prompt timeout in seconds (default: 10.0).")
    p_cap.add_argument("--motion-limit", type=float, default=10.0, help="Subject motion limit in mm (default: 10.0).")
    p_cap.add_argument("--max-retakes", type=int, default=3, help="Max retakes per stop (default: 3).")

    # 2. check-camera (Stage 27 hardware sanity check snapshot)
    p_chk = subparsers.add_parser("check-camera", help="Verify RealSense orientation, FOV, and depth sanity (no session).")
    p_chk.add_argument("--portrait", type=str, choices=["cw", "ccw"], default="cw", help="Portrait rotation ('cw' or 'ccw').")
    p_chk.add_argument("--out", type=str, default="scan_output", help="Output directory to save check snapshot into DIR/check/.")
    p_chk.add_argument("--fake", action="store_true", help="Use synthetic FakeCamera for headless testing.")

    # 3. scan
    p_scan = subparsers.add_parser("scan", help="Run full-body alignment, fusion, and measurement.")
    p_scan.add_argument("--capture", type=str, required=True, help="Path to raw capture directory.")
    p_scan.add_argument("--out", type=str, default="scan_output/fullbody", help="Output directory.")
    p_scan.add_argument("--stops", type=int, default=8, help="Number of stops (default: 8).")
    p_scan.add_argument("--voxel", type=float, default=0.005, help="TSDF voxel length in metres.")
    p_scan.add_argument("--no-viz", action="store_true", help="Run without opening GUI windows.")
    p_scan.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    # 4. measure
    p_meas = subparsers.add_parser("measure", help="Extract measurements from an existing fused scan.")
    p_meas.add_argument("--scan", type=str, required=True, help="Directory containing fused.ply or scan_result.json.")

    # 5. validate
    p_val = subparsers.add_parser("validate", help="Compare scan measurements with physical tape values.")
    p_val.add_argument("--scan", type=str, required=True, help="Scan output directory.")
    p_val.add_argument("--tape", type=str, default="data/tape_measurements.json", help="Tape measurements JSON.")

    # 6. sweet-spot
    p_sw = subparsers.add_parser("sweet-spot", help="Evaluate capture geometry information score.")
    p_sw.add_argument("--scan", type=str, required=True, help="Scan directory or capture path.")

    # 7. review
    p_rev = subparsers.add_parser("review", help="Review fused 3D assets in Open3D.")
    p_rev.add_argument("--scan", type=str, required=True, help="Scan output directory.")

    # 8. keypoints-check
    p_kpt = subparsers.add_parser(
        "keypoints-check",
        help="Run MediaPipe Pose on color image, overlay 33 joints, and print confidence.",
    )
    p_kpt.add_argument("color", type=str, help="Path to input color image (PNG/JPG).")
    p_kpt.add_argument("depth", type=str, nargs="?", default=None, help="Optional path to depth array (.npy).")
    p_kpt.add_argument("--out", type=str, default=None, help="Optional output image path for skeleton overlay.")

    return parser


def _run_capture(args: argparse.Namespace) -> int:
    """Execute live 8-stop capture controller with operator preview."""
    from realsense_measure.fullbody.capture_controller import CaptureController, CaptureThresholds, ControllerState
    from realsense_measure.fullbody.capture_ui import run_ui_loop
    from realsense_measure.fullbody.session_format import SessionWriter, validate_anonymous_subject_id

    try:
        subject_id = validate_anonymous_subject_id(args.subject_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "fake", False):
        from tests.fullbody.fake_camera import FakeCamera
        cam = FakeCamera(
            width=480,
            height=848,
            depth_scale=0.001,
            depth_units="sensor_units",
            fps=30.0,
        )
    else:
        from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
        cam_cfg = full_body_camera_config(portrait_rotation=args.portrait)
        cam = RealSenseCaptureAdapter(camera_config=cam_cfg, portrait_rotation=args.portrait)

    try:
        cam.start()
    except Exception as exc:
        print(f"Error: Failed to start camera: {exc}", file=sys.stderr)
        return 1

    accepted_stops = 0
    controller: Optional[CaptureController] = None

    try:
        writer = SessionWriter(
            root=out_dir,
            subject_id=subject_id,
            n_stops=args.stops,
            turn_direction=args.turn,
            camera_portrait_rotation=args.portrait,
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            device_info=cam.device_info(),
        )

        thresholds = CaptureThresholds(
            burst_frames=args.burst_frames,
            motion_limit_mm=getattr(args, "motion_limit", 10.0),
            max_retakes=getattr(args, "max_retakes", 3),
            dist_min_m=getattr(args, "dist_min", 1.10),
            dist_max_m=getattr(args, "dist_max", 1.50),
            require_head_feet=not getattr(args, "no_head_feet_check", False),
            turn_prompt_timeout_s=getattr(args, "turn_prompt_timeout", 10.0),
        )

        landmark_fn = None
        try:
            from realsense_measure.fullbody.keypoints import MediaPipePoseDetector
            detector = MediaPipePoseDetector()
            def landmark_extractor(color_bgr: np.ndarray, depth_arr: np.ndarray) -> Optional[dict]:
                try:
                    pts_2d, conf, vis = detector.detect(color_bgr)
                    return {
                        "landmarks_2d": {k: [float(v[0]), float(v[1])] for k, v in pts_2d.items()},
                        "confidence": {k: float(v) for k, v in conf.items()},
                        "visibility": {k: float(v) for k, v in vis.items()},
                    }
                except Exception as k_exc:
                    logger.warning(f"Keypoint extraction warning: {k_exc}")
                    return None
            landmark_fn = landmark_extractor
        except Exception:
            landmark_fn = None

        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            landmark_fn=landmark_fn,
        )

        headless = bool(getattr(args, "headless", False))
        run_ui_loop(controller, headless=headless)

    except KeyboardInterrupt:
        print("\nCapture interrupted by operator (Ctrl-C).")
    except Exception as exc:
        print(f"\nCapture error: {exc}", file=sys.stderr)
    finally:
        cam.stop()

        # Check how many stops were accepted
        for stop_idx in range(args.stops):
            s_json = out_dir / f"stop_{stop_idx:02d}" / "stop.json"
            if s_json.exists():
                try:
                    with open(s_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("accepted_attempt") is not None:
                        accepted_stops += 1
                except Exception:
                    pass

    if controller is not None and controller.state == ControllerState.DONE:
        print(f"\nCapture session successfully finalized at: {out_dir}")
        print(f"Accepted all {args.stops} stops.")
        return 0
    else:
        print(f"\nSession unfinalized at {out_dir}. Accepted stops: {accepted_stops} of {args.stops}.")
        print("Raw attempt data and integrity hashes remain intact on disk.")
        return 1


def _run_check_camera(args: argparse.Namespace) -> int:
    """Verify RealSense orientation, FOV, and depth sanity (no session)."""
    if getattr(args, "fake", False):
        from tests.fullbody.fake_camera import FakeCamera
        cam = FakeCamera(width=480, height=848, depth_scale=0.001, depth_units="sensor_units")
    else:
        from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
        cam_cfg = full_body_camera_config(portrait_rotation=args.portrait)
        cam = RealSenseCaptureAdapter(camera_config=cam_cfg, portrait_rotation=args.portrait)

    try:
        cam.start()
    except Exception as exc:
        print(f"Error starting camera: {exc}", file=sys.stderr)
        return 1

    last_color: Optional[np.ndarray] = None
    last_depth: Optional[np.ndarray] = None

    try:
        # Stream for approx 5 seconds or until frames acquired
        start_time = time.time()
        while (time.time() - start_time) < 5.0:
            c, r, f, ts = cam.read_raw_filtered()
            if c is not None and r is not None:
                last_color = c
                last_depth = r
            time.sleep(0.033)
    finally:
        cam.stop()

    if last_color is None or last_depth is None:
        print("Error: Could not acquire valid frames from camera.", file=sys.stderr)
        return 1

    dev_info = cam.device_info()
    intr = cam.intrinsics
    scale = cam.depth_scale
    h, w = last_depth.shape[:2]

    d_m = last_depth.astype(np.float32) * (scale if cam.depth_units == "sensor_units" else 1.0)
    roi = d_m[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
    valid_roi = roi[(roi > 0.3) & (roi < 3.5)]
    med_depth_m = float(np.median(valid_roi)) if len(valid_roi) > 0 else 0.0
    valid_frac = float(len(valid_roi)) / float(roi.size)

    print("\n" + "=" * 60)
    print("REALSENSE CAMERA CHECK REPORT")
    print("=" * 60)
    print(f"Device Name:        {dev_info.get('name', 'Intel RealSense')}")
    print(f"Serial Number:      {dev_info.get('serial_number', 'UNKNOWN')}")
    print(f"Firmware Version:   {dev_info.get('firmware_version', 'UNKNOWN')}")
    print(f"Depth Scale:        {scale}")
    print(f"Stream Resolution:  {intr.get('width')}x{intr.get('height')}")
    print(f"Intrinsics (Rot):   fx={intr.get('fx'):.2f}, fy={intr.get('fy'):.2f}, cx={intr.get('cx'):.2f}, cy={intr.get('cy'):.2f}")
    print(f"Portrait Rotation:  {args.portrait} (applied: {intr.get('is_post_rotation')})")
    print(f"ROI Median Depth:   {med_depth_m:.3f} m")
    print(f"ROI Valid Fraction: {valid_frac * 100.0:.1f} %")
    print("=" * 60 + "\n")

    check_dir = Path(args.out) / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    from realsense_measure.fullbody.capture_ui import colorize_depth_m
    depth_vis = colorize_depth_m(d_m)

    cv2.imwrite(str(check_dir / "check_color.png"), last_color)
    cv2.imwrite(str(check_dir / "check_depth_colorized.png"), depth_vis)

    metrics_txt = check_dir / "check_metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"device_name: {dev_info.get('name')}\n")
        f.write(f"serial_number: {dev_info.get('serial_number')}\n")
        f.write(f"firmware_version: {dev_info.get('firmware_version')}\n")
        f.write(f"depth_scale: {scale}\n")
        f.write(f"resolution: {intr.get('width')}x{intr.get('height')}\n")
        f.write(f"intrinsics: fx={intr.get('fx')}, fy={intr.get('fy')}, cx={intr.get('cx')}, cy={intr.get('cy')}\n")
        f.write(f"portrait_rotation: {args.portrait}\n")
        f.write(f"roi_median_depth_m: {med_depth_m:.3f}\n")
        f.write(f"roi_valid_fraction: {valid_frac:.4f}\n")

    print(f"Snapshot saved to: {check_dir}")
    print("  - check_color.png")
    print("  - check_depth_colorized.png")
    print("  - check_metrics.txt\n")
    return 0


def run_pipeline(capture_dir: Path, out_dir: Path, cfg: Any) -> Any:
    """Execute the complete full-body reconstruction and measurement pipeline."""
    from realsense_measure.fullbody.alignment import align_stops
    from realsense_measure.fullbody.confidence import compute_confidence_map
    from realsense_measure.fullbody.floor import fit_floor_plane, verify_floor_consistency
    from realsense_measure.fullbody.fusion import fuse_aligned_stops
    from realsense_measure.fullbody.io import load_capture_session, save_scan_result
    from realsense_measure.fullbody.keypoints import MediaPipePoseDetector, lift_keypoints_to_3d
    from realsense_measure.fullbody.measure_scan import measure_fused_scan
    from realsense_measure.fullbody.types import ScanResult
    import open3d as o3d

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

        # 4. Point cloud in camera frame
        color_rgb = np.ascontiguousarray(s.color[:, :, ::-1])
        depth_m = s.depth.copy()
        depth_m[depth_m < cfg.depth_min_m] = 0.0
        depth_m[depth_m > cfg.depth_max_m] = 0.0
        if s.confidence_map is not None:
            depth_m[s.confidence_map < cfg.min_confidence_threshold] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,
            depth_trunc=cfg.depth_max_m,
            convert_rgb_to_intensity=False,
        )
        s.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, s.intrinsics)

    if detected_floors:
        is_consistent, rep_floor = verify_floor_consistency(detected_floors, cfg)
        if not is_consistent:
            logger.warning("Floor planes across stops are not fully consistent; using consensus plane.")
    else:
        logger.warning("No floor planes detected across stops; falling back to nominal gravity floor.")
        from realsense_measure.fullbody.types import FloorPlane
        rep_floor = FloorPlane(
            coefficients=np.array([0.0, -1.0, 0.0, 1.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            point_on_plane=np.array([0.0, 1.0, 1.3]),
            inlier_count=0,
            inlier_ratio=0.0,
            rmse_m=0.0,
            gravity_R_cam=np.eye(3),
        )
    logger.info(f"Floor plane verification complete (normal={rep_floor.normal}, height={rep_floor.point_on_plane[1]:.3f} m).")

    # Alignment
    logger.info("Aligning multi-stop point clouds with joint prior and 4-DoF MAP-ICP ...")
    aligned = align_stops(stops, rep_floor, cfg)

    # Fusion
    logger.info("Fusing aligned point clouds via Open3D TSDF Integration ...")
    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg)

    # Anthropometric measurements
    logger.info("Extracting tape-style anthropometric measurements ...")
    ref_kpts = aligned[0].keypoints_canonical if aligned else None
    measurements = measure_fused_scan(fused_pcd, ref_kpts, cfg, hole_map=hole_map)

    subj_id = "subject_01"
    meta_p = Path(capture_dir) / "meta.json"
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as mf:
                subj_id = json.load(mf).get("subject_id", subj_id)
        except Exception:
            pass

    scan_res = ScanResult(
        subject_id=subj_id,
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

    if args.subcommand == "capture":
        return _run_capture(args)

    elif args.subcommand == "check-camera":
        return _run_check_camera(args)

    elif args.subcommand == "scan":
        from realsense_measure.fullbody.config import FullBodyConfig
        cfg = FullBodyConfig()
        cfg.output_dir = Path(args.out)
        cfg.num_stops = args.stops
        cfg.tsdf_voxel_length_m = args.voxel
        cfg.verbose = args.verbose

        result = run_pipeline(Path(args.capture), Path(args.out), cfg)
        print("\n" + result.summary_table() + "\n")
        return 0

    elif args.subcommand == "measure":
        from realsense_measure.fullbody.io import load_scan_result
        p = Path(args.scan)
        scan = load_scan_result(p)
        print("\n" + scan.summary_table() + "\n")
        return 0

    elif args.subcommand == "validate":
        from realsense_measure.fullbody.io import load_scan_result
        from realsense_measure.fullbody.validate import compare_with_tape, format_validation_table, load_tape_measurements
        scan = load_scan_result(Path(args.scan))
        tape_vals, rep_floor, subj = load_tape_measurements(Path(args.tape))
        report = compare_with_tape(scan.measurements, tape_vals, rep_floor)
        print("\n" + format_validation_table(report) + "\n")
        return 0

    elif args.subcommand == "sweet-spot":
        from realsense_measure.fullbody.config import FullBodyConfig
        from realsense_measure.fullbody.io import load_scan_result
        from realsense_measure.fullbody.sweet_spot import evaluate_capture_configuration
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
        import open3d as o3d
        from realsense_measure.fullbody.io import load_scan_result
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


if __name__ == "__main__":
    sys.exit(main())
