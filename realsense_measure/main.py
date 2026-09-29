"""
Command-line entry point for realsense_measure.

Captures multiple views of an object with a RealSense D455(f), registers
them into a unified point cloud, and measures the target's dimensions.

Example usage
-------------
Scan a box with default settings (auto-capture every 2.0s, Open3D windows enabled)::

    python main.py

Scan with faster continuous capture (e.g. every 1.5s)::

    python main.py --interval 1.5

Manual SPACE-triggered capture::

    python main.py --manual-capture

Specify output directory and target type::

    python main.py --target box --out my_scan

Skip the interactive Open3D stage windows::

    python main.py --no-viz

Use a finer voxel grid for higher-resolution scans (slower)::

    python main.py --voxel 0.003
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import PipelineConfig
from pipeline import ScanPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Measure a 3-D object or human face/head with a RealSense D455(f) depth camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target",
        default="box",
        choices=["box", "head", "body"],
        help="what to scan/measure",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["pointcloud", "rgbd_human"],
        help="registration mode: pointcloud (geometric) or rgbd_human (RGB-D odometry + colored ICP)",
    )
    parser.add_argument(
        "--fusion",
        default="pointcloud",
        choices=["pointcloud", "tsdf"],
        help="reconstruction fusion method: pointcloud (concatenation) or tsdf (volumetric integration)",
    )
    parser.add_argument(
        "--show-pairs",
        action="store_true",
        help="visualize pairwise registration before/after comparison",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="visualize estimated camera trajectory with motion jump detection",
    )
    parser.add_argument(
        "--out",
        default="scan_output",
        metavar="DIR",
        help="output directory for point clouds + measurement.json",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="skip the Open3D popup window after each pipeline stage",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=None,
        metavar="M",
        help="override voxel downsample size in metres "
             "(applies to both preprocessing and registration)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SEC",
        help="override automatic capture interval in seconds (default: 2.0s for box, 0.4s for human)",
    )
    parser.add_argument(
        "--manual-capture",
        action="store_true",
        help="disable automatic capture and require pressing SPACE for each frame",
    )
    parser.add_argument(
        "--loop-closure",
        action="store_true",
        help="run diagnostic loop-closure check between last and first frame",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="disable multiway pose-graph registration and use sequential registration instead",
    )
    parser.add_argument(
        "--reference-dims",
        type=str,
        default=None,
        metavar="L,W,H",
        help="comma-separated ground-truth box dimensions in mm (e.g. '300,200,150') for accuracy comparison",
    )
    parser.add_argument(
        "--manual-roi",
        action="store_true",
        help="interactively select a 2D bounding box ROI on the first frame to focus isolation",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- build configuration -------------------------------------------
    cfg = PipelineConfig(output_dir=Path(args.out))

    cfg.target.name        = args.target
    cfg.show_stage_windows = not args.no_viz

    if args.manual_roi:
        cfg.target.use_manual_roi = True

    if args.reference_dims is not None:
        try:
            raw_vals = [float(x.strip()) for x in args.reference_dims.split(",") if x.strip()]
            if len(raw_vals) != 3:
                raise ValueError(f"Expected exactly 3 dimensions (length, width, height in mm), got {len(raw_vals)}")
            # Convert mm to metres and sort descending
            cfg.reference_dims_m = sorted([v / 1000.0 for v in raw_vals], reverse=True)
        except Exception as err:
            parser.error(f"Invalid value for --reference-dims '{args.reference_dims}': {err}")

    if args.mode is not None:
        cfg.registration.registration_mode = args.mode
    elif args.target in ("head", "body"):
        cfg.registration.registration_mode = "rgbd_human"

    cfg.fusion_method = args.fusion
    if args.show_pairs:
        cfg.registration.show_registration_pairs = True
    if args.show_trajectory:
        cfg.registration.show_camera_trajectory = True

    if args.voxel is not None:
        cfg.preprocess.voxel_size_m   = args.voxel
        cfg.registration.voxel_size_m = args.voxel

    if args.interval is not None:
        cfg.camera.capture_interval_s = args.interval
        cfg.camera.human_capture_interval_s = args.interval

    if args.manual_capture:
        cfg.camera.auto_capture = False

    if args.sequential:
        cfg.registration.use_multiway = False

    if args.loop_closure:
        cfg.attempt_loop_closure = True

    # ---- run -----------------------------------------------------------
    try:
        ScanPipeline(cfg).run()
    except KeyboardInterrupt:
        print("\nScan aborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
