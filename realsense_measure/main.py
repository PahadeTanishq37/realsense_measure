"""
Command-line entry point for realsense_measure.

Captures multiple views of an object with a RealSense D455(f), registers
them into a unified point cloud, and measures the target's dimensions.

Example usage
-------------
Scan a box with default settings (auto-capture every 3.0s, Open3D windows enabled)::

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
        description="Measure a 3-D object with a RealSense D455(f) depth camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target",
        default="box",
        choices=["box"],  # more choices land here once head.py / body.py exist
        help="what to measure",
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
        help="override automatic capture interval in seconds (default: 3.0)",
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- build configuration -------------------------------------------
    cfg = PipelineConfig(output_dir=Path(args.out))

    cfg.target.name        = args.target
    cfg.show_stage_windows = not args.no_viz

    if args.voxel is not None:
        cfg.preprocess.voxel_size_m   = args.voxel
        cfg.registration.voxel_size_m = args.voxel

    if args.interval is not None:
        cfg.camera.capture_interval_s = args.interval

    if args.manual_capture:
        cfg.camera.auto_capture = False

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
