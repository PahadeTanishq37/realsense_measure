"""
Command-line entry point for realsense_measure.

Captures multiple views of an object with a RealSense D455(f), registers
them into a unified point cloud, and measures the target's dimensions.

Example usage
-------------
Scan a box with default settings (Open3D windows enabled, 4 mm voxel)::

    python main.py

Specify output directory and target type::

    python main.py --target box --out my_scan

Skip the interactive Open3D stage windows (useful on headless machines or
when scripting back-to-back scans)::

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
        help="override automatic frame capture interval in seconds",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- build configuration -------------------------------------------
    cfg = PipelineConfig(output_dir=Path(args.out))

    cfg.target.name        = args.target
    cfg.show_stage_windows = not args.no_viz

    if args.interval is not None:
        cfg.camera.capture_interval_s = args.interval

    if args.voxel is not None:
        cfg.preprocess.voxel_size_m   = args.voxel
        cfg.registration.voxel_size_m = args.voxel

    # ---- run -----------------------------------------------------------
    try:
        ScanPipeline(cfg).run()
    except KeyboardInterrupt:
        print("\nScan aborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
