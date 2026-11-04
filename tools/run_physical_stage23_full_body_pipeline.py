#!/usr/bin/env python3
"""
tools/run_physical_stage23_full_body_pipeline.py
=================================================
Stage 23: Full-Body Reconstruction and Anthropometric Pipeline Integration.

Integrates:
- Stage 22 right-handed ISB coordinate geometry and torso isolation.
- Existing physical multi-stop RealSense D455f point clouds.
- Anny parametric human-body model fitting & regularized Laplacian surface deformation.
- Full-body anthropometric measurements suite with honest observability disclosures.
- Comprehensive JSON and 3D visual deliverable artifacts.

Usage:
  python tools/run_physical_stage23_full_body_pipeline.py [--out-dir stage23_full_body_integration_output]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

# Configure repository sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(PACKAGE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realsense_measure.measurements.full_body_pipeline_orchestrator import (
    Stage23FullBodyPipelineOrchestrator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage23_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 23: Full-Body Pipeline Integration")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT.parent / "stage23_full_body_integration_output",
        help="Destination directory for Stage 23 deliverables",
    )
    args = parser.parse_args()

    logger.info("Initializing Stage 23 Full-Body Pipeline Orchestrator...")
    orchestrator = Stage23FullBodyPipelineOrchestrator(
        repo_root=REPO_ROOT.parent,
        output_dir=args.out_dir,
    )

    t0 = time.time()
    result = orchestrator.run_pipeline()
    elapsed = time.time() - t0

    logger.info(f"Stage 23 completed in {elapsed:.2f} seconds with status: {result['status']}")
    logger.info(f"Generated {result['measurements_count']} integrated anthropometric measurements.")
    logger.info(f"Deliverables directory: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

