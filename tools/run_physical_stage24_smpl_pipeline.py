#!/usr/bin/env python3
"""
tools/run_physical_stage24_smpl_pipeline.py
============================================
Stage 24: SMPL Integration and RealSense-Based Human Mesh Reconstruction.

Executes:
- SMPL model setup, licensing audit, and asset verification.
- Canonical 6,890-vertex SMPL mesh generation and quality validation.
- Ingestion of Stage 23 multi-stop RealSense physical point cloud.
- Gradient-based SMPL alignment and fitting (translation, orientation, scale, shape).
- Full anthropometric measurement extraction on the fitted SMPL mesh.
- Comparative multi-view visualization contrasting Anny baseline and SMPL.

Usage:
  python tools/run_physical_stage24_smpl_pipeline.py [--out-dir stage24_smpl_integration_output]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

# Configure python path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(PACKAGE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realsense_measure.measurements.smpl_orchestrator import (
    Stage24SMPLOrchestrator,
    Stage24Verdict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage24_smpl_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 24: SMPL Integration Pipeline")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT.parent / "stage24_smpl_integration_output",
        help="Destination directory for Stage 24 deliverables",
    )
    args = parser.parse_args()

    logger.info("Initializing Stage 24 SMPL Integration Orchestrator...")
    orchestrator = Stage24SMPLOrchestrator(
        repo_root=REPO_ROOT.parent,
        output_dir=args.out_dir,
    )

    t0 = time.time()
    result = orchestrator.run_stage24()
    elapsed = time.time() - t0

    logger.info(f"Stage 24 completed in {elapsed:.2f} seconds.")
    logger.info(f"Final Verdict: {result['verdict']}")
    logger.info(f"Deliverables directory: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

