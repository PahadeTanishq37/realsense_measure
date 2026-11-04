#!/usr/bin/env python3
"""
tools/run_physical_stage25_smpl_personalization.py
===================================================
Stage 25: RealSense-Driven SMPL Mesh Personalization and Surface Deformation Runner.

Executes:
- Stage 24 SMPL model status & setup audit.
- Ingestion of Stage 23 RealSense physical point cloud and Stage 24 fitted SMPL mesh.
- Point-cloud-driven regularized Laplacian surface deformation (SMPL 6,890 vertices, 13,776 faces).
- Dirichlet boundary pinning of unobserved lower limbs (calves, feet, crown).
- 15-measurement anthropometric extraction on the personalized mesh.
- Generation of all 5 diagnostic figures and 6 JSON artifact summaries.
- Generation of the Stage 25 Markdown Audit Report.

Usage:
  python tools/run_physical_stage25_smpl_personalization.py [--out-dir stage25_smpl_personalization_output]
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

from realsense_measure.measurements.smpl_personalization_orchestrator import (
    Stage25SMPLPersonalizationOrchestrator,
    Stage25Verdict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage25_smpl_personalization")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 25: SMPL Mesh Personalization Pipeline")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT.parent / "stage25_smpl_personalization_output",
        help="Destination directory for Stage 25 deliverables",
    )
    args = parser.parse_args()

    logger.info("Initializing Stage 25 SMPL Mesh Personalization Orchestrator...")
    orchestrator = Stage25SMPLPersonalizationOrchestrator(
        repo_root=REPO_ROOT.parent,
        output_dir=args.out_dir,
    )

    t0 = time.time()
    try:
        results = orchestrator.run_stage25()
        elapsed = time.time() - t0
        logger.info(f"Stage 25 personalization successfully completed in {elapsed:.2f}s.")
        logger.info(f"Final Stage 25 Verdict: {results['verdict']}")
        logger.info(f"Report: {results['report_path']}")
        return 0
    except Exception as exc:
        logger.exception(f"Stage 25 pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

