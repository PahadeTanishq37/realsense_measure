#!/usr/bin/env python3
"""
tools/run_physical_stage26_anthropometric_audit.py
===================================================
Stage 26: Anthropometric Measurement Pipeline Audit, Anatomical Landmark Correction & Validation.

Executes:
- Root-cause defect audit of Stage 25 measurement discrepancies.
- Mathematical coordinate frame standardization into Stage 22 ISB frame (+X Lateral, +Y Superior, +Z Anterior).
- Extraction of corrected 15-measurement anthropometric suite on baseline fitted and personalized meshes.
- Perimeter-weighted physical point cloud surface coverage calculation.
- Synthetic geometric validation on analytical box and cylinder under SO(3) rotations and scale.
- Deliverables generation in `stage26_anthropometric_validation_output/`.

Usage:
  python tools/run_physical_stage26_anthropometric_audit.py [--out-dir stage26_anthropometric_validation_output]
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

from realsense_measure.measurements.stage26_orchestrator import (
    Stage26AnthropometricAuditOrchestrator,
    Stage26Verdict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage26_anthropometric_audit")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 26: Anthropometric Audit Pipeline")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT.parent / "stage26_anthropometric_validation_output",
        help="Destination directory for Stage 26 deliverables",
    )
    args = parser.parse_args()

    logger.info("Initializing Stage 26 Anthropometric Audit Orchestrator...")
    orchestrator = Stage26AnthropometricAuditOrchestrator(
        repo_root=REPO_ROOT.parent,
        output_dir=args.out_dir,
    )

    t0 = time.time()
    try:
        results = orchestrator.run_stage26()
        elapsed = time.time() - t0
        logger.info(f"Stage 26 audit successfully completed in {elapsed:.2f}s.")
        logger.info(f"Final Stage 26 Verdict: {results['verdict']}")
        logger.info(f"Report: {results['report_path']}")
        return 0
    except Exception as exc:
        logger.exception(f"Stage 26 pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

