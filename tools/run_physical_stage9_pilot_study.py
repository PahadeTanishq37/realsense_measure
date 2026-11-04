#!/usr/bin/env python3
"""
tools/run_physical_stage9_pilot_study.py
========================================
Problem 4 — Stage 9: Physical Pilot Study & Repeatability Evaluation.

Executes:
1. Pilot study protocol definition (anonymous subject IDs, session tracking, environmental controls).
2. Capture configuration specification (distance, height, yaw stops, resolution).
3. Full-body readiness evaluation using sensor-grounded Stage 8 diagnostics.
4. Repeatability analysis for multiple captures of the same subject with ICC(2,1) and within-subject CV.
5. Manual anthropometric ground-truth validation bridge (Stage 6 integration).
6. Empirical configuration comparison without ungrounded superiority claims.
7. Standardized output generation in stage9_pilot_study_output/.

Usage:
    python tools/run_physical_stage9_pilot_study.py \
        --raw-dir raw_captures/body_capture_01 \
        --out-dir stage9_pilot_study_output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Configure python path to access repository modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from body.pilot_study import (
    FullBodyReadinessChecker,
    PilotConfigurationComparator,
    PilotManualValidationBridge,
    PilotStudyOrchestrator,
    PilotStudyProtocolConfig,
    RepeatabilityEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_physical_stage9_pilot_study")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 9: Physical Pilot Study & Repeatability Evaluation"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw_captures/body_capture_01"),
        help="Path to physical raw capture directory (default: raw_captures/body_capture_01)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stage9_pilot_study_output"),
        help="Output directory (default: stage9_pilot_study_output)",
    )
    parser.add_argument(
        "--stage6-template",
        type=Path,
        default=Path("stage6_validation_output/manual_validation_template.csv"),
        help="Path to Stage 6 manual validation template CSV",
    )
    parser.add_argument(
        "--stage5-dual-mesh",
        type=Path,
        default=Path("stage5_anthropometric_output/dual_mesh_measurement_comparison.csv"),
        help="Path to Stage 5 dual-mesh comparison CSV",
    )
    parser.add_argument(
        "--stage8-calib",
        type=Path,
        default=Path("stage8_capture_quality_audit_output/capture_quality_calibration_report.json"),
        help="Path to Stage 8 quality calibration report JSON",
    )

    args = parser.parse_args()

    logger.info("============================================================")
    logger.info("  Problem 4 — Stage 9: Physical Pilot Study & Repeatability")
    logger.info("============================================================")
    logger.info(f"  Raw capture directory:   {args.raw_dir}")
    logger.info(f"  Stage 6 template CSV:    {args.stage6_template}")
    logger.info(f"  Stage 5 dual-mesh CSV:   {args.stage5_dual_mesh}")
    logger.info(f"  Stage 8 calib report:    {args.stage8_calib}")
    logger.info(f"  Output directory:        {args.out_dir}")

    orchestrator = PilotStudyOrchestrator(output_dir=args.out_dir)
    summary = orchestrator.run()

    logger.info("------------------------------------------------------------")
    logger.info("Stage 9 Pilot Study Summary:")
    logger.info(f"  Capture Readiness:      {summary['audited_capture_readiness']['overall_status']}")
    logger.info(f"  Coverage Ratio:         {summary['audited_capture_readiness']['coverage_ratio']:.1%}")
    logger.info(f"  Missing Regions:        {', '.join(summary['audited_capture_readiness']['missing_regions'])}")
    logger.info(f"  Repeatability Status:   {summary['repeatability_statistics_status']}")
    logger.info(f"  Manual Validation:      {summary['manual_validation_status']}")
    logger.info(f"  Configuration Decision: {summary['optimal_configuration_conclusion']}")
    logger.info("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

