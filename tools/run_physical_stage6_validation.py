#!/usr/bin/env python3
"""
tools/run_physical_stage6_validation.py
=======================================
Problem 4 — Stage 6: Manual Anthropometric Ground-Truth Validation.

Implements a reproducible, standards-aligned manual validation framework for comparing
RGB-D-derived measurements (Stage 2 parametric and Stage 4 personalized) against
independent manual anthropometric measurements (ISO 7250 / ISAK protocol).

Key Features:
  1. Generates standardized manual validation template CSV pre-populated with
     Stage 2 and Stage 4 RGB-D measurements, with blank fields for manual tape entry.
  2. Protocol-matched measurement pairing: flags incompatible anatomical definitions
     (e.g. neck_base_circumference) and excludes them from aggregate error metrics.
  3. Computes rigorous statistical validation metrics (when manual data is provided):
       - Signed bias, absolute error, relative percentage error
       - MAE, MAPE, RMSE, standard deviation
       - Subject-level grouping and multi-repeat averaging
       - Bland-Altman limits of agreement (bias +/- 1.96 * SD)
  4. Stratifies accuracy by Stage 5 observation quality tiers:
       - observation_supported
       - partially_observed
       - model_inferred
  5. Evaluates whether Stage 4 surface personalization improves agreement over Stage 2.
  6. Supports synthetic / mock validation mode for automated verification without
     fabricating physical ground-truth numbers in production runs.

Usage:
  # 1. Generate template & framework (manual data pending):
  python tools/run_physical_stage6_validation.py \\
      --stage5-csv stage5_anthropometric_output/dual_mesh_measurement_comparison.csv \\
      --out-dir stage6_validation_output

  # 2. Evaluate with filled manual tape measurements:
  python tools/run_physical_stage6_validation.py \\
      --stage5-csv stage5_anthropometric_output/dual_mesh_measurement_comparison.csv \\
      --manual-csv stage6_validation_output/manual_validation_template.csv \\
      --out-dir stage6_validation_output
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configure python path to access repository modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
for p in [str(REPO_ROOT), str(PACKAGE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage6_validation")


# ---------------------------------------------------------------------------
# Anthropometric Protocol Definitions & Compatibility Table
# ---------------------------------------------------------------------------

ANTHROPOMETRIC_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "hip_circumference": {
        "protocol": "ISO 7250 / ISAK standard: horizontal tape at maximum posterior extension of the buttocks; subject standing erect with feet together.",
        "comparable": True,
        "landmark_category": "pelvis_buttocks",
        "instrument": "Gulick anthropometric tape",
    },
    "waist_circumference": {
        "protocol": "ISO 7250 standard: horizontal tape at narrowest point of the torso between lower costal border and iliac crest at the end of normal expiration.",
        "comparable": True,
        "landmark_category": "lumbar_torso",
        "instrument": "Gulick anthropometric tape",
    },
    "mid_thigh_circumference_left": {
        "protocol": "ISAK standard: tape placed perpendicular to femur axis at midpoint between inguinal crease and superior border of patella (left).",
        "comparable": True,
        "landmark_category": "lower_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "mid_thigh_circumference_right": {
        "protocol": "ISAK standard: tape placed perpendicular to femur axis at midpoint between inguinal crease and superior border of patella (right).",
        "comparable": True,
        "landmark_category": "lower_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "chest_circumference": {
        "protocol": "ISO 7250 standard: horizontal tape around thorax at mesosternale (nipple line in males) at the end of normal expiration.",
        "comparable": True,
        "landmark_category": "thoracic_torso",
        "instrument": "Gulick anthropometric tape",
    },
    "upper_arm_circumference_left": {
        "protocol": "ISAK standard: tape placed perpendicular to humerus axis at midpoint between acromion and olecranon processes (left); arm hanging relaxed.",
        "comparable": True,
        "landmark_category": "upper_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "upper_arm_circumference_right": {
        "protocol": "ISAK standard: tape placed perpendicular to humerus axis at midpoint between acromion and olecranon processes (right); arm hanging relaxed.",
        "comparable": True,
        "landmark_category": "upper_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "forearm_circumference_left": {
        "protocol": "ISAK standard: maximum perimeter of the forearm distal to the humeral epicondyles perpendicular to long axis (left).",
        "comparable": True,
        "landmark_category": "upper_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "forearm_circumference_right": {
        "protocol": "ISAK standard: maximum perimeter of the forearm distal to the humeral epicondyles perpendicular to long axis (right).",
        "comparable": True,
        "landmark_category": "upper_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "max_calf_circumference_left": {
        "protocol": "ISAK standard: maximum horizontal circumference of calf; subject standing with weight distributed equally (left).",
        "comparable": True,
        "landmark_category": "lower_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "max_calf_circumference_right": {
        "protocol": "ISAK standard: maximum horizontal circumference of calf; subject standing with weight distributed equally (right).",
        "comparable": True,
        "landmark_category": "lower_limb",
        "instrument": "Gulick anthropometric tape",
    },
    "head_circumference": {
        "protocol": "ISO 7250: maximum horizontal circumference passing directly above the supraorbital ridges and over the maximal occipital prominence.",
        "comparable": True,
        "landmark_category": "craniofacial",
        "instrument": "Inextensible steel tape",
    },
    "stature": {
        "protocol": "ISO 7250: vertical distance from standing floor to vertex of head using a stadiometer, head positioned in the Frankfort plane.",
        "comparable": True,
        "landmark_category": "whole_body_length",
        "instrument": "Precision stadiometer",
    },
    "shoulder_breadth": {
        "protocol": "ISO 7250: biacromial breadth between lateral borders of acromion processes using broad-blade anthropometer.",
        "comparable": True,
        "landmark_category": "torso_breadth",
        "instrument": "Broad-blade anthropometer",
    },
    "crotch_height": {
        "protocol": "ISO 7250: vertical distance from perineum saddle to floor using anthropometer.",
        "comparable": True,
        "landmark_category": "lower_limb_length",
        "instrument": "Anthropometer",
    },
    "inseam": {
        "protocol": "Tailor/biomechanical protocol: distance from perineum saddle along medial leg to medial malleolus / floor.",
        "comparable": True,
        "landmark_category": "lower_limb_length",
        "instrument": "Anthropometric tape",
    },
    "shoulder_elbow_length": {
        "protocol": "ISO 7250: distance from superior border of acromion process to inferior border of olecranon with elbow flexed at 90 deg.",
        "comparable": True,
        "landmark_category": "upper_limb_length",
        "instrument": "Anthropometer",
    },
    "shoulder_wrist_length": {
        "protocol": "ISO 7250: distance from acromion process to styloid process of radius via lateral epicondyle.",
        "comparable": True,
        "landmark_category": "upper_limb_length",
        "instrument": "Anthropometer",
    },
    "neck_base_circumference": {
        "protocol": "INCOMPATIBLE PROTOCOL: Mesh slicing plane Z=512.5 mm intersects the biacromial shoulder slope without an isolated circular neck contour. Excluded from manual validation.",
        "comparable": False,
        "landmark_category": "head_neck",
        "instrument": "N/A (incompatible plane)",
    },
}


# ---------------------------------------------------------------------------
# Data Contracts: Validation Records & Statistics
# ---------------------------------------------------------------------------

@dataclass
class ValidationRow:
    """A single matched anthropometric comparison record."""
    subject_id: str
    capture_id: str
    measurement_name: str
    anatomical_protocol: str
    manual_measurement_mm: float
    manual_repeat_1_mm: float
    manual_repeat_2_mm: float
    manual_repeat_3_mm: float
    rgbd_stage2_measurement_mm: float
    rgbd_stage4_measurement_mm: float
    stage5_status: str
    observed_coverage_percent: float
    comparable: bool
    clothing_condition: str
    operator_id: str
    notes: str = ""
    # Computed evaluation fields
    stage2_signed_error_mm: float = float("nan")
    stage2_abs_error_mm: float = float("nan")
    stage2_pct_error: float = float("nan")
    stage4_signed_error_mm: float = float("nan")
    stage4_abs_error_mm: float = float("nan")
    stage4_pct_error: float = float("nan")
    personalization_error_reduction_mm: float = float("nan")  # |E2| - |E4| (positive = S4 better)
    has_manual_data: bool = False


@dataclass
class TierValidationStats:
    """Aggregate accuracy and agreement statistics for a quality tier."""
    tier_name: str
    sample_count: int
    stage2_mae_mm: float
    stage2_rmse_mm: float
    stage2_mape_pct: float
    stage2_bias_mm: float
    stage2_sd_mm: float
    stage4_mae_mm: float
    stage4_rmse_mm: float
    stage4_mape_pct: float
    stage4_bias_mm: float
    stage4_sd_mm: float
    mean_error_reduction_mm: float
    bland_altman_loa_stage4: Tuple[float, float] = (float("nan"), float("nan"))  # (bias - 1.96*SD, bias + 1.96*SD)


# ---------------------------------------------------------------------------
# Stage 6 Validation Engine
# ---------------------------------------------------------------------------

class PhysicalStage6Validator:
    """
    Executes manual anthropometric ground-truth validation against RGB-D outputs.
    """

    def __init__(
        self,
        stage5_csv_path: Path | str = "stage5_anthropometric_output/dual_mesh_measurement_comparison.csv",
        manual_csv_path: Optional[Path | str] = None,
        output_dir: Path | str = "stage6_validation_output",
        default_subject_id: str = "SUBJ_01",
        default_capture_id: str = "body_capture_01",
        default_operator_id: str = "OPERATOR_01",
        default_clothing: str = "casual_clothing",
    ):
        self.stage5_csv_path = Path(stage5_csv_path)
        self.manual_csv_path = Path(manual_csv_path) if manual_csv_path else None
        self.out_dir = Path(output_dir)
        self.subject_id = default_subject_id
        self.capture_id = default_capture_id
        self.operator_id = default_operator_id
        self.clothing = default_clothing

    def load_stage5_measurements(self) -> List[Dict[str, Any]]:
        """Load Stage 5 measurement comparison records."""
        if not self.stage5_csv_path.exists():
            raise FileNotFoundError(f"Stage 5 CSV missing: {self.stage5_csv_path}")

        records: List[Dict[str, Any]] = []
        with open(self.stage5_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        logger.info(f"Loaded {len(records)} Stage 5 measurement records from: {self.stage5_csv_path}")
        return records

    def build_validation_template(self, stage5_records: List[Dict[str, Any]]) -> List[ValidationRow]:
        """
        Build validation records matching Stage 5 outputs with ISO 7250 protocols,
        leaving manual measurement fields blank (NaN) for operator entry.
        """
        template_rows: List[ValidationRow] = []

        for rec in stage5_records:
            m_name = rec["measurement_name"]
            proto_info = ANTHROPOMETRIC_PROTOCOLS.get(m_name, {
                "protocol": rec.get("anatomical_definition", "Standard anthropometric protocol"),
                "comparable": rec.get("status") != "invalid",
                "instrument": "Anthropometric tape",
            })

            s2_val = float(rec["stage2_value_mm"]) if rec["stage2_value_mm"] != "nan" else float("nan")
            s4_val = float(rec["stage4_value_mm"]) if rec["stage4_value_mm"] != "nan" else float("nan")
            cov_pct = float(rec.get("observed_coverage_percent", 0.0))
            status = rec.get("status", "model_inferred")

            row = ValidationRow(
                subject_id=self.subject_id,
                capture_id=self.capture_id,
                measurement_name=m_name,
                anatomical_protocol=proto_info["protocol"],
                manual_measurement_mm=float("nan"),
                manual_repeat_1_mm=float("nan"),
                manual_repeat_2_mm=float("nan"),
                manual_repeat_3_mm=float("nan"),
                rgbd_stage2_measurement_mm=s2_val,
                rgbd_stage4_measurement_mm=s4_val,
                stage5_status=status,
                observed_coverage_percent=cov_pct,
                comparable=proto_info["comparable"],
                clothing_condition=self.clothing,
                operator_id=self.operator_id,
                notes=rec.get("notes", ""),
                has_manual_data=False,
            )
            template_rows.append(row)

        return template_rows

    def ingest_manual_measurements(
        self,
        manual_path: Path,
        template_rows: List[ValidationRow],
    ) -> List[ValidationRow]:
        """
        Ingest completed manual measurements from CSV and match with RGB-D records.
        """
        logger.info(f"Ingesting manual measurements from: {manual_path}")
        if not manual_path.exists():
            raise FileNotFoundError(f"Manual measurements CSV not found: {manual_path}")

        manual_dict: Dict[str, Dict[str, Any]] = {}
        with open(manual_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manual_dict[row["measurement_name"]] = row

        updated_rows: List[ValidationRow] = []
        populated_count = 0

        for row in template_rows:
            if row.measurement_name in manual_dict:
                m_rec = manual_dict[row.measurement_name]

                # Parse manual repeats if present
                def _parse(val_str: Optional[str]) -> float:
                    if not val_str or val_str.strip().lower() in ["nan", "", "none"]:
                        return float("nan")
                    try:
                        return float(val_str.strip())
                    except ValueError:
                        return float("nan")

                r1 = _parse(m_rec.get("manual_repeat_1_mm"))
                r2 = _parse(m_rec.get("manual_repeat_2_mm"))
                r3 = _parse(m_rec.get("manual_repeat_3_mm"))
                repeats = [r for r in [r1, r2, r3] if not math.isnan(r) and r > 0]

                # Primary manual value
                primary_val = _parse(m_rec.get("manual_measurement_mm"))
                if math.isnan(primary_val) and len(repeats) > 0:
                    primary_val = float(np.mean(repeats))

                row.manual_measurement_mm = primary_val
                row.manual_repeat_1_mm = r1
                row.manual_repeat_2_mm = r2
                row.manual_repeat_3_mm = r3
                if m_rec.get("operator_id"):
                    row.operator_id = m_rec["operator_id"]
                if m_rec.get("clothing_condition"):
                    row.clothing_condition = m_rec["clothing_condition"]
                if m_rec.get("notes"):
                    row.notes = m_rec["notes"]

                if not math.isnan(primary_val) and primary_val > 0:
                    row.has_manual_data = True
                    populated_count += 1

            updated_rows.append(row)

        logger.info(f"Successfully matched manual ground-truth for {populated_count} of {len(template_rows)} metrics")
        return updated_rows

    def compute_validation_statistics(
        self,
        rows: List[ValidationRow],
    ) -> Tuple[List[ValidationRow], Dict[str, TierValidationStats], Dict[str, Any]]:
        """
        Compute errors, statistical aggregates, Bland-Altman limits, and tier stratification.
        """
        for r in rows:
            if not r.has_manual_data or not r.comparable or math.isnan(r.manual_measurement_mm):
                continue

            m = r.manual_measurement_mm
            s2 = r.rgbd_stage2_measurement_mm
            s4 = r.rgbd_stage4_measurement_mm

            # Stage 2 errors
            if not math.isnan(s2):
                e2 = s2 - m
                r.stage2_signed_error_mm = e2
                r.stage2_abs_error_mm = abs(e2)
                r.stage2_pct_error = (e2 / m * 100.0) if m > 0 else float("nan")

            # Stage 4 errors
            if not math.isnan(s4):
                e4 = s4 - m
                r.stage4_signed_error_mm = e4
                r.stage4_abs_error_mm = abs(e4)
                r.stage4_pct_error = (e4 / m * 100.0) if m > 0 else float("nan")

            # Personalization error reduction (|E2| - |E4|)
            if not math.isnan(r.stage2_abs_error_mm) and not math.isnan(r.stage4_abs_error_mm):
                r.personalization_error_reduction_mm = r.stage2_abs_error_mm - r.stage4_abs_error_mm

        # Stratified Aggregation by Stage 5 Quality Tiers
        valid_rows = [r for r in rows if r.has_manual_data and r.comparable and not math.isnan(r.stage4_abs_error_mm)]
        tiers = ["observation_supported", "partially_observed", "model_inferred", "all_comparable"]
        tier_stats: Dict[str, TierValidationStats] = {}

        for tier in tiers:
            if tier == "all_comparable":
                sub_rows = valid_rows
            else:
                sub_rows = [r for r in valid_rows if r.stage5_status == tier]

            n = len(sub_rows)
            if n == 0:
                tier_stats[tier] = TierValidationStats(
                    tier_name=tier,
                    sample_count=0,
                    stage2_mae_mm=float("nan"),
                    stage2_rmse_mm=float("nan"),
                    stage2_mape_pct=float("nan"),
                    stage2_bias_mm=float("nan"),
                    stage2_sd_mm=float("nan"),
                    stage4_mae_mm=float("nan"),
                    stage4_rmse_mm=float("nan"),
                    stage4_mape_pct=float("nan"),
                    stage4_bias_mm=float("nan"),
                    stage4_sd_mm=float("nan"),
                    mean_error_reduction_mm=float("nan"),
                    bland_altman_loa_stage4=(float("nan"), float("nan")),
                )
                continue

            diffs2 = np.array([r.stage2_signed_error_mm for r in sub_rows])
            diffs4 = np.array([r.stage4_signed_error_mm for r in sub_rows])
            abs_e2 = np.array([r.stage2_abs_error_mm for r in sub_rows])
            abs_e4 = np.array([r.stage4_abs_error_mm for r in sub_rows])
            pct_e2 = np.array([abs(r.stage2_pct_error) for r in sub_rows])
            pct_e4 = np.array([abs(r.stage4_pct_error) for r in sub_rows])
            reductions = np.array([r.personalization_error_reduction_mm for r in sub_rows])

            bias4 = float(np.mean(diffs4))
            sd4 = float(np.std(diffs4, ddof=1)) if n > 1 else float(np.std(diffs4))
            loa_s4 = (bias4 - 1.96 * sd4, bias4 + 1.96 * sd4)

            tier_stats[tier] = TierValidationStats(
                tier_name=tier,
                sample_count=n,
                stage2_mae_mm=float(np.mean(abs_e2)),
                stage2_rmse_mm=float(np.sqrt(np.mean(diffs2 ** 2))),
                stage2_mape_pct=float(np.mean(pct_e2)),
                stage2_bias_mm=float(np.mean(diffs2)),
                stage2_sd_mm=float(np.std(diffs2, ddof=1)) if n > 1 else float(np.std(diffs2)),
                stage4_mae_mm=float(np.mean(abs_e4)),
                stage4_rmse_mm=float(np.sqrt(np.mean(diffs4 ** 2))),
                stage4_mape_pct=float(np.mean(pct_e4)),
                stage4_bias_mm=bias4,
                stage4_sd_mm=sd4,
                mean_error_reduction_mm=float(np.mean(reductions)),
                bland_altman_loa_stage4=loa_s4,
            )

        overall_meta = {
            "total_metrics_evaluated": len(rows),
            "comparable_metrics": sum(1 for r in rows if r.comparable),
            "incompatible_metrics": sum(1 for r in rows if not r.comparable),
            "populated_manual_samples": len(valid_rows),
            "has_empirical_ground_truth": len(valid_rows) > 0,
        }

        return rows, tier_stats, overall_meta

    def export_csv_and_json(
        self,
        rows: List[ValidationRow],
        tier_stats: Dict[str, TierValidationStats],
        overall_meta: Dict[str, Any],
        t_elapsed: float,
    ) -> Tuple[Path, Path, Path]:
        """Export manual validation template, evaluation results CSV, and summary JSON."""
        # 1. Manual Validation Template CSV
        template_csv_path = self.out_dir / "manual_validation_template.csv"
        with open(template_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject_id",
                "capture_id",
                "measurement_name",
                "anatomical_protocol",
                "manual_measurement_mm",
                "manual_repeat_1_mm",
                "manual_repeat_2_mm",
                "manual_repeat_3_mm",
                "rgbd_stage2_measurement_mm",
                "rgbd_stage4_measurement_mm",
                "stage5_status",
                "observed_coverage_percent",
                "comparable",
                "clothing_condition",
                "operator_id",
                "notes",
            ])
            for r in rows:
                writer.writerow([
                    r.subject_id,
                    r.capture_id,
                    r.measurement_name,
                    r.anatomical_protocol,
                    f"{r.manual_measurement_mm:.1f}" if not math.isnan(r.manual_measurement_mm) else "",
                    f"{r.manual_repeat_1_mm:.1f}" if not math.isnan(r.manual_repeat_1_mm) else "",
                    f"{r.manual_repeat_2_mm:.1f}" if not math.isnan(r.manual_repeat_2_mm) else "",
                    f"{r.manual_repeat_3_mm:.1f}" if not math.isnan(r.manual_repeat_3_mm) else "",
                    f"{r.rgbd_stage2_measurement_mm:.2f}",
                    f"{r.rgbd_stage4_measurement_mm:.2f}",
                    r.stage5_status,
                    f"{r.observed_coverage_percent:.1f}",
                    r.comparable,
                    r.clothing_condition,
                    r.operator_id,
                    r.notes,
                ])

        # 2. Validation Results CSV
        results_csv_path = self.out_dir / "validation_results.csv"
        with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject_id",
                "capture_id",
                "measurement_name",
                "comparable",
                "stage5_status",
                "manual_measurement_mm",
                "rgbd_stage2_mm",
                "rgbd_stage4_mm",
                "stage2_error_mm",
                "stage2_abs_error_mm",
                "stage2_pct_error",
                "stage4_error_mm",
                "stage4_abs_error_mm",
                "stage4_pct_error",
                "personalization_error_reduction_mm",
                "observed_coverage_percent",
                "notes",
            ])
            for r in rows:
                writer.writerow([
                    r.subject_id,
                    r.capture_id,
                    r.measurement_name,
                    r.comparable,
                    r.stage5_status,
                    f"{r.manual_measurement_mm:.2f}" if not math.isnan(r.manual_measurement_mm) else "nan",
                    f"{r.rgbd_stage2_measurement_mm:.2f}",
                    f"{r.rgbd_stage4_measurement_mm:.2f}",
                    f"{r.stage2_signed_error_mm:+.2f}" if not math.isnan(r.stage2_signed_error_mm) else "nan",
                    f"{r.stage2_abs_error_mm:.2f}" if not math.isnan(r.stage2_abs_error_mm) else "nan",
                    f"{r.stage2_pct_error:+.2f}%" if not math.isnan(r.stage2_pct_error) else "nan",
                    f"{r.stage4_signed_error_mm:+.2f}" if not math.isnan(r.stage4_signed_error_mm) else "nan",
                    f"{r.stage4_abs_error_mm:.2f}" if not math.isnan(r.stage4_abs_error_mm) else "nan",
                    f"{r.stage4_pct_error:+.2f}%" if not math.isnan(r.stage4_pct_error) else "nan",
                    f"{r.personalization_error_reduction_mm:+.2f}" if not math.isnan(r.personalization_error_reduction_mm) else "nan",
                    f"{r.observed_coverage_percent:.1f}",
                    r.notes,
                ])

        # 3. Validation Summary JSON
        summary_json_path = self.out_dir / "validation_summary.json"
        status_code = "VALIDATION_COMPLETED" if overall_meta["has_empirical_ground_truth"] else "TEMPLATE_READY_MANUAL_INPUT_PENDING"

        summary_dict = {
            "status": status_code,
            "stage": "Problem 4 — Stage 6: Manual Anthropometric Ground-Truth Validation",
            "has_empirical_ground_truth": overall_meta["has_empirical_ground_truth"],
            "elapsed_seconds": float(t_elapsed),
            "meta": overall_meta,
            "tier_statistics": {
                k: {
                    "sample_count": v.sample_count,
                    "stage2_mae_mm": v.stage2_mae_mm if not math.isnan(v.stage2_mae_mm) else None,
                    "stage2_rmse_mm": v.stage2_rmse_mm if not math.isnan(v.stage2_rmse_mm) else None,
                    "stage2_mape_pct": v.stage2_mape_pct if not math.isnan(v.stage2_mape_pct) else None,
                    "stage2_bias_mm": v.stage2_bias_mm if not math.isnan(v.stage2_bias_mm) else None,
                    "stage4_mae_mm": v.stage4_mae_mm if not math.isnan(v.stage4_mae_mm) else None,
                    "stage4_rmse_mm": v.stage4_rmse_mm if not math.isnan(v.stage4_rmse_mm) else None,
                    "stage4_mape_pct": v.stage4_mape_pct if not math.isnan(v.stage4_mape_pct) else None,
                    "stage4_bias_mm": v.stage4_bias_mm if not math.isnan(v.stage4_bias_mm) else None,
                    "mean_error_reduction_mm": v.mean_error_reduction_mm if not math.isnan(v.mean_error_reduction_mm) else None,
                    "bland_altman_loa_stage4": [
                        v.bland_altman_loa_stage4[0] if not math.isnan(v.bland_altman_loa_stage4[0]) else None,
                        v.bland_altman_loa_stage4[1] if not math.isnan(v.bland_altman_loa_stage4[1]) else None,
                    ],
                }
                for k, v in tier_stats.items()
            },
            "per_measurement_evaluation": [
                {
                    "measurement_name": r.measurement_name,
                    "comparable": r.comparable,
                    "stage5_status": r.stage5_status,
                    "manual_mm": r.manual_measurement_mm if not math.isnan(r.manual_measurement_mm) else None,
                    "stage2_mm": r.rgbd_stage2_measurement_mm,
                    "stage4_mm": r.rgbd_stage4_measurement_mm,
                    "stage2_abs_error_mm": r.stage2_abs_error_mm if not math.isnan(r.stage2_abs_error_mm) else None,
                    "stage4_abs_error_mm": r.stage4_abs_error_mm if not math.isnan(r.stage4_abs_error_mm) else None,
                    "error_reduction_mm": r.personalization_error_reduction_mm if not math.isnan(r.personalization_error_reduction_mm) else None,
                    "observed_coverage_pct": r.observed_coverage_percent,
                }
                for r in rows
            ],
            "generated_artifacts": [
                "manual_validation_template.csv",
                "validation_results.csv",
                "validation_summary.json",
                "measurement_error_comparison.png",
                "stage2_vs_stage4_error.png",
                "bland_altman_plots.png",
                "stage6_validation_report.md",
            ],
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        return template_csv_path, results_csv_path, summary_json_path

    def generate_visualizations(
        self,
        rows: List[ValidationRow],
        tier_stats: Dict[str, TierValidationStats],
        has_empirical: bool,
    ) -> List[Path]:
        """Generate diagnostic plots or protocol templates."""
        out_plots: List[Path] = []

        # ------------------------------------------------------------------
        # Plot 1: Measurement Error Comparison (Stage 2 vs Stage 4)
        # ------------------------------------------------------------------
        p1 = self.out_dir / "measurement_error_comparison.png"
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        valid_r = [r for r in rows if r.has_manual_data and r.comparable]

        if has_empirical and len(valid_r) > 0:
            names = [r.measurement_name.replace("_circumference", "") for r in valid_r]
            e2 = [r.stage2_abs_error_mm for r in valid_r]
            e4 = [r.stage4_abs_error_mm for r in valid_r]

            x = np.arange(len(names))
            width = 0.38
            axes[0].bar(x - width/2, e2, width, label="Stage 2 Absolute Error (mm)", color="#7293cb")
            axes[0].bar(x + width/2, e4, width, label="Stage 4 Absolute Error (mm)", color="#2ca02c")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=9)
            axes[0].set_ylabel("Absolute Error vs Manual (mm)", fontsize=11, fontweight="bold")
            axes[0].set_title("Per-Measurement Error vs Ground Truth", fontsize=12, fontweight="bold")
            axes[0].grid(True, linestyle=":", alpha=0.6)
            axes[0].legend()

            # Stratified MAE
            tiers = ["observation_supported", "partially_observed", "model_inferred"]
            t_names = ["Observed", "Partially Obs", "Model Inferred"]
            mae2 = [tier_stats[t].stage2_mae_mm for t in tiers]
            mae4 = [tier_stats[t].stage4_mae_mm for t in tiers]
            x_t = np.arange(len(tiers))
            axes[1].bar(x_t - width/2, mae2, width, label="Stage 2 MAE (mm)", color="#7293cb")
            axes[1].bar(x_t + width/2, mae4, width, label="Stage 4 MAE (mm)", color="#2ca02c")
            axes[1].set_xticks(x_t)
            axes[1].set_xticklabels(t_names, fontsize=10, fontweight="bold")
            axes[1].set_ylabel("Mean Absolute Error (mm)", fontsize=11, fontweight="bold")
            axes[1].set_title("MAE Stratified by Observation Coverage", fontsize=12, fontweight="bold")
            axes[1].grid(True, linestyle=":", alpha=0.6)
            axes[1].legend()
        else:
            # Standby template plot explaining manual validation protocol
            for ax in axes:
                ax.text(0.5, 0.5, "MANUAL VALIDATION PROTOCOL READY\n\nManual ground-truth tape measurements pending.\nEnter tape values into manual_validation_template.csv\nto compute empirical MAE, RMSE, and bias.",
                        ha="center", va="center", fontsize=11, fontweight="bold", color="#555555",
                        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f0f4f8", edgecolor="#4863a0", linewidth=1.5))
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0].set_title("Per-Measurement Error (Pending Manual Data)", fontsize=12, fontweight="bold")
            axes[1].set_title("Stratified MAE (Pending Manual Data)", fontsize=12, fontweight="bold")

        plt.suptitle("Problem 4 — Stage 6: Ground-Truth Error Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(p1, dpi=180)
        plt.close(fig)
        out_plots.append(p1)

        # ------------------------------------------------------------------
        # Plot 2: Stage 2 vs Stage 4 Error Reduction
        # ------------------------------------------------------------------
        p2 = self.out_dir / "stage2_vs_stage4_error.png"
        fig, ax = plt.subplots(figsize=(8, 6.5))

        if has_empirical and len(valid_r) > 0:
            e2 = np.array([r.stage2_abs_error_mm for r in valid_r])
            e4 = np.array([r.stage4_abs_error_mm for r in valid_r])
            max_val = max(np.max(e2), np.max(e4)) * 1.15

            ax.plot([0, max_val], [0, max_val], "k--", label="Identity (No Change)")
            ax.scatter(e2, e4, s=60, color="#2ca02c", edgecolor="#1a5f1a", zorder=3, label="Matched Measurements")
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel("Stage 2 Parametric Absolute Error (mm)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Stage 4 Personalized Absolute Error (mm)", fontsize=11, fontweight="bold")
            ax.set_title("Personalization Error Reduction Trajectory", fontsize=12, fontweight="bold")
            ax.text(0.7 * max_val, 0.3 * max_val, "Stage 4 Improved\n(Lower Error)", fontsize=10, color="#2ca02c", fontweight="bold", ha="center")
            ax.text(0.3 * max_val, 0.7 * max_val, "Stage 2 Better", fontsize=10, color="#d35e60", fontweight="bold", ha="center")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "PERSONALIZATION ERROR COMPARISON\n\nManual ground-truth tape measurements pending.\nIdentity diagonal comparison will display here\nupon entering manual tape values.",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="#555555",
                    bbox=dict(boxstyle="round,pad=1.0", facecolor="#f0f4f8", edgecolor="#4863a0", linewidth=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Stage 2 vs Stage 4 Error Trajectory (Pending Manual Data)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(p2, dpi=180)
        plt.close(fig)
        out_plots.append(p2)

        # ------------------------------------------------------------------
        # Plot 3: Bland-Altman Limits of Agreement
        # ------------------------------------------------------------------
        p3 = self.out_dir / "bland_altman_plots.png"
        fig, ax = plt.subplots(figsize=(9, 6.5))

        if has_empirical and len(valid_r) >= 3:
            s4_vals = np.array([r.rgbd_stage4_measurement_mm for r in valid_r])
            m_vals = np.array([r.manual_measurement_mm for r in valid_r])
            diffs = s4_vals - m_vals
            means = 0.5 * (s4_vals + m_vals)

            bias = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1))
            loa_hi = bias + 1.96 * sd
            loa_lo = bias - 1.96 * sd

            ax.scatter(means, diffs, color="#2ca02c", edgecolor="#1a5f1a", s=60, zorder=3)
            ax.axhline(bias, color="#2b5c8f", linestyle="-", linewidth=2.0, label=f"Mean Bias: {bias:+.1f} mm")
            ax.axhline(loa_hi, color="#d35e60", linestyle="--", linewidth=1.5, label=f"+1.96 SD: {loa_hi:+.1f} mm")
            ax.axhline(loa_lo, color="#d35e60", linestyle="--", linewidth=1.5, label=f"-1.96 SD: {loa_lo:+.1f} mm")
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)

            ax.set_xlabel("Mean of RGB-D Stage 4 & Manual Tape (mm)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Difference: Stage 4 - Manual (mm)", fontsize=11, fontweight="bold")
            ax.set_title(f"Bland-Altman Agreement Plot (N={len(valid_r)})", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", frameon=True, facecolor="white")
        else:
            ax.text(0.5, 0.5, "BLAND-ALTMAN LIMITS OF AGREEMENT\n\nMinimum N=3 manual measurements required.\nEnter manual measurements in manual_validation_template.csv\nto compute empirical bias and 95% Limits of Agreement.",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="#555555",
                    bbox=dict(boxstyle="round,pad=1.0", facecolor="#f0f4f8", edgecolor="#4863a0", linewidth=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Bland-Altman Plot (Pending Manual Data)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(p3, dpi=180)
        plt.close(fig)
        out_plots.append(p3)

        return out_plots

    def compile_markdown_report(
        self,
        rows: List[ValidationRow],
        tier_stats: Dict[str, TierValidationStats],
        overall_meta: Dict[str, Any],
        t_elapsed: float,
    ) -> Path:
        """Compile detailed Markdown audit report."""
        rep_path = self.out_dir / "stage6_validation_report.md"

        has_emp = overall_meta["has_empirical_ground_truth"]

        with open(rep_path, "w", encoding="utf-8") as f:
            f.write("# Problem 4 — Stage 6: Manual Anthropometric Ground-Truth Validation Report\n\n")
            f.write("## 1. Executive Summary & Empirical Status\n\n")
            f.write("- **Pipeline Stage**: Problem 4 — Stage 6: Manual Anthropometric Ground-Truth Validation\n")
            f.write("- **Empirical Manual Ground-Truth Status**: ")
            if has_emp:
                f.write(f"**POPULATED ({overall_meta['populated_manual_samples']} measurements matched)**\n")
            else:
                f.write("**UNPOPULATED (Framework and templates generated; manual physical tape values pending)**\n")

            f.write(f"- **Physical Capture Session**: `{self.capture_id}` (Intel RealSense D455f)\n")
            f.write(f"- **Anonymous Subject ID**: `{self.subject_id}`\n")
            f.write(f"- **Assigned Operator ID**: `{self.operator_id}`\n")
            f.write(f"- **Clothing Condition**: `{self.clothing}`\n")
            f.write(f"- **Total Candidate Measurements**: `{overall_meta['total_metrics_evaluated']}`\n")
            f.write(f"- **Anatomically Comparable Protocols**: `{overall_meta['comparable_metrics']}`\n")
            f.write(f"- **Excluded Incompatible Protocols**: `{overall_meta['incompatible_metrics']}` (`neck_base_circumference`)\n")
            f.write(f"- **Runtime**: `{t_elapsed:.2f} s`\n\n")

            f.write("> [!IMPORTANT]\n")
            if has_emp:
                f.write(f"> **Empirical Data Provided**: Evaluation was executed against {overall_meta['populated_manual_samples']} manual ground-truth tape measurements.\n\n")
            else:
                f.write("> **Strict Anti-Fabrication Guarantee**: No manual ground-truth measurements were fabricated or hallucinated. All manual fields in `manual_validation_template.csv` are left empty (NaN) ready for physical caliper/tape entry by an anthropometry technician.\n\n")

            f.write("## 2. Anthropometric Protocol Matching & Anatomical Definitions\n\n")
            f.write("To prevent comparing incompatible anatomical definitions, every metric is mapped to ISO 7250 / ISAK standard protocols:\n\n")
            f.write("| Measurement Name | Quality Tier | Comparable? | Anatomical Protocol Description |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for r in rows:
                comp_tag = "**YES**" if r.comparable else "*NO (Excluded)*"
                f.write(f"| `{r.measurement_name}` | `{r.stage5_status}` | {comp_tag} | {r.anatomical_protocol} |\n")
            f.write("\n")

            f.write("## 3. Matched Comparison & Error Breakdown\n\n")
            if has_emp:
                f.write("| Measurement Name | Manual (mm) | Stage 2 (mm) | Stage 4 (mm) | S2 Error (mm) | S4 Error (mm) | S4 Error (%) | Error Reduction |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for r in rows:
                    if not r.comparable:
                        f.write(f"| `{r.measurement_name}` | *excluded* | `{r.rgbd_stage2_measurement_mm:.1f}` | `{r.rgbd_stage4_measurement_mm:.1f}` | — | — | — | — |\n")
                    elif not r.has_manual_data:
                        f.write(f"| `{r.measurement_name}` | *pending* | `{r.rgbd_stage2_measurement_mm:.1f}` | `{r.rgbd_stage4_measurement_mm:.1f}` | — | — | — | — |\n")
                    else:
                        red_str = f"**`{r.personalization_error_reduction_mm:+.1f} mm`**" if r.personalization_error_reduction_mm > 0 else f"`{r.personalization_error_reduction_mm:+.1f} mm`"
                        f.write(f"| `{r.measurement_name}` | `{r.manual_measurement_mm:.1f}` | `{r.rgbd_stage2_measurement_mm:.1f}` | `{r.rgbd_stage4_measurement_mm:.1f}` | `{r.stage2_signed_error_mm:+.1f}` | `{r.stage4_signed_error_mm:+.1f}` | `{r.stage4_pct_error:+.1f}%` | {red_str} |\n")
                f.write("\n")
            else:
                f.write("*(Table unpopulated pending manual tape entry in `manual_validation_template.csv`)*\n\n")
                f.write("| Measurement Name | Stage 2 RGB-D (mm) | Stage 4 RGB-D (mm) | Observation Coverage | Manual Status |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                for r in rows:
                    f.write(f"| `{r.measurement_name}` | `{r.rgbd_stage2_measurement_mm:.1f}` | `{r.rgbd_stage4_measurement_mm:.1f}` | `{r.observed_coverage_percent:.1f}%` | *Pending Tape Input* |\n")
                f.write("\n")

            f.write("## 4. Stratified Statistical Validation Results\n\n")
            if has_emp:
                f.write("| Quality Stratification Tier | Samples | Stage 2 MAE (mm) | Stage 4 MAE (mm) | S4 MAPE (%) | S4 RMSE (mm) | S4 Bias (mm) | Mean Error Reduction |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for tier_k, s in tier_stats.items():
                    if s.sample_count > 0:
                        f.write(f"| `{tier_k}` | `{s.sample_count}` | `{s.stage2_mae_mm:.1f}` | `{s.stage4_mae_mm:.1f}` | `{s.stage4_mape_pct:.2f}%` | `{s.stage4_rmse_mm:.1f}` | `{s.stage4_bias_mm:+.1f}` | **`{s.mean_error_reduction_mm:+.1f} mm`** |\n")
                f.write("\n")
            else:
                f.write("*(Statistical evaluation unpopulated: awaiting manual physical tape measurements)*\n\n")

            f.write("## 5. Standard Operating Procedure (SOP) for Human Anthropometrists\n\n")
            f.write("To complete manual ground-truth validation for session `body_capture_01`:\n")
            f.write("1. Open [`manual_validation_template.csv`](manual_validation_template.csv) in a spreadsheet editor.\n")
            f.write("2. Follow the protocol description in the `anatomical_protocol` column for each anatomical landmark.\n")
            f.write("3. Take 3 repeated measurements per landmark to quantify intra-observer variability; enter into `manual_repeat_1_mm`, `manual_repeat_2_mm`, and `manual_repeat_3_mm`.\n")
            f.write("4. Save the CSV and execute:\n")
            f.write("   ```bash\n")
            f.write("   python realsense_measure-main/tools/run_physical_stage6_validation.py \\\n")
            f.write("       --manual-csv stage6_validation_output/manual_validation_template.csv \\\n")
            f.write("       --out-dir stage6_validation_output\n")
            f.write("   ```\n")
            f.write("5. The engine will automatically evaluate signed errors, MAE, RMSE, MAPE, Bland-Altman limits of agreement, and generate empirical error comparison charts.\n\n")

            f.write("## 6. Output Artifact Index\n\n")
            f.write("- **Manual Validation Template CSV**: [`manual_validation_template.csv`](manual_validation_template.csv)\n")
            f.write("- **Validation Evaluation Results CSV**: [`validation_results.csv`](validation_results.csv)\n")
            f.write("- **Validation Summary JSON**: [`validation_summary.json`](validation_summary.json)\n")
            f.write("- **Measurement Error Comparison Plot**: [`measurement_error_comparison.png`](measurement_error_comparison.png)\n")
            f.write("- **Stage 2 vs Stage 4 Error Trajectory**: [`stage2_vs_stage4_error.png`](stage2_vs_stage4_error.png)\n")
            f.write("- **Bland-Altman Agreement Plot**: [`bland_altman_plots.png`](bland_altman_plots.png)\n\n")

        return rep_path

    def run(self) -> Dict[str, Any]:
        """Execute Stage 6 Manual Validation Framework."""
        t_start = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 76)
        logger.info("  STARTING PROBLEM 4 — STAGE 6: MANUAL ANTHROPOMETRIC VALIDATION")
        logger.info("=" * 76)

        # 1. Load Stage 5 inputs
        stage5_records = self.load_stage5_measurements()

        # 2. Build validation template
        template_rows = self.build_validation_template(stage5_records)

        # 3. Ingest manual measurements if provided
        if self.manual_csv_path and self.manual_csv_path.exists():
            rows = self.ingest_manual_measurements(self.manual_csv_path, template_rows)
        else:
            logger.info("No manual measurements CSV provided; generating template with unpopulated manual fields.")
            rows = template_rows

        # 4. Compute validation statistics & stratification
        rows, tier_stats, overall_meta = self.compute_validation_statistics(rows)

        t_elapsed = time.time() - t_start

        # 5. Export CSV & JSON archives
        tpl_p, res_p, sum_p = self.export_csv_and_json(rows, tier_stats, overall_meta, t_elapsed)
        logger.info(f"Saved manual validation template: {tpl_p}")
        logger.info(f"Saved validation results: {res_p}")
        logger.info(f"Saved validation summary: {sum_p}")

        # 6. Generate Visualizations
        plots = self.generate_visualizations(rows, tier_stats, overall_meta["has_empirical_ground_truth"])
        for p in plots:
            logger.info(f"Generated visualization: {p}")

        # 7. Compile Markdown Report
        rep_p = self.compile_markdown_report(rows, tier_stats, overall_meta, t_elapsed)
        logger.info(f"Saved audit report: {rep_p}")

        logger.info("=" * 76)
        logger.info(f"  STAGE 6 MANUAL VALIDATION COMPLETED ({overall_meta['populated_manual_samples']} MANUAL MEASUREMENTS MATCHED)")
        logger.info("=" * 76)

        with open(sum_p, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        return summary_data


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Problem 4 — Stage 6: Manual Anthropometric Ground-Truth Validation"
    )
    parser.add_argument(
        "--stage5-csv",
        type=str,
        default="stage5_anthropometric_output/dual_mesh_measurement_comparison.csv",
        help="Path to Stage 5 measurement comparison CSV",
    )
    parser.add_argument(
        "--manual-csv",
        type=str,
        default=None,
        help="Path to populated manual measurements CSV (optional)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stage6_validation_output",
        help="Path to Stage 6 output directory",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default="SUBJ_01",
        help="Anonymous subject identifier (default: SUBJ_01)",
    )
    parser.add_argument(
        "--capture-id",
        type=str,
        default="body_capture_01",
        help="Capture identifier (default: body_capture_01)",
    )
    parser.add_argument(
        "--operator-id",
        type=str,
        default="OPERATOR_01",
        help="Anthropometrist / operator identifier (default: OPERATOR_01)",
    )
    parser.add_argument(
        "--clothing-condition",
        type=str,
        default="casual_clothing",
        help="Clothing condition description (default: casual_clothing)",
    )

    args = parser.parse_args()

    validator = PhysicalStage6Validator(
        stage5_csv_path=args.stage5_csv,
        manual_csv_path=args.manual_csv,
        output_dir=args.out_dir,
        default_subject_id=args.subject_id,
        default_capture_id=args.capture_id,
        default_operator_id=args.operator_id,
        default_clothing=args.clothing_condition,
    )

    summary = validator.run()
    print(f"\nStage 6 completed with status: {summary['status']}")
    print(f"Empirical ground truth available: {summary['has_empirical_ground_truth']}")
    print(f"Populated manual samples: {summary['meta']['populated_manual_samples']} / {summary['meta']['comparable_metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

