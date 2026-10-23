"""
evaluate_head_scan_offline.py - Standalone offline evaluator for HeadTarget segmentation.

Evaluates existing PLY point clouds through the active production HeadTarget pipeline
without accessing the live camera or modifying production code, thresholds, or scan files.

Features:
  - Evaluates a single PLY file or a directory (with optional recursive discovery).
  - Reports input and preprocessing geometry (point counts, finite points, AABB extents).
  - Invokes active production HeadTarget.segment().
  - Explicitly labels unexposed internal segmentation decisions as 'not_exposed'
    rather than silently recreating a potentially divergent version of the algorithm.
  - Invokes active production HeadTarget.is_plausible_head() and HeadTarget.measure().
  - Outputs machine-readable JSON and concise CSV summaries.
  - Safe against missing, empty, or corrupt point clouds without aborting batches.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

# Ensure repository root and package are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / 'realsense_measure'
for p in [str(PACKAGE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import PipelineConfig, PreprocessConfig
from targets.head import HeadTarget, _sanitize_points


def get_git_revision() -> str:
    """Return the current Git commit hash if available, otherwise a fallback string."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return rev
    except Exception:
        return "unknown"


def _compute_aabb_info(pts: np.ndarray) -> Dict[str, Any]:
    """Compute axis-aligned bounding box min, max, and extents in millimetres."""
    if len(pts) == 0:
        return {
            "min_mm": [0.0, 0.0, 0.0],
            "max_mm": [0.0, 0.0, 0.0],
            "extent_mm": [0.0, 0.0, 0.0],
            "max_span_mm": 0.0,
            "min_span_mm": 0.0,
        }
    mins = np.min(pts, axis=0) * 1000.0
    maxs = np.max(pts, axis=0) * 1000.0
    exts = maxs - mins
    sorted_exts = sorted(exts)
    return {
        "min_mm": [round(float(v), 2) for v in mins],
        "max_mm": [round(float(v), 2) for v in maxs],
        "extent_mm": [round(float(v), 2) for v in exts],
        "max_span_mm": round(float(sorted_exts[-1]), 2),
        "min_span_mm": round(float(sorted_exts[0]), 2),
    }


def evaluate_single_ply(
    ply_path: Path,
    target: Optional[HeadTarget] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single PLY scan using the active HeadTarget pipeline.

    Returns a structured dictionary with input, segmentation, downstream validation,
    and diagnostic information. Never raises an uncaught exception.
    """
    if target is None:
        target = HeadTarget()

    record: Dict[str, Any] = {
        "input_path": str(ply_path),
        "input_filename": ply_path.name,
        "execution_error": None,
        "input_and_preprocessing": {},
        "segmentation_diagnostics": {},
        "downstream_validation": {},
        "anatomical_success_claim": "none (geometric observation only; no ground truth available)",
    }

    if not ply_path.exists():
        record["execution_error"] = f"File not found: {ply_path}"
        return record

    if not ply_path.is_file():
        record["execution_error"] = f"Path is not a regular file: {ply_path}"
        return record

    # 1. Load point cloud
    try:
        # Pre-check header to distinguish corrupt non-PLY files from valid empty PLY
        with ply_path.open("rb") as fp:
            header_start = fp.read(4)
            if header_start != b"ply\n" and header_start != b"ply\r":
                record["execution_error"] = f"Corrupt or invalid PLY header in: {ply_path}"
                return record

        pcd = o3d.io.read_point_cloud(str(ply_path))
    except Exception as err:
        record["execution_error"] = f"Failed to read point cloud: {err}"
        return record

    raw_pts = np.asarray(pcd.points)
    n_raw = len(raw_pts)
    finite_pts = _sanitize_points(raw_pts)
    n_finite = len(finite_pts)

    in_aabb = _compute_aabb_info(finite_pts)
    record["input_and_preprocessing"] = {
        "raw_point_count": n_raw,
        "finite_point_count": n_finite,
        "dominant_cluster_point_count": "not_exposed",
        "config_used": {
            "cluster_eps_m": getattr(target.cfg, "cluster_eps_m", 0.025),
            "cluster_min_points": getattr(target.cfg, "cluster_min_points", 20),
            "voxel_size_m": getattr(target.cfg, "voxel_size_m", 0.005),
        },
        "aabb_min_mm": in_aabb["min_mm"],
        "aabb_max_mm": in_aabb["max_mm"],
        "aabb_extent_mm": in_aabb["extent_mm"],
        "max_span_mm": in_aabb["max_span_mm"],
        "min_span_mm": in_aabb["min_span_mm"],
    }

    if n_finite == 0:
        record["execution_error"] = "Point cloud contains zero finite coordinates"
        record["segmentation_diagnostics"] = {
            "output_point_count": 0,
            "point_reduction_count": 0,
            "retention_ratio_raw": 0.0,
            "retention_ratio_dominant": "not_exposed",
            "cut_detected_by_reduction": False,
            "is_fallback_unsegmented": True,
            "longitudinal_span_gate_passed": "not_exposed",
            "longitudinal_axis_u": "not_exposed",
            "orientation_decision": "not_exposed",
            "cross_sectional_areas_evaluated": "not_exposed",
            "candidate_indices_evaluated": "not_exposed",
            "candidate_criteria_results": "not_exposed",
            "cut_attempted": "not_exposed",
            "cut_or_fallback_reason": "Zero finite points",
        }
        record["downstream_validation"] = {
            "is_plausible": False,
            "plausibility_reason": "Zero finite points",
            "measurement_status": "unverified",
            "validated_volume_m3": None,
            "estimated_volume_m3": 0.0,
            "cranial_envelope_mm": {"breadth_mm": 0.0, "length_mm": 0.0, "height_mm": 0.0},
            "validation_diagnostics": ["Zero finite points"],
        }
        return record

    # 2. Production Segmentation
    try:
        seg_pcd = target.segment(pcd)
    except Exception as err:
        record["execution_error"] = f"Exception in HeadTarget.segment(): {err}"
        return record

    seg_pts = _sanitize_points(np.asarray(seg_pcd.points))
    n_seg = len(seg_pts)
    out_aabb = _compute_aabb_info(seg_pts)

    retention_raw = round(n_seg / max(n_raw, 1), 4)
    point_reduction = n_raw - n_seg
    cut_detected_by_reduction = bool(point_reduction > 0)

    # Note on internal variables:
    # As audited in Problem 10B, HeadTarget.segment() executes internal operations
    # (PCA axis calculation, 25-slice areas, candidate criteria) but does not record
    # or return them. Per specification, we explicitly record them as 'not_exposed'
    # rather than creating a potentially divergent duplicate algorithm.
    record["segmentation_diagnostics"] = {
        "output_point_count": n_seg,
        "point_reduction_count": point_reduction,
        "retention_ratio_raw": retention_raw,
        "retention_ratio_dominant": "not_exposed",
        "cut_detected_by_reduction": cut_detected_by_reduction,
        "reduction_caveat": (
            "Point reduction reflects combined effects of DBSCAN outlier removal and/or constriction cutting. "
            "A point count drop does not guarantee an anatomically valid neck cut."
        ),
        "longitudinal_span_gate_passed": "not_exposed",
        "longitudinal_axis_u": "not_exposed",
        "orientation_decision": "not_exposed",
        "cross_sectional_areas_evaluated": "not_exposed",
        "candidate_indices_evaluated": "not_exposed",
        "candidate_criteria_results": "not_exposed",
        "cut_attempted": "not_exposed",
        "is_fallback_unsegmented": bool(n_seg == n_raw),
        "cut_or_fallback_reason": (
            "not_exposed (HeadTarget.segment returns only PointCloud; "
            + ("point count unchanged, zero cut applied" if n_seg == n_raw else "point count changed")
            + ")"
        ),
        "output_aabb_min_mm": out_aabb["min_mm"],
        "output_aabb_max_mm": out_aabb["max_mm"],
        "output_aabb_extent_mm": out_aabb["extent_mm"],
        "output_max_span_mm": out_aabb["max_span_mm"],
        "output_min_span_mm": out_aabb["min_span_mm"],
    }

    # 3. Downstream Plausibility Check
    try:
        is_plaus, plaus_reason = target.is_plausible_head(seg_pcd)
    except Exception as err:
        is_plaus = False
        plaus_reason = f"Exception in is_plausible_head(): {err}"

    # 4. Downstream Measurement
    try:
        meas_result = target.measure(seg_pcd)
    except Exception as err:
        meas_result = {
            "status": "unverified",
            "error": f"Exception in HeadTarget.measure(): {err}",
            "cranial_envelope_3d": {"breadth_mm": 0.0, "length_mm": 0.0, "height_mm": 0.0},
            "oriented_bbox": {"volume_m3": None, "estimated_volume_m3": 0.0},
            "diagnostics": [f"Exception in measure(): {err}"],
        }

    cranial = meas_result.get("cranial_envelope_3d", {})
    obb = meas_result.get("oriented_bbox", {})
    record["downstream_validation"] = {
        "is_plausible": is_plaus,
        "plausibility_reason": plaus_reason,
        "measurement_status": meas_result.get("status", "unverified"),
        "cranial_envelope_mm": {
            "breadth_mm": cranial.get("breadth_mm", 0.0),
            "length_mm": cranial.get("length_mm", 0.0),
            "height_mm": cranial.get("height_mm", 0.0),
        },
        "validated_volume_m3": meas_result.get("volume_m3", obb.get("volume_m3")),
        "estimated_volume_m3": obb.get("estimated_volume_m3", 0.0),
        "symmetry_residual_mm": meas_result.get("symmetry_residual_mm"),
        "validation_diagnostics": meas_result.get("diagnostics", []),
        "boundary_checks": meas_result.get("boundary_checks", {}),
    }

    return record


def discover_ply_files(
    input_path: Path,
    recursive: bool = False,
    pattern: str = "*.ply",
) -> List[Path]:
    """Discover eligible PLY files from a file path or directory."""
    if not input_path.exists():
        return [input_path]  # return so evaluator can report FileNotFoundError

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        if recursive:
            return sorted(list(input_path.rglob(pattern)))
        else:
            return sorted(list(input_path.glob(pattern)))

    return []


def run_evaluation(
    inputs: List[Path],
    output_dir: Optional[Path] = None,
    output_name: str = "head_eval_summary",
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Run offline evaluation on a list of input PLY paths.

    Writes machine-readable JSON and CSV summaries if output_dir is specified.
    Returns the comprehensive summary dictionary.
    """
    git_rev = get_git_revision()
    target = HeadTarget()

    results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    if not quiet:
        print("=" * 80)
        print(f"  HeadTarget Offline Segmentation Evaluator")
        print(f"  Revision : {git_rev}")
        print(f"  Inputs   : {len(inputs)} files to evaluate")
        print("=" * 80)

    for idx, p in enumerate(inputs, 1):
        if not quiet:
            print(f"[{idx}/{len(inputs)}] Evaluating: {p}")

        rec = evaluate_single_ply(p, target=target)
        results.append(rec)

        # Build concise CSV row
        inp_sec = rec.get("input_and_preprocessing", {})
        seg_sec = rec.get("segmentation_diagnostics", {})
        down_sec = rec.get("downstream_validation", {})
        cranial = down_sec.get("cranial_envelope_mm", {})

        csv_row = {
            "input_path": rec.get("input_path", str(p)),
            "input_filename": rec.get("input_filename", p.name),
            "execution_error": rec.get("execution_error") or "",
            "raw_point_count": inp_sec.get("raw_point_count", 0),
            "finite_point_count": inp_sec.get("finite_point_count", 0),
            "dominant_cluster_points": inp_sec.get("dominant_cluster_point_count", "not_exposed"),
            "input_max_span_mm": inp_sec.get("max_span_mm", 0.0),
            "output_point_count": seg_sec.get("output_point_count", 0),
            "point_reduction_count": seg_sec.get("point_reduction_count", 0),
            "retention_ratio_raw": seg_sec.get("retention_ratio_raw", 0.0),
            "cut_detected_by_reduction": seg_sec.get("cut_detected_by_reduction", False),
            "is_fallback_unsegmented": seg_sec.get("is_fallback_unsegmented", False),
            "segmentation_diagnostics": "not_exposed",
            "is_plausible": down_sec.get("is_plausible", False),
            "plausibility_reason": down_sec.get("plausibility_reason", ""),
            "measurement_status": down_sec.get("measurement_status", "unverified"),
            "breadth_mm": cranial.get("breadth_mm", 0.0),
            "length_mm": cranial.get("length_mm", 0.0),
            "height_mm": cranial.get("height_mm", 0.0),
            "validated_volume_m3": down_sec.get("validated_volume_m3") or "",
            "estimated_volume_m3": down_sec.get("estimated_volume_m3", 0.0),
            "symmetry_residual_mm": down_sec.get("symmetry_residual_mm") or "",
            "diagnostics_count": len(down_sec.get("validation_diagnostics", [])),
        }
        csv_rows.append(csv_row)

        if not quiet:
            err = rec.get("execution_error")
            if err:
                print(f"    [!] Error: {err}")
            else:
                n_in = inp_sec.get("raw_point_count", 0)
                n_out = seg_sec.get("output_point_count", 0)
                st = down_sec.get("measurement_status", "unverified")
                b = cranial.get("breadth_mm", 0.0)
                l = cranial.get("length_mm", 0.0)
                h = cranial.get("height_mm", 0.0)
                print(f"    Points: {n_in:,} -> {n_out:,} | Status: {st.upper()} | Dims (BxLxH mm): {b:.1f} x {l:.1f} x {h:.1f}")

    summary: Dict[str, Any] = {
        "evaluator_name": "evaluate_head_scan_offline",
        "evaluator_version": "1.0.0",
        "git_commit": git_rev,
        "total_files_evaluated": len(inputs),
        "successful_evaluations": sum(1 for r in results if not r.get("execution_error")),
        "failed_evaluations": sum(1 for r in results if r.get("execution_error")),
        "validated_measurements": sum(
            1 for r in results if r.get("downstream_validation", {}).get("measurement_status") == "validated"
        ),
        "unverified_measurements": sum(
            1 for r in results if r.get("downstream_validation", {}).get("measurement_status") == "unverified"
        ),
        "records": results,
    }

    # Write outputs if output directory is provided
    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        json_path = out_dir_path / f"{output_name}.json"
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        csv_path = out_dir_path / f"{output_name}.csv"
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        if not quiet:
            print("-" * 80)
            print(f"  Summary saved to:")
            print(f"    JSON: {json_path}")
            print(f"    CSV : {csv_path}")
            print("=" * 80)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_head_scan_offline.py",
        description="Evaluate existing PLY head scans offline using active HeadTarget logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more PLY files or directories to evaluate.",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_opt",
        help="Optional single file or directory to evaluate (alternative to positional inputs).",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively discover PLY files within input directories.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ply",
        help="Glob pattern for discovering PLY files in directories.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save JSON and CSV reports (never overwrites input scans).",
    )
    parser.add_argument(
        "--output-name",
        default="head_eval_summary",
        help="Base filename for generated JSON and CSV summaries.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console progress output.",
    )

    args = parser.parse_args()

    # Collect input paths
    all_raw_paths: List[str] = list(args.inputs)
    if args.input_opt:
        all_raw_paths.append(args.input_opt)

    if not all_raw_paths:
        parser.error("No input files or directories provided. Use 'python evaluate_head_scan_offline.py <file_or_dir>'.")

    discovered: List[Path] = []
    for raw in all_raw_paths:
        p = Path(raw)
        discovered.extend(discover_ply_files(p, recursive=args.recursive, pattern=args.pattern))

    # Remove duplicates while preserving order
    seen = set()
    unique_inputs = []
    for p in discovered:
        resolved = str(p.resolve()) if p.exists() else str(p)
        if resolved not in seen:
            seen.add(resolved)
            unique_inputs.append(p)

    output_dir = Path(args.output_dir) if args.output_dir else None
    run_evaluation(
        inputs=unique_inputs,
        output_dir=output_dir,
        output_name=args.output_name,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
