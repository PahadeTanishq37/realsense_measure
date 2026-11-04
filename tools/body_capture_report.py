#!/usr/bin/env python3
"""
tools/body_capture_report.py - Body-capture diagnostics and depth completeness analysis.

Analyzes raw RGB-D capture recordings for --target body scans:
- Person mask extraction via MediaPipe selfie segmentation (with depth-based fallback)
- Valid depth percentage across 5 vertical bands (head, chest, waist, thighs, shins)
- Valid depth percentage for left and right halves
- Median depth of the subject mask
- Percentage of mask pixels within working range (0.40 - 3.00 m)
- Share of zero-depth pixels inside mask whose colour is dark (HSV Value V < 0.20),
  testing the dark-clothing absorption hypothesis
- Console summary table
- Coverage visualization saved to coverage_by_frame.png

Usage:
    python tools/body_capture_report.py <raw dir> [--out-dir <out dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


try:
    from realsense_measure.segmentation import get_person_mask, is_model_available
    SEGMENTATION_AVAILABLE = True
except Exception:
    SEGMENTATION_AVAILABLE = False

try:
    from realsense_measure.camera.recording import load_raw_capture
except Exception:
    load_raw_capture = None


def extract_person_mask(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract binary person mask using MediaPipe or depth foreground fallback.
    """
    mask = None
    if SEGMENTATION_AVAILABLE:
        try:
            mask = get_person_mask(color_bgr)
        except Exception:
            mask = None

    if mask is not None and np.sum(mask) > 100:
        return mask

    # Fallback 1: Background subtraction if background depth reference is available
    if bg_depth_m is not None:
        diff = bg_depth_m - depth_m
        fg = (depth_m >= 0.40) & (depth_m <= 3.00) & (diff > 0.08)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        if n_lbl > 1:
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[largest, cv2.CC_STAT_AREA] >= 200:
                return (lbls == largest)

    # Fallback 2: Connected component of valid foreground depth in 0.4 - 3.0m
    valid_d = (depth_m >= 0.40) & (depth_m <= 3.00)
    if np.any(valid_d):
        d_med = float(np.median(depth_m[valid_d]))
        slice_m = (depth_m >= max(0.40, d_med - 0.45)) & (depth_m <= d_med + 0.45)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(slice_m.astype(np.uint8), connectivity=8)
        if n_lbl > 1:
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[largest, cv2.CC_STAT_AREA] >= 200:
                return (lbls == largest)

    return np.zeros_like(depth_m, dtype=bool)


def analyze_body_frame(
    frame_idx: int,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute body completeness and diagnostic metrics for a single frame.
    """
    h, w = depth_m.shape[:2]
    mask = extract_person_mask(color_bgr, depth_m, bg_depth_m)
    mask_pixels = int(np.sum(mask))

    if mask_pixels < 50:
        return {
            "frame_idx": frame_idx,
            "mask_pixels": mask_pixels,
            "valid_pct": 0.0,
            "median_depth_m": 0.0,
            "in_range_pct": 0.0,
            "head_valid_pct": 0.0,
            "chest_valid_pct": 0.0,
            "waist_valid_pct": 0.0,
            "thighs_valid_pct": 0.0,
            "shins_valid_pct": 0.0,
            "left_valid_pct": 0.0,
            "right_valid_pct": 0.0,
            "dark_zero_depth_share_pct": 0.0,
            "zero_depth_count": 0,
            "dark_zero_count": 0,
            "has_mask": False,
        }

    # Bounding extents of person mask
    ys, xs = np.where(mask)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    h_person = max(1, ymax - ymin)
    x_mid = (xmin + xmax) / 2.0

    # 5 vertical bands: head, chest, waist, thighs, shins
    b0 = ymin
    b1 = ymin + 0.20 * h_person
    b2 = ymin + 0.40 * h_person
    b3 = ymin + 0.60 * h_person
    b4 = ymin + 0.80 * h_person
    b5 = ymax + 1

    row_indices = np.arange(h)[:, None]
    col_indices = np.arange(w)[None, :]

    band_head = mask & (row_indices >= b0) & (row_indices < b1)
    band_chest = mask & (row_indices >= b1) & (row_indices < b2)
    band_waist = mask & (row_indices >= b2) & (row_indices < b3)
    band_thighs = mask & (row_indices >= b3) & (row_indices < b4)
    band_shins = mask & (row_indices >= b4) & (row_indices <= b5)

    def _band_valid(b_mask: np.ndarray) -> float:
        total = np.sum(b_mask)
        if total == 0:
            return 0.0
        valid = np.sum(b_mask & (depth_m > 0.0))
        return float(valid / total * 100.0)

    head_pct = _band_valid(band_head)
    chest_pct = _band_valid(band_chest)
    waist_pct = _band_valid(band_waist)
    thighs_pct = _band_valid(band_thighs)
    shins_pct = _band_valid(band_shins)

    # Left and Right halves
    mask_left = mask & (col_indices < x_mid)
    mask_right = mask & (col_indices >= x_mid)

    left_tot = np.sum(mask_left)
    left_pct = float(np.sum(mask_left & (depth_m > 0.0)) / left_tot * 100.0) if left_tot > 0 else 0.0

    right_tot = np.sum(mask_right)
    right_pct = float(np.sum(mask_right & (depth_m > 0.0)) / right_tot * 100.0) if right_tot > 0 else 0.0

    # Overall valid depth & median depth
    mask_valid = mask & (depth_m > 0.0)
    valid_count = int(np.sum(mask_valid))
    valid_pct = float(valid_count / mask_pixels * 100.0)

    med_depth = float(np.median(depth_m[mask_valid])) if valid_count > 0 else 0.0

    # Inside 0.40 - 3.00 m
    in_range_count = int(np.sum(mask & (depth_m >= 0.40) & (depth_m <= 3.00)))
    in_range_pct = float(in_range_count / mask_pixels * 100.0)

    # Dark clothing hypothesis: share of zero-depth pixels inside mask with V < 0.20
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    v_norm = hsv[:, :, 2].astype(np.float32) / 255.0

    zero_mask = mask & (depth_m == 0.0)
    zero_count = int(np.sum(zero_mask))
    dark_zero_mask = zero_mask & (v_norm < 0.20)
    dark_zero_count = int(np.sum(dark_zero_mask))

    dark_share_pct = float(dark_zero_count / zero_count * 100.0) if zero_count > 0 else 0.0

    return {
        "frame_idx": frame_idx,
        "mask_pixels": mask_pixels,
        "valid_pct": valid_pct,
        "median_depth_m": med_depth,
        "in_range_pct": in_range_pct,
        "head_valid_pct": head_pct,
        "chest_valid_pct": chest_pct,
        "waist_valid_pct": waist_pct,
        "thighs_valid_pct": thighs_pct,
        "shins_valid_pct": shins_pct,
        "left_valid_pct": left_pct,
        "right_valid_pct": right_pct,
        "dark_zero_depth_share_pct": dark_share_pct,
        "zero_depth_count": zero_count,
        "dark_zero_count": dark_zero_count,
        "has_mask": True,
    }


def plot_coverage_report(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate and save coverage_by_frame.png diagnostic visualization.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not available; skipping plot generation.")
        return

    frames = [r["frame_idx"] for r in results]
    if not frames:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Body Capture Coverage & Completeness Report", fontsize=14, fontweight="bold")

    # 1. 5 vertical bands valid depth %
    ax1.plot(frames, [r["head_valid_pct"] for r in results], "o-", label="Head (0-20%)", color="#1f77b4", lw=1.8)
    ax1.plot(frames, [r["chest_valid_pct"] for r in results], "s-", label="Chest (20-40%)", color="#2ca02c", lw=1.8)
    ax1.plot(frames, [r["waist_valid_pct"] for r in results], "^-", label="Waist (40-60%)", color="#ff7f0e", lw=1.8)
    ax1.plot(frames, [r["thighs_valid_pct"] for r in results], "d-", label="Thighs (60-80%)", color="#9467bd", lw=1.8)
    ax1.plot(frames, [r["shins_valid_pct"] for r in results], "v-", label="Shins (80-100%)", color="#8c564b", lw=1.8)
    ax1.plot(frames, [r["valid_pct"] for r in results], "k--", label="Overall Body", lw=2.2)
    ax1.axhline(70.0, color="green", linestyle=":", alpha=0.6, label="70% Good Gate")
    ax1.set_ylabel("Valid Depth (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", ncol=3, fontsize=9)
    ax1.set_title("Valid Depth Percentage by Vertical Band")

    # 2. Left vs Right Halves & Range (0.4 - 3.0 m)
    ax2.plot(frames, [r["left_valid_pct"] for r in results], "o-", label="Left Half", color="#3498db", lw=1.8)
    ax2.plot(frames, [r["right_valid_pct"] for r in results], "s-", label="Right Half", color="#e74c3c", lw=1.8)
    ax2.plot(frames, [r["in_range_pct"] for r in results], "g--", label="In Range [0.4, 3.0] m", lw=1.8)
    ax2.set_ylabel("Valid Depth (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", ncol=3, fontsize=9)
    ax2.set_title("Bilateral Coverage & Working Range Compliance")

    # 3. Dark clothing hypothesis: zero-depth V < 0.20 share & median distance
    ax3_twin = ax3.twinx()
    bars = ax3.bar(
        frames,
        [r["dark_zero_depth_share_pct"] for r in results],
        width=0.4,
        color="#34495e",
        alpha=0.75,
        label="Dark Clothing Dropout (V < 0.20)",
    )
    line = ax3_twin.plot(
        frames,
        [r["median_depth_m"] for r in results],
        "o-",
        color="#d35400",
        lw=2.0,
        label="Median Distance (m)",
    )
    ax3_twin.axhline(2.2, color="red", linestyle=":", alpha=0.7, label="2.2m Too-Far Limit")
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Dark Zero-Depth Share (%)", color="#34495e")
    ax3_twin.set_ylabel("Median Distance (m)", color="#d35400")
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Combined legend for ax3 and ax3_twin
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)
    ax3.set_title("Dark-Clothing Dropout Share (V < 0.20) vs Working Distance")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  [Report] Coverage plot saved to: {output_path}")


def generate_body_report(
    raw_dir: Path | str,
    out_dir: Optional[Path | str] = None,
    allow_aborted: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run body capture report on a directory containing recorded raw RGB-D frames.
    Validates trial status via load_raw_capture() before consuming recorded frame data.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")

    # Re-use load_raw_capture to validate trial status and load raw frames
    if load_raw_capture is not None:
        raw_frames = load_raw_capture(raw_path, target_name="body", allow_aborted=allow_aborted)
    else:
        raise RuntimeError("load_raw_capture is not available.")

    if not raw_frames:
        raise ValueError(f"No frames loaded from {raw_path}")

    # Read meta.json for optional burst capture metrics
    meta_path = raw_path / "meta.json"
    frames_meta_by_id: Dict[int, Dict[str, Any]] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_doc = json.load(f)
            for item in meta_doc.get("frames", []):
                fid = int(item.get("frame_id", item.get("order", 0)))
                frames_meta_by_id[fid] = item
        except Exception:
            pass

    # Check background reference
    bg_depth_m = getattr(raw_frames, "background_ref_depth", None)
    if bg_depth_m is None and len(raw_frames) > 0:
        bg_depth_m = getattr(raw_frames[0], "background_ref_depth", None)

    results: List[Dict[str, Any]] = []

    print(f"\nAnalyzing {len(raw_frames)} frames from: {raw_path}/ ...")

    for f in raw_frames:
        fid = f.frame_id
        color_bgr = f.color_bgr
        depth_m = f.depth_m

        res = analyze_body_frame(fid, color_bgr, depth_m, bg_depth_m)
        item = frames_meta_by_id.get(fid, {})
        res["burst_frames_used"] = item.get("burst_frames_used")
        res["burst_motion_mm"] = item.get("burst_motion_mm")
        res["burst_motion_flagged"] = item.get("burst_motion_flagged")
        results.append(res)

    # Print Table
    header = (
        f"{'Frame':>5} | {'MaskPts':>8} | {'Valid%':>6} | {'MedDist':>7} | "
        f"{'In-Range':>8} | {'Head%':>6} | {'Chest%':>6} | {'Waist%':>6} | "
        f"{'Thigh%':>6} | {'Shin%':>6} | {'Left%':>6} | {'Right%':>6} | {'DarkDrop%':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['frame_idx']:5d} | "
            f"{r['mask_pixels']:8d} | "
            f"{r['valid_pct']:5.1f}% | "
            f"{r['median_depth_m']:6.2f}m | "
            f"{r['in_range_pct']:7.1f}% | "
            f"{r['head_valid_pct']:5.1f}% | "
            f"{r['chest_valid_pct']:5.1f}% | "
            f"{r['waist_valid_pct']:5.1f}% | "
            f"{r['thighs_valid_pct']:5.1f}% | "
            f"{r['shins_valid_pct']:5.1f}% | "
            f"{r['left_valid_pct']:5.1f}% | "
            f"{r['right_valid_pct']:5.1f}% | "
            f"{r['dark_zero_depth_share_pct']:8.1f}%"
        )
    print(sep)

    # Summary row
    if results:
        m_valid = np.mean([r["valid_pct"] for r in results])
        m_dist = np.mean([r["median_depth_m"] for r in results])
        m_in_range = np.mean([r["in_range_pct"] for r in results])
        m_head = np.mean([r["head_valid_pct"] for r in results])
        m_chest = np.mean([r["chest_valid_pct"] for r in results])
        m_waist = np.mean([r["waist_valid_pct"] for r in results])
        m_thighs = np.mean([r["thighs_valid_pct"] for r in results])
        m_shins = np.mean([r["shins_valid_pct"] for r in results])
        m_left = np.mean([r["left_valid_pct"] for r in results])
        m_right = np.mean([r["right_valid_pct"] for r in results])
        m_dark = np.mean([r["dark_zero_depth_share_pct"] for r in results])

        print(
            f"{'MEAN':>5} | "
            f"{'':>8} | "
            f"{m_valid:5.1f}% | "
            f"{m_dist:6.2f}m | "
            f"{m_in_range:7.1f}% | "
            f"{m_head:5.1f}% | "
            f"{m_chest:5.1f}% | "
            f"{m_waist:5.1f}% | "
            f"{m_thighs:5.1f}% | "
            f"{m_shins:5.1f}% | "
            f"{m_left:5.1f}% | "
            f"{m_right:5.1f}% | "
            f"{m_dark:8.1f}%"
        )
        print(sep + "\n")

    has_burst = any(r.get("burst_frames_used") is not None for r in results)
    if has_burst:
        b_hdr = f"{'Frame':>5} | {'Burst Frames':>12} | {'Motion (mm)':>12} | {'Stillness Status':>20}"
        b_sep = "-" * len(b_hdr)
        print("\n" + "=" * len(b_hdr))
        print("  BURST CAPTURE & STILLNESS MOTION SUMMARY")
        print("=" * len(b_hdr))
        print(b_hdr)
        print(b_sep)
        for r in results:
            bf = r.get("burst_frames_used")
            bm = r.get("burst_motion_mm")
            bflag = r.get("burst_motion_flagged")
            bf_str = f"{bf} frames" if bf is not None else "--"
            bm_str = f"{bm:.1f} mm" if bm is not None else "--"
            status_str = "FLAGGED (>10 mm)" if bflag else ("STILL (<=10 mm)" if bm is not None else "--")
            print(f"{r['frame_idx']:5d} | {bf_str:>12} | {bm_str:>12} | {status_str:>20}")
        print(b_sep + "\n")

    # Output directory
    target_out = Path(out_dir) if out_dir else raw_path
    target_out.mkdir(parents=True, exist_ok=True)

    # Save coverage_by_frame.png
    png_path = target_out / "coverage_by_frame.png"
    plot_coverage_report(results, png_path)

    # Save CSV
    csv_path = target_out / "coverage_by_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_idx", "mask_pixels", "valid_pct", "median_depth_m", "in_range_pct",
            "head_valid_pct", "chest_valid_pct", "waist_valid_pct", "thighs_valid_pct", "shins_valid_pct",
            "left_valid_pct", "right_valid_pct", "dark_zero_depth_share_pct",
            "zero_depth_count", "dark_zero_count",
            "burst_frames_used", "burst_motion_mm", "burst_motion_flagged"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  [Report] CSV table saved to: {csv_path}")

    return results


# Documented alias
generate_body_report_from_raw = generate_body_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze body capture depth completeness, vertical bands, and dark clothing hypothesis."
    )
    parser.add_argument("raw_dir", help="Path to raw capture directory")
    parser.add_argument("--out-dir", default=None, help="Directory to save coverage_by_frame.png and CSV (default: <raw_dir>)")
    parser.add_argument("--allow-aborted", action="store_true", help="Allow analyzing aborted or incomplete captures")

    args = parser.parse_args()
    generate_body_report(args.raw_dir, args.out_dir, allow_aborted=args.allow_aborted)


if __name__ == "__main__":
    main()
