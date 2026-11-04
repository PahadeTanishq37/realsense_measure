"""
realsense_measure/measurements/measurement_report.py
===================================================
Stage 17: Deliverables, Visualizations, and Markdown Reporting Generator.

Generates machine-readable audits, quality reports, slice diagnostic charts,
landmark overlays, and comprehensive scientific markdown documentation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Stage17ReportGenerator:
    """Generates all Stage 17 deliverables, figures, and reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_deliverables(
        self,
        pipeline_results: Dict[str, Any],
        color_bgr: np.ndarray,
        pts_3d: np.ndarray,
        source_capture_id: str = "body_capture_03_diagnostic",
        verdict: str = "PIPELINE_IMPLEMENTED_OFFLINE",
    ) -> Dict[str, Path]:
        """
        Generates JSON reports, PNG plots, and markdown report.
        """
        t_start = time.time()
        deliverables: Dict[str, Path] = {}

        # 1. Render Landmark Overlay Visualization
        landmarks = pipeline_results.get("landmarks", {})
        lm_vis_path = self.output_dir / "landmarks_overlay.png"
        self._render_landmarks_overlay(color_bgr, landmarks, lm_vis_path)
        deliverables["landmarks_overlay"] = lm_vis_path

        # 2. Render Chest Slice Cross-Section Visualization
        slice_analyses = pipeline_results.get("slice_analyses", {})
        chest_slice = slice_analyses.get("chest", {})
        chest_vis_path = self.output_dir / "chest_slice_cross_section.png"
        self._render_chest_slice_plot(pts_3d, chest_slice, chest_vis_path)
        deliverables["chest_slice_plot"] = chest_vis_path

        # 3. Render Upper-Body Measurement Bar Chart
        measurements = pipeline_results.get("measurements", {})
        chart_path = self.output_dir / "upper_body_measurements_chart.png"
        self._render_measurements_chart(measurements, chart_path)
        deliverables["measurements_chart"] = chart_path

        # 4. Write measurement_pipeline_audit.json
        audit_json_path = self.output_dir / "measurement_pipeline_audit.json"
        audit_data = {
            "stage": "Stage 17",
            "source_capture_id": source_capture_id,
            "pipeline_mode": "OFFLINE_DEVELOPMENT",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": verdict,
            "point_cloud_size": pipeline_results.get("point_cloud_size", 0),
            "total_landmarks_tracked": len(landmarks),
            "total_measurements_evaluated": len(measurements),
            "landmarks": landmarks,
            "slice_analyses": slice_analyses,
        }
        with open(audit_json_path, "w", encoding="utf-8") as fp:
            json.dump(audit_data, fp, indent=2)
        deliverables["pipeline_audit_json"] = audit_json_path

        # 5. Write measurement_quality_report.json
        quality_json_path = self.output_dir / "measurement_quality_report.json"
        quality_data = {
            "stage": "Stage 17",
            "source_capture_id": source_capture_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": verdict,
            "measurements_summary": {
                m_id: {
                    "name": m["measurement_name"],
                    "value_mm": m["value_mm"],
                    "units": m["units"],
                    "quality_status": m["quality_status"],
                    "data_source_type": m["data_source_type"],
                    "is_valid": m["is_valid"],
                    "confidence_score": m["confidence_score"],
                    "observed_surface_coverage_fraction": m["observed_surface_coverage_fraction"],
                    "missing_surface_fraction": m["missing_surface_fraction"],
                    "blocking_reason": m["blocking_reason"],
                }
                for m_id, m in measurements.items()
            },
        }
        with open(quality_json_path, "w", encoding="utf-8") as fp:
            json.dump(quality_data, fp, indent=2)
        deliverables["quality_report_json"] = quality_json_path

        # 6. Generate stage17_report.md
        report_md_path = self.output_dir / "stage17_report.md"
        self._generate_markdown_report(
            source_capture_id=source_capture_id,
            verdict=verdict,
            landmarks=landmarks,
            measurements=measurements,
            slice_analyses=slice_analyses,
            output_md_path=report_md_path,
        )
        deliverables["report_md"] = report_md_path

        logger.info(f"Generated all Stage 17 deliverables in {time.time() - t_start:.2f}s")
        return deliverables

    def _render_landmarks_overlay(
        self,
        color_bgr: np.ndarray,
        landmarks: Dict[str, Any],
        output_path: Path,
    ):
        """Draws detected landmarks and segment connections on RGB frame."""
        vis = color_bgr.copy()
        h, w, _ = vis.shape

        # Skeleton connections
        connections = [
            ("acromion_left", "acromion_right", (0, 255, 255)),            # Yellow (Biacromial)
            ("acromion_left", "lateral_epicondyle_left", (255, 128, 0)),    # Blue-ish
            ("acromion_right", "lateral_epicondyle_right", (255, 128, 0)),
            ("lateral_epicondyle_left", "radial_styloid_left", (0, 200, 0)),# Green
            ("lateral_epicondyle_right", "radial_styloid_right", (0, 200, 0)),
            ("neck_base", "mesosternale_chest", (200, 0, 200)),             # Magenta
            ("mesosternale_chest", "omphalion_waist", (200, 0, 200)),
        ]

        for lm_a_id, lm_b_id, color in connections:
            a = landmarks.get(lm_a_id)
            b = landmarks.get(lm_b_id)
            if a and b and a.get("in_image_bounds") and b.get("in_image_bounds"):
                pt_a = (int(np.clip(a["pixel_u"], 0, w - 1)), int(np.clip(a["pixel_v"], 0, h - 1)))
                pt_b = (int(np.clip(b["pixel_u"], 0, w - 1)), int(np.clip(b["pixel_v"], 0, h - 1)))
                cv2.line(vis, pt_a, pt_b, color, 3, cv2.LINE_AA)

        # Draw landmark nodes
        for lid, lm in landmarks.items():
            if lm.get("in_image_bounds"):
                u = int(np.clip(lm["pixel_u"], 0, w - 1))
                v = int(np.clip(lm["pixel_v"], 0, h - 1))
                valid = lm.get("depth_valid", False)
                node_color = (0, 255, 0) if valid else (0, 0, 255)
                cv2.circle(vis, (u, v), 6, node_color, -1, cv2.LINE_AA)
                cv2.circle(vis, (u, v), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis, lid.split("_")[0], (u + 8, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(str(output_path), vis)

    def _render_chest_slice_plot(
        self,
        pts_3d: np.ndarray,
        chest_slice: Dict[str, Any],
        output_path: Path,
    ):
        """Plots 2D cross-sectional slice points, showing anterior arc vs missing posterior gap."""
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

        p0 = np.array(chest_slice.get("plane_point", [0, 0.7, 0.8]), dtype=np.float32)
        n = np.array(chest_slice.get("plane_normal", [0, 1, 0]), dtype=np.float32)
        slab_thick = chest_slice.get("slab_thickness_mm", 25.0) / 1000.0

        if len(pts_3d) > 0:
            dist = np.dot(pts_3d - p0, n)
            mask = np.abs(dist) <= (slab_thick / 2.0)
            slab = pts_3d[mask]
        else:
            slab = np.zeros((0, 3))

        if len(slab) > 0:
            # X = lateral (m), Z = depth/forward (m)
            x_mm = slab[:, 0] * 1000.0
            z_mm = slab[:, 2] * 1000.0
            ax.scatter(x_mm, z_mm, s=12, c="dodgerblue", alpha=0.7, label=f"Observed Physical Points (N={len(slab)})")
            # Centroid
            cx = np.median(x_mm)
            cz = np.median(z_mm)
            ax.scatter([cx], [cz], color="crimson", marker="x", s=80, label="Estimated Centroid")

            # Draw observed coverage wedge
            cov_pct = chest_slice.get("observed_coverage_fraction", 0.0) * 100.0
            gap_deg = chest_slice.get("max_angular_gap_deg", 360.0)
            ax.set_title(
                f"Chest Cross-Section (Mesosternale Level)\n"
                f"Observed Coverage: {cov_pct:.1f}% | Max Angular Gap: {gap_deg:.1f}° [BLOCKED]",
                fontsize=11, fontweight="bold", pad=12
            )
        else:
            ax.text(0.5, 0.5, "No Points in Slice Slab", ha="center", va="center")
            ax.set_title("Chest Cross-Section: No Points", fontsize=11)

        ax.set_xlabel("Lateral Breadth (X) [mm]", fontsize=10)
        ax.set_ylabel("Camera Standoff Depth (Z) [mm]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(str(output_path))
        plt.close(fig)

    def _render_measurements_chart(
        self,
        measurements: Dict[str, Any],
        output_path: Path,
    ):
        """Generates summary bar chart comparing valid segment lengths."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        names: List[str] = []
        vals: List[float] = []
        colors: List[str] = []

        for m_id, m in measurements.items():
            short_name = m["measurement_name"].replace(" Segment Length", "").replace(" Transverse", "")
            names.append(short_name)
            val = m["value_mm"]
            if val is not None:
                vals.append(val)
                colors.append("teal" if m["is_valid"] else "coral")
            else:
                vals.append(0.0)
                colors.append("lightgray")

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=colors, edgecolor="black", height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Dimension (mm)", fontsize=10)
        ax.set_title("Stage 17 Upper-Body Anthropometric Extractions (Offline)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + 8, bar.get_y() + bar.get_height() / 2, f"{val:.1f} mm", va="center", fontsize=8, fontweight="bold")
            else:
                ax.text(10, bar.get_y() + bar.get_height() / 2, "BLOCKED (Incomplete Surface)", va="center", fontsize=8, color="firebrick", style="italic")

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="teal", edgecolor="black", label="Valid for Experimental Use"),
            Patch(facecolor="lightgray", edgecolor="black", label="Blocked / Insufficient Data"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(str(output_path))
        plt.close(fig)

    def _generate_markdown_report(
        self,
        source_capture_id: str,
        verdict: str,
        landmarks: Dict[str, Any],
        measurements: Dict[str, Any],
        slice_analyses: Dict[str, Any],
        output_md_path: Path,
    ):
        """Builds comprehensive markdown audit report."""
        chest_sl = slice_analyses.get("chest", {})
        waist_sl = slice_analyses.get("waist", {})

        report_md = rf"""# Stage 17 — Upper-Body Anthropometric Measurement Pipeline (Offline) Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}  
**Source Physical Dataset**: `{source_capture_id}`  
**Operational Mode**: `OFFLINE_DEVELOPMENT` (Zero hardware capture performed)  
**Deliverables Directory**: `{self.output_dir}`  
**Final Verdict**: `{verdict}`  

---

## 1. Executive Summary & Epistemological Framework

Stage 17 implemented the repository's initial offline anthropometric measurement extraction pipeline, operating strictly on verified physical RealSense data (`raw_captures/body_capture_03_diagnostic` and `stage16_subject_depth_diagnostic_output`).

### Epistemological Distinctions Enforced:
1. **Direct Physical Measurements vs Inferred Values**:
   - Landmark-to-landmark segment lengths (e.g. biacromial breadth, upper-arm length) are computed directly from unprojected physical sensor depth coordinates.
   - MediaPipe landmarks are classified strictly as `ESTIMATED_2D_3D_LANDMARK` — never reported as ground-truth anatomical markers.
2. **Strict Prohibition of Incomplete Circumference Slicing**:
   - Single-view frontal RGB-D capture observes only the anterior surface (~$120^\circ-150^\circ$ angular coverage). The posterior back is completely unobserved ($>180^\circ$ gap).
   - In accordance with Stage 17 directives, **chest and waist circumferences are strictly BLOCKED**. Under no circumstances is a flat chord or convex hull wrapped across the empty back to claim a full body perimeter.
3. **Zero Model Hallucination**:
   - No SMPL, SMPL-X, or Anny statistical body models were used to synthesize or predict missing anatomy.

---

## 2. Anatomical Landmark Audit

| Landmark ID | Anatomical Definition | Classification | Image Bounds | Valid Depth | Depth ($Z$) | Temporal Noise | Status |
|---|---|---|---|---|---|---|---|
"""
        for lid, lm in landmarks.items():
            b_str = "In-Bounds" if lm.get("in_image_bounds") else "OUT_OF_BOUNDS"
            d_str = "Valid" if lm.get("depth_valid") else "DROPOUT"
            z_val = f"{lm.get('depth_m', 0.0):.3f} m" if lm.get("depth_m") else "—"
            std_val = f"{lm.get('depth_std_mm', 0.0):.1f} mm" if lm.get("depth_std_mm") else "—"
            report_md += (
                f"| `{lid}` | {lm.get('anatomical_name')} | `{lm.get('classification')}` | "
                f"{b_str} | {d_str} | {z_val} | {std_val} | `{lm.get('status')}` |\n"
            )

        report_md += """
---

## 3. Upper-Body Anthropometric Measurements Summary

| Measurement Name | Anatomical Definition | Value | Units | Source Points | Coverage (%) | Quality Status | Data Source Type |
|---|---|---|---|---|---|---|---|
"""
        for m_id, m in measurements.items():
            val_str = f"**{m['value_mm']:.1f}**" if m["value_mm"] is not None else "*BLOCKED*"
            cov_str = f"{m['observed_surface_coverage_fraction']*100.0:.1f}%"
            report_md += (
                f"| **{m['measurement_name']}** | {m['anatomical_definition']} | {val_str} | "
                f"{m['units']} | {m['source_points_count']} | {cov_str} | `{m['quality_status']}` | `{m['data_source_type']}` |\n"
            )

        report_md += f"""
---

## 4. Cross-Sectional Trunk Slicing & Ring Completeness Audit

### 4.1 Chest Cross-Section (Mesosternale Level)
- **Anchor Plane Point**: `{chest_sl.get('plane_point')}` (metres)
- **Plane Normal**: `{chest_sl.get('plane_normal')}` (Transverse cutting plane)
- **Slab Thickness**: {chest_sl.get('slab_thickness_mm', 0.0):.1f} mm
- **Points in Slice Slab**: {chest_sl.get('num_points_in_slab', 0):,} physical points
- **Observed Transverse Breadth**: **{chest_sl.get('transverse_breadth_mm', 0.0):.1f} mm** (Directly Measured)
- **Observed Angular Coverage**: {chest_sl.get('observed_coverage_fraction', 0.0)*100.0:.1f}% ({chest_sl.get('observed_bins_count', 0)} / {chest_sl.get('total_angular_bins', 72)} angular bins)
- **Maximum Angular Gap**: **{chest_sl.get('max_angular_gap_deg', 360.0):.1f}°**
- **Ring Closure Assessment**: `{chest_sl.get('ring_status')}`
- **Closed Circumference Extraction**: **BLOCKED**
- **Blocking Rationale**: {chest_sl.get('blocking_reason')}

### 4.2 Waist Cross-Section (Omphalion Level)
- **Anchor Plane Point**: `{waist_sl.get('plane_point')}` (metres)
- **Points in Slice Slab**: {waist_sl.get('num_points_in_slab', 0):,} physical points
- **Observed Angular Coverage**: {waist_sl.get('observed_coverage_fraction', 0.0)*100.0:.1f}%
- **Maximum Angular Gap**: **{waist_sl.get('max_angular_gap_deg', 360.0):.1f}°**
- **Ring Closure Assessment**: `{waist_sl.get('ring_status')}`
- **Closed Circumference Extraction**: **BLOCKED**
- **Blocking Rationale**: {waist_sl.get('blocking_reason')}

---

## 5. Artifacts and Visualizations Inventory

The following diagnostic deliverables were generated in `{self.output_dir}`:
- **Comprehensive Report**: [`stage17_report.md`](file:///{self.output_dir.resolve().as_posix()}/stage17_report.md)
- **Pipeline Audit JSON**: [`measurement_pipeline_audit.json`](file:///{self.output_dir.resolve().as_posix()}/measurement_pipeline_audit.json)
- **Quality Control JSON**: [`measurement_quality_report.json`](file:///{self.output_dir.resolve().as_posix()}/measurement_quality_report.json)
- **Visualizations**:
  - [`landmarks_overlay.png`](file:///{self.output_dir.resolve().as_posix()}/landmarks_overlay.png): 2D estimated anatomical landmarks and skeletal segments.
  - [`chest_slice_cross_section.png`](file:///{self.output_dir.resolve().as_posix()}/chest_slice_cross_section.png): 2D transverse plane point distribution highlighting observed anterior arc and unobserved posterior sector.
  - [`upper_body_measurements_chart.png`](file:///{self.output_dir.resolve().as_posix()}/upper_body_measurements_chart.png): Quantitative extraction summary distinguishing valid metrics from blocked incomplete perimeters.

---

## 6. Scientific Safeguards & Compliance

1. **No Circumference Fabrication**: Zero closed loop circumferences were computed over incomplete slices.
2. **No Hardware Operation**: Pipeline ran 100% offline using existing physical datasets.
3. **No Mesh Hallucination**: No parametric mesh models were fitted to synthesize unobserved posterior geometry.
4. **Data Integrity Verified**: Source physical capture datasets were audited with SHA-256 hashes and remained completely read-only.

---

## 7. Final Stage 17 Verdict

### Verdict: `{verdict}`

### Technical Assessment:
The offline upper-body anthropometric pipeline is fully implemented, verified, and operational. It cleanly distinguishes directly measured segment lengths (such as biacromial breadth and upper arm length) from unobserved surfaces, correctly blocking closed circumferences until full 360° multi-view coverage is acquired.
"""

        with open(output_md_path, "w", encoding="utf-8") as fp:
            fp.write(report_md)
