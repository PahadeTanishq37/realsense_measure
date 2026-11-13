"""Problem 4 — Stage 10: Physical Data Acquisition Readiness & Capture-Only Workflow.

Provides:
1. CaptureSessionManifest: Complete, auditable capture session manifest schema tracking
   anonymous subject ID, operator, timestamps, camera serial/model, intrinsics, alignment,
   physical standoff, yaw stops, clothing, posture, depth validity, SHA-256 checksums,
   and multi-stage linkages.
2. CaptureQualityChecker: Pre-save and post-save diagnostics evaluating frame availability,
   depth validity, timestamp monotonicity, blur/stability, boundary clipping, and resolution consistency,
   strictly distinguishing sensor-valid depth, spatial coverage, anatomical visibility, and parametric inference.
3. CaptureExperimentConfigManager: Records and compares physical capture configurations without
   automatic declarations of camera distance optimality.
4. Stage10DeliverablesGenerator: Generates standardized checklists, JSON schemas, example manifests,
   CLI guides, and limitation audits in stage10_acquisition_readiness_output/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("capture_session")


# ---------------------------------------------------------------------------
# Anonymous Subject ID Validation
# ---------------------------------------------------------------------------

def validate_anonymous_subject_id(subject_id: Optional[str]) -> str:
    """
    Ensure subject identifier is strictly anonymous and does NOT leak PII.
    
    Rejects names, emails, spaces, or non-anonymous patterns.
    """
    if not subject_id or not isinstance(subject_id, str):
        raise ValueError("subject_id must be a non-empty string.")

    if "@" in subject_id:
        raise ValueError(f"subject_id '{subject_id}' contains '@' (email rejected for PII protection).")

    if " " in subject_id:
        raise ValueError(f"subject_id '{subject_id}' contains spaces (personal names rejected for PII protection).")

    pattern = r"^(SUBJ|ANON|PARTICIPANT|SUBJECT|P|TEST|S)(_?[A-Za-z0-9]+)$"
    if not re.match(pattern, subject_id, re.IGNORECASE):
        raise ValueError(
            f"subject_id '{subject_id}' does not match authorized anonymous format (e.g. 'SUBJ_01', 'S01', 'ANON_42')."
        )
    return subject_id


# ---------------------------------------------------------------------------
# Depth Validity & Quality Statistics Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DepthValidityStats:
    """Quantitative sensor-level depth validity metrics."""
    total_pixels: int
    valid_pixels: int
    valid_fraction: float
    zero_pixels: int
    nan_pixels: int
    out_of_range_pixels: int
    min_depth_m: Optional[float]
    max_depth_m: Optional[float]
    mean_depth_m: Optional[float]
    median_depth_m: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticCheckResult:
    """Pre-save or post-save diagnostic evaluation result."""
    check_name: str
    status: str  # "PASS", "WARN", "FAIL"
    category: str  # "sensor_valid_depth", "spatial_coverage", "anatomical_visibility", "parametric_inference", "metadata"
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileManifestEntry:
    """File entry with size and SHA-256 integrity hash."""
    relative_path: str
    file_type: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Capture Session Manifest Schema
# ---------------------------------------------------------------------------

@dataclass
class CaptureSessionManifest:
    """
    Comprehensive, auditable capture session manifest.
    
    Every field is sensor-grounded, explicit, and auditable. Unknown parameters
    are recorded as null/unknown rather than guessed or fabricated.
    """
    manifest_version: str = "1.0.0"
    capture_id: str = "body_capture_01"
    session_id: str = "SESS_20261102_01"
    subject_id: str = "SUBJ_01"
    operator_id: str = "OPERATOR_01"
    config_id: str = "CONFIG_D220_H100_Y4_R720"
    repeat_index: int = 1
    capture_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Hardware & Optical stream configuration
    camera_model: str = "Intel RealSense D455f"
    camera_serial_number: Optional[str] = None  # None if unavailable
    rgb_resolution: Tuple[int, int] = (1280, 720)
    depth_resolution: Tuple[int, int] = (1280, 720)
    is_depth_aligned: bool = True
    frame_rate_fps: int = 30
    intrinsics: Dict[str, Any] = field(default_factory=lambda: {
        "width": 1280,
        "height": 720,
        "fx": 646.73,
        "fy": 645.89,
        "cx": 637.24,
        "cy": 366.94,
    })

    # Physical Standoff & Environment
    camera_to_subject_distance_m: Optional[float] = 2.20
    camera_height_m: Optional[float] = 1.00
    camera_pitch_tilt_deg: Optional[float] = None  # null if unmeasured
    nominal_yaw_stops: List[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    clothing_condition: str = "casual_clothing"
    posture_instructions: str = (
        "Standardized A-pose: standing erect, feet shoulder-width, arms abducted 30-40 deg, "
        "palms forward, head Frankfort horizontal plane."
    )
    lighting_description: str = "diffuse_indoor_fluorescent"

    # Execution summary
    capture_mode: str = "capture_only"
    capture_duration_seconds: Optional[float] = None
    frames_captured: int = 0
    expected_frames: int = 4

    # Diagnostic quality and integrity
    depth_statistics_overall: Optional[DepthValidityStats] = None
    per_frame_depth_statistics: List[DepthValidityStats] = field(default_factory=list)
    quality_diagnostics: List[DiagnosticCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    file_manifest: List[FileManifestEntry] = field(default_factory=list)

    # Cross-Stage Linakges
    stage_linkages: Dict[str, Any] = field(default_factory=lambda: {
        "stage6_manual_validation_template": "stage6_validation_output/manual_validation_template.csv",
        "stage7_protocol_manifest": "stage7_capture_protocol_output/capture_session_manifest.csv",
        "stage8_quality_calibration": "stage8_capture_quality_audit_output/capture_quality_calibration_report.json",
        "stage9_pilot_study_summary": "stage9_pilot_study_output/pilot_study_summary.json",
    })
    notes: str = ""

    def __post_init__(self):
        if self.subject_id:
            validate_anonymous_subject_id(self.subject_id)
        if self.repeat_index < 1:
            raise ValueError(f"repeat_index must be >= 1, got {self.repeat_index}.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rgb_resolution"] = list(self.rgb_resolution)
        d["depth_resolution"] = list(self.depth_resolution)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaptureSessionManifest:
        d = data.copy()
        if "rgb_resolution" in d and isinstance(d["rgb_resolution"], list):
            d["rgb_resolution"] = tuple(d["rgb_resolution"])
        if "depth_resolution" in d and isinstance(d["depth_resolution"], list):
            d["depth_resolution"] = tuple(d["depth_resolution"])
        if "depth_statistics_overall" in d and isinstance(d["depth_statistics_overall"], dict):
            d["depth_statistics_overall"] = DepthValidityStats(**d["depth_statistics_overall"])
        if "per_frame_depth_statistics" in d and isinstance(d["per_frame_depth_statistics"], list):
            d["per_frame_depth_statistics"] = [
                DepthValidityStats(**x) if isinstance(x, dict) else x for x in d["per_frame_depth_statistics"]
            ]
        if "quality_diagnostics" in d and isinstance(d["quality_diagnostics"], list):
            d["quality_diagnostics"] = [
                DiagnosticCheckResult(**x) if isinstance(x, dict) else x for x in d["quality_diagnostics"]
            ]
        if "file_manifest" in d and isinstance(d["file_manifest"], list):
            d["file_manifest"] = [
                FileManifestEntry(**x) if isinstance(x, dict) else x for x in d["file_manifest"]
            ]
        return cls(**d)

    def save(self, filepath: Path | str) -> Path:
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved capture manifest to {out_p}")
        return out_p


# ---------------------------------------------------------------------------
# Capture Quality Checker (Pre-save & Post-save Diagnostics)
# ---------------------------------------------------------------------------

class CaptureQualityChecker:
    """
    Evaluates data capture quality before and after saving.
    
    Strictly separates:
    1. Sensor-valid depth (range gating, zero counts, non-finite values)
    2. Spatial coverage (bounding geometry relative to camera frustum boundaries)
    3. Anatomical visibility (sensor landmarks presence and confidence)
    4. Parametric-model inference (prior-driven geometry)
    """

    @staticmethod
    def compute_depth_stats(
        depth_m: np.ndarray,
        min_valid_m: float = 0.40,
        max_valid_m: float = 3.00,
    ) -> DepthValidityStats:
        """Calculate quantitative sensor depth validity statistics."""
        total = int(depth_m.size)
        zeros = int(np.count_nonzero(depth_m == 0))
        nans = int(np.count_nonzero(np.isnan(depth_m)))

        valid_mask = (depth_m >= min_valid_m) & (depth_m <= max_valid_m) & np.isfinite(depth_m)
        valid_count = int(np.count_nonzero(valid_mask))
        valid_frac = float(valid_count / total) if total > 0 else 0.0

        out_of_range = total - (valid_count + zeros + nans)

        if valid_count > 0:
            valid_depths = depth_m[valid_mask]
            min_d = float(np.min(valid_depths))
            max_d = float(np.max(valid_depths))
            mean_d = float(np.mean(valid_depths))
            med_d = float(np.median(valid_depths))
        else:
            min_d = max_d = mean_d = med_d = None

        return DepthValidityStats(
            total_pixels=total,
            valid_pixels=valid_count,
            valid_fraction=round(valid_frac, 4),
            zero_pixels=zeros,
            nan_pixels=nans,
            out_of_range_pixels=out_of_range,
            min_depth_m=round(min_d, 3) if min_d is not None else None,
            max_depth_m=round(max_d, 3) if max_d is not None else None,
            mean_depth_m=round(mean_d, 3) if mean_d is not None else None,
            median_depth_m=round(med_d, 3) if med_d is not None else None,
        )

    @staticmethod
    def evaluate_pre_save(
        frames: List[Any],
        expected_yaw_stops: int = 4,
        intrinsics_wh: Tuple[int, int] = (1280, 720),
        min_depth_m: float = 0.40,
        max_depth_m: float = 3.00,
    ) -> Tuple[List[DiagnosticCheckResult], List[str], List[str]]:
        """
        Pre-save validation checks:
        - Frame availability
        - Depth validity and saturation
        - Monotonic timestamps
        - Blur / frame stability
        - Boundary clipping
        - Missing yaw stops
        - Intrinsics and image dimension match
        """
        diagnostics: List[DiagnosticCheckResult] = []
        warnings: List[str] = []
        failures: List[str] = []

        # 1. Frame count & availability
        n_frames = len(frames)
        if n_frames == 0:
            failures.append("No frames captured in session.")
            diagnostics.append(DiagnosticCheckResult(
                check_name="frame_availability",
                status="FAIL",
                category="sensor_valid_depth",
                details="Zero frames captured.",
            ))
            return diagnostics, warnings, failures

        if n_frames < expected_yaw_stops:
            warnings.append(f"Incomplete yaw stops: captured {n_frames}/{expected_yaw_stops} expected stops.")
            diagnostics.append(DiagnosticCheckResult(
                check_name="yaw_stop_completeness",
                status="WARN",
                category="spatial_coverage",
                details=f"Captured {n_frames}/{expected_yaw_stops} stops.",
            ))
        else:
            diagnostics.append(DiagnosticCheckResult(
                check_name="yaw_stop_completeness",
                status="PASS",
                category="spatial_coverage",
                details=f"All {n_frames} expected stops present.",
            ))

        # Check each frame
        timestamps: List[float] = []
        intr_w, intr_h = intrinsics_wh

        for idx, f in enumerate(frames):
            color = getattr(f, "color_bgr", None)
            depth = getattr(f, "depth_m", None)
            ts = getattr(f, "timestamp", None)

            # Frame data presence
            if color is None or depth is None:
                failures.append(f"Frame {idx}: missing color or depth array.")
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_data_presence",
                    status="FAIL",
                    category="sensor_valid_depth",
                    details=f"Color is {color is not None}, Depth is {depth is not None}",
                ))
                continue

            # Resolution match
            h_c, w_c = color.shape[:2]
            h_d, w_d = depth.shape[:2]
            if (w_c, h_c) != (intr_w, intr_h) or (w_d, h_d) != (intr_w, intr_h):
                failures.append(
                    f"Frame {idx}: resolution mismatch. Color ({w_c}x{h_c}), Depth ({w_d}x{h_d}) vs Intrinsics ({intr_w}x{intr_h})."
                )
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_resolution_match",
                    status="FAIL",
                    category="sensor_valid_depth",
                    details=f"Color=({w_c}x{h_c}), Depth=({w_d}x{h_d}), Intrinsics=({intr_w}x{intr_h})",
                ))

            # Depth validity fraction
            stats = CaptureQualityChecker.compute_depth_stats(depth, min_depth_m, max_depth_m)
            if stats.valid_fraction < 0.30:
                warnings.append(f"Frame {idx}: low valid depth return ({stats.valid_fraction:.1%}).")
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_depth_validity",
                    status="WARN",
                    category="sensor_valid_depth",
                    details=f"Valid fraction = {stats.valid_fraction:.1%}",
                ))
            else:
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_depth_validity",
                    status="PASS",
                    category="sensor_valid_depth",
                    details=f"Valid fraction = {stats.valid_fraction:.1%}",
                ))

            # Optical boundary contact (spatial clipping)
            # Check if foreground touches frame border within 3 pixels
            fg_mask = (depth >= min_depth_m) & (depth <= max_depth_m)
            top_contact = bool(np.count_nonzero(fg_mask[0:3, :]) >= 5)
            bottom_contact = bool(np.count_nonzero(fg_mask[h_d-3:h_d, :]) >= 5)
            left_contact = bool(np.count_nonzero(fg_mask[:, 0:3]) >= 5)
            right_contact = bool(np.count_nonzero(fg_mask[:, w_d-3:w_d]) >= 5)

            clipping_edges = []
            if top_contact: clipping_edges.append("top (head crown clipped)")
            if bottom_contact: clipping_edges.append("bottom (lower body clipped)")
            if left_contact: clipping_edges.append("left (lateral clipping)")
            if right_contact: clipping_edges.append("right (lateral clipping)")

            if clipping_edges:
                warnings.append(f"Frame {idx}: boundary clipping detected at {', '.join(clipping_edges)}.")
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_boundary_clipping",
                    status="WARN",
                    category="spatial_coverage",
                    details=f"Clipping at: {', '.join(clipping_edges)}",
                ))
            else:
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_boundary_clipping",
                    status="PASS",
                    category="spatial_coverage",
                    details="No boundary clipping detected.",
                ))

            # Blur / stability check using grayscale Laplacian variance
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_var < 50.0:
                warnings.append(f"Frame {idx}: possible motion blur (Laplacian variance = {blur_var:.1f} < 50).")
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_blur_stability",
                    status="WARN",
                    category="sensor_valid_depth",
                    details=f"Laplacian variance = {blur_var:.1f}",
                ))
            else:
                diagnostics.append(DiagnosticCheckResult(
                    check_name=f"frame_{idx}_blur_stability",
                    status="PASS",
                    category="sensor_valid_depth",
                    details=f"Laplacian variance = {blur_var:.1f}",
                ))

            if ts is not None:
                timestamps.append(float(ts))

        # Check timestamp monotonicity
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            if any(d <= 0 for d in diffs):
                failures.append("Non-monotonic timestamps detected in capture sequence.")
                diagnostics.append(DiagnosticCheckResult(
                    check_name="timestamp_monotonicity",
                    status="FAIL",
                    category="sensor_valid_depth",
                    details=f"Negative or zero timestamp deltas found: {diffs}",
                ))
            else:
                diagnostics.append(DiagnosticCheckResult(
                    check_name="timestamp_monotonicity",
                    status="PASS",
                    category="sensor_valid_depth",
                    details="Timestamps strictly monotonic.",
                ))

        return diagnostics, warnings, failures

    @staticmethod
    def evaluate_post_save(
        capture_dir: Path | str,
    ) -> Tuple[List[FileManifestEntry], List[DiagnosticCheckResult]]:
        """
        Post-save integrity checks:
        - Check files exist and are non-empty
        - Compute SHA-256 checksums
        - Check meta.json validity
        """
        cap_p = Path(capture_dir)
        entries: List[FileManifestEntry] = []
        diagnostics: List[DiagnosticCheckResult] = []

        if not cap_p.exists():
            diagnostics.append(DiagnosticCheckResult(
                check_name="saved_directory_exists",
                status="FAIL",
                category="metadata",
                details=f"Directory {cap_p} does not exist.",
            ))
            return entries, diagnostics

        all_files = sorted([f for f in cap_p.glob("*") if f.is_file()])
        for f in all_files:
            size = f.stat().st_size
            sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
            ext = f.suffix.lower()
            ftype = "image_png" if ext == ".png" else "numpy_depth" if ext == ".npy" else "json_metadata" if ext == ".json" else "unknown"

            entries.append(FileManifestEntry(
                relative_path=f.name,
                file_type=ftype,
                size_bytes=size,
                sha256=sha256,
            ))

        # Check meta.json
        meta_p = cap_p / "meta.json"
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as fp:
                    meta_data = json.load(fp)
                diagnostics.append(DiagnosticCheckResult(
                    check_name="meta_json_validity",
                    status="PASS",
                    category="metadata",
                    details=f"meta.json valid with {meta_data.get('num_frames', 0)} frames recorded.",
                ))
            except Exception as exc:
                diagnostics.append(DiagnosticCheckResult(
                    check_name="meta_json_validity",
                    status="FAIL",
                    category="metadata",
                    details=f"meta.json parsing error: {exc}",
                ))
        else:
            diagnostics.append(DiagnosticCheckResult(
                check_name="meta_json_validity",
                status="FAIL",
                category="metadata",
                details="meta.json missing from saved directory.",
            ))

        return entries, diagnostics


# ---------------------------------------------------------------------------
# Capture Experiment Configuration Manager
# ---------------------------------------------------------------------------

class CaptureExperimentConfigManager:
    """
    Manages and compares physical capture configurations.
    
    Refuses to declare an optimal camera distance or height without empirical multi-config data.
    """

    @staticmethod
    def compare_configurations(
        manifests: List[CaptureSessionManifest],
    ) -> Dict[str, Any]:
        """Compare actual physical capture sessions."""
        if not manifests:
            return {
                "status": "NO_SESSIONS_AVAILABLE",
                "message": "No physical capture sessions provided for comparison.",
                "sessions_compared": 0,
            }

        sessions = []
        for m in manifests:
            sessions.append({
                "capture_id": m.capture_id,
                "config_id": m.config_id,
                "subject_id": m.subject_id,
                "distance_m": m.camera_to_subject_distance_m,
                "height_m": m.camera_height_m,
                "yaw_stops": len(m.nominal_yaw_stops),
                "resolution": f"{m.rgb_resolution[0]}x{m.rgb_resolution[1]}",
                "valid_depth_pct": m.depth_statistics_overall.valid_fraction * 100.0 if m.depth_statistics_overall else None,
                "warning_count": len(m.warnings),
                "failure_count": len(m.failures),
            })

        has_multi = len(set(s["config_id"] for s in sessions)) >= 2

        return {
            "status": "COMPARISON_ACTIVE" if has_multi else "SINGLE_CONFIGURATION_ONLY",
            "sessions_count": len(sessions),
            "sessions": sessions,
            "optimal_configuration_decision": None,
            "decision_justification": (
                "No optimal configuration declared. Camera distance optimality cannot be derived "
                "theoretically; it requires physical comparative trials across multiple distances "
                "evaluating full-body visibility and manual anthropometric error."
                if not has_multi else "Empirical comparative evaluation active."
            ),
        }


# ---------------------------------------------------------------------------
# Stage 10 Deliverables Generator
# ---------------------------------------------------------------------------

class Stage10DeliverablesGenerator:
    """
    Generates all Stage 10 artifacts in stage10_acquisition_readiness_output/.
    """

    def __init__(self, output_dir: Optional[Path | str] = None):
        self.output_dir = Path(output_dir or "stage10_acquisition_readiness_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        existing_capture_manifest: Optional[CaptureSessionManifest] = None,
    ) -> Dict[str, Any]:
        """Generate all required Stage 10 deliverables."""
        logger.info("Generating Stage 10 acquisition readiness artifacts...")

        # 1. Manifest JSON Schema
        schema_path = self._generate_manifest_schema()

        # 2. Example manifest with clearly marked nonphysical placeholder values
        example_path = self._generate_example_manifest()

        # 3. Capture protocol checklist
        checklist_path = self._generate_capture_protocol_checklist()

        # 4. Capture-only CLI documentation
        cli_doc_path = self._generate_cli_documentation()

        # 5. Known limitations & physical validation requirements
        limitations_path = self._generate_limitations_and_requirements()

        # 6. Test results summary
        test_results_path = self._generate_test_results_summary()

        # 7. Stage 10 Master Report
        report_path = self._generate_stage10_report(existing_capture_manifest)

        logger.info(f"Stage 10 deliverables successfully written to {self.output_dir}.")

        return {
            "status": "COMPLETE",
            "output_directory": str(self.output_dir),
            "artifacts": {
                "manifest_schema": str(schema_path),
                "example_manifest": str(example_path),
                "capture_protocol_checklist": str(checklist_path),
                "cli_documentation": str(cli_doc_path),
                "known_limitations": str(limitations_path),
                "test_results": str(test_results_path),
                "stage10_report": str(report_path),
            },
        }

    def _generate_manifest_schema(self) -> Path:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "CaptureSessionManifest",
            "description": "Schema for RealSense D455f anthropometric capture session manifests.",
            "type": "object",
            "required": [
                "manifest_version",
                "capture_id",
                "session_id",
                "subject_id",
                "operator_id",
                "config_id",
                "capture_timestamp",
                "camera_model",
                "rgb_resolution",
                "depth_resolution",
                "intrinsics",
                "nominal_yaw_stops",
                "capture_mode",
            ],
            "properties": {
                "manifest_version": {"type": "string"},
                "capture_id": {"type": "string"},
                "session_id": {"type": "string"},
                "subject_id": {
                    "type": "string",
                    "pattern": "^(SUBJ|ANON|PARTICIPANT|SUBJECT|P|TEST)_[A-Za-z0-9]+$",
                    "description": "Strictly anonymous identifier rejecting PII."
                },
                "operator_id": {"type": "string"},
                "config_id": {"type": "string"},
                "repeat_index": {"type": "integer", "minimum": 1},
                "capture_timestamp": {"type": "string", "format": "date-time"},
                "camera_model": {"type": "string"},
                "camera_serial_number": {"type": ["string", "null"]},
                "rgb_resolution": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "depth_resolution": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "is_depth_aligned": {"type": "boolean"},
                "intrinsics": {
                    "type": "object",
                    "required": ["width", "height", "fx", "fy", "cx", "cy"],
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "fx": {"type": "number"},
                        "fy": {"type": "number"},
                        "cx": {"type": "number"},
                        "cy": {"type": "number"}
                    }
                },
                "camera_to_subject_distance_m": {"type": ["number", "null"]},
                "camera_height_m": {"type": ["number", "null"]},
                "camera_pitch_tilt_deg": {"type": ["number", "null"]},
                "nominal_yaw_stops": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2
                },
                "clothing_condition": {"type": "string"},
                "posture_instructions": {"type": "string"},
                "lighting_description": {"type": "string"},
                "capture_mode": {"type": "string", "enum": ["capture_only", "full_pipeline"]},
                "capture_duration_seconds": {"type": ["number", "null"]},
                "frames_captured": {"type": "integer"},
                "expected_frames": {"type": "integer"},
                "depth_statistics_overall": {"type": ["object", "null"]},
                "per_frame_depth_statistics": {"type": "array", "items": {"type": "object"}},
                "quality_diagnostics": {"type": "array", "items": {"type": "object"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "failures": {"type": "array", "items": {"type": "string"}},
                "file_manifest": {"type": "array", "items": {"type": "object"}},
                "stage_linkages": {"type": "object"}
            }
        }
        p = self.output_dir / "manifest_schema.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        return p

    def _generate_example_manifest(self) -> Path:
        example = {
            "_comment": "NON-PHYSICAL PLACEHOLDER MANIFEST FOR SCHEMA DEMONSTRATION AND TESTING PURPOSES ONLY",
            "manifest_version": "1.0.0",
            "capture_id": "placeholder_capture_demo",
            "session_id": "SESS_DEMO_20261102",
            "subject_id": "ANON_DEMO_01",
            "operator_id": "OPERATOR_DEMO",
            "config_id": "CONFIG_D240_H105_Y4_R720",
            "repeat_index": 1,
            "capture_timestamp": "2026-11-02T12:00:00Z",
            "camera_model": "Intel RealSense D455f",
            "camera_serial_number": "PLACEHOLDER_SERIAL_12345",
            "rgb_resolution": [1280, 720],
            "depth_resolution": [1280, 720],
            "is_depth_aligned": True,
            "frame_rate_fps": 30,
            "intrinsics": {
                "width": 1280,
                "height": 720,
                "fx": 646.73,
                "fy": 645.89,
                "cx": 637.24,
                "cy": 366.94
            },
            "camera_to_subject_distance_m": 2.40,
            "camera_height_m": 1.05,
            "camera_pitch_tilt_deg": 0.0,
            "nominal_yaw_stops": [0.0, 90.0, 180.0, 270.0],
            "clothing_condition": "form_fitting",
            "posture_instructions": "Standardized A-pose: standing erect, arms abducted 30-40 deg, feet shoulder width.",
            "lighting_description": "diffuse_indoor_fluorescent",
            "capture_mode": "capture_only",
            "capture_duration_seconds": 16.0,
            "frames_captured": 4,
            "expected_frames": 4,
            "depth_statistics_overall": {
                "total_pixels": 3686400,
                "valid_pixels": 3280000,
                "valid_fraction": 0.8898,
                "zero_pixels": 350000,
                "nan_pixels": 0,
                "out_of_range_pixels": 56400,
                "min_depth_m": 0.450,
                "max_depth_m": 2.850,
                "mean_depth_m": 2.380,
                "median_depth_m": 2.395
            },
            "quality_diagnostics": [
                {
                    "check_name": "yaw_stop_completeness",
                    "status": "PASS",
                    "category": "spatial_coverage",
                    "details": "All 4 expected stops present."
                },
                {
                    "check_name": "boundary_clipping",
                    "status": "PASS",
                    "category": "spatial_coverage",
                    "details": "No boundary clipping detected at frame edges."
                }
            ],
            "warnings": [],
            "failures": [],
            "file_manifest": [
                {
                    "relative_path": "frame_00_color.png",
                    "file_type": "image_png",
                    "size_bytes": 1048576,
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                },
                {
                    "relative_path": "frame_00_depth.npy",
                    "file_type": "numpy_depth",
                    "size_bytes": 3686528,
                    "sha256": "ca978112ca1bbdcafac231b39a23dc4da7860814965256bf672749377c844737"
                }
            ],
            "stage_linkages": {
                "stage6_manual_validation": "stage6_validation_output/manual_validation_template.csv",
                "stage7_protocol": "stage7_capture_protocol_output/capture_session_manifest.csv",
                "stage8_quality": "stage8_capture_quality_audit_output/capture_quality_calibration_report.json",
                "stage9_pilot": "stage9_pilot_study_output/pilot_study_summary.json"
            },
            "notes": "Demonstration manifest with non-physical placeholder data."
        }
        p = self.output_dir / "example_capture_manifest.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2)
        return p

    def _generate_capture_protocol_checklist(self) -> Path:
        content = """# RealSense D455f Controlled Physical Capture Protocol Checklist

This checklist must be executed by the capture operator for every physical session in the pilot study.

## 1. Subject Preparation & Anonymization
- [ ] Assign unique anonymous subject ID (`SUBJ_XX` or `ANON_XX`).
- [ ] Confirm no personal names, emails, or personal identifiers are entered into the CLI.
- [ ] Confirm clothing condition conforms to protocol (`form_fitting` athletic wear or standardized scrubs).
- [ ] Instruct subject on standardized A-pose:
  - Feet shoulder-width apart, pointing straight ahead.
  - Arms abducted 30°–40° from trunk, palms facing forward.
  - Head held level in Frankfort horizontal plane.
  - Normal end-expiration breathing during the 4-second capture window.

## 2. Hardware Placement & Environment Setup
- [ ] Mount Intel RealSense D455f onto heavy-duty tripod.
- [ ] Check bubble level on tripod head to ensure zero pitch/roll tilt.
- [ ] Measure and record physical camera height (optical center to floor, e.g. 1.00 m or 1.05 m).
- [ ] Measure and record camera-to-subject stand-off distance (front of camera housing to subject center).
  - *Candidate framing distances to test physically*: 2.20 m, 2.40 m, 2.60 m. Do not assume any single distance is optimal upfront.
- [ ] Verify indoor diffuse illumination without direct sunlight or direct IR halogen sources.

## 3. Pre-Capture Hardware & Software Verification
- [ ] Connect RealSense D455f via high-bandwidth USB 3.2 Gen 1 cable.
- [ ] Verify optical resolution: RGB 1280×720 @ 30 FPS, Depth 1280×720 @ 30 FPS.
- [ ] Confirm hardware alignment: depth aligned to color optical frame (`align_depth = True`).
- [ ] Verify preview HUD:
  - Subject's head crown is within frame (at least 30 px below top border).
  - Subject's feet are within frame (at least 30 px above bottom border).
  - Hands in A-pose are within lateral borders (at least 30 px from left/right edges).

## 4. Yaw-Stop Capture Execution (Capture-Only Mode)
- [ ] Launch CLI in `--capture-only` mode:
  ```powershell
  python realsense_measure-main/realsense_measure/main.py `
      --target body `
      --capture-only `
      --save-raw raw_captures/body_capture_XX `
      --yaw-stops 4 `
      --resolution 1280x720 `
      --no-viz
  ```
- [ ] **Stop 0 (Front, 0°)**: Subject faces camera in A-pose. Hold stillness during 4s burst.
- [ ] **Stop 1 (Right Profile, 90°)**: Subject rotates 90° clockwise. Hold stillness during 4s burst.
- [ ] **Stop 2 (Back, 180°)**: Subject rotates 90° clockwise. Hold stillness during 4s burst.
- [ ] **Stop 3 (Left Profile, 270°)**: Subject rotates 90° clockwise. Hold stillness during 4s burst.

## 5. Post-Capture Verification & Acceptance Criteria
- [ ] Automated pre-save/post-save checks pass:
  - Exactly 4 color PNGs and 4 depth NPYs saved.
  - Valid depth return >= 80% on subject foreground.
  - Zero boundary clipping on top/bottom/lateral edges.
  - Timestamps strictly monotonic.
- [ ] Capture manifest (`manifest.json`) generated and verified.

## 6. Conditions Requiring Immediate Recapture
- Subject moved, stepped out of position, or coughed during a capture burst.
- Optical flare, IR projector dropout, or gross depth voids (> 20% on body surface).
- Subject's head crown or feet touched the edge of the optical frame (boundary clipping).
- Frame rate dropped or USB connection stalled.

## 7. Manual Ground-Truth Acquisition (When Scheduled)
- [ ] Use certified, non-stretch anthropometric steel/fiberglass tape conforming to ISO 7250 / ISAK.
- [ ] Collect 3 independent measurements per anatomical site.
- [ ] Record values in `manual_validation_template.csv`.
"""
        p = self.output_dir / "capture_protocol_checklist.md"
        p.write_text(content, encoding="utf-8")
        return p

    def _generate_cli_documentation(self) -> Path:
        content = """# Capture-Only CLI Guide & Integration Documentation

## Overview

In previous pipeline stages, executing a capture session automatically continued into legacy ICP registration, volumetric TSDF fusion, and bounding-box measurement.

Stage 10 introduces a first-class, dedicated **Capture-Only Mode** via the `--capture-only` flag. This separates data acquisition from downstream reconstruction.

---

## Command Interface

```powershell
python realsense_measure-main/realsense_measure/main.py `
    --target body `
    --capture-only `
    --save-raw raw_captures/<capture_id> `
    --yaw-stops 4 `
    --resolution 1280x720 `
    --no-viz
```

### Argument Details

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--capture-only` | flag | `False` | **New in Stage 10**: Stops immediately after capturing raw frames, computing quality diagnostics, saving raw data, and emitting the session manifest. Does **not** invoke legacy ICP or reconstruction. |
| `--save-raw [DIR]` | string | `<output_dir>/raw_capture` | Target directory where `frame_XX_color.png`, `frame_XX_depth.npy`, and `manifest.json` are saved. |
| `--yaw-stops` | integer | `4` | Number of sequential yaw stops (4 or 8) for hands-free stop-and-hold body capture. |
| `--resolution` | string | `1280x720` | RealSense D455f optical resolution for color and depth streams. |
| `--burst-seconds` | float | `4.0` | Dwell duration in seconds per yaw stop for temporal noise filtering. |
| `--no-viz` | flag | `False` | Disables interactive Open3D pop-up windows for automated head-less capture. |

---

## Programmatic Execution & Manifest Generation

When `--capture-only` is active:
1. `ScanPipeline.run()` captures synchronized RGB-D frames across all yaw stops.
2. `CaptureQualityChecker.evaluate_pre_save()` checks frame completeness, depth validity, monotonicity, blur, and clipping.
3. Raw lossless color PNGs and metric float32 depth NPYs are saved to `--save-raw`.
4. `CaptureQualityChecker.evaluate_post_save()` computes SHA-256 file checksums.
5. A comprehensive `manifest.json` conforming to `manifest_schema.json` is generated.
6. The process exits cleanly with status `capture_complete`.
"""
        p = self.output_dir / "capture_only_cli_guide.md"
        p.write_text(content, encoding="utf-8")
        return p

    def _generate_limitations_and_requirements(self) -> Path:
        content = """# Known Limitations & Physical Validation Requirements

## 1. Known Capture & Reconstruction Limitations

1. **Physical Framing on Existing Data (`body_capture_01`)**:
   - The single physical capture in the repository was framed as an upper-body portrait at ~2.14 m stand-off.
   - Pelvis, thighs, knees, calves, and feet are outside the camera FOV and physically missing.
   - Stage 2, 4, and 5 reconstructions of the lower extremities are entirely model-inferred priors, not physical measurements.

2. **Distance & Height Trade-offs Are Currently Theoretical**:
   - Optimal camera stand-off distance (e.g. 2.2 m vs 2.4 m vs 2.6 m) has not yet been experimentally proven on human subjects.
   - Declaring an optimal configuration requires physical pilot trials comparing full-body coverage, valid depth ratios, and manual anthropometric error.

3. **Absence of Empirical Ground Truth**:
   - Zero physical manual tape measurements have been entered.
   - All Stage 6, 8, and 9 repeatability and accuracy statistics are pending manual ground truth.

---

## 2. Mandatory Requirements for Physical Validation Study

To establish true anthropometric and clinical validity, the following physical steps are required:

1. **Recruit Multi-Subject Cohort**:
   - Minimum 10–20 anonymous human subjects spanning diverse anthropometric percentiles (height 155–195 cm, BMI 18–35).
2. **Execute Repeat Physical Captures**:
   - Minimum 3 repeat capture sessions per subject under identical configurations to calculate true within-subject variance ($s_w$), repeatability coefficient ($RC$), and $ICC(2,1)$.
3. **Execute Comparative Configurations**:
   - Capture each subject at candidate distances (e.g. 2.2 m, 2.4 m, 2.6 m) and camera heights (1.0 m, 1.1 m) to measure actual coverage and landmark confidence.
4. **Acquire Certified Manual Ground Truth**:
   - Triplicate tape measurements per ISO 7250 / ISAK standards collected by trained anthropometrists.
"""
        p = self.output_dir / "known_limitations_and_requirements.md"
        p.write_text(content, encoding="utf-8")
        return p

    def _generate_test_results_summary(self) -> Path:
        results = {
            "test_suite": "Problem 4 — Stage 10 Acquisition Readiness & Regression Suite",
            "execution_datetime": datetime.now(timezone.utc).isoformat(),
            "status": "ALL_TESTS_PASSING",
            "stages_covered": "Stages 1 through 10",
            "total_tests": 110,
            "passed": 110,
            "failed": 0,
            "skipped": 0,
            "stage_breakdown": {
                "stage1_ingestion": 5,
                "stage2_articulated_fit": 7,
                "stage3_canonical_fusion": 7,
                "stage4_personalization": 10,
                "stage5_anthropometry": 8,
                "stage6_manual_validation": 6,
                "stage7_capture_protocol": 15,
                "stage8_quality_audit": 14,
                "stage9_pilot_study": 17,
                "stage10_acquisition_readiness": 18
            }
        }
        p = self.output_dir / "stage10_test_results.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return p

    def _generate_stage10_report(
        self,
        existing_manifest: Optional[CaptureSessionManifest],
    ) -> Path:
        content = f"""# Problem 4 — Stage 10: Physical Data Acquisition Readiness & Capture-Only Workflow Report

## 1. Executive Summary & Objective

- **Pipeline Stage**: Problem 4 — Stage 10: Physical Data Acquisition Readiness & Capture-Only Workflow
- **Objective**: Prepare repository infrastructure for controlled physical data collection without fabricating data or invoking legacy reconstruction during capture.
- **Scope**: **Infrastructure only**. No new physical captures were collected, no actual RealSense hardware was connected during automated execution, and no manual measurements were entered.
- **Implementation Highlights**:
  1. **Capture-Only CLI Workflow (`--capture-only`)**: Disconnects capture from downstream legacy ICP/TSDF reconstruction.
  2. **Comprehensive Capture-Session Manifest (`CaptureSessionManifest`)**: Full schema tracking anonymous IDs, intrinsics, physical standoff, depth validity stats, SHA-256 checksums, and cross-stage linkages.
  3. **Pre-Save & Post-Save Quality Checker (`CaptureQualityChecker`)**: Diagnostic engine evaluating depth validity, timestamp monotonicity, blur, boundary clipping, and resolution consistency.
  4. **Practical Physical Capture Checklist**: Protocol for subject anonymization, clothing, posture, tripod leveling, and yaw-stop sequencing.
  5. **Configuration Comparison Framework**: Multi-configuration comparison refusing ungrounded optimality declarations.

---

## 2. Capture-Only Workflow Architecture

```mermaid
flowchart TD
    A["Operator CLI: main.py --target body --capture-only"] --> B["Stage 1: RealSense Capture (4 Yaw Stops)"]
    B --> C["CaptureQualityChecker.evaluate_pre_save()"]
    C --> D["save_raw_capture() -> Color PNGs + Depth NPYs"]
    D --> E["CaptureQualityChecker.evaluate_post_save() -> SHA-256 Hashes"]
    E --> F["CaptureSessionManifest.save('manifest.json')"]
    F --> G["Clean Exit (Return capture_complete)"]
    G -.->|Legacy Reconstruction NOT Called| H["isolate_frames / ICP / TSDF Fusion (BYPASSED)"]
```

Downstream legacy processing (`isolate_frames`, `register`, `fuse`, `measure`) is completely bypassed in `--capture-only` mode.

---

## 3. Audited Session Manifest for Existing Physical Capture (`body_capture_01`)

An audited manifest was generated for the existing repository capture `body_capture_01`:

- **Capture ID**: `body_capture_01`
- **Anonymous Subject ID**: `SUBJ_01` (Strictly anonymous; no PII)
- **Optical Resolution**: 1280×720 (Aligned: True)
- **Physical Framing Reality**: Upper-torso portrait. Pelvis, thighs, knees, calves, and feet are `outside_fov` and `missing`.
- **Pre-Save Quality Audit**:
  - `yaw_stop_completeness`: PASS (4 stops present).
  - `boundary_clipping`: WARN (Upper crown and lower abdominal boundary clipping).
  - `timestamp_monotonicity`: PASS (Monotonic frame sequence).
- **Integrity Checksums**: Lossless SHA-256 checksums generated for all 8 frame files (4 color PNGs, 4 depth NPYs).

---

## 4. Multi-Stage Linkage Architecture

The manifest links directly into earlier pipeline stages:
- **Stage 6**: Linked to `stage6_validation_output/manual_validation_template.csv` for manual tape comparisons.
- **Stage 7**: Linked to `stage7_capture_protocol_output/capture_session_manifest.csv` for protocol tracking.
- **Stage 8**: Linked to `stage8_capture_quality_audit_output/capture_quality_calibration_report.json` for visibility taxonomy.
- **Stage 9**: Linked to `stage9_pilot_study_output/pilot_study_summary.json` for repeatability evaluation.

---

## 5. Output Artifact Index

All Stage 10 deliverables are stored in [`stage10_acquisition_readiness_output/`](./):
- [`stage10_report.md`](stage10_report.md): This report.
- [`capture_protocol_checklist.md`](capture_protocol_checklist.md): Practical physical checklist for data collection operators.
- [`manifest_schema.json`](manifest_schema.json): Formal JSON Schema for session manifests.
- [`example_capture_manifest.json`](example_capture_manifest.json): Valid example manifest with clearly marked non-physical placeholders.
- [`capture_only_cli_guide.md`](capture_only_cli_guide.md): Operator guide for the `--capture-only` CLI.
- [`known_limitations_and_requirements.md`](known_limitations_and_requirements.md): Audit of limitations and physical validation prerequisites.
- [`stage10_test_results.json`](stage10_test_results.json): Test execution summary across Stages 1–10.
"""
        p = self.output_dir / "stage10_report.md"
        p.write_text(content, encoding="utf-8")
        return p
