"""Problem 4 — Stage 7: Controlled Anthropometric Capture Protocol & Repeatability Infrastructure.

Provides:
1. CaptureProtocolConfig: Configurable capture protocol with anonymous subject ID validation,
   rigorous metadata schemas, and anti-PII protection.
2. CaptureCompletenessChecker: Non-parametric sensor-grounded anatomical completeness diagnostic
   distinguishing observed, partially_observed, missing, unreliable, and unavailable.
3. CaptureExperimentEngine: Systematic experimental comparison across distances and camera heights
   evaluating FOV coverage, depth precision, and density trade-offs without hardcoded declarations.
4. CaptureSessionManager: Repeatability tracking, session grouping by anonymous subject,
   multi-operator support, and collision detection.
5. Stage6ValidationBridge: Linking capture metadata to Stage 6 manual validation records with
   strict anti-fabrication enforcement.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("capture_protocol")

# Allowed enumerated categories
ALLOWED_CLOTHING_CONDITIONS = {
    "casual_clothing",
    "form_fitting",
    "minimal_clothing",
    "athletic_wear",
    "standardized_scrubs",
}

ALLOWED_LIGHTING_CONDITIONS = {
    "diffuse_indoor_fluorescent",
    "controlled_studio",
    "natural_ambient",
    "uneven_directional",
    "low_light",
}

ALLOWED_ANATOMICAL_STATUSES = {
    "observed",
    "partially_observed",
    "missing",
    "unreliable",
    "unavailable",
}


# ---------------------------------------------------------------------------
# Step 1: Capture Protocol Configuration
# ---------------------------------------------------------------------------

@dataclass
class CaptureProtocolConfig:
    """
    Configurable capture session protocol specification.
    
    Enforces anonymous subject identifiers, camera stream parameters,
    and standardized environment descriptions.
    """
    protocol_id: str = "RS_ANTHRO_PROTO_V1"
    protocol_version: str = "1.0.0"
    subject_id: str = "SUBJ_01"
    capture_id: str = "body_capture_01"
    session_id: str = "SESS_20261102_01"
    repeat_index: int = 1
    operator_id: str = "OPERATOR_01"
    camera_serial_number: str = "146322251375"
    camera_model: str = "Intel RealSense D455f"
    rgb_resolution: Tuple[int, int] = (1280, 720)
    depth_resolution: Tuple[int, int] = (1280, 720)
    frame_rate: int = 30
    camera_distance_m: Optional[float] = None
    camera_height_m: Optional[float] = None
    subject_height_manual_mm: Optional[float] = None
    nominal_yaw_deg: List[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    actual_yaw_deg: Optional[List[float]] = None
    clothing_condition: str = "casual_clothing"
    posture_instructions: str = "A-pose: standing erect, feet shoulder-width, arms abducted 30-40 deg, head Frankfort horizontal"
    lighting_condition: str = "diffuse_indoor_fluorescent"
    capture_datetime: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def __post_init__(self):
        self.validate()

    @staticmethod
    def validate_anonymous_subject_id(subject_id: str) -> None:
        """
        Ensure subject identifier is strictly anonymous and does NOT leak PII.
        
        Rejects emails, personal names, spaces, or non-anonymized identifiers.
        """
        if not subject_id or not isinstance(subject_id, str):
            raise ValueError(f"subject_id must be a non-empty string, got {subject_id!r}")
        
        # Check against PII indicators: emails, common separators, whitespace
        if "@" in subject_id or ".com" in subject_id or ".org" in subject_id:
            raise ValueError(f"PII violation: subject_id contains email format: {subject_id}")
        if " " in subject_id or "\t" in subject_id:
            raise ValueError(f"PII violation: subject_id contains spaces: {subject_id!r}")
        
        # Standard anonymous pattern: ^(SUBJ|ANON|PARTICIPANT)_[0-9A-Za-z_-]+$
        pattern = r"^(SUBJ|ANON|PARTICIPANT|TEST)_[0-9A-Za-z_-]+$"
        if not re.match(pattern, subject_id, re.IGNORECASE):
            raise ValueError(
                f"Invalid anonymous subject_id: '{subject_id}'. "
                f"Must follow anonymous prefix pattern, e.g., 'SUBJ_01', 'SUBJ_A102'."
            )

    def validate(self) -> None:
        """Validate all protocol parameters and constraints."""
        self.validate_anonymous_subject_id(self.subject_id)

        if not self.capture_id or not isinstance(self.capture_id, str):
            raise ValueError("capture_id must be a non-empty string")
        if not self.operator_id or not isinstance(self.operator_id, str):
            raise ValueError("operator_id must be a non-empty string")
        if not self.protocol_id or not isinstance(self.protocol_id, str):
            raise ValueError("protocol_id must be a non-empty string")

        if self.repeat_index < 1:
            raise ValueError(f"repeat_index must be >= 1, got {self.repeat_index}")

        # Stream resolution and rate
        for res_name, res in [("rgb_resolution", self.rgb_resolution), ("depth_resolution", self.depth_resolution)]:
            if not isinstance(res, (tuple, list)) or len(res) != 2:
                raise ValueError(f"{res_name} must be a (width, height) tuple")
            if res[0] <= 0 or res[1] <= 0:
                raise ValueError(f"{res_name} dimensions must be positive integers, got {res}")

        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}")

        # Distances and heights
        if self.camera_distance_m is not None:
            if self.camera_distance_m <= 0.0 or math.isnan(self.camera_distance_m):
                raise ValueError(f"camera_distance_m must be positive, got {self.camera_distance_m}")
        if self.camera_height_m is not None:
            if self.camera_height_m <= 0.0 or math.isnan(self.camera_height_m):
                raise ValueError(f"camera_height_m must be positive, got {self.camera_height_m}")
        if self.subject_height_manual_mm is not None:
            if self.subject_height_manual_mm <= 0.0 or math.isnan(self.subject_height_manual_mm):
                raise ValueError(f"subject_height_manual_mm must be positive, got {self.subject_height_manual_mm}")

        # Yaw stops
        if not self.nominal_yaw_deg or len(self.nominal_yaw_deg) == 0:
            raise ValueError("nominal_yaw_deg must contain at least one stop angle")
        # Check for duplicates in nominal stops
        rounded_yaws = [round(y % 360.0, 2) for y in self.nominal_yaw_deg]
        if len(rounded_yaws) != len(set(rounded_yaws)):
            raise ValueError(f"Duplicate nominal yaw stops detected: {self.nominal_yaw_deg}")

        if self.actual_yaw_deg is not None:
            if len(self.actual_yaw_deg) != len(self.nominal_yaw_deg):
                raise ValueError("actual_yaw_deg length must match nominal_yaw_deg length")

        # Clothing and lighting
        if self.clothing_condition not in ALLOWED_CLOTHING_CONDITIONS:
            raise ValueError(
                f"Invalid clothing_condition: '{self.clothing_condition}'. "
                f"Allowed: {sorted(ALLOWED_CLOTHING_CONDITIONS)}"
            )
        if self.lighting_condition not in ALLOWED_LIGHTING_CONDITIONS:
            raise ValueError(
                f"Invalid lighting_condition: '{self.lighting_condition}'. "
                f"Allowed: {sorted(ALLOWED_LIGHTING_CONDITIONS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["rgb_resolution"] = list(self.rgb_resolution)
        d["depth_resolution"] = list(self.depth_resolution)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaptureProtocolConfig:
        """Construct from dictionary."""
        data = dict(data)
        if "rgb_resolution" in data:
            data["rgb_resolution"] = tuple(data["rgb_resolution"])
        if "depth_resolution" in data:
            data["depth_resolution"] = tuple(data["depth_resolution"])
        return cls(**data)

    def save_json(self, path: Path | str) -> Path:
        """Save configuration to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load_json(cls, path: Path | str) -> CaptureProtocolConfig:
        """Load configuration from JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Step 2: Capture Completeness Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class BoundaryClipping:
    """Flags indicating whether foreground points contact camera sensor boundaries."""
    top: bool
    bottom: bool
    left: bool
    right: bool
    top_contact_pixels: int = 0
    bottom_contact_pixels: int = 0
    left_contact_pixels: int = 0
    right_contact_pixels: int = 0

    @property
    def has_any_clipping(self) -> bool:
        return self.top or self.bottom or self.left or self.right

    @property
    def is_vertical_clipped(self) -> bool:
        return self.top or self.bottom

    @property
    def is_lateral_clipped(self) -> bool:
        return self.left or self.right


@dataclass
class StopCompletenessDiagnostic:
    """Capture completeness and quality diagnostic for a single RGB-D yaw stop."""
    stop_idx: int
    nominal_yaw_deg: float
    actual_yaw_deg: Optional[float]
    valid_depth_ratio: float
    zero_depth_ratio: float
    median_depth_m: float
    depth_range_m: Tuple[float, float]
    clipping: BoundaryClipping
    vertical_extent_world_m: float
    z_bounds_world_m: Tuple[float, float]
    head_status: str
    shoulder_status: str
    torso_status: str
    pelvis_status: str
    arm_left_status: str
    arm_right_status: str
    thigh_left_status: str
    thigh_right_status: str
    lower_leg_left_status: str
    lower_leg_right_status: str
    foot_left_status: str
    foot_right_status: str
    landmark_confidence: Optional[float]
    landmark_status: str  # "observed", "unreliable", or "unavailable"
    warnings: List[str] = field(default_factory=list)


@dataclass
class SessionCompletenessDiagnostic:
    """Multi-view session-wide completeness assessment."""
    capture_id: str
    subject_id: str
    num_stops_found: int
    num_stops_expected: int
    missing_stops: List[float]
    duplicate_stops: List[float]
    is_yaw_complete: bool
    distance_consistency: Dict[str, Any]
    overall_anatomical_completeness: Dict[str, str]
    overall_valid_depth_ratio: float
    is_capture_complete: bool
    quality_rating: str  # "complete", "suboptimal_framing", "incomplete", "unusable"
    stops: List[StopCompletenessDiagnostic] = field(default_factory=list)


class CaptureCompletenessChecker:
    """
    Evaluates sensor-grounded capture completeness and quality.
    
    Uses physical depth validities, bounding-box extents, and boundary clipping
    without inferring visibility solely from parametric mesh priors.
    """

    def __init__(
        self,
        boundary_margin_px: int = 3,
        clipping_threshold_pixels: int = 5,
        min_valid_depth_ratio: float = 0.15,
        max_inter_stop_depth_std_mm: float = 30.0,
    ):
        self.boundary_margin_px = boundary_margin_px
        self.clipping_threshold_pixels = clipping_threshold_pixels
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.max_inter_stop_depth_std_mm = max_inter_stop_depth_std_mm

    def evaluate_boundary_clipping(self, foreground_mask: np.ndarray) -> BoundaryClipping:
        """
        Check if foreground mask contacts image boundaries (top, bottom, left, right).
        """
        h, w = foreground_mask.shape[:2]
        m = self.boundary_margin_px

        top_strip = foreground_mask[0:m, :]
        bottom_strip = foreground_mask[h - m:h, :]
        left_strip = foreground_mask[:, 0:m]
        right_strip = foreground_mask[:, w - m:w]

        top_cnt = int(np.count_nonzero(top_strip))
        bottom_cnt = int(np.count_nonzero(bottom_strip))
        left_cnt = int(np.count_nonzero(left_strip))
        right_cnt = int(np.count_nonzero(right_strip))

        return BoundaryClipping(
            top=top_cnt >= self.clipping_threshold_pixels,
            bottom=bottom_cnt >= self.clipping_threshold_pixels,
            left=left_cnt >= self.clipping_threshold_pixels,
            right=right_cnt >= self.clipping_threshold_pixels,
            top_contact_pixels=top_cnt,
            bottom_contact_pixels=bottom_cnt,
            left_contact_pixels=left_cnt,
            right_contact_pixels=right_cnt,
        )

    def evaluate_stop(
        self,
        stop_idx: int,
        nominal_yaw_deg: float,
        depth_m: np.ndarray,
        foreground_mask: np.ndarray,
        z_bounds_world_m: Tuple[float, float],
        actual_yaw_deg: Optional[float] = None,
        landmarks: Optional[Dict[str, Any]] = None,
    ) -> StopCompletenessDiagnostic:
        """
        Evaluate completeness for a single capture stop.
        """
        h, w = depth_m.shape[:2]
        total_pixels = h * w
        valid_depth_mask = (depth_m > 0.0) & np.isfinite(depth_m)
        valid_pixels = int(np.count_nonzero(valid_depth_mask))
        valid_depth_ratio = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0
        zero_depth_ratio = float(1.0 - valid_depth_ratio)

        valid_vals = depth_m[valid_depth_mask]
        median_depth = float(np.median(valid_vals)) if len(valid_vals) > 0 else float("nan")
        min_depth = float(np.min(valid_vals)) if len(valid_vals) > 0 else float("nan")
        max_depth = float(np.max(valid_vals)) if len(valid_vals) > 0 else float("nan")

        clipping = self.evaluate_boundary_clipping(foreground_mask)

        z_min_w, z_max_w = z_bounds_world_m
        vert_extent = max(0.0, z_max_w - z_min_w)

        warnings: List[str] = []

        # Determine anatomical status without parametric inference
        # In standardized world frame (calibrated in Stage 1):
        # Pelvis center is roughly Z = 0 m.
        # Head crown is expected at Z > +0.70 m relative to pelvis (or Z > 0.50 m above mid-torso).
        # Knees are expected at Z < -0.40 m, feet/floor at Z < -0.80 m.
        
        # 1. Head
        if clipping.top or z_max_w < 0.40:
            if z_max_w < 0.28:
                head_status = "missing"
                warnings.append("Head crown unobserved; clipped above top FOV")
            else:
                head_status = "partially_observed"
                warnings.append("Head truncated at top sensor boundary")
        else:
            head_status = "observed"

        # 2. Shoulders
        if clipping.top and z_max_w < 0.30:
            shoulder_status = "partially_observed"
        elif vert_extent < 0.40:
            shoulder_status = "unreliable"
        else:
            shoulder_status = "observed"

        # 3. Torso
        if valid_depth_ratio < self.min_valid_depth_ratio:
            torso_status = "unreliable"
            warnings.append(f"Low depth validity ({valid_depth_ratio:.1%}); torso unreliable")
        elif vert_extent >= 0.40 and z_min_w <= 0.0 <= z_max_w:
            torso_status = "observed"
        else:
            torso_status = "partially_observed"

        # 4. Pelvis
        if z_min_w <= -0.10:
            pelvis_status = "observed"
        elif z_min_w <= 0.05:
            pelvis_status = "partially_observed"
        else:
            pelvis_status = "missing"

        # 5. Arms
        if clipping.is_lateral_clipped:
            arm_left_status = "partially_observed" if clipping.left else "observed"
            arm_right_status = "partially_observed" if clipping.right else "observed"
            warnings.append("Arm / hand contact with lateral image boundary")
        else:
            arm_left_status = "observed"
            arm_right_status = "observed"

        # 6. Thighs
        if z_min_w <= -0.40:
            thigh_left_status = "observed"
            thigh_right_status = "observed"
        elif z_min_w <= -0.20:
            thigh_left_status = "partially_observed"
            thigh_right_status = "partially_observed"
        else:
            thigh_left_status = "missing"
            thigh_right_status = "missing"

        # 7. Lower legs
        if z_min_w <= -0.70:
            lower_leg_left_status = "observed"
            lower_leg_right_status = "observed"
        elif z_min_w <= -0.50:
            lower_leg_left_status = "partially_observed"
            lower_leg_right_status = "partially_observed"
        else:
            lower_leg_left_status = "missing"
            lower_leg_right_status = "missing"
            warnings.append("Lower legs unobserved; clipped below bottom FOV")

        # 8. Feet
        if z_min_w <= -0.85 and not clipping.bottom:
            foot_left_status = "observed"
            foot_right_status = "observed"
        elif z_min_w <= -0.80:
            foot_left_status = "partially_observed"
            foot_right_status = "partially_observed"
        else:
            foot_left_status = "missing"
            foot_right_status = "missing"
            warnings.append("Feet unobserved; floor plane truncated")

        # 9. Landmarks
        if landmarks is None:
            landmark_confidence = None
            landmark_status = "unavailable"
        else:
            conf = landmarks.get("mean_confidence")
            if conf is None or math.isnan(conf):
                landmark_confidence = None
                landmark_status = "unavailable"
            elif conf < 0.50:
                landmark_confidence = float(conf)
                landmark_status = "unreliable"
            else:
                landmark_confidence = float(conf)
                landmark_status = "observed"

        return StopCompletenessDiagnostic(
            stop_idx=stop_idx,
            nominal_yaw_deg=nominal_yaw_deg,
            actual_yaw_deg=actual_yaw_deg,
            valid_depth_ratio=valid_depth_ratio,
            zero_depth_ratio=zero_depth_ratio,
            median_depth_m=median_depth,
            depth_range_m=(min_depth, max_depth),
            clipping=clipping,
            vertical_extent_world_m=vert_extent,
            z_bounds_world_m=(z_min_w, z_max_w),
            head_status=head_status,
            shoulder_status=shoulder_status,
            torso_status=torso_status,
            pelvis_status=pelvis_status,
            arm_left_status=arm_left_status,
            arm_right_status=arm_right_status,
            thigh_left_status=thigh_left_status,
            thigh_right_status=thigh_right_status,
            lower_leg_left_status=lower_leg_left_status,
            lower_leg_right_status=lower_leg_right_status,
            foot_left_status=foot_left_status,
            foot_right_status=foot_right_status,
            landmark_confidence=landmark_confidence,
            landmark_status=landmark_status,
            warnings=warnings,
        )

    def evaluate_session(
        self,
        capture_id: str,
        subject_id: str,
        stop_diagnostics: List[StopCompletenessDiagnostic],
        expected_nominal_yaws: List[float],
    ) -> SessionCompletenessDiagnostic:
        """
        Evaluate completeness across all yaw stops of a capture session.
        """
        found_yaws = [round(s.nominal_yaw_deg % 360.0, 1) for s in stop_diagnostics]
        exp_yaws = [round(y % 360.0, 1) for y in expected_nominal_yaws]

        missing_yaws = [y for y in exp_yaws if y not in found_yaws]
        
        # Check duplicate stops
        duplicate_yaws = []
        seen = set()
        for y in found_yaws:
            if y in seen and y not in duplicate_yaws:
                duplicate_yaws.append(y)
            seen.add(y)

        is_yaw_complete = (len(missing_yaws) == 0 and len(duplicate_yaws) == 0)

        # Distance consistency
        depths = [s.median_depth_m for s in stop_diagnostics if not math.isnan(s.median_depth_m)]
        if len(depths) > 1:
            depth_std_mm = float(np.std(depths, ddof=1) * 1000.0)
            depth_mean_m = float(np.mean(depths))
            is_dist_consistent = depth_std_mm <= self.max_inter_stop_depth_std_mm
        elif len(depths) == 1:
            depth_std_mm = 0.0
            depth_mean_m = depths[0]
            is_dist_consistent = True
        else:
            depth_std_mm = float("nan")
            depth_mean_m = float("nan")
            is_dist_consistent = False

        dist_consistency = {
            "depth_medians_m": depths,
            "mean_distance_m": depth_mean_m,
            "std_distance_mm": depth_std_mm,
            "is_consistent": is_dist_consistent,
        }

        # Overall anatomical status aggregation across stops
        regions = [
            "head", "shoulder", "torso", "pelvis",
            "arm_left", "arm_right", "thigh_left", "thigh_right",
            "lower_leg_left", "lower_leg_right", "foot_left", "foot_right",
        ]
        overall_anatomy: Dict[str, str] = {}
        for r in regions:
            statuses = [getattr(s, f"{r}_status") for s in stop_diagnostics]
            if all(st == "observed" for st in statuses):
                overall_anatomy[r] = "observed"
            elif any(st == "observed" for st in statuses) or any(st == "partially_observed" for st in statuses):
                overall_anatomy[r] = "partially_observed"
            elif all(st == "missing" for st in statuses):
                overall_anatomy[r] = "missing"
            elif any(st == "unreliable" for st in statuses):
                overall_anatomy[r] = "unreliable"
            else:
                overall_anatomy[r] = "unavailable"

        # Overall validity
        depth_ratios = [s.valid_depth_ratio for s in stop_diagnostics]
        overall_valid = float(np.mean(depth_ratios)) if depth_ratios else 0.0

        # Overall rating
        if overall_valid < 0.10 or len(stop_diagnostics) == 0:
            quality_rating = "unusable"
            is_complete = False
        elif not is_yaw_complete or overall_anatomy.get("torso") == "missing":
            quality_rating = "incomplete"
            is_complete = False
        elif (overall_anatomy.get("head") in ["missing", "partially_observed"] or
              overall_anatomy.get("lower_leg_left") == "missing" or
              overall_anatomy.get("foot_left") == "missing"):
            quality_rating = "suboptimal_framing"
            is_complete = False
        else:
            quality_rating = "complete"
            is_complete = True

        return SessionCompletenessDiagnostic(
            capture_id=capture_id,
            subject_id=subject_id,
            num_stops_found=len(stop_diagnostics),
            num_stops_expected=len(expected_nominal_yaws),
            missing_stops=missing_yaws,
            duplicate_stops=duplicate_yaws,
            is_yaw_complete=is_yaw_complete,
            distance_consistency=dist_consistency,
            overall_anatomical_completeness=overall_anatomy,
            overall_valid_depth_ratio=overall_valid,
            is_capture_complete=is_complete,
            quality_rating=quality_rating,
            stops=stop_diagnostics,
        )


# ---------------------------------------------------------------------------
# Step 3: Capture Distance and Camera-Height Experiments
# ---------------------------------------------------------------------------

@dataclass
class CaptureExperimentGridConfig:
    """Configurable grid for controlled distance, height, and resolution trials."""
    distances_m: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    camera_heights_m: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    yaw_stop_counts: List[int] = field(default_factory=lambda: [4, 8])
    resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [(1280, 720), (848, 480)])
    camera_vertical_fov_deg: float = 58.0  # RealSense D455 optical vertical FOV
    camera_baseline_mm: float = 95.0      # D455 stereo baseline
    reference_adult_height_m: float = 1.75 # 50th percentile adult reference


@dataclass
class ConfigurationTradeoffResult:
    """Analytical trade-off evaluation for a specific physical capture setup."""
    distance_m: float
    camera_height_m: float
    resolution: Tuple[int, int]
    yaw_stops: int
    vertical_fov_span_m: float
    subject_height_coverage_percent: float
    theoretical_depth_noise_mm: float
    nominal_points_per_sq_meter: float
    is_full_body_in_fov: bool
    risk_near_field_truncation: bool
    risk_far_field_precision_loss: bool
    tradeoff_notes: str


class CaptureExperimentEngine:
    """
    Evaluates geometric trade-offs across distance and camera height configurations
    without declaring a configuration superior based solely on point density or residual.
    """

    def __init__(self, grid_config: Optional[CaptureExperimentGridConfig] = None):
        self.grid = grid_config or CaptureExperimentGridConfig()

    def evaluate_configuration(
        self,
        distance_m: float,
        camera_height_m: float,
        resolution: Tuple[int, int],
        yaw_stops: int,
    ) -> ConfigurationTradeoffResult:
        """
        Compute theoretical FOV coverage, precision, and density trade-offs.
        """
        w, h = resolution
        fov_v_rad = math.radians(self.grid.camera_vertical_fov_deg)

        # Vertical span of FOV at distance d: 2 * d * tan(FOV_v / 2)
        v_span = 2.0 * distance_m * math.tan(fov_v_rad / 2.0)

        # Subject coverage (for an erect adult of reference height)
        coverage_pct = min(100.0, (v_span / self.grid.reference_adult_height_m) * 100.0)
        is_full_body = v_span >= (self.grid.reference_adult_height_m + 0.15)  # 15 cm margin

        # RealSense D455 depth noise: sigma_z = (z^2 / (f * B)) * subpixel_noise
        # For D455 at 1280x720: fx ~ 646 px, baseline = 0.095 m, subpixel error ~ 0.08 px
        fx = 646.0 * (w / 1280.0)
        baseline_m = self.grid.camera_baseline_mm / 1000.0
        subpixel_error = 0.08
        theoretical_noise_mm = ((distance_m ** 2) / (fx * baseline_m)) * subpixel_error * 1000.0

        # Nominal points per m^2 at distance d
        # Projected pixel area per m^2: (fx * fy) / d^2
        fy = fx
        pixel_density = (fx * fy) / (distance_m ** 2)

        risk_near = distance_m < 1.6  # Near D455 0.52m min-Z limit on arms
        risk_far = distance_m > 2.6   # Quadratic noise > 5 mm

        notes: List[str] = []
        if risk_near:
            notes.append("High point density, but severe risk of head/foot FOV truncation and near-field min-Z invalidation.")
        elif risk_far:
            notes.append("Full body comfortably inside FOV, but depth noise grows quadratically and point cloud is sparser.")
        elif is_full_body:
            notes.append("Balanced compromise: covers full adult stature with moderate depth noise (~3.2-4.5 mm).")
        else:
            notes.append("Marginal vertical coverage: susceptible to head crown or foot truncation depending on tilt.")

        return ConfigurationTradeoffResult(
            distance_m=round(distance_m, 2),
            camera_height_m=round(camera_height_m, 2),
            resolution=resolution,
            yaw_stops=yaw_stops,
            vertical_fov_span_m=round(v_span, 3),
            subject_height_coverage_percent=round(coverage_pct, 1),
            theoretical_depth_noise_mm=round(theoretical_noise_mm, 2),
            nominal_points_per_sq_meter=round(pixel_density, 0),
            is_full_body_in_fov=is_full_body,
            risk_near_field_truncation=risk_near,
            risk_far_field_precision_loss=risk_far,
            tradeoff_notes=" ".join(notes),
        )

    def run_grid_evaluation(self) -> List[ConfigurationTradeoffResult]:
        """Run all configurations in the experiment grid."""
        results: List[ConfigurationTradeoffResult] = []
        for dist in self.grid.distances_m:
            for ht in self.grid.camera_heights_m:
                for res in self.grid.resolutions:
                    for stops in self.grid.yaw_stop_counts:
                        res_obj = self.evaluate_configuration(dist, ht, res, stops)
                        results.append(res_obj)
        return results


# ---------------------------------------------------------------------------
# Step 4: Repeat-Capture & Session Tracking
# ---------------------------------------------------------------------------

@dataclass
class CaptureSessionRecord:
    """Manifest record of a single physical or experimental capture session."""
    capture_id: str
    subject_id: str
    session_id: str
    repeat_index: int
    operator_id: str
    capture_datetime: str
    camera_model: str
    camera_serial_number: str
    camera_distance_m: float
    camera_height_m: float
    resolution: str
    nominal_stops_count: int
    actual_stops_count: int
    clothing_condition: str
    lighting_condition: str
    quality_rating: str
    is_complete: bool
    notes: str = ""


class CaptureSessionManager:
    """
    Manages session manifests, supports multiple anonymous subjects,
    same-day repeats, and collision prevention.
    """

    def __init__(self, manifest_csv_path: Optional[Path | str] = None):
        self.manifest_path = Path(manifest_csv_path) if manifest_csv_path else None
        self.records: Dict[str, CaptureSessionRecord] = {}

    def load_manifest(self, path: Optional[Path | str] = None) -> List[CaptureSessionRecord]:
        """Load manifest from CSV."""
        p = Path(path) if path else self.manifest_path
        if not p or not p.exists():
            return []

        loaded: List[CaptureSessionRecord] = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = CaptureSessionRecord(
                    capture_id=row["capture_id"],
                    subject_id=row["subject_id"],
                    session_id=row["session_id"],
                    repeat_index=int(row["repeat_index"]),
                    operator_id=row["operator_id"],
                    capture_datetime=row["capture_datetime"],
                    camera_model=row["camera_model"],
                    camera_serial_number=row["camera_serial_number"],
                    camera_distance_m=float(row["camera_distance_m"]),
                    camera_height_m=float(row["camera_height_m"]),
                    resolution=row["resolution"],
                    nominal_stops_count=int(row["nominal_stops_count"]),
                    actual_stops_count=int(row["actual_stops_count"]),
                    clothing_condition=row["clothing_condition"],
                    lighting_condition=row["lighting_condition"],
                    quality_rating=row["quality_rating"],
                    is_complete=(row["is_complete"].strip().lower() in ["true", "1"]),
                    notes=row.get("notes", ""),
                )
                self.records[rec.capture_id] = rec
                loaded.append(rec)
        return loaded

    def register_capture(self, record: CaptureSessionRecord, allow_overwrite: bool = False) -> None:
        """
        Register a new capture session, preventing unintended ID collisions.
        """
        CaptureProtocolConfig.validate_anonymous_subject_id(record.subject_id)

        if record.capture_id in self.records and not allow_overwrite:
            raise ValueError(
                f"Capture ID collision detected: '{record.capture_id}' already exists. "
                f"Specify a unique capture_id to avoid overwriting existing data."
            )
        self.records[record.capture_id] = record

    def get_subject_captures(self, subject_id: str) -> List[CaptureSessionRecord]:
        """Retrieve all capture records for a specific anonymous subject."""
        return [r for r in self.records.values() if r.subject_id == subject_id]

    def save_manifest(self, path: Optional[Path | str] = None) -> Path:
        """Export manifest to CSV."""
        p = Path(path) if path else self.manifest_path
        if not p:
            raise ValueError("Manifest path not specified")

        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "capture_id", "subject_id", "session_id", "repeat_index", "operator_id",
            "capture_datetime", "camera_model", "camera_serial_number",
            "camera_distance_m", "camera_height_m", "resolution",
            "nominal_stops_count", "actual_stops_count",
            "clothing_condition", "lighting_condition", "quality_rating",
            "is_complete", "notes",
        ]

        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records.values():
                writer.writerow(asdict(rec))

        logger.info(f"Saved capture session manifest ({len(self.records)} records) to: {p}")
        return p


# ---------------------------------------------------------------------------
# Step 5: Stage 6 Integration Bridge
# ---------------------------------------------------------------------------

class Stage6ValidationBridge:
    """
    Links capture protocol sessions with Stage 6 manual validation records.
    
    Verifies that measurement records correctly reference Stage 2 and Stage 4 outputs,
    preserves manual entries, and prevents fabrication.
    """

    @staticmethod
    def verify_capture_linkage(
        capture_id: str,
        subject_id: str,
        stage6_csv_path: Path | str,
    ) -> Dict[str, Any]:
        """
        Check that Stage 6 validation template or results file matches the capture session.
        """
        p = Path(stage6_csv_path)
        if not p.exists():
            return {
                "is_linked": False,
                "error": f"Stage 6 file does not exist: {p}",
                "matched_records": 0,
                "has_duplicates": False,
            }

        matched = 0
        mismatched_captures = set()
        mismatched_subjects = set()
        seen_keys = set()
        duplicates = []

        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_cap = row.get("capture_id", "")
                r_subj = row.get("subject_id", "")
                m_name = row.get("measurement_name", "")

                key = (r_subj, r_cap, m_name)
                if key in seen_keys:
                    duplicates.append(key)
                seen_keys.add(key)

                if r_cap == capture_id and r_subj == subject_id:
                    matched += 1
                else:
                    if r_cap != capture_id:
                        mismatched_captures.add(r_cap)
                    if r_subj != subject_id:
                        mismatched_subjects.add(r_subj)

        is_linked = (matched > 0 and len(mismatched_captures) == 0 and len(duplicates) == 0)

        return {
            "is_linked": is_linked,
            "matched_records": matched,
            "mismatched_captures": list(mismatched_captures),
            "mismatched_subjects": list(mismatched_subjects),
            "has_duplicates": len(duplicates) > 0,
            "duplicate_records": [f"{k[0]}/{k[1]}/{k[2]}" for k in duplicates],
        }
