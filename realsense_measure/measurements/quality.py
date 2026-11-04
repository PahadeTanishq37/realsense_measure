"""
realsense_measure/measurements/quality.py
========================================
Stage 17: Anthropometric Measurement Quality Control & Governance Engine.

Provides quality status classification, data provenance tracking, and strict
scientific validation checks for all anthropometric measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MeasurementQualityStatus(str, Enum):
    """Quality status enum required by Stage 17 specifications."""
    VALID_FOR_EXPERIMENTAL_USE = "VALID_FOR_EXPERIMENTAL_USE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class DataSourceType(str, Enum):
    """Epistemological category of measurement value."""
    DIRECTLY_MEASURED = "DIRECTLY_MEASURED"      # Derived directly from physical sensor points
    ESTIMATED_GEOMETRIC = "ESTIMATED_GEOMETRIC"  # Sliced or fitted geometric primitive
    INFERRED = "INFERRED"                        # Statistically or anatomically estimated
    BLOCKED = "BLOCKED"                          # Measurement prevented by data/coverage limits


@dataclass
class QualityEvaluation:
    """Evaluation result for a candidate anthropometric measurement."""
    measurement_id: str
    status: MeasurementQualityStatus
    data_source: DataSourceType
    is_valid: bool
    confidence_score: float  # [0.0, 1.0]
    checks_passed: List[str]
    checks_failed: List[str]
    blocking_reason: Optional[str] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["data_source"] = self.data_source.value
        return d


class MeasurementQualityAuditor:
    """
    Audits measurement candidates against documented scientific criteria.
    """

    @classmethod
    def evaluate_landmark_distance(
        cls,
        measurement_id: str,
        dist_mm: Optional[float],
        landmark_a: Any,
        landmark_b: Any,
        expected_range_mm: Tuple[float, float] = (100.0, 1000.0),
    ) -> QualityEvaluation:
        """
        Audits Euclidean landmark-to-landmark segment length.
        """
        passed: List[str] = []
        failed: List[str] = []
        notes: List[str] = []

        if landmark_a is None or landmark_b is None:
            failed.append("Missing required landmark object")
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.NOT_AVAILABLE,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason="One or both required anatomical landmarks are null.",
            )

        # Check bounds
        if not landmark_a.in_image_bounds or not landmark_b.in_image_bounds:
            failed.append("Landmark truncated outside camera field of view")
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.NOT_AVAILABLE,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason=f"Landmark truncation ({landmark_a.landmark_id}: in_bounds={landmark_a.in_image_bounds}, {landmark_b.landmark_id}: in_bounds={landmark_b.in_image_bounds}).",
            )
        passed.append("Both landmarks within camera image bounds")

        # Check depth validity
        if not landmark_a.depth_valid or not landmark_b.depth_valid:
            failed.append("Infrared depth dropout at landmark location")
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.INSUFFICIENT_DATA,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason=f"Depth dropout at landmark ({landmark_a.landmark_id}: depth={landmark_a.depth_valid}, {landmark_b.landmark_id}: depth={landmark_b.depth_valid}).",
            )
        passed.append("Both landmarks have valid physical sensor depth")

        if dist_mm is None or dist_mm <= 0.0:
            failed.append("Non-positive distance value")
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.INVALID_GEOMETRY,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason="Computed distance is non-positive or undefined.",
            )

        # Check anatomical range
        min_mm, max_mm = expected_range_mm
        if not (min_mm <= dist_mm <= max_mm):
            failed.append(f"Distance {dist_mm:.1f} mm outside plausible anatomical range [{min_mm}, {max_mm}] mm")
            status = MeasurementQualityStatus.INVALID_GEOMETRY
            conf = 0.3
        else:
            passed.append(f"Distance {dist_mm:.1f} mm within plausible anatomical range [{min_mm}, {max_mm}] mm")
            status = MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE
            conf = 0.9

        # Noise penalty
        max_std = max(landmark_a.depth_std_mm, landmark_b.depth_std_mm)
        if max_std > 40.0:
            notes.append(f"Elevated landmark depth noise: {max_std:.1f} mm")
            if status == MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE:
                status = MeasurementQualityStatus.LOW_CONFIDENCE
                conf = 0.65

        min_vis = min(landmark_a.visibility_confidence, landmark_b.visibility_confidence)
        if min_vis < 0.60:
            notes.append(f"Marginal detector confidence: {min_vis:.2f}")
            if status == MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE:
                status = MeasurementQualityStatus.LOW_CONFIDENCE
                conf = 0.60

        return QualityEvaluation(
            measurement_id=measurement_id,
            status=status,
            data_source=DataSourceType.DIRECTLY_MEASURED,
            is_valid=(status == MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE),
            confidence_score=conf,
            checks_passed=passed,
            checks_failed=failed,
            blocking_reason=None,
            notes=notes,
        )

    @classmethod
    def evaluate_slice_circumference(
        cls,
        measurement_id: str,
        slice_analysis: Any,
    ) -> QualityEvaluation:
        """
        Audits cross-sectional circumference measurement.
        """
        passed: List[str] = []
        failed: List[str] = []

        if slice_analysis.num_points_in_slab < 30:
            failed.append("Insufficient point cloud density in slicing slab")
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.INSUFFICIENT_DATA,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason=slice_analysis.blocking_reason or "Insufficient point density in slab.",
            )
        passed.append(f"Point density in slab: {slice_analysis.num_points_in_slab} points")

        if not slice_analysis.is_closed_ring:
            failed.append(
                f"Incomplete ring perimeter (Coverage: {slice_analysis.observed_coverage_fraction*100.0:.1f}%, "
                f"Max angular gap: {slice_analysis.max_angular_gap_deg:.1f} deg)"
            )
            return QualityEvaluation(
                measurement_id=measurement_id,
                status=MeasurementQualityStatus.INSUFFICIENT_DATA,
                data_source=DataSourceType.BLOCKED,
                is_valid=False,
                confidence_score=0.0,
                checks_passed=passed,
                checks_failed=failed,
                blocking_reason=slice_analysis.blocking_reason,
                notes=[
                    "Circumference calculation blocked to prevent false perimeter under-reporting.",
                    "Single-view capture lacks posterior (back) observations.",
                ],
            )

        passed.append("Cross-sectional ring is closed and fully observed")
        return QualityEvaluation(
            measurement_id=measurement_id,
            status=MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE,
            data_source=DataSourceType.ESTIMATED_GEOMETRIC,
            is_valid=True,
            confidence_score=0.90,
            checks_passed=passed,
            checks_failed=failed,
            blocking_reason=None,
        )
