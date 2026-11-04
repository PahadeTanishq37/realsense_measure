"""
realsense_measure/measurements/anthropometry.py
==============================================
Stage 17: Upper-Body Anthropometric Measurement Pipeline (Offline).

Computes defensible upper-body anthropometric measurements from physical RGB-D
frames or verified subject point clouds.

SAFEGUARDS ENFORCED:
1. MediaPipe landmarks are tracked and labeled as ESTIMATED_2D_3D_LANDMARK.
2. Segment lengths (shoulders, arms) are computed via 3D Euclidean distances between
   unprojected valid physical points.
3. Cross-sectional slices (chest, waist) audit angular bin coverage.
4. If a ring is incomplete (e.g. single-view capture lacking posterior back surface),
   circumference calculation is strictly BLOCKED.
5. Missing anatomy is NEVER hallucinated or filled with statistical body models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .landmarks import AnatomicalLandmark, AnatomicalLandmarksExtractor
from .quality import (
    DataSourceType,
    MeasurementQualityAuditor,
    MeasurementQualityStatus,
    QualityEvaluation,
)
from .slicing import AnatomicalPlaneSlicer, SliceRingAnalysis

logger = logging.getLogger(__name__)


@dataclass
class AnthropometricMeasurementRecord:
    """Standardized record for an extracted or audited anthropometric measurement."""
    measurement_id: str
    measurement_name: str
    anatomical_definition: str
    units: str                          # "mm"
    value_mm: Optional[float]           # None if blocked or unavailable
    observed_surface_coverage_fraction: float  # [0.0, 1.0]
    missing_surface_fraction: float     # 1.0 - coverage
    source_points_count: int
    plane_or_landmark_definition: str
    data_source_type: str               # DataSourceType
    quality_status: str                 # MeasurementQualityStatus
    is_valid: bool
    confidence_score: float             # [0.0, 1.0]
    blocking_reason: Optional[str] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UpperBodyAnthropometryPipeline:
    """
    Offline Upper-Body Anthropometric Measurement Pipeline.
    """

    def __init__(
        self,
        intrinsics: Dict[str, float],
        min_depth_m: float = 0.30,
        max_depth_m: float = 3.50,
    ):
        self.intrinsics = dict(intrinsics)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.landmark_extractor = AnatomicalLandmarksExtractor(
            intrinsics=self.intrinsics,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )

    def process_rgbd_frame(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
        depth_std_m: Optional[np.ndarray] = None,
        point_cloud_3d: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete measurement pipeline on RGB-D frame.
        """
        # Step 1: Extract and audit landmarks
        landmarks = self.landmark_extractor.extract_landmarks(
            color_bgr=color_bgr,
            depth_m=depth_m,
            person_mask=person_mask,
            depth_std_m=depth_std_m,
        )

        # Step 2: Build or obtain subject 3D point cloud
        if point_cloud_3d is not None and len(point_cloud_3d) > 0:
            pts_3d = point_cloud_3d
        else:
            pts_3d = self._build_subject_cloud(color_bgr, depth_m, person_mask)

        # Step 3: Compute segment lengths
        records: Dict[str, AnthropometricMeasurementRecord] = {}

        # 3.1 Biacromial Breadth (Shoulder Breadth)
        rec_sh = self._measure_biacromial_breadth(landmarks)
        records[rec_sh.measurement_id] = rec_sh

        # 3.2 Left Upper-Arm Length
        rec_ual = self._measure_limb_segment(
            landmarks=landmarks,
            m_id="upper_arm_length_left",
            m_name="Left Upper-Arm Segment Length",
            anat_def="3D Euclidean distance between left acromion and left lateral epicondyle (elbow).",
            lm_a_id="acromion_left",
            lm_b_id="lateral_epicondyle_left",
            expected_range_mm=(180.0, 480.0),
        )
        records[rec_ual.measurement_id] = rec_ual

        # 3.3 Right Upper-Arm Length
        rec_uar = self._measure_limb_segment(
            landmarks=landmarks,
            m_id="upper_arm_length_right",
            m_name="Right Upper-Arm Segment Length",
            anat_def="3D Euclidean distance between right acromion and right lateral epicondyle (elbow).",
            lm_a_id="acromion_right",
            lm_b_id="lateral_epicondyle_right",
            expected_range_mm=(180.0, 480.0),
        )
        records[rec_uar.measurement_id] = rec_uar

        # 3.4 Left Forearm Length
        rec_fal = self._measure_limb_segment(
            landmarks=landmarks,
            m_id="forearm_length_left",
            m_name="Left Forearm Segment Length",
            anat_def="3D Euclidean distance between left lateral epicondyle (elbow) and left radial styloid (wrist).",
            lm_a_id="lateral_epicondyle_left",
            lm_b_id="radial_styloid_left",
            expected_range_mm=(160.0, 380.0),
        )
        records[rec_fal.measurement_id] = rec_fal

        # 3.5 Right Forearm Length
        rec_far = self._measure_limb_segment(
            landmarks=landmarks,
            m_id="forearm_length_right",
            m_name="Right Forearm Segment Length",
            anat_def="3D Euclidean distance between right lateral epicondyle (elbow) and right radial styloid (wrist).",
            lm_a_id="lateral_epicondyle_right",
            lm_b_id="radial_styloid_right",
            expected_range_mm=(160.0, 380.0),
        )
        records[rec_far.measurement_id] = rec_far

        # Step 4: Cross-sectional trunk slicing
        # 4.1 Chest Plane (Mesosternale)
        rec_chest_circ, rec_chest_breadth, chest_slice_analysis = self._slice_trunk_plane(
            pts_3d=pts_3d,
            landmarks=landmarks,
            ref_lm_id="mesosternale_chest",
            fallback_relative_height=0.35,
            plane_prefix="chest",
            plane_display_name="Chest (Mesosternale Level)",
            anat_def_circ="Closed cross-sectional outer perimeter of the trunk at mesosternale level.",
            anat_def_breadth="Maximum transverse lateral dimension across trunk at mesosternale level.",
        )
        records[rec_chest_circ.measurement_id] = rec_chest_circ
        records[rec_chest_breadth.measurement_id] = rec_chest_breadth

        # 4.2 Waist Plane (Omphalion)
        rec_waist_circ, rec_waist_breadth, waist_slice_analysis = self._slice_trunk_plane(
            pts_3d=pts_3d,
            landmarks=landmarks,
            ref_lm_id="omphalion_waist",
            fallback_relative_height=0.70,
            plane_prefix="waist",
            plane_display_name="Waist (Omphalion Level)",
            anat_def_circ="Closed cross-sectional outer perimeter of the trunk at omphalion level.",
            anat_def_breadth="Maximum transverse lateral dimension across trunk at omphalion level.",
        )
        records[rec_waist_circ.measurement_id] = rec_waist_circ
        records[rec_waist_breadth.measurement_id] = rec_waist_breadth

        return {
            "landmarks": {k: v.to_dict() for k, v in landmarks.items()},
            "measurements": {k: v.to_dict() for k, v in records.items()},
            "slice_analyses": {
                "chest": chest_slice_analysis.to_dict(),
                "waist": waist_slice_analysis.to_dict(),
            },
            "point_cloud_size": len(pts_3d),
        }

    # -----------------------------------------------------------------------
    # Helper Measurement Methods
    # -----------------------------------------------------------------------

    def _measure_biacromial_breadth(
        self,
        landmarks: Dict[str, AnatomicalLandmark],
    ) -> AnthropometricMeasurementRecord:
        """Measures shoulder breadth between acromion landmarks."""
        lm_l = landmarks.get("acromion_left")
        lm_r = landmarks.get("acromion_right")

        dist_mm = None
        if lm_l and lm_r and lm_l.depth_valid and lm_r.depth_valid and lm_l.point_3d_camera_m and lm_r.point_3d_camera_m:
            p_l = np.array(lm_l.point_3d_camera_m)
            p_r = np.array(lm_r.point_3d_camera_m)
            dist_mm = float(np.linalg.norm(p_l - p_r) * 1000.0)

        eval_res = MeasurementQualityAuditor.evaluate_landmark_distance(
            measurement_id="biacromial_breadth",
            dist_mm=dist_mm,
            landmark_a=lm_l,
            landmark_b=lm_r,
            expected_range_mm=(250.0, 600.0),
        )

        return AnthropometricMeasurementRecord(
            measurement_id="biacromial_breadth",
            measurement_name="Biacromial Breadth (Shoulder Breadth)",
            anatomical_definition="3D straight-line distance between left and right acromion process landmarks.",
            units="mm",
            value_mm=round(dist_mm, 1) if dist_mm is not None and eval_res.is_valid else None,
            observed_surface_coverage_fraction=1.0 if eval_res.is_valid else 0.0,
            missing_surface_fraction=0.0 if eval_res.is_valid else 1.0,
            source_points_count=2 if eval_res.is_valid else 0,
            plane_or_landmark_definition="acromion_left <-> acromion_right 3D Euclidean distance",
            data_source_type=eval_res.data_source.value,
            quality_status=eval_res.status.value,
            is_valid=eval_res.is_valid,
            confidence_score=eval_res.confidence_score,
            blocking_reason=eval_res.blocking_reason,
            notes=eval_res.notes,
        )

    def _measure_limb_segment(
        self,
        landmarks: Dict[str, AnatomicalLandmark],
        m_id: str,
        m_name: str,
        anat_def: str,
        lm_a_id: str,
        lm_b_id: str,
        expected_range_mm: Tuple[float, float],
    ) -> AnthropometricMeasurementRecord:
        """Measures 3D Euclidean segment length between two limb landmarks."""
        lm_a = landmarks.get(lm_a_id)
        lm_b = landmarks.get(lm_b_id)

        dist_mm = None
        if lm_a and lm_b and lm_a.depth_valid and lm_b.depth_valid and lm_a.point_3d_camera_m and lm_b.point_3d_camera_m:
            p_a = np.array(lm_a.point_3d_camera_m)
            p_b = np.array(lm_b.point_3d_camera_m)
            dist_mm = float(np.linalg.norm(p_a - p_b) * 1000.0)

        eval_res = MeasurementQualityAuditor.evaluate_landmark_distance(
            measurement_id=m_id,
            dist_mm=dist_mm,
            landmark_a=lm_a,
            landmark_b=lm_b,
            expected_range_mm=expected_range_mm,
        )

        has_numeric_val = (dist_mm is not None) and (
            eval_res.status in [MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE, MeasurementQualityStatus.LOW_CONFIDENCE]
        )

        return AnthropometricMeasurementRecord(
            measurement_id=m_id,
            measurement_name=m_name,
            anatomical_definition=anat_def,
            units="mm",
            value_mm=round(dist_mm, 1) if has_numeric_val else None,
            observed_surface_coverage_fraction=1.0 if has_numeric_val else 0.0,
            missing_surface_fraction=0.0 if has_numeric_val else 1.0,
            source_points_count=2 if has_numeric_val else 0,
            plane_or_landmark_definition=f"{lm_a_id} <-> {lm_b_id} 3D Euclidean distance",
            data_source_type=eval_res.data_source.value,
            quality_status=eval_res.status.value,
            is_valid=eval_res.is_valid,
            confidence_score=eval_res.confidence_score,
            blocking_reason=eval_res.blocking_reason,
            notes=eval_res.notes,
        )

    def _slice_trunk_plane(
        self,
        pts_3d: np.ndarray,
        landmarks: Dict[str, AnatomicalLandmark],
        ref_lm_id: str,
        fallback_relative_height: float,
        plane_prefix: str,
        plane_display_name: str,
        anat_def_circ: str,
        anat_def_breadth: str,
    ) -> Tuple[AnthropometricMeasurementRecord, AnthropometricMeasurementRecord, SliceRingAnalysis]:
        """Slices point cloud at specified anatomical trunk level and evaluates breadth and circumference."""
        ref_lm = landmarks.get(ref_lm_id)

        # In RealSense optical frame (+X right, +Y down, +Z forward):
        # Transverse slicing plane has normal pointing vertically: n = [0, 1, 0]
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        if ref_lm and ref_lm.depth_valid and ref_lm.point_3d_camera_m:
            plane_anchor = np.array(ref_lm.point_3d_camera_m, dtype=np.float32)
        elif len(pts_3d) > 50:
            # Fallback based on point cloud Y extent (Y min = top, Y max = bottom in camera optical frame)
            y_min = np.min(pts_3d[:, 1])
            y_max = np.max(pts_3d[:, 1])
            y_anchor = y_min + fallback_relative_height * (y_max - y_min)
            plane_anchor = np.array([np.median(pts_3d[:, 0]), y_anchor, np.median(pts_3d[:, 2])], dtype=np.float32)
        else:
            plane_anchor = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Slice
        slab_pts, slice_analysis = AnatomicalPlaneSlicer.slice_point_cloud(
            points_3d=pts_3d,
            plane_point=plane_anchor,
            plane_normal=plane_normal,
            plane_name=plane_display_name,
            slab_thickness_m=0.025,  # 25 mm slab
        )

        # Evaluate circumference quality
        eval_circ = MeasurementQualityAuditor.evaluate_slice_circumference(
            measurement_id=f"{plane_prefix}_circumference",
            slice_analysis=slice_analysis,
        )

        rec_circ = AnthropometricMeasurementRecord(
            measurement_id=f"{plane_prefix}_circumference",
            measurement_name=f"{plane_display_name} Circumference",
            anatomical_definition=anat_def_circ,
            units="mm",
            value_mm=round(slice_analysis.closed_circumference_mm, 1) if slice_analysis.closed_circumference_mm is not None else None,
            observed_surface_coverage_fraction=slice_analysis.observed_coverage_fraction,
            missing_surface_fraction=slice_analysis.missing_surface_fraction,
            source_points_count=slice_analysis.num_points_in_slab,
            plane_or_landmark_definition=f"Transverse plane Y={plane_anchor[1]:.3f}m (normal: [0, 1, 0], thickness: {slice_analysis.slab_thickness_mm:.0f}mm)",
            data_source_type=eval_circ.data_source.value,
            quality_status=eval_circ.status.value,
            is_valid=eval_circ.is_valid,
            confidence_score=eval_circ.confidence_score,
            blocking_reason=eval_circ.blocking_reason,
            notes=eval_circ.notes,
        )

        # Transverse Breadth is directly observable across the visible front points
        breadth_valid = (slice_analysis.num_points_in_slab >= 30) and (slice_analysis.transverse_breadth_mm > 150.0)
        rec_breadth = AnthropometricMeasurementRecord(
            measurement_id=f"{plane_prefix}_transverse_breadth",
            measurement_name=f"{plane_display_name} Transverse Breadth",
            anatomical_definition=anat_def_breadth,
            units="mm",
            value_mm=round(slice_analysis.transverse_breadth_mm, 1) if breadth_valid else None,
            observed_surface_coverage_fraction=slice_analysis.observed_coverage_fraction,
            missing_surface_fraction=slice_analysis.missing_surface_fraction,
            source_points_count=slice_analysis.num_points_in_slab,
            plane_or_landmark_definition=f"Transverse plane Y={plane_anchor[1]:.3f}m (normal: [0, 1, 0])",
            data_source_type=DataSourceType.DIRECTLY_MEASURED.value if breadth_valid else DataSourceType.BLOCKED.value,
            quality_status=MeasurementQualityStatus.VALID_FOR_EXPERIMENTAL_USE.value if breadth_valid else MeasurementQualityStatus.INSUFFICIENT_DATA.value,
            is_valid=breadth_valid,
            confidence_score=0.85 if breadth_valid else 0.0,
            blocking_reason=None if breadth_valid else "Insufficient points or non-physical breadth.",
            notes=[f"Observed anterior arc length: {slice_analysis.observed_arc_length_mm:.1f} mm"],
        )

        return rec_circ, rec_breadth, slice_analysis

    def _build_subject_cloud(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        person_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Unprojects valid masked depth to 3D point cloud array (metres)."""
        valid_mask = (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m) & (depth_m < 60.0)
        if person_mask is not None:
            valid_mask = valid_mask & person_mask

        v, u = np.where(valid_mask)
        z = depth_m[valid_mask]
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1).astype(np.float32)
