"""
realsense_measure/fullbody/sweet_spot.py

Capture configuration evaluator:
Scores the depth acquisition geometry by integrating information weight (valid points / sigma(z)^2)
across anatomical body parts using the quadratic temporal noise model from diagnostics.
Identifies under-covered anatomical regions (feet, crown, underarms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.parts import BodyPart, label_points_by_body_part
from realsense_measure.fullbody.types import Keypoints3D

logger = logging.getLogger(__name__)


@dataclass
class SweetSpotReport:
    """Summary of capture setup quality score and body part coverage."""
    total_score: float
    per_part_scores: Dict[str, float] = field(default_factory=dict)
    under_covered_parts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def evaluate_capture_configuration(
    pcd: o3d.geometry.PointCloud,
    keypoints: Optional[Keypoints3D],
    cfg: FullBodyConfig,
) -> SweetSpotReport:
    """
    Score depth coverage across body parts using inverse-variance noise weighting:
    Score = sum over points of 1 / sigma(z)^2.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return SweetSpotReport(
            total_score=0.0,
            under_covered_parts=["all"],
            recommendations=["No points in point cloud; check camera connection and distance."],
        )

    # 1. Label points by body part
    if keypoints is not None:
        labels = label_points_by_body_part(pts, keypoints, cfg)
    else:
        # Fallback to height-based vertical slice labeling if keypoints unavailable
        y = pts[:, 1]
        y_min, y_max = float(np.min(y)), float(np.max(y))
        span = max(0.1, y_max - y_min)
        labels = np.zeros(len(pts), dtype=np.int32)
        labels[y > y_min + 0.85 * span] = int(BodyPart.HEAD)
        labels[(y >= y_min + 0.50 * span) & (y <= y_min + 0.85 * span)] = int(BodyPart.TORSO)
        labels[(y >= y_min + 0.25 * span) & (y < y_min + 0.50 * span)] = int(BodyPart.THIGH_LEFT)
        labels[y < y_min + 0.25 * span] = int(BodyPart.SHANK_LEFT)

    # 2. Quadratic noise model: sigma(z) = sigma_base * (z / z_ref)^2
    z = np.maximum(0.2, np.abs(pts[:, 2]))
    sigma_z = cfg.noise_sigma_base_m * np.square(z / cfg.noise_reference_distance_m)
    # Information weight: 1 / sigma(z)^2
    weights = 1.0 / np.maximum(1e-6, np.square(sigma_z))

    total_score = float(np.sum(weights) / 1e6)  # Scale for readable numbers

    # 3. Per-part scores
    per_part_scores: Dict[str, float] = {}
    under_covered: List[str] = []
    recommendations: List[str] = []

    # Nominal expected minimal point thresholds
    thresholds = {
        "HEAD": 1500,
        "TORSO": 4000,
        "PELVIS": 1500,
        "UPPER_ARMS": 1000,
        "FOREARMS": 800,
        "THIGHS": 2500,
        "SHANKS_FEET": 2000,
    }

    # Group counts
    head_cnt = int(np.sum(labels == int(BodyPart.HEAD)))
    torso_cnt = int(np.sum(labels == int(BodyPart.TORSO)))
    pelvis_cnt = int(np.sum(labels == int(BodyPart.PELVIS)))
    arms_cnt = int(np.sum(np.isin(labels, [int(BodyPart.UPPER_ARM_LEFT), int(BodyPart.UPPER_ARM_RIGHT)])))
    forearms_cnt = int(np.sum(np.isin(labels, [int(BodyPart.FOREARM_LEFT), int(BodyPart.FOREARM_RIGHT)])))
    thighs_cnt = int(np.sum(np.isin(labels, [int(BodyPart.THIGH_LEFT), int(BodyPart.THIGH_RIGHT)])))
    shanks_cnt = int(np.sum(np.isin(labels, [int(BodyPart.SHANK_LEFT), int(BodyPart.SHANK_RIGHT)])))

    counts = {
        "HEAD": head_cnt,
        "TORSO": torso_cnt,
        "PELVIS": pelvis_cnt,
        "UPPER_ARMS": arms_cnt,
        "FOREARMS": forearms_cnt,
        "THIGHS": thighs_cnt,
        "SHANKS_FEET": shanks_cnt,
    }

    for name, cnt in counts.items():
        min_expected = thresholds[name]
        part_score = float(np.sum(weights[labels == getattr(BodyPart, name, 0)]) / 1e6) if getattr(BodyPart, name, None) else float(cnt)
        per_part_scores[name] = round(part_score, 2)
        if cnt < min_expected:
            under_covered.append(name)

    # Check feet and crown clipping specifically
    y = pts[:, 1]
    if np.min(y) > 0.08:
        under_covered.append("FEET_CLIPPED")
        recommendations.append("Feet appear clipped: tilt camera downwards slightly or increase camera distance.")
    if np.max(y) < 1.4:
        under_covered.append("CROWN_LOW_COVERAGE")
        recommendations.append("Crown has low coverage: ensure person is centered vertically.")

    return SweetSpotReport(
        total_score=round(total_score, 2),
        per_part_scores=per_part_scores,
        under_covered_parts=under_covered,
        recommendations=recommendations,
    )

