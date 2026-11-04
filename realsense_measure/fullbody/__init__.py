"""
realsense_measure/fullbody

Model-free metric full-body scanning, alignment, fusion, and anthropometry.
Strictly clean commercial licensing (Apache 2.0 / BSD / MIT).
Zero SMPL, SMPL-X, STAR, or Anny dependencies.
"""

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import (
    AlignedStop,
    FloorPlane,
    Keypoints3D,
    MeasurementResult,
    QualityFlag,
    ScanResult,
    StopData,
)

__all__ = [
    "FullBodyConfig",
    "StopData",
    "AlignedStop",
    "FloorPlane",
    "Keypoints3D",
    "MeasurementResult",
    "QualityFlag",
    "ScanResult",
]

