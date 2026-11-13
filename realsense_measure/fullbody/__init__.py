"""
realsense_measure/fullbody

Model-free metric full-body scanning, alignment, fusion, and anthropometry.
Strictly clean commercial licensing (Apache 2.0 / BSD / MIT).
Zero SMPL, SMPL-X, STAR, or Anny dependencies.
"""

from realsense_measure.fullbody.config import FullBodyConfig

def __getattr__(name: str):
    # Lazy import types only when requested to prevent open3d DLL block on lightweight modules
    if name in (
        "StopData",
        "AlignedStop",
        "FloorPlane",
        "Keypoints3D",
        "MeasurementResult",
        "QualityFlag",
        "ScanResult",
    ):
        from realsense_measure.fullbody import types
        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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


