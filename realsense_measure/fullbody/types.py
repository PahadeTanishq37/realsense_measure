"""
realsense_measure/fullbody/types.py

Data structures for the full-body RGB-D scanning pipeline.
Canonical subject frame (ISB convention):
  +X: Subject's left
  +Y: Superior (upwards toward crown)
  +Z: Anterior (forward from chest)
  Origin: Floor plane under the subject's vertical centre-of-mass axis.
Units: Metres (unless explicitly suffixed as _mm or _deg).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import open3d as o3d


class QualityFlag(str, Enum):
    """Measurement quality classification."""
    OK = "OK"
    PARTIAL = "PARTIAL"
    NOT_MEASURABLE = "NOT_MEASURABLE"


@dataclass
class FloorPlane:
    """
    Parametric floor plane equation: a*x + b*y + c*z + d = 0.

    Normal (a, b, c) is normalized to unit length and oriented toward
    the camera / subject superior (+Y in gravity frame).
    """
    coefficients: np.ndarray  # shape (4,), float64 [a, b, c, d]
    normal: np.ndarray        # shape (3,), float64 unit normal
    point_on_plane: np.ndarray  # shape (3,), float64
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rmse_m: float = 0.0
    gravity_R_cam: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))


@dataclass
class Keypoints3D:
    """
    Set of 3D skeletal landmarks.

    All points are expressed in the coordinate frame given by `frame`.
    """
    names: List[str]
    points: np.ndarray      # shape (K, 3), float32/float64 metres
    confidence: np.ndarray  # shape (K,), float32 in [0, 1]
    visibility: np.ndarray  # shape (K,), float32 in [0, 1]
    frame: str = "camera"   # "camera", "gravity", or "canonical"

    def get(self, name: str) -> Tuple[Optional[np.ndarray], float, float]:
        """Return (3D point, confidence, visibility) for named joint or (None, 0.0, 0.0)."""
        if name in self.names:
            idx = self.names.index(name)
            return self.points[idx].copy(), float(self.confidence[idx]), float(self.visibility[idx])
        return None, 0.0, 0.0

    def transform(self, T: np.ndarray, new_frame: str) -> Keypoints3D:
        """Transform all points by 4x4 matrix T and return new Keypoints3D."""
        R = T[:3, :3]
        t = T[:3, 3]
        new_pts = (R @ self.points.T).T + t
        return Keypoints3D(
            names=list(self.names),
            points=new_pts.astype(np.float32),
            confidence=self.confidence.copy(),
            visibility=self.visibility.copy(),
            frame=new_frame,
        )


@dataclass
class StopData:
    """
    Data captured for a single subject rotation stop.
    """
    stop_id: int
    nominal_angle_deg: float
    color: np.ndarray                         # HxWx3 uint8 BGR
    depth: np.ndarray                         # HxW float32 metres
    intrinsics: o3d.camera.PinholeCameraIntrinsic
    depth_std: Optional[np.ndarray] = None    # HxW float32 metres
    valid_counts: Optional[np.ndarray] = None # HxW int32
    timestamp: float = 0.0
    person_mask: Optional[np.ndarray] = None  # HxW bool
    keypoints_camera: Optional[Keypoints3D] = None
    floor_plane: Optional[FloorPlane] = None
    confidence_map: Optional[np.ndarray] = None  # HxW float32 in [0, 1]
    pcd: Optional[o3d.geometry.PointCloud] = None
    raw_burst_colors: List[np.ndarray] = field(default_factory=list)
    raw_burst_depths: List[np.ndarray] = field(default_factory=list)


@dataclass
class AlignedStop:
    """
    Result of aligning a stop into the canonical subject coordinate frame.
    """
    stop_id: int
    nominal_angle_deg: float
    T_canonical_camera: np.ndarray  # 4x4 matrix mapping camera coords -> canonical coords
    prior_yaw_deg: float
    refined_yaw_deg: float
    prior_translation_m: np.ndarray  # shape (3,)
    refined_translation_m: np.ndarray  # shape (3,)
    residual_before_mm: float
    residual_after_mm: float
    inlier_fraction: float
    accepted: bool
    rejection_reason: str = ""
    pcd_canonical: Optional[o3d.geometry.PointCloud] = None
    keypoints_canonical: Optional[Keypoints3D] = None
    confidence_canonical: Optional[np.ndarray] = None
    body_part_labels: Optional[np.ndarray] = None  # per-point integer label


@dataclass
class MeasurementResult:
    """
    Individual anthropometric measurement result.
    """
    name: str
    value_mm: float
    method: str
    coverage: float                   # fraction of contour observed [0, 1]
    interpolated_fraction: float      # fraction interpolated across gaps [0, 1]
    quality_flag: str                 # "OK", "PARTIAL", "NOT_MEASURABLE"
    uncertainty_mm: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value_mm": round(float(self.value_mm), 2) if np.isfinite(self.value_mm) else None,
            "method": self.method,
            "coverage": round(float(self.coverage), 4),
            "interpolated_fraction": round(float(self.interpolated_fraction), 4),
            "quality_flag": self.quality_flag,
            "uncertainty_mm": round(float(self.uncertainty_mm), 2),
            "notes": self.notes,
        }


@dataclass
class ScanResult:
    """
    Complete output of the full-body scan and measurement pipeline.
    """
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = 0.0
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    pcd: Optional[o3d.geometry.PointCloud] = None
    aligned_stops: List[AlignedStop] = field(default_factory=list)
    measurements: Dict[str, MeasurementResult] = field(default_factory=dict)
    landmarks: Dict[str, np.ndarray] = field(default_factory=dict)
    hole_map: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Produce a clean text table of measurement results."""
        lines = [
            f"{'Measurement':<32} | {'Value (mm)':>10} | {'Method':<16} | {'Cov':>5} | {'Quality':<8} | {'Notes'}",
            "-" * 95,
        ]
        for m in self.measurements.values():
            val_str = f"{m.value_mm:10.1f}" if np.isfinite(m.value_mm) else "       NaN"
            cov_str = f"{m.coverage * 100:4.0f}%"
            lines.append(
                f"{m.name:<32} | {val_str} | {m.method:<16} | {cov_str} | {m.quality_flag:<8} | {m.notes}"
            )
        return "\n".join(lines)

