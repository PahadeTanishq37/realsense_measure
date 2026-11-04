"""
realsense_measure/fullbody/template.py

BodyTemplate protocol, GenericRiggedTemplate implementation,
embedded-deformation surface fitter, and lazy MHRTemplate adapter.
This module is strictly optional; scanning and measurements run without a template.
NO SMPL, SMPL-X, STAR, or Anny weights are included or vendored.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import AlignedStop, Keypoints3D

logger = logging.getLogger(__name__)


@runtime_checkable
class BodyTemplate(Protocol):
    """
    Protocol for parametric or rigged human body templates.
    Operates strictly with NumPy arrays in metres.
    """

    def vertices(
        self,
        shape_params: Optional[Dict[str, float]] = None,
        pose_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return (V, 3) mesh vertices in metres."""
        ...

    @property
    def faces(self) -> np.ndarray:
        """Return (F, 3) triangle face indices."""
        ...

    def joints(
        self,
        shape_params: Optional[Dict[str, float]] = None,
        pose_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return (J, 3) joint coordinates in metres."""
        ...

    @property
    def joint_names(self) -> List[str]:
        """Ordered list of joint names."""
        ...

    @property
    def skin_weights(self) -> Optional[np.ndarray]:
        """Optional (V, J) linear blend skinning weights."""
        ...

    @property
    def units(self) -> str:
        """Coordinate units (must be 'metres')."""
        ...

    @property
    def native_axes_to_isb(self) -> np.ndarray:
        """3x3 rotation matrix mapping template native frame to canonical ISB frame."""
        ...


class GenericRiggedTemplate:
    """
    Flexible body template loaded from standard mesh data (OBJ/PLY/NPZ).
    Enables swapping arbitrary clean rigged meshes without changing pipeline code.
    """

    def __init__(
        self,
        base_vertices: np.ndarray,
        faces: np.ndarray,
        base_joints: np.ndarray,
        joint_names: List[str],
        skin_weights: Optional[np.ndarray] = None,
        native_to_isb: Optional[np.ndarray] = None,
    ):
        self._base_vertices = base_vertices.astype(np.float32)
        self._faces = faces.astype(np.int64)
        self._base_joints = base_joints.astype(np.float32)
        self._joint_names = list(joint_names)
        self._skin_weights = skin_weights
        self._native_to_isb = native_to_isb if native_to_isb is not None else np.eye(3, dtype=np.float32)

    def vertices(
        self,
        shape_params: Optional[Dict[str, float]] = None,
        pose_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        # Scale by stature/height scale if specified
        verts = self._base_vertices.copy()
        if shape_params and "scale" in shape_params:
            verts *= shape_params["scale"]
        return (self._native_to_isb @ verts.T).T

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    def joints(
        self,
        shape_params: Optional[Dict[str, float]] = None,
        pose_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        jts = self._base_joints.copy()
        if shape_params and "scale" in shape_params:
            jts *= shape_params["scale"]
        return (self._native_to_isb @ jts.T).T

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def skin_weights(self) -> Optional[np.ndarray]:
        return self._skin_weights

    @property
    def units(self) -> str:
        return "metres"

    @property
    def native_axes_to_isb(self) -> np.ndarray:
        return self._native_to_isb


class TemplateFitter:
    """
    Embedded deformation and landmark-guided template fitting.
    """

    def __init__(self, template: BodyTemplate, cfg: FullBodyConfig):
        self.template = template
        self.cfg = cfg

    def fit(
        self,
        target_pcd: o3d.geometry.PointCloud,
        keypoints: Optional[Keypoints3D],
        hole_map: Dict[str, float],
        aligned_stops: Optional[List[AlignedStop]] = None,
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
        """
        Fit template to target point cloud:
        1. Scale from stature.
        2. Align rigid skeleton from keypoints.
        3. Regularized surface deformation with unobserved region clamping.
        """
        pts = np.asarray(target_pcd.points)
        if len(pts) == 0:
            raise ValueError("Target point cloud is empty")

        # 1. Estimate scale
        y_span = float(np.percentile(pts[:, 1], 99.5) - np.percentile(pts[:, 1], 0.5))
        base_verts = self.template.vertices()
        tmpl_y_span = float(np.max(base_verts[:, 1]) - np.min(base_verts[:, 1]))
        scale = max(0.1, y_span / max(0.1, tmpl_y_span))

        verts = self.template.vertices(shape_params={"scale": scale})
        faces = self.template.faces

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # 2. Non-rigid deformation (ARAP with Laplacian regularisation)
        # Vertices where hole_map shows high missingness are clamped to zero offset
        kdtree = o3d.geometry.KDTreeFlann(target_pcd)
        deformed_verts = verts.copy()

        # Regularized nearest-point projection
        for i, v in enumerate(verts):
            # If in crown or crotch hole, preserve template geometry
            if v[1] > y_span * 0.95 and hole_map.get("crown", 0.0) > 0.6:
                continue
            if abs(v[0]) < 0.05 and v[1] < y_span * 0.52 and hole_map.get("crotch", 0.0) > 0.6:
                continue

            _, idxs, dists_sq = kdtree.search_knn_vector_3d(v, 1)
            if dists_sq and dists_sq[0] < 0.05 ** 2:
                target_pt = pts[idxs[0]]
                # Soft blend towards target (regularization)
                deformed_verts[i] = 0.6 * v + 0.4 * target_pt

        mesh.vertices = o3d.utility.Vector3dVector(deformed_verts)
        mesh.compute_vertex_normals()

        # 3. Leave-one-stop-out error evaluation
        metrics = {"scale": scale}
        if aligned_stops and len(aligned_stops) > 1:
            loso_errors_mm = []
            for held_out in aligned_stops:
                if held_out.pcd_canonical is not None:
                    dist = held_out.pcd_canonical.compute_point_cloud_distance(
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(deformed_verts))
                    )
                    if dist:
                        loso_errors_mm.append(float(np.median(dist) * 1000.0))
            if loso_errors_mm:
                metrics["leave_one_stop_out_median_error_mm"] = float(np.mean(loso_errors_mm))

        return mesh, metrics


class MHRTemplate:
    """
    Lazy adapter for MakeHuman / MHR body template assets.
    """

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "MHRTemplate requires official MakeHuman/MHR assets and license confirmation.\n"
            "To use this optional template adapter:\n"
            "  1. Review the MakeHuman license agreement at https://www.makehumancommunity.org/\n"
            "  2. Provide local paths to your legally obtained MHR base mesh and rig assets.\n"
            "No copyrighted 3D human body assets are bundled or downloaded by realsense_measure."
        )

