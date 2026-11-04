"""
realsense_measure/body/appearance.py
====================================
Appearance modeling and visual identity layer for reconstructed 3D body models.

Core Principles:
1. Strict Decoupling: Geometry, skeletal topology, and anthropometric measurement
   pipelines remain 100% mathematically and bitwise invariant.
2. Conservative Multi-View Radiance Aggregation: Multi-stop camera projection,
   backface culling (>80 deg rejection), and depth buffer Z-buffer occlusion testing.
3. Explicit Neutral Fallback: Hidden or unobserved regions carry a documented
   neutral color (RGB [128, 128, 128]) and explicit boolean visibility flags.
4. Comprehensive Quality Diagnostics: Per-stop sampled/rejected counts, rejection
   taxonomies, coverage fraction, multi-view consistency standard deviation.
5. Standard Exporters: Stanford PLY with vertex colors and Wavefront OBJ extended
   with `v x y z r g b` vertex colors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame
from .adapter import AdaptedStopData
from .synth import SimulatedStopData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass
class AppearanceQualityDiagnostics:
    """
    Quality diagnostics and rejection metrics for appearance projection and sampling.

    Attributes
    ----------
    total_vertices : int
        Total vertices in the canonical mesh (e.g. 13,718).
    observed_vertices : int
        Count of vertices with at least one valid RGB observation.
    unobserved_vertices : int
        Count of vertices without any valid RGB observations.
    coverage_pct : float
        Percentage of vertices with observed appearance (0.0 - 100.0).
    mean_observations_per_observed_vertex : float
        Average number of contributing stops for observed vertices.
    max_observations_per_vertex : int
        Maximum observations across all stops for any single vertex.
    single_view_observed_count : int
        Number of vertices observed in exactly one stop.
    multi_view_observed_count : int
        Number of vertices observed in 2 or more distinct stops.
    per_stop_sampled_points : Dict[int, int]
        Number of valid surface vertices sampled from each acquisition stop.
    per_stop_rejected_points : Dict[int, int]
        Number of vertices rejected at each acquisition stop.
    rejection_reasons : Dict[str, int]
        Categorical rejection counts: 'backface', 'out_of_bounds', 'background_mask',
        'occluded_zbuffer', 'invalid_depth', 'no_color_image'.
    per_stop_rejection_reasons : Dict[int, Dict[str, int]]
        Per-stop breakdown of rejection counts by category.
    color_consistency_std_rgb : float
        Mean RGB standard deviation across overlapping views for multi-observed vertices.
    mean_color_std_rgb : Optional[Tuple[float, float, float]]
        Mean per-channel standard deviation (R, G, B) across multi-observed vertices.
    status : str
        Quality status: 'success', 'partial', or 'failed'.
    """
    total_vertices: int
    observed_vertices: int
    unobserved_vertices: int
    coverage_pct: float
    mean_observations_per_observed_vertex: float
    max_observations_per_vertex: int
    single_view_observed_count: int = 0
    multi_view_observed_count: int = 0
    per_stop_sampled_points: Dict[int, int] = field(default_factory=dict)
    per_stop_rejected_points: Dict[int, int] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    per_stop_rejection_reasons: Dict[int, Dict[str, int]] = field(default_factory=dict)
    color_consistency_std_rgb: float = 0.0
    mean_color_std_rgb: Optional[Tuple[float, float, float]] = None
    status: str = "success"

    @property
    def observed_vertex_count(self) -> int:
        return self.observed_vertices

    @property
    def unobserved_vertex_count(self) -> int:
        return self.unobserved_vertices

    @property
    def coverage_percentage(self) -> float:
        return self.coverage_pct

    @property
    def coverage_fraction(self) -> float:
        return self.coverage_pct / 100.0 if self.total_vertices > 0 else 0.0

    @property
    def per_stop_sampled_vertices(self) -> Dict[int, int]:
        return self.per_stop_sampled_points

    @property
    def per_stop_rejected_vertices(self) -> Dict[int, int]:
        return self.per_stop_rejected_points

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_vertices": self.total_vertices,
            "observed_vertices": self.observed_vertices,
            "unobserved_vertices": self.unobserved_vertices,
            "coverage_pct": self.coverage_pct,
            "coverage_percentage": self.coverage_pct,
            "coverage_fraction": self.coverage_fraction,
            "single_view_observed_count": self.single_view_observed_count,
            "multi_view_observed_count": self.multi_view_observed_count,
            "mean_observations_per_observed_vertex": self.mean_observations_per_observed_vertex,
            "max_observations_per_vertex": self.max_observations_per_vertex,
            "per_stop_sampled_points": self.per_stop_sampled_points,
            "per_stop_rejected_points": self.per_stop_rejected_points,
            "rejection_reasons": self.rejection_reasons,
            "per_stop": [
                {
                    "stop_idx": s_idx,
                    "sampled": self.per_stop_sampled_points.get(s_idx, 0),
                    "rejected": self.per_stop_rejected_points.get(s_idx, 0),
                    "rejection_reasons": self.per_stop_rejection_reasons.get(s_idx, {}),
                }
                for s_idx in sorted(self.per_stop_sampled_points.keys())
            ],
            "color_consistency_std_rgb": self.color_consistency_std_rgb,
            "mean_color_std_rgb": list(self.mean_color_std_rgb) if self.mean_color_std_rgb else None,
            "status": self.status,
        }


@dataclass
class VertexAppearanceResult:
    """
    Structured outcome of appearance projection and multi-view aggregation.

    Attributes
    ----------
    colors_rgb : np.ndarray
        (V, 3) float32 array in [0.0, 1.0] representing per-vertex RGB colors.
        Unobserved vertices carry a documented neutral gray [128/255, 128/255, 128/255].
    colors_uint8 : np.ndarray
        (V, 3) uint8 array in [0, 255] representing per-vertex RGB colors.
    is_observed_mask : np.ndarray
        (V,) bool array indicating True for vertices with direct RGB observations.
    observation_counts : np.ndarray
        (V,) int32 array tracking how many distinct stops observed each vertex.
    confidence_weights : np.ndarray
        (V,) float32 array tracking cumulative incident cosine weights.
    contributing_stops : List[List[int]]
        List of length V containing stop indices that contributed to each vertex.
    diagnostics : AppearanceQualityDiagnostics
        Summary and granular rejection diagnostics.
    """
    colors_rgb: np.ndarray
    colors_uint8: np.ndarray
    is_observed_mask: np.ndarray
    observation_counts: np.ndarray
    confidence_weights: np.ndarray
    contributing_stops: List[List[int]]
    diagnostics: AppearanceQualityDiagnostics

    @property
    def vertex_colors_rgb(self) -> np.ndarray:
        return self.colors_rgb

    @property
    def vertex_colors_uint8(self) -> np.ndarray:
        return self.colors_uint8

    @property
    def observation_count_per_vertex(self) -> np.ndarray:
        return self.observation_counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "diagnostics": self.diagnostics.as_dict(),
            "observed_vertices": self.diagnostics.observed_vertices,
            "unobserved_vertices": self.diagnostics.unobserved_vertices,
            "coverage_pct": self.diagnostics.coverage_pct,
            "coverage_percentage": self.diagnostics.coverage_pct,
            "coverage_fraction": self.diagnostics.coverage_fraction,
            "is_observed_count": int(np.sum(self.is_observed_mask)),
            "unobserved_count": int(np.sum(~self.is_observed_mask)),
            "mean_confidence": float(np.mean(self.confidence_weights[self.is_observed_mask])) if np.any(self.is_observed_mask) else 0.0,
        }


# ---------------------------------------------------------------------------
# Geometry & Projection Utilities
# ---------------------------------------------------------------------------

def compute_mesh_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex unit normals from triangular faces.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
    faces : np.ndarray, shape (F, 3)

    Returns
    -------
    np.ndarray, shape (V, 3)
        Unit normal vectors.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
    vn = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vn_len = np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8
    return (vn / vn_len).astype(np.float32)


def _parse_intrinsics(intr: Any) -> Tuple[float, float, float, float, int, int]:
    """Extract (fx, fy, cx, cy, width, height) from any intrinsics representation."""
    if hasattr(intr, "get_focal_length") and hasattr(intr, "get_principal_point"):
        fx, fy = intr.get_focal_length()
        cx, cy = intr.get_principal_point()
        w = int(getattr(intr, "width", 640))
        h = int(getattr(intr, "height", 480))
        return float(fx), float(fy), float(cx), float(cy), w, h
    if isinstance(intr, dict):
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        w = int(intr.get("width", 640))
        h = int(intr.get("height", 480))
        return fx, fy, cx, cy, w, h
    if hasattr(intr, "fx") and hasattr(intr, "fy"):
        fx = float(intr.fx)
        fy = float(intr.fy)
        cx = float(intr.cx)
        cy = float(intr.cy)
        w = int(getattr(intr, "width", 640))
        h = int(getattr(intr, "height", 480))
        return fx, fy, cx, cy, w, h
    if hasattr(intr, "intrinsic_matrix"):
        mat = intr.intrinsic_matrix
        w = int(getattr(intr, "width", 640))
        h = int(getattr(intr, "height", 480))
        return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2]), w, h
    raise ValueError(f"Unrecognized intrinsics representation: {type(intr)}")


class CameraProjectionResult:
    """Holds camera projection outcomes and supports tuple unpacking, dictionary indexing, and attributes."""
    def __init__(
        self,
        pts_cam: np.ndarray,
        uv_proj: np.ndarray,
        normals_cam: np.ndarray,
        cos_view_angles: np.ndarray,
        is_in_frame: np.ndarray,
        is_visible: np.ndarray,
    ):
        self.pts_cam = pts_cam
        self.uv_proj = uv_proj
        self.normals_cam = normals_cam
        self.cos_view_angles = cos_view_angles
        self.is_in_frame = is_in_frame
        self.is_visible = is_visible

    def __iter__(self):
        return iter((self.pts_cam, self.uv_proj, self.normals_cam, self.cos_view_angles))

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.pts_cam, self.uv_proj, self.normals_cam, self.cos_view_angles)[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


def project_canonical_to_camera(
    verts_canon: np.ndarray,
    normals_canon: np.ndarray,
    pose: Any,
    distance_m: float,
    intrinsics: Any,
    raw_dict: Optional[dict] = None,
    canonical_joints: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
) -> CameraProjectionResult:
    """
    Forward-project canonical mesh vertices and normals into RealSense camera optical frame.

    Applies articulated arm kinematics, yaw rotation, and camera-to-world rigid transforms.
    """
    verts = np.asarray(verts_canon, dtype=np.float32).copy()
    normals = np.asarray(normals_canon, dtype=np.float32).copy()

    if isinstance(pose, dict):
        yaw_deg = float(pose.get("yaw_deg", 0.0))
        trans = np.asarray(pose.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        abd_L = float(pose.get("arm_abd_L_deg", 0.0))
        abd_R = float(pose.get("arm_abd_R_deg", 0.0))
    else:
        yaw_deg = float(getattr(pose, "yaw_deg", 0.0))
        trans = np.asarray(getattr(pose, "translation_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        abd_L = float(getattr(pose, "arm_abd_L_deg", 0.0))
        abd_R = float(getattr(pose, "arm_abd_R_deg", 0.0))

    # 1. Articulate arms if abduction is non-zero
    if (abs(abd_L) > 1e-3 or abs(abd_R) > 1e-3) and raw_dict and canonical_joints is not None and joint_names is not None:
        v_weights = raw_dict.get("vertex_bone_weights")
        v_indices = raw_dict.get("vertex_bone_indices")
        bl = raw_dict.get("bone_labels")
        if v_weights is not None and v_indices is not None and bl is not None:
            if hasattr(v_weights, "cpu"):
                v_weights = v_weights.cpu().numpy()
            if hasattr(v_indices, "cpu"):
                v_indices = v_indices.cpu().numpy()

            jmap = {name: i for i, name in enumerate(joint_names)}

            for side, abd_deg in [("L", abd_L), ("R", abd_R)]:
                if abs(abd_deg) < 1e-3:
                    continue
                sh_idx = jmap.get(f"upperarm01.{side}", jmap.get(f"UpperArm.{side}", 0))
                pivot = canonical_joints[sh_idx]

                arm_bones = [b for b in bl if f".{side}" in b and any(k in b.lower() for k in [
                    "upperarm", "lowerarm", "wrist", "hand", "finger", "thumb"
                ])]
                b_idxs = set(bl.index(b) for b in arm_bones if b in bl)
                if not b_idxs:
                    continue

                w_side = np.zeros(len(verts), dtype=np.float32)
                for k in range(v_indices.shape[1]):
                    in_b = np.isin(v_indices[:, k], list(b_idxs))
                    w_side += np.where(in_b, v_weights[:, k], 0.0)
                w_side = np.clip(w_side, 0.0, 1.0)[:, None]

                rad = np.radians(abd_deg)
                cos_a = np.cos(rad)
                sin_a = np.sin(rad)

                diff = verts - pivot
                rot_x = diff[:, 0] * cos_a + diff[:, 2] * sin_a
                rot_z = -diff[:, 0] * sin_a + diff[:, 2] * cos_a
                rot_diff = np.stack([rot_x, diff[:, 1], rot_z], axis=-1)
                verts += (rot_diff - diff) * w_side

                n_x = normals[:, 0] * cos_a + normals[:, 2] * sin_a
                n_z = -normals[:, 0] * sin_a + normals[:, 2] * cos_a
                rot_n = np.stack([n_x, normals[:, 1], n_z], axis=-1)
                normals += (rot_n - normals) * w_side
                normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    # 2. Apply yaw rotation around Z and translation
    rad_y = np.radians(yaw_deg)
    cos_y = np.cos(rad_y)
    sin_y = np.sin(rad_y)

    rx = verts[:, 0] * cos_y - verts[:, 1] * sin_y
    ry = verts[:, 0] * sin_y + verts[:, 1] * cos_y
    rz = verts[:, 2]
    pts_world = np.stack([rx, ry, rz], axis=-1) + trans

    nx = normals[:, 0] * cos_y - normals[:, 1] * sin_y
    ny = normals[:, 0] * sin_y + normals[:, 1] * cos_y
    nz = normals[:, 2]
    normals_world = np.stack([nx, ny, nz], axis=-1)

    # 3. Transform world coordinates to camera optical coordinates
    # World: +X Lat, +Y Ant, +Z Up; Camera: +X Right, +Y Down, +Z Fwd
    # Camera optical center is at (0, -distance_m, 0)
    X_c = pts_world[:, 0]
    Y_c = -pts_world[:, 2]
    Z_c = pts_world[:, 1] + float(distance_m)
    pts_cam = np.stack([X_c, Y_c, Z_c], axis=-1)

    N_cx = normals_world[:, 0]
    N_cy = -normals_world[:, 2]
    N_cz = normals_world[:, 1]
    normals_cam = np.stack([N_cx, N_cy, N_cz], axis=-1)
    normals_cam /= (np.linalg.norm(normals_cam, axis=-1, keepdims=True) + 1e-8)

    # 4. Pinhole projection
    fx, fy, cx, cy, width, height = _parse_intrinsics(intrinsics)

    safe_z = np.where(Z_c > 0.05, Z_c, 0.05)
    u = fx * (X_c / safe_z) + cx
    v = fy * (Y_c / safe_z) + cy
    uv_proj = np.stack([u, v], axis=-1)

    # 5. Incident viewing angle
    ray_len = np.linalg.norm(pts_cam, axis=-1, keepdims=True) + 1e-8
    v_ray = pts_cam / ray_len
    cos_view_angles = -np.sum(normals_cam * v_ray, axis=-1)

    is_in_frame = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height) & (Z_c > 0.05)
    is_visible = is_in_frame & (cos_view_angles > 0.0)

    return CameraProjectionResult(
        pts_cam=pts_cam,
        uv_proj=uv_proj,
        normals_cam=normals_cam,
        cos_view_angles=cos_view_angles,
        is_in_frame=is_in_frame,
        is_visible=is_visible,
    )


# ---------------------------------------------------------------------------
# Appearance Modeler
# ---------------------------------------------------------------------------

class AppearanceModeler:
    """
    Conservative appearance reconstruction layer mapping multi-stop RGB observations
    onto canonical reconstructed mesh surfaces.
    """

    def __init__(
        self,
        min_cos_angle: float = 0.1736,   # cos(80 deg): normal pointing towards camera
        depth_margin_m: float = 0.025,   # 25 mm depth consistency tolerance for occlusion
        unobserved_color_rgb: Tuple[float, float, float] = (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0),
    ):
        self.min_cos_angle = float(min_cos_angle)
        self.depth_margin_m = float(depth_margin_m)
        self.unobserved_color_rgb = np.asarray(unobserved_color_rgb, dtype=np.float32)

    def extract_appearance(
        self,
        stops: List[Union[RGBDFrame, AdaptedStopData, SimulatedStopData]],
        canonical_vertices: np.ndarray,
        faces: np.ndarray,
        fit_result: Optional[Any] = None,
        per_stop_poses: Optional[List[Any]] = None,
        distance_m: Optional[Union[float, List[float]]] = None,
        canonical_joints: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        raw_dict: Optional[dict] = None,
    ) -> VertexAppearanceResult:
        """
        Extract vertex colors by projecting canonical mesh geometry into each acquisition stop.
        """
        # Resolve poses
        poses: List[Any] = []
        if per_stop_poses is not None:
            poses = per_stop_poses
        elif fit_result is not None and hasattr(fit_result, "per_stop_poses"):
            poses = fit_result.per_stop_poses
            if distance_m is None and hasattr(fit_result, "distance_m"):
                distance_m = fit_result.distance_m

        V = len(canonical_vertices)
        if V == 0:
            diag_empty = AppearanceQualityDiagnostics(
                total_vertices=0,
                observed_vertices=0,
                unobserved_vertices=0,
                coverage_pct=0.0,
                mean_observations_per_observed_vertex=0.0,
                max_observations_per_vertex=0,
                status="failed",
            )
            return VertexAppearanceResult(
                colors_rgb=np.zeros((0, 3), dtype=np.float32),
                colors_uint8=np.zeros((0, 3), dtype=np.uint8),
                is_observed_mask=np.zeros(0, dtype=bool),
                observation_counts=np.zeros(0, dtype=np.int32),
                confidence_weights=np.zeros(0, dtype=np.float32),
                contributing_stops=[],
                diagnostics=diag_empty,
            )

        normals_canon = compute_mesh_vertex_normals(canonical_vertices, faces)

        color_accum = np.zeros((V, 3), dtype=np.float64)
        weight_accum = np.zeros(V, dtype=np.float64)
        obs_counts = np.zeros(V, dtype=np.int32)
        contributing_stops: List[List[int]] = [[] for _ in range(V)]
        stop_samples_per_vertex: List[List[np.ndarray]] = [[] for _ in range(V)]

        per_stop_sampled: Dict[int, int] = {}
        per_stop_rejected: Dict[int, int] = {}
        per_stop_rejection_reasons: Dict[int, Dict[str, int]] = {}
        rejection_reasons_global: Dict[str, int] = {
            "out_of_frame": 0,
            "backface": 0,
            "background_mask": 0,
            "occluded": 0,
            "invalid_depth": 0,
            "no_color_image": 0,
        }

        # Process each acquisition stop
        for s_idx, stop in enumerate(stops):
            pose = poses[s_idx] if s_idx < len(poses) else {}

            stop_reasons = {
                "out_of_frame": 0,
                "backface": 0,
                "background_mask": 0,
                "occluded": 0,
                "invalid_depth": 0,
                "no_color_image": 0,
            }

            color_bgr = getattr(stop, "color_bgr", None)
            if color_bgr is None:
                c_img = getattr(stop, "color_image", None)
                if c_img is not None:
                    color_bgr = c_img[:, :, [2, 1, 0]] if c_img.shape[-1] == 3 else c_img

            depth_img = getattr(stop, "depth_image", getattr(stop, "depth_m", None))
            mask = getattr(stop, "foreground_mask", None)
            intr = getattr(stop, "intrinsics", None)

            if intr is None:
                logger.warning("Stop %d is missing intrinsics; skipping appearance sampling.", s_idx)
                continue

            # Fallback distance_m
            if distance_m is not None:
                if isinstance(distance_m, (list, tuple, np.ndarray)):
                    dist_m = float(distance_m[s_idx]) if s_idx < len(distance_m) else float(distance_m[0])
                else:
                    dist_m = float(distance_m)
            elif isinstance(intr, dict) and "distance_m" in intr:
                dist_m = float(intr["distance_m"])
            else:
                dist_m = float(getattr(stop, "distance_m", 2.0))

            if color_bgr is None:
                stop_reasons["no_color_image"] += V
                rejection_reasons_global["no_color_image"] += V
                per_stop_sampled[s_idx] = 0
                per_stop_rejected[s_idx] = V
                per_stop_rejection_reasons[s_idx] = stop_reasons
                continue

            H, W = color_bgr.shape[:2]

            pts_cam, uv_proj, normals_cam, cos_view = project_canonical_to_camera(
                verts_canon=canonical_vertices,
                normals_canon=normals_canon,
                pose=pose,
                distance_m=dist_m,
                intrinsics=intr,
                raw_dict=raw_dict,
                canonical_joints=canonical_joints,
                joint_names=joint_names,
            )

            u_coords = uv_proj[:, 0]
            v_coords = uv_proj[:, 1]
            z_cam = pts_cam[:, 2]

            # Bounds check
            in_bounds = (u_coords >= 1.0) & (u_coords < W - 1.0) & (v_coords >= 1.0) & (v_coords < H - 1.0) & (z_cam > 0.10)

            # Backface culling
            front_facing = cos_view > self.min_cos_angle

            u_int = np.clip(np.round(u_coords).astype(np.int32), 0, W - 1)
            v_int = np.clip(np.round(v_coords).astype(np.int32), 0, H - 1)

            # Foreground mask check
            if mask is not None:
                in_foreground = mask[v_int, u_int]
            else:
                in_foreground = np.ones(V, dtype=bool)

            # Z-buffer depth occlusion check
            is_depth_valid = np.zeros(V, dtype=bool)
            not_occluded = np.zeros(V, dtype=bool)

            if depth_img is not None:
                meas_depth = depth_img[v_int, u_int]
                valid_d_mask = np.isfinite(meas_depth) & (meas_depth > 0.10)
                is_depth_valid = valid_d_mask
                diff_depth = z_cam - meas_depth
                not_occluded = valid_d_mask & (diff_depth <= self.depth_margin_m) & (diff_depth >= -self.depth_margin_m * 2.0)
            else:
                is_depth_valid = np.ones(V, dtype=bool)
                not_occluded = np.ones(V, dtype=bool)

            sampled_this_stop = 0
            rejected_this_stop = 0

            for i in range(V):
                if not in_bounds[i]:
                    stop_reasons["out_of_frame"] += 1
                    rejection_reasons_global["out_of_frame"] += 1
                    rejected_this_stop += 1
                elif not front_facing[i]:
                    stop_reasons["backface"] += 1
                    rejection_reasons_global["backface"] += 1
                    rejected_this_stop += 1
                elif not in_foreground[i]:
                    stop_reasons["background_mask"] += 1
                    rejection_reasons_global["background_mask"] += 1
                    rejected_this_stop += 1
                elif depth_img is not None and not is_depth_valid[i]:
                    stop_reasons["invalid_depth"] += 1
                    rejection_reasons_global["invalid_depth"] += 1
                    rejected_this_stop += 1
                elif depth_img is not None and not not_occluded[i]:
                    stop_reasons["occluded"] += 1
                    rejection_reasons_global["occluded"] += 1
                    rejected_this_stop += 1
                else:
                    sampled_this_stop += 1

            per_stop_sampled[s_idx] = sampled_this_stop
            per_stop_rejected[s_idx] = rejected_this_stop
            per_stop_rejection_reasons[s_idx] = stop_reasons

            vis_mask = in_bounds & front_facing & in_foreground & is_depth_valid & not_occluded
            vis_indices = np.where(vis_mask)[0]

            if len(vis_indices) > 0:
                u_vis = u_int[vis_indices]
                v_vis = v_int[vis_indices]
                bgr_samples = color_bgr[v_vis, u_vis].astype(np.float64) / 255.0
                rgb_samples = bgr_samples[:, [2, 1, 0]]

                w_samples = np.clip(cos_view[vis_indices], 0.0, 1.0)

                np.add.at(color_accum, vis_indices, rgb_samples * w_samples[:, None])
                np.add.at(weight_accum, vis_indices, w_samples)
                obs_counts[vis_indices] += 1

                for v_id, rgb_val in zip(vis_indices, rgb_samples):
                    contributing_stops[v_id].append(s_idx)
                    stop_samples_per_vertex[v_id].append(rgb_val)

        # ------------------------------------------------------------------
        # Final Multi-View Radiance Aggregation
        # ------------------------------------------------------------------
        is_observed = weight_accum > 1e-6
        colors_rgb = np.zeros((V, 3), dtype=np.float32)

        colors_rgb[is_observed] = (color_accum[is_observed] / weight_accum[is_observed, None]).astype(np.float32)
        colors_rgb[~is_observed] = self.unobserved_color_rgb
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        colors_uint8 = np.clip(np.round(colors_rgb * 255.0), 0, 255).astype(np.uint8)

        multi_obs_stds: List[np.ndarray] = []
        for i in range(V):
            if len(stop_samples_per_vertex[i]) >= 2:
                samples = np.stack(stop_samples_per_vertex[i], axis=0)  # (K, 3)
                multi_obs_stds.append(np.std(samples, axis=0))

        if multi_obs_stds:
            stds_arr = np.stack(multi_obs_stds, axis=0)  # (M, 3)
            mean_color_std_rgb = (float(np.mean(stds_arr[:, 0])), float(np.mean(stds_arr[:, 1])), float(np.mean(stds_arr[:, 2])))
            mean_color_std = float(np.mean(stds_arr))
        else:
            mean_color_std_rgb = None
            mean_color_std = 0.0

        n_obs = int(np.sum(is_observed))
        cov_pct = float(n_obs / V * 100.0)
        mean_obs = float(np.mean(obs_counts[is_observed])) if n_obs > 0 else 0.0
        max_obs = int(np.max(obs_counts)) if n_obs > 0 else 0
        single_view_count = int(np.sum(obs_counts == 1))
        multi_view_count = int(np.sum(obs_counts >= 2))

        status = "success" if cov_pct >= 25.0 else ("partial" if n_obs > 0 else "failed")

        diagnostics = AppearanceQualityDiagnostics(
            total_vertices=V,
            observed_vertices=n_obs,
            unobserved_vertices=V - n_obs,
            coverage_pct=cov_pct,
            mean_observations_per_observed_vertex=mean_obs,
            max_observations_per_vertex=max_obs,
            single_view_observed_count=single_view_count,
            multi_view_observed_count=multi_view_count,
            per_stop_sampled_points=per_stop_sampled,
            per_stop_rejected_points=per_stop_rejected,
            rejection_reasons=rejection_reasons_global,
            per_stop_rejection_reasons=per_stop_rejection_reasons,
            color_consistency_std_rgb=mean_color_std,
            mean_color_std_rgb=mean_color_std_rgb,
            status=status,
        )

        return VertexAppearanceResult(
            colors_rgb=colors_rgb,
            colors_uint8=colors_uint8,
            is_observed_mask=is_observed,
            observation_counts=obs_counts,
            confidence_weights=weight_accum.astype(np.float32),
            contributing_stops=contributing_stops,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Public Functions & File Exporters
# ---------------------------------------------------------------------------

def extract_vertex_appearance(
    *args,
    stops: Optional[List[Any]] = None,
    canonical_vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    fit_result: Optional[Any] = None,
    per_stop_poses: Optional[List[Any]] = None,
    distance_m: Optional[Union[float, List[float]]] = None,
    canonical_joints: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
    raw_dict: Optional[dict] = None,
    depth_margin_m: float = 0.025,
    min_cos_angle: float = 0.1736,
    **kwargs,
) -> VertexAppearanceResult:
    """
    Convenience function to extract vertex appearance.

    Flexibly accepts arguments in multiple orderings:
    1. extract_vertex_appearance(stops, canonical_vertices, faces, fit_result=...)
    2. extract_vertex_appearance(canonical_vertices, faces, stops, per_stop_poses=...)
    3. Keyword arguments.
    """
    if len(args) >= 1:
        arg0 = args[0]
        if isinstance(arg0, (list, tuple)) and (len(arg0) == 0 or not isinstance(arg0[0], (int, float))):
            stops = arg0
            if len(args) >= 2:
                canonical_vertices = args[1]
            if len(args) >= 3:
                faces = args[2]
            if len(args) >= 4:
                if hasattr(args[3], "per_stop_poses"):
                    fit_result = args[3]
                elif isinstance(args[3], (list, tuple)):
                    per_stop_poses = args[3]
        elif isinstance(arg0, np.ndarray):
            canonical_vertices = arg0
            if len(args) >= 2:
                faces = args[1]
            if len(args) >= 3:
                stops = args[2]
            if len(args) >= 4:
                if hasattr(args[3], "per_stop_poses"):
                    fit_result = args[3]
                elif isinstance(args[3], (list, tuple)):
                    per_stop_poses = args[3]

    if stops is None or canonical_vertices is None or faces is None:
        raise ValueError("extract_vertex_appearance requires 'stops', 'canonical_vertices', and 'faces'.")

    modeler = AppearanceModeler(
        depth_margin_m=depth_margin_m,
        min_cos_angle=min_cos_angle,
    )
    return modeler.extract_appearance(
        stops=stops,
        canonical_vertices=canonical_vertices,
        faces=faces,
        fit_result=fit_result,
        per_stop_poses=per_stop_poses,
        distance_m=distance_m,
        canonical_joints=canonical_joints,
        joint_names=joint_names,
        raw_dict=raw_dict,
    )


def export_vertex_colored_ply(
    filepath: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    colors_uint8: np.ndarray,
) -> Path:
    """
    Export vertex-colored triangular mesh to Stanford PLY format.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    V = len(vertices)
    F = len(faces)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated by Realsense Measure Appearance Layer\n")
        f.write(f"element vertex {V}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {F}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for (vx, vy, vz), (r, g, b) in zip(vertices, colors_uint8):
            f.write(f"{vx:.6f} {vy:.6f} {vz:.6f} {int(r)} {int(g)} {int(b)}\n")

        for f_idx in faces:
            f.write(f"3 {int(f_idx[0])} {int(f_idx[1])} {int(f_idx[2])}\n")

    return path


def export_vertex_colored_obj(
    filepath: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    colors_rgb: np.ndarray,
) -> Path:
    """
    Export vertex-colored triangular mesh to Wavefront OBJ format (`v x y z r g b`).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Wavefront OBJ with Vertex Colors\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        for (vx, vy, vz), (r, g, b) in zip(vertices, colors_rgb):
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return path
