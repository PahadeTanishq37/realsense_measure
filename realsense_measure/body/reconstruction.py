"""
realsense_measure/body/reconstruction.py
=======================================
Multi-stop body reconstruction workflow connecting:
1. RGB-D stop adaptation (frame_to_body_stop)
2. Joint phenotype fitting via DifferentiableBodyFitter
3. Canonical unposing via BodyUnposer
4. Observation consolidation & provenance tracking
5. Anatomically constrained personalized surface fitting via PersonalisedSurfaceFitter
6. Dual mesh anthropometry (model baseline vs. personalized) via measure_mesh
7. Observed-ring diagnostics via ObservedRingExtractor
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from camera.frame import RGBDFrame
from .adapter import AdaptedStopData, frame_to_body_stop
from .appearance import (
    VertexAppearanceResult,
    export_vertex_colored_obj,
    export_vertex_colored_ply,
    extract_vertex_appearance,
)
from .anny_backend import AnnyBackend
from .fit import DifferentiableBodyFitter, FitResult
from .measure import Measurement, is_contour_closed, measure_mesh
from .observed_measure import ObservedRingExtractor, ObservedRingResult
from .personalized_surface import (
    ANATOMICAL_REGIONS,
    PersonalisedSurfaceFitter,
    PersonalisedSurfaceResult,
    _compute_vertex_regions,
)
from .prior import AdultShapePrior
from .synth import SimulatedStopData
from .unpose import BodyUnposer, UnposedStopData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts: Multi-Stop Fit
# ---------------------------------------------------------------------------

@dataclass
class MultiStopFitResult:
    """
    Structured outcome of the multi-stop parametric fitting pipeline.

    Attributes
    ----------
    fitted_phenotypes : Dict[str, float]
        Shared phenotype values in [0, 1] (gender, age, muscle, weight, height, proportions).
    measurements : Dict[str, Measurement]
        Anthropometric measurements computed on the canonical fitted mesh.
    per_stop_poses : List[Dict[str, Any]]
        Per-stop optimized poses (yaw_deg, translation_m, arm_abd_L_deg, arm_abd_R_deg).
    unposed_stops : List[UnposedStopData]
        Per-stop unposed observations, surface-to-model residuals, and canonical correspondence.
    canonical_mesh : Dict[str, Any]
        Fitted canonical mesh containing 'vertices' (V, 3), 'faces' (F, 3),
        'joints' (J, 3), 'joint_names', and 'raw' metadata.
    fit_result : FitResult
        The low-level fitting outcome from DifferentiableBodyFitter.
    adapted_stops : List[Union[AdaptedStopData, SimulatedStopData]]
        The adapted or simulated stop data structures evaluated during fitting.
    diagnostics : Dict[str, Any]
        Optimization and alignment diagnostics (loss breakdown, iterations, elapsed_sec, per-stop residuals).
    warnings : List[str]
        Non-fatal warnings recorded during ingestion, fitting, or unposing.
    is_valid : bool
        Overall validity flag reflecting convergence and non-empty outputs.
    """
    fitted_phenotypes: Dict[str, float]
    measurements: Dict[str, Measurement]
    per_stop_poses: List[Dict[str, Any]]
    unposed_stops: List[UnposedStopData]
    canonical_mesh: Dict[str, Any]
    fit_result: FitResult
    adapted_stops: List[Union[AdaptedStopData, SimulatedStopData]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True
    appearance: Optional[VertexAppearanceResult] = None

    @property
    def distance_m(self) -> List[float]:
        """Distance in metres for each adapted stop."""
        dists = []
        for st in self.adapted_stops:
            intr = getattr(st, "intrinsics", {})
            d = intr.get("distance_m", getattr(st, "distance_m", 2.0)) if isinstance(intr, dict) else 2.0
            dists.append(float(d))
        return dists

    @property
    def all_unposed_points(self) -> np.ndarray:
        """Concatenated unposed points from all stops in canonical frame (M, 3)."""
        pts = [s.unposed_points for s in self.unposed_stops if len(s.unposed_points) > 0]
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(pts)

    @property
    def all_residuals_mm(self) -> np.ndarray:
        """Concatenated point-to-model distance residuals in mm across all stops."""
        res = [s.point_to_model_distances * 1000.0 for s in self.unposed_stops if len(s.point_to_model_distances) > 0]
        if not res:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(res)

    @property
    def mean_residual_mm(self) -> float:
        """Overall mean point-to-model distance residual in mm across all unposed points."""
        arr = self.all_residuals_mm
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    @property
    def median_residual_mm(self) -> float:
        """Overall median point-to-model distance residual in mm across all unposed points."""
        arr = self.all_residuals_mm
        return float(np.median(arr)) if len(arr) > 0 else float("nan")

    @property
    def p90_residual_mm(self) -> float:
        """Overall 90th percentile point-to-model distance residual in mm across all unposed points."""
        arr = self.all_residuals_mm
        return float(np.percentile(arr, 90)) if len(arr) > 0 else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        """Convert summary results to a serializable dictionary."""
        return {
            "fitted_phenotypes": self.fitted_phenotypes,
            "per_stop_poses": self.per_stop_poses,
            "num_stops": len(self.unposed_stops),
            "total_unposed_points": int(len(self.all_unposed_points)),
            "mean_residual_mm": self.mean_residual_mm,
            "median_residual_mm": self.median_residual_mm,
            "p90_residual_mm": self.p90_residual_mm,
            "final_loss": self.fit_result.final_loss,
            "converged": self.fit_result.converged,
            "iterations": self.fit_result.iterations,
            "elapsed_sec": self.fit_result.elapsed_sec,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Data Contracts: Measurement Comparison & Personalized Result
# ---------------------------------------------------------------------------

@dataclass
class MeasurementComparison:
    """
    Comparison between fitted parametric model measurement and personalized mesh measurement.

    Attributes
    ----------
    name : str
        Measurement identifier (e.g. 'chest_circumference').
    model_value_mm : float
        Measurement on the fitted canonical template mesh in mm.
    personalized_value_mm : float
        Measurement on the personalized surface mesh in mm.
    absolute_change_mm : float
        Personalized value minus model value (mm).
    relative_change_pct : float
        Percentage change relative to model baseline (%).
    method : str
        Slicing / contour calculation method used.
    source : str
        Measurement provenance flag ('model', 'personalized_mesh', etc.).
    notes : str
        Anatomical landmarks and notes.
    """
    name: str
    model_value_mm: float
    personalized_value_mm: float
    absolute_change_mm: float
    relative_change_pct: float
    method: str
    source: str
    notes: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_value_mm": self.model_value_mm,
            "personalized_value_mm": self.personalized_value_mm,
            "absolute_change_mm": self.absolute_change_mm,
            "relative_change_pct": self.relative_change_pct,
            "method": self.method,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass
class PersonalizedReconstructionResult:
    """
    Structured outcome of the end-to-end multi-stop personalized reconstruction pipeline.

    Attributes
    ----------
    multi_stop_fit : MultiStopFitResult
        Parametric joint fitting and unposing outcome.
    personalized_surface : PersonalisedSurfaceResult
        Personalised surface fitting outcome including optimal lambda, LOO residuals, and diagnostics.
    canonical_fitted_mesh : Dict[str, Any]
        Base fitted canonical mesh (unmodified template with fitted phenotypes).
    personalized_mesh : Dict[str, Any]
        Personalized canonical mesh after anatomically constrained Laplacian deformation.
    model_measurements : Dict[str, Measurement]
        Measurements on the fitted canonical mesh.
    personalized_measurements : Dict[str, Measurement]
        Measurements on the personalized surface mesh.
    measurement_comparisons : Dict[str, MeasurementComparison]
        Comparison delta metrics between model and personalized measurements.
    observed_rings : Dict[str, ObservedRingResult]
        Direct observed point-cloud ring circumferences with coverage and open-contour flags.
    diagnostics : Dict[str, Any]
        Comprehensive quality diagnostics and convergence metrics.
    status : str
        Reconstruction quality status: 'success', 'partial', or 'failed'.
    warnings : List[str]
        Warnings accumulated during fitting, unposing, and personalization.
    is_valid : bool
        Overall validity flag reflecting finiteness, topological integrity, and non-empty outputs.
    """
    multi_stop_fit: MultiStopFitResult
    personalized_surface: PersonalisedSurfaceResult
    canonical_fitted_mesh: Dict[str, Any]
    personalized_mesh: Dict[str, Any]
    model_measurements: Dict[str, Measurement]
    personalized_measurements: Dict[str, Measurement]
    measurement_comparisons: Dict[str, MeasurementComparison]
    observed_rings: Dict[str, ObservedRingResult]
    diagnostics: Dict[str, Any]
    status: str
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True
    appearance: Optional[VertexAppearanceResult] = None

    @property
    def displacement_magnitudes_mm(self) -> np.ndarray:
        """Euclidean displacement norm in mm for each vertex between model and personalized mesh."""
        v_fit = self.canonical_fitted_mesh["vertices"]
        v_pers = self.personalized_mesh["vertices"]
        return np.linalg.norm(v_pers - v_fit, axis=-1) * 1000.0

    @property
    def mean_displacement_mm(self) -> float:
        """Mean per-vertex displacement magnitude in mm."""
        arr = self.displacement_magnitudes_mm
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    @property
    def median_displacement_mm(self) -> float:
        """Median per-vertex displacement magnitude in mm."""
        arr = self.displacement_magnitudes_mm
        return float(np.median(arr)) if len(arr) > 0 else float("nan")

    @property
    def p90_displacement_mm(self) -> float:
        """90th percentile per-vertex displacement magnitude in mm."""
        arr = self.displacement_magnitudes_mm
        return float(np.percentile(arr, 90)) if len(arr) > 0 else float("nan")

    @property
    def max_displacement_mm(self) -> float:
        """Maximum per-vertex displacement magnitude in mm."""
        arr = self.displacement_magnitudes_mm
        return float(np.max(arr)) if len(arr) > 0 else float("nan")

    def export_obj(
        self,
        filepath: Union[str, Path],
        mesh_type: str = "personalized",
        include_vertex_colors: bool = False,
    ) -> Path:
        """
        Export canonical or personalized mesh as Wavefront OBJ.

        Parameters
        ----------
        filepath : Union[str, Path]
            Target .obj filepath.
        mesh_type : str
            "personalized" (default) or "fitted".
        include_vertex_colors : bool
            If True and appearance is available, write vertex colors as
            `v x y z r g b` (rgb in [0, 1]).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = self.personalized_mesh if mesh_type == "personalized" else self.canonical_fitted_mesh
        verts = mesh["vertices"]
        faces = mesh["faces"]

        if include_vertex_colors and self.appearance is not None:
            return export_vertex_colored_obj(path, verts, faces, self.appearance.vertex_colors_rgb)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Anny Canonical {mesh_type.capitalize()} Mesh\n")
            f.write(f"# Vertices: {len(verts)}, Faces: {len(faces)}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        return path

    def export_ply(
        self,
        filepath: Union[str, Path],
        mesh_type: str = "personalized",
    ) -> Path:
        """
        Export canonical or personalized mesh as Stanford PLY with vertex colors.

        If appearance data is available, vertex colors are included as uchar (0-255).
        Otherwise neutral grey (128, 128, 128) is written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = self.personalized_mesh if mesh_type == "personalized" else self.canonical_fitted_mesh
        verts = mesh["vertices"]
        faces = mesh["faces"]

        if self.appearance is not None:
            colors = self.appearance.vertex_colors_uint8
        else:
            colors = np.full((len(verts), 3), 128, dtype=np.uint8)

        return export_vertex_colored_ply(path, verts, faces, colors)

    def export_json(self, filepath: Union[str, Path]) -> Path:
        """Export comprehensive reconstruction metadata and anthropometry to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)
        return path

    def as_dict(self) -> Dict[str, Any]:
        """Convert complete reconstruction result to a JSON-serializable dictionary."""
        d = {
            "status": self.status,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "fitted_phenotypes": self.multi_stop_fit.fitted_phenotypes,
            "displacement_summary_mm": {
                "mean": self.mean_displacement_mm,
                "median": self.median_displacement_mm,
                "p90": self.p90_displacement_mm,
                "max": self.max_displacement_mm,
            },
            "selected_lambda": self.personalized_surface.selected_lambda,
            "vertex_coverage_fraction": self.personalized_surface.vertex_coverage_fraction,
            "model_measurements": {
                k: {
                    "name": m.name,
                    "value_mm": m.value_mm,
                    "method": m.method,
                    "source": m.source,
                }
                for k, m in self.model_measurements.items()
            },
            "personalized_measurements": {
                k: {
                    "name": m.name,
                    "value_mm": m.value_mm,
                    "method": m.method,
                    "source": m.source,
                }
                for k, m in self.personalized_measurements.items()
            },
            "measurement_comparisons": {
                k: c.as_dict() for k, c in self.measurement_comparisons.items()
            },
            "observed_rings": {
                k: {
                    "name": r.name,
                    "value_mm": r.value_mm,
                    "coverage_fraction": r.coverage_fraction,
                    "source": r.source,
                    "method": r.method,
                    "num_points_in_slab": r.num_points_in_slab,
                    "is_valid": r.is_valid(),
                }
                for k, r in self.observed_rings.items()
            },
            "diagnostics": self.diagnostics,
        }
        if self.appearance is not None:
            d["appearance"] = self.appearance.as_dict()
        return d


# ---------------------------------------------------------------------------
# Session Validation
# ---------------------------------------------------------------------------

def validate_multi_stop_session(
    stops: List[Union[AdaptedStopData, SimulatedStopData]],
    raise_on_empty: bool = True,
) -> List[str]:
    """
    Validate multi-stop session consistency and contract compliance.
    """
    if not stops or len(stops) == 0:
        raise ValueError("Stop list cannot be empty.")

    warnings: List[str] = []

    # 1. Stop index uniqueness
    stop_indices = [s.stop_idx for s in stops]
    if len(stop_indices) != len(set(stop_indices)):
        raise ValueError(f"Duplicate stop indices detected in stops list: {stop_indices}")

    # 2. Reference dimensions & intrinsics from stop 0
    ref_s = stops[0]
    if not hasattr(ref_s, "intrinsics") or ref_s.intrinsics is None:
        raise ValueError("Stop 0 is missing camera intrinsics dictionary.")

    ref_w = int(ref_s.intrinsics.get("width", 0))
    ref_h = int(ref_s.intrinsics.get("height", 0))
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError(f"Stop 0 has invalid dimensions in intrinsics: width={ref_w}, height={ref_h}")

    # 3. Check consistency across all stops
    for idx, s in enumerate(stops):
        # Nominal yaw validity
        yaw = getattr(s, "nominal_yaw_deg", None)
        if yaw is None or not np.isfinite(yaw):
            raise ValueError(f"Invalid nominal yaw at stop {idx}: {yaw}. Must be a finite number.")

        # Intrinsics validity
        intr = getattr(s, "intrinsics", None)
        if intr is None:
            raise ValueError(f"Stop {idx} is missing camera intrinsics dictionary.")

        fx = float(intr.get("fx", 0.0))
        fy = float(intr.get("fy", 0.0))
        dist_m = float(intr.get("distance_m", 0.0))
        w = int(intr.get("width", 0))
        h = int(intr.get("height", 0))

        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"Invalid focal lengths at stop {idx}: fx={fx}, fy={fy}. Must be positive.")
        if dist_m <= 0.10 or not np.isfinite(dist_m):
            raise ValueError(f"Invalid distance_m at stop {idx}: {dist_m}. Must be finite and >= 0.10 m.")

        if (w, h) != (ref_w, ref_h):
            raise ValueError(
                f"Mismatched image dimensions across stops: stop 0 is ({ref_w}x{ref_h}), "
                f"stop {idx} is ({w}x{h})."
            )

        # Depth image shape consistency if present
        if hasattr(s, "depth_image") and s.depth_image is not None:
            dh, dw = s.depth_image.shape[:2]
            if (dw, dh) != (w, h):
                raise ValueError(
                    f"Stop {idx} depth_image shape ({dw}x{dh}) does not match intrinsics ({w}x{h})."
                )

        # Point cloud and mask emptiness
        n_pts = len(s.point_cloud_world) if hasattr(s, "point_cloud_world") and s.point_cloud_world is not None else 0
        n_mask = int(np.sum(s.foreground_mask)) if hasattr(s, "foreground_mask") and s.foreground_mask is not None else 0

        if n_pts == 0 or n_mask == 0:
            msg = f"Stop {idx} (stop_idx={s.stop_idx}) has empty points (N={n_pts}) or foreground mask (pixels={n_mask})."
            if raise_on_empty:
                raise ValueError(msg)
            else:
                warnings.append(msg)
                logger.warning(msg)

    return warnings


# ---------------------------------------------------------------------------
# Multi-Stop Orchestrator
# ---------------------------------------------------------------------------

class MultiStopOrchestrator:
    """
    Orchestrates end-to-end multi-stop body reconstruction:
    RGB-D Adaptation -> Validation -> Joint Anny Fitting -> Canonical Unposing
    -> Anatomically Constrained Personalization -> Dual Anthropometry -> Quality Diagnostics.
    """

    def __init__(
        self,
        backend: Optional[AnnyBackend] = None,
        prior: Optional[AdultShapePrior] = None,
        fitter: Optional[DifferentiableBodyFitter] = None,
        surface_fitter: Optional[PersonalisedSurfaceFitter] = None,
    ):
        self.backend = backend or AnnyBackend()
        self.prior = prior or AdultShapePrior(self.backend._model)
        self.fitter = fitter or DifferentiableBodyFitter(self.backend, self.prior)
        self.surface_fitter = surface_fitter

    def run(
        self,
        stops: Union[List[RGBDFrame], List[AdaptedStopData], List[SimulatedStopData]],
        nominal_yaws_deg: Optional[List[float]] = None,
        distance_m: Optional[Union[float, List[float]]] = None,
        known_height_mm: Optional[float] = None,
        use_floor_constraint: bool = False,
        floor_z: Optional[float] = None,
        n_coarse_iter: int = 40,
        n_fine_iter: int = 30,
        multi_starts: int = 3,
        seed: int = 42,
        ablation_remove_term: Optional[str] = None,
        raise_on_empty: bool = True,
    ) -> MultiStopFitResult:
        """
        Execute multi-stop fitting and canonical unposing.
        """
        if not stops or len(stops) == 0:
            raise ValueError("Stop list cannot be empty.")

        n_stops = len(stops)
        warnings_list: List[str] = []

        if nominal_yaws_deg is not None:
            if len(nominal_yaws_deg) != n_stops:
                raise ValueError(
                    f"nominal_yaws_deg length ({len(nominal_yaws_deg)}) does not match stops count ({n_stops})."
                )

        if isinstance(distance_m, (list, tuple)):
            if len(distance_m) != n_stops:
                raise ValueError(
                    f"distance_m list length ({len(distance_m)}) does not match stops count ({n_stops})."
                )

        # 1. Adapt RGBDFrames if necessary
        adapted_stops: List[Union[AdaptedStopData, SimulatedStopData]] = []
        for i, item in enumerate(stops):
            if isinstance(item, RGBDFrame):
                if nominal_yaws_deg is not None:
                    yaw = float(nominal_yaws_deg[i])
                elif hasattr(item, "nominal_yaw_deg"):
                    yaw = float(getattr(item, "nominal_yaw_deg"))
                else:
                    yaw = float(i * (360.0 / n_stops))

                if distance_m is not None:
                    if isinstance(distance_m, (list, tuple)):
                        dist = float(distance_m[i])
                    else:
                        dist = float(distance_m)
                elif hasattr(item, "distance_m"):
                    dist = float(getattr(item, "distance_m"))
                else:
                    depth = item.depth_m
                    mask = item.foreground_mask if getattr(item, "foreground_mask", None) is not None else None
                    if mask is None:
                        from .adapter import get_or_create_foreground_mask
                        mask = get_or_create_foreground_mask(item.color_bgr, depth)
                    valid_d = depth[mask & np.isfinite(depth) & (depth > 0.1)]
                    if len(valid_d) > 0:
                        dist = float(np.median(valid_d))
                        w_msg = f"distance_m not specified for stop {i}; inferred subject median depth {dist:.3f} m."
                        warnings_list.append(w_msg)
                        logger.info(w_msg)
                    else:
                        raise ValueError(f"distance_m not provided and could not be inferred for stop {i}.")

                adapted = frame_to_body_stop(
                    frame=item,
                    nominal_yaw_deg=yaw,
                    distance_m=dist,
                    stop_idx=i,
                    raise_on_empty=raise_on_empty,
                )
                adapted_stops.append(adapted)
            elif isinstance(item, (AdaptedStopData, SimulatedStopData)):
                if nominal_yaws_deg is not None:
                    item.nominal_yaw_deg = float(nominal_yaws_deg[i])
                adapted_stops.append(item)
            else:
                raise TypeError(f"Unsupported stop type at index {i}: {type(item)}")

        # 2. Validate session compatibility
        session_warnings = validate_multi_stop_session(adapted_stops, raise_on_empty=raise_on_empty)
        warnings_list.extend(session_warnings)

        # 3. Fit Anny body model jointly across all stops
        logger.info("Executing joint multi-stop fitting across %d stops...", len(adapted_stops))
        fit_result: FitResult = self.fitter.fit(
            stops=adapted_stops,
            known_height_mm=known_height_mm,
            use_floor_constraint=use_floor_constraint,
            floor_z=floor_z,
            n_coarse_iter=n_coarse_iter,
            n_fine_iter=n_fine_iter,
            multi_starts=multi_starts,
            seed=seed,
            ablation_remove_term=ablation_remove_term,
        )

        # 4. Generate canonical rest mesh from fitted phenotypes
        fitted_output = self.backend.rest_pose(phenotypes=fit_result.fitted_phenotypes)
        canon_verts = fitted_output.numpy_vertices(0)
        canon_joints = fitted_output.numpy_joints(0)
        canon_faces = fitted_output.faces.cpu().numpy()
        joint_names = list(fitted_output.joint_names)
        raw_dict = fitted_output.raw if hasattr(fitted_output, "raw") else {}

        # Compute full anthropometric measurements on canonical rest mesh
        measurements = measure_mesh(
            canon_verts,
            canon_joints,
            canon_faces,
            joint_names,
            raw_dict,
        )

        # Package canonical mesh dictionary
        canonical_mesh = {
            "vertices": canon_verts,
            "joints": canon_joints,
            "faces": canon_faces,
            "joint_names": joint_names,
            "raw": raw_dict,
        }

        # 5. Canonical unposing of all stops
        logger.info("Unposing %d stops into canonical A-pose frame...", len(adapted_stops))
        unposer = BodyUnposer(
            canon_vertices=canon_verts,
            canon_joints=canon_joints,
            joint_names=joint_names,
            raw_dict=raw_dict,
        )
        unposed_stops: List[UnposedStopData] = unposer.unpose_all_stops(
            stops=adapted_stops,
            per_stop_poses=fit_result.per_stop_poses,
        )

        # 6. Collate diagnostics
        diagnostics = {
            "final_loss": fit_result.final_loss,
            "loss_breakdown": fit_result.loss_breakdown.as_dict(),
            "converged": fit_result.converged,
            "iterations": fit_result.iterations,
            "elapsed_sec": fit_result.elapsed_sec,
            "num_stops": len(unposed_stops),
            "per_stop_residuals": [
                {
                    "stop_idx": u.stop_idx,
                    "num_points": len(u.unposed_points),
                    "mean_residual_mm": u.mean_residual_mm,
                    "median_residual_mm": u.median_residual_mm,
                    "p90_residual_mm": u.p90_residual_mm,
                }
                for u in unposed_stops
            ],
        }

        is_valid = fit_result.converged and all(np.isfinite(p) for p in fit_result.fitted_phenotypes.values())

        return MultiStopFitResult(
            fitted_phenotypes=fit_result.fitted_phenotypes,
            measurements=measurements,
            per_stop_poses=fit_result.per_stop_poses,
            unposed_stops=unposed_stops,
            canonical_mesh=canonical_mesh,
            fit_result=fit_result,
            adapted_stops=adapted_stops,
            diagnostics=diagnostics,
            warnings=warnings_list,
            is_valid=is_valid,
        )

    def reconstruct_and_personalize(
        self,
        stops: Union[List[RGBDFrame], List[AdaptedStopData], List[SimulatedStopData]],
        nominal_yaws_deg: Optional[List[float]] = None,
        distance_m: Optional[Union[float, List[float]]] = None,
        known_height_mm: Optional[float] = None,
        use_floor_constraint: bool = False,
        floor_z: Optional[float] = None,
        n_coarse_iter: int = 40,
        n_fine_iter: int = 30,
        multi_starts: int = 3,
        seed: int = 42,
        ablation_remove_term: Optional[str] = None,
        max_correspondence_dist_m: float = 0.040,
        candidate_lambdas: Optional[List[float]] = None,
        run_observed_rings: bool = True,
        extract_appearance: bool = True,
        appearance_depth_margin_m: float = 0.025,
        appearance_min_cos_angle: float = 0.1736,
        raise_on_empty: bool = True,
    ) -> PersonalizedReconstructionResult:
        """
        Execute complete multi-stop reconstruction:
        Parametric fitting -> Canonical unposing -> Anatomically constrained surface personalization
        -> Dual anthropometric measurement -> Observed-ring extraction -> Quality diagnostics.
        """
        t0 = time.time()
        warnings_list: List[str] = []

        # 1. Run multi-stop parametric fitting and unposing
        fit_res = self.run(
            stops=stops,
            nominal_yaws_deg=nominal_yaws_deg,
            distance_m=distance_m,
            known_height_mm=known_height_mm,
            use_floor_constraint=use_floor_constraint,
            floor_z=floor_z,
            n_coarse_iter=n_coarse_iter,
            n_fine_iter=n_fine_iter,
            multi_starts=multi_starts,
            seed=seed,
            ablation_remove_term=ablation_remove_term,
            raise_on_empty=raise_on_empty,
        )
        warnings_list.extend(fit_res.warnings)

        # 2. Extract and strictly protect base canonical mesh (deep copy)
        base_verts = np.copy(fit_res.canonical_mesh["vertices"]).astype(np.float64)
        faces = np.copy(fit_res.canonical_mesh["faces"]).astype(np.int64)
        joints = np.copy(fit_res.canonical_mesh["joints"]).astype(np.float32)
        joint_names = list(fit_res.canonical_mesh["joint_names"])
        raw_dict = fit_res.canonical_mesh.get("raw", {})
        nV = len(base_verts)

        # Filter valid unposed points across all stops
        valid_unposed_stops: List[UnposedStopData] = []
        for st in fit_res.unposed_stops:
            if len(st.unposed_points) == 0:
                continue
            # Remove non-finite points if any
            finite_mask = np.all(np.isfinite(st.unposed_points), axis=-1)
            if not np.all(finite_mask):
                w_msg = f"Stop {st.stop_idx} contained {np.sum(~finite_mask)} non-finite unposed points; filtered."
                warnings_list.append(w_msg)
                filtered_pts = st.unposed_points[finite_mask]
                filtered_dists = st.point_to_model_distances[finite_mask] if len(st.point_to_model_distances) == len(st.unposed_points) else st.point_to_model_distances
                filtered_nn = st.nearest_vertex_indices[finite_mask] if len(st.nearest_vertex_indices) == len(st.unposed_points) else st.nearest_vertex_indices
                valid_st = UnposedStopData(
                    stop_idx=st.stop_idx,
                    unposed_points=filtered_pts,
                    camera_view_dir=st.camera_view_dir,
                    point_to_model_distances=filtered_dists,
                    mean_residual_mm=float(np.mean(filtered_dists * 1000.0)) if len(filtered_dists) > 0 else 0.0,
                    median_residual_mm=float(np.median(filtered_dists * 1000.0)) if len(filtered_dists) > 0 else 0.0,
                    p90_residual_mm=float(np.percentile(filtered_dists * 1000.0, 90)) if len(filtered_dists) > 0 else 0.0,
                    nearest_vertex_indices=filtered_nn,
                    arm_point_labels=st.arm_point_labels[finite_mask] if st.arm_point_labels is not None else None,
                    point_arm_weights=st.point_arm_weights[finite_mask] if st.point_arm_weights is not None else None,
                    diagnostics=st.diagnostics,
                )
                valid_unposed_stops.append(valid_st)
            else:
                valid_unposed_stops.append(st)

        # 3. Fit anatomically constrained personalized surface
        logger.info("Executing anatomically constrained surface personalisation...")
        s_fitter = self.surface_fitter or PersonalisedSurfaceFitter(
            canonical_vertices=base_verts,
            faces=faces,
            candidate_lambdas=candidate_lambdas,
            raw_dict=raw_dict,
            max_dist_m=max_correspondence_dist_m,
            canonical_joints=joints,
            joint_names=joint_names,
        )

        surf_res = s_fitter.fit_with_loo(
            unposed_stops=valid_unposed_stops,
            max_dist_m=max_correspondence_dist_m,
        )

        pers_verts = np.asarray(surf_res.personalised_vertices, dtype=np.float32)

        # Verify vertex finiteness and preserve mesh topology
        non_finite_count = int(np.sum(~np.isfinite(pers_verts)))
        if non_finite_count > 0:
            w_msg = f"Personalised mesh produced {non_finite_count} non-finite vertices! Reverting invalid vertices."
            warnings_list.append(w_msg)
            logger.error(w_msg)
            pers_verts = np.where(np.isfinite(pers_verts), pers_verts, base_verts.astype(np.float32))

        # 4. Package meshes
        fitted_canonical_mesh = {
            "vertices": base_verts.astype(np.float32),
            "faces": faces.astype(np.int64),
            "joints": joints.astype(np.float32),
            "joint_names": joint_names,
            "raw": raw_dict,
        }

        personalized_mesh = {
            "vertices": pers_verts,
            "faces": faces.astype(np.int64),
            "joints": joints.astype(np.float32),
            "joint_names": joint_names,
            "raw": raw_dict,
        }

        # 5. Compute dual anthropometric measurements
        logger.info("Computing dual anthropometric measurements on model and personalized mesh...")
        model_measurements = fit_res.measurements
        pers_measurements = measure_mesh(
            pers_verts,
            joints,
            faces,
            joint_names,
            raw_dict,
        )

        # Compare model vs personalized measurements
        comparisons: Dict[str, MeasurementComparison] = {}
        for m_name, m_meas in model_measurements.items():
            p_meas = pers_measurements.get(m_name)
            m_val = m_meas.value_mm
            p_val = p_meas.value_mm if p_meas is not None else float("nan")

            if np.isfinite(m_val) and np.isfinite(p_val):
                abs_diff = float(p_val - m_val)
                rel_diff = float((abs_diff / m_val) * 100.0) if abs(m_val) > 1e-6 else 0.0
            else:
                abs_diff = float("nan")
                rel_diff = float("nan")

            comparisons[m_name] = MeasurementComparison(
                name=m_name,
                model_value_mm=m_val,
                personalized_value_mm=p_val,
                absolute_change_mm=abs_diff,
                relative_change_pct=rel_diff,
                method=p_meas.method if p_meas is not None else m_meas.method,
                source="personalized_mesh",
                notes=p_meas.notes if p_meas is not None else m_meas.notes,
            )

        # 6. Observed-ring extraction with contour closure diagnostics
        observed_rings: Dict[str, ObservedRingResult] = {}
        if run_observed_rings and len(valid_unposed_stops) > 0:
            logger.info("Extracting observed ring diagnostics...")
            ring_extractor = ObservedRingExtractor(
                canon_verts=base_verts.astype(np.float32),
                canon_joints=joints,
                faces=faces,
                joint_names=joint_names,
                raw_dict=raw_dict,
            )
            raw_rings = ring_extractor.measure_all_observed_rings(valid_unposed_stops)

            for r_name, r_res in raw_rings.items():
                is_closed, closure_reason = is_contour_closed(r_res.contour_pts)
                # If coverage is low or contour is open, ensure source is marked as mixed
                source_label = "observed" if (r_res.coverage_fraction >= 0.95 and is_closed) else "mixed"
                observed_rings[r_name] = ObservedRingResult(
                    name=r_res.name,
                    value_mm=r_res.value_mm,
                    coverage_fraction=r_res.coverage_fraction,
                    source=source_label,
                    method=r_res.method,
                    unposed_residual_mm=r_res.unposed_residual_mm,
                    num_points_in_slab=r_res.num_points_in_slab,
                    num_covered_bins=r_res.num_covered_bins,
                    total_bins=r_res.total_bins,
                    contour_pts=r_res.contour_pts,
                    covered_bin_mask=r_res.covered_bin_mask,
                )

        # 7. Anatomical region displacement analysis
        disp_norms_mm = np.linalg.norm(pers_verts - base_verts, axis=-1) * 1000.0
        v_regs, _ = _compute_vertex_regions(nV, raw_dict)
        disp_by_region: Dict[str, Dict[str, float]] = {}
        under_covered_regions: List[str] = []

        if v_regs is not None:
            for reg in ANATOMICAL_REGIONS:
                reg_mask = (v_regs == reg)
                if np.any(reg_mask):
                    reg_disps = disp_norms_mm[reg_mask]
                    reg_cov = float(np.mean(surf_res.vertex_has_data_mask[reg_mask]))
                    disp_by_region[reg] = {
                        "vertex_count": int(np.sum(reg_mask)),
                        "coverage_fraction": reg_cov,
                        "mean_displacement_mm": float(np.mean(reg_disps)),
                        "median_displacement_mm": float(np.median(reg_disps)),
                        "p90_displacement_mm": float(np.percentile(reg_disps, 90)),
                        "max_displacement_mm": float(np.max(reg_disps)),
                    }
                    if reg_cov < 0.20:
                        under_covered_regions.append(reg)

        # 8. Quality status determination
        total_valid_pts = len(fit_res.all_unposed_points)
        is_converged = fit_res.fit_result.converged
        has_adequate_coverage = surf_res.vertex_coverage_fraction >= 0.20
        all_finite = (non_finite_count == 0) and all(np.isfinite(p) for p in fit_res.fitted_phenotypes.values())

        if is_converged and all_finite and has_adequate_coverage and len(valid_unposed_stops) >= 2:
            status = "success"
        elif all_finite and total_valid_pts > 0:
            status = "partial"
            if not is_converged:
                warnings_list.append(
                    f"Parametric fitting final loss ({fit_res.fit_result.final_loss:.2f}) "
                    f"did not meet nominal convergence threshold (< 8.0)."
                )
            if not has_adequate_coverage:
                warnings_list.append(
                    f"Personalized surface vertex coverage ({surf_res.vertex_coverage_fraction * 100:.1f}%) "
                    f"is below nominal threshold (20%)."
                )
        else:
            status = "failed"

        is_valid_result = all_finite and (total_valid_pts > 0)

        elapsed_total = time.time() - t0

        diagnostics: Dict[str, Any] = {
            "status": status,
            "elapsed_sec_total": elapsed_total,
            "parametric_fit_elapsed_sec": fit_res.fit_result.elapsed_sec,
            "parametric_final_loss": fit_res.fit_result.final_loss,
            "parametric_converged": fit_res.fit_result.converged,
            "num_input_stops": len(stops),
            "total_valid_unposed_points": total_valid_pts,
            "per_stop_point_counts": [len(s.unposed_points) for s in valid_unposed_stops],
            "unposed_residual_summary_mm": {
                "mean": fit_res.mean_residual_mm,
                "median": fit_res.median_residual_mm,
                "p90": fit_res.p90_residual_mm,
            },
            "personalization": {
                "selected_lambda": surf_res.selected_lambda,
                "loo_scores": surf_res.loo_scores,
                "vertex_coverage_fraction": surf_res.vertex_coverage_fraction,
                "displacement_mean_mm": float(np.mean(disp_norms_mm)),
                "displacement_median_mm": float(np.median(disp_norms_mm)),
                "displacement_p90_mm": float(np.percentile(disp_norms_mm, 90)),
                "displacement_max_mm": float(np.max(disp_norms_mm)),
                "displacement_by_region": disp_by_region,
                "under_covered_regions": under_covered_regions,
            },
            "correspondence_diagnostics": (
                {
                    "accepted_points": surf_res.diagnostics.accepted_points,
                    "rejected_points": surf_res.diagnostics.rejected_points,
                    "confidence_mean": surf_res.diagnostics.confidence_mean,
                    "rejection_reasons": surf_res.diagnostics.rejection_reasons,
                }
                if surf_res.diagnostics is not None
                else {}
            ),
            "mesh_topology_intact": (len(pers_verts) == nV and len(faces) == len(fit_res.canonical_mesh["faces"])),
            "invalid_nonfinite_vertex_count": non_finite_count,
        }

        # 9. Extract vertex appearance and visual identity layer
        appearance_res = None
        if extract_appearance:
            try:
                logger.info("Extracting multi-stop vertex appearance...")
                appearance_res = extract_vertex_appearance(
                    stops=stops,
                    canonical_vertices=pers_verts,
                    faces=faces,
                    fit_result=fit_res.fit_result,
                    distance_m=fit_res.distance_m,
                    canonical_joints=joints,
                    joint_names=joint_names,
                    raw_dict=raw_dict,
                    depth_margin_m=appearance_depth_margin_m,
                    min_cos_angle=appearance_min_cos_angle,
                )
                if appearance_res is not None:
                    diagnostics["appearance"] = appearance_res.diagnostics.as_dict()
            except Exception as e:
                logger.warning(f"Appearance extraction failed: {e}")
                warnings_list.append(f"Appearance extraction failed: {e}")

        return PersonalizedReconstructionResult(
            multi_stop_fit=fit_res,
            personalized_surface=surf_res,
            canonical_fitted_mesh=fitted_canonical_mesh,
            personalized_mesh=personalized_mesh,
            model_measurements=model_measurements,
            personalized_measurements=pers_measurements,
            measurement_comparisons=comparisons,
            observed_rings=observed_rings,
            appearance=appearance_res,
            diagnostics=diagnostics,
            status=status,
            warnings=warnings_list,
            is_valid=is_valid_result,
        )


# ---------------------------------------------------------------------------
# Functional Public Interface
# ---------------------------------------------------------------------------

def fit_and_unpose_multi_stop(
    stops: Union[List[RGBDFrame], List[AdaptedStopData], List[SimulatedStopData]],
    nominal_yaws_deg: Optional[List[float]] = None,
    distance_m: Optional[Union[float, List[float]]] = None,
    backend: Optional[AnnyBackend] = None,
    prior: Optional[AdultShapePrior] = None,
    fitter: Optional[DifferentiableBodyFitter] = None,
    known_height_mm: Optional[float] = None,
    use_floor_constraint: bool = False,
    floor_z: Optional[float] = None,
    n_coarse_iter: int = 40,
    n_fine_iter: int = 30,
    multi_starts: int = 3,
    seed: int = 42,
    ablation_remove_term: Optional[str] = None,
    raise_on_empty: bool = True,
) -> MultiStopFitResult:
    """
    Convenience function to run multi-stop fitting and unposing in a single call.
    """
    orchestrator = MultiStopOrchestrator(
        backend=backend,
        prior=prior,
        fitter=fitter,
    )
    return orchestrator.run(
        stops=stops,
        nominal_yaws_deg=nominal_yaws_deg,
        distance_m=distance_m,
        known_height_mm=known_height_mm,
        use_floor_constraint=use_floor_constraint,
        floor_z=floor_z,
        n_coarse_iter=n_coarse_iter,
        n_fine_iter=n_fine_iter,
        multi_starts=multi_starts,
        seed=seed,
        ablation_remove_term=ablation_remove_term,
        raise_on_empty=raise_on_empty,
    )


def reconstruct_and_personalize(
    stops: Union[List[RGBDFrame], List[AdaptedStopData], List[SimulatedStopData]],
    nominal_yaws_deg: Optional[List[float]] = None,
    distance_m: Optional[Union[float, List[float]]] = None,
    backend: Optional[AnnyBackend] = None,
    prior: Optional[AdultShapePrior] = None,
    fitter: Optional[DifferentiableBodyFitter] = None,
    surface_fitter: Optional[PersonalisedSurfaceFitter] = None,
    known_height_mm: Optional[float] = None,
    use_floor_constraint: bool = False,
    floor_z: Optional[float] = None,
    n_coarse_iter: int = 40,
    n_fine_iter: int = 30,
    multi_starts: int = 3,
    seed: int = 42,
    ablation_remove_term: Optional[str] = None,
    max_correspondence_dist_m: float = 0.040,
    candidate_lambdas: Optional[List[float]] = None,
    run_observed_rings: bool = True,
    extract_appearance: bool = True,
    appearance_depth_margin_m: float = 0.025,
    appearance_min_cos_angle: float = 0.1736,
    raise_on_empty: bool = True,
) -> PersonalizedReconstructionResult:
    """
    Convenience function to run end-to-end multi-stop reconstruction, surface personalisation,
    and anthropometric measurements in a single call.
    """
    orchestrator = MultiStopOrchestrator(
        backend=backend,
        prior=prior,
        fitter=fitter,
        surface_fitter=surface_fitter,
    )
    return orchestrator.reconstruct_and_personalize(
        stops=stops,
        nominal_yaws_deg=nominal_yaws_deg,
        distance_m=distance_m,
        known_height_mm=known_height_mm,
        use_floor_constraint=use_floor_constraint,
        floor_z=floor_z,
        n_coarse_iter=n_coarse_iter,
        n_fine_iter=n_fine_iter,
        multi_starts=multi_starts,
        seed=seed,
        ablation_remove_term=ablation_remove_term,
        max_correspondence_dist_m=max_correspondence_dist_m,
        candidate_lambdas=candidate_lambdas,
        run_observed_rings=run_observed_rings,
        extract_appearance=extract_appearance,
        appearance_depth_margin_m=appearance_depth_margin_m,
        appearance_min_cos_angle=appearance_min_cos_angle,
        raise_on_empty=raise_on_empty,
    )
