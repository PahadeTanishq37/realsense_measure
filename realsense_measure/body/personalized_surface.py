"""
realsense_measure/body/personalized_surface.py
=============================================
Personalised canonical surface reconstruction via Laplacian regularisation
with independent anatomical association and constrained point-to-mesh correspondence.

Separates anatomical association (Operation A: skeletal landmark / joint segment envelopes)
from mesh correspondence (Operation B: constrained candidate search), eliminating
circular labeling dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .measure import (
    Measurement,
    _loop_perimeter,
    _slice_mesh_at_z,
    _slice_mesh_plane,
    measure_mesh,
)
from .unpose import UnposedStopData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anatomical Region Definitions & Constants
# ---------------------------------------------------------------------------

ANATOMICAL_REGIONS: List[str] = [
    "torso",
    "pelvis",
    "head_neck",
    "upperarm_L",
    "forearm_L",
    "hand_L",
    "upperarm_R",
    "forearm_R",
    "hand_R",
    "thigh_L",
    "lowerleg_L",
    "thigh_R",
    "lowerleg_R",
]

REGION_COMPATIBILITY: Dict[str, set] = {
    "hand_L": {"hand_L", "forearm_L"},
    "hand_R": {"hand_R", "forearm_R"},
    "forearm_L": {"forearm_L", "hand_L", "upperarm_L"},
    "forearm_R": {"forearm_R", "hand_R", "upperarm_R"},
    "upperarm_L": {"upperarm_L", "forearm_L", "torso"},
    "upperarm_R": {"upperarm_R", "forearm_R", "torso"},
    "thigh_L": {"thigh_L", "lowerleg_L"},
    "thigh_R": {"thigh_R", "lowerleg_R"},
    "lowerleg_L": {"lowerleg_L", "thigh_L"},
    "lowerleg_R": {"lowerleg_R", "thigh_R"},
    "torso": {"torso", "pelvis", "head_neck", "upperarm_L", "upperarm_R"},
    "pelvis": {"pelvis", "torso"},
    "head_neck": {"head_neck", "torso"},
}

LEFT_LIMBS = {"upperarm_L", "forearm_L", "hand_L", "thigh_L", "lowerleg_L"}
RIGHT_LIMBS = {"upperarm_R", "forearm_R", "hand_R", "thigh_R", "lowerleg_R"}
ARM_REGIONS = {"upperarm_L", "forearm_L", "hand_L", "upperarm_R", "forearm_R", "hand_R"}
HAND_REGIONS = {"hand_L", "hand_R"}
FOREARM_REGIONS = {"forearm_L", "forearm_R"}
UPPERARM_REGIONS = {"upperarm_L", "upperarm_R"}
LEG_REGIONS = {"thigh_L", "lowerleg_L", "thigh_R", "lowerleg_R"}
TRUNK_REGIONS = {"torso", "pelvis"}

# Cache for default Anny skinning template (invariant across phenotypes)
_CACHED_ANNY_SKINNING: Optional[dict] = None


def _get_cached_default_anny_skinning() -> Optional[dict]:
    """Lazily load and cache base topological skinning and joint data from anny."""
    global _CACHED_ANNY_SKINNING
    if _CACHED_ANNY_SKINNING is None:
        try:
            from .anny_backend import AnnyBackend
            backend = AnnyBackend(verbose=False)
            body = backend.rest_pose()
            _CACHED_ANNY_SKINNING = {
                "bone_labels": list(body.raw["bone_labels"]),
                "vertex_bone_weights": (
                    body.raw["vertex_bone_weights"].numpy()
                    if hasattr(body.raw["vertex_bone_weights"], "numpy")
                    else body.raw["vertex_bone_weights"]
                ),
                "vertex_bone_indices": (
                    body.raw["vertex_bone_indices"].numpy()
                    if hasattr(body.raw["vertex_bone_indices"], "numpy")
                    else body.raw["vertex_bone_indices"]
                ),
                "canonical_joints": body.numpy_joints(0),
                "joint_names": list(body.joint_names),
            }
        except Exception as exc:
            logger.debug("Default Anny skinning could not be loaded: %s", exc)
            _CACHED_ANNY_SKINNING = None
    return _CACHED_ANNY_SKINNING


def _classify_bone_name(bone_name: str) -> str:
    """Map a bone name from Anny's 104-bone rig to a canonical anatomical region."""
    bl = bone_name.lower()
    if ".l" in bl or "_l" in bl or "left" in bl:
        if any(x in bl for x in ["finger", "metacarpal", "wrist", "hand", "thumb"]):
            return "hand_L"
        if "lowerarm" in bl or "elbow" in bl:
            return "forearm_L"
        if any(x in bl for x in ["upperarm", "shoulder", "clavicle"]):
            return "upperarm_L"
        if any(x in bl for x in ["foot", "toe", "lowerleg", "ankle"]):
            return "lowerleg_L"
        if "upperleg" in bl or "thigh" in bl:
            return "thigh_L"
    if ".r" in bl or "_r" in bl or "right" in bl:
        if any(x in bl for x in ["finger", "metacarpal", "wrist", "hand", "thumb"]):
            return "hand_R"
        if "lowerarm" in bl or "elbow" in bl:
            return "forearm_R"
        if any(x in bl for x in ["upperarm", "shoulder", "clavicle"]):
            return "upperarm_R"
        if any(x in bl for x in ["foot", "toe", "lowerleg", "ankle"]):
            return "lowerleg_R"
        if "upperleg" in bl or "thigh" in bl:
            return "thigh_R"
    if any(x in bl for x in ["pelvis", "root"]):
        return "pelvis"
    if any(x in bl for x in ["neck", "head", "eye", "jaw"]):
        return "head_neck"
    return "torso"


def _normalize_point_region(reg: str, pt_x: float) -> str:
    """Normalize a user or metadata region name to canonical region strings."""
    r = reg.strip()
    rl = r.lower()
    if rl in ["unknown", "ambiguous", "unassigned", "none", "null"]:
        return "unknown"
    if rl in ["hand", "wrist", "fingers"]:
        return "hand_L" if pt_x >= 0.0 else "hand_R"
    if rl in ["forearm", "lowerarm", "elbow"]:
        return "forearm_L" if pt_x >= 0.0 else "forearm_R"
    if rl in ["upperarm", "shoulder", "arm"]:
        return "upperarm_L" if pt_x >= 0.0 else "upperarm_R"
    if rl in ["thigh", "upperleg"]:
        return "thigh_L" if pt_x >= 0.0 else "thigh_R"
    if rl in ["lowerleg", "calf", "foot", "shin", "ankle", "leg"]:
        return "lowerleg_L" if pt_x >= 0.0 else "lowerleg_R"
    if rl in ["head", "neck"]:
        return "head_neck"
    if rl in ["torso", "chest", "waist", "abdomen", "back", "spine"]:
        return "torso"
    if rl in ["pelvis", "hip", "buttocks", "root"]:
        return "pelvis"
    return r


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex unit surface normals from mesh topology."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
    vn = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vn /= (np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8)
    return vn


def _compute_vertex_regions(
    nV: int,
    raw_dict: Optional[dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Classify each mesh vertex into its dominant anatomical region based on skinning.

    Returns
    -------
    dom_regions : (nV,) str or None
    dom_weights : (nV,) float or None
    """
    if raw_dict is None:
        if nV == 13718:
            raw_dict = _get_cached_default_anny_skinning()
        else:
            return None, None

    if raw_dict is None:
        return None, None

    bl = raw_dict.get("bone_labels")
    vw = raw_dict.get("vertex_bone_weights")
    vi = raw_dict.get("vertex_bone_indices")
    if bl is None or vw is None or vi is None:
        return None, None

    if hasattr(vw, "cpu"):
        vw = vw.cpu().numpy()
    if hasattr(vi, "cpu"):
        vi = vi.cpu().numpy()

    if len(vw) != nV:
        return None, None

    bone_regions = [_classify_bone_name(b) for b in bl]
    regs = ANATOMICAL_REGIONS
    reg_to_id = {r: i for i, r in enumerate(regs)}

    reg_matrix = np.zeros((nV, len(regs)), dtype=np.float32)
    for k in range(vi.shape[1]):
        col_idx = vi[:, k]
        col_w = vw[:, k]
        for b_i in range(len(bl)):
            mask = (col_idx == b_i)
            if np.any(mask):
                r_id = reg_to_id[bone_regions[b_i]]
                reg_matrix[mask, r_id] += col_w[mask]

    dom_ids = np.argmax(reg_matrix, axis=1)
    dom_regions = np.array([regs[i] for i in dom_ids])
    dom_weights = np.max(reg_matrix, axis=1).astype(np.float64)
    return dom_regions, dom_weights


# ---------------------------------------------------------------------------
# Diagnostics & Results Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrespondenceDiagnostics:
    """Audit diagnostics for anatomically constrained point-to-mesh correspondence."""
    total_input_points: int = 0
    accepted_points: int = 0
    rejected_points: int = 0
    unknown_points: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    per_region_counts: Dict[str, int] = field(default_factory=dict)
    confidence_mean: float = 0.0
    confidence_median: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    observed_vertices_count: int = 0
    unobserved_vertices_count: int = 0


@dataclass
class PersonalisedSurfaceResult:
    """Result of personalised surface reconstruction."""
    canonical_vertices: np.ndarray       # (V, 3) base canonical mesh
    personalised_vertices: np.ndarray    # (V, 3) personalised mesh V0 + delta_V
    faces: np.ndarray                    # (F, 3)
    delta_v_norm_mean_mm: float
    delta_v_norm_max_mm: float
    selected_lambda: float
    loo_scores: Dict[float, float]       # lambda -> mean held-out residual (mm)
    vertex_coverage_fraction: float      # Fraction of all vertices with observed data within 5 mm
    vertex_has_data_mask: np.ndarray     # (V,) bool
    diagnostics: Optional[CorrespondenceDiagnostics] = None


# ---------------------------------------------------------------------------
# Personalised Surface Fitter
# ---------------------------------------------------------------------------

class PersonalisedSurfaceFitter:
    """
    Fits per-vertex 3D displacements to canonical mesh using sparse Laplacian
    regularisation and leave-one-stop-out validation, constrained by independent
    anatomical association, normal consistency, and boundary ambiguity rejection.
    """

    def __init__(
        self,
        canonical_vertices: np.ndarray,
        faces: np.ndarray,
        candidate_lambdas: Optional[List[float]] = None,
        raw_dict: Optional[dict] = None,
        max_dist_m: float = 0.040,
        canonical_joints: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        r_support_m: float = 0.12,
        d_transition_m: float = 0.03,
        gamma_clamp: float = 200.0,
    ):
        self.v0 = np.asarray(canonical_vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.nV = len(self.v0)
        self.candidate_lambdas = candidate_lambdas or [5.0, 15.0, 30.0, 60.0, 120.0]
        self.max_dist_m = float(max_dist_m)
        self.raw_dict = raw_dict
        self.r_support_m = float(r_support_m)
        self.d_transition_m = float(d_transition_m)
        self.gamma_clamp = float(gamma_clamp)

        # Precompute uniform graph Laplacian L
        self.L, self.LtL = self._build_laplacian()
        self.base_tree = cKDTree(self.v0)

        # Precompute canonical surface normals
        self.vertex_normals = _compute_vertex_normals(self.v0, self.faces)

        # Precompute canonical anatomical regions & skinning weights
        self.vertex_regions, self.vertex_region_weights = _compute_vertex_regions(
            self.nV, self.raw_dict
        )

        # Skeletal joints for independent anatomical association (Operation A)
        self.canonical_joints = canonical_joints
        self.joint_names = joint_names
        if self.canonical_joints is None:
            if self.raw_dict and "rest_bone_heads" in self.raw_dict:
                rbh = self.raw_dict["rest_bone_heads"]
                if hasattr(rbh, "cpu"):
                    rbh = rbh.cpu().numpy()
                self.canonical_joints = np.asarray(rbh[0] if rbh.ndim == 3 else rbh, dtype=np.float64)
                self.joint_names = list(self.raw_dict.get("bone_labels", []))
            elif self.nV == 13718:
                def_data = _get_cached_default_anny_skinning()
                if def_data is not None and "canonical_joints" in def_data:
                    self.canonical_joints = def_data["canonical_joints"]
                    self.joint_names = def_data.get("joint_names")

        self.last_diagnostics: Optional[CorrespondenceDiagnostics] = None

    def _build_laplacian(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        adj = [set() for _ in range(self.nV)]
        for f in self.faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[f[i]].add(f[j])
                    adj[f[j]].add(f[i])

        rows, cols, vals = [], [], []
        for i in range(self.nV):
            deg = len(adj[i])
            if deg > 0:
                rows.append(i); cols.append(i); vals.append(1.0)
                inv_deg = -1.0 / deg
                for j in adj[i]:
                    rows.append(i); cols.append(j); vals.append(inv_deg)

        L = sp.csr_matrix((vals, (rows, cols)), shape=(self.nV, self.nV))
        LtL = L.T @ L
        return L, LtL

    def associate_anatomical_regions(
        self,
        points: np.ndarray,
        canonical_joints: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Operation A: Independently determine which anatomical region each 3D observation
        belongs to, using skeletal joint segments and spatial envelopes.

        Returns an array of strings of shape (N,) containing canonical region names
        ('torso', 'pelvis', 'thigh_L', etc.) or 'unknown'.

        CRITICAL: This operation does NOT use nearest-template mesh vertices,
        preventing circular validation loops.
        """
        pts = np.asarray(points, dtype=np.float64)
        N = len(pts)
        if N == 0:
            return np.zeros(0, dtype=object)

        joints = canonical_joints if canonical_joints is not None else self.canonical_joints
        jnames = joint_names if joint_names is not None else self.joint_names

        if joints is None or jnames is None:
            return np.array(["unknown"] * N, dtype=object)

        jmap = {n: i for i, n in enumerate(jnames)}

        def _get_j(name: str, fallback: np.ndarray) -> np.ndarray:
            return joints[jmap[name]] if name in jmap else fallback

        root = _get_j("root", np.array([0.0, 0.0, 0.0]))
        spine04 = _get_j("spine04", np.array([0.0, 0.032, 0.068]))
        spine01 = _get_j("spine01", np.array([0.0, 0.035, 0.345]))
        neck01 = _get_j("neck01", np.array([0.0, 0.006, 0.513]))
        head = _get_j("head", np.array([0.0, -0.030, 0.609]))

        sh_L = _get_j("upperarm01.L", np.array([0.165, -0.013, 0.427]))
        el_L = _get_j("lowerarm01.L", np.array([0.334, -0.019, 0.239]))
        wr_L = _get_j("wrist.L", np.array([0.450, -0.174, 0.121]))
        hand_axis_L = wr_L + (wr_L - el_L) * 0.85

        sh_R = _get_j("upperarm01.R", np.array([-0.165, -0.013, 0.427]))
        el_R = _get_j("lowerarm01.R", np.array([-0.334, -0.019, 0.239]))
        wr_R = _get_j("wrist.R", np.array([-0.450, -0.174, 0.121]))
        hand_axis_R = wr_R + (wr_R - el_R) * 0.85

        hip_L = _get_j("upperleg01.L", np.array([0.102, 0.004, -0.007]))
        kn_L = _get_j("lowerleg01.L", np.array([0.133, -0.028, -0.396]))
        ft_L = _get_j("foot.L", np.array([0.170, 0.000, -0.796]))

        hip_R = _get_j("upperleg01.R", np.array([-0.102, 0.004, -0.007]))
        kn_R = _get_j("lowerleg01.R", np.array([-0.133, -0.028, -0.396]))
        ft_R = _get_j("foot.R", np.array([-0.170, 0.000, -0.796]))

        segments = {
            "head_neck": (neck01, head + np.array([0.0, 0.0, 0.12])),
            "torso": (spine04, spine01),
            "pelvis": (root + np.array([0.0, 0.0, -0.09]), root + np.array([0.0, 0.0, 0.05])),
            "upperarm_L": (sh_L, el_L),
            "forearm_L": (el_L, wr_L),
            "hand_L": (wr_L, hand_axis_L),
            "upperarm_R": (sh_R, el_R),
            "forearm_R": (el_R, wr_R),
            "hand_R": (wr_R, hand_axis_R),
            "thigh_L": (hip_L, kn_L),
            "lowerleg_L": (kn_L, ft_L),
            "thigh_R": (hip_R, kn_R),
            "lowerleg_R": (kn_R, ft_R),
        }

        max_radii = {
            "head_neck": 0.16,
            "torso": 0.25,
            "pelvis": 0.20,
            "upperarm_L": 0.12,
            "forearm_L": 0.09,
            "hand_L": 0.10,
            "upperarm_R": 0.12,
            "forearm_R": 0.09,
            "hand_R": 0.10,
            "thigh_L": 0.15,
            "lowerleg_L": 0.16,
            "thigh_R": 0.15,
            "lowerleg_R": 0.16,
        }

        seg_keys = list(segments.keys())

        # Vectorized segment distance calculation
        dists = np.zeros((N, len(seg_keys)), dtype=np.float64)
        for i, k in enumerate(seg_keys):
            a, b = segments[k]
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))
            if ab_len_sq < 1e-8:
                dists[:, i] = np.linalg.norm(pts - a, axis=-1)
            else:
                t = np.clip(np.sum((pts - a) * ab, axis=-1) / ab_len_sq, 0.0, 1.0)
                proj = a + t[:, None] * ab
                dists[:, i] = np.linalg.norm(pts - proj, axis=-1)

        norm_dists = dists / np.array([max_radii[k] for k in seg_keys])

        # Bilateral sagittal gate
        px = pts[:, 0]
        for i, k in enumerate(seg_keys):
            if "_L" in k:
                norm_dists[px < -0.015, i] = 999.0
            elif "_R" in k:
                norm_dists[px > 0.015, i] = 999.0

        sorted_idx = np.argsort(norm_dists, axis=1)
        best_i = sorted_idx[:, 0]
        second_i = sorted_idx[:, 1]

        best_norm = norm_dists[np.arange(N), best_i]
        second_norm = norm_dists[np.arange(N), second_i]

        assigned = []
        for idx in range(N):
            b_k = seg_keys[best_i[idx]]
            s_k = seg_keys[second_i[idx]]
            b_d = best_norm[idx]
            s_d = second_norm[idx]

            # Outside all plausible anatomical envelopes -> unknown
            if b_d > 1.25:
                assigned.append("unknown")
                continue

            # Check boundary ambiguity between conflicting disjoint body parts
            b_is_arm = any(x in b_k for x in ["upperarm", "forearm", "hand"])
            s_is_arm = any(x in s_k for x in ["upperarm", "forearm", "hand"])
            b_is_trunk = b_k in ["torso", "pelvis"]
            s_is_trunk = s_k in ["torso", "pelvis"]

            # Arm-trunk contact / axilla ambiguity
            if (b_is_arm and s_is_trunk) or (b_is_trunk and s_is_arm):
                if s_d < b_d * 1.30:
                    assigned.append("unknown")
                    continue

            # Medial thigh collision zone ambiguity
            if (b_k == "thigh_L" and s_k == "thigh_R") or (b_k == "thigh_R" and s_k == "thigh_L"):
                if s_d < b_d * 1.25 or abs(px[idx]) < 0.015:
                    assigned.append("unknown")
                    continue

            assigned.append(b_k)

        return np.array(assigned, dtype=object)

    def compute_correspondences(
        self,
        points: np.ndarray,
        point_normals: Optional[np.ndarray] = None,
        point_regions: Optional[Union[np.ndarray, List[str]]] = None,
        max_dist_m: Optional[float] = None,
        k_neighbors: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CorrespondenceDiagnostics]:
        """
        Operation B: Compute anatomically constrained correspondences for input points.

        Evaluates multiple nearest candidates per point against:
        1. Independent anatomical region (Operation A).
        2. Distance threshold (d_max).
        3. Anatomical compatibility matrix.
        4. Bilateral sagittal symmetry constraints.
        5. Surface normal compatibility.
        6. Boundary ambiguity rejection across disjoint body parts.
        7. Confidence score weighting.
        """
        pts = np.asarray(points, dtype=np.float64)
        n_pts = len(pts)

        d_max = float(max_dist_m if max_dist_m is not None else self.max_dist_m)
        rejection_reasons: Dict[str, int] = {
            "unknown_region": 0,
            "out_of_range": 0,
            "cross_talk": 0,
            "normal_incompatible": 0,
            "ambiguous_boundary": 0,
            "low_confidence": 0,
        }
        per_region_counts: Dict[str, int] = {r: 0 for r in ANATOMICAL_REGIONS}

        if n_pts == 0:
            diag = CorrespondenceDiagnostics(
                total_input_points=0,
                accepted_points=0,
                rejected_points=0,
                unknown_points=0,
                rejection_reasons=rejection_reasons,
                per_region_counts=per_region_counts,
                observed_vertices_count=0,
                unobserved_vertices_count=self.nV,
            )
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                diag,
            )

        # Independent anatomical association (Operation A) if not explicitly provided
        if point_regions is None:
            point_regions = self.associate_anatomical_regions(pts)

        k = min(k_neighbors, self.nV)
        all_dists, all_nns = self.base_tree.query(pts, k=k)
        if k == 1:
            all_dists = all_dists[:, None]
            all_nns = all_nns[:, None]

        assigned_v: List[int] = []
        acc_pts: List[np.ndarray] = []
        conf_weights: List[float] = []

        has_normals = point_normals is not None and len(point_normals) == n_pts
        has_regions = point_regions is not None and len(point_regions) == n_pts

        for i in range(n_pts):
            p = pts[i]
            dists_i = all_dists[i]
            nns_i = all_nns[i]

            # Point region tag
            pt_reg: Optional[str] = None
            if has_regions and point_regions[i] is not None:
                pt_reg = _normalize_point_region(str(point_regions[i]), p[0])

            # Unknown anatomical regions cannot deform the mesh
            if pt_reg == "unknown" or pt_reg is None:
                rejection_reasons["unknown_region"] += 1
                continue

            # Direct bilateral check: if point is explicitly tagged as a limb region
            # but point coordinates violate the sagittal plane (X=0)
            if pt_reg in LEFT_LIMBS and p[0] < -0.015:
                rejection_reasons["cross_talk"] += 1
                continue
            if pt_reg in RIGHT_LIMBS and p[0] > 0.015:
                rejection_reasons["cross_talk"] += 1
                continue

            # Distance check
            within_range = dists_i < d_max
            if not np.any(within_range):
                rejection_reasons["out_of_range"] += 1
                continue

            # Candidate ambiguity check across conflicting disjoint body parts
            cand_regs = [self.vertex_regions[int(nns_i[j])] for j in range(k) if within_range[j] and self.vertex_regions is not None]
            has_thigh_L = "thigh_L" in cand_regs
            has_thigh_R = "thigh_R" in cand_regs
            has_hand = any(r in HAND_REGIONS for r in cand_regs)
            has_forearm = any(r in FOREARM_REGIONS for r in cand_regs)
            has_trunk = any(r in TRUNK_REGIONS for r in cand_regs)

            if (has_thigh_L and has_thigh_R) or ((has_hand or has_forearm) and has_trunk):
                rejection_reasons["ambiguous_boundary"] += 1
                continue

            # Point normal if available and reliable
            pt_n: Optional[np.ndarray] = None
            if has_normals:
                raw_n = point_normals[i]
                if raw_n is not None:
                    n_arr = np.asarray(raw_n, dtype=np.float64)
                    n_len = np.linalg.norm(n_arr)
                    if n_len > 0.5:
                        pt_n = n_arr / (n_len + 1e-8)

            # Evaluate each candidate within range
            valid_candidates: List[Tuple[int, float, str, float]] = []  # (v_idx, dist, reg, cos_th)
            fail_cross_talk = False
            fail_normal = False

            for j in range(k):
                if not within_range[j]:
                    continue
                v_idx = int(nns_i[j])
                d = float(dists_i[j])
                v_reg = self.vertex_regions[v_idx] if self.vertex_regions is not None else "unknown"

                # A. Anatomical Region Compatibility Check
                if v_reg != "unknown":
                    allowed_regs = REGION_COMPATIBILITY.get(pt_reg, {pt_reg})
                    if v_reg not in allowed_regs:
                        fail_cross_talk = True
                        continue

                    # Axillary safeguard: prevent upperarm <-> torso cross-talk below shoulder line
                    if (v_reg in UPPERARM_REGIONS and pt_reg == "torso") or (v_reg == "torso" and pt_reg in UPPERARM_REGIONS):
                        if p[2] < 0.32 or self.v0[v_idx, 2] < 0.32:
                            fail_cross_talk = True
                            continue

                # B. Bilateral Sagittal Constraint Check
                if v_reg in LEFT_LIMBS and p[0] < -0.015:
                    fail_cross_talk = True
                    continue
                if v_reg in RIGHT_LIMBS and p[0] > 0.015:
                    fail_cross_talk = True
                    continue

                # C. Surface Normal Compatibility Check
                cos_th = 1.0
                if pt_n is not None:
                    mesh_n = self.vertex_normals[v_idx]
                    cos_th = float(np.dot(pt_n, mesh_n))
                    if cos_th < 0.20:
                        fail_normal = True
                        continue

                valid_candidates.append((v_idx, d, v_reg, cos_th))

            if not valid_candidates:
                if fail_normal and not fail_cross_talk:
                    rejection_reasons["normal_incompatible"] += 1
                else:
                    rejection_reasons["cross_talk"] += 1
                continue

            # D. Ambiguous Boundary Check among valid candidates
            best_v, best_d, best_reg, best_cos = min(valid_candidates, key=lambda c: c[1])

            is_ambiguous = False
            for v_idx, d, v_reg, _ in valid_candidates:
                if d <= best_d + 0.012 and v_reg != best_reg and v_reg != "unknown" and best_reg != "unknown":
                    if (best_reg in HAND_REGIONS or best_reg in FOREARM_REGIONS) and (v_reg in TRUNK_REGIONS or v_reg in LEG_REGIONS):
                        is_ambiguous = True
                        break
                    if (v_reg in HAND_REGIONS or v_reg in FOREARM_REGIONS) and (best_reg in TRUNK_REGIONS or best_reg in LEG_REGIONS):
                        is_ambiguous = True
                        break
                    if best_reg == "thigh_L" and v_reg == "thigh_R":
                        is_ambiguous = True
                        break
                    if best_reg == "thigh_R" and v_reg == "thigh_L":
                        is_ambiguous = True
                        break

            if is_ambiguous:
                rejection_reasons["ambiguous_boundary"] += 1
                continue

            # E. Confidence Weighting
            s_dist = 1.0 - (best_d / d_max)
            s_norm = max(0.0, best_cos) if pt_n is not None else 1.0
            s_reg = float(self.vertex_region_weights[best_v]) if self.vertex_region_weights is not None else 1.0
            s_reg_factor = min(1.0, 0.5 + 0.5 * (s_reg / 0.5))

            conf = s_dist * (0.30 + 0.70 * s_norm) * s_reg_factor
            if conf < 0.15:
                rejection_reasons["low_confidence"] += 1
                continue

            # Accepted correspondence
            assigned_v.append(best_v)
            acc_pts.append(p)
            conf_weights.append(conf)
            if best_reg in per_region_counts:
                per_region_counts[best_reg] += 1

        acc_arr = np.array(acc_pts, dtype=np.float64) if acc_pts else np.zeros((0, 3), dtype=np.float64)
        v_arr = np.array(assigned_v, dtype=np.int64) if assigned_v else np.zeros(0, dtype=np.int64)
        w_arr = np.array(conf_weights, dtype=np.float64) if conf_weights else np.zeros(0, dtype=np.float64)

        n_acc = len(v_arr)
        n_rej = n_pts - n_acc
        obs_v_count = len(np.unique(v_arr)) if n_acc > 0 else 0
        unobs_v_count = self.nV - obs_v_count

        conf_mean = float(np.mean(w_arr)) if n_acc > 0 else 0.0
        conf_median = float(np.median(w_arr)) if n_acc > 0 else 0.0
        conf_min = float(np.min(w_arr)) if n_acc > 0 else 0.0
        conf_max = float(np.max(w_arr)) if n_acc > 0 else 0.0

        diag = CorrespondenceDiagnostics(
            total_input_points=n_pts,
            accepted_points=n_acc,
            rejected_points=n_rej,
            unknown_points=rejection_reasons["unknown_region"],
            rejection_reasons=rejection_reasons,
            per_region_counts=per_region_counts,
            confidence_mean=conf_mean,
            confidence_median=conf_median,
            confidence_min=conf_min,
            confidence_max=conf_max,
            observed_vertices_count=obs_v_count,
            unobserved_vertices_count=unobs_v_count,
        )

        return v_arr, acc_arr, w_arr, diag

    def _solve_offsets(
        self,
        points: np.ndarray,
        lam: float,
        point_normals: Optional[np.ndarray] = None,
        point_regions: Optional[Union[np.ndarray, List[str]]] = None,
        max_dist_m: Optional[float] = None,
        r_support_m: Optional[float] = None,
        d_transition_m: Optional[float] = None,
        gamma_clamp: Optional[float] = None,
    ) -> np.ndarray:
        """
        Solve linear system (W + lam * LtL + Gamma) delta_V = W (P_bar - V0)
        using confidence-weighted anatomically constrained correspondences and
        Dirichlet boundary pinning to eliminate cross-region deformation leakage.
        """
        assigned_v, acc_pts, conf_w, diags = self.compute_correspondences(
            points,
            point_normals=point_normals,
            point_regions=point_regions,
            max_dist_m=max_dist_m,
        )
        self.last_diagnostics = diags

        if len(acc_pts) < 10:
            return np.zeros((self.nV, 3), dtype=np.float64)

        r_supp = r_support_m if r_support_m is not None else self.r_support_m
        d_trans = d_transition_m if d_transition_m is not None else self.d_transition_m
        g_clamp = gamma_clamp if gamma_clamp is not None else self.gamma_clamp

        # Identify actively observed anatomical regions from correspondences
        active_regs = (
            set(self.vertex_regions[assigned_v])
            if self.vertex_regions is not None
            else set()
        )

        # Build Dirichlet boundary pinning & anatomical stiffness vector gamma
        gamma = np.zeros(self.nV, dtype=np.float64)
        tapered_conf_w = conf_w.copy()

        if self.vertex_regions is not None and len(active_regs) > 0:
            inactive_mask = ~np.isin(self.vertex_regions, list(active_regs))
            active_mask = ~inactive_mask

            obs_tree = cKDTree(self.v0[assigned_v])
            dists_to_obs, _ = obs_tree.query(self.v0)

            if np.any(inactive_mask):
                inactive_tree = cKDTree(self.v0[inactive_mask])
                dists_to_inactive, _ = inactive_tree.query(self.v0)

                # Taper correspondence confidence near inactive boundaries to prevent surface tearing
                assigned_d_in = dists_to_inactive[assigned_v]
                near_bound = assigned_d_in < d_trans
                if np.any(near_bound):
                    tapered_conf_w[near_bound] *= (assigned_d_in[near_bound] / d_trans)
            else:
                dists_to_inactive = np.full(self.nV, 999.0)

            # Inactive regions receive Dirichlet clamping to ensure zero displacement
            gamma[inactive_mask] = g_clamp

            # Active vertices: clamp if far from observations or smoothly taper near inactive boundaries
            d_o = dists_to_obs[active_mask]
            d_in = dists_to_inactive[active_mask]

            if r_supp > 0:
                g_obs = np.where(
                    d_o > r_supp,
                    g_clamp,
                    np.where(
                        d_o > r_supp * 0.5,
                        g_clamp * ((d_o - r_supp * 0.5) / (r_supp * 0.5)) ** 2,
                        0.0,
                    ),
                )
            else:
                g_obs = np.zeros(np.sum(active_mask), dtype=np.float64)

            g_bound = np.where(
                d_in < d_trans,
                g_clamp * ((d_trans - d_in) / d_trans) ** 2,
                0.0,
            )
            gamma[active_mask] = np.maximum(g_obs, g_bound)

        w = np.zeros(self.nV, dtype=np.float64)
        p_sum = np.zeros((self.nV, 3), dtype=np.float64)

        np.add.at(w, assigned_v, tapered_conf_w)
        np.add.at(p_sum, assigned_v, acc_pts * tapered_conf_w[:, None])

        has_data = w > 0.0
        p_bar = self.v0.copy()
        p_bar[has_data] = p_sum[has_data] / w[has_data, None]

        W_diag = sp.diags(w)
        Gamma_diag = sp.diags(gamma)
        A = W_diag + lam * self.LtL + Gamma_diag
        b = W_diag @ (p_bar - self.v0)

        delta_v = np.zeros((self.nV, 3), dtype=np.float64)
        for d in range(3):
            sol, _ = spla.cg(A, b[:, d], rtol=1e-5, maxiter=250)
            delta_v[:, d] = sol

        return delta_v

    def fit_displacement(
        self,
        unposed_stops: Union[List[UnposedStopData], np.ndarray],
        max_dist_m: Optional[float] = None,
    ) -> PersonalisedSurfaceResult:
        """
        Fit anatomically constrained personalized surface displacements.
        Accepts either a list of UnposedStopData or a raw point cloud array (N, 3).
        """
        if isinstance(unposed_stops, list):
            return self.fit_with_loo(unposed_stops, max_dist_m=max_dist_m)
        else:
            pts = np.asarray(unposed_stops, dtype=np.float32)
            dists, nns = self.base_tree.query(pts)
            dummy = UnposedStopData(
                stop_idx=0,
                unposed_points=pts,
                camera_view_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                point_to_model_distances=dists,
                mean_residual_mm=float(np.mean(dists * 1000.0)) if len(dists) > 0 else 0.0,
                median_residual_mm=float(np.median(dists * 1000.0)) if len(dists) > 0 else 0.0,
                p90_residual_mm=float(np.percentile(dists * 1000.0, 90)) if len(dists) > 0 else 0.0,
                nearest_vertex_indices=nns,
            )
            return self.fit_with_loo([dummy], max_dist_m=max_dist_m)

    def fit_with_loo(
        self,
        unposed_stops: List[UnposedStopData],
        max_dist_m: Optional[float] = None,
    ) -> PersonalisedSurfaceResult:
        """
        Select optimal lambda via leave-one-stop-out validation, then fit on all stops.
        Derives point anatomical regions strictly independently of nearest-template indices.
        """
        all_stop_points = []
        all_stop_regions = []

        for st in unposed_stops:
            if len(st.unposed_points) == 0:
                continue
            all_stop_points.append(st.unposed_points)
            # Operation A: Independently establish anatomical regions using skeletal joint envelopes.
            # CRITICAL: We do NOT use st.nearest_vertex_indices to avoid circular labeling!
            stop_regs = self.associate_anatomical_regions(st.unposed_points)
            if hasattr(st, "arm_point_labels") and st.arm_point_labels is not None and len(st.arm_point_labels) == len(stop_regs):
                # Integrate upstream independent arm identification from unposing
                arm_mask = np.isin(st.arm_point_labels, list(ARM_REGIONS))
                stop_regs[arm_mask] = st.arm_point_labels[arm_mask]
                unk_mask = st.arm_point_labels == "unknown"
                stop_regs[unk_mask] = "unknown"
            all_stop_regions.append(stop_regs)

        n_stops = len(all_stop_points)
        loo_scores: Dict[float, float] = {}
        d_max = max_dist_m if max_dist_m is not None else self.max_dist_m

        if n_stops >= 3 and len(self.candidate_lambdas) > 1:
            held_out_idx = n_stops - 1
            train_pts = np.vstack([
                all_stop_points[k] for k in range(n_stops) if k != held_out_idx
            ])
            train_regs = (
                np.concatenate([all_stop_regions[k] for k in range(n_stops) if k != held_out_idx])
                if all(r is not None for r in all_stop_regions)
                else None
            )
            held_out_pts = all_stop_points[held_out_idx]

            # Downsample held out points for fast evaluation
            if len(held_out_pts) > 5000:
                held_eval = held_out_pts[:: len(held_out_pts) // 5000]
            else:
                held_eval = held_out_pts

            for lam in self.candidate_lambdas:
                delta_v_fold = self._solve_offsets(
                    train_pts, lam, point_regions=train_regs, max_dist_m=d_max
                )
                v_fold = self.v0 + delta_v_fold

                fold_tree = cKDTree(v_fold)
                dists, _ = fold_tree.query(held_eval)
                valid_d = dists[dists < d_max]
                loo_scores[lam] = float(np.mean(valid_d) * 1000.0) if len(valid_d) > 0 else 999.0

            best_lambda = min(loo_scores.keys(), key=lambda l: loo_scores[l])
        else:
            best_lambda = self.candidate_lambdas[0]
            loo_scores[best_lambda] = 0.0

        # Final fit on all stops combined using best_lambda
        all_pts = np.vstack(all_stop_points) if all_stop_points else np.zeros((0, 3), dtype=np.float64)
        all_regs = (
            np.concatenate(all_stop_regions)
            if (all_stop_regions and all(r is not None for r in all_stop_regions))
            else None
        )

        delta_v_final = self._solve_offsets(
            all_pts, best_lambda, point_regions=all_regs, max_dist_m=d_max
        )
        final_diagnostics = self.last_diagnostics
        v_personalised = self.v0 + delta_v_final

        # Compute data coverage: fraction of vertices with observed points within 5 mm
        if len(all_pts) > 0:
            obs_tree = cKDTree(all_pts)
            dists_to_obs, _ = obs_tree.query(v_personalised)
            has_data_mask = (dists_to_obs <= 0.005)
            overall_cov = float(np.mean(has_data_mask))
        else:
            has_data_mask = np.zeros(self.nV, dtype=bool)
            overall_cov = 0.0

        delta_norms_mm = np.linalg.norm(delta_v_final, axis=-1) * 1000.0

        return PersonalisedSurfaceResult(
            canonical_vertices=self.v0,
            personalised_vertices=v_personalised,
            faces=self.faces,
            delta_v_norm_mean_mm=float(np.mean(delta_norms_mm)),
            delta_v_norm_max_mm=float(np.max(delta_norms_mm)),
            selected_lambda=best_lambda,
            loo_scores=loo_scores,
            vertex_coverage_fraction=overall_cov,
            vertex_has_data_mask=has_data_mask,
            diagnostics=final_diagnostics,
        )

    def measure_personalised_rings(
        self,
        surf_res: PersonalisedSurfaceResult,
        canonical_joints: np.ndarray,
        joint_names: List[str],
        raw_dict: Optional[dict] = None,
    ) -> Dict[str, Measurement]:
        """
        Measure all rings directly on the watertight personalised canonical mesh
        using measure_mesh, and attach exact ring vertex coverage.
        """
        r_dict = raw_dict if raw_dict is not None else self.raw_dict
        meas = measure_mesh(
            surf_res.personalised_vertices,
            canonical_joints,
            surf_res.faces,
            joint_names,
            r_dict,
        )

        for k, m in meas.items():
            if "circumference" in k:
                if surf_res.vertex_coverage_fraction >= 0.95:
                    m.source = "observed"
                else:
                    m.source = "mixed"
                m.notes = f"{m.notes} [Coverage: {surf_res.vertex_coverage_fraction*100:.1f}%, LOO lambda={surf_res.selected_lambda:.1f}]"

        return meas
