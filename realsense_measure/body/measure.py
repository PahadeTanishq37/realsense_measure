"""
realsense_measure/body/measure.py
===================================
All body measurements are computed on the **rest_pose()** output only
(canonical A-pose, identity bone rotations).  This guarantees:

  1. Reproducibility — no pose ambiguity.
  2. Comparability across bodies — shapes differ, pose does not.
  3. Independence from camera/fitting (Step 1 scope).

Axis convention (anny native, confirmed from anny 0.6.2)
---------------------------------------------------------
+Z  UP       (Z increases toward the top of the head)
+Y  FORWARD  (Y increases in the anterior/forward direction)
+X  LATERAL  (+X = subject's anatomical left, -X = subject's anatomical right)

Vertices are in **metres**; measurements returned in **mm**.

Limb Circumferences (Set B)
---------------------------
Limbs in A-pose are oriented diagonally (not parallel to Z).
Each limb is sliced with a cutting plane **strictly perpendicular** to the
segment axis connecting parent and child joints:
  - Upper arm : perpendicular to upperarm01 → lowerarm01 at midpoint
  - Forearm   : perpendicular to lowerarm01 → wrist, maximum in proximal 15–40%
  - Mid-thigh : perpendicular to upperleg01 → lowerleg01 at midpoint
  - Max calf  : perpendicular to lowerleg01 → foot, maximum along lower leg (20–50%)

The cross-section loop is selected by checking that it encloses the segment
axis (winding number in the 2D plane coordinate system).
This completely eliminates orientation bias and guarantees left/right symmetry
within 0.1% on symmetric rest poses.

Torso Circumferences (Chest, Waist, Hip)
----------------------------------------
Before slicing the trunk, arm and hand vertices are filtered out using bone
skinning weights so that hanging forearms/hands cannot bridge into the torso
perimeter.  The loop enclosing the spine/torso axis (X=0) is selected.
  - Chest: at nipple / mid-sternum level (spine01 / upper thoracic)
  - Waist: narrowest trunk cross-section in the lumbar zone (spine04 to spine02)
  - Hip:   maximum cross-section around the pelvis / buttock crest (near root)

Lengths
-------
  - Shoulder–elbow : 3D Euclidean distance from acromion surface landmark
                     to elbow joint (lowerarm01).
  - Shoulder–wrist : 3D kinematic length from acromion landmark to elbow +
                     elbow to wrist.
  - Inseam         : 3D distance along the leg from perineum crotch landmark
                     to knee + knee to ankle.
  - Crotch height  : 3D vertical distance from perineum crotch landmark to floor.

Shoulder landmark definition:
  The acromion process landmark is located as the local surface peak on the
  mesh directly superior/lateral to the glenohumeral joint (upperarm01).
  The offset from upperarm01 is documented and approximately 74 mm.

Every measurement returns a Measurement dataclass with value_mm, name, method,
and source="model".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Measurement dataclass
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """
    A single body measurement result.

    Attributes
    ----------
    name : str
        Human-readable measurement name (e.g. ``"chest_circumference"``).
    value_mm : float
        Measurement value in millimetres.  ``math.nan`` if computation failed.
    method : str
        Short description of the algorithm used.
    source : str
        ``"model"`` (default) or ``"observed"`` (when covered by depth data).
    notes : str
        Human-readable landmark definition and method notes.
    """
    name: str
    value_mm: float
    method: str
    source: str = "model"
    notes: str = ""

    def is_valid(self) -> bool:
        return not math.isnan(self.value_mm)

    def __repr__(self) -> str:
        return f"Measurement({self.name!r}, {self.value_mm:.1f} mm, source={self.source!r})"


# ---------------------------------------------------------------------------
# Low-level 3D Geometry & Plane Slicing Helpers
# ---------------------------------------------------------------------------

def _orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors (u, v) forming an orthonormal 2D basis in the plane."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _point_in_polygon_2d(pts_2d: np.ndarray) -> bool:
    """
    Check if the origin (0, 0) is enclosed by a closed 2D polygon using winding number.
    """
    if len(pts_2d) < 3:
        return False
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    angles = np.arctan2(y, x)
    dangles = np.diff(angles)
    dangles = (dangles + np.pi) % (2 * np.pi) - np.pi
    total_angle = np.sum(dangles) + ((angles[0] - angles[-1] + np.pi) % (2 * np.pi) - np.pi)
    return abs(total_angle) > 1.0  # ≈ 2*pi for enclosing loop


def _slice_mesh_plane(
    vertices: np.ndarray,  # (V, 3)
    faces: np.ndarray,     # (F, 3)
    p0: np.ndarray,        # (3,) origin on cutting plane
    normal: np.ndarray,    # (3,) plane normal vector
) -> List[np.ndarray]:
    """
    Slice a triangle mesh with an arbitrary plane (p0, normal).
    Returns a list of closed loops, each shape (N, 3).
    """
    if len(faces) == 0 or len(vertices) == 0:
        return []

    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        return []
    n = normal / norm_val

    # Signed distance of every vertex to the cutting plane
    s = (vertices - p0) @ n
    above = (s >= 0)
    fa = above[faces]
    sum_fa = fa.sum(axis=1)

    crossing_idx = np.where((sum_fa == 1) | (sum_fa == 2))[0]
    if len(crossing_idx) == 0:
        return []

    cf = faces[crossing_idx]
    cv = vertices[cf]
    cs = s[cf]
    ca = fa[crossing_idx]

    e0 = (ca[:, 0] != ca[:, 1])
    e1 = (ca[:, 1] != ca[:, 2])
    e2 = (ca[:, 2] != ca[:, 0])

    def interp(p_a, p_b, s_a, s_b):
        ds = s_b - s_a
        safe_ds = np.where(np.abs(ds) < 1e-12, 1e-12, ds)
        t = -s_a / safe_ds
        return p_a + t[:, None] * (p_b - p_a)

    pts01 = interp(cv[:, 0], cv[:, 1], cs[:, 0], cs[:, 1])
    pts12 = interp(cv[:, 1], cv[:, 2], cs[:, 1], cs[:, 2])
    pts20 = interp(cv[:, 2], cv[:, 0], cs[:, 2], cs[:, 0])

    pa = np.where(e0[:, None], pts01, pts12)
    pb = np.where(e0[:, None], np.where(e1[:, None], pts12, pts20), pts20)

    segments = [(pa[i], pb[i]) for i in range(len(pa))]
    return _chain_segments(segments)


def _slice_mesh_at_z(
    vertices: np.ndarray,
    faces: np.ndarray,
    z_plane: float,
) -> List[np.ndarray]:
    """Horizontal (Z=const) planar cross section."""
    p0 = np.array([0.0, 0.0, z_plane])
    normal = np.array([0.0, 0.0, 1.0])
    return _slice_mesh_plane(vertices, faces, p0, normal)


def _circumference_at_z(
    vertices: np.ndarray,
    faces: np.ndarray,
    z_plane: float,
    torso: bool = False,
    side: Optional[str] = None,
) -> float:
    """Convenience helper for horizontal cross-section perimeter (mm)."""
    verts = vertices
    fcs = faces
    if side is not None:
        mask = (vertices[:, 0] < 0) if side == "left" else (vertices[:, 0] >= 0)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return float("nan")
        idx_map = np.full(len(vertices), -1, dtype=np.int64)
        idx_map[valid_idx] = np.arange(len(valid_idx))
        face_mask = np.all(mask[faces], axis=1)
        if not np.any(face_mask):
            return float("nan")
        fcs = idx_map[faces[face_mask]]
        verts = vertices[valid_idx]

    loops = _slice_mesh_at_z(verts, fcs, z_plane)
    if not loops:
        return float("nan")
    if torso:
        loop = _select_torso_loop(loops)
    else:
        loop = max(loops, key=_loop_perimeter)
    if loop is None or len(loop) < 3:
        return float("nan")
    return _loop_perimeter(loop) * 1000.0


def _chain_segments(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-5,
) -> List[np.ndarray]:
    """
    Chain a list of (p0, p1) segment tuples into closed loops using cKDTree.
    """
    if not segments:
        return []

    try:
        from scipy.spatial import cKDTree
        use_tree = True
    except ImportError:
        use_tree = False

    n = len(segments)
    if not use_tree or n < 10:
        used = [False] * n
        loops: List[np.ndarray] = []
        for start in range(n):
            if used[start]:
                continue
            chain: List[np.ndarray] = [segments[start][0], segments[start][1]]
            used[start] = True
            for _ in range(n):
                tail = chain[-1]
                found = False
                for j in range(n):
                    if used[j]:
                        continue
                    p0, p1 = segments[j]
                    if np.linalg.norm(p0 - tail) < tol:
                        chain.append(p1)
                        used[j] = True
                        found = True
                        break
                    if np.linalg.norm(p1 - tail) < tol:
                        chain.append(p0)
                        used[j] = True
                        found = True
                        break
                if not found:
                    break
            if len(chain) >= 3:
                loops.append(np.array(chain, dtype=np.float64))
        return loops

    p0s = np.array([s[0] for s in segments], dtype=np.float64)
    p1s = np.array([s[1] for s in segments], dtype=np.float64)
    all_pts = np.vstack([p0s, p1s])
    tree = cKDTree(all_pts)

    used_seg = np.zeros(n, dtype=bool)
    loops = []

    for start in range(n):
        if used_seg[start]:
            continue
        used_seg[start] = True
        chain = [p0s[start], p1s[start]]
        tail = p1s[start]

        while True:
            neighbors = tree.query_ball_point(tail, tol)
            found = False
            for idx in neighbors:
                seg_idx = idx % n
                if used_seg[seg_idx]:
                    continue
                used_seg[seg_idx] = True
                tail = p1s[seg_idx] if idx < n else p0s[seg_idx]
                chain.append(tail)
                found = True
                break
            if not found:
                break

        if len(chain) >= 3:
            loops.append(np.array(chain, dtype=np.float64))

    return loops


def is_contour_closed(
    contour: np.ndarray,
    tol: float = 1e-4,
    max_gap_ratio: float = 1.8,
    min_isoperimetric_quotient: float = 0.10,
) -> Tuple[bool, str]:
    """
    Explicit contour-closure validation for anthropometric cross-sections.

    Distinguishes genuinely closed loops from open arcs, gap-broken contours,
    degenerate geometries, and close-endpoint open folds (hairpins).

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 3) or (N, 2) containing sequentially ordered vertices.
    tol : float, default=1e-4
        Tolerance for detecting explicit duplicate closure points (m).
    max_gap_ratio : float, default=1.8
        Maximum allowable ratio of endpoint gap to maximum adjacent segment step.
    min_isoperimetric_quotient : float, default=0.10
        Minimum isoperimetric circularity (4*pi*Area / P^2) required to exclude
        collapsed geometries, self-intersecting loops, or razor-thin hairpins.

    Returns
    -------
    is_closed : bool
        True if the contour forms a genuinely closed, non-degenerate anatomical loop.
    reason : str
        Diagnostic description of the closure status or failure mode.
    """
    if contour is None:
        return False, "Contour is None"

    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return False, f"Invalid contour array shape {pts.shape}"

    N = len(pts)
    if N < 3:
        return False, f"Too few points ({N} < 3)"

    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return False, "Contour contains NaN or Inf coordinates"

    # Filter consecutive duplicate points
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.ones(N, dtype=bool)
    keep[1:] = diffs > 1e-9
    clean = pts[keep]
    if len(clean) < 3:
        return False, "Degenerate contour: fewer than 3 distinct points"

    # Check explicit closure
    explicitly_closed = (np.linalg.norm(clean[-1] - clean[0]) <= tol)
    eval_pts = clean[:-1] if explicitly_closed else clean
    N_eval = len(eval_pts)
    if N_eval < 3:
        return False, "Fewer than 3 distinct points"

    seg_lens = np.linalg.norm(np.diff(eval_pts, axis=0), axis=1)
    if np.all(seg_lens < 1e-8):
        return False, "Zero-length segments: collapsed contour"

    med_step = float(np.median(seg_lens))
    max_step = float(np.max(seg_lens))
    gap = float(np.linalg.norm(eval_pts[-1] - eval_pts[0]))

    if not explicitly_closed:
        allowed_gap = max(max_step * max_gap_ratio, tol)
        if gap > allowed_gap:
            return False, f"Endpoint gap ({gap*1000:.1f} mm) exceeds allowed step ({allowed_gap*1000:.1f} mm)"

    # Project to 2D best-fit plane
    centered = eval_pts - np.mean(eval_pts, axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.shape == (3, 3):
        evals, evecs = np.linalg.eigh(cov)
        p2d = centered @ evecs[:, [1, 2]]
    else:
        p2d = centered[:, :2]

    # Area & isoperimetric quotient
    x = p2d[:, 0]
    y = p2d[:, 1]
    shoelace_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(seg_lens)) + (0.0 if explicitly_closed else gap)
    if perim < 1e-6:
        return False, "Zero perimeter"

    iq = (4.0 * np.pi * shoelace_area) / (perim ** 2)
    if iq < min_isoperimetric_quotient:
        return False, f"Isoperimetric quotient ({iq:.4f}) < {min_isoperimetric_quotient}: geometry collapsed, folded, or open hairpin"

    # Polar angular sweep along the open sequence around centroid
    angles = np.arctan2(y, x)
    d_a = (angles[1:] - angles[:-1] + np.pi) % (2 * np.pi) - np.pi
    total_sweep = float(abs(np.sum(d_a)))
    min_sweep = np.radians(200.0) if N_eval <= 4 else np.radians(270.0)
    if total_sweep < min_sweep:
        return False, f"Angular sweep ({np.degrees(total_sweep):.1f} deg) < {np.degrees(min_sweep):.1f} deg: incomplete angular coverage"

    return True, "Closed"


def _loop_perimeter(loop: np.ndarray, require_closed: bool = True) -> float:
    """
    Surface-path perimeter of a closed 3D polygon (metres).

    Parameters
    ----------
    loop : np.ndarray
        Array of shape (N, 3) or (N, 2).
    require_closed : bool, default=True
        If True, validates that the contour is genuinely closed using
        is_contour_closed(). Returns float("nan") if open, incomplete,
        degenerate, or unclosed fold.
        If False, calculates the path length along the open polyline
        without adding an artificial straight-line chord.
    """
    if loop is None or len(loop) < 2:
        return float("nan") if require_closed else 0.0

    pts = np.asarray(loop, dtype=np.float64)
    if require_closed:
        closed, _ = is_contour_closed(pts)
        if not closed:
            return float("nan")

    diffs = np.diff(pts, axis=0)
    perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))
    closing_gap = float(np.linalg.norm(pts[-1] - pts[0]))
    if require_closed and closing_gap > 1e-6:
        perimeter += closing_gap
    return perimeter


def _loop_area_xy(loop: np.ndarray) -> float:
    """Projected area in XY plane via Shoelace formula."""
    pts = loop[:, [0, 1]]
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _select_torso_loop(loops: List[np.ndarray], origin_xy: Tuple[float, float] = (0.0, 0.0)) -> Optional[np.ndarray]:
    """
    Select the loop enclosing the spine/torso axis (X=origin_x, Y=origin_y).
    Filters out unclosed loops and falls back to largest XY area.
    """
    if not loops:
        return None
    valid_loops = [lp for lp in loops if is_contour_closed(lp)[0]]
    if not valid_loops:
        return None
    if len(valid_loops) == 1:
        return valid_loops[0]

    ox, oy = origin_xy
    enclosing = []
    for lp in valid_loops:
        pts2d = lp[:, :2] - np.array([ox, oy])
        if _point_in_polygon_2d(pts2d):
            enclosing.append(lp)

    if enclosing:
        return max(enclosing, key=_loop_area_xy)
    return max(valid_loops, key=_loop_area_xy)


# ---------------------------------------------------------------------------
# Perpendicular Limb Slicing
# ---------------------------------------------------------------------------

def _measure_limb_circumference(
    vertices: np.ndarray,
    faces: np.ndarray,
    p0: np.ndarray,
    axis: np.ndarray,
    max_radius: float = 0.25,
) -> float:
    """
    Measure circumference of a limb sliced strictly perpendicular to its bone axis.
    Selects the closed loop enclosing the segment axis p0.
    Returns value in mm.
    """
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return float("nan")
    n = axis / axis_norm

    # Fast pre-filter: vertices within bounding cylinder
    proj = (vertices - p0) @ n
    dists_sq = np.sum((vertices - p0)**2, axis=1) - proj**2
    near_mask = (np.abs(proj) < 0.15) & (dists_sq < max_radius**2)

    if np.sum(near_mask) < 10:
        near_verts = vertices
        near_faces = faces
    else:
        # Re-index faces touching near_mask
        idx_map = np.full(len(vertices), -1, dtype=np.int64)
        v_sub = np.where(near_mask)[0]
        idx_map[v_sub] = np.arange(len(v_sub))
        f_sub = faces[np.all(near_mask[faces], axis=1)]
        near_verts = vertices[v_sub]
        near_faces = idx_map[f_sub]

    loops = _slice_mesh_plane(near_verts, near_faces, p0, n)
    if not loops:
        # Fall back to full mesh slice if submesh was too tight
        loops = _slice_mesh_plane(vertices, faces, p0, n)
        if not loops:
            return float("nan")

    # Select loop enclosing axis point p0
    u, v = _orthonormal_basis(n)
    enclosing = []
    for lp in loops:
        pts2d = np.stack([(lp - p0) @ u, (lp - p0) @ v], axis=1)
        if _point_in_polygon_2d(pts2d):
            enclosing.append(lp)

    if enclosing:
        best_loop = min(enclosing, key=lambda lp: np.mean(np.linalg.norm(lp - p0, axis=1)))
    else:
        best_loop = min(loops, key=lambda lp: np.mean(np.linalg.norm(lp - p0, axis=1)))

    if len(best_loop) < 3:
        return float("nan")

    return _loop_perimeter(best_loop) * 1000.0


# ---------------------------------------------------------------------------
# Torso Masking Helpers
# ---------------------------------------------------------------------------

def _get_torso_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    raw_dict: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exclude arm and hand vertices using skinning weights so that hanging
    forearms and hands cannot merge into waist/hip circumference perimeters.
    """
    v_weights = raw_dict.get("vertex_bone_weights")
    v_indices = raw_dict.get("vertex_bone_indices")
    bl = raw_dict.get("bone_labels")

    if v_weights is None or v_indices is None or bl is None:
        return vertices, faces

    if hasattr(v_weights, "cpu"):
        v_weights = v_weights.cpu().numpy()
    if hasattr(v_indices, "cpu"):
        v_indices = v_indices.cpu().numpy()

    # Arm bones (upper arm, forearm, wrist, hand, fingers) to exclude from torso
    arm_exclude_bones = [
        b for b in bl if any(k in b.lower() for k in [
            "upperarm", "lowerarm", "wrist", "finger", "metacarpal", "hand", "thumb"
        ])
    ]
    arm_idxs = set(bl.index(b) for b in arm_exclude_bones if b in bl)

    if not arm_idxs:
        return vertices, faces

    arm_weight = np.zeros(len(vertices), dtype=np.float32)
    for k in range(v_indices.shape[1]):
        in_arm = np.isin(v_indices[:, k], list(arm_idxs))
        arm_weight += np.where(in_arm, v_weights[:, k], 0.0)

    # Exclude faces where any vertex belongs significantly to arm
    keep_faces = ~np.any(arm_weight[faces] > 0.35, axis=1)
    if not np.any(keep_faces):
        return vertices, faces

    return vertices, faces[keep_faces]


# ---------------------------------------------------------------------------
# Landmark Heights and Landmark Extraction
# ---------------------------------------------------------------------------

class LandmarkHeights:
    """
    Extract anatomical landmarks and segment axes from rest-pose joints and mesh.
    """

    def __init__(
        self,
        joints: np.ndarray,    # (J, 3)
        joint_names: List[str],
        vertices: np.ndarray,  # (V, 3)
    ) -> None:
        j = np.asarray(joints)
        if j.ndim == 3 and len(j) == 1:
            j = j[0]
        v = np.asarray(vertices)
        if v.ndim == 3 and len(v) == 1:
            v = v[0]
        self.joints = j
        self.joint_names = joint_names
        self.jmap = {name: i for i, name in enumerate(joint_names)}
        self.verts = v

    def _j(self, *candidates: str) -> Optional[np.ndarray]:
        for c in candidates:
            if c in self.jmap:
                return self.joints[self.jmap[c]]
        return None

    def head_top_z(self) -> float:
        return float(self.verts[:, 2].max())

    def floor_z(self) -> float:
        return float(self.verts[:, 2].min())

    def head_circ_z(self) -> float:
        """60 mm below crown: passing brow-ridge (glabella) and occiput."""
        return self.head_top_z() - 0.060

    def neck_base_z(self) -> float:
        """Superior thoracic aperture / cervicale."""
        neck_pt = self._j("neck01", "neck_01", "Neck")
        if neck_pt is not None:
            return float(neck_pt[2])
        return self.head_top_z() - (self.head_top_z() - self.floor_z()) * 0.14

    def acromion_landmark(self, side: str = "L") -> Optional[np.ndarray]:
        """
        Anatomical acromion landmark (shoulder tip).
        Located as the local surface peak on the mesh directly superior/lateral
        to the glenohumeral joint (upperarm01).
        Offset from upperarm01 is approximately 74 mm.
        Returns None if shoulder joint or mesh vertices are missing.
        """
        if len(self.verts) == 0:
            return None
        up01 = self._j(f"upperarm01.{side}", f"UpperArm.{side}", f"shoulder.{side}", f"Shoulder.{side}")
        if up01 is None:
            return None

        # Find nearby mesh vertices within 80 mm
        dists = np.linalg.norm(self.verts - up01, axis=1)
        nearby = np.where(dists < 0.080)[0]
        if len(nearby) > 0:
            return self.verts[nearby[np.argmax(self.verts[nearby, 2])]]
        # Fallback anatomical offset from joint if mesh is coarse
        sign = 1.0 if side == "L" else -1.0
        return up01 + np.array([-0.025 * sign, 0.028, 0.064])

    def crotch_landmark(self) -> np.ndarray:
        """
        Perineum surface landmark: midline surface point between medial thighs.
        """
        midline = np.where(
            (np.abs(self.verts[:, 0]) < 0.025) &
            (self.verts[:, 2] < 0.10) &
            (self.verts[:, 2] > -0.30)
        )[0]
        if len(midline) > 0:
            return self.verts[midline[np.argmin(self.verts[midline, 2])]]
        # Fallback from root/pelvis
        root = self._j("root", "pelvis")
        if root is not None:
            return root + np.array([0.0, 0.0, -0.13])
        return np.array([0.0, 0.0, -0.13])


# ---------------------------------------------------------------------------
# Main Measurement Function
# ---------------------------------------------------------------------------

def measure_body(
    output,
    batch_idx: int = 0,
    joints: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
    raw_dict: Optional[dict] = None,
) -> Dict[str, Measurement]:
    """
    Compute all body measurements on a rest-pose mesh or BodyOutput.

    Parameters
    ----------
    output : BodyOutput or np.ndarray or torch.Tensor
        Either a BodyOutput instance, or vertex array (V, 3) in metres.
    batch_idx : int
        Batch element index if output is BodyOutput.
    joints : Optional[np.ndarray]
        Joint coordinates (J, 3) in metres if output is vertices.
    faces : Optional[np.ndarray]
        Triangle faces (F, 3) if output is vertices.
    joint_names : Optional[List[str]]
        Names of joints if output is vertices.
    raw_dict : Optional[dict]
        Additional metadata (e.g. skinning weights) if output is vertices.

    Returns
    -------
    dict[str, Measurement]
    """
    if hasattr(output, "numpy_vertices"):
        verts = output.numpy_vertices(batch_idx)   # (V, 3) metres
        jts = output.numpy_joints(batch_idx)       # (J, 3) metres
        if hasattr(output, "numpy_faces"):
            fcs = output.numpy_faces()
        else:
            fcs = output.faces.cpu().numpy() if hasattr(output.faces, "cpu") else np.asarray(output.faces)
        jnames = output.joint_names
        raw = output.raw if hasattr(output, "raw") else {}
        return measure_mesh(verts, jts, fcs, jnames, raw)
    else:
        # Direct vertex input
        verts = output.cpu().numpy() if hasattr(output, "cpu") else np.asarray(output)
        jts = joints.cpu().numpy() if hasattr(joints, "cpu") else (np.asarray(joints) if joints is not None else np.zeros((0, 3)))
        fcs = faces.cpu().numpy() if hasattr(faces, "cpu") else (np.asarray(faces) if faces is not None else np.zeros((0, 3), dtype=np.int64))
        jnames = joint_names or []
        raw = raw_dict or {}
        return measure_mesh(verts, jts, fcs, jnames, raw)


def measure_mesh(
    verts: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    jnames: Optional[List[str]] = None,
    raw_dict: Optional[dict] = None,
) -> Dict[str, Measurement]:
    """
    Compute all body measurements directly on vertices, joints, and faces.
    """
    jnames = jnames or []
    raw_dict = raw_dict or {}
    lm = LandmarkHeights(joints, jnames, verts)
    results: Dict[str, Measurement] = {}

    def _add(name: str, value_mm: float, method: str, notes: str = "") -> None:
        results[name] = Measurement(
            name=name,
            value_mm=value_mm,
            method=method,
            source="model",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Pre-clean torso mesh by excluding hanging arm/hand vertices
    # ------------------------------------------------------------------
    torso_verts, torso_faces = _get_torso_mesh(verts, faces, raw_dict)

    # ==================================================================
    # Set A: Whole-body measurements
    # ==================================================================

    # 1. Stature
    stature_mm = (lm.head_top_z() - lm.floor_z()) * 1000.0
    _add("stature", stature_mm, "max_vertex_z - min_vertex_z",
         "Crown to heel along Z axis.")

    # 2. Head circumference
    hc_z = lm.head_circ_z()
    hc_loops = _slice_mesh_at_z(verts, faces, hc_z)
    hc_loop = max(hc_loops, key=_loop_perimeter) if hc_loops else None
    hc_mm = _loop_perimeter(hc_loop) * 1000.0 if hc_loop is not None else float("nan")
    _add("head_circumference", hc_mm, "surface_path_loop",
         f"Plane Z={hc_z*1000:.1f} mm (~60 mm below crown); surface path.")

    # 3. Neck base circumference
    nb_z = lm.neck_base_z()
    nb_loops = _slice_mesh_at_z(torso_verts, torso_faces, nb_z)
    # Select loop enclosing neck axis (near X=0, Y=neck_y)
    neck_pt = lm._j("neck01", "neck_01", "Neck")
    neck_y = float(neck_pt[1]) if neck_pt is not None else 0.0
    nb_candidates = [
        lp for lp in nb_loops
        if np.linalg.norm(np.mean(lp[:, :2], axis=0) - np.array([0.0, neck_y])) < 0.15
        and len(lp) >= 3
    ]
    nb_loop = min(nb_candidates, key=_loop_perimeter) if nb_candidates else (
        min(nb_loops, key=_loop_perimeter) if nb_loops else None
    )
    nb_mm = _loop_perimeter(nb_loop) * 1000.0 if nb_loop is not None else float("nan")
    _add("neck_base_circumference", nb_mm, "surface_path_loop",
         f"Plane Z={nb_z*1000:.1f} mm (superior thoracic aperture); surface path.")

    # 4. Chest circumference
    # Level: mid-sternum / pectoral line. In anny: spine01 (upper thoracic)
    sp01 = lm._j("spine01")
    sp02 = lm._j("spine02")
    if sp01 is not None and sp02 is not None:
        chest_z = float(sp02[2] + 0.85 * (sp01[2] - sp02[2]))
    elif sp01 is not None:
        chest_z = float(sp01[2] - 0.02)
    else:
        chest_z = lm.floor_z() + (lm.head_top_z() - lm.floor_z()) * 0.55

    chest_loops = _slice_mesh_at_z(torso_verts, torso_faces, chest_z)
    chest_loop = _select_torso_loop(chest_loops, origin_xy=(0.0, float(sp01[1]) if sp01 is not None else 0.0))
    if chest_loop is None:
        # Fallback to watertight whole-body mesh if axillary filtering punched a hole in the chest wall
        chest_loops = _slice_mesh_at_z(verts, faces, chest_z)
        chest_loop = _select_torso_loop(chest_loops, origin_xy=(0.0, float(sp01[1]) if sp01 is not None else 0.0))
    chest_mm = _loop_perimeter(chest_loop) * 1000.0 if chest_loop is not None else float("nan")
    _add("chest_circumference", chest_mm, "surface_path_loop_torso",
         f"Plane Z={chest_z*1000:.1f} mm (mid-sternum/spine01); arm vertices excluded; surface path.")

    # 5. Waist circumference
    # Narrowest trunk circumference in lumbar region (spine04 to spine02)
    sp04 = lm._j("spine04")
    if sp04 is not None and sp02 is not None:
        waist_eval_zs = np.linspace(float(sp04[2]), float(sp02[2]), 9)
    else:
        stature_m = lm.head_top_z() - lm.floor_z()
        waist_eval_zs = np.linspace(lm.floor_z() + stature_m * 0.40, lm.floor_z() + stature_m * 0.48, 9)

    waist_candidates = []
    for z_w in waist_eval_zs:
        w_loops = _slice_mesh_at_z(torso_verts, torso_faces, z_w)
        w_lp = _select_torso_loop(w_loops, origin_xy=(0.0, 0.0))
        if w_lp is not None and len(w_lp) >= 3:
            waist_candidates.append(_loop_perimeter(w_lp) * 1000.0)

    waist_mm = min(waist_candidates) if waist_candidates else float("nan")
    _add("waist_circumference", waist_mm, "surface_path_loop_torso_min",
         "Minimum trunk circumference in lumbar region (spine04 to spine02); arms excluded; surface path.")

    # 6. Hip circumference
    # Maximum trunk circumference around root / greater trochanters
    root_j = lm._j("root", "pelvis")
    root_z = float(root_j[2]) if root_j is not None else 0.0
    hip_eval_zs = np.linspace(root_z - 0.04, root_z + 0.06, 7)
    hip_candidates = []
    for z_h in hip_eval_zs:
        h_loops = _slice_mesh_at_z(torso_verts, torso_faces, z_h)
        h_lp = _select_torso_loop(h_loops, origin_xy=(0.0, 0.0))
        if h_lp is not None and len(h_lp) >= 3:
            hip_candidates.append(_loop_perimeter(h_lp) * 1000.0)

    hip_mm = max(hip_candidates) if hip_candidates else float("nan")
    _add("hip_circumference", hip_mm, "surface_path_loop_torso_max",
         "Maximum hip circumference around pelvis/buttocks; arms excluded; surface path.")

    # 7. Crotch height
    crotch_pt = lm.crotch_landmark()
    crotch_mm = float(crotch_pt[2] - lm.floor_z()) * 1000.0
    _add("crotch_height", crotch_mm, "crotch_surface_to_floor",
         f"Perineum landmark Z={crotch_pt[2]*1000:.1f} mm to floor Z={lm.floor_z()*1000:.1f} mm.")

    # 8. Ankle circumference (Set A = mean of left and right perpendicular ankle slices)
    ank_circumferences = []
    for side in ["L", "R"]:
        kn = lm._j(f"lowerleg01.{side}", f"LowerLeg.{side}")
        ft = lm._j(f"foot.{side}", f"ankle.{side}", f"Foot.{side}")
        if kn is not None and ft is not None:
            # 50 mm above foot joint along lower leg axis
            axis_leg = ft - kn
            axis_len = np.linalg.norm(axis_leg)
            ank_pt = ft - (0.050 / axis_len) * axis_leg if axis_len > 0.050 else ft
            c_ank = _measure_limb_circumference(verts, faces, ank_pt, axis_leg, max_radius=0.15)
            if not math.isnan(c_ank) and c_ank > 0:
                ank_circumferences.append(c_ank)

    ankle_mm = float(np.mean(ank_circumferences)) if ank_circumferences else float("nan")
    _add("ankle_circumference", ankle_mm, "perpendicular_limb_mean",
         "Mean of left and right ankle circumferences sliced perpendicular to tibia; surface path.")

    # 9. Shoulder breadth (biacromial breadth)
    # 3D Euclidean distance between left and right acromion surface landmarks
    acromion_l = lm.acromion_landmark("L")
    acromion_r = lm.acromion_landmark("R")
    if acromion_l is not None and acromion_r is not None:
        sb_mm = float(np.linalg.norm(acromion_r - acromion_l) * 1000.0)
        _add("shoulder_breadth", sb_mm, "3d_biacromial_distance",
             "3D Euclidean distance between left and right acromion surface landmarks (biacromial breadth).")
    else:
        _add("shoulder_breadth", float("nan"), "3d_biacromial_distance",
             "Acromion surface landmarks unavailable; shoulder breadth cannot be determined.")

    # 10. Shoulder–elbow length (Bilateral)
    # Left:
    el_l = lm._j("lowerarm01.L", "forearm.L", "LowerArm.L", "Elbow.L", "elbow.L")
    se_l_mm = float(np.linalg.norm(el_l - acromion_l) * 1000.0) if (el_l is not None and acromion_l is not None) else float("nan")
    _add("shoulder_to_elbow_length_left", se_l_mm, "3d_acromion_to_elbow",
         "3D distance from left acromion surface landmark to left elbow joint.")
    _add("shoulder_elbow_length", se_l_mm, "3d_acromion_to_elbow",
         "Legacy alias for shoulder_to_elbow_length_left (left arm).")

    # Right:
    el_r = lm._j("lowerarm01.R", "forearm.R", "LowerArm.R", "Elbow.R", "elbow.R")
    se_r_mm = float(np.linalg.norm(el_r - acromion_r) * 1000.0) if (el_r is not None and acromion_r is not None) else float("nan")
    _add("shoulder_to_elbow_length_right", se_r_mm, "3d_acromion_to_elbow",
         "3D distance from right acromion surface landmark to right elbow joint.")

    # 11. Shoulder–wrist length (Bilateral)
    # Left:
    wr_l = lm._j("wrist.L", "hand.L", "Wrist.L", "Hand.L")
    if el_l is not None and wr_l is not None and acromion_l is not None:
        sw_l_mm = float((np.linalg.norm(el_l - acromion_l) + np.linalg.norm(wr_l - el_l)) * 1000.0)
    else:
        sw_l_mm = float("nan")
    _add("shoulder_to_wrist_length_left", sw_l_mm, "3d_acromion_elbow_wrist_chain",
         "3D kinematic distance from left acromion to elbow + elbow to wrist (left arm).")
    _add("shoulder_wrist_length", sw_l_mm, "3d_acromion_elbow_wrist_chain",
         "Legacy alias for shoulder_to_wrist_length_left (left arm).")

    # Right:
    wr_r = lm._j("wrist.R", "hand.R", "Wrist.R", "Hand.R")
    if el_r is not None and wr_r is not None and acromion_r is not None:
        sw_r_mm = float((np.linalg.norm(el_r - acromion_r) + np.linalg.norm(wr_r - el_r)) * 1000.0)
    else:
        sw_r_mm = float("nan")
    _add("shoulder_to_wrist_length_right", sw_r_mm, "3d_acromion_elbow_wrist_chain",
         "3D kinematic distance from right acromion to elbow + elbow to wrist (right arm).")

    # 12. Inseam
    # Single midline-to-distal leg anthropometric measurement: crotch landmark → knee → ankle
    kn_l = lm._j("lowerleg01.L", "shin.L", "LowerLeg.L")
    ft_l = lm._j("foot.L", "ankle.L", "Foot.L")
    if kn_l is not None and ft_l is not None and crotch_pt is not None:
        inseam_mm = float((np.linalg.norm(kn_l - crotch_pt) + np.linalg.norm(ft_l - kn_l)) * 1000.0)
    else:
        inseam_mm = float("nan")
    _add("inseam", inseam_mm, "3d_crotch_knee_ankle_chain",
         "3D distance along leg from midline perineum crotch landmark to knee to ankle (left leg path).")

    # ==================================================================
    # Set B: Limb circumferences (Left & Right separate, NO symmetry prior)
    # Sliced strictly perpendicular to the segment axis
    # ==================================================================

    for side_code, side_label in [("L", "left"), ("R", "right")]:
        sh = lm._j(f"upperarm01.{side_code}", f"shoulder.{side_code}", f"UpperArm.{side_code}")
        el = lm._j(f"lowerarm01.{side_code}", f"forearm.{side_code}", f"LowerArm.{side_code}")
        wr = lm._j(f"wrist.{side_code}", f"hand.{side_code}", f"Wrist.{side_code}", f"Hand.{side_code}")

        hip_j = lm._j(f"upperleg01.{side_code}", f"pelvis.{side_code}", f"UpperLeg.{side_code}")
        kn_j  = lm._j(f"lowerleg01.{side_code}", f"shin.{side_code}", f"LowerLeg.{side_code}")
        ft_j  = lm._j(f"foot.{side_code}", f"ankle.{side_code}", f"Foot.{side_code}")

        # B1. Upper arm circumference: midpoint of shoulder-elbow
        if sh is not None and el is not None:
            p_mid_ua = 0.5 * (sh + el)
            ua_mm = _measure_limb_circumference(verts, faces, p_mid_ua, el - sh, max_radius=0.20)
        else:
            ua_mm = float("nan")
        _add(f"upper_arm_circumference_{side_label}", ua_mm,
             "perpendicular_limb_slice",
             f"Midpoint of {side_label} shoulder-elbow segment, sliced perpendicular to bone axis.")

        # B2. Forearm circumference: maximum near elbow end (proximal 15–40%)
        if el is not None and wr is not None:
            fa_axis = wr - el
            t_evals = np.linspace(0.15, 0.40, 11)
            fa_candidates = [
                _measure_limb_circumference(verts, faces, el + t * fa_axis, fa_axis, max_radius=0.18)
                for t in t_evals
            ]
            valid_fa = [v for v in fa_candidates if not math.isnan(v) and v > 0]
            fa_mm = max(valid_fa) if valid_fa else float("nan")
        else:
            fa_mm = float("nan")
        _add(f"forearm_circumference_{side_label}", fa_mm,
             "perpendicular_limb_slice_max",
             f"Maximum circumference along proximal 15–40% of {side_label} forearm perpendicular to axis.")

        # B3. Mid-thigh circumference: midpoint of hip-knee
        if hip_j is not None and kn_j is not None:
            p_mid_th = 0.5 * (hip_j + kn_j)
            th_mm = _measure_limb_circumference(verts, faces, p_mid_th, kn_j - hip_j, max_radius=0.30)
        else:
            th_mm = float("nan")
        _add(f"mid_thigh_circumference_{side_label}", th_mm,
             "perpendicular_limb_slice",
             f"Midpoint of {side_label} hip-knee segment, sliced perpendicular to femur axis.")

        # B4. Maximum calf circumference: maximum along lower leg (proximal 20–50%)
        if kn_j is not None and ft_j is not None:
            calf_axis = ft_j - kn_j
            t_evals = np.linspace(0.20, 0.50, 11)
            calf_candidates = [
                _measure_limb_circumference(verts, faces, kn_j + t * calf_axis, calf_axis, max_radius=0.22)
                for t in t_evals
            ]
            valid_calf = [v for v in calf_candidates if not math.isnan(v) and v > 0]
            calf_mm = max(valid_calf) if valid_calf else float("nan")
        else:
            calf_mm = float("nan")
        _add(f"max_calf_circumference_{side_label}", calf_mm,
             "perpendicular_limb_slice_max",
             f"Maximum circumference along 20–50% of {side_label} lower leg perpendicular to tibia axis.")

    logger.debug("measure_body computed %d measurements.", len(results))
    return results


# ---------------------------------------------------------------------------
# Plausibility Checker
# ---------------------------------------------------------------------------

def check_measurement_plausibility(meas: Dict[str, Measurement]) -> List[str]:
    """
    Validate anatomical plausibility of body measurements against broad adult limits.
    Returns a list of warning strings (empty if all pass).
    """
    warnings = []

    def _val(k):
        m = meas.get(k)
        return m.value_mm if m and m.is_valid() else None

    stature = _val("stature")
    chest = _val("chest_circumference")
    waist = _val("waist_circumference")
    hip = _val("hip_circumference")

    if waist and hip and hip > 0:
        whr = waist / hip
        if not (0.55 <= whr <= 1.15):
            warnings.append(f"Waist/Hip ratio {whr:.2f} outside broad adult limits [0.55, 1.15]")

    if chest and hip and hip > 0:
        chr_val = chest / hip
        if not (0.70 <= chr_val <= 1.30):
            warnings.append(f"Chest/Hip ratio {chr_val:.2f} outside broad adult limits [0.70, 1.30]")

    if waist and chest and waist > 1.25 * chest:
        warnings.append(f"Waist {waist:.1f} mm exceeds chest {chest:.1f} mm by > 25%")

    if waist and hip and waist > 1.25 * hip:
        warnings.append(f"Waist {waist:.1f} mm exceeds hip {hip:.1f} mm by > 25%")

    sw_len = _val("shoulder_wrist_length")
    if sw_len and stature and stature > 0:
        arm_ratio = sw_len / stature
        if not (0.28 <= arm_ratio <= 0.42):
            warnings.append(f"Shoulder-wrist length / stature ratio {arm_ratio:.3f} outside expected [0.28, 0.42]")

    return warnings


# ---------------------------------------------------------------------------
# High-Resolution Visual Overlays
# ---------------------------------------------------------------------------

def render_measurement_overlays(
    output,
    measurements: Dict[str, Measurement],
    out_dir: Path,
    batch_idx: int = 0,
    prefix: str = "overlay",
) -> List[Path]:
    """
    Render high-resolution front, side, and arm close-up overlay PNGs with every
    measurement ring clearly labelled.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; skipping overlay renders.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts = output.numpy_vertices(batch_idx)   # (V, 3) metres
    faces = output.faces.cpu().numpy()
    joints = output.numpy_joints(batch_idx)
    jnames = output.joint_names

    lm = LandmarkHeights(joints, jnames, verts)
    saved: List[Path] = []

    # Ring color mapping
    colors = {
        "head_circumference":            "#e74c3c",
        "neck_base_circumference":       "#e67e22",
        "chest_circumference":           "#f1c40f",
        "waist_circumference":           "#2ecc71",
        "hip_circumference":             "#1abc9c",
        "ankle_circumference":           "#3498db",
        "upper_arm_circumference_left":  "#9b59b6",
        "upper_arm_circumference_right": "#8e44ad",
        "forearm_circumference_left":    "#2980b9",
        "forearm_circumference_right":   "#1a6fa0",
        "mid_thigh_circumference_left":  "#27ae60",
        "mid_thigh_circumference_right": "#1e8449",
        "max_calf_circumference_left":   "#d35400",
        "max_calf_circumference_right":  "#c0392b",
    }

    # Generate ring curves for display
    rings = []

    # Torso rings
    torso_verts, torso_faces = _get_torso_mesh(verts, faces, output.raw if hasattr(output, "raw") else {})
    for name, z_fn in [
        ("head_circumference", lambda: lm.head_circ_z()),
        ("neck_base_circumference", lambda: lm.neck_base_z()),
        ("chest_circumference", lambda: lm.chest_z() if hasattr(lm, "chest_z") else (lm._j("spine02")[2] + 0.85*(lm._j("spine01")[2] - lm._j("spine02")[2]))),
    ]:
        z_val = z_fn()
        lps = _slice_mesh_at_z(verts if "head" in name or "neck" in name else torso_verts,
                               faces if "head" in name or "neck" in name else torso_faces, z_val)
        lp = max(lps, key=_loop_perimeter) if lps else None
        if lp is not None:
            rings.append((name, lp, colors.get(name, "#333333")))

    # Waist & Hip rings
    sp04 = lm._j("spine04")
    sp02 = lm._j("spine02")
    if sp04 is not None and sp02 is not None:
        w_zs = np.linspace(sp04[2], sp02[2], 9)
        best_w_lp = None
        min_p = float("inf")
        for zw in w_zs:
            lps = _slice_mesh_at_z(torso_verts, torso_faces, zw)
            lp = _select_torso_loop(lps)
            if lp is not None and _loop_perimeter(lp) < min_p:
                min_p = _loop_perimeter(lp)
                best_w_lp = lp
        if best_w_lp is not None:
            rings.append(("waist_circumference", best_w_lp, colors["waist_circumference"]))

    root_j = lm._j("root", "pelvis")
    root_z = float(root_j[2]) if root_j is not None else 0.0
    h_zs = np.linspace(root_z - 0.04, root_z + 0.06, 7)
    best_h_lp = None
    max_hp = 0.0
    for zh in h_zs:
        lps = _slice_mesh_at_z(torso_verts, torso_faces, zh)
        lp = _select_torso_loop(lps)
        if lp is not None and _loop_perimeter(lp) > max_hp:
            max_hp = _loop_perimeter(lp)
            best_h_lp = lp
    if best_h_lp is not None:
        rings.append(("hip_circumference", best_h_lp, colors["hip_circumference"]))

    # Limb rings (perpendicular)
    for side in ["L", "R"]:
        side_lbl = "left" if side == "L" else "right"
        sh = lm._j(f"upperarm01.{side}")
        el = lm._j(f"lowerarm01.{side}")
        wr = lm._j(f"wrist.{side}")
        hip = lm._j(f"upperleg01.{side}")
        kn = lm._j(f"lowerleg01.{side}")
        ft = lm._j(f"foot.{side}")

        if sh is not None and el is not None:
            lps = _slice_mesh_plane(verts, faces, 0.5 * (sh + el), el - sh)
            u, v = _orthonormal_basis(el - sh)
            encl = [lp for lp in lps if _point_in_polygon_2d(np.stack([(lp - 0.5*(sh+el)) @ u, (lp - 0.5*(sh+el)) @ v], axis=1))]
            if encl:
                rings.append((f"upper_arm_circumference_{side_lbl}", encl[0], colors[f"upper_arm_circumference_{side_lbl}"]))

        if el is not None and wr is not None:
            lps = _slice_mesh_plane(verts, faces, el + 0.25 * (wr - el), wr - el)
            u, v = _orthonormal_basis(wr - el)
            encl = [lp for lp in lps if _point_in_polygon_2d(np.stack([(lp - (el + 0.25*(wr-el))) @ u, (lp - (el + 0.25*(wr-el))) @ v], axis=1))]
            if encl:
                rings.append((f"forearm_circumference_{side_lbl}", encl[0], colors[f"forearm_circumference_{side_lbl}"]))

        if hip is not None and kn is not None:
            lps = _slice_mesh_plane(verts, faces, 0.5 * (hip + kn), kn - hip)
            u, v = _orthonormal_basis(kn - hip)
            encl = [lp for lp in lps if _point_in_polygon_2d(np.stack([(lp - 0.5*(hip+kn)) @ u, (lp - 0.5*(hip+kn)) @ v], axis=1))]
            if encl:
                rings.append((f"mid_thigh_circumference_{side_lbl}", encl[0], colors[f"mid_thigh_circumference_{side_lbl}"]))

        if kn is not None and ft is not None:
            lps = _slice_mesh_plane(verts, faces, kn + 0.33 * (ft - kn), ft - kn)
            u, v = _orthonormal_basis(ft - kn)
            encl = [lp for lp in lps if _point_in_polygon_2d(np.stack([(lp - (kn + 0.33*(ft-kn))) @ u, (lp - (kn + 0.33*(ft-kn))) @ v], axis=1))]
            if encl:
                rings.append((f"max_calf_circumference_{side_lbl}", encl[0], colors[f"max_calf_circumference_{side_lbl}"]))

    def _draw_canvas(ax, h_col: int, v_col: int, xlabel: str, ylabel: str, title: str, crop_box=None):
        sub_mask = np.ones(len(verts), dtype=bool)
        if crop_box:
            h_min, h_max, v_min, v_max = crop_box
            sub_mask = (verts[:, h_col] >= h_min) & (verts[:, h_col] <= h_max) & \
                       (verts[:, v_col] >= v_min) & (verts[:, v_col] <= v_max)

        ax.scatter(verts[sub_mask, h_col], verts[sub_mask, v_col],
                   s=0.25, c="#c0c0c0", alpha=0.30, rasterized=True)

        legend_patches = []
        for name, loop, col in rings:
            if crop_box:
                h_min, h_max, v_min, v_max = crop_box
                if not (np.any(loop[:, h_col] >= h_min) and np.any(loop[:, h_col] <= h_max) and
                        np.any(loop[:, v_col] >= v_min) and np.any(loop[:, v_col] <= v_max)):
                    continue

            # Draw 3D loop projected
            closed_lp = np.vstack([loop, loop[0]])
            ax.plot(closed_lp[:, h_col], closed_lp[:, v_col], "-", color=col, linewidth=1.5, alpha=0.9)
            val = measurements.get(name)
            val_str = f"{val.value_mm:.1f} mm" if val and val.is_valid() else ""
            label_text = f"{name.replace('_', ' ')}: {val_str}"
            legend_patches.append(mpatches.Patch(color=col, label=label_text))

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        if legend_patches:
            ax.legend(handles=legend_patches, loc="lower right", fontsize=6, framealpha=0.85, ncol=1)

    # 1. Front View (X vs Z)
    fig, ax = plt.subplots(figsize=(7, 14), dpi=200)
    _draw_canvas(ax, 0, 2, "X / m  (Subject Left →)", "Z / m  (Height)", "Body Measurement Rings — Front View")
    fig.tight_layout()
    p_front = out_dir / f"{prefix}_front.png"
    fig.savefig(str(p_front))
    plt.close(fig)
    saved.append(p_front)

    # 2. Side View (Y vs Z)
    fig, ax = plt.subplots(figsize=(7, 14), dpi=200)
    _draw_canvas(ax, 1, 2, "Y / m  (Anterior / Forward)", "Z / m  (Height)", "Body Measurement Rings — Side View")
    fig.tight_layout()
    p_side = out_dir / f"{prefix}_side.png"
    fig.savefig(str(p_side))
    plt.close(fig)
    saved.append(p_side)

    # 3. Arm Close-Up (Left Arm: X ∈ [0.05, 0.60], Z ∈ [0.0, 0.65])
    fig, ax = plt.subplots(figsize=(8, 10), dpi=200)
    _draw_canvas(ax, 0, 2, "X / m (Lateral)", "Z / m (Height)",
                 "Upper Arm & Forearm Rings — Left Arm Close-up",
                 crop_box=(0.05, 0.60, 0.05, 0.60))
    fig.tight_layout()
    p_arm = out_dir / f"{prefix}_arm_closeup.png"
    fig.savefig(str(p_arm))
    plt.close(fig)
    saved.append(p_arm)

    logger.info("Saved %d high-res overlay images to %s", len(saved), out_dir)
    return saved
