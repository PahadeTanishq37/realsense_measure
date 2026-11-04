"""
3D face landmark detection for landmark-guided registration.

This is the only file in the project that imports mediapipe, so it can be
swapped out later without touching anything else (same pattern as
camera/realsense_capture.py being the only file that imports pyrealsense2).

Landmarks are used to compute a RIGID TRANSFORM between frames (see
registration.py's rigid_transform_from_correspondences and
register_frame_pair_landmark_guided) — never to move or snap the actual
captured points. The raw geometry stays exactly as captured; only the
frame-to-frame alignment uses landmarks as reliable correspondence points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import urllib.request
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from config import LandmarkConfig

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# A small, fixed set of stable landmark indices, out of mediapipe's full 468,
# chosen because they sit on rigid bone/cartilage (not lips/cheeks, which
# deform with expression) and are reliably detected across head angles.
LANDMARK_INDICES: dict[str, int] = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "nose_tip": 1,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "chin": 152,
    "nose_bridge": 168,
    "left_nostril": 49,
    "right_nostril": 279,
    "upper_lip_center": 0,
    "lower_lip_center": 17,
}

_landmarker = None  # lazy singleton -- model load is slow, do it once


def _ensure_model_downloaded() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not auto-download the face landmarker model ({e}). "
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {MODEL_PATH}"
        ) from e


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
        _ensure_model_downloaded()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_face_landmarks_3d(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic,
    depth_patch: int = 3,
    quiet: bool = False,
    cfg: LandmarkConfig | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Detect facial landmarks and back-project them to 3D camera-space meters with depth quality scoring.

    Parameters
    ----------
    color_bgr:
        Color frame as captured elsewhere in this project (BGR, OpenCV
        convention).
    depth_m:
        Depth frame in meters, same resolution/alignment as color_bgr.
    intrinsics_o3d:
        Camera intrinsics, as already built elsewhere in this project
        (camera/realsense_capture.py).
    depth_patch:
        Side length of the square patch used for a median depth read at
        each landmark, instead of a single noisy pixel.
    quiet:
        If True, suppress landmark detection print logging (used during live capture loop).
    cfg:
        Optional LandmarkConfig for variance and valid ratio quality thresholds.

    Returns
    -------
    dict mapping each name in LANDMARK_INDICES to a dict:
        {"position": [x, y, z], "valid_ratio": float, "variance": float, "low_confidence": bool}
    for every landmark that was both detected by mediapipe AND had valid (non-zero) depth nearby.
    Returns None if no face was detected in the frame at all.
    """
    import mediapipe as mp

    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]  # num_faces=1
    h, w = color_bgr.shape[:2]
    fx, fy = intrinsics_o3d.get_focal_length()
    cx, cy = intrinsics_o3d.get_principal_point()

    half = depth_patch // 2
    out: dict[str, dict[str, Any]] = {}
    max_var = getattr(cfg, "max_depth_variance_m2", 1e-5) if cfg is not None else 1e-5
    min_vr = getattr(cfg, "min_valid_ratio", 0.5) if cfg is not None else 0.5

    for name, idx in LANDMARK_INDICES.items():
        lm = face[idx]
        px = int(round(float(lm.x * w)))
        py = int(round(float(lm.y * h)))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        y0, y1 = max(0, py - half), min(h, py + half + 1)
        x0, x1 = max(0, px - half), min(w, px + half + 1)
        patch = depth_m[y0:y1, x0:x1]
        patch_size = patch.size

        z_center = depth_m[py, px]
        center_valid = bool(np.isfinite(z_center) and z_center > 0.0)

        valid_mask = np.isfinite(patch) & (patch > 0.0)
        valid = patch[valid_mask]
        used_patch_size = patch_size

        if valid.size == 0:
            # Expand search patch up to 7x7 if local pixel had a minor void/dropout
            y0_w, y1_w = max(0, py - 3), min(h, py + 4)
            x0_w, x1_w = max(0, px - 3), min(w, px + 4)
            wide_patch = depth_m[y0_w:y1_w, x0_w:x1_w]
            used_patch_size = wide_patch.size
            valid_mask = np.isfinite(wide_patch) & (wide_patch > 0.0)
            valid = wide_patch[valid_mask]
            if valid.size == 0:
                continue

        step_thresh = getattr(cfg, "max_surface_step_m", 0.03) if cfg is not None else 0.03
        valid_sorted = np.sort(valid)
        split_indices = np.where(np.diff(valid_sorted) > step_thresh)[0] + 1
        clusters = np.split(valid_sorted, split_indices)

        if not center_valid:
            # When center depth is invalid and patch contains only one depth cluster,
            # there is no reliable evidence of a distinct face surface (e.g. flat background wall) -> reject!
            if len(clusters) < 2:
                continue
            # When multiple clusters exist, require at least 2 pixels supporting the foreground cluster C_0
            if len(clusters[0]) < 2:
                continue
            surface_pixels = clusters[0]
            is_discontinuity = True
        else:
            if len(clusters) > 1:
                # Multiple clusters (depth discontinuity).
                # If center pixel belongs to a farther cluster rather than foreground C_0, reject ambiguous sample!
                if np.abs(float(z_center) - np.median(clusters[0])) > step_thresh:
                    continue
                surface_pixels = clusters[0]
                is_discontinuity = True
            else:
                surface_pixels = clusters[0]
                is_discontinuity = False

        total_px = used_patch_size
        valid_ratio = float(surface_pixels.size / total_px) if total_px > 0 else 0.0
        local_variance = float(np.var(surface_pixels)) if surface_pixels.size > 1 else 0.0
        low_confidence = bool(
            valid_ratio < min_vr
            or local_variance > max_var
            or not center_valid
            or is_discontinuity
            or used_patch_size > patch_size
        )

        z = float(np.median(surface_pixels))
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        pos = np.array([x, y, z], dtype=np.float64)

        out[name] = {
            "position": pos,
            "valid_ratio": valid_ratio,
            "variance": local_variance,
            "low_confidence": low_confidence,
        }

    if out and not quiet:
        n_low = sum(1 for v in out.values() if v["low_confidence"])
        low_msg = f" ({n_low} low-confidence/noisy)" if n_low > 0 else ""
        print(f"      [landmarks] Detected {len(out)}/{len(LANDMARK_INDICES)} landmarks{low_msg}: {list(out.keys())}")

    return out if out else None


def project_landmarks_3d_to_pixel(
    landmarks_3d: dict[str, Any],
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic,
) -> dict[str, np.ndarray]:
    """
    Project 3D camera-space landmark points back to (u, v) image coordinates.
    Accepts either raw [x, y, z] vectors or quality dicts with a 'position' key.
    """
    fx, fy = intrinsics_o3d.get_focal_length()
    cx, cy = intrinsics_o3d.get_principal_point()
    out: dict[str, np.ndarray] = {}
    for name, item in landmarks_3d.items():
        pt = item["position"] if isinstance(item, dict) and "position" in item else item
        if pt[2] > 0.01:
            u = (pt[0] * fx / pt[2]) + cx
            v = (pt[1] * fy / pt[2]) + cy
            out[name] = np.array([u, v], dtype=np.float32)
    return out


@dataclass
class LandmarkSample3D:
    """
    Structured 3D landmark observation preserving 2D pixel coordinates, raw sampled depth,
    metric camera position, and quality metrics.

    Distinguishes requested 2D detector landmark pixels from actual sensor depth source pixels,
    and differentiates estimated landmark coordinates from direct raw sensor observations.

    Attributes
    ----------
    name:
        Identifier of the anatomical landmark (e.g. 'nose_tip', 'chin', 'left_eye_outer').
    pixel_uv:
        Requested 2D detector landmark coordinate (px, py). (Backward-compatible alias).
    depth_m:
        Robust estimated depth in metres (0.0 if invalid/unavailable).
    point_3d:
        3D camera-space metric coordinate [X, Y, Z] in metres (None if unavailable).
        When estimated along the landmark ray: X = (px - cx)*z/fx, Y = (py - cy)*z/fy, Z = z.
    is_valid:
        True ONLY if the sample meets depth validity, step discontinuity, and variance criteria.
        Callers MUST check `is_valid` before using `point_3d` for measurement or fitting.
    point_index:
        0-based index of the point corresponding to point_3d in IndexedPointCloud.
        Must be -1 if point_3d is a synthetic patch-median estimate or does not exactly
        correspond to an indexed point.
    source_point_index:
        0-based index of the raw indexed observation corresponding to observed_point_3d
        in IndexedPointCloud. May be non-negative even when point_index == -1.
    landmark_pixel_uv:
        Requested 2D detector landmark coordinate (px, py) on the image grid.
    source_pixel_uv:
        Actual depth pixel coordinate(s) used for the observation.
        If the center pixel is valid, this is (px, py).
        If center depth was invalid and a neighbor fallback was used, this is the
        neighbor's (u_nb, v_nb) coordinate.
        None if out of bounds or no valid depth exists in the patch.
    has_point_3d:
        True if a 3D coordinate could be computed (point_3d is not None), independently
        of validity.
    is_observed_point:
        True ONLY when point_3d is a direct observation and its coordinate corresponds
        to the indexed source point within tolerance (<= 0.1mm).
    observed_point_3d:
        Exact sensor observation projected using its actual source pixel and raw depth.
    source_pixels:
        List of all (u, v) pixel coordinates that contributed to the foreground surface cluster.
    confidence_info:
        Quality diagnostics from depth patch sampling (valid_ratio, variance, low_confidence,
        is_discontinuity, error, etc.).
    """

    name: str
    pixel_uv: tuple[int, int]
    depth_m: float
    point_3d: np.ndarray | None
    is_valid: bool
    point_index: int = -1
    source_point_index: int = -1
    landmark_pixel_uv: tuple[int, int] | None = None
    source_pixel_uv: tuple[int, int] | None = None
    has_point_3d: bool = False
    is_observed_point: bool = False
    observed_point_3d: np.ndarray | None = None
    source_pixels: list[tuple[int, int]] = field(default_factory=list)
    confidence_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.landmark_pixel_uv is None:
            self.landmark_pixel_uv = self.pixel_uv
        if self.pixel_uv is None:
            self.pixel_uv = self.landmark_pixel_uv
        self.has_point_3d = self.point_3d is not None


def sample_landmark_3d(
    pixel_uv: tuple[int, int],
    depth_m: np.ndarray,
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic,
    name: str = "custom",
    depth_patch: int = 3,
    cfg: LandmarkConfig | None = None,
    indexed_pcd: Any | None = None,
    frame_id: int | None = None,
) -> LandmarkSample3D:
    """
    Sample 3D position and depth quality at a 2D landmark pixel using raw depth map patch statistics.

    Maintains complete consistency with detect_face_landmarks_3d by performing identical
    multi-modal surface clustering, step discontinuity rejection, and variance checks.
    Distinguishes requested landmark ray estimates from genuine raw sensor observations.

    Parameters
    ----------
    pixel_uv:
        (u, v) integer pixel coordinates on the RGB image grid.
    depth_m:
        HxW float32 metric depth array.
    intrinsics_o3d:
        Camera pinhole intrinsics.
    name:
        Landmark name identifier.
    depth_patch:
        Square patch size for depth sampling.
    cfg:
        Optional LandmarkConfig thresholds.
    indexed_pcd:
        Optional IndexedPointCloud or RGBDFrame to retrieve point index.
    frame_id:
        Frame ID for indexed_pcd query.

    Returns
    -------
    LandmarkSample3D
    """
    px, py = int(round(pixel_uv[0])), int(round(pixel_uv[1]))
    h, w = depth_m.shape[:2]
    if px < 0 or px >= w or py < 0 or py >= h:
        return LandmarkSample3D(
            name=name,
            pixel_uv=(px, py),
            depth_m=0.0,
            point_3d=None,
            is_valid=False,
            point_index=-1,
            source_point_index=-1,
            landmark_pixel_uv=(px, py),
            source_pixel_uv=None,
            has_point_3d=False,
            is_observed_point=False,
            observed_point_3d=None,
            source_pixels=[],
            confidence_info={"error": "out_of_bounds"},
        )

    fx, fy = intrinsics_o3d.get_focal_length()
    cx, cy = intrinsics_o3d.get_principal_point()

    half = depth_patch // 2
    y0, y1 = max(0, py - half), min(h, py + half + 1)
    x0, x1 = max(0, px - half), min(w, px + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    patch_size = patch.size
    yy, xx = np.mgrid[y0:y1, x0:x1]

    z_center = depth_m[py, px]
    center_valid = bool(np.isfinite(z_center) and z_center > 0.0)

    valid_mask = np.isfinite(patch) & (patch > 0.0)
    valid = patch[valid_mask]
    valid_xs = xx[valid_mask]
    valid_ys = yy[valid_mask]
    used_patch_size = patch_size

    if valid.size == 0:
        y0_w, y1_w = max(0, py - 3), min(h, py + 4)
        x0_w, x1_w = max(0, px - 3), min(w, px + 4)
        wide_patch = depth_m[y0_w:y1_w, x0_w:x1_w]
        yy_w, xx_w = np.mgrid[y0_w:y1_w, x0_w:x1_w]
        used_patch_size = wide_patch.size
        valid_mask = np.isfinite(wide_patch) & (wide_patch > 0.0)
        valid = wide_patch[valid_mask]
        valid_xs = xx_w[valid_mask]
        valid_ys = yy_w[valid_mask]
        if valid.size == 0:
            return LandmarkSample3D(
                name=name,
                pixel_uv=(px, py),
                depth_m=0.0,
                point_3d=None,
                is_valid=False,
                point_index=-1,
                source_point_index=-1,
                landmark_pixel_uv=(px, py),
                source_pixel_uv=None,
                has_point_3d=False,
                is_observed_point=False,
                observed_point_3d=None,
                source_pixels=[],
                confidence_info={"error": "no_valid_depth_in_patch"},
            )

    step_thresh = getattr(cfg, "max_surface_step_m", 0.03) if cfg is not None else 0.03
    max_var = getattr(cfg, "max_depth_variance_m2", 1e-5) if cfg is not None else 1e-5
    min_vr = getattr(cfg, "min_valid_ratio", 0.5) if cfg is not None else 0.5

    order = np.argsort(valid)
    valid_sorted = valid[order]
    sorted_xs = valid_xs[order]
    sorted_ys = valid_ys[order]

    split_indices = np.where(np.diff(valid_sorted) > step_thresh)[0] + 1
    clusters_depth = np.split(valid_sorted, split_indices)
    clusters_x = np.split(sorted_xs, split_indices)
    clusters_y = np.split(sorted_ys, split_indices)

    surface_depths = clusters_depth[0]
    surface_xs = clusters_x[0]
    surface_ys = clusters_y[0]
    source_pixels = [(int(x), int(y)) for x, y in zip(surface_xs, surface_ys)]

    if not center_valid:
        if len(surface_depths) < 2:
            return LandmarkSample3D(
                name=name,
                pixel_uv=(px, py),
                depth_m=0.0,
                point_3d=None,
                is_valid=False,
                point_index=-1,
                source_point_index=-1,
                landmark_pixel_uv=(px, py),
                source_pixel_uv=None,
                has_point_3d=False,
                is_observed_point=False,
                observed_point_3d=None,
                source_pixels=source_pixels,
                confidence_info={"error": "invalid_center_depth_and_insufficient_surface"},
            )
        is_discontinuity = True
    else:
        if len(clusters_depth) > 1:
            if np.abs(float(z_center) - np.median(surface_depths)) > step_thresh:
                return LandmarkSample3D(
                    name=name,
                    pixel_uv=(px, py),
                    depth_m=0.0,
                    point_3d=None,
                    is_valid=False,
                    point_index=-1,
                    source_point_index=-1,
                    landmark_pixel_uv=(px, py),
                    source_pixel_uv=(px, py),
                    has_point_3d=False,
                    is_observed_point=False,
                    observed_point_3d=None,
                    source_pixels=source_pixels,
                    confidence_info={"error": "center_on_background_cluster"},
                )
            is_discontinuity = True
        else:
            is_discontinuity = False

    total_px = used_patch_size
    valid_ratio = float(surface_depths.size / total_px) if total_px > 0 else 0.0
    local_variance = float(np.var(surface_depths)) if surface_depths.size > 1 else 0.0
    low_confidence = bool(
        valid_ratio < min_vr
        or local_variance > max_var
        or not center_valid
        or is_discontinuity
        or used_patch_size > patch_size
    )

    z = float(np.median(surface_depths))
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    pos = np.array([x, y, z], dtype=np.float64)

    is_valid = not low_confidence

    # Determine source pixel, observed point, and indexed point associations
    if center_valid:
        source_uv = (px, py)
        # Check if center pixel exists in indexed_pcd
        src_p_idx = -1
        if indexed_pcd is not None and hasattr(indexed_pcd, "get_point_index_at_pixel"):
            src_p_idx = indexed_pcd.get_point_index_at_pixel(px, py, frame_id=frame_id, unambiguous_only=True)

        if src_p_idx >= 0 and indexed_pcd is not None:
            # Genuine 1-to-1 sensor observation exists in the point cloud
            obs_pt = np.asarray(indexed_pcd.points[src_p_idx], dtype=np.float64)
            is_exact_obs = bool(surface_depths.size == 1 or np.isclose(z, float(z_center), atol=1e-4))
            source_pt_idx = src_p_idx
        else:
            obs_x = (px - cx) * float(z_center) / fx
            obs_y = (py - cy) * float(z_center) / fy
            obs_pt = np.array([obs_x, obs_y, float(z_center)], dtype=np.float64)
            is_exact_obs = False if indexed_pcd is not None else bool(surface_depths.size == 1 or np.isclose(z, float(z_center), atol=1e-4))
            source_pt_idx = -1

        # point_index strictly corresponds to point_3d:
        # Must be -1 if point_3d is a synthetic patch-median estimate or if sample failed quality checks
        if is_exact_obs and is_valid and source_pt_idx >= 0:
            pt_idx = source_pt_idx
        else:
            pt_idx = -1
    else:
        # Center was invalid; fallback to nearest valid neighbor in foreground surface cluster
        dists_sq = (surface_xs - px) ** 2 + (surface_ys - py) ** 2
        nearest_i = int(np.argmin(dists_sq))
        nb_u = int(surface_xs[nearest_i])
        nb_v = int(surface_ys[nearest_i])
        source_uv = (nb_u, nb_v)
        nb_z = float(surface_depths[nearest_i])

        # Project observed point using the neighbour's own pixel coordinates
        nb_x = (nb_u - cx) * nb_z / fx
        nb_y = (nb_v - cy) * nb_z / fy
        obs_pt = np.array([nb_x, nb_y, nb_z], dtype=np.float64)

        # Look up neighbour's point in indexed_pcd if available
        source_pt_idx = -1
        if indexed_pcd is not None and hasattr(indexed_pcd, "get_point_index_at_pixel"):
            source_pt_idx = indexed_pcd.get_point_index_at_pixel(nb_u, nb_v, frame_id=frame_id, unambiguous_only=True)

        # point_index for the landmark ray is strictly -1 (landmark ray was not directly observed)
        pt_idx = -1
        is_exact_obs = False

    return LandmarkSample3D(
        name=name,
        pixel_uv=(px, py),
        depth_m=z,
        point_3d=pos,
        is_valid=is_valid,
        point_index=pt_idx,
        source_point_index=source_pt_idx,
        landmark_pixel_uv=(px, py),
        source_pixel_uv=source_uv,
        has_point_3d=True,
        is_observed_point=is_exact_obs and is_valid,
        observed_point_3d=obs_pt,
        source_pixels=source_pixels,
        confidence_info={
            "valid_ratio": valid_ratio,
            "variance": local_variance,
            "low_confidence": low_confidence,
            "is_discontinuity": is_discontinuity,
        },
    )
