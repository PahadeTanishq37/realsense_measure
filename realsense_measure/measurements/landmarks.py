"""
realsense_measure/measurements/landmarks.py
===========================================
Stage 17: Anatomical Landmark Extractor & 3D Projector.

Extracts anatomical landmarks from RGB-D frames using MediaPipe PoseLandmarker
and calibrated pinhole intrinsics.

CRITICAL RESEARCH GUIDELINE:
All landmarks produced by machine learning detectors are labeled explicitly as
'ESTIMATED_2D_3D_LANDMARK'. They must NEVER be claimed as ground-truth anatomical
markers. Missing, truncated, or infrared-absorbed landmarks are reported as
unavailable/truncated rather than fabricated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnatomicalLandmark:
    """Represents a single anatomical landmark with 2D and 3D coordinates."""
    landmark_id: str
    anatomical_name: str
    classification: str  # Strictly "ESTIMATED_2D_3D_LANDMARK" (never ground-truth)
    pixel_u: float
    pixel_v: float
    in_image_bounds: bool
    point_3d_camera_m: Optional[List[float]]  # [X, Y, Z] in metres in RealSense optical frame
    depth_m: Optional[float]
    depth_valid: bool
    depth_std_mm: float
    visibility_confidence: float
    status: str  # "VALID", "TRUNCATED_OUT_OF_BOUNDS", "DROPOUT_NO_DEPTH", "LOW_CONFIDENCE"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnatomicalLandmarksExtractor:
    """
    Extracts upper-body anatomical landmarks from aligned RGB-D frames.
    """

    # Model path
    DEFAULT_MODEL_PATH = Path("realsense_measure-main/realsense_measure/models/pose_landmarker_heavy.task")

    LANDMARK_DEFINITIONS = {
        "acromion_left": "Left Shoulder (Acromion / Glenohumeral joint)",
        "acromion_right": "Right Shoulder (Acromion / Glenohumeral joint)",
        "lateral_epicondyle_left": "Left Elbow (Lateral epicondyle)",
        "lateral_epicondyle_right": "Right Elbow (Lateral epicondyle)",
        "radial_styloid_left": "Left Wrist (Radial styloid)",
        "radial_styloid_right": "Right Wrist (Radial styloid)",
        "neck_base": "Neck Base (Suprasternale / Cervicale)",
        "mesosternale_chest": "Chest Reference (Mesosternale level)",
        "omphalion_waist": "Waist Reference (Omphalion level)",
        "axilla_left": "Left Anterior Axillary Fold",
        "axilla_right": "Right Anterior Axillary Fold",
        "hip_left": "Left Hip (Greater Trochanter reference)",
        "hip_right": "Right Hip (Greater Trochanter reference)",
    }

    def __init__(
        self,
        intrinsics: Dict[str, float],
        model_path: Optional[Path] = None,
        patch_radius_px: int = 10,
        min_depth_m: float = 0.30,
        max_depth_m: float = 3.50,
    ):
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["cx"])
        self.cy = float(intrinsics["cy"])
        self.width = int(intrinsics.get("width", 1280))
        self.height = int(intrinsics.get("height", 720))
        self.patch_radius = int(patch_radius_px)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            # Try alternate repo path
            alt = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_heavy.task"
            if alt.exists():
                self.model_path = alt

    def extract_landmarks(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
        depth_std_m: Optional[np.ndarray] = None,
    ) -> Dict[str, AnatomicalLandmark]:
        """
        Executes landmark estimation, depth sampling, and 3D pinhole unprojection.
        """
        h, w = depth_m.shape
        raw_mp_landmarks = self._run_detector(color_bgr)
        landmarks: Dict[str, AnatomicalLandmark] = {}

        if raw_mp_landmarks is None:
            logger.warning("Pose detector failed to find landmarks in color frame.")
            for lid, aname in self.LANDMARK_DEFINITIONS.items():
                landmarks[lid] = AnatomicalLandmark(
                    landmark_id=lid,
                    anatomical_name=aname,
                    classification="ESTIMATED_2D_3D_LANDMARK",
                    pixel_u=0.0,
                    pixel_v=0.0,
                    in_image_bounds=False,
                    point_3d_camera_m=None,
                    depth_m=None,
                    depth_valid=False,
                    depth_std_mm=0.0,
                    visibility_confidence=0.0,
                    status="NOT_AVAILABLE",
                    notes="Detector found zero landmarks.",
                )
            return landmarks

        # Extract MediaPipe standard keypoint coordinates
        def get_mp(idx: int) -> Tuple[float, float, float, bool]:
            lm = raw_mp_landmarks[idx]
            u = lm.x * w
            v = lm.y * h
            in_bounds = (0.0 <= lm.x <= 1.0) and (0.0 <= lm.y <= 1.0)
            vis = float(getattr(lm, "visibility", 0.0) or 0.0)
            return u, v, vis, in_bounds

        # Standard MediaPipe indices
        # 11: left_shoulder, 12: right_shoulder
        # 13: left_elbow, 14: right_elbow
        # 15: left_wrist, 16: right_wrist
        # 23: left_hip, 24: right_hip
        u_ls, v_ls, vis_ls, b_ls = get_mp(11)
        u_rs, v_rs, vis_rs, b_rs = get_mp(12)
        u_le, v_le, vis_le, b_le = get_mp(13)
        u_re, v_re, vis_re, b_re = get_mp(14)
        u_lw, v_lw, vis_lw, b_lw = get_mp(15)
        u_rw, v_rw, vis_rw, b_rw = get_mp(16)
        u_lh, v_lh, vis_lh, b_lh = get_mp(23)
        u_rh, v_rh, vis_rh, b_rh = get_mp(24)

        # Derived anatomical reference points
        # 1. Neck base: midway between shoulders, elevated slightly toward cervicale
        u_neck = (u_ls + u_rs) / 2.0
        v_neck = min(v_ls, v_rs) - 0.05 * abs(v_ls - v_rs)
        vis_neck = min(vis_ls, vis_rs)
        b_neck = (0 <= u_neck < w) and (0 <= v_neck < h)

        # 2. Chest reference (mesosternale level): ~35% down trunk from shoulder mid
        sh_mid_v = (v_ls + v_rs) / 2.0
        hip_mid_v = (v_lh + v_rh) / 2.0 if (b_lh or b_rh) else min(h - 1.0, sh_mid_v + 250.0)
        v_chest = sh_mid_v + 0.35 * (hip_mid_v - sh_mid_v)
        u_chest = (u_ls + u_rs) / 2.0
        vis_chest = min(vis_ls, vis_rs)
        b_chest = (0 <= u_chest < w) and (0 <= v_chest < h)

        # 3. Waist reference (omphalion / narrowest waist): ~70% down trunk
        v_waist = sh_mid_v + 0.70 * (hip_mid_v - sh_mid_v)
        u_waist = (u_ls + u_rs) / 2.0
        vis_waist = min(vis_ls, vis_rs)
        b_waist = (0 <= u_waist < w) and (0 <= v_waist < h) and (b_lh or b_rh)

        # 4. Axillary regions: anterior axillary fold (slightly inferior-lateral to shoulder)
        u_ax_l = u_ls + 0.15 * (u_ls - u_rs)
        v_ax_l = v_ls + 0.20 * (hip_mid_v - sh_mid_v)
        b_ax_l = (0 <= u_ax_l < w) and (0 <= v_ax_l < h)

        u_ax_r = u_rs - 0.15 * (u_ls - u_rs)
        v_ax_r = v_rs + 0.20 * (hip_mid_v - sh_mid_v)
        b_ax_r = (0 <= u_ax_r < w) and (0 <= v_ax_r < h)

        keypoint_map = {
            "acromion_left": (u_ls, v_ls, vis_ls, b_ls),
            "acromion_right": (u_rs, v_rs, vis_rs, b_rs),
            "lateral_epicondyle_left": (u_le, v_le, vis_le, b_le),
            "lateral_epicondyle_right": (u_re, v_re, vis_re, b_re),
            "radial_styloid_left": (u_lw, v_lw, vis_lw, b_lw),
            "radial_styloid_right": (u_rw, v_rw, vis_rw, b_rw),
            "neck_base": (u_neck, v_neck, vis_neck, b_neck),
            "mesosternale_chest": (u_chest, v_chest, vis_chest, b_chest),
            "omphalion_waist": (u_waist, v_waist, vis_waist, b_waist),
            "axilla_left": (u_ax_l, v_ax_l, vis_ls * 0.9, b_ax_l),
            "axilla_right": (u_ax_r, v_ax_r, vis_rs * 0.9, b_ax_r),
            "hip_left": (u_lh, v_lh, vis_lh, b_lh),
            "hip_right": (u_rh, v_rh, vis_rh, b_rh),
        }

        for lid, (u_val, v_val, conf, in_bounds) in keypoint_map.items():
            aname = self.LANDMARK_DEFINITIONS[lid]
            if not in_bounds:
                landmarks[lid] = AnatomicalLandmark(
                    landmark_id=lid,
                    anatomical_name=aname,
                    classification="ESTIMATED_2D_3D_LANDMARK",
                    pixel_u=float(u_val),
                    pixel_v=float(v_val),
                    in_image_bounds=False,
                    point_3d_camera_m=None,
                    depth_m=None,
                    depth_valid=False,
                    depth_std_mm=0.0,
                    visibility_confidence=float(conf),
                    status="TRUNCATED_OUT_OF_BOUNDS",
                    notes="Landmark falls outside sensor camera frame bounds.",
                )
                continue

            # Sample depth in neighborhood
            u_i = int(np.clip(round(u_val), 0, w - 1))
            v_i = int(np.clip(round(v_val), 0, h - 1))

            r = self.patch_radius
            u_min, u_max = max(0, u_i - r), min(w, u_i + r + 1)
            v_min, v_max = max(0, v_i - r), min(h, v_i + r + 1)

            patch_d = depth_m[v_min:v_max, u_min:u_max]
            valid_mask = (patch_d >= self.min_depth_m) & (patch_d <= self.max_depth_m)
            if person_mask is not None:
                patch_p = person_mask[v_min:v_max, u_min:u_max]
                valid_mask = valid_mask & patch_p

            valid_depths = patch_d[valid_mask]

            if len(valid_depths) < 5:
                landmarks[lid] = AnatomicalLandmark(
                    landmark_id=lid,
                    anatomical_name=aname,
                    classification="ESTIMATED_2D_3D_LANDMARK",
                    pixel_u=float(u_val),
                    pixel_v=float(v_val),
                    in_image_bounds=True,
                    point_3d_camera_m=None,
                    depth_m=None,
                    depth_valid=False,
                    depth_std_mm=0.0,
                    visibility_confidence=float(conf),
                    status="DROPOUT_NO_DEPTH",
                    notes=f"Infrared depth dropout in neighborhood (only {len(valid_depths)} valid points in {2*r}x{2*r} patch).",
                )
                continue

            z_med = float(np.median(valid_depths))
            # Temporal / patch std
            if depth_std_m is not None:
                patch_std = depth_std_m[v_min:v_max, u_min:u_max][valid_mask]
                std_mm = float(np.median(patch_std) * 1000.0) if len(patch_std) > 0 else 0.0
            else:
                std_mm = float(np.std(valid_depths) * 1000.0)

            # Unproject to RealSense optical camera coordinates (metres)
            # +X right, +Y down, +Z forward
            x_m = (u_val - self.cx) * z_med / self.fx
            y_m = (v_val - self.cy) * z_med / self.fy
            p3d = [float(x_m), float(y_m), float(z_med)]

            status = "VALID" if (conf >= 0.50 and std_mm < 60.0) else "LOW_CONFIDENCE"
            note = "Valid anatomical landmark unprojected to metric 3D." if status == "VALID" else "Low confidence or high depth jitter."

            landmarks[lid] = AnatomicalLandmark(
                landmark_id=lid,
                anatomical_name=aname,
                classification="ESTIMATED_2D_3D_LANDMARK",
                pixel_u=float(u_val),
                pixel_v=float(v_val),
                in_image_bounds=True,
                point_3d_camera_m=p3d,
                depth_m=z_med,
                depth_valid=True,
                depth_std_mm=std_mm,
                visibility_confidence=float(conf),
                status=status,
                notes=note,
            )

        return landmarks

    def _run_detector(self, color_bgr: np.ndarray) -> Optional[List[Any]]:
        """Runs MediaPipe PoseLandmarker on BGR image."""
        if not self.model_path.exists():
            logger.warning(f"MediaPipe pose landmarker model not found at {self.model_path}")
            return None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
            )
            landmarker = PoseLandmarker.create_from_options(options)
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)
            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                return res.pose_landmarks[0]
        except Exception as exc:
            logger.error(f"Error executing MediaPipe PoseLandmarker: {exc}")
        return None

