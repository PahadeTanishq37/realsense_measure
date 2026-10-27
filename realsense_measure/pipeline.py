"""
Orchestrates the full scan pipeline: capture → isolate → register → fuse → measure.

Each stage prints progress to stdout and (when cfg.show_stage_windows is True)
opens a blocking Open3D visualisation window so you can inspect the intermediate
result before the pipeline advances to the next step.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame
from camera.realsense_capture import (
    DeviceDisconnectedError,
    RealSenseCamera,
    StreamNegotiationError,
)
from config import HeadCaptureConfig, PipelineConfig
from landmarks import detect_face_landmarks_3d, project_landmarks_3d_to_pixel
from preprocessing import isolate_object, isolate_with_segmentation, reject_positional_outliers
from reconstruction import (
    compute_reconstruction_quality_metrics,
    filter_depth_pixel_quality,
    fuse_head_scalable_tsdf,
    fuse_point_clouds,
    fuse_tsdf_volume,
    mask_frame_to_head_roi,
    refine_poses_model_based,
)
from registration import (
    compute_rotation_deg,
    register_frame_pair,
    register_sequence,
    register_sequence_multiway,
    register_sequence_rgbd_human,
)
from targets.base import get_target
import targets.box   # noqa: F401 — registers BoxTarget via @register_target
import targets.head  # noqa: F401 — registers HeadTarget via @register_target
from visualizer import (
    color_frames_distinctly,
    draw_camera_trajectory,
    obb_lineset,
    prepare_display_geometry,
    show,
    visualize_registration_pair,
)


def select_roi_on_frame(
    color_bgr: np.ndarray,
    window_name: str = "Select Object ROI (Drag Box & press SPACE/ENTER)",
) -> tuple[int, int, int, int] | None:
    """
    Open an OpenCV interactive window letting the user drag a bounding box around the target object.

    Returns (x, y, w, h) in pixels, or None if cancelled.
    """
    rect = cv2.selectROI(window_name, color_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


class HeadCaptureTracker:
    """
    Lightweight real-time feedback tracker for human/head capture.

    Tracks landmark lock quality and frame-to-frame pixel motion without
    costly 3D registration or ICP.
    """

    def __init__(self, cfg: HeadCaptureConfig | None = None) -> None:
        self.cfg = cfg or HeadCaptureConfig()
        self.frames_since_good_lock: int = 0
        self.prev_landmarks_px: dict[str, np.ndarray] | None = None
        self.last_face_bbox: tuple[int, int, int, int] | None = None
        self.frames_since_face_bbox: int = 0
        self.avg_motion_px: float = 0.0

    def update(
        self,
        landmarks_px: dict[str, np.ndarray] | None,
    ) -> tuple[str | None, str | None, float, int]:
        """
        Update tracker with current frame landmarks.

        Parameters
        ----------
        landmarks_px:
            Dict mapping landmark name to (x, y) pixel coordinates, or None.

        Returns
        -------
        warning_banner:
            Warning message to display on HUD, or None if nominal.
        warning_color:
            'yellow', 'red', or None.
        avg_motion_px:
            Average pixel displacement between matched landmarks and previous frame.
        frames_since_good_lock:
            Rolling count of frames since >= min_landmarks_for_lock were detected.
        """
        num_landmarks = len(landmarks_px) if landmarks_px is not None else 0
        if num_landmarks >= self.cfg.min_landmarks_for_lock:
            self.frames_since_good_lock = 0
        else:
            self.frames_since_good_lock += 1

        avg_motion_px = 0.0
        if landmarks_px is not None and self.prev_landmarks_px is not None:
            matched = set(landmarks_px.keys()) & set(self.prev_landmarks_px.keys())
            if matched:
                displacements = [
                    float(np.linalg.norm(np.asarray(landmarks_px[k]) - np.asarray(self.prev_landmarks_px[k])))
                    for k in matched
                ]
                avg_motion_px = float(np.mean(displacements))

        self.avg_motion_px = avg_motion_px

        if landmarks_px is not None and len(landmarks_px) >= self.cfg.min_landmarks_for_lock:
            coords = np.array(list(landmarks_px.values()))
            pad = 12
            x1 = max(0, int(np.min(coords[:, 0])) - pad)
            y1 = max(0, int(np.min(coords[:, 1])) - pad)
            x2 = int(np.max(coords[:, 0])) + pad
            y2 = int(np.max(coords[:, 1])) + pad
            self.last_face_bbox = (x1, y1, x2, y2)
            self.frames_since_face_bbox = 0
        else:
            self.frames_since_face_bbox += 1

        if landmarks_px is not None and len(landmarks_px) > 0:
            self.prev_landmarks_px = landmarks_px

        warning_banner: str | None = None
        warning_color: str | None = None

        if self.frames_since_good_lock >= self.cfg.max_frames_without_lock:
            warning_banner = "COVERAGE GAP LIKELY -- CONSIDER RE-CAPTURING THIS ANGLE"
            warning_color = "red"
        elif avg_motion_px > self.cfg.max_landmark_px_motion:
            warning_banner = "TOO FAST -- SLOW DOWN ROTATION"
            warning_color = "red"
        elif self.frames_since_good_lock >= self.cfg.min_frames_for_low_lock:
            warning_banner = "LOW LANDMARK LOCK -- FACE CAMERA MORE DIRECTLY"
            warning_color = "yellow"

        return warning_banner, warning_color, avg_motion_px, self.frames_since_good_lock


def compute_head_roi_quality(
    depth_m: np.ndarray,
    landmarks_px: dict[str, np.ndarray] | None,
    cfg: HeadCaptureConfig,
    tracker: HeadCaptureTracker | None = None,
) -> dict[str, Any]:
    """
    Compute real-time depth quality metrics in the face/head region.

    Uses landmark bounding box when a face is detected; otherwise a multi-tier fallback:
      (a) last known face box from tracker (previous N frames)
      (b) bounding box of nearest connected foreground region inside central 60%
      (c) skip quality check and return 'no subject found'
    """
    h, w = depth_m.shape[:2]
    num_lms = len(landmarks_px) if landmarks_px is not None else 0
    face_detected = num_lms >= cfg.min_landmarks_for_lock

    roi_bbox: tuple[int, int, int, int] | None = None
    roi_type: str = "none"

    if face_detected and landmarks_px:
        coords = np.array(list(landmarks_px.values()))
        pad = 12
        x1 = max(0, int(np.min(coords[:, 0])) - pad)
        y1 = max(0, int(np.min(coords[:, 1])) - pad)
        x2 = min(w, int(np.max(coords[:, 0])) + pad)
        y2 = min(h, int(np.max(coords[:, 1])) + pad)
        roi_bbox = (x1, y1, x2, y2)
        roi_type = f"face_bbox ({num_lms} pts)"
        if tracker is not None:
            tracker.last_face_bbox = roi_bbox
            tracker.frames_since_face_bbox = 0
    else:
        # (a) Last known face box from previous N frames
        max_gap = getattr(cfg, "fallback_face_box_max_frames", 30)
        if (
            tracker is not None
            and getattr(tracker, "last_face_bbox", None) is not None
            and getattr(tracker, "frames_since_face_bbox", 999) <= max_gap
        ):
            x1, y1, x2, y2 = tracker.last_face_bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            roi_bbox = (x1, y1, x2, y2)
            roi_type = f"last_face_bbox (gap={tracker.frames_since_face_bbox})"
        else:
            # (b) Bounding box of nearest connected foreground region inside central 60% of image
            cx1 = int(w * 0.20)
            cx2 = int(w * 0.80)
            cy1 = int(h * 0.20)
            cy2 = int(h * 0.80)
            sub_d = depth_m[cy1:cy2, cx1:cx2]
            valid_fg = (sub_d > 0.10) & (sub_d <= getattr(cfg, "warn_dist_max_m", 0.95) + 0.35)
            n_fg_pts = int(np.count_nonzero(valid_fg))
            if n_fg_pts >= 100:
                d_min_sub = float(np.percentile(sub_d[valid_fg], 5))
                if d_min_sub <= 1.30:
                    slice_mask = (sub_d > 0.10) & (sub_d <= d_min_sub + 0.25)
                    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(slice_mask.astype(np.uint8), connectivity=8)
                    if n_lbl > 1:
                        best_lbl = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                        if stats[best_lbl, cv2.CC_STAT_AREA] >= 150:
                            bx = stats[best_lbl, cv2.CC_STAT_LEFT]
                            by = stats[best_lbl, cv2.CC_STAT_TOP]
                            bw = stats[best_lbl, cv2.CC_STAT_WIDTH]
                            bh = stats[best_lbl, cv2.CC_STAT_HEIGHT]
                            pad_px = 10
                            x1 = max(0, cx1 + bx - pad_px)
                            y1 = max(0, cy1 + by - pad_px)
                            x2 = min(w, cx1 + bx + bw + pad_px)
                            y2 = min(h, cy1 + by + bh + pad_px)
                            roi_bbox = (x1, y1, x2, y2)
                            roi_type = "nearest_foreground_roi"

    # (c) Skip quality check and show 'no subject found' if no valid ROI found
    if roi_bbox is None:
        return {
            "roi_median_depth_m": 0.0,
            "roi_min_depth_m": 0.0,
            "roi_valid_depth_frac": 0.0,
            "face_detected": False,
            "roi_bbox": None,
            "roi_type": "none",
            "dist_status_text": "NO SUBJECT",
            "dist_color": "red",
            "is_gate_pass": False,
            "gate_reason": "No subject found",
            "gate_reasons": ["No subject found"],
            "gate_mode": getattr(cfg, "gate_mode", "off"),
            "advisory_reasons": ["No subject found"],
        }

    x1, y1, x2, y2 = roi_bbox
    roi_depth = depth_m[y1:y2, x1:x2]
    total_px = roi_depth.size
    valid_mask = roi_depth > 0
    valid_px = roi_depth[valid_mask]
    n_valid = int(valid_px.size)

    valid_fraction = float(n_valid / total_px) if total_px > 0 else 0.0
    median_depth_m = float(np.median(valid_px)) if n_valid > 0 else 0.0
    min_depth_m = float(np.min(valid_px)) if n_valid > 0 else 0.0

    # Distance guidance & HUD color band
    if n_valid == 0 or median_depth_m <= 0.0:
        dist_status_text = "NO DEPTH"
        dist_color = "red"
    elif median_depth_m < cfg.critical_dist_min_m:
        dist_status_text = "MOVE BACK"
        dist_color = "red"
    elif median_depth_m < cfg.warn_dist_min_m:
        dist_status_text = "MOVE BACK"
        dist_color = "yellow"
    elif median_depth_m > cfg.warn_dist_max_m:
        dist_status_text = "TOO FAR"
        dist_color = "red"
    elif median_depth_m > cfg.target_dist_max_m:
        dist_status_text = "TOO FAR"
        dist_color = "yellow"
    else:
        dist_status_text = "OK"
        dist_color = "green"

    # Capture gate evaluation
    gate_mode = getattr(cfg, "gate_mode", "off")
    if getattr(cfg, "capture_gate", False) and gate_mode == "off":
        gate_mode = "hard"

    advisory_reasons: list[str] = []
    if cfg.gate_require_face and not face_detected:
        advisory_reasons.append("Face not detected")
    if valid_fraction < cfg.gate_min_valid_depth_frac:
        advisory_reasons.append(f"ROI valid depth too low ({valid_fraction:.1%} < {cfg.gate_min_valid_depth_frac:.0%})")
    if median_depth_m < cfg.gate_min_dist_m:
        advisory_reasons.append(f"Subject too close ({median_depth_m:.2f}m < {cfg.gate_min_dist_m:.2f}m)")
    elif median_depth_m > cfg.gate_max_dist_m:
        advisory_reasons.append(f"Subject too far ({median_depth_m:.2f}m > {cfg.gate_max_dist_m:.2f}m)")

    is_gate_pass = True
    gate_reason: str | None = None
    gate_reasons: list[str] = []

    if gate_mode == "off":
        is_gate_pass = True
        gate_reason = None
        gate_reasons = []
    elif gate_mode == "advisory":
        # In advisory mode, HUD shows the reasons in colour, but still allows capture
        is_gate_pass = True
        gate_reasons = advisory_reasons
        gate_reason = "; ".join(advisory_reasons) if advisory_reasons else None
    elif gate_mode == "hard":
        # Hard-block only on catastrophic conditions by default
        hard_min_valid = getattr(cfg, "gate_hard_min_valid_depth_frac", 0.35)
        hard_min_dist = getattr(cfg, "gate_hard_min_dist_m", 0.50)
        hard_reasons = []
        if valid_fraction < hard_min_valid:
            hard_reasons.append(f"ROI valid depth too low ({valid_fraction:.1%} < {hard_min_valid:.0%})")
        if min_depth_m < hard_min_dist or median_depth_m < hard_min_dist:
            d_val = min_depth_m if min_depth_m > 0 else median_depth_m
            hard_reasons.append(f"Subject too close ({d_val:.2f}m < {hard_min_dist:.2f}m)")
        elif median_depth_m > cfg.gate_max_dist_m:
            hard_reasons.append(f"Subject too far ({median_depth_m:.2f}m > {cfg.gate_max_dist_m:.2f}m)")
        if cfg.gate_require_face and not face_detected:
            hard_reasons.append("Face not detected")

        if hard_reasons:
            is_gate_pass = False
            gate_reason = "; ".join(hard_reasons)
            gate_reasons = hard_reasons
        else:
            is_gate_pass = True
            gate_reason = None
            gate_reasons = []

    return {
        "roi_median_depth_m": median_depth_m,
        "roi_min_depth_m": min_depth_m,
        "roi_valid_depth_frac": valid_fraction,
        "face_detected": face_detected,
        "roi_bbox": roi_bbox,
        "roi_type": roi_type,
        "dist_status_text": dist_status_text,
        "dist_color": dist_color,
        "is_gate_pass": is_gate_pass,
        "gate_reason": gate_reason,
        "gate_reasons": gate_reasons,
        "gate_mode": gate_mode,
        "advisory_reasons": advisory_reasons,
    }


def _draw_hud(
    panel: np.ndarray,
    depth_m: np.ndarray,
    n_frames: int,
    n_pts: int,
    vis_min_m: float,
    vis_max_m: float,
    auto_capture: bool = True,
    mode_title: str = "D455f STATUS",
    is_human: bool = False,
    warning_banner: str | None = None,
    warning_color: str | tuple[int, int, int] | None = None,
    lock_status: str | None = None,
    roi_quality: dict[str, Any] | None = None,
    gate_enabled: bool = False,
    stillness_status: str | None = None,
    turn_guidance: str | None = None,
    bg_status: str | None = None,
) -> None:
    """
    Render the D455f status HUD onto the depth panel in-place.
    """
    valid = depth_m[depth_m > 0]
    n_valid   = int(valid.size)
    d_median  = float(np.median(valid)) if n_valid else 0.0
    d_min     = float(valid.min())      if n_valid else 0.0
    d_max     = float(valid.max())      if n_valid else 0.0

    cloud_status = f"{n_pts:,} pts" if n_pts > 0 else "--"

    guide_lines: list[tuple[str, tuple[int, int, int] | None, bool]] = []
    # entry tuple: (text, color_override, is_distance_band)

    if is_human:
        if roi_quality is not None:
            if roi_quality.get("roi_type") == "none":
                guide_lines.append(("Distance (ROI)  : -- [NO SUBJECT FOUND]", (0, 0, 255), True))
            else:
                d_med = roi_quality["roi_median_depth_m"]
                d_nearest = roi_quality["roi_min_depth_m"]
                v_frac = roi_quality["roi_valid_depth_frac"]
                status_txt = roi_quality["dist_status_text"]
                c_tag = roi_quality["dist_color"]
                if c_tag == "green":
                    status_bgr = (0, 255, 0)
                elif c_tag == "yellow":
                    status_bgr = (0, 255, 255)
                else:
                    status_bgr = (0, 0, 255)

                dist_str = f"{d_med:.2f} m" if d_med > 0 else "--"
                guide_lines.append((f"Distance (ROI)  : {dist_str} [{status_txt}]", status_bgr, True))
                guide_lines.append((f"ROI valid depth : {v_frac:.1%} (nearest: {d_nearest:.2f} m)", None, False))

            gate_mode = roi_quality.get("gate_mode", "hard" if gate_enabled else "off")
            if gate_mode == "hard":
                if roi_quality["is_gate_pass"]:
                    guide_lines.append(("Capture Gate    : HARD [READY]", (0, 255, 0), False))
                else:
                    guide_lines.append((f"Capture Gate    : [HOLD] {roi_quality['gate_reason']}", (0, 0, 255), False))
            elif gate_mode == "advisory":
                if roi_quality.get("advisory_reasons"):
                    adv_str = "; ".join(roi_quality["advisory_reasons"])
                    guide_lines.append((f"Capture Gate    : [ADVISORY] {adv_str}", (0, 255, 255), False))
                else:
                    guide_lines.append(("Capture Gate    : [ADVISORY OK]", (0, 255, 0), False))
            elif gate_enabled:
                if roi_quality["is_gate_pass"]:
                    guide_lines.append(("Capture Gate    : ARMED [READY]", (0, 255, 0), False))
                else:
                    guide_lines.append((f"Capture Gate    : [HOLD] {roi_quality['gate_reason']}", (0, 0, 255), False))
        else:
            if d_median > 0:
                if d_median < 0.50:
                    dist_guide = "Distance        : TOO CLOSE (<0.50m) -- BACK UP"
                    c = (0, 0, 255)
                elif d_median > 0.95:
                    dist_guide = "Distance        : TOO FAR (>0.95m) -- MOVE CLOSER"
                    c = (0, 255, 255)
                else:
                    dist_guide = "Distance        : OPTIMAL (0.50m - 0.90m) [OK]"
                    c = (0, 255, 0)
            else:
                dist_guide = "Distance        : NO DEPTH"
                c = (0, 0, 255)
            guide_lines.append((dist_guide, c, True))

        if bg_status:
            bg_c = (0, 255, 0) if "LOCKED" in bg_status else (0, 255, 255)
            guide_lines.append((f"Background Ref  : {bg_status}", bg_c, False))
        if stillness_status:
            st_c = (0, 255, 0) if "HOLDING" in stillness_status else (0, 255, 255)
            guide_lines.append((f"Stillness Hold  : {stillness_status}", st_c, False))
        if turn_guidance:
            guide_lines.append((f"Turn Target     : {turn_guidance}", (0, 255, 255), False))

    if is_human:
        if warning_banner:
            legend = f"[!] {warning_banner}"
        else:
            legend = "HUMAN 180 SWEEP: move slowly, keep subject still -- ENTER: finish"
    elif auto_capture:
        legend = "AUTO-CAPTURING -- move camera around object -- press ENTER when done"
    else:
        legend = "SPACE: capture  ENTER: finish (2+)  ESC: abort"

    lines: list[tuple[str, tuple[int, int, int] | None, bool]] = [
        (mode_title, (0, 255, 255), False),
        ("", None, False),
        (f"Frames captured : {n_frames}", None, False),
        (f"Valid depth px  : {n_valid:,}", None, False),
        (f"Point cloud pts : {cloud_status}", None, False),
        ("", None, False),
        (f"Depth median    : {d_median:.3f} m", None, False),
        (f"Depth min       : {d_min:.3f} m", None, False),
        (f"Depth max       : {d_max:.3f} m", None, False),
    ]

    if is_human and lock_status:
        lines.append((f"Landmark lock   : {lock_status}", None, False))

    if is_human and guide_lines:
        lines.append(("", None, False))
        lines.extend(guide_lines)

    lines.extend([
        ("", None, False),
        (f"Vis range  min  : {vis_min_m:.2f} m", None, False),
        (f"Vis range  max  : {vis_max_m:.2f} m", None, False),
        ("", None, False),
        (legend, None, False),
    ])

    y0, dy = 28, 22
    for i, (line_text, color_override, is_dist_band) in enumerate(lines):
        y = y0 + i * dy
        if not line_text:
            continue

        # Shadow (readability over any background color)
        cv2.putText(panel, line_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)

        # Color resolution
        if warning_banner and line_text == legend:
            if warning_color == "yellow":
                color = (0, 255, 255)  # Yellow
            elif warning_color == "red":
                color = (0, 0, 255)    # Red
            elif isinstance(warning_color, tuple):
                color = warning_color
            else:
                color = (0, 165, 255)  # Orange warning alert
        elif color_override is not None:
            color = color_override
        else:
            color = (255, 255, 255)

        cv2.putText(panel, line_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 1, cv2.LINE_AA)

        # Visual color band block for working distance
        if is_dist_band and color_override is not None:
            band_x1 = 285
            band_y1 = y - 13
            cv2.rectangle(panel, (band_x1, band_y1), (band_x1 + 18, band_y1 + 14), color_override, -1)
            cv2.rectangle(panel, (band_x1, band_y1), (band_x1 + 18, band_y1 + 14), (0, 0, 0), 1)


def _depth_mean_diff(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute mean absolute difference between valid depth pixels in two frames (m)."""
    mask = (d1 > 0) & (d2 > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(d1[mask] - d2[mask])))


def _rough_motion_estimate(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_dist_m: float = 0.05,
) -> tuple[float, float]:
    """
    Lightweight, fast rough motion estimate between consecutive frames during live capture.
    Returns (estimated_translation_m, estimated_rotation_deg).
    """
    if len(source_pcd.points) < 50 or len(target_pcd.points) < 50:
        return 0.0, 0.0

    try:
        src = source_pcd.voxel_down_sample(0.015) if len(source_pcd.points) > 500 else source_pcd
        tgt = target_pcd.voxel_down_sample(0.015) if len(target_pcd.points) > 500 else target_pcd

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_dist_m,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15),
        )

        t_m = float(np.linalg.norm(reg.transformation[:3, 3]))
        r_deg = float(compute_rotation_deg(reg.transformation[:3, :3]))

        # If ICP correspondence fitness is low (< 20%), also consider centroid separation
        if reg.fitness < 0.20:
            c_src = np.mean(np.asarray(src.points), axis=0)
            c_tgt = np.mean(np.asarray(tgt.points), axis=0)
            c_dist = float(np.linalg.norm(c_src - c_tgt))
            t_m = max(t_m, c_dist)

        return t_m, r_deg
    except Exception:
        return 0.0, 0.0


def get_next_scan_folder(base_dir: str = "stage_wise_output") -> Path:
    """
    Returns a new, empty folder for this scan run, auto-numbered:
    stage_wise_output/scan_test_1/, scan_test_2/, etc. -- never overwrites
    a previous scan's folder, so every run's output is permanently kept
    and reviewable later, independent of whatever scan_output/ currently
    does.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("scan_test_")]
    numbers = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
    next_num = max(numbers, default=0) + 1
    new_folder = base / f"scan_test_{next_num}"
    new_folder.mkdir()
    return new_folder


class ScanPipeline:
    """
    End-to-end orchestration of a multi-view scan (BOX or HUMAN HEAD/BODY).

    Usage::

        cfg = PipelineConfig()
        result = ScanPipeline(cfg).run()
        print(result)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        if self.cfg.random_seed is not None:
            import random
            import open3d as o3d
            random.seed(self.cfg.random_seed)
            np.random.seed(self.cfg.random_seed)
            o3d.utility.random.seed(self.cfg.random_seed)
            logger.info(f"  Random seed initialized: {self.cfg.random_seed}")

        # If targeting human head or body without explicit registration override, resolve defaults
        cfg.registration.resolve_defaults(cfg.target.name)

        self._clean_output_dir()
        self.target = get_target(cfg.target.name)
        self.scan_folder = get_next_scan_folder()
        logger.info(f"  Saving organized stage outputs to: {self.scan_folder}/")

    def _clean_output_dir(self) -> None:
        """
        Safely empty the output directory before running a new scan.

        Removes all files and subdirectories inside cfg.output_dir so results from
        previous runs are never mixed with fresh scan outputs.
        """
        out_dir = self.cfg.output_dir
        if out_dir.exists() and out_dir.is_dir():
            for item in out_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"  [!] Warning: Failed to remove old output item {item.name}: {e}")
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: capture
    # ------------------------------------------------------------------

    def _log_capture_guidance(self, is_human_mode: bool, capture_interval: float) -> None:
        """Display stage 1 operator guidance in terminal."""
        logger.info("\n" + "=" * 60)
        logger.info("  Stage 1: Capture")
        if is_human_mode:
            logger.info("  MODE: HUMAN 180° SCAN (RGB-D Odometry + Colored ICP)")
            logger.info("  OPERATOR GUIDANCE:")
            logger.info("    * Keep the subject still (subject rigidity is assumed).")
            logger.info("    * Move camera slowly in a smooth ~180° arc around the front.")
            logger.info("    * Maintain substantial overlap between consecutive views.")
            logger.info("    * Keep the subject's face/head approximately centered.")
            logger.info("    * Avoid sudden camera rotations or translations.")
            logger.info("    * Complete the 180° sweep smoothly.")
            logger.info(f"  AUTO-CAPTURE interval: {capture_interval:.2f}s")
            logger.info("  ENTER = finish (requires 2+ frames) | ESC = abort")
        else:
            logger.info("  MODE: BOX / OBJECT SCAN (Point Cloud)")
            if self.cfg.camera.auto_capture:
                logger.info(f"  AUTO-CAPTURE: move camera around object (interval: "
                      f"{capture_interval:.2f}s)")
                logger.info("  TIP: for small or irregular objects, --manual-capture "
                      "(press SPACE per angle) gives much more consistent results "
                      "than auto-capture.")
                logger.info("  ENTER = finish (requires 2+ frames) | ESC = abort")
            else:
                logger.info("  MANUAL CAPTURE: SPACE = capture frame | ENTER = finish | ESC = abort")
        logger.info("=" * 60)

    def _init_manual_roi(
        self,
        cam: RealSenseCamera,
        is_human_mode: bool,
    ) -> tuple[int, int, int, int] | None:
        """Acquire initial frame for interactive ROI selection if configured."""
        if not (self.cfg.target.use_manual_roi and not is_human_mode):
            return None
        logger.info("\n  [MANUAL ROI MODE] Acquiring initial frame for ROI selection...")
        init_color, init_depth = None, None
        for _ in range(40):
            init_color, init_depth, _ = cam.read_with_colorized()
            if init_color is not None and init_depth is not None:
                break
            time.sleep(0.05)
        if init_color is not None:
            logger.info("  --> Drag a rectangle around the target object in the window, then press SPACE or ENTER.")
            manual_roi = select_roi_on_frame(init_color)
            if manual_roi is not None:
                rx, ry, rw, rh = manual_roi
                logger.info(f"  --> Manual ROI locked: x={rx}, y={ry}, w={rw}, h={rh} px\n")
                return manual_roi
            else:
                logger.warning("  [!] Manual ROI selection cancelled -- falling back to full-frame.\n")
        return None

    def _render_capture_preview(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        depth_color: np.ndarray,
        cam: RealSenseCamera,
        head_tracker: HeadCaptureTracker | None,
        manual_roi: tuple[int, int, int, int] | None,
        is_human_mode: bool,
        motion_warn_text: str | None,
        motion_warn_until: float,
        n_frames: int,
        last_n_pts: int,
        gate_warn_text: str | None = None,
        gate_warn_until: float = 0.0,
        stillness_status: str | None = None,
        turn_guidance: str | None = None,
        bg_status: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Compose dual-panel color + depth HUD preview image for live display."""
        if depth_color.shape[:2] != color_bgr.shape[:2]:
            h, w = color_bgr.shape[:2]
            depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)

        depth_panel = depth_color.copy()
        hud_title = "HUMAN 180° SCAN" if is_human_mode else "D455f STATUS"

        head_warn_text: str | None = None
        head_warn_color: str | None = None
        lock_status_str: str | None = None
        live_landmarks_px: dict[str, np.ndarray] | None = None

        if head_tracker is not None:
            intr = getattr(cam, "intrinsics_o3d", None)
            if intr is None:
                h_f, w_f = color_bgr.shape[:2]
                intr = o3d.camera.PinholeCameraIntrinsic(w_f, h_f, w_f * 0.8, w_f * 0.8, w_f / 2.0, h_f / 2.0)
            try:
                lm_3d = detect_face_landmarks_3d(color_bgr, depth_m, intr, quiet=True)
                if lm_3d:
                    live_landmarks_px = project_landmarks_3d_to_pixel(lm_3d, intr)
            except Exception:
                pass

            head_warn_text, head_warn_color, avg_disp, lock_gap = head_tracker.update(live_landmarks_px)
            n_lm = len(live_landmarks_px) if live_landmarks_px else 0
            if n_lm >= head_tracker.cfg.min_landmarks_for_lock:
                lock_status_str = f"LOCKED ({n_lm} pts)"
            elif lock_gap < head_tracker.cfg.max_frames_without_lock:
                lock_status_str = f"LOW LOCK ({n_lm} pts, gap={lock_gap})"
            else:
                lock_status_str = f"NO LOCK (gap={lock_gap})"

        # Compute face/head ROI quality (with tracker history for fallback)
        roi_quality = None
        head_tracker_cfg = getattr(self.cfg.camera, "head_capture", getattr(self.cfg, "head_capture", HeadCaptureConfig()))
        if is_human_mode:
            roi_quality = compute_head_roi_quality(depth_m, live_landmarks_px, head_tracker_cfg, tracker=head_tracker)

        now_ts = time.time()
        if now_ts < gate_warn_until and gate_warn_text:
            active_warn = gate_warn_text
            active_warn_color = "red"
        elif is_human_mode and head_warn_text:
            active_warn = head_warn_text
            active_warn_color = head_warn_color
        elif now_ts < motion_warn_until:
            active_warn = motion_warn_text
            active_warn_color = "red"
        else:
            active_warn = None
            active_warn_color = None

        _draw_hud(
            depth_panel,
            depth_m,
            n_frames=n_frames,
            n_pts=last_n_pts,
            vis_min_m=self.cfg.camera.depth_vis.visual_min_m,
            vis_max_m=self.cfg.camera.depth_vis.visual_max_m,
            auto_capture=self.cfg.camera.auto_capture,
            mode_title=hud_title,
            is_human=is_human_mode,
            warning_banner=active_warn,
            warning_color=active_warn_color,
            lock_status=lock_status_str,
            roi_quality=roi_quality,
            gate_enabled=bool(head_tracker_cfg.capture_gate),
            stillness_status=stillness_status,
            turn_guidance=turn_guidance,
            bg_status=bg_status,
        )

        display_color = color_bgr.copy()
        if is_human_mode:
            h_c, w_c = display_color.shape[:2]
            cx_c, cy_c = w_c // 2, h_c // 2
            cv2.ellipse(display_color, (cx_c, cy_c), (int(w_c * 0.20), int(h_c * 0.35)), 0, 0, 360, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.drawMarker(display_color, (cx_c, cy_c), (0, 220, 220), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            cv2.putText(display_color, "HEAD GUIDE", (cx_c - 45, cy_c - int(h_c * 0.37)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 220), 1, cv2.LINE_AA)
            if live_landmarks_px:
                for pt in live_landmarks_px.values():
                    cv2.circle(display_color, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

            if roi_quality is not None and roi_quality.get("roi_bbox") is not None:
                rx1, ry1, rx2, ry2 = roi_quality["roi_bbox"]
                box_c = (0, 255, 0) if roi_quality["dist_color"] == "green" else (
                    (0, 255, 255) if roi_quality["dist_color"] == "yellow" else (0, 0, 255)
                )
                cv2.rectangle(display_color, (rx1, ry1), (rx2, ry2), box_c, 2)
                roi_label = f"HEAD ROI: {roi_quality['dist_status_text']} ({roi_quality['roi_median_depth_m']:.2f}m, {roi_quality['roi_valid_depth_frac']:.0%})"
                cv2.putText(display_color, roi_label, (rx1 + 4, max(ry1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_c, 1, cv2.LINE_AA)
            elif roi_quality is not None and roi_quality.get("roi_type") == "none":
                cv2.putText(display_color, "HEAD ROI: NO SUBJECT FOUND", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        if manual_roi is not None:
            rx, ry, rw, rh = manual_roi
            cv2.rectangle(display_color, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display_color, "ROI", (rx + 4, max(ry - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        return np.hstack([display_color, depth_panel]), roi_quality

    def _capture_burst(
        self,
        cam: RealSenseCamera,
        burst_count: int = 10,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture N frames in quick succession and compute pixel-wise median depth."""
        if burst_count <= 1:
            c, d, _ = cam.read_with_colorized()
            return c, d
        burst_colors: list[np.ndarray] = []
        burst_depths: list[np.ndarray] = []
        for _ in range(burst_count):
            c, d, _ = cam.read_with_colorized()
            if c is not None and d is not None:
                burst_colors.append(c)
                burst_depths.append(d)
            time.sleep(0.008)
        if not burst_depths:
            return None, None
        from camera.frame import compute_burst_median_frame
        min_v = getattr(self.cfg.head_capture, "burst_min_valid_frac", 0.50)
        return compute_burst_median_frame(burst_colors, burst_depths, min_valid_frac=min_v)

    def _try_capture_human_frame(
        self,
        cam: RealSenseCamera,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        raw_frames: list[o3d.geometry.PointCloud | RGBDFrame],
        now: float,
        roi_quality: dict[str, Any] | None = None,
    ) -> tuple[RGBDFrame | None, int, str | None, float]:
        """Construct and validate a human RGBDFrame from current sensor observations."""
        rgbd_frame = cam.create_rgbd_frame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            frame_id=len(raw_frames),
            timestamp=now,
        )
        if roi_quality is not None:
            rgbd_frame.quality_metrics = {
                "roi_median_depth_m": roi_quality.get("roi_median_depth_m"),
                "roi_min_depth_m": roi_quality.get("roi_min_depth_m"),
                "roi_valid_depth_frac": roi_quality.get("roi_valid_depth_frac"),
                "face_detected": roi_quality.get("face_detected"),
                "roi_type": roi_quality.get("roi_type"),
            }
        n_pts = len(rgbd_frame.pcd.points)
        if n_pts < self.cfg.target.min_object_points:
            logger.debug(f"  Frame discarded: only {n_pts} pts in range -- check distance to subject (0.4m - 1.2m)")
            return None, 0, None, 0.0

        warn_text = None
        warn_until = 0.0
        if raw_frames:
            prev_frame = raw_frames[-1]
            prev_pcd = prev_frame.pcd if isinstance(prev_frame, RGBDFrame) else prev_frame
            est_t_m, est_r_deg = _rough_motion_estimate(rgbd_frame.pcd, prev_pcd)
            max_jump_t = getattr(self.cfg.registration, "max_pose_jump_translation_m", 0.15)
            max_jump_r = getattr(self.cfg.registration, "max_pose_jump_rotation_deg", 45.0)
            if est_t_m > max_jump_t or est_r_deg > max_jump_r:
                logger.info(
                    f"  [!] WARNING: large motion since last frame "
                    f"({est_t_m * 100:.0f}cm, {est_r_deg:.0f}deg) "
                    f"-- consider moving slower through this section, or reduce "
                    f"--interval for denser capture"
                )
                warn_text = f"LARGE MOTION ({est_t_m * 100:.0f}cm, {est_r_deg:.0f}deg) -- MOVE SLOWER"
                warn_until = now + 3.0

        return rgbd_frame, n_pts, warn_text, warn_until

    def _try_capture_object_frame(
        self,
        cam: RealSenseCamera,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        manual_roi: tuple[int, int, int, int] | None,
        raw_frames: list[o3d.geometry.PointCloud | RGBDFrame],
        captured_centroids: list[np.ndarray],
        last_centroid: np.ndarray | None,
        is_manual: bool = False,
    ) -> tuple[o3d.geometry.PointCloud | None, np.ndarray | None, int]:
        """Construct and isolate a pointcloud from object observation."""
        pcd = cam.to_point_cloud(color_bgr, depth_m, roi=manual_roi)
        iso_test = isolate_object(pcd, self.cfg.preprocess, self.cfg.target, last_centroid=last_centroid)
        n_obj_pts = len(iso_test.points)
        if n_obj_pts < self.cfg.target.min_object_points:
            hint = "try adjusting angle" if is_manual else "keep moving"
            logger.debug(f"  Frame discarded: isolated to only {n_obj_pts} pts "
                  f"(need >= {self.cfg.target.min_object_points}) -- "
                  f"this angle didn't capture usable object data, {hint}")
            return None, last_centroid, 0

        centroid = np.asarray(iso_test.points).mean(axis=0)
        dev = 0.0
        is_outlier = False
        if captured_centroids:
            median_c = np.median(np.array(captured_centroids), axis=0)
            dev = float(np.linalg.norm(centroid - median_c))
            if dev > self.cfg.target.max_centroid_dev_m:
                is_outlier = True

        raw_n_pts = len(pcd.points)
        frame_num = len(raw_frames) + 1
        if is_outlier:
            logger.debug(f"  Captured frame {frame_num:2d}: {n_obj_pts:>5} object "
                  f"pts, but {dev*100:.0f} cm from your other frames -- "
                  f"[WARNING] this may not be the same object, keep it centered")
        else:
            logger.debug(f"  Captured frame {frame_num:2d}: "
                  f"{raw_n_pts:>7} raw pts -> {n_obj_pts:>5} object pts [OK]")

        return pcd, centroid, raw_n_pts

    def capture_frames(self) -> list[o3d.geometry.PointCloud | RGBDFrame]:
        """
        Live camera capture loop or replay from raw recording.

        Returns
        -------
        list[PointCloud | RGBDFrame]
            In pointcloud mode: raw PointClouds.
            In rgbd_human mode: synchronized RGBDFrame instances containing color,
            depth, point clouds with normals, timestamps, and intrinsics.
        """
        if self.cfg.replay_dir is not None:
            from camera.recording import load_raw_capture
            logger.info(f"\n  [REPLAY MODE] Loading raw frames from: {self.cfg.replay_dir}")
            return load_raw_capture(self.cfg.replay_dir)

        try:
            cam = RealSenseCamera(self.cfg.camera).start()
        except StreamNegotiationError as exc:
            logger.error(f"\n  [ERROR] RealSense camera stream negotiation failed: {exc}")
            raise
        is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human") or (self.cfg.target.name in ("head", "body"))

        # In human mode, use the configured human_capture_interval_s (default 3.0s) unless overridden
        capture_interval = self.cfg.camera.capture_interval_s
        if is_human_mode and self.cfg.camera.human_capture_interval_s is not None:
            capture_interval = self.cfg.camera.human_capture_interval_s

        raw_frames: list[o3d.geometry.PointCloud | RGBDFrame] = []
        captured_centroids: list[np.ndarray] = []
        last_centroid: np.ndarray | None = None

        self._log_capture_guidance(is_human_mode, capture_interval)

        start_capture_time = time.time()
        last_capture_time = 0.0  # triggers capture on the first valid frame
        last_captured_depth: np.ndarray | None = None
        last_skip_warn_time = 0.0
        last_n_pts = 0  # point count of the most recently captured cloud
        motion_warn_text: str | None = None
        motion_warn_until = 0.0
        gate_warn_text: str | None = None
        gate_warn_until = 0.0
        last_gate_log_time = 0.0
        last_res_warn_time = 0.0
        head_tracker_cfg = getattr(self.cfg.camera, "head_capture", getattr(self.cfg, "head_capture", HeadCaptureConfig()))
        use_landmarks_capture = getattr(self.cfg.registration, "use_landmarks", True) and getattr(head_tracker_cfg, "enabled", True)
        head_tracker = HeadCaptureTracker(head_tracker_cfg) if (is_human_mode and use_landmarks_capture) else None

        manual_roi = self._init_manual_roi(cam, is_human_mode)

        # Effective resolution logging and range check
        eff_w = self.cfg.camera.width
        eff_h = self.cfg.camera.height
        logger.info(f"  Effective stream resolution: {eff_w}x{eff_h}")
        res_min_dist = (
            self.cfg.camera.min_range_1280x720_m
            if (eff_w >= 1280 and eff_h >= 720)
            else self.cfg.camera.min_range_848x480_m
        )

        # Stillness tracking state for --auto-hold
        stillness_start_time: float | None = None
        prev_still_roi_depth: np.ndarray | None = None
        spacing_satisfied: bool = True  # first frame can trigger immediately once still

        # Subject-turns tracking
        cumulative_yaw_deg: float = 0.0

        window_name = "Stage 1: Human 180 deg Capture" if is_human_mode else "Stage 1: D455f Capture (RGB | Depth)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_initialized = False

        try:
            while True:
                color_bgr, depth_m, depth_color = cam.read_with_colorized()
                if color_bgr is None or depth_m is None:
                    # Pump GUI messages and check abort so window never freezes during frame drop
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        logger.info("  Capture aborted by user.")
                        raise KeyboardInterrupt
                    time.sleep(0.005)
                    continue  # dropped frame — just retry

                now = time.time()

                # Background status string
                bg_status_str = None
                if head_tracker_cfg.subject_turns:
                    if self.cfg.background_ref_depth is not None:
                        bg_status_str = "LOCKED [OK]"
                    else:
                        bg_status_str = "EMPTY SCENE (Press 'B' to Lock)"

                # Stillness tracking
                stillness_status_str = None
                if head_tracker_cfg.auto_hold and is_human_mode:
                    is_still = False
                    if head_tracker is not None and getattr(head_tracker, "last_face_bbox", None) is not None:
                        rx1, ry1, rx2, ry2 = head_tracker.last_face_bbox
                        curr_roi_d = depth_m[ry1:ry2, rx1:rx2]
                        if prev_still_roi_depth is not None and prev_still_roi_depth.shape == curr_roi_d.shape:
                            depth_step_diff = _depth_mean_diff(curr_roi_d, prev_still_roi_depth)
                            cam_motion_ok = True
                            if not head_tracker_cfg.subject_turns:
                                cam_motion_ok = (head_tracker.avg_motion_px < 15.0)
                            if depth_step_diff < head_tracker_cfg.stillness_depth_thresh_m and cam_motion_ok:
                                is_still = True
                            if depth_step_diff >= head_tracker_cfg.stillness_min_motion_spacing_m:
                                spacing_satisfied = True
                        prev_still_roi_depth = curr_roi_d.copy()

                    if is_still:
                        if stillness_start_time is None:
                            stillness_start_time = now
                        still_duration = now - stillness_start_time
                        if not spacing_satisfied:
                            stillness_status_str = "MOVE/TURN FIRST [WAIT]"
                        else:
                            stillness_status_str = f"HOLDING ({still_duration:.1f}s / {head_tracker_cfg.stillness_time_s:.1f}s)"
                    else:
                        stillness_start_time = None
                        still_duration = 0.0
                        stillness_status_str = "WAITING FOR STILLNESS" if spacing_satisfied else "MOVE/TURN FIRST"

                # Turn guidance string
                turn_guidance_str = None
                if head_tracker_cfg.subject_turns:
                    if cumulative_yaw_deg >= 180.0:
                        turn_guidance_str = f"DONE: ~{cumulative_yaw_deg:.0f} deg covered -- Press ENTER"
                    elif len(raw_frames) > 0:
                        target_step = getattr(head_tracker_cfg, "target_yaw_step_deg", 20.0)
                        turn_guidance_str = f"Turn about {target_step:.0f} deg more (covered: ~{cumulative_yaw_deg:.0f} deg)"
                    else:
                        turn_guidance_str = "Facing forward -- Ready for first shot"

                preview, roi_quality = self._render_capture_preview(
                    color_bgr, depth_m, depth_color, cam, head_tracker, manual_roi,
                    is_human_mode, motion_warn_text, motion_warn_until,
                    len(raw_frames), last_n_pts,
                    gate_warn_text=gate_warn_text, gate_warn_until=gate_warn_until,
                    stillness_status=stillness_status_str,
                    turn_guidance=turn_guidance_str,
                    bg_status=bg_status_str,
                )

                # Check resolution minimum distance warning
                if roi_quality and roi_quality.get("roi_median_depth_m", 0.0) > 0:
                    med_d = roi_quality["roi_median_depth_m"]
                    if med_d < res_min_dist and (now - last_res_warn_time >= 3.0):
                        logger.warning(
                            f"  [!] Subject depth ({med_d:.2f}m) is below min recommended range "
                            f"({res_min_dist:.2f}m) for {eff_w}x{eff_h} resolution."
                        )
                        last_res_warn_time = now

                if not window_initialized:
                    disp_w = min(1280, preview.shape[1])
                    disp_h = int(preview.shape[0] * (disp_w / preview.shape[1]))
                    cv2.resizeWindow(window_name, disp_w, disp_h)
                    window_initialized = True

                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                # ---- Key 'B' / 'b' for background reference ----
                if is_human_mode and head_tracker_cfg.subject_turns and (key in (ord('b'), ord('B'))):
                    logger.info("  [SUBJECT-TURNS] Capturing background reference burst (10 frames)...")
                    bg_c, bg_d = self._capture_burst(cam, burst_count=head_tracker_cfg.burst_count)
                    if bg_d is not None:
                        self.cfg.background_ref_color = bg_c
                        self.cfg.background_ref_depth = bg_d
                        gate_warn_text = "BACKGROUND REFERENCE LOCKED [OK]"
                        gate_warn_until = now + 2.0
                        logger.info("  [SUBJECT-TURNS] Background reference locked successfully.")

                # ---- Auto-hold stillness trigger ----
                if head_tracker_cfg.auto_hold and is_human_mode:
                    if (stillness_start_time is not None) and (now - stillness_start_time >= head_tracker_cfg.stillness_time_s) and spacing_satisfied:
                        gate_blocks = False
                        gate_reason = ""
                        if head_tracker_cfg.gate_mode == "hard" and roi_quality:
                            if not roi_quality["is_gate_pass"]:
                                gate_blocks = True
                                gate_reason = roi_quality["gate_reason"] or "quality criteria failed"

                        if gate_blocks:
                            if now - last_gate_log_time >= 0.5:
                                logger.warning(f"  [CAPTURE GATE REJECTED] Frame {len(raw_frames)} attempt blocked: {gate_reason}")
                                last_gate_log_time = now
                            gate_warn_text = f"GATE REJECTED: {gate_reason.upper()} -- RETRYING"
                            gate_warn_until = now + 1.2
                        else:
                            logger.info(f"  [AUTO-HOLD] Stillness held for {now - stillness_start_time:.2f}s -> Capturing burst frame {len(raw_frames) + 1}...")
                            burst_c, burst_d = self._capture_burst(cam, burst_count=head_tracker_cfg.burst_count)
                            if burst_c is None:
                                burst_c, burst_d = color_bgr, depth_m
                            frame, n_pts, m_warn, m_until = self._try_capture_human_frame(
                                cam, burst_c, burst_d, raw_frames, now, roi_quality=roi_quality
                            )
                            if frame is not None:
                                raw_frames.append(frame)
                                last_n_pts = n_pts
                                last_capture_time = now
                                last_captured_depth = burst_d.copy()
                                spacing_satisfied = False
                                stillness_start_time = None
                                if len(raw_frames) >= 2:
                                    est_t, est_r_deg = _rough_motion_estimate(raw_frames[-1].pcd, raw_frames[-2].pcd)
                                    cumulative_yaw_deg += est_r_deg
                                if m_warn is not None:
                                    motion_warn_text = m_warn
                                    motion_warn_until = m_until
                                logger.info(f"  Captured human frame {len(raw_frames):2d}: {n_pts:>6} pts with RGB-D [OK]")

                # ---- Standard timer-based auto-capture logic --------------------------------
                elif self.cfg.camera.auto_capture:
                    if now - last_capture_time >= capture_interval:
                        is_duplicate = False
                        if (self.cfg.camera.skip_near_duplicate_frames
                                 and last_captured_depth is not None):
                            diff_m = _depth_mean_diff(depth_m, last_captured_depth)
                            if diff_m < self.cfg.camera.duplicate_depth_diff_threshold_m:
                                is_duplicate = True

                        if is_duplicate:
                            if now - last_skip_warn_time >= 1.0:
                                logger.debug("  ... camera hasn't moved enough, still waiting ...")
                                last_skip_warn_time = now
                        elif is_human_mode and head_tracker_cfg.gate_mode == "hard" and (roi_quality and not roi_quality["is_gate_pass"]):
                            reason = roi_quality["gate_reason"] or "quality criteria failed"
                            if now - last_gate_log_time >= 0.5:
                                logger.warning(f"  [CAPTURE GATE REJECTED] Frame {len(raw_frames)} attempt blocked: {reason}")
                                last_gate_log_time = now
                            gate_warn_text = f"GATE REJECTED: {reason.upper()} -- RETRYING"
                            gate_warn_until = now + 1.2
                            # Note: last_capture_time is intentionally NOT advanced so auto-capture
                            # retries immediately on the next valid frame without waiting a full interval.
                        else:
                            if is_human_mode:
                                frame, n_pts, m_warn, m_until = self._try_capture_human_frame(
                                    cam, color_bgr, depth_m, raw_frames, now, roi_quality=roi_quality
                                )
                                if frame is not None:
                                    raw_frames.append(frame)
                                    last_n_pts = n_pts
                                    last_capture_time = now
                                    last_captured_depth = depth_m.copy()
                                    if len(raw_frames) >= 2:
                                        est_t, est_r_deg = _rough_motion_estimate(raw_frames[-1].pcd, raw_frames[-2].pcd)
                                        cumulative_yaw_deg += est_r_deg
                                    if m_warn is not None:
                                        motion_warn_text = m_warn
                                        motion_warn_until = m_until
                                    logger.debug(f"  Captured human frame {len(raw_frames):2d}: {n_pts:>6} pts with RGB-D [OK]")
                            else:
                                pcd, centroid, raw_pts = self._try_capture_object_frame(
                                    cam, color_bgr, depth_m, manual_roi, raw_frames,
                                    captured_centroids, last_centroid, is_manual=False
                                )
                                if pcd is not None:
                                    raw_frames.append(pcd)
                                    captured_centroids.append(centroid)
                                    last_centroid = centroid
                                    last_n_pts = raw_pts
                                    last_capture_time = now
                                    last_captured_depth = depth_m.copy()

                # ---- Manual capture handling ---------------------------
                elif key == 32:  # SPACE — manual capture
                    if is_human_mode:
                        gate_blocks = False
                        gate_reason = ""
                        if head_tracker_cfg.gate_mode == "hard" and roi_quality:
                            if not roi_quality["is_gate_pass"]:
                                gate_blocks = True
                                gate_reason = roi_quality["gate_reason"] or "quality criteria failed"

                        if gate_blocks:
                            logger.warning(f"  [CAPTURE GATE REJECTED] Manual capture attempt blocked: {gate_reason}")
                            gate_warn_text = f"GATE REJECTED: {gate_reason.upper()} -- RETRYING"
                            gate_warn_until = now + 1.2
                        else:
                            logger.info(f"  [MANUAL] Capturing burst frame {len(raw_frames) + 1}...")
                            burst_c, burst_d = self._capture_burst(cam, burst_count=head_tracker_cfg.burst_count)
                            if burst_c is None:
                                burst_c, burst_d = color_bgr, depth_m
                            frame, n_pts, m_warn, m_until = self._try_capture_human_frame(
                                cam, burst_c, burst_d, raw_frames, time.time(), roi_quality=roi_quality
                            )
                            if frame is not None:
                                raw_frames.append(frame)
                                last_n_pts = n_pts
                                last_captured_depth = burst_d.copy()
                                spacing_satisfied = False
                                if len(raw_frames) >= 2:
                                    est_t, est_r_deg = _rough_motion_estimate(raw_frames[-1].pcd, raw_frames[-2].pcd)
                                    cumulative_yaw_deg += est_r_deg
                                if m_warn is not None:
                                    motion_warn_text = m_warn
                                    motion_warn_until = m_until
                                logger.info(f"  Captured human frame {len(raw_frames):2d}: {n_pts:>6} pts with RGB-D [OK]")
                    else:
                        pcd, centroid, raw_pts = self._try_capture_object_frame(
                            cam, color_bgr, depth_m, manual_roi, raw_frames,
                            captured_centroids, last_centroid, is_manual=True
                        )
                        if pcd is not None:
                            raw_frames.append(pcd)
                            captured_centroids.append(centroid)
                            last_centroid = centroid
                            last_n_pts = raw_pts
                            last_captured_depth = depth_m.copy()

                # ---- Finish / Abort keys -------------------------------
                if key == 13:  # ENTER — finish
                    if len(raw_frames) < 2:
                        logger.warning("  ⚠  Need at least 2 frames before finishing. "
                              "Keep capturing.")
                    else:
                        elapsed = time.time() - start_capture_time
                        n_frames = len(raw_frames)
                        avg_interval = elapsed / n_frames if n_frames > 0 else 0.0
                        logger.info(f"\n  Capture complete — {n_frames} frames captured in "
                              f"{elapsed:.1f}s (effective avg interval: {avg_interval:.2f}s/frame).")
                        break

                elif key == 27:  # ESC — abort
                    logger.info("  Capture aborted by user.")
                    raise KeyboardInterrupt

        except DeviceDisconnectedError as exc:
            logger.error(f"\n  [ERROR] RealSense camera disconnected during capture: {exc}")
            raise
        finally:
            cam.stop()
            cv2.destroyAllWindows()

        return raw_frames

    # ------------------------------------------------------------------
    # Stage 2: isolate
    # ------------------------------------------------------------------

    def isolate_frames(
        self,
        raw_frames: list[o3d.geometry.PointCloud | RGBDFrame],
    ) -> list[o3d.geometry.PointCloud | RGBDFrame]:
        """
        Run per-frame isolation.

        Parameters
        ----------
        raw_frames:
            List of raw PointClouds or RGBDFrames from :meth:`capture_frames`.

        Returns
        -------
        list[PointCloud | RGBDFrame]
            Isolated, cleaned per-frame objects.
        """
        logger.info("\n" + "=" * 60)
        logger.info("  Stage 2: Isolating object per frame ...")
        logger.info("=" * 60)

        pre_cfg = self.cfg.preprocess
        target_cfg = self.cfg.target

        # Branch for human RGB-D frames: retain RGBDFrame representation
        if raw_frames and isinstance(raw_frames[0], RGBDFrame):
            isolated_rgbd: list[RGBDFrame] = []
            last_iso_centroid: np.ndarray | None = None

            is_subject_turns = bool(getattr(self.cfg.head_capture, "subject_turns", False))
            bg_depth = getattr(self.cfg, "background_ref_depth", None)
            if bg_depth is None:
                bg_depth = getattr(raw_frames, "background_ref_depth", getattr(raw_frames[0], "background_ref_depth", None))

            if is_subject_turns and bg_depth is not None:
                from preprocessing import isolate_subject_by_background_subtraction
                logger.info("  Stage 2: using background-subtraction isolation (fixed-camera subject-turns mode, no 3D DBSCAN).")
                for i, f in enumerate(raw_frames):
                    n_before = len(f.pcd.points)
                    iso, fg_mask = isolate_subject_by_background_subtraction(
                        f,
                        bg_depth,
                        diff_threshold_m=self.cfg.head_capture.bg_diff_threshold_m,
                        depth_min_m=self.cfg.camera.depth_min_m,
                        depth_max_m=self.cfg.camera.depth_max_m,
                    )
                    f.pcd = iso
                    f.foreground_mask = fg_mask
                    n_after = len(iso.points)
                    if n_after < target_cfg.min_object_points:
                        logger.debug(f"  frame {i:02d}: {n_before} -> {n_after} pts -- DISCARDED (< {target_cfg.min_object_points})")
                        continue
                    isolated_rgbd.append(f)

                if not isolated_rgbd:
                    raise RuntimeError(
                        "No frames survived background-subtraction isolation. "
                        "Ensure the subject is within camera depth range and background reference was captured."
                    )
                if self.cfg.show_stage_windows:
                    show(
                        color_frames_distinctly([f.pcd for f in isolated_rgbd]),
                        window_name="Stage 2: preprocessed human frames (background-subtracted)",
                    )
                if self.cfg.save_intermediate:
                    for i, f in enumerate(isolated_rgbd):
                        out = self.cfg.output_dir / f"frame_{i:02d}_isolated.ply"
                        o3d.io.write_point_cloud(str(out), f.pcd)
                        logger.info(f"  Saved: {out}")
                return isolated_rgbd

            use_segmentation = (
                pre_cfg.isolation_mode == "segmentation"
                and hasattr(raw_frames[0], "color_bgr")
                and raw_frames[0].color_bgr is not None
            )
            if use_segmentation:
                logger.info("  Stage 2: using segmentation-guided isolation.")
            for i, f in enumerate(raw_frames):
                pcd = f.pcd
                n_before = len(pcd.points)
                if use_segmentation:
                    iso, method = isolate_with_segmentation(
                        f, pre_cfg, target_cfg, last_centroid=last_iso_centroid
                    )
                    logger.debug(
                        f"  frame {i:02d}: {n_before:>7} pts -> {len(iso.points):>6} pts "
                        f"[method={method}]"
                    )
                else:
                    # Full 3-step isolation: downsample+denoise, dominant plane removal (walls/floor), and single plausible cluster selection
                    iso = isolate_object(pcd, pre_cfg, target_cfg, last_centroid=last_iso_centroid)
                n_after = len(iso.points)
                if n_after < target_cfg.min_object_points:
                    logger.debug(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts -- "
                          f"DISCARDED (< {target_cfg.min_object_points} pts, unusable object data)")
                    continue
                last_iso_centroid = np.asarray(iso.points).mean(axis=0)
                f.pcd = iso
                if not use_segmentation:
                    logger.debug(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts [ACCEPTED]")
                isolated_rgbd.append(f)

            # Positional consistency: drop frames whose centroid jumps away
            keep, rejected_pos = reject_positional_outliers(
                [f.pcd for f in isolated_rgbd],
                target_cfg.max_centroid_dev_m,
                target_name=target_cfg.name,
            )
            for idx, dist in rejected_pos:
                logger.debug(f"  isolated #{idx:02d}: centroid {dist:.2f} m deviation -- "
                      f"DISCARDED (likely background, not the object)")
            isolated_rgbd = [isolated_rgbd[i] for i in keep]

            if not isolated_rgbd:
                raise RuntimeError(
                    "No frames survived object isolation. "
                    "Ensure the subject is within camera depth range (0.4m - 1.2m) and centered in frame."
                )

            if self.cfg.show_stage_windows:
                show(
                    color_frames_distinctly([f.pcd for f in isolated_rgbd]),
                    window_name="Stage 2: preprocessed human frames",
                )

            if self.cfg.save_intermediate:
                for i, f in enumerate(isolated_rgbd):
                    out = self.cfg.output_dir / f"frame_{i:02d}_isolated.ply"
                    o3d.io.write_point_cloud(str(out), f.pcd)
                    logger.info(f"  Saved: {out}")

            return isolated_rgbd

        # Box / standard PointCloud path:
        isolated: list[o3d.geometry.PointCloud] = []
        last_iso_centroid: np.ndarray | None = None
        for i, frame in enumerate(raw_frames):
            n_before = len(frame.points)
            iso = isolate_object(frame, pre_cfg, target_cfg, last_centroid=last_iso_centroid)
            n_after = len(iso.points)
            if n_after < target_cfg.min_object_points:
                logger.debug(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts -- "
                      f"DISCARDED (< {target_cfg.min_object_points} pts, unusable object data)")
                continue
            last_iso_centroid = np.asarray(iso.points).mean(axis=0)
            logger.debug(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts [ACCEPTED]")
            isolated.append(iso)

        # Positional consistency: a background blob can be object-sized, so size
        # checks alone let it through.  Drop frames whose centroid jumps away.
        keep, rejected_pos = reject_positional_outliers(
            isolated,
            target_cfg.max_centroid_dev_m,
            target_name=target_cfg.name,
        )
        for idx, dist in rejected_pos:
            logger.debug(f"  isolated #{idx:02d}: centroid {dist:.2f} m from median position -- "
                  f"DISCARDED (likely background, not the object)")
        isolated = [isolated[i] for i in keep]

        if not isolated:
            raise RuntimeError(
                "No frames survived object isolation. "
                "Ensure the target object is within camera depth range and on a clear surface."
            )

        if self.cfg.show_stage_windows:
            show(
                color_frames_distinctly(isolated),
                window_name="Stage 2: isolated per-frame clouds",
            )

        if self.cfg.save_intermediate:
            for i, pcd in enumerate(isolated):
                out = self.cfg.output_dir / f"frame_{i:02d}_isolated.ply"
                o3d.io.write_point_cloud(str(out), pcd)
                logger.info(f"  Saved: {out}")

        return isolated

    # ------------------------------------------------------------------
    # Stage 3: register
    # ------------------------------------------------------------------

    def _log_human_registration_summary(
        self,
        diagnostics: list[tuple[float, float, bool]],
        summary_stats: dict,
        isolated_frames: list,
    ) -> None:
        """Log human registration method diagnostics and tier summary."""
        frame_methods = summary_stats.get("frame_methods", {})
        frame_scores = summary_stats.get("frame_scores", {})
        rejection_reasons = summary_stats.get("rejection_reasons", {})
        fusion_eligible = summary_stats.get("fusion_eligible", {})
        fusion_reasons = summary_stats.get("fusion_reasons", {})
        logger.info("\n" + "=" * 60)
        logger.info("HUMAN RGB-D REGISTRATION SUMMARY (Method Diagnostics)")
        logger.info("-----------------------------------------------------")
        for i, diag in enumerate(diagnostics):
            if i == 0:
                logger.debug("    frame 00: reference frame (origin) [ACCEPTED]")
            elif diag is not None:
                fit, rmse, acc = diag
                method = frame_methods.get(i, "none")
                score_str = frame_scores.get(i, f"fitness={fit:.2f} rmse={rmse * 1000.0:.1f}mm")
                if acc:
                    if not fusion_eligible.get(i, True):
                        reason_ex = fusion_reasons.get(i, "")
                        logger.debug(f"    frame {i:02d}: pose accepted, EXCLUDED from fusion ({reason_ex})")
                    else:
                        logger.debug(f"    frame {i:02d}: tier={method} {score_str} ACCEPTED")
                elif i in rejection_reasons:
                    logger.info(
                        f"    frame {i:02d}: tier={method} {score_str}\n"
                        f"              REJECTED -- {rejection_reasons[i]}"
                    )
                else:
                    logger.debug(f"    frame {i:02d}: tier=none REJECTED")
        logger.info("=" * 60)

        n_accepted = summary_stats.get("accepted", sum(1 for d in diagnostics if d is not None and d[2]))
        n_total = summary_stats.get("captured", len(isolated_frames))
        n_rejected = n_total - n_accepted
        tier_summary = summary_stats.get(
            "tier_summary_str",
            f"{sum(1 for m in frame_methods.values() if m == 'landmark')} landmark, "
            f"{sum(1 for m in frame_methods.values() if m in ('colored_icp', 'rgbd_odometry_colored_icp'))} colored_icp, "
            f"{sum(1 for m in frame_methods.values() if m == 'fpfh_fallback')} fpfh_fallback, "
            f"{n_rejected} rejected"
        )
        logger.info(f"\n  Registration tier summary: {tier_summary}")
        logger.info(f"  Registration summary: {n_accepted} of {n_total} frames accepted into the reconstruction"
              f" ({n_rejected} rejected -- excluded from fusion).")
        if n_total > 0 and (n_accepted / n_total) < 0.60:
            logger.warning("  Tip: low acceptance ratio (< 60%). Try moving the camera more slowly for higher frame overlap.")

    def _visualize_registration_diagnostics(
        self,
        isolated_frames: list[o3d.geometry.PointCloud | RGBDFrame],
        aligned: list[o3d.geometry.PointCloud],
        diagnostics: list[tuple[float, float, bool]],
        poses: list[np.ndarray],
    ) -> None:
        """Display optional inspection windows for registration pairs and camera trajectory."""
        if self.cfg.registration.show_registration_pairs and self.cfg.show_stage_windows:
            logger.info("\n  Displaying pairwise registration inspection windows...")
            for idx in range(min(2, len(isolated_frames) - 1)):
                if isinstance(isolated_frames[idx], RGBDFrame) and isinstance(isolated_frames[idx + 1], RGBDFrame):
                    T_src_to_tgt = np.linalg.inv(poses[idx + 1]) @ poses[idx]
                    visualize_registration_pair(
                        isolated_frames[idx].pcd,
                        isolated_frames[idx + 1].pcd,
                        T_src_to_tgt,
                        title=f"Pair {idx:02d} -> {idx+1:02d}",
                    )

        if self.cfg.registration.show_camera_trajectory and self.cfg.show_stage_windows:
            logger.info("\n  Displaying estimated camera trajectory...")
            traj_geoms = draw_camera_trajectory(poses, scale=0.04)
            if traj_geoms:
                acc_clouds = [f for f, d in zip(aligned, diagnostics) if d is not None and d[2]]
                show(traj_geoms + acc_clouds, window_name="Estimated Camera Trajectory")

    def _save_and_display_stage3(
        self,
        aligned: list[o3d.geometry.PointCloud],
        diagnostics: list[tuple[float, float, bool]],
        is_human_mode: bool,
    ) -> None:
        """Save colored stage3 output point cloud and optionally render the Open3D inspection window."""
        colored_aligned_frames = color_frames_distinctly(aligned)
        if hasattr(self, "scan_folder") and self.scan_folder is not None:
            combined = o3d.geometry.PointCloud()
            for frame in colored_aligned_frames:
                combined += frame
            o3d.io.write_point_cloud(str(self.scan_folder / "stage3_output.ply"), combined)
            logger.info(f"  Saved: {self.scan_folder / 'stage3_output.ply'}")

        if self.cfg.show_stage_windows:
            rejected_indices = [i for i, d in enumerate(diagnostics) if d is not None and not d[2]]
            if rejected_indices:
                if is_human_mode:
                    logger.warning(f"\n  [!] Note: Stage 3 visualization displays all frames including rejected frame(s) {rejected_indices}.")
                    logger.info("      Rejected frames are excluded from Stage 4 fusion.")
                else:
                    logger.warning(f"\n  [!] Note: Visualizing all frames including rejected frame(s) {rejected_indices}.")
            screenshot_path = (self.scan_folder / "stage3_output.png") if (hasattr(self, "scan_folder") and self.scan_folder is not None) else None
            display_frames = prepare_display_geometry(
                colored_aligned_frames,
                voxel_size_m=self.cfg.stage3_4_display_voxel_size_m,
            )
            win_name = "Stage 3: registered/aligned human clouds" if is_human_mode else "Stage 3: registered/aligned clouds"
            show(
                display_frames,
                window_name=win_name,
                screenshot_path=screenshot_path,
            )

    def _check_pointcloud_loop_closure(
        self,
        isolated_frames: list[o3d.geometry.PointCloud | RGBDFrame],
        diagnostics: list[tuple[float, float, bool]],
    ) -> None:
        """Check loop closure consistency between first and last accepted frame if configured."""
        if not self.cfg.attempt_loop_closure:
            return
        accepted_indices = [idx for idx, (*_, acc) in enumerate(diagnostics) if acc]
        if len(accepted_indices) >= 8:
            last_idx = accepted_indices[-1]
            try:
                src = isolated_frames[last_idx].pcd if hasattr(isolated_frames[last_idx], "pcd") else isolated_frames[last_idx]
                dst = isolated_frames[0].pcd if hasattr(isolated_frames[0], "pcd") else isolated_frames[0]
                _, loop_fitness, loop_rmse = register_frame_pair(
                    src, dst, self.cfg.registration
                )
                if loop_fitness >= self.cfg.registration.min_accept_fitness:
                    logger.info(f"\n  Loop closure check: fitness={loop_fitness:.4f} (rmse={loop_rmse * 1000:.2f} mm) -- looks like a consistent full orbit.")
                else:
                    logger.warning(f"\n  [!] Loop closure check: low overlap (fitness={loop_fitness:.4f}) with initial frame.")
                    logger.info("      Drift may have accumulated over the full orbit.")
            except Exception as exc:
                logger.warning(f"\n  [!] Loop closure check skipped: {exc}")

    def _register_human(
        self,
        isolated_frames: list[o3d.geometry.PointCloud | RGBDFrame],
    ) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]], list[np.ndarray] | None]:
        """Align human RGB-D frames using odometry + Colored ICP + pose graph."""
        logger.info("  Stage 3: Registering human frames (RGB-D Odometry + Colored ICP + Pose Graph) ...")
        logger.info("=" * 60)
        aligned, diagnostics, poses, summary_stats = register_sequence_rgbd_human(
            isolated_frames, self.cfg.registration
        )
        self.last_registration_stats = summary_stats

        report = summary_stats.get("registration_report")
        if report is not None:
            if hasattr(self, "scan_folder") and self.scan_folder is not None:
                try:
                    with open(self.scan_folder / "registration_report.json", "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                except Exception as e:
                    logger.warning(f"  Failed to write registration_report.json to {self.scan_folder}: {e}")
            if hasattr(self.cfg, "output_dir") and self.cfg.output_dir is not None:
                try:
                    with open(self.cfg.output_dir / "registration_report.json", "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                except Exception as e:
                    logger.warning(f"  Failed to write registration_report.json to {self.cfg.output_dir}: {e}")

        self._log_human_registration_summary(diagnostics, summary_stats, isolated_frames)
        self._visualize_registration_diagnostics(isolated_frames, aligned, diagnostics, poses)
        self._check_pointcloud_loop_closure(isolated_frames, diagnostics)
        self._save_and_display_stage3(aligned, diagnostics, is_human_mode=True)
        return aligned, diagnostics, poses

    def _register_pointcloud(
        self,
        isolated_frames: list[o3d.geometry.PointCloud | RGBDFrame],
    ) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]], list[np.ndarray] | None]:
        """Align object point cloud frames using multiway pose graph or sequential FPFH+ICP."""
        if self.cfg.registration.use_multiway:
            logger.info("  Stage 3: Registering frames (Pose Graph Multiway Registration) ...")
        else:
            logger.info("  Stage 3: Registering frames (Sequential FPFH+RANSAC -> ICP) ...")
        logger.info("=" * 60)

        pcds = [f.pcd if hasattr(f, "pcd") else f for f in isolated_frames]
        if self.cfg.registration.use_multiway:
            aligned, diagnostics = register_sequence_multiway(pcds, self.cfg.registration)
        else:
            aligned, diagnostics = register_sequence(pcds, self.cfg.registration)

        pc_pairs = []
        rejected_indices: list[int] = []
        for i, diag in enumerate(diagnostics):
            if diag is None:
                continue
            fitness, rmse, accepted = diag
            status = "ACCEPTED" if accepted else "REJECTED -- excluded from fusion, consider recapturing this angle"
            if not accepted:
                rejected_indices.append(i)
            logger.debug(f"  frame {i:02d}: fitness={fitness:.4f}  "
                  f"rmse={rmse * 1000:.3f} mm  {status}")

            pc_pairs.append({
                "source": i,
                "target": max(0, i - 1),
                "success": bool(accepted),
                "method": "pointcloud_multiway" if self.cfg.registration.use_multiway else "pointcloud_sequential",
                "fitness": float(fitness),
                "rmse_mm": float(rmse * 1000.0) if np.isfinite(rmse) else 999.0,
                "rotation_deg": 0.0,
                "translation_mm": 0.0,
                "src_roi_pts": len(pcds[i].points) if (i < len(pcds) and hasattr(pcds[i], "points")) else 0,
                "tgt_roi_pts": len(pcds[max(0, i - 1)].points) if (max(0, i - 1) < len(pcds) and hasattr(pcds[max(0, i - 1)], "points")) else 0,
                "trans_disagree_mm": None,
                "rot_disagree_deg": None,
                "residual_mm": None,
                "reason": "accepted" if accepted else "rejected",
                "is_face_pair": False,
                "is_primary": True,
            })

        n_accepted = sum(1 for d in diagnostics if d is not None and d[2])
        n_total = len(diagnostics)
        logger.info(f"\n  Registration summary: {n_accepted} of {n_total} frames accepted into the reconstruction.")
        if n_total > 0 and (n_accepted / n_total) < 0.70:
            logger.warning("  Tip: low acceptance ratio (< 70%). Try moving the camera more slowly")
            logger.info("       or reducing --interval for higher frame overlap.")

        pc_rmses = [p["rmse_mm"] for p in pc_pairs if p["success"] and p["rmse_mm"] < 990.0]
        pc_report = {
            "head_only_registration": False,
            "total_pairs_evaluated": len(pc_pairs),
            "accepted_pairs_count": sum(1 for p in pc_pairs if p["success"]),
            "tier_counts": {"pointcloud": len(pc_pairs)},
            "mean_rmse_mm": float(np.mean(pc_rmses)) if pc_rmses else 0.0,
            "median_rmse_mm": float(np.median(pc_rmses)) if pc_rmses else 0.0,
            "max_rmse_mm": float(np.max(pc_rmses)) if pc_rmses else 0.0,
            "pairs": pc_pairs,
        }
        self.last_registration_stats = {
            "captured": n_total,
            "accepted": n_accepted,
            "rejected": n_total - n_accepted,
            "avg_rmse_mm": float(np.mean(pc_rmses)) if pc_rmses else 0.0,
            "median_rmse_mm": float(np.median(pc_rmses)) if pc_rmses else 0.0,
            "tier_counts": {"pointcloud": len(pc_pairs)},
            "tier_summary_str": f"{n_accepted} pointcloud accepted, {n_total - n_accepted} rejected",
            "diagnostics": diagnostics,
            "n_edges": n_accepted,
            "n_loop": 0,
            "registration_report": pc_report,
        }
        if hasattr(self, "scan_folder") and self.scan_folder is not None:
            try:
                with open(self.scan_folder / "registration_report.json", "w", encoding="utf-8") as f:
                    json.dump(pc_report, f, indent=2)
            except Exception as e:
                logger.warning(f"  Failed to write registration_report.json to {self.scan_folder}: {e}")
        if hasattr(self.cfg, "output_dir") and self.cfg.output_dir is not None:
            try:
                with open(self.cfg.output_dir / "registration_report.json", "w", encoding="utf-8") as f:
                    json.dump(pc_report, f, indent=2)
            except Exception as e:
                logger.warning(f"  Failed to write registration_report.json to {self.cfg.output_dir}: {e}")

        self._check_pointcloud_loop_closure(isolated_frames, diagnostics)
        self._save_and_display_stage3(aligned, diagnostics, is_human_mode=False)
        return aligned, diagnostics, None

    def register(
        self,
        isolated_frames: list[o3d.geometry.PointCloud | RGBDFrame],
    ) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]], list[np.ndarray] | None]:
        """
        Align all isolated frames into the shared reference coordinate system.

        Returns
        -------
        aligned:
            List of all aligned point clouds in frame-0 coordinates.
        diagnostics:
            List of (fitness, rmse, accepted) tuples per frame.
        poses:
            List of 4x4 camera poses (or None in sequential pointcloud mode).
        """
        logger.info("\n" + "=" * 60)
        is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human") or (
            self.cfg.target.name in ("head", "body") and self.cfg.registration.registration_mode != "pointcloud"
        )
        if is_human_mode:
            return self._register_human(isolated_frames)
        else:
            return self._register_pointcloud(isolated_frames)

    # ------------------------------------------------------------------
    # Stage 4: fuse
    # ------------------------------------------------------------------

    def _filter_fusion_eligible_frames(
        self,
        aligned_frames: list[o3d.geometry.PointCloud],
        diagnostics: list[tuple[float, float, bool]] | None,
    ) -> tuple[list[o3d.geometry.PointCloud], list[int]]:
        """Filter registered frames based on diagnostics and secondary fusion thresholds."""
        if diagnostics is None:
            return aligned_frames, list(range(len(aligned_frames)))

        is_human_mode = (
            (getattr(self.cfg.registration, "registration_mode", "") == "rgbd_human")
            or (getattr(self.cfg.target, "name", "") in ("head", "body"))
        )
        min_fit_fusion = getattr(self.cfg.registration, "min_fitness_for_fusion", 0.60)
        max_lm_res_fusion = getattr(self.cfg.registration, "max_landmark_residual_for_fusion_mm", 5.0)
        min_face_fit = getattr(self.cfg.registration, "min_face_pair_fitness_for_fusion", 0.75)
        max_face_rmse = getattr(self.cfg.registration, "max_face_pair_rmse_for_fusion_mm", 4.0)
        face_y_thresh = getattr(self.cfg.registration, "face_region_center_y_threshold_m", 0.0)

        frame_scores = {}
        fusion_eligible = {}
        fusion_reasons = {}
        residuals = {}
        frame_methods = {}
        if hasattr(self, "last_registration_stats") and self.last_registration_stats:
            frame_scores = self.last_registration_stats.get("frame_scores", {})
            fusion_eligible = self.last_registration_stats.get("fusion_eligible", {})
            fusion_reasons = self.last_registration_stats.get("fusion_reasons", {})
            residuals = self.last_registration_stats.get("residuals") or self.last_registration_stats.get("frame_residuals", {})
            frame_methods = self.last_registration_stats.get("frame_methods", {})

        from registration import is_upper_face_frame

        accepted_indices = []
        gate_report_records = []
        gate_exclusions = {
            "min_fitness_for_fusion": 0,
            "max_landmark_residual_for_fusion_mm": 0,
            "min_face_pair_fitness_for_fusion": 0,
            "max_face_pair_rmse_for_fusion_mm": 0,
        }

        for i, d in enumerate(diagnostics):
            reg_accepted = bool(d is not None and d[2]) if i > 0 else True
            fit = float(d[0]) if (d is not None and d[0] is not None) else (1.0 if i == 0 else 0.0)
            rmse_m = float(d[1]) if (d is not None and d[1] is not None) else (0.0 if i == 0 else float("inf"))
            rmse_mm = (rmse_m * 1000.0) if np.isfinite(rmse_m) else None

            score_str = frame_scores.get(i, "")
            lm_res = residuals.get(i)
            if lm_res is None and "residual=" in score_str:
                try:
                    lm_res = float(score_str.split("residual=")[1].split("mm")[0])
                except Exception:
                    pass

            method = frame_methods.get(i, "reference" if i == 0 else ("none" if not reg_accepted else "registered"))
            is_upper = is_upper_face_frame(aligned_frames[i], face_y_thresh) if i < len(aligned_frames) else False

            # Evaluate each gate vs threshold
            failed_gates = {}
            if fit < min_fit_fusion:
                failed_gates["min_fitness_for_fusion"] = {
                    "actual": round(fit, 4),
                    "threshold": float(min_fit_fusion),
                }
            if lm_res is not None and lm_res > max_lm_res_fusion:
                failed_gates["max_landmark_residual_for_fusion_mm"] = {
                    "actual": round(float(lm_res), 2),
                    "threshold": float(max_lm_res_fusion),
                }
            if fit < min_face_fit:
                failed_gates["min_face_pair_fitness_for_fusion"] = {
                    "actual": round(fit, 4),
                    "threshold": float(min_face_fit),
                }
            if rmse_mm is not None and rmse_mm > max_face_rmse:
                failed_gates["max_face_pair_rmse_for_fusion_mm"] = {
                    "actual": round(rmse_mm, 2),
                    "threshold": float(max_face_rmse),
                }

            is_fused = False
            reasons = []

            if i == 0:
                is_fused = True
                accepted_indices.append(0)
            elif not reg_accepted:
                is_fused = False
                reasons.append("rejected in Stage 3 registration")
            elif is_human_mode:
                if fusion_eligible and i in fusion_eligible:
                    if fusion_eligible[i]:
                        is_fused = True
                        accepted_indices.append(i)
                    else:
                        is_fused = False
                        reason_ex = fusion_reasons.get(i, "stricter fusion gate")
                        reasons.append(reason_ex)
                        logger.debug(f"  frame {i}: pose accepted, EXCLUDED from fusion ({reason_ex})")
                else:
                    if fit < min_fit_fusion:
                        reasons.append(f"fitness {fit:.2f} < {min_fit_fusion:.2f} fusion threshold")
                    if lm_res is not None and lm_res > max_lm_res_fusion:
                        reasons.append(f"landmark residual {lm_res:.1f}mm > {max_lm_res_fusion:.1f}mm fusion threshold")

                    if not reasons:
                        is_fused = True
                        accepted_indices.append(i)
                    else:
                        is_fused = False
                        for r_msg in reasons:
                            logger.debug(f"  frame {i}: pose accepted, EXCLUDED from fusion ({r_msg})")
            else:
                is_fused = True
                accepted_indices.append(i)

            decision_str = "fused" if is_fused else "excluded"

            # Tally gate exclusions for registered frames that were dropped by fusion gates
            if not is_fused and reg_accepted:
                for g_name in failed_gates:
                    if g_name in gate_exclusions:
                        gate_exclusions[g_name] += 1

            record = {
                "frame_index": i,
                "registration_method": method,
                "fitness": round(fit, 4),
                "rmse": round(rmse_m, 6) if np.isfinite(rmse_m) else None,
                "landmark_residual_mm": round(float(lm_res), 2) if lm_res is not None else None,
                "upper_lower_face_region": "upper" if is_upper else "lower",
                "is_upper_face_region": is_upper,
                "failed_gates": failed_gates,
                "final_decision": decision_str,
                "reason": "; ".join(reasons) if reasons else "accepted for fusion",
            }
            gate_report_records.append(record)

        accepted_frames = [aligned_frames[i] for i in accepted_indices]
        if not accepted_frames:
            logger.warning("  [!] Warning: No frames met acceptance criteria; falling back to frame 0.")
            accepted_frames = aligned_frames[:1]
            accepted_indices = [0]
            if gate_report_records:
                gate_report_records[0]["final_decision"] = "fused"
                gate_report_records[0]["reason"] = "fallback to frame 0"

        # One-line summary
        n_registered = sum(1 for d in diagnostics if d is not None and d[2])
        n_fused = len(accepted_indices)
        gate_summary = (
            f"Stage 4 Gate Summary: {n_registered} frames registered, {n_fused} fused, "
            f"excluded by gates: min_fitness_for_fusion={gate_exclusions['min_fitness_for_fusion']}, "
            f"max_landmark_residual_for_fusion_mm={gate_exclusions['max_landmark_residual_for_fusion_mm']}, "
            f"min_face_pair_fitness_for_fusion={gate_exclusions['min_face_pair_fitness_for_fusion']}, "
            f"max_face_pair_rmse_for_fusion_mm={gate_exclusions['max_face_pair_rmse_for_fusion_mm']}"
        )
        logger.info(gate_summary)
        print(gate_summary)

        gate_report = {
            "summary": {
                "frames_total": len(diagnostics),
                "frames_registered": n_registered,
                "frames_fused": n_fused,
                "frames_excluded": len(diagnostics) - n_fused,
                "excluded_by_gate": gate_exclusions,
            },
            "thresholds": {
                "min_fitness_for_fusion": float(min_fit_fusion),
                "max_landmark_residual_for_fusion_mm": float(max_lm_res_fusion),
                "min_face_pair_fitness_for_fusion": float(min_face_fit),
                "max_face_pair_rmse_for_fusion_mm": float(max_face_rmse),
            },
            "frames": gate_report_records,
        }

        save_dirs = [self.cfg.output_dir]
        if hasattr(self, "scan_folder") and self.scan_folder is not None:
            save_dirs.append(self.scan_folder)

        for d_path in save_dirs:
            try:
                d_path.mkdir(parents=True, exist_ok=True)
                out_json = d_path / "stage4_gate_report.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(gate_report, f, indent=2)
                logger.info(f"  Saved: {out_json}")
            except Exception as exc:
                logger.warning(f"  [!] Failed to save stage4_gate_report.json to {d_path}: {exc}")

        return accepted_frames, accepted_indices

    def _reconstruct_volume(
        self,
        accepted_frames: list[o3d.geometry.PointCloud],
        accepted_indices: list[int],
        raw_frames: list | None,
        poses: list[np.ndarray] | None,
    ) -> tuple[o3d.geometry.PointCloud, str]:
        """Perform 3D reconstruction via volumetric TSDF integration or point cloud concatenation."""
        if (
            self.cfg.refine
            and raw_frames is not None
            and raw_frames
            and isinstance(raw_frames[0], RGBDFrame)
            and poses is not None
        ):
            logger.info("\n" + "=" * 60)
            logger.info("  Reconstruction mode: HEAD REFINEMENT (--refine)")
            logger.info("=" * 60)
            refine_cfg = self.cfg.head_refinement
            acc_raw = [raw_frames[i] for i in accepted_indices]
            acc_poses = [poses[i] for i in accepted_indices]

            # 1. Filter frames by landmark count (< 6 dropped)
            frames_filtered = []
            poses_filtered = []
            lm_data = []
            for frame, pose in zip(acc_raw, acc_poses):
                lm_dict = detect_face_landmarks_3d(frame.color_bgr, frame.depth_m, frame.intrinsics, quiet=True)
                n_lm = len(lm_dict) if lm_dict is not None else 0
                if n_lm < refine_cfg.min_fusion_landmarks:
                    logger.info(
                        f"  [head-refine] Excluded frame {frame.frame_id}: "
                        f"landmark count ({n_lm}) < {refine_cfg.min_fusion_landmarks}"
                    )
                    continue
                frames_filtered.append(frame)
                poses_filtered.append(pose)
                lm_data.append(lm_dict)

            if len(frames_filtered) >= 2:
                acc_raw = frames_filtered
                acc_poses = poses_filtered
            else:
                logger.warning("  [head-refine] Too few frames met landmark threshold; keeping all accepted frames.")
                lm_data = [detect_face_landmarks_3d(f.color_bgr, f.depth_m, f.intrinsics, quiet=True) for f in acc_raw]

            # 2. Derive global world head center from detected landmarks
            world_centers = []
            for f, p, lm in zip(acc_raw, acc_poses, lm_data):
                if lm:
                    pos = [v["position"] for v in lm.values()]
                    c_cam = np.mean(pos, axis=0)
                    c_world = (p[:3, :3] @ c_cam) + p[:3, 3]
                    world_centers.append(c_world)
            global_world_center = np.mean(world_centers, axis=0) if world_centers else np.zeros(3)

            # 3. Apply per-pixel depth quality filter and head-only masking
            head_frames = []
            head_pcds = []
            for f, p, lm in zip(acc_raw, acc_poses, lm_data):
                if lm:
                    pos = [v["position"] for v in lm.values()]
                    c_cam = np.mean(pos, axis=0)
                    chin_pos = np.array(lm["chin"]["position"]) if "chin" in lm else None
                    if chin_pos is not None:
                        v_chin = chin_pos - c_cam
                        norm_v = np.linalg.norm(v_chin)
                        down_vec = v_chin / norm_v if norm_v > 1e-4 else np.array([0.0, 1.0, 0.0])
                        chin_dist = float(norm_v)
                    else:
                        down_vec = np.array([0.0, 1.0, 0.0])
                        chin_dist = 0.10
                else:
                    inv_p = np.linalg.inv(p)
                    c_cam = (inv_p[:3, :3] @ global_world_center) + inv_p[:3, 3]
                    down_vec = np.array([0.0, 1.0, 0.0])
                    chin_dist = 0.10

                # Per-pixel depth quality filter (drop normals > 70 deg and depth jumps > 8 mm)
                depth_q, q_stats = filter_depth_pixel_quality(
                    f.depth_m,
                    f.intrinsics,
                    max_normal_angle_deg=refine_cfg.filter_normal_angle_deg,
                    max_depth_jump_m=refine_cfg.filter_depth_jump_mm / 1000.0,
                )

                # Head-only fusion mask (sphere radius ~0.13 m, chin extension <= 25 mm)
                masked_depth, masked_color, dropped_count, total_valid = mask_frame_to_head_roi(
                    color_bgr=f.color_bgr,
                    depth_m=depth_q,
                    intrinsics=f.intrinsics,
                    head_center_cam=c_cam,
                    down_vec=down_vec,
                    chin_dist=chin_dist,
                    radius_m=refine_cfg.head_fusion_radius_m,
                    chin_margin_m=refine_cfg.head_fusion_chin_margin_m,
                )
                pct_dropped = (dropped_count / total_valid * 100.0) if total_valid > 0 else 0.0
                logger.info(
                    f"  [head-only fusion] Frame {f.frame_id}: "
                    f"dropped {dropped_count:,} / {total_valid:,} pixels ({pct_dropped:.1f}%) outside head region"
                )

                f_head = f.clone()
                f_head.depth_m = masked_depth
                f_head.color_bgr = masked_color

                # Extract head ROI point cloud for ICP & metrics
                h_img, w_img = masked_depth.shape[:2]
                fx_c, fy_c = f.intrinsics.get_focal_length()
                cx_c, cy_c = f.intrinsics.get_principal_point()
                u_g, v_g = np.meshgrid(np.arange(w_img), np.arange(h_img))
                valid_head = (masked_depth > 0.05) & np.isfinite(masked_depth)

                pts_x = (u_g[valid_head] - cx_c) * masked_depth[valid_head] / fx_c
                pts_y = (v_g[valid_head] - cy_c) * masked_depth[valid_head] / fy_c
                pts_z = masked_depth[valid_head]
                pcd_h = o3d.geometry.PointCloud()
                pcd_h.points = o3d.utility.Vector3dVector(np.column_stack([pts_x, pts_y, pts_z]))
                colors_h = (masked_color[valid_head][:, ::-1] / 255.0).astype(np.float64)
                pcd_h.colors = o3d.utility.Vector3dVector(colors_h)
                f_head.pcd = pcd_h

                head_frames.append(f_head)
                head_pcds.append(pcd_h)

            # 4. Model-based refinement
            logger.info("  Running model-based pose refinement ...")
            refined_poses, iter_stats, _ = refine_poses_model_based(
                head_frames,
                acc_poses,
                head_pcds,
                voxel_length_m=0.002,
                sdf_trunc_m=0.008,
                depth_trunc_m=refine_cfg.depth_trunc_m,
                max_distance_m=refine_cfg.refine_max_distance_mm / 1000.0,
                max_trans_m=refine_cfg.refine_max_trans_mm / 1000.0,
                max_rot_deg=refine_cfg.refine_max_rot_deg,
                max_iterations=refine_cfg.refine_iterations,
                early_stop_improvement_m=refine_cfg.refine_early_stop_mm / 1000.0,
            )

            # 5. Scalable TSDF fusion with refined poses
            logger.info(
                f"  Integrating {len(head_frames)} head frames into ScalableTSDFVolume "
                f"(voxel {refine_cfg.voxel_length_mm} mm, sdf_trunc {refine_cfg.sdf_trunc_mm} mm) ..."
            )
            final_mesh, fused = fuse_head_scalable_tsdf(
                head_frames,
                refined_poses,
                voxel_length_m=refine_cfg.voxel_length_mm / 1000.0,
                sdf_trunc_m=refine_cfg.sdf_trunc_mm / 1000.0,
                depth_trunc_m=refine_cfg.depth_trunc_m,
                taubin_iterations=refine_cfg.taubin_iterations,
                min_component_ratio=refine_cfg.min_component_ratio,
            )

            if len(fused.points) < 100 and len(final_mesh.vertices) > 0:
                fused = o3d.geometry.PointCloud(final_mesh.vertices)
                fused.colors = final_mesh.vertex_colors
                fused.normals = final_mesh.vertex_normals

            # Save fused_mesh.ply
            mesh_out = self.cfg.output_dir / "fused_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_out), final_mesh)
            logger.info(f"  Saved fused mesh: {mesh_out}")
            if hasattr(self, "scan_folder") and self.scan_folder is not None:
                o3d.io.write_triangle_mesh(str(self.scan_folder / "fused_mesh.ply"), final_mesh)
                logger.info(f"  Saved: {self.scan_folder / 'fused_mesh.ply'}")

            # 6. Quality metrics
            metrics = compute_reconstruction_quality_metrics(
                final_mesh,
                head_frames,
                acc_poses,
                refined_poses,
                head_pcds,
                refine_cfg,
            )

            # Save reconstruction_quality.json
            qual_path = self.cfg.output_dir / "reconstruction_quality.json"
            with qual_path.open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, indent=2)
            logger.info(f"  Saved quality metrics: {qual_path}")
            if hasattr(self, "scan_folder") and self.scan_folder is not None:
                with (self.scan_folder / "reconstruction_quality.json").open("w", encoding="utf-8") as fp:
                    json.dump(metrics, fp, indent=2)

            # Log quality metrics summary table
            res_b = metrics.get("residual_before_refinement_mm", {})
            res_a = metrics.get("residual_after_refinement_mm", {})
            rep = metrics.get("repeatability_odd_even_mm", {})
            logger.info("\n" + "=" * 60)
            logger.info("  RECONSTRUCTION QUALITY METRICS (--refine)")
            logger.info("=" * 60)
            logger.info(f"  Residual before refinement: median = {res_b.get('median')} mm, p90 = {res_b.get('p90')} mm")
            logger.info(f"  Residual after refinement : median = {res_a.get('median')} mm, p90 = {res_a.get('p90')} mm")
            logger.info("  Repeatability (odd vs even frames):")
            logger.info(f"    - Symmetric Chamfer dist: {rep.get('chamfer_distance_mm')} mm")
            logger.info(f"    - 95th-percentile dist  : {rep.get('p95_distance_mm')} mm")
            logger.info("  Mesh topology:")
            logger.info(f"    - Vertices  : {metrics.get('mesh_vertex_count'):,}")
            logger.info(f"    - Triangles : {metrics.get('mesh_triangle_count'):,}")
            logger.info(f"    - Holes     : {metrics.get('mesh_hole_count')}")
            logger.info("=" * 60 + "\n")

            fusion_name = "SCALABLE TSDF (REFINED)"
        elif (
            self.cfg.fusion_method == "tsdf"
            and raw_frames is not None
            and raw_frames
            and isinstance(raw_frames[0], RGBDFrame)
            and poses is not None
        ):
            logger.info("  Reconstruction mode: TSDF Volumetric Integration")
            acc_raw = [raw_frames[i] for i in accepted_indices]
            acc_poses = [poses[i] for i in accepted_indices]
            is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human") or (self.cfg.target.name in ("head", "body"))
            fused = fuse_tsdf_volume(
                acc_raw,
                acc_poses,
                self.cfg.preprocess,
                voxel_length=self.cfg.tsdf_voxel_length_m,
                sdf_trunc=self.cfg.tsdf_trunc_m,
                skip_final_clustering=is_human_mode,
            )
            fusion_name = "TSDF"
        else:
            logger.info("  Reconstruction mode: Point Cloud Concatenation")
            is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human") or (self.cfg.target.name in ("head", "body"))
            human_voxel_size = 0.002 if is_human_mode else None
            fused = fuse_point_clouds(
                accepted_frames,
                self.cfg.preprocess,
                skip_final_clustering=is_human_mode,
                skip_outlier_removal=is_human_mode,
                voxel_size=human_voxel_size,
                reg_cfg=self.cfg.registration if is_human_mode else None,
            )
            fusion_name = "POINT CLOUD"

        logger.info(f"  Reconstruction:\n    method: {fusion_name}\n    points: {len(fused.points):,}")
        return fused, fusion_name

    def _save_and_display_stage4(self, fused: o3d.geometry.PointCloud, fusion_name: str) -> None:
        """Save stage4 reconstructed point cloud and optionally display preview."""
        if hasattr(self, "scan_folder") and self.scan_folder is not None:
            o3d.io.write_point_cloud(str(self.scan_folder / "stage4_output.ply"), fused)
            logger.info(f"  Saved: {self.scan_folder / 'stage4_output.ply'}")

        if self.cfg.show_stage_windows:
            display_pcd = prepare_display_geometry(
                fused,
                voxel_size_m=self.cfg.stage3_4_display_voxel_size_m,
            )
            if not display_pcd.has_colors():
                display_pcd.paint_uniform_color([0.65, 0.65, 0.65])
            screenshot_path = (self.scan_folder / "stage4_output.png") if (hasattr(self, "scan_folder") and self.scan_folder is not None) else None
            show(
                [display_pcd],
                window_name=f"Stage 4: fused reconstruction ({fusion_name})",
                screenshot_path=screenshot_path,
            )

        if self.cfg.save_intermediate:
            out = self.cfg.output_dir / "fused.ply"
            o3d.io.write_point_cloud(str(out), fused)
            logger.info(f"  Saved: {out}")

    def fuse(
        self,
        aligned_frames: list[o3d.geometry.PointCloud],
        diagnostics: list[tuple[float, float, bool]] | None = None,
        raw_frames: list | None = None,
        poses: list[np.ndarray] | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Merge registered frames into one clean point cloud.

        Only frames with accepted=True in diagnostics are included in the final fusion.
        Supports volumetric TSDF integration for human RGB-D frames.

        Parameters
        ----------
        aligned_frames:
            Registered per-frame clouds from :meth:`register`.
        diagnostics:
            Optional registration diagnostics tuples (fitness, rmse, accepted).
        raw_frames:
            Optional original raw / isolated frames (e.g. RGBDFrame instances).
        poses:
            Optional list of 4x4 camera poses.

        Returns
        -------
        o3d.geometry.PointCloud
            Fused, denoised reconstruction.
        """
        logger.info("\n" + "=" * 60)
        logger.info("  Stage 4: Fusing aligned frames ...")
        logger.info("=" * 60)

        accepted_frames, accepted_indices = self._filter_fusion_eligible_frames(
            aligned_frames, diagnostics
        )
        n_skipped = len(aligned_frames) - len(accepted_frames)
        skip_msg = f" (excluded {n_skipped} rejected frame{'s' if n_skipped != 1 else ''})" if n_skipped > 0 else ""
        logger.info(f"  Fusing {len(accepted_frames)} accepted frame(s){skip_msg} ...")

        fused, fusion_name = self._reconstruct_volume(
            accepted_frames, accepted_indices, raw_frames, poses
        )
        self._save_and_display_stage4(fused, fusion_name)
        return fused

    # ------------------------------------------------------------------
    # Stage 5: measure
    # ------------------------------------------------------------------

    def measure(self, fused_pcd: o3d.geometry.PointCloud) -> dict:
        """
        Segment the fused cloud and return dimensional measurements.

        The ``"geometry_for_viz"`` key is removed from the result before it
        is JSON-serialised and returned (Open3D objects are not serialisable).

        Parameters
        ----------
        fused_pcd:
            Fused reconstruction from :meth:`fuse`.

        Returns
        -------
        dict
            JSON-serialisable measurement result.
        """
        logger.info("\n" + "=" * 60)
        logger.info("  Stage 5: Segmenting and measuring ...")
        logger.info("=" * 60)

        segmented = self.target.segment(fused_pcd)
        logger.info(f"  Segmented cloud: {len(segmented.points):>7} points")

        # Fast shape-plausibility pre-check (e.g. for BoxTarget or HeadTarget)
        if hasattr(self.target, "is_plausible_box"):
            is_plausible, reason = self.target.is_plausible_box(segmented)
            if not is_plausible:
                logger.info("\n" + "!" * 60)
                logger.error("  [!] NOT BOX SHAPED -- MEASUREMENT REFUSED")
                logger.info("  " + "=" * 56)
                logger.info(f"  Reason: {reason}")
                logger.info("!" * 60)
                return {
                    "target": "box",
                    "status": "not_box_shaped",
                    "reason": reason,
                }
        elif hasattr(self.target, "is_plausible_head"):
            is_plausible, reason = self.target.is_plausible_head(segmented)
            if not is_plausible:
                logger.info("\n" + "!" * 60)
                logger.error("  [!] NOT HEAD SHAPED -- MEASUREMENT REFUSED")
                logger.info("  " + "=" * 56)
                logger.info(f"  Reason: {reason}")
                logger.info("!" * 60)
                return {
                    "target": "head",
                    "status": "not_head_shaped",
                    "reason": reason,
                }

        result = self.target.measure(segmented)

        # Pop the Open3D geometry before printing / saving — it is not
        # JSON-serialisable and must be kept separate for visualisation.
        obb = result.pop("geometry_for_viz", None)

        if result.get("target") == "head" or "cranial_envelope_3d" in result:
            cranial = result.get("cranial_envelope_3d", {})
            status = result.get("status", "unverified")
            logger.info(f"\n  Target : HEAD (3D Geometric Cranial Envelope)")
            logger.info(f"  Status : {status.upper()}")
            logger.info(f"  Breadth: {cranial.get('breadth_mm', 0.0):>7.1f} mm  (Left-Right lateral width)")
            logger.info(f"  Length : {cranial.get('length_mm', 0.0):>7.1f} mm  (Anterior-Posterior depth)")
            logger.info(f"  Height : {cranial.get('height_mm', 0.0):>7.1f} mm  (Superior-Inferior height)")
            vol = result.get("volume_m3") or result.get("oriented_bbox", {}).get("volume_m3")
            est_vol = result.get("oriented_bbox", {}).get("estimated_volume_m3")
            if vol is not None:
                logger.info(f"  Volume : {vol * 1e6:.1f} cm^3 (validated cranial envelope volume)")
            elif est_vol is not None:
                logger.info(f"  Volume : UNVERIFIED (estimated: {est_vol * 1e6:.1f} cm^3 -- diagnostic only)")
            else:
                logger.info("  Volume : UNVERIFIED (null)")
            diags = result.get("diagnostics", [])
            if diags:
                logger.info("  Diagnostics:")
                for d in diags:
                    logger.info(f"    - {d}")
        else:
            planarity = result.get("planarity_check")
            if planarity:
                logger.info("\n  " + "-" * 50)
                logger.info("  PLANARITY CHECK (Face RMS Deviations):")
                rms_list = planarity.get("rms_deviation_mm", [])
                if len(rms_list) == 6:
                    logger.info(f"    Axis 0 faces (near/far): {rms_list[0]:.2f} mm | {rms_list[1]:.2f} mm")
                    logger.info(f"    Axis 1 faces (near/far): {rms_list[2]:.2f} mm | {rms_list[3]:.2f} mm")
                    logger.info(f"    Axis 2 faces (near/far): {rms_list[4]:.2f} mm | {rms_list[5]:.2f} mm")
                else:
                    logger.info(f"    RMS deviations         : {rms_list}")
                logger.info(f"    Max Face RMS Deviation : {planarity.get('max_rms_mm', 0.0):.2f} mm (limit: 4.0 mm)")

                warning = planarity.get("warning")
                if warning:
                    logger.info("\n  " + "!" * 58)
                    logger.warning(f"  [!] WARNING: {warning.upper()}")
                    logger.info("  " + "!" * 58)
                else:
                    logger.info("    Box Shape Validation   : PASS (Rigid planar box)")
                logger.info("  " + "-" * 50)

            obb_info = result.get("oriented_bbox", {})
            meas_dims_m = sorted(
                [obb_info.get("length_m", 0.0), obb_info.get("width_m", 0.0), obb_info.get("height_m", 0.0)],
                reverse=True,
            )

            # ---- ground-truth reference comparison (if provided) -----------
            if self.cfg.reference_dims_m is not None:
                ref_dims_m = sorted(self.cfg.reference_dims_m, reverse=True)
                axis_names = ["length", "width", "height"]
                ref_comparison = {}
                for name, m_val, r_val in zip(axis_names, meas_dims_m, ref_dims_m):
                    err_mm = abs(m_val - r_val) * 1000.0
                    err_pct = (abs(m_val - r_val) / r_val * 100.0) if r_val > 0 else 0.0
                    ref_comparison[name] = {
                        "measured_m": round(m_val, 4),
                        "reference_m": round(r_val, 4),
                        "measured_mm": round(m_val * 1000.0, 1),
                        "reference_mm": round(r_val * 1000.0, 1),
                        "error_mm": round(err_mm, 2),
                        "error_pct": round(err_pct, 2),
                    }
                result["reference_comparison"] = ref_comparison

            ref_comp = result.get("reference_comparison", {})

            logger.info(f"\n  Target : {result.get('target', '?')}")

            def _fmt_dim_line(label: str, key: str) -> str:
                val_mm = obb_info.get(f"{key}_m", 0.0) * 1000.0
                line = f"  {label:<6} : {val_mm:>7.1f} mm"
                if key in ref_comp:
                    info = ref_comp[key]
                    ref_mm = info["reference_mm"]
                    err_mm = info["error_mm"]
                    err_pct = info["error_pct"]
                    line += f"  (ref: {ref_mm:5.1f} mm | error: {err_mm:4.1f} mm, {err_pct:4.1f}%)"
                return line

            logger.info(_fmt_dim_line("Length", "length"))
            logger.info(_fmt_dim_line("Width", "width"))
            logger.info(_fmt_dim_line("Height", "height"))
            vol = obb_info.get("volume_m3")
            est_vol = obb_info.get("estimated_volume_m3")
            if vol is not None:
                logger.info(f"  Volume : {vol * 1e6:.1f} cm^3 (validated)")
            elif est_vol is not None:
                logger.info(f"  Volume : UNVERIFIED (estimated: {est_vol * 1e6:.1f} cm^3 -- underlying dimensions unverified)")
            else:
                logger.info("  Volume : UNVERIFIED")
            aabb_ext = result.get("axis_aligned_bbox_extent_m", [])
            if aabb_ext:
                ext_mm = [f"{v * 1000:.1f}" for v in aabb_ext]
                logger.info(f"  AABB   : {' x '.join(ext_mm)} mm (axis-aligned reference)")

        obb_ls = obb_lineset(obb) if obb is not None else None

        if hasattr(self, "scan_folder") and self.scan_folder is not None:
            o3d.io.write_point_cloud(str(self.scan_folder / "stage5_output.ply"), segmented)
            logger.info(f"  Saved: {self.scan_folder / 'stage5_output.ply'}")

        if self.cfg.show_stage_windows and obb is not None:
            # TASK 1 & 2: Display-only downsampled copy for interactive Stage 5 window.
            # The actual measurements (head breadth/length/height, OBB fitting) and the
            # saved files (segmented_object.ply, stage5_output.ply, measurement.json)
            # strictly keep using the full-density cloud to avoid losing measurement accuracy.
            seg_display = prepare_display_geometry(
                segmented,
                voxel_size_m=self.cfg.stage5_display_voxel_size_m,
            )
            if not seg_display.has_colors():
                seg_display.paint_uniform_color([0.18, 0.72, 0.38])
            geoms_to_show = [seg_display, obb_ls] if obb_ls is not None else [seg_display]
            screenshot_path = (self.scan_folder / "stage5_output.png") if (hasattr(self, "scan_folder") and self.scan_folder is not None) else None
            show(
                geoms_to_show,
                window_name="Stage 5: measured object",
                screenshot_path=screenshot_path,
            )

        # ---- persist results -------------------------------------------
        meas_path = self.cfg.output_dir / "measurement.json"
        with meas_path.open("w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        logger.info(f"\n  Saved measurement: {meas_path}")

        seg_path = self.cfg.output_dir / "segmented_object.ply"
        o3d.io.write_point_cloud(str(seg_path), segmented)
        logger.info(f"  Saved segmented cloud: {seg_path}")

        if hasattr(self, "scan_folder") and self.scan_folder is not None and meas_path.exists():
            target_meas = self.scan_folder / "measurement.json"
            if meas_path.resolve() != target_meas.resolve():
                shutil.copy(meas_path, target_meas)
                logger.info(f"  Saved: {target_meas}")

        return result

    # ------------------------------------------------------------------
    # Sufficiency Gate & Full pipeline
    # ------------------------------------------------------------------

    def check_measurement_sufficiency(
        self,
        diagnostics: list[tuple[float, float, bool]] | None,
        fused_pcd: o3d.geometry.PointCloud,
        aligned_frames: list[o3d.geometry.PointCloud] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate whether the registered and fused scan has sufficient data for reliable measurement.

        Checks:
          1. Number of accepted frames >= 3.
          2. Fused point count >= min_object_points * 3.
          3. Viewpoint spread among accepted frames' centroids >= 0.03 m (3.0 cm).

        Returns
        -------
        tuple[bool, list[str]]
            (is_sufficient, reasons_if_failed)
        """
        reasons: list[str] = []
        n_accepted = sum(1 for d in (diagnostics or []) if d is not None and d[2])
        n_total = len(diagnostics) if diagnostics else 0

        # Check 1: at least 3 accepted frames
        if n_accepted < 3:
            reasons.append(f"only {n_accepted} of {n_total} frames were accepted (need >= 3)")

        # Check 2: fused cloud point count
        min_fused = self.cfg.target.min_object_points * 3
        fused_pts = len(fused_pcd.points)
        if fused_pts < min_fused:
            reasons.append(f"fused cloud has only {fused_pts} points (need >= {min_fused})")

        # Check 3: viewpoint spread among accepted frames
        if aligned_frames and diagnostics:
            accepted_clouds = [
                f for f, d in zip(aligned_frames, diagnostics)
                if d is not None and d[2] and len(f.points) > 0
            ]
            if len(accepted_clouds) >= 2:
                centroids = [np.mean(np.asarray(c.points), axis=0) for c in accepted_clouds]
                max_spread_m = float(max(
                    np.linalg.norm(c1 - c2)
                    for i, c1 in enumerate(centroids)
                    for c2 in centroids[i + 1:]
                ))
                if max_spread_m < 0.03:
                    reasons.append(
                        f"accepted frames only span {max_spread_m * 100.0:.1f} cm (need >= 3.0 cm) "
                        "— camera barely moved between shots"
                    )
            elif n_accepted >= 2:
                reasons.append("fewer than 2 valid point clouds to compute viewpoint spread")

        return (len(reasons) == 0, reasons)

    def run(self) -> dict:
        """
        Execute all pipeline stages in sequence.

        BOX mode:   capture → isolate → register → fuse → measure (dimensions)
        HUMAN mode: capture → isolate → register → fuse → summary  (no box measurement)

        Returns
        -------
        dict
            JSON-serialisable result.  For HUMAN mode this contains reconstruction
            statistics rather than dimensional measurements.
        """
        if self.cfg.random_seed is not None:
            import random
            import open3d as o3d
            random.seed(self.cfg.random_seed)
            np.random.seed(self.cfg.random_seed)
            o3d.utility.random.seed(self.cfg.random_seed)

        is_body_mode = (self.cfg.target.name == "body" and self.cfg.registration.registration_mode == "rgbd_human")

        raw_frames = self.capture_frames()

        if self.cfg.save_raw:
            from camera.recording import save_raw_capture
            save_dir = self.cfg.save_raw_dir
            if save_dir is None:
                save_dir = self.scan_folder / "raw_capture"
            save_raw_capture(raw_frames, save_dir, self.cfg)

        isolated                    = self.isolate_frames(raw_frames)
        aligned, diagnostics, poses = self.register(isolated)
        fused                       = self.fuse(aligned, diagnostics, raw_frames=isolated, poses=poses)

        if is_body_mode:
            # Human full-body reconstruction: print a clean summary
            n_accepted = sum(1 for d in (diagnostics or []) if d is not None and d[2])
            n_total    = len(diagnostics) if diagnostics else 0
            logger.info("\n" + "=" * 60)
            logger.info("  HUMAN BODY RECONSTRUCTION COMPLETE")
            logger.info("  ============================")
            logger.info(f"  Fusion mode    : {self.cfg.fusion_method.upper()}")
            logger.info(f"  Frames total   : {n_total}")
            logger.info(f"  Frames accepted: {n_accepted}")
            logger.info(f"  Frames rejected: {n_total - n_accepted}")
            logger.info(f"  Final points   : {len(fused.points):,}")
            logger.info("=" * 60)

            out = self.cfg.output_dir / "human_reconstruction.ply"
            import open3d as o3d
            o3d.io.write_point_cloud(str(out), fused)
            logger.info(f"  Saved: {out}")

            result = {
                "target": self.cfg.target.name,
                "mode": "human_reconstruction",
                "fusion_method": self.cfg.fusion_method,
                "frames_total": n_total,
                "frames_accepted": n_accepted,
                "frames_rejected": n_total - n_accepted,
                "final_points": len(fused.points),
                "points_in_stage4": len(fused.points),
                "output": str(out),
            }
        else:
            is_sufficient, reasons = self.check_measurement_sufficiency(diagnostics, fused, aligned)
            if not is_sufficient:
                n_accepted = sum(1 for d in (diagnostics or []) if d is not None and d[2])
                n_total    = len(diagnostics) if diagnostics else 0
                logger.info("\n" + "!" * 60)
                logger.error("  [!] SCAN INSUFFICIENT -- MEASUREMENT REFUSED")
                logger.info("  " + "=" * 56)
                logger.info("  The scan did not meet the minimum data sufficiency criteria:")
                for r in reasons:
                    logger.info(f"    - {r}")
                logger.info("!" * 60)

                result = {
                    "target": self.cfg.target.name,
                    "status": "insufficient_data",
                    "reasons": reasons,
                    "frames_accepted": n_accepted,
                    "frames_total": n_total,
                    "fused_points": len(fused.points),
                    "points_in_stage4": len(fused.points),
                }
            else:
                result = self.measure(fused)
                result["points_in_stage4"] = len(fused.points)

        logger.info("\n" + "=" * 60)
        logger.info(f"  Done. Outputs written to: {self.cfg.output_dir.resolve()}")
        logger.info("=" * 60 + "\n")

        return result
