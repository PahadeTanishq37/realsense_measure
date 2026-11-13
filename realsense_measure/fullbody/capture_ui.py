"""
realsense_measure/fullbody/capture_ui.py

Real-time operator preview and HUD rendering for Stage 27 CaptureController.
Strictly imports cv2, numpy, and standard library only.
Headless-testable rendering functions without GUI or blocking dependencies.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def colorize_depth_m(
    depth_m: np.ndarray,
    min_m: float = 0.30,
    max_m: float = 3.00,
) -> np.ndarray:
    """Convert float32 metric depth array to BGR JET colormap visualization."""
    valid_mask = (depth_m > 0.05) & (depth_m <= max_m)
    depth_clamped = np.clip(depth_m, min_m, max_m)
    norm = ((depth_clamped - min_m) / max(1e-3, (max_m - min_m)) * 255.0).astype(np.uint8)
    # Invert so near is warm (red) and far is cool (blue)
    norm_inv = 255 - norm
    colorized = cv2.applyColorMap(norm_inv, cv2.COLORMAP_JET)
    colorized[~valid_mask] = [0, 0, 0]
    return colorized


def render_frame(
    state_snapshot: Dict[str, Any],
    color: np.ndarray,
    depth_filtered: Optional[np.ndarray],
) -> np.ndarray:
    """
    Pure headless rendering function creating operator HUD.
    Returns composite BGR image of shape (H, W*2, 3).
    """
    h, w = color.shape[:2]

    # Generate colorized depth
    if depth_filtered is not None:
        if depth_filtered.shape[:2] != (h, w):
            depth_resized = cv2.resize(depth_filtered, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_resized = depth_filtered
        depth_vis = colorize_depth_m(depth_resized)
    else:
        depth_vis = np.zeros_like(color)

    # Side-by-side composite
    composite = np.hstack([color.copy(), depth_vis])
    comp_w = composite.shape[1]

    # Extract state snapshot data
    state = str(state_snapshot.get("state", "WAIT_FOR_SUBJECT"))
    stop_idx = int(state_snapshot.get("stop_index", 0))
    n_stops = int(state_snapshot.get("n_stops", 8))
    turn_dir = str(state_snapshot.get("turn_direction", "cw")).lower()
    retake_idx = int(state_snapshot.get("current_retake", 0))

    dist_m = state_snapshot.get("distance_m")
    valid_pct = state_snapshot.get("valid_depth_pct")
    motion_mm = state_snapshot.get("motion_mm")
    head_ok = bool(state_snapshot.get("head_visible", True))
    feet_ok = bool(state_snapshot.get("feet_visible", True))
    burst_prog = float(np.clip(state_snapshot.get("burst_progress", 0.0), 0.0, 1.0))
    rejection_reason = state_snapshot.get("last_rejection_reason")

    # 1. Body-ROI guide rectangle on RGB feed (left side)
    guide_x0, guide_y0 = int(0.20 * w), int(0.05 * h)
    guide_x1, guide_y1 = int(0.80 * w), int(0.95 * h)
    cv2.rectangle(composite, (guide_x0, guide_y0), (guide_x1, guide_y1), (0, 255, 255), 1)

    # 2. Header banner with semi-transparent overlay
    banner_h = 80
    overlay = composite.copy()
    cv2.rectangle(overlay, (0, 0), (comp_w, banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, composite, 0.25, 0, composite)

    # State colors
    state_colors = {
        "WAIT_FOR_SUBJECT": (0, 200, 255),  # Yellow
        "POSITION_CHECK": (0, 200, 255),
        "TURN_PROMPT": (255, 200, 0),     # Cyan
        "WAIT_STILL": (255, 150, 0),
        "CAPTURE_BURST": (0, 255, 255),   # Yellow
        "EVALUATE": (200, 200, 200),
        "ACCEPT": (0, 255, 0),            # Green
        "DONE": (0, 255, 0),
        "RETAKE": (0, 0, 255),            # Red
        "ABORT": (0, 0, 255),
    }
    state_color = state_colors.get(state, (255, 255, 255))

    # Large state text
    cv2.putText(composite, f"STATE: {state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

    # Stop counter
    stop_text = f"Stop {stop_idx + 1} of {n_stops}"
    if retake_idx > 0:
        stop_text += f" (Retake {retake_idx})"
    cv2.putText(composite, stop_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Turning instruction text (computed from turn_direction)
    turn_deg = int(360 // max(1, n_stops))
    if turn_dir == "cw":
        instr_text = f"Turn {turn_deg} degrees to your RIGHT"
    else:
        instr_text = f"Turn {turn_deg} degrees to your LEFT"

    (txt_w, txt_h), _ = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    instr_x = max(w + 5, comp_w - txt_w - 15)
    cv2.putText(composite, instr_text, (instr_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 3. Live metrics HUD bar
    hud_y = banner_h + 25
    x_off = 20

    # Distance metric [1.10 - 1.50]
    if dist_m is not None:
        d_color = (0, 255, 0) if (1.10 <= dist_m <= 1.50) else (0, 0, 255)
        d_str = f"Dist: {dist_m:.2f}m"
    else:
        d_color = (180, 180, 180)
        d_str = "Dist: --"
    cv2.putText(composite, d_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, d_color, 2)
    x_off += 150

    # Valid depth % (>= 30%)
    if valid_pct is not None:
        v_color = (0, 255, 0) if valid_pct >= 30.0 else (0, 0, 255)
        v_str = f"Depth: {valid_pct:.1f}%"
    else:
        v_color = (180, 180, 180)
        v_str = "Depth: --"
    cv2.putText(composite, v_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)
    x_off += 160

    # Motion mm (<= 10.0 mm)
    if motion_mm is not None:
        m_color = (0, 255, 0) if motion_mm <= 10.0 else (0, 0, 255)
        m_str = f"Motion: {motion_mm:.1f}mm"
    else:
        m_color = (180, 180, 180)
        m_str = "Motion: --"
    cv2.putText(composite, m_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 2)
    x_off += 180

    # Head and feet visibility
    hf_color = (0, 255, 0) if (head_ok and feet_ok) else (0, 0, 255)
    hf_str = f"Head: {'OK' if head_ok else 'CLIP'} | Feet: {'OK' if feet_ok else 'CLIP'}"
    cv2.putText(composite, hf_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hf_color, 2)

    # 4. Last rejection reason
    if rejection_reason:
        cv2.putText(
            composite,
            f"REJECTED: {rejection_reason}",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # 5. Burst progress bar at bottom
    bar_y0, bar_y1 = h - 40, h - 25
    bar_x0, bar_x1 = 20, comp_w - 20
    bar_total_w = bar_x1 - bar_x0
    # Background frame
    cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)
    if burst_prog > 0.0:
        fill_w = int(bar_total_w * burst_prog)
        cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1), (0, 255, 0), -1)
    cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x1, bar_y1), (180, 180, 180), 1)

    # 6. Keybindings prompt
    keys_text = "[SPACE] Confirm  |  [R] Retake  |  [S] Skip  |  [Q/ESC] Abort"
    cv2.putText(composite, keys_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return composite


def run_ui_loop(
    controller: Any,
    window_name: str = "Full-Body 8-Stop Turntable Capture",
    headless: bool = False,
    max_steps: Optional[int] = None,
) -> None:
    """
    Drive controller.step() with live UI preview and keyboard input handling.
    In headless mode, runs pure stepping without calling cv2.imshow or cv2.waitKey.
    """
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    step_count = 0
    try:
        while True:
            now = time.time()
            state = controller.step(now)
            step_count += 1

            if state in ("DONE", "ABORT"):
                break

            if max_steps is not None and step_count >= max_steps:
                break

            if not headless:
                # Build snapshot from controller
                snapshot = {
                    "state": controller.state.value if hasattr(controller.state, "value") else str(controller.state),
                    "stop_index": getattr(controller, "current_stop", 0),
                    "n_stops": getattr(controller, "n_stops", 8),
                    "turn_direction": getattr(controller.writer, "turn_direction", "cw"),
                    "current_retake": getattr(controller, "current_retake", 0),
                    "distance_m": getattr(controller, "_latest_distance_m", None),
                    "valid_depth_pct": getattr(controller, "_latest_valid_depth_pct", None),
                    "motion_mm": getattr(controller, "_latest_motion_mm", None),
                    "head_visible": getattr(controller, "_latest_head_visible", True),
                    "feet_visible": getattr(controller, "_latest_feet_visible", True),
                    "burst_progress": len(getattr(controller, "_burst_depths", [])) / max(1, controller.thresholds.burst_frames),
                    "last_rejection_reason": getattr(controller, "_latest_rejection_reason", None),
                }

                # Get latest frame for display if available
                color = getattr(controller, "_latest_color", None)
                depth = getattr(controller, "_latest_filtered", None)
                if color is None:
                    color = np.zeros((480, 270, 3), dtype=np.uint8)

                frame = render_frame(snapshot, color, depth)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    controller.send_command("confirm")
                elif key in (ord("r"), ord("R")):
                    controller.send_command("retake")
                elif key in (ord("s"), ord("S")):
                    controller.send_command("skip")
                elif key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    controller.send_command("abort")

            time.sleep(0.01)

    finally:
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

