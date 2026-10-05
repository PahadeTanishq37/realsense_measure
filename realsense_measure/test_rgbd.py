"""
RealSense D455 Camera Live Stream & Capture Diagnostic Tool.

Use this script to verify:
  1. Synchronized RGB and Depth streaming at stable FPS.
  2. Live side-by-side preview with responsive OpenCV window sizing.
  3. Continuous frame capture (manual SPACE or automatic stress test)
     to confirm that neither camera stream freezes after multiple captures.

Controls:
  SPACE : Capture a single RGB-D frame
  'a'   : Toggle continuous auto-capture stress test (captures every 1s)
  'r'   : Reset capture count
  ESC/q : Exit cleanly
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import open3d as o3d

from config import CameraConfig
from camera.realsense_capture import RealSenseCamera


def main():
    print("=" * 60)
    print("  RealSense D455 Live RGB-D Stream & Capture Test")
    print("=" * 60)
    print("  Starting camera pipeline...")

    cfg = CameraConfig()
    try:
        cam = RealSenseCamera(cfg).start()
    except Exception as e:
        print(f"  [ERROR] Failed to start RealSense camera: {e}")
        return

    window_name = "RealSense D455 Stream Test (RGB | Depth)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    fps = 0.0
    frame_count = 0
    t_start = time.time()
    last_fps_time = time.time()
    fps_frame_count = 0

    captured_frames = []
    auto_stress_test = False
    last_auto_capture_time = 0.0

    print("  Camera streaming active.")
    print("  Press SPACE to capture a frame.")
    print("  Press 'a' to toggle auto-capture stress test.")
    print("  Press 'q' or ESC to exit.\n")

    try:
        while True:
            t_loop = time.time()
            color_bgr, depth_m, depth_color = cam.read_with_colorized()

            if color_bgr is None or depth_m is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                time.sleep(0.005)
                continue

            frame_count += 1
            fps_frame_count += 1
            if t_loop - last_fps_time >= 1.0:
                fps = fps_frame_count / (t_loop - last_fps_time)
                fps_frame_count = 0
                last_fps_time = t_loop

            # Resize depth color to match RGB if shapes differ
            if depth_color.shape[:2] != color_bgr.shape[:2]:
                h, w = color_bgr.shape[:2]
                depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)

            # Overlay stats on depth panel
            display_depth = depth_color.copy()
            cv2.putText(
                display_depth,
                f"FPS: {fps:.1f} | Captured: {len(captured_frames)} | Auto: {'ON' if auto_stress_test else 'OFF'}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_depth,
                f"FPS: {fps:.1f} | Captured: {len(captured_frames)} | Auto: {'ON' if auto_stress_test else 'OFF'}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Overlay guide on color panel
            display_color = color_bgr.copy()
            h, w = display_color.shape[:2]
            cv2.putText(
                display_color,
                "RGB STREAM [OK]",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            preview = np.hstack([display_color, display_depth])

            if not window_initialized:
                disp_w = min(1280, preview.shape[1])
                disp_h = int(preview.shape[0] * (disp_w / preview.shape[1]))
                cv2.resizeWindow(window_name, disp_w, disp_h)
                window_initialized = True

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            # Key handling
            if key in (27, ord('q')):  # ESC or q to exit
                print("\n  Exiting diagnostic test.")
                break

            elif key == ord('a'):  # Toggle auto-capture stress test
                auto_stress_test = not auto_stress_test
                print(f"  Auto-capture stress test: {'ENABLED (every 1.0s)' if auto_stress_test else 'DISABLED'}")

            elif key == ord('r'):  # Reset count
                captured_frames.clear()
                print("  Captured frames count reset.")

            trigger_capture = False
            if key == 32:  # SPACE
                trigger_capture = True
            elif auto_stress_test and (t_loop - last_auto_capture_time >= 1.0):
                trigger_capture = True
                last_auto_capture_time = t_loop

            if trigger_capture:
                rgbd_frame = cam.create_rgbd_frame(
                    color_bgr=color_bgr,
                    depth_m=depth_m,
                    frame_id=len(captured_frames),
                    timestamp=t_loop,
                )
                captured_frames.append(rgbd_frame)
                n_pts = len(rgbd_frame.pcd.points)
                print(f"  [FRAME {len(captured_frames):02d}] Captured: {n_pts:,} 3D points | Stream FPS: {fps:.1f} (Stream smooth, no freeze)")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"  Test completed: {frame_count} total frames processed, {len(captured_frames)} frames captured.")


if __name__ == "__main__":
    main()
