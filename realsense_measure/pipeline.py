"""
Orchestrates the full scan pipeline: capture → isolate → register → fuse → measure.

Each stage prints progress to stdout and (when cfg.show_stage_windows is True)
opens a blocking Open3D visualisation window so you can inspect the intermediate
result before the pipeline advances to the next step.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d

from camera.realsense_capture import RealSenseCamera
from config import PipelineConfig
from preprocessing import isolate_object
from reconstruction import fuse_point_clouds
from registration import register_frame_pair, register_sequence
from targets.base import get_target
import targets.box  # noqa: F401 — registers BoxTarget via @register_target
from visualizer import color_frames_distinctly, obb_lineset, show


def _draw_hud(
    panel: np.ndarray,
    depth_m: np.ndarray,
    n_frames: int,
    n_pts: int,
    vis_min_m: float,
    vis_max_m: float,
    auto_capture: bool = True,
) -> None:
    """
    Render the D455f status HUD onto the depth panel in-place.

    Parameters
    ----------
    panel:
        The BGR image that will appear on the RIGHT side of the preview
        window (the SDK-colorized depth image).  Modified in-place.
    depth_m:
        The RAW float32 depth array in metres — used only to compute
        statistics (valid pixel count, median, min, max).  Never displayed
        as depth values.
    n_frames:
        Number of frames captured so far.
    n_pts:
        Number of points in the most recently captured point cloud
        (0 if no frame captured yet).
    vis_min_m, vis_max_m:
        Configured visualization range (from DepthVisConfig).
    auto_capture:
        Whether continuous automatic capture mode is active.
    """
    valid = depth_m[depth_m > 0]
    n_valid   = int(valid.size)
    d_median  = float(np.median(valid)) if n_valid else 0.0
    d_min     = float(valid.min())      if n_valid else 0.0
    d_max     = float(valid.max())      if n_valid else 0.0

    cloud_status = f"{n_pts:,} pts" if n_pts > 0 else "--"

    if auto_capture:
        legend = "AUTO-CAPTURING -- move camera around object -- press ENTER when done"
    else:
        legend = "SPACE: capture  ENTER: finish (2+)  ESC: abort"

    lines = [
        "D455f STATUS",
        "",
        f"Frames captured : {n_frames}",
        f"Valid depth px  : {n_valid:,}",
        f"Point cloud pts : {cloud_status}",
        "",
        f"Depth median    : {d_median:.3f} m",
        f"Depth min       : {d_min:.3f} m",
        f"Depth max       : {d_max:.3f} m",
        "",
        f"Vis range  min  : {vis_min_m:.2f} m",
        f"Vis range  max  : {vis_max_m:.2f} m",
        "",
        legend,
    ]

    y0, dy = 28, 22
    for i, line in enumerate(lines):
        y = y0 + i * dy
        # Shadow (readability over any background color)
        cv2.putText(panel, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        # White text
        color = (0, 255, 255) if line == "D455f STATUS" else (255, 255, 255)
        cv2.putText(panel, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 1, cv2.LINE_AA)


def _depth_mean_diff(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute mean absolute difference between valid depth pixels in two frames (m)."""
    mask = (d1 > 0) & (d2 > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(d1[mask] - d2[mask])))


class ScanPipeline:
    """
    End-to-end orchestration of a multi-view box measurement scan.

    Usage::

        cfg = PipelineConfig()
        result = ScanPipeline(cfg).run()
        print(result)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.target = get_target(cfg.target.name)

    # ------------------------------------------------------------------
    # Stage 1: capture
    # ------------------------------------------------------------------

    def capture_frames(self) -> list[o3d.geometry.PointCloud]:
        """
        Live camera capture loop — returns a list of raw per-frame PointClouds.

        Controls:
          AUTO-CAPTURE: continuously captures frames at cfg.camera.capture_interval_s
          SPACE  — capture current frame (when manual capture is enabled)
          ENTER  — finish capture (requires ≥ 2 frames)
          ESC    — abort (raises KeyboardInterrupt)

        Returns
        -------
        list[o3d.geometry.PointCloud]
            Raw point clouds, one per captured frame, in capture order.
        """
        cam = RealSenseCamera(self.cfg.camera).start()
        raw_frames: list[o3d.geometry.PointCloud] = []

        print("\n" + "=" * 60)
        print("  Stage 1: Capture")
        if self.cfg.camera.auto_capture:
            print(f"  AUTO-CAPTURE: move camera around object (interval: "
                  f"{self.cfg.camera.capture_interval_s:.2f}s)")
            print("  ENTER = finish (requires 2+ frames) | ESC = abort")
        else:
            print("  MANUAL CAPTURE: SPACE = capture frame | ENTER = finish | ESC = abort")
        print("=" * 60)

        start_capture_time = time.time()
        last_capture_time = 0.0  # triggers capture on the first valid frame
        last_captured_depth: np.ndarray | None = None
        last_skip_warn_time = 0.0
        last_n_pts = 0  # point count of the most recently captured cloud

        try:
            while True:
                color_bgr, depth_m, depth_color = cam.read_with_colorized()
                if color_bgr is None or depth_m is None:
                    continue  # dropped frame — just retry

                # depth_color is the SDK-colorized BGR image (VISUALIZATION ONLY).
                # depth_m is the raw float32 metres array (used for point cloud).

                # Ensure both panels are the same size before hstack.
                if depth_color.shape[:2] != color_bgr.shape[:2]:
                    h, w = color_bgr.shape[:2]
                    depth_color = cv2.resize(depth_color, (w, h),
                                             interpolation=cv2.INTER_NEAREST)

                # Draw the rich HUD onto the depth panel (in-place, copy first
                # so we don't clobber the original colorized image).
                depth_panel = depth_color.copy()
                _draw_hud(
                    depth_panel,
                    depth_m,
                    n_frames=len(raw_frames),
                    n_pts=last_n_pts,
                    vis_min_m=self.cfg.camera.depth_vis.visual_min_m,
                    vis_max_m=self.cfg.camera.depth_vis.visual_max_m,
                    auto_capture=self.cfg.camera.auto_capture,
                )

                preview = np.hstack([color_bgr, depth_panel])
                cv2.imshow("Stage 1: D455f Capture (RGB | Depth)", preview)
                key = cv2.waitKey(1) & 0xFF

                now = time.time()

                # ---- Auto-capture logic --------------------------------
                if self.cfg.camera.auto_capture:
                    if now - last_capture_time >= self.cfg.camera.capture_interval_s:
                        is_duplicate = False
                        if (self.cfg.camera.skip_near_duplicate_frames
                                and last_captured_depth is not None):
                            diff_m = _depth_mean_diff(depth_m, last_captured_depth)
                            if diff_m < self.cfg.camera.duplicate_depth_diff_threshold_m:
                                is_duplicate = True

                        if is_duplicate:
                            if now - last_skip_warn_time >= 1.0:
                                print("  ... camera hasn't moved enough, still waiting ...")
                                last_skip_warn_time = now
                        else:
                            pcd = cam.to_point_cloud(color_bgr, depth_m)
                            raw_frames.append(pcd)
                            last_n_pts = len(pcd.points)
                            last_capture_time = now
                            last_captured_depth = depth_m.copy()
                            print(f"  Captured frame {len(raw_frames):2d}: "
                                  f"{last_n_pts:>7} raw points")

                # ---- Manual capture handling ---------------------------
                elif key == 32:  # SPACE — manual capture
                    pcd = cam.to_point_cloud(color_bgr, depth_m)
                    raw_frames.append(pcd)
                    last_n_pts = len(pcd.points)
                    last_captured_depth = depth_m.copy()
                    print(f"  Captured frame {len(raw_frames):2d}: "
                          f"{last_n_pts:>7} raw points")

                # ---- Finish / Abort keys -------------------------------
                if key == 13:  # ENTER — finish
                    if len(raw_frames) < 2:
                        print("  ⚠  Need at least 2 frames before finishing. "
                              "Keep capturing.")
                    else:
                        elapsed = time.time() - start_capture_time
                        n_frames = len(raw_frames)
                        avg_interval = elapsed / n_frames if n_frames > 0 else 0.0
                        print(f"\n  Capture complete — {n_frames} frames captured in "
                              f"{elapsed:.1f}s (effective avg interval: {avg_interval:.2f}s/frame).")
                        break

                elif key == 27:  # ESC — abort
                    print("  Capture aborted by user.")
                    raise KeyboardInterrupt

        finally:
            cam.stop()
            cv2.destroyAllWindows()

        return raw_frames

    # ------------------------------------------------------------------
    # Stage 2: isolate
    # ------------------------------------------------------------------

    def isolate_frames(
        self,
        raw_frames: list[o3d.geometry.PointCloud],
    ) -> list[o3d.geometry.PointCloud]:
        """
        Run per-frame isolation (downsample → plane removal → cluster).

        Parameters
        ----------
        raw_frames:
            List of raw PointClouds from :meth:`capture_frames`.

        Returns
        -------
        list[o3d.geometry.PointCloud]
            Isolated, cleaned per-frame object clouds.
        """
        print("\n" + "=" * 60)
        print("  Stage 2: Isolating object per frame ...")
        print("=" * 60)

        pre_cfg = self.cfg.preprocess
        isolated: list[o3d.geometry.PointCloud] = []

        for i, frame in enumerate(raw_frames):
            n_before = len(frame.points)
            iso = isolate_object(frame, pre_cfg)
            n_after = len(iso.points)
            print(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts")
            isolated.append(iso)

        if self.cfg.show_stage_windows:
            show(
                color_frames_distinctly(isolated),
                window_name="Stage 2: isolated per-frame clouds",
            )

        if self.cfg.save_intermediate:
            for i, pcd in enumerate(isolated):
                out = self.cfg.output_dir / f"frame_{i:02d}_isolated.ply"
                o3d.io.write_point_cloud(str(out), pcd)
                print(f"  Saved: {out}")

        return isolated

    # ------------------------------------------------------------------
    # Stage 3: register
    # ------------------------------------------------------------------

    def register(
        self,
        isolated_frames: list[o3d.geometry.PointCloud],
    ) -> tuple[list[o3d.geometry.PointCloud], list[tuple[float, float, bool]]]:
        """
        Align all isolated frames into the shared reference coordinate system.

        Uses multi-candidate registration with acceptance gating against
        cfg.registration.min_accept_fitness to protect against reference corruption.

        Parameters
        ----------
        isolated_frames:
            Cleaned per-frame clouds from :meth:`isolate_frames`.

        Returns
        -------
        aligned:
            List of all aligned point clouds in frame-0 coordinates.
        diagnostics:
            List of (fitness, rmse, accepted) tuples per frame.
        """
        print("\n" + "=" * 60)
        print("  Stage 3: Registering frames (FPFH+RANSAC -> ICP) ...")
        print("=" * 60)

        aligned, diagnostics = register_sequence(isolated_frames, self.cfg.registration)

        rejected_indices: list[int] = []
        for i, (fitness, rmse, accepted) in enumerate(diagnostics):
            status = "ACCEPTED" if accepted else "REJECTED -- excluded from fusion, consider recapturing this angle"
            if not accepted:
                rejected_indices.append(i)
            print(f"  frame {i:02d}: fitness={fitness:.4f}  "
                  f"rmse={rmse * 1000:.3f} mm  {status}")

        n_accepted = sum(1 for *_, acc in diagnostics if acc)
        n_total = len(diagnostics)
        print(f"\n  Registration summary: {n_accepted} of {n_total} frames accepted into the reconstruction.")
        if n_total > 0 and (n_accepted / n_total) < 0.70:
            print("  Tip: low acceptance ratio (< 70%). Try moving the camera more slowly")
            print("       or reducing --interval for higher frame overlap.")

        # Optional diagnostic loop-closure check (gated behind cfg.attempt_loop_closure)
        if self.cfg.attempt_loop_closure:
            accepted_indices = [idx for idx, (*_, acc) in enumerate(diagnostics) if acc]
            if len(accepted_indices) >= 8:
                last_idx = accepted_indices[-1]
                try:
                    # Register the last accepted raw frame directly against the initial frame (frame 0)
                    _, loop_fitness, loop_rmse = register_frame_pair(
                        isolated_frames[last_idx], isolated_frames[0], self.cfg.registration
                    )
                    if loop_fitness >= self.cfg.registration.min_accept_fitness:
                        print(f"\n  Loop closure check: fitness={loop_fitness:.4f} (rmse={loop_rmse * 1000:.2f} mm) -- looks like a consistent full orbit.")
                    else:
                        print(f"\n  [!] Loop closure check: low overlap (fitness={loop_fitness:.4f}) with initial frame.")
                        print("      Drift may have accumulated over the full orbit.")
                        print("      Tip: orbit more slowly/steadily or reduce capture interval.")
                        # Note: implementing full pose-graph optimization across all frame pairs
                        # (e.g. o3d.pipelines.registration.global_optimization) would be the next step
                        # if drift is persistent, but is out of scope for this pass.
                except Exception as exc:
                    print(f"\n  [!] Loop closure check skipped: {exc}")
            else:
                print("\n  [!] Loop closure check skipped: requires at least 8 accepted frames.")

        if self.cfg.show_stage_windows:
            if rejected_indices:
                print(f"\n  [!] Note: Visualizing all frames including rejected frame(s) {rejected_indices}.")
                print("      They may appear misaligned or disconnected from the coherent reconstruction.")
            show(
                color_frames_distinctly(aligned),
                window_name="Stage 3: registered/aligned clouds",
            )

        return aligned, diagnostics

    # ------------------------------------------------------------------
    # Stage 4: fuse
    # ------------------------------------------------------------------

    def fuse(
        self,
        aligned_frames: list[o3d.geometry.PointCloud],
        diagnostics: list[tuple[float, float, bool]] | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Merge registered frames into one clean point cloud.

        Only frames with accepted=True in diagnostics are included in the final fusion.

        Parameters
        ----------
        aligned_frames:
            Registered per-frame clouds from :meth:`register`.
        diagnostics:
            Optional registration diagnostics tuples (fitness, rmse, accepted).
            If provided, rejected frames are excluded from fusion.

        Returns
        -------
        o3d.geometry.PointCloud
            Fused, denoised reconstruction.
        """
        print("\n" + "=" * 60)
        print("  Stage 4: Fusing aligned frames ...")
        print("=" * 60)

        if diagnostics is not None:
            accepted_frames = [f for f, (*_, acc) in zip(aligned_frames, diagnostics) if acc]
            if not accepted_frames:
                print("  [!] Warning: No frames met acceptance criteria; falling back to frame 0.")
                accepted_frames = aligned_frames[:1]
        else:
            accepted_frames = aligned_frames

        n_skipped = len(aligned_frames) - len(accepted_frames)
        skip_msg = f" (excluded {n_skipped} rejected frame{'s' if n_skipped != 1 else ''})" if n_skipped > 0 else ""
        print(f"  Fusing {len(accepted_frames)} accepted frame(s){skip_msg} ...")

        fused = fuse_point_clouds(accepted_frames, self.cfg.preprocess)
        print(f"  Fused cloud: {len(fused.points):>7} points")

        if self.cfg.show_stage_windows:
            display_pcd = o3d.geometry.PointCloud(fused)
            display_pcd.paint_uniform_color([0.65, 0.65, 0.65])  # flat grey
            show(display_pcd, window_name="Stage 4: fused reconstruction")

        if self.cfg.save_intermediate:
            out = self.cfg.output_dir / "fused.ply"
            o3d.io.write_point_cloud(str(out), fused)
            print(f"  Saved: {out}")

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
        print("\n" + "=" * 60)
        print("  Stage 5: Segmenting and measuring ...")
        print("=" * 60)

        segmented = self.target.segment(fused_pcd)
        print(f"  Segmented cloud: {len(segmented.points):>7} points")

        result = self.target.measure(segmented)

        # Pop the Open3D geometry before printing / saving — it is not
        # JSON-serialisable and must be kept separate for visualisation.
        obb = result.pop("geometry_for_viz", None)

        obb_info = result.get("oriented_bbox", {})
        print(f"\n  Target : {result.get('target', '?')}")
        print(f"  Length : {obb_info.get('length_m', 0) * 1000:.1f} mm")
        print(f"  Width  : {obb_info.get('width_m',  0) * 1000:.1f} mm")
        print(f"  Height : {obb_info.get('height_m', 0) * 1000:.1f} mm")
        print(f"  Volume : {obb_info.get('volume_m3', 0) * 1e6:.1f} cm^3")
        aabb_ext = result.get("axis_aligned_bbox_extent_m", [])
        if aabb_ext:
            ext_mm = [f"{v * 1000:.1f}" for v in aabb_ext]
            print(f"  AABB   : {' x '.join(ext_mm)} mm (axis-aligned reference)")

        if self.cfg.show_stage_windows and obb is not None:
            seg_display = o3d.geometry.PointCloud(segmented)
            seg_display.paint_uniform_color([0.18, 0.72, 0.38])  # green
            show(
                [seg_display, obb_lineset(obb)],
                window_name="Stage 5: measured object",
            )

        # ---- persist results -------------------------------------------
        meas_path = self.cfg.output_dir / "measurement.json"
        with meas_path.open("w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        print(f"\n  Saved measurement: {meas_path}")

        seg_path = self.cfg.output_dir / "segmented_object.ply"
        o3d.io.write_point_cloud(str(seg_path), segmented)
        print(f"  Saved segmented cloud: {seg_path}")

        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute all pipeline stages in sequence and return the measurement.

        Stages
        ------
        1. capture_frames  — interactive live-camera loop
        2. isolate_frames  — per-frame background removal
        3. register        — global + ICP multi-view alignment
        4. fuse            — merge into one reconstruction
        5. measure         — segment + dimensional measurement

        Returns
        -------
        dict
            JSON-serialisable measurement result from :meth:`measure`.
        """
        raw_frames           = self.capture_frames()
        isolated             = self.isolate_frames(raw_frames)
        aligned, diagnostics = self.register(isolated)
        fused                = self.fuse(aligned, diagnostics)
        result               = self.measure(fused)

        print("\n" + "=" * 60)
        print(f"  Done. Outputs written to: {self.cfg.output_dir.resolve()}")
        print("=" * 60 + "\n")

        return result
