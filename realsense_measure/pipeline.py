"""
Orchestrates the full scan pipeline: capture → isolate → register → fuse → measure.

Each stage prints progress to stdout and (when cfg.show_stage_windows is True)
opens a blocking Open3D visualisation window so you can inspect the intermediate
result before the pipeline advances to the next step.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from camera.realsense_capture import RealSenseCamera
from config import PipelineConfig
from preprocessing import isolate_object
from reconstruction import fuse_point_clouds
from registration import register_sequence
from targets.base import get_target
import targets.box  # noqa: F401 — registers BoxTarget via @register_target
from visualizer import color_frames_distinctly, obb_lineset, show


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
          SPACE  — capture current frame
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
        print("  SPACE = capture frame | ENTER = finish | ESC = abort")
        print("=" * 60)

        try:
            while True:
                color_bgr, depth_m = cam.read()
                if color_bgr is None or depth_m is None:
                    continue  # dropped frame — just retry

                # ---- depth colourmap for side-by-side preview ----------
                depth_abs = cv2.convertScaleAbs(
                    depth_m,
                    alpha=255.0 / self.cfg.camera.depth_max_m,
                )
                depth_color = cv2.applyColorMap(depth_abs, cv2.COLORMAP_JET)

                # Make both the same height before stacking (decimation might
                # have changed the depth resolution before the read() resize).
                if depth_color.shape[:2] != color_bgr.shape[:2]:
                    h, w = color_bgr.shape[:2]
                    depth_color = cv2.resize(depth_color, (w, h))

                preview = np.hstack([color_bgr, depth_color])

                # ---- overlay HUD text ----------------------------------
                n = len(raw_frames)
                lines = [
                    f"Frames captured: {n}",
                    "SPACE: capture | ENTER: finish (need 2+) | ESC: abort",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(
                        preview, line,
                        (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA,   # dark outline
                    )
                    cv2.putText(
                        preview, line,
                        (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA,  # white text
                    )

                cv2.imshow("Stage 1: Capture (colour | depth)", preview)
                key = cv2.waitKey(1) & 0xFF

                # ---- key handling --------------------------------------
                if key == 32:  # SPACE — capture frame
                    pcd = cam.to_point_cloud(color_bgr, depth_m)
                    raw_frames.append(pcd)
                    print(f"  Captured frame {len(raw_frames):2d}: "
                          f"{len(pcd.points):>7} raw points")

                elif key == 13:  # ENTER — finish
                    if len(raw_frames) < 2:
                        print("  ⚠  Need at least 2 frames before finishing. "
                              "Keep capturing.")
                    else:
                        print(f"  Capture complete — {len(raw_frames)} frames.")
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
        print("  Stage 2: Isolating object per frame …")
        print("=" * 60)

        pre_cfg = self.cfg.preprocess
        isolated: list[o3d.geometry.PointCloud] = []

        for i, frame in enumerate(raw_frames):
            n_before = len(frame.points)
            iso = isolate_object(frame, pre_cfg)
            n_after = len(iso.points)
            print(f"  frame {i:02d}: {n_before:>7} pts → {n_after:>6} pts")
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
    ) -> list[o3d.geometry.PointCloud]:
        """
        Align all isolated frames into the shared reference coordinate system.

        Uses the growing-reference strategy from :func:`~registration.register_sequence`.
        Prints per-frame fitness and RMSE; flags weak alignments.

        Parameters
        ----------
        isolated_frames:
            Cleaned per-frame clouds from :meth:`isolate_frames`.

        Returns
        -------
        list[o3d.geometry.PointCloud]
            All frames expressed in frame-0 coordinates.
        """
        print("\n" + "=" * 60)
        print("  Stage 3: Registering frames (FPFH+RANSAC → ICP) …")
        print("=" * 60)

        aligned, diagnostics = register_sequence(isolated_frames, self.cfg.registration)

        for i, (fitness, rmse) in enumerate(diagnostics):
            note = ""
            if i > 0 and fitness <= 0.3:
                note = "  ⚠  weak alignment — consider recapturing this angle"
            print(f"  frame {i:02d}: fitness={fitness:.4f}  "
                  f"rmse={rmse * 1000:.3f} mm{note}")

        if self.cfg.show_stage_windows:
            show(
                color_frames_distinctly(aligned),
                window_name="Stage 3: registered/aligned clouds",
            )

        return aligned

    # ------------------------------------------------------------------
    # Stage 4: fuse
    # ------------------------------------------------------------------

    def fuse(
        self,
        aligned_frames: list[o3d.geometry.PointCloud],
    ) -> o3d.geometry.PointCloud:
        """
        Merge all registered frames into one clean point cloud.

        Parameters
        ----------
        aligned_frames:
            Registered per-frame clouds from :meth:`register`.

        Returns
        -------
        o3d.geometry.PointCloud
            Fused, denoised reconstruction.
        """
        print("\n" + "=" * 60)
        print("  Stage 4: Fusing aligned frames …")
        print("=" * 60)

        fused = fuse_point_clouds(aligned_frames, self.cfg.preprocess)
        print(f"  Fused cloud: {len(fused.points):>7} points")

        if self.cfg.show_stage_windows:
            display = fused.__copy__() if hasattr(fused, "__copy__") else fused
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
        print("  Stage 5: Segmenting and measuring …")
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
        print(f"  Volume : {obb_info.get('volume_m3', 0) * 1e6:.1f} cm³")
        aabb_ext = result.get("axis_aligned_bbox_extent_m", [])
        if aabb_ext:
            ext_mm = [f"{v * 1000:.1f}" for v in aabb_ext]
            print(f"  AABB   : {' × '.join(ext_mm)} mm (axis-aligned reference)")

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
        raw_frames = self.capture_frames()
        isolated   = self.isolate_frames(raw_frames)
        aligned    = self.register(isolated)
        fused      = self.fuse(aligned)
        result     = self.measure(fused)

        print("\n" + "=" * 60)
        print(f"  Done. Outputs written to: {self.cfg.output_dir.resolve()}")
        print("=" * 60 + "\n")

        return result
