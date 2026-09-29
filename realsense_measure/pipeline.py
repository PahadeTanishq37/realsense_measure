"""
Orchestrates the full scan pipeline: capture → isolate → register → fuse → measure.

Each stage prints progress to stdout and (when cfg.show_stage_windows is True)
opens a blocking Open3D visualisation window so you can inspect the intermediate
result before the pipeline advances to the next step.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame
from camera.realsense_capture import RealSenseCamera
from config import PipelineConfig
from preprocessing import isolate_object, reject_positional_outliers
from reconstruction import fuse_point_clouds, fuse_tsdf_volume
from registration import (
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
    show,
    visualize_registration_pair,
)


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
    mode_title:
        Title header displayed at the top of the HUD.
    is_human:
        Whether human 180° scan mode is active.
    """
    valid = depth_m[depth_m > 0]
    n_valid   = int(valid.size)
    d_median  = float(np.median(valid)) if n_valid else 0.0
    d_min     = float(valid.min())      if n_valid else 0.0
    d_max     = float(valid.max())      if n_valid else 0.0

    cloud_status = f"{n_pts:,} pts" if n_pts > 0 else "--"

    if is_human:
        legend = "HUMAN 180 SWEEP: move slowly, keep subject still -- ENTER: finish"
    elif auto_capture:
        legend = "AUTO-CAPTURING -- move camera around object -- press ENTER when done"
    else:
        legend = "SPACE: capture  ENTER: finish (2+)  ESC: abort"

    lines = [
        mode_title,
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
        # White / yellow text
        color = (0, 255, 255) if line == mode_title else (255, 255, 255)
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
    End-to-end orchestration of a multi-view scan (BOX or HUMAN HEAD/BODY).

    Usage::

        cfg = PipelineConfig()
        result = ScanPipeline(cfg).run()
        print(result)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        # If targeting human head or body, default to rgbd_human registration mode unless overridden
        if cfg.target.name in ("head", "body") and cfg.registration.registration_mode == "pointcloud":
            cfg.registration.registration_mode = "rgbd_human"

        self._clean_output_dir()
        self.target = get_target(cfg.target.name)

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
                    print(f"  [!] Warning: Failed to remove old output item {item.name}: {e}")
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: capture
    # ------------------------------------------------------------------

    def capture_frames(self) -> list[o3d.geometry.PointCloud | RGBDFrame]:
        """
        Live camera capture loop.

        Returns
        -------
        list[PointCloud | RGBDFrame]
            In pointcloud mode: raw PointClouds.
            In rgbd_human mode: synchronized RGBDFrame instances containing color,
            depth, point clouds with normals, timestamps, and intrinsics.
        """
        cam = RealSenseCamera(self.cfg.camera).start()
        is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human")

        # In human mode, use the configured human_capture_interval_s (default 0.4s) unless overridden
        capture_interval = self.cfg.camera.capture_interval_s
        if is_human_mode and self.cfg.camera.human_capture_interval_s is not None:
            capture_interval = self.cfg.camera.human_capture_interval_s

        raw_frames: list[o3d.geometry.PointCloud | RGBDFrame] = []

        print("\n" + "=" * 60)
        print("  Stage 1: Capture")
        if is_human_mode:
            print("  MODE: HUMAN 180° SCAN (RGB-D Odometry + Colored ICP)")
            print("  OPERATOR GUIDANCE:")
            print("    * Keep the subject still (subject rigidity is assumed).")
            print("    * Move camera slowly in a smooth ~180° arc around the front.")
            print("    * Maintain substantial overlap between consecutive views.")
            print("    * Keep the subject's face/head approximately centered.")
            print("    * Avoid sudden camera rotations or translations.")
            print("    * Complete the 180° sweep smoothly.")
            print(f"  AUTO-CAPTURE interval: {capture_interval:.2f}s")
            print("  ENTER = finish (requires 2+ frames) | ESC = abort")
        else:
            print("  MODE: BOX / OBJECT SCAN (Point Cloud)")
            if self.cfg.camera.auto_capture:
                print(f"  AUTO-CAPTURE: move camera around object (interval: "
                      f"{capture_interval:.2f}s)")
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

                # Draw the rich HUD onto the depth panel
                depth_panel = depth_color.copy()
                hud_title = "HUMAN 180° SCAN" if is_human_mode else "D455f STATUS"
                _draw_hud(
                    depth_panel,
                    depth_m,
                    n_frames=len(raw_frames),
                    n_pts=last_n_pts,
                    vis_min_m=self.cfg.camera.depth_vis.visual_min_m,
                    vis_max_m=self.cfg.camera.depth_vis.visual_max_m,
                    auto_capture=self.cfg.camera.auto_capture,
                    mode_title=hud_title,
                    is_human=is_human_mode,
                )

                preview = np.hstack([color_bgr, depth_panel])
                window_name = "Stage 1: Human 180° Capture" if is_human_mode else "Stage 1: D455f Capture (RGB | Depth)"
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                now = time.time()

                # ---- Auto-capture logic --------------------------------
                if self.cfg.camera.auto_capture:
                    if now - last_capture_time >= capture_interval:
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
                            if is_human_mode:
                                rgbd_frame, _ = cam.capture_rgbd_frame(frame_id=len(raw_frames))
                                if rgbd_frame is not None:
                                    n_pts = len(rgbd_frame.pcd.points)
                                    if n_pts < self.cfg.target.min_object_points:
                                        print(f"  Frame discarded: only {n_pts} pts in range -- check distance to subject (0.4m - 1.2m)")
                                    else:
                                        raw_frames.append(rgbd_frame)
                                        last_n_pts = n_pts
                                        last_capture_time = now
                                        last_captured_depth = depth_m.copy()
                                        print(f"  Captured human frame {len(raw_frames):2d}: {n_pts:>6} pts with RGB-D [OK]")
                            else:
                                pcd = cam.to_point_cloud(color_bgr, depth_m)
                                iso_test = isolate_object(pcd, self.cfg.preprocess, self.cfg.target)
                                n_obj_pts = len(iso_test.points)
                                if n_obj_pts < self.cfg.target.min_object_points:
                                    print(f"  Frame discarded: isolated to only {n_obj_pts} pts "
                                          f"(need >= {self.cfg.target.min_object_points}) -- "
                                          f"this angle didn't capture usable object data, keep moving")
                                else:
                                    raw_frames.append(pcd)
                                    last_n_pts = len(pcd.points)
                                    last_capture_time = now
                                    last_captured_depth = depth_m.copy()
                                    print(f"  Captured frame {len(raw_frames):2d}: "
                                          f"{last_n_pts:>7} raw pts -> {n_obj_pts:>5} object pts [OK]")

                # ---- Manual capture handling ---------------------------
                elif key == 32:  # SPACE — manual capture
                    if is_human_mode:
                        rgbd_frame, _ = cam.capture_rgbd_frame(frame_id=len(raw_frames))
                        if rgbd_frame is not None:
                            n_pts = len(rgbd_frame.pcd.points)
                            if n_pts < self.cfg.target.min_object_points:
                                print(f"  Frame discarded: only {n_pts} pts in range -- check distance to subject")
                            else:
                                raw_frames.append(rgbd_frame)
                                last_n_pts = n_pts
                                last_captured_depth = depth_m.copy()
                                print(f"  Captured human frame {len(raw_frames):2d}: {n_pts:>6} pts with RGB-D [OK]")
                    else:
                        pcd = cam.to_point_cloud(color_bgr, depth_m)
                        iso_test = isolate_object(pcd, self.cfg.preprocess, self.cfg.target)
                        n_obj_pts = len(iso_test.points)
                        if n_obj_pts < self.cfg.target.min_object_points:
                            print(f"  Frame discarded: isolated to only {n_obj_pts} pts "
                                  f"(need >= {self.cfg.target.min_object_points}) -- "
                                  f"this angle didn't capture usable object data, try adjusting angle")
                        else:
                            raw_frames.append(pcd)
                            last_n_pts = len(pcd.points)
                            last_captured_depth = depth_m.copy()
                            print(f"  Captured frame {len(raw_frames):2d}: "
                                  f"{last_n_pts:>7} raw pts -> {n_obj_pts:>5} object pts [OK]")

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
        print("\n" + "=" * 60)
        print("  Stage 2: Isolating object per frame ...")
        print("=" * 60)

        pre_cfg = self.cfg.preprocess
        target_cfg = self.cfg.target

        # Branch for human RGB-D frames: retain RGBDFrame representation
        if raw_frames and isinstance(raw_frames[0], RGBDFrame):
            isolated_rgbd: list[RGBDFrame] = []
            for i, f in enumerate(raw_frames):
                pcd = f.pcd
                n_before = len(pcd.points)
                # Denoise point cloud while preserving RGB-D arrays
                cl, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=pre_cfg.outlier_neighbors,
                    std_ratio=pre_cfg.outlier_std_ratio,
                )
                f.pcd = cl
                isolated_rgbd.append(f)
                print(f"  frame {i:02d}: {n_before:>7} pts -> {len(cl.points):>6} clean pts [ACCEPTED]")

            if self.cfg.show_stage_windows:
                show(
                    color_frames_distinctly([f.pcd for f in isolated_rgbd]),
                    window_name="Stage 2: preprocessed human frames",
                )
            return isolated_rgbd

        # Box / standard PointCloud path:
        isolated: list[o3d.geometry.PointCloud] = []
        for i, frame in enumerate(raw_frames):
            n_before = len(frame.points)
            iso = isolate_object(frame, pre_cfg, target_cfg)
            n_after = len(iso.points)
            if n_after < target_cfg.min_object_points:
                print(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts -- "
                      f"DISCARDED (< {target_cfg.min_object_points} pts, unusable object data)")
                continue
            print(f"  frame {i:02d}: {n_before:>7} pts -> {n_after:>6} pts [ACCEPTED]")
            isolated.append(iso)

        # Positional consistency: a background blob can be object-sized, so size
        # checks alone let it through.  Drop frames whose centroid jumps away.
        keep, rejected_pos = reject_positional_outliers(isolated, target_cfg.max_centroid_dev_m)
        for idx, dist in rejected_pos:
            print(f"  isolated #{idx:02d}: centroid {dist:.2f} m from median position -- "
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
                print(f"  Saved: {out}")

        return isolated

    # ------------------------------------------------------------------
    # Stage 3: register
    # ------------------------------------------------------------------

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
        print("\n" + "=" * 60)
        is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human")

        if is_human_mode:
            print("  Stage 3: Registering human frames (RGB-D Odometry + Colored ICP + Pose Graph) ...")
            print("=" * 60)
            aligned, diagnostics, poses, summary_stats = register_sequence_rgbd_human(
                isolated_frames, self.cfg.registration
            )

            # Output the required Section 21 diagnostic report
            print("\n" + "=" * 60)
            print("HUMAN RGB-D REGISTRATION")
            print("------------------------")
            print(f"Frames captured: {summary_stats.get('captured', len(isolated_frames))}")
            print(f"Frames accepted: {summary_stats.get('accepted', 0)}")
            print(f"Frames rejected: {summary_stats.get('rejected', 0)}")
            print("")
            print(f"Average pairwise RMSE: {summary_stats.get('avg_rmse_mm', 0.0):.2f} mm")
            print(f"Median pairwise RMSE:  {summary_stats.get('median_rmse_mm', 0.0):.2f} mm")
            print(f"Maximum translation jump: {summary_stats.get('max_trans_jump_mm', 0.0):.1f} mm")
            print(f"Maximum rotation jump:    {summary_stats.get('max_rot_jump_deg', 0.0):.1f} deg")
            print("")
            print("Pose graph:")
            print(f"    nodes: {summary_stats.get('n_nodes', 0)}")
            print(f"    edges: {summary_stats.get('n_edges', 0)}")
            print(f"    loop closures: {summary_stats.get('n_loop', 0)}")
            print("")
            first_rej = summary_stats.get("first_rejected_frame")
            if first_rej is not None:
                print(f"First rejected or suspicious frame: Frame {first_rej:02d}")
            else:
                print("First rejected or suspicious frame: None (all accepted)")
            print("=" * 60)

            # Pairwise inspection visualization
            if self.cfg.registration.show_registration_pairs and self.cfg.show_stage_windows:
                print("\n  Displaying pairwise registration inspection windows...")
                for idx in range(min(2, len(isolated_frames) - 1)):
                    if isinstance(isolated_frames[idx], RGBDFrame) and isinstance(isolated_frames[idx + 1], RGBDFrame):
                        # T maps source (idx) INTO target (idx+1):
                        #   pose_idx   = world_T_cam_idx
                        #   pose_idx+1 = world_T_cam_(idx+1)
                        #   T_src_to_tgt = inv(world_T_cam_(idx+1)) @ world_T_cam_idx
                        # BUT visualize_registration_pair moves the SOURCE cloud,
                        # so we need the transform that maps source PCD into target PCD space.
                        # Both PCDs are already in camera coords, so the relative pose is:
                        #   T = inv(pose_{idx+1}) @ pose_idx
                        T_src_to_tgt = np.linalg.inv(poses[idx + 1]) @ poses[idx]
                        visualize_registration_pair(
                            isolated_frames[idx].pcd,
                            isolated_frames[idx + 1].pcd,
                            T_src_to_tgt,
                            title=f"Pair {idx:02d} -> {idx+1:02d}",
                        )

            # Camera trajectory visualization
            if self.cfg.registration.show_camera_trajectory and self.cfg.show_stage_windows:
                print("\n  Displaying estimated camera trajectory...")
                traj_geoms = draw_camera_trajectory(poses, scale=0.04)
                if traj_geoms:
                    acc_clouds = [f for f, (*_, acc) in zip(aligned, diagnostics) if acc]
                    show(traj_geoms + acc_clouds, window_name="Estimated Camera Trajectory")

            if self.cfg.show_stage_windows:
                show(
                    color_frames_distinctly(aligned),
                    window_name="Stage 3: registered/aligned human clouds",
                )

            return aligned, diagnostics, poses

        # Standard box / pointcloud path:
        if self.cfg.registration.use_multiway:
            print("  Stage 3: Registering frames (Pose Graph Multiway Registration) ...")
        else:
            print("  Stage 3: Registering frames (Sequential FPFH+RANSAC -> ICP) ...")
        print("=" * 60)

        if self.cfg.registration.use_multiway:
            aligned, diagnostics = register_sequence_multiway(isolated_frames, self.cfg.registration)
        else:
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

        if self.cfg.attempt_loop_closure:
            accepted_indices = [idx for idx, (*_, acc) in enumerate(diagnostics) if acc]
            if len(accepted_indices) >= 8:
                last_idx = accepted_indices[-1]
                try:
                    _, loop_fitness, loop_rmse = register_frame_pair(
                        isolated_frames[last_idx], isolated_frames[0], self.cfg.registration
                    )
                    if loop_fitness >= self.cfg.registration.min_accept_fitness:
                        print(f"\n  Loop closure check: fitness={loop_fitness:.4f} (rmse={loop_rmse * 1000:.2f} mm) -- looks like a consistent full orbit.")
                    else:
                        print(f"\n  [!] Loop closure check: low overlap (fitness={loop_fitness:.4f}) with initial frame.")
                        print("      Drift may have accumulated over the full orbit.")
                except Exception as exc:
                    print(f"\n  [!] Loop closure check skipped: {exc}")

        if self.cfg.show_stage_windows:
            if rejected_indices:
                print(f"\n  [!] Note: Visualizing all frames including rejected frame(s) {rejected_indices}.")
            show(
                color_frames_distinctly(aligned),
                window_name="Stage 3: registered/aligned clouds",
            )

        return aligned, diagnostics, None

    # ------------------------------------------------------------------
    # Stage 4: fuse
    # ------------------------------------------------------------------

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
        print("\n" + "=" * 60)
        print("  Stage 4: Fusing aligned frames ...")
        print("=" * 60)

        if diagnostics is not None:
            # diagnostics entries can be None for frames rejected before pose-graph assignment
            accepted_indices = [
                i for i, d in enumerate(diagnostics)
                if d is not None and d[2]
            ]
            accepted_frames = [aligned_frames[i] for i in accepted_indices]
            if not accepted_frames:
                print("  [!] Warning: No frames met acceptance criteria; falling back to frame 0.")
                accepted_frames = aligned_frames[:1]
                accepted_indices = [0]
        else:
            accepted_frames = aligned_frames
            accepted_indices = list(range(len(aligned_frames)))

        n_skipped = len(aligned_frames) - len(accepted_frames)
        skip_msg = f" (excluded {n_skipped} rejected frame{'s' if n_skipped != 1 else ''})" if n_skipped > 0 else ""
        print(f"  Fusing {len(accepted_frames)} accepted frame(s){skip_msg} ...")

        # Volumetric TSDF or Point Cloud Concatenation
        if (
            self.cfg.fusion_method == "tsdf"
            and raw_frames is not None
            and raw_frames
            and isinstance(raw_frames[0], RGBDFrame)
            and poses is not None
        ):
            print("  Reconstruction mode: TSDF Volumetric Integration")
            acc_raw = [raw_frames[i] for i in accepted_indices]
            acc_poses = [poses[i] for i in accepted_indices]
            fused = fuse_tsdf_volume(
                acc_raw,
                acc_poses,
                self.cfg.preprocess,
                voxel_length=self.cfg.tsdf_voxel_length_m,
                sdf_trunc=self.cfg.tsdf_trunc_m,
            )
            fusion_name = "TSDF"
        else:
            print("  Reconstruction mode: Point Cloud Concatenation")
            fused = fuse_point_clouds(accepted_frames, self.cfg.preprocess)
            fusion_name = "POINT CLOUD"

        print(f"  Reconstruction:\n    method: {fusion_name}\n    points: {len(fused.points):,}")

        if self.cfg.show_stage_windows:
            display_pcd = o3d.geometry.PointCloud(fused)
            if not display_pcd.has_colors():
                display_pcd.paint_uniform_color([0.65, 0.65, 0.65])
            show(display_pcd, window_name=f"Stage 4: fused reconstruction ({fusion_name})")

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

        planarity = result.get("planarity_check")
        if planarity:
            print("\n  " + "-" * 50)
            print("  PLANARITY CHECK (Face RMS Deviations):")
            rms_list = planarity.get("rms_deviation_mm", [])
            if len(rms_list) == 6:
                print(f"    Axis 0 faces (near/far): {rms_list[0]:.2f} mm | {rms_list[1]:.2f} mm")
                print(f"    Axis 1 faces (near/far): {rms_list[2]:.2f} mm | {rms_list[3]:.2f} mm")
                print(f"    Axis 2 faces (near/far): {rms_list[4]:.2f} mm | {rms_list[5]:.2f} mm")
            else:
                print(f"    RMS deviations         : {rms_list}")
            print(f"    Max Face RMS Deviation : {planarity.get('max_rms_mm', 0.0):.2f} mm (limit: 4.0 mm)")

            warning = planarity.get("warning")
            if warning:
                print("\n  " + "!" * 58)
                print(f"  [!] WARNING: {warning.upper()}")
                print("  " + "!" * 58)
            else:
                print("    Box Shape Validation   : PASS (Rigid planar box)")
            print("  " + "-" * 50)

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

        print(f"\n  Target : {result.get('target', '?')}")

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

        print(_fmt_dim_line("Length", "length"))
        print(_fmt_dim_line("Width", "width"))
        print(_fmt_dim_line("Height", "height"))
        print(f"  Volume : {obb_info.get('volume_m3', 0) * 1e6:.1f} cm^3")
        aabb_ext = result.get("axis_aligned_bbox_extent_m", [])
        if aabb_ext:
            ext_mm = [f"{v * 1000:.1f}" for v in aabb_ext]
            print(f"  AABB   : {' x '.join(ext_mm)} mm (axis-aligned reference)")

        if self.cfg.show_stage_windows and obb is not None:
            seg_display = o3d.geometry.PointCloud(segmented)
            if not seg_display.has_colors():
                seg_display.paint_uniform_color([0.18, 0.72, 0.38])
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
        is_human_mode = (self.cfg.registration.registration_mode == "rgbd_human")

        raw_frames                  = self.capture_frames()
        isolated                    = self.isolate_frames(raw_frames)
        aligned, diagnostics, poses = self.register(isolated)
        fused                       = self.fuse(aligned, diagnostics, raw_frames=isolated, poses=poses)

        if is_human_mode:
            # Human reconstruction: print a clean summary instead of box dimensions.
            n_accepted = sum(1 for d in (diagnostics or []) if d is not None and d[2])
            n_total    = len(diagnostics) if diagnostics else 0
            print("\n" + "=" * 60)
            print("  HUMAN RECONSTRUCTION COMPLETE")
            print("  ============================")
            print(f"  Fusion mode    : {self.cfg.fusion_method.upper()}")
            print(f"  Frames total   : {n_total}")
            print(f"  Frames accepted: {n_accepted}")
            print(f"  Frames rejected: {n_total - n_accepted}")
            print(f"  Final points   : {len(fused.points):,}")
            print("=" * 60)

            out = self.cfg.output_dir / "human_reconstruction.ply"
            import open3d as o3d
            o3d.io.write_point_cloud(str(out), fused)
            print(f"  Saved: {out}")

            result = {
                "target": self.cfg.target.name,
                "mode": "human_reconstruction",
                "fusion_method": self.cfg.fusion_method,
                "frames_total": n_total,
                "frames_accepted": n_accepted,
                "frames_rejected": n_total - n_accepted,
                "final_points": len(fused.points),
                "output": str(out),
            }
        else:
            is_sufficient, reasons = self.check_measurement_sufficiency(diagnostics, fused, aligned)
            if not is_sufficient:
                n_accepted = sum(1 for d in (diagnostics or []) if d is not None and d[2])
                n_total    = len(diagnostics) if diagnostics else 0
                print("\n" + "!" * 60)
                print("  [!] SCAN INSUFFICIENT -- MEASUREMENT REFUSED")
                print("  " + "=" * 56)
                print("  The scan did not meet the minimum data sufficiency criteria:")
                for r in reasons:
                    print(f"    - {r}")
                print("!" * 60)

                result = {
                    "target": self.cfg.target.name,
                    "status": "insufficient_data",
                    "reasons": reasons,
                    "frames_accepted": n_accepted,
                    "frames_total": n_total,
                    "fused_points": len(fused.points),
                }
            else:
                result = self.measure(fused)

        print("\n" + "=" * 60)
        print(f"  Done. Outputs written to: {self.cfg.output_dir.resolve()}")
        print("=" * 60 + "\n")

        return result
