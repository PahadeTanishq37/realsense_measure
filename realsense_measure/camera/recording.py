"""
Record and replay functionality for raw RGB-D captures.

Enables reproducible benchmarking and offline pipeline evaluation without requiring
a physical RealSense camera or display server during replay.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import open3d as o3d

from camera.frame import RGBDFrame, create_pcd_from_rgbd

if TYPE_CHECKING:
    from config import PipelineConfig

logger = logging.getLogger(__name__)


def save_raw_capture(
    raw_frames: list[RGBDFrame],
    output_dir: Path | str,
    cfg: PipelineConfig,
) -> Path:
    """
    Save raw captured frames before Stage 2 object isolation.

    For each frame, saves:
      - color: lossless PNG
      - depth: float32 numpy array (.npy) with metric depth values in metres
      - timestamp & order: recorded in a single meta.json along with intrinsics and CameraConfig

    Parameters
    ----------
    raw_frames:
        List of raw RGBDFrame instances from Stage 1 capture.
    output_dir:
        Target directory to store images, numpy arrays, and meta.json.
    cfg:
        Active PipelineConfig providing camera and stream configuration.

    Returns
    -------
    Path
        Path to the raw capture folder.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_frames:
        logger.warning(f"  [save-raw] No frames to save in {out_path}")
        return out_path

    # Extract intrinsics from first frame (or cfg)
    first_frame = raw_frames[0]
    intr = first_frame.intrinsics
    intr_mat = intr.intrinsic_matrix

    intr_meta = {
        "width": intr.width,
        "height": intr.height,
        "fx": float(intr_mat[0, 0]),
        "fy": float(intr_mat[1, 1]),
        "cx": float(intr_mat[0, 2]),
        "cy": float(intr_mat[1, 2]),
    }

    cam_meta = {
        "width": cfg.camera.width,
        "height": cfg.camera.height,
        "fps": cfg.camera.fps,
        "use_decimation": cfg.camera.use_decimation,
        "decimation_magnitude": cfg.camera.decimation_magnitude,
        "use_spatial_filter": cfg.camera.use_spatial_filter,
        "use_temporal_filter": cfg.camera.use_temporal_filter,
        "use_hole_filling": cfg.camera.use_hole_filling,
        "depth_min_m": cfg.camera.depth_min_m,
        "depth_max_m": cfg.camera.depth_max_m,
        "human_face_quality_mode": cfg.camera.human_face_quality_mode,
    }

    frames_meta = []
    for i, frame in enumerate(raw_frames):
        fid = getattr(frame, "frame_id", i)
        ts = getattr(frame, "timestamp", float(i))

        color_filename = f"frame_{i:02d}_color.png"
        depth_filename = f"frame_{i:02d}_depth.npy"

        color_file = out_path / color_filename
        depth_file = out_path / depth_filename

        # Save color as lossless PNG
        cv2.imwrite(str(color_file), frame.color_bgr)

        # Save depth in metres as float32 .npy
        np.save(str(depth_file), frame.depth_m.astype(np.float32))

        frames_meta.append({
            "frame_id": fid,
            "order": i,
            "timestamp": ts,
            "color_file": color_filename,
            "depth_file": depth_filename,
        })

    meta_doc = {
        "intrinsics": intr_meta,
        "camera_config": cam_meta,
        "target_name": cfg.target.name,
        "num_frames": len(raw_frames),
        "frames": frames_meta,
    }

    meta_path = out_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)

    logger.info(f"  [save-raw] Saved {len(raw_frames)} raw frames and meta.json to: {out_path}/")
    return out_path


def load_raw_capture(input_dir: Path | str) -> list[RGBDFrame]:
    """
    Load raw RGB-D frames from a previous recording without importing or requiring pyrealsense2.

    Rebuilds synchronized RGBDFrame instances (including point clouds) exactly as live capture does.

    Parameters
    ----------
    input_dir:
        Directory containing meta.json, PNG color images, and .npy depth files.

    Returns
    -------
    list[RGBDFrame]
        Reconstructed RGBDFrame list ready for Stages 2-5.
    """
    in_path = Path(input_dir)
    meta_path = in_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Raw capture metadata not found at {meta_path}. "
            "Ensure the directory contains a valid meta.json from --save-raw."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_doc = json.load(f)

    intr_meta = meta_doc["intrinsics"]
    intr = o3d.camera.PinholeCameraIntrinsic(
        int(intr_meta["width"]),
        int(intr_meta["height"]),
        float(intr_meta["fx"]),
        float(intr_meta["fy"]),
        float(intr_meta["cx"]),
        float(intr_meta["cy"]),
    )

    cam_cfg = meta_doc.get("camera_config", {})
    depth_min_m = float(cam_cfg.get("depth_min_m", 0.40))
    depth_max_m = float(cam_cfg.get("depth_max_m", 1.20))

    frames_meta = meta_doc.get("frames", [])
    raw_frames: list[RGBDFrame] = []

    for item in frames_meta:
        fid = int(item["frame_id"])
        ts = float(item["timestamp"])
        color_path = in_path / item["color_file"]
        depth_path = in_path / item["depth_file"]

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Failed to read color frame image at {color_path}")

        depth_m = np.load(str(depth_path)).astype(np.float32)

        pcd = create_pcd_from_rgbd(
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=intr,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )

        frame = RGBDFrame(
            frame_id=fid,
            color_bgr=color_bgr,
            depth_m=depth_m,
            pcd=pcd,
            timestamp=ts,
            intrinsics=intr,
        )
        raw_frames.append(frame)

    logger.info(f"  [replay] Rebuilt {len(raw_frames)} frames from {in_path}/ (zero camera dependencies)")
    return raw_frames


def parse_named_config(config_str: str, base_cfg: PipelineConfig) -> tuple[PipelineConfig, list[str]]:
    """
    Parse a configuration name or compound specification (e.g. 'multiway+tsdf', 'head_only+multiway+tsdf').

    Returns (configured_pipeline_config, capture_time_only_warnings).
    """
    import copy
    cfg = copy.deepcopy(base_cfg)

    tokens = [t.strip().lower() for t in config_str.replace(",", "+").replace("|", "+").split("+") if t.strip()]
    warnings = []

    for tok in tokens:
        if tok in ("baseline", "default"):
            pass
        elif tok == "multiway":
            cfg.registration.use_multiway = True
        elif tok == "sequential":
            cfg.registration.use_multiway = False
        elif tok == "tsdf":
            cfg.fusion_method = "tsdf"
        elif tok == "pointcloud":
            cfg.fusion_method = "pointcloud"
        elif tok in ("head_only", "head_only_registration"):
            cfg.registration.head_only_registration = True
        elif "face_quality" in tok:
            warnings.append(f"--face-quality setting '{tok}' is capture-time only and cannot be modified at replay")
        elif "decimation" in tok or "resolution" in tok or "filter" in tok:
            warnings.append(f"Camera setting '{tok}' is capture-time only and cannot be modified at replay")
        else:
            logger.warning(f"  [replay-batch] Unrecognized config token: '{tok}'")

    return cfg, warnings


def run_replay_batch(
    raw_dir: Path | str,
    config_names: list[str],
    output_base_dir: Path | str = "batch_runs",
    target_name: str = "head",
) -> Path:
    """
    Execute a list of named configurations against the recorded raw capture.

    Writes per-config runs to separate subdirectories, and aggregates a single
    comparison_summary.csv with registration tier counts, pair RMSE metrics,
    edges, stage 4 points, fusion method, and stage 5 measurements.

    Parameters
    ----------
    raw_dir:
        Directory containing the saved raw capture.
    config_names:
        List of configuration specifiers (e.g. ['baseline', 'head_only+multiway+tsdf']).
    output_base_dir:
        Root folder for batch execution outputs.
    target_name:
        Target type (defaults to 'head').

    Returns
    -------
    Path
        Path to comparison_summary.csv.
    """
    from config import PipelineConfig
    from pipeline import ScanPipeline

    raw_path = Path(raw_dir)
    out_base = Path(output_base_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for name in config_names:
        logger.info("\n" + "=" * 65)
        logger.info(f"  REPLAY BATCH: Running configuration '{name}'")
        logger.info("=" * 65)

        base_cfg = PipelineConfig(
            output_dir=out_base / name,
            show_stage_windows=False,
            replay_dir=raw_path,
        )
        base_cfg.target.name = target_name
        base_cfg.registration.resolve_defaults(target_name)

        run_cfg, cap_warnings = parse_named_config(name, base_cfg)

        for w in cap_warnings:
            logger.warning(f"  [CAPTURE-TIME ONLY] {w}")

        pipeline = ScanPipeline(run_cfg)
        pipeline.scan_folder = run_cfg.output_dir
        result = pipeline.run()

        # Gather summary metrics
        reg_stats = getattr(pipeline, "last_registration_stats", {})
        tier_summary = reg_stats.get("tier_summary_str", "")

        # Compute pair RMSE metrics
        diagnostics = reg_stats.get("diagnostics", [])
        rmses_mm = [
            float(d[1]) * 1000.0
            for d in diagnostics
            if d is not None and len(d) > 1 and d[1] is not None and np.isfinite(d[1]) and d[1] > 0
        ]
        mean_rmse = float(np.mean(rmses_mm)) if rmses_mm else 0.0
        median_rmse = float(np.median(rmses_mm)) if rmses_mm else 0.0
        max_rmse = float(np.max(rmses_mm)) if rmses_mm else 0.0

        n_edges = reg_stats.get("n_edges", 0)
        n_loop = reg_stats.get("n_loop", 0)
        primary_edges = max(0, n_edges - n_loop)
        edges_str = f"{primary_edges} primary, {n_loop} loop"

        stage4_pts = result.get("points_in_stage4") or reg_stats.get("fused_points", 0)
        if stage4_pts == 0 and (run_cfg.output_dir / "stage4_output.ply").exists():
            try:
                p_test = o3d.io.read_point_cloud(str(run_cfg.output_dir / "stage4_output.ply"))
                stage4_pts = len(p_test.points)
            except Exception:
                pass

        fusion_method = run_cfg.fusion_method.upper()

        # Stage 5 measurement
        stage5_meas = ""
        head_meas = result.get("head_measurements") or result.get("dimensions_m")
        if head_meas:
            if isinstance(head_meas, dict):
                parts = [f"{k}={v*1000.0:.1f}mm" if isinstance(v, float) else f"{k}={v}" for k, v in head_meas.items()]
                stage5_meas = "; ".join(parts)
            else:
                stage5_meas = str(head_meas)
        elif "oriented_bbox" in result:
            obb = result["oriented_bbox"]
            stage5_meas = f"L={obb.get('length_m', 0)*1000.0:.1f}mm, W={obb.get('width_m', 0)*1000.0:.1f}mm, H={obb.get('height_m', 0)*1000.0:.1f}mm"

        summary_rows.append({
            "config_name": name,
            "registration_tier_counts": tier_summary,
            "mean_pair_rmse_mm": f"{mean_rmse:.2f}",
            "median_pair_rmse_mm": f"{median_rmse:.2f}",
            "max_pair_rmse_mm": f"{max_rmse:.2f}",
            "edges": edges_str,
            "points_in_stage4": stage4_pts,
            "fusion_method": fusion_method,
            "stage5_measurement": stage5_meas,
            "capture_time_notes": "; ".join(cap_warnings) if cap_warnings else "none",
        })

    csv_path = out_base / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config_name",
            "registration_tier_counts",
            "mean_pair_rmse_mm",
            "median_pair_rmse_mm",
            "max_pair_rmse_mm",
            "edges",
            "points_in_stage4",
            "fusion_method",
            "stage5_measurement",
            "capture_time_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info(f"\n  [replay-batch] Comparison summary written to: {csv_path}")
    return csv_path
