"""
realsense_measure/fullbody/io.py

I/O utilities for full-body capture sessions and scan results.
Loads raw capture sessions into list[StopData] supporting both standard
meta.json capture formats and per-stop burst directories.
Saves and loads ScanResult (.json + .ply/.npz).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import (
    AlignedStop,
    MeasurementResult,
    ScanResult,
    StopData,
)

logger = logging.getLogger(__name__)


def load_capture_session(
    capture_dir: Union[str, Path],
    cfg: FullBodyConfig,
) -> List[StopData]:
    """
    Load a raw full-body capture directory into a list of StopData.

    Supported layout 1 (standard recording with meta.json):
        capture_dir/
            meta.json
            frame_00_color.png
            frame_00_depth.npy
            ...

    Supported layout 2 (per-stop subfolders):
        capture_dir/
            stop_00/
                color.png, depth.npy (or burst_00.png, burst_00.npy, ...)
            stop_01/
                ...

    Raises
    ------
    FileNotFoundError:
        If capture directory or required files do not exist.
    ValueError:
        If the directory format is unrecognized or invalid.
    """
    p = Path(capture_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Capture directory does not exist: {p}")

    meta_file = p / "meta.json"
    if meta_file.exists():
        return _load_meta_json_session(p, meta_file, cfg)

    # Check for per-stop subdirectories
    subdirs = sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("stop_")])
    if subdirs:
        return _load_subdirs_session(subdirs, cfg)

    raise ValueError(
        f"Unrecognized capture layout in '{p}'. Expected either:\n"
        f"  (1) 'meta.json' with frame_XX_color.png and frame_XX_depth.npy files, or\n"
        f"  (2) 'stop_00/', 'stop_01/' subdirectories containing color and depth files."
    )


def _load_meta_json_session(
    root: Path,
    meta_path: Path,
    cfg: FullBodyConfig,
) -> List[StopData]:
    """Parse session recorded via standard realsense_measure save_raw_capture."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    intr_dict = meta.get("intrinsics", {})
    w = intr_dict.get("width", 848)
    h = intr_dict.get("height", 480)
    fx = intr_dict.get("fx", 428.0)
    fy = intr_dict.get("fy", 428.0)
    cx = intr_dict.get("cx", 422.0)
    cy = intr_dict.get("cy", 244.0)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"No frames defined in {meta_path}")

    stops: List[StopData] = []
    n_frames = len(frames)
    step_deg = 360.0 / max(1, n_frames)

    turn_dir = meta.get("turn_direction", cfg.turn_direction)
    cam_rot = meta.get("camera_portrait_rotation", cfg.camera_portrait_rotation)

    for i, f_entry in enumerate(frames):
        c_rel = f_entry.get("color_file", f"frame_{i:02d}_color.png")
        d_rel = f_entry.get("depth_file", f"frame_{i:02d}_depth.npy")

        c_path = root / c_rel
        d_path = root / d_rel

        if not c_path.exists() or not d_path.exists():
            raise FileNotFoundError(f"Missing frame data for stop {i}: {c_path} or {d_path}")

        color_bgr = cv2.imread(str(c_path))
        if color_bgr is None:
            raise ValueError(f"Failed to read image: {c_path}")

        # Depth may be .npy or .png in 16-bit millimetres
        if d_path.suffix == ".npy":
            depth_m = np.load(str(d_path)).astype(np.float32)
        else:
            raw_depth_u16 = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
            depth_m = raw_depth_u16.astype(np.float32) * 0.001

        # Check portrait orientation swap
        if color_bgr.shape[0] != h or color_bgr.shape[1] != w:
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                color_bgr.shape[1], color_bgr.shape[0], fx, fy, cx, cy
            )

        nominal_angle = (i * step_deg) if turn_dir == "cw" else (-i * step_deg) % 360.0
        t_stamp = float(f_entry.get("timestamp", 0.0))

        stops.append(
            StopData(
                stop_id=i,
                nominal_angle_deg=nominal_angle,
                color=color_bgr,
                depth=depth_m,
                intrinsics=intrinsics,
                timestamp=t_stamp,
            )
        )

    logger.info(f"  [io] Loaded {len(stops)} stops from {root}")
    return stops


def _load_subdirs_session(
    subdirs: List[Path],
    cfg: FullBodyConfig,
) -> List[StopData]:
    """Parse session formatted as individual stop directories."""
    stops: List[StopData] = []
    n_stops = len(subdirs)
    step_deg = 360.0 / max(1, n_stops)

    for i, s_dir in enumerate(subdirs):
        # Look for color and depth files inside stop folder
        color_files = sorted(list(s_dir.glob("*color*.png")) + list(s_dir.glob("*.jpg")))
        depth_files = sorted(list(s_dir.glob("*depth*.npy")) + list(s_dir.glob("*depth*.png")))

        if not color_files or not depth_files:
            continue

        color_bgr = cv2.imread(str(color_files[0]))
        if depth_files[0].suffix == ".npy":
            depth_m = np.load(str(depth_files[0])).astype(np.float32)
        else:
            depth_m = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001

        h, w = depth_m.shape[:2]
        # Default portrait pinhole intrinsics if no meta.json is inside subfolder
        fx = fy = float(max(h, w))
        cx = float(w / 2.0)
        cy = float(h / 2.0)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        nominal_angle = (i * step_deg) if cfg.turn_direction == "cw" else (-i * step_deg) % 360.0
        stops.append(
            StopData(
                stop_id=i,
                nominal_angle_deg=nominal_angle,
                color=color_bgr,
                depth=depth_m,
                intrinsics=intrinsics,
            )
        )

    logger.info(f"  [io] Loaded {len(stops)} stops from subdirectories in {subdirs[0].parent}")
    return stops


def save_scan_result(
    result: ScanResult,
    output_dir: Union[str, Path],
) -> Path:
    """
    Save ScanResult to directory:
    - scan_result.json
    - mesh.ply
    - fused.ply
    - alignment.npz
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. JSON report
    data: Dict[str, Any] = {
        "subject_id": result.subject_id,
        "session_id": result.session_id,
        "timestamp": result.timestamp,
        "measurements": {k: m.to_dict() for k, m in result.measurements.items()},
        "landmarks": {k: [round(float(coord), 4) for coord in v] for k, v in result.landmarks.items()},
        "hole_map": result.hole_map,
        "metadata": result.metadata,
        "num_aligned_stops": len(result.aligned_stops),
    }

    json_path = out / "scan_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 2. 3D geometry
    if result.mesh is not None and len(result.mesh.vertices) > 0:
        o3d.io.write_triangle_mesh(str(out / "mesh.ply"), result.mesh)
    if result.pcd is not None and len(result.pcd.points) > 0:
        o3d.io.write_point_cloud(str(out / "fused.ply"), result.pcd)

    # 3. Alignment transforms NPZ
    if result.aligned_stops:
        transforms = {
            f"stop_{a.stop_id:02d}": a.T_canonical_camera
            for a in result.aligned_stops if a.T_canonical_camera is not None
        }
        yaws = np.array([a.refined_yaw_deg for a in result.aligned_stops])
        np.savez(out / "alignment.npz", yaws_deg=yaws, **transforms)

    logger.info(f"  [io] Saved scan results and 3D assets to {out}")
    return out


def load_scan_result(
    output_dir: Union[str, Path],
) -> ScanResult:
    """Load previously saved ScanResult from directory."""
    p = Path(output_dir)
    json_path = p / "scan_result.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No scan_result.json found in {p}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = ScanResult(
        subject_id=data.get("subject_id"),
        session_id=data.get("session_id"),
        timestamp=float(data.get("timestamp", 0.0)),
        hole_map=data.get("hole_map", {}),
        metadata=data.get("metadata", {}),
    )

    for k, m_dict in data.get("measurements", {}).items():
        res.measurements[k] = MeasurementResult(
            name=m_dict["name"],
            value_mm=float(m_dict["value_mm"]) if m_dict["value_mm"] is not None else float("nan"),
            method=m_dict["method"],
            coverage=float(m_dict["coverage"]),
            interpolated_fraction=float(m_dict["interpolated_fraction"]),
            quality_flag=m_dict["quality_flag"],
            uncertainty_mm=float(m_dict.get("uncertainty_mm", 0.0)),
            notes=m_dict.get("notes", ""),
        )

    for k, coords in data.get("landmarks", {}).items():
        res.landmarks[k] = np.array(coords, dtype=np.float64)

    mesh_file = p / "mesh.ply"
    if mesh_file.exists():
        res.mesh = o3d.io.read_triangle_mesh(str(mesh_file))

    pcd_file = p / "fused.ply"
    if pcd_file.exists():
        res.pcd = o3d.io.read_point_cloud(str(pcd_file))

    return res

