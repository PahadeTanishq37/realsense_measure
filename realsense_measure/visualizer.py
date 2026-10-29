"""
Lightweight Open3D visualisation helpers.

These functions exist so you can SEE what the pipeline is doing at each
stage rather than trusting numbers blindly.  Every function is side-effect-
free with respect to its input geometries — callers' clouds are never mutated.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette — 8 perceptually distinct, sensor-friendly colours.
# The palette cycles if there are more frames than entries.
# ---------------------------------------------------------------------------

_PALETTE: list[tuple[float, float, float]] = [
    (0.918, 0.263, 0.208),  # vivid red
    (0.129, 0.588, 0.953),  # vivid blue
    (0.298, 0.686, 0.314),  # vivid green
    (1.000, 0.596, 0.000),  # vivid orange
    (0.612, 0.153, 0.690),  # vivid purple
    (0.000, 0.737, 0.831),  # vivid cyan
    (1.000, 0.922, 0.231),  # vivid yellow
    (1.000, 0.341, 0.553),  # vivid pink
]


def color_frames_distinctly(
    frames: list[o3d.geometry.PointCloud],
) -> list[o3d.geometry.PointCloud]:
    """
    Return deep copies of each frame painted a distinct flat colour.

    After multi-view registration, coloured frames should visually COINCIDE
    into one coherent, multi-coloured shape.  If you see several separate
    blobs of different colours instead of them overlapping, registration has
    failed for those frames.

    Parameters
    ----------
    frames:
        Input point clouds (not mutated).

    Returns
    -------
    list[o3d.geometry.PointCloud]
        Copies of the input clouds, each painted with a colour from the
        built-in palette (cycling if ``len(frames) > len(_PALETTE)``).
    """
    coloured: list[o3d.geometry.PointCloud] = []
    for i, pcd in enumerate(frames):
        pcd_copy = copy.deepcopy(pcd)
        rgb = _PALETTE[i % len(_PALETTE)]
        pcd_copy.paint_uniform_color(rgb)
        coloured.append(pcd_copy)
    return coloured


def compute_robust_bounding_box(
    geometries: Sequence[o3d.geometry.Geometry] | o3d.geometry.Geometry,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute a robust bounding box ignoring far outliers (1st to 99th percentile per axis).

    Parameters
    ----------
    geometries:
        Geometry or list/sequence of Open3D geometries.
    p_low:
        Lower percentile (default: 1.0).
    p_high:
        Upper percentile (default: 99.0).

    Returns
    -------
    centre:
        np.ndarray (3,) center of the robust bounding box: (p1 + p99) / 2.
    max_extent:
        float, largest extent across the 3 axes: max(p99 - p1).
    p1:
        np.ndarray (3,) 1st percentile coordinates per axis.
    p99:
        np.ndarray (3,) 99th percentile coordinates per axis.
    """
    if isinstance(geometries, (list, tuple)):
        geom_list = list(geometries)
    elif hasattr(geometries, "__iter__") and not isinstance(geometries, o3d.geometry.Geometry):
        geom_list = list(geometries)
    else:
        geom_list = [geometries]

    pts_list: list[np.ndarray] = []
    for g in geom_list:
        if isinstance(g, o3d.geometry.PointCloud) and g.has_points():
            pts_list.append(np.asarray(g.points))
        elif isinstance(g, o3d.geometry.TriangleMesh) and g.has_vertices():
            pts_list.append(np.asarray(g.vertices))
        elif isinstance(g, o3d.geometry.LineSet) and g.has_points():
            pts_list.append(np.asarray(g.points))

    if pts_list:
        all_pts = np.vstack(pts_list)
        if len(all_pts) > 0:
            p_lo = np.percentile(all_pts, p_low, axis=0)
            p_hi = np.percentile(all_pts, p_high, axis=0)
            centre = (p_lo + p_hi) / 2.0
            extent = p_hi - p_lo
            max_extent = float(np.max(extent))
            return centre, max_extent, p_lo, p_hi

    # Fallback to standard min/max bounds if points are not directly available
    min_b = np.array([float("inf"), float("inf"), float("inf")])
    max_b = np.array([float("-inf"), float("-inf"), float("-inf")])
    for g in geom_list:
        if hasattr(g, "get_min_bound") and hasattr(g, "get_max_bound"):
            min_b = np.minimum(min_b, g.get_min_bound())
            max_b = np.maximum(max_b, g.get_max_bound())

    if np.all(np.isfinite(min_b)) and np.all(np.isfinite(max_b)):
        centre = (min_b + max_b) / 2.0
        extent = max_b - min_b
        return centre, float(np.max(extent)), min_b, max_b

    return np.zeros(3), 1.0, np.zeros(3), np.ones(3)


def frame_view(
    geometries: Sequence[o3d.geometry.Geometry] | o3d.geometry.Geometry | o3d.visualization.Visualizer,
    vis: o3d.visualization.Visualizer | Sequence[o3d.geometry.Geometry] | o3d.geometry.Geometry | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Frame the Open3D camera on the robust bounding box (1st-99th percentile) of geometries.

    Focuses explicitly on the robust center and scales zoom so the robust object
    (e.g., ~0.25 m head) fills ~70% of the window regardless of far outlier points.
    Preserves existing upright camera front and up orientations.

    Parameters
    ----------
    geometries:
        Open3D geometry or sequence of geometries, or Visualizer instance.
    vis:
        Active Visualizer instance (optional if passed as first argument).

    Returns
    -------
    centre:
        np.ndarray (3,) center of robust bounding box.
    extent:
        float, largest robust extent in metres.
    zoom:
        float, applied zoom level.
    """
    # Allow flexible argument order: frame_view(vis, geometries) or frame_view(geometries, vis)
    if isinstance(geometries, o3d.visualization.Visualizer):
        vis, geometries = geometries, vis

    if isinstance(geometries, (list, tuple)):
        geom_list = list(geometries)
    elif hasattr(geometries, "__iter__") and not isinstance(geometries, o3d.geometry.Geometry):
        geom_list = list(geometries)
    else:
        geom_list = [geometries] if geometries is not None else []

    centre, robust_extent, p1, p99 = compute_robust_bounding_box(geom_list)

    # Compute full extent across all geometries to evaluate outlier dispersion
    full_min = np.array([float("inf"), float("inf"), float("inf")])
    full_max = np.array([float("-inf"), float("-inf"), float("-inf")])
    for g in geom_list:
        if hasattr(g, "get_min_bound") and hasattr(g, "get_max_bound"):
            full_min = np.minimum(full_min, g.get_min_bound())
            full_max = np.maximum(full_max, g.get_max_bound())

    if np.all(np.isfinite(full_min)) and np.all(np.isfinite(full_max)):
        full_extent = float(np.max(full_max - full_min))
    else:
        full_extent = robust_extent

    full_extent = max(full_extent, 1e-4)
    robust_extent = max(robust_extent, 1e-4)

    # In Open3D, camera distance is proportional to full_extent * zoom.
    # To have the robust box fill ~70% of the window:
    # zoom = 0.70 * (robust_extent / full_extent)
    # Clamped between 0.1 and 1.0.
    zoom = float(np.clip(0.70 * (robust_extent / full_extent), 0.1, 1.0))

    if vis is not None:
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        R = cam.extrinsic[:3, :3]
        front = -R[2, :]
        up = -R[1, :]

        vc.set_lookat(centre.tolist())
        vc.set_front(front.tolist())
        vc.set_up(up.tolist())
        vc.set_zoom(zoom)

        # In presence of extreme distant outliers where zoom hits the 0.1 limit,
        # adjust camera distance via pinhole parameters so robust box fills 70% of window height:
        fy = cam.intrinsic.intrinsic_matrix[1, 1]
        target_height_px = 800.0 * 0.70
        desired_dist = fy * robust_extent / target_height_px
        E = np.eye(4)
        E[:3, :3] = R
        t = -R @ centre
        t[2] += desired_dist
        E[:3, 3] = t
        cam.extrinsic = E
        vc.convert_from_pinhole_camera_parameters(cam)

    return centre, robust_extent, zoom


def show(
    geometries: Sequence[o3d.geometry.Geometry] | o3d.geometry.Geometry,
    window_name: str = "stage",
    point_size: float = 3.0,
    screenshot_path: str | Path | None = None,
) -> None:
    """
    Open an interactive Open3D window displaying all given geometries.

    Uses a 1280x800 window with point size 3.0. Bounding boxes are logged
    for diagnosis before opening. The camera is framed on the robust 1st-99th
    percentile box so the subject fills ~70% of the viewport. Pressing 'R'
    re-applies the robust framing.
    """
    if isinstance(geometries, (list, tuple)):
        geom_list = list(geometries)
    elif hasattr(geometries, "__iter__") and not isinstance(geometries, o3d.geometry.Geometry):
        geom_list = list(geometries)
    else:
        geom_list = [geometries]

    # --- 1. DIAGNOSTIC LOGGING OF BOUNDING BOXES ---
    print(f"\n========================================================================")
    print(f"  [VIEWER DIAGNOSTIC] Opening Window: '{window_name}'")
    print(f"  Total Geometries: {len(geom_list)}")
    print(f"------------------------------------------------------------------------")

    combined_min = np.array([float("inf"), float("inf"), float("inf")])
    combined_max = np.array([float("-inf"), float("-inf"), float("-inf")])

    for i, geom in enumerate(geom_list):
        gtype = type(geom).__name__
        n_pts = 0
        if hasattr(geom, "points"):
            n_pts = len(geom.points)
        elif hasattr(geom, "vertices"):
            n_pts = len(geom.vertices)

        gmin = geom.get_min_bound() if hasattr(geom, "get_min_bound") else np.zeros(3)
        gmax = geom.get_max_bound() if hasattr(geom, "get_max_bound") else np.zeros(3)
        gext = gmax - gmin

        combined_min = np.minimum(combined_min, gmin)
        combined_max = np.maximum(combined_max, gmax)

        print(f"  Geometry [{i}] {gtype} ({n_pts} pts):")
        print(f"    min:    [{gmin[0]:.4f}, {gmin[1]:.4f}, {gmin[2]:.4f}] m")
        print(f"    max:    [{gmax[0]:.4f}, {gmax[1]:.4f}, {gmax[2]:.4f}] m")
        print(f"    extent: [{gext[0]:.4f}, {gext[1]:.4f}, {gext[2]:.4f}] m (max: {np.max(gext):.4f} m)")

    combined_ext = combined_max - combined_min
    r_centre, r_extent, r_p1, r_p99 = compute_robust_bounding_box(geom_list)

    print(f"  ----------------------------------------------------------------------")
    print(f"  COMBINED AABB (Full Box with Outliers):")
    print(f"    min:    [{combined_min[0]:.4f}, {combined_min[1]:.4f}, {combined_min[2]:.4f}] m")
    print(f"    max:    [{combined_max[0]:.4f}, {combined_max[1]:.4f}, {combined_max[2]:.4f}] m")
    print(f"    extent: [{combined_ext[0]:.4f}, {combined_ext[1]:.4f}, {combined_ext[2]:.4f}] m (max: {np.max(combined_ext):.4f} m)")
    print(f"  ROBUST AABB (1st-99th percentile):")
    print(f"    centre: [{r_centre[0]:.4f}, {r_centre[1]:.4f}, {r_centre[2]:.4f}] m")
    print(f"    p1:     [{r_p1[0]:.4f}, {r_p1[1]:.4f}, {r_p1[2]:.4f}] m")
    print(f"    p99:    [{r_p99[0]:.4f}, {r_p99[1]:.4f}, {r_p99[2]:.4f}] m")
    print(f"    extent: [{r_p99[0]-r_p1[0]:.4f}, {r_p99[1]-r_p1[1]:.4f}, {r_p99[2]-r_p1[2]:.4f}] m (max: {r_extent:.4f} m)")
    print(f"  Controls: Press 'R' in window to re-center/frame robust view | Drag=Rotate | Shift+Drag=Pan")
    print(f"========================================================================\n")

    logger.info(
        f"[Viewer] '{window_name}': {len(geom_list)} geoms. Combined ext: {np.max(combined_ext):.3f} m, "
        f"Robust ext: {r_extent:.3f} m, Robust centre: {r_centre.round(3)}"
    )

    # --- 2. CREATE WINDOW & REGISTER KEY CALLBACK ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_title = f"{window_name} [Press 'R' to Frame]"
    vis.create_window(window_name=window_title, width=1280, height=800)

    for geom in geom_list:
        vis.add_geometry(geom)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.10, 0.10, 0.12])  # near-black

    # --- 3. APPLY ROBUST CAMERA FRAMING ---
    frame_view(geom_list, vis=vis)

    # --- 4. KEY 'R' CALLBACK TO RE-APPLY FRAMING ---
    def on_reframe(v: o3d.visualization.Visualizer) -> bool:
        print("  [Viewer] 'R' pressed -> Re-applying robust camera framing...")
        frame_view(geom_list, vis=v)
        v.update_renderer()
        return False

    vis.register_key_callback(ord('R'), on_reframe)
    vis.register_key_callback(ord('r'), on_reframe)

    # Optional auto-close timer if O3D_AUTOCLOSE_MS environment variable is set
    autoclose_ms = os.environ.get("O3D_AUTOCLOSE_MS")
    if autoclose_ms:
        try:
            close_after_sec = float(autoclose_ms) / 1000.0
            start_time = time.time()

            def on_tick(v: o3d.visualization.Visualizer) -> bool:
                if time.time() - start_time >= close_after_sec:
                    v.close()
                return False

            vis.register_animation_callback(on_tick)
        except ValueError:
            pass

    vis.run()      # blocks until the user closes the window
    if screenshot_path is not None:
        vis.capture_screen_image(str(screenshot_path), do_render=True)
    vis.destroy_window()


def prepare_display_geometry(
    geometry: o3d.geometry.Geometry | Sequence[o3d.geometry.Geometry],
    voxel_size_m: float = 0.006,
) -> o3d.geometry.Geometry | list[o3d.geometry.Geometry]:
    """
    Return a display-only downsampled deep copy of geometry.

    For point clouds, creates a separate deep copy downsampled to ``voxel_size_m``
    so on-screen rendering is clean and uncluttered rather than overly dense fuzz.
    Non-point-cloud geometries (e.g. LineSet, TriangleMesh) are returned as copies.
    """
    if isinstance(geometry, (list, tuple)):
        return [
            g.voxel_down_sample(voxel_size=voxel_size_m)
            if isinstance(g, o3d.geometry.PointCloud) and g.has_points() and voxel_size_m > 0
            else copy.deepcopy(g)
            for g in geometry
        ]
    elif isinstance(geometry, o3d.geometry.PointCloud):
        if geometry.has_points() and voxel_size_m > 0:
            return geometry.voxel_down_sample(voxel_size=voxel_size_m)
        return copy.deepcopy(geometry)
    return copy.deepcopy(geometry)


def obb_lineset(
    obb: o3d.geometry.OrientedBoundingBox,
    color: tuple[float, float, float] = (1.0, 0.8, 0.0),
) -> o3d.geometry.LineSet:
    """
    Convert an OrientedBoundingBox into a coloured wireframe LineSet.

    Draws the 12 edges of the fitted bounding box on top of the point cloud
    so you can visually verify the measurement result.

    Parameters
    ----------
    obb:
        Fitted oriented bounding box, e.g. as returned by
        ``BoxTarget.measure()["geometry_for_viz"]``.
    color:
        RGB colour in [0, 1] range.  Default is a golden-yellow that is
        visible against both dark and light point clouds.

    Returns
    -------
    o3d.geometry.LineSet
        Wireframe representation of ``obb``, ready to pass into :func:`show`.
    """
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lineset.paint_uniform_color(color)
    return lineset


def visualize_registration_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    title: str = "Pairwise Registration",
    point_size: float = 2.5,
) -> None:
    """
    Display pairwise registration result (source in red, target in cyan).

    Shows:
      1. BEFORE REGISTRATION: raw source and target in their respective coordinates.
      2. AFTER REGISTRATION: source transformed onto target coordinates.
    """
    src_before = copy.deepcopy(source)
    tgt_before = copy.deepcopy(target)
    src_before.paint_uniform_color([0.92, 0.25, 0.20])   # red
    tgt_before.paint_uniform_color([0.10, 0.75, 0.90])   # cyan

    show(
        [src_before, tgt_before],
        window_name=f"{title} -- BEFORE alignment (Red=Source, Cyan=Target)",
        point_size=point_size,
    )

    src_after = copy.deepcopy(source).transform(transform)
    tgt_after = copy.deepcopy(target)
    src_after.paint_uniform_color([0.92, 0.25, 0.20])    # red
    tgt_after.paint_uniform_color([0.10, 0.75, 0.90])    # cyan

    show(
        [src_after, tgt_after],
        window_name=f"{title} -- AFTER alignment (Coincidence check)",
        point_size=point_size,
    )


def draw_camera_trajectory(
    poses: list[np.ndarray],
    frame_size: float = 0.04,
    max_jump_m: float = 0.15,
) -> list[o3d.geometry.Geometry]:
    """
    Generate 3D camera trajectory geometries showing optical centers, orientations, and jumps.

    Parameters
    ----------
    poses:
        List of 4x4 camera-to-world transformation matrices.
    frame_size:
        Size of the coordinate axis indicator at each camera pose in metres.
    max_jump_m:
        Translation threshold beyond which an inter-frame jump is flagged as suspicious.

    Returns
    -------
    list[o3d.geometry.Geometry]
        LineSet connecting trajectory centers, plus coordinate frame meshes for each pose.
    """
    if not poses:
        return []

    geometries: list[o3d.geometry.Geometry] = []

    # 1. Coordinate frame at each camera position
    centers: list[np.ndarray] = []
    for i, pose in enumerate(poses):
        center = pose[:3, 3]
        centers.append(center)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        axis.transform(pose)
        geometries.append(axis)

    # 2. Path line segments with jump detection
    if len(centers) >= 2:
        centers_arr = np.array(centers)
        lines: list[list[int]] = []
        colors: list[list[float]] = []

        for i in range(len(centers) - 1):
            dist = float(np.linalg.norm(centers_arr[i + 1] - centers_arr[i]))
            lines.append([i, i + 1])
            if dist > max_jump_m:
                print(f"  [!] SUSPICIOUS TRAJECTORY JUMP: Camera C{i} -> C{i + 1} "
                      f"moved {dist * 1000:.1f} mm (> {max_jump_m * 1000:.0f} mm limit)")
                colors.append([1.0, 0.1, 0.1])  # Bright red for suspicious jump
            else:
                colors.append([0.2, 0.9, 0.2])  # Green for normal trajectory

        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(centers_arr),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(ls)

    return geometries

