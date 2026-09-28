"""
Lightweight Open3D visualisation helpers.

These functions exist so you can SEE what the pipeline is doing at each
stage rather than trusting numbers blindly.  Every function is side-effect-
free with respect to its input geometries — callers' clouds are never mutated.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import open3d as o3d


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


def show(
    geometries: Sequence[o3d.geometry.Geometry] | o3d.geometry.Geometry,
    window_name: str = "stage",
    point_size: float = 2.0,
) -> None:
    """
    Open a blocking Open3D window displaying all given geometries.

    The window blocks until the user closes it, then is destroyed cleanly.
    All rendering options (background colour, point size) are set before the
    event loop starts.

    Parameters
    ----------
    geometries:
        Any Open3D geometry object or list of geometry objects to display
        (PointCloud, LineSet, TriangleMesh, OrientedBoundingBox, etc.).
    window_name:
        Title bar text -- useful for identifying which pipeline stage you are
        looking at.
    point_size:
        Rendered radius of each point in pixels.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    if isinstance(geometries, (list, tuple)):
        geom_list = list(geometries)
    elif hasattr(geometries, "__iter__") and not isinstance(geometries, o3d.geometry.Geometry):
        geom_list = list(geometries)
    else:
        geom_list = [geometries]

    for geom in geom_list:
        vis.add_geometry(geom)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.10, 0.10, 0.12])  # near-black

    vis.run()      # blocks until the user closes the window
    vis.destroy_window()


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

