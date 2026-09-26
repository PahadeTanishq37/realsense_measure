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
