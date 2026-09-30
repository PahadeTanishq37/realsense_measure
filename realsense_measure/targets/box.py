"""
BoxTarget: measure a rigid, roughly box-shaped object.

Registered under the name "box".  See targets/base.py for the extension
contract — adding a new target type (head, body) follows the same pattern.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from config import PreprocessConfig
from preprocessing import extract_largest_cluster, remove_dominant_plane
from targets.base import Target, register_target

# Below this, a "face" is considered flat (sensor-noise-only deviation).
# Above it, the surface at that end of that axis is curved and/or the
# registration that put it there is misaligned — either way the reported
# dimension for that axis should not be trusted at face value.
DEFAULT_FLATNESS_WARN_M = 0.006  # 6 mm


def fit_plane_to_slab(
    local_pts: np.ndarray,
    axis: int,
    end: str,
    slab_fraction: float = 0.08,
) -> dict:
    """
    Fit a plane to the outer slab of points at one end of one local axis.

    Parameters
    ----------
    local_pts:
        Nx3 points already expressed in the OBB's local frame (see
        ``BoxTarget.measure`` for how this is built).
    axis:
        Which local axis (0, 1, or 2) to take the slab from.
    end:
        ``"lo"`` for the slab at the minimum end of this axis, ``"hi"`` for
        the slab at the maximum end.
    slab_fraction:
        Fraction of points (by percentile along this axis) that make up the
        slab, e.g. 0.08 takes the outermost 8%.

    Returns
    -------
    dict with keys:
        ``coord``   -- the slab centroid's coordinate along ``axis`` (this
                       is what "where is this face" means for the extent
                       calculation)
        ``rms_m``   -- RMS distance of slab points from the fitted plane;
                       near sensor noise (~1-3 mm) for a real flat face,
                       much higher for a curved surface or a misaligned
                       double-wall from imperfect registration
        ``n_points`` -- how many points made up the slab (a very small
                       count means the fit itself is unreliable)
    """
    v = local_pts[:, axis]
    if end == "lo":
        cutoff = np.percentile(v, slab_fraction * 100.0)
        slab = local_pts[v <= cutoff]
    else:
        cutoff = np.percentile(v, 100.0 - slab_fraction * 100.0)
        slab = local_pts[v >= cutoff]

    if len(slab) < 10:
        return {"coord": float(cutoff), "rms_m": float("nan"), "n_points": len(slab)}

    centroid = slab.mean(axis=0)
    _, _, vt = np.linalg.svd(slab - centroid)
    normal = vt[-1]
    residuals = (slab - centroid) @ normal
    return {
        "coord": float(centroid[axis]),
        "rms_m": float(residuals.std()),
        "n_points": int(len(slab)),
    }


@register_target
class BoxTarget(Target):
    """
    Segment and measure a rigid box-shaped object from a fused point cloud.

    The measurement pipeline is:
      1. :meth:`segment` — strip any residual planar background, keep the
         single largest connected component.
      2. :meth:`measure` — fit a minimum-volume oriented bounding box, then
         measure each axis by fitting a plane to the two opposing faces
         (rather than trusting raw point extent), and flag any axis whose
         "face" isn't actually flat.
    """

    name: str = "box"

    def __init__(
        self,
        preprocess_cfg: PreprocessConfig | None = None,
        flatness_warn_m: float = DEFAULT_FLATNESS_WARN_M,
        slab_fraction: float = 0.08,
    ) -> None:
        """
        Parameters
        ----------
        preprocess_cfg:
            Preprocessing settings shared with the rest of the pipeline.
            If ``None``, sensible defaults are used (``PreprocessConfig()``).
        flatness_warn_m:
            RMS-from-plane threshold above which a face is flagged as not
            flat (see :data:`DEFAULT_FLATNESS_WARN_M`).
        slab_fraction:
            Fraction of points taken as each face's slab for plane fitting
            (see :func:`fit_plane_to_slab`).
        """
        self.cfg: PreprocessConfig = preprocess_cfg or PreprocessConfig()
        self.flatness_warn_m = flatness_warn_m
        self.slab_fraction = slab_fraction

    # ------------------------------------------------------------------
    # Target interface
    # ------------------------------------------------------------------

    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove residual planar background and return the dominant cluster.

        Even though the fused cloud has already been processed per-frame,
        small amounts of table/wall geometry can be dragged back in during
        registration (especially near the edges of the reference cloud).
        This extra plane-removal pass mops those up before measurement.
        """
        no_plane, _ = remove_dominant_plane(pcd, self.cfg)

        if not no_plane.has_points():
            # Plane removal consumed everything — return the original cloud
            # rather than propagating an empty cloud to measure().
            return pcd

        return extract_largest_cluster(no_plane, self.cfg)

    def measure(
        self,
        pcd: o3d.geometry.PointCloud,
        trim_percentile: float = 0.5,
    ) -> dict:
        """
        Fit an oriented bounding box, then measure each axis from its two
        fitted faces rather than raw point extent.

        Parameters
        ----------
        pcd:
            Segmented box cloud, as returned by :meth:`segment`.
        trim_percentile:
            Percentile used for the legacy percentile-trimmed extent, kept
            in the output under ``"percentile_trim_extent_m"`` purely for
            comparison/debugging — it is no longer what ``length_m`` /
            ``width_m`` / ``height_m`` are computed from.

        Returns
        -------
        dict
            JSON-serialisable measurement dict with one extra key
            ``"geometry_for_viz"`` holding the fitted OBB for the
            visualizer, and a ``"warnings"`` list that is non-empty when
            one or more axes did not look like real flat box faces.

        Raises
        ------
        ValueError
            If the cloud has fewer than 10 points after segmentation.
        """
        if len(pcd.points) < 10:
            raise ValueError(
                "Not enough points survived segmentation to measure the box "
                f"(got {len(pcd.points)}, need at least 10)."
            )

        # Fit a minimum-volume oriented bounding box.
        # get_minimal_oriented_bounding_box was added in Open3D >= 0.18;
        # fall back to get_oriented_bounding_box for older installs.
        try:
            obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        except AttributeError:
            obb = pcd.get_oriented_bounding_box(robust=True)

        pts = np.asarray(pcd.points)
        R = np.asarray(obb.R)           # 3x3 rotation matrix (OBB axes as columns)
        center = np.asarray(obb.center)
        local_pts = (pts - center) @ R  # points expressed in the OBB's local frame

        # ------------------------------------------------------------------
        # Legacy percentile-trimmed extent — kept for comparison only.
        # This is measurably biased HIGH on real sensor data (a handful of
        # noisy edge points inflate every axis by several millimetres), and
        # gives no indication of WHETHER a face is actually flat.
        # ------------------------------------------------------------------
        lo_pct = np.percentile(local_pts, trim_percentile, axis=0)
        hi_pct = np.percentile(local_pts, 100.0 - trim_percentile, axis=0)
        percentile_extent = hi_pct - lo_pct

        # ------------------------------------------------------------------
        # PRIMARY measurement: fit a plane to each of the six faces and
        # measure the perpendicular distance between opposing plane pairs.
        # This is both more accurate (a plane fit is not moved by a single
        # outlier point the way a percentile cutoff can be) and
        # self-validating: the RMS-from-plane residual tells you directly
        # whether that face was actually flat, instead of silently
        # returning a confident-looking number for a curved or
        # misregistered surface.
        # ------------------------------------------------------------------
        order = np.argsort(percentile_extent)[::-1]  # longest -> shortest local axis
        axis_names = ["length", "width", "height"]

        face_extent_m = [0.0, 0.0, 0.0]
        face_rms_m = [0.0, 0.0, 0.0]
        warnings: list[str] = []

        for rank, axis in enumerate(order):
            lo = fit_plane_to_slab(local_pts, axis, "lo", self.slab_fraction)
            hi = fit_plane_to_slab(local_pts, axis, "hi", self.slab_fraction)
            extent = hi["coord"] - lo["coord"]
            rms = max(
                lo["rms_m"] if lo["rms_m"] == lo["rms_m"] else 0.0,   # nan-safe
                hi["rms_m"] if hi["rms_m"] == hi["rms_m"] else 0.0,
            )
            face_extent_m[rank] = float(extent)
            face_rms_m[rank] = float(rms)

            if rms > self.flatness_warn_m:
                warnings.append(
                    f"{axis_names[rank]} axis: face RMS-from-flat = {rms * 1000:.1f} mm "
                    f"(threshold {self.flatness_warn_m * 1000:.0f} mm) -- this face is not "
                    f"flat. The object may not be a rigid box, or registration may be "
                    f"misaligned along this axis. Treat this dimension as unreliable."
                )
            if lo["n_points"] < 20 or hi["n_points"] < 20:
                warnings.append(
                    f"{axis_names[rank]} axis: only {min(lo['n_points'], hi['n_points'])} "
                    f"points on one face -- too few to fit reliably."
                )

        length, width, height = face_extent_m

        aabb = pcd.get_axis_aligned_bounding_box()
        aabb_extent = [round(float(v), 4) for v in np.asarray(aabb.get_extent())]

        return {
            "target": "box",
            "num_points": len(pcd.points),
            "warnings": warnings,
            "oriented_bbox": {
                "length_m": round(length, 4),
                "width_m": round(width, 4),
                "height_m": round(height, 4),
                "volume_m3": round(length * width * height, 6),
                "face_rms_m": [round(v, 5) for v in face_rms_m],
                "center": center.tolist(),
                "rotation": R.tolist(),
            },
            # Legacy percentile-trim result, for comparison / debugging only.
            "percentile_trim_extent_m": [
                round(float(v), 4) for v in np.sort(percentile_extent)[::-1]
            ],
            "axis_aligned_bbox_extent_m": aabb_extent,
            "geometry_for_viz": obb,
        }
