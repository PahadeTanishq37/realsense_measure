"""
Provenance-preserving RGB-D Point Cloud Representation.

Maintains complete traceability from every 3D reconstructed point back to:
- Original RGB and depth pixel coordinates (u, v)
- Source capture frame ID
- Hardware and system timestamp
- Calibrated camera intrinsics and coordinate frame
- Depth validity, raw metric depth, and observation weight
- Aggregation count (distinguishing directly observed K=1 from downsampled K>1)
- Contributing frame IDs for multi-frame fused voxels

Provides vectorized back-projection, provenance-preserving filtering (ROI, plane,
DBSCAN, statistical outliers), aggregation-aware voxel downsampling, Open3D
interoperability, and compressed NPZ serialization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import open3d as o3d


@dataclass
class FrameMetadata:
    """
    Metadata describing the sensor parameters and capture conditions for a frame.

    Attributes
    ----------
    frame_id:
        Unique identifier for the capture frame.
    timestamp:
        Hardware or system capture timestamp in seconds.
    intrinsics:
        3x3 pinhole intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    width:
        Image width in pixels.
    height:
        Image height in pixels.
    depth_scale:
        Scale factor used to convert depth buffer units to metres.
    coordinate_frame:
        Name of the coordinate frame (default: 'camera').
    extra_info:
        Optional dictionary of additional metadata (e.g. quality metrics, exposure).
    """

    frame_id: int
    timestamp: float = 0.0
    intrinsics: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    width: int = 848
    height: int = 480
    depth_scale: float = 1.0
    coordinate_frame: str = "camera"
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-serializable dictionary."""
        return {
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "intrinsics": self.intrinsics.tolist(),
            "width": int(self.width),
            "height": int(self.height),
            "depth_scale": float(self.depth_scale),
            "coordinate_frame": str(self.coordinate_frame),
            "extra_info": self.extra_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameMetadata":
        """Deserialize metadata from dictionary."""
        intr = np.asarray(data.get("intrinsics", np.eye(3)), dtype=np.float64)
        return cls(
            frame_id=int(data.get("frame_id", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            intrinsics=intr,
            width=int(data.get("width", 848)),
            height=int(data.get("height", 480)),
            depth_scale=float(data.get("depth_scale", 1.0)),
            coordinate_frame=str(data.get("coordinate_frame", "camera")),
            extra_info=dict(data.get("extra_info", {})),
        )

    @classmethod
    def from_o3d_intrinsics(
        cls,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        frame_id: int = 0,
        timestamp: float = 0.0,
        depth_scale: float = 1.0,
        coordinate_frame: str = "camera",
        extra_info: dict[str, Any] | None = None,
    ) -> "FrameMetadata":
        """Create FrameMetadata from an Open3D PinholeCameraIntrinsic object."""
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            intrinsics=np.array(intrinsics.intrinsic_matrix, dtype=np.float64),
            width=intrinsics.width,
            height=intrinsics.height,
            depth_scale=depth_scale,
            coordinate_frame=coordinate_frame,
            extra_info={} if extra_info is None else extra_info,
        )

    def to_o3d_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        """Convert to an Open3D PinholeCameraIntrinsic object."""
        mat = self.intrinsics
        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width,
            height=self.height,
            fx=float(mat[0, 0]),
            fy=float(mat[1, 1]),
            cx=float(mat[0, 2]),
            cy=float(mat[1, 2]),
        )


class IndexedPointCloud:
    """
    Structured 3D point cloud retaining full sensor and pixel provenance.

    Every 3D point retains its source pixel coordinates (u, v), depth value,
    frame ID, validity, weight, and aggregation history across multi-frame merges
    and voxel downsampling operations.
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        source_frame_id: np.ndarray,
        source_pixel_uv: np.ndarray,
        source_depth_uv: np.ndarray | None = None,
        source_depth_m: np.ndarray | None = None,
        valid_depth: np.ndarray | None = None,
        observation_weight: np.ndarray | None = None,
        aggregation_count: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        contributing_frame_ids: list[list[int]] | None = None,
        cf_flat: np.ndarray | None = None,
        cf_offsets: np.ndarray | None = None,
        frames_meta: dict[int, FrameMetadata] | None = None,
        coordinate_frame: str = "camera",
        transform_matrix: np.ndarray | None = None,
    ):
        n = len(points)
        self.points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        self.colors = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)

        if normals is not None:
            self.normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3)
        else:
            self.normals = None

        self.source_frame_id = np.ascontiguousarray(source_frame_id, dtype=np.int32).reshape(-1)
        self.source_pixel_uv = np.ascontiguousarray(source_pixel_uv, dtype=np.int32).reshape(-1, 2)

        if source_depth_uv is not None:
            self.source_depth_uv = np.ascontiguousarray(source_depth_uv, dtype=np.int32).reshape(-1, 2)
        else:
            self.source_depth_uv = self.source_pixel_uv.copy()

        if source_depth_m is not None:
            self.source_depth_m = np.ascontiguousarray(source_depth_m, dtype=np.float32).reshape(-1)
        else:
            self.source_depth_m = self.points[:, 2].copy()

        if valid_depth is not None:
            self.valid_depth = np.ascontiguousarray(valid_depth, dtype=bool).reshape(-1)
        else:
            self.valid_depth = np.ones(n, dtype=bool)

        if observation_weight is not None:
            self.observation_weight = np.ascontiguousarray(observation_weight, dtype=np.float32).reshape(-1)
        else:
            self.observation_weight = np.ones(n, dtype=np.float32)

        if aggregation_count is not None:
            self.aggregation_count = np.ascontiguousarray(aggregation_count, dtype=np.int32).reshape(-1)
        else:
            self.aggregation_count = np.ones(n, dtype=np.int32)

        # Compact contributor-frame storage (ragged array: cf_flat and cf_offsets)
        # Single-frame / un-aggregated clouds use cf_flat=None, cf_offsets=None (zero per-point lists)
        if cf_flat is not None and cf_offsets is not None:
            self.cf_flat = np.ascontiguousarray(cf_flat, dtype=np.int32).reshape(-1)
            self.cf_offsets = np.ascontiguousarray(cf_offsets, dtype=np.int64).reshape(-1)
        elif contributing_frame_ids is not None:
            is_implicit = all(
                len(c) == 1 and (len(self.source_frame_id) <= i or c[0] == self.source_frame_id[i])
                for i, c in enumerate(contributing_frame_ids)
            )
            if is_implicit:
                self.cf_flat = None
                self.cf_offsets = None
            else:
                flat = []
                offsets = [0]
                for c in contributing_frame_ids:
                    flat.extend(c)
                    offsets.append(len(flat))
                self.cf_flat = np.ascontiguousarray(flat, dtype=np.int32)
                self.cf_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        else:
            self.cf_flat = None
            self.cf_offsets = None

        self.frames_meta = {} if frames_meta is None else copy.deepcopy(frames_meta)
        self.coordinate_frame = coordinate_frame
        self._pixel_index_cache: dict[tuple[int, bool], np.ndarray] = {}

        if transform_matrix is not None:
            self.transform_matrix = np.ascontiguousarray(transform_matrix, dtype=np.float64).reshape(4, 4)
        else:
            self.transform_matrix = np.eye(4, dtype=np.float64)

        # Enforce consistent lengths
        if len(self.colors) != n or len(self.source_frame_id) != n or len(self.source_pixel_uv) != n:
            raise ValueError(
                f"Array length mismatch: points={len(self.points)}, colors={len(self.colors)}, "
                f"source_frame_id={len(self.source_frame_id)}, source_pixel_uv={len(self.source_pixel_uv)}"
            )
        if self.cf_offsets is not None and len(self.cf_offsets) != n + 1:
            raise ValueError(
                f"cf_offsets length {len(self.cf_offsets)} does not match point count + 1 ({n + 1})"
            )

    def __len__(self) -> int:
        return len(self.points)

    def has_points(self) -> bool:
        return len(self.points) > 0

    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def contributing_frame_ids(self) -> list[list[int]]:
        """
        Return contributing frame IDs as a list of lists for backward compatibility.
        Constructed dynamically on demand without storing persistent Python list objects.
        """
        n = len(self.points)
        if self.cf_flat is None or self.cf_offsets is None:
            return [[int(fid)] for fid in self.source_frame_id]
        return [
            self.cf_flat[self.cf_offsets[i] : self.cf_offsets[i + 1]].tolist()
            for i in range(n)
        ]

    @contributing_frame_ids.setter
    def contributing_frame_ids(self, value: list[list[int]] | None) -> None:
        if value is None:
            self.cf_flat = None
            self.cf_offsets = None
        else:
            is_implicit = all(
                len(c) == 1 and (len(self.source_frame_id) <= i or c[0] == self.source_frame_id[i])
                for i, c in enumerate(value)
            )
            if is_implicit:
                self.cf_flat = None
                self.cf_offsets = None
            else:
                flat = []
                offsets = [0]
                for c in value:
                    flat.extend(c)
                    offsets.append(len(flat))
                self.cf_flat = np.ascontiguousarray(flat, dtype=np.int32)
                self.cf_offsets = np.ascontiguousarray(offsets, dtype=np.int64)

    def get_contributing_frame_ids(self, point_idx: int) -> list[int]:
        """
        Return contributing frame IDs for a specific point without allocating lists for all points.
        """
        if point_idx < 0 or point_idx >= len(self.points):
            raise IndexError(f"Point index {point_idx} out of range [0, {len(self.points)})")
        if self.cf_flat is None or self.cf_offsets is None:
            return [int(self.source_frame_id[point_idx])]
        start = self.cf_offsets[point_idx]
        end = self.cf_offsets[point_idx + 1]
        return self.cf_flat[start:end].tolist()

    def get_pixel_index_map(
        self,
        frame_id: int | None = None,
        unambiguous_only: bool = True,
    ) -> np.ndarray:
        """
        Construct or retrieve a 2D integer mapping from pixel coordinates (u, v) to point indices:
            pixel_to_point[v, u] = point_index

        Coordinates (u, v) correspond strictly to the RGB/aligned depth image plane for `frame_id`.
        If the sensor captured a rotated image (e.g. 90 deg portrait), (u, v) indexes the rotated grid.

        Parameters
        ----------
        frame_id:
            The capture frame ID to build the map for. If None and the cloud has metadata
            for exactly one frame, that frame ID is used.
        unambiguous_only:
            If True, only observations with aggregation_count == 1 (pure un-aggregated 1-to-1 rays)
            are mapped. Voxel centroids with aggregation_count > 1 (which merged multiple rays)
            are left as -1 to prevent false unique pixel claims.
            If False, representative points of downsampled voxels are also mapped.

        Returns
        -------
        np.ndarray:
            (H, W) array of dtype np.int32 initialized to -1. Valid points contain their 0-based
            index in self.points.
        """
        if frame_id is None:
            if len(self.frames_meta) == 1:
                frame_id = next(iter(self.frames_meta.keys()))
            elif len(self.frames_meta) == 0:
                frame_id = 0
            else:
                raise ValueError(
                    f"Point cloud contains multiple frames {list(self.frames_meta.keys())}; "
                    "frame_id must be specified explicitly."
                )

        target_fid = int(frame_id)
        cache_key = (target_fid, bool(unambiguous_only))
        if hasattr(self, "_pixel_index_cache") and cache_key in self._pixel_index_cache:
            return self._pixel_index_cache[cache_key]

        # Determine dimensions from frame metadata
        if target_fid in self.frames_meta:
            h = int(self.frames_meta[target_fid].height)
            w = int(self.frames_meta[target_fid].width)
        else:
            h, w = 480, 848

        grid = np.full((h, w), -1, dtype=np.int32)
        if len(self.points) == 0:
            if not hasattr(self, "_pixel_index_cache"):
                self._pixel_index_cache = {}
            self._pixel_index_cache[cache_key] = grid
            return grid

        # Filter points belonging to this frame
        frame_mask = (self.source_frame_id == target_fid)
        all_pt_indices = np.nonzero(frame_mask)[0]
        if len(all_pt_indices) == 0:
            if not hasattr(self, "_pixel_index_cache"):
                self._pixel_index_cache = {}
            self._pixel_index_cache[cache_key] = grid
            return grid

        u_all = self.source_pixel_uv[all_pt_indices, 0]
        v_all = self.source_pixel_uv[all_pt_indices, 1]
        in_bounds = (u_all >= 0) & (u_all < w) & (v_all >= 0) & (v_all < h)
        if not np.any(in_bounds):
            if not hasattr(self, "_pixel_index_cache"):
                self._pixel_index_cache = {}
            self._pixel_index_cache[cache_key] = grid
            return grid

        pt_indices = all_pt_indices[in_bounds]
        u = u_all[in_bounds]
        v = v_all[in_bounds]

        lin_idx = v.astype(np.int64) * w + u.astype(np.int64)
        pixel_counts = np.bincount(lin_idx, minlength=w * h)

        if unambiguous_only:
            # An observation is unambiguous iff:
            # 1. Exactly one surviving point in this cloud claims (target_fid, u, v) (pixel_counts == 1)
            # 2. That point has not been merged across voxels (aggregation_count == 1)
            is_unique = (pixel_counts[lin_idx] == 1)
            is_unaggregated = (self.aggregation_count[pt_indices] == 1)
            valid_mask = is_unique & is_unaggregated
            if np.any(valid_mask):
                valid_pt_indices = pt_indices[valid_mask]
                grid[v[valid_mask], u[valid_mask]] = valid_pt_indices.astype(np.int32)
        else:
            # If unambiguous_only is False, map representative points deterministically.
            # In case of duplicate keys, the first occurrence is mapped.
            _, unique_first_indices = np.unique(lin_idx, return_index=True)
            rep_pt_indices = pt_indices[unique_first_indices]
            rep_u = u[unique_first_indices]
            rep_v = v[unique_first_indices]
            grid[rep_v, rep_u] = rep_pt_indices.astype(np.int32)

        if not hasattr(self, "_pixel_index_cache"):
            self._pixel_index_cache = {}
        self._pixel_index_cache[cache_key] = grid
        return grid

    def get_point_index_at_pixel(
        self,
        u: int,
        v: int,
        frame_id: int | None = None,
        unambiguous_only: bool = True,
    ) -> int:
        """
        Retrieve the 0-based point index in self.points corresponding to pixel (u, v).

        Parameters
        ----------
        u:
            Horizontal pixel coordinate (column index, 0 <= u < width).
        v:
            Vertical pixel coordinate (row index, 0 <= v < height).
        frame_id:
            Source frame ID. Defaults to the unique frame if single-frame.
        unambiguous_only:
            If True, returns -1 if the point at (u, v) is an aggregated voxel centroid.

        Returns
        -------
        int:
            Point index in [0, len(self.points) - 1], or -1 if no valid point exists or if out of bounds.
        """
        if frame_id is None and len(self.frames_meta) == 1:
            target_fid = next(iter(self.frames_meta.keys()))
        elif frame_id is not None:
            target_fid = int(frame_id)
        else:
            target_fid = 0

        if target_fid in self.frames_meta:
            w = int(self.frames_meta[target_fid].width)
            h = int(self.frames_meta[target_fid].height)
        else:
            w, h = 848, 480

        # Bounds checking
        u_int, v_int = int(u), int(v)
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            return -1

        grid = self.get_pixel_index_map(frame_id=target_fid, unambiguous_only=unambiguous_only)
        return int(grid[v_int, u_int])

    def get_point_at_pixel(
        self,
        u: int,
        v: int,
        frame_id: int | None = None,
        unambiguous_only: bool = True,
    ) -> np.ndarray | None:
        """
        Retrieve the 3D metric coordinate np.ndarray (3,) for pixel (u, v), or None if invalid.
        """
        idx = self.get_point_index_at_pixel(
            u, v, frame_id=frame_id, unambiguous_only=unambiguous_only
        )
        if idx >= 0:
            return self.points[idx]
        return None

    def clone(self) -> "IndexedPointCloud":
        """Return a deep copy of the IndexedPointCloud."""
        return IndexedPointCloud(
            points=self.points.copy(),
            colors=self.colors.copy(),
            source_frame_id=self.source_frame_id.copy(),
            source_pixel_uv=self.source_pixel_uv.copy(),
            source_depth_uv=self.source_depth_uv.copy(),
            source_depth_m=self.source_depth_m.copy(),
            valid_depth=self.valid_depth.copy(),
            observation_weight=self.observation_weight.copy(),
            aggregation_count=self.aggregation_count.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            cf_flat=self.cf_flat.copy() if self.cf_flat is not None else None,
            cf_offsets=self.cf_offsets.copy() if self.cf_offsets is not None else None,
            frames_meta=copy.deepcopy(self.frames_meta),
            coordinate_frame=self.coordinate_frame,
            transform_matrix=self.transform_matrix.copy(),
        )

    def select_by_mask(self, mask: np.ndarray) -> "IndexedPointCloud":
        """Select a subset of points by boolean mask, preserving all provenance."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self.points):
            raise ValueError(f"Mask length {len(mask)} does not match point count {len(self.points)}")

        sub_cf_flat = None
        sub_cf_offsets = None
        if self.cf_flat is not None and self.cf_offsets is not None:
            counts = self.cf_offsets[1:] - self.cf_offsets[:-1]
            sub_counts = counts[mask]
            sub_cf_offsets = np.zeros(len(sub_counts) + 1, dtype=np.int64)
            np.cumsum(sub_counts, out=sub_cf_offsets[1:])

            pt_indices = np.repeat(np.arange(len(mask), dtype=np.int64), counts)
            elem_mask = mask[pt_indices]
            sub_cf_flat = self.cf_flat[elem_mask]

        return IndexedPointCloud(
            points=self.points[mask],
            colors=self.colors[mask],
            source_frame_id=self.source_frame_id[mask],
            source_pixel_uv=self.source_pixel_uv[mask],
            source_depth_uv=self.source_depth_uv[mask],
            source_depth_m=self.source_depth_m[mask],
            valid_depth=self.valid_depth[mask],
            observation_weight=self.observation_weight[mask],
            aggregation_count=self.aggregation_count[mask],
            normals=self.normals[mask] if self.normals is not None else None,
            cf_flat=sub_cf_flat,
            cf_offsets=sub_cf_offsets,
            frames_meta=self.frames_meta,
            coordinate_frame=self.coordinate_frame,
            transform_matrix=self.transform_matrix.copy(),
        )

    def select_by_index(
        self, indices: np.ndarray | Sequence[int], invert: bool = False
    ) -> "IndexedPointCloud":
        """Select a subset of points by integer index list/array, preserving provenance."""
        n = len(self.points)
        mask = np.zeros(n, dtype=bool)
        if len(indices) > 0:
            mask[np.asarray(indices, dtype=np.int64)] = True
        if invert:
            mask = ~mask
        return self.select_by_mask(mask)

    def filter_roi(
        self,
        min_bound: Sequence[float] | np.ndarray,
        max_bound: Sequence[float] | np.ndarray,
    ) -> "IndexedPointCloud":
        """Filter points strictly within the 3D axis-aligned bounding box [min_bound, max_bound]."""
        if self.is_empty():
            return self.clone()
        min_b = np.asarray(min_bound, dtype=np.float32).reshape(3)
        max_b = np.asarray(max_bound, dtype=np.float32).reshape(3)
        in_box = (
            (self.points[:, 0] >= min_b[0])
            & (self.points[:, 0] <= max_b[0])
            & (self.points[:, 1] >= min_b[1])
            & (self.points[:, 1] <= max_b[1])
            & (self.points[:, 2] >= min_b[2])
            & (self.points[:, 2] <= max_b[2])
        )
        return self.select_by_mask(in_box)

    def filter_statistical_outliers(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> tuple["IndexedPointCloud", np.ndarray]:
        """
        Remove statistical outliers using Open3D's algorithm while preserving provenance.

        Returns
        -------
        cleaned_pcd:
            Filtered IndexedPointCloud retaining exact original pixel coordinates.
        inlier_indices:
            Array of surviving inlier indices into the original cloud.
        """
        if len(self.points) < nb_neighbors:
            return self.clone(), np.arange(len(self.points), dtype=np.int64)

        o3d_pcd = self.to_open3d()
        _, inliers = o3d_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        inlier_arr = np.asarray(inliers, dtype=np.int64)
        return self.select_by_index(inlier_arr), inlier_arr

    def filter_plane(
        self,
        distance_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000,
        return_plane_model: bool = True,
    ) -> tuple["IndexedPointCloud", list[float]]:
        """
        Segment and remove the dominant plane using RANSAC.

        Returns
        -------
        non_plane_pcd:
            Cloud with plane inliers removed, retaining provenance for all retained points.
        plane_model:
            [a, b, c, d] plane equation ax + by + cz + d = 0.
        """
        if len(self.points) < ransac_n:
            return self.clone(), [0.0, 0.0, 0.0, 0.0]

        o3d_pcd = self.to_open3d()
        plane_model, inliers = o3d_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        non_plane = self.select_by_index(inliers, invert=True)
        return non_plane, list(plane_model)

    def segment_plane(
        self,
        distance_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> tuple[list[float], list[int]]:
        """Open3D-compatible plane segmentation returning (plane_model, inlier_indices_list)."""
        if len(self.points) < ransac_n:
            return [0.0, 0.0, 0.0, 0.0], []
        o3d_pcd = self.to_open3d()
        plane_model, inliers = o3d_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        return list(plane_model), list(inliers)

    def voxel_down_sample(self, voxel_size: float) -> "IndexedPointCloud":
        """Open3D-compatible alias for voxel_downsample."""
        return self.voxel_downsample(voxel_size)

    def remove_statistical_outlier(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> tuple["IndexedPointCloud", list[int]]:
        """Open3D-compatible alias returning (filtered_cloud, inlier_indices_list)."""
        filtered_pcd, inliers = self.filter_statistical_outliers(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return filtered_pcd, inliers.tolist()

    def get_axis_aligned_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """Return the Open3D axis-aligned bounding box enclosing this cloud."""
        if self.is_empty():
            return o3d.geometry.AxisAlignedBoundingBox()
        min_b = np.min(self.points, axis=0).astype(np.float64)
        max_b = np.max(self.points, axis=0).astype(np.float64)
        return o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)

    def get_min_bound(self) -> np.ndarray:
        """Return the minimum 3D coordinate bound."""
        if self.is_empty():
            return np.zeros(3, dtype=np.float64)
        return np.min(self.points, axis=0).astype(np.float64)

    def get_max_bound(self) -> np.ndarray:
        """Return the maximum 3D coordinate bound."""
        if self.is_empty():
            return np.zeros(3, dtype=np.float64)
        return np.max(self.points, axis=0).astype(np.float64)

    def get_center(self) -> np.ndarray:
        """Return the geometric center (mean of points)."""
        if self.is_empty():
            return np.zeros(3, dtype=np.float64)
        return np.mean(self.points, axis=0).astype(np.float64)

    def crop(
        self,
        bounding_box: o3d.geometry.AxisAlignedBoundingBox | Sequence[Sequence[float]],
    ) -> "IndexedPointCloud":
        """Crop points inside an AxisAlignedBoundingBox or [min_bound, max_bound]."""
        if hasattr(bounding_box, "get_min_bound"):
            min_b = bounding_box.get_min_bound()
            max_b = bounding_box.get_max_bound()
        else:
            min_b, max_b = bounding_box[0], bounding_box[1]
        return self.filter_roi(min_b, max_b)

    def paint_uniform_color(self, color: Sequence[float]) -> "IndexedPointCloud":
        """Assign uniform color in [0, 1] across all points."""
        col = np.asarray(color, dtype=np.float32).reshape(1, 3)
        self.colors[:] = col
        return self

    def cluster_dbscan(
        self, eps: float = 0.05, min_points: int = 10, print_progress: bool = False, *args, **kwargs
    ) -> np.ndarray:
        """
        Run DBSCAN clustering on 3D coordinates.

        Returns
        -------
        labels:
            Cluster label for each point (-1 indicates noise).
        """
        if self.is_empty():
            return np.array([], dtype=np.int32)
        o3d_pcd = self.to_open3d()
        labels = o3d_pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress
        )
        return np.asarray(labels, dtype=np.int32)

    def voxel_downsample(
        self, voxel_size: float, aggregation_strategy: str = "centroid_nearest"
    ) -> "IndexedPointCloud":
        """
        Voxel downsampling with complete provenance preservation and aggregation history.

        Each output voxel contains:
        - Geometric centroid: arithmetic mean of contributing 3D points.
        - Color: arithmetic mean of contributing RGB values.
        - Normals: normalized mean of contributing normals (if present).
        - aggregation_count: sum of contributing aggregation counts (K = sum(k_i)).
          Points directly observed have K=1; downsampled points have K > 1.
        - Provenance representative: based on `aggregation_strategy`.
          For 'centroid_nearest', the point in the voxel closest to the voxel centroid
          provides the representative (u, v) pixel coordinate, source frame ID,
          depth value, and weight.
        - contributing_frame_ids: list of all unique frame IDs that contributed
          to that voxel.
        """
        if self.is_empty():
            return self.clone()
        if voxel_size <= 0.0:
            return self.clone()

        # Compute integer voxel grid coordinates
        voxel_coords = np.floor(self.points / voxel_size).astype(np.int64)

        # Group points by unique voxel coordinates
        unique_voxels, inverse_indices, counts = np.unique(
            voxel_coords, axis=0, return_inverse=True, return_counts=True
        )
        num_voxels = len(unique_voxels)

        # 1. Geometric Centroids
        sum_pts = np.zeros((num_voxels, 3), dtype=np.float64)
        np.add.at(sum_pts, inverse_indices, self.points)
        centroid_points = (sum_pts / counts[:, None]).astype(np.float32)

        # 2. Colors
        sum_colors = np.zeros((num_voxels, 3), dtype=np.float64)
        np.add.at(sum_colors, inverse_indices, self.colors)
        new_colors = (sum_colors / counts[:, None]).astype(np.float32)

        # 3. Normals (if present)
        new_normals = None
        if self.normals is not None:
            sum_normals = np.zeros((num_voxels, 3), dtype=np.float64)
            np.add.at(sum_normals, inverse_indices, self.normals)
            norms = np.linalg.norm(sum_normals, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_normals = (sum_normals / norms).astype(np.float32)

        # 4. Aggregation count (K = sum of original aggregation counts)
        sum_agg = np.zeros(num_voxels, dtype=np.int32)
        np.add.at(sum_agg, inverse_indices, self.aggregation_count)
        new_aggregation_count = sum_agg

        # 5. Provenance Representative Selection
        # Find the point in each voxel that is closest to the voxel centroid
        dists_sq = np.sum((self.points - centroid_points[inverse_indices]) ** 2, axis=1)

        # Lexsort by inverse_indices (primary key) and dists_sq (secondary key)
        order = np.lexsort((dists_sq, inverse_indices))
        # Unique finds the first occurrence of each voxel in the sorted order (minimum distance)
        _, first_indices = np.unique(inverse_indices[order], return_index=True)
        best_pt_indices = order[first_indices]

        new_frame_id = self.source_frame_id[best_pt_indices]
        new_pixel_uv = self.source_pixel_uv[best_pt_indices]
        new_depth_uv = self.source_depth_uv[best_pt_indices]
        new_depth_m = self.source_depth_m[best_pt_indices]
        new_valid_depth = self.valid_depth[best_pt_indices]
        new_weight = self.observation_weight[best_pt_indices]

        # 6. Multi-frame Contributing IDs
        # Collect distinct contributing frames for each voxel as compact ragged arrays
        if self.cf_flat is None or self.cf_offsets is None:
            # Each input point i has frame self.source_frame_id[i]
            pairs = np.column_stack([inverse_indices, self.source_frame_id])
        else:
            # Each input point i has frames self.cf_flat[cf_offsets[i]:cf_offsets[i+1]]
            counts = self.cf_offsets[1:] - self.cf_offsets[:-1]
            pt_indices = np.repeat(np.arange(len(self.points), dtype=np.int64), counts)
            voxel_for_elem = inverse_indices[pt_indices]
            pairs = np.column_stack([voxel_for_elem, self.cf_flat])

        unique_pairs = np.unique(pairs, axis=0)
        voxel_counts = np.bincount(unique_pairs[:, 0], minlength=num_voxels)
        new_cf_offsets = np.zeros(num_voxels + 1, dtype=np.int64)
        np.cumsum(voxel_counts, out=new_cf_offsets[1:])
        new_cf_flat = unique_pairs[:, 1].astype(np.int32)

        return IndexedPointCloud(
            points=centroid_points,
            colors=new_colors,
            source_frame_id=new_frame_id,
            source_pixel_uv=new_pixel_uv,
            source_depth_uv=new_depth_uv,
            source_depth_m=new_depth_m,
            valid_depth=new_valid_depth,
            observation_weight=new_weight,
            aggregation_count=new_aggregation_count,
            normals=new_normals,
            cf_flat=new_cf_flat,
            cf_offsets=new_cf_offsets,
            frames_meta=self.frames_meta,
            coordinate_frame=self.coordinate_frame,
            transform_matrix=self.transform_matrix.copy(),
        )

    def transform(self, matrix: np.ndarray) -> "IndexedPointCloud":
        """
        Apply a 4x4 rigid transformation to 3D point positions and normals in place.

        Sensor provenance (u, v pixel coordinates, original depth, intrinsics)
        remains unchanged. The cumulative transform is updated.
        """
        mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        r = mat[:3, :3].astype(np.float32)
        t = mat[:3, 3].astype(np.float32)

        self.points = (self.points @ r.T) + t

        if self.normals is not None:
            self.normals = self.normals @ r.T
            norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            self.normals = self.normals / norms

        self.transform_matrix = mat @ self.transform_matrix
        return self

    def estimate_normals(
        self,
        search_param: o3d.geometry.KDTreeSearchParam | None = None,
    ) -> None:
        """Estimate normals in place using Open3D's KDTree normal estimation."""
        if self.is_empty():
            return
        o3d_pcd = self.to_open3d()
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        o3d_pcd.estimate_normals(search_param=search_param)
        self.normals = np.asarray(o3d_pcd.normals, dtype=np.float32)

    def merge(self, other: "IndexedPointCloud") -> "IndexedPointCloud":
        """Merge this cloud with another IndexedPointCloud, concatenating provenance arrays."""
        return self.concatenate([self, other])

    @classmethod
    def concatenate(cls, clouds: Sequence["IndexedPointCloud"]) -> "IndexedPointCloud":
        """
        Concatenate multiple IndexedPointCloud instances into one.

        Preserves all individual points, colors, normals, and complete sensor provenance.
        Does not merge or alter conflicting 3D coordinates.

        Handling of Duplicate Pixel Observations:
        If input clouds contain repeated frame IDs and overlapping pixel coordinates (frame_id, u, v),
        all duplicate points are preserved in the concatenated cloud.
        When querying via `get_pixel_index_map(frame_id, unambiguous_only=True)` or
        `get_point_index_at_pixel(u, v, frame_id, unambiguous_only=True)`, any pixel
        possessing multiple surviving observations will return -1 to signal ambiguity.
        Ambiguity is resolved either when the cloud is filtered such that only one
        observation survives, or when `voxel_downsample()` explicitly merges the observations
        into a fused centroid.
        """
        valid_clouds = [c for c in clouds if not c.is_empty()]
        if not valid_clouds:
            if clouds:
                return clouds[0].clone()
            raise ValueError("Cannot concatenate empty list of point clouds")

        all_points = np.vstack([c.points for c in valid_clouds])
        all_colors = np.vstack([c.colors for c in valid_clouds])
        all_fid = np.concatenate([c.source_frame_id for c in valid_clouds])
        all_uv = np.vstack([c.source_pixel_uv for c in valid_clouds])
        all_depth_uv = np.vstack([c.source_depth_uv for c in valid_clouds])
        all_depth_m = np.concatenate([c.source_depth_m for c in valid_clouds])
        all_valid = np.concatenate([c.valid_depth for c in valid_clouds])
        all_weights = np.concatenate([c.observation_weight for c in valid_clouds])
        all_agg = np.concatenate([c.aggregation_count for c in valid_clouds])

        has_normals = any(c.normals is not None for c in valid_clouds)
        all_normals = None
        if has_normals:
            normal_list = []
            for c in valid_clouds:
                if c.normals is not None:
                    normal_list.append(c.normals)
                else:
                    normal_list.append(np.zeros((len(c), 3), dtype=np.float32))
            all_normals = np.vstack(normal_list)

        any_aggregated = any(c.cf_flat is not None and c.cf_offsets is not None for c in valid_clouds)
        all_cf_flat = None
        all_cf_offsets = None
        if any_aggregated:
            flat_list = []
            offsets_list = []
            current_offset = 0
            for c in valid_clouds:
                if c.cf_flat is not None and c.cf_offsets is not None:
                    c_flat = c.cf_flat
                    c_counts = c.cf_offsets[1:] - c.cf_offsets[:-1]
                else:
                    c_flat = c.source_frame_id.astype(np.int32)
                    c_counts = np.ones(len(c), dtype=np.int64)
                flat_list.append(c_flat)
                c_offsets = np.zeros(len(c_counts) + 1, dtype=np.int64)
                np.cumsum(c_counts, out=c_offsets[1:])
                offsets_list.append(c_offsets[:-1] + current_offset)
                current_offset += len(c_flat)
            offsets_list.append(np.array([current_offset], dtype=np.int64))
            all_cf_flat = np.concatenate(flat_list).astype(np.int32)
            all_cf_offsets = np.concatenate(offsets_list).astype(np.int64)

        merged_frames_meta: dict[int, FrameMetadata] = {}
        for c in valid_clouds:
            merged_frames_meta.update(copy.deepcopy(c.frames_meta))

        return cls(
            points=all_points,
            colors=all_colors,
            source_frame_id=all_fid,
            source_pixel_uv=all_uv,
            source_depth_uv=all_depth_uv,
            source_depth_m=all_depth_m,
            valid_depth=all_valid,
            observation_weight=all_weights,
            aggregation_count=all_agg,
            normals=all_normals,
            cf_flat=all_cf_flat,
            cf_offsets=all_cf_offsets,
            frames_meta=merged_frames_meta,
            coordinate_frame=valid_clouds[0].coordinate_frame,
            transform_matrix=valid_clouds[0].transform_matrix.copy(),
        )

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Convert to standard Open3D PointCloud for visualization and geometry processing."""
        pcd = o3d.geometry.PointCloud()
        if not self.is_empty():
            pcd.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float64))
            if self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals.astype(np.float64))
        return pcd

    @classmethod
    def from_open3d(
        cls,
        pcd: o3d.geometry.PointCloud,
        frame_id: int = 0,
        intrinsics: o3d.camera.PinholeCameraIntrinsic | None = None,
        timestamp: float = 0.0,
    ) -> "IndexedPointCloud":
        """
        Wrap an existing Open3D PointCloud as an IndexedPointCloud with baseline provenance.
        """
        points = np.asarray(pcd.points, dtype=np.float32)
        n = len(points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
        else:
            colors = np.ones((n, 3), dtype=np.float32) * 0.7

        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

        source_frame_id = np.full(n, frame_id, dtype=np.int32)
        source_pixel_uv = np.full((n, 2), -1, dtype=np.int32)
        source_depth_m = points[:, 2].copy() if n > 0 else np.zeros(0, dtype=np.float32)

        frames_meta = {}
        if intrinsics is not None:
            frames_meta[frame_id] = FrameMetadata.from_o3d_intrinsics(
                intrinsics, frame_id=frame_id, timestamp=timestamp
            )

        return cls(
            points=points,
            colors=colors,
            source_frame_id=source_frame_id,
            source_pixel_uv=source_pixel_uv,
            source_depth_uv=source_pixel_uv.copy(),
            source_depth_m=source_depth_m,
            valid_depth=np.ones(n, dtype=bool),
            observation_weight=np.ones(n, dtype=np.float32),
            aggregation_count=np.ones(n, dtype=np.int32),
            normals=normals,
            cf_flat=None,
            cf_offsets=None,
            frames_meta=frames_meta,
        )

    def save_npz(self, filepath: str | Path) -> Path:
        """
        Save the complete point cloud and provenance to a compressed NPZ file.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta_dict = {
            fid: fm.to_dict() for fid, fm in self.frames_meta.items()
        }
        meta_json_str = json.dumps(meta_dict)

        save_dict = {
            "points": self.points,
            "colors": self.colors,
            "source_frame_id": self.source_frame_id,
            "source_pixel_uv": self.source_pixel_uv,
            "source_depth_uv": self.source_depth_uv,
            "source_depth_m": self.source_depth_m,
            "valid_depth": self.valid_depth,
            "observation_weight": self.observation_weight,
            "aggregation_count": self.aggregation_count,
            "transform_matrix": self.transform_matrix,
            "coordinate_frame": np.array(self.coordinate_frame),
            "meta_json": np.array(meta_json_str),
        }

        if self.normals is not None:
            save_dict["normals"] = self.normals

        # Store contributing_frame_ids as a compact flattened ragged array with offsets
        if self.cf_flat is not None and self.cf_offsets is not None:
            save_dict["cf_flat"] = self.cf_flat
            save_dict["cf_offsets"] = self.cf_offsets

        np.savez_compressed(out_path, **save_dict)
        return out_path

    @classmethod
    def load_npz(cls, filepath: str | Path) -> "IndexedPointCloud":
        """
        Load an IndexedPointCloud from a compressed NPZ archive.
        """
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"IndexedPointCloud NPZ file not found: {in_path}")

        data = np.load(in_path, allow_pickle=False)

        points = data["points"]
        colors = data["colors"]
        source_frame_id = data["source_frame_id"]
        source_pixel_uv = data["source_pixel_uv"]
        source_depth_uv = data["source_depth_uv"]
        source_depth_m = data["source_depth_m"]
        valid_depth = data["valid_depth"]
        observation_weight = data["observation_weight"]
        aggregation_count = data["aggregation_count"]
        transform_matrix = data["transform_matrix"]
        coordinate_frame = str(data["coordinate_frame"])

        normals = data["normals"] if "normals" in data else None

        cf_flat = data["cf_flat"] if "cf_flat" in data else None
        cf_offsets = data["cf_offsets"] if "cf_offsets" in data else None

        meta_dict = json.loads(str(data["meta_json"]))
        frames_meta = {
            int(fid): FrameMetadata.from_dict(fdict)
            for fid, fdict in meta_dict.items()
        }

        return cls(
            points=points,
            colors=colors,
            source_frame_id=source_frame_id,
            source_pixel_uv=source_pixel_uv,
            source_depth_uv=source_depth_uv,
            source_depth_m=source_depth_m,
            valid_depth=valid_depth,
            observation_weight=observation_weight,
            aggregation_count=aggregation_count,
            normals=normals,
            cf_flat=cf_flat,
            cf_offsets=cf_offsets,
            frames_meta=frames_meta,
            coordinate_frame=coordinate_frame,
            transform_matrix=transform_matrix,
        )


def create_indexed_pcd_from_rgbd(
    color: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic | np.ndarray | dict[str, float],
    depth_scale: float = 1.0,
    depth_min_m: float = 0.10,
    depth_max_m: float = 3.00,
    roi: tuple[int, int, int, int] | None = None,
    mask: np.ndarray | None = None,
    frame_id: int = 0,
    timestamp: float = 0.0,
    is_bgr: bool = True,
    coordinate_frame: str = "camera",
) -> IndexedPointCloud:
    """
    Vectorized back-projection of synchronized RGB-D arrays into an IndexedPointCloud.

    Derives 3D metric coordinates while preserving exact pixel provenance:
        Z = D(u, v) * depth_scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Parameters
    ----------
    color:
        HxWx3 uint8 color image array (BGR or RGB).
    depth_m:
        HxW metric depth array in metres (float32 or uint16 with depth_scale).
    intrinsics:
        Open3D PinholeCameraIntrinsic, 3x3 numpy array, or dictionary with fx, fy, cx, cy.
    depth_scale:
        Scale multiplier to convert depth_m to metres (1.0 if already float metres).
    depth_min_m:
        Minimum valid depth threshold in metres.
    depth_max_m:
        Maximum valid depth threshold in metres.
    roi:
        Optional (x, y, w, h) bounding box to crop valid points in pixel space.
    mask:
        Optional HxW boolean or uint8 foreground mask.
    frame_id:
        Zero-based frame sequence index.
    timestamp:
        Capture timestamp in seconds.
    is_bgr:
        True if color array is BGR (OpenCV format); False if RGB.
    coordinate_frame:
        Name of reference coordinate system.

    Returns
    -------
    IndexedPointCloud
        Fully attributed point cloud preserving original (u, v) pixel coordinates.
    """
    # 1. Parse Intrinsics
    if isinstance(intrinsics, o3d.camera.PinholeCameraIntrinsic):
        mat = np.array(intrinsics.intrinsic_matrix, dtype=np.float64)
        img_w, img_h = intrinsics.width, intrinsics.height
    elif isinstance(intrinsics, np.ndarray):
        mat = np.asarray(intrinsics, dtype=np.float64)
        img_h, img_w = depth_m.shape[:2]
    elif isinstance(intrinsics, dict):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        img_w = int(intrinsics.get("width", depth_m.shape[1]))
        img_h = int(intrinsics.get("height", depth_m.shape[0]))
        mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise TypeError(f"Unsupported intrinsics type: {type(intrinsics)}")

    fx = float(mat[0, 0])
    fy = float(mat[1, 1])
    cx = float(mat[0, 2])
    cy = float(mat[1, 2])

    frame_meta = FrameMetadata(
        frame_id=frame_id,
        timestamp=timestamp,
        intrinsics=mat,
        width=img_w,
        height=img_h,
        depth_scale=depth_scale,
        coordinate_frame=coordinate_frame,
    )

    # 2. Build Depth Validity Mask
    depth_f32 = depth_m.astype(np.float32) * depth_scale
    valid = np.isfinite(depth_f32) & (depth_f32 >= depth_min_m) & (depth_f32 <= depth_max_m)

    if roi is not None:
        rx, ry, rw, rh = roi
        roi_mask = np.zeros_like(valid, dtype=bool)
        h, w = depth_f32.shape[:2]
        r_y0 = max(0, ry)
        r_y1 = min(h, ry + rh)
        r_x0 = max(0, rx)
        r_x1 = min(w, rx + rw)
        if r_y1 > r_y0 and r_x1 > r_x0:
            roi_mask[r_y0:r_y1, r_x0:r_x1] = True
        valid = valid & roi_mask

    if mask is not None:
        valid = valid & (mask > 0)

    # 3. Vectorized Back-Projection
    # v is row index (y), u is column index (x)
    v_indices, u_indices = np.nonzero(valid)
    num_pts = len(v_indices)

    if num_pts == 0:
        return IndexedPointCloud(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            source_frame_id=np.zeros(0, dtype=np.int32),
            source_pixel_uv=np.zeros((0, 2), dtype=np.int32),
            source_depth_uv=np.zeros((0, 2), dtype=np.int32),
            source_depth_m=np.zeros(0, dtype=np.float32),
            valid_depth=np.zeros(0, dtype=bool),
            observation_weight=np.zeros(0, dtype=np.float32),
            aggregation_count=np.zeros(0, dtype=np.int32),
            frames_meta={frame_id: frame_meta},
            coordinate_frame=coordinate_frame,
        )

    z = depth_f32[v_indices, u_indices]
    u_f = u_indices.astype(np.float32)
    v_f = v_indices.astype(np.float32)

    x = (u_f - cx) * z / fx
    y = (v_f - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # 4. Color Extraction
    if is_bgr:
        # BGR to RGB normalized to [0, 1]
        colors = color[v_indices, u_indices, ::-1].astype(np.float32) / 255.0
    else:
        colors = color[v_indices, u_indices].astype(np.float32) / 255.0

    # 5. Provenance Arrays
    source_frame_id = np.full(num_pts, frame_id, dtype=np.int32)
    source_pixel_uv = np.stack([u_indices, v_indices], axis=-1).astype(np.int32)
    source_depth_uv = source_pixel_uv.copy()
    source_depth_m = z.copy()
    valid_depth = np.ones(num_pts, dtype=bool)
    observation_weight = np.ones(num_pts, dtype=np.float32)
    aggregation_count = np.ones(num_pts, dtype=np.int32)

    return IndexedPointCloud(
        points=points,
        colors=colors,
        source_frame_id=source_frame_id,
        source_pixel_uv=source_pixel_uv,
        source_depth_uv=source_depth_uv,
        source_depth_m=source_depth_m,
        valid_depth=valid_depth,
        observation_weight=observation_weight,
        aggregation_count=aggregation_count,
        normals=None,
        cf_flat=None,
        cf_offsets=None,
        frames_meta={frame_id: frame_meta},
        coordinate_frame=coordinate_frame,
    )
