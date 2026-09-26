"""
Thin wrapper around pyrealsense2 for the RealSense D455(f).

This is the ONLY file in the project that imports pyrealsense2.  Everything
RealSense-specific — stream configuration, post-processing filters, depth
scaling, intrinsic calibration — lives here.  The rest of the pipeline only
ever deals with plain numpy arrays and Open3D point clouds, which means:

  * The sensor can be swapped (e.g. to an Azure Kinect or a ZED) by
    replacing just this file.
  * preprocessing.py, registration.py, reconstruction.py, and targets/ never
    need touching when hardware changes.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "pyrealsense2 is not installed. "
        "On the machine with the D455f plugged in, run:  pip install pyrealsense2"
    ) from exc

from config import CameraConfig, DepthVisConfig



class RealSenseCamera:
    """
    Manages a RealSense D455(f) depth+colour stream.

    Typical usage::

        with RealSenseCamera(cfg).start() as cam:
            color, depth = cam.read()
            pcd = cam.to_point_cloud(color, depth)

    Or without the context manager::

        cam = RealSenseCamera(cfg).start()
        try:
            ...
        finally:
            cam.stop()
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg

        # ---- pipeline & stream configuration ----------------------------
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
        )
        self.rs_config.enable_stream(
            rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
        )

        # Align depth onto the colour frame so both are in the same pixel
        # grid (required for accurate RGBD point-cloud creation).
        self.align = rs.align(rs.stream.color)

        # ---- post-processing filters -------------------------------------
        # Spatial and temporal filters work in *disparity* space (inverse
        # depth), not depth space — they are designed and tuned that way.
        # The pair of disparity transforms bracket those two filters:
        #   depth → disparity → spatial → temporal → depth → hole-filling

        self._decimation = rs.decimation_filter()
        self._decimation.set_option(
            rs.option.filter_magnitude, cfg.decimation_magnitude
        )
        self._to_disparity   = rs.disparity_transform(True)   # depth → disparity
        self._spatial        = rs.spatial_filter()
        self._temporal       = rs.temporal_filter()
        self._to_depth       = rs.disparity_transform(False)  # disparity → depth
        self._hole_filling   = rs.hole_filling_filter()

        # ---- SDK colorizer (visualization ONLY — never touches raw depth) --
        # Configured properly in start() once we know the DepthVisConfig.
        self._colorizer = rs.colorizer()

        # ---- state initialised in start() -------------------------------
        self.depth_scale: float | None = None
        self.intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic | None = None
        self._profile = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "RealSenseCamera":
        """
        Start the RealSense pipeline and populate calibration attributes.

        Returns ``self`` so the call can be chained::

            cam = RealSenseCamera(cfg).start()

        Returns
        -------
        RealSenseCamera
        """
        self._profile = self.pipeline.start(self.rs_config)

        # ---- depth scale (converts raw u16 units to metres) -------------
        depth_sensor = (
            self._profile
            .get_device()
            .first_depth_sensor()
        )
        self.depth_scale = depth_sensor.get_depth_scale()

        # ---- colour-stream intrinsics → Open3D format -------------------
        stream_profile = (
            self._profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        intr = stream_profile.get_intrinsics()
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
        )

        # ---- configure SDK colorizer -------------------------------------
        vis = self.cfg.depth_vis
        self._colorizer.set_option(rs.option.color_scheme,
                                   float(vis.color_scheme))
        self._colorizer.set_option(rs.option.min_distance,
                                   vis.visual_min_m)
        self._colorizer.set_option(rs.option.max_distance,
                                   vis.visual_max_m)
        self._colorizer.set_option(rs.option.histogram_equalization_enabled,
                                   1.0 if vis.histogram_equalization else 0.0)

        return self

    def stop(self) -> None:
        """Stop the RealSense pipeline and release device resources."""
        self.pipeline.stop()

    # Context-manager support so the camera can be used in a `with` block.
    def __enter__(self) -> "RealSenseCamera":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_depth(self, depth_frame: rs.depth_frame) -> rs.frame:
        """
        Apply the configured post-processing filter chain to a depth frame.

        Filter order:
          decimation (spatial resolution ↓) →
          depth-to-disparity →
          spatial filter (edge-preserving smooth) →
          temporal filter (cross-frame averaging) →
          disparity-to-depth →
          hole-filling

        Each filter is only applied if its corresponding flag in
        :class:`~config.CameraConfig` is ``True``.

        Parameters
        ----------
        depth_frame:
            Raw depth frame from the RealSense SDK.

        Returns
        -------
        rs.frame
            Filtered depth frame.
        """
        frame = depth_frame

        if self.cfg.use_decimation:
            frame = self._decimation.process(frame)

        # Disparity transform is always applied around spatial/temporal
        # filters because those filters require disparity-space input.
        frame = self._to_disparity.process(frame)

        if self.cfg.use_spatial_filter:
            frame = self._spatial.process(frame)

        if self.cfg.use_temporal_filter:
            frame = self._temporal.process(frame)

        frame = self._to_depth.process(frame)

        if self.cfg.use_hole_filling:
            frame = self._hole_filling.process(frame)

        return frame

    # ------------------------------------------------------------------
    # Public frame API
    # ------------------------------------------------------------------

    def read(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Capture one aligned colour + depth frame pair.

        Applies all configured post-processing filters.  Depth decimation can
        shrink the depth frame's resolution below the colour frame's; if that
        happens the depth image is upsampled with nearest-neighbour
        interpolation so both arrays share the same spatial shape.

        Returns
        -------
        color_bgr:
            HxWx3 uint8 array in BGR order, or ``None`` on a dropped frame.
        depth_m:
            HxW float32 array of depth values in metres (0 where invalid),
            or ``None`` on a dropped frame.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_frame = self._filter_depth(depth_frame)

        # ---- convert to numpy ------------------------------------------
        depth_m = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * self.depth_scale
        )
        color_bgr = np.asanyarray(color_frame.get_data())

        # Decimation shrinks the depth image; resize it back up to the colour
        # resolution so to_point_cloud() receives matching-shaped arrays.
        if depth_m.shape[:2] != color_bgr.shape[:2]:
            import cv2  # local import — only needed when shapes diverge
            h, w = color_bgr.shape[:2]
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        return color_bgr, depth_m

    def colorize_depth(
        self,
        depth_frame: "rs.frame",
    ) -> np.ndarray:
        """
        Colorize a raw (filtered) depth frame using the RealSense SDK colorizer.

        This is the ONLY method that should produce a colored depth image.
        The result is intended exclusively for human visualization — it must
        never be used as actual depth data.

        The SDK colorizer maps the configured visual_min_m → visual_max_m
        range onto the full color gradient.  Pixels with no valid depth
        (value 0) are rendered black by the SDK, clearly distinguishable
        from valid near-depth pixels.

        Parameters
        ----------
        depth_frame:
            A raw or filtered rs.depth_frame (BEFORE converting to numpy).
            Must still be an SDK frame object, not a numpy array.

        Returns
        -------
        np.ndarray
            HxWx3 uint8 BGR array suitable for cv2.imshow().
        """
        # rs.colorizer produces an RGB frame; convert to BGR for OpenCV.
        colorized_rgb = np.asanyarray(
            self._colorizer.colorize(depth_frame).get_data()
        )
        return colorized_rgb[:, :, ::-1]  # RGB → BGR

    def read_with_colorized(
        self,
    ) -> tuple[
        "np.ndarray | None",
        "np.ndarray | None",
        "np.ndarray | None",
    ]:
        """
        Capture one aligned frame and return raw data PLUS a visualization image.

        Strictly separates the two data paths:

        RAW DEPTH (depth_m):
            float32 numpy array in metres.
            Used for XYZ calculation, point cloud, measurement.
            Never modified by colorization.

        COLORIZED DEPTH (depth_color_bgr):
            uint8 BGR numpy array, produced by the RealSense SDK colorizer.
            Used ONLY for the live preview window.
            Never fed into point-cloud or measurement code.

        Returns
        -------
        color_bgr:
            HxWx3 uint8 colour image, or None on a dropped frame.
        depth_m:
            HxW float32 depth in metres, or None on a dropped frame.
        depth_color_bgr:
            HxWx3 uint8 SDK-colorized depth image, or None on a dropped frame.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        # Apply post-processing filters to the raw depth frame.
        depth_frame = self._filter_depth(depth_frame)

        # --- VISUALIZATION PATH (SDK colorizer, BEFORE converting to numpy) --
        # Must be done on the rs.frame object, not the numpy array.
        depth_color_bgr = self.colorize_depth(depth_frame)

        # --- RAW DEPTH PATH (numpy, for XYZ / point cloud) ------------------
        depth_m = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * self.depth_scale
        )
        color_bgr = np.asanyarray(color_frame.get_data())

        # Decimation can shrink the depth image; resize both arrays if needed.
        if depth_m.shape[:2] != color_bgr.shape[:2]:
            import cv2  # local import — only needed when shapes diverge
            h, w = color_bgr.shape[:2]
            depth_m        = cv2.resize(depth_m,        (w, h),
                                        interpolation=cv2.INTER_NEAREST)
            depth_color_bgr = cv2.resize(depth_color_bgr, (w, h),
                                         interpolation=cv2.INTER_NEAREST)

        return color_bgr, depth_m, depth_color_bgr

    def to_point_cloud(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Convert a colour + depth frame pair into an Open3D point cloud.

        Parameters
        ----------
        color_bgr:
            HxWx3 uint8 BGR image from :meth:`read`.
        depth_m:
            HxW float32 depth image in metres from :meth:`read`.

        Returns
        -------
        o3d.geometry.PointCloud
            Coloured point cloud in the camera's coordinate frame, flipped
            to Open3D's display convention (Y-up, Z-toward-viewer).
        """
        # BGR → RGB (Open3D expects RGB)
        color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])

        # Clip depth to the usable range; zeroed pixels become "no data"
        # and are excluded from the RGBD image by depth_trunc.
        depth_clipped = depth_m.copy()
        depth_clipped[depth_clipped < self.cfg.depth_min_m] = 0.0
        depth_clipped[depth_clipped > self.cfg.depth_max_m] = 0.0

        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_clipped)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,                    # already in metres
            depth_trunc=self.cfg.depth_max_m,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsics_o3d,
        )

        # Coordinate system: RealSense native convention is kept.
        #   +X = right,  +Y = down,  +Z = forward  (units: metres)
        # No flip is applied here so the point cloud remains in the camera
        # frame that preprocessing, registration, and reconstruction expect.
        return pcd
