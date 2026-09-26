"""
Central configuration for the realsense_measure pipeline.

All tunable parameters live here, grouped by pipeline stage.  This design
means that adding a new target type (e.g. "head", "body") never requires
touching pipeline.py, preprocessing.py, registration.py, or any other
stage module.  The only two things needed are:

  1. Add any new fields to the relevant dataclass(es) below
     (or create a new one and attach it to PipelineConfig).
  2. Add a new Target implementation under targets/.

Everything else — cameras, filters, registration thresholds, output paths —
is already parameterised here and flows through PipelineConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Stage: Camera / acquisition
# ---------------------------------------------------------------------------

@dataclass
class DepthVisConfig:
    """
    Depth-visualisation settings for the live capture preview.

    These parameters control ONLY what is shown on screen — they have
    absolutely no effect on the raw depth values used for point-cloud
    creation or measurement.

    RealSense colorizer color_scheme values
    ----------------------------------------
    0  Jet          (blue → green → yellow → red)
    1  Classic      (white → black)
    2  WhiteToBlack (white → black, reversed)
    3  BlackToWhite (black → white)
    4  Bio          (green bio-style gradient)
    5  Cold         (blue/purple cold gradient)
    6  Warm         (warm red/orange gradient)
    7  Quantized    (8-level discrete palette)
    8  Pattern      (cross-hatch pattern)
    """

    # Metric range mapped onto the full color gradient.
    # Anything outside this range is clamped to the nearest end color.
    visual_min_m: float = 0.15   # near end of gradient (matches depth_min_m)
    visual_max_m: float = 3.00   # far end  — wider than depth_max_m so the
                                  # full scene is visible in the live view

    # RealSense colorizer color scheme index (see table above).
    color_scheme: int = 0         # 0 = Jet (blue-near → red-far)

    # Enable the SDK's built-in histogram equalization.
    # Improves contrast in scenes where most depth values cluster in a
    # narrow band (e.g. a flat wall filling most of the frame).
    histogram_equalization: bool = True


@dataclass
class CameraConfig:
    """RealSense D455(f) stream and post-processing filter settings."""

    # Stream resolution & frame rate
    width: int = 848
    height: int = 480
    fps: int = 30

    # Post-processing filters (applied in order: decimation → spatial →
    # temporal → hole-filling)
    use_decimation: bool = True
    decimation_magnitude: int = 2     # halves spatial resolution ÷2

    use_spatial_filter: bool = True   # edge-preserving spatial smooth
    use_temporal_filter: bool = True  # temporal averaging across frames
    use_hole_filling: bool = True     # fill small depth voids

    # Depth range to keep (metres).  The D455 is unreliable below ~15 cm;
    # 1.2 m clips far background before it pollutes the point cloud.
    depth_min_m: float = 0.15
    depth_max_m: float = 1.20

    # Auto-capture settings
    auto_capture: bool = True
    capture_interval_s: float = 3.0   # how often a frame is automatically captured (s)
    skip_near_duplicate_frames: bool = True
    duplicate_depth_diff_threshold_m: float = 0.01   # min mean abs depth diff (m) to avoid duplicate capture

    # Depth visualisation settings (preview window only — no effect on data).
    depth_vis: DepthVisConfig = field(default_factory=DepthVisConfig)


# ---------------------------------------------------------------------------
# Stage: Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Point-cloud cleaning and segmentation parameters."""

    # Voxel down-sampling: 4 mm grid keeps enough detail for dimensional
    # measurement while keeping computation tractable.
    voxel_size_m: float = 0.004

    # Statistical outlier removal
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 1.5

    # RANSAC plane segmentation (table/floor removal)
    plane_dist_threshold_m: float = 0.006
    plane_ransac_n: int = 3
    plane_num_iterations: int = 2_000

    # DBSCAN clustering (isolate the target object)
    cluster_eps_m: float = 0.02
    cluster_min_points: int = 60


# ---------------------------------------------------------------------------
# Stage: Registration
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """Multi-view point-cloud registration (FPFH + RANSAC → ICP) settings."""

    # Voxel size used when computing FPFH features.
    # Set to 3 mm (denser points feeding feature extraction and matching —
    # trades some speed for significantly better correspondence counts on low-texture objects).
    voxel_size_m: float = 0.003

    # FPFH search radius multipliers (relative to voxel_size_m)
    fpfh_radius_normal_mult: float = 2.0
    fpfh_radius_feature_mult: float = 8.0   # 8x voxel size captures wider surface context

    # RANSAC global registration
    ransac_distance_mult: float = 1.5       # × voxel_size_m → correspondence dist
    ransac_max_iter: int = 4_000_000
    ransac_confidence: float = 0.999

    # ICP refinement
    icp_max_dist_m: float = 0.01            # max correspondence distance
    icp_max_iterations: int = 100

    # Acceptance threshold: frames with fitness below this are considered
    # unreliable, excluded from fusion, and not merged into the growing reference.
    min_accept_fitness: float = 0.35

    # Fast-path ICP threshold: if identity-seeded ICP achieves this fitness,
    # skip expensive FPFH+RANSAC global registration.
    icp_only_fitness_threshold: float = 0.60


# ---------------------------------------------------------------------------
# Stage: Target
# ---------------------------------------------------------------------------

@dataclass
class TargetConfig:
    """Which object type to measure and sanity-check bounds."""

    # Selects the Target implementation loaded by the pipeline.
    # Supported values: "box" | "head" | "body"
    name: str = "box"

    # Reject any fitted dimension smaller than this (likely a degenerate fit).
    min_dimension_m: float = 0.02


# ---------------------------------------------------------------------------
# Top-level: PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Master configuration object passed into every pipeline stage.

    Instantiate with defaults::

        cfg = PipelineConfig()

    Or override individual fields::

        cfg = PipelineConfig(
            camera=CameraConfig(fps=15),
            target=TargetConfig(name="head"),
            show_stage_windows=False,
        )
    """

    output_dir: Path = field(default_factory=lambda: Path("scan_output"))

    # Per-stage configs — constructed with their own defaults automatically.
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Developer / debug toggles
    show_stage_windows: bool = True   # pop up an Open3D window after each stage
    save_intermediate: bool = True    # write .ply / .json to output_dir after each stage

    # Diagnostic full-orbit check
    attempt_loop_closure: bool = False   # check last vs first frame alignment after registration

    def __post_init__(self) -> None:
        # Ensure output_dir is always a Path, even if a plain string was passed.
        self.output_dir = Path(self.output_dir)
