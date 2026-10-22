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
class LandmarkConfig:
    """Settings for facial landmark depth sampling and quality scoring."""
    max_depth_variance_m2: float = 1e-5        # ~3mm std depth variance threshold (m^2)
    min_valid_ratio: float = 0.5               # minimum non-zero depth pixel ratio in patch
    max_surface_step_m: float = 0.03           # max depth step (30mm) before treating as separate surface


@dataclass
class HeadCaptureConfig:
    """Real-time feedback thresholds during human/head capture."""
    enabled: bool = True                       # enable real-time landmark tracking during capture
    min_landmarks_for_lock: int = 4            # minimum landmarks detected to consider tracking locked
    min_frames_for_low_lock: int = 3           # frames without good lock before yellow warning
    max_frames_without_lock: int = 8           # hard cap frames without lock before red coverage gap warning
    max_landmark_px_motion: float = 40.0       # max average pixel displacement between frames before red too-fast warning


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

    # Depth range to keep (metres).
    # D455 Min-Z is ~0.52 m at 1280x720 (Intel spec) and somewhat less at 848x480.
    # The ideal range starts at 0.6 m. Hold the object >= 0.5-0.6 m from the camera for reliable depth.
    depth_min_m: float = 0.40
    depth_max_m: float = 1.20

    # Auto-capture settings
    auto_capture: bool = True
    capture_interval_s: float = 3.0   # how often a frame is automatically captured for box scans (s)
    human_capture_interval_s: float = 2.0 # capture interval for human head/body scans (s)
    skip_near_duplicate_frames: bool = True
    duplicate_depth_diff_threshold_m: float = 0.01   # min mean abs depth diff (m) to avoid duplicate capture

    # Depth visualisation settings (preview window only — no effect on data).
    depth_vis: DepthVisConfig = field(default_factory=DepthVisConfig)

    # Real-time feedback settings for human/head capture
    head_capture: HeadCaptureConfig = field(default_factory=HeadCaptureConfig)

    # High-quality capture mode for human face/head scans
    human_face_quality_mode: bool = False

    def resolve_defaults(self, target_name: str = "box") -> "CameraConfig":
        """Apply human face quality settings when enabled and target is head or body."""
        if self.human_face_quality_mode and target_name in ("head", "body"):
            self.use_decimation = False
            self.use_hole_filling = False
            self.width = 1280
            self.height = 720
        return self


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
    min_accept_fitness: float | None = None

    # Fast-path ICP threshold: if identity-seeded ICP achieves this fitness,
    # skip expensive FPFH+RANSAC global registration.
    icp_only_fitness_threshold: float = 0.60

    # Registration strategy mode:
    # "pointcloud" : legacy geometric FPFH+RANSAC -> ICP (optimal for box)
    # "rgbd_human" : calibrated RGB-D odometry -> colored ICP -> quality gate -> pose graph (optimal for human face/head/body)
    registration_mode: str | None = None

    # Facial landmark guidance (for "rgbd_human" mode)
    use_landmarks: bool = True                 # seed pairwise registration with 3D facial landmarks when available
    min_seed_dense_fitness: float = 0.30       # minimum dense-surface fitness required to accept landmark Kabsch seed
    max_depth_variance_m2: float = 1e-5        # max depth variance threshold (m^2)
    landmark_cfg: LandmarkConfig = field(default_factory=LandmarkConfig)
    enable_landmark_fpfh_crosscheck: bool = True               # independent FPFH+RANSAC geometric cross-check
    max_landmark_fpfh_disagreement_trans_m: float = 0.02       # max translation disagreement between landmark and FPFH (20 mm)
    max_landmark_fpfh_disagreement_rot_deg: float = 10.0       # max rotation angle disagreement between landmark and FPFH (10 deg)
    enable_landmark_face_fitness_check: bool = True            # localized face-region fitness evaluation
    face_crop_radius_m: float = 0.08                           # radius around nose tip/face anchor for localized fitness evaluation (80 mm)
    min_landmark_face_fitness: float = 0.25                    # minimum local face-region fitness required to accept landmark seed
    face_fitness_dist_m: float = 0.005                         # correspondence search distance for localized face fitness (5 mm)

    # Landmark geometric conditioning gates (Problem 4)
    enable_landmark_conditioning_check: bool = True            # evaluate matched landmark geometry before Kabsch
    landmark_min_unique_dist_m: float = 0.005                  # min separation between distinct landmarks (5 mm)
    landmark_min_spread_m: float = 0.025                       # min maximum-pairwise-distance / diameter (25 mm)
    landmark_min_secondary_spread_m: float = 0.005             # min RMS spread along 2nd principal axis (5 mm)
    landmark_max_collinear_ratio: float = 10.0                 # max ratio of 1st to 2nd singular value (sigma_1 / sigma_2)

    # Multiway pose-graph registration settings
    use_multiway: bool = False  # sequential is the safer default; set True to use the (corrected, gated) pose graph
    loop_closure_search_window: int = 4   # how many frames back, beyond the immediate previous frame, to also test for a good registration — catches cases where frame i overlaps with frame i-3 or i-4 even though it wasn't captured immediately after it
    pose_graph_edge_prune_threshold: float = 0.25  # Open3D's own default, controls how aggressively weak/inconsistent edges get discarded during global optimization

    # RGB-D Odometry settings (for "rgbd_human" mode)
    rgbd_odometry_method: str = "hybrid"   # "hybrid" (depth + color) or "color"

    # Colored ICP refinement settings (for "rgbd_human" mode)
    use_colored_icp: bool = True
    colored_icp_max_dist_m: float = 0.02    # 20 mm correspondence search radius
    colored_icp_lambda_geom: float = 0.968  # Open3D recommended balance (0.968 geometric, 0.032 photometric)
    colored_icp_max_iterations: int = 50

    # Motion sanity / physical plausibility quality gates
    max_rotation_deg_per_frame: float = 25.0   # reject inter-frame rotation jumps > 25 deg
    max_translation_m_per_frame: float = 0.15  # reject inter-frame translation jumps > 150 mm
    max_pose_jump_translation_m: float = 0.15  # reject pose jump translation > 0.15m from previous accepted frame
    max_pose_jump_rotation_deg: float = 45.0   # reject pose jump rotation > 45deg from previous accepted frame
    max_pose_jump_step_cap: int = 6            # maximum frame-skip multiplier cap for pose-jump bounds
    min_odometry_fitness: float = 0.35         # minimum inlier correspondence ratio for odometry
    min_colored_icp_fitness: float | None = None
    min_loop_closure_fitness: float = 0.65     # minimum inlier fitness ratio for non-consecutive loop-closure edges (prevents weak cross-profile tension)
    max_colored_icp_rmse_m: float = 0.008      # maximum allowable inlier RMSE (8 mm)
    max_pair_centroid_dist_m: float = 0.50     # maximum centroid distance between candidate source and target frames to attempt Colored ICP
    search_accepted_window: int = 4            # search up to 4 previously accepted frames when consecutive fails

    # Bounded registration recovery settings (Problem 7A)
    enable_bounded_recovery: bool = True               # enable bounded bridge buffer recovery
    max_bridge_buffer_size: int = 2                    # maximum consecutive provisional bridge frames
    max_bridge_crosscheck_trans_m: float = 0.020       # max translation disagreement for independent validation (20 mm)
    max_bridge_crosscheck_rot_deg: float = 10.0        # max rotation disagreement for independent validation (10 deg)

    # Fusion eligibility thresholds (stricter secondary gate for Stage 4 fusion)
    min_fitness_for_fusion: float = 0.60                  # frames with fitness < 0.60 keep pose in graph but are excluded from dense Stage 4 fusion
    max_landmark_residual_for_fusion_mm: float = 5.0      # landmark frames with residual > 5.0mm excluded from dense Stage 4 fusion
    min_face_pair_fitness_for_fusion: float = 0.75        # stricter fusion fitness threshold for upper (face/head) pairs
    max_face_pair_rmse_for_fusion_mm: float = 4.0         # stricter fusion RMSE threshold (mm) for upper (face/head) pairs
    face_region_center_y_threshold_m: float = 0.0         # Y threshold (m): bbox_center_y <= threshold is upper (face/head)

    # Post-fusion artifact cleanup for head/body mode (linearity-aware streak removal)
    enable_streak_removal: bool = True                 # enable streak/speckle post-fusion cleanup for human scans
    streak_linearity_ratio_threshold: float = 8.0      # PCA eigenvalue ratio above which a neighborhood is considered "linear" (streak)
    streak_removal_radius_mult: float = 2.5            # radius-pass radius = this * fusion voxel_size (2 mm) ≈ 5 mm

    # Visualisation toggles
    show_registration_pairs: bool = False      # display before/after pairwise alignment window
    show_camera_trajectory: bool = False       # display 3D camera trajectory with jump diagnostics

    def resolve_defaults(self, target_name: str = "box") -> "RegistrationConfig":
        """Apply box/head/body-specific values when fields are still None."""
        if target_name in ("head", "body") or self.registration_mode == "rgbd_human":
            if self.min_accept_fitness is None:
                self.min_accept_fitness = 0.50
            if self.min_colored_icp_fitness is None:
                self.min_colored_icp_fitness = 0.50
            if self.registration_mode is None:
                self.registration_mode = "rgbd_human"
        else:
            if self.min_accept_fitness is None:
                self.min_accept_fitness = 0.35
            if self.min_colored_icp_fitness is None:
                self.min_colored_icp_fitness = 0.45
            if self.registration_mode is None:
                self.registration_mode = "pointcloud"
        return self




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

    # Physical plausibility bounds for isolated object clusters (metres)
    expected_min_size_m: float | None = None   # smallest plausible dimension
    expected_max_size_m: float | None = None   # largest plausible dimension (well above object, below room clutter)
    min_object_points: int | None = None        # minimum points needed for a valid object capture
    max_centroid_dev_m: float | None = None   # reject frames whose object centroid is farther than this from the median centroid
    use_manual_roi: bool = False       # interactively select 2D pixel ROI on first frame for cropped point cloud isolation

    def __post_init__(self) -> None:
        """Set target-specific physical bounds and rejection tolerances if still None (sentinel)."""
        if self.name == "head":
            if self.expected_min_size_m is None:
                self.expected_min_size_m = 0.12
            if self.expected_max_size_m is None:
                self.expected_max_size_m = 0.55
            if self.min_object_points is None:
                self.min_object_points = 200
            if self.max_centroid_dev_m is None:
                self.max_centroid_dev_m = 0.25
        elif self.name == "body":
            if self.expected_min_size_m is None:
                self.expected_min_size_m = 0.20
            if self.expected_max_size_m is None:
                self.expected_max_size_m = 1.20
            if self.min_object_points is None:
                self.min_object_points = 300
            if self.max_centroid_dev_m is None:
                self.max_centroid_dev_m = 0.35
        else:
            if self.expected_min_size_m is None:
                self.expected_min_size_m = 0.03
            if self.expected_max_size_m is None:
                self.expected_max_size_m = 0.60
            if self.min_object_points is None:
                self.min_object_points = 150
            if self.max_centroid_dev_m is None:
                self.max_centroid_dev_m = 0.15




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
    head_capture: HeadCaptureConfig = field(default_factory=HeadCaptureConfig)

    # Developer / debug toggles
    show_stage_windows: bool = True   # pop up an Open3D window after each stage
    save_intermediate: bool = True    # write .ply / .json to output_dir after each stage

    # Reconstruction fusion method: "pointcloud" (concatenation) or "tsdf" (volumetric TSDF integration)
    fusion_method: str = "pointcloud"
    tsdf_voxel_length_m: float | None = None   # default 0.004 (box) / 0.002 (human)
    tsdf_trunc_m: float | None = None          # default 0.02 (box) / 4.0 * voxel_length (human)

    # Diagnostic full-orbit check
    attempt_loop_closure: bool = False   # check last vs first frame alignment after registration

    # Ground-truth reference dimensions for accuracy comparison (metres, sorted descending)
    reference_dims_m: list[float] | None = None

    # Display-only downsampling settings for interactive preview windows (metres).
    # Coarser display resolution prevents windows from looking overly dense/cluttered,
    # while all saved .ply files and dimensional measurements remain at 100% full resolution.
    stage5_display_voxel_size_m: float = 0.006      # 6 mm display resolution for Stage 5 window
    stage3_4_display_voxel_size_m: float = 0.0025    # 2.5 mm display resolution for Stage 3/4 preview windows (loosened from 6mm)

    def __post_init__(self) -> None:
        # Ensure output_dir is always a Path, even if a plain string was passed.
        self.output_dir = Path(self.output_dir)

        # Resolve registration target-specific defaults if still None
        self.registration.resolve_defaults(self.target.name)

        # Resolve camera target-specific defaults (e.g. human face quality mode)
        self.camera.resolve_defaults(self.target.name)

        # Human head/body mode: apply finer TSDF voxel resolution if still None
        if self.target.name in ("head", "body") or self.registration.registration_mode == "rgbd_human":
            if self.tsdf_voxel_length_m is None:
                self.tsdf_voxel_length_m = 0.002
            if self.tsdf_trunc_m is None:
                self.tsdf_trunc_m = 4.0 * self.tsdf_voxel_length_m
        else:
            if self.tsdf_voxel_length_m is None:
                self.tsdf_voxel_length_m = 0.004
            if self.tsdf_trunc_m is None:
                self.tsdf_trunc_m = 0.02

        self.camera.head_capture = self.head_capture

