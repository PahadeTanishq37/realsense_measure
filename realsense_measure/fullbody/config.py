"""
realsense_measure/fullbody/config.py

Configuration parameters for full-body scanning, alignment, fusion,
and anthropometric measurement.
All thresholds are defined here; modules must not use hard-coded magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FullBodyConfig:
    """
    Central configuration for the fullbody reconstruction and measurement pipeline.
    """
    # -------------------------------------------------------------------------
    # Protocol & Stops
    # -------------------------------------------------------------------------
    num_stops: int = 8
    expected_step_deg: float = 45.0
    turn_direction: str = "cw"                   # "cw" (clockwise from above) or "ccw"
    camera_portrait_rotation: str = "cw"         # "cw" (90 deg clockwise) or "ccw" (90 deg counter-clockwise)
    max_angle_deviation_deg: float = 25.0
    burst_frame_count: int = 30
    camera_height_m: float = 0.90
    camera_distance_m: float = 1.30

    # -------------------------------------------------------------------------
    # Burst processing
    # -------------------------------------------------------------------------
    burst_min_valid_frac: float = 0.50
    flying_pixel_depth_gradient_threshold_m: float = 0.04
    hole_fill_max_radius_px: int = 3
    temporal_std_outlier_sigma: float = 3.0
    depth_min_m: float = 0.40
    depth_max_m: float = 2.60

    # -------------------------------------------------------------------------
    # Floor detection
    # -------------------------------------------------------------------------
    floor_lower_image_fraction: float = 0.35  # bottom 35% of image rows
    floor_distance_threshold_m: float = 0.015  # 15 mm
    floor_ransac_iterations: int = 2000
    floor_normal_min_up_alignment: float = 0.70  # cos(45 deg) approx
    floor_max_angle_diff_deg: float = 4.0        # max normal discrepancy across stops

    # -------------------------------------------------------------------------
    # Confidence weighting
    # -------------------------------------------------------------------------
    min_confidence_threshold: float = 0.08
    incidence_angle_power: float = 2.0
    edge_distance_falloff_px: float = 8.0  # increased from 5 → 8 px: makes ring coverage robust to sensor noise at body edges
    noise_sigma_base_m: float = 0.001            # 1 mm at 0.6 m
    noise_reference_distance_m: float = 0.60

    # -------------------------------------------------------------------------
    # 3D Keypoints & Skeletal Estimation
    # -------------------------------------------------------------------------
    keypoint_window_size_px: int = 5
    keypoint_min_confidence: float = 0.40
    keypoint_min_visibility: float = 0.50
    keypoint_depth_mad_scale: float = 2.5

    # -------------------------------------------------------------------------
    # Body Part Segmentation (Capsules)
    # -------------------------------------------------------------------------
    capsule_radius_head_m: float = 0.13
    capsule_radius_torso_m: float = 0.22
    capsule_radius_pelvis_m: float = 0.18
    capsule_radius_upper_arm_m: float = 0.09
    capsule_radius_forearm_m: float = 0.07
    capsule_radius_thigh_m: float = 0.13
    capsule_radius_shank_m: float = 0.09

    # -------------------------------------------------------------------------
    # Multi-Stop Alignment
    # -------------------------------------------------------------------------
    alignment_max_yaw_correction_deg: float = 8.0
    alignment_max_translation_correction_m: float = 0.04
    alignment_icp_distance_threshold_m: float = 0.04
    alignment_max_iterations: int = 15
    alignment_convergence_eps: float = 1e-4
    alignment_min_inlier_fraction: float = 0.40
    arm_refinement_max_rotation_deg: float = 20.0
    arm_refinement_max_translation_m: float = 0.05
    rigid_parts_only: bool = True

    # -------------------------------------------------------------------------
    # TSDF Fusion
    # -------------------------------------------------------------------------
    tsdf_voxel_length_m: float = 0.005  # 5 mm
    tsdf_sdf_trunc_m: float = 0.020     # 20 mm (4x voxel)
    tsdf_depth_trunc_m: float = 2.50
    min_mesh_component_ratio: float = 0.02
    keep_dominant_mesh_component: bool = False

    # -------------------------------------------------------------------------
    # Anthropometric Measurements
    # -------------------------------------------------------------------------
    slice_thickness_m: float = 0.020    # 20 mm slab (increased from 15 mm: more points in slab reduces chest underestimate)
    num_angular_bins: int = 180         # 2 deg resolution
    min_angular_coverage_for_ok: float = 0.70
    min_angular_coverage_for_partial: float = 0.35
    crown_hole_threshold: float = 0.20  # hole map fraction above which crown is flagged unobserved
    bootstrap_iterations: int = 200
    crown_edge_margin_px: int = 12

    # -------------------------------------------------------------------------
    # Body Template (Optional Plugin)
    # -------------------------------------------------------------------------
    arap_max_iterations: int = 20
    arap_regularization_weight: float = 50.0

    # -------------------------------------------------------------------------
    # Output & Diagnostics
    # -------------------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("scan_output/fullbody"))
    save_debug_visualizations: bool = False
    verbose: bool = False

