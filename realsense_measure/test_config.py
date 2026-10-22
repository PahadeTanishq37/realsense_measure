"""
Unit tests for config.py exercising the sentinel-based override logic (Fix #1).

Fix #1 replaced equality-to-default checks with None-sentinel checks so that:
1. Explicitly passing a value equal to an old default (e.g. min_accept_fitness=0.35)
   is NOT silently overwritten by target-specific logic (e.g. head/body mode).
2. Leaving a field unset (defaulting to None sentinel) applies the correct
   target-specific default for box, head, or body targets.
"""

from pathlib import Path
import pytest

from config import (
    CameraConfig,
    DepthVisConfig,
    HeadCaptureConfig,
    LandmarkConfig,
    PipelineConfig,
    PreprocessConfig,
    RegistrationConfig,
    TargetConfig,
)


class TestSentinelOverrideLogic:
    """Tests verifying sentinel-based defaults in RegistrationConfig, TargetConfig, and PipelineConfig."""

    # -----------------------------------------------------------------------
    # RegistrationConfig Sentinel Tests
    # -----------------------------------------------------------------------

    def test_registration_config_sentinel_preserves_explicit_old_defaults(self):
        """
        Confirm that explicitly passing values equal to old defaults:
          min_accept_fitness = 0.35
          min_colored_icp_fitness = 0.45
          registration_mode = "pointcloud"
        are NOT silently overwritten when resolving defaults for "head" or "body".
        """
        reg_cfg = RegistrationConfig(
            min_accept_fitness=0.35,
            min_colored_icp_fitness=0.45,
            registration_mode="pointcloud",
        )

        # Resolve for head target
        reg_cfg.resolve_defaults(target_name="head")

        assert reg_cfg.min_accept_fitness == 0.35, (
            "Explicit min_accept_fitness=0.35 was silently overwritten by head default"
        )
        assert reg_cfg.min_colored_icp_fitness == 0.45, (
            "Explicit min_colored_icp_fitness=0.45 was silently overwritten by head default"
        )
        assert reg_cfg.registration_mode == "pointcloud", (
            "Explicit registration_mode='pointcloud' was silently overwritten by head default"
        )

        # Resolve for body target
        reg_cfg_body = RegistrationConfig(
            min_accept_fitness=0.35,
            min_colored_icp_fitness=0.45,
            registration_mode="pointcloud",
        )
        reg_cfg_body.resolve_defaults(target_name="body")

        assert reg_cfg_body.min_accept_fitness == 0.35
        assert reg_cfg_body.min_colored_icp_fitness == 0.45
        assert reg_cfg_body.registration_mode == "pointcloud"

    def test_registration_config_unset_fields_apply_target_defaults(self):
        """
        Confirm that leaving fields unset (None) applies the correct
        target-specific defaults.
        """
        # Head target defaults
        head_reg = RegistrationConfig()
        assert head_reg.min_accept_fitness is None
        assert head_reg.min_colored_icp_fitness is None
        assert head_reg.registration_mode is None

        head_reg.resolve_defaults(target_name="head")
        assert head_reg.min_accept_fitness == 0.50
        assert head_reg.min_colored_icp_fitness == 0.50
        assert head_reg.registration_mode == "rgbd_human"

        # Body target defaults
        body_reg = RegistrationConfig()
        body_reg.resolve_defaults(target_name="body")
        assert body_reg.min_accept_fitness == 0.50
        assert body_reg.min_colored_icp_fitness == 0.50
        assert body_reg.registration_mode == "rgbd_human"

        # Box target defaults
        box_reg = RegistrationConfig()
        box_reg.resolve_defaults(target_name="box")
        assert box_reg.min_accept_fitness == 0.35
        assert box_reg.min_colored_icp_fitness == 0.45
        assert box_reg.registration_mode == "pointcloud"

    def test_registration_config_partial_explicit_override(self):
        """
        Confirm that setting one field explicitly (to old default) while leaving
        another unset (None) preserves the explicit value and defaults the unset one.
        """
        reg_cfg = RegistrationConfig(min_accept_fitness=0.35)
        assert reg_cfg.min_colored_icp_fitness is None
        assert reg_cfg.registration_mode is None

        reg_cfg.resolve_defaults(target_name="head")

        assert reg_cfg.min_accept_fitness == 0.35  # preserved explicit old default
        assert reg_cfg.min_colored_icp_fitness == 0.50  # resolved target default
        assert reg_cfg.registration_mode == "rgbd_human"  # resolved target default

    # -----------------------------------------------------------------------
    # TargetConfig Sentinel Tests
    # -----------------------------------------------------------------------

    def test_target_config_preserves_explicit_old_defaults(self):
        """
        Confirm that passing explicit physical bounds equal to box defaults
        on a head TargetConfig is not overwritten in __post_init__.
        """
        target = TargetConfig(
            name="head",
            expected_min_size_m=0.03,
            expected_max_size_m=0.60,
            min_object_points=150,
            max_centroid_dev_m=0.15,
        )

        assert target.expected_min_size_m == 0.03
        assert target.expected_max_size_m == 0.60
        assert target.min_object_points == 150
        assert target.max_centroid_dev_m == 0.15

    def test_target_config_unset_fields_apply_target_defaults(self):
        """
        Confirm that TargetConfig applies distinct target defaults when
        fields are left unset (None).
        """
        head = TargetConfig(name="head")
        assert head.expected_min_size_m == 0.12
        assert head.expected_max_size_m == 0.55
        assert head.min_object_points == 200
        assert head.max_centroid_dev_m == 0.25

        body = TargetConfig(name="body")
        assert body.expected_min_size_m == 0.20
        assert body.expected_max_size_m == 1.20
        assert body.min_object_points == 300
        assert body.max_centroid_dev_m == 0.35

        box = TargetConfig(name="box")
        assert box.expected_min_size_m == 0.03
        assert box.expected_max_size_m == 0.60
        assert box.min_object_points == 150
        assert box.max_centroid_dev_m == 0.15

    # -----------------------------------------------------------------------
    # PipelineConfig Top-Level Sentinel Tests
    # -----------------------------------------------------------------------

    def test_pipeline_config_preserves_explicit_old_defaults(self):
        """
        Confirm that PipelineConfig preserves explicit registration and TSDF
        parameters equal to old box defaults even when target is 'head'.
        """
        reg_cfg = RegistrationConfig(
            min_accept_fitness=0.35,
            min_colored_icp_fitness=0.45,
            registration_mode="pointcloud",
        )
        pipe_cfg = PipelineConfig(
            target=TargetConfig(name="head"),
            registration=reg_cfg,
            tsdf_voxel_length_m=0.004,
            tsdf_trunc_m=0.02,
        )

        # Registration fields should NOT have been overwritten
        assert pipe_cfg.registration.min_accept_fitness == 0.35
        assert pipe_cfg.registration.min_colored_icp_fitness == 0.45
        assert pipe_cfg.registration.registration_mode == "pointcloud"

        # TSDF fields should NOT have been overwritten
        assert pipe_cfg.tsdf_voxel_length_m == 0.004
        assert pipe_cfg.tsdf_trunc_m == 0.02

    def test_pipeline_config_unset_fields_apply_target_defaults(self):
        """
        Confirm that PipelineConfig resolves None sentinels for both registration
        and TSDF parameters based on target type.
        """
        head_pipe = PipelineConfig(target=TargetConfig(name="head"))
        assert head_pipe.registration.min_accept_fitness == 0.50
        assert head_pipe.registration.min_colored_icp_fitness == 0.50
        assert head_pipe.registration.registration_mode == "rgbd_human"
        assert head_pipe.tsdf_voxel_length_m == 0.002
        assert head_pipe.tsdf_trunc_m == pytest.approx(0.008)

        box_pipe = PipelineConfig(target=TargetConfig(name="box"))
        assert box_pipe.registration.min_accept_fitness == 0.35
        assert box_pipe.registration.min_colored_icp_fitness == 0.45
        assert box_pipe.registration.registration_mode == "pointcloud"
        assert box_pipe.tsdf_voxel_length_m == 0.004
        assert box_pipe.tsdf_trunc_m == 0.02
