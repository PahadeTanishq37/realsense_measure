"""
tests/body/test_synth.py
========================
Tests for synthetic ground truth generation, smooth displacement field,
and virtual D455 depth camera raycasting with Open3D.
"""

import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import (
    CameraConfig,
    SyntheticTruthSubject,
    generate_smooth_displacement_field,
)


@pytest.fixture(scope="module")
def backend():
    return AnnyBackend()


@pytest.fixture(scope="module")
def prior(backend):
    return AdultShapePrior(backend._model)


def test_displacement_field_statistics():
    # Synthetic sphere
    pts = np.random.randn(2000, 3) * 0.5
    disp = generate_smooth_displacement_field(
        pts,
        correlation_length_m=0.05,
        std_mm=6.5,
        seed=42,
    )
    std_mm = np.std(disp) * 1000.0
    assert 5.0 <= std_mm <= 8.0


def test_synthetic_truth_subject_measurements(backend, prior):
    subject = SyntheticTruthSubject(backend, prior, subject_id=0, seed=42)
    m = subject.true_measurements
    assert "stature" in m
    assert 1300.0 < m["stature"].value_mm < 2400.0
    assert "chest_circumference" in m
    assert 600.0 < m["chest_circumference"].value_mm < 1600.0


def test_virtual_camera_raycasting(backend, prior):
    subject = SyntheticTruthSubject(backend, prior, subject_id=0, seed=42)
    cfg = CameraConfig.portrait(pessimistic=False)
    stops = subject.simulate_stops(cfg, n_stops=2)

    assert len(stops) == 2
    for s in stops:
        assert s.depth_image.shape == (cfg.height, cfg.width)
        assert s.foreground_mask.shape == (cfg.height, cfg.width)
        assert len(s.point_cloud) > 10000
        # Depth values should be in realistic 0.5 to 2.5 m range
        assert 0.5 < s.depth_image[s.foreground_mask].min() < 2.0
        assert 0.8 < s.depth_image[s.foreground_mask].max() < 3.0

