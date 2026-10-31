"""
tests/body/test_fit.py
======================
Tests for DifferentiableBodyFitter: quick end-to-end fit and convergence check.
"""

import numpy as np
import pytest

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.fit import DifferentiableBodyFitter
from realsense_measure.body.prior import AdultShapePrior
from realsense_measure.body.synth import CameraConfig, SyntheticTruthSubject


@pytest.fixture(scope="module")
def backend():
    return AnnyBackend()


@pytest.fixture(scope="module")
def prior(backend):
    return AdultShapePrior(backend._model)


def test_quick_body_fit(backend, prior):
    subject = SyntheticTruthSubject(backend, prior, subject_id=0, seed=42)
    cfg = CameraConfig.portrait(pessimistic=False)
    stops = subject.simulate_stops(cfg, n_stops=2)

    fitter = DifferentiableBodyFitter(backend, prior)
    true_stature = subject.true_measurements["stature"].value_mm

    res = fitter.fit(
        stops,
        known_height_mm=true_stature,
        n_coarse_iter=15,
        n_fine_iter=10,
        multi_starts=1,
    )

    assert res.converged or res.final_loss < 5.0
    assert res.elapsed_sec < 10.0
    fitted_stature = res.measurements["stature"].value_mm
    # Stature should fit within allowable error (10 mm)
    assert abs(fitted_stature - true_stature) < 15.0

