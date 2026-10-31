"""
tests/body/test_prior.py
========================
Tests for AdultShapePrior: adult sampling, conditional prior distributions,
log-probability evaluation, and required baselines.
"""

import numpy as np
import pytest
import torch

from realsense_measure.body.prior import (
    ADULT_AGE_MAX,
    ADULT_AGE_MIN,
    ADULT_AGE_NEUTRAL,
    AdultShapePrior,
)


@pytest.fixture(scope="module")
def prior():
    return AdultShapePrior()


def test_adult_sampling_within_bounds(prior):
    samples = prior.sample_adult(n=50, seed=123)
    assert len(samples["age"]) == 50
    assert np.all(samples["age"] >= ADULT_AGE_MIN)
    assert np.all(samples["age"] <= ADULT_AGE_MAX)
    assert np.all(samples["height"] >= 0.0)
    assert np.all(samples["height"] <= 1.0)
    assert np.all(samples["weight"] >= 0.0)
    assert np.all(samples["weight"] <= 1.0)


def test_log_prob_differentiable(prior):
    p = torch.tensor([0.5, ADULT_AGE_NEUTRAL, 0.45, 0.45, 0.45, 0.5], requires_grad=True)
    lp = prior.log_prob(p)
    assert torch.isfinite(lp)
    lp.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert not torch.isnan(p.grad).any()


def test_baseline_average_adult(prior):
    avg = prior.get_average_adult_phenotypes()
    assert avg["age"] == ADULT_AGE_NEUTRAL
    assert 0.2 < avg["height"] < 0.8
    assert 0.2 < avg["weight"] < 0.8


def test_baseline_height_conditioned(prior):
    cond_tall = prior.get_height_conditioned_phenotypes(1900.0)
    cond_short = prior.get_height_conditioned_phenotypes(1500.0)
    assert cond_tall["height"] > cond_short["height"]

