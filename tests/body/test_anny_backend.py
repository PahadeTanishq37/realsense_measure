"""
tests/body/test_anny_backend.py
================================
Integration tests for AnnyBackend and measure_body using anny.
"""

import math
import pytest
import torch
import numpy as np

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.model import BodyModel, BodyOutput
from realsense_measure.body.measure import measure_body, Measurement


@pytest.fixture(scope="module")
def anny_model():
    return AnnyBackend(verbose=False)


def test_anny_backend_interface(anny_model):
    assert isinstance(anny_model, BodyModel)
    assert len(anny_model.phenotype_labels) >= 6
    expected_phenos = ["gender", "age", "muscle", "weight", "height", "proportions"]
    for p in expected_phenos:
        assert p in anny_model.phenotype_labels

    assert anny_model.num_vertices > 10000
    assert anny_model.num_faces > 20000
    assert len(anny_model.joint_names) == 104


def test_anny_backend_rest_pose(anny_model):
    output = anny_model.rest_pose()
    assert isinstance(output, BodyOutput)
    assert isinstance(output.vertices, torch.Tensor)
    assert output.vertices.dim() == 3
    assert output.vertices.shape[0] == 1
    assert output.vertices.shape[2] == 3

    assert isinstance(output.joints, torch.Tensor)
    assert output.joints.dim() == 3
    assert output.joints.shape[0] == 1
    assert output.joints.shape[2] == 3

    # Check Z-up axis convention and realistic human scale in metres
    v = output.numpy_vertices(0)
    z_span = v[:, 2].max() - v[:, 2].min()
    assert 1.2 < z_span < 2.2, f"Expected height ~1.2-2.2m, got {z_span:.2f}m"


def test_anny_backend_batch_rest_pose(anny_model):
    batch_size = 2
    phenotypes = {
        "height": torch.tensor([0.2, 0.8]),
        "weight": torch.tensor([0.3, 0.7]),
    }
    output = anny_model.rest_pose(phenotypes=phenotypes, batch_size=batch_size)
    assert output.batch_size == 2
    assert output.vertices.shape[0] == 2
    # Verify that the two bodies in batch differ
    diff = torch.norm(output.vertices[0] - output.vertices[1]).item()
    assert diff > 0.01


def test_anny_measure_body_sets(anny_model):
    out = anny_model.rest_pose()
    meas = measure_body(out)

    required_set_a = [
        "stature",
        "head_circumference",
        "neck_base_circumference",
        "chest_circumference",
        "waist_circumference",
        "hip_circumference",
        "crotch_height",
        "ankle_circumference",
        "shoulder_elbow_length",
        "shoulder_wrist_length",
        "inseam",
    ]

    required_set_b = [
        "upper_arm_circumference_left",
        "upper_arm_circumference_right",
        "forearm_circumference_left",
        "forearm_circumference_right",
        "mid_thigh_circumference_left",
        "mid_thigh_circumference_right",
        "max_calf_circumference_left",
        "max_calf_circumference_right",
    ]

    for k in required_set_a + required_set_b:
        assert k in meas, f"Missing {k}"
        m = meas[k]
        assert isinstance(m, Measurement)
        assert m.source == "model"
        assert not math.isnan(m.value_mm), f"{k} is NaN"
        assert m.value_mm > 0.0, f"{k} non-positive: {m.value_mm}"

