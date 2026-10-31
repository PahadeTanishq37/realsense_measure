"""
tests/body/test_symmetry.py
===========================
Symmetry sanity test for Set B rehab measurements.

On the symmetric rest pose of any symmetric parametric model (such as anny),
left and right limbs must agree within 2% across random adult bodies.
"""

import math
import numpy as np
import pytest
import torch

from realsense_measure.body.anny_backend import AnnyBackend
from realsense_measure.body.measure import measure_body


@pytest.fixture(scope="module")
def anny_model():
    return AnnyBackend(verbose=False)


def test_set_b_left_right_symmetry_50_adult_bodies(anny_model):
    """
    Generate 50 random adult bodies and verify that every Set B measurement
    agrees between left and right within 2% relative difference.
    """
    rng = np.random.default_rng(12345)
    labels = anny_model.phenotype_labels

    set_b_pairs = [
        ("upper_arm_circumference_left", "upper_arm_circumference_right"),
        ("forearm_circumference_left",   "forearm_circumference_right"),
        ("mid_thigh_circumference_left",  "mid_thigh_circumference_right"),
        ("max_calf_circumference_left",   "max_calf_circumference_right"),
    ]

    # In Anny, adult age is in [0.77, 0.83] (approx 20 - 64 years old)
    max_discrepancies = {f"{l}/{r}": 0.0 for l, r in set_b_pairs}
    failures = []

    for i in range(50):
        # Sample adult phenotypes
        phenos = {
            "gender": float(rng.uniform(0.0, 1.0)),
            "age": float(rng.uniform(0.77, 0.83)),
            "muscle": float(rng.uniform(0.1, 0.9)),
            "weight": float(rng.uniform(0.1, 0.9)),
            "height": float(rng.uniform(0.1, 0.9)),
            "proportions": float(rng.uniform(0.1, 0.9)),
        }

        out = anny_model.rest_pose(phenotypes=phenos)
        meas = measure_body(out)

        for left_key, right_key in set_b_pairs:
            l_val = meas[left_key].value_mm
            r_val = meas[right_key].value_mm
            pair_name = f"{left_key}/{right_key}"

            assert not math.isnan(l_val), f"Body {i}: {left_key} is NaN"
            assert not math.isnan(r_val), f"Body {i}: {right_key} is NaN"

            mean_val = 0.5 * (l_val + r_val)
            diff_pct = abs(l_val - r_val) / mean_val * 100.0 if mean_val > 0 else 0.0
            if diff_pct > max_discrepancies[pair_name]:
                max_discrepancies[pair_name] = diff_pct

            if diff_pct > 2.0:
                failures.append(
                    f"Body {i} ({pair_name}): Left={l_val:.1f} mm, "
                    f"Right={r_val:.1f} mm, Diff={diff_pct:.2f}% (>2%)"
                )

    print("\n--- Maximum Left/Right Discrepancies Across 50 Adult Bodies ---")
    for pair_name, max_diff in max_discrepancies.items():
        print(f"  {pair_name:60s}: max diff = {max_diff:.2f}%")

    assert not failures, (
        f"Symmetry sanity test failed for {len(failures)} measurements:\n"
        + "\n".join(failures[:10])
    )

