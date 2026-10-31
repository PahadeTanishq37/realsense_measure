"""
tests/body/test_gradient_check.py
==================================
Requirement 0: Gradient check for Anny phenotype parameters.
Compares autograd d(vertices)/d(phenotype) against central finite
differences (step 0.02) in both float32 and float64 for all 6 phenotypes:
['gender', 'age', 'muscle', 'weight', 'height', 'proportions'].
"""

import pytest
import torch
import anny

LABELS = ["gender", "age", "muscle", "weight", "height", "proportions"]
BASE_P = [0.45, 0.79, 0.45, 0.45, 0.45, 0.45]
H = 0.02


def compute_exact_jacobians(dtype: torch.dtype, h: float = H):
    """
    Computes exact autograd Jacobian (V, 3, 6) and finite difference Jacobian
    (V, 3, 6) across all 13,718 vertices in the mesh.
    """
    model = anny.Anny(facial_actions="none", local_changes="none").to(dtype=dtype)
    base_tensor = torch.tensor(BASE_P, dtype=dtype)

    def get_verts(p):
        _, pheno, lc, fa = model.get_tensor_inputs(None, p.unsqueeze(0), None, None)
        coeffs = model._get_phenotype_blendshape_coefficients(pheno, lc, fa)
        return model.get_rest_vertices(coeffs)[0]

    # Exact autograd Jacobian via forward-mode AD vectorization
    jac_auto = torch.func.jacfwd(get_verts)(base_tensor)

    # Central finite differences
    jac_fd = torch.zeros_like(jac_auto)
    for j in range(6):
        p_plus = base_tensor.clone()
        p_plus[j] += h
        p_minus = base_tensor.clone()
        p_minus[j] -= h
        v_plus = get_verts(p_plus)
        v_minus = get_verts(p_minus)
        jac_fd[:, :, j] = (v_plus - v_minus) / (2.0 * h)

    return jac_auto, jac_fd


@pytest.mark.parametrize("dtype, max_rel_err", [
    (torch.float32, 5e-4),   # float32 has precision limit ~1e-4 with h=0.02
    (torch.float64, 1e-10),  # float64 achieves machine precision ~1e-13
])
def test_autograd_vs_finite_difference_per_phenotype(dtype, max_rel_err):
    """
    Verifies that autograd d(vertices)/d(phenotype) matches central finite
    differences within tolerance across all 13,718 mesh vertices for every phenotype.
    """
    jac_auto, jac_fd = compute_exact_jacobians(dtype)

    for j, name in enumerate(LABELS):
        diff_norm = torch.linalg.norm(jac_auto[:, :, j] - jac_fd[:, :, j]).item()
        fd_norm = torch.linalg.norm(jac_fd[:, :, j]).item()
        rel_err = diff_norm / (fd_norm + 1e-12)

        assert not torch.isnan(jac_auto[:, :, j]).any(), f"NaN in autograd gradient for {name}"
        assert not torch.isinf(jac_auto[:, :, j]).any(), f"Inf in autograd gradient for {name}"
        assert fd_norm > 0.1, f"Zero finite difference norm for {name}"
        assert rel_err < max_rel_err, (
            f"Relative error for {name} ({rel_err:.3e}) exceeded tolerance {max_rel_err:.3e} "
            f"in {dtype}"
        )

