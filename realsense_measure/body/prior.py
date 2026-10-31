"""
realsense_measure/body/prior.py
================================
Phenotype population distribution and priors wrapping
``anny.shape_distribution.SimpleShapeDistribution``.

Provides:
  1. Population shape prior conditioned on age and gender for:
     - height
     - weight
     - muscle
     - proportions
  2. Adult sampler sampling specifically from the adult age range
     [0.7726, 0.8248] (20–60 years).
  3. Differentiable log-prob prior for MAP optimization.
  4. Required baseline estimators:
     (i)  Average adult from the prior
     (ii) Prior conditioned on true height only

Constraint note:
  This module explicitly contains NO symmetry priors.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from anny.shape_distribution import MorphologicalAgeMapping, SimpleShapeDistribution

logger = logging.getLogger(__name__)

# Adult age interval in anny phenotype units (20–60 years)
ADULT_AGE_MIN = 0.7726
ADULT_AGE_MAX = 0.8248
ADULT_AGE_NEUTRAL = 0.7922  # ~35 years

PHENOTYPE_NAMES = ["gender", "age", "muscle", "weight", "height", "proportions"]


class AdultShapePrior(nn.Module):
    """
    Population prior for Anny phenotype parameters.
    """

    def __init__(self, anny_model=None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        if anny_model is None:
            import anny
            anny_model = anny.Anny(facial_actions="none", local_changes="none")
        self.ssd = SimpleShapeDistribution(anny_model).to(self.device)
        self.age_mapping = MorphologicalAgeMapping()

    def sample_adult(
        self,
        n: int = 1,
        seed: Optional[int] = None,
        gender: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample n random adult phenotype vectors.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : Optional[int]
            Random seed for reproducibility.
        gender : Optional[float]
            If specified in [0, 1], gender is fixed. Otherwise uniformly sampled.

        Returns
        -------
        Dict[str, np.ndarray] of 1D arrays of length n.
        """
        rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        ages = rng.uniform(ADULT_AGE_MIN, ADULT_AGE_MAX, size=n).astype(np.float64)
        if gender is not None:
            genders = np.full(n, fill_value=gender, dtype=np.float64)
        else:
            genders = rng.uniform(0.0, 1.0, size=n).astype(np.float64)

        dtype = self.ssd.boys_conditional_height_distribution.age_anchors.dtype
        age_t = torch.tensor(ages, dtype=dtype, device=self.device)
        gen_t = torch.tensor(genders, dtype=dtype, device=self.device)

        # Conditional distributions
        def _sample_feat(dist_b, dist_g):
            db = dist_b.get_torch_distribution(age_t).sample()
            dg = dist_g.get_torch_distribution(age_t).sample()
            return torch.where(gen_t <= 0.5, db, dg).cpu().numpy()

        heights = _sample_feat(
            self.ssd.boys_conditional_height_distribution,
            self.ssd.girls_conditional_height_distribution,
        )
        weights = _sample_feat(
            self.ssd.boys_conditional_weight_distribution,
            self.ssd.girls_conditional_weight_distribution,
        )
        muscles = _sample_feat(
            self.ssd.boys_conditional_muscle_distribution,
            self.ssd.girls_conditional_muscle_distribution,
        )
        props = _sample_feat(
            self.ssd.boys_conditional_proportions_distribution,
            self.ssd.girls_conditional_proportions_distribution,
        )

        return {
            "gender": genders,
            "age": ages,
            "muscle": muscles,
            "weight": weights,
            "height": heights,
            "proportions": props,
        }

    def log_prob(
        self,
        phenotypes: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute total log probability log p(phenotypes).

        Differentiable wrt phenotype parameters.
        """
        if isinstance(phenotypes, dict):
            p = torch.stack(
                [phenotypes[k].squeeze() for k in PHENOTYPE_NAMES],
                dim=-1,
            )
        else:
            p = phenotypes

        if p.dim() == 1:
            p = p.unsqueeze(0)

        # Components: [gender, age, muscle, weight, height, proportions]
        gender = torch.clamp(p[:, 0], 1e-4, 1.0 - 1e-4)
        age = torch.clamp(p[:, 1], 1e-4, 1.0 - 1e-4)
        muscle = torch.clamp(p[:, 2], 1e-4, 1.0 - 1e-4)
        weight = torch.clamp(p[:, 3], 1e-4, 1.0 - 1e-4)
        height = torch.clamp(p[:, 4], 1e-4, 1.0 - 1e-4)
        proportions = torch.clamp(p[:, 5], 1e-4, 1.0 - 1e-4)

        # Age prior: penalty for leaving adult range [ADULT_AGE_MIN, ADULT_AGE_MAX]
        age_mid = 0.5 * (ADULT_AGE_MIN + ADULT_AGE_MAX)
        age_half_width = 0.5 * (ADULT_AGE_MAX - ADULT_AGE_MIN)
        age_z = (age - age_mid) / (age_half_width + 1e-6)
        log_p_age = -0.5 * torch.relu(torch.abs(age_z) - 1.0) ** 2 * 20.0

        # Feature conditional mixture log probability
        def _mixture_log_prob(dist_b, dist_g, val):
            # Beta distribution parameters from age
            db = dist_b.get_torch_distribution(age)
            dg = dist_g.get_torch_distribution(age)
            lp_b = db.log_prob(val)
            lp_g = dg.log_prob(val)
            # Mixture: (1 - gender) * p_b + gender * p_g
            w_b = torch.log(1.0 - gender + 1e-8)
            w_g = torch.log(gender + 1e-8)
            return torch.logaddexp(w_b + lp_b, w_g + lp_g)

        lp_h = _mixture_log_prob(
            self.ssd.boys_conditional_height_distribution,
            self.ssd.girls_conditional_height_distribution,
            height,
        )
        lp_w = _mixture_log_prob(
            self.ssd.boys_conditional_weight_distribution,
            self.ssd.girls_conditional_weight_distribution,
            weight,
        )
        lp_m = _mixture_log_prob(
            self.ssd.boys_conditional_muscle_distribution,
            self.ssd.girls_conditional_muscle_distribution,
            muscle,
        )
        lp_p = _mixture_log_prob(
            self.ssd.boys_conditional_proportions_distribution,
            self.ssd.girls_conditional_proportions_distribution,
            proportions,
        )

        total_log_prob = log_p_age + lp_h + lp_w + lp_m + lp_p
        return total_log_prob.squeeze()

    def get_average_adult_phenotypes(
        self,
        gender: float = 0.5,
    ) -> Dict[str, float]:
        """
        Baseline (i): Average adult from the prior.
        """
        dtype = self.ssd.boys_conditional_height_distribution.age_anchors.dtype
        age_t = torch.tensor([ADULT_AGE_NEUTRAL], dtype=dtype, device=self.device)
        gen_t = torch.tensor([gender], dtype=dtype, device=self.device)

        def _cond_mean(dist_b, dist_g):
            mb = dist_b.get_torch_distribution(age_t).mean
            mg = dist_g.get_torch_distribution(age_t).mean
            return float(torch.where(gen_t <= 0.5, mb, mg).item())

        h_mean = _cond_mean(
            self.ssd.boys_conditional_height_distribution,
            self.ssd.girls_conditional_height_distribution,
        )
        w_mean = _cond_mean(
            self.ssd.boys_conditional_weight_distribution,
            self.ssd.girls_conditional_weight_distribution,
        )
        m_mean = _cond_mean(
            self.ssd.boys_conditional_muscle_distribution,
            self.ssd.girls_conditional_muscle_distribution,
        )
        p_mean = _cond_mean(
            self.ssd.boys_conditional_proportions_distribution,
            self.ssd.girls_conditional_proportions_distribution,
        )

        return {
            "gender": float(gender),
            "age": ADULT_AGE_NEUTRAL,
            "muscle": m_mean,
            "weight": w_mean,
            "height": h_mean,
            "proportions": p_mean,
        }

    def get_height_conditioned_phenotypes(
        self,
        true_stature_mm: float,
        gender: float = 0.5,
    ) -> Dict[str, float]:
        """
        Baseline (ii): Prior conditioned on true height only.
        Inverts true stature to find the corresponding height phenotype,
        and sets other phenotypes to their conditional prior means.
        """
        # Empirical linear mapping for neutral adult:
        # stature_mm ≈ 1360.0 + 960.0 * height_phenotype
        # Invert:
        h_pheno = float(np.clip((true_stature_mm - 1360.0) / 960.0, 0.05, 0.95))

        base = self.get_average_adult_phenotypes(gender=gender)
        base["height"] = h_pheno
        return base
