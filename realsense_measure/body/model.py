"""
realsense_measure/body/model.py
================================
Abstract interface for parametric 3D body models.

Axis convention (IDENTICAL for all implementations in this sub-package)
-----------------------------------------------------------------------
  +Z  UP       (Z increases towards the top of the head)
  +Y  FORWARD  (Y increases in the direction the subject faces / anterior)
  +X  LATERAL  (X > 0 = subject's own right)

This is anny's native MakeHuman coordinate frame confirmed from the
actual model outputs (anny 0.6.2):
  - floor (heel)  ≈  Z = -0.87 m
  - crown (head)  ≈  Z = +0.76 m
  - anterior      ≈  Y > 0
  - subject right ≈  X > 0

All vertices are in **metres** (float32 by default).

Thread-safety
-------------
Instances are NOT thread-safe.  Create one instance per thread.

Reproducibility
---------------
Every call to forward() and rest_pose() is deterministic given the same
inputs and random seed.  No internal random state is maintained after
construction.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BodyOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class BodyOutput:
    """
    Output of :meth:`BodyModel.forward`.

    All tensors are on the same device as the model and have a leading
    batch dimension (B).

    Attributes
    ----------
    vertices : torch.Tensor, shape (B, V, 3), float32
        Posed mesh vertices in **metres**.
        Axis convention: +Y up, +Z forward, +X right.
    faces : torch.Tensor, shape (F, 3), int64
        Triangle face indices (shared across the batch; no batch dim).
    joints : torch.Tensor, shape (B, J, 3), float32
        3-D joint positions in **metres**, same coordinate frame.
        J = number of joints reported by the backend (model-dependent).
    joint_names : List[str]
        Length-J list of human-readable joint names, aligned with the
        joints axis.
    phenotypes : dict[str, float]
        Phenotype values that produced this output (for logging /
        reproducibility).  Copied from the input; not re-derived.
    raw : dict
        Model-specific raw output dictionary for debugging.  Keys and
        shapes are backend-dependent.
    """

    vertices: torch.Tensor          # (B, V, 3) metres
    faces: torch.Tensor             # (F, 3)
    joints: torch.Tensor            # (B, J, 3) metres
    joint_names: List[str]
    phenotypes: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[1]

    @property
    def num_joints(self) -> int:
        return self.joints.shape[1]

    def numpy_vertices(self, batch_idx: int = 0):
        """Return vertices as a numpy array (V, 3) for batch index *batch_idx*."""
        return self.vertices[batch_idx].detach().cpu().numpy()

    def numpy_joints(self, batch_idx: int = 0):
        """Return joints as a numpy array (J, 3) for batch index *batch_idx*."""
        return self.joints[batch_idx].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# BodyModel abstract base class
# ---------------------------------------------------------------------------

class BodyModel(abc.ABC):
    """
    Abstract interface for a differentiable parametric body model.

    Sub-classes must implement :meth:`forward` and :meth:`rest_pose`.

    All implementations MUST satisfy:

    * **Metres** – vertex coordinates are always in metres.
    * **Differentiable** – the computation graph is preserved so callers
      can back-propagate through vertices and joints.
    * **Batch-capable** – every parameter can carry a leading batch
      dimension; scalar inputs are broadcast automatically.
    * **Deterministic** – given the same inputs the outputs are
      bit-identical.  Do NOT use dropout or stochastic operations.
    * **No symmetry priors** – the model does NOT enforce left/right
      bilateral symmetry.  Left and right limb shapes are allowed to
      differ independently (important for rehab applications).

    Axis convention
    ---------------
    +Y up, +Z forward, +X right.  This is the MakeHuman / anny native
    frame and must be preserved by all implementations.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def forward(
        self,
        phenotypes: Optional[dict] = None,
        pose: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        """
        Run the body model and return a :class:`BodyOutput`.

        Parameters
        ----------
        phenotypes : dict[str, float | torch.Tensor], optional
            High-level shape parameters.  Keys are model-specific label
            strings (see :attr:`phenotype_labels`).  Values are scalar
            floats or (B,) tensors in the range expected by the model
            (typically [0, 1]).  Missing keys default to the neutral
            midpoint (0.5 for [0,1]-ranged parameters).
        pose : torch.Tensor, optional
            Bone pose parameters.  Shape is model-specific.  When *None*,
            the model's canonical A-pose is used (all bones at identity).
        batch_size : int
            Number of bodies in the batch.  Ignored when *phenotypes*
            contains tensor values with an explicit batch dimension.
        **kwargs
            Backend-specific extra arguments passed through unchanged.

        Returns
        -------
        BodyOutput
        """

    @abc.abstractmethod
    def rest_pose(
        self,
        phenotypes: Optional[dict] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        """
        Return the body in **canonical A-pose** with the given shape.

        All bone rotations are identity; no pose blend-shapes are applied.
        This is the pose used for all body measurements in
        :mod:`realsense_measure.body.measure`.

        Parameters
        ----------
        phenotypes : dict[str, float | torch.Tensor], optional
            Same meaning as in :meth:`forward`.
        batch_size : int
            Number of bodies in the batch.

        Returns
        -------
        BodyOutput
            Vertices in A-pose, joints in A-pose.
        """

    # ------------------------------------------------------------------
    # Informational properties (must be provided by sub-classes)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def phenotype_labels(self) -> List[str]:
        """Ordered list of phenotype parameter names."""

    @property
    @abc.abstractmethod
    def joint_names(self) -> List[str]:
        """Ordered list of joint names, aligned with the joints tensor."""

    @property
    @abc.abstractmethod
    def num_vertices(self) -> int:
        """Total number of vertices in the mesh topology."""

    @property
    @abc.abstractmethod
    def num_faces(self) -> int:
        """Total number of triangular faces."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Torch device on which model parameters live."""

    # ------------------------------------------------------------------
    # Default neutral phenotype helper
    # ------------------------------------------------------------------

    def neutral_phenotypes(self, value: float = 0.5) -> dict:
        """
        Return a dict with all phenotypes set to *value* (default 0.5).

        Useful as a starting point for sweeps or optimisation.
        """
        return {k: value for k in self.phenotype_labels}
