"""
realsense_measure/body/model.py
================================
Abstract interface for parametric 3D body models.

Axis convention (IDENTICAL for all implementations in this sub-package)
-----------------------------------------------------------------------
  +Z  UP       (Z increases towards the top of the head)
  +Y  FORWARD  (Y increases in the direction the subject faces / anterior)
  +X  LATERAL  (X > 0 = subject's own right)

All vertices are in **metres** (float32 by default).
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BodyOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class BodyOutput:
    """
    Output of :meth:`BodyModel.forward`.

    Attributes
    ----------
    vertices : np.ndarray or torch.Tensor, shape (B, V, 3) or (V, 3)
        Posed mesh vertices in **metres**.
        Axis convention: +Y up, +Z forward, +X right.
    faces : np.ndarray or torch.Tensor, shape (F, 3)
        Triangle face indices.
    joints : np.ndarray or torch.Tensor, shape (B, J, 3) or (J, 3)
        3-D joint positions in **metres**, same coordinate frame.
    joint_names : List[str]
        Length-J list of human-readable joint names, aligned with the joints axis.
    phenotypes : dict[str, float]
        Phenotype values that produced this output.
    raw : dict
        Model-specific raw output dictionary for debugging.
    """

    vertices: Any
    faces: Any
    joints: Any
    joint_names: List[str]
    phenotypes: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        shape = getattr(self.vertices, "shape", (1, 0))
        return shape[0] if len(shape) == 3 else 1

    @property
    def num_vertices(self) -> int:
        shape = getattr(self.vertices, "shape", (0, 0))
        return shape[1] if len(shape) == 3 else shape[0]

    @property
    def num_joints(self) -> int:
        shape = getattr(self.joints, "shape", (0, 0))
        return shape[1] if len(shape) == 3 else shape[0]

    def numpy_vertices(self, batch_idx: int = 0) -> np.ndarray:
        """Return vertices as a numpy array (V, 3) for batch index *batch_idx*."""
        v = self.vertices
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v)
        if v.ndim == 3:
            return v[batch_idx]
        return v

    def numpy_joints(self, batch_idx: int = 0) -> np.ndarray:
        """Return joints as a numpy array (J, 3) for batch index *batch_idx*."""
        j = self.joints
        if hasattr(j, "detach"):
            j = j.detach().cpu().numpy()
        j = np.asarray(j)
        if j.ndim == 3:
            return j[batch_idx]
        return j

    def numpy_faces(self) -> np.ndarray:
        """Return faces as a numpy array (F, 3)."""
        f = self.faces
        if hasattr(f, "detach"):
            f = f.detach().cpu().numpy()
        return np.asarray(f)


# ---------------------------------------------------------------------------
# BodyModel abstract base class
# ---------------------------------------------------------------------------

class BodyModel(abc.ABC):
    """
    Abstract interface for a parametric body model.

    Sub-classes must implement :meth:`forward` and :meth:`rest_pose`.
    """

    @abc.abstractmethod
    def forward(
        self,
        phenotypes: Optional[dict] = None,
        pose: Optional[Any] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        """Run the body model and return a :class:`BodyOutput`."""

    @abc.abstractmethod
    def rest_pose(
        self,
        phenotypes: Optional[dict] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        """Return the body in **canonical A-pose** with the given shape."""

    @property
    @abc.abstractmethod
    def phenotype_labels(self) -> List[str]:
        """Ordered list of phenotype parameter names."""

    @property
    @abc.abstractmethod
    def joint_names(self) -> List[str]:
        """Ordered list of joint names, aligned with the joints array."""

    @property
    @abc.abstractmethod
    def num_vertices(self) -> int:
        """Total number of vertices in the mesh topology."""

    @property
    @abc.abstractmethod
    def num_faces(self) -> int:
        """Total number of triangular faces."""

    @property
    def device(self) -> Any:
        """Device on which model parameters live, if applicable."""
        return "cpu"

    def neutral_phenotypes(self, value: float = 0.5) -> dict:
        """Return a dict with all phenotypes set to *value* (default 0.5)."""
        return {k: value for k in self.phenotype_labels}
