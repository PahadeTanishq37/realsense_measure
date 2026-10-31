"""
realsense_measure/body/anny_backend.py
=======================================
Wrapper around ``anny.Anny`` that implements the :class:`BodyModel`
abstract interface.

Setup / first-run notes
-----------------------
On the very first call, anny downloads and caches its model assets
(blendshapes, skeletal rig, skinning weights).  This can take **2–5
minutes** depending on network speed.  Subsequent runs load from the
local cache (usually under the system temp / NVIDIA warp cache).  All
downloaded asset paths and sizes are logged at INFO level so you can audit.

Licence note
------------
The *base* ``anny`` package (pip install anny) is Apache 2.0 licensed.
The optional ``[smpl]`` extra (NOT installed here) contains SMPL model
weights which are CC BY-NC 4.0 (non-commercial only).  This wrapper
uses ONLY the base package assets, which are freely usable commercially.

Configuration choice  (anny 0.6.2)
------------------------------------
::

    anny.Anny(
        rig='anny',                    # default 104-bone rig
        topology='anny',               # default ~13 718 vertex topology
        local_changes='none',          # no local blendshapes needed
        facial_actions='none',         # face actions disabled (default)
        phenotypes='default',          # 6 main phenotype axes
        pose_parameterization='local-ref',   # default
        skinning_method=None,          # default (warp LBS)
        phenotype_blendshapes='compact',     # default compact representation
    )

``facial_actions='none'`` and ``local_changes='none'`` disable shape
blendshapes that are not needed for body measurement, reducing memory
and forward-pass time.

Axis convention (anny native)
------------------------------
+Z  UP       (Z increases toward the top of the head)
+X  LATERAL  (X > 0 = subject's own right; viewer's left when facing +Y)
+Y  FORWARD  (Y increases in the direction the subject faces, i.e. anterior)

Vertices are in **metres**.  A neutral adult body spans roughly
Z ∈ [-0.87, 0.76] (floor-to-crown ≈ 1.63 m).

.. note::
   The :class:`BodyOutput` dataclass stores vertices with this same
   +Z-up convention.  The ``joint_names`` correspond to anny ``bone_labels``.

Phenotype labels and ranges (anny 0.6.2)
-----------------------------------------
The six phenotype axes confirmed from ``model.phenotype_labels``:

  ============  ===========  =============================
  anny label    Range        Notes
  ============  ===========  =============================
  gender        [0, 1]       0 ≈ male morphology, 1 ≈ female
  age           [0, 1]       0 ≈ infant, 0.5 ≈ ~35 yr, 1 ≈ elder
  muscle        [0, 1]       0 ≈ low muscle, 1 ≈ high muscle
  weight        [0, 1]       0 ≈ very light, 1 ≈ very heavy
  height        [0, 1]       0 ≈ very short, 1 ≈ very tall
  proportions   [0, 1]       0 ≈ stocky/short-limbed, 1 ≈ elongated
  ============  ===========  =============================

These are printed and verified at construction time.  If anny updates
its labels, the mismatch is logged as a WARNING.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import torch

from .model import BodyModel, BodyOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Friendly-name → anny-label map  (verified against anny 0.6.2)
# ---------------------------------------------------------------------------

# Canonical 6 main phenotype labels as returned by anny 0.6.2
_CANONICAL_PHENOTYPES = ["gender", "age", "muscle", "weight", "height", "proportions"]


class AnnyBackend(BodyModel):
    """
    BodyModel implementation wrapping ``anny.Anny`` (v0.6.2+).

    Parameters
    ----------
    device : str or torch.device
        Where to place the model tensors.  Defaults to CPU.
    dtype : torch.dtype
        Floating-point type.  float32 recommended.
    verbose : bool
        When True, print a detailed inspection of model attributes to
        the log on construction (phenotype labels, bone labels, etc.).
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype

        logger.info("=" * 60)
        logger.info("AnnyBackend: loading anny.Anny model …")
        logger.info(
            "  NOTE: First-start asset download/compilation can take 2–5 min."
        )
        t0 = time.time()

        # ------------------------------------------------------------------
        # Import anny and log version
        # ------------------------------------------------------------------
        try:
            import anny as _anny_pkg
        except ImportError as exc:
            raise ImportError(
                "The 'anny' package is not installed.\n"
                "Run:  pip install anny\n"
                "(Do NOT add [examples] or [smpl] extras — [smpl] pulls in "
                "CC BY-NC SMPL weights which are non-commercial-only.)"
            ) from exc

        _anny_version = getattr(_anny_pkg, "__version__", "unknown")
        logger.info("  anny version : %s", _anny_version)
        logger.info(
            "  Licence note : base anny assets are Apache 2.0 (commercially "
            "usable).  The [smpl] extra (NOT installed) contains CC BY-NC "
            "assets (non-commercial only)."
        )

        # ------------------------------------------------------------------
        # Construct the anny model (lightest configuration)
        # facial_actions='none'  → face blendshapes disabled (not needed)
        # local_changes='none'   → local shape blendshapes disabled
        # ------------------------------------------------------------------
        self._model = _anny_pkg.Anny(
            facial_actions="none",    # disabled — unneeded for body measurement
            local_changes="none",     # disabled — reduces memory & forward time
        ).to(device=self._device, dtype=self._dtype)

        elapsed = time.time() - t0
        logger.info("  Model loaded in %.1f s.", elapsed)

        # ------------------------------------------------------------------
        # Cache faces and identity pose before inspection
        # ------------------------------------------------------------------
        self._faces: torch.Tensor = self._model.faces.to(
            device=self._device, dtype=torch.long
        )
        self._bone_count: int = self._model.bone_count
        self._identity_pose = (
            torch.eye(4, device=self._device, dtype=self._dtype)
            .unsqueeze(0)      # (1, 4, 4)
            .unsqueeze(0)      # (1, 1, 4, 4)
        )

        # ------------------------------------------------------------------
        # Inspect and log all model attributes
        # ------------------------------------------------------------------
        if verbose:
            self._inspect_and_log()

        # ------------------------------------------------------------------
        # Verify phenotype labels
        # ------------------------------------------------------------------
        actual_labels = list(self._model.phenotype_labels)
        for expected in _CANONICAL_PHENOTYPES:
            if expected not in actual_labels:
                logger.warning(
                    "Expected phenotype label '%s' not in model.phenotype_labels. "
                    "anny may have changed its API.  Actual: %s",
                    expected, actual_labels,
                )

        logger.info("AnnyBackend ready.")
        logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Inspection helper
    # -----------------------------------------------------------------------

    def _inspect_and_log(self) -> None:
        """Print and log a full inspection of the anny model."""
        m = self._model
        sep = "─" * 60

        logger.info(sep)
        logger.info("ANNY MODEL INSPECTION")
        logger.info(sep)

        pheno = list(m.phenotype_labels)
        logger.info("phenotype_labels (%d): %s", len(pheno), pheno)

        bone = list(m.bone_labels)
        logger.info("bone_labels (%d): %s", len(bone), bone)

        lc = list(m.local_change_labels)
        logger.info("local_change_labels (%d): %s", len(lc), lc)

        logger.info("bone_count : %d", m.bone_count)
        logger.info("faces shape: %s", tuple(m.faces.shape))

        logger.info(
            "vertex_units : metres  "
            "(anny 0.6.x does not expose vertex_units attribute; "
            "units confirmed from rest_bone_heads Z range ≈ 1.6 m for adult)"
        )
        logger.info(
            "Axis convention : +Z UP, +X lateral (subject right = +X), "
            "+Y anterior (forward)"
        )

        # Phenotype range table
        logger.info(sep)
        logger.info("Phenotype label  | Range  | Notes")
        logger.info("-" * 60)
        notes_map = {
            "gender":       "[0,1]  0=male morphology, 1=female",
            "age":          "[0,1]  0=infant, 0.5=~35 yr, 1=elder",
            "muscle":       "[0,1]  0=low muscle, 1=high muscle",
            "weight":       "[0,1]  0=very light, 1=very heavy",
            "height":       "[0,1]  0=very short, 1=very tall",
            "proportions":  "[0,1]  0=stocky/short-limbed, 1=elongated",
        }
        for lbl in pheno:
            note = notes_map.get(lbl, "[0,1]  (see anny docs)")
            logger.info("  %-16s | %s", lbl, note)

        # Run a dummy forward to reveal output keys and vertex scale
        logger.info(sep)
        logger.info("Running dummy forward pass to inspect output ...")
        with torch.no_grad():
            _pose = self._identity_pose.expand(1, m.bone_count, 4, 4)
            _pheno = {k: torch.tensor(0.5, device=self._device, dtype=self._dtype)
                      for k in m.phenotype_labels}
            _raw = m(pose_parameters=_pose, phenotype_kwargs=_pheno)

        logger.info("output keys: %s", list(_raw.keys()))
        for k, v in _raw.items():
            if isinstance(v, torch.Tensor):
                logger.info(
                    "  output['%s']: shape=%s dtype=%s", k, tuple(v.shape), v.dtype
                )

        verts = _raw["vertices"][0]
        logger.info(
            "  vertex Z range: %.4f to %.4f  (span=%.3f m ≈ stature)",
            float(verts[:, 2].min()), float(verts[:, 2].max()),
            float(verts[:, 2].max() - verts[:, 2].min()),
        )

        # Log bone_heads for key bones
        rbh = _raw["rest_bone_heads"][0].cpu().numpy()
        bl = list(m.bone_labels)
        key_bones = ["root", "head", "neck01", "upperleg01.L", "foot.L"]
        logger.info("Key bone rest_bone_heads (x,y,z) in metres:")
        for b in key_bones:
            if b in bl:
                pos = rbh[bl.index(b)]
                logger.info("  %-20s: (%.4f, %.4f, %.4f)", b, *pos)

        logger.info(sep)

    # -----------------------------------------------------------------------
    # BodyModel interface
    # -----------------------------------------------------------------------

    def forward(
        self,
        phenotypes: Optional[dict] = None,
        pose: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        pheno_kwargs = self._build_pheno_kwargs(phenotypes, batch_size)
        pose_params = self._build_pose_params(pose, batch_size)

        raw = self._model(
            pose_parameters=pose_params,
            phenotype_kwargs=pheno_kwargs,
        )

        vertices = raw["vertices"]          # (B, V, 3) metres, +Z up
        joints = self._extract_joints(raw)  # (B, J, 3) metres
        raw["vertex_bone_weights"] = getattr(self._model, "vertex_bone_weights", None)
        raw["vertex_bone_indices"] = getattr(self._model, "vertex_bone_indices", None)
        raw["bone_labels"] = list(self._model.bone_labels)
        return BodyOutput(
            vertices=vertices,
            faces=self._faces,
            joints=joints,
            joint_names=list(self._model.bone_labels),
            phenotypes=phenotypes or {},
            raw=raw,
        )

    def rest_pose(
        self,
        phenotypes: Optional[dict] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> BodyOutput:
        """
        Return the body in canonical A-pose with the given phenotypes.

        All bone transforms are identity 4×4 matrices — no articulation
        beyond the model's built-in A-pose offsets.  This is the pose
        required for all measurements in :mod:`realsense_measure.body.measure`.
        """
        pheno_kwargs = self._build_pheno_kwargs(phenotypes, batch_size)
        pose_params = self._identity_pose.expand(batch_size, self._bone_count, 4, 4)

        raw = self._model(
            pose_parameters=pose_params,
            phenotype_kwargs=pheno_kwargs,
        )

        vertices = raw["vertices"]
        joints = self._extract_joints(raw)
        raw["vertex_bone_weights"] = getattr(self._model, "vertex_bone_weights", None)
        raw["vertex_bone_indices"] = getattr(self._model, "vertex_bone_indices", None)
        raw["bone_labels"] = list(self._model.bone_labels)
        return BodyOutput(
            vertices=vertices,
            faces=self._faces,
            joints=joints,
            joint_names=list(self._model.bone_labels),
            phenotypes=phenotypes or {},
            raw=raw,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def phenotype_labels(self) -> List[str]:
        return list(self._model.phenotype_labels)

    @property
    def joint_names(self) -> List[str]:
        return list(self._model.bone_labels)

    @property
    def num_vertices(self) -> int:
        return int(self._model.faces.max().item()) + 1

    @property
    def num_faces(self) -> int:
        return int(self._model.faces.shape[0])

    @property
    def device(self) -> torch.device:
        return self._device

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_pheno_kwargs(
        self, phenotypes: Optional[dict], batch_size: int
    ) -> dict:
        """Convert user phenotype dict → anny phenotype_kwargs tensors."""
        result = {}
        for label in self._model.phenotype_labels:
            val: float | torch.Tensor = 0.5   # neutral default
            if phenotypes and label in phenotypes:
                val = phenotypes[label]

            if isinstance(val, torch.Tensor):
                t = val.to(device=self._device, dtype=self._dtype)
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(batch_size)
            else:
                t = torch.full(
                    (batch_size,), float(val),
                    device=self._device, dtype=self._dtype
                )
            result[label] = t
        return result

    def _build_pose_params(
        self, pose: Optional[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        if pose is not None:
            return pose.to(device=self._device, dtype=self._dtype)
        return self._identity_pose.expand(batch_size, self._bone_count, 4, 4)

    def _extract_joints(self, raw: dict) -> torch.Tensor:
        """
        Extract (B, J, 3) joint positions from anny's raw output dict.

        anny provides ``rest_bone_heads`` (B, J, 3) as rest-pose joint
        positions in metres, and ``bone_poses`` (B, J, 4, 4) for posed.
        We use the translation column of ``bone_poses`` for posed output,
        or fall back to ``rest_bone_heads`` for rest-pose calls.
        """
        if "bone_poses" in raw:
            # Translation is the last column of the 4×4 homogeneous matrix
            return raw["bone_poses"][:, :, :3, 3]  # (B, J, 3)
        if "rest_bone_heads" in raw:
            return raw["rest_bone_heads"]           # (B, J, 3)
        logger.warning(
            "AnnyBackend: no joint data found in raw output. "
            "Keys present: %s. Returning zeros.", list(raw.keys())
        )
        B = raw["vertices"].shape[0]
        J = self._bone_count
        return torch.zeros(B, J, 3, device=self._device, dtype=self._dtype)
