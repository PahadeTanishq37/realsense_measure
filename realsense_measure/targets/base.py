"""
Extension seam for measurement targets.

This file is the ONLY place the pipeline needs to know about in order to
support new object types.  Adding a "head" or "body" target later is purely
additive:

  1. Create ``targets/head.py`` (or ``targets/body.py``).
  2. Implement the :class:`Target` interface (``segment`` + ``measure``).
  3. Decorate the class with ``@register_target``.
  4. Import the new module anywhere before the pipeline runs
     (``targets/__init__.py`` is the natural place).

Nothing in ``pipeline.py``, ``registration.py``, ``reconstruction.py``, or
``preprocessing.py`` needs to change.  The pipeline always calls
:func:`get_target` with the name from ``PipelineConfig.target.name`` and
works entirely through the :class:`Target` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import open3d as o3d


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Target(ABC):
    """
    Interface that every measurable object type must implement.

    Subclasses should:
    - Set the ``name`` class attribute to a short, unique identifier string
      (e.g. ``"box"``, ``"head"``).
    - Decorate themselves with :func:`register_target` so the pipeline can
      look them up by name at runtime.
    - Implement :meth:`segment` and :meth:`measure`.
    """

    #: Short identifier used to look up this target in the registry.
    #: Override in every concrete subclass.
    name: str = "base"

    @abstractmethod
    def segment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Refine the fused object cloud down to just this target's points.

        The input cloud is already reasonably clean (fused, denoised, largest
        cluster extracted) but may still contain geometry that is irrelevant
        to the specific measurement.  This method performs any final,
        target-specific cropping or segmentation needed before measuring.

        Examples:
        - **Box**: one more plane-removal pass to strip any residual table
          surface that survived the per-frame preprocessing.
        - **Head**: crop to the face region using a bounding-sphere heuristic.
        - **Body**: separate limbs from torso before measuring each segment.

        Parameters
        ----------
        pcd:
            Fused, mostly-clean object cloud in the shared reference frame.

        Returns
        -------
        o3d.geometry.PointCloud
            Subset (or transformed version) of ``pcd`` ready for measurement.
        """

    @abstractmethod
    def measure(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        Compute and return all measurements for this target.

        The returned dict must be **JSON-serialisable** (no numpy arrays, no
        Open3D objects) *except* for one reserved key:

        ``"geometry_for_viz"``
            An Open3D geometry object (e.g. ``OrientedBoundingBox``,
            ``TriangleMesh``) that the visualizer will draw on top of the
            point cloud.  May be ``None`` if no extra geometry is needed.

        All other keys should be plain Python scalars or strings so the dict
        can be written directly to a ``.json`` file by ``pipeline.py``.

        Parameters
        ----------
        pcd:
            Segmented target cloud, as returned by :meth:`segment`.

        Returns
        -------
        dict
            Measurement results.  Always includes ``"geometry_for_viz"``.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Target]] = {}


def register_target(cls: type[Target]) -> type[Target]:
    """
    Class decorator that registers a :class:`Target` subclass by its ``name``.

    Usage::

        @register_target
        class BoxTarget(Target):
            name = "box"
            ...

    The decorator stores ``cls`` in the module-level ``_REGISTRY`` dict keyed
    by ``cls.name``, then returns ``cls`` unchanged so normal class machinery
    is unaffected.

    Parameters
    ----------
    cls:
        A concrete :class:`Target` subclass with a unique ``name`` attribute.

    Returns
    -------
    type[Target]
        ``cls`` itself (unchanged).
    """
    _REGISTRY[cls.name] = cls
    return cls


def get_target(name: str) -> Target:
    """
    Instantiate a registered :class:`Target` by name.

    Parameters
    ----------
    name:
        The short identifier string matching a subclass's ``name`` attribute
        (e.g. ``"box"``, ``"head"``).

    Returns
    -------
    Target
        A freshly constructed instance of the matching target class.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry, with a message listing all
        currently registered target names.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none registered>"
        raise ValueError(
            f"Unknown target {name!r}.  "
            f"Available targets: {available}.  "
            f"To add a new one, create targets/<name>.py, implement Target, "
            f"and decorate with @register_target."
        )
    return _REGISTRY[name]()
