"""
realsense_measure/body/__init__.py

Body measurement sub-package.  Camera / depth-fitting is NOT included in
Step 1; this package covers model-based measurements from parametric body
models only.

Public surface
--------------
  BodyModel     – abstract interface (body/model.py)
  BodyOutput    – dataclass returned by BodyModel.forward()
  AnnyBackend   – anny.Anny wrapper (body/anny_backend.py)
  measure_body  – compute all measurements (body/measure.py)
  Measurement   – dataclass for a single measurement result
"""

from .model import BodyModel, BodyOutput
from .anny_backend import AnnyBackend
from .measure import Measurement, measure_body

__all__ = [
    "BodyModel",
    "BodyOutput",
    "AnnyBackend",
    "Measurement",
    "measure_body",
]

