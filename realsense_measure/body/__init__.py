"""
realsense_measure/body/__init__.py

Core body measurement and capture support sub-package.
"""

from .model import BodyModel, BodyOutput
from .anny_backend import AnnyBackend
from .measure import Measurement, measure_body, is_contour_closed
from .prior import AdultShapePrior
from .capture_session import (
    validate_anonymous_subject_id,
    CaptureQualityChecker,
    CaptureSessionManifest,
)
from .capture_motion import (
    evaluate_burst_stillness,
    StillnessResult,
)
from .hardware_smoke_test import (
    CollisionError,
    check_capture_collision,
)

__all__ = [
    "BodyModel",
    "BodyOutput",
    "AnnyBackend",
    "Measurement",
    "measure_body",
    "is_contour_closed",
    "AdultShapePrior",
    "validate_anonymous_subject_id",
    "CaptureQualityChecker",
    "CaptureSessionManifest",
    "evaluate_burst_stillness",
    "StillnessResult",
    "CollisionError",
    "check_capture_collision",
]
