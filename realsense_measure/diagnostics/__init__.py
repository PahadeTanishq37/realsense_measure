"""
realsense_measure/diagnostics/__init__.py
========================================
Diagnostic tools for sensor calibration, temporal depth noise evaluation,
and capture quality assessment.
"""

from __future__ import annotations

try:
    from realsense_measure.diagnostics.temporal_noise import (
        MatchedPopulationComparison,
        SingleStreamNoiseMetrics,
        TemporalNoiseComparison,
        compare_raw_vs_filtered_noise,
        compute_temporal_depth_noise,
    )
except ImportError:
    from diagnostics.temporal_noise import (
        MatchedPopulationComparison,
        SingleStreamNoiseMetrics,
        TemporalNoiseComparison,
        compare_raw_vs_filtered_noise,
        compute_temporal_depth_noise,
    )

__all__ = [
    "SingleStreamNoiseMetrics",
    "MatchedPopulationComparison",
    "TemporalNoiseComparison",
    "compute_temporal_depth_noise",
    "compare_raw_vs_filtered_noise",
]
