"""
Synthetic test harness for real-time human/head capture feedback.

Tests:
  (a) Steady frontal face -> frames_since_good_lock stays 0, no warnings.
  (b) Profile turn with landmarks dropping below 4 -> verifies LOW LANDMARK LOCK (yellow)
      at frame 3, COVERAGE GAP LIKELY (red) at frame 8, and reset on recovery.
  (c) Sudden large landmark jump -> verifies TOO FAST -- SLOW DOWN ROTATION (red) fires.
  (d) Config threshold customization and box mode separation.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import HeadCaptureConfig, PipelineConfig, TargetConfig
from pipeline import HeadCaptureTracker, ScanPipeline


def _make_frontal_landmarks(offset_x: float = 0.0, offset_y: float = 0.0) -> dict[str, np.ndarray]:
    """Create 8 synthetic face landmarks centered at (320 + offset_x, 240 + offset_y)."""
    base = {
        "nose_tip": np.array([320.0, 240.0]),
        "nose_bridge": np.array([320.0, 215.0]),
        "left_eye_inner": np.array([300.0, 200.0]),
        "right_eye_inner": np.array([340.0, 200.0]),
        "left_eye_outer": np.array([280.0, 195.0]),
        "right_eye_outer": np.array([360.0, 195.0]),
        "upper_lip_center": np.array([320.0, 260.0]),
        "lower_lip_center": np.array([320.0, 275.0]),
    }
    return {k: v + np.array([offset_x, offset_y]) for k, v in base.items()}


def _make_profile_landmarks() -> dict[str, np.ndarray]:
    """Create only 2 visible landmarks simulating an extreme profile turn (< 4 threshold)."""
    return {
        "chin": np.array([330.0, 310.0]),
        "left_eye_outer": np.array([285.0, 195.0]),
    }


def test_steady_frontal_face():
    """Case (a): steady frontal face -> lock stays 0, no warnings."""
    cfg = HeadCaptureConfig(
        min_landmarks_for_lock=4,
        min_frames_for_low_lock=3,
        max_frames_without_lock=8,
        max_landmark_px_motion=40.0,
    )
    tracker = HeadCaptureTracker(cfg)

    print("\n--- Test (a): Steady Frontal Face Tracking ---")
    for frame_idx in range(10):
        # Small natural micro-movement (sub-pixel / 1px jitter)
        jitter_x = np.sin(frame_idx * 0.5) * 1.5
        jitter_y = np.cos(frame_idx * 0.5) * 1.0
        lm = _make_frontal_landmarks(jitter_x, jitter_y)

        warn_text, warn_color, motion_px, lock_gap = tracker.update(lm)
        print(f"  Frame {frame_idx:02d}: {len(lm)} landmarks, lock_gap={lock_gap}, "
              f"motion={motion_px:.2f}px, warn={warn_text}")

        assert lock_gap == 0, f"Expected lock_gap 0 for steady face, got {lock_gap}"
        assert warn_text is None, f"Expected no warning for steady face, got {warn_text}"
        assert warn_color is None
        assert motion_px < cfg.max_landmark_px_motion

    print("  [PASS] Steady frontal face: frames_since_good_lock stays 0, no warnings fired.")


def test_profile_turn_and_coverage_gap():
    """
    Case (b): profile turn where landmarks drop below 4:
    - frames 1-2: lock_gap increments, no warning yet
    - frame 3: LOW LANDMARK LOCK (yellow) fires at count 3
    - frames 4-7: LOW LANDMARK LOCK persists
    - frame 8: COVERAGE GAP LIKELY (red) fires at count 8
    - frame 9: face returns -> lock_gap resets to 0, warning clears
    """
    cfg = HeadCaptureConfig(
        min_landmarks_for_lock=4,
        min_frames_for_low_lock=3,
        max_frames_without_lock=8,
        max_landmark_px_motion=40.0,
    )
    tracker = HeadCaptureTracker(cfg)

    print("\n--- Test (b): Profile Turn and Coverage Gap Tracking ---")

    # Initial frame: good frontal lock
    warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_frontal_landmarks())
    assert lock_gap == 0
    assert warn_text is None
    print(f"  Frame 00 (Initial lock): lock_gap={lock_gap}, warn={warn_text}")

    # Frames 1-2: Partial profile, only 2 landmarks (< 4 threshold)
    for f in [1, 2]:
        warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_profile_landmarks())
        print(f"  Frame {f:02d} (<4 landmarks): lock_gap={lock_gap}, warn={warn_text}")
        assert lock_gap == f
        assert warn_text is None, f"Frame {f} should not warn yet before min_frames_for_low_lock=3"
        assert warn_color is None

    # Frame 3: lock_gap reaches 3 -> YELLOW warning fires
    warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_profile_landmarks())
    print(f"  Frame 03 (Threshold=3): lock_gap={lock_gap}, warn={warn_text} (color={warn_color})")
    assert lock_gap == 3
    assert warn_text == "LOW LANDMARK LOCK -- FACE CAMERA MORE DIRECTLY"
    assert warn_color == "yellow"

    # Frames 4 to 7: lock_gap continues incrementing, YELLOW warning persists
    for f in range(4, 8):
        warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_profile_landmarks())
        print(f"  Frame {f:02d} (Persistent low lock): lock_gap={lock_gap}, warn={warn_text}")
        assert lock_gap == f
        assert warn_text == "LOW LANDMARK LOCK -- FACE CAMERA MORE DIRECTLY"
        assert warn_color == "yellow"

    # Frame 8: lock_gap reaches max_frames_without_lock=8 -> RED COVERAGE GAP warning fires
    warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_profile_landmarks())
    print(f"  Frame 08 (Hard cap=8): lock_gap={lock_gap}, warn={warn_text} (color={warn_color})")
    assert lock_gap == 8
    assert warn_text == "COVERAGE GAP LIKELY -- CONSIDER RE-CAPTURING THIS ANGLE"
    assert warn_color == "red"

    # Frame 9: Operator turns face back toward camera -> Full 8 landmarks detected
    warn_text, warn_color, motion_px, lock_gap = tracker.update(_make_frontal_landmarks())
    print(f"  Frame 09 (Face recovered): lock_gap={lock_gap}, warn={warn_text}")
    assert lock_gap == 0, "lock_gap must reset to 0 upon recovering >= 4 landmarks"
    assert warn_text is None, "Warning must clear upon recovering tracking lock"
    assert warn_color is None

    print("  [PASS] Profile turn & coverage gap: low lock yellow (3) and coverage gap red (8) verified.")


def test_sudden_large_landmark_jump():
    """Case (c): sudden large landmark jump -> TOO FAST -- SLOW DOWN ROTATION fires."""
    cfg = HeadCaptureConfig(
        min_landmarks_for_lock=4,
        min_frames_for_low_lock=3,
        max_frames_without_lock=8,
        max_landmark_px_motion=40.0,
    )
    tracker = HeadCaptureTracker(cfg)

    print("\n--- Test (c): Sudden Large Landmark Jump (Motion Warning) ---")

    # Frame 0: steady frontal face at base position
    lm0 = _make_frontal_landmarks(0.0, 0.0)
    warn_text, warn_color, motion_px, lock_gap = tracker.update(lm0)
    assert lock_gap == 0
    assert warn_text is None
    print(f"  Frame 00 (Origin): lock_gap={lock_gap}, motion={motion_px:.2f}px, warn={warn_text}")

    # Frame 1: camera whip / rapid turn causing a 65 px jump (> 40.0 px threshold)
    lm1 = _make_frontal_landmarks(65.0, 0.0)
    warn_text, warn_color, motion_px, lock_gap = tracker.update(lm1)
    print(f"  Frame 01 (65px jump): lock_gap={lock_gap}, motion={motion_px:.2f}px, warn={warn_text} (color={warn_color})")
    assert lock_gap == 0
    assert abs(motion_px - 65.0) < 1e-4
    assert warn_text == "TOO FAST -- SLOW DOWN ROTATION"
    assert warn_color == "red"

    # Frame 2: camera slows down, motion drops to 3 px
    lm2 = _make_frontal_landmarks(68.0, 0.0)
    warn_text, warn_color, motion_px, lock_gap = tracker.update(lm2)
    print(f"  Frame 02 (Settled): lock_gap={lock_gap}, motion={motion_px:.2f}px, warn={warn_text}")
    assert lock_gap == 0
    assert abs(motion_px - 3.0) < 1e-4
    assert warn_text is None
    assert warn_color is None

    print("  [PASS] Sudden landmark jump: TOO FAST -- SLOW DOWN ROTATION (red) fired and cleared.")


def test_box_mode_unaffected():
    """Confirm box mode pipeline has no head tracking overhead and does not activate head feedback."""
    print("\n--- Test (d): Target Configuration Scoping ---")
    cfg_box = PipelineConfig(target=TargetConfig(name="box"))
    assert cfg_box.target.name == "box"
    assert cfg_box.registration.registration_mode == "pointcloud"

    cfg_head = PipelineConfig(target=TargetConfig(name="head"))
    assert cfg_head.target.name == "head"
    assert cfg_head.registration.registration_mode == "rgbd_human"
    assert isinstance(cfg_head.head_capture, HeadCaptureConfig)
    assert cfg_head.head_capture.max_landmark_px_motion == 40.0
    assert cfg_head.head_capture.min_frames_for_low_lock == 3
    assert cfg_head.head_capture.max_frames_without_lock == 8
    print("  [PASS] Box mode registration and target settings unaffected; Head capture config verified.")


if __name__ == "__main__":
    test_steady_frontal_face()
    test_profile_turn_and_coverage_gap()
    test_sudden_large_landmark_jump()
    test_box_mode_unaffected()
    print("\n=================================================================")
    print("  ALL REAL-TIME HUMAN CAPTURE FEEDBACK HARNESS TESTS PASSED!")
    print("=================================================================\n")
